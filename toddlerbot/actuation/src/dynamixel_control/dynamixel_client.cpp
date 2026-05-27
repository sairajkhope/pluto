#include "dynamixel_client.h"
#include <iostream>
#include <cstring>
#include <thread>
#include <chrono>

#define ERRBIT_ALERT 128

// static members
std::set<DynamixelClient *> DynamixelClient::OPEN_CLIENTS;
std::once_flag DynamixelClient::cleanup_flag;

inline double DynamixelClient::now()
{
  return std::chrono::duration<double>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

DynamixelClient::DynamixelClient(const std::vector<int> &motor_ids,
                                 const std::string &port,
                                 int baudrate,
                                 bool lazy_connect)
    : motor_ids_(motor_ids),
      port_name_(port),
      baudrate_(baudrate),
      lazy_connect_(lazy_connect),
      port_handler_(dynamixel::PortHandler::getPortHandler(port.c_str())),
      packet_handler_(dynamixel::PacketHandler::getPacketHandler(PROTOCOL_VERSION)),
      bulk_reader_(port_handler_, packet_handler_)
{
  std::call_once(cleanup_flag, []()
                 { atexit(dynamixel_cleanup_handler); });
  if (OPEN_CLIENTS.count(this))
    throw std::runtime_error("Client already registered for port " + port_name_);
  OPEN_CLIENTS.insert(this);

  // init bulk reader params
  for (int id : motor_ids_)
    bulk_reader_.addParam((uint8_t)id, ADDR_PRESENT_POS_VEL_CUR, LEN_PRESENT_POS_VEL_CUR);

  // init data buffers
  data_dict_["pos"] = std::vector<float>(motor_ids_.size());
  data_dict_["vel"] = std::vector<float>(motor_ids_.size());
  data_dict_["cur"] = std::vector<float>(motor_ids_.size());

  if (!lazy_connect_)
    connect();
}

DynamixelClient::~DynamixelClient()
{
  disconnect();
}

void DynamixelClient::connect()
{
  if (is_connected())
    return;
  if (!port_handler_->openPort())
    throw std::runtime_error("Failed to open port " + port_name_);
  is_open_ = true;
  if (!port_handler_->setBaudRate(baudrate_))
  {
    port_handler_->closePort();
    throw std::runtime_error("Failed to set baudrate");
  }
}

void DynamixelClient::disconnect()
{
  if (!is_connected())
    return;
  set_torque_enabled(motor_ids_, false, 0);

  for (auto &p : sync_readers_)
    delete p.second;
  for (auto &p : sync_writers_)
    delete p.second;

  sync_readers_.clear();
  sync_writers_.clear();

  port_handler_->closePort();
  is_open_ = false;
  OPEN_CLIENTS.erase(this);
}

bool DynamixelClient::is_connected() const
{
  return is_open_;
}

void DynamixelClient::set_torque_enabled(const std::vector<int> &ids,
                                         bool en,
                                         int retries,
                                         float retry_interval)
{
  auto rem = ids;
  while (!rem.empty())
  {
    rem = write_byte(rem, en ? 1 : 0, ADDR_TORQUE_ENABLE);
    if (rem.empty() || retries-- == 0)
      break;
    std::this_thread::sleep_for(std::chrono::duration<float>(retry_interval));
  }
}

void DynamixelClient::reboot(const std::vector<int> &ids)
{
  check_connected();
  for (int id : ids)
    packet_handler_->reboot(port_handler_, (uint8_t)id);
}

void DynamixelClient::clear_multi_turn(const std::vector<int> &ids)
{
  check_connected();
  for (int id : ids)
  {
    packet_handler_->clearMultiTurn(port_handler_, (uint8_t)id);
  }
}

void DynamixelClient::clear_error(const std::vector<int> &ids)
{
  check_connected();
  for (int id : ids)
  {
    packet_handler_->clearError(port_handler_, (uint8_t)id);
  }
}

// ------------------------------------------------------------------
// sync_read
std::pair<double, std::vector<float>>
DynamixelClient::sync_read(uint16_t addr, uint16_t len, float scale)
{
  check_connected();
  auto key = std::make_pair(addr, len);
  if (!sync_readers_.count(key))
  {
    auto *r = new dynamixel::GroupSyncRead(port_handler_, packet_handler_, addr, len);
    for (int id : motor_ids_)
      r->addParam((uint8_t)id);
    sync_readers_[key] = r;
  }
  auto *gr = sync_readers_[key];

  auto t_tx = std::chrono::high_resolution_clock::now();
  int comm;
  {
    std::lock_guard<std::mutex> lock(comms_mutex_);
    comm = gr->txRxPacket();
  }
  auto t_rx = std::chrono::high_resolution_clock::now();
  double latency_ms = std::chrono::duration<double, std::milli>(t_rx - t_tx).count();
  if (!handle_packet_result(comm, 0, -1, "sync_read"))
  {
    throw std::runtime_error("sync_read failed");
  }

  std::vector<float> out(motor_ids_.size(), 0.0f);
  for (size_t i = 0; i < motor_ids_.size(); ++i)
  {
    uint8_t id = motor_ids_[i];
    if (!gr->isAvailable(id, addr, len))
      continue;
    uint32_t raw = gr->getData(id, addr, len);
    out[i] = unsigned_to_signed(raw, len) * scale;
  }
  return {latency_ms, out};
}

// ------------------------------------------------------------------
// bulk_read
std::pair<double, std::map<std::string, std::vector<float>>>
DynamixelClient::bulk_read(const std::vector<std::string> &attrs, int retries)
{
  check_connected();
  auto t_tx = std::chrono::high_resolution_clock::now();

  bool success = false;
  int attempt = 0;
  while (true)
  {
    attempt++;
    int comm;
    {
      std::lock_guard<std::mutex> lock(comms_mutex_);
      comm = bulk_reader_.txPacket();
      if (comm == COMM_SUCCESS)
      {
        comm = bulk_reader_.rxPacket();
      }
    }
    if (handle_packet_result(comm, 0, -1, "bulk_read"))
    {
      success = true;
      break; // success
    }
    // Debug: print port and motor IDs on failure
    std::cout << "[Dynamixel DEBUG] bulk_read FAILED on port: " << port_name_
              << " | attempt: " << attempt
              << " | motor_ids: [";
    for (size_t i = 0; i < motor_ids_.size(); ++i)
    {
      std::cout << motor_ids_[i];
      if (i < motor_ids_.size() - 1)
        std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    if (retries == 0)
    {
      break; // give up
    }
    retries--;
  }

  auto t_rx = std::chrono::high_resolution_clock::now();
  double latency_ms = std::chrono::duration<double, std::milli>(t_rx - t_tx).count();

  std::map<std::string, std::vector<float>> res;
  for (auto &a : attrs)
  {
    res[a] = std::vector<float>(motor_ids_.size(), 0.0f);
  }

  // Debug: track which motors failed to respond
  std::vector<int> failed_ids;
  for (size_t i = 0; i < motor_ids_.size(); ++i)
  {
    uint8_t id = motor_ids_[i];
    if (!bulk_reader_.isAvailable(id, ADDR_PRESENT_POS_VEL_CUR, LEN_PRESENT_POS_VEL_CUR))
    {
      failed_ids.push_back(id);
      continue;
    }

    uint32_t p = bulk_reader_.getData(id, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION);
    res["pos"][i] = unsigned_to_signed(p, LEN_PRESENT_POSITION) * DEFAULT_POS_SCALE;

    uint32_t v = bulk_reader_.getData(id, ADDR_PRESENT_VELOCITY, LEN_PRESENT_VELOCITY);
    res["vel"][i] = unsigned_to_signed(v, LEN_PRESENT_VELOCITY) * DEFAULT_VEL_SCALE;

    uint16_t c = bulk_reader_.getData(id, ADDR_PRESENT_CURRENT, LEN_PRESENT_CURRENT);
    if (cur_scale_arr_.empty())
      cur_scale_arr_ = get_cur_scale();
    res["cur"][i] = unsigned_to_signed(c, LEN_PRESENT_CURRENT) * cur_scale_arr_[i];
  }

  // Debug: report which specific motors didn't respond
  if (!failed_ids.empty())
  {
    std::cout << "[Dynamixel DEBUG] bulk_read on port: " << port_name_
              << " | motors NOT responding: [";
    for (size_t i = 0; i < failed_ids.size(); ++i)
    {
      std::cout << failed_ids[i];
      if (i < failed_ids.size() - 1)
        std::cout << ", ";
    }
    std::cout << "] (out of " << motor_ids_.size() << " total)" << std::endl;
  }

  return {latency_ms, res};
}

// ------------------------------------------------------------------
// sync_write
void DynamixelClient::sync_write(const std::vector<int> &ids,
                                 const std::vector<int> &vals,
                                 uint16_t addr, uint16_t len)
{
  check_connected();
  auto key = std::make_pair(addr, len);
  if (!sync_writers_.count(key))
  {
    sync_writers_[key] = new dynamixel::GroupSyncWrite(
        port_handler_, packet_handler_, addr, len);
  }
  auto *gw = sync_writers_[key];

  for (size_t i = 0; i < ids.size(); ++i)
  {
    uint8_t buf[4] = {};
    std::memcpy(buf, &vals[i], len);
    gw->addParam((uint8_t)ids[i], buf);
  }
  int comm;
  {
    std::lock_guard<std::mutex> lock(comms_mutex_);
    comm = gw->txPacket();
  }
  handle_packet_result(comm, 0, -1, "sync_write");
  gw->clearParam();
}

void DynamixelClient::write_desired_pos(const std::vector<int> &ids, const std::vector<float> &ps)
{
  std::vector<int> s;
  for (auto p : ps)
    s.push_back((int)(p / DEFAULT_POS_SCALE));
  sync_write(ids, s, ADDR_GOAL_POSITION, LEN_GOAL_POSITION);
}

void DynamixelClient::write_desired_cur(const std::vector<int> &ids, const std::vector<float> &cs)
{
  if (cur_scale_arr_.empty())
    cur_scale_arr_ = get_cur_scale();
  std::vector<int> s;
  for (size_t i = 0; i < cs.size(); ++i)
    s.push_back((int)(cs[i] / cur_scale_arr_[i]));
  sync_write(ids, s, ADDR_GOAL_CURRENT, LEN_GOAL_CURRENT);
}

std::vector<float> DynamixelClient::get_cur_scale(int retries)
{
  auto arr = read_model_number(retries).second;
  std::vector<float> sc(arr.size(), 1.0f);
  for (size_t i = 0; i < arr.size(); ++i)
  {
    if (arr[i] == 1030 || arr[i] == 1020) // XM430-210 or XM430-W350
      sc[i] = 2.69f;
  }
  return sc;
}

void DynamixelClient::check_connected()
{
  if (lazy_connect_ && !is_connected())
    connect();
  if (!is_connected())
    throw std::runtime_error("Must connect");
}

bool DynamixelClient::handle_packet_result(int comm, int dxl_err, int id, const std::string &ctx)
{
  std::string e;
  if (comm != COMM_SUCCESS)
    e = packet_handler_->getTxRxResult(comm);
  else if (dxl_err)
    e = packet_handler_->getRxPacketError(dxl_err);
  if (!e.empty())
  {
    std::cout << "[Dynamixel] " << ctx << " id=" << id << " -- " << e << std::endl;
    if (dxl_err & ERRBIT_ALERT)
    {
      std::cout << "[Dynamixel] " << ctx << " id=" << id << " -- Rebooting..." << std::endl;
      // handle the hardware error by rebooting
      reboot({id});
      // wait for servo to initialize after reboot
      std::this_thread::sleep_for(std::chrono::milliseconds(500));
      return true;
    }
    else
      return false;
  }
  return true;
}

std::vector<int> DynamixelClient::write_byte(const std::vector<int> &ids, int val, uint16_t addr)
{
  check_connected();
  std::vector<int> f;
  for (int id : ids)
  {
    uint8_t xe = 0;
    int r = packet_handler_->write1ByteTxRx(port_handler_, id, addr, val, &xe);
    if (!handle_packet_result(r, xe, id, "write_byte"))
      f.push_back(id);
  }
  return f;
}

int DynamixelClient::unsigned_to_signed(int v, int s)
{
  int bits = 8 * s;
  if (v & (1 << (bits - 1)))
    v -= 1 << bits;
  return v;
}
int DynamixelClient::signed_to_unsigned(int v, int s)
{
  if (v < 0)
    v = (1 << (8 * s)) + v;
  return v;
}

void dynamixel_cleanup_handler()
{
  for (auto c : DynamixelClient::get_open_clients())
    try
    {
      c->disconnect();
    }
    catch (...)
    {
    }
}

// Add missing implementations for these methods:

std::pair<double, std::vector<float>> DynamixelClient::read_vin(int retries)
{
  return sync_read(ADDR_PRESENT_V_IN, LEN_PRESENT_V_IN, DEFAULT_V_IN_SCALE);
}

std::pair<double, std::vector<float>> DynamixelClient::read_pos(int retries)
{
  return sync_read(ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION, DEFAULT_POS_SCALE);
}

std::tuple<double, std::vector<float>, std::vector<float>, std::vector<float>>
DynamixelClient::read_pos_vel_cur(int retries)
{
  auto res = bulk_read({"pos", "vel", "cur"}, retries);
  return {res.first, res.second.at("pos"), res.second.at("vel"), res.second.at("cur")};
}

std::pair<double, std::vector<float>> DynamixelClient::read_model_number(int retries)
{
  return sync_read(ADDR_MODEL_NUMBER, LEN_MODEL_NUMBER, 1.0f);
}
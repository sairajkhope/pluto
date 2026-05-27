#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "dynamixel_control.h"
#include "dynamixel_sdk/port_handler.h"
#include "dynamixel_sdk/packet_handler.h"
#include <future>
#include <thread>
#include <vector>
#include <tuple>
#include <iostream>
#include <filesystem>
#include <regex>
#include <cstdlib>  // for std::system
#include <cstdio>   // for popen, fgets, pclose
#include <unistd.h> // for uname
#include <sys/utsname.h>

namespace py = pybind11;
namespace fs = std::filesystem;

// You can pass this in if you don't want global state
std::vector<DynamixelControl *> controllers;

std::vector<int> scan_port(const std::string &port_name,
                           int baudrate,
                           int protocol_version = 2.0,
                           int max_motor_id = 32)
{
  using namespace dynamixel;

  std::vector<int> found_ids;

  PortHandler *portHandler = PortHandler::getPortHandler(port_name.c_str());
  PacketHandler *packetHandler = PacketHandler::getPacketHandler(protocol_version);

  if (!portHandler->openPort())
  {
    std::cout << "[scan_port] Failed to open " << port_name << std::endl;
    return found_ids;
  }

  if (!portHandler->setBaudRate(baudrate))
  {
    std::cout << "[scan_port] Failed to set baudrate on " << port_name << std::endl;
    portHandler->closePort();
    return found_ids;
  }

  for (int id = 0; id < max_motor_id; ++id)
  {
    uint16_t model_number;
    uint8_t error;
    int result = packetHandler->ping(portHandler, id, &model_number, &error);
    if (result == COMM_SUCCESS && error == 0)
    {
      found_ids.push_back(id);
    }
  }

  portHandler->closePort();
  return found_ids;
}

void set_latency_timer(const std::string &port, int latency_value)
{
  std::string os_type;
  struct utsname buffer;
  if (uname(&buffer) == 0)
  {
    os_type = buffer.sysname;
  }
  else
  {
    throw std::runtime_error("Failed to detect OS type");
  }

  std::string command;

  if (os_type == "Linux")
  {
    std::string port_name = port.substr(port.find_last_of('/') + 1);
    command = "echo " + std::to_string(latency_value) +
              " | sudo tee /sys/bus/usb-serial/devices/" + port_name + "/latency_timer";
  }
  else if (os_type == "Darwin")
  {
    command = "./toddlerbot/actuation/latency_timer_setter_macOS/set_latency_timer -l " +
              std::to_string(latency_value);
  }
  else
  {
    throw std::runtime_error("Unsupported OS: " + os_type);
  }

  // Use popen to capture output
  FILE *pipe = popen(command.c_str(), "r");
  if (!pipe)
  {
    throw std::runtime_error("Failed to run latency timer command");
  }

  char buffer_output[128];
  std::string result;
  while (fgets(buffer_output, sizeof(buffer_output), pipe) != nullptr)
  {
    result += buffer_output;
  }

  int return_code = pclose(pipe);
  if (return_code != 0)
  {
    throw std::runtime_error("Latency timer command failed with code " + std::to_string(return_code));
  }

  if (!result.empty() && result.back() == '\n')
  {
    result.pop_back();
  }
  std::cout << "Latency Timer set: " << result << std::endl;
}

std::vector<std::shared_ptr<DynamixelControl>> create_controllers(
    const std::string &port_pattern,
    const std::vector<float> &kp,
    const std::vector<float> &kd,
    const std::vector<float> &zero_pos,
    const std::vector<std::string> &control_mode,
    int baudrate,
    int return_delay)
{
  std::vector<std::shared_ptr<DynamixelControl>> controllers;
  std::map<std::string, std::vector<int>> port_to_ids;

  if (port_pattern.find("ttyUSB") != std::string::npos ||
      port_pattern.find("ttyACM") != std::string::npos)
  {
    set_latency_timer(port_pattern, 1);
  }

  std::regex pattern(port_pattern);
  for (const auto &entry : fs::directory_iterator("/dev"))
  {
    std::string filename = entry.path().filename().string();
    if (!std::regex_match(filename, pattern))
      continue;

    std::string full_path = entry.path();
    try
    {
      auto ids = scan_port(full_path, baudrate);
      if (!ids.empty())
        port_to_ids[full_path] = ids;
    }
    catch (const std::exception &e)
    {
      std::cout << "[create_controllers] Error scanning " << full_path << ": " << e.what() << std::endl;
    }
  }

  // Collect and sort all detected motor IDs
  std::vector<int> all_ids;
  for (const auto &pair : port_to_ids)
    all_ids.insert(all_ids.end(), pair.second.begin(), pair.second.end());

  std::sort(all_ids.begin(), all_ids.end());

  // Build a map from motor ID to index in global kp/kd arrays
  std::unordered_map<int, int> id_to_index;
  for (size_t i = 0; i < all_ids.size(); ++i)
    id_to_index[all_ids[i]] = i;

  // For each port, extract the corresponding slice of controller params
  for (const auto &[port, ids] : port_to_ids)
  {
    std::vector<float> kp_local, kd_local, zero_local;
    std::vector<std::string> mode_local;

    for (int id : ids)
    {
      int idx = id_to_index.at(id);
      kp_local.push_back(kp[idx]);
      kd_local.push_back(kd[idx]);
      zero_local.push_back(zero_pos[idx]);
      mode_local.push_back(control_mode[idx]);
    }

    std::cout << "Detected motors on " << port << ": ";
    for (int id : ids)
      std::cout << id << " ";
    std::cout << std::endl;

    controllers.emplace_back(std::make_shared<DynamixelControl>(
        port, ids, kp_local, kd_local, zero_local, mode_local, baudrate, return_delay));
  }
  return controllers;
}

void initialize_motors(
    const std::vector<std::shared_ptr<DynamixelControl>> &ctrls)
{
  size_t N = ctrls.size();
  std::vector<std::thread> threads;
  threads.reserve(N);

  // Initialize each controller in parallel with staggered start times
  for (size_t i = 0; i < ctrls.size(); ++i)
  {
    threads.emplace_back([ctrl = ctrls[i], delay_ms = i * 50]()
                         { 
                           std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
                           ctrl->initialize_motors(); });
  }

  for (auto &t : threads)
    t.join();
}

std::map<std::string, std::vector<int>>
get_motor_ids(const std::vector<DynamixelControl *> &controllers)
{
  std::map<std::string, std::vector<int>> motor_ids;
  for (size_t i = 0; i < controllers.size(); ++i)
  {
    motor_ids["controller_" + std::to_string(i)] = controllers[i]->get_motor_ids();
  }
  return motor_ids;
}

std::map<std::string, std::map<std::string, std::vector<float>>>
get_motor_states(const std::vector<DynamixelControl *> &controllers, int retries = 0)
{
  size_t N = controllers.size();
  std::vector<std::future<std::map<std::string, std::vector<float>>>> futures;
  futures.reserve(N);

  // Launch threads
  for (auto *ctrl : controllers)
  {
    futures.emplace_back(std::async(std::launch::async, [ctrl, retries]()
                                    { return ctrl->get_state(retries); }));
  }

  std::map<std::string, std::map<std::string, std::vector<float>>> states;

  for (size_t i = 0; i < N; ++i)
  {
    std::string key = "controller_" + std::to_string(i);
    try
    {
      states[key] = futures[i].get();
    }
    catch (const std::exception &e)
    {
      std::cout << "[get_motor_states] Exception for " << key << " ids: ";
      for (auto id : controllers[i]->get_motor_ids())
        std::cout << id << " ";
      std::cout << "-- " << e.what() << std::endl;

      // Insert an empty map to preserve key
      states[key] = std::map<std::string, std::vector<float>>{};
    }
  }
  return states;
}

void set_motor_pos(const std::vector<DynamixelControl *> &controllers,
                   const std::vector<std::vector<float>> &pos_vecs)
{
  size_t N = controllers.size();
  std::vector<std::future<void>> futures;
  futures.reserve(N);

  for (size_t i = 0; i < N && i < pos_vecs.size(); ++i)
  {
    futures.emplace_back(std::async(std::launch::async,
                                    [ctrl = controllers[i], pos = pos_vecs[i]]()
                                    {
                                      ctrl->set_pos(pos);
                                    }));
  }

  for (auto &f : futures)
    f.get(); // Wait and propagate any exceptions
}

void disable_motors(const std::vector<std::shared_ptr<DynamixelControl>> &ctrls)
{
  for (const auto &ctrl : ctrls)
  {
    ctrl->disable_motors();
  }
}

void enable_motors(const std::vector<std::shared_ptr<DynamixelControl>> &ctrls)
{
  for (const auto &ctrl : ctrls)
  {
    ctrl->enable_motors();
  }
}

void close_motors(const std::vector<std::shared_ptr<DynamixelControl>> &ctrls)
{
  for (const auto &ctrl : ctrls)
  {
    ctrl->close_motors();
  }
}

PYBIND11_MODULE(dynamixel_cpp, m)
{
  py::class_<DynamixelControl, std::shared_ptr<DynamixelControl>>(m, "DynamixelControl");
  m.def("create_controllers", &create_controllers, py::arg("port_pattern"),
        py::arg("kp"), py::arg("kd"), py::arg("zero_pos"), py::arg("control_mode"),
        py::arg("baudrate"), py::arg("return_delay"));
  m.def("scan_port", &scan_port,
        py::arg("port_name"),
        py::arg("baudrate"),
        py::arg("protocol_version"),
        py::arg("max_motor_id"));
  m.def("initialize",
        &initialize_motors,
        py::arg("controllers"),
        "Connect to client and initialize motors on each controller");
  m.def("get_motor_states", &get_motor_states, py::arg("controllers"), py::arg("retries"),
        py::call_guard<py::gil_scoped_release>(),
        "Get state of all motors across all controllers");
  m.def("get_motor_ids", &get_motor_ids, py::arg("controllers"),
        "Get motor IDs for each controller");
  m.def("set_motor_pos", &set_motor_pos, py::arg("controllers"), py::arg("pos_vecs"),
        py::call_guard<py::gil_scoped_release>(),
        "Set position for each controller's motors");
  m.def("disable_motors", &disable_motors, py::arg("controllers"),
        "Disable torque on all motors across all controllers");
  m.def("enable_motors", &enable_motors, py::arg("controllers"),
        "Enable torque on all motors across all controllers");
  m.def("close", &close_motors, py::arg("controllers"),
        "Disable torque and disconnect all specified DynamixelControl instances");
}

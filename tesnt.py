from dynamixel_sdk import *

# -----------------------------
# Settings
PROTOCOL_VERSION = 2.0
DXL_ID = 1
BAUDRATE = 2000000
DEVICENAME = "/dev/tty.usbserial-FTAK8D39"   # Windows: "COM3"

ADDR_PRESENT_POSITION = 132

# -----------------------------
# Initialize handlers
portHandler = PortHandler(DEVICENAME)
packetHandler = PacketHandler(PROTOCOL_VERSION)

# Open port
if not portHandler.openPort():
    raise RuntimeError("Failed to open port")

# Set baudrate
if not portHandler.setBaudRate(BAUDRATE):
    raise RuntimeError("Failed to set baudrate")

# -----------------------------
# Read position ONLY
dxl_position, dxl_comm_result, dxl_error = packetHandler.read4ByteTxRx(
    portHandler,
    DXL_ID,
    ADDR_PRESENT_POSITION
)

if dxl_comm_result != COMM_SUCCESS:
    print("Communication error:", packetHandler.getTxRxResult(dxl_comm_result))
elif dxl_error != 0:
    print("Dynamixel error:", packetHandler.getRxPacketError(dxl_error))
else:
    print("Present Position:", dxl_position)

# -----------------------------
# Close port
portHandler.closePort()

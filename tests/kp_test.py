from dynamixel_sdk import *
import time
import csv
import math
import numpy as np
import argparse
import sys

# ================= USER CONFIG =================
PORT = "/dev/tty.usbserial-FTAK8D39"
DXL_ID = 1
BAUDRATE = 2000000

KP_VALUE = 1000                 # Register value being tested
CURRENT_LIMIT_mA = 800

LOG_DURATION_SEC = 5           # Reduced since we're doing single angle
SAMPLE_DT = 0.02

# ================ CONTROL TABLE (XC430) =========
ADDR_TORQUE_ENABLE     = 64
ADDR_OPERATING_MODE    = 11
ADDR_CURRENT_LIMIT     = 38
ADDR_P_GAIN            = 84
ADDR_I_GAIN            = 82
ADDR_D_GAIN            = 80
ADDR_GOAL_POSITION     = 116
ADDR_PRESENT_POSITION  = 132
ADDR_PRESENT_CURRENT   = 126

PROTOCOL_VERSION = 2.0

# ================= PARSE ARGUMENTS =================
parser = argparse.ArgumentParser(
    description='Measure Kp by applying controlled perturbations',
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog='''
Examples:
  python kp_test.py 1    # Apply 1 degree perturbation
  python kp_test.py 5    # Apply 5 degree perturbation
  python kp_test.py -3   # Apply -3 degree perturbation (opposite direction)
    '''
)
parser.add_argument(
    'angle',
    type=float,
    help='Perturbation angle in degrees (1-15, or negative for opposite direction)'
)
parser.add_argument(
    '--kp',
    type=int,
    default=KP_VALUE,
    help=f'Kp register value to test (default: {KP_VALUE})'
)
parser.add_argument(
    '--port',
    type=str,
    default=PORT,
    help=f'Serial port (default: {PORT})'
)
parser.add_argument(
    '--id',
    type=int,
    default=DXL_ID,
    help=f'Motor ID (default: {DXL_ID})'
)

args = parser.parse_args()

# Validate angle
if abs(args.angle) < 0.1 or abs(args.angle) > 15:
    print(f"ERROR: Angle must be between 0.1 and 15 degrees (got {args.angle})")
    sys.exit(1)

PERTURBATION_DEG = args.angle
KP_VALUE = args.kp
PORT = args.port
DXL_ID = args.id
STABILITY_WAIT = 0.5             # Wait for motor to stabilize after perturbation

# ================= SETUP SDK ====================
print(f"Opening port: {PORT}")
portHandler = PortHandler(PORT)
packetHandler = PacketHandler(PROTOCOL_VERSION)

if not portHandler.openPort():
    print(f"ERROR: Failed to open port {PORT}")
    print("Check if the port exists and is not in use by another program.")
    exit(1)
print("Port opened successfully.")

if not portHandler.setBaudRate(BAUDRATE):
    print(f"ERROR: Failed to set baudrate {BAUDRATE}")
    portHandler.closePort()
    exit(1)
print(f"Baudrate set to {BAUDRATE}")

# ================= VERIFY MOTOR CONNECTION ==================
print(f"Verifying connection to motor ID {DXL_ID}...")
# Try to read model number (address 0) to verify motor is reachable
ADDR_MODEL_NUMBER = 0
model_number, comm_result, error = packetHandler.read2ByteTxRx(portHandler, DXL_ID, ADDR_MODEL_NUMBER)
if comm_result != COMM_SUCCESS:
    print(f"ERROR: Cannot communicate with motor ID {DXL_ID}")
    print(f"Communication error: {packetHandler.getTxRxResult(comm_result)}")
    print("\nTroubleshooting:")
    print(f"  1. Verify motor ID is {DXL_ID}")
    print(f"  2. Check motor power is ON")
    print(f"  3. Verify physical connection to port {PORT}")
    print(f"  4. Try different baudrate (current: {BAUDRATE})")
    print(f"  5. Check if another program is using the port")
    portHandler.closePort()
    exit(1)
if error != 0:
    print(f"WARNING: Motor error: {packetHandler.getRxPacketError(error)}")
print(f"Motor ID {DXL_ID} found (Model: {model_number})")

# ================= CONFIGURE MOTOR ==============
print("Configuring motor...")
comm_result, error = packetHandler.write1ByteTxRx(portHandler, DXL_ID, ADDR_TORQUE_ENABLE, 0)
if comm_result != COMM_SUCCESS:
    print(f"ERROR: Failed to disable torque: {packetHandler.getTxRxResult(comm_result)}")
    print(f"Check: Port {PORT}, Motor ID {DXL_ID}, Baudrate {BAUDRATE}")
    portHandler.closePort()
    exit(1)

comm_result, error = packetHandler.write1ByteTxRx(portHandler, DXL_ID, ADDR_OPERATING_MODE, 3)
if comm_result != COMM_SUCCESS:
    print(f"ERROR: Failed to set operating mode: {packetHandler.getTxRxResult(comm_result)}")
    portHandler.closePort()
    exit(1)

comm_result, error = packetHandler.write2ByteTxRx(portHandler, DXL_ID, ADDR_D_GAIN, 0)
if comm_result != COMM_SUCCESS:
    print(f"ERROR: Failed to set D gain: {packetHandler.getTxRxResult(comm_result)}")
    portHandler.closePort()
    exit(1)

comm_result, error = packetHandler.write2ByteTxRx(portHandler, DXL_ID, ADDR_I_GAIN, 0)
if comm_result != COMM_SUCCESS:
    print(f"ERROR: Failed to set I gain: {packetHandler.getTxRxResult(comm_result)}")
    portHandler.closePort()
    exit(1)

comm_result, error = packetHandler.write2ByteTxRx(portHandler, DXL_ID, ADDR_P_GAIN, KP_VALUE)
if comm_result != COMM_SUCCESS:
    print(f"ERROR: Failed to set P gain: {packetHandler.getTxRxResult(comm_result)}")
    portHandler.closePort()
    exit(1)

current_limit_raw = int(CURRENT_LIMIT_mA / 2.69)
comm_result, error = packetHandler.write2ByteTxRx(
    portHandler, DXL_ID, ADDR_CURRENT_LIMIT, current_limit_raw
)
if comm_result != COMM_SUCCESS:
    print(f"ERROR: Failed to set current limit: {packetHandler.getTxRxResult(comm_result)}")
    portHandler.closePort()
    exit(1)

# ================= LOCK POSITION =================
print("Reading initial position...")
goal_pos, comm_result, error = packetHandler.read4ByteTxRx(
    portHandler, DXL_ID, ADDR_PRESENT_POSITION
)
if comm_result != COMM_SUCCESS:
    print(f"ERROR: Failed to read position: {packetHandler.getTxRxResult(comm_result)}")
    print(f"Motor may not be responding. Check connection and power.")
    portHandler.closePort()
    exit(1)

# Convert initial position to degrees (flip sign for orientation)
initial_pos_deg = -(goal_pos * 360.0 / 4096.0)

# Read initial current
initial_cur_raw, comm_result, error = packetHandler.read2ByteTxRx(
    portHandler, DXL_ID, ADDR_PRESENT_CURRENT
)
if comm_result != COMM_SUCCESS:
    print(f"WARNING: Failed to read initial current: {packetHandler.getTxRxResult(comm_result)}")
    initial_cur_A = 0.0
else:
    # Convert signed current
    if initial_cur_raw > 32767:
        initial_cur_raw -= 65536
    initial_cur_A = initial_cur_raw * 0.00269  # Convert to Amps

print(f"Initial position: {goal_pos} counts = {initial_pos_deg:+.2f} degrees")
print(f"Initial current: {initial_cur_A:+.4f} A")

# Enable torque and set initial goal position
comm_result, error = packetHandler.write4ByteTxRx(
    portHandler, DXL_ID, ADDR_GOAL_POSITION, goal_pos
)
if comm_result != COMM_SUCCESS:
    print(f"ERROR: Failed to set initial goal position: {packetHandler.getTxRxResult(comm_result)}")
    portHandler.closePort()
    exit(1)

comm_result, error = packetHandler.write1ByteTxRx(portHandler, DXL_ID, ADDR_TORQUE_ENABLE, 1)
if comm_result != COMM_SUCCESS:
    print(f"ERROR: Failed to enable torque: {packetHandler.getTxRxResult(comm_result)}")
    portHandler.closePort()
    exit(1)

print(f"\nMotor locked at initial position.")
print(f"Applying {PERTURBATION_DEG:+.1f} degree perturbation...")

# ================= APPLY PERTURBATION =================
# Convert degrees to encoder counts (4096 counts = 360 degrees)
# Flip sign: positive input angle means moving down (negative in encoder)
perturbation_counts = int(-PERTURBATION_DEG * 4096.0 / 360.0)
perturbed_goal = goal_pos + perturbation_counts

# Set new goal position to create perturbation
comm_result, error = packetHandler.write4ByteTxRx(
    portHandler, DXL_ID, ADDR_GOAL_POSITION, perturbed_goal
)
if comm_result != COMM_SUCCESS:
    print(f"ERROR: Failed to set perturbed goal position: {packetHandler.getTxRxResult(comm_result)}")
    portHandler.closePort()
    exit(1)

# Wait for motor to stabilize at new position
print("Waiting for motor to stabilize...")
time.sleep(STABILITY_WAIT)

# Measure actual stall position and current
stall_pos, stall_pos_comm_result, error = packetHandler.read4ByteTxRx(
    portHandler, DXL_ID, ADDR_PRESENT_POSITION
)
if stall_pos_comm_result != COMM_SUCCESS:
    print(f"WARNING: Failed to read stall position: {packetHandler.getTxRxResult(stall_pos_comm_result)}")
    stall_pos_deg = 0
    stall_cur_A = 0.0
    stall_read_success = False
else:
    stall_read_success = True
    # Convert stall position to degrees (flip sign for orientation)
    stall_pos_deg = -(stall_pos * 360.0 / 4096.0)
    position_error_deg = stall_pos_deg - initial_pos_deg
    
    # Read stall current
    stall_cur_raw, comm_result, error = packetHandler.read2ByteTxRx(
        portHandler, DXL_ID, ADDR_PRESENT_CURRENT
    )
    if comm_result != COMM_SUCCESS:
        print(f"WARNING: Failed to read stall current: {packetHandler.getTxRxResult(comm_result)}")
        stall_cur_A = 0.0
    else:
        # Convert signed current
        if stall_cur_raw > 32767:
            stall_cur_raw -= 65536
        stall_cur_A = stall_cur_raw * 0.00269  # Convert to Amps
    
    print(f"Stall position: {stall_pos} counts = {stall_pos_deg:+.2f} degrees")
    print(f"Position error: {position_error_deg:+.2f} degrees (target: {PERTURBATION_DEG:+.1f} degrees)")
    print(f"Stall current: {stall_cur_A:+.4f} A")

# ================= DATA LOGGING ==================
print(f"Measuring current at {PERTURBATION_DEG:+.1f} degree perturbation...")
data = []
start_time = time.time()
consecutive_failures = 0
max_consecutive_failures = 50

while time.time() - start_time < LOG_DURATION_SEC:
    pos_result, pos_comm_result, pos_error = packetHandler.read4ByteTxRx(
        portHandler, DXL_ID, ADDR_PRESENT_POSITION
    )
    cur_result, cur_comm_result, cur_error = packetHandler.read2ByteTxRx(
        portHandler, DXL_ID, ADDR_PRESENT_CURRENT
    )

    # Check for communication errors
    if pos_comm_result != COMM_SUCCESS or cur_comm_result != COMM_SUCCESS:
        consecutive_failures += 1
        if consecutive_failures >= max_consecutive_failures:
            print(f"\n\nERROR: Too many consecutive communication failures ({consecutive_failures}).")
            print("Motor is not responding. Check:")
            print(f"  - Port: {PORT}")
            print(f"  - Motor ID: {DXL_ID}")
            print(f"  - Motor power and connection")
            print(f"  - Baudrate: {BAUDRATE}")
            portHandler.closePort()
            exit(1)
        time.sleep(SAMPLE_DT)
        continue
    
    # Reset failure counter on success
    consecutive_failures = 0
    
    if pos_error != 0 or cur_error != 0:
        time.sleep(SAMPLE_DT)
        continue

    # Convert signed current
    if cur_result > 32767:
        cur_result -= 65536

    # Calculate actual error from initial position (flip sign for orientation)
    error_deg = -(goal_pos - pos_result) * 360.0 / 4096.0
    current_A = cur_result * 0.00269

    print(
        f"Error: {error_deg:+6.2f} deg | "
        f"Current: {current_A:+.3f} A",
        end="\r",
        flush=True
    )

    # Log all data points
    data.append((error_deg, current_A))

    time.sleep(SAMPLE_DT)

print("\n\nLogging complete.")

# ================= POST-PROCESSING =================
if len(data) < 10:
    print("\nERROR: Insufficient data collected.")
    portHandler.closePort()
    exit(1)

errors = np.array([d[0] for d in data])
currents = np.array([d[1] for d in data])

# Calculate statistics
mean_error = np.mean(errors)
mean_current = np.mean(currents)
std_error = np.std(errors)
std_current = np.std(currents)

# Simple linear fit through origin (I = K * error)
# Since we're at a single angle, we can estimate K = I / error
if abs(mean_error) > 0.01:  # Avoid division by zero
    estimated_kp = mean_current / mean_error
    kp_A_per_deg = estimated_kp
    kp_A_per_rad = kp_A_per_deg * (180.0 / math.pi)
else:
    print("\nWARNING: Error too small, cannot calculate Kp")
    estimated_kp = 0
    kp_A_per_deg = 0
    kp_A_per_rad = 0

print("\n===== Kp MEASUREMENT RESULT =====")
print(f"Commanded Position P Gain (register): {KP_VALUE}")
print(f"Perturbation angle: {PERTURBATION_DEG:+.1f} degrees")
print(f"Number of data points: {len(data)}")
print(f"\n--- Initial State ---")
print(f"Position: {initial_pos_deg:+.2f} degrees")
print(f"Current: {initial_cur_A:+.4f} A")
print(f"\n--- Stall State ---")
if stall_read_success:
    print(f"Position: {stall_pos_deg:+.2f} degrees (error: {stall_pos_deg - initial_pos_deg:+.2f} deg)")
    print(f"Current: {stall_cur_A:+.4f} A")
else:
    print("Stall position/current: Failed to read")
print(f"\n--- Measurement Statistics ---")
print(f"Mean error: {mean_error:+.3f} ± {std_error:.3f} deg")
print(f"Mean current: {mean_current:+.4f} ± {std_current:.4f} A")
print(f"\n--- Kp Estimates ---")
print(f"Kp (current/error): {kp_A_per_deg:.4f} A/deg = {kp_A_per_rad:.4f} A/rad")

# ================= GET WEIGHING SCALE READING =================
print("\n" + "="*50)
print("Please provide weighing scale reading:")
print("Apparatus length: 11.4 cm")
try:
    weight_grams = float(input("Enter mean weight reading (grams): "))
    
    # Calculate torque from weight and lever arm
    # torque = force × distance = (mass × g) × distance
    # mass in grams → convert to kg
    # g = 9.81 m/s²
    # distance = 11.4 cm = 0.114 m
    LEVER_ARM_LENGTH_M = 0.114  # 11.4 cm in meters
    GRAVITY = 9.81  # m/s²
    
    mass_kg = weight_grams / 1000.0
    force_N = mass_kg * GRAVITY
    measured_torque_Nm = force_N * LEVER_ARM_LENGTH_M
    
    print(f"\n--- External Torque Measurement ---")
    print(f"Weight: {weight_grams:.2f} g = {mass_kg:.4f} kg")
    print(f"Force: {force_N:.4f} N")
    print(f"Torque: {measured_torque_Nm:.4f} N⋅m (at {LEVER_ARM_LENGTH_M*100:.1f} cm)")
    
    # Calculate Kp from torque
    if abs(mean_error) > 0.01:
        kp_torque_per_deg = measured_torque_Nm / mean_error
        kp_torque_per_rad = kp_torque_per_deg * (180.0 / math.pi)
        print(f"\nKp (torque/error): {kp_torque_per_deg:.4f} N⋅m/deg = {kp_torque_per_rad:.4f} N⋅m/rad")
    
except ValueError:
    print("WARNING: Invalid input. Skipping torque calculation.")
    measured_torque_Nm = None
except KeyboardInterrupt:
    print("\nWARNING: Input cancelled. Skipping torque calculation.")
    measured_torque_Nm = None

print("\nNOTE: This is the effective stiffness at this angle.")
print("For full characterization, run with multiple angles (1, 2, 3, 4, 5 degrees).")

# ================= SAVE CSV ======================
filename = f"xc430_kp_{KP_VALUE}_angle_{abs(PERTURBATION_DEG):.1f}deg.csv"
with open(filename, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["error_deg", "current_A"])
    writer.writerows(zip(errors, currents))
print(f"\nData saved to: {filename}")

# ================= RESET TO INITIAL POSITION =================
print("\nResetting motor to initial position...")
comm_result, error = packetHandler.write4ByteTxRx(
    portHandler, DXL_ID, ADDR_GOAL_POSITION, goal_pos
)
if comm_result != COMM_SUCCESS:
    print(f"WARNING: Failed to reset position: {packetHandler.getTxRxResult(comm_result)}")
else:
    # Wait for motor to return to initial position
    time.sleep(STABILITY_WAIT)
    # Verify position
    current_pos, comm_result, error = packetHandler.read4ByteTxRx(
        portHandler, DXL_ID, ADDR_PRESENT_POSITION
    )
    if comm_result == COMM_SUCCESS:
        # Flip sign for orientation
        pos_error_deg = -(goal_pos - current_pos) * 360.0 / 4096.0
        current_pos_deg = -(current_pos * 360.0 / 4096.0)
        if abs(pos_error_deg) < 1.0:  # Within 1 degree
            print(f"Motor reset to initial position: {current_pos_deg:+.2f} deg (error: {pos_error_deg:+.2f} deg)")
        else:
            print(f"WARNING: Motor position: {current_pos_deg:+.2f} deg, error: {pos_error_deg:+.2f} deg")

# ================= CLEANUP =======================
print("Disabling torque...")
packetHandler.write1ByteTxRx(portHandler, DXL_ID, ADDR_TORQUE_ENABLE, 0)
portHandler.closePort()
print("Port closed.")

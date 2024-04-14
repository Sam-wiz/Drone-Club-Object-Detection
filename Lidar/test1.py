import pyserial
ser = serial.Serial('/dev/serial0', 115200, timeout=1)  # Open serial port

try:
    while True:
        line = ser.readline().strip().decode('utf-8')
        if line:
            print("Distance:", line)  # Print the received data
except KeyboardInterrupt:
    print("Exiting program.")
    ser.close()  # Close serial port on Ctrl+C


"""
ESP32-S3 Firmware Template
Arduino/PlatformIO code for the MCU side
"""

# This file contains the ESP32-S3 firmware as a string constant
# It can be compiled and uploaded using PlatformIO

ESP32_FIRMWARE_INO = '''
/*
 * AegisAI ESP32-S3 Firmware
 * Handles sensors, actuators, and communication with Raspberry Pi
 */

#include <Arduino.h>
#include <Wire.h>
#include <MPU6050.h>
#include <VL53L0X.h>
#include <INA219.h>

// Communication Protocol
#define SYNC1 0xAE
#define SYNC2 0x5A
#define MAX_PAYLOAD 250

// Packet Types
#define PKT_COMMAND 0x01
#define PKT_SENSOR_DATA 0x02
#define PKT_ACK 0x03
#define PKT_ERROR 0x04

// Sensor Types
#define SENSOR_IMU 0x10
#define SENSOR_DISTANCE 0x11
#define SENSOR_CURRENT 0x12
#define SENSOR_BATTERY 0x17

// Pin Definitions (ESP32-S3)
#define MOTOR1_PWM 4
#define MOTOR2_PWM 5
#define MOTOR3_PWM 6
#define MOTOR4_PWM 7
#define MOTOR1_DIR 15
#define MOTOR2_DIR 16
#define MOTOR3_DIR 17
#define MOTOR4_DIR 18

#define SERVO1_PIN 8
#define SERVO2_PIN 9

#define I2C_SDA 21
#define I2C_SCL 22

#define LED_PIN 48  // Built-in RGB LED on ESP32-S3

// PWM Configuration
#define PWM_FREQ 5000
#define PWM_RESOLUTION 8

// Sensors
MPU6050 mpu;
VL53L0X distanceSensor;
INA219 currentSensor;

// State
bool sensorsInitialized = false;
uint32_t lastSensorRead = 0;
uint32_t sensorInterval = 10;  // 100Hz

// Buffer for sending data
uint8_t txBuffer[MAX_PAYLOAD + 6];

// Motor speeds
int16_t motorSpeeds[4] = {0, 0, 0, 0};

// Function declarations
void initSensors();
void readSensors();
void processCommand(uint8_t cmd, uint8_t* payload, uint8_t len);
void sendSensorPacket(uint8_t sensorType, uint8_t* data, uint8_t dataLen);
uint16_t calculateCRC16(uint8_t* data, uint8_t len);
void setMotorSpeed(int motor, int16_t speed);

void setup() {
  // Initialize serial communication
  Serial.begin(115200);
  while (!Serial && millis() < 3000);
  
  // Initialize I2C
  Wire.begin(I2C_SDA, I2C_SCL);
  Wire.setClock(400000);  // 400kHz
  
  // Initialize LED
  pinMode(LED_PIN, OUTPUT);
  
  // Initialize motor PWM channels
  ledcSetup(0, PWM_FREQ, PWM_RESOLUTION);
  ledcSetup(1, PWM_FREQ, PWM_RESOLUTION);
  ledcSetup(2, PWM_FREQ, PWM_RESOLUTION);
  ledcSetup(3, PWM_FREQ, PWM_RESOLUTION);
  
  ledcAttachPin(MOTOR1_PWM, 0);
  ledcAttachPin(MOTOR2_PWM, 1);
  ledcAttachPin(MOTOR3_PWM, 2);
  ledcAttachPin(MOTOR4_PWM, 3);
  
  // Motor direction pins
  pinMode(MOTOR1_DIR, OUTPUT);
  pinMode(MOTOR2_DIR, OUTPUT);
  pinMode(MOTOR3_DIR, OUTPUT);
  pinMode(MOTOR4_DIR, OUTPUT);
  
  // Initialize sensors
  initSensors();
  
  Serial.println("AegisAI ESP32-S3 Ready");
}

void loop() {
  // Read incoming commands
  static uint8_t rxBuffer[MAX_PAYLOAD + 6];
  static int rxIdx = 0;
  static int expectedLen = 0;
  
  while (Serial.available()) {
    uint8_t b = Serial.read();
    
    // Looking for sync
    if (rxIdx == 0 && b != SYNC1) continue;
    if (rxIdx == 1 && b != SYNC2) { rxIdx = 0; continue; }
    
    rxBuffer[rxIdx++] = b;
    
    // Got header, know expected length
    if (rxIdx == 4) {
      expectedLen = 4 + rxBuffer[3] + 2;  // header + payload + crc
    }
    
    // Complete packet received
    if (rxIdx >= 6 && rxIdx == expectedLen) {
      // Verify CRC
      uint16_t rxCRC = (rxBuffer[expectedLen-2] << 8) | rxBuffer[expectedLen-1];
      uint16_t calcCRC = calculateCRC16(&rxBuffer[4], rxBuffer[3]);
      
      if (rxCRC == calcCRC) {
        uint8_t pktType = rxBuffer[2];
        uint8_t cmd = rxBuffer[4];
        uint8_t payloadLen = rxBuffer[3] - 1;
        
        if (pktType == PKT_COMMAND) {
          processCommand(cmd, &rxBuffer[5], payloadLen);
        }
      }
      
      rxIdx = 0;
      expectedLen = 0;
    }
    
    if (rxIdx >= sizeof(rxBuffer)) rxIdx = 0;
  }
  
  // Read and send sensor data
  if (millis() - lastSensorRead >= sensorInterval) {
    lastSensorRead = millis();
    readSensors();
  }
}

void initSensors() {
  // Initialize MPU6050
  mpu.initialize();
  if (mpu.testConnection()) {
    mpu.setFullScaleAccelRange(MPU6050_ACCEL_FS_4);
    mpu.setFullScaleGyroRange(MPU6050_GYRO_FS_500);
    Serial.println("MPU6050 OK");
  } else {
    Serial.println("MPU6050 FAIL");
  }
  
  // Initialize VL53L0X
  if (distanceSensor.init()) {
    distanceSensor.setTimeout(500);
    distanceSensor.startContinuous();
    Serial.println("VL53L0X OK");
  } else {
    Serial.println("VL53L0X FAIL");
  }
  
  // Initialize INA219
  if (currentSensor.begin()) {
    currentSensor.setCalibration_32V_2A();
    Serial.println("INA219 OK");
  } else {
    Serial.println("INA219 FAIL");
  }
  
  sensorsInitialized = true;
}

void readSensors() {
  if (!sensorsInitialized) return;
  
  // Read IMU
  int16_t ax, ay, az, gx, gy, gz;
  mpu.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);
  
  // Convert to floats (scale to g and deg/s)
  float imuData[6];
  imuData[0] = ax / 8192.0;  // 4g range
  imuData[1] = ay / 8192.0;
  imuData[2] = az / 8192.0;
  imuData[3] = gx / 65.5;    // 500 deg/s range
  imuData[4] = gy / 65.5;
  imuData[5] = gz / 65.5;
  
  sendSensorPacket(SENSOR_IMU, (uint8_t*)imuData, 24);
  
  // Read distance
  uint16_t distance = distanceSensor.readRangeContinuousMillimeters();
  if (!distanceSensor.timeoutOccurred()) {
    uint8_t distData[2];
    distData[0] = (distance >> 8) & 0xFF;
    distData[1] = distance & 0xFF;
    sendSensorPacket(SENSOR_DISTANCE, distData, 2);
  }
  
  // Read current/voltage
  float current_mA = currentSensor.getCurrent_mA();
  float voltage_V = currentSensor.getBusVoltage_V();
  int16_t currentData[2];
  currentData[0] = (int16_t)current_mA;
  currentData[1] = (int16_t)(voltage_V * 1000);
  sendSensorPacket(SENSOR_CURRENT, (uint8_t*)currentData, 4);
}

void processCommand(uint8_t cmd, uint8_t* payload, uint8_t len) {
  switch (cmd) {
    case 0x01:  // Init
      Serial.println("Init received");
      sendAck(0x01);
      break;
      
    case 0x02:  // Stop
      setMotorSpeed(0, 0);
      setMotorSpeed(1, 0);
      setMotorSpeed(2, 0);
      setMotorSpeed(3, 0);
      sendAck(0x02);
      break;
      
    case 0x04:  // Calibrate
      mpu.CalibrateAccel(6);
      mpu.CalibrateGyro(6);
      Serial.println("CAL_OK");
      sendAck(0x04);
      break;
      
    case 0x10:  // Actuator command
      if (len >= 3) {
        uint8_t actType = payload[0];
        uint8_t numValues = payload[1];
        
        if (actType == 0x20 && numValues <= 4) {  // Motors
          for (int i = 0; i < numValues; i++) {
            int16_t speed = (payload[2 + i*2] << 8) | payload[3 + i*2];
            setMotorSpeed(i, speed);
          }
        }
      }
      sendAck(0x10);
      break;
      
    default:
      sendAck(cmd);
      break;
  }
}

void setMotorSpeed(int motor, int16_t speed) {
  if (motor < 0 || motor > 3) return;
  
  motorSpeeds[motor] = speed;
  
  // Set direction
  uint8_t dirPin;
  switch (motor) {
    case 0: dirPin = MOTOR1_DIR; break;
    case 1: dirPin = MOTOR2_DIR; break;
    case 2: dirPin = MOTOR3_DIR; break;
    case 3: dirPin = MOTOR4_DIR; break;
    default: return;
  }
  
  digitalWrite(dirPin, speed >= 0 ? HIGH : LOW);
  
  // Set PWM (0-255)
  uint8_t pwm = map(abs(speed), 0, 255, 0, 255);
  ledcWrite(motor, pwm);
}

void sendSensorPacket(uint8_t sensorType, uint8_t* data, uint8_t dataLen) {
  // Build payload: sensor_type + num_values + data
  uint8_t payload[MAX_PAYLOAD];
  payload[0] = sensorType;
  payload[1] = dataLen;
  memcpy(&payload[2], data, dataLen);
  uint8_t payloadLen = dataLen + 2;
  
  // Build packet
  txBuffer[0] = SYNC1;
  txBuffer[1] = SYNC2;
  txBuffer[2] = PKT_SENSOR_DATA;
  txBuffer[3] = payloadLen;
  memcpy(&txBuffer[4], payload, payloadLen);
  
  // Calculate and append CRC
  uint16_t crc = calculateCRC16(payload, payloadLen);
  txBuffer[4 + payloadLen] = (crc >> 8) & 0xFF;
  txBuffer[5 + payloadLen] = crc & 0xFF;
  
  // Send
  Serial.write(txBuffer, 6 + payloadLen);
}

void sendAck(uint8_t cmd) {
  uint8_t payload[1] = {cmd};
  
  txBuffer[0] = SYNC1;
  txBuffer[1] = SYNC2;
  txBuffer[2] = PKT_ACK;
  txBuffer[3] = 1;
  txBuffer[4] = cmd;
  
  uint16_t crc = calculateCRC16(payload, 1);
  txBuffer[5] = (crc >> 8) & 0xFF;
  txBuffer[6] = crc & 0xFF;
  
  Serial.write(txBuffer, 7);
}

uint16_t calculateCRC16(uint8_t* data, uint8_t len) {
  uint16_t crc = 0xFFFF;
  
  for (int i = 0; i < len; i++) {
    crc ^= (uint16_t)data[i] << 8;
    for (int j = 0; j < 8; j++) {
      if (crc & 0x8000) {
        crc = (crc << 1) ^ 0x1021;
      } else {
        crc <<= 1;
      }
    }
  }
  
  return crc;
}
'''

PLATFORMIO_INI = '''
; PlatformIO Configuration for AegisAI ESP32-S3 Firmware

[env:esp32-s3-devkitc-1]
platform = espressif32
board = esp32-s3-devkitc-1
framework = arduino

; Upload settings
upload_speed = 921600
monitor_speed = 115200

; Build flags
build_flags = 
    -DARDUINO_USB_CDC_ON_BOOT=1
    -DARDUINO_USB_MODE=1
    -DCORE_DEBUG_LEVEL=0

; Library dependencies
lib_deps =
    electroniccats/MPU6050@^1.0.0
    pololu/VL53L0X@^1.3.1
    adafruit/Adafruit INA219@^1.2.1
    Wire

; Memory settings
board_build.partitions = default.csv
board_upload.flash_size = 16MB
'''


class ESP32FirmwareManager:
    """Manage ESP32 firmware generation and upload."""
    
    def __init__(self):
        self.firmware_code = ESP32_FIRMWARE_INO
        self.platformio_ini = PLATFORMIO_INI
    
    def generate_firmware_files(self, output_dir: str):
        """
        Generate firmware files for PlatformIO project.
        
        Args:
            output_dir: Directory to create PlatformIO project
        """
        from pathlib import Path
        
        output_path = Path(output_dir)
        
        # Create directory structure
        (output_path / "src").mkdir(parents=True, exist_ok=True)
        (output_path / "include").mkdir(exist_ok=True)
        (output_path / "lib").mkdir(exist_ok=True)
        
        # Write main.cpp
        with open(output_path / "src" / "main.cpp", 'w') as f:
            f.write(self.firmware_code)
        
        # Write platformio.ini
        with open(output_path / "platformio.ini", 'w') as f:
            f.write(self.platformio_ini)
        
        print(f"ESP32 firmware project created at {output_path}")
        print("To compile and upload:")
        print(f"  cd {output_path}")
        print("  pio run -t upload")
    
    def get_firmware_code(self) -> str:
        """Get the firmware source code."""
        return self.firmware_code

#include <Wire.h>
#include <EEPROM.h>
#include <Adafruit_PWMServoDriver.h>

struct HeadServoBinding {
  const char* key;
  uint8_t channel;
  int minDeg;
  int maxDeg;
  uint16_t minPulse;
  uint16_t maxPulse;
  int currentDeg;
};

Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver(0x40);

const HeadServoBinding DEFAULT_HEAD_SERVOS[] = {
  {"HY", 0, 35, 145, 110, 510, 90},
  {"HP", 1, 55, 125, 120, 500, 90},
  {"NR", 2, 65, 115, 135, 475, 90},
  {"JW", 3, 10, 55, 130, 320, 20},
  {"EX", 4, 60, 120, 165, 430, 90},
  {"EY", 5, 70, 110, 180, 410, 90},
  {"LU", 6, 70, 120, 180, 430, 92},
  {"LL", 7, 60, 110, 170, 410, 88},
  {"RU", 8, 60, 110, 170, 410, 88},
  {"RL", 9, 70, 120, 180, 430, 92},
};

HeadServoBinding headServos[] = {
  {"HY", 0, 35, 145, 110, 510, 90},
  {"HP", 1, 55, 125, 120, 500, 90},
  {"NR", 2, 65, 115, 135, 475, 90},
  {"JW", 3, 10, 55, 130, 320, 20},
  {"EX", 4, 60, 120, 165, 430, 90},
  {"EY", 5, 70, 110, 180, 410, 90},
  {"LU", 6, 70, 120, 180, 430, 92},
  {"LL", 7, 60, 110, 170, 410, 88},
  {"RU", 8, 60, 110, 170, 410, 88},
  {"RL", 9, 70, 120, 180, 430, 92},
};

static const size_t HEAD_SERVO_COUNT = sizeof(headServos) / sizeof(headServos[0]);
static const uint32_t CONFIG_MAGIC = 0x41494254UL;
static const uint16_t CONFIG_VERSION = 1;
String serialBuffer;
unsigned long lastSeq = 0;

struct StoredServoConfig {
  uint8_t channel;
  int16_t minDeg;
  int16_t maxDeg;
  uint16_t minPulse;
  uint16_t maxPulse;
  int16_t currentDeg;
};

struct PersistedHeadConfig {
  uint32_t magic;
  uint16_t version;
  uint16_t count;
  StoredServoConfig servos[HEAD_SERVO_COUNT];
};

int clampDeg(int value, int minDeg, int maxDeg) {
  if (value < minDeg) return minDeg;
  if (value > maxDeg) return maxDeg;
  return value;
}

void sanitizeBinding(HeadServoBinding& binding) {
  if (binding.maxDeg <= binding.minDeg) {
    binding.maxDeg = binding.minDeg + 1;
  }
  if (binding.minPulse < 50) {
    binding.minPulse = 50;
  }
  if (binding.maxPulse <= binding.minPulse) {
    binding.maxPulse = binding.minPulse + 1;
  }
  binding.currentDeg = clampDeg(binding.currentDeg, binding.minDeg, binding.maxDeg);
}

void loadDefaultConfig() {
  for (size_t i = 0; i < HEAD_SERVO_COUNT; ++i) {
    headServos[i].channel = DEFAULT_HEAD_SERVOS[i].channel;
    headServos[i].minDeg = DEFAULT_HEAD_SERVOS[i].minDeg;
    headServos[i].maxDeg = DEFAULT_HEAD_SERVOS[i].maxDeg;
    headServos[i].minPulse = DEFAULT_HEAD_SERVOS[i].minPulse;
    headServos[i].maxPulse = DEFAULT_HEAD_SERVOS[i].maxPulse;
    headServos[i].currentDeg = DEFAULT_HEAD_SERVOS[i].currentDeg;
    sanitizeBinding(headServos[i]);
  }
}

bool loadPersistedConfig() {
  PersistedHeadConfig persisted;
  EEPROM.get(0, persisted);
  if (persisted.magic != CONFIG_MAGIC || persisted.version != CONFIG_VERSION || persisted.count != HEAD_SERVO_COUNT) {
    return false;
  }
  for (size_t i = 0; i < HEAD_SERVO_COUNT; ++i) {
    headServos[i].channel = persisted.servos[i].channel;
    headServos[i].minDeg = persisted.servos[i].minDeg;
    headServos[i].maxDeg = persisted.servos[i].maxDeg;
    headServos[i].minPulse = persisted.servos[i].minPulse;
    headServos[i].maxPulse = persisted.servos[i].maxPulse;
    headServos[i].currentDeg = persisted.servos[i].currentDeg;
    sanitizeBinding(headServos[i]);
  }
  return true;
}

void savePersistedConfig() {
  PersistedHeadConfig persisted;
  persisted.magic = CONFIG_MAGIC;
  persisted.version = CONFIG_VERSION;
  persisted.count = HEAD_SERVO_COUNT;
  for (size_t i = 0; i < HEAD_SERVO_COUNT; ++i) {
    persisted.servos[i].channel = headServos[i].channel;
    persisted.servos[i].minDeg = headServos[i].minDeg;
    persisted.servos[i].maxDeg = headServos[i].maxDeg;
    persisted.servos[i].minPulse = headServos[i].minPulse;
    persisted.servos[i].maxPulse = headServos[i].maxPulse;
    persisted.servos[i].currentDeg = headServos[i].currentDeg;
  }
  EEPROM.put(0, persisted);
}

uint8_t parseHexByte(const String& hex) {
  return (uint8_t) strtoul(hex.c_str(), nullptr, 16);
}

uint8_t computeChecksum(const String& payload) {
  uint8_t checksum = 0;
  for (size_t i = 0; i < payload.length(); ++i) {
    checksum ^= (uint8_t) payload[i];
  }
  return checksum;
}

uint16_t degToPulse(const HeadServoBinding& binding, int deg) {
  long pulse = map(deg, binding.minDeg, binding.maxDeg, binding.minPulse, binding.maxPulse);
  if (pulse < binding.minPulse) pulse = binding.minPulse;
  if (pulse > binding.maxPulse) pulse = binding.maxPulse;
  return (uint16_t)pulse;
}

void applyServo(const HeadServoBinding& binding) {
  pwm.setPWM(binding.channel, 0, degToPulse(binding, binding.currentDeg));
}

void applyField(const String& key, const String& value) {
  const int deg = value.toInt();
  for (size_t i = 0; i < HEAD_SERVO_COUNT; ++i) {
    if (key.equals(headServos[i].key)) {
      headServos[i].currentDeg = clampDeg(deg, headServos[i].minDeg, headServos[i].maxDeg);
      applyServo(headServos[i]);
      return;
    }
  }
}

bool applyConfigField(const String& key, const String& value) {
  for (size_t i = 0; i < HEAD_SERVO_COUNT; ++i) {
    String prefix = String(headServos[i].key) + "_";
    if (!key.startsWith(prefix)) {
      continue;
    }
    String suffix = key.substring(prefix.length());
    if (suffix.equals("CH")) {
      headServos[i].channel = (uint8_t) constrain(value.toInt(), 0, 15);
    } else if (suffix.equals("MIND")) {
      headServos[i].minDeg = value.toInt();
    } else if (suffix.equals("MAXD")) {
      headServos[i].maxDeg = value.toInt();
    } else if (suffix.equals("MINP")) {
      headServos[i].minPulse = (uint16_t) max(50, value.toInt());
    } else if (suffix.equals("MAXP")) {
      headServos[i].maxPulse = (uint16_t) max((int) headServos[i].minPulse + 1, value.toInt());
    } else {
      return false;
    }
    sanitizeBinding(headServos[i]);
    applyServo(headServos[i]);
    return true;
  }
  return false;
}

bool parsePayload(const String& payload, bool configMode) {
  bool profileOk = false;
  bool configChanged = false;
  bool saveRequested = false;
  bool resetRequested = false;
  int start = 0;
  unsigned long seq = lastSeq;

  while (start < (int)payload.length()) {
    int commaIdx = payload.indexOf(',', start);
    if (commaIdx < 0) {
      commaIdx = payload.length();
    }
    const String token = payload.substring(start, commaIdx);
    const int eqIdx = token.indexOf('=');
    if (eqIdx > 0) {
      const String key = token.substring(0, eqIdx);
      const String value = token.substring(eqIdx + 1);
      if (key.equals("SEQ")) {
        seq = (unsigned long)value.toInt();
      } else if (key.equals("PROFILE")) {
        profileOk = value.equals("HEAD_PCA9685");
      } else if (configMode && key.equals("SAVE")) {
        saveRequested = value.toInt() != 0;
      } else if (configMode && key.equals("RESET")) {
        resetRequested = value.toInt() != 0;
      }
    }
    start = commaIdx + 1;
  }

  if (!profileOk) {
    Serial.println("ERR profile");
    return false;
  }

  if (configMode && resetRequested) {
    loadDefaultConfig();
    configChanged = true;
  }

  start = 0;
  while (start < (int)payload.length()) {
    int commaIdx = payload.indexOf(',', start);
    if (commaIdx < 0) {
      commaIdx = payload.length();
    }
    const String token = payload.substring(start, commaIdx);
    const int eqIdx = token.indexOf('=');
    if (eqIdx > 0) {
      const String key = token.substring(0, eqIdx);
      const String value = token.substring(eqIdx + 1);
      if (key.equals("SEQ")) {
        seq = (unsigned long)value.toInt();
      } else if (!key.equals("PROFILE") && !key.equals("T") && !key.equals("SAVE") && !key.equals("RESET")) {
        if (configMode) {
          if (applyConfigField(key, value)) {
            configChanged = true;
          }
        } else {
          applyField(key, value);
        }
      }
    }
    start = commaIdx + 1;
  }

  if (configMode && (configChanged || saveRequested)) {
    savePersistedConfig();
  }

  lastSeq = seq;
  Serial.print(configMode ? "OK cfg seq=" : "OK head seq=");
  Serial.println(lastSeq);
  return true;
}

bool parseFrame(const String& frame) {
  const bool isMotionFrame = frame.startsWith("$AIBOT,");
  const bool isConfigFrame = frame.startsWith("$AIBOTCFG,");
  if (!isMotionFrame && !isConfigFrame) {
    return false;
  }
  const int starIdx = frame.indexOf('*');
  if (starIdx < 0) {
    return false;
  }

  const String payload = frame.substring(1, starIdx);
  const String checksumHex = frame.substring(starIdx + 1);
  if (checksumHex.length() < 2) {
    return false;
  }
  const uint8_t expected = parseHexByte(checksumHex.substring(0, 2));
  const uint8_t actual = computeChecksum(payload);
  if (expected != actual) {
    Serial.print("ERR checksum expected=");
    Serial.print(expected, HEX);
    Serial.print(" actual=");
    Serial.println(actual, HEX);
    return false;
  }
  return parsePayload(payload, isConfigFrame);
}

void setup() {
  Serial.begin(115200);
  serialBuffer.reserve(512);
  loadDefaultConfig();
  bool loadedFromEeprom = loadPersistedConfig();
  pwm.begin();
  pwm.setPWMFreq(50);
  delay(10);

  for (size_t i = 0; i < HEAD_SERVO_COUNT; ++i) {
    applyServo(headServos[i]);
  }

  Serial.println(loadedFromEeprom ? "AIBOT HEAD PCA9685 receiver ready cfg=eeprom" : "AIBOT HEAD PCA9685 receiver ready cfg=defaults");
}

void loop() {
  while (Serial.available() > 0) {
    char ch = (char)Serial.read();
    if (ch == '\r') {
      continue;
    }
    if (ch == '\n') {
      if (serialBuffer.length() > 0) {
        parseFrame(serialBuffer);
        serialBuffer = "";
      }
      continue;
    }
    if (serialBuffer.length() < 511) {
      serialBuffer += ch;
    } else {
      serialBuffer = "";
      Serial.println("ERR frame_too_long");
    }
  }
}

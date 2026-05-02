#include <Servo.h>

struct ServoBinding {
  const char* key;
  uint8_t pin;
  int minDeg;
  int maxDeg;
  int currentDeg;
  Servo servo;
};

ServoBinding bindings[] = {
  {"HY", 2, 35, 145, 90},
  {"HP", 3, 55, 125, 90},
  {"NR", 4, 65, 115, 90},
  {"JW", 5, 10, 55, 20},
  {"LO", 6, 0, 80, 15},
  {"RO", 7, 0, 80, 15},
  {"LS", 8, 0, 85, 20},
  {"RS", 9, 0, 85, 20},
  {"LE", 10, 5, 95, 30},
  {"RE", 11, 5, 95, 30},
  {"LW", 12, 20, 160, 90},
  {"RW", 13, 20, 160, 90},
  {"LT", 22, 0, 180, 20},
  {"RT", 23, 0, 180, 20},
  {"LI", 24, 0, 180, 20},
  {"RI", 25, 0, 180, 20},
};

static const size_t BINDING_COUNT = sizeof(bindings) / sizeof(bindings[0]);
String serialBuffer;
unsigned long lastSeq = 0;

int clampDeg(int value, int minDeg, int maxDeg) {
  if (value < minDeg) return minDeg;
  if (value > maxDeg) return maxDeg;
  return value;
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

void applyField(const String& key, const String& value) {
  const int deg = value.toInt();
  for (size_t i = 0; i < BINDING_COUNT; ++i) {
    if (key.equals(bindings[i].key)) {
      bindings[i].currentDeg = clampDeg(deg, bindings[i].minDeg, bindings[i].maxDeg);
      bindings[i].servo.write(bindings[i].currentDeg);
      return;
    }
  }
}

bool parseFrame(const String& frame) {
  if (!frame.startsWith("$AIBOT,")) {
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
        seq = (unsigned long) value.toInt();
      } else if (!key.equals("T")) {
        applyField(key, value);
      }
    }
    start = commaIdx + 1;
  }

  lastSeq = seq;
  Serial.print("OK seq=");
  Serial.println(lastSeq);
  return true;
}

void setup() {
  Serial.begin(115200);
  serialBuffer.reserve(256);
  for (size_t i = 0; i < BINDING_COUNT; ++i) {
    bindings[i].servo.attach(bindings[i].pin);
    bindings[i].servo.write(bindings[i].currentDeg);
  }
  Serial.println("AIBOT receiver ready");
}

void loop() {
  while (Serial.available() > 0) {
    char ch = (char) Serial.read();
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
    if (serialBuffer.length() < 255) {
      serialBuffer += ch;
    } else {
      serialBuffer = "";
      Serial.println("ERR frame_too_long");
    }
  }
}

/*
 * signglass OLED 브리지 펌웨어
 * ---------------------------------------------------------------
 * PC ──USB 시리얼──▶ 이 보드 ──SPI──▶ SSD1306/1309 OLED (128x64)
 *
 * PC 에는 I2C/SPI 포트가 없으므로 이 브리지가 반드시 필요합니다.
 * 프로토콜: 0xA5 0x5A + 1024바이트 프레임버퍼 (SSD1306 페이지 포맷)
 *
 * 라이브러리: Arduino IDE > 라이브러리 매니저에서
 *   - Adafruit SSD1306
 *   - Adafruit GFX Library
 *
 * 배선 (SPI, 7핀 OLED 모듈 기준)
 *   OLED        Uno/Nano      Pico(Arduino core)
 *   GND         GND           GND
 *   VCC         3.3V          3V3
 *   SCL/D0(CLK) 13            GP18
 *   SDA/D1(MOSI)11            GP19
 *   RES         9             GP20
 *   DC          8             GP21
 *   CS          10            GP17
 *
 * ※ 대부분의 OLED 모듈은 3.3V 로직입니다. Uno/Nano(5V) 를 쓸 때
 *   모듈에 레귤레이터/레벨시프터가 없으면 3.3V 보드(Pico)를 쓰세요.
 *
 * ※ 보드 선택 팁
 *   - Nano/Uno(16MHz): 921600bps 가 불안정하면 아래 BAUD 를 500000 으로 낮추세요.
 *   - Pico/ESP32: USB CDC 라 보율과 무관하게 빠릅니다. 이쪽을 권장합니다.
 */

#include <SPI.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

#define OLED_W    128
#define OLED_H    64
#define FRAME_SZ  (OLED_W * OLED_H / 8)   // 1024

#define PIN_MOSI  11
#define PIN_CLK   13
#define PIN_DC     8
#define PIN_RST    9
#define PIN_CS    10

#define BAUD      921600
#define TIMEOUT_MS 200

Adafruit_SSD1306 display(OLED_W, OLED_H, &SPI, PIN_DC, PIN_RST, PIN_CS);

/* ★ 수신용 버퍼를 따로 두지 않고 라이브러리의 프레임버퍼에 바로 받습니다.
 *
 *   Adafruit_SSD1306 이 이미 1024바이트를 잡고 있는데 여기에 uint8_t
 *   frame[1024] 를 더 두면 합계 2048바이트입니다. ATmega328P(Uno/Nano)의
 *   SRAM 이 정확히 2048바이트라 스택이 남지 않아, 컴파일은 되지만 실행하면
 *   조용히 리셋되거나 화면이 깨집니다. 위 배선표에 Uno/Nano 가 적혀 있으므로
 *   실제로 밟게 되는 문제입니다.
 *
 *   바로 받으면 메모리가 절반이 되고 memcpy 1024바이트도 사라집니다.
 *   중간에 끊긴 프레임은 버퍼가 반만 바뀐 채로 남지만, display() 를
 *   호출하지 않으므로 화면에는 나타나지 않고 다음 정상 프레임이 통째로
 *   덮어씁니다.
 */

void setup() {
  Serial.begin(BAUD);
  Serial.setTimeout(TIMEOUT_MS);   // 기본값 1000ms 면 아래 재동기화가 늦습니다

  if (!display.begin(SSD1306_SWITCHCAPVCC)) {
    // 부팅 실패 시 LED 로 알림
    pinMode(LED_BUILTIN, OUTPUT);
    while (true) { digitalWrite(LED_BUILTIN, !digitalRead(LED_BUILTIN)); delay(200); }
  }

  display.clearDisplay();
  display.setTextSize(1);
  display.setTextColor(SSD1306_WHITE);
  display.setCursor(4, 28);
  display.println(F("signglass ready"));
  display.display();
}

/* 헤더 0xA5 0x5A 를 찾을 때까지 버림.
   이게 없으면 한 바이트만 밀려도 이후 모든 프레임이 줄무늬로 깨집니다. */
bool waitHeader() {
  static uint8_t prev = 0;
  while (Serial.available()) {
    uint8_t b = Serial.read();
    if (prev == 0xA5 && b == 0x5A) { prev = 0; return true; }
    prev = b;
  }
  return false;
}

bool readFrame() {
  uint8_t *buf = display.getBuffer();
  uint16_t got = 0;
  unsigned long t0 = millis();
  while (got < FRAME_SZ) {
    if (Serial.available()) {
      got += Serial.readBytes((char*)buf + got, FRAME_SZ - got);
      t0 = millis();
    } else if (millis() - t0 > TIMEOUT_MS) {
      return false;   // 끊긴 프레임은 버리고 재동기화 (화면에는 안 나감)
    }
  }
  return true;
}

void loop() {
  if (!waitHeader()) return;
  if (!readFrame())  return;
  display.display();
}

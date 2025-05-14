const int ADC_PIN = 34;              // ADC1_CHANNEL_6
const int TARGET_INTERVAL_US = 100;  // 100 µs = 10 kHz

void setup() {
  Serial.begin(500000);              // Match Python
  while (!Serial);                   // Optional: wait for connection
}

void loop() {
  static unsigned long last_sample_time = 0;
  unsigned long now = micros();

  if (now - last_sample_time >= TARGET_INTERVAL_US) {
    last_sample_time = now;

    int adc_val = analogRead(ADC_PIN);      // 0–4095
    Serial.print("ADC,");
    Serial.print(now);
    Serial.print(",");
    Serial.println(adc_val);
  }
}

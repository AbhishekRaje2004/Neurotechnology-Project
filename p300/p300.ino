// Arduino Code to send EEG data
#define EKG A0  // Analog input for EEG signal

void setup() {
  Serial.begin(115200);  // Start serial communication
}

void loop() {
  int eeg_signal = analogRead(EKG);  // Read EEG signal from the analog pin
  Serial.println(eeg_signal);  // Send the signal to Python over serial
  delay(1);  // Delay to adjust the sampling rate
}

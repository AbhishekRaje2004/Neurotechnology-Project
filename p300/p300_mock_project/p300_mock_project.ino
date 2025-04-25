#define EEG_PIN A0

// Variables for EEG acquisition
unsigned long sampleTime = 0; // Time for next sample
unsigned long startTime = 0;  // Reference start time 
const int samplingRate = 250; // 250 Hz
const int samplingInterval = 4; // 1000/250 = 4ms per sample
int triggerValue = -1;

// Buffer for averaging to reduce noise
const int bufferSize = 4;
int eegBuffer[bufferSize];
int bufferIndex = 0;

void setup() {
  Serial.begin(115200);
  pinMode(EEG_PIN, INPUT);
  
  // Initialize the buffer
  for (int i = 0; i < bufferSize; i++) {
    eegBuffer[i] = 0;
  }
  
  // Wait for serial connection
  while (!Serial);
  
  // Set start time reference
  startTime = millis();
  sampleTime = startTime;
  
  // Send initialization message
  Serial.println("EEG,Arduino_Init,250Hz");
}

void loop() {
  // Check for incoming triggers from PC
  if (Serial.available() > 0) {
    String input = Serial.readStringUntil('\n');
    triggerValue = input.toInt();
  }
  
  // Check if it's time to take a sample
  unsigned long currentTime = millis();
  
  if (currentTime >= sampleTime) {
    // Read EEG value and add to buffer for smoothing
    int eeg_signal = analogRead(EEG_PIN);
    eegBuffer[bufferIndex] = eeg_signal;
    bufferIndex = (bufferIndex + 1) % bufferSize;
    
    // Calculate average (simple moving average)
    long sum = 0;
    for (int i = 0; i < bufferSize; i++) {
      sum += eegBuffer[i];
    }
    int eeg_avg = sum / bufferSize;
    
    // Format output as CSV: timestamp,eeg_value,trigger_value
    Serial.print(currentTime);
    Serial.print(",");
    Serial.print(eeg_avg);
    Serial.print(",");
    Serial.println(triggerValue);
    
    // Reset trigger after sending it once
    if (triggerValue >= 0) {
      triggerValue = -1;
    }
    
    // Calculate next sample time based on the original start time to prevent drift
    // This is more accurate than just adding samplingInterval to the current time
    unsigned long elapsedTime = currentTime - startTime;
    unsigned long expectedSamples = elapsedTime / samplingInterval;
    sampleTime = startTime + ((expectedSamples + 1) * samplingInterval);
    
    // If we've fallen behind (processing took too long), catch up
    if (sampleTime < currentTime) {
      sampleTime = currentTime + 1; // Schedule next sample ASAP
    }
  }
}

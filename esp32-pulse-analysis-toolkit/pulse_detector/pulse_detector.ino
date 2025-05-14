/*
 * ESP32 Pulse Signal Recorder
 * 
 * Records pulses from an external source (connected to pin 34)
 * Designed to detect pulses with:
 * - Period: ~0.8s
 * - ON time: ~0.4ms
 * 
 * Pin usage:
 * - Input: Pin 34 (ADC)
 * - Output indicator: Pin 2 (LED)
 */

// Pin Configuration for ESP32
const int sensorPin = 34;       // Analog input pin (ADC1_CH6)
const int ledPin = 2;           // ESP32 built-in LED pin

// Timing Configuration
const unsigned long sampleInterval = 1;  // 1ms between samples (1kHz)
const float expectedPulsePeriod = 800.0; // Expected period in milliseconds (0.8s)

// Detection Parameters
int threshold = 1800;           // Default threshold for pulse detection (12-bit ADC)
const int debounceDelay = 0;   // Minimum samples to confirm state change

// Variables
unsigned long lastSampleTime = 0;    // Last sample time
unsigned long lastPulseTime = 0;     // Time of last detected pulse
int consecutiveSamplesAbove = 0;     // Consecutive samples above threshold
int consecutiveSamplesBelow = 0;     // Consecutive samples below threshold
bool inPulse = false;               // Currently inside a pulse?

void setup() {
  // Initialize serial communication
  Serial.begin(115200);
  
  // Initialize pins
  pinMode(ledPin, OUTPUT);
  digitalWrite(ledPin, LOW);
  
  // Configure ESP32 ADC
  analogReadResolution(12);         // ESP32 has 12-bit ADC resolution (0-4095)
  analogSetAttenuation(ADC_11db);   // Full range: 0-3.3V
  
  Serial.println("ESP32 Pulse Recording System");
  Serial.println("Timestamp(ms),Value,PulseDetected,TimeSinceLastPulse(ms)");
}

void loop() {
  // Check if it's time for a new sample
  unsigned long currentTime = millis();
  if (currentTime - lastSampleTime >= sampleInterval) {
    lastSampleTime = currentTime;
    
    // Read sensor value
    int sensorValue = analogRead(sensorPin);
    
    // Default pulse state is 0 (no pulse)
    int pulseDetected = 0;
    unsigned long timeSinceLastPulse = 0;
    
    // State machine for pulse detection with debouncing
    if (sensorValue > threshold) {
      // Signal is above threshold
      consecutiveSamplesAbove++;
      consecutiveSamplesBelow = 0;
      
      // Check for start of new pulse (with debouncing)
      if (!inPulse && consecutiveSamplesAbove >= debounceDelay) {
        inPulse = true;
        digitalWrite(ledPin, HIGH);  // Visual feedback
        
        // Calculate time since last pulse
        timeSinceLastPulse = currentTime - lastPulseTime;
        lastPulseTime = currentTime;
        pulseDetected = 1;  // Indicate pulse detected
      }
    } else {
      // Signal is below threshold
      consecutiveSamplesBelow++;
      consecutiveSamplesAbove = 0;
      
      // Check for end of pulse (with debouncing)
      if (inPulse && consecutiveSamplesBelow >= debounceDelay) {
        inPulse = false;
        digitalWrite(ledPin, LOW);
      }
    }
    
    // Output the data in CSV format
    Serial.print(currentTime);
    Serial.print(",");
    Serial.print(sensorValue);
    Serial.print(",");
    Serial.print(pulseDetected);
    Serial.print(",");
    Serial.println(timeSinceLastPulse);
  }
}
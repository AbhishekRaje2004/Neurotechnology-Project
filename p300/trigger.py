import serial
import csv
import time
import numpy as np
import pandas as pd

sampling_rate = 1000  
sampling_interval = 1 / sampling_rate  

trigger_times = []
with open('trigger_times.csv', 'r') as file:
    reader = csv.reader(file)
    next(reader) 
    for row in reader:
        trigger_times.append(float(row[2]))  
arduino_port = 'COM3'
ser = serial.Serial(arduino_port, 115200)
time.sleep(2)  
combined_data = []

trigger_index = 0
eeg_sample_number = 0
start_time = time.time()

while trigger_index < len(trigger_times):
    if ser.in_waiting > 0:
        eeg_signal = int(ser.readline().strip())  
        
        current_time = time.time() - start_time  
        
        if abs(trigger_times[trigger_index] - current_time) < sampling_interval:
            combined_data.append([eeg_sample_number, current_time, eeg_signal, 'Trigger'])
            trigger_index += 1
        else:
            combined_data.append([eeg_sample_number, current_time, eeg_signal, ''])
        
        eeg_sample_number += 1

df = pd.DataFrame(combined_data, columns=["EEG Sample Number", "EEG Sample Time (s)", "EEG Signal", "Trigger"])

df.to_csv('combined_eeg_trigger_data.csv', index=False)

ser.close()
print(df)

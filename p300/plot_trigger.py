import pandas as pd

import matplotlib.pyplot as plt

# Read the CSV file
csv_file = 'trigger_times.csv'
data = pd.read_csv(csv_file)

# Ensure the column 'trigger_time' exists
if 'Time' not in data.columns:
    raise ValueError("The CSV file must contain a 'trigger_time' column.")

# Extract trigger times
trigger_times = data['Time']

# Create the tick plot
plt.figure(figsize=(10, 2))
plt.eventplot(trigger_times, orientation='horizontal', colors='blue')
plt.title('Trigger Time Tick Plot')
plt.xlabel('Time')
plt.yticks([])  # Remove y-axis ticks for clarity
plt.grid(axis='x', linestyle='--', alpha=0.7)

# Show the plot
plt.tight_layout()
plt.show()
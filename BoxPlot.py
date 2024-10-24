import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file into a DataFrame
df = pd.read_csv('api_latency.csv')

# Generate a boxplot for the latency of each test case
plt.figure(figsize=(10, 6))
df.boxplot(by="Test Case", column=["Latency (s)"])
plt.title("API Latency Boxplot")
plt.suptitle('')
plt.xlabel("Test Case")
plt.ylabel("Latency (seconds)")
plt.show()

import pytest
import requests
import time
import csv
import os
import pandas as pd
import matplotlib.pyplot as plt

# URL of the deployed API
API_URL = "http://serve-sentiment-new-env.eba-xivp2d2x.us-east-2.elasticbeanstalk.com/predict"  # Replace with your actual API URL

# Sample test cases
test_cases = [
    {"article": "This is a fake news article."},
    {"article": "Another fake news example."},
    {"article": "This is a real news."},
    {"article": "Another real news example."}
]

# Number of API calls for each test case
ITERATIONS = 100

@pytest.mark.parametrize("test_case", test_cases)
def test_api_performance(test_case):
    """
    Test the performance of the API by sending 100 requests for each test case
    and recording the response times. Save each test case's results in a separate CSV file.
    """
    latencies = []

    for _ in range(ITERATIONS):
        start_time = time.time()
        response = requests.post(API_URL, json=test_case)
        end_time = time.time()

        # Check if response is successful
        assert response.status_code == 200, f"Expected status code 200, got {response.status_code}"

        latency = end_time - start_time
        latencies.append(latency)

    # Save the performance data to a separate CSV file for each test case
    csv_filename = save_to_csv(test_case['article'], latencies)

    # Generate boxplot after saving the CSV
    generate_boxplot(csv_filename)


def save_to_csv(test_case, latencies):
    """
    Save the latency data for each test case to a separate CSV file.
    The file name will be based on the article text.
    """
    # Create a safe file name from the test case article
    safe_filename = "".join([c if c.isalnum() else "_" for c in test_case])[:50]  # Limit to 50 characters

    # CSV file name for this specific test case
    csv_filename = f"{safe_filename}_performance.csv"

    # Write data to the CSV file
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Test Case", "Latency (s)"])  # Header
        for latency in latencies:
            writer.writerow([test_case, latency])  # Write latency data

    print(f"Saved performance data for '{test_case}' to {csv_filename}")
    return csv_filename  # Return the file name for further processing


def generate_boxplot(csv_filename):
    """
    Generate a boxplot from the latency data stored in a CSV file.
    """
    # Load the CSV file into a DataFrame
    df = pd.read_csv(csv_filename)

    # Create the boxplot
    plt.figure(figsize=(8, 6))
    plt.boxplot(df['Latency (s)'])
    plt.title(f'Performance Boxplot for {csv_filename}')
    plt.ylabel('Latency (seconds)')
    plt.xticks([1], [df['Test Case'][0][:50]])  # Use the first 50 characters of the test case for the label
    plt.grid(True)

    # Save the boxplot as a PNG image
    boxplot_filename = csv_filename.replace(".csv", "_boxplot.png")
    plt.savefig(boxplot_filename)
    print(f"Boxplot saved as {boxplot_filename}")
    plt.close()


if __name__ == '__main__':
    pytest.main()

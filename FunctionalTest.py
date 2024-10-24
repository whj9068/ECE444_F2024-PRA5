import pytest
import requests

# URL of the deployed API
API_URL = "http://serve-sentiment-new-env.eba-xivp2d2x.us-east-2.elasticbeanstalk.com/predict"  # Replace with your actual API URL

# Sample test cases
test_cases = [
    ("This is a fake news article.", "FAKE"),
    ("Another fake news example.", "FAKE"),
    ("This is a real news.", "REAL"),
    ("Another real news example.", "REAL")
]

@pytest.mark.parametrize("article, expected_prediction", test_cases)
def test_predict(article, expected_prediction):
    """
    Test the /predict route with various news articles (both fake and real).
    """
    # Create the payload
    test_data = {"article": article}
    
    # Send POST request
    response = requests.post(API_URL, json=test_data)
    
    # Assertions
    assert response.status_code == 200, f"Expected status code 200 but got {response.status_code}"
    json_response = response.json()
    assert 'prediction' in json_response, "Response does not contain 'prediction' field"
    assert json_response['prediction'] == expected_prediction, f"Expected '{expected_prediction}' but got {json_response['prediction']}"

def test_no_article_provided():
    """
    Test the /predict route when no article is provided in the request.
    """
    # Create the payload with no article
    test_data = {}
    
    # Send POST request
    response = requests.post(API_URL, json=test_data)
    
    # Assertions
    assert response.status_code == 400, f"Expected status code 400 but got {response.status_code}"
    json_response = response.json()
    assert 'error' in json_response, "Response does not contain 'error' field"
    assert json_response['error'] == 'No article provided', "Error message mismatch"


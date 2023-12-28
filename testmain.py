# test_main.py
import requests

def test_predict_url():
    url = "http://127.0.0.1:8000/predict_url?url=https://www.nni.com.sg/patient-care/specialties-services/PublishingImages/CT%20Scan_Brain.jpg"
    response = requests.get(url)
    assert response.status_code == 200
    assert "your_expected_response" in response.text

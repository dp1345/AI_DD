import requests

url = 'http://127.0.0.1:5000/predict'
video_path = 'r3.mp4'  # Change to your actual test video path

with open(video_path, 'rb') as video_file:
    files = {'file': video_file}
    response = requests.post(url, files=files)

print("Status Code:", response.status_code)
print("Response JSON:", response.json())

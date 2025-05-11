import requests

url = 'http://127.0.0.1:5000/predict'
image_path = 'sydney2.jpg'  # Replace with an image path

with open(image_path, 'rb') as img_file:
    response = requests.post(url, files={'image': img_file})

print(response.json())

import requests

url = 'https://licentaserver.onrender.com/predict'  # Your Render app URL
image_path = 'sydney2.jpg'  # Path to your local image

with open(image_path, 'rb') as img_file:
    response = requests.post(url, files={'image': img_file})

print(response.json())

import requests

url = 'https://licentaserver.onrender.com/predict'  
image_path = 'sydney2.jpg' 

with open(image_path, 'rb') as img_file:
    response = requests.post(url, files={'image': img_file})

print(response.json())

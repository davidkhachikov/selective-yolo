from PIL import Image

image_path = "/home/user/Desktop/selective-yolo/resources/universal_attack/adversarial_karpin.jpg"
try:
    img = Image.open(image_path)
    img.show()  # To display the image
    print("Image loaded successfully!")
except Exception as e:
    print(f"Error loading image: {e}")

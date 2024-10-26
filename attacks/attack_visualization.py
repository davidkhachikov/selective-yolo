from ultralytics import YOLO

model = YOLO("yolo11n.pt")
results = model(["test_image/no_cat_image.jpg", "test_image/obvious_attack_image.jpg"])

# Process results list
for result in results:
    boxes = result.boxes
    masks = result.masks
    keypoints = result.keypoints
    probs = result.probs
    obb = result.obb
    result.show()

from simple_noise_attack import simple_noise_attack
from obvious_attack import obvious_attack
import os
import cv2
from tqdm import tqdm

test_path = "coco-2017/test/data"
noise_path = "coco-2017/test-noise-attack/data"
grad_path = "coco-2017/test-grad-attack/data"
gradient = cv2.imread("gradient.jpg")
for image in tqdm(os.listdir(test_path)):
    abs_image = os.path.join(test_path, image)
    abs_noise = os.path.join(noise_path, image)
    abs_grad = os.path.join(grad_path, image)
    original = cv2.imread(abs_image)
    cv2.imwrite(abs_noise, simple_noise_attack(original, 20))
    cv2.imwrite(abs_grad, obvious_attack(original, gradient, opacity=0.03))

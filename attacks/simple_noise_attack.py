import numpy as np
import cv2


def simple_noise_attack(orig: np.array, strength: int = 5) -> np.array:
    h, w, c = orig.shape
    n = int(h * w * c)
    noise = np.concatenate((np.random.poisson(strength, n//2),
                           1 - (np.random.poisson(strength, n//2))))
    np.random.shuffle(noise)
    noise = noise.reshape((h, w, c))
    attack = orig + noise
    return attack


if __name__ == "__main__":
    original = cv2.imread("test_image/no_noise_image.jpg")
    cv2.imwrite("test_image/simple_noise_attack_image.jpg", simple_noise_attack(original, 20))

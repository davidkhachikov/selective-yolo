import numpy as np
import cv2


def obvious_attack(original: np.array, dif_image: np.array, mask: np.array = None, opacity: float = 1.0) -> np.array:
    copy_dif = dif_image.copy()
    copy_orig = original.copy()
    if dif_image.shape != original.shape:
        h, w, c = original.shape
        copy_dif = cv2.resize(copy_dif, (w, h))
    if mask is not None and mask.shape != original.shape:
        h, w, c = original.shape
        mask = cv2.resize(mask, (w, h))
    if mask is not None:
        copy_dif = cv2.bitwise_and(copy_dif, copy_dif, mask=mask)
        copy_orig = cv2.bitwise_and(original, original, mask=cv2.bitwise_not(mask))
    attack = cv2.bitwise_or(copy_orig, copy_dif)
    if opacity < 1:
        attack = cv2.addWeighted(attack, opacity, original, 1 - opacity, 0)
    return attack


if __name__ == "__main__":
    original = cv2.imread("test_image/no_cat_image.jpg")
    different = cv2.imread("test_image/cat_image.jpg")
    assert different.shape == original.shape
    _, mask = cv2.threshold(different, 20, 255, cv2.THRESH_BINARY)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("test_image/obvious_attack_image.jpg", obvious_attack(original, different, mask, 0.1))


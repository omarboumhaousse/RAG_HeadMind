"""Step 2: Image preprocessing — binarization, deskew, denoising."""
import numpy as np
from PIL import Image
import cv2

def preprocess(image: np.ndarray) -> np.ndarray:
    """Take an RGB page image, return a cleaned binary image.

    Steps:
        1. Convert to grayscale
        2. Binarize (Otsu or Sauvola thresholding)
        3. Deskew (projection profile variance maximization) Avec OpenCV pour avoir des prmeiers résultats
        4. Denoise (morphological operations) J'obtiens pour l'insant de meilleurs résultats sans denoise sur les textes 
        de Omnidobench qui sont très lisibles, je décide donc d'omettre cette étape dans la fonction preprocess
    """
    
    gray = convertgrayscale(image)
    binarizer = SauvolaBinarizer()

    binary = binarizer.binarize(gray)
    img = deskew(binary)
    return img


def convertgrayscale(image:np.ndarray) -> np.ndarray:
  
    if image.ndim == 2:
        # déjà en grayscale
        return image.astype(np.float64)

    if image.shape[2] != 3:
        raise ValueError("Image RGB attendue")

    # formule standard de luminance
    r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]

    gray = 0.299 * r + 0.587 * g + 0.114 * b

    return gray.astype(np.float64)

class SauvolaBinarizer:

    def __init__(self, window_size=31, k=0.35, R=128):

        self.window_size = window_size
        self.k = k
        self.R = R
        # k et R sont arbitraires selon ce qui marche le mieux dans la littérature
    
    def binarize(self, image : np.ndarray) -> np.ndarray:

        if image.ndim != 2:
            raise ValueError("Image grayscale requise")

        r = self.window_size // 2
        padded = np.pad(image, r, mode="reflect")
        H, W = padded.shape

        integral = np.zeros((H + 1, W + 1))
        integral_sq = np.zeros((H + 1, W + 1))

        integral[1:, 1:] = np.cumsum(np.cumsum(padded, axis=0), axis=1)
        integral_sq[1:, 1:] = np.cumsum(np.cumsum(padded**2, axis=0), axis=1)

        h, w = image.shape
        area = self.window_size * self.window_size

        y0 = np.arange(h)
        x0 = np.arange(w)
        y1 = y0
        x1 = x0
        y2 = y0 + self.window_size
        x2 = x0 + self.window_size

        sum_ = (
            integral[y2[:, None], x2]
            - integral[y1[:, None], x2]
            - integral[y2[:, None], x1]
            + integral[y1[:, None], x1]
        )

        sum_sq = (
            integral_sq[y2[:, None], x2]
            - integral_sq[y1[:, None], x2]
            - integral_sq[y2[:, None], x1]
            + integral_sq[y1[:, None], x1]
        )

        mean = sum_ / area
        var = sum_sq / area - mean**2
        std = np.sqrt(np.maximum(var, 0))

        T = mean * (1 + self.k * (std / self.R - 1))
        binary = np.where(image < T, 0, 255).astype(np.uint8)
        return binary



def deskew(image: np.ndarray) -> np.ndarray:
    coords = np.column_stack(np.where(image == 0))
    angle = cv2.minAreaRect(coords)[-1]

    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        image,
        M,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )

    return rotated


def morph(image: np.ndarray) -> np.ndarray:
    kernel = np.ones((2,2), np.uint8)
    img = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(img)
    min_area = 10
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            img[labels == i] = 0
    return img





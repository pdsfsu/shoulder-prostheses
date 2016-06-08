import cv2
import numpy as np

def open_center_kernel():
    return np.array([[1, 1, 1, 1, 1],
                    [1, 0, 0, 0, 1],
                    [1, 0, 0, 0, 1],
                    [1, 0, 0, 0, 1],
                    [1, 1, 1, 1, 1]], np.uint8)

def open_cross_kernel():
    return np.array([[1, 1, 0, 1, 1],
                    [1, 1, 0, 1, 1],
                    [0, 0, 0, 0, 0],
                    [1, 1, 0, 1, 1],
                    [1, 1, 0, 1, 1]], np.uint8)

def open_elliptical_kernel():
    return np.array([[0, 0, 1, 0, 0],
                    [1, 1, 0, 1, 1],
                    [1, 0, 0, 0, 1],
                    [1, 1, 0, 1, 1],
                    [0, 0, 1, 0, 0]], np.uint8)

def sm_open_cross_kernel():
    return np.array([[1, 0, 1],
                    [0, 0, 0],
                    [1, 0, 1]], np.uint8)

def rectangular_kernel(size=5):
    return cv2.getStructuringElement(cv2.MORPH_RECT,(size,size))

def elliptical_kernel(size=5):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(size,size));

def cross_kernel(size=5):
    return cv2.getStructuringElement(cv2.MORPH_CROSS,(size,size));

def erosion(img, kernel=None, iterations=1):
    if kernel is None:
        kernel = rectangular_kernel()
    return cv2.erode(img, kernel, iterations)

def dilation(img, kernel=None, iterations=1):
    if kernel is None:
        kernel = rectangular_kernel()
    return cv2.dilate(img, kernel, iterations)

def opening(img, kernel=None):
    if kernel is None:
        kernel = rectangular_kernel()
    return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

def closing(img, kernel=None):
    if kernel is None:
        kernel = rectangular_kernel()
    return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
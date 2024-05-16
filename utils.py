import numpy as np
import cv2 as cv


def get_limits(color):
    c = np.uint8([[color]])
    hsvC = cv.cvtColor(c, cv.COLOR_BGR2HSV)  # Convert BGR to HSV
    hue_value = hsvC[0][0][0]

    if 160 <= hue_value <= 180 or 0 <= hue_value <= 20:
        # Red hues, handle wrap-around
        lower_limit_1 = np.array([max(hue_value - 10, 0), 100, 100], dtype=np.uint8)
        upper_limit_1 = np.array([min(hue_value + 10, 179), 255, 255], dtype=np.uint8)
        lower_limit_2 = np.array([(hue_value + 170) % 180, 100, 100], dtype=np.uint8)
        upper_limit_2 = np.array([min((hue_value + 190) % 180, 179), 255, 255], dtype=np.uint8)
        return (lower_limit_1, upper_limit_1), (lower_limit_2, upper_limit_2)
    else:
        # General case
        lower_limit = np.array([max(hue_value - 10, 0), 100, 100], dtype=np.uint8)
        upper_limit = np.array([min(hue_value + 10, 179), 255, 255], dtype=np.uint8)
        return (lower_limit, upper_limit), None
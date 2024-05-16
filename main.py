import cv2 as cv
import numpy as np
from utils import get_limits  # Assuming utils is the name of your file/module
from PIL import Image


def get_color_from_user():
    color_input = input("Enter BGR values (comma separated): ")
    color = list(map(int, color_input.split(',')))
    if len(color) != 3:
        raise ValueError("Please provide exactly 3 integers for BGR values.")
    return color


# Obtain user input for the target color in BGR format
target_color = get_color_from_user()

# Initialize webcam capturing (default camera is at index 0)
cam = cv.VideoCapture(0)

while True:
    retval, frame = cam.read()  # Read a frame from the webcam
    if not retval:  # If frame is not captured, break the loop
        break

    # Convert the captured frame from BGR to HSV colorspace
    hsv_img = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    # Get the HSV limits for the user specified color
    limits_1, limits_2 = get_limits(target_color)
    lower_limit_1, upper_limit_1 = limits_1
    mask1 = cv.inRange(hsv_img, lower_limit_1, upper_limit_1)

    if limits_2 is not None:
        lower_limit_2, upper_limit_2 = limits_2
        mask2 = cv.inRange(hsv_img, lower_limit_2, upper_limit_2)
        # Combine both masks if it's a color with wrap-around (like red)
        mask = cv.bitwise_or(mask1, mask2)
    else:
        mask = mask1

    # Convert the mask from numpy array to PIL Image to use getbbox method
    mask_pil = Image.fromarray(mask)
    bbox = mask_pil.getbbox()  # Get bounding box coordinates for the white areas of the mask

    if bbox is not None:  # If a bounding box is found, draw the rectangle
        x1, y1, x2, y2 = bbox
        frame = cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)  # Draw a green rectangle on the detected area

    # Display the frame with the detected color object
    cv.imshow('Webcam frame', frame)

    # Exit the loop if 'q' key is pressed
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cam.release()
cv.destroyAllWindows()
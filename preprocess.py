import cv2
import numpy as np

resize_width, resize_height = 240, 240 # TODO: test performance of different scaling

def preprocess_img(path):
    img = cv2.imread(path)

    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    smoothed = cv2.GaussianBlur(img_hsv, ksize=(3,3), sigmaX=1.5, sigmaY=1.5)

    dim = (resize_width, resize_height) # subject to change

    resized = cv2.resize(smoothed, dim, interpolation = cv2.INTER_AREA)

    return resized

# if run as script, apply preprocessing to demo image and show result
if __name__ == "__main__":
    img_path = "./downloads/gsv_0.jpg"
    orig_img = cv2.imread(img_path)
    processed_img = preprocess_img(img_path)
    cv2.imshow("original", orig_img)
    cv2.imshow("preprocessed", processed_img)
    cv2.waitKey(0)
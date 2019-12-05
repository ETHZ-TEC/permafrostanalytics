# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 14:59:40 2019

@author: RGArt4
"""

import cv2
import numpy as np

# import matplotlib.pyplot as plt

input_path = r"C:\Users\RGArt4\Desktop\Mountaineer detection\input\20170826_071210.jpg"
# input_path = r"C:\Users\RGArt4\Desktop\Mountaineer detection\input\20170826_071610.jpg"

# input_path = r"C:\Users\RGArt4\Desktop\Mountaineer detection\input\hr\20170830_074410_hr.jpg"
# input_path = r"C:\Users\RGArt4\Desktop\Mountaineer detection\input\hr\20170830_074811.jpg"

# cap = cv2.VideoCapture(0)
frame = cv2.imread(input_path)

# FFT
# img = cv2.imread(input_path)
# img_float32 = np.float32(img)
# dft = cv2.dft(img_float32, flags = cv2.DFT_COMPLEX_OUTPUT)
# dft_shift = np.fft.fftshift(dft)
# magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))
# plt.subplot(121),plt.imshow(img, cmap = 'gray')
# plt.title('Input Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
# plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
# plt.show()

# while(1):
# _, frame = cap.read()
# It converts the BGR color space of image to HSV color space
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

# Threshold of blue in HSV space // RGB?
# lower_blue = np.array([35, 140, 60])
# upper_blue = np.array([255, 255, 180])

# Threshold of blue in HSV space // RGB?
lower_blue = np.array([110, 140, 50])
upper_blue = np.array([130, 255, 255])

lower_green = np.array([50, 140, 50])
upper_green = np.array([70, 255, 255])

lower_yellow = np.array([20, 140, 50])
upper_yellow = np.array([40, 255, 255])


# Threshold of blue in HSV space // RGB?
# lower_red = np.array([80, 140, 60])
# upper_red = np.array([255, 255, 180])

# Threshold of red in HSV space // RGB?
lower_rock = np.array([0, 30, 30])
upper_rock = np.array([255, 80, 80])

# Threshold of red in HSV space
lower_red = np.array([0, 160, 100])
upper_red = np.array([20, 200, 255])

# lower_red = np.array([0, 160, 50])
# upper_red = np.array([10, 255, 255])


# preparing the mask to overlay
mask_red = cv2.inRange(hsv, lower_red, upper_red)
mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
mask_green = cv2.inRange(hsv, lower_green, upper_green)

masks = cv2.bitwise_or(mask_red, mask_blue)  # ,mask_yellow,mask_green)


# The black region in the mask has the value of 0,
# so when multiplied with original image removes all non-blue regions
# result1 = cv2.bitwise_and(frame, frame, mask = mask)
# result2 = cv2.bitwise_and(frame, frame, mask = mask2)

result = cv2.bitwise_or(frame, frame, mask=masks)

cv2.imshow("frame", frame)
cv2.imshow("red mask", mask_red)
cv2.imshow("blue mask", mask_blue)
cv2.imshow("yellow mask", mask_yellow)
cv2.imshow("green mask", mask_green)
cv2.imshow("All masks", masks)
# cv2.imshow('result1', result1)
# cv2.imshow('result2', result2)
cv2.imshow("result", result)

# cv2.imwrite(r"C:\Users\RGArt4\Desktop\Mountaineer detection\examples\1_frame.jpg", frame)
# cv2.imwrite(r"C:\Users\RGArt4\Desktop\Mountaineer detection\examples\1_red.jpg", mask_red)
# cv2.imwrite(r"C:\Users\RGArt4\Desktop\Mountaineer detection\examples\1_blue.jpg", mask_blue)
# cv2.imwrite(r"C:\Users\RGArt4\Desktop\Mountaineer detection\examples\1_yellow.jpg", mask_yellow)
# cv2.imwrite(r"C:\Users\RGArt4\Desktop\Mountaineer detection\examples\1_green.jpg", mask_green)
# cv2.imwrite(r"C:\Users\RGArt4\Desktop\Mountaineer detection\examples\1_masks.jpg", masks)
# cv2.imshow('result1', result1)
# cv2.imshow('result2', result2)
# cv2.imwrite(r"C:\Users\RGArt4\Desktop\Mountaineer detection\examples\1_result.jpg", result)


cv2.waitKey(0)

cv2.destroyAllWindows()
# frame.release()
exit()

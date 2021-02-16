#!/usr/bin/env python3
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import math


def getHoughLines(img):
    rho, theta, thresh = 2, np.pi/180, 1
    lines = cv2.HoughLines(img, rho, theta, thresh,)
    return lines[:50]


def drawHoughLines(lines, img):
    for line in lines:
        rho, theta = line[0]
        if(theta < math.radians(5) or theta > math.radians(175)):
            continue
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 10000*(-b))
        y1 = int(y0 + 10000*(a))
        x2 = int(x0 - 10000*(-b))
        y2 = int(y0 - 10000*(a))

        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 1)
    return img


def drawHoughLinesP(img, orig):
    minLineLength = 100
    maxLineGap = 10
    lines = cv2.HoughLinesP(img, 1, np.pi/180, 100,
                            minLineLength, maxLineGap)
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(orig, (x1, y1), (x2, y2), (0, 255, 0), 1)
    return orig


def getCorners(img):
    # Apply Shi-Tomasi corner detection
    corners = cv2.goodFeaturesToTrack(img, maxCorners=500,
                                      qualityLevel=0.01,
                                      minDistance=10)
    corners = np.int0(corners)
    return corners


def drawCorners2(img, orig):
    dst = cv2.cornerHarris(img, blockSize=2, ksize=3, k=.04)
    img_2 = orig.copy()
    img_2[dst > 0.01*dst.max()] = [255, 0, 0]
    return img_2


def drawCorners(corners, img):
    img_2 = img.copy()
    for i in corners:
        x, y = i.ravel()
        cv2.circle(img_2, center=(x, y),
                   radius=5, color=255, thickness=-1)
    return img_2


###############
# line detection
###############
orig = cv2.imread("data/living_space.jpg")
img = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
width, height = img.shape
kernel_size = math.floor(min([width, height]) * 0.05)


kernel = np.ones((kernel_size, kernel_size), np.uint8)
# This gets rid of small featuers
# bilateral = cv2.bilateralFilter(img, 20, sigmaSpace=75, sigmaColor=75)
opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)


med_val = np.median(opened)
lower = int(max(0, .7*med_val))
upper = int(min(255, 1.3*med_val))
# img = cv2.blur(img, ksize=(9, 9))
canny = cv2.Canny(image=opened, threshold1=lower, threshold2=upper+100)

blank = np.zeros(orig.shape)
lines = getHoughLines(canny)
img = drawHoughLines(lines, orig)

# Houghlines P not as good,and cant segment via rho,theta
# results = drawHoughLinesP(img, blank)


# detect corners?
# corners = getCorners(img)
# results = drawCorners(corners, orig)
# results = drawCorners2(img, orig)

Image.fromarray(orig).show("1")
# Image.fromarray(bilateral).show("2")
Image.fromarray(opened).show("3")
Image.fromarray(canny).show("4")
Image.fromarray(img).show("5")

Image.fromarray(img).save("img.jpg")

# plt.imshow()
# plt.axis('off')
# plt.show()


# cv2.imwrite('step2.jpg', img)

#########################################
# Steps:
#########################################
# 1. Load image
# 2. Apply bilateral filter
# 3. Apply opening (To get rid of small features)
# 4. Apply canny edge detection
# 5. Find hough lines
# 6. Cluster hough lines TODO
# 7. Find intersections of the lines TODO


# camera pose estimation
# https://www.elderlab.yorku.ca/wp-content/uploads/2013/08/ElAssalCRV17.pdf
# https://www.researchgate.net/publication/232639681_Camera_Pan_and_Tilt_Estimation_in_Soccer_Scenes_Based_on_Vanishing_Points

# Vanishing point detection notes:
# https://dsp.stackexchange.com/questions/46253/given-a-set-of-lines-find-only-those-who-are-parallel-perspective
#

import cv2
import numpy as np
import matplotlib.pyplot as plt


class OnnxInferer:
    def __init__(self):
        pass

    def getHoughLines(self, img):
    rho, theta, thresh = 2, np.pi/180, 1
    lines = cv2.HoughLines(img, rho, theta, thresh,)
    return lines[:50]

    def drawHoughLines(self, lines, img):
        for line in lines:
            rho, theta = line[0]
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

    def drawHoughLinesP(self, img, orig):
        minLineLength = 100
        maxLineGap = 10
        lines = cv2.HoughLinesP(img, 1, np.pi/180, 100,
                                minLineLength, maxLineGap)
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(orig, (x1, y1), (x2, y2), (0, 255, 0), 1)
        return orig

    def getCorners(self, img):
        # Apply Shi-Tomasi corner detection
        corners = cv2.goodFeaturesToTrack(img, maxCorners=500,
                                          qualityLevel=0.01,
                                          minDistance=10)
        corners = np.int0(corners)
        return corners

    def drawCorners2(self, img, orig):
        dst = cv2.cornerHarris(img, blockSize=2, ksize=3, k=.04)
        img_2 = orig.copy()
        img_2[dst > 0.01*dst.max()] = [255, 0, 0]
        return img_2

    def drawCorners(self, corners, img):
        img_2 = img.copy()
        for i in corners:
            x, y = i.ravel()
            cv2.circle(img_2, center=(x, y),
                       radius=5, color=255, thickness=-1)
        return img_2

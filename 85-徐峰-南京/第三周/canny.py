#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author： xufeng
# datetime： 2021/12/15 0:56 
# ide： PyCharm


import cv2 as cv
import numpy as np

'''
cv2.Canny(image, threshold1, threshold2[, edges[, apertureSize[, L2gradient ]]])   
必要参数：
第一个参数是需要处理的原图像，该图像必须为单通道的灰度图；
第二个参数是滞后阈值1；
第三个参数是滞后阈值2。
'''

img = cv.imread("../../../../BaiduNetdiskDownload/lenna.png", 1)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow("canny", cv.Canny(gray, 200, 250))
cv.imshow("ori", img)
cv.waitKey()
cv.destroyAllWindows()
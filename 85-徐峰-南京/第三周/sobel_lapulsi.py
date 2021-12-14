#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author： xufeng
# datetime： 2021/12/15 1:02 
# ide： PyCharm


import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread("../../../../BaiduNetdiskDownload/lenna.png", 1)
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)


'''
Sobel算子
Sobel算子函数原型如下：
dst = cv2.Sobel(src, ddepth, dx, dy[, dst[, ksize[, scale[, delta[, borderType]]]]]) 
前四个是必须的参数：
第一个参数是需要处理的图像；
第二个参数是图像的深度，-1表示采用的是与原图像相同的深度。目标图像的深度必须大于等于原图像的深度；
dx和dy表示的是求导的阶数，0表示这个方向上没有求导，一般为0、1、2。
其后是可选的参数：
dst是目标图像；
ksize是Sobel算子的大小，必须为1、3、5、7。
scale是缩放导数的比例常数，默认情况下没有伸缩系数；
delta是一个可选的增量，将会加到最终的dst中，同样，默认情况下没有额外的值加到dst中；
borderType是判断图像边界的模式。这个参数默认值为cv2.BORDER_DEFAULT。
'''

img_sobelX = cv.Sobel(img_gray, cv.CV_64F, 1, 0, ksize=3) # x方向求导
img_sobelY = cv.Sobel(img_gray, cv.CV_64F, 0, 1, ksize=3) # y 方向求导

# laplace 算子
img_laplace = cv.Laplacian(img_gray, cv.CV_64F, ksize=3)

# canny 算子
img_canny= cv.Canny(img_gray, 100, 150)

plt.subplot(231), plt.imshow(img_gray), plt.title('original')
plt.subplot(232), plt.imshow(img_sobelX, "gray"), plt.title("Sobel_x")
plt.subplot(233), plt.imshow(img_sobelY, "gray"), plt.title("Sobel_y")
plt.subplot(234), plt.imshow(img_laplace,  "gray"), plt.title("Laplace")
plt.subplot(235), plt.imshow(img_canny, "gray"), plt.title("Canny")
plt.show()


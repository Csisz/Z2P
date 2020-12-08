import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('people.jpg')
grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

edge_kernel = np.array([[0,-.5,0 ],[-.5,3,-.5 ],[0,-.5,0]])
img0 = cv2.filter2D(grayscale, -1, edge_kernel)

cv2.imshow('sharpen_kernel_new',img0)
cv2.imshow('original',img)
cv2.waitKey(0)
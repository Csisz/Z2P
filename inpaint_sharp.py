import numpy as np
import cv2
import os
from matplotlib import pyplot as plt

input_path = 'imgs/test/input'
mask_path = 'imgs/test/mask'
output_path = 'imgs/test/output/'

filenames = os.listdir(os.path.join(input_path))
filenames.sort()
for filename in filenames:
    print(filename)
    image = cv2.imread(os.path.join(input_path,filename), 1)
    mask_image = cv2.imread(os.path.join(mask_path,filename), 0)
    telea_image = cv2.inpaint(image, mask_image, 5, cv2.INPAINT_TELEA)
    # ns_image = cv2.inpaint(image, mask_image, 5, cv2.INPAINT_NS)
    
    grayscale = cv2.cvtColor(telea_image, cv2.COLOR_BGR2GRAY)
    sharpen_kernel = np.array([[0,-.5,0 ],[-.5,3,-.5 ],[0,-.5,0]])
    img0 = cv2.filter2D(grayscale, -1, sharpen_kernel)
    
    cv2.imwrite(os.path.join(output_path,filename), img0)
    # cv2.imwrite(os.path.join(output_path,'NS_'+filename), ns_image)




import numpy as np
import cv2
import os

image = cv2.imread("grand_bad.jpg", 1)
mask_image = cv2.imread("mask2.png", 0)

input_path = 'test/input'
mask_path = 'test/mask'
output_path = 'test/output'

filenames = os.listdir(os.path.join(input_path))
filenames.sort()
for filename in filenames:
    print(filename)
    image = cv2.imread(os.path.join(input_path,filename), 1)
    mask_image = cv2.imread(os.path.join(mask_path,filename), 0)
    telea_image = cv2.inpaint(image, mask_image, 5, cv2.INPAINT_TELEA)
    ns_image = cv2.inpaint(image, mask_image, 5, cv2.INPAINT_NS)
    cv2.imwrite(os.path.join(output_path,filename), telea_image)
    cv2.imwrite(os.path.join(output_path,'NS_'+filename), ns_image)



# cv2.imshow("Orignal Image", image)
# cv2.imshow("Mask Image", mask_image)

# cv2.imshow("TELEA Restored Image", telea_image)
# cv2.imshow("NS Restored Image", ns_image)

# cv2.waitKey(0)
# cv2.destroyAllWindows() 

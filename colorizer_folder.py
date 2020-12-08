from PIL import Image,ImageEnhance
import cv2
import numpy as np
import argparse
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from colorizers import *
import os

input_path = 'imgs/test/output/'
output_path = 'imgs/final_output/colorized/'

filenames = os.listdir(os.path.join(input_path))
filenames.sort()
for filename in filenames:
    print('Start colorizing-----> '+filename)

    # load colorizerscolor_imgs
    colorizer_siggraph17 = siggraph17(pretrained=True).eval()
    colorizer_siggraph17.cuda()

    # default size to process images is 256x256
    # grab L channel in both original ("orig") and resized ("rs") resolutions
    img = cv2.imread(os.path.join(input_path,filename), 1)
    (tens_l_orig, tens_l_rs) = preprocess_img(img, HW=(256,256))
    tens_l_rs = tens_l_rs.cuda()

    # colorizer outputs 256x256 as map
    # resize and concatenate to original L channel
    img_bw = postprocess_tens(tens_l_orig, torch.cat((0*tens_l_orig,0*tens_l_orig),dim=1))
    out_img_siggraph17 = postprocess_tens(tens_l_orig, colorizer_siggraph17(tens_l_rs).cpu())

    filename = "{0}_{2}{1}".format(*os.path.splitext(filename) + ('coloriZed',))
    prefix = output_path + filename
    plt.imsave('%s_coloriZed.png'%prefix, out_img_siggraph17)
    print("File saved: "+prefix)


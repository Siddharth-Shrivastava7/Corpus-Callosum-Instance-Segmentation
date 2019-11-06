import glob
import cv2
import numpy as np


import matplotlib.pyplot as plt

direc = glob.glob('./train/masko/*.jpg') 

direc = sorted(direc)

for k in direc:

    img = plt.imread(k)

    imgn = cv2.flip(img,0)

    save_path = k.replace('masko','mask') 

    plt.imsave(save_path,imgn) 

print('finished it')
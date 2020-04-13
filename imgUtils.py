# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 12:18:40 2018

@author: renjch
"""

#!/usr/bin/python
# -*- coding: UTF-8 -*-
import os  
import os.path
from xml.etree.ElementTree import parse, Element
import numpy as np  
import cv2  
#create a black use numpy,size is:512*512


#Year_Joint_Foreground_background.jpg / .xml
def joint_foreground_background(foreground, background):
    '''
    merge image and return path & image
    save image and annotation xml to the standard location
    '''
    foreImage = cv2.imread(foreground)
    bkImage = cv2.imread(background)
    
    foreImageW = foreImage.shape[1]
    foreImageH = foreImage.shape[0]
    
    bkImageW = bkImage.shape[1]
    bkImageH = bkImage.shape[0]
    
#    print(foreImageW, foreImageH, bkImageW, bkImageH)
    
    h = max(foreImageH, bkImageH)
#    print(h)
    
    img = np.zeros((h, foreImageW + bkImageW, 3), np.uint8) 
#    print(img.shape)
    img[0:foreImageH, 0:foreImageW] = foreImage  
    img[0:bkImageH, foreImageW:foreImageW + bkImageW] = bkImage
    
    return img
    #fill the image with white
#    cv2.imshow('', img)
#    cv2.waitKey(0)
#    
#    cv2.destroyAllWindows()
    
    
if __name__ == '__main__':
    img1 = '1.png'
    xml = '20181212_195115521.xml'
    img2 = '20180927111205766.bmp'        
    
    img = joint_foreground_background(img1, img2)
    
    cv2.imshow('', img)
    cv2.waitKey(0)
    
    cv2.destroyAllWindows()
    
    
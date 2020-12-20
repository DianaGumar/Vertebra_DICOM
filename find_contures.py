#!/usr/bin/env python
# coding: utf-8

# In[3]:


import sys
import numpy as np
import cv2 as cv
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors
from matplotlib.colors import hsv_to_rgb
from scipy.interpolate import splprep, splev
import codecs, json 
import os
import glob
import shutil


# In[2]:


# общая функция
# забрасываем маску, получаем контура

def GetCounters(im_thresh):
    contours, hierarchy = cv.findContours(im_thresh.copy(), cv.RETR_CCOMP, cv.CHAIN_APPROX_NONE)
    # вычисляем площади замкнутых контуров, 
    # для отсеивания мусора
    areas = []
    for i in contours:
        areas.append(cv.contourArea (i))
    areas_average = sum(areas) / len(areas)
    # убираем небольшие контура
    contours_new = []
    for i in contours:
        if cv.contourArea(i) > areas_average :
            contours_new.append(i)   
    #уменьшение количества точек и аппроксимация контуров
    contours_appr = contours_new.copy()
    for i in range(len(contours_new)):
        epsilon = 0.0007*cv.arcLength(contours_new[i], True)
        approx = cv.approxPolyDP(contours_new[i], epsilon, True)
        contours_appr[i] = approx
    
    return contours_appr

def Blur(img):
    median = cv.medianBlur(img.copy(),7)
    b = 5
    blur = cv.GaussianBlur(median,(b,b),0)
    return blur

    
def SaveImgConrures(img, contures, path):
    cv.imwrite(path + "img.jpg", img)
    for i in range(len(contures)):
        b = contures[i].tolist()
        file_path = path + str(i) + ".json"
        json.dump(b, codecs.open(file_path, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)
          
def FindDicomCounter(path_to_img, img_name):
    img = cv.imread(path_to_img + img_name) 
    img = cv.bitwise_not(img)
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
#     hsv[:,:,2] = [[max(pixel - 25, 0) if pixel < 210 else min(pixel + 25, 255) for pixel in row] for row in hsv[:,:,2]]
    hsv_min = np.array((0, 0, 0), np.uint8)
    hsv_max = np.array((0, 0.02, 215), np.uint8)
    hsv = Blur(hsv)
    thresh = cv.inRange(hsv, hsv_min, hsv_max)
    counters = GetCounters(thresh)
    img_contures = cv.drawContours(img.copy(), counters, -1, (255, 0, 0), 1, cv.LINE_AA, None, 1)
    SaveImgConrures(img_contures, counters, path_to_img)
    

def find_counters_in_folder(path_to_folder):
    for img in glob.glob(path_to_folder + "/*.jpg"):
        #n= cv2.imread(img)
        print(img)
        new_path = os.path.splitext(img)[0] + "/"
        img_name = os.path.basename(img) 
        os.mkdir(new_path)
        shutil.move(img, new_path + img_name)
        FindDicomCounter(new_path, img_name)


# In[4]:


if __name__ == "__main__":
    find_counters_in_folder(sys.argv[1])


# In[ ]:





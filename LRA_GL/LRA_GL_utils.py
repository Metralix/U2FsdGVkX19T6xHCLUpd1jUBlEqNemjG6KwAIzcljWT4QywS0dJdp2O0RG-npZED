# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 21:00:10 2019

@author: haoyang li
"""

import numpy as np
import math
import matplotlib.pyplot as plt  
import matplotlib.image as mpimg  
import os

"""
function:read the picture from the os folder
input:folder path
output:(train_vis,train_nir,gallary,probe) in the format of matrix
"""
def readPicture(folder_path=r"C:\Users\54164\Desktop\LRA_GL\dataset_CASIA"):
    #读取 train_vis 训练的彩色照片，并转化为灰度图
    train_vis = []
    for i in range(30840):
        train_vis_path = os.path.join(folder_path,'train_vis','{0:05d}.jpg'.format(i+1))
        if os.path.exists(train_vis_path):
            I = mpimg.imread(train_vis_path)  
            train_vis.append(rgb2gray(I))
    print("readed train_vis")
    #读取 train_nir 训练的黑白照片 
    train_nir = []
    for i in range(30840):
        suffix = ['a','b','c','d','e']
        path_a = os.path.join(folder_path,'train_nir','{0:05d}_a.jpg'.format(i+1))
        if os.path.exists(path_a):
            for c in suffix:
                path_all = os.path.join(folder_path,'train_nir','{0:05d}_{1}.jpg'.format(i+1,c))
                I = mpimg.imread(path_all)  
                train_nir.append(I)
    print("readed train_nir")
    #读取 test_vis gallary 测试的彩照  
    test_vis = []
    for i in range(30840):
        test_vis_path = os.path.join(folder_path,'test_vis','{0:05d}.jpg'.format(i+1))
        if os.path.exists(test_vis_path):
            I = mpimg.imread(test_vis_path)  
            test_vis.append(rgb2gray(I))
    print("readed test_vis")
    #读取 test_nir probe 测试的黑白照
    test_nir = []
    for i in range(30840):
        test_nir_path = os.path.join(folder_path,'test_nir','{0:05d}.jpg'.format(i+1))
        if os.path.exists(test_nir_path):
            I = mpimg.imread(test_nir_path)  
            test_nir.append(I)
    
    print("readed test_nir")        
    return train_vis,train_nir,test_vis,test_nir



"""
lbp encode the picture set
input:picture sets of 2d or 3d reality photos
output:lbp code set of these pictures 
"""
def lbp_encode(picture_set,cell_size=20):
    for i in range(picture_set.shape[0]):
        if i % 50 == 0:
            print("read set's no. {0} pictures".format(i))
        hist = np.zeros(256*cell_size*cell_size)
        hist_idx = 0
        pic = picture_set[i,:,:].copy()
        height = math.floor(pic.shape[0]/cell_size) #20
        width = math.floor(pic.shape[1]/cell_size) #20
        for cell_x in range(cell_size): #0-10
            for cell_y in range(cell_size):
                for x in range(width):
                    x = x + cell_x * width
                    for y in range(height): 
                        y = y + cell_y * height
                        if x-2>=0 and y-2>=0 and x+2<pic.shape[1] and y+2<pic.shape[0]: #边界值忽略
	                        code = []
	                        code.append(pic[y-1,x-1]>pic[y,x])
	                        code.append(pic[y,x-1]>pic[y,x])
	                        code.append(pic[y+1,x-1]>pic[y,x])
	                        code.append(pic[y+1,x]>pic[y,x])
	                        code.append(pic[y+1,x+1]>pic[y,x])
	                        code.append(pic[y,x+1]>pic[y,x])
	                        code.append(pic[y-1,x+1]>pic[y,x])
	                        code.append(pic[y-1,x]>pic[y,x])
	                        code = np.array(code)
	                        code = code + 0
	                        code = bin2oct(code)
	                        hist[code + hist_idx*256] += 1
                hist_idx += 1
        hist = hist/(height*width)
        hist = hist.reshape([256*cell_size*cell_size,1])
        if i == 0:
            coverted_matrix = hist.copy()
        else:
            coverted_matrix = np.hstack((coverted_matrix,hist.copy()))

    return coverted_matrix
                        
"""
calculate the intro_class variants
input:the train_vis and trian_nir datasets
output:get the intro_class variant of the trian datasets
       the out size is the same as a 840*480 gary picture
"""
def intro_class_variant(vis_pics,nir_pics):
	variant_set = []
	for i in range(nir_pics.shape[0]):
		pic = nir_pics[i,:,:]-vis_pics[i//5,:,:]
		variant_set.append(pic.copy())

	return np.array(variant_set)

"""
input:result matrix of the algorithm
output:calculate the accuracy of the model
"""
def calculate_accuracy(res_matrix):
    res_matrix = res_matrix + 10
    max_list = np.max(res_matrix,axis=0)
    for i in range(res_matrix.shape[1]):
        res_matrix[:,i] = res_matrix[:,i]-max_list[i]+1e-5
    
    res_matrix = res_matrix >= 0
    correct_cnt = 0
    for i in range(res_matrix.shape[1]):
        if res_matrix[i,i] == 1:
            correct_cnt += 1
    
    return correct_cnt / res_matrix.shape[1]


"""
input:result matrix of the algorithm
output:calculate the accuracy_top5 of the model
"""
def calculate_accuracy_top5(res_matrix):
    res_matrix = res_matrix + 10
    #max_list = np.max(res_matrix,axis=0)
    
    max5_list = []
    for i in range(res_matrix.shape[1]):
        cow_list = res_matrix[:,i].copy()
        cow_list.sort()
        max5_list.append(cow_list[res_matrix.shape[0]-5])
    
    for i in range(res_matrix.shape[1]):
        res_matrix[:,i] = res_matrix[:,i]-max5_list[i]+1e-5
    
    res_matrix = res_matrix >= 0
    correct_cnt = 0
    for i in range(res_matrix.shape[1]):
        if res_matrix[i,i] == 1:
            correct_cnt += 1
    
    return correct_cnt / res_matrix.shape[1]


"""
transform the RGB pattern into gray
"""
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

"""
transform the bin array into an oct number
"""
def bin2oct(bin_list):
    summ = 0
    for i in range(len(bin_list)):
        summ += int(bin_list[len(bin_list)-1-i])*pow(2,i)
    return summ

     
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 20:56:28 2019

@author: 54164
"""


import cv2
import os
import numpy as np
import matplotlib.pyplot as plt  
from shutil import copyfile

def cut_face_into_folder(data_set,folder_name):
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    face_set = []
    for i in range(data_set.shape[0]):
        img = np.array(data_set[i,:,:],dtype='uint8')
        face = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=15, minSize=(150, 150), flags=cv2.CASCADE_SCALE_IMAGE)
        if face != ():
            print("readed {0} faces".format(i))
            x = face[0,0]
            y = face[0,1]
            w = face[0,2]
            h = face[0,3]
            #face_set.append(img[y:y+h,x:x+w].copy())
            img_new = cv2.resize(img[y:y+h,x:x+w].copy(), (200,200), interpolation = cv2.INTER_AREA )
            img_new = cv2.resize(img_new[10:190,25:175], (150,150), interpolation = cv2.INTER_AREA )
            cv2.imwrite("./{0}/{1}.jpg".format(folder_name,i),img_new)
            print("saved {0} faces in {1}".format(i,folder_name))
        else:
            print("Ooops!!!!!!!!!")
            

def reload_new_faces(train_vis_path="./face\train_vis_face",
					 train_nir_path="./face\train_nir_face",
					 test_vis_path="./face\test_vis_face",
					 test_nir_path="./face\test_nir_face",):
	# test_nir and test_vis
	test_nir = []
	test_vis = []
	for i in range(360):
		test_nir_path_all = os.path.join(test_nir_path,"{0}.jpg".format(i))
		test_vis_path_all = os.path.join(test_vis_path,"{0}.jpg".format(i))
		if os.path.exists(test_nir_path_all) and os.path.exists(test_vis_path_all):
			test_nir.append(cv2.imread(test_nir_path_all,cv2.IMREAD_GRAYSCALE))
			test_vis.append(cv2.imread(test_vis_path_all,cv2.IMREAD_GRAYSCALE))
		elif os.path.exists(test_nir_path_all):
			os.remove(test_nir_path_all)
		elif os.path.exists(test_vis_path_all):
			os.remove(test_vis_path_all)
            
	# train_vis and train_nir
	train_vis = []
	train_nir = []
	for i in range(360):
		train_vis_path_all = os.path.join(train_vis_path,"{0}.jpg".format(i))		
		if os.path.exists(train_vis_path_all):
			nir_all_exist = True
			for j in range(5):
				if os.path.exists(os.path.join(train_nir_path,"{0}.jpg".format(i*5+j))) == False:
					nir_all_exist = False
			if nir_all_exist:
				for j in range(5):
					train_nir_path_all = os.path.join(train_nir_path,"{0}.jpg".format(i*5+j))
					train_nir.append(cv2.imread(train_nir_path_all,cv2.IMREAD_GRAYSCALE))
				train_vis.append(cv2.imread(train_vis_path_all,cv2.IMREAD_GRAYSCALE))
			else:
				for j in range(5):
					train_nir_path_all = os.path.join(train_nir_path,"{0}.jpg".format(i*5+j))
					if os.path.exists(train_nir_path_all):
						os.remove(train_nir_path_all)
				os.remove(train_vis_path_all)
            
	return np.array(train_vis),np.array(train_nir),np.array(test_vis),np.array(test_nir)

def rename_pictures(train_vis_path="./train_vis_face",
					 train_nir_path="./train_nir_face",
					 test_vis_path="./test_vis_face",
					 test_nir_path="./test_nir_face",):
	# rename 
	if os.path.exists(os.path.join(train_vis_path,"new")) == False:
		os.mkdir(os.path.join(train_vis_path,"new"))
	idx = 0
	for i in range(360):
		train_vis_path_all = os.path.join(train_vis_path,"{0}.jpg".format(i))
		if os.path.exists(train_vis_path_all):
			copyfile(train_vis_path_all,os.path.join(train_vis_path,"new","{0}.jpg".format(idx)))
			idx += 1
	if os.path.exists(os.path.join(train_nir_path,"new")) == False:
		os.mkdir(os.path.join(train_nir_path,"new"))
	idx = 0
	for i in range(360):
		train_nir_path_all = os.path.join(train_nir_path,"{0}.jpg".format(i))
		if os.path.exists(train_nir_path_all):
			copyfile(train_nir_path_all,os.path.join(train_nir_path,"new","{0}.jpg".format(idx)))
			idx += 1
	if os.path.exists(os.path.join(test_vis_path,"new")) == False:
		os.mkdir(os.path.join(test_vis_path,"new"))
	idx = 0
	for i in range(360):
		test_vis_path_all = os.path.join(test_vis_path,"{0}.jpg".format(i))
		if os.path.exists(test_vis_path_all):
			copyfile(test_vis_path_all,os.path.join(test_vis_path,"new","{0}.jpg".format(idx)))
			idx += 1
	if os.path.exists(os.path.join(test_nir_path,"new")) == False:
		os.mkdir(os.path.join(test_nir_path,"new"))
	idx = 0
	for i in range(360):
		test_nir_path_all = os.path.join(test_nir_path,"{0}.jpg".format(i))
		if os.path.exists(test_nir_path_all):
			copyfile(test_nir_path_all,os.path.join(test_nir_path,"new","{0}.jpg".format(idx)))
			idx += 1
	print("rename finished!!")

    
    

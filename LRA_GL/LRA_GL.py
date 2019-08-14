# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 21:08:18 2019

@author: 54164
"""
import numpy as np
import matplotlib.pyplot as plt  
import time
import LRA_GL_utils as utils
import data_pre_process as dprocess

time_start=time.time()

train_vis,train_nir,test_vis,test_nir =utils.readPicture(folder_path="./dataset_CASIA")
#train_vis = np.load('train_vis.npy')
#train_nir = np.load('train_nir.npy')
#test_vis = np.load('test_vis.npy')
#test_nir = np.load('test_nir.npy')

pre process the pictures
dprocess.cut_face_into_folder(train_vis,"train_vis_face")
dprocess.cut_face_into_folder(train_nir,"train_nir_face")
dprocess.cut_face_into_folder(test_vis,"test_vis_face")
dprocess.cut_face_into_folder(test_nir,"test_nir_face")

train_vis_face,train_nir_face,test_vis_face,test_nir_face = dprocess.reload_new_faces()



k = test_nir_face.shape[0]
m = train_nir_face.shape[0]
Y_gl = np.hstack((np.eye(k),np.zeros([k,m])))
#Y_gl = np.hstack((np.eye(k),np.zeros([k,1000])))

probe = utils.lbp_encode(test_nir_face)
print("encoded test_nir !!!")
X = utils.lbp_encode(test_vis_face)
print("encoded test_vis !!!")
variants = utils.intro_class_variant(train_vis_face,train_nir_face)
print("extracted the varients !!!")
variants_lbp = utils.lbp_encode(variants)
print("encoded varients !!!")

X_gl = np.hstack((X,variants_lbp))
#X_gl = np.hstack((X,variants_lbp[:,0:1000]))

X_gl_mat = np.mat(X_gl)
X_gl_inv = X_gl_mat.I
X_gl_inv = np.array(X_gl_inv)

W_gl = np.dot(Y_gl,X_gl_inv)
Y_gl_hat = np.dot(W_gl,probe)

accuracy = utils.calculate_accuracy(Y_gl_hat)
accuracy_top5 = utils.calculate_accuracy_top5(Y_gl_hat)

time_end=time.time()

print('totally cost : ',time_end-time_start)




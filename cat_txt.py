import numpy as np
import cv2
import os, sys
import matplotlib.pyplot as plt
def load_annotations_2(ann_file):
    data_infos_d1 = {}
    data_infos_d2 = {}
    with open(ann_file) as f:
        samples = [x.strip().split(' ') for x in f.readlines()]
        # print(len(samples[0]))
        for i in range (len(samples)):
            j1=samples[i][0]
            j2 = samples[i][1]
    return data_infos_d1,data_infos_d2

def load_annotations_nint(ann_file):
    data_infos1 = {}
    with open(ann_file) as f:
        samples = [x.strip().split(' ') for x in f.readlines()]
        # print(len(samples[0]))
        for i in range (len(samples)):
            I=[]
            for j in range(0,50):
                I.append(np.array((samples[i][j]), dtype=np.int16))
            data_infos1[i] = I
    return data_infos1

def load_annotations_nfloat(ann_file):
    data_infos1 = {}
    with open(ann_file) as f:
        samples = [x.strip().split(' ') for x in f.readlines()]
        # print(len(samples[0]))
        for i in range(len(samples)):
            I = []
            for j in range(0, 50):
                I.append(np.array((samples[i][j]), dtype=np.float64))
            data_infos1[i] = I
    return data_infos1


img_label1,img_label2 = load_annotations_2('./weights/pass50/pixel_attention2/cluster/train_labeled.txt')
D_1 = list(img_label1.values())
D_2 = list(img_label2.values())

img_label1_I = load_annotations_nint('./weights/pass50/pixel_attention2/cluster/train_I_50.txt')
I_12 = list(img_label1_I.values())

img_label1_D = load_annotations_nfloat('./weights/pass50/pixel_attention2/cluster/train_D_50.txt')
D_13 = list(img_label1_D.values())


co=0
for i in range (len(img_label1)):
    arr=D_1[i] + ' ' + str(D_2[i])+ ' '+ '*'
    for j in range (0,50):
        arr=arr+ ' '+str(I_12[i][j])
    arr = arr+ ' ' + '*'
    for j in range (0,50):
        arr=arr+ ' '+str(D_13[i][j])
    with open(os.path.join('./weights/pass50/pixel_attention2/cluster', "train_I_D_50.txt"), "a") as f:
        f.write(arr)
        f.write("\n")
    co = co + 1

print(co)

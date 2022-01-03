# 1. Only add your code inside the function (including newly improted packages). 
#  You can design a new function and call the new function in the given functions. 
# 2. For bonus: Give your own picturs. If you have N pictures, name your pictures such as ["t3_1.png", "t3_2.png", ..., "t3_N.png"], and put them inside the folder "images".
# 3. Not following the project guidelines will result in a 10% reduction in grades

import cv2
import numpy as np
import matplotlib.pyplot as plt
import json


def stitch(imgmark, N, savepath=''): #For bonus: change your input(N=*) here as default if the number of your input pictures is not 4.
    "The output image should be saved in the savepath."
    "The intermediate overlap relation should be returned as NxN a one-hot(only contains 0 or 1) array."
    "Do NOT modify the code provided."
    imgpath = [f'./images/{imgmark}_{n}.png' for n in range(1,N+1)]
    imgs = []
    for ipath in imgpath:
        img = cv2.imread(ipath)
        imgs.append(img)
    "Start you code here"

    overlap_arr = check_onehot(imgs, N)
    img = stitch_background(imgs[1],imgs[0])
    for x in range(1,N-1):
            img = stitch_background(img, imgs[x+1])
    
    cv2.imwrite(savepath, img)   
    return overlap_arr


def check_onehot(imgs,N):
    """ Quick Fucntion to check One-Hot:"""
    "Quicker to check with lesser features to compare in Sift and return array"

    import numpy
    onehot = numpy.zeros((N,N))
    print("Start One-Hot Calculation")
    for x in range(N):
        for y in range(N):
                if x == y:
                    onehot[x][y] = 1;
                    continue
                img1_gray = imgs[x]
                img2_gray = imgs[y]
                sift = cv2.xfeatures2d.SIFT_create(100)    
                kp_1, des_1 = sift.detectAndCompute(img1_gray, None)
                kp_2, des_2 = sift.detectAndCompute(img2_gray, None)
                good_match = []
                for i in range(len(des_1)):
                    ssd = []
                    for j in range(len(des_2)):
                        ssd.append(((sum((des_1[i] - des_2[j])**2)),j))
                    ssd.sort(key=lambda tup: tup[0])
                    if (ssd[0][0]/ssd[1][0])<0.8:
                        good_match.append((i,ssd[0][1]))
                #print(x,y, len(good_match))
                if(len(good_match) > 100*0.18):
                    onehot[x][y] = onehot[y][x] = 1
                else:
                    onehot[x][y] = onehot[y][x] = 0
                y += 1
    print("Done One-Hot Calculation")
    return onehot


def stitch_background(img1, img2):
    
    img1_gray = img1
    img2_gray = img2
    sift = cv2.xfeatures2d.SIFT_create(500)
    kp_1, des_1 = sift.detectAndCompute(img1_gray, None)
    kp_2, des_2 = sift.detectAndCompute(img2_gray, None)
    good_match = []

    for i in range(len(des_1)):
        ssd = []
        for j in range(len(des_2)):
            ssd.append(((sum((des_1[i] - des_2[j])**2)),j))
        ssd.sort(key=lambda tup: tup[0])
        if (ssd[0][0]/ssd[1][0])<0.7:
            good_match.append((i,ssd[0][1]))

    src_index = [a_tuple[0] for a_tuple in good_match]
    dst_index = [a_tuple[1] for a_tuple in good_match]

    src = np.float32([kp_1[m].pt for m in src_index]).reshape(-1, 1, 2)
    dest = np.float32([kp_2[m].pt for m in dst_index]).reshape(-1, 1, 2)
    
    M, mask = cv2.findHomography(src, dest, cv2.RANSAC, 5.0)
    #print("matrix", M) 
    #print("mask",mask)
    
    from numpy import float32, concatenate, array, int32
    h1,w1,d = img1.shape
    h2,w2,d = img2.shape
    pts1 = float32([[0,0],[0,h1],[w1,h1],[w1,0]]).reshape(-1,1,2)
    pts2 = float32([[0,0],[0,h2],[w2,h2],[w2,0]]).reshape(-1,1,2)
    pts1_ = cv2.perspectiveTransform(pts1, M)
    pts = concatenate((pts2, pts1_), axis=0)
    [xmin, ymin] = int32(pts.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = int32(pts.max(axis=0).ravel() + 0.5)
    t = [-xmin,-ymin]
    Mt = array([[1,0,t[0]],[0,1,t[1]],[0,0,1]]) # translate
    
    result = cv2.warpPerspective(img1, Mt.dot(M), (xmax-xmin, ymax-ymin))
    result[t[1]:h2+t[1],t[0]:w2+t[0]] = img2

    return result


if __name__ == "__main__":
    #task2
    overlap_arr = stitch('t2', N=4, savepath='task2.png')
    with open('t2_overlap.txt', 'w') as outfile:
        json.dump(overlap_arr.tolist(), outfile)
    #bonus
    overlap_arr2 = stitch('t3', N=3, savepath='task3.png')
    with open('t3_overlap.txt', 'w') as outfile:
        json.dump(overlap_arr2.tolist(), outfile)

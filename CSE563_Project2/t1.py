#Only add your code inside the function (including newly improted packages)
# You can design a new function and call the new function in the given functions. 
# Not following the project guidelines will result in a 10% reduction in grades

import cv2
import numpy as np
import matplotlib.pyplot as plt


def stitch_background(img1, img2, savepath=''):
    "The output image should be saved in the savepath."
    "Do NOT modify the code provided."
    
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
 
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
    
    for x in range(h2):
        for y in range(w2):
            if sum(result[t[1]+x,t[0]+y]) + 70 < sum(img2[x,y]):
                result[t[1]+x,t[0]+y] = img2[x,y]
            
    cv2.imwrite(savepath, result)

    return

    
if __name__ == "__main__":
    img1 = cv2.imread('./images/t1_1.png')
    img2 = cv2.imread('./images/t1_2.png')
    savepath = 'task1.png'
    stitch_background(img1, img2, savepath=savepath)


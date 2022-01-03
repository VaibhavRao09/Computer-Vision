#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 03:20:54 2021

@author: saloni
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2
import os, json, glob
import argparse
import face_recognition
import matplotlib.pyplot as plt

import numpy as np
from scipy.spatial.distance import cdist


class KMeans(object):

    def __init__(self, n_clusters):
        self.n_clusters = n_clusters

    def fit(self, X, iter_max=100):
        I = np.eye(self.n_clusters)
        np.random.seed(43)
        centers = X[np.random.choice(len(X), self.n_clusters, replace=False)]
        for _ in range(iter_max):
            prev_centers = np.copy(centers)
            D = cdist(X, centers)
            cluster_index = np.argmin(D, axis=1)
            cluster_index_onehot = I[cluster_index]
            centers = np.sum(X[:, None, :] * cluster_index_onehot[:, :, None], axis=0) / np.sum(cluster_index_onehot, axis=0)[:, None]
            #if np.allclose(prev_centers, centers):
            if np.all(prev_centers == centers):
                print("BRAKING")
                break
        self.centers = centers
        self.label = cluster_index


def read_image(img_path):
    """Reads an image into memory as a grayscale array.
    """
    img = cv2.imread(img_path)#, cv2.IMREAD_GRAYSCALE)
    return img


def Detection(image, K):

    encodings = []
    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
    #faceCascade = cv2.CascadeClassifier("haarcascade_profileface.xml")
    
    for img in image:
        #X_face_locations = face_recognition.face_locations(img[1])

        faces = faceCascade.detectMultiScale(img[1], 1.1, 4)
        (x, y, width, height) = faces[0]
        #print({"iname": img[0], "bbox": [int(x), int(y), int(width), int(height)]})
        y = top = y
        left = x
        bottom = y + height
        right = x + width
        #faces = faceCascade.detectMultiScale(img[1], 1.09, 4)
        face_enc = face_recognition.face_encodings(img[1], known_face_locations=[(top, right, bottom, left)])
        #rgb = cv2.cvtColor(img[1], cv2.COLOR_RGB2GRAY)
        #print((X_face_locations)[0][1])
        #img_face = rgb[X_face_locations[0]:X_face_locations[2], X_face_locations[3]:X_face_locations[1]]
        #plt.imshow(img_face)
        encodings.append(face_enc[0])        
        
    #X_TRAIN = np.array(encodings[:][1])
    encodings = np.array(encodings)
    kmean = KMeans(int(K))
    kmean.fit(encodings,50)
    print(kmean)
    
    plt.scatter(encodings[:, 0], encodings[:, 1], s=50);
    plt.scatter(kmean.centers[:, 0], kmean.centers[:, 1], c='black', s=200, alpha=0.5);
    
    labels = kmean.label
    #pred = kmean.predict(encodings)
    #cluster = [a_tuple[0] for a_tuple in image]
    #cluster = zip(cluster, pred)

    result = []
    for i in range(int(K)):
        elements = (np.array(image)[:,0][labels == i]).tolist()
        elements.sort(key=lambda x: int(x.split('.')[0]))
        #elements.sort(key=lambda x: int(x.split('_')[1]))
        result.append({"cluster_no": i, "elements": elements})
    print(result)
    
        #print(cluster[1]np.where(pred == i)[0]])        
        
        #for (top, right, bottom, left) in (X_face_locations):
        	# draw the predicted face name on the image
            #y = top
            #x = left
            #height = bottom - y
            #width = right - x            
            #result.append({"iname": img[0], "bbox": [x, y, width, height]})
            #result.append({"iname": img[0], "bbox": [int(top), int(right), int(bottom), int(left)]})
            #img_face = rgb[top:bottom, left:right]
            #cv2.rectangle(img[1], (left, top), (right, bottom), (0, 255, 0), 2)        
            #cv2.imshow("Image", img[1])
            #cv2.waitKey(0)
        
    return result
    

def save_results(bbox, result_dir):
    """
    Donot modify this code
    """
    results = bbox
    with open(os.path.join(result_dir, 'clusters.json'), "w") as file:
        json.dump(results, file)


def main():
    #args = parse_args()
    parser = argparse.ArgumentParser(description="CSE 573 project 3")
    parser.add_argument(
        "path", type=str, nargs='?', default="faceCluster_5",
        help="path to the image folder to be used for face-cluster")
    args = parser.parse_args()

    K = args.path.split('_')[1]
    face_images = []

    all_face_imgs = glob.glob(args.path+ "/*")
    #all_face_imgs.sort()
    
    for each_face in all_face_imgs :
        file_name = "{}".format(os.path.split(each_face)[-1])
        face_images.append([file_name, read_image(each_face)])
    
    results = Detection(face_images, K)

    save_results(results, ".")
    

if __name__ == "__main__":
    main()


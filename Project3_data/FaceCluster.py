#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2
import os, json, glob
import argparse
import face_recognition
import numpy as np

#Calculate Eucledian Distance between 2 points
def eucdist(X, centers):
    dist = np.zeros(X.shape[0])
    for j in range(len(centers)):
        dist_linalg = np.linalg.norm(X - centers[j,:],axis=1)
        dist = np.vstack((dist, dist_linalg))
    dist = np.transpose(dist[1:])
    return dist

class KMeans(object):

    def __init__(self, num_clusters):
        self.num_clusters = num_clusters
    
    #Train Function to train and get labels
    def fit(self, X, maxiter=100):
        I = np.eye(self.num_clusters)
        np.random.seed(43)
        #Choose random 'UNIQUE' points from the centres.
        centers = X[np.random.choice(len(X), self.num_clusters, replace=False)]
        for _ in range(maxiter):
            prev_centers = np.copy(centers)
            dist = eucdist(X, centers)
            cluster_index = np.argmin(dist, axis=1)
            #Convert to one-hot
            cluster_index_onehot = I[cluster_index]
            #Find new centres from the mean of the points
            centers = np.sum(X[:, None, :] * cluster_index_onehot[:, :, None], axis=0) / np.sum(cluster_index_onehot, axis=0)[:, None]
            if np.all(prev_centers == centers):
                break
        #Used for visualizations.
        self.centers = centers
        #The Last cluster indexes will be our labels
        self.label = cluster_index
        


def read_image(img_path):
    img = cv2.imread(img_path)
    return img


def Detection(image, K):

    encodings = []
    cropped_face = []
    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
    
    for img in image:
        faces = faceCascade.detectMultiScale(img[1], 1.1, 4)
        if len(faces) == 0:
            #print("noface")
            faces = [(1,1,1,1)]
        (x, y, width, height) = faces[0]
        top = y
        left = x
        bottom = y + height
        right = x + width
        
        face_enc = face_recognition.face_encodings(img[1], known_face_locations=[(top, right, bottom, left)])
        
        face = img[1][top:bottom, left:right]
        cropped_face.append(face)            
        encodings.append(face_enc[0])  
    
    encodings = np.array(encodings)
    kmean = KMeans(K)
    kmean.fit(encodings)
    labels = kmean.label

    result = []
    for i in range(K):
        
        elements = (np.array(image)[:,0][labels == i]).tolist()
        face_crop = (np.array(cropped_face)[labels == i]).tolist()
        
        for x in range(len(face_crop)): 
            face_crop[x] = cv2.resize(face_crop[x],(120,120))
            
        cv2.imwrite('cluster_'+str(i)+'.jpg', np.hstack(face_crop))
        
        #Sorting as Image name is 1.jpg (Sort by int(1))
        elements.sort(key=lambda x: int(x.split('.')[0]))
        result.append({"cluster_no": i, "elements": elements})
    
    return result
    

def save_results(cluster, result_dir):
    results = cluster
    with open(os.path.join(result_dir, 'clusters.json'), "w") as file:
        json.dump(results, file)


def main():
    parser = argparse.ArgumentParser(description="CSE 573 project 3")
    parser.add_argument(
        "path", type=str, nargs='?', default="./faceCluster_5",
        help="path to the image folder to be used for face-cluster")
    
    args = parser.parse_args()
    '''The path should be of the format "/Users/vaibhav/faceCluster_K". 
    Should not have a trailing '/'
    Extract K by splitting with '_' and taking the last split i.e. -1'''
    K = args.path.split('_')[-1]

    face_images = []

    all_face_imgs = glob.glob(args.path+ "/*")
    
    for each_face in all_face_imgs :
        file_name = "{}".format(os.path.split(each_face)[-1])
        face_images.append([file_name, read_image(each_face)])
    
    results = Detection(face_images, int(K))

    save_results(results, ".")
    

if __name__ == "__main__":
    main()


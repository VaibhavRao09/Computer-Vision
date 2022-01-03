#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2
import os, json, glob
import argparse


def read_image(img_path):
    """Reads an image into memory as a grayscale array.
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    return img


def Detection(image):

    result = []
    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
    
    for img in image:
        faces = faceCascade.detectMultiScale(img[1], 1.06, 5)
        
        for (x, y, width, height) in (faces):
            result.append({"iname": img[0], "bbox": [int(x), int(y), int(width), int(height)]})
        
    return result
    

def save_results(bbox, result_dir):
    results = bbox
    with open(os.path.join(result_dir, 'results.json'), "w") as file:
        json.dump(results, file)


def main():
    parser = argparse.ArgumentParser(description="CSE 573 project 3 Task1")
    parser.add_argument(
        "path", type=str, nargs='?', default="./Validation folder",
        help="path to the image folder to be used for face-detection")
    args = parser.parse_args()

    face_images = []

    all_face_imgs = glob.glob(args.path+ "/images/*")
    
    ''' Spliting file name with _ as file name is img_1.jpg. Extracting 1 with 
    the below line of code to sort the files numerically. If this throws error
    please comment this line below'''
    all_face_imgs.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    
    for each_face in all_face_imgs :
        file_name = "{}".format(os.path.split(each_face)[-1])
        face_images.append([file_name, read_image(each_face)])
    
    results = Detection(face_images)

    save_results(results, ".")
    

if __name__ == "__main__":
    main()


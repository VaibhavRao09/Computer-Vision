"""
Character Detection

The goal of this task is to implement an optical character recognition system consisting of Enrollment, Detection and Recognition sub tasks

Please complete all the functions that are labelled with '# TODO'. When implementing the functions,
comment the lines 'raise NotImplementedError' instead of deleting them.

Do NOT modify the code provided.
Please follow the guidelines mentioned in the project1.pdf
Do NOT import any library (function, module, etc.).
"""

import argparse
import json
import os
import glob
import cv2
import numpy as np

def read_image(img_path, show=False):
    """Reads an image into memory as a grayscale array.
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if show:
        show_image(img)

    return img

def show_image(img, delay=1000):
    """Shows an image.
    """
    cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('image', img)
    cv2.waitKey(delay)
    cv2.destroyAllWindows()

def parse_args():
    parser = argparse.ArgumentParser(description="cse 473/573 project 1.")
    parser.add_argument(
        "--test_img", type=str, default="./data/test_img.jpg",
        help="path to the image used for character detection (do not change this arg)")
    parser.add_argument(
        "--character_folder_path", type=str, default="./data/characters",
        help="path to the characters folder")
    parser.add_argument(
        "--result_saving_directory", dest="rs_directory", type=str, default="./",
        help="directory to which results are saved (do not change this arg)")
    args = parser.parse_args()
    return args

import copy 

def ocr(test_img, characters):
    """Step 1 : Enroll a set of characters. Also, you may store features in an intermediate file.
       Step 2 : Use connected component labeling to detect various characters in an test_img.
       Step 3 : Taking each of the character detected from previous step,
         and your features for each of the enrolled characters, you are required to a recognition or matching.

    Args:
        test_img : image that contains character to be detected.
        characters_list: list of characters along with name for each character.

    Returns:
    a nested list, where each element is a dictionary with {"bbox" : (x(int), y (int), w (int), h (int)), "name" : (string)},
        x: row that the character appears (starts from 0).
        y: column that the character appears (starts from 0).
        w: width of the detected character.
        h: height of the detected character.
        name: name of character provided or "UNKNOWN".
        Note : the order of detected characters should follow english text reading pattern, i.e.,
            list should start from top left, then move from left to right. After finishing the first line, go to the next line and continue.
        
    """
    # TODO Add your code here. Do not modify the return and input arguments
    
    '''Had to create a copy since Sift doesn't seem to work on bin images'''
    original_img = copy.deepcopy(test_img)

    feature = enrollment(characters)
    
    bbox = detection(test_img)
    
    result = recognition(original_img, feature, bbox)
    
    return result    
    #raise NotImplementedError

def enrollment(characters):
    """ Args:
        You are free to decide the input arguments.
    Returns:
    You are free to decide the return.
    """
    # TODO: Step 1 : Your Enrollment code should go here.

    sift = cv2.SIFT_create()
        
    feature = []
    for x in characters:
        x[1] = cv2.resize(x[1],(30,30))
        cv2.imwrite('./Res/subpixel_'+x[0]+'.png',x[1])
        keypoints_sift, descriptors = sift.detectAndCompute(x[1], None)
        feature.append([x[0], descriptors])
     
    ''' return the descriptors computed via SIFT and label information back'''
    return feature
    #raise NotImplementedError

  

def detection(test_img):
    """ 
    Use connected component labeling to detect various characters in an test_img.
    Args:
        You are free to decide the input arguments.
    Returns:
    You are free to decide the return.
    """
    # TODO: Step 2 : Your Detection code should go here.

    '''Covert image to binary for easy traversal and thresholding'''
    
    test_img[test_img <= 127] = 1    #Treat as foreground pixel
    test_img[test_img > 127] = 0   #Treat as background pixel

    #initialize 
    row,col = test_img.shape
    cc_label_mat = np.zeros((row,col))
    curr_label = 0

    def mark_label(i,j):
        '''A Recursive function to traverse and mark all
        connected points with the current label till we find a 
        boundary and return'''
        
        if i < 0 or i == row:   # handle corner case of 1st row and last row(Out of bound)
            return  
        if j < 0 or j == col:   # handle corner case of 1st col and last col(Out of bound)
            return
        '''Condition if it is not a foreground pixel or we have already marked 
           this locatopn before. we can return, Nothing to-do
           '''
        if (not test_img[i][j]) or cc_label_mat[i][j]: 
            return
        
        # 4 - directional recursive traversal        
        dir_x = [+1,0,-1,0]
        dir_y = [0,+1,0,-1]
        
        cc_label_mat[i][j] = curr_label ### Mark Cell with our current label
        
        for quad_direct in range(4):
            #call recursively with next pixel
            mark_label( i + dir_x[quad_direct], j + dir_y[quad_direct] )
    '''  ****** FUNCTION mark_label END ****** '''
      
    for x in range(row):
        for y in range(col):
            if( (not cc_label_mat[x][y]) and test_img[x][y] ):
                curr_label += 1
                mark_label(x,y)
                
    bbox = []
    for cc in range(0, curr_label):
        temp_r,temp_c = np.where(cc_label_mat == cc+1)
        y_min = int(min(temp_r))
        x_min = int(min(temp_c))
        y_max = int(max(temp_r))
        x_max = int(max(temp_c))
        w = y_max - y_min
        h = x_max - x_min
        bbox.append([x_min,y_min,h,w])
        
    return bbox
    #raise NotImplementedError

def recognition(original_img, feature, bbox):
    """ 
    Args:
        You are free to decide the input arguments.
    Returns:
    You are free to decide the return.
    """
    # TODO: Step 3 : Your Recognition code should go here.
    
    sift = cv2.SIFT_create()

    descriptors_testimg = []
    result = []
    cropped = []

    for i in bbox:
        result.append({"bbox": i, "name": "UNKNOWN"})
        x,y,h,w = i
        cropped.append(original_img[y-1:y + w +1, x-1:x + h+1])
        cropped_resize = cv2.resize(cropped[-1],(30,30))
        kp, des = sift.detectAndCompute(cropped_resize,None)
        descriptors_testimg.append(des)
        cv2.rectangle(original_img, (x, y), (x+h, y+w), (255,0 , 0), 2)

    
    cv2.imwrite('exp.jpg', original_img)
    ''' For loop to test all Cropped images from test_img and the extracted 
    features as above with the descriptor of each character that was enrolled.
    Later save the max enrolled character match less than threshold (320 selected)
    based on trial and error and save the label information in 'result'(name)'''

    bf = cv2.BFMatcher()
    i=0    
    for des in descriptors_testimg:
        max_match = []

        if (des is not None):
            for char in feature:
                matches = bf.match(des,char[1])
                sorted(matches, key = lambda x:x.distance)                
                max_match.append([matches[0].distance, char[0]])

        max_match = sorted(max_match ,key=lambda l:l[0])
        if (max_match) and max_match[0][0] < 500:
            result[i]['name'] = max_match[0][1]
        i += 1
                
    return result        
    #raise NotImplementedError


def save_results(coordinates, rs_directory):
    """
    Donot modify this code
    """
    results = coordinates
    with open(os.path.join(rs_directory, 'results.json'), "w") as file:
        json.dump(results, file)


def main():
    args = parse_args()
    
    characters = []

    all_character_imgs = glob.glob(args.character_folder_path+ "/*")
    
    for each_character in all_character_imgs :
        character_name = "{}".format(os.path.split(each_character)[-1].split('.')[0])
        characters.append([character_name, read_image(each_character, show=False)])

    test_img = read_image(args.test_img)

    results = ocr(test_img, characters)

    save_results(results, args.rs_directory)


if __name__ == "__main__":
    main()

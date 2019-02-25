import os
import cv2
import numpy as np
import sqlite3
import dlib
from sklearn.model_selection import train_test_split

class DataLoader:
    '''
    Connects to the database and gets data is useful form
    
    Methods implemented in this class are -
    
    # def get_raw_data(self, image_path, database_path)
    # def process_raw_data(self)
    # def intersection_over_union(self, boxA, boxB)
    # def selective_search(self, image)
    # def region_of_proposals(self, image)
    # def split_data(self, data)
    
    '''
    
    # Path variables
    images_path = '../input/aflw/AFLW/aflw-images/data/flickr/'
    database_path = '../input/aflw/AFLW/aflw-db/data/aflw.sqlite'
    
    
    
    def __init__(self, image_path, database_path):
        self.image_path = image_path
        self.database_path = database_path

    def get_raw_data(self, image_path, database_path):
        '''
        Opens the database and gets the data in required form
        '''
        images = []
        landmarks = []
        visibility = []
        pose = []
        gender = []

        counter = 1

        conn = sqlite3.connect(database_path)
        c = conn.cursor()

        select_string = "faceimages.filepath, faces.face_id, facepose.roll, facepose.pitch, facepose.yaw, facerect.x, facerect.y, facerect.w, facerect.h, facemetadata.sex"
        from_string = "faceimages, faces, facepose, facerect, facemetadata"
        where_string = "faces.face_id = facepose.face_id and faces.file_id = faceimages.file_id and faces.face_id = facerect.face_id and faces.face_id = facemetadata.face_id"
        query_string = "SELECT " + select_string + " FROM " + from_string + " WHERE " + where_string

        print("Processing", end = '')
        for row in c.execute(query_string):

            #Using our specific query_string, the "row" variable will contain:
            # row[0] = image path
            # row[1] = face id
            # row[2] = roll
            # row[3] = pitch
            # row[4] = yaw
            # row[5] = face coord x
            # row[6] = face coord y
            # row[7] = face width
            # row[8] = face heigh
            # row[9] = sex

            input_path = image_path + str(row[0])
        
            if(os.path.isfile(input_path)  == True):
                image = cv2.imread(input_path, 1)
                images.append(image)
                
                #Image dimensions
                image_h, image_w, image_channels = image.shape
                
                #Roll, pitch and yaw
                roll   = row[2]
                pitch  = row[3]
                yaw    = row[4]
                pose.append(np.asarray([roll, pitch, yaw]))
                
                #Face rectangle coords
                #face_x = row[5]
                #face_y = row[6]
                #face_w = row[7]
                #face_h = row[8]
                
                #Gender
                sex = (1 if row[9] == 'm' else 0)
                gender.append(sex)        
                
                
                # Gets the landmarks corresponding to a particular image id
                select_str = "coords.feature_id, coords.x, coords.y"
                from_str = "featurecoords coords"
                where_str = "coords.face_id = {}".format(row[1])
                query_str = "SELECT " + select_str + " FROM " + from_str + " WHERE " + where_str
                lm = np.zeros((21,2)).astype(np.float32)
                v = np.zeros((21)).astype(np.int32)
                
                c2 = conn.cursor()
                
                for q in c2.execute(query_str):
                    lm[q[0]-1][0] = q[1]
                    lm[q[0]-1][1] = q[2]
                    v[q[0]-1] = 1
                
                lm = lm.reshape(42)
                
                landmarks.append(lm)
                visibility.append(v)
        
                c2.close()
                
                print(".", end = '')
                counter = counter + 1
                
                # For testing purpose
                if counter == 10:
                    break
            else:
                #raise ValueError('Error: I cannot find the file specified: ' + str(input_path))
                continue
        
        c.close()
        
        return images, landmarks, visibility, pose, gender
    
    
    def process_raw_data(self):
        '''
        # Processes images and creates region proposals
        # Converts everything to numpy arrays
        '''
        
        # implement selective search
        # implement iou
        # return everything as numpy array
        
        
        
        return
    
    def intersection_over_union(self,boxA, boxB):
        '''
        Calculates intersection over union of two bounding boxes
        '''
        top_left_x1, top_left_y1, bottom_right_x1, bottom_right_y1 = boxA
        top_left_x2, top_left_y2, bottom_right_x2, bottom_right_y2 = boxB

        intersect_top_left_x = max(top_left_x1, top_left_x2)
        intersect_top_left_y = max(top_left_y1, top_left_y2)
        intersect_bottom_right_x = min(bottom_right_x1, bottom_right_x2)
        intersect_bottom_right_y = min(bottom_right_y1, bottom_right_y2)

        intersect_area = (intersect_bottom_right_x - intersect_top_left_x + 1) * (intersect_bottom_right_y - intersect_top_left_y + 1)

        total_area = (
            (bottom_right_x1 - top_left_x1 + 1) * (bottom_right_y1 - top_left_y1 + 1)
            + (bottom_right_x2 - top_left_x2 + 1) * (bottom_right_y2 - top_left_y2 + 1)
            - intersect_area
            )

        iou = float(intersect_area) / float(total_area + 0.0)

        return iou
    
    def selective_search(self,image):
        '''
        Uses dlib built in find_candidate_object_location function to
        give face location proposals
        '''
        rects = []
        dlib.find_candidate_object_locations(image, rects, min_size = 500)
        
        return rects
    
    def region_of_proposals(self, image):
        '''
        Return parts of images which have an iou >= 0.50
        '''
        return
    
    
    
    def split_data(self, data):
        '''
        Splits data in train and validation sets
        '''
        images, face, landmarks, visibility, pose, gender = data
        
        x_train, x_test, y_train_face, y_test_face, y_train_landmarks, \
        y_test_landmarks, y_train_visibility, y_test_visibility, \
        y_train_pose, y_test_pose, y_train_gender, y_test_gender = \
        train_test_split(images, face, landmarks, visibility, pose, gender, test_size = 0.10, shuffle = True)
        
        train_data = x_train, y_train_face, y_train_landmarks, y_train_visibility, y_train_pose, y_train_gender
        test_data = x_test, y_test_face, y_test_landmarks, y_test_visibility, y_test_pose, y_test_gender
        
        return (train_data, test_data)
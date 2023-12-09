# Utilities for object detector.

import numpy as np
import sys
import tensorflow as tf
import os
from threading import Thread
from datetime import datetime
import cv2
from handtracking.utils import label_map_util
from collections import defaultdict
from PIL import Image
from ultralytics import YOLO

detection_graph = tf.compat.v1.Graph()
sys.path.append("..")

# score threshold for showing bounding boxes.
_score_thresh = 0.27

MODEL_NAME = 'handtracking/hand_inference_graph'
# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
NEW_PATH_TO_CKPT = MODEL_NAME + '/save_model_format.pb'
# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join(MODEL_NAME, 'hand_label_map.pbtxt')

NUM_CLASSES = 1
# load label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(
    label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)




# Load a pretrained YOLOv8n model
#model = YOLO('yolov8n.pt')

# Load a frozen inference graph into memory

def load_inference_graph():
    # Load frozen TensorFlow model into memory
    print("> ====== Loading HAND frozen graph into memory")

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.compat.v1.GraphDef()
        with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    print(">  ====== Hand Inference graph loaded.")
    return detection_graph

# param: hands is [[(x1,y1), (x2,y2))], [(x1,y1), (x2,y2)]]
def check_object_in_between(hands,image_np):
    # left hand 
    x1 = hands[0][0][0] # left 
    x2 = hands[1][0][0] # right 
    y1 = hands[0][0][1] # top 
    y2 = hands[1][0][1] # bottom 
    #midpoint1 = (int((x1+y1)/2), int((x2+y2)/2)) 
    # right hand 
    x3 = hands[0][1][0] # left 
    x4 = hands[1][1][0] # right 
    y3 = hands[0][1][1] # top 
    y4 = hands[1][1][1] # bottom 
    top = max(y1,y3) # top coord
    bottom = min(y2,y4) # bottom coord 
    left = x1 
    right = x4 
    midpoint2 = (int((x1+y1)/2), int((x2+y2)/2)) 
    # put point in segment anything 
    w=100
    #im1 = image_np[midpoint1[1]:midpoint2[1],midpoint1[0]-w:midpoint1[0]+w 
    image = Image.fromarray(image_np) 
    #im1 = image.crop((midpoint1[0]-w,midpoint1[1],midpoint1[0]+w,midpoint2[ 1]))
    im1 = image.crop((min(left,right),max(top,bottom),min(right,left),max(bottom,right)))
    im1.save("object.jpg")  
    #prediction = model.predict('object.jpg', save=True, imgsz=320, conf=0.5)   
    # ADD EMBEDDING HERE 
    #return "keys"

# Draw the detected bounding boxes on the images
# You can modify this to also draw a label.
def draw_box_on_image(num_hands_detect, score_thresh, scores, boxes, im_width, im_height, image_np):
    #hands_found =0
    hands = []
    for i in range(num_hands_detect):
        if (scores[i] > score_thresh):
            (left, right, top, bottom) = (boxes[i][1] * im_width, boxes[i][3] * im_width,
                                          boxes[i][0] * im_height, boxes[i][2] * im_height)
            p1 = (int(left), int(top))
            p2 = (int(right), int(bottom))
            hands.append([p1,p2]) 
            cv2.rectangle(image_np, p1, p2, (77, 255, 9), 3, 1)
    if len(hands)==2:   
        #check_object_in_between(hands,image_np) 
        return True 
    return False # no hands detected 
     # if hands are still detected but object is not 
# Show fps value on image.
def draw_fps_on_image(fps, image_np):
    cv2.putText(image_np, fps, (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (77, 255, 9), 2)

# Actual detection .. generate scores and bounding boxes given an image
def detect_objects(image_np, detection_graph):
    # Definite input and output Tensors for detection_graph
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    image_np_expanded = np.expand_dims(image_np, axis=0)

    with detection_graph.as_default():
        with tf.compat.v1.Session() as sess:
            (boxes, scores, classes, num) = sess.run(
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})
     
    return np.squeeze(boxes), np.squeeze(scores)


# Code to thread reading camera input.
# Source : Adrian Rosebrock
# https://www.pyimagesearch.com/2017/02/06/faster-video-file-fps-with-cv2-videocapture-and-opencv/
class WebcamVideoStream:
    def __init__(self, src, width, height):
        # initialize the video camera stream and read the first frame
        # from the stream
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        (self.grabbed, self.frame) = self.stream.read()

        # initialize the variable used to indicate if the thread should
        # be stopped
        self.stopped = False

    def start(self):
        # start the thread to read frames from the video stream
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        # keep looping infinitely until the thread is stopped
        while True:
            # if the thread indicator variable is set, stop the thread
            if self.stopped:
                return

            # otherwise, read the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        # return the frame most recently read
        return self.frame

    def size(self):
        # return size of the capture device
        return self.stream.get(3), self.stream.get(4)

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True

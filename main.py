
from __future__ import division
import argparse
import datetime
import imutils
import time
import cv2
import numpy as np
from lib.mouse import Mouse
from lib.polygon import drawQuadrilateral
from lib.user_interaction import getPerpectiveCoordinates
from lib.heatmap import Heatmap
from lib.coordinate_transform import windowToFieldCoordinates
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import zipfile


from collections import defaultdict
from io import StringIO
from PIL import Image
import cv2

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util



sys.path.append("..")

# Path to frozen detection graph. This is the actual model that is used for the object detection.
# Note: Model used for SSDLite_Mobilenet_v2
PATH_TO_CKPT = 'frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = 'label_map.pbtxt'

NUM_CLASSES = 90


detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)



ap = argparse.ArgumentParser()
#ap.add_argument("-v", "--video", help="path to the video file")
ap.add_argument("-fw", "--fieldwidth", type=int, default=280, help="top-view field width")
ap.add_argument("-fh", "--fieldheight", type=int, default=334, help="top-view field height")
args = vars(ap.parse_args())

print(args)



# init frameCount and create mouse object
frameCount = 0
mouse = Mouse()



# get top-view field dimensions
resultWidth = args["fieldwidth"]
resultHeight = args["fieldheight"]
padding = 20

# minimum area of detected object(s)
objectMinArea = resultWidth*0.1 * resultHeight*0.2

# create a black image/frame where the top-view field will be drawn
field = np.zeros((resultHeight + padding*2,resultWidth + padding*2,3), np.uint8)

# top-view rectangle coordinates
(xb1, yb1) = (padding, padding)
(xb2, yb2) = (padding + resultWidth, padding)
(xb3, yb3) = (padding + resultWidth, padding + resultHeight)
(xb4, yb4) = (padding, padding + resultHeight)

# draw the 2D top-view field
drawQuadrilateral(field, [(xb1, yb1), (xb2, yb2), (xb3, yb3), (xb4, yb4)], 0, 255, 0, 2)

# crea heatmap object
heatmap = Heatmap(field, resultWidth, resultHeight)

#from tf-------------------------------------------------------------------------------
filename = 'soccer1.mp4'
cap = cv2.VideoCapture(filename) 



# Running the tensorflow session
with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
    frameCount = 0
    while (True):
        ret, frame = cap.read()
        frameCount += 1
        if ret:
            # resize the frame, to lessen the burden on CPU
            frame = imutils.resize(frame, width=800)
            height = frame.shape[0]
            width = frame.shape[1]
        if not ret:
            break
        # freeze first frame util user provides the area of the field
            # (4 points should be given by mouse clicks)
        if frameCount == 1:
            coords = getPerpectiveCoordinates(frame, 'frame', mouse)
        
        # draw perspective field
        drawQuadrilateral(frame, coords, 0, 255, 0, 2)
        

	    
	         

        if frameCount % 1 == 0:
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            frame_expanded = np.expand_dims(frame, axis=0)
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image where a particular object was detected.
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            # Actual detection.
            (boxes, scores, classes, num_detections) = sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: frame_expanded})
            # Visualization of the results of a detection.
            print(len(boxes[0]))
            vis_util.visualize_boxes_and_labels_on_image_array(
                frame,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=3,
                min_score_thresh=0.6)
            
            frame_number = frameCount
            loc = {}

           
            
            for n in range(len(boxes[0])):
                x = 0.0
                y = 0.0
                w = 0.0
                h = 0.0
                if scores[0][n] > 0.60:
                    # Calculate position
                    ymin = int(boxes[0][n][0] * height)
                    xmin = int(boxes[0][n][1] * width)
                    ymax = int(boxes[0][n][2] * height)
                    xmax = int(boxes[0][n][3] * width)
                    
                    x = xmin
                    y = ymin
                    w = xmax - xmin
                    h = ymax - ymin
                
                basePoint = (int(x + (w/2)), (y + h))
                
                # get the top-view relative coordinates
                (xbRel, ybRel) = heatmap.getPosRelativeCoordinates(basePoint, coords)

                if xbRel < 0 or xbRel > resultWidth or ybRel < 0 or ybRel > resultHeight:
                    print ("Skipped a contour, not a cool contour")
                else:
                    # draw rectangle around the detected object and a red point in the center of its base
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.circle(frame, basePoint, 3, (0, 0, 255), 2)
                    cv2.imshow('frame',frame)
                    
                    # get the top-view absolute coordinates
                    (xb, yb) = heatmap.getPosAbsoluteCoordinates((xbRel, ybRel), (xb1, yb1))
                    # draw overlayed opacity circle every 5 frames
                    if frameCount % 5 == 0:
                        heatmap.drawOpacityCircle((xb, yb), 255, 0, 0, 0, 15)
                        
            # display all windows
            cv2.imshow('frame',frame)
            #cv2.imshow('thresh',thresh)
            cv2.imshow('field',field)
            def leftClickDebug(event, x, y, flags, param):
                if event == cv2.EVENT_LBUTTONDOWN:
                    resultCoord = windowToFieldCoordinates((x, y), coords, resultWidth, resultHeight)
                    print ("Coordinates to real coordinates", resultCoord)
            # UNCOMMENT IF YOU WANT TO DEBUG
            # cv2.setMouseCallback('frame', leftClickDebug)
            # wait for key press
            key = cv2.waitKey(1) & 0xFF

            # if the `q` key is pressed, break from the loop
            if key == ord("q"):
                break
    

cv2.destroyAllWindows()
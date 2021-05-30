from genericpath import exists
import cv2
# import numpy as np
# import matplotlib.pyplot as plt
import os
import sys
from datetime import datetime

# print(os.path.join('/User','steph','face_detection','Steph'))
# input_path = os.path.join('/User','steph','face_detection')
if len(sys.argv) >= 2:
  DATA_FOLDER = os.path.join('./data', sys.argv[1])
  print('Any captured frames are saved in ', DATA_FOLDER)
  # print(os.getcwd())
  os.makedirs(DATA_FOLDER, exist_ok=True)
SAVE_FORMAT = '.jpg'

cap  = cv2.VideoCapture(0)

MODEL = 'yolo/yolov3-face.cfg'
WEIGHT = 'yolo/yolov3-wider_16000.weights'

net = cv2.dnn.readNetFromDarknet(MODEL, WEIGHT)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

IMG_WIDTH, IMG_HEIGHT = 416, 416

if not cap.isOpened():
  raise IOError("Cannot open webcam")

while True:
  ret, frame = cap.read()

  cv2.imshow('Input', frame)

  c = cv2.waitKey(1)

  blob = cv2.dnn.blobFromImage(frame,
                            1/255, (IMG_WIDTH, IMG_HEIGHT),
                            [0,0,0], 1, crop=False)
  # Set model input 
  net.setInput(blob)

  # Define the layers that we want to get the outputs from
  output_layers = net.getUnconnectedOutLayersNames()

  # Run 'prediction'
  outs = net.forward(output_layers)

  frame_height = frame.shape[0]
  frame_width = frame.shape[1]

  # Scan through all the bounding boxes output from the network and keep only
  # the ones with high confidence scores. Assign the box's class label as the
  # class with the highest score.

  confidences = []
  boxes = []

  # Each frame produces 3 outs corresponding to 3 output layers
  for out in outs:
      # One out has multiple predictions for multiple captured objects.
      for detection in out:
          confidence = detection[-1]
          if confidence > 0.5:
              center_x = int(detection[0] * frame_width)
              center_y = int(detection[1] * frame_height)
              width = int(detection[2] * frame_width)
              height = int(detection[3] * frame_height)

              topleft_x = int(center_x - width/2)
              topleft_y = int(center_y - height/2)
              # print("det", detection)
              # print(topleft_y)     

              confidences.append(float(confidence))
              boxes.append([topleft_x, topleft_y, width, height])

  # Perform non-maximum suppression to eliminate 
  # redundant overlapping boxes with lower confidences.
  indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

  result = frame.copy()
  final_boxes = []
  for i in indices:
      i = i[0]
      box = boxes[i]
      final_boxes.append(box)

      # Extract position data
      left = box[0]
      top = box[1]
      width = box[2]
      height = box[3]

      # Draw bouding box with the above measurements
      ### YOUR CODE HERE
      cv2.rectangle(result, (left, top), (left+width,top+height), (0, 255,0), 1)	
    # Display text about confidence rate above each box
      text = f'{confidences[i]:.2f}'
      ### YOUR CODE HERE
      # white = Scalar(255,255,255)
      cv2.putText(result, text, (left,top), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0), 1, cv2.LINE_AA)
      num_faces = f'Number of faces detected: {len(final_boxes)}'
      cv2.putText(result, num_faces, (20,20), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 1, cv2.LINE_AA)
  # Display text about number of detected faces on topleft corner
      screenshot = frame[left:left+width, top:top+height]
      

  cv2.imshow('face detection', result)
  # cv2.waitKey(0)
  # cv2.destroyAllWindows()
  # stop when hitting ESC
  if c == 27:
    break

  if c == ord('s'):
        fname = os.path.join(DATA_FOLDER, datetime.today().strftime('%y%m%d%H%M%S')+SAVE_FORMAT)
        print('Save captured frame as ', fname)
        cv2.imwrite(fname, frame)

cap.release()
cv2.destroyAllWindows()
sys.exit()
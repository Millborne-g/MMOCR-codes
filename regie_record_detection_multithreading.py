import os
import argparse
import cv2
import numpy as np
import sys
import importlib.util
import imutils
import re
import pytesseract
from PIL import Image
from threading import Thread
import time
#pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
pytesseract.pytesseract.tesseract_cmd = r"C:\Users\emielyn\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"

from mmocr.apis import TextRecInferencer
inferencer = TextRecInferencer(model='SATRN', weights=r'C:\Users\emielyn\mmocr\best_IC15_recog_word_acc_epoch_77.pth')



# Define and parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--modeldir', help='Folder the .tflite file is located in',
                    required=True)
parser.add_argument('--graph', help='Name of the .tflite file, if different than detect.tflite',
                    default='detect.tflite')
parser.add_argument('--labels', help='Name of the labelmap file, if different than labelmap.txt',
                    default='labelmap.txt')
parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',
                    default=0.5)
parser.add_argument('--video', help='Name of the video file',
                    default='newCamVid1.mp4')
parser.add_argument('--edgetpu', help='Use Coral Edge TPU Accelerator to speed up detection',
                    action='store_true')

args = parser.parse_args()

MODEL_NAME = args.modeldir
GRAPH_NAME = args.graph
LABELMAP_NAME = args.labels
VIDEO_NAME = args.video
min_conf_threshold = float(args.threshold)
use_TPU = args.edgetpu

# Import TensorFlow libraries
# If tflite_runtime is installed, import interpreter from tflite_runtime, else import from regular tensorflow
# If using Coral Edge TPU, import the load_delegate library
pkg = importlib.util.find_spec('tflite_runtime')
if pkg:
    from tflite_runtime.interpreter import Interpreter
    if use_TPU:
        from tflite_runtime.interpreter import load_delegate
else:
    from tensorflow.lite.python.interpreter import Interpreter
    if use_TPU:
        from tensorflow.lite.python.interpreter import load_delegate

# If using Edge TPU, assign filename for Edge TPU model
if use_TPU:
    # If user has specified the name of the .tflite file, use that name, otherwise use default 'edgetpu.tflite'
    if (GRAPH_NAME == 'detect.tflite'):
        GRAPH_NAME = 'edgetpu.tflite'   

# Get path to current working directory
CWD_PATH = os.getcwd()

# Path to video file
VIDEO_PATH = os.path.join(CWD_PATH,VIDEO_NAME)

# Path to .tflite file, which contains the model that is used for object detection
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,GRAPH_NAME)

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_NAME,LABELMAP_NAME)

# Load the label map
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Have to do a weird fix for label map if using the COCO "starter model" from
# https://www.tensorflow.org/lite/models/object_detection/overview
# First label is '???', which has to be removed.
if labels[0] == '???':
    del(labels[0])

# Load the Tensorflow Lite model.
# If using Edge TPU, use special load_delegate argument
if use_TPU:
    interpreter = Interpreter(model_path=PATH_TO_CKPT,
                              experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
    print(PATH_TO_CKPT)
else:
    interpreter = Interpreter(model_path=PATH_TO_CKPT)

interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5

# Check output layer name to determine if this model was created with TF2 or TF1,
# because outputs are ordered differently for TF2 and TF1 models
outname = output_details[0]['name']

if ('StatefulPartitionedCall' in outname): # This is a TF2 model
    boxes_idx, classes_idx, scores_idx = 1, 3, 0
else: # This is a TF1 model
    boxes_idx, classes_idx, scores_idx = 0, 1, 2


saveImageDelay = False

def ocr():
    global saveImageDelay
    
    while True:
        try:
            directory = r'C:\\tflite1\\mmocr\\result'

            img = cv2.imread(r"C:\Users\emielyn\mmocr\iMAGE.jpg")
            #img_resized = cv2.resize(img, (width, height))
            img_resized = cv2.resize(img,None, fx=0.5 , fy =0.5)

            # Pass preprocessed image to OCR model
            result = inferencer(img_resized, print_result=True)
            text = result['predictions'][0]['text']

            # Print OCR results
            print('Prediction: ',text)
            # saveImageDelay = True
        
        except Exception as e:
            print('')


def detection():
    # Open video file
    global saveImageDelay
    video = cv2.VideoCapture(VIDEO_PATH)
    imW = video.get(cv2.CAP_PROP_FRAME_WIDTH)
    imH = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
    last_save_time = time.time()
    while(video.isOpened()):

        # Acquire frame and resize to expected shape [1xHxWx3]
        ret, frame = video.read()
        if not ret:
            print('Reached the end of the video!')
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (width, height))
        input_data = np.expand_dims(frame_resized, axis=0)

        # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
        if floating_model:
            input_data = (np.float32(input_data) - input_mean) / input_std

        # Perform the actual detection by running the model with the image as input
        interpreter.set_tensor(input_details[0]['index'],input_data)
        interpreter.invoke()

        # Retrieve detection results
        boxes = interpreter.get_tensor(output_details[boxes_idx]['index'])[0] # Bounding box coordinates of detected objects
        classes = interpreter.get_tensor(output_details[classes_idx]['index'])[0] # Class index of detected objects
        scores = interpreter.get_tensor(output_details[scores_idx]['index'])[0] # Confidence of detected objects
        area = [(1,357),(639,357),(639,450),(1,450)] #sa laptop cam
        # Loop over all detections and draw detection box if confidence is above minimum threshold
        for i in range(len(scores)):
            if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):
                
                # Get bounding box coordinates and draw box
                # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
                ymin = int(max(1,(boxes[i][0] * imH)))
                xmin = int(max(1,(boxes[i][1] * imW)))
                ymax = int(min(imH,(boxes[i][2] * imH)))
                xmax = int(min(imW,(boxes[i][3] * imW)))
                cx = int((xmin + xmax)/2)
                cy = int((ymin + ymax)/2)
                result = cv2.pointPolygonTest(np.array(area, np.int32), (int(cx), int(cy)), False)
                # if  result >= 0:
                detected = True
                # cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)
                object_name = labels[int(classes[i])] 
                label = '%s: %d%%' % (object_name, int(scores[i]*100)) 
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) 
                label_ymin = max(ymin, labelSize[1] + 10) 
                cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED)
                cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) 
                # cv2.circle(frame,(cx,cy),5,(10, 255, 0),-1)
                imgRoi = frame[ymin:ymax, xmin:xmax]
                cv2.imwrite("iMAGE.jpg", imgRoi)
                
                    # Save image every 3 seconds
                # elapsed_time = time.time() - last_save_time
                # if elapsed_time >= 2.7:
                # # if saveImageDelay == True:
                #     cv2.imwrite("iMAGE.jpg", imgRoi)
                #     last_save_time = time.time()
                #     saveImageDelay = False
                
                # ocr()
                
                
                # else:
                #     detected = False
            

        # All the results have been drawn on the frame, so it's time to display it.
        frame1 = imutils.resize(frame, width=640)
        cv2.imshow('Object detector', frame1)

        # Press 'q' to quit
        if cv2.waitKey(1) == ord('q'):
            break

    # Clean up
    video.release()
    cv2.destroyAllWindows()


task1 = Thread(target=detection)
task2 = Thread(target=ocr)

while True:
    task1.start()
    task2.start()

    task1.join()
    task2.join()
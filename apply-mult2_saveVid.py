import os
import argparse
import cv2
import numpy as np
import sys
import time
from threading import Thread
import importlib.util
import pytesseract
import pyrebase
from datetime import date
from datetime import datetime
import imutils

from mmocr.apis import TextRecInferencer
inferencer = TextRecInferencer(model='SATRN', weights=r'D:\testCudaPytorch2\best_IC15_recog_word_acc_epoch_77.pth')

firebaseConfig = {
  "apiKey": "AIzaSyB_4cNoh3klH4mKPSd7dhJzr5QUGoLihy8",
  "authDomain": "scanmemaster-9da58.firebaseapp.com",
  "projectId": "scanmemaster-9da58",
  "databaseURL" : "https://scanmemaster-9da58-default-rtdb.firebaseio.com/",
  "storageBucket": "scanmemaster-9da58.appspot.com",
  "messagingSenderId": "270970295536",
  "appId": "1:270970295536:web:02ecd24ee665578e6d9e35",
  "measurementId": "G-27WEKS22GB"
}

firebase = pyrebase.initialize_app(firebaseConfig)
db = firebase.database()


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

min_conf_threshold = float(args.threshold)
use_TPU = args.edgetpu
# Keep track if we are using TPU
TPU_in_use = False

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

video = cv2.VideoCapture(VIDEO_PATH)
imW = video.get(cv2.CAP_PROP_FRAME_WIDTH)
imH = video.get(cv2.CAP_PROP_FRAME_HEIGHT)

detected = False
exit = 0
image_output = "iMAGE.jpg"

frame_rate_calc = 1
freq = cv2.getTickFrequency()

def detection():
    global frame_rate_calc
    global detected
    global exit
    global video
    while True:
        t1 = cv2.getTickCount()
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
        boxes = interpreter.get_tensor(output_details[1]['index'])[0] # Bounding box coordinates of detected objects
        classes = interpreter.get_tensor(output_details[3]['index'])[0] # Class index of detected objects
        scores = interpreter.get_tensor(output_details[0]['index'])[0] # Confidence of detected objects

        #area = [(1,433),(1278,436),(1278,750),(1,750)]
        #area = [(10,466),(1730,466),(1730,1060),(10,1060)]
        area = [(10,200),(1730,200),(1730,1060),(10,1060)]

        # Loop over all detections and draw detection box if confidence is above minimum threshold
        for i in range(len(scores)):
            if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):

                ymin = int(max(1,(boxes[i][0] * imH)))
                xmin = int(max(1,(boxes[i][1] * imW)))
                ymax = int(min(imH,(boxes[i][2] * imH)))
                xmax = int(min(imW,(boxes[i][3] * imW)))

                cx = int((xmin + xmax)/2)
                cy = int((ymin + ymax)/2)

                result = cv2.pointPolygonTest(np.array(area, np.int32), (int(cx), int(cy)), False)
                if result >= 0:
                    detected = True
                    # Draw label
                    object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
                    label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
                    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
                    label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
                    #cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
                    cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (174, 241, 214), 2) # Draw label text
                    #cv2.circle(frame,(cx,cy),5,(10, 255, 0),-1)
                    imgRoi = frame[ymin:ymax, xmin:xmax]
                    cv2.imwrite("iMAGE.jpg", imgRoi)
                else:
                    detected = False
        for i in area:
            cv2.polylines(frame,[np.array(area, np.int32)], True, (15,220,10),6)
            
        cv2.putText(frame,'FPS: {0:.2f}'.format(frame_rate_calc),(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)
        frame1 = imutils.resize(frame, width=700)

        cv2.imshow('Object detector', frame1)

        t2 = cv2.getTickCount()
        time1 = (t2-t1)/freq
        frame_rate_calc= 1/time1

        key = cv2.waitKey(1)
        if key == ord('q'):
            exit =1
            break

    video.stop()
    cv2.destroyAllWindows()

def ocr():
        global detected
        global exit
        global prev_txt
        while True: 
            if exit == 0:    
                    if os.path.exists(image_output):
                        try:
                            img_ocr = cv2.imread(image_output)
                            img_ocr = cv2.resize(img_ocr,None, fx=0.5 , fy =0.5)
                            if detected == True:
                                result = inferencer(img_ocr, print_result=True)
                                text = result['predictions'][0]['text']
                                print('Prediction: ',text)
   
                        except Exception as e:
                            print("")
                    else:
                        continue
            else:
                break

def saveForQuery():
    global exit
    filename = "scanned_platenumbers.txt"
    prevPN = ''
    # Create the file if it doesn't exist
    if not os.path.isfile(filename):
        open(filename, "w").close()

    while True:
        if exit == 0:

            #Read the latest scanned on the database
            plateNum = db.child("ScannedQuery").child("PlateNumber").get()
            if plateNum.val() != prevPN:
                # Open the file in append mode
                with open(filename, "a") as file:
                    # Get the text to append from the user
                    plateNum = plateNum.val()
                    # Append the text to the end of the file
                    file.write(plateNum+ "\n")
                    # Close the file
                    file.close()
                #print('checkdatabase')
                prevPN = plateNum
                #time.sleep(1)
        else:
            break

prev_txt = []

def checkExist():
    global exit
    global prev_txt
    while True:
        if exit == 0:
            filename = "scanned_platenumbers.txt"
            first_line = ""
            # Open the file for reading and writing
            with open(filename, "r+") as file:
                # Read the first line of the file
                first_line = file.readline().strip()
                # Read the remaining lines of the file
                remaining_lines = file.readlines()
                # Overwrite the file with the remaining lines
                file.seek(0)
                file.writelines(remaining_lines)
                file.truncate()
                # Close the file
                file.close()
            plateNum = first_line
            try:
                exist = db.child("Vehicle_with_criminal_offense").child(plateNum).child("plateNumber").get()
                #print(exist.val())
                if exist.val() != None:
                    isApprehended = db.child("Vehicle_with_criminal_offense").child(plateNum).child("apprehended").get()
                    #print("isApprehended "+isApprehended.val())
                    if isApprehended.val() != 'yes':
                        # Create Data
                        nowD = datetime.now()
                        dateToday = str(date.today())
                        timeToday = nowD.strftime("%H:%M:%S")
                        crimeScanned = db.child("Vehicle_with_criminal_offense").child(plateNum).child("criminalOffense").get()
                        data = {"PlateNumber":plateNum, "Location": "Lapasan Zone 4", "Date": dateToday, "Time": timeToday, "Notification": "on", "Apprehended": "no", "CriminalOffense": crimeScanned.val()}
                        print(len(prev_txt))
                        if plateNum not in prev_txt:
                            db.child("Scanned").child((dateToday+" "+timeToday)).set(data)
                            crime = db.child("Vehicle_with_criminal_offense").child(plateNum).child("criminalOffense").get()
                            dataPlateNumber = {"PlateNumber":plateNum, "Apprehended": "no","CriminalOffense": crime.val()}
                            db.child("ScannedPlateNumber").child(plateNum).set(dataPlateNumber)

                            #For Notification
                            db.child("ScannedNotification").set(data)
                            db.child("ScannedPlateNumberNotification").set(dataPlateNumber)
                            prev_txt.append(plateNum)

                    #print("Plate Number dont't exist")

            except Exception as e:
                print(" ")

        else:
            break

def clear_list():
    global exit
    while True:
        if exit == 0:
            time.sleep(6)
            prev_txt.clear()
            print("--------------------------")
        else:
            break



def record_video(vidOutput = "testVid.mp4", duration=10):
    # cap = cv2.VideoCapture("rtsp://camuser1:caiustpuser1@192.168.254.115:554/cam/realmonitor?channel=1&subtype=0")
    cap= cv2.VideoCapture(0)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(vidOutput, fourcc, fps, (width, height))

    frames_to_record = fps * duration
    frames_recorded = 0

    while frames_recorded < frames_to_record:
        ret, frame = cap.read()

        if not ret:
            break

        out.write(frame)

        frames_recorded += 1
    
    print()
    print()
    print()
    print()
    print()
    print('done save vid')
    print()
    print()
    print()
    print()
    print()

    cap.release()
    out.release()

def checkVid():


    filename = 'testVid.mp4'

    try:
        if os.path.exists(filename):
            print(f"{filename} exists!")
        else:
            print(f"{filename} does not exist.")
    except Exception as e:
        print(f"Error: {e}")



task1 = Thread(target=detection)
task2 = Thread(target=ocr)
task5 = Thread(target=record_video)
#task3 = Thread(target=saveForQuery)
#task4 = Thread(target=checkExist)

while True:

    task1.start()
    task2.start()
    
    task5.start()
#    task3.start()
#    task4.start()
    
    task1.join()
    task2.join()
    task5.join()
#    task3.join()
#    task4.join()
#    task5.join()    

    
   


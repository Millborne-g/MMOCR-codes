import os
import argparse
import cv2
import numpy as np
import sys
import time
from threading import Thread
import importlib.util
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Users\emielyn\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"
import pyrebase
from datetime import date
from datetime import datetime
import imutils
import Levenshtein

from mmocr.apis import TextRecInferencer
inferencer = TextRecInferencer(model='SATRN', weights=r'C:\Users\emielyn\mmocr\best_IC15_recog_word_acc_epoch_77.pth')

# Initialize the Firebase app with your service account credentials

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


class VideoStream:
    """Camera object that controls video streaming from the Picamera"""
    def __init__(self,resolution=(640,480),framerate=30): 
    # def __init__(self,resolution=(420,480),framerate=30):820
        self.stream = cv2.VideoCapture(0)

        # self.stream = cv2.VideoCapture("newCamVid1.mp4")

        #self.stream = cv2.VideoCapture("rtsp://thesis:thesisisit@10.0.254.12/stream2")

        ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        ret = self.stream.set(3,resolution[0])
        ret = self.stream.set(4,resolution[1])
            
        (self.grabbed, self.frame) = self.stream.read()

        self.stopped = False

    def start(self):
        Thread(target=self.update,args=()).start()
        return self

    def update(self):
        while True:
            if self.stopped:
                self.stream.release()
                return

            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True

parser = argparse.ArgumentParser()
parser.add_argument('--modeldir', help='Folder the .tflite file is located in',
                    required=True)
parser.add_argument('--graph', help='Name of the .tflite file, if different than detect.tflite',
                    default='detect.tflite')
parser.add_argument('--labels', help='Name of the labelmap file, if different than labelmap.txt',
                    default='labelmap.txt')
parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',
                    default=0.5)
parser.add_argument('--resolution', help='Desired webcam resolution in WxH. If the webcam does not support the resolution entered, errors may occur.',
                    default='640x480')
parser.add_argument('--edgetpu', help='Use Coral Edge TPU Accelerator to speed up detection',
                    action='store_true')

args = parser.parse_args()

MODEL_NAME = args.modeldir
GRAPH_NAME = args.graph
LABELMAP_NAME = args.labels
min_conf_threshold = float(args.threshold)
resW, resH = args.resolution.split('x')
imW, imH = int(resW), int(resH)
use_TPU = args.edgetpu


pkg = importlib.util.find_spec('tflite_runtime')
if pkg:
    from tflite_runtime.interpreter import Interpreter
    if use_TPU:
        from tflite_runtime.interpreter import load_delegate
else:
    from tensorflow.lite.python.interpreter import Interpreter
    if use_TPU:
        from tensorflow.lite.python.interpreter import load_delegate

if use_TPU:
    if (GRAPH_NAME == 'detect.tflite'):
        GRAPH_NAME = 'edgetpu.tflite'       

CWD_PATH = os.getcwd()

PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,GRAPH_NAME)

PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_NAME,LABELMAP_NAME)

with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

if labels[0] == '???':
    del(labels[0])

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


outname = output_details[0]['name']

if ('StatefulPartitionedCall' in outname): 
    boxes_idx, classes_idx, scores_idx = 1, 3, 0
else: 
    boxes_idx, classes_idx, scores_idx = 0, 1, 2

frame_rate_calc = 1
freq = cv2.getTickFrequency()

videostream = VideoStream(resolution=(imW,imH),framerate=30).start()
count = 0
exit = 0
detected = False
image_output = "iMAGE.jpg"


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

            # print('check '+plateNum)

            try:
                if len(plateNum)>0:
                    # Get all plate numbers in "Vehicle_with_criminal_offense" node
                    plate_nums = db.child("Vehicle_with_criminal_offense").shallow().get().val()
                    
                    # Find closest match to input
                    global closest_match
                    closest_match = None
                    min_distance = float('inf')
                    for num in plate_nums:
                        distance = Levenshtein.distance(plateNum, num)
                        if distance < min_distance:
                            closest_match = num
                            min_distance = distance
                    
                    confidence = round((1 - (min_distance / len(plateNum))) * 100, 2)
                    if confidence >= 60 and closest_match not in prev_txt:
                        exist = db.child("Vehicle_with_criminal_offense").child(closest_match).child("plateNumber").get()
                        #print(exist.val())
                        if exist.val() != None:
                            isApprehended = db.child("Vehicle_with_criminal_offense").child(closest_match).child("apprehended").get()
                            #print("isApprehended "+isApprehended.val())
                            if isApprehended.val() != 'yes':
                                    print('Notify '+plateNum)
                                    # Create Data
                                    nowD = datetime.now()
                                    dateToday = str(date.today())
                                    timeToday = nowD.strftime("%H:%M:%S")
                                    crimeScanned = db.child("Vehicle_with_criminal_offense").child(closest_match).child("criminalOffense").get()

                                    color = ''
                                    if confidence >= 60 and confidence <= 75:
                                        color='yellow'
                                    elif confidence > 75 and confidence <= 100:
                                        color='red'

                                    data = {"PlateNumber":closest_match, "Location": "Lapasan Zone 4", "Date": dateToday, "Time": timeToday, "Notification": "on", "Apprehended": "no", "CriminalOffense": crimeScanned.val(), 'Color': color, 'DetectedPN': plateNum}
                                    db.child("Scanned").child((dateToday+" "+timeToday)).set(data)
                                    dataPlateNumber = {"PlateNumber":closest_match, "Apprehended": "no","CriminalOffense": crimeScanned.val()}
                                    db.child("ScannedPlateNumber").child(closest_match).set(dataPlateNumber)

                                    #For Notification
                                    db.child("ScannedNotification").set(data)
                                    db.child("ScannedPlateNumberNotification").set(dataPlateNumber)
                                    prev_txt.append(closest_match)
                        else:
                            print(" ")
                            #print("Plate Number dont't exist")
            except Exception as e:
                print("err "+str(e))
                #print("Plate Number dont't exist "+ str(e))
            #print()
            #print('checkDatabase')
            #print('Latest data:', plateNum)
            #print()
            #time.sleep(1)
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

def clear_list():
    global exit
    while True:
        if exit == 0:
            time.sleep(30)
            prev_txt.clear()
            print("--------------------------")
        else:
            break


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
                                    # txt =pytesseract.image_to_string(img_ocr, config='-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 8 --oem 3')
                                    # print(txt)  
                                    # Pass preprocessed image to OCR model
                                    result = inferencer(img_ocr, print_result=True)
                                    text = result['predictions'][0]['text']

                                    # Print OCR results
                                    print('Prediction: ',text)
                                    data = {"PlateNumber":text}
                                    db.child("ScannedQuery").set(data)
                                try:
                                    os.remove(image_output)
                                except OSError as e:
                                    print(f"Error: {image_output} path could not be delete. {e}")
                            except Exception as e:
                                print("")
                                #print("An error occured:", str(e))
                        else:
                            
                    
                            continue
                            
                else:
                    break

def detection():
    global frame_rate_calc
    global detected
    global exit
    while True:
        t1 = cv2.getTickCount()
        frame1 = videostream.read()

        frame = frame1.copy()
        # frame = imutils.resize(frame1, width=820)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (width, height))
        input_data = np.expand_dims(frame_resized, axis=0)

        if floating_model:
            input_data = (np.float32(input_data) - input_mean) / input_std

        interpreter.set_tensor(input_details[0]['index'],input_data)
        interpreter.invoke()

        boxes = interpreter.get_tensor(output_details[boxes_idx]['index'])[0]
        classes = interpreter.get_tensor(output_details[classes_idx]['index'])[0] 
        scores = interpreter.get_tensor(output_details[scores_idx]['index'])[0]

        # area = [(1,200),(817,200),(817,450),(1,450)] #Bahog ug video

        area = [(1,257),(639,257),(639,480),(1,480)] #sa laptop cam
        # area = [(2,243),(637,243),(637,360),(2,360)] #sa CCTV

        for i in range(len(scores)):
            if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):

                ymin = int(max(1,(boxes[i][0] * imH)))
                xmin = int(max(1,(boxes[i][1] * imW)))
                ymax = int(min(imH,(boxes[i][2] * imH)))
                xmax = int(min(imW,(boxes[i][3] * imW)))
                
                cx = int((xmin + xmax)/2)
                cy = int((ymin + ymax)/2)
                result = cv2.pointPolygonTest(np.array(area, np.int32), (int(cx), int(cy)), False)
                if  result >= 0:
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
             
                else:
                    detected = False
        for i in area:
            cv2.polylines(frame,[np.array(area, np.int32)], True, (15,220,10),6)

        cv2.putText(frame,'FPS: {0:.2f}'.format(frame_rate_calc),(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)
        
        # frame1 = imutils.resize(frame, width=650)
        cv2.imshow('Object detector', frame)

        
        t2 = cv2.getTickCount()
        time1 = (t2-t1)/freq
        frame_rate_calc= 1/time1
        

        if cv2.waitKey(1) == ord('q'):
            exit =1
            break
    videostream.stop()
    cv2.destroyAllWindows()

task1 = Thread(target=detection)
task2 = Thread(target=ocr)
task3 = Thread(target=saveForQuery)
task4 = Thread(target=checkExist)
task5 = Thread(target=clear_list)

while True:
    task1.start()
    task2.start()
    task3.start()
    task4.start()
    task5.start()


    task1.join()
    task2.join()
    task3.join()
    task4.join()
    task5.join()
    if exit ==1:
        print("Done executing")
        break
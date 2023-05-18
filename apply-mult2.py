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
import Levenshtein


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
storage = firebase.storage()

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
                    default='UserTestingVidFinal.mp4')
parser.add_argument('--edgetpu', help='Use Coral Edge TPU Accelerator to speed up detection',
                    action='store_true')

args = parser.parse_args()

MODEL_NAME = args.modeldir
GRAPH_NAME = args.graph
LABELMAP_NAME = args.labels
VIDEO_NAME = args.video
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
directory = image_output
setCopyPath = False
copyPath = "check_copy.jpg"
prevImgFname = []

frame_rate_calc = 1
freq = cv2.getTickFrequency()

def detection():
    global frame_rate_calc
    global detected
    global exit
    global video
    global setCopyPath
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

                    if setCopyPath == False:
                        cv2.imwrite("check_copy.jpg", imgRoi)
                        setCopyPath = True
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

        cv2.waitKey(2)

        if cv2.waitKey(1) == ord('q'):
            exit =1
            break

    video.stop()
    cv2.destroyAllWindows()

def ocr():
    global saveImageDelay
    global directory
    global detected
    global exit
    global prev_txt
    global prevImgFname
    global setCopyPath
    filename = "scanned_platenumbers.txt"
    prevPN = ''
    # Create the file if it doesn't exist
    if not os.path.isfile(filename):
        open(filename, "w").close()
    while True:
        if exit == 0:
            if os.path.exists(directory):
                try:
                    img = cv2.imread(directory)
                    img_resized = cv2.resize(img,None, fx=0.5 , fy =0.5)
                    start_time = time.time()
                    result = inferencer(img_resized, print_result=True)
                    text = result['predictions'][0]['text']
                    end_time = time.time()
                    elapsed_time = end_time - start_time

                    if text != prevPN:
                    # Open the file in append mode
                        with open(filename, "a") as file:
                            # Get the text to append from the user
                            plateNum = text
                            # Append the text to the end of the file
                            file.write(plateNum+ "\n")
                            # Close the file
                            file.close()
                        #print('checkdatabase')
                        prevPN = text
                        # data = {"PlateNumber":text}
                        # db.child("ScannedQuery").set(data)
                    print('Prediction: '+text+' TimeElapse: '+str(elapsed_time))
                    if text not in prevImgFname:
                                    
                        new_original_filename = f"scanned_platenumbers\{text}.jpg"

                        # Rename the original image with the new filename
                        os.rename(copyPath, new_original_filename)
                        setCopyPath = False
                        prevImgFname.append(text)
                        
                    # Print OCR results
                    
                    
                except Exception as e:
                    print('error '+str(e))
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

            # print('check '+plateNum)

            try:
                if len(plateNum)>0:
                    # Find closest matches to input with at least 50% confidence score
                    plate_nums = db.child("Vehicle_with_criminal_offense").shallow().get().val()

                    global closest_matches
                    closest_matches = []
                    # min_distance = float('inf')
                    for num in plate_nums:
                        distance = Levenshtein.distance(plateNum, num)
                        confidence = round((1 - (distance / len(plateNum))) * 100, 2)
                        if confidence >= 60:
                            num_value = db.child("Vehicle_with_criminal_offense").child(num).child('criminalOffense').get().val()
                            closest_matches.append((num, num_value, confidence))

                    # Sort matches by descending confidence
                    closest_matches = sorted(closest_matches, key=lambda x: x[2], reverse=True)
                    if closest_matches[0][0] not in prev_txt:
                        exist = db.child("Vehicle_with_criminal_offense").child(closest_matches[0][0]).child("plateNumber").get()
                        #print(exist.val())
                        if exist.val() != None:
                            isApprehended = db.child("Vehicle_with_criminal_offense").child(closest_matches[0][0]).child("apprehended").get()
                            #print("isApprehended "+isApprehended.val())
                            if isApprehended.val() != 'yes':
                                    print()
                                    print()
                                    print()
                                    print('Notify '+plateNum)   
                                    print()
                                    print()
                                    print()
                                    # imgFilename = f"{plateNum}.jpg"
                                    # imgFilename_path = "scanned_platenumbers"
                                    
                                    # if os.path.exists(os.path.join(imgFilename_path, imgFilename)):
                                    # Create Data
                                    nowD = datetime.now()
                                    dateToday = str(date.today())
                                    timeToday = nowD.strftime("%H:%M:%S")
                                    crimeScanned = db.child("Vehicle_with_criminal_offense").child(closest_matches[0][0]).child("criminalOffense").get()

                                    detected_PN_filename = f"scanned_platenumbers\{plateNum}.jpg"
                                    # storageFilename = dateToday+" "+timeToday+detected_PN_filename.split('\\')[-1]

                                    # Upload image to Firebase storage
                                    storage.child(f"{dateToday} {timeToday} {plateNum}.jpg").put(detected_PN_filename)

                                    # Get view link of uploaded file
                                    view_link = storage.child(f"{dateToday} {timeToday} {plateNum}.jpg").get_url(None)

                                    # color = ''
                                    # if confidence >= 60 and confidence <= 75:
                                    #     color='yellow'
                                    # elif confidence > 75 and confidence <= 100:
                                    #     color='red'

                                    data = {"PlateNumber":closest_matches[0][0], "Location": "Corales Barangay 29", "Date": dateToday, "Time": timeToday, "Notification": "on", "Apprehended": "no", "CriminalOffense": crimeScanned.val(), 'DetectedPN': plateNum, 'ClosestMatches':str(closest_matches), 'ImageLink':view_link}
                                    db.child("Scanned").child((dateToday+" "+timeToday)).set(data)
                                    dataPlateNumber = {"PlateNumber":closest_matches[0][0], "Apprehended": "no","CriminalOffense": crimeScanned.val()}
                                    db.child("ScannedPlateNumber").child(closest_matches[0][0]).set(dataPlateNumber)

                                    #For Notification
                                    db.child("ScannedNotification").set(data)
                                    db.child("ScannedPlateNumberNotification").set(dataPlateNumber)
                                    prev_txt.append(closest_matches[0][0])
                                    
                        else:
                            print("check exist "+str(exist.val()))
                            #print("Plate Number dont't exist")
            except Exception as e:
                print("err CheckExist "+str(e))
                #print("Plate Number dont't exist "+ str(e))
            #print()
            #print('checkDatabase')
            #print('Latest data:', plateNum)
            #print()
            #time.sleep(1)
        else:
            break

prev_txt = []

def clear_list():
    global exit
    while True:
        if exit == 0:
            time.sleep(60)
            
            folder_path = 'scanned_platenumbers'  # Replace with the path to your folder

            # Loop through all the files in the folder and remove them
            for filename in os.listdir(folder_path):
                file_path = os.path.join(folder_path, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    print(f'Error deleting {file_path}: {e}')
            prev_txt.clear()
            prevImgFname.clear()
            print()
            print("-----------Clear List------------")
            print()
        else:
            break


def first_clear_list():
    global directory
    global copyPath
    folder_path = 'scanned_platenumbers'  # Replace with the path to your folder
     # Loop through all the files in the folder and remove them
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f'Error deleting {file_path}: {e}')
    prev_txt.clear()
    prevImgFname.clear()
    try:
        os.remove(directory)
        os.remove(copyPath)
    except:
        print()
    print()
    print("-----------First Clear List------------")
    print()


first_clear_list()


task1 = Thread(target=detection)
task2 = Thread(target=ocr)
task3 = Thread(target=saveForQuery)
task4 = Thread(target=checkExist)

while True:

    task1.start()
    task2.start()
    task3.start()
    task4.start()
    
    task1.join()
    task2.join()
    task3.join()
    task4.join()
#    task5.join()    

    
   


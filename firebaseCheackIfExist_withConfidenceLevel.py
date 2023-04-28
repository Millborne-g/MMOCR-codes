import pyrebase
from datetime import date
import time
from datetime import datetime
import random
import threading
import time
import os
import Levenshtein

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

def checkExist(plateNum):
    
    try:
        plate_nums = db.child("Vehicle_with_criminal_offense").shallow().get().val()

        # Find closest matches to input with at least 50% confidence score
        global closest_matches
        closest_matches = []
        min_distance = float('inf')
        for num in plate_nums:
            distance = Levenshtein.distance(plateNum, num)
            confidence = round((1 - (distance / len(plateNum))) * 100, 2)
            if confidence >= 50:
                closest_matches.append((num, confidence))

        # Sort matches by descending confidence
        closest_matches = sorted(closest_matches, key=lambda x: x[1], reverse=True)
        print(str(closest_matches[0][0]))
        
        if True:
            exist = db.child("Vehicle_with_criminal_offense").child(closest_matches[0][0]).child("plateNumber").get()
            # # print(f"Closest match found: {closest_match}")
            # print(f"Closest match found in db: {plateNum}")
            # print(f"Confidence level: {confidence}%")
            if exist.val() != None:
                print()
                isApprehended = db.child("Vehicle_with_criminal_offense").child(closest_matches[0][0]).child("apprehended").get()
                print("isApprehended "+isApprehended.val())
                if isApprehended.val() != 'yes':
                    # Create Data
                    nowD = datetime.now()
                    dateToday = str(date.today())
                    timeToday = nowD.strftime("%H:%M:%S")
                    crimeScanned = db.child("Vehicle_with_criminal_offense").child(closest_matches[0][0]).child("criminalOffense").get()

                    color = ''
                    if confidence >= 50 and confidence <= 70:
                        color='yellow'
                    elif confidence > 70 and confidence <= 100:
                        color='red'

                    data = {"PlateNumber":closest_matches[0][0], "Location": "Lapasan Zone 4", "Date": dateToday, "Time": timeToday, "Notification": "on", "Apprehended": "no", "CriminalOffense": crimeScanned.val(), 'Color': color, 'DetectedPN': plateNum, 'ClosestMatches':str(closest_matches)}
                    db.child("Scanned").child((dateToday+" "+timeToday)).set(data)
                    dataPlateNumber = {"PlateNumber":closest_matches[0][0], "Apprehended": "no","CriminalOffense": crimeScanned.val()}
                    db.child("ScannedPlateNumber").child(closest_matches[0][0]).set(dataPlateNumber)

                    #For Notification
                    db.child("ScannedNotification").set(data)
                    db.child("ScannedPlateNumberNotification").set(dataPlateNumber)
            else:
                print("Plate Number don't exist ssss")
        else:
            print("Plate Number don't exist dddd")
            # print("No match found for input")
    except Exception as e:
        print("Error: " + str(e))
        
    print()
    print('checkDatabase')
    print('Latest data:', plateNum)
    print()
    time.sleep(1)

checkExist("4AD1781")
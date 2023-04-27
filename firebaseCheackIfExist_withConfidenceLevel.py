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
        # Get all plate numbers in "Vehicle_with_criminal_offense" node
        plate_nums = db.child("Vehicle_with_criminal_offense").shallow().get().val()
        
        # Find closest match to input
        closest_match = None
        min_distance = float('inf')
        for num in plate_nums:
            distance = Levenshtein.distance(plateNum, num)
            if distance < min_distance:
                closest_match = num
                min_distance = distance
        
        confidence = round((1 - (min_distance / len(plateNum))) * 100, 2)
        
        if confidence >= 50:
            exist = db.child("Vehicle_with_criminal_offense").child(closest_match).child("plateNumber").get()
            print(f"Closest match found: {closest_match}")
            print(f"Closest match found in db: {plateNum}")
            print(f"Confidence level: {confidence}%")
            if exist.val() != None:
                print()
                isApprehended = db.child("Vehicle_with_criminal_offense").child(closest_match).child("apprehended").get()
                print("isApprehended "+isApprehended.val())
                if isApprehended.val() != 'yes':
                    # Create Data
                    nowD = datetime.now()
                    dateToday = str(date.today())
                    timeToday = nowD.strftime("%H:%M:%S")
                    crimeScanned = db.child("Vehicle_with_criminal_offense").child(closest_match).child("criminalOffense").get()

                    color = ''
                    if confidence >= 50 and confidence <= 70:
                        color='yellow'
                    elif confidence > 70 and confidence <= 100:
                        color='red'

                    data = {"PlateNumber":closest_match, "Location": "Lapasan Zone 4", "Date": dateToday, "Time": timeToday, "Notification": "on", "Apprehended": "no", "CriminalOffense": crimeScanned.val(), 'Color': color, 'DetectedPN': plateNum}
                    db.child("Scanned").child((dateToday+" "+timeToday)).set(data)
                    dataPlateNumber = {"PlateNumber":closest_match, "Apprehended": "no","CriminalOffense": crimeScanned.val()}
                    db.child("ScannedPlateNumber").child(closest_match).set(dataPlateNumber)

                    #For Notification
                    db.child("ScannedNotification").set(data)
                    db.child("ScannedPlateNumberNotification").set(dataPlateNumber)
            else:
                print("Plate Number don't exist")
        else:
            print("Plate Number don't exist")
            # print("No match found for input")
    except Exception as e:
        print("Error: " + str(e))
        
    print()
    print('checkDatabase')
    print('Latest data:', plateNum)
    print()
    time.sleep(1)

checkExist("4AD1781")
import pyrebase
from datetime import date
import time
from datetime import datetime
import random

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

def submitPlateNumber(PN):
    try:
        image_file = "image.jpg"
        now = datetime.now()
        dateToday = str(date.today())
        timeToday = now.strftime("%H:%M:%S")
        isApprehended = db.child("Vehicle_with_criminal_offense").child(PN).child("apprehended").get()
        crimeScanned = db.child("Vehicle_with_criminal_offense").child(PN).child("criminalOffense").get()

        # Upload image to Firebase storage
        storage.child(image_file).put(image_file)

        # Get view link of uploaded file
        view_link = storage.child(image_file).get_url(None)

        # Scanned
        data = {"PlateNumber":PN, "Location": "Lapasan Zone 4", "Date": dateToday, "Time": timeToday, "Notification": "on", "Apprehended": isApprehended.val(), "CriminalOffense": crimeScanned.val(), "ImageLink": view_link}
        db.child("Scanned").child((dateToday+" "+timeToday)).set(data)

        # ScannedPlateNmber
        dataPlateNumber = {"PlateNumber":PN, "Apprehended": isApprehended.val(),"CriminalOffense": crimeScanned.val(), "ImageLink": view_link}
        db.child("ScannedPlateNumber").child(PN).set(dataPlateNumber)

        #For Notification
        db.child("ScannedNotification").set(data)
        db.child("ScannedPlateNumberNotification").set(dataPlateNumber)

        print('plate number submitted to db')
    except Exception as e:
        
        print("Plate Number dont't exist "+ str(e))

# my_strings = ["001", "x", "321", "AAD1781", "y"]
# while True:  
#   plateNumber = random.choice(my_strings)
#   print(plateNumber)
#   submitPlateNumber(plateNumber)
#   time.sleep(1)

print("AAD1781")
submitPlateNumber("AAD1781")

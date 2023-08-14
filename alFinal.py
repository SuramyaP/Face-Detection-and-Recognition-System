import cv2
import numpy as np
import os
import face_recognition
import smtplib
import imghdr
from email.message import EmailMessage

def msg_owner():
    email_id = 'example@example.com'
    email_pass = 'passcode'

    msg = EmailMessage()

    msg['Subject'] = "HOME UPDATE!!!"
    msg['From'] = email_id
    msg['To'] = 'rjtdulal@gmail.com'
    msg.set_content("Here is the picture of the individual trying to enter your house.")

    with open('Unknown/image.jpg') as m:
        file_data = open('Unknown/image.jpg', 'rb').read()
        file_type = imghdr.what(m.name)
        # file_name = m.name
        file_name = "Person Identity"
        # print(file_name)

    msg.add_attachment(file_data, maintype = 'image',subtype = file_type, filename = file_name)

    with smtplib.SMTP_SSL('smtp.gmail.com',465) as smtp:
        smtp.login(email_id, email_pass)
        smtp.send_message(msg)


def capture_photo():
    face_classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml") 
    def face_cropped(img):
        RGBface = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
        faces = face_classifier.detectMultiScale(RGBface, 1.3, 5)
        
        if faces is ():
            return None
        for (x,y,w,h) in faces:
            # cropped_face = img[y:y+h,x:x+w]
            cropped_face = img
        return cropped_face
    
    cap = cv2.VideoCapture(0)
    img_id = 0
    
    while True:
        ret, frame = cap.read()
        if face_cropped(frame) is not None:
            img_id+=1 
            face = cv2.resize(face_cropped(frame), (200,200))
            # face = face_cropped(frame)
            #face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            file_name_path = "Unknown/"+"image.jpg"
            
            cv2.imwrite(file_name_path, face)
            
            
            
            #cv2.putText(face, str(img_id), (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2 )
            
            cv2.imshow("Cropped_Face", frame)
            if cv2.waitKey(1)==13 or int(img_id)==1:#<---- this value determines how many images captured after face appears 
                break
                
    cap.release()
    cv2.destroyAllWindows()
    print("Face Capture Completed !!!")

capture_photo()
    
KNOWN_FACES_DIR = "Known"
# UNKNOWN_FACES_DIR = "Unknown"
TOLERENCE = 0.6
FRAME_THICKNESS = 3
FONT_THICKNESS = 2
MODEL = "hog"

video = cv2.VideoCapture(1)
print("Loading Known Faces")

known_faces = []
known_names = []

for name in os.listdir(KNOWN_FACES_DIR):
    for filename in os.listdir(f"{KNOWN_FACES_DIR}/{name}"):
        print(filename)
        image = face_recognition.load_image_file(f"{KNOWN_FACES_DIR}/{name}/{filename}")
        print(image.shape)
        # image = cv2.resize(image,(1000,1000))
        # cv2.imshow("mero", image)
        # cv2.waitKey(0)
        try:
            encoding = face_recognition.face_encodings(image)[0]
            known_faces.append(encoding)
            known_names.append(name)
        except IndexError as e:
            print(e)

print("Processing Unknown Faces")
# for filename in os.listdir(UNKNOWN_FACES_DIR):
while True:
    # print(filename)
    # image = face_recognition.load_image_file(f"{UNKNOWN_FACES_DIR}/{filename}")
    ret, image = video.read()
    locations = face_recognition.face_locations(image, model = MODEL)
    encodings = face_recognition.face_encodings(image, locations)[0]
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    for face_encoding, face_location in zip(encodings, locations):
        results = face_recognition.compare_faces(known_faces, face_encoding, TOLERENCE)
        match = None
        if True in results:
            match = known_names[results.index(True)]
            print(f"Match found: {match}")
            
            top_left = (face_location[3], face_location[0])
            bottom_right = (face_location[1], face_location[2])
            color = [0,255,0]
            cv2.rectangle(image, top_left, bottom_right, color, FRAME_THICKNESS)
            
            top_left = (face_location[3], face_location[2])
            bottom_right = (face_location[1], face_location[2]+22)
            cv2.rectangle(image, top_left, bottom_right, color, cv2.FILLED)
            cv2.putText(image, match,(face_location[3]+10, face_location[2]+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), FONT_THICKNESS)
        else:
            print("Match Not Found")
                    
            
    cv2.imshow(filename, image)
            # cv2.waitKey(0)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        cv2.destroyAllWindows()
        break
            # cv2.destroyWindow(filename)

            
try:
    msg_owner()
except:
    print("Internet Not Available")

    

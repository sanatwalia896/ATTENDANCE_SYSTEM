import cv2
import numpy as np

# Initialise camera
cap = cv2.VideoCapture(0)

# Face Detection
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

skip = 0
face_data = []
dataset_path = './data/'
file_name = input("Enter the name of the person: ")

while True:
    ret,frame = cap.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if ret == False:
        continue

    faces = face_cascade.detectMultiScale(frame,1.3,5)
    faces = sorted(faces, key = lambda f : f[2]*f[3], reverse = True) # Reverse sorting gives the largest face first

    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)

        # Extract (crop out the required face) : region of interest
        offset = 10
        face_section = frame[y-offset : y+h+offset, x-offset : x+w+offset]
        face_section = cv2.resize(face_section,(100,100))

        skip+=1
        if skip%10 == 0:
            face_data.append(face_section)
            print(len(face_data))

    cv2.imshow("Frame",frame)

    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed == ord('q'):
        break

# Convert face list to a numpy array
face_data = np.asarray(face_data)
face_data = face_data.reshape((face_data.shape[0],-1))
print(face_data.shape)

#Save this data into file system
np.save(dataset_path + file_name + '.npy', face_data)
print("Data successfully saved at "+ dataset_path + file_name + '.npy')

cap.release()
cv2.destroyAllWindows()
import cv2
import json


cam = cv2.VideoCapture(0)
cam.set(3, 640)  # Set video width
cam.set(4, 480)  # Set video height
faceDetector = cv2.CascadeClassifier('haarcascade.xml')

faceId = input('\n Enter User ID (1, 2, 3, etc...): ')
faceName = input('\n Enter User Name: ')
print("\n [INFO] Wait the minutes ...")

# Write User Name to json file
with open('dataset.json', 'r') as jsonFile:  # Load from json file
    data = json.load(jsonFile)

if int(faceId) < len(data):  # Take face sample again
    if faceName != '':
        data[int(faceId)] = faceName
else:
    data.append(faceName)

with open('dataset.json', 'w') as jsonFile:  # Write to json file
    json.dump(data, jsonFile)

count = 0

while True:
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceDetector.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        count += 1
        # Save the captured image into the datasets folder
        cv2.imwrite("dataset/User." + str(faceId) + '.' +
                    str(count) + ".jpg", gray[y:y+h, x:x+w])
        cv2.imshow('image', img)

    k = cv2.waitKey(100) & 0xff  # Press 'ESC' for exiting video
    if k == 27:
        break
    elif count >= 30:  # Take 30 face sample and stop video
        break

print("\n [INFO] Done!")
cam.release()
cv2.destroyAllWindows()

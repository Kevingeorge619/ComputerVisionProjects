import cv2
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh #facemesh is a module inside mediapipe
face_mesh = mp_face_mesh.FaceMesh()

cap = cv2.VideoCapture(0) # 0 is port number of the camera, VideoCapture is a class and cap is the object

while cap.isOpened():
    key, img  = cap.read()
    if not key:
        break
   # key, img = cap.read() # reads the video and returns 2 values a key and an image
    rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # convert color
    results = face_mesh.process(rgb_frame)  # face landmarks processing and puts in results

    if results.multi_face_landmarks:
        for landmarks in results.multi_face_landmarks:
            for point in landmarks.landmark:
                x, y = int(point.x * img.shape[1]), int(point.y * img.shape[0])
                cv2.circle(img, (x, y), 1, (0, 255, 0), -1)

    cv2.imshow("my_video",img) #to show the image - my_video is the window name
    cv2.waitKey(1) # to wait to capture one image at a time

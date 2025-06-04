import cv2
import mediapipe as mp
import pygame
import time
import os
from email.message import EmailMessage
import smtplib
FROM_EMAIL = ""
TO_EMAIL = ""
APP_PASSWORD = ""

pygame.init()
pygame.mixer.init()

alert_sound = pygame.mixer.Sound("Alert.mp3")
alert_played = False  # To avoid repeated alerts
previously_detected = False

# Flags to avoid repeating actions
alert_played = False
image_captured = False  # ✅ Define this here before using it
last_capture_time = 0
capture_interval = 10  # seconds

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    current_time = time.time()

    if results.pose_landmarks:
        mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        print("Human Detected")
        if not alert_played:
            alert_sound.play()
            alert_played = True

            if current_time - last_capture_time > capture_interval:
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                filename = f"human_detected_{timestamp}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Image saved: {filename}")
                last_capture_time = current_time

                msg = EmailMessage()
                msg["Subject"] = "📷 Human Detected - Screenshot Attached"
                msg["From"] = FROM_EMAIL
                msg["To"] = TO_EMAIL
                msg.set_content("A screenshot was captured upon detecting a human. Please find it attached.")

                with open(filename, "rb") as f:
                    file_data = f.read()
                    file_name = os.path.basename(filename)
                    msg.add_attachment(file_data, maintype="image", subtype="png", filename=file_name)

                try:
                    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
                        smtp.login(FROM_EMAIL, APP_PASSWORD)
                        smtp.send_message(msg)
                    print("✅ Email sent successfully!")
                except Exception as e:
                    print(f"❌ Failed to send email: {e}")

        else:
            alert_played = False  # Reset if no human is detected
            image_captured = False # reset if no human is detected
            previously_detected = False

    cv2.imshow('Pose Estimation', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

import cv2
from deepface import DeepFace
import os
import time

# Create folder to store registered faces
os.makedirs("registered_faces", exist_ok=True)

print("=== FACE RECOGNITION TEST ===")
print("Step 1: We will take your photo as registered person")
print("Step 2: Camera will check if you are authorised or not")
print("")

# Step 1 - Take photo of registered person
cap = cv2.VideoCapture(0)
print("Look at camera. Press SPACE to capture your photo...")

while True:
    ret, frame = cap.read()
    cv2.imshow("Capture - Press SPACE", frame)
    
    if cv2.waitKey(1) & 0xFF == ord(' '):
        # Save the photo
        cv2.imwrite("registered_faces/person1.jpg", frame)
        print("Photo saved as registered person!")
        break

cap.release()
cv2.destroyAllWindows()

print("")
print("Now testing recognition. Starting camera...")
time.sleep(2)

# Step 2 - Test recognition
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]
        
        try:
            # Save current face temporarily
            cv2.imwrite("temp_face.jpg", face_img)
            
            # Compare with registered face
            result = DeepFace.verify(
                "temp_face.jpg",
                "registered_faces/person1.jpg",
                enforce_detection=False
            )
            
            if result["verified"]:
                color = (0, 255, 0)  # Green
                label = "AUTHORISED"
            else:
                color = (0, 0, 255)  # Red
                label = "UNAUTHORISED"
                
        except Exception as e:
            color = (255, 165, 0)  # Orange
            label = "UNIDENTIFIED"

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (x, y-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("Face Recognition Test - Press Q to quit", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
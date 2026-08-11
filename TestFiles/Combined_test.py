from ultralytics import YOLO
from deepface import DeepFace
import cv2
import time
import os

# Settings
EXPECTED_COUNT = 2
INACTIVITY_THRESHOLD = 30  # seconds
POSITION_THRESHOLD = 80    # pixels

# Load models
model = YOLO("yolov8n.pt")
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# Variables
last_child_position = None
inactivity_start = None
os.makedirs("registered_faces", exist_ok=True)
os.makedirs("snapshots", exist_ok=True)

cap = cv2.VideoCapture(0)
print("Combined system test running. Press Q to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Check brightness
    brightness = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean()
    if brightness < 50:
        cv2.putText(frame, "LOW LIGHT ALERT", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        print("LOW LIGHT DETECTED")
        cv2.imshow("Combined Test", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    # Person detection and count
    results = model(frame, verbose=False)
    person_count = 0
    child_position = None

    for result in results:
        for box in result.boxes:
            if int(box.cls[0]) == 0:
                person_count += 1
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                if person_count == 1:
                    child_position = (cx, cy)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, "Child", (x1, y1-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                else:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # Person count alert
    if person_count > EXPECTED_COUNT:
        cv2.putText(frame, "EXTRA PERSON ALERT", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        print("ALERT: Extra person detected")
        timestamp = time.strftime("%H%M%S")
        cv2.imwrite(f"snapshots/alert_{timestamp}.jpg", frame)

    # Inactivity detection
    if child_position:
        if last_child_position:
            distance = ((child_position[0] - last_child_position[0])**2 +
                       (child_position[1] - last_child_position[1])**2)**0.5

            if distance < POSITION_THRESHOLD:
                if inactivity_start is None:
                    inactivity_start = time.time()
                else:
                    inactive_time = time.time() - inactivity_start
                    timer_text = f"Still: {int(inactive_time)}s / {INACTIVITY_THRESHOLD}s"
                    cv2.putText(frame, timer_text, (10, 110),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)

                    if inactive_time >= INACTIVITY_THRESHOLD:
                        print("ALERT: Child inactivity detected")
                        cv2.putText(frame, "INACTIVITY ALERT", (10, 150),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            else:
                inactivity_start = None

        last_child_position = child_position

    # Display count
    cv2.putText(frame, f"Persons: {person_count}", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow("Combined Test - Press Q to quit", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
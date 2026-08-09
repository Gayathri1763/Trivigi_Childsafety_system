from ultralytics import YOLO
import cv2

# Load YOLOv8 nano model
model = YOLO("yolov8n.pt")

# Set expected number of persons
EXPECTED_COUNT = 1

cap = cv2.VideoCapture(0)
print("Person counting started. Press Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run detection
    results = model(frame, verbose=False)

    # Count only persons (class 0 in YOLO is person)
    person_count = 0
    for result in results:
        for box in result.boxes:
            if int(box.cls[0]) == 0:  # Person class
                person_count += 1
                # Draw box around person
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, "Person", (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # Show count and alert
    count_text = f"Count: {person_count} / Expected: {EXPECTED_COUNT}"
    cv2.putText(frame, count_text, (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    if person_count > EXPECTED_COUNT:
        alert_text = "ALERT: Extra person detected!"
        cv2.putText(frame, alert_text, (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        print(f"ALERT: {person_count} persons detected, expected {EXPECTED_COUNT}")

    cv2.imshow("Person Count Test - Press Q to quit", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
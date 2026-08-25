import cv2
import os
import time
import numpy as np
from deepface import DeepFace
from ultralytics import YOLO

# ── Load models ───────────────────────────────────────────────────
model        = YOLO("yolov8n.pt")
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

REGISTERED_DIR = "registered_faces"
SNAPSHOT_DIR   = "snapshots"
os.makedirs(REGISTERED_DIR, exist_ok=True)
os.makedirs(SNAPSHOT_DIR,   exist_ok=True)


def save_snapshot(alert_type, frame):
    ts       = time.strftime("%Y%m%d_%H%M%S")
    filename = f"{alert_type}_{ts}.jpg"
    path     = os.path.join(SNAPSHOT_DIR, filename)
    cv2.imwrite(path, frame)
    return filename


def is_face_covered(face_img):
    if face_img is None or face_img.size == 0:
        return False
    h, w = face_img.shape[:2]
    if h < 20 or w < 20:
        return False
    upper     = cv2.cvtColor(face_img[:h//2, :], cv2.COLOR_BGR2GRAY)
    lower     = cv2.cvtColor(face_img[h//2:, :], cv2.COLOR_BGR2GRAY)
    upper_var = np.var(upper)
    lower_var = np.var(lower)
    if upper_var > 0 and lower_var / (upper_var + 1e-5) < 0.35:
        return True
    return False


def get_registered_photos():
    if not os.path.exists(REGISTERED_DIR):
        return []
    return [
        os.path.join(REGISTERED_DIR, f)
        for f in os.listdir(REGISTERED_DIR)
        if f.lower().endswith((".jpg", ".png", ".jpeg"))
    ]


def classify_face(face_crop, registered_photos):
    if is_face_covered(face_crop):
        return "UNIDENTIFIED", (0, 165, 255)
    if not registered_photos:
        return "NO REGISTERED FACE", (200, 200, 200)
    try:
        temp_path = os.path.join(SNAPSHOT_DIR, "temp_check.jpg")
        cv2.imwrite(temp_path, face_crop)
        for reg_photo in registered_photos:
            result = DeepFace.verify(
                temp_path,
                reg_photo,
                enforce_detection=False,
                silent=True
            )
            if result["verified"]:
                return "AUTHORIZED", (0, 255, 0)
        return "UNAUTHORIZED", (0, 0, 255)
    except Exception:
        return "UNIDENTIFIED", (0, 165, 255)


def process_frame(frame, settings, last_state):
    alerts             = []
    expected_count     = settings.get("expected_count", 2)
    inactivity_seconds = settings.get("inactivity_seconds", 30)
    position_threshold = settings.get("position_threshold", 80)
    mode               = settings.get("mode", "child")

    last_child_pos   = last_state.get("last_child_pos")
    inactivity_start = last_state.get("inactivity_start")

    # ── Low light ─────────────────────────────────────────────────
    brightness = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean()
    if brightness < 50:
        cv2.putText(frame, "LOW LIGHT — Detection paused",
                    (10, 35), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 255), 2)
        alerts.append({
            "type":     "LOW_LIGHT",
            "message":  "Room visibility insufficient",
            "time":     time.strftime("%H:%M:%S"),
            "snapshot": None
        })
        return frame, alerts, last_state

    # ── Person detection ──────────────────────────────────────────
    results      = model(frame, verbose=False, imgsz=320)
    person_boxes = []
    for result in results:
        for box in result.boxes:
            if int(box.cls[0]) == 0:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                person_boxes.append((x1, y1, x2, y2))

    person_count = len(person_boxes)

    # ── Count alert ───────────────────────────────────────────────
    if person_count > expected_count:
        snap = save_snapshot("count_alert", frame)
        alerts.append({
            "type":     "COUNT_ALERT",
            "message":  f"{person_count} persons detected — expected {expected_count}",
            "time":     time.strftime("%H:%M:%S"),
            "snapshot": snap
        })
        cv2.putText(frame,
                    f"COUNT ALERT: {person_count}/{expected_count}",
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (0, 0, 255), 2)

    # ── Face recognition ──────────────────────────────────────────
    gray              = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces             = face_cascade.detectMultiScale(gray, 1.1, 4)
    registered_photos = get_registered_photos()

    for (fx, fy, fw, fh) in faces:
        face_crop      = frame[fy:fy+fh, fx:fx+fw]
        status, color  = classify_face(face_crop, registered_photos)

        cv2.rectangle(frame, (fx, fy), (fx+fw, fy+fh), color, 2)
        cv2.putText(frame, status, (fx, fy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

        if status == "UNAUTHORIZED":
            snap = save_snapshot("unauthorized", frame)
            alerts.append({
                "type":     "UNAUTHORIZED",
                "message":  "Unauthorized person detected",
                "time":     time.strftime("%H:%M:%S"),
                "snapshot": snap
            })
        elif status == "UNIDENTIFIED":
            snap = save_snapshot("unidentified", frame)
            alerts.append({
                "type":     "UNIDENTIFIED",
                "message":  "Unidentified person — face covered or unclear",
                "time":     time.strftime("%H:%M:%S"),
                "snapshot": snap
            })

    # ── Child tracking and inactivity ─────────────────────────────
    if mode == "child" and person_boxes:
        x1, y1, x2, y2 = person_boxes[0]
        cx              = (x1 + x2) // 2
        cy              = (y1 + y2) // 2
        child_pos       = (cx, cy)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, "Child", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        for x1b, y1b, x2b, y2b in person_boxes[1:]:
            cv2.rectangle(frame, (x1b, y1b), (x2b, y2b), (255, 0, 0), 2)
            cv2.putText(frame, "Person", (x1b, y1b - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        if last_child_pos is not None:
            dist = (
                (child_pos[0] - last_child_pos[0]) ** 2 +
                (child_pos[1] - last_child_pos[1]) ** 2
            ) ** 0.5

            if dist < position_threshold:
                if inactivity_start is None:
                    inactivity_start = time.time()
                else:
                    inactive_secs = int(time.time() - inactivity_start)
                    cv2.putText(frame,
                                f"Still: {inactive_secs}s/{inactivity_seconds}s",
                                (10, 110), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (255, 165, 0), 2)
                    if inactive_secs >= inactivity_seconds:
                        snap = save_snapshot("inactivity", frame)
                        alerts.append({
                            "type":     "INACTIVITY",
                            "message":  "Child has not changed position",
                            "time":     time.strftime("%H:%M:%S"),
                            "snapshot": snap
                        })
                        inactivity_start = None
            else:
                inactivity_start = None

        last_child_pos = child_pos

    elif mode == "infant":
        cv2.putText(frame, "INFANT MODE",
                    (10, 35), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 0), 2)

    # ── Update state ──────────────────────────────────────────────
    last_state["last_child_pos"]   = last_child_pos
    last_state["inactivity_start"] = inactivity_start

    # ── Person count display ──────────────────────────────────────
    cv2.putText(frame,
                f"Persons: {person_count}/{expected_count}",
                (10, frame.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return frame, alerts, last_state
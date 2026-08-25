import cv2
import os
import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from flask import Flask, render_template, Response, request, jsonify, send_from_directory
from flask_socketio import SocketIO

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")
executor = ThreadPoolExecutor(max_workers=2)

# ── Paths ─────────────────────────────────────────────────────────
REGISTERED_DIR = "registered_faces"
SNAPSHOT_DIR   = "snapshots"
SETTINGS_FILE  = "settings.json"
os.makedirs(REGISTERED_DIR, exist_ok=True)
os.makedirs(SNAPSHOT_DIR,   exist_ok=True)

# ── Default settings ──────────────────────────────────────────────
DEFAULT_SETTINGS = {
    "expected_count":     2,
    "inactivity_seconds": 30,
    "mode":               "child",
    "quick_entry_alert":  True
}

# ── Shared state ──────────────────────────────────────────────────
alerts_log       = []
current_frame    = None
frame_lock       = threading.Lock()
last_child_pos   = None
inactivity_start = None
last_alert_time  = {}
face_pending     = False


# ── Settings ──────────────────────────────────────────────────────
def load_settings():
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE) as f:
                return json.load(f)
        except Exception:
            return DEFAULT_SETTINGS.copy()
    return DEFAULT_SETTINGS.copy()


def save_settings_file(data):
    with open(SETTINGS_FILE, "w") as f:
        json.dump(data, f)


# ── Snapshot and Alerts ───────────────────────────────────────────
def take_snapshot(label, frame):
    ts       = time.strftime("%Y%m%d_%H%M%S")
    filename = f"{label}_{ts}.jpg"
    cv2.imwrite(os.path.join(SNAPSHOT_DIR, filename), frame)
    return filename


def add_alert(alert_type, message, snapshot=None, cooldown=5):
    now = time.time()
    if alert_type in last_alert_time and (now - last_alert_time[alert_type]) < cooldown:
        return
    last_alert_time[alert_type] = now
    alert_item = {
        "type":     alert_type,
        "message":  message,
        "time":     time.strftime("%H:%M:%S"),
        "snapshot": snapshot,
        "resolved": None
    }
    alerts_log.append(alert_item)
    socketio.emit("new_alert", alert_item)


# ── Face helpers ──────────────────────────────────────────────────
def get_registered():
    return [
        os.path.join(REGISTERED_DIR, f)
        for f in os.listdir(REGISTERED_DIR)
        if f.lower().endswith((".jpg", ".png", ".jpeg"))
    ]


def is_covered(face_img):
    if face_img is None or face_img.size == 0:
        return False
    h, w = face_img.shape[:2]
    if h < 20 or w < 20:
        return False
    upper_var = float(cv2.cvtColor(
        face_img[:h//2, :], cv2.COLOR_BGR2GRAY).var())
    lower_var = float(cv2.cvtColor(
        face_img[h//2:, :], cv2.COLOR_BGR2GRAY).var())
    return upper_var > 0 and lower_var / (upper_var + 1e-5) < 0.35


def check_face_async(face_crop):
    global face_pending
    try:
        from deepface import DeepFace
        if is_covered(face_crop):
            status = "UNIDENTIFIED"
        else:
            reg = get_registered()
            if not reg:
                status = "NO_REG"
            else:
                tmp = os.path.join(
                    SNAPSHOT_DIR,
                    f"tmp_check_{threading.get_ident()}.jpg"
                )
                cv2.imwrite(tmp, face_crop)
                status = "UNAUTHORIZED"
                try:
                    for r in reg:
                        res = DeepFace.verify(
                            tmp, r,
                            enforce_detection=False,
                            silent=True
                        )
                        if res.get("verified", False):
                            status = "AUTHORIZED"
                            break
                finally:
                    if os.path.exists(tmp):
                        os.remove(tmp)

        if status == "UNAUTHORIZED":
            add_alert("UNAUTHORIZED", "Unauthorized person detected")
        elif status == "UNIDENTIFIED":
            add_alert("UNIDENTIFIED", "Face covered or unclear")

    except Exception as e:
        print(f"[ERROR] Face processing error: {e}")
    finally:
        face_pending = False


# ── Background camera thread ──────────────────────────────────────
def camera_loop():
    global current_frame, last_child_pos, inactivity_start, face_pending

    from ultralytics import YOLO
    from picamera2 import Picamera2

    # ── Initialize Picamera2 ──────────────────────────────────────
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"size": (640, 480), "format": "XBGR8888"}
    )
    picam2.configure(config)
    picam2.start()
    time.sleep(2)
    print("[INFO] Picamera2 started successfully")

    # ── Load YOLO ─────────────────────────────────────────────────
    yolo = YOLO("yolov8n.pt")

    # ── Load face cascade ─────────────────────────────────────────
    cascade_path = "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"
    fc = cv2.CascadeClassifier(cascade_path)
    if fc.empty():
        print("[WARN] Face cascade not loaded")
        fc = None

    face_check_interval = 10
    frame_idx = 0

    while True:
        # ── Capture frame from Picamera2 ──────────────────────────
        frame_rgba = picam2.capture_array()
        frame = cv2.cvtColor(frame_rgba, cv2.COLOR_BGRA2BGR)

        settings     = load_settings()
        expected     = settings.get("expected_count", 2)
        inact_thresh = settings.get("inactivity_seconds", 30)
        mode         = settings.get("mode", "child")
        pos_thresh   = 80
        frame_idx   += 1

        # ── Low light check ───────────────────────────────────────
        bright = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean()
        if bright < 50:
            cv2.putText(frame, "LOW LIGHT", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            add_alert("LOW_LIGHT", "Room too dark")
            with frame_lock:
                current_frame = frame.copy()
            time.sleep(0.04)
            continue

        # ── YOLO person detection ─────────────────────────────────
        results      = yolo(frame, verbose=False, imgsz=320)
        person_boxes = []
        for r in results:
            for b in r.boxes:
                if int(b.cls[0]) == 0:
                    x1, y1, x2, y2 = map(int, b.xyxy[0])
                    person_boxes.append((x1, y1, x2, y2))

        count = len(person_boxes)

        # ── Count alert ───────────────────────────────────────────
        if count > expected:
            snap = take_snapshot("count", frame)
            add_alert("COUNT_ALERT",
                      f"{count} persons detected, expected {expected}",
                      snap)

        # ── Face check every N frames ─────────────────────────────
        if fc is not None and frame_idx % face_check_interval == 0:
            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = fc.detectMultiScale(gray, 1.1, 4)
            for (fx, fy, fw, fh) in faces:
                cv2.rectangle(frame, (fx, fy),
                              (fx+fw, fy+fh), (255, 165, 0), 2)
                if not face_pending:
                    face_pending = True
                    crop = frame[fy:fy+fh, fx:fx+fw].copy()
                    executor.submit(check_face_async, crop)

        # ── Draw person boxes ─────────────────────────────────────
        colors_box = [(0,255,0),(255,0,0),(0,255,255),(255,0,255)]
        for i, (x1, y1, x2, y2) in enumerate(person_boxes):
            c = colors_box[min(i, len(colors_box)-1)]
            cv2.rectangle(frame, (x1,y1), (x2,y2), c, 2)
            label = "Child" if i == 0 and mode == "child" else "Person"
            cv2.putText(frame, label, (x1, y1-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2)

        # ── Inactivity detection ──────────────────────────────────
        if mode == "child" and person_boxes:
            x1, y1, x2, y2 = person_boxes[0]
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            pos = (cx, cy)

            if last_child_pos:
                dist = ((pos[0]-last_child_pos[0])**2 +
                        (pos[1]-last_child_pos[1])**2)**0.5
                if dist < pos_thresh:
                    if inactivity_start is None:
                        inactivity_start = time.time()
                    else:
                        secs = int(time.time()-inactivity_start)
                        cv2.putText(frame,
                                    f"Still:{secs}s/{inact_thresh}s",
                                    (10,60), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7, (255,165,0), 2)
                        if secs >= inact_thresh:
                            snap = take_snapshot("inact", frame)
                            add_alert("INACTIVITY",
                                      "Child not moved", snap)
                            inactivity_start = None
                else:
                    inactivity_start = None
            last_child_pos = pos

        # ── HUD ───────────────────────────────────────────────────
        cv2.putText(frame,
                    f"Persons:{count}/{expected} | {mode.upper()} MODE",
                    (10, frame.shape[0]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2)

        with frame_lock:
            current_frame = frame.copy()

        time.sleep(0.03)


# ── MJPEG stream ──────────────────────────────────────────────────
def generate():
    while True:
        with frame_lock:
            if current_frame is None:
                time.sleep(0.05)
                continue
            ret, buf = cv2.imencode(
                ".jpg", current_frame,
                [cv2.IMWRITE_JPEG_QUALITY, 70]
            )
            if not ret:
                continue
            data = buf.tobytes()

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n"
               + data + b"\r\n")
        time.sleep(0.04)


# ── Routes ────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html",
                           settings=load_settings(),
                           alerts=alerts_log[-5:])


@app.route("/video_feed")
def video_feed():
    return Response(generate(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/register", methods=["GET"])
def register():
    faces = [f for f in os.listdir(REGISTERED_DIR)
             if f.lower().endswith((".jpg", ".png", ".jpeg"))]
    return render_template("register.html", faces=faces)


@app.route("/upload_face", methods=["POST"])
def upload_face():
    if "photo" not in request.files:
        return jsonify({"success": False, "message": "No file"})
    f     = request.files["photo"]
    label = request.form.get("label", "person")
    ts    = time.strftime("%Y%m%d_%H%M%S")
    name  = f"{label}_{ts}.jpg"
    f.save(os.path.join(REGISTERED_DIR, name))
    return jsonify({"success": True, "message": f"Saved: {name}"})


@app.route("/delete_face/<filename>", methods=["POST"])
def delete_face(filename):
    p = os.path.join(REGISTERED_DIR, filename)
    if os.path.exists(p):
        os.remove(p)
    return jsonify({"success": True})


@app.route("/alerts")
def alerts_page():
    return render_template("alerts.html",
                           alerts=list(reversed(alerts_log)))


@app.route("/resolve_alert", methods=["POST"])
def resolve_alert():
    d      = request.get_json()
    idx    = d.get("index", -1)
    status = d.get("status", "safe")
    rev    = list(reversed(alerts_log))
    if 0 <= idx < len(rev):
        rev[idx]["resolved"] = status
    return jsonify({"success": True})


@app.route("/clear_alerts", methods=["POST"])
def clear_alerts():
    alerts_log.clear()
    return jsonify({"success": True})


@app.route("/settings")
def settings_page():
    return render_template("settings.html", settings=load_settings())


@app.route("/save_settings", methods=["POST"])
def save_settings_route():
    save_settings_file(request.get_json())
    return jsonify({"success": True, "message": "Saved"})


@app.route("/snapshots/<path:filename>")
def snap(filename):
    return send_from_directory(SNAPSHOT_DIR, filename)


@app.route("/registered_faces/<path:filename>")
def reg_face(filename):
    return send_from_directory(REGISTERED_DIR, filename)


@app.route("/shutdown", methods=["POST"])
def shutdown():
    os.system("sudo shutdown now")
    return jsonify({"success": True})


# ── SocketIO events ───────────────────────────────────────────────
@socketio.on("connect")
def handle_connect():
    from flask_socketio import emit
    emit("connected", {"message": "Connected to Trivigi"})


# ── Entry point ───────────────────────────────────────────────────
if __name__ == "__main__":
    t = threading.Thread(target=camera_loop, daemon=True)
    t.start()
    print("Trivigi running → http://localhost:5000")
    socketio.run(app, host="0.0.0.0", port=5000, debug=False)
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
CHILD_DIR      = "registered_faces/child"
TRUSTED_DIR    = "registered_faces/trusted"
SNAPSHOT_DIR   = "snapshots"
SETTINGS_FILE  = "settings.json"

os.makedirs(CHILD_DIR,    exist_ok=True)
os.makedirs(TRUSTED_DIR,  exist_ok=True)
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

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
def get_photos_from(folder):
    if not os.path.exists(folder):
        return []
    return [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith((".jpg", ".png", ".jpeg"))
    ]


def is_covered(face_img):
    """
    Detects if face is masked or covered using variance ratio method.
    Upper half of face (eyes/forehead) has high variance.
    Lower half (nose/mouth) covered by mask has low variance.
    """
    if face_img is None or face_img.size == 0:
        return False
    h, w = face_img.shape[:2]
    if h < 30 or w < 30:
        return False

    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)

    upper_half = gray[:h//2, :]
    lower_half = gray[h//2:, :]

    upper_var = float(upper_half.var())
    lower_var = float(lower_half.var())

    # Additional edge detection check for mask boundary
    edges = cv2.Canny(gray, 50, 150)
    upper_edges = edges[:h//2, :].sum()
    lower_edges = edges[h//2:, :].sum()

    # If lower half has significantly less variance and fewer edges
    variance_ratio = lower_var / (upper_var + 1e-5)
    edge_ratio = lower_edges / (upper_edges + 1e-5)

    if variance_ratio < 0.4 and edge_ratio < 0.6:
        return True

    return False


def verify_against_folder(tmp_path, folder):
    """Returns True if face matches any photo in folder."""
    from deepface import DeepFace
    photos = get_photos_from(folder)
    for photo in photos:
        try:
            result = DeepFace.verify(
                tmp_path, photo,
                enforce_detection=False,
                silent=True
            )
            if result.get("verified", False):
                return True
        except Exception:
            continue
    return False


def check_face_async(face_crop):
    """
    Three state classification:
    1. Check if face is covered → UNIDENTIFIED
    2. Check if matches child folder → tracked as child
    3. Check if matches trusted folder → AUTHORIZED
    4. No match → UNAUTHORIZED
    """
    global face_pending
    try:
        # Step 1 — Mask check first
        if is_covered(face_crop):
            add_alert("UNIDENTIFIED",
                      "Covered or masked face detected — flagged as suspicious")
            return

        tmp = os.path.join(SNAPSHOT_DIR,
                           f"tmp_{threading.get_ident()}.jpg")
        cv2.imwrite(tmp, face_crop)

        try:
            # Step 2 — Check child folder
            if verify_against_folder(tmp, CHILD_DIR):
                # Child recognized — no alert, tracking handles this
                return

            # Step 3 — Check trusted folder
            if verify_against_folder(tmp, TRUSTED_DIR):
                # Trusted member — authorized, no alert
                return

            # Step 4 — No match found
            add_alert("UNAUTHORIZED",
                      "Unauthorized person detected")

        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

    except Exception as e:
        print(f"[ERROR] Face check error: {e}")
        add_alert("UNIDENTIFIED", "Face unclear or unidentifiable")
    finally:
        face_pending = False


# ── Background camera thread ──────────────────────────────────────
def camera_loop():
    global current_frame, last_child_pos, inactivity_start, face_pending

    from ultralytics import YOLO
    from picamera2 import Picamera2
    from picamera2 import controls

    settings  = load_settings()
    mode      = settings.get("mode", "child")
    last_mode = mode

    def start_camera(current_mode):
        cam = Picamera2()
        if current_mode == "infant":
            # Full wide angle — maximum zoom out
            cfg = cam.create_preview_configuration(
                main={"size": (1920, 1080), "format": "XBGR8888"},
                controls={"ScalerCrop": cam.camera_properties["ScalerCropMaximum"]}
            )
        else:
            # Child mode — standard view
            cfg = cam.create_preview_configuration(
                main={"size": (640, 480), "format": "XBGR8888"}
            )
        cam.configure(cfg)
        cam.start()

        if current_mode == "infant":
            # Set zoom to minimum (fully zoomed out)
            cam.set_controls({"ScalerCrop": cam.camera_properties["ScalerCropMaximum"]})

        time.sleep(2)
        print(f"[INFO] Camera started — mode: {current_mode}")
        return cam

    picam2 = start_camera(mode)

    yolo = YOLO("yolov8n.pt")

    cascade_path = "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"
    fc = cv2.CascadeClassifier(cascade_path)
    if fc.empty():
        fc = None

    face_check_interval = 10
    frame_idx           = 0

    while True:
        # Check if mode changed — restart camera with new settings
        settings = load_settings()
        mode     = settings.get("mode", "child")

        if mode != last_mode:
            print(f"[INFO] Mode changed to {mode} — restarting camera")
            picam2.stop()
            picam2.close()
            picam2    = start_camera(mode)
            last_mode = mode
            frame_idx = 0
            continue

        # Capture frame
        frame_rgba = picam2.capture_array()
        frame      = cv2.cvtColor(frame_rgba, cv2.COLOR_BGRA2BGR)

        # Resize infant mode frame for processing
        # (keep display full wide but process smaller version)
        if mode == "infant":
            process_frame = cv2.resize(frame, (640, 360))
        else:
            process_frame = frame

        expected     = settings.get("expected_count", 2)
        inact_thresh = settings.get("inactivity_seconds", 30)
        pos_thresh   = 80
        frame_idx   += 1

        # Low light check
        bright = cv2.cvtColor(process_frame, cv2.COLOR_BGR2GRAY).mean()
        if bright < 50:
            cv2.putText(frame, "LOW LIGHT — Detection paused",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 0, 255), 2)
            add_alert("LOW_LIGHT", "Room too dark")
            with frame_lock:
                current_frame = cv2.resize(frame, (854, 480)) if mode == "infant" else frame.copy()
            time.sleep(0.04)
            continue

        # YOLO on smaller frame for speed
        results      = yolo(process_frame, verbose=False, imgsz=320)
        person_boxes = []
        scale_x = frame.shape[1] / process_frame.shape[1]
        scale_y = frame.shape[0] / process_frame.shape[0]

        for r in results:
            for b in r.boxes:
                if int(b.cls[0]) == 0:
                    x1, y1, x2, y2 = map(int, b.xyxy[0])
                    # Scale boxes back to original frame size
                    x1 = int(x1 * scale_x)
                    y1 = int(y1 * scale_y)
                    x2 = int(x2 * scale_x)
                    y2 = int(y2 * scale_y)
                    person_boxes.append((x1, y1, x2, y2))

        count = len(person_boxes)

        # Count alert
        if count > expected:
            snap = take_snapshot("count", frame)
            add_alert("COUNT_ALERT",
                      f"{count} persons detected — expected {expected}",
                      snap)

        # Face recognition
        if fc is not None and frame_idx % face_check_interval == 0:
            gray  = cv2.cvtColor(process_frame, cv2.COLOR_BGR2GRAY)
            faces = fc.detectMultiScale(gray, 1.1, 4)
            for (fx, fy, fw, fh) in faces:
                if not face_pending:
                    face_pending = True
                    crop = process_frame[fy:fy+fh, fx:fx+fw].copy()
                    executor.submit(check_face_async, crop)

        # Draw boxes
        colors_box = [(0,255,0),(255,0,0),(0,255,255),(255,0,255)]
        for i, (x1, y1, x2, y2) in enumerate(person_boxes):
            c = colors_box[min(i, len(colors_box)-1)]
            cv2.rectangle(frame, (x1,y1), (x2,y2), c, 2)
            label = "Child" if i == 0 and mode == "child" else "Person"
            cv2.putText(frame, label, (x1, y1-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2)

        # Child mode inactivity
        if mode == "child" and person_boxes:
            x1, y1, x2, y2 = person_boxes[0]
            cx  = (x1+x2)//2
            cy  = (y1+y2)//2
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
                                    f"Still: {secs}s / {inact_thresh}s",
                                    (10, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7, (255,165,0), 2)
                        if secs >= inact_thresh:
                            snap = take_snapshot("inact", frame)
                            add_alert("INACTIVITY",
                                      "Child has not changed position",
                                      snap)
                            inactivity_start = None
                else:
                    inactivity_start = None
            last_child_pos = pos

        elif mode == "infant":
            cv2.putText(frame,
                        "INFANT MODE — Full room view",
                        (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (255,255,0), 2)

        # HUD
        cv2.putText(frame,
                    f"Persons: {count} / Expected: {expected} | {mode.upper()} MODE",
                    (10, frame.shape[0]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2)

        # Resize for streaming — keep manageable size
        display = cv2.resize(frame, (854, 480)) if mode == "infant" else frame

        with frame_lock:
            current_frame = display.copy()

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
    child_faces   = [f for f in os.listdir(CHILD_DIR)
                     if f.lower().endswith((".jpg",".png",".jpeg"))]
    trusted_faces = [f for f in os.listdir(TRUSTED_DIR)
                     if f.lower().endswith((".jpg",".png",".jpeg"))]
    return render_template("register.html",
                           child_faces=child_faces,
                           trusted_faces=trusted_faces)


@app.route("/upload_face", methods=["POST"])
def upload_face():
    if "photo" not in request.files:
        return jsonify({"success": False, "message": "No file"})
    f        = request.files["photo"]
    label    = request.form.get("label", "person")
    category = request.form.get("category", "trusted")
    ts       = time.strftime("%Y%m%d_%H%M%S")
    name     = f"{label}_{ts}.jpg"

    folder = CHILD_DIR if category == "child" else TRUSTED_DIR
    f.save(os.path.join(folder, name))
    return jsonify({"success": True,
                    "message": f"Registered in {category}: {name}"})


@app.route("/delete_face/<category>/<filename>", methods=["POST"])
def delete_face(category, filename):
    folder = CHILD_DIR if category == "child" else TRUSTED_DIR
    p = os.path.join(folder, filename)
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


@app.route("/registered_faces/child/<path:filename>")
def reg_child(filename):
    return send_from_directory(CHILD_DIR, filename)


@app.route("/registered_faces/trusted/<path:filename>")
def reg_trusted(filename):
    return send_from_directory(TRUSTED_DIR, filename)


@app.route("/shutdown", methods=["POST"])
def shutdown():
    os.system("sudo shutdown now")
    return jsonify({"success": True})


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
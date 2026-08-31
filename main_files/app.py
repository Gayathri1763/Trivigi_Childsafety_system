import cv2
import os
import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from flask import (Flask, render_template, Response,
                   request, jsonify, send_from_directory)
from flask_socketio import SocketIO

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*",
                    async_mode="threading")
executor = ThreadPoolExecutor(max_workers=2)

# ── Paths ──────────────────────────────────────────────────────────
CHILD_DIR     = "registered_faces/child"
TRUSTED_DIR   = "registered_faces/trusted"
SNAPSHOT_DIR  = "snapshots"
SETTINGS_FILE = "settings.json"

os.makedirs(CHILD_DIR,    exist_ok=True)
os.makedirs(TRUSTED_DIR,  exist_ok=True)
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

# ── Defaults ───────────────────────────────────────────────────────
DEFAULT_SETTINGS = {
    "expected_count":     2,
    "inactivity_seconds": 30,
    "mode":               "child",
    "quick_entry_alert":  True,
    "zone_size":          250
}

# ── Shared state ───────────────────────────────────────────────────
alerts_log         = []
current_frame      = None
frame_lock         = threading.Lock()
last_child_pos     = None
inactivity_start   = None
last_alert_time    = {}
face_results       = {}
face_results_lock  = threading.Lock()
face_check_running = set()


# ── Settings ───────────────────────────────────────────────────────
def load_settings():
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return DEFAULT_SETTINGS.copy()


def save_settings_file(data):
    with open(SETTINGS_FILE, "w") as f:
        json.dump(data, f)


# ── Alerts ─────────────────────────────────────────────────────────
def take_snapshot(label, frame):
    ts       = time.strftime("%Y%m%d_%H%M%S")
    filename = f"{label}_{ts}.jpg"
    cv2.imwrite(os.path.join(SNAPSHOT_DIR, filename), frame)
    return filename


def add_alert(alert_type, message, snapshot=None, cooldown=5):
    now = time.time()
    if alert_type in last_alert_time and \
       (now - last_alert_time[alert_type]) < cooldown:
        return
    last_alert_time[alert_type] = now
    item = {
        "type":     alert_type,
        "message":  message,
        "time":     time.strftime("%H:%M:%S"),
        "snapshot": snapshot,
        "resolved": None
    }
    alerts_log.append(item)
    socketio.emit("new_alert", item)


# ── Face helpers ───────────────────────────────────────────────────
def get_photos(folder):
    if not os.path.exists(folder):
        return []
    return [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith((".jpg", ".png", ".jpeg"))
    ]


def is_covered(face_bgr):
    """
    Checks if lower half of face is uniform (masked).
    Only triggers on clearly covered faces to avoid
    false positives from lighting variation.
    """
    if face_bgr is None or face_bgr.size == 0:
        return False
    h, w = face_bgr.shape[:2]
    if h < 40 or w < 40:
        return False
    gray        = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    upper_var   = float(gray[:h//2, :].var())
    lower_var   = float(gray[h//2:, :].var())
    edges       = cv2.Canny(gray, 50, 150)
    upper_edges = int(edges[:h//2, :].sum())
    lower_edges = int(edges[h//2:, :].sum())
    var_ratio   = lower_var   / (upper_var   + 1e-5)
    edge_ratio  = lower_edges / (upper_edges + 1e-5)
    # Stricter thresholds to reduce false positives
    return var_ratio < 0.3 and edge_ratio < 0.4


def verify_folder(face_rgb_array, folder):
    """
    Passes RGB numpy array directly to DeepFace — avoids
    file I/O errors and BGR/RGB colour space confusion.
    Uses ArcFace with standard cosine distance threshold 0.68.
    """
    from deepface import DeepFace
    photos = get_photos(folder)
    if not photos:
        return False
    for photo in photos:
        try:
            res = DeepFace.verify(
                img1_path=face_rgb_array,
                img2_path=photo,
                model_name="ArcFace",
                enforce_detection=False,
                silent=True,
                threshold=0.68
            )
            if res.get("verified", False):
                return True
        except Exception:
            continue
    return False


def classify_face_worker(person_idx, face_crop_bgr,
                         clean_frame_copy):
    """
    Classifies one face crop.
    face_crop_bgr must be from the CLEAN unannotated frame.
    clean_frame_copy used only for snapshots.
    """
    global face_check_running
    try:
        # Convert to RGB for DeepFace
        face_rgb = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2RGB)

        if is_covered(face_crop_bgr):
            result = "UNIDENTIFIED"
            snap   = take_snapshot("unidentified", clean_frame_copy)
            add_alert("UNIDENTIFIED",
                      "Covered face — flagged suspicious", snap)

        elif verify_folder(face_rgb, CHILD_DIR):
            result = "CHILD"

        elif verify_folder(face_rgb, TRUSTED_DIR):
            result = "AUTHORIZED"

        else:
            result = "UNAUTHORIZED"
            snap   = take_snapshot("unauthorized", clean_frame_copy)
            add_alert("UNAUTHORIZED",
                      "Unauthorized person detected", snap)

        with face_results_lock:
            face_results[person_idx] = result

    except Exception as e:
        print(f"[ERROR] classify_face_worker idx={person_idx}: {e}")
        with face_results_lock:
            face_results[person_idx] = "UNAUTHORIZED"
    finally:
        face_check_running.discard(person_idx)


# ── Camera loop ────────────────────────────────────────────────────
def camera_loop():
    global current_frame, last_child_pos, inactivity_start

    from ultralytics import YOLO
    from picamera2 import Picamera2

    settings  = load_settings()
    mode      = settings.get("mode", "child")
    last_mode = mode

    def start_cam(m):
        cam  = Picamera2()
        size = (1920, 1080) if m == "infant" else (640, 480)
        cfg  = cam.create_preview_configuration(
            main={"size": size, "format": "XBGR8888"}
        )
        cam.configure(cfg)
        cam.start()
        time.sleep(2)
        print(f"[INFO] Camera {m} {size}")
        return cam

    picam2 = start_cam(mode)
    yolo   = YOLO("yolov8n.pt")

    cascade_path = (
        "/usr/share/opencv4/haarcascades/"
        "haarcascade_frontalface_default.xml"
    )
    cascade = cv2.CascadeClassifier(cascade_path)
    if cascade.empty():
        print("[WARN] Cascade not loaded")
        cascade = None

    frame_idx = 0

    while True:
        settings  = load_settings()
        mode      = settings.get("mode", "child")
        expected  = settings.get("expected_count", 2)
        inact_sec = settings.get("inactivity_seconds", 30)
        zone_size = settings.get("zone_size", 250)

        # ZONE_RADIUS is the actual comparison distance.
        # This matches the drawn circle radius exactly.
        ZONE_RADIUS = zone_size // 2

        frame_idx += 1

        # Mode change → restart camera
        if mode != last_mode:
            picam2.stop()
            picam2.close()
            picam2    = start_cam(mode)
            last_mode = mode
            frame_idx = 0
            with face_results_lock:
                face_results.clear()
            last_child_pos   = None
            inactivity_start = None
            continue

        # ── Capture two versions of frame ──────────────────────────
        # clean_frame: unannotated — used for face crops and snapshots
        # display_frame: annotated — used for drawing and streaming
        raw         = picam2.capture_array()
        clean_frame = cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)
        display_frame = clean_frame.copy()

        # ── Low light check ────────────────────────────────────────
        if cv2.cvtColor(clean_frame,
                        cv2.COLOR_BGR2GRAY).mean() < 50:
            cv2.putText(display_frame, "LOW LIGHT",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 0, 255), 2)
            add_alert("LOW_LIGHT", "Room too dark")
            with frame_lock:
                current_frame = display_frame.copy()
            time.sleep(0.05)
            continue

        # ── YOLO on clean frame ────────────────────────────────────
        results      = yolo(clean_frame, verbose=False,
                            imgsz=320, conf=0.5, iou=0.45)
        person_boxes = []
        for r in results:
            for b in r.boxes:
                if int(b.cls[0]) == 0:
                    x1, y1, x2, y2 = map(int, b.xyxy[0])
                    person_boxes.append((x1, y1, x2, y2))

        total = len(person_boxes)

        # ── Face recognition — crops from CLEAN frame ──────────────
        if cascade is not None and frame_idx % 15 == 0:
            gray  = cv2.cvtColor(clean_frame, cv2.COLOR_BGR2GRAY)
            faces = cascade.detectMultiScale(gray, 1.1, 4)

            for f_idx, (fx, fy, fw, fh) in enumerate(faces):
                # Add 20% padding around face for better ArcFace accuracy
                pad_w = int(fw * 0.20)
                pad_h = int(fh * 0.20)
                fy1   = max(0, fy - pad_h)
                fy2   = min(clean_frame.shape[0], fy + fh + pad_h)
                fx1   = max(0, fx - pad_w)
                fx2   = min(clean_frame.shape[1], fx + fw + pad_w)

                # Crop from CLEAN (unannotated) frame
                crop = clean_frame[fy1:fy2, fx1:fx2].copy()

                # Match this face to nearest YOLO person box
                face_cx = fx + fw // 2
                face_cy = fy + fh // 2

                best_idx  = None
                best_dist = 999999
                for p_idx, (x1, y1, x2, y2) in \
                        enumerate(person_boxes):
                    box_cx = (x1 + x2) // 2
                    box_cy = (y1 + y2) // 2
                    d = ((face_cx - box_cx) ** 2 +
                         (face_cy - box_cy) ** 2) ** 0.5
                    if d < best_dist:
                        best_dist = d
                        best_idx  = p_idx

                if best_idx is not None and \
                   best_idx not in face_check_running:
                    face_check_running.add(best_idx)
                    executor.submit(
                        classify_face_worker,
                        best_idx,
                        crop,
                        clean_frame.copy()   # clean snapshot
                    )

        # ── Draw boxes on display_frame ────────────────────────────
        child_box_idx = None
        with face_results_lock:
            results_snap = dict(face_results)

        for p_idx, (x1, y1, x2, y2) in enumerate(person_boxes):
            result = results_snap.get(p_idx, "CHECKING")

            if result == "CHILD":
                color = (0, 255, 0)
                label = "Child"
                child_box_idx = p_idx
            elif result == "AUTHORIZED":
                color = (0, 200, 255)
                label = "Authorized"
            elif result == "UNAUTHORIZED":
                color = (0, 0, 255)
                label = "Unauthorized"
            elif result == "UNIDENTIFIED":
                color = (0, 165, 255)
                label = "Unidentified"
            else:
                color = (180, 180, 180)
                label = "Checking..."

            cv2.rectangle(display_frame,
                          (x1, y1), (x2, y2), color, 2)
            cv2.putText(display_frame, label,
                        (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, color, 2)

        # ── Child mode — zone-based inactivity ─────────────────────
        if mode == "child":

            # Find child position — prefer recognized child box
            # Fall back to smallest box if not yet recognized
            child_center = None
            if child_box_idx is not None:
                x1, y1, x2, y2 = person_boxes[child_box_idx]
            elif person_boxes:
                x1, y1, x2, y2 = min(
                    person_boxes,
                    key=lambda b: (b[2]-b[0]) * (b[3]-b[1])
                )
            else:
                x1 = y1 = x2 = y2 = None

            if x1 is not None:
                # Use upper body center — less affected by arm movements
                # Track point is at 1/3 from top of bounding box
                cx = (x1 + x2) // 2
                cy = y1 + (y2 - y1) // 3
                child_center = (cx, cy)

            if child_center:
                cx, cy = child_center

                if last_child_pos is None:
                    # First time — set anchor
                    last_child_pos   = (cx, cy)
                    inactivity_start = time.time()
                else:
                    dist = (
                        (cx - last_child_pos[0]) ** 2 +
                        (cy - last_child_pos[1]) ** 2
                    ) ** 0.5

                    if dist > ZONE_RADIUS:
                        # Genuine movement — reset anchor and timer
                        last_child_pos   = (cx, cy)
                        inactivity_start = time.time()
                    else:
                        # Still within zone — check timer
                        secs = int(time.time() - inactivity_start)
                        cv2.putText(display_frame,
                                    f"Still: {secs}s / {inact_sec}s",
                                    (10, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7, (255, 165, 0), 2)

                        if secs >= inact_sec:
                            snap = take_snapshot(
                                "inactivity", clean_frame)
                            add_alert("INACTIVITY",
                                      "Child has not changed position",
                                      snap)
                            inactivity_start = time.time()

                # Draw zone circle on display_frame
                # Radius matches ZONE_RADIUS exactly
                cv2.circle(display_frame,
                           last_child_pos,
                           ZONE_RADIUS,
                           (0, 200, 255), 1)

            # Count — exclude recognized child from total
            child_count  = 1 if child_box_idx is not None else 0
            other_count  = total - child_count
            exp_others   = max(0, expected - 1)
            if other_count > exp_others:
                snap = take_snapshot("count", clean_frame)
                add_alert("COUNT_ALERT",
                          f"{other_count} extra persons detected",
                          snap)

        # ── Infant mode ────────────────────────────────────────────
        elif mode == "infant":
            cv2.putText(display_frame, "INFANT MODE",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 255, 0), 2)
            if total > expected:
                snap = take_snapshot("count", clean_frame)
                add_alert("COUNT_ALERT",
                          f"{total} persons — expected {expected}",
                          snap)

        # ── HUD ───────────────────────────────────────────────────
        cv2.putText(display_frame,
                    f"Persons:{total} Expected:{expected} "
                    f"| {mode.upper()}",
                    (10, display_frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1)

        with frame_lock:
            current_frame = display_frame.copy()

        time.sleep(0.03)


# ── MJPEG stream ───────────────────────────────────────────────────
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


# ── Routes ─────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html",
                           settings=load_settings(),
                           alerts=alerts_log[-5:])


@app.route("/video_feed")
def video_feed():
    return Response(generate(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/register")
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
    folder   = CHILD_DIR if category == "child" else TRUSTED_DIR
    f.save(os.path.join(folder, name))
    return jsonify({"success": True,
                    "message": f"Saved in {category}: {name}"})


@app.route("/delete_face/<category>/<filename>",
           methods=["POST"])
def delete_face(category, filename):
    folder = CHILD_DIR if category == "child" else TRUSTED_DIR
    p = os.path.join(folder, filename)
    if os.path.exists(p):
        os.remove(p)
    return jsonify({"success": True})


@app.route("/alerts")
def alerts_page():
    return render_template(
        "alerts.html",
        alerts=list(reversed(alerts_log))
    )


@app.route("/resolve_alert", methods=["POST"])
def resolve_alert():
    d   = request.get_json()
    idx = d.get("index", -1)
    st  = d.get("status", "safe")
    rev = list(reversed(alerts_log))
    if 0 <= idx < len(rev):
        rev[idx]["resolved"] = st
    return jsonify({"success": True})


@app.route("/clear_alerts", methods=["POST"])
def clear_alerts():
    alerts_log.clear()
    return jsonify({"success": True})


@app.route("/settings")
def settings_page():
    return render_template("settings.html",
                           settings=load_settings())


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


if __name__ == "__main__":
    t = threading.Thread(target=camera_loop, daemon=True)
    t.start()
    print("Trivigi running → http://localhost:5000")
    socketio.run(app, host="0.0.0.0",
                 port=5000, debug=False)
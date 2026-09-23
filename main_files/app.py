import cv2
import os
import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from flask import (Flask, render_template, Response,
                   request, jsonify, send_from_directory)
from flask_socketio import SocketIO

import gpio_devices
import servo_control

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
ALERTS_LOG_MAX     = 500
current_frame      = None
current_frame_jpeg = None
frame_lock         = threading.Lock()
last_child_pos     = None
inactivity_start   = None
child_absent_start      = None   # time.time() when the child was last seen present; None while present
child_not_found_alerted = False  # fires CHILD_NOT_FOUND once per absence, not every frame
last_alert_time    = {}
# Keyed by YOLO/ByteTrack persistent track_id — NOT by a box's position
# in the current frame's list. Positional indices are unstable frame to
# frame (a new person entering can reshuffle the list), which is what
# caused the "child" label to jump onto a stranger. A track_id follows
# the same physical person across frames instead.
face_results        = {}
face_first_seen     = {}   # track_id -> time.time() first submitted for classification
face_results_lock   = threading.Lock()
face_check_running  = set()
FACE_RESULTS_MAX     = 2000   # safety cap for very long-running sessions
FACE_CHECK_TIMEOUT_S = 10     # grace window before an inconclusive read gives up
CHILD_NOT_FOUND_TIMEOUT_S = 10  # child mode: how long child can be absent before alerting


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


def add_alert(alert_type, message, frame=None, cooldown=5,
              snapshot_label=None):
    """
    frame is the raw frame to snapshot — NOT a pre-taken snapshot.
    The snapshot is only captured if this call actually passes the
    cooldown check, so a suppressed repeat alert never touches disk.
    """
    now = time.time()
    if alert_type in last_alert_time and \
       (now - last_alert_time[alert_type]) < cooldown:
        return
    last_alert_time[alert_type] = now

    snapshot = None
    if frame is not None:
        snapshot = take_snapshot(
            snapshot_label or alert_type.lower(), frame)

    item = {
        "type":     alert_type,
        "message":  message,
        "time":     time.strftime("%H:%M:%S"),
        "snapshot": snapshot,
        "resolved": None
    }
    alerts_log.append(item)
    if len(alerts_log) > ALERTS_LOG_MAX:
        del alerts_log[:-ALERTS_LOG_MAX]
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

    Returns True if matched, False if it was actually compared
    against at least one reference photo and matched none, or None
    if no comparison could be completed at all — no reference photos
    registered, or every DeepFace attempt errored out. None must NOT
    be treated as a confirmed non-match by the caller.
    """
    from deepface import DeepFace
    photos = get_photos(folder)
    if not photos:
        return None
    attempted = False
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
            attempted = True
            if res.get("verified", False):
                return True
        except Exception:
            continue
    return False if attempted else None


def classify_face_worker(track_id, face_crop_bgr,
                         clean_frame_copy, first_seen):
    """
    Classifies one face crop.
    face_crop_bgr must be from the CLEAN unannotated frame.
    clean_frame_copy used only for snapshots.
    track_id is the YOLO/ByteTrack persistent id for this person —
    NOT their position in any single frame's box list — so the result
    stays attached to the same physical person across frames.

    first_seen is when track_id first entered the classification
    pipeline. A single bad-angle or motion-blurred frame shouldn't
    brand someone UNIDENTIFIED — an inconclusive read (no reference
    photos to compare against, or DeepFace erroring) leaves the
    person at "CHECKING" and gets retried on the next cycle, only
    finalising to UNIDENTIFIED once FACE_CHECK_TIMEOUT_S has passed
    with no clear read. A genuine CHILD/AUTHORIZED/UNAUTHORIZED match
    is written immediately — the grace window only applies to "can't
    tell yet".
    """
    global face_check_running
    timed_out = (time.time() - first_seen) >= FACE_CHECK_TIMEOUT_S
    try:
        # Convert to RGB for DeepFace
        face_rgb = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2RGB)

        result = None   # None = still inconclusive, try again next cycle

        if is_covered(face_crop_bgr):
            result = "UNIDENTIFIED"
            add_alert("UNIDENTIFIED",
                      "Covered face — flagged suspicious",
                      clean_frame_copy)

        else:
            child_match = verify_folder(face_rgb, CHILD_DIR)
            if child_match is True:
                result = "CHILD"
            else:
                trusted_match = verify_folder(face_rgb, TRUSTED_DIR)
                if trusted_match is True:
                    result = "AUTHORIZED"
                elif child_match is False and trusted_match is False:
                    # Actually compared against both reference sets,
                    # matched neither — a clear result, no need to
                    # wait out the grace window.
                    result = "UNAUTHORIZED"
                    add_alert("UNAUTHORIZED",
                              "Unauthorized person detected",
                              clean_frame_copy)
                elif timed_out:
                    # Still not clear after the full grace window —
                    # no reference photos to compare against, or every
                    # DeepFace attempt errored the whole time.
                    result = "UNIDENTIFIED"
                    add_alert(
                        "UNIDENTIFIED",
                        f"Could not verify identity within "
                        f"{FACE_CHECK_TIMEOUT_S}s — flagged for review",
                        clean_frame_copy)
                # else: leave result as None, still within the grace
                # window — retried automatically on the next cycle.

        if result is not None:
            with face_results_lock:
                face_results[track_id] = result
                if len(face_results) > FACE_RESULTS_MAX:
                    face_results.clear()
                    face_first_seen.clear()

    except Exception as e:
        print(f"[ERROR] classify_face_worker track_id={track_id}: {e}")
        if timed_out:
            with face_results_lock:
                face_results[track_id] = "UNIDENTIFIED"
            add_alert("UNIDENTIFIED",
                      "Face classification failed — flagged for review",
                      clean_frame_copy)
    finally:
        face_check_running.discard(track_id)


# ── Camera loop ────────────────────────────────────────────────────
def camera_loop():
    global current_frame, current_frame_jpeg, last_child_pos, inactivity_start
    global child_absent_start, child_not_found_alerted

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
    last_frame_time = time.time()
    fps_smoothed = 0.0

    # NOTE: YOLO used to be throttled to every other frame to save CPU,
    # but that broke ByteTrack identity continuity — skipping frames
    # doubles the apparent motion between track() calls, so a moving
    # child would frequently get reassigned a brand new track_id. A
    # fresh id has no classification yet, so the servo would see
    # child_box=None and start "scanning for lost child" even though
    # the child never left frame. Running track() every frame trades
    # some CPU for correct, continuous tracking.
    YOLO_EVERY_N_FRAMES = 1
    person_boxes     = []
    person_track_ids = []   # parallel list — track_ids[i] identifies person_boxes[i]
    person_first_seen = {}  # track_id -> time.time() first seen as a person,
                             # regardless of whether a face was ever found for
                             # them — covers a face the cascade can't detect
                             # at all (e.g. covered by a mask/shawl), which
                             # would otherwise never reach classify_face_worker
                             # and sit at "Checking..." forever.

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

        now_ts = time.time()
        frame_dt = now_ts - last_frame_time
        last_frame_time = now_ts
        if frame_dt > 0:
            instant_fps  = 1.0 / frame_dt
            fps_smoothed = (instant_fps if fps_smoothed == 0
                            else fps_smoothed * 0.9 + instant_fps * 0.1)

        # Mode change → restart camera
        if mode != last_mode:
            picam2.stop()
            picam2.close()
            picam2    = start_cam(mode)
            last_mode = mode
            frame_idx        = 0
            person_boxes      = []
            person_track_ids  = []
            person_first_seen = {}
            with face_results_lock:
                face_results.clear()
                face_first_seen.clear()
            face_check_running.clear()
            last_child_pos   = None
            inactivity_start = None
            child_absent_start      = None
            child_not_found_alerted = False
            continue

        # ── Capture two versions of frame ──────────────────────────
        # clean_frame: unannotated — used for face crops and snapshots
        # display_frame: annotated — used for drawing and streaming
        raw         = picam2.capture_array()
        clean_frame = cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)
        display_frame = clean_frame.copy()

        cv2.putText(display_frame, f"FPS: {fps_smoothed:.1f}",
                    (display_frame.shape[1] - 120, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 255), 2)

        # ── Low light check ────────────────────────────────────────
        if cv2.cvtColor(clean_frame,
                        cv2.COLOR_BGR2GRAY).mean() < 50:
            cv2.putText(display_frame, "LOW LIGHT",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 0, 255), 2)
            add_alert("LOW_LIGHT", "Room too dark")
            ok, jpeg_buf = cv2.imencode(
                ".jpg", display_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            with frame_lock:
                current_frame = display_frame.copy()
                if ok:
                    current_frame_jpeg = jpeg_buf.tobytes()
            time.sleep(0.05)
            continue

        # ── YOLO + ByteTrack on clean frame ─────────────────────────
        # Throttled to every Nth frame — see YOLO_EVERY_N_FRAMES above.
        # .track(persist=True) assigns a stable track_id to each person
        # that follows them across frames, instead of a raw per-frame
        # detection list whose ORDER can reshuffle the instant someone
        # else enters frame. Classification results are keyed by that
        # track_id (see classify_face_worker / face_results below), so
        # a new person appearing can no longer make the "CHILD" label
        # jump onto the wrong box.
        if frame_idx % YOLO_EVERY_N_FRAMES == 0:
            results = yolo.track(clean_frame, verbose=False,
                                 imgsz=320, conf=0.5, iou=0.45,
                                 persist=True, tracker="bytetrack.yaml")
            person_boxes     = []
            person_track_ids = []
            for r in results:
                ids = r.boxes.id
                for i, b in enumerate(r.boxes):
                    if int(b.cls[0]) != 0:
                        continue
                    x1, y1, x2, y2 = map(int, b.xyxy[0])
                    tid = int(ids[i]) if ids is not None else None
                    person_boxes.append((x1, y1, x2, y2))
                    person_track_ids.append(tid)

            now_seen = time.time()
            for tid in person_track_ids:
                if tid is not None:
                    person_first_seen.setdefault(tid, now_seen)
            if len(person_first_seen) > FACE_RESULTS_MAX:
                person_first_seen.clear()

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

                # Match this face to nearest YOLO person box, then use
                # THAT box's persistent track_id — not its position in
                # this frame's list — as the classification key.
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

                best_track_id = (person_track_ids[best_idx]
                                 if best_idx is not None else None)

                if best_track_id is not None and \
                   best_track_id not in face_check_running:
                    with face_results_lock:
                        first_seen = face_first_seen.setdefault(
                            best_track_id, time.time())
                    face_check_running.add(best_track_id)
                    executor.submit(
                        classify_face_worker,
                        best_track_id,
                        crop,
                        clean_frame.copy(),  # clean snapshot
                        first_seen
                    )

        # ── No face ever found for this person ─────────────────────
        # The Haar cascade needs a visible mouth/nose/eyes region — a
        # mask, shawl or hand over the lower face routinely means NO
        # face is ever detected, so classify_face_worker never runs
        # and is_covered() never gets a crop to inspect. Without this,
        # that person sits at "Checking..." forever. Same grace window
        # as an inconclusive classification (FACE_CHECK_TIMEOUT_S).
        now = time.time()
        for tid in person_track_ids:
            if tid is None or tid in face_check_running:
                continue
            with face_results_lock:
                already_resolved = tid in face_results
            if already_resolved:
                continue
            started = person_first_seen.get(tid)
            if started is not None and (now - started) >= FACE_CHECK_TIMEOUT_S:
                with face_results_lock:
                    face_results[tid] = "UNIDENTIFIED"
                add_alert(
                    "UNIDENTIFIED",
                    f"No face detected for {FACE_CHECK_TIMEOUT_S}s — "
                    "possibly covered — flagged for review",
                    clean_frame)

        # ── Draw boxes on display_frame ────────────────────────────
        child_box_idx = None
        with face_results_lock:
            results_snap = dict(face_results)

        for p_idx, (x1, y1, x2, y2) in enumerate(person_boxes):
            tid    = person_track_ids[p_idx]
            result = (results_snap.get(tid, "CHECKING")
                      if tid is not None else "CHECKING")

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

            # Inactivity tracking only applies to the CONFIRMED child
            # (child_box_idx) — no more falling back to "smallest box
            # in frame". Tracking a stranger as a stand-in for the
            # child was causing bogus inactivity alerts once the real
            # child left frame while someone else remained.
            if child_box_idx is not None:
                x1, y1, x2, y2 = person_boxes[child_box_idx]
                # Use upper body center — less affected by arm movements
                # Track point is at 1/3 from top of bounding box
                cx = (x1 + x2) // 2
                cy = y1 + (y2 - y1) // 3

                # Child is present — clear the "not found" timer.
                child_absent_start      = None
                child_not_found_alerted = False

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
                            add_alert("INACTIVITY",
                                      "Child has not changed position",
                                      clean_frame)
                            inactivity_start = time.time()

                # Draw zone circle on display_frame
                # Radius matches ZONE_RADIUS exactly
                cv2.circle(display_frame,
                           last_child_pos,
                           ZONE_RADIUS,
                           (0, 200, 255), 1)

            else:
                # Child not recognized as present this frame — pause
                # inactivity tracking entirely (don't measure a
                # stranger's stillness, and don't let a stale timer
                # fire the instant the real child reappears), and time
                # how long the child has been missing from view.
                last_child_pos   = None
                inactivity_start = None

                if child_absent_start is None:
                    child_absent_start = time.time()
                else:
                    absent_secs = time.time() - child_absent_start
                    cv2.putText(display_frame,
                                f"Child not found: {int(absent_secs)}s",
                                (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (0, 0, 255), 2)
                    if absent_secs >= CHILD_NOT_FOUND_TIMEOUT_S and \
                       not child_not_found_alerted:
                        child_not_found_alerted = True
                        add_alert("CHILD_NOT_FOUND",
                                  "Child not found",
                                  clean_frame)

            # ── Servo tracking ──────────────────────────────────────
            # Tracks the child's FACE, not just their body box centre,
            # so framing stays on the face as they move. Face detection
            # here runs on a small crop of just the child's body box —
            # cheap enough to do every frame (unlike the full-frame
            # cascade pass above, which is throttled). Falls back to
            # the body box when no face is found this frame (e.g. the
            # child has turned away), so tracking doesn't drop out.
            child_box = (person_boxes[child_box_idx]
                         if child_box_idx is not None else None)

            child_track_box = child_box
            if child_box is not None and cascade is not None:
                bx1, by1, bx2, by2 = child_box
                bx1c = max(0, bx1)
                by1c = max(0, by1)
                bx2c = min(clean_frame.shape[1], bx2)
                by2c = min(clean_frame.shape[0], by2)
                if bx2c > bx1c and by2c > by1c:
                    crop_gray = cv2.cvtColor(
                        clean_frame[by1c:by2c, bx1c:bx2c],
                        cv2.COLOR_BGR2GRAY)
                    faces_in_crop = cascade.detectMultiScale(
                        crop_gray, 1.1, 4)
                    if len(faces_in_crop) > 0:
                        fx, fy, fw, fh = max(
                            faces_in_crop, key=lambda f: f[2] * f[3])
                        child_track_box = (
                            bx1c + fx, by1c + fy,
                            bx1c + fx + fw, by1c + fy + fh)

            servo_control.update(
                child_track_box,
                clean_frame.shape[1],
                clean_frame.shape[0],
                total,
                lambda: add_alert(
                    "CHILD_NOT_VISIBLE",
                    "Child not visible — please check")
            )

            # Count — exclude recognized child from total
            child_count  = 1 if child_box_idx is not None else 0
            other_count  = total - child_count
            exp_others   = max(0, expected - 1)
            if other_count > exp_others:
                add_alert("COUNT_ALERT",
                          f"{other_count} extra persons detected",
                          clean_frame, snapshot_label="count")

        # ── Infant mode ────────────────────────────────────────────
        elif mode == "infant":
            cv2.putText(display_frame, "INFANT MODE",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 255, 0), 2)
            if total > expected:
                add_alert("COUNT_ALERT",
                          f"{total} persons — expected {expected}",
                          clean_frame, snapshot_label="count")

        # ── HUD ───────────────────────────────────────────────────
        cv2.putText(display_frame,
                    f"Persons:{total} Expected:{expected} "
                    f"| {mode.upper()}",
                    (10, display_frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1)

        # Encode once here — generate() just serves these cached bytes,
        # so N connected viewers don't each re-encode the same frame.
        ok, jpeg_buf = cv2.imencode(
            ".jpg", display_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        with frame_lock:
            current_frame = display_frame.copy()
            if ok:
                current_frame_jpeg = jpeg_buf.tobytes()

        time.sleep(0.03)


# ── MJPEG stream ───────────────────────────────────────────────────
def generate():
    while True:
        with frame_lock:
            data = current_frame_jpeg
        if data is None:
            time.sleep(0.05)
            continue
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

    # ── SOS button + ultrasonic proximity ────────────────────────
    def get_current_frame_copy():
        with frame_lock:
            return (current_frame.copy()
                    if current_frame is not None else None)

    def on_sos_press():
        add_alert("SOS",
                  "SOS button pressed — immediate attention required",
                  get_current_frame_copy(), cooldown=0)

    def on_proximity(distance_m):
        add_alert("PROXIMITY",
                  f"Person within {distance_m:.2f}m of camera",
                  get_current_frame_copy())

    gpio_devices.start_sos_button(on_sos_press)
    gpio_devices.start_ultrasonic_monitor(on_proximity)

    # ── Ngrok remote access ───────────────────────────────────────
    # Needs an authtoken configured once via:
    #   ngrok config add-authtoken <token>
    try:
        from pyngrok import ngrok
        public_url = ngrok.connect(5000, "http")
        print(f"[INFO] Ngrok tunnel live at {public_url}")
    except Exception as e:
        print(f"[WARN] Ngrok tunnel not started: {e}")

    t = threading.Thread(target=camera_loop, daemon=True)
    t.start()
    print("Trivigi running → http://localhost:5000")
    socketio.run(app, host="0.0.0.0",
                 port=5000, debug=False,
                 allow_unsafe_werkzeug=True)
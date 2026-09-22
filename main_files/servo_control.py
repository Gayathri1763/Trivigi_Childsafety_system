import time

# ── Hardware init ──────────────────────────────────────────────────
# Wrapped in try/except — lets the rest of the app run normally on a
# Pi where the pan-tilt board isn't connected yet or I2C isn't
# enabled (raspi-config → Interface Options → I2C).
try:
    from adafruit_servokit import ServoKit
    kit = ServoKit(channels=16)
    _hardware_ready = True
except Exception as e:
    print(f"[WARN] Pan-tilt servo hardware not detected: {e}")
    kit = None
    _hardware_ready = False

PAN_CHANNEL  = 0
TILT_CHANNEL = 1

EDGE_MARGIN_PX             = 80    # trigger zone from the frame edge
SAFE_MARGIN_PX             = 160   # child must clear back past this before the opposite edge can trigger again
STEP_DEGREES               = 1     # small per-frame step for smooth recentring
MOVE_COOLDOWN_SECONDS      = 0.04  # floor on how often a pan step can fire
SCAN_STEP_DEGREES          = 1
SCAN_COOLDOWN_SECONDS      = 0.03
SCAN_START_DELAY_SECONDS   = 0.5
NOT_FOUND_TIMEOUT_SECONDS  = 30

pan_angle  = 90
tilt_angle = 90

_last_seen_time        = time.time()
_scan_direction         = 1
_not_found_alert_fired  = False
_last_pan_move_time     = 0.0
_last_scan_move_time    = 0.0
_last_edge_direction    = 0   # -1 = last corrected toward the left edge, +1 = right, 0 = centred/unset

if _hardware_ready:
    kit.servo[PAN_CHANNEL].angle  = pan_angle
    kit.servo[TILT_CHANNEL].angle = tilt_angle


def _move_pan(delta):
    global pan_angle
    pan_angle = max(0, min(180, pan_angle + delta))
    if _hardware_ready:
        kit.servo[PAN_CHANNEL].angle = pan_angle


def update(child_box, frame_width, on_child_lost):
    """
    Called once per camera_loop iteration in child mode.
    child_box is (x1, y1, x2, y2) of the recognised child, or None
    if the child is not visible this frame.

    The camera does NOT rotate continuously — only when the child
    nears a frame edge (recentre) or has left the frame (slow scan).
    Recentring moves in small (STEP_DEGREES) steps, rate-limited by
    MOVE_COOLDOWN_SECONDS, for smooth motion. A hysteresis band
    (SAFE_MARGIN_PX) blocks the opposite edge from re-triggering
    until the child is well clear of both margins, so a box merely
    jittering near the trigger line doesn't cause left/right hunting.
    on_child_lost() fires once after NOT_FOUND_TIMEOUT_SECONDS of the
    child staying missing.
    """
    global _last_seen_time, _scan_direction, _not_found_alert_fired
    global _last_pan_move_time, _last_scan_move_time, _last_edge_direction

    if not _hardware_ready:
        return

    now = time.time()

    if child_box is not None:
        _last_seen_time        = now
        _not_found_alert_fired = False

        x1, _, x2, _ = child_box

        near_left  = x1 < EDGE_MARGIN_PX
        near_right = x2 > frame_width - EDGE_MARGIN_PX
        clear_zone = (x1 > SAFE_MARGIN_PX) and                      (x2 < frame_width - SAFE_MARGIN_PX)

        if clear_zone:
            _last_edge_direction = 0

        can_move = (now - _last_pan_move_time) >= MOVE_COOLDOWN_SECONDS

        if near_left and _last_edge_direction != 1 and can_move:
            _move_pan(-STEP_DEGREES)
            _last_edge_direction = -1
            _last_pan_move_time  = now
        elif near_right and _last_edge_direction != -1 and can_move:
            _move_pan(STEP_DEGREES)
            _last_edge_direction = 1
            _last_pan_move_time  = now
        return

    missing_for = now - _last_seen_time
    if missing_for < SCAN_START_DELAY_SECONDS:
        return

    # Slow scan to relocate the child — reverse direction at the
    # pan limits instead of a continuous sweep.
    if (now - _last_scan_move_time) >= SCAN_COOLDOWN_SECONDS:
        if pan_angle <= 0 or pan_angle >= 180:
            _scan_direction = -_scan_direction
        _move_pan(SCAN_STEP_DEGREES * _scan_direction)
        _last_scan_move_time = now

    if missing_for >= NOT_FOUND_TIMEOUT_SECONDS and        not _not_found_alert_fired:
        _not_found_alert_fired = True
        on_child_lost()

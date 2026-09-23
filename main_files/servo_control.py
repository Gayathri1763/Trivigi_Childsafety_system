import time

PAN_CHANNEL  = 0
TILT_CHANNEL = 1

# Centre-tracking thresholds — measured from the middle of the frame,
# not the frame edge. Trigger starts a correction once the child's
# box centre drifts this far from centre; release stops it once back
# within the tighter band, so a box hovering right at the trigger
# line doesn't cause back-and-forth hunting.
PAN_TRIGGER_PX             = 60
PAN_RELEASE_PX             = 20
TILT_TRIGGER_PX            = 40
TILT_RELEASE_PX            = 15

# Proportional step: small correction when nearly centred, up to
# MAX_STEP when badly off-centre, so tracking is quick to catch up on
# real movement without the fixed-step crawl feeling unresponsive.
PAN_GAIN                   = 0.04  # degrees per pixel of pan error
PAN_MIN_STEP                = 1
PAN_MAX_STEP                = 6

TILT_GAIN                  = 0.04  # degrees per pixel of tilt error
TILT_MIN_STEP                = 1
TILT_MAX_STEP                = 5

MOVE_COOLDOWN_SECONDS      = 0.04  # floor on how often a step can fire, per axis

SCAN_STEP_DEGREES          = 1
SCAN_COOLDOWN_SECONDS      = 0.03
SCAN_START_DELAY_SECONDS   = 0.5
NOT_FOUND_TIMEOUT_SECONDS  = 30

HW_RETRY_INTERVAL_SECONDS  = 5.0   # how often to re-probe the I2C board while it's down

pan_angle  = 90
tilt_angle = 90

_last_seen_time         = time.time()
_scan_direction          = 1
_scan_direction_locked   = False   # set once per loss event, from the child's last known side
_last_known_pan_error    = 0.0     # + = child was right of centre when last seen
_not_found_alert_fired   = False

_pan_tracking            = False
_last_pan_move_time      = 0.0

_tilt_tracking           = False
_last_tilt_move_time     = 0.0

_last_scan_move_time     = 0.0

kit                  = None
_hardware_ready      = False
_last_hw_retry_time  = 0.0


def _try_init_hardware():
    """
    (Re)attempts to bring up the PCA9685 board over I2C. Called once
    at import, and re-tried periodically from update() while hardware
    is down. The I2C connection on this rig has repeatedly dropped
    and reseated itself mid-session — previously the only way to pick
    that back up was restarting the whole app; now it self-heals.
    """
    global kit, _hardware_ready, _last_hw_retry_time
    _last_hw_retry_time = time.time()
    was_ready = _hardware_ready
    try:
        from adafruit_servokit import ServoKit
        new_kit = ServoKit(channels=16)
        new_kit.servo[PAN_CHANNEL].angle  = pan_angle
        new_kit.servo[TILT_CHANNEL].angle = tilt_angle
        kit = new_kit
        _hardware_ready = True
        if not was_ready:
            print("[INFO] Pan-tilt servo hardware (re)connected")
    except Exception as e:
        kit = None
        _hardware_ready = False
        if was_ready:
            print(f"[WARN] Pan-tilt servo hardware lost: {e}")


_try_init_hardware()
if not _hardware_ready:
    print("[WARN] Pan-tilt servo hardware not detected "
          "(will keep retrying every "
          f"{HW_RETRY_INTERVAL_SECONDS:.0f}s)")


def _move_pan(delta):
    global pan_angle
    pan_angle = max(0, min(180, pan_angle + delta))
    if _hardware_ready:
        try:
            kit.servo[PAN_CHANNEL].angle = pan_angle
        except OSError as e:
            print(f"[WARN] Pan servo write failed (I2C glitch?): {e}")


def _move_tilt(delta):
    global tilt_angle
    tilt_angle = max(0, min(180, tilt_angle + delta))
    if _hardware_ready:
        try:
            kit.servo[TILT_CHANNEL].angle = tilt_angle
        except OSError as e:
            print(f"[WARN] Tilt servo write failed (I2C glitch?): {e}")


def _pan_step(pan_error):
    magnitude = max(PAN_MIN_STEP, min(PAN_MAX_STEP, abs(pan_error) * PAN_GAIN))
    return magnitude if pan_error > 0 else -magnitude


def _tilt_step(tilt_error):
    magnitude = max(TILT_MIN_STEP, min(TILT_MAX_STEP, abs(tilt_error) * TILT_GAIN))
    return -magnitude if tilt_error > 0 else magnitude


def update(child_box, frame_width, frame_height, person_count, on_child_lost):
    """
    Called once per camera_loop iteration in child mode.
    child_box is (x1, y1, x2, y2) of the recognised child, or None
    if the child is not visible this frame.

    Keeps the child centred in frame on both axes: once the box
    centre drifts past PAN_TRIGGER_PX / TILT_TRIGGER_PX from the
    middle of the frame, the matching servo nudges back by a step
    proportional to how far off-centre it is (see _pan_step /
    _tilt_step) until the box centre is back within the tighter
    *_RELEASE_PX band. The gap between trigger and
    release is the hysteresis that stops rapid direction reversals
    right at the threshold. When the child leaves frame entirely AND
    nobody else is visible either, pan performs a slow side-to-side
    scan to relocate them — if someone else is still in frame (just
    not yet confirmed as the child), the camera holds position
    instead of sweeping past them. on_child_lost() fires once after
    NOT_FOUND_TIMEOUT_SECONDS missing, regardless of whether a scan
    happened.
    """
    global _last_seen_time, _scan_direction, _not_found_alert_fired
    global _scan_direction_locked, _last_known_pan_error
    global _pan_tracking, _last_pan_move_time
    global _tilt_tracking, _last_tilt_move_time
    global _last_scan_move_time

    now = time.time()

    if not _hardware_ready and        (now - _last_hw_retry_time) >= HW_RETRY_INTERVAL_SECONDS:
        _try_init_hardware()

    if not _hardware_ready:
        return

    if child_box is not None:
        _last_seen_time        = now
        _not_found_alert_fired = False

        x1, y1, x2, y2 = child_box
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        pan_error  = cx - frame_width  / 2   # + = child right of centre
        tilt_error = cy - frame_height / 2   # + = child below centre

        _last_known_pan_error  = pan_error
        _scan_direction_locked = False

        # ── Pan ──
        if not _pan_tracking:
            if abs(pan_error) > PAN_TRIGGER_PX:
                _pan_tracking = True
        elif abs(pan_error) <= PAN_RELEASE_PX:
            _pan_tracking = False

        if _pan_tracking and (now - _last_pan_move_time) >= MOVE_COOLDOWN_SECONDS:
            _move_pan(_pan_step(pan_error))
            _last_pan_move_time = now

        # ── Tilt ──
        if not _tilt_tracking:
            if abs(tilt_error) > TILT_TRIGGER_PX:
                _tilt_tracking = True
        elif abs(tilt_error) <= TILT_RELEASE_PX:
            _tilt_tracking = False

        if _tilt_tracking and (now - _last_tilt_move_time) >= MOVE_COOLDOWN_SECONDS:
            _move_tilt(_tilt_step(tilt_error))
            _last_tilt_move_time = now

        return

    missing_for = now - _last_seen_time
    if missing_for < SCAN_START_DELAY_SECONDS:
        return

    if person_count == 0:
        # Room is genuinely empty — actively sweep to search.
        # First frame of the scan: head toward whichever side the
        # child was last seen on, instead of an arbitrary direction.
        if not _scan_direction_locked:
            _scan_direction = 1 if _last_known_pan_error > 0 else -1
            _scan_direction_locked = True

        # Reverse direction at the pan limits instead of a continuous
        # one-way sweep.
        if (now - _last_scan_move_time) >= SCAN_COOLDOWN_SECONDS:
            if pan_angle <= 0 or pan_angle >= 180:
                _scan_direction = -_scan_direction
            _move_pan(SCAN_STEP_DEGREES * _scan_direction)
            _last_scan_move_time = now
    else:
        # Someone is still in frame, just not confirmed as the child
        # yet — hold position rather than sweep past/away from them.
        _scan_direction_locked = False

    if missing_for >= NOT_FOUND_TIMEOUT_SECONDS and        not _not_found_alert_fired:
        _not_found_alert_fired = True
        on_child_lost()

import time
import threading
import RPi.GPIO as GPIO

# ── Pin map ────────────────────────────────────────────────────────
# SOS button — momentary push button, internal pull-up.
# Works as a hardware interrupt even if all software detection fails.
SOS_BUTTON_PIN = 22

# HC-SR04 ultrasonic sensor.
# ECHO is stepped down from 5V to ~3.27V by the 1kΩ/2.2kΩ voltage
# divider before reaching GPIO 27 — see wiring doc section 4.
TRIG_PIN = 17
ECHO_PIN = 27

PROXIMITY_METERS      = 1.0
SOS_DEBOUNCE_SECONDS  = 2
SOUND_SPEED_M_S       = 343

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)


# ── SOS push button ───────────────────────────────────────────────
def start_sos_button(on_press):
    """
    Watches the SOS button on a background thread.
    on_press() fires once per press — debounced so a single press
    cannot queue the alert twice.
    """
    try:
        GPIO.setup(SOS_BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    except Exception as e:
        print(f"[WARN] SOS button not available: {e}")
        return None

    def watch():
        last_press = 0
        pressed    = False
        while True:
            state = GPIO.input(SOS_BUTTON_PIN)
            if state == GPIO.LOW and not pressed:
                pressed = True
                now = time.time()
                if now - last_press > SOS_DEBOUNCE_SECONDS:
                    last_press = now
                    on_press()
            elif state == GPIO.HIGH:
                pressed = False
            time.sleep(0.05)

    t = threading.Thread(target=watch, daemon=True)
    t.start()
    print(f"[INFO] SOS button armed on GPIO {SOS_BUTTON_PIN}")
    return t


# ── HC-SR04 ultrasonic ────────────────────────────────────────────
def _measure_distance_m():
    """
    Sends a 10us trigger pulse and times the echo return.
    Returns distance in metres, or None on timeout — treated as
    "nothing in range" rather than an error.
    """
    GPIO.output(TRIG_PIN, False)
    time.sleep(0.0002)
    GPIO.output(TRIG_PIN, True)
    time.sleep(0.00001)
    GPIO.output(TRIG_PIN, False)

    timeout      = time.time() + 0.04
    pulse_start  = time.time()
    while GPIO.input(ECHO_PIN) == 0:
        pulse_start = time.time()
        if pulse_start > timeout:
            return None

    timeout     = time.time() + 0.04
    pulse_end   = time.time()
    while GPIO.input(ECHO_PIN) == 1:
        pulse_end = time.time()
        if pulse_end > timeout:
            return None

    pulse_duration = pulse_end - pulse_start
    return (pulse_duration * SOUND_SPEED_M_S) / 2


def start_ultrasonic_monitor(on_proximity, poll_interval=0.3, cooldown=5):
    """
    Polls the HC-SR04 continuously on a background thread.
    Fires on_proximity(distance_m) whenever something is within
    PROXIMITY_METERS of the camera — independent of face recognition,
    fires regardless of who it is.
    """
    try:
        GPIO.setup(TRIG_PIN, GPIO.OUT)
        GPIO.setup(ECHO_PIN, GPIO.IN)
        GPIO.output(TRIG_PIN, False)
        time.sleep(0.5)
    except Exception as e:
        print(f"[WARN] Ultrasonic sensor not available: {e}")
        return None

    def watch():
        last_fire = 0
        while True:
            try:
                dist = _measure_distance_m()
                if dist is not None and dist < PROXIMITY_METERS:
                    now = time.time()
                    if now - last_fire > cooldown:
                        last_fire = now
                        on_proximity(dist)
            except Exception as e:
                print(f"[ERROR] ultrasonic read: {e}")
            time.sleep(poll_interval)

    t = threading.Thread(target=watch, daemon=True)
    t.start()
    print(f"[INFO] Ultrasonic monitor armed on GPIO {TRIG_PIN}/{ECHO_PIN}")
    return t

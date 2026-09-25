# Trivigi — Intelligent Child Safety Monitoring System

**MSc Robotics Thesis | Gayathri Satheesh L | M01087828 | Middlesex University Dubai | 2026**

---

## What is Trivigi?

An intelligent child safety monitoring system for UAE households where both parents work and a child is left with a domestic helper. Trivigi actively classifies every person in the room, tracks the child's position, and alerts parents in real time through a cloud-free web dashboard.

---

## Features

- Three-state person classification — Child, Authorised, Unauthorised, Unidentified
- Child positional inactivity detection with zone-based tracking
- Covered face detection — automatic suspicious flag when face is masked
- Context-aware person count alert
- Servo motor pan-tilt tracking — camera follows registered child
- HC-SR04 proximity alert — snapshot when person comes within 1 metre
- SOS push button — physical emergency alert
- Flask web dashboard — live feed, alerts, registration, settings
- Ngrok remote access — view dashboard from anywhere
- 24-hour auto-deletion of all snapshots
- Privacy by design — all processing local, no cloud storage

---

## Hardware

| Component | Model |
|---|---|
| Processor | Raspberry Pi 5 4GB |
| Cooler | Official RPi Active Cooler SC1148 |
| Camera | RPi NoIR Camera Module 3 Wide |
| Pan-Tilt | Arducam Upgraded V2.0 |
| Proximity Sensor | HC-SR04 Ultrasonic |
| SOS Button | PBS-11B 12mm Momentary |
| Storage | SanDisk 256GB MicroSD |
| Power Backup | NIVAPRO 10000mAh Pass-Through |
| Power Adapter | Official RPi 27W USB-C |

---

## Software Stack

| Library | Purpose |
|---|---|
| Picamera2 | Frame capture from NoIR camera |
| OpenCV | Image processing and face detection |
| YOLOv8 nano + ByteTrack | Person detection and persistent tracking |
| DeepFace ArcFace | Face recognition and classification |
| Flask + Flask-SocketIO | Web dashboard and real-time alerts |
| pyngrok | Cloud-free remote access tunnel |
| RPi.GPIO | SOS button and ultrasonic sensor |
| adafruit-servokit | Pan-tilt servo control via I2C |
| Raspberry Pi OS 64-bit | Operating system |

---

## How to Use the System

### Step 1 — Power On
Connect power adapter to Raspberry Pi. Wait 60 seconds for system to boot. The trivigi service starts automatically — no manual action needed.

### Step 2 — Open Dashboard

**On home WiFi:**
```
http://trivigi.local:5000
```

**From anywhere (Ngrok remote access):**
```
https://your-ngrok-url.ngrok-free.app
```
Get current Ngrok URL by SSHing into Pi and running:
```bash
curl http://localhost:4040/api/tunnels
```

### Step 3 — Register Faces
1. Open dashboard → click **Register** tab
2. Click **Register Child** → upload 3 to 5 clear photos of child from different angles
3. Click **Register Trusted Member** → upload photos of maid and family members
4. Click Register button after each upload

### Step 4 — Configure Settings
Open dashboard → click **Settings** tab and set:
- **Monitoring Mode** — Child mode or Infant mode
- **Expected Persons** — how many people should normally be in the room
- **Inactivity Threshold** — seconds before inactivity alert fires (default 1800)
- **Zone Size** — how far child must move to reset inactivity timer

### Step 5 — Monitor
Go back to **Dashboard** tab. Live camera feed shows with bounding boxes:

| Box Colour | Label | Meaning |
|---|---|---|
| Green | Child | Registered child — tracking active |
| Cyan | Authorised | Registered trusted member |
| Red | Unauthorised | Unknown person — alert sent |
| Orange | Unidentified | Face covered — alert sent |

### Step 6 — Manage Alerts
Open **Alerts** tab to view all alerts with snapshots. Click **Safe** or **Unsafe** to resolve each alert.

---

## Service Commands (via SSH)

```bash
sudo systemctl status trivigi     # check if running
sudo systemctl restart trivigi    # restart after changes
sudo systemctl stop trivigi       # stop service
journalctl -u trivigi -f          # view live logs
```

## Safe Shutdown

```bash
sudo shutdown now
```
Wait for green LED to stop before unplugging.
Now this is not required as the booting is automatic.

---

## Privacy

- All processing runs locally on Raspberry Pi
- No footage sent to any external server
- Snapshots auto-deleted after 24 hours
- Ngrok forwards connection only — stores nothing

---

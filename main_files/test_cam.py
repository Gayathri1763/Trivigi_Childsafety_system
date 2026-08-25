import cv2

indices = [0, 2, 3, 4, 5]

for idx in indices:
    # Try V4L2 backend with YUYV format
    cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'YUYV'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        for _ in range(5):
            cap.grab()
            
        ret, frame = cap.read()
        cap.release()
        
        if ret and frame is not None and frame.mean() > 5:
            print(f"SUCCESS: Working camera found at index {idx} using YUYV!")
            break

    # Fallback try without forcing V4L2/FOURCC
    cap = cv2.VideoCapture(idx)
    if cap.isOpened():
        for _ in range(5):
            cap.grab()
        ret, frame = cap.read()
        cap.release()
        
        if ret and frame is not None and frame.mean() > 5:
            print(f"SUCCESS: Working camera found at index {idx} using Default Backend!")
            break
        
    print(f"Index {idx}: Failed")

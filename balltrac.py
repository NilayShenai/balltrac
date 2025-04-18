import cv2
import numpy as np
import argparse
import imutils
import time
import mediapipe as mp

# --- Argument parsing for command-line options ---
ap = argparse.ArgumentParser()
ap.add_argument("-s", "--screen", action="store_true", help="Start in screen mode")
ap.add_argument("-c", "--calibrate", action="store_true", help="Enable HSV calibration sliders")
args = vars(ap.parse_args())

# --- Color range presets for tennis ball detection ---
HSV_RANGES = {
    "real": ((22, 150, 100), (33, 255, 255)),   # For real-world tennis balls
    "screen": ((18, 80, 80), (45, 255, 255))    # For balls on screens/videos
}
# HSV range for green (used in calibration box)
GREEN_LOWER = (40, 40, 40)
GREEN_UPPER = (80, 255, 255)

# --- Global state variables ---
auto_hsv = None           # Stores auto-calibrated HSV range
auto_calibrated = False   # Flag for whether auto calibration is active
show_box = True           # Whether to show the green calibration box
easter_egg_mode = False   # Easter egg (gesture detection) mode flag
calibration_message = ""  # Message to display after calibration

def nothing(x): pass  # Placeholder for trackbar callback

def create_trackbars():
    """Create HSV trackbars for manual calibration."""
    cv2.namedWindow("Trackbars")
    cv2.createTrackbar("H Min", "Trackbars", 18, 179, nothing)
    cv2.createTrackbar("H Max", "Trackbars", 45, 179, nothing)
    cv2.createTrackbar("S Min", "Trackbars", 80, 255, nothing)
    cv2.createTrackbar("S Max", "Trackbars", 255, 255, nothing)
    cv2.createTrackbar("V Min", "Trackbars", 80, 255, nothing)
    cv2.createTrackbar("V Max", "Trackbars", 255, 255, nothing)

def get_trackbar_values():
    """Read current HSV values from the sliders."""
    h_min = cv2.getTrackbarPos("H Min", "Trackbars")
    h_max = cv2.getTrackbarPos("H Max", "Trackbars")
    s_min = cv2.getTrackbarPos("S Min", "Trackbars")
    s_max = cv2.getTrackbarPos("S Max", "Trackbars")
    v_min = cv2.getTrackbarPos("V Min", "Trackbars")
    v_max = cv2.getTrackbarPos("V Max", "Trackbars")
    return (h_min, s_min, v_min), (h_max, s_max, v_max)

def enhance_contrast(hsv):
    """Enhance the contrast of the V (brightness) channel for screen mode."""
    h, s, v = cv2.split(hsv)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    v = clahe.apply(v)
    return cv2.merge([h, s, v])

def auto_calibrate(hsv, screen_mode, show_box):
    """
    Auto-calibrate HSV bounds using the color inside the green box.
    Only works if the calibration box is visible.
    """
    if not show_box:
        print("[INFO] Calibration box is hidden. Press 'b' to show it.")
        return None, "[INFO] Calibration box is hidden. Press 'b' to show it."
    h, w = hsv.shape[:2]
    cx, cy = w // 2, h // 2
    box = 50
    region = hsv[cy - box:cy + box, cx - box:cx + box]
    if screen_mode:
        # For screen mode, look for green pixels
        mask = cv2.inRange(region, GREEN_LOWER, GREEN_UPPER)
    else:
        # For real mode, look for yellow-green pixels (typical tennis ball)
        mask = cv2.inRange(region, (15, 50, 50), (45, 255, 255))
    filtered_pixels = region[mask > 0]
    if filtered_pixels.shape[0] < 50:
        # Not enough pixels found for calibration; fallback to default
        return (HSV_RANGES["screen"] if screen_mode else HSV_RANGES["real"]), "[WARN] Not enough pixels found for auto calibration. Using default."
    # Calculate 5th and 95th percentiles to set robust HSV bounds
    h_vals, s_vals, v_vals = filtered_pixels[:, 0], filtered_pixels[:, 1], filtered_pixels[:, 2]
    h_min, h_max = np.percentile(h_vals, [5, 95])
    s_min, s_max = np.percentile(s_vals, [5, 95])
    v_min, v_max = np.percentile(v_vals, [5, 95])
    lower = (int(h_min), int(s_min), int(v_min))
    upper = (int(h_max), int(s_max), int(v_max))
    return (lower, upper), "[AUTO] Calibrated HSV: Lower={} Upper={}".format(lower, upper)

def draw_ui(frame, fps, mode, auto_calibrated, tennisLower, tennisUpper, show_box, calibration_message, easter_egg_mode):
    """
    Draws all overlays and instructions on the frame.
    """
    if not easter_egg_mode:
        # Display FPS and current mode
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Mode: {'SCREEN' if mode else 'REAL'}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        # Show auto-calibrated HSV range if in use
        if auto_calibrated and tennisLower and tennisUpper:
            cv2.putText(frame, f"AUTO HSV: {tennisLower}-{tennisUpper}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        # Draw the green calibration box in the center
        if show_box:
            h, w = frame.shape[:2]
            cx, cy = w // 2, h // 2
            box = 50
            cv2.rectangle(frame, (cx - box, cy - box), (cx + box, cy + box), (0, 255, 0), 2)
            cv2.putText(frame, "Place ball in green box & press 'a' to calibrate", (cx - 150, cy - box - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        # Show any calibration message (e.g., warnings or confirmation)
        if calibration_message:
            cv2.putText(frame, calibration_message, (10, frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        # Display user controls in the top right
        controls = [
            "'q': Quit",
            "'s': Toggle Screen/Real Mode",
            "'a': Auto-Calibrate (when box visible)",
            "'b': Show/Hide Calibration Box",
            "'f': Easter Egg Mode!"
        ]
        for i, text in enumerate(controls):
            cv2.putText(frame, text, (frame.shape[1] - 350, 30 + 25*i),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    else:
        # Easter egg mode: show a fun message and instructions
        cv2.putText(frame, "EASTER EGG: FLIPPING OFF DETECTOR!", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        cv2.putText(frame, "Show your hand to the camera.", (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "Press 'f' again to return.", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

def process_mask(mask, screen_mode):
    """
    Clean up the binary mask to remove noise and fill gaps.
    """
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    if screen_mode:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    return mask

def detect_flipping_off(frame, mp_hands, hands, mp_drawing):
    """
    Detects if the user is showing only the middle finger (flipping off).
    Uses MediaPipe Hands for hand landmark detection.
    """
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    flipping = False
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            lm = hand_landmarks.landmark
            h, w, _ = frame.shape
            # Check if middle finger is extended and others are folded
            middle_tip_y = lm[12].y
            middle_pip_y = lm[10].y
            folded = True
            for idx, pip in zip([8, 16, 20], [6, 14, 18]):
                if lm[idx].y < lm[pip].y:
                    folded = False
            if (middle_tip_y < middle_pip_y) and folded:
                flipping = True
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                cv2.putText(frame, "FLIPPING OFF DETECTED!", (80, int(h/2)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 4)
    return frame, flipping

def main():
    # Use global state variables
    global auto_calibrated, auto_hsv, show_box, easter_egg_mode, calibration_message
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam")
        return

    # If started with -c, show HSV trackbars for manual calibration
    if args["calibrate"]:
        create_trackbars()

    fps = 0
    last_time = time.time()
    smoothing = 0.9

    # Initialize MediaPipe for hand tracking (used in Easter egg mode)
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )

    print(f"[INFO] Started in {'SCREEN' if args['screen'] else 'REAL-WORLD'} mode")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARNING] Failed to grab frame")
            break

        # Resize frame for consistent processing and display
        frame = imutils.resize(frame, width=800)
        frame_disp = frame.copy()

        if not easter_egg_mode:
            # --- Tennis Ball Tracking Mode ---
            blurred = cv2.GaussianBlur(frame, (11, 11), 0)
            hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
            if args["screen"]:
                hsv = enhance_contrast(hsv)

            # Select HSV bounds: auto, manual, or default
            if auto_calibrated and auto_hsv:
                tennisLower, tennisUpper = auto_hsv
            elif args["calibrate"]:
                tennisLower, tennisUpper = get_trackbar_values()
            else:
                tennisLower, tennisUpper = HSV_RANGES["screen"] if args["screen"] else HSV_RANGES["real"]

            # Create and clean up the mask
            mask = cv2.inRange(hsv, tennisLower, tennisUpper)
            mask = process_mask(mask, args["screen"])

            # Find contours (potential balls) in the mask
            cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)

            found_ball = False
            if cnts:
                # Ignore small contours (noise)
                cnts = [c for c in cnts if cv2.contourArea(c) > (80 if args["screen"] else 300)]
                if cnts:
                    c = max(cnts, key=cv2.contourArea)
                    ((x, y), radius) = cv2.minEnclosingCircle(c)
                    min_radius = 5 if args["screen"] else 15
                    if radius > min_radius:
                        found_ball = True
                        # Draw circle and coordinates on the detected ball
                        cv2.circle(frame_disp, (int(x), int(y)), int(radius), (203, 73, 255), 2)
                        cv2.putText(frame_disp, f"Ball: ({int(x)}, {int(y)})", (10, 120),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Calculate and smooth FPS for display
            current_time = time.time()
            instant_fps = 1.0 / (current_time - last_time)
            fps = (fps * smoothing) + (instant_fps * (1 - smoothing))
            last_time = current_time

            # Draw all overlays and UI
            draw_ui(frame_disp, fps, args["screen"], auto_calibrated, tennisLower, tennisUpper, show_box, calibration_message, easter_egg_mode)

            cv2.imshow("Tennis Ball Tracking", frame_disp)
            cv2.imshow("Mask", mask)
        else:
            # --- Easter Egg Mode: Flipping Off Detector ---
            frame_disp, flipping = detect_flipping_off(frame_disp, mp_hands, hands, mp_drawing)
            draw_ui(frame_disp, fps, args["screen"], auto_calibrated, None, None, show_box, calibration_message, easter_egg_mode)
            cv2.imshow("Tennis Ball Tracking", frame_disp)
            # Hide mask window in easter egg mode, but do not crash if not open
            try:
                cv2.destroyWindow("Mask")
            except cv2.error:
                pass

        key = cv2.waitKey(1) & 0xFF
        # Only clear the calibration message after displaying it for one frame
        if calibration_message:
            calibration_message = ""
        if key == ord("q"):
            break
        elif key == ord("s") and not easter_egg_mode:
            # Toggle between real and screen mode
            args["screen"] = not args["screen"]
            auto_calibrated = False
            auto_hsv = None
            print(f"[INFO] Switched to {'SCREEN' if args['screen'] else 'REAL-WORLD'} mode")
        elif key == ord("a") and not easter_egg_mode:
            # Auto-calibrate HSV using the green box
            result, msg = auto_calibrate(hsv, args["screen"], show_box)
            if result:
                auto_hsv = result
                auto_calibrated = True
            calibration_message = msg
            print(msg)
        elif key == ord("b") and not easter_egg_mode:
            # Show or hide the calibration box
            show_box = not show_box
            status = "shown" if show_box else "hidden"
            print(f"[INFO] Calibration box {status}.")
        elif key == ord("f"):
            # Toggle Easter Egg mode
            easter_egg_mode = not easter_egg_mode
            if easter_egg_mode:
                print("[EASTER EGG] Flipping off detector activated!")
            else:
                print("[EASTER EGG] Back to tennis ball tracking.")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

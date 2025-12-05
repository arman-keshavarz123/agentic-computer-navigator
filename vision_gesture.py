import cv2
import mediapipe as mp
import threading
import time
import queue
from collections import deque, Counter

class GestureStream(threading.Thread):
    def __init__(self):
        super().__init__()
        self.daemon = True
        self.running = True
        self.paused = False
        
        # Communication Queue
        self.gesture_queue = queue.Queue()
        
        # Configuration
        self.cam_id = 0
        
        # SMOOTHNESS TWEAK: Reduced history length (7 -> 5) for faster reaction time
        self.history_length = 5 
        self.gesture_history = deque(maxlen=self.history_length)
        
        # MediaPipe Setup
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            # SMOOTHNESS TWEAK: Lowered confidence slightly to prevent "flickering" detection
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5
        )

    def get_finger_state(self, lm_list):
        """
        Returns a list of 5 booleans [Thumb, Index, Middle, Ring, Pinky]
        True if finger is extended (up), False if curled.
        """
        fingers = []
        
        # Thumb logic (Right Hand optimized, works for mirrored Left)
        if lm_list[4][1] < lm_list[3][1]: 
            fingers.append(True)
        else:
            fingers.append(False)

        # 4 Fingers logic
        tip_ids = [8, 12, 16, 20]
        for id in tip_ids:
            if lm_list[id][2] < lm_list[id - 2][2]:
                fingers.append(True)
            else:
                fingers.append(False)
        
        return fingers

    def detect_gesture_name(self, fingers):
        """Maps finger states to string names."""
        if all(fingers):
            return "OPEN_PALM"
        
        if not any(fingers):
            return "CLOSED_FIST"
        
        if fingers[1] and not fingers[2] and not fingers[3] and not fingers[4]:
            return "POINTING_INDEX"
            
        if fingers[1] and fingers[2] and not fingers[3] and not fingers[4]:
            return "PEACE"
            
        if fingers[0] and fingers[4] and not fingers[1] and not fingers[2] and not fingers[3]:
            return "SHAKA" 
            
        return "UNKNOWN"

    def run(self):
        print("VISION SYSTEM ONLINE (Webcam Active)")
        cap = cv2.VideoCapture(self.cam_id)
        
        last_sent_gesture = None
        
        while self.running:
            success, img = cap.read()
            if not success:
                time.sleep(0.1)
                continue

            if self.paused:
                time.sleep(0.1)
                continue

            # 1. Image Preprocessing
            img = cv2.flip(img, 1) 
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # 2. MediaPipe Detection
            results = self.hands.process(img_rgb)
            
            detected_this_frame = "UNKNOWN"

            if results.multi_hand_landmarks:
                for hand_lms in results.multi_hand_landmarks:
                    # Optional: Calculation for finer details if needed later
                    h, w, c = img.shape
                    lm_list = []
                    for id, lm in enumerate(hand_lms.landmark):
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        lm_list.append([id, cx, cy])

                    if len(lm_list) != 0:
                        fingers = self.get_finger_state(lm_list)
                        detected_this_frame = self.detect_gesture_name(fingers)

            # 3. Stabilization Logic
            self.gesture_history.append(detected_this_frame)
            
            # Only process if history buffer is full
            if len(self.gesture_history) == self.history_length:
                # Get most common gesture in history
                most_common, count = Counter(self.gesture_history).most_common(1)[0]
                
                # Consistency check (e.g. 4 out of 5 frames must match)
                if count >= (self.history_length - 1) and most_common != "UNKNOWN":
                    
                    if most_common != last_sent_gesture:
                        self.gesture_queue.put(most_common)
                        last_sent_gesture = most_common
                else:
                    if most_common == "UNKNOWN":
                        last_sent_gesture = None

            time.sleep(0.01) 
                
        cap.release()
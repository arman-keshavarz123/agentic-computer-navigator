import os
import cv2
import time
import queue
import threading
import torch
import numpy as np
import mediapipe as mp
from collections import deque
from PIL import Image
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

# --- CONFIGURATION ---
# If you have specific SignGemma weights, put the path here.
# Otherwise, PaliGemma is the best drop-in replacement for VLM tasks.
MODEL_ID = "google/paligemma-3b-mix-224" 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class GestureStream(threading.Thread):
    def __init__(self):
        super().__init__()
        self.daemon = True
        self.running = True
        self.gesture_queue = queue.Queue()
        
        # 1. The "Wake Word" Detector (MediaPipe)
        # We use this ONLY to detect when to start recording. It's cheap and fast.
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # 2. The "Brain" (SignGemma / VLM)
        print(f"[Vision] Loading VLM Model ({MODEL_ID}) to {DEVICE}...")
        try:
            self.model = PaliGemmaForConditionalGeneration.from_pretrained(
                MODEL_ID, 
                torch_dtype=torch.bfloat16,
                device_map=DEVICE
            ).eval()
            self.processor = AutoProcessor.from_pretrained(MODEL_ID)
            print("[Vision] Model Loaded Successfully.")
        except Exception as e:
            print(f"[Vision] CRITICAL ERROR loading model: {e}")
            self.running = False

        # Recording State
        self.is_recording = False
        self.frame_buffer = deque(maxlen=30) # Buffer for context
        self.recording_buffer = []

    def detect_trigger(self, img):
        """
        Cheap check for 'Closed Fist' to trigger the heavy VLM.
        """
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)
        
        if results.multi_hand_landmarks:
            for hand_lms in results.multi_hand_landmarks:
                # Simple logic: Are all fingertips below knuckles?
                lm = hand_lms.landmark
                fingers_curled = 0
                
                # Check 4 fingers (Index to Pinky)
                if lm[8].y > lm[6].y: fingers_curled += 1
                if lm[12].y > lm[10].y: fingers_curled += 1
                if lm[16].y > lm[14].y: fingers_curled += 1
                if lm[20].y > lm[18].y: fingers_curled += 1
                
                # If Fist Detected
                if fingers_curled >= 3:
                    return True
        return False

    def process_video_sequence(self, frames):
        """
        Takes a list of CV2 frames, formats them for the VLM, and runs inference.
        """
        print("[Vision] Processing Sign Language Sequence...")
        
        # 1. Preprocess: Stack frames into a grid or select keyframes
        # VLM usually takes one image, so we create a 'contact sheet' or pick the middle frame
        # For true video models, you pass the stack. Here we use the Contact Sheet method for PaliGemma.
        
        # Take 4 evenly spaced frames from the recording
        indices = np.linspace(0, len(frames)-1, 4, dtype=int)
        selected_frames = [frames[i] for i in indices]
        
        # Convert to PIL
        pil_images = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in selected_frames]
        
        # 2. Prompting
        # We ask the model to translate the gesture to a command
        prompt = "describe the hand gesture action"
        
        # 3. Inference
        try:
            # We treat the sequence as inputs. 
            # Note: For simple PaliGemma, we might just feed the last frame or a grid.
            # Let's try the last clear frame for this specific implementation to keep it robust
            # unless you are using a specific Video-Llava checkpoint.
            input_image = pil_images[-1] 
            
            model_inputs = self.processor(text=prompt, images=input_image, return_tensors="pt").to(DEVICE)
            input_len = model_inputs["input_ids"].shape[-1]

            with torch.inference_mode():
                generation = self.model.generate(**model_inputs, max_new_tokens=20, do_sample=False)
                generation = generation[0][input_len:]
                decoded = self.processor.decode(generation, skip_special_tokens=True)
                
            print(f"[Vision] VLM Output: '{decoded}'")
            return decoded.strip()
            
        except Exception as e:
            print(f"[Vision] Inference Failed: {e}")
            return None

    def run(self):
        cap = cv2.VideoCapture(0)
        print("[Vision] Camera Active. Show CLOSED FIST to trigger SignGemma.")
        
        trigger_frames = 0
        RECORD_DURATION = 40 # Frames to record after trigger (~1.5s)
        
        while self.running:
            success, frame = cap.read()
            if not success: continue
            
            frame = cv2.flip(frame, 1)
            
            # STATE 1: IDLE (Waiting for Trigger)
            if not self.is_recording:
                if self.detect_trigger(frame):
                    trigger_frames += 1
                    if trigger_frames > 10: # Hold fist for ~0.3s
                        print("[Vision] 🔴 Trigger Detected! Recording Sign...")
                        self.is_recording = True
                        self.recording_buffer = [] # Reset buffer
                        trigger_frames = 0
                        # Visual Feedback
                        os.system("afplay /System/Library/Sounds/Hero.aiff") 
                else:
                    trigger_frames = 0
                    self.frame_buffer.append(frame)

            # STATE 2: RECORDING (Collecting Frames for GPU)
            else:
                self.recording_buffer.append(frame)
                
                # Visual Indicator (Simple overlay)
                cv2.circle(frame, (50, 50), 20, (0, 0, 255), -1) 
                
                if len(self.recording_buffer) >= RECORD_DURATION:
                    print("[Vision] 🛑 Recording Complete. Sending to GPU...")
                    self.is_recording = False
                    
                    # Run Inference in a separate thread to not freeze camera
                    # (Though here we block briefly, with a cluster it's fast)
                    cmd = self.process_video_sequence(self.recording_buffer)
                    
                    if cmd:
                        self.gesture_queue.put(cmd)
                        os.system("afplay /System/Library/Sounds/Tink.aiff")
            
            # Optional: Show feed for debug
            # cv2.imshow("SignGemma Feed", frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'): break
            
            time.sleep(0.01)
            
        cap.release()
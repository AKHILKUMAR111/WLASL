import cv2
import mediapipe as mp
import torch
import torch.nn as nn
import numpy as np
import math
import argparse
import os
import json
from collections import deque

# --- Define Model Architecture ---
class PositionalEncoding(nn.Module):
    """Adds positional information to the input sequence embeddings."""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1) # Shape: (max_len, 1, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """Args: x: Tensor, shape [seq_len, batch_size, embedding_dim]"""
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

# --- This is the CORRECTED PoseTransformer class for realtime_gloss.py ---
# It is now an exact copy of your training notebook's version.

class PoseTransformer(nn.Module):
    """Transformer model for pose sequence classification."""
    def __init__(self, input_dim, embed_dim, num_heads, num_layers, dim_feedforward, num_classes, max_seq_len, dropout=0.1):
        super(PoseTransformer, self).__init__()
        self.embed_dim = embed_dim
        
        # Input embedding layer (Linear layer to project input features)
        self.input_embedding = nn.Linear(input_dim, embed_dim)
        
        # Positional Encoding
        self.pos_encoder = PositionalEncoding(embed_dim, dropout, max_len=max_seq_len)
        
        # Standard PyTorch Transformer Encoder
        # --- FIX 1: Removed norm_first=True ---
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, 
                                                   dim_feedforward=dim_feedforward, dropout=dropout, 
                                                   batch_first=False) # Must match training (False)
        encoder_norm = nn.LayerNorm(embed_dim)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, norm=encoder_norm)
        
        # Classification head
        self.fc_out = nn.Linear(embed_dim, num_classes)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.input_embedding.weight.data.uniform_(-initrange, initrange)
        self.input_embedding.bias.data.zero_()
        self.fc_out.weight.data.uniform_(-initrange, initrange)
        self.fc_out.bias.data.zero_()

    def forward(self, src):
        """
        Args:
            src: Tensor, shape [batch_size, seq_len, input_dim]
        """
        # --- Reshape for PyTorch Transformer ---
        # Input shape: (Batch, Seq, Feature) -> (Seq, Batch, Feature)
        src = src.transpose(0, 1) 
        
        # --- Processing ---
        src = self.input_embedding(src) * math.sqrt(self.embed_dim) # Embed input
        src = self.pos_encoder(src) # Add positional encoding
        output = self.transformer_encoder(src) # Pass through Transformer Encoder
        
        # --- Classification ---
        # --- FIX 2: Use first token, not mean pooling ---
        output = output[0, :, :] # Take output of the first time step
        
        output = self.fc_out(output) # Final classification layer
        return output # Logits output

# --- <<< CRITICAL FIX: DEFINE CONSTANTS GLOBALLY >>> ---
POSE_LANDMARKS = 33
HAND_LANDMARKS = 21
NUM_POSE_VALUES = 4 # x,y,z,vis
NUM_HAND_VALUES = 4 # x,y,z,vis (dummy)
NUM_VALUES_PER_LANDMARK = 4 # General value for reshaping

# Total features: (33*4) + (21*4) + (21*4) = 132 + 84 + 84 = 300
INPUT_FEATURES = (POSE_LANDMARKS * NUM_POSE_VALUES) + (HAND_LANDMARKS * NUM_HAND_VALUES) * 2
# --- <<< END FIX >>> ---


# --- Helper Functions ---
def extract_landmarks_from_results(results):
    """Processes MediaPipe results and returns a 300-feature flattened array."""
    current_frame_features = []
    try:
        # 1. Pose Landmarks
        if results.pose_landmarks:
            for landmark in results.pose_landmarks.landmark:
                current_frame_features.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
        else:
            current_frame_features.extend([0.0] * POSE_LANDMARKS * NUM_POSE_VALUES)
        # 2. Left Hand
        if results.left_hand_landmarks:
            for landmark in results.left_hand_landmarks.landmark:
                current_frame_features.extend([landmark.x, landmark.y, landmark.z, 1.0]) 
        else:
            current_frame_features.extend([0.0] * HAND_LANDMARKS * NUM_HAND_VALUES)
        # 3. Right Hand
        if results.right_hand_landmarks:
            for landmark in results.right_hand_landmarks.landmark:
                current_frame_features.extend([landmark.x, landmark.y, landmark.z, 1.0])
        else:
            current_frame_features.extend([0.0] * HAND_LANDMARKS * NUM_HAND_VALUES)
            
        return np.array(current_frame_features, dtype=np.float32)
    except Exception as e:
        print(f"Error in extract_landmarks_from_results: {e}")
        return np.zeros(INPUT_FEATURES, dtype=np.float32)

def is_sign_visible(landmarks_flat: np.ndarray, threshold: float = 0.05) -> bool:
    """Checks if a significant number of *hand* landmarks are non-zero."""
    LH_START_IDX = POSE_LANDMARKS * NUM_POSE_VALUES # 132
    RH_END_IDX = INPUT_FEATURES # 300
    TOTAL_HAND_FEATURES = (HAND_LANDMARKS * NUM_HAND_VALUES) * 2 # 168

    if landmarks_flat is None or landmarks_flat.size != INPUT_FEATURES:
        return False
    hand_landmarks = landmarks_flat[LH_START_IDX:RH_END_IDX]
    visible_points = np.count_nonzero(hand_landmarks)
    if TOTAL_HAND_FEATURES == 0: return False
    return (visible_points / TOTAL_HAND_FEATURES) > threshold
    
def normalize_landmarks(landmarks):
    """Normalizes landmarks (shape [frames, 300]) by centering on the nose."""
    normalized_video = np.copy(landmarks)
    num_frames = landmarks.shape[0]
    
    for frame_idx in range(num_frames):
        frame_data = landmarks[frame_idx]
        # Now these constants are globally defined and accessible
        frame_reshaped = frame_data.reshape((POSE_LANDMARKS + HAND_LANDMARKS * 2, NUM_VALUES_PER_LANDMARK))
        pose_landmarks = frame_reshaped[:POSE_LANDMARKS]
        nose_x, nose_y = pose_landmarks[0, 0], pose_landmarks[0, 1]
        nose_visibility = pose_landmarks[0, 3]
        
        if nose_visibility > 0.1:
            frame_reshaped[:, 0] -= nose_x
            frame_reshaped[:, 1] -= nose_y
        else:
            frame_reshaped[:, :3] = 0.0
        
        normalized_video[frame_idx] = frame_reshaped.flatten()
    return normalized_video

def pad_truncate(landmarks, max_len):
    """Pads or truncates landmark sequence to a fixed length."""
    num_frames = landmarks.shape[0]
    num_features = landmarks.shape[1]
    padded_landmarks = np.zeros((max_len, num_features), dtype=np.float32)
    seq_len_to_use = min(num_frames, max_len)
    padded_landmarks[:seq_len_to_use] = landmarks[:seq_len_to_use]
    return padded_landmarks

def draw_text(image, text, conf):
    """Draws prediction text on the image."""
    text_to_show = f"{text} ({conf*100:.1f}%)" if conf > 0.01 else text
    cv2.putText(image, text_to_show, (10, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

# --- Main execution block ---
def main(args):
    
    MODEL_PATH = args.weights
    JSON_MAP_PATH = args.id2gloss
    NUM_CLASSES = args.classes
    CAMERA_INDEX = args.camera
    SHOW_SKELETON = args.show_skeleton
    CONF_THRESHOLD = args.conf_threshold
    
    # Constants from training
    MAX_SEQ_LENGTH = 100
    EMBED_DIM = 256
    NUM_HEADS = 8
    NUM_ENCODER_LAYERS = 3
    DIM_FEEDFORWARD = 512
    DROPOUT = 0.3
    
    # Load Label Mapping (Handles list or dict)
    print(f"Loading label mapping from {JSON_MAP_PATH}...")
    if not os.path.exists(JSON_MAP_PATH):
        print(f"Error: Label map file not found at {JSON_MAP_PATH}")
        return
    try:
        with open(JSON_MAP_PATH, 'r') as f:
            loaded_map = json.load(f)
        if isinstance(loaded_map, list):
            print("Loaded label map as a LIST. Converting to dictionary.")
            id_to_gloss = {i: gloss for i, gloss in enumerate(loaded_map)}
        elif isinstance(loaded_map, dict):
            print("Loaded label map as a DICTIONARY. Converting keys to integers.")
            id_to_gloss = {int(k): v for k, v in loaded_map.items()}
        else:
            raise TypeError("Label map file is not a valid list or dictionary.")
            
        loaded_num_classes = len(id_to_gloss)
        if loaded_num_classes != NUM_CLASSES:
            print(f"Warning: --classes arg ({NUM_CLASSES}) does not match loaded map ({loaded_num_classes}). Using value from map: {loaded_num_classes}")
            NUM_CLASSES = loaded_num_classes
        print(f"Label mapping loaded successfully with {NUM_CLASSES} classes.")
    except Exception as e:
        print(f"Error loading id_to_gloss from {JSON_MAP_PATH}: {e}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Model (with 300 INPUT_FEATURES)
    print("Loading trained model...")
    model = PoseTransformer(
        input_dim=INPUT_FEATURES, # 300
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_ENCODER_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        num_classes=NUM_CLASSES,
        max_seq_len=MAX_SEQ_LENGTH,
        dropout=DROPOUT
    ).to(device)

    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}")
        return
    
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device)) 
    model.eval()
    print("Model loaded successfully.")

    # MediaPipe Setup (Using Holistic)
    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=2,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    mp_drawing = mp.solutions.drawing_utils

    print(f"Starting webcam feed from camera index {CAMERA_INDEX}...")
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"Error: Could not open camera {CAMERA_INDEX}.")
        return

    # --- Variables for Sign Completion Logic ---
    buffer = []
    last_pred, last_conf = "", 0.0
    recording_state = "WAITING" 
    SIGN_END_PATIENCE = 15
    sign_end_counter = 0
    frame_index = 0
    
    print("Starting real-time recognition... (Press 'c' to clear, 'ESC' to quit)")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_index += 1
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27: # ESC
            print("ESC pressed, quitting.")
            break
        if key == ord('c'):
            print("Buffer cleared by user.")
            buffer.clear()
            last_pred, last_conf = "", 0.0
            recording_state = "WAITING"
            sign_end_counter = 0

        frame = cv2.flip(frame, 1)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = holistic.process(image_rgb)
        image_rgb.flags.writeable = True
        frame = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        landmark_flat = extract_landmarks_from_results(results) # Gets 300 features
        sign_visible = is_sign_visible(landmark_flat) # Checks hands
        
        if SHOW_SKELETON:
            # Draw Pose, Left Hand, Right Hand
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, landmark_drawing_spec=mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=2))
            mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        # --- State Machine Logic ---
        if recording_state == "WAITING":
            if sign_visible:
                recording_state = "RECORDING"
                buffer.clear()
                buffer.append(landmark_flat)
                sign_end_counter = 0
        
        elif recording_state == "RECORDING":
            buffer.append(landmark_flat) 
            
            if sign_visible:
                sign_end_counter = 0 
            else:
                sign_end_counter += 1
                if sign_end_counter >= SIGN_END_PATIENCE:
                    print(f"[Frame {frame_index}] Sign ended. Processing {len(buffer)} frames.")
                    
                    if len(buffer) >= args.min_frames:
                        seq = np.stack(buffer, axis=0).astype(np.float32)
                        seq = normalize_landmarks(seq) # Normalize (300 features)
                        seq = pad_truncate(seq, MAX_SEQ_LENGTH) # Pad/Truncate
                        tens = torch.from_numpy(seq).unsqueeze(0).to(device) # Shape [1, 100, 300]

                        with torch.no_grad():
                            logits = model(tens)
                            probs = torch.softmax(logits, dim=-1)[0]
                            conf, pred = torch.max(probs, dim=-1)
                            pred_idx = int(pred.item())
                            last_conf = float(conf.item())
                            
                            if last_conf < 0.1:
                                last_pred = ""
                            else:
                                last_pred = id_to_gloss.get(pred_idx, f"class_{pred_idx}")
                    else:
                        print(f"Sign too short ({len(buffer)} frames < {args.min_frames}). Skipping.")
                        last_pred, last_conf = "", 0.0
                        
                    buffer.clear()
                    recording_state = "WAITING"
                    sign_end_counter = 0
        
        # --- Visualization ---
        draw_text(frame, last_pred, last_conf)
        cv2.putText(frame, f"State: {recording_state}", (10, frame.shape[0] - 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Buffer: {len(buffer)}", (10, frame.shape[0] - 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1, cv2.LINE_AA)
        
        cv2.imshow("Sign-to-Gloss (Pose Transformer)", frame)

    # --- Cleanup ---
    cap.release()
    cv2.destroyAllWindows()
    holistic.close()
    print("Webcam feed stopped.")

# --- Argument Parser Setup ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Real-time Sign Language Recognition')
    parser.add_argument('--weights', type=str, required=True, 
                        help='Path to the trained model .pth file')
    parser.add_argument('--classes', type=int, required=True, 
                        help='Number of classes the model was trained on')
    parser.add_argument('--id2gloss', type=str, required=True, 
                        help='Path to the id_to_gloss .json file')
    parser.add_argument('--camera', type=int, default=0, 
                        help='Camera index (e.g., 0 for default webcam)')
    parser.add_argument('--show-skeleton', action='store_true', 
                        help='Set this flag to draw the pose skeleton')
    # Add your other args
    parser.add_argument('--min-frames', type=int, default=10, 
                        help='Minimum number of frames to process a sign')
    parser.add_argument('--stride', type=int, default=5, 
                        help='(Not used in this logic) Inference cadence')
    parser.add_argument('--conf-threshold', type=float, default=0.4, 
                        help='Confidence threshold to display prediction')

    args = parser.parse_args()
    
    # Run the main function
    main(args)
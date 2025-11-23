# Real-Time Sign Language Recognition (Cafe Subset)

This project implements a **Pose-Based Transformer** model for real-time recognition of American Sign Language (ASL). It utilizes a specific subset of the **WLASL** dataset relevant to a cafe setting (e.g., "coffee", "tea", "pay", "thank you").

Unlike the original WLASL repository (which uses computationally heavy 3D-CNNs like I3D), this implementation uses **MediaPipe Holistic** for lightweight feature extraction and a custom **PyTorch Transformer** for sequence modeling. This allows the system to run efficiently on standard CPUs for real-time inference.

## 📚 Dataset Citation

This project is built upon the WLASL dataset. If you use this code or data, please cite the original paper:

> **Word-level Deep Sign Language Recognition from Video: A New Large-scale Dataset and Methods Comparison**
> Dongxu Li, Cristian Rodriguez Opazo, Xin Yu, Hongdong Li.
> *The IEEE Winter Conference on Applications of Computer Vision (WACV)*, 2020.

## ⚙️ How It Works

The recognition pipeline consists of four main stages:

1.  **Feature Extraction (MediaPipe Holistic):**
    * The system captures video frames from the webcam.
    * **MediaPipe Holistic** extracts skeletal landmarks for the **Pose**, **Left Hand**, and **Right Hand**.
    * We utilize 33 Pose landmarks and 21 landmarks per hand (x, y, z, visibility), totaling **300 features per frame**. (Face landmarks are excluded to reduce noise).

2.  **Preprocessing & Normalization:**
    * Landmarks are **normalized** by centering them around the user's nose (landmark 0). This makes the model robust to the user's position in the frame.
    * Real-time data is buffered into a sliding window.

3.  **Sign Detection Logic:**
    * The system monitors hand visibility. When hands appear, it enters a **RECORDING** state.
    * When hands disappear for a set duration (patience), the system assumes the sign is finished.
    * The captured sequence is padded or truncated to a fixed length of **100 frames**.

4.  **Classification (Pose Transformer):**
    * The processed sequence is fed into a custom **PyTorch Transformer Encoder**.
    * The model uses **Positional Encoding** to understand the order of movements.
    * It outputs a probability distribution over the target classes (14 cafe-related signs), and the gloss with the highest confidence is displayed.

## 📦 Installation

Ensure you have Python 3.8+ installed. You will also need **FFmpeg** installed on your system.

Install the required Python dependencies:

```bash
pip install torch torchvision torchaudio numpy opencv-python mediapipe scikit-learn matplotlib
🚀 How to Run
1. Real-Time Recognition Demo
To run the live webcam demo using the pre-trained model:

Bash

python start_kit/realtime_gloss.py --weights start_kit/final_pose_transformer.pth --classes 14 --id2gloss start_kit/id_to_gloss.json --camera 0 --show-skeleton
--weights: Path to the trained model file (.pth).

--classes: Number of signs the model was trained on (e.g., 14).

--id2gloss: Path to the JSON file mapping IDs to word labels.

--show-skeleton: (Optional) Visualizes the skeleton overlay on the video feed.

2. Reproducing the Training Pipeline
Follow these steps to recreate the dataset and retrain the model from scratch:

Step 1: Create Dataset Subset Generate the specific cafe vocabulary subset (WLASL_Cafe.json) from the original dataset.

Bash

python subsetCreater.py
Step 2: Download Raw Videos Download the source videos listed in your subset file.

Bash

cd start_kit
python video_downloader.py
Step 3: Convert Flash Videos Some older videos are in .swf format. Convert them to .mp4 using the provided script (requires a bash environment like Git Bash on Windows).

Bash

bash scripts/swf2mp4.sh
Step 4: Preprocess Videos Trim the raw videos to the exact start/end frames for each sign.

Bash

python preprocess.py
You should now see cleaned video clips in the videos/ directory.

Step 5: Extract Landmarks Open landmark_extracter.ipynb in Jupyter Notebook and run all cells.

This script processes every video in videos/ using MediaPipe Holistic.

It saves the extracted features (300 landmarks per frame) as .npy files in the landmarks/ folder.

Step 6: Train the Model Open train_transformer.ipynb in Jupyter Notebook and run the cells sequentially.

Data Preparation: Loads landmarks, normalizes data, applies augmentation (noise, rotation, time masking), and splits data.

Training: Trains the Pose Transformer model using Cross-Entropy loss and Adam optimizer.

Evaluation: Checks accuracy on validation/test sets.

Save: Saves the final trained weights to final_pose_transformer.pth.

📂 Project Structure
start_kit/: Inference scripts and model artifacts.

realtime_gloss.py: Main real-time application.

video_downloader.py: Script to download WLASL videos.

train_transformer.ipynb: Notebook for training the model.

landmark_extracter.ipynb: Notebook for extracting MediaPipe features.

preprocess.py: Script to trim raw videos.

subsetCreater.py: Script to filter the main dataset.

WLASL_Cafe.json: Subset definition file.

videos/: Directory containing processed video clips.

landmarks/: Directory containing extracted .npy feature files.

📊 Performance
The model achieves approximately 60-62% accuracy on the validation set using 5-Fold Cross-Validation. Real-time performance is optimized for CPU usage, achieving high FPS on standard laptops.
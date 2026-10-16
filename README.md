# HoloMotion: Real-Time Object Detection & Tracking

A clean, beginner-friendly computer vision project that detects everyday objects in real time and tracks them smoothly across video frames. Built with Python, OpenCV, Ultralytics YOLO, and ByteTrack.

Optimized to run locally on your **CPU** (tested on Fedora Linux with an AMD Ryzen processor) — no dedicated GPU required!

---

## 💡 The "Why Doesn't YOLO See My Pen?" Problem

If you've ever held up a pen, pencil, or pair of glasses in front of standard object detection models (like YOLOv8 Nano) and wondered why nothing shows up, here is what is happening under the hood:

- **The Standard COCO Dataset**: Most off-the-shelf YOLO models are pre-trained on COCO's 80 everyday categories (people, cars, cell phones, cups, bottles, books, laptops, etc.).
- **What COCO Lacks**: It does **not** include pens, pencils, glasses, keys, or watches. Since the base model was never trained on them, it simply cannot recognize them!

### How HoloMotion Fixes This
1. **Open-Vocabulary Mode (`--world`)**: HoloMotion supports **YOLO-World**, an open-vocabulary model that can detect almost any custom item you want — including pens, pencils, glasses, watches, and notebooks.
2. **Small Handheld Object Tuning**: Small items held in your hands often have lower confidence scores (~0.15–0.28). We:
   - Lowered the default confidence threshold to `0.20` (and support lowering down to `0.05`).
   - Added live keyboard controls (`[` and `]`) so you can nudge confidence up or down instantly.
   - Added a `P` hotkey to hide/show the person bounding box so small held items are easy to see.

---

## 🎯 Getting the Best Results: Why "Person" Detects Easily & How to Detect Smaller Objects

When sitting in front of a laptop webcam, you'll immediately notice that **Person** is detected with high confidence (75%–90%+), while smaller objects like phones, cups, pens, or bottles can sometimes be trickier. 

### Why is this?
1. **Physical Scale**: You occupy 60%–80% of the camera frame, giving the AI thousands of pixels of detail. A pen or phone in your hand might only be 20–40 pixels wide.
2. **Hand Occlusion**: When holding an object, your fingers usually cover 30%–50% of it, breaking its standard outline.
3. **Webcam Motion Blur & Indoor Light**: Laptop webcams adjust exposure for your face, which often blurs fast-moving hands and small items.
4. **Vocabulary Mismatch**: Standard YOLOv8 does not know what a "pen" or "glasses" is at all.

---

### 🛠️ 6 Quick Tips to Detect Smaller Objects Reliably

| Tip | What to Do | Why It Helps |
| :--- | :--- | :--- |
| **1. Increase Image Resolution** | Add `--imgsz 800` (e.g. `python main.py --imgsz 800`) | Gives the AI 2x more pixels to resolve small items like pens, glasses, and phones. |
| **2. Lower the Confidence Threshold** | Press **`[`** in the video window (or start with `--conf 0.15`) | Lowers the bar from 0.20 to 0.15 or 0.10 so fainter handheld objects appear immediately. |
| **3. Hide the Person Box** | Press **`P`** in the video window | Hides the giant body box so your handheld item isn't buried inside it. |
| **4. Hold Items to the Side** | Hold your cup, bottle, or phone slightly to the side of your torso | Clean contrast against the background wall/room makes the object contour 10x sharper. |
| **5. Pause for a Split Second** | Hold the item steady for half a second | Eliminates webcam motion blur so the neural network gets a clean, sharp frame. |
| **6. Focus YOLO-World Classes** | Pass only the items you want (e.g. `--classes "pen, phone, cup"`) | Sharper text embeddings give significantly higher detection confidence scores. |

---

## 🚀 Quick Mode Selection Guide

| What You Want to Detect | Which Command to Run | Notes |
| :--- | :--- | :--- |
| **Standard Desk Items**<br>*(Phone, Laptop, Cup, Bottle, Book, Chair, Person)* | `python main.py` | Ultra-fast (45–60 FPS). Uses standard YOLOv8n. |
| **Pens, Pencils, Glasses & Watches** | `python main.py --world` | Open-vocabulary YOLO-World. Detects items standard COCO missed. |
| **High-Precision Small Object Mode** | `python main.py --world --imgsz 800 --conf 0.15` | Best accuracy for small pens, thin glasses, and handheld items. |
| **Custom Object List** | `python main.py --world --classes "pen, notebook, headphones"` | Looks specifically for your chosen items. |
| **Video File (No Webcam Needed)** | `python main.py --source assets/sample_video.mp4` | Loops through recorded video automatically. |

---

## 💻 Environment Setup

If you're starting fresh, create and activate a virtual environment:

```bash
# Navigate to the repository
cd HoloMotion

# Create a virtual environment
python3 -m venv .venv

# Activate it
source .venv/bin/activate
```

### Install Dependencies

To keep things fast and avoid downloading gigabytes of CUDA packages when running on CPU:

```bash
# 1. Install lightweight PyTorch for CPU
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 2. Install the remaining requirements
pip install -r requirements.txt
```

---

## 🎥 Running the Application

### 1. Recommended: High-Clarity Mode for Small Desk Items
Best for detecting pens, pencils, glasses, and phones with higher resolution:

```bash
python main.py --world --imgsz 800 --conf 0.15
```

### 2. Standard YOLOv8 Mode (Maximum FPS)
Best for general objects (laptop, cup, bottle, book, phone, person):

```bash
python main.py
```

### 3. Video File Mode
Run detection and tracking on a video file:

```bash
# Play and loop continuously:
python main.py --source assets/sample_video.mp4

# Play once and stop at the end:
python main.py --source assets/sample_video.mp4 --no-loop
```

---

## ⌨️ Live Keyboard Controls

While the video window is active, click into it and use these shortcuts:

| Key | Action | Purpose |
| :--- | :--- | :--- |
| **`[`** | **Lower Confidence** (-0.05) | Makes fainter or smaller objects (like thin pens) show up immediately. |
| **`]`** | **Raise Confidence** (+0.05) | Cleans up faint false positives. |
| **`P`** | **Toggle Person Box** | Hides the large body box so handheld objects aren't obscured. |
| **`Q` / `ESC`** | **Quit** | Safely stops video capture and closes windows cleanly. |

---

## 🔍 Understanding the Screen Overlay

- **Bounding Box**: Outlines the object (amber for handheld items, teal for other objects).
- **Label & Percentage**: Object name and detection confidence (e.g. `pen 82%` or `#1 laptop 91%`).
- **Tracking ID (`#1`, `#2`)**: A persistent ID assigned by ByteTrack that follows each object as it moves across the screen. Newly appearing objects display their label immediately and receive a `#ID` once tracked.
- **Top HUD**: Shows live frames-per-second (FPS), number of tracked items, current confidence threshold, and person filter state.

---

## 🧪 Testing & Verification

A self-contained verification script is included to confirm your environment and models are working properly:

```bash
python test_pipeline.py
```

This verifies:
1. Python dependencies and tracker modules are properly imported.
2. Standard YOLOv8 loads on CPU.
3. Open-Vocabulary YOLO-World loads with custom vocabulary.
4. Video file processing and ByteTrack tracking loop run across consecutive frames.
5. Camera access and invalid path error handling work gracefully.

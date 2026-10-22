# SecureVision — AI Pipeline & Raspberry Pi 5 Integration Guide

## 1. Purpose

SecureVision is a smart door authentication system developed for our ECE 257 Senior Design project.

This repository contains the Rust-based facial authentication pipeline that I developed and tested on macOS. The next step is to integrate it onto the **Raspberry Pi 5 with the Raspberry Pi Camera Module 3**, and eventually connect authentication results to the ESP32-S3 lock controller.

**Important:** The AI pipeline has been developed and tested on macOS. It has not yet been verified end-to-end on the Raspberry Pi 5. The goal is to reuse the existing AI code rather than rewrite the models.

## 2. Getting My Code

My completed work is on the GitHub branch:

`jeeva/face-detection`

Repository: https://github.com/ifrahimm/securevision

If you already have the repository cloned:

```bash
git fetch origin
git switch --track origin/jeeva/face-detection
```

If the branch already exists locally, use `git switch jeeva/face-detection` and `git pull`.

If you are cloning the repository for the first time:

```bash
git clone -b jeeva/face-detection https://github.com/ifrahimm/securevision.git
cd securevision
```

All subsequent commands assume you're in the repository root unless stated otherwise.

## 3. Project Structure

The Rust workspace is located in `smart-lock/`.

```text
securevision/
├── README.md
└── smart-lock/
    ├── Cargo.toml
    ├── Cargo.lock
    ├── apps/
    │   └── pi-controller/
    │       └── src/main.rs
    └── crates/
        ├── lock-ai/
        │   ├── Cargo.toml
        │   ├── src/
        │   │   ├── lib.rs
        │   │   ├── face_detection.rs
        │   │   ├── face_alignment.rs
        │   │   ├── face_embedding.rs
        │   │   ├── face_recognition.rs
        │   │   ├── face_database.rs
        │   │   └── face_antispoof.rs
        │   ├── models/
        │   │   ├── yolov8n-face.onnx
        │   │   ├── w600k_mbf.onnx
        │   │   └── minifasnet_v2.onnx
        │   └── examples/
        │       ├── enroll_face.rs
        │       ├── recognize_face.rs
        │       ├── recognize_stream.rs
        │       ├── antispoof_detect_stream.rs
        │       └── securevision_demo.rs
        ├── lock-core/
        └── lock-protocol/
```

The three ONNX model files are committed to GitHub. You do not need to download them separately.

## 4. How the AI Pipeline Works

The facial authentication pipeline consists of four main stages:

**Step 1 — Face Detection: YOLOv8n-Face**

- Detects a face in the input image.
- Produces a bounding box and facial landmarks.
- Implemented in `src/face_detection.rs`.

**Step 2 — Face Alignment**

- Uses facial landmarks to normalize the face before recognition.
- Implemented in `src/face_alignment.rs`.

**Step 3 — Face Recognition: MobileFaceNet**

- Uses `w600k_mbf.onnx`.
- Produces a 512-dimensional face embedding.
- Compares the embedding with enrolled users using cosine similarity.
- Implemented in `face_embedding.rs`, `face_recognition.rs`, and `face_database.rs`.
- The current demo uses a similarity threshold of approximately `0.40`. This is a prototype setting, not a security-validated threshold.

**Step 4 — Anti-Spoofing: MiniFASNetV2**

- Uses `minifasnet_v2.onnx`.
- Intended to distinguish a live face from presentation attacks such as printed photographs or phone displays.
- Implemented in `src/face_antispoof.rs`.
- The model's output-class mapping still needs verification. Do not assume that a particular class index always means a live face.

The eventual access decision should require both successful recognition and successful liveness verification.

## 5. Build and Test the Code

Install a compatible Rust toolchain using rustup, then navigate to the workspace:

```bash
cd smart-lock
cargo check -p lock-ai
cargo check -p pi-controller
cargo test -p lock-core
cargo test -p lock-protocol
```

These commands have passed on macOS.

The `lock-core` crate includes authentication-decision logic. The `lock-protocol` crate defines messages intended for communication between the Raspberry Pi and ESP32-S3.

A warning about the dependency `block v0.1.6` was observed on macOS. It did not prevent compilation, but platform-specific build problems may still occur on Raspberry Pi OS.

## 6. Running the Existing AI Examples

The examples were developed and tested on macOS. They may need camera-backend or dependency adjustments on Raspberry Pi OS.

Run the examples from inside `lock-ai` because the existing code uses relative paths for models and enrollment data:

```bash
cd smart-lock/crates/lock-ai
```

If you're already inside `smart-lock`, run `cd crates/lock-ai` instead.

**Facial recognition using an image:**

```bash
cargo run --example recognize_face
```

**Live facial recognition:**

```bash
cargo run --example recognize_stream
```

**Live anti-spoofing:**

```bash
cargo run --example antispoof_detect_stream
```

**Combined demonstration:**

```bash
cargo run --example securevision_demo
```

The combined demo was tested on macOS and provides facial recognition and anti-spoofing demonstration modes.

Known keyboard controls for the combined demo:

- `1` — Recognition mode
- `2` — Anti-spoofing mode
- `Space` — Capture
- `R` — Reset
- `Esc` — Exit

Some examples may require an enrollment database or particular image inputs. Check the example's source code for its expected arguments before running it.

## 7. Enrollment Database

The facial recognition system uses:

`smart-lock/crates/lock-ai/enrollments.json`

This file contains enrolled face embeddings. It is intentionally excluded from Git because it contains biometric information.

**You will need to generate your own enrollment database.**

The repository includes:

`examples/enroll_face.rs`

Read this example to determine its input requirements and use it to create your own enrollment data.

Do not commit `enrollments.json`, personal face images, or other biometric test data to GitHub.

## 8. Your Main Task — Raspberry Pi 5 Integration

The priority is to get the existing AI pipeline working with the Raspberry Pi Camera Module 3.

### Task A — Set Up Raspberry Pi

Use Raspberry Pi OS 64-bit, install Rust, and clone the repository.

Verify the workspace builds on the Pi:

```bash
cd smart-lock
cargo check -p lock-ai
cargo check -p pi-controller
```

If a dependency fails, identify whether the issue is related to ARM64, native libraries, or the camera backend before modifying the AI inference code.

### Task B — Get Camera Module 3 Working

Before integrating AI, verify the Pi can capture an image from Camera Module 3.

On current Raspberry Pi OS versions, test with:

```bash
rpicam-hello
rpicam-still -o test.jpg
```

If these commands are unavailable, check the installed Raspberry Pi camera utilities and operating system version.

**Goal:** Successfully capture and save a photograph using Camera Module 3.

### Task C — Run Face Detection on a Captured Image

Start with a still image rather than a live video stream.

Use the existing YOLOv8n-Face implementation and verify that it detects a face in an image captured on the Pi.

The main files to inspect are:

- `src/face_detection.rs`
- `src/face_alignment.rs`
- `examples/recognize_face.rs`

You may need to adjust the example's image-loading code or input paths.

**Goal:** Produce a valid face bounding box and facial landmarks from a Pi camera image.

### Task D — Run Face Recognition

After face detection works:

1. Align the detected face.
2. Generate the MobileFaceNet embedding.
3. Load a locally generated enrollment database.
4. Compare the embedding against enrolled users.
5. Print the matched identity and cosine similarity.

**Goal:** Correctly identify an enrolled user and reject an unknown user in basic testing.

### Task E — Run Anti-Spoofing

Use `minifasnet_v2.onnx` on the detected face.

Test with:

- A live face.
- A printed face photograph.
- A face displayed on a phone.

Before interpreting the results as live/spoof decisions, verify the model's output-class labels and preprocessing requirements.

**Goal:** Obtain repeatable inference outputs and establish the correct class mapping.

### Task F — Integrate Live Camera Frames

Once still-image inference works, replace the macOS-oriented webcam capture with a Raspberry Pi-compatible camera source.

Possible approaches include a V4L2-compatible capture interface or a `libcamera`/`rpicam`-based capture pipeline, depending on the installed Raspberry Pi OS and camera configuration.

Try to keep camera acquisition separate from the AI pipeline so the existing inference code can be reused.

**Goal:** Run detection and recognition repeatedly on live frames and measure performance.

### Task G — Connect Authentication Results to Pi Controller

The relevant crates are:

`crates/lock-core`

This contains:

- `FaceAuthResult`
- `AccessDecision`
- `evaluate_face_auth()`

The current access-decision logic grants access only when `face_matched` and `liveness_passed` are both true.

`crates/lock-protocol`

This contains the `Message` enum and byte encoding:

| Message | Byte |
|---|---|
| AuthSuccess | `0x01` |
| AuthFailure | `0x02` |
| UnlockRequest | `0x03` |
| LockRequest | `0x04` |
| Ack | `0x05` |
| Error | `0x06` |

`apps/pi-controller/src/main.rs`

This is the starting point for Raspberry Pi-side controller integration.

**Important:** The message definitions exist, but end-to-end UART communication and physical lock control still need to be integrated and tested. Do not assume these functions are already complete.

Initially, log authentication decisions rather than activating a physical lock. Once liveness verification and UART communication are validated, connect the results to the ESP32-S3.

## 9. Recommended Integration Order

Work in this order to avoid debugging multiple systems simultaneously:

1. Build the Rust crates on Raspberry Pi OS.
2. Capture an image using Camera Module 3.
3. Run YOLOv8n-Face detection on that image.
4. Run MobileFaceNet recognition using local enrollment data.
5. Run MiniFASNetV2 and verify its output labels.
6. Connect the camera to continuous inference.
7. Measure detection, recognition, and anti-spoofing latency.
8. Integrate the authentication decision with the Pi controller.
9. Test UART communication with the ESP32-S3.
10. Perform complete system validation before enabling lock actuation.

## 10. What to Report Back

For our next integration update, please record:

- Raspberry Pi OS version and Rust version.
- Whether `lock-ai` and `pi-controller` compile.
- Whether Camera Module 3 captures images successfully.
- Whether the three ONNX models load and run.
- Face recognition and anti-spoofing test results.
- Approximate processing time per frame.
- Any source changes needed for Raspberry Pi compatibility.
- Remaining integration blockers.

## 11. Current Project Status

**Already implemented or tested on macOS:**

- YOLOv8n-Face detection.
- Face alignment.
- MobileFaceNet embedding generation and face matching.
- Local enrollment database support.
- MiniFASNetV2 inference.
- Combined Rust demonstration.
- Authentication decision logic and unit tests.
- UART message encoding and decoding with unit tests.
- Initial Pi controller crate.

**Still to complete or verify:**

- Raspberry Pi 5 ARM64 compatibility.
- Camera Module 3 capture integration.
- End-to-end live inference on the Pi.
- Anti-spoofing output-class validation.
- Performance and reliability testing.
- UART communication with the ESP32-S3.
- Physical locking mechanism integration.

**Main objective:** Reuse the existing AI pipeline, adapt camera capture for Raspberry Pi 5, and produce reliable authentication results that can eventually be passed to the ESP32-S3 lock controller.

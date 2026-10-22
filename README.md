SecureVision — AI Pipeline & Raspberry Pi Integration Guide
Branch: jeeva/face-detection
Reference commit: bb05e56
Status: AI components compile on the developer's macOS machine; Raspberry Pi 5 / Camera Module 3 integration is not yet verified.

1. What is SecureVision?
SecureVision is a prototype smart-door access system. The Raspberry Pi 5 is intended to handle camera capture and facial authentication; an ESP32-S3 is intended to handle sensors and lock actuation. The goal is to make an access decision using both identity matching and liveness (anti-spoofing), then communicate with the controller over UART.

Important: This branch provides AI and integration foundations, not a fully integrated or security-validated smart lock. Do not attach the demo's decisions directly to a real door lock.
2. Get the code
If you already have the team's repository:
git fetch origin
git switch --track origin/jeeva/face-detection
If the branch already exists locally, use git switch jeeva/face-detection and git pull instead. Save or commit your own changes before switching branches.

If you need to clone from scratch:
git clone https://github.com/ifrahimm/securevision.git
cd securevision
git switch --track origin/jeeva/face-detection
The Rust workspace is under smart-lock/:
cd smart-lock
cargo check -p lock-ai
cargo check -p pi-controller
cargo test -p lock-core
cargo test -p lock-protocol
These commands require a current Rust toolchain (rustup, cargo, rustc). A successful macOS check does not establish compatibility with Raspberry Pi OS or its camera backend.
3. Repository map
smart-lock/
├── Cargo.toml                   # Rust workspace members
├── Cargo.lock                   # Resolved dependency versions
├── apps/
│   ├── cli/                     # Existing CLI app
│   └── pi-controller/
│       ├── Cargo.toml
│       └── src/main.rs          # Pi-side controller starting point
├── crates/
│   ├── lock-ai/
│   │   ├── Cargo.toml
│   │   ├── models/
│   │   │   ├── yolov8n-face.onnx
│   │   │   ├── w600k_mbf.onnx
│   │   │   └── minifasnet_v2.onnx
│   │   ├── src/
│   │   │   ├── face_detection.rs
│   │   │   ├── face_alignment.rs
│   │   │   ├── face_embedding.rs
│   │   │   ├── face_recognition.rs
│   │   │   ├── face_database.rs
│   │   │   └── face_antispoof.rs
│   │   └── examples/
│   │       ├── enroll_face.rs
│   │       ├── recognize_face.rs
│   │       ├── recognize_stream.rs
│   │       ├── antispoof_detect_stream.rs
│   │       └── securevision_demo.rs
│   ├── lock-core/src/lib.rs     # FaceAuthResult and access decision
│   ├── lock-protocol/src/lib.rs # Shared Pi/ESP32 message definitions
│   └── lock-hal/                # Hardware abstraction crate
└── ...
The firmware/proto-v0 area exists separately; changes to ESP32 firmware were not part of the referenced AI commit.
4. How the AI pipeline fits together
Camera frame
    ↓
YOLOv8n-Face → face box + landmarks
    ├──→ Face alignment → MobileFaceNet → 512-value embedding
    │                                       ↓
    │                            Compare with enrolled embeddings
    │                                       ↓
    │                                  Face matched?
    └──→ Face crop → MiniFASNetV2 → Anti-spoof class scores
                                            ↓
                                   Liveness passed?*
                                            ↓
                     lock-core: evaluate_face_auth(...)
                                            ↓
                                 Granted / Denied
                                            ↓
                      Future Pi ↔ ESP32 UART integration
The MiniFASNetV2 output-class meaning is not yet verified. Do not interpret a class index as a trusted live/spoof verdict until validated against known real faces and presentation attacks.
Component	Purpose	Where to start
YOLOv8n-Face	Locate a face and facial landmarks	src/face_detection.rs
Face alignment	Normalize a detected face before embedding	src/face_alignment.rs
MobileFaceNet (w600k_mbf.onnx)	Generate a 512-dimensional identity embedding	src/face_embedding.rs
Face recognition	Compare embeddings by cosine similarity	src/face_recognition.rs
Enrollment database	Store/load identities and embeddings	src/face_database.rs
MiniFASNetV2	Produce anti-spoof class scores	src/face_antispoof.rs
Access decision	Require both identity and liveness to pass	../lock-core/src/lib.rs
UART protocol	Encode/decode one-byte control messages	../lock-protocol/src/lib.rs


The webcam demonstration uses a 0.40 cosine-similarity threshold for demonstration only. This is not a calibrated production security threshold.
5. Run the webcam demonstration (developer machine)
Run from the lock-ai crate directory, because the demo loads models and enrollment data with relative paths:
cd smart-lock/crates/lock-ai  # If starting at repository root
cargo run --example securevision_demo
Before running, you need:
- A working webcam recognized by the nokhwa camera backend (the demo currently opens camera index 0).
- A display/desktop environment; this demo uses a graphical window.
- The three ONNX files in models/ (included in this branch).
- A local enrollments.json in the lock-ai directory. This file is intentionally not committed because it can contain personal biometric embeddings.
The examples/enroll_face.rs source is included as a starting point for creating an enrollment. Read its source to check required inputs and invocation before running it; do not assume it is an interactive enrollment wizard.
Demo keyboard controls
Key	Action
1	Facial recognition mode
2	Anti-spoofing mode
Space	Analyze the current camera frame
R	Clear previous result
Esc	Exit


The demo expects exactly one detected face for analysis. Recognition compares the captured face against the locally enrolled embeddings. Anti-spoof mode reports a model class and confidence, not a verified live/spoof verdict.

Other example entry points worth reading are recognize_face.rs, recognize_stream.rs, and antispoof_detect_stream.rs. Their arguments and camera assumptions may differ; inspect their source before use.
6. Shared access-decision logic
In crates/lock-core/src/lib.rs:
pub struct FaceAuthResult {
    pub face_matched: bool,
    pub liveness_passed: bool,
}

pub enum AccessDecision {
    Granted,
    Denied,
}
evaluate_face_auth returns Granted only when both booleans are true. The four truth-table cases have unit tests.

This logic is implemented, but wiring real AI predictions into it safely is future integration work. Until anti-spoof classes are verified, use mocked decisions or a non-actuating test harness.
7. Pi ↔ ESP32 protocol foundation
crates/lock-protocol/src/lib.rs defines these one-byte messages:
Message	Byte
AuthSuccess	0x01
AuthFailure	0x02
UnlockRequest	0x03
LockRequest	0x04
Ack	0x05
Error	0x06


Use Message::to_byte() to encode and Message::from_byte() to decode. An unknown byte returns None.

Protocol limitation: The current message enum defines byte values; it does not by itself implement serial-port configuration, framing, retries, timeouts, authentication, or end-to-end communication. Those must be designed and tested with the ESP32 firmware.
8. Raspberry Pi 5 integration checklist
Work through these steps in order; do not assume the macOS webcam demo will run unchanged on Raspberry Pi OS.
- [ ] Clone the branch on Raspberry Pi 5 and confirm rustc --version and cargo --version.
- [ ] Run cargo check -p lock-ai and cargo check -p pi-controller; record any ARM64/Linux dependency issues.
- [ ] Verify the Pi Camera Module 3 independently using Raspberry Pi OS camera tools (rpicam-*, where available).
- [ ] Choose a camera capture interface compatible with the Pi camera stack; adapt the current nokhwa/webcam code as needed.
- [ ] Test YOLO face detection on a saved camera frame first, then face alignment and MobileFaceNet embedding.
- [ ] Create a local enrollment database on the Pi; do not copy another person's private embeddings without permission.
- [ ] Measure detection, embedding, and total authentication latency on the Pi.
- [ ] Validate MiniFASNetV2 class mapping with known live and spoof samples; document thresholds and failure cases.
- [ ] Connect model results to lock-core in a test harness without actuating the lock.
- [ ] Implement/test UART communication with ESP32-S3, including acknowledgements and error handling.
- [ ] Only after validation, test controlled lock actuation in a safe bench setup.
9. Troubleshooting
No such file or directory: models/... — Run the example from smart-lock/crates/lock-ai/, not from the workspace root.

Missing enrollments.json — The file is intentionally excluded from Git. Create a local enrollment using the supplied enrollment example after checking its expected inputs.

No camera / camera index error — The demo opens camera index 0. On Raspberry Pi OS, Camera Module 3 may not be exposed through the same backend; validate the camera separately before changing the AI model code.

No single face — The demo requires exactly one detected face. Try one person in frame with adequate lighting.

block v0.1.6 future-incompatibility warning — This appeared during successful macOS builds; it was not a build failure. Reassess dependencies if Rust version changes or the warning becomes an error.

cargo check passes but demo fails — cargo check verifies compilation, not webcam availability, ONNX runtime performance, enrollment data, or physical hardware.
10. What has been verified vs. what remains
Verified on the developer's macOS machine: cargo check -p lock-ai; cargo check -p pi-controller; four lock-core unit tests; two lock-protocol unit tests; successful Git push of commit bb05e56.

Still to verify: clean-clone build, Raspberry Pi OS/ARM64 build, Pi Camera Module 3 capture, end-to-end live Pi inference, anti-spoof class semantics, UART transport with ESP32-S3, and secure physical lock operation.
11. Collaboration notes
Work in your own branch for Pi-specific changes and open a pull request when ready. Keep changes to the camera backend and platform integration separate from model-processing logic when possible. Include test commands, hardware details, error output, and measured latency in pull requests.

Do not commit enrollments.json, face photos, test captures, personal embeddings, credentials, or device-specific secrets. The three .onnx model files are already tracked in the shared branch.
Suggested first task for the Pi teammate: Build lock-ai on Raspberry Pi 5, verify Camera Module 3 capture, and run face detection on one saved frame. This isolates camera/platform issues before attempting live recognition or ESP32 integration.
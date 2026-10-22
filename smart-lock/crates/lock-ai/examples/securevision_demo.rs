//! SecureVision webcam demonstration. Run from the lock-ai crate root.
//! Keys: 1 = recognition, 2 = anti-spoofing, SPACE = analyze, R = clear, ESC = quit.
use image::{imageops::FilterType, Rgb, RgbImage};
use lock_ai::face_alignment::align_face;
use lock_ai::face_antispoof::{crop_antispoof_face, predict_antispoof};
use lock_ai::face_database::FaceDatabase;
use lock_ai::face_detection::{decode_output, Detection};
use lock_ai::face_embedding::generate_embedding_from_image;
use lock_ai::face_recognition::cosine_similarity;
use minifb::{Key, KeyRepeat, Window, WindowOptions};
use nokhwa::{pixel_format::RgbFormat, utils::{CameraIndex, RequestedFormat, RequestedFormatType}, Camera};
use powerboxesrs::nms::nms_slice;
use std::{error::Error, sync::mpsc::{self, Receiver, Sender}, thread, time::{Duration, Instant}};
use tract_onnx::prelude::*;

const WIDTH: usize = 960;
const HEIGHT: usize = 720;
const VIDEO_HEIGHT: usize = 540;
const THRESHOLD: f32 = 0.40; // Demonstration threshold only.
const CROP_SCALE: f32 = 2.7; // Experimental crop scale.

type DemoResult<T> = Result<T, Box<dyn Error + Send + Sync>>;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode { Recognition, AntiSpoof }
impl Mode { fn label(self) -> &'static str { match self { Self::Recognition => "FACIAL RECOGNITION", Self::AntiSpoof => "ANTI-SPOOFING" } } }

struct Job { img: RgbImage, mode: Mode }
struct Outcome { mode: Mode, bbox: Option<[f32; 4]>, title: String, detail: String, latency_ms: f64 }

fn detect_face(img: &RgbImage, detector: &TypedRunnableModel) -> TractResult<Option<Detection>> {
    let (w, h) = img.dimensions();
    if w == 0 || h == 0 { return Err(std::io::Error::other("Empty camera frame").into()); }
    let scale = (640.0 / w as f32).min(640.0 / h as f32);
    let nw = (w as f32 * scale).round() as u32;
    let nh = (h as f32 * scale).round() as u32;
    let px = (640 - nw) / 2;
    let py = (640 - nh) / 2;
    let resized = image::imageops::resize(img, nw, nh, FilterType::Triangle);
    let mut letterboxed = RgbImage::from_pixel(640, 640, Rgb([114, 114, 114]));
    image::imageops::replace(&mut letterboxed, &resized, px as i64, py as i64);
    let mut input = tract_ndarray::Array4::<f32>::zeros((1, 3, 640, 640));
    for y in 0..640 { for x in 0..640 {
        let p = letterboxed.get_pixel(x, y);
        for c in 0..3 { input[[0, c, y as usize, x as usize]] = p[c] as f32 / 255.0; }
    }}
    let outputs = std::sync::Arc::new(detector.clone()).run(tvec!(input.into_tensor().into()))?;
    let mut candidates = Vec::new();
    for (output, stride) in outputs.iter().zip([8.0_f32, 16.0, 32.0]) {
        candidates.append(&mut decode_output(output, stride, 0.25)?);
    }
    let mut boxes = Vec::new(); let mut scores = Vec::new();
    for d in &candidates {
        boxes.extend_from_slice(&[d.x1, d.y1, d.x2, d.y2]);
        scores.push(d.confidence as f64);
    }
    let keep = nms_slice(&boxes, &scores, 0.45, 0.25);
    if keep.len() != 1 { return Ok(None); }
    let mut face = candidates[keep[0]].clone();
    face.x1 = (face.x1 - px as f32) / scale;
    face.y1 = (face.y1 - py as f32) / scale;
    face.x2 = (face.x2 - px as f32) / scale;
    face.y2 = (face.y2 - py as f32) / scale;
    face.landmarks = face.landmarks.map(|(x, y)| ((x - px as f32) / scale, (y - py as f32) / scale));
    Ok(Some(face))
}

fn run_job(job: Job, detector: &TypedRunnableModel, recognition: &TypedRunnableModel,
           antispoof: &std::sync::Arc<TypedRunnableModel>, db: &FaceDatabase) -> DemoResult<Outcome> {
    let start = Instant::now();
    let mode = job.mode;
    let Some(face) = detect_face(&job.img, detector)? else {
        return Ok(Outcome { mode, bbox: None, title: "NO SINGLE FACE".into(),
            detail: "Need exactly one detected face".into(), latency_ms: start.elapsed().as_secs_f64()*1000.0 });
    };
    let bbox = Some([face.x1, face.y1, face.x2, face.y2]);
    let (title, detail) = match mode {
        Mode::AntiSpoof => {
            let crop = crop_antispoof_face(&job.img, &face, CROP_SCALE)
                .map_err(std::io::Error::other)?;
            let p = predict_antispoof(&crop, antispoof)?;
            let cls = (0..3).max_by(|&a, &b| p[a].total_cmp(&p[b])).unwrap();
            (format!("CLASS {cls} - {:.2}%", p[cls]*100.0),
             "Class mapping unverified; test only".to_string())
        }
        Mode::Recognition => {
            let aligned = align_face(&job.img, &face.landmarks).map_err(std::io::Error::other)?;
            let embedding = generate_embedding_from_image(&aligned, recognition)?;
            let mut best: Option<(&str, f32)> = None;
            for user in &db.users {
                if user.embedding.len() != 512 { continue; }
                if let Some(score) = cosine_similarity(&user.embedding, &embedding) {
                    if score.is_finite() && best.as_ref().is_none_or(|(_, s)| score > *s) {
                        best = Some((&user.name, score));
                    }
                }
            }
            match best {
                Some((name, score)) if score >= THRESHOLD =>
                    (format!("MATCH: {name}"), format!("Similarity {score:.4} - demo only")),
                Some((_, score)) => ("UNKNOWN FACE".into(), format!("Best similarity {score:.4}")),
                None => ("UNKNOWN FACE".into(), "No valid enrolled matches".into()),
            }
        }
    };
    Ok(Outcome { mode, bbox, title, detail, latency_ms: start.elapsed().as_secs_f64()*1000.0 })
}

fn worker(rx: Receiver<Job>, tx: Sender<Result<Outcome, String>>) -> DemoResult<()> {
    let detector = tract_onnx::onnx().model_for_path("models/yolov8n-face.onnx")?
        .with_input_fact(0, f32::fact([1, 3, 640, 640]).into())?.into_optimized()?.into_runnable()?;
    let recognition = tract_onnx::onnx().model_for_path("models/w600k_mbf.onnx")?
        .with_input_fact(0, f32::fact([1, 3, 112, 112]).into())?.into_optimized()?.into_runnable()?;
    let antispoof = std::sync::Arc::new(tract_onnx::onnx().model_for_path("models/minifasnet_v2.onnx")?
        .with_input_fact(0, f32::fact([1, 3, 80, 80]).into())?.into_optimized()?.into_runnable()?);
    let db = FaceDatabase::load("enrollments.json")?;
    while let Ok(job) = rx.recv() {
        let result = run_job(job, &detector, &recognition, &antispoof, &db).map_err(|e| e.to_string());
        if tx.send(result).is_err() { break; }
    }
    Ok(())
}

fn rect(buffer: &mut [u32], x1: i32, y1: i32, x2: i32, y2: i32, color: u32) {
    let (x1, x2) = (x1.max(0).min(WIDTH as i32-1), x2.max(0).min(WIDTH as i32-1));
    let (y1, y2) = (y1.max(0).min(VIDEO_HEIGHT as i32-1), y2.max(0).min(VIDEO_HEIGHT as i32-1));
    for y in y1..=y2 { for x in x1..=x2 {
        if y-y1 < 3 || y2-y < 3 || x-x1 < 3 || x2-x < 3 {
            buffer[y as usize*WIDTH+x as usize] = color;
        }
    }}
}

fn text(buffer: &mut [u32], x: usize, y: usize, message: &str, color: u32, scale: usize) {
    use font8x8::{UnicodeFonts, BASIC_FONTS};
    let mut pen = x;
    for ch in message.chars() {
        if let Some(glyph) = BASIC_FONTS.get(ch) {
            for (gy, bits) in glyph.iter().enumerate() {
                for gx in 0..8 {
                    if bits & (1 << gx) != 0 {
                        for sy in 0..scale { for sx in 0..scale {
                            let px = pen+gx*scale+sx; let py = y+gy*scale+sy;
                            if px < WIDTH && py < HEIGHT { buffer[py*WIDTH+px] = color; }
                        }}
                    }
                }
            }
        }
        pen += 8*scale+scale;
        if pen >= WIDTH.saturating_sub(8*scale) { break; }
    }
}

fn render(buffer: &mut [u32], frame: &RgbImage, mode: Mode, result: Option<&Outcome>, busy: bool, status: &str) {
    buffer.fill(0x101827);
    let fw = frame.width() as f32; let fh = frame.height() as f32;
    let scale = (WIDTH as f32/fw).min(VIDEO_HEIGHT as f32/fh);
    let dw = (fw*scale).round() as usize; let dh = (fh*scale).round() as usize;
    let ox = (WIDTH-dw)/2; let oy = (VIDEO_HEIGHT-dh)/2;
    // Nearest-neighbor preview is deliberately cheap; inference uses the original frame.
    for y in 0..dh { for x in 0..dw {
        let sx = ((x as f32/scale) as u32).min(frame.width()-1);
        let sy = ((y as f32/scale) as u32).min(frame.height()-1);
        let p = frame.get_pixel(sx, sy);
        buffer[(oy+y)*WIDTH+ox+x] = ((p[0] as u32)<<16)|((p[1] as u32)<<8)|p[2] as u32;
    }}
    if let Some(r) = result {
        if r.mode == mode {
            if let Some([x1,y1,x2,y2]) = r.bbox {
                rect(buffer, (ox as f32+x1*scale) as i32, (oy as f32+y1*scale) as i32,
                     (ox as f32+x2*scale) as i32, (oy as f32+y2*scale) as i32, 0x00ff99);
            }
        }
    }
    text(buffer, 18, 555, mode.label(), 0x7dd3fc, 2);
    if busy { text(buffer, 18, 586, "ANALYZING...", 0xffcc55, 2); }
    else if let Some(r) = result.filter(|r| r.mode == mode) {
        text(buffer, 18, 586, &r.title.to_ascii_uppercase(), 0x80ffaa, 2);
        text(buffer, 18, 616, &format!("{} | {:.0} MS", r.detail, r.latency_ms).to_ascii_uppercase(), 0xffffff, 1);
    } else { text(buffer, 18, 586, status, 0xffffff, 2); }
    text(buffer, 18, 679, "1 RECOGNITION   2 ANTI-SPOOF   SPACE CAPTURE   R RESET   ESC EXIT", 0xb7c8d8, 1);
    text(buffer, 18, 697, "DEMO ONLY - NO LOCK ACTUATION", 0xffbb88, 1);
}

fn main() -> DemoResult<()> {
    let (job_tx, job_rx) = mpsc::channel::<Job>();
    let (result_tx, result_rx) = mpsc::channel::<Result<Outcome, String>>();
    thread::spawn(move || { if let Err(e) = worker(job_rx, result_tx.clone()) {
        let _ = result_tx.send(Err(format!("Model initialization failed: {e}")));
    }});
    let requested = RequestedFormat::new::<RgbFormat>(RequestedFormatType::AbsoluteHighestResolution);
    let mut camera = Camera::new(CameraIndex::Index(0), requested)?;
    camera.open_stream()?;
    let mut window = Window::new("SecureVision - Live Demo", WIDTH, HEIGHT, WindowOptions::default())?;
    window.set_target_fps(30);
    let mut buffer = vec![0_u32; WIDTH*HEIGHT];
    let mut mode = Mode::Recognition;
    let mut outcome: Option<Outcome> = None;
    let mut busy = false;
    let mut status = "PRESS SPACE TO ANALYZE".to_string();
    while window.is_open() && !window.is_key_down(Key::Escape) {
        // Always consume camera frames to avoid stale buffered captures.
        let frame = camera.frame()?;
        let img = frame.decode_image::<RgbFormat>()?;
        if window.is_key_pressed(Key::Key1, KeyRepeat::No) { mode = Mode::Recognition; outcome = None; }
        if window.is_key_pressed(Key::Key2, KeyRepeat::No) { mode = Mode::AntiSpoof; outcome = None; }
        if window.is_key_pressed(Key::R, KeyRepeat::No) { outcome = None; status = "PRESS SPACE TO ANALYZE".into(); }
        if window.is_key_pressed(Key::Space, KeyRepeat::No) && !busy {
            job_tx.send(Job { img: img.clone(), mode })?;
            busy = true;
            outcome = None;
        }
        if let Ok(reply) = result_rx.try_recv() {
            busy = false;
            match reply {
                Ok(r) => { println!("{}: {} | {} | {:.1} ms", r.mode.label(), r.title, r.detail, r.latency_ms); outcome = Some(r); }
                Err(e) => { eprintln!("Demo error: {e}"); status = "ERROR - SEE TERMINAL".into(); outcome = None; }
            }
        }
        render(&mut buffer, &img, mode, outcome.as_ref(), busy, &status);
        window.update_with_buffer(&buffer, WIDTH, HEIGHT)?;
    }
    camera.stop_stream()?;
    Ok(())
}

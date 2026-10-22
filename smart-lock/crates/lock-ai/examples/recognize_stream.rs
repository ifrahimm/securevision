
use nokhwa::{
    pixel_format::RgbFormat,
    utils::{CameraIndex, RequestedFormat, RequestedFormatType},
    Camera,
};

use std::thread;
use std::time::Duration;
use image::imageops::FilterType;
use image::{Rgb, RgbImage};

use powerboxesrs::nms::nms_slice;
use tract_onnx::prelude::*;

use lock_ai::face_alignment::align_face;
use lock_ai::face_database::FaceDatabase;
use lock_ai::face_detection::{decode_output, Detection};
use lock_ai::face_embedding::generate_embedding_from_image;
use lock_ai::face_recognition::cosine_similarity;

use std::io::{self, Write};
use std::time::Instant;

const DEMO_THRESHOLD: f32 = 0.40;

fn recognize_image(
    image_path: &str,
    detector: &TypedRunnableModel,
    recognition_model: &TypedRunnableModel,
    database: &FaceDatabase,
) -> TractResult<()> {
    let start = Instant::now();

    // =====================================================
    // 1. IMAGE LOADING & PREPROCESSING
    // =====================================================

    let preprocessing_start = Instant::now();

    let img = image::open(image_path)?.to_rgb8();

    let original_width = img.width();
    let original_height = img.height();

    if original_width == 0 || original_height == 0 {
        return Err(io::Error::other("Empty image").into());
    }

    // Letterbox image to 640x640
    let scale = f32::min(
        640.0 / original_width as f32,
        640.0 / original_height as f32,
    );

    let new_width = (original_width as f32 * scale).round() as u32;
    let new_height = (original_height as f32 * scale).round() as u32;

    let resized = image::imageops::resize(
        &img,
        new_width,
        new_height,
        FilterType::Triangle,
    );

    let mut letterboxed =
        RgbImage::from_pixel(640, 640, Rgb([114, 114, 114]));

    let pad_x = (640 - new_width) / 2;
    let pad_y = (640 - new_height) / 2;

    image::imageops::replace(
        &mut letterboxed,
        &resized,
        pad_x as i64,
        pad_y as i64,
    );

    // Create YOLO input tensor
    let mut input =
        tract_ndarray::Array4::<f32>::zeros((1, 3, 640, 640));

    for y in 0..640 {
        for x in 0..640 {
            let pixel = letterboxed.get_pixel(x, y);

            for c in 0..3 {
                input[[0, c, y as usize, x as usize]] =
                    pixel[c] as f32 / 255.0;
            }
        }
    }

    let preprocessing_time = preprocessing_start.elapsed();

    // =====================================================
    // 2. YOLOv8n-FACE INFERENCE
    // =====================================================

    let detection_start = Instant::now();

    let outputs = std::sync::Arc::new(detector.clone())
        .run(tvec!(input.into_tensor().into()))?;

    let detection_time = detection_start.elapsed();

    // =====================================================
    // 3. FACE DETECTION POSTPROCESSING
    // =====================================================

    let postprocessing_start = Instant::now();

    let strides = [8.0f32, 16.0, 32.0];
    let mut candidates = Vec::<Detection>::new();

    for (output, stride) in outputs.iter().zip(strides.iter()) {
        candidates.append(
            &mut decode_output(output, *stride, 0.25)?
        );
    }

    let mut boxes = Vec::new();
    let mut scores = Vec::new();

    for detection in &candidates {
        boxes.extend_from_slice(&[
            detection.x1,
            detection.y1,
            detection.x2,
            detection.y2,
        ]);

        scores.push(detection.confidence as f64);
    }

    let keep = nms_slice(&boxes, &scores, 0.45, 0.25);

    let detections: Vec<Detection> = keep
        .into_iter()
        .map(|i| candidates[i].clone())
        .collect();

    let postprocessing_time = postprocessing_start.elapsed();

    println!("Faces detected: {}", detections.len());

    // Reject zero or multiple faces
    if detections.len() != 1 {
        println!(
            "Authentication: REJECTED (requires exactly one face)"
        );

        println!("\n--- Latency Breakdown ---");
        println!(
            "Preprocessing:       {:?}",
            preprocessing_time
        );
        println!(
            "YOLO inference:      {:?}",
            detection_time
        );
        println!(
            "Detection postproc:  {:?}",
            postprocessing_time
        );
        println!(
            "Total processing:    {:?}",
            start.elapsed()
        );

        return Ok(());
    }

    // =====================================================
    // 4. LANDMARK COORDINATE CONVERSION
    // =====================================================

    let landmarks = detections[0].landmarks.map(|(x, y)| {
        (
            (x - pad_x as f32) / scale,
            (y - pad_y as f32) / scale,
        )
    });

    // =====================================================
    // 5. FACE ALIGNMENT
    // =====================================================

    let alignment_start = Instant::now();

    let aligned = align_face(&img, &landmarks)
        .map_err(|e| {
            io::Error::other(
                format!("Face alignment failed: {e}")
            )
        })?;

    let alignment_time = alignment_start.elapsed();

    // =====================================================
    // 6. MOBILEFACENET EMBEDDING GENERATION
    // =====================================================

    let embedding_start = Instant::now();

    let embedding =
        generate_embedding_from_image(
            &aligned,
            recognition_model,
        )?;

    let embedding_time = embedding_start.elapsed();

    // =====================================================
    // 7. COSINE SIMILARITY MATCHING
    // =====================================================

    let matching_start = Instant::now();

    let mut best_match: Option<(&str, f32)> = None;

    for user in &database.users {
        if user.embedding.len() != 512 {
            continue;
        }

        if let Some(score) =
            cosine_similarity(&user.embedding, &embedding)
        {
            if !score.is_finite() {
                continue;
            }

            println!("{}: {:.4}", user.name, score);

            if best_match
                .as_ref()
                .is_none_or(|(_, best)| score > *best)
            {
                best_match = Some((&user.name, score));
            }
        }
    }

    let matching_time = matching_start.elapsed();

    // =====================================================
    // 8. RECOGNITION RESULT
    // =====================================================

    println!("\n--- Recognition Result ---");

    match best_match {
        Some((name, score)) if score >= DEMO_THRESHOLD => {
            println!("Best match: {}", name);
            println!("Similarity: {:.4}", score);
            println!("Authentication: ACCEPTED (DEMO ONLY)");
        }

        Some((name, score)) => {
            println!("Closest match: {}", name);
            println!("Similarity: {:.4}", score);
            println!("Authentication: REJECTED");
        }

        None => {
            println!(
                "Authentication: REJECTED (no valid matches)"
            );
        }
    }

    // =====================================================
    // 9. LATENCY BREAKDOWN
    // =====================================================

    println!("\n--- Latency Breakdown ---");

    println!(
        "Preprocessing:       {:.3} ms",
        preprocessing_time.as_secs_f64() * 1000.0
    );

    println!(
        "YOLO inference:      {:.3} ms",
        detection_time.as_secs_f64() * 1000.0
    );

    println!(
        "Detection postproc:  {:.3} ms",
        postprocessing_time.as_secs_f64() * 1000.0
    );

    println!(
        "Face alignment:      {:.3} ms",
        alignment_time.as_secs_f64() * 1000.0
    );

    println!(
        "MobileFaceNet:       {:.3} ms",
        embedding_time.as_secs_f64() * 1000.0
    );

    println!(
        "Similarity matching: {:.3} ms",
        matching_time.as_secs_f64() * 1000.0
    );

    println!(
        "Total processing:    {:.3} ms",
        start.elapsed().as_secs_f64() * 1000.0
    );

    Ok(())
}
fn main() -> TractResult<()> {
    println!("==================================");
    println!("SecureVision - Recognition Stream");
    println!("==================================");

    let startup = Instant::now();

    // Load YOLO once
    println!("Loading YOLOv8n-Face...");

    let detector = tract_onnx::onnx()
        .model_for_path("models/yolov8n-face.onnx")?
        .with_input_fact(0, f32::fact([1, 3, 640, 640]).into())?
        .into_optimized()?
        .into_runnable()?;

    // Load MobileFaceNet once
    println!("Loading MobileFaceNet...");

    let recognition_model = tract_onnx::onnx()
        .model_for_path("models/w600k_mbf.onnx")?
        .with_input_fact(0, f32::fact([1, 3, 112, 112]).into())?
        .into_optimized()?
        .into_runnable()?;

    // Load enrollment database once
    println!("Loading enrollment database...");

    let database = FaceDatabase::load("enrollments.json")?;

    println!("Enrolled users: {}", database.users.len());
    println!("Startup time: {:?}", startup.elapsed());

        println!("\nModels loaded. Starting webcam...");

    let requested = RequestedFormat::new::<RgbFormat>(
        RequestedFormatType::AbsoluteHighestResolution,
    );

    let mut camera =
        Camera::new(CameraIndex::Index(0), requested)?;

    camera.open_stream()?;

    // Warm up the camera once.
    println!("Warming up camera...");

    for _ in 0..30 {
        let _ = camera.frame()?;
        thread::sleep(Duration::from_millis(100));
    }

    println!("Camera ready.");
    println!("Press ENTER to capture, or type 'quit' to exit.");

    let stdin = io::stdin();

    loop {
        print!("\nCapture > ");
        io::stdout().flush()?;

        let mut input = String::new();

        if stdin.read_line(&mut input)? == 0 {
            break;
        }

        if input.trim().eq_ignore_ascii_case("quit") {
            break;
        }

        // Capture a webcam frame.
        let frame = camera.frame()?;
        let decoded = frame.decode_image::<RgbFormat>()?;

        // Save the frame for the existing recognition pipeline.
        let image_path = "webcam_capture.jpg";
        decoded.save(image_path)?;

        println!("Image captured.");

        // Run the existing recognition pipeline.
        if let Err(error) = recognize_image(
            image_path,
            &detector,
            &recognition_model,
            &database,
        ) {
            eprintln!("Recognition error: {error}");
            println!("Authentication: REJECTED");
        }
    }

    camera.stop_stream()?;

    println!("SecureVision stopped.");

    Ok(())
}
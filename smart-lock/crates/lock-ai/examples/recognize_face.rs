
use image::imageops::FilterType;
use image::{Rgb, RgbImage};

use powerboxesrs::nms::nms_slice;
use tract_onnx::prelude::*;

use lock_ai::face_alignment::align_face;
use lock_ai::face_antispoof::{
    crop_antispoof_face,
    predict_antispoof,
};
use lock_ai::face_database::FaceDatabase;
use lock_ai::face_detection::{decode_output, Detection};
use lock_ai::face_embedding::generate_embedding_from_image;
use lock_ai::face_recognition::cosine_similarity;

use std::sync::Arc;
use std::time::Instant;

const DEMO_THRESHOLD: f32 = 0.40;
const ANTISPOOF_CROP_SCALE: f32 = 2.7;

fn main() -> TractResult<()> {
    let total_start = Instant::now();

    let args: Vec<String> = std::env::args().collect();

    if args.len() != 2 {
        eprintln!(
            "Usage: cargo run --example recognize_face -- <input_image>"
        );
        std::process::exit(1);
    }

    let image_path = &args[1];

    println!("SecureVision - Integrated Face Recognition");
    println!("Input: {}", image_path);

    // --------------------------------------------------
    // 1. Load original image
    // --------------------------------------------------
    let preprocessing_start = Instant::now();

    let img = image::open(image_path)?.to_rgb8();

    let original_width = img.width();
    let original_height = img.height();

    if original_width == 0 || original_height == 0 {
        return Err(std::io::Error::other(
            "Empty input image"
        ).into());
    }

    // --------------------------------------------------
    // 2. Letterbox to 640x640
    // --------------------------------------------------
    let scale = f32::min(
        640.0 / original_width as f32,
        640.0 / original_height as f32,
    );

    let new_width =
        (original_width as f32 * scale).round() as u32;

    let new_height =
        (original_height as f32 * scale).round() as u32;

    let resized = image::imageops::resize(
        &img,
        new_width,
        new_height,
        FilterType::Triangle,
    );

    let mut letterboxed =
        RgbImage::from_pixel(
            640,
            640,
            Rgb([114, 114, 114]),
        );

    let pad_x = (640 - new_width) / 2;
    let pad_y = (640 - new_height) / 2;

    image::imageops::replace(
        &mut letterboxed,
        &resized,
        pad_x as i64,
        pad_y as i64,
    );

    // --------------------------------------------------
    // 3. Create YOLO input tensor
    // --------------------------------------------------
    let mut input =
        tract_ndarray::Array4::<f32>::zeros(
            (1, 3, 640, 640)
        );

    for y in 0..640 {
        for x in 0..640 {
            let pixel = letterboxed.get_pixel(x, y);

            for c in 0..3 {
                input[[0, c, y as usize, x as usize]] =
                    pixel[c] as f32 / 255.0;
            }
        }
    }

    let preprocessing_time =
        preprocessing_start.elapsed();

    // --------------------------------------------------
    // 4. Load YOLOv8 face detector
    // --------------------------------------------------
    println!("Loading YOLOv8n-Face...");

    let detector_load_start = Instant::now();

    let detector = tract_onnx::onnx()
        .model_for_path("models/yolov8n-face.onnx")?
        .with_input_fact(
            0,
            f32::fact([1, 3, 640, 640]).into(),
        )?
        .into_optimized()?
        .into_runnable()?;

    let detector_load_time =
        detector_load_start.elapsed();

    // --------------------------------------------------
    // 5. Run YOLO detection and NMS
    // --------------------------------------------------
    let detection_start = Instant::now();

    let outputs =
        detector.run(tvec!(input.into_tensor().into()))?;

    let strides = [8.0f32, 16.0, 32.0];

    let mut candidates = Vec::<Detection>::new();

    for (output, stride) in
        outputs.iter().zip(strides.iter())
    {
        candidates.append(
            &mut decode_output(output, *stride, 0.25)?
        );
    }

    let mut boxes = Vec::new();
    let mut scores = Vec::new();

    for d in &candidates {
        boxes.extend_from_slice(&[
            d.x1,
            d.y1,
            d.x2,
            d.y2,
        ]);

        scores.push(d.confidence as f64);
    }

    let keep =
        nms_slice(&boxes, &scores, 0.45, 0.25);

    let detections: Vec<Detection> = keep
        .into_iter()
        .map(|i| candidates[i].clone())
        .collect();

    let detection_time =
        detection_start.elapsed();

    println!("Faces detected: {}", detections.len());

    if detections.len() != 1 {
        println!(
            "Identity match: NO (requires exactly one face)"
        );
        println!("Liveness decision: NOT ENABLED");
        println!(
            "Total elapsed: {:?}",
            total_start.elapsed()
        );
        return Ok(());
    }

    // --------------------------------------------------
    // 6. Convert detection to original coordinates
    // --------------------------------------------------
    let detection = &detections[0];

    let original_detection = Detection {
        x1: (detection.x1 - pad_x as f32) / scale,
        y1: (detection.y1 - pad_y as f32) / scale,
        x2: (detection.x2 - pad_x as f32) / scale,
        y2: (detection.y2 - pad_y as f32) / scale,

        confidence: detection.confidence,

        landmarks: detection.landmarks.map(|(x, y)| {
            (
                (x - pad_x as f32) / scale,
                (y - pad_y as f32) / scale,
            )
        }),
    };

    // --------------------------------------------------
    // 7. Create anti-spoofing crop
    // --------------------------------------------------
    let crop_start = Instant::now();

    let antispoof_crop = crop_antispoof_face(
        &img,
        &original_detection,
        ANTISPOOF_CROP_SCALE,
    )
    .map_err(std::io::Error::other)?;

    let crop_time = crop_start.elapsed();

    // Save for visual inspection during development.
    antispoof_crop.save(
        "test_images/antispoof_crop.png"
    )?;

    println!("Anti-spoof crop saved successfully.");

    // --------------------------------------------------
    // 8. Load MiniFASNetV2
    // --------------------------------------------------
    println!("Loading MiniFASNetV2...");

    let antispoof_load_start = Instant::now();

    // Arc is required by the Tract 0.23.8 run() API
    // used in face_antispoof.rs.
    let antispoof_model = Arc::new(
        tract_onnx::onnx()
            .model_for_path("models/minifasnet_v2.onnx")?
            .with_input_fact(
                0,
                f32::fact([1, 3, 80, 80]).into(),
            )?
            .into_optimized()?
            .into_runnable()?
    );

    let antispoof_load_time =
        antispoof_load_start.elapsed();

    // --------------------------------------------------
    // 9. Run anti-spoofing inference
    // --------------------------------------------------
    let antispoof_start = Instant::now();

    let probabilities = predict_antispoof(
        &antispoof_crop,
        &antispoof_model,
    )?;

    let antispoof_time =
        antispoof_start.elapsed();

    println!("\n--- Anti-Spoofing Result ---");

    for (class, probability) in
        probabilities.iter().enumerate()
    {
        println!(
            "Class {}: {:.2}%",
            class,
            probability * 100.0
        );
    }

    let predicted_class = probabilities
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(index, _)| index)
        .unwrap_or(0);

    println!("Predicted class: {}", predicted_class);
    println!("Class labels: NOT VERIFIED");
    println!("Liveness decision: NOT ENABLED");

    // --------------------------------------------------
    // 10. Align face for recognition
    // --------------------------------------------------
    let alignment_start = Instant::now();

    let aligned = align_face(
        &img,
        &original_detection.landmarks,
    )
    .map_err(|e| {
        std::io::Error::other(
            format!("Face alignment failed: {e}")
        )
    })?;

    let alignment_time =
        alignment_start.elapsed();

    // --------------------------------------------------
    // 11. Load MobileFaceNet
    // --------------------------------------------------
    println!("\nLoading MobileFaceNet...");

    let recognition_load_start = Instant::now();

    let recognition_model = tract_onnx::onnx()
        .model_for_path("models/w600k_mbf.onnx")?
        .with_input_fact(
            0,
            f32::fact([1, 3, 112, 112]).into(),
        )?
        .into_optimized()?
        .into_runnable()?;

    let recognition_load_time =
        recognition_load_start.elapsed();

    // --------------------------------------------------
    // 12. Generate embedding
    // --------------------------------------------------
    let embedding_start = Instant::now();

    let embedding = generate_embedding_from_image(
        &aligned,
        &recognition_model,
    )?;

    let embedding_time =
        embedding_start.elapsed();

    // --------------------------------------------------
    // 13. Load database and compare embeddings
    // --------------------------------------------------
    let matching_start = Instant::now();

    let database =
        FaceDatabase::load("enrollments.json")?;

    println!(
        "Enrolled users: {}",
        database.users.len()
    );

    let mut best_match: Option<(&str, f32)> = None;

    for user in &database.users {
        if user.embedding.len() != 512 {
            continue;
        }

        if let Some(score) = cosine_similarity(
            &user.embedding,
            &embedding,
        ) {
            if !score.is_finite() {
                continue;
            }

            println!(
                "{}: {:.4}",
                user.name,
                score
            );

            if best_match
                .as_ref()
                .is_none_or(|(_, best)| score > *best)
            {
                best_match =
                    Some((&user.name, score));
            }
        }
    }

    let matching_time =
        matching_start.elapsed();

    // --------------------------------------------------
    // 14. Report recognition result
    // --------------------------------------------------
    println!("\n--- Recognition Result ---");

    match best_match {
        Some((name, score))
            if score >= DEMO_THRESHOLD =>
        {
            println!("Best match: {}", name);
            println!("Similarity: {:.4}", score);
            println!(
                "Identity match: YES (DEMO ONLY)"
            );
        }

        _ => {
            println!("Identity match: NO");
        }
    }

    // Anti-spoofing class labels and preprocessing
    // have not yet been validated against the
    // model's reference implementation.
    println!("Full authentication: NOT ENABLED");

    // --------------------------------------------------
    // 15. Performance measurements
    // --------------------------------------------------
    let total_time =
        total_start.elapsed();

    let processing_time =
        preprocessing_time
        + detection_time
        + crop_time
        + antispoof_time
        + alignment_time
        + embedding_time
        + matching_time;

    let model_loading_overhead =
        detector_load_time
        + antispoof_load_time
        + recognition_load_time;

    println!(
        "\n--- Performance Measurements ---"
    );

    println!(
        "Image preprocessing: {:?}",
        preprocessing_time
    );

    println!(
        "YOLO model loading: {:?}",
        detector_load_time
    );

    println!(
        "YOLO detection + NMS: {:?}",
        detection_time
    );

    println!(
        "Anti-spoof crop: {:?}",
        crop_time
    );

    println!(
        "MiniFASNetV2 loading: {:?}",
        antispoof_load_time
    );

    println!(
        "MiniFASNetV2 inference: {:?}",
        antispoof_time
    );

    println!(
        "Face alignment: {:?}",
        alignment_time
    );

    println!(
        "MobileFaceNet loading: {:?}",
        recognition_load_time
    );

    println!(
        "Embedding generation: {:?}",
        embedding_time
    );

    println!(
        "Database matching: {:?}",
        matching_time
    );

    println!("----------------------------------");

    println!(
        "Processing subtotal: {:?}",
        processing_time
    );

    println!(
        "Model loading overhead: {:?}",
        model_loading_overhead
    );

    println!(
        "Total elapsed: {:?}",
        total_time
    );

    Ok(())
}

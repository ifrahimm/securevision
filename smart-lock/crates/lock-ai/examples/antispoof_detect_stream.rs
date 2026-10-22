
use image::imageops::FilterType;
use image::{Rgb, RgbImage};
use lock_ai::face_antispoof::{crop_antispoof_face, predict_antispoof};
use lock_ai::face_detection::{decode_output, Detection};
use nokhwa::{
    pixel_format::RgbFormat,
    utils::{CameraIndex, RequestedFormat, RequestedFormatType},
    Camera,
};
use powerboxesrs::nms::nms_slice;
use std::io::{self, Write};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};
use tract_onnx::prelude::*;

const CROP_SCALE: f32 = 2.7; // Experimental; original scale unverified.

fn detect_face(
    img: &RgbImage,
    detector: &TypedRunnableModel,
) -> TractResult<Option<Detection>> {
    let width = img.width();
    let height = img.height();

    if width == 0 || height == 0 {
        return Err(io::Error::other("Empty image").into());
    }

    let scale = (640.0 / width as f32)
        .min(640.0 / height as f32);

    let new_w = (width as f32 * scale).round() as u32;
    let new_h = (height as f32 * scale).round() as u32;

    let resized = image::imageops::resize(
        img, new_w, new_h, FilterType::Triangle
    );

    let pad_x = (640 - new_w) / 2;
    let pad_y = (640 - new_h) / 2;

    let mut letterboxed =
        RgbImage::from_pixel(640, 640, Rgb([114, 114, 114]));

    image::imageops::replace(
        &mut letterboxed, &resized, pad_x as i64, pad_y as i64
    );

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

    let outputs = Arc::new(detector.clone())
        .run(tvec!(input.into_tensor().into()))?;

    let strides = [8.0f32, 16.0, 32.0];
    let mut candidates = Vec::<Detection>::new();

    for (output, stride) in outputs.iter().zip(strides.iter()) {
        candidates.append(&mut decode_output(output, *stride, 0.25)?);
    }

    let mut boxes = Vec::new();
    let mut scores = Vec::new();

    for d in &candidates {
        boxes.extend_from_slice(&[d.x1, d.y1, d.x2, d.y2]);
        scores.push(d.confidence as f64);
    }

    let keep = nms_slice(&boxes, &scores, 0.45, 0.25);

    println!("Faces detected: {}", keep.len());

    if keep.len() != 1 {
        return Ok(None);
    }

    let mut face = candidates[keep[0]].clone();

    // Convert from letterboxed coordinates to camera coordinates.
    face.x1 = (face.x1 - pad_x as f32) / scale;
    face.y1 = (face.y1 - pad_y as f32) / scale;
    face.x2 = (face.x2 - pad_x as f32) / scale;
    face.y2 = (face.y2 - pad_y as f32) / scale;

    Ok(Some(face))
}

fn main() -> TractResult<()> {
    println!("SecureVision - Integrated Anti-Spoofing Test");

    println!("Loading YOLOv8n-Face...");
    let detector = tract_onnx::onnx()
        .model_for_path("models/yolov8n-face.onnx")?
        .with_input_fact(0, f32::fact([1, 3, 640, 640]).into())?
        .into_optimized()?
        .into_runnable()?;

    println!("Loading MiniFASNetV2...");
    let antispoof = Arc::new(
        tract_onnx::onnx()
            .model_for_path("models/minifasnet_v2.onnx")?
            .with_input_fact(0, f32::fact([1, 3, 80, 80]).into())?
            .into_optimized()?
            .into_runnable()?
    );

    let requested = RequestedFormat::new::<RgbFormat>(
        RequestedFormatType::AbsoluteHighestResolution,
    );

    let mut camera = Camera::new(CameraIndex::Index(0), requested)?;
    camera.open_stream()?;

    println!("Warming up camera...");
    for _ in 0..10 {
        let _ = camera.frame()?;
        thread::sleep(Duration::from_millis(100));
    }

    println!("Press ENTER to capture, or type 'quit' to exit.");

    let stdin = io::stdin();
    let mut attempt = 0usize;

    loop {
        print!("\nCapture > ");
        io::stdout().flush()?;

        let mut command = String::new();
        if stdin.read_line(&mut command)? == 0
            || command.trim().eq_ignore_ascii_case("quit")
        {
            break;
        }

        attempt += 1;
        let start = Instant::now();

        // Discard potentially stale buffered frames.
        println!("Refreshing camera frames...");

        for _ in 0..15 {
            let _ = camera.frame()?;
        }

        // Capture a fresh frame.
        let frame = camera.frame()?;
        let img = frame.decode_image::<RgbFormat>()?;

        println!("Fresh frame captured.");

        img.save(format!("antispoof_full_{attempt}.png"))?;

        let Some(face) = detect_face(&img, &detector)? else {
            println!("REJECTED: Expected exactly one face.");
            continue;
        };

        let crop = match crop_antispoof_face(
            &img, &face, CROP_SCALE
        ) {
            Ok(crop) => crop,
            Err(error) => {
                println!("Crop failed: {error}");
                continue;
            }
        };

        let crop_path = format!("antispoof_crop_{attempt}.png");
        crop.save(&crop_path)?;

        let inference_start = Instant::now();
        let probabilities = predict_antispoof(&crop, &antispoof)?;
        let inference_ms =
            inference_start.elapsed().as_secs_f64() * 1000.0;

        println!("\n--- Anti-Spoofing Results ---");
        println!("Saved crop: {crop_path}");
        println!("Class 0: {:.2}%", probabilities[0] * 100.0);
        println!("Class 1: {:.2}%", probabilities[1] * 100.0);
        println!("Class 2: {:.2}%", probabilities[2] * 100.0);

        println!("Class labels: UNVERIFIED");
        println!("Liveness: UNDETERMINED (TEST ONLY)");
        println!("Inference: {:.2} ms", inference_ms);
        println!(
            "Total: {:.2} ms",
            start.elapsed().as_secs_f64() * 1000.0
        );
    }

    camera.stop_stream()?;
    Ok(())
}

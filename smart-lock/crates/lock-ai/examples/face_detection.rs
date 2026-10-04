use image::imageops::FilterType;
use image::{Rgb, RgbImage};

use imageproc::drawing::draw_hollow_rect_mut;
use imageproc::rect::Rect;

use powerboxesrs::nms::nms_slice;

use tract_onnx::prelude::*;

use lock_ai::face_detection::{Detection, decode_output};

// --------------------------------------------------
// Draw rectangle on original image
// --------------------------------------------------
fn draw_rectangle(image: &mut RgbImage, x1: u32, y1: u32, x2: u32, y2: u32) {
    let color = Rgb([255, 0, 0]);

    for thickness in 0..4u32 {
        let left = x1.saturating_add(thickness);
        let top = y1.saturating_add(thickness);
        let right = x2.saturating_sub(thickness);
        let bottom = y2.saturating_sub(thickness);

        if right > left && bottom > top {
            let rect =
                Rect::at(left as i32, top as i32).of_size(right - left + 1, bottom - top + 1);

            draw_hollow_rect_mut(image, rect, color);
        }
    }
}
// ==================================================
// MAIN
// ==================================================

fn main() -> TractResult<()> {
    println!("SecureVision - Visual Face Detection");

    // --------------------------------------------------
    // 1. Load original camera image
    // --------------------------------------------------
    let img = image::open("securevision_test.jpg")?.to_rgb8();

    let original_width = img.width();
    let original_height = img.height();

    println!("Original image: {} x {}", original_width, original_height);

    // --------------------------------------------------
    // 2. Letterbox to 640 x 640
    // --------------------------------------------------
    let target_width = 640u32;
    let target_height = 640u32;

    let scale = f32::min(
        target_width as f32 / original_width as f32,
        target_height as f32 / original_height as f32,
    );

    let new_width = (original_width as f32 * scale).round() as u32;

    let new_height = (original_height as f32 * scale).round() as u32;

    let resized = image::imageops::resize(&img, new_width, new_height, FilterType::Triangle);

    let mut letterboxed = RgbImage::from_pixel(target_width, target_height, Rgb([114, 114, 114]));

    let pad_x = (target_width - new_width) / 2;

    let pad_y = (target_height - new_height) / 2;

    image::imageops::replace(&mut letterboxed, &resized, pad_x as i64, pad_y as i64);

    println!("Resized: {} x {}", new_width, new_height);

    println!("Padding: x={}, y={}", pad_x, pad_y);

    // --------------------------------------------------
    // 3. Create [1,3,640,640] input tensor
    // --------------------------------------------------
    let mut input = tract_ndarray::Array4::<f32>::zeros((1, 3, 640, 640));

    for y in 0..640 {
        for x in 0..640 {
            let pixel = letterboxed.get_pixel(x, y);

            input[[0, 0, y as usize, x as usize]] = pixel[0] as f32 / 255.0;

            input[[0, 1, y as usize, x as usize]] = pixel[1] as f32 / 255.0;

            input[[0, 2, y as usize, x as usize]] = pixel[2] as f32 / 255.0;
        }
    }

    println!("Input tensor: {:?}", input.shape());

    // --------------------------------------------------
    // 4. Load model
    // --------------------------------------------------
    println!("Loading YOLOv8n-Face...");

    let model = tract_onnx::onnx()
        .model_for_path("models/yolov8n-face.onnx")?
        .with_input_fact(0, f32::fact([1, 3, 640, 640]).into())?
        .into_optimized()?
        .into_runnable()?;

    println!("Model ready.");

    // --------------------------------------------------
    // 5. Run inference
    // --------------------------------------------------
    println!("Running inference...");

    let outputs = model.run(tvec!(input.into_tensor().into()))?;

    println!("Inference completed!");

    // --------------------------------------------------
    // 6. Decode outputs
    // --------------------------------------------------
    let strides = [8.0f32, 16.0f32, 32.0f32];

    let mut candidates = Vec::<Detection>::new();

    for (output, stride) in outputs.iter().zip(strides.iter()) {
        let mut decoded = decode_output(output, *stride, 0.25)?;

        candidates.append(&mut decoded);
    }

    println!("Candidates before NMS: {}", candidates.len());

    // --------------------------------------------------
    // 7. NMS
    // --------------------------------------------------
    let mut boxes = Vec::with_capacity(candidates.len() * 4);
    let mut scores = Vec::with_capacity(candidates.len());

    for detection in &candidates {
        boxes.extend_from_slice(&[detection.x1, detection.y1, detection.x2, detection.y2]);

        scores.push(detection.confidence as f64);
    }

    let keep_indices = nms_slice(&boxes, &scores, 0.45, 0.25);

    let final_detections: Vec<Detection> = keep_indices
        .into_iter()
        .map(|index| candidates[index].clone())
        .collect();

    println!("Faces detected after NMS: {}", final_detections.len());

    // --------------------------------------------------
    // 8. Copy original image for drawing
    // --------------------------------------------------
    let mut result_image = img.clone();

    // --------------------------------------------------
    // 9. Convert model coordinates back to
    //    original camera coordinates
    // --------------------------------------------------
    for (i, detection) in final_detections.iter().enumerate() {
        let original_x1 = (detection.x1 - pad_x as f32) / scale;

        let original_y1 = (detection.y1 - pad_y as f32) / scale;

        let original_x2 = (detection.x2 - pad_x as f32) / scale;

        let original_y2 = (detection.y2 - pad_y as f32) / scale;

        // Clamp coordinates to image boundaries
        let x1 = original_x1.clamp(0.0, (original_width - 1) as f32) as u32;

        let y1 = original_y1.clamp(0.0, (original_height - 1) as f32) as u32;

        let x2 = original_x2.clamp(0.0, (original_width - 1) as f32) as u32;

        let y2 = original_y2.clamp(0.0, (original_height - 1) as f32) as u32;

        println!();

        println!("Face {}:", i + 1);

        println!("  Confidence: {:.1}%", detection.confidence * 100.0);

        println!("  Original image box: ({}, {}) -> ({}, {})", x1, y1, x2, y2);

        draw_rectangle(&mut result_image, x1, y1, x2, y2);
    }

    // --------------------------------------------------
    // 10. Save result
    // --------------------------------------------------
    result_image.save("detected_face.jpg")?;

    println!();
    println!("Success!");
    println!("Saved result as detected_face.jpg");

    Ok(())
}

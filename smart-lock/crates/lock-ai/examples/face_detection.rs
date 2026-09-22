use image::{Rgb, RgbImage};
use image::imageops::FilterType;
use tract_onnx::prelude::*;

#[derive(Debug, Clone)]
struct Detection {
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
    confidence: f32,
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

// --------------------------------------------------
// Decode one YOLO output tensor
// --------------------------------------------------
fn decode_output(
    output: &TValue,
    stride: f32,
    confidence_threshold: f32,
) -> TractResult<Vec<Detection>> {
    let array = output.to_plain_array_view::<f32>()?;
    let shape = array.shape();

    let height = shape[2];
    let width = shape[3];

    let mut detections = Vec::new();

    for y in 0..height {
        for x in 0..width {
            // Channel 64 = face confidence
            let raw_confidence = array[[0, 64, y, x]];
            let confidence = sigmoid(raw_confidence);

            if confidence < confidence_threshold {
                continue;
            }

            // Channels 0-63 = DFL bounding box
            let mut distances = [0.0f32; 4];

            for side in 0..4 {
                let start_channel = side * 16;

                // Stable softmax
                let mut max_value = f32::NEG_INFINITY;

                for bin in 0..16 {
                    let value =
                        array[[0, start_channel + bin, y, x]];

                    if value > max_value {
                        max_value = value;
                    }
                }

                let mut exp_values = [0.0f32; 16];
                let mut sum = 0.0;

                for bin in 0..16 {
                    let value =
                        array[[0, start_channel + bin, y, x]];

                    let e = (value - max_value).exp();

                    exp_values[bin] = e;
                    sum += e;
                }

                let mut distance = 0.0;

                for bin in 0..16 {
                    let probability =
                        exp_values[bin] / sum;

                    distance +=
                        probability * bin as f32;
                }

                distances[side] = distance;
            }

            let grid_x = x as f32 + 0.5;
            let grid_y = y as f32 + 0.5;

            let x1 =
                (grid_x - distances[0]) * stride;

            let y1 =
                (grid_y - distances[1]) * stride;

            let x2 =
                (grid_x + distances[2]) * stride;

            let y2 =
                (grid_y + distances[3]) * stride;

            detections.push(Detection {
                x1,
                y1,
                x2,
                y2,
                confidence,
            });
        }
    }

    Ok(detections)
}

// --------------------------------------------------
// IoU
// --------------------------------------------------
fn calculate_iou(
    a: &Detection,
    b: &Detection,
) -> f32 {
    let xx1 = a.x1.max(b.x1);
    let yy1 = a.y1.max(b.y1);

    let xx2 = a.x2.min(b.x2);
    let yy2 = a.y2.min(b.y2);

    let width =
        (xx2 - xx1).max(0.0);

    let height =
        (yy2 - yy1).max(0.0);

    let intersection =
        width * height;

    let area_a =
        (a.x2 - a.x1).max(0.0)
        * (a.y2 - a.y1).max(0.0);

    let area_b =
        (b.x2 - b.x1).max(0.0)
        * (b.y2 - b.y1).max(0.0);

    let union =
        area_a + area_b - intersection;

    if union <= 0.0 {
        return 0.0;
    }

    intersection / union
}

// --------------------------------------------------
// Non-Maximum Suppression
// --------------------------------------------------
fn nms(
    mut detections: Vec<Detection>,
    iou_threshold: f32,
) -> Vec<Detection> {
    detections.sort_by(|a, b| {
        b.confidence
            .partial_cmp(&a.confidence)
            .unwrap()
    });

    let mut kept = Vec::new();

    while !detections.is_empty() {
        let best = detections.remove(0);

        detections.retain(|candidate| {
            calculate_iou(
                &best,
                candidate,
            ) <= iou_threshold
        });

        kept.push(best);
    }

    kept
}

// --------------------------------------------------
// Draw rectangle on original image
// --------------------------------------------------
fn draw_rectangle(
    image: &mut RgbImage,
    x1: u32,
    y1: u32,
    x2: u32,
    y2: u32,
) {
    // Red box
    let color = Rgb([255, 0, 0]);

    // Make line 4 pixels thick
    for thickness in 0..4u32 {
        let left = x1.saturating_add(thickness);
        let top = y1.saturating_add(thickness);

        let right = x2.saturating_sub(thickness);
        let bottom = y2.saturating_sub(thickness);

        // Top and bottom
        for x in left..=right {
            if top < image.height() {
                image.put_pixel(x, top, color);
            }

            if bottom < image.height() {
                image.put_pixel(x, bottom, color);
            }
        }

        // Left and right
        for y in top..=bottom {
            if left < image.width() {
                image.put_pixel(left, y, color);
            }

            if right < image.width() {
                image.put_pixel(right, y, color);
            }
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
    let img =
        image::open("securevision_test.jpg")?
            .to_rgb8();

    let original_width = img.width();
    let original_height = img.height();

    println!(
        "Original image: {} x {}",
        original_width,
        original_height
    );

    // --------------------------------------------------
    // 2. Letterbox to 640 x 640
    // --------------------------------------------------
    let target_width = 640u32;
    let target_height = 640u32;

    let scale = f32::min(
        target_width as f32
            / original_width as f32,
        target_height as f32
            / original_height as f32,
    );

    let new_width =
        (original_width as f32 * scale)
            .round() as u32;

    let new_height =
        (original_height as f32 * scale)
            .round() as u32;

    let resized =
        image::imageops::resize(
            &img,
            new_width,
            new_height,
            FilterType::Triangle,
        );

    let mut letterboxed =
        RgbImage::from_pixel(
            target_width,
            target_height,
            Rgb([114, 114, 114]),
        );

    let pad_x =
        (target_width - new_width) / 2;

    let pad_y =
        (target_height - new_height) / 2;

    image::imageops::replace(
        &mut letterboxed,
        &resized,
        pad_x as i64,
        pad_y as i64,
    );

    println!(
        "Resized: {} x {}",
        new_width,
        new_height
    );

    println!(
        "Padding: x={}, y={}",
        pad_x,
        pad_y
    );

    // --------------------------------------------------
    // 3. Create [1,3,640,640] input tensor
    // --------------------------------------------------
    let mut input =
        tract_ndarray::Array4::<f32>::zeros(
            (1, 3, 640, 640)
        );

    for y in 0..640 {
        for x in 0..640 {
            let pixel =
                letterboxed.get_pixel(x, y);

            input[[0, 0, y as usize, x as usize]] =
                pixel[0] as f32 / 255.0;

            input[[0, 1, y as usize, x as usize]] =
                pixel[1] as f32 / 255.0;

            input[[0, 2, y as usize, x as usize]] =
                pixel[2] as f32 / 255.0;
        }
    }

    println!(
        "Input tensor: {:?}",
        input.shape()
    );

    // --------------------------------------------------
    // 4. Load model
    // --------------------------------------------------
    println!("Loading YOLOv8n-Face...");

    let model =
        tract_onnx::onnx()
            .model_for_path(
                "models/yolov8n-face.onnx"
            )?
            .with_input_fact(
                0,
                f32::fact(
                    [1, 3, 640, 640]
                ).into(),
            )?
            .into_optimized()?
            .into_runnable()?;

    println!("Model ready.");

    // --------------------------------------------------
    // 5. Run inference
    // --------------------------------------------------
    println!("Running inference...");

    let outputs =
        model.run(
            tvec!(
                input.into_tensor().into()
            )
        )?;

    println!("Inference completed!");

    // --------------------------------------------------
    // 6. Decode outputs
    // --------------------------------------------------
    let strides =
        [8.0f32, 16.0f32, 32.0f32];

    let mut candidates =
        Vec::<Detection>::new();

    for (output, stride) in
        outputs.iter().zip(strides.iter())
    {
        let mut decoded =
            decode_output(
                output,
                *stride,
                0.25,
            )?;

        candidates.append(
            &mut decoded
        );
    }

    println!(
        "Candidates before NMS: {}",
        candidates.len()
    );

    // --------------------------------------------------
    // 7. NMS
    // --------------------------------------------------
    let final_detections =
        nms(candidates, 0.45);

    println!(
        "Faces detected after NMS: {}",
        final_detections.len()
    );

    // --------------------------------------------------
    // 8. Copy original image for drawing
    // --------------------------------------------------
    let mut result_image = img.clone();

    // --------------------------------------------------
    // 9. Convert model coordinates back to
    //    original camera coordinates
    // --------------------------------------------------
    for (i, detection) in
        final_detections.iter().enumerate()
    {
        let original_x1 =
            (detection.x1 - pad_x as f32)
                / scale;

        let original_y1 =
            (detection.y1 - pad_y as f32)
                / scale;

        let original_x2 =
            (detection.x2 - pad_x as f32)
                / scale;

        let original_y2 =
            (detection.y2 - pad_y as f32)
                / scale;

        // Clamp coordinates to image boundaries
        let x1 =
            original_x1
                .clamp(
                    0.0,
                    (original_width - 1) as f32,
                ) as u32;

        let y1 =
            original_y1
                .clamp(
                    0.0,
                    (original_height - 1) as f32,
                ) as u32;

        let x2 =
            original_x2
                .clamp(
                    0.0,
                    (original_width - 1) as f32,
                ) as u32;

        let y2 =
            original_y2
                .clamp(
                    0.0,
                    (original_height - 1) as f32,
                ) as u32;

        println!();

        println!(
            "Face {}:",
            i + 1
        );

        println!(
            "  Confidence: {:.1}%",
            detection.confidence * 100.0
        );

        println!(
            "  Original image box: ({}, {}) -> ({}, {})",
            x1,
            y1,
            x2,
            y2
        );

        draw_rectangle(
            &mut result_image,
            x1,
            y1,
            x2,
            y2,
        );
    }

    // --------------------------------------------------
    // 10. Save result
    // --------------------------------------------------
    result_image.save(
        "detected_face.jpg"
    )?;

    println!();
    println!("Success!");
    println!("Saved result as detected_face.jpg");

    Ok(())
}
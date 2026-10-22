
use image::{imageops::FilterType, RgbImage};
use tract_onnx::prelude::*;

use crate::face_detection::Detection;
use std::sync::Arc;

/// Crop an expanded region around a detected face.
///
/// The expanded crop includes surrounding context that
/// can be useful for presentation-attack detection.
///
/// This version clamps the crop to image boundaries.
pub fn crop_antispoof_face(
    image: &RgbImage,
    detection: &Detection,
    scale: f32,
) -> Result<RgbImage, String> {
    if image.width() == 0 || image.height() == 0 {
        return Err("Empty input image".into());
    }

    if !scale.is_finite() || scale <= 0.0 {
        return Err("Invalid crop scale".into());
    }

    let coords = [
        detection.x1,
        detection.y1,
        detection.x2,
        detection.y2,
    ];

    if coords.iter().any(|v| !v.is_finite()) {
        return Err("Invalid face coordinates".into());
    }

    let face_width = detection.x2 - detection.x1;
    let face_height = detection.y2 - detection.y1;

    if face_width <= 0.0 || face_height <= 0.0 {
        return Err("Invalid face bounding box".into());
    }

    let center_x = (detection.x1 + detection.x2) / 2.0;
    let center_y = (detection.y1 + detection.y2) / 2.0;

    let crop_width = face_width * scale;
    let crop_height = face_height * scale;

    let x1 = (center_x - crop_width / 2.0)
        .floor()
        .clamp(0.0, image.width() as f32) as u32;

    let y1 = (center_y - crop_height / 2.0)
        .floor()
        .clamp(0.0, image.height() as f32) as u32;

    let x2 = (center_x + crop_width / 2.0)
        .ceil()
        .clamp(0.0, image.width() as f32) as u32;

    let y2 = (center_y + crop_height / 2.0)
        .ceil()
        .clamp(0.0, image.height() as f32) as u32;

    if x2 <= x1 || y2 <= y1 {
        return Err("Crop lies outside image bounds".into());
    }

    let cropped = image::imageops::crop_imm(
        image,
        x1,
        y1,
        x2 - x1,
        y2 - y1,
    )
    .to_image();

    Ok(image::imageops::resize(
        &cropped,
        80,
        80,
        FilterType::Triangle,
    ))
}

/// Run MiniFASNetV2 inference on an 80x80 RGB crop.
///
/// Converts RGB to BGR and uses raw 0-255 pixel values.
///
/// Returns softmax probabilities for three output classes.
///
/// IMPORTANT:
/// - This preprocessing is experimental.
/// - Class labels have not yet been verified.
/// - Results must not directly authorize lock actuation.
pub fn predict_antispoof(
    crop: &RgbImage,
    model: &Arc<TypedRunnableModel>,
) -> TractResult<[f32; 3]> {
    if crop.width() != 80 || crop.height() != 80 {
        return Err(std::io::Error::other(
            "Anti-spoof input must be 80x80",
        )
        .into());
    }

    // Create NCHW tensor: [batch, channels, height, width].
    let mut input =
        tract_ndarray::Array4::<f32>::zeros((1, 3, 80, 80));

    for y in 0..80 {
        for x in 0..80 {
            let pixel = crop.get_pixel(x, y);

            // BGR channel order, raw 0-255 values.
            input[[0, 0, y as usize, x as usize]] =
                pixel[2] as f32;

            input[[0, 1, y as usize, x as usize]] =
                pixel[1] as f32;

            input[[0, 2, y as usize, x as usize]] =
                pixel[0] as f32;
        }
    }

    // Run ONNX inference.
    let outputs = model.run(
        tvec!(input.into_tensor().into())
    )?;

    // Extract three output logits.
    let scores = outputs[0]
        .to_plain_array_view::<f32>()?;

    let values: Vec<f32> =
        scores.iter().copied().collect();

    if values.len() != 3 {
        return Err(std::io::Error::other(
            "Expected three anti-spoofing scores",
        )
        .into());
    }

    if values.iter().any(|v| !v.is_finite()) {
        return Err(std::io::Error::other(
            "Non-finite anti-spoofing scores",
        )
        .into());
    }

    // Numerically stable softmax.
    let max_score = values
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);

    let exps: Vec<f32> = values
        .iter()
        .map(|v| (v - max_score).exp())
        .collect();

    let sum: f32 = exps.iter().sum();

    if !sum.is_finite() || sum <= 0.0 {
        return Err(std::io::Error::other(
            "Invalid anti-spoofing probability sum",
        )
        .into());
    }

    Ok([
        exps[0] / sum,
        exps[1] / sum,
        exps[2] / sum,
    ])
}


/// Standard ArcFace-style reference landmarks
/// for a 112 x 112 aligned face.
///
/// Order: left eye, right eye, nose,
/// left mouth corner, right mouth corner.

pub const REFERENCE_LANDMARKS: [(f32, f32); 5] = [
    (38.2946, 51.6963),
    (73.5318, 51.5014),
    (56.0252, 71.7366),
    (41.5493, 92.3655),
    (70.7299, 92.2041),
];


use image::{Rgb, RgbImage};

/// Align a face to 112x112 using five facial landmarks.
///
/// Fits a similarity transformation (rotation, uniform scale,
/// and translation) from the detected landmarks to the
/// standard recognition reference landmarks.
pub fn align_face(
    image: &RgbImage,
    landmarks: &[(f32, f32); 5],
) -> Result<RgbImage, String> {
    const SIZE: u32 = 112;
    const N: f32 = 5.0;

    // Compute landmark centroids.
    let mut src_mean = (0.0f32, 0.0f32);
    let mut dst_mean = (0.0f32, 0.0f32);

    for i in 0..5 {
        src_mean.0 += landmarks[i].0;
        src_mean.1 += landmarks[i].1;
        dst_mean.0 += REFERENCE_LANDMARKS[i].0;
        dst_mean.1 += REFERENCE_LANDMARKS[i].1;
    }

    src_mean.0 /= N;
    src_mean.1 /= N;
    dst_mean.0 /= N;
    dst_mean.1 /= N;

    // Fit destination = [a -b; b a] * source + translation.
    let mut denominator = 0.0f32;
    let mut numerator_a = 0.0f32;
    let mut numerator_b = 0.0f32;

    for i in 0..5 {
        let sx = landmarks[i].0 - src_mean.0;
        let sy = landmarks[i].1 - src_mean.1;

        let dx = REFERENCE_LANDMARKS[i].0 - dst_mean.0;
        let dy = REFERENCE_LANDMARKS[i].1 - dst_mean.1;

        denominator += sx * sx + sy * sy;
        numerator_a += sx * dx + sy * dy;
        numerator_b += sx * dy - sy * dx;
    }

    if !denominator.is_finite() || denominator < 1e-6 {
        return Err("Invalid or degenerate facial landmarks".into());
    }

    let a = numerator_a / denominator;
    let b = numerator_b / denominator;

    let tx = dst_mean.0 - a * src_mean.0 + b * src_mean.1;
    let ty = dst_mean.1 - b * src_mean.0 - a * src_mean.1;

    let determinant = a * a + b * b;

    if !determinant.is_finite() || determinant < 1e-10 {
        return Err("Invalid face alignment transformation".into());
    }

    let mut aligned = RgbImage::new(SIZE, SIZE);

    // Inverse mapping: each output pixel samples the original.
    for y in 0..SIZE {
        for x in 0..SIZE {
            let dx = x as f32 - tx;
            let dy = y as f32 - ty;

            let src_x = (a * dx + b * dy) / determinant;
            let src_y = (-b * dx + a * dy) / determinant;

            // Nearest-neighbor sampling for this first prototype.
            let ix = src_x.round() as i32;
            let iy = src_y.round() as i32;

            let pixel = if ix >= 0
                && iy >= 0
                && (ix as u32) < image.width()
                && (iy as u32) < image.height()
            {
                *image.get_pixel(ix as u32, iy as u32)
            } else {
                Rgb([0, 0, 0])
            };

            aligned.put_pixel(x, y, pixel);
        }
    }

    Ok(aligned)
}

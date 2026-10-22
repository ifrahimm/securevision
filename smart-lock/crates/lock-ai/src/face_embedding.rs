use image::RgbImage;
use tract_onnx::prelude::*;

// Existing file-based interface.
// Used by enroll_face.rs and identify_face.rs.
pub fn generate_embedding(
    image_path: &str,
    model: &TypedRunnableModel,
) -> TractResult<Vec<f32>> {
    let img = image::open(image_path)?.to_rgb8();
    generate_embedding_from_image(&img, model)
}

// New in-memory interface.
// Used by the integrated recognition pipeline.
pub fn generate_embedding_from_image(
    img: &RgbImage,
    model: &TypedRunnableModel,
) -> TractResult<Vec<f32>> {
    if img.dimensions() != (112, 112) {
        return Err(std::io::Error::other(
            "Expected a 112x112 aligned face"
        ).into());
    }

    let mut input =
        tract_ndarray::Array4::<f32>::zeros((1, 3, 112, 112));

    for y in 0..112 {
        for x in 0..112 {
            let pixel = img.get_pixel(x, y);

            for c in 0..3 {
                input[[0, c, y as usize, x as usize]] =
                    (pixel[c] as f32 - 127.5) / 127.5;
            }
        }
    }

    let outputs = std::sync::Arc::new(model.clone())
        .run(tvec!(input.into_tensor().into()))?;

    let embedding: Vec<f32> = outputs[0]
        .to_plain_array_view::<f32>()?
        .iter()
        .copied()
        .collect();

    Ok(embedding)
}
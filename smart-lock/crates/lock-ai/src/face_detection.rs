use tract_onnx::prelude::*;

#[derive(Debug, Clone)]
pub struct Detection {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
    pub confidence: f32,
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

pub fn decode_output(
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
                    let value = array[[0, start_channel + bin, y, x]];

                    if value > max_value {
                        max_value = value;
                    }
                }

                let mut exp_values = [0.0f32; 16];
                let mut sum = 0.0;

                for bin in 0..16 {
                    let value = array[[0, start_channel + bin, y, x]];
                    let e = (value - max_value).exp();

                    exp_values[bin] = e;
                    sum += e;
                }

                let mut distance = 0.0;

                for bin in 0..16 {
                    let probability = exp_values[bin] / sum;
                    distance += probability * bin as f32;
                }

                distances[side] = distance;
            }

            let grid_x = x as f32 + 0.5;
            let grid_y = y as f32 + 0.5;

            let x1 = (grid_x - distances[0]) * stride;
            let y1 = (grid_y - distances[1]) * stride;
            let x2 = (grid_x + distances[2]) * stride;
            let y2 = (grid_y + distances[3]) * stride;

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

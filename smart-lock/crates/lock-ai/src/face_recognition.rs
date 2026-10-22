
/// Calculate cosine similarity between two face embeddings.
///
/// Returns a value between -1.0 and 1.0 for valid embeddings.
/// Higher values indicate greater similarity.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> Option<f32> {
    if a.is_empty() || a.len() != b.len() {
        return None;
    }

    let mut dot = 0.0f64;
    let mut norm_a = 0.0f64;
    let mut norm_b = 0.0f64;

    for (&x, &y) in a.iter().zip(b.iter()) {
        if !x.is_finite() || !y.is_finite() {
            return None;
        }

        dot += (x as f64) * (y as f64);
        norm_a += (x as f64) * (x as f64);
        norm_b += (y as f64) * (y as f64);
    }

    if norm_a <= 0.0 || norm_b <= 0.0 {
        return None;
    }

    Some((dot / (norm_a.sqrt() * norm_b.sqrt())) as f32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_embeddings() {
        let a = [1.0, 2.0, 3.0];
        let similarity = cosine_similarity(&a, &a).unwrap();

        assert!((similarity - 1.0).abs() < 1e-6);
    }

    #[test]
    fn orthogonal_embeddings() {
        let a = [1.0, 0.0];
        let b = [0.0, 1.0];

        assert!(cosine_similarity(&a, &b).unwrap().abs() < 1e-6);
    }

    #[test]
    fn rejects_zero_embedding() {
        assert!(cosine_similarity(&[0.0, 0.0], &[1.0, 2.0]).is_none());
    }

    #[test]
    fn rejects_mismatched_lengths() {
        assert!(cosine_similarity(&[1.0], &[1.0, 2.0]).is_none());
    }
}

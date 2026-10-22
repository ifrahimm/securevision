
#![no_std]

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FaceAuthResult {
    pub face_matched: bool,
    pub liveness_passed: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccessDecision {
    Granted,
    Denied,
}

pub fn evaluate_face_auth(result: FaceAuthResult) -> AccessDecision {
    if result.face_matched && result.liveness_passed {
        AccessDecision::Granted
    } else {
        AccessDecision::Denied
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn both_checks_pass() {
        let result = FaceAuthResult {
            face_matched: true,
            liveness_passed: true,
        };

        assert_eq!(evaluate_face_auth(result), AccessDecision::Granted);
    }

    #[test]
    fn face_match_fails() {
        let result = FaceAuthResult {
            face_matched: false,
            liveness_passed: true,
        };

        assert_eq!(evaluate_face_auth(result), AccessDecision::Denied);
    }

    #[test]
    fn liveness_fails() {
        let result = FaceAuthResult {
            face_matched: true,
            liveness_passed: false,
        };

        assert_eq!(evaluate_face_auth(result), AccessDecision::Denied);
    }

    #[test]
    fn both_checks_fail() {
        let result = FaceAuthResult {
            face_matched: false,
            liveness_passed: false,
        };

        assert_eq!(evaluate_face_auth(result), AccessDecision::Denied);
    }
}

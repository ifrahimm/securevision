
use lock_core::{evaluate_face_auth, AccessDecision, FaceAuthResult};
use lock_protocol::Message;

fn main() {
    println!("SecureVision Pi Controller");
    println!("Starting authentication...\n");

    // Simulated results from our future AI system.
    let face_result = FaceAuthResult {
        face_matched: true,
        liveness_passed: false,
    };

    // Evaluate authentication using lock-core.
    let decision = evaluate_face_auth(face_result);

    // Convert the decision into a protocol message.
    let message = match decision {
        AccessDecision::Granted => Message::AuthSuccess,
        AccessDecision::Denied => Message::AuthFailure,
    };

    println!("Authentication decision: {:?}", decision);
    println!("Protocol message: {:?}", message);
    println!("Encoded byte: 0x{:02X}", message.to_byte());
}

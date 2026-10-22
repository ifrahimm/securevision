// Rust usually uses std (standard lib) but ESP32 firmware runs w/o traditional firmware
#![no_std]

// Messages shared between Raspberry Pi 5 and ESP32-S3.
/* Useful Traits:
    Debug	    Allows printing with {:?}
    Clone	    Allows explicit duplication using .clone()
    Copy	    Allows inexpensive implicit copying
    PartialEq	Allows equality comparisons using ==
    Eq	        Indicates equality follows the full equivalence rules
*/
#[derive(Debug, Clone, Copy, PartialEq, Eq)]

// public enumeration
pub enum Message {
    AuthSuccess,
    AuthFailure,
    UnlockRequest,
    LockRequest,
    Ack,
    Error,
}

// to_byte: converts message --> byte
// from_byte: converts byte --> message
impl Message {
    // Convert a message into a single byte for UART.
    pub const fn to_byte(self) -> u8 {
        // like a switch statement
        match self {
            Message::AuthSuccess => 0x01,
            Message::AuthFailure => 0x02,
            Message::UnlockRequest => 0x03,
            Message::LockRequest => 0x04,
            Message::Ack => 0x05,
            Message::Error => 0x06,
        }
    }

    // Convert a received UART byte back into a message.
    pub const fn from_byte(byte: u8) -> Option<Self> {
        // like a switch statement
        match byte {
            0x01 => Some(Message::AuthSuccess),
            0x02 => Some(Message::AuthFailure),
            0x03 => Some(Message::UnlockRequest),
            0x04 => Some(Message::LockRequest),
            0x05 => Some(Message::Ack),
            0x06 => Some(Message::Error),
            _ => None,
        }
    }
}

// tells Rust to compile this module when building tests
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn message_round_trip() {
        let message = Message::UnlockRequest;
        let encoded = message.to_byte();
        let decoded = Message::from_byte(encoded);

        assert_eq!(decoded, Some(message));
    }

    #[test]
    fn invalid_message_is_rejected() {
        assert_eq!(Message::from_byte(0xFF), None);
    }
}

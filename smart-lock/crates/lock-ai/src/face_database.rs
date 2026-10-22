use serde::{Deserialize, Serialize};
use std::fs;
use std::io;
use std::path::Path;

#[derive(Debug, Serialize, Deserialize)]
pub struct EnrolledUser {
    pub id: String,
    pub name: String,
    pub embedding: Vec<f32>,
}

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct FaceDatabase {
    pub users: Vec<EnrolledUser>,
}

impl FaceDatabase {
    pub fn load(path: &str) -> io::Result<Self> {
        if !Path::new(path).exists() {
            return Ok(Self::default());
        }

        let contents = fs::read_to_string(path)?;

        serde_json::from_str(&contents)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
    }

    pub fn save(&self, path: &str) -> io::Result<()> {
        let contents = serde_json::to_string_pretty(self)
            .map_err(io::Error::other)?;

        fs::write(path, contents)
    }

    pub fn enroll(
        &mut self,
        id: String,
        name: String,
        embedding: Vec<f32>,
    ) -> Result<(), String> {
        if embedding.len() != 512
            || !embedding.iter().all(|v| v.is_finite())
            || embedding.iter().all(|v| *v == 0.0)
        {
            return Err("Invalid face embedding".into());
        }

        if id.trim().is_empty() || name.trim().is_empty() {
            return Err("ID and name cannot be empty".into());
        }

        if self.users.iter().any(|user| user.id == id) {
            return Err("User ID already enrolled".into());
        }

        self.users.push(EnrolledUser {
            id,
            name,
            embedding,
        });

        Ok(())
    }
}
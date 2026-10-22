use lock_ai::face_database::FaceDatabase;
use lock_ai::face_embedding::generate_embedding;
use tract_onnx::prelude::*;

fn main() -> TractResult<()> {
    let args: Vec<String> = std::env::args().collect();

    if args.len() != 4 {
        eprintln!(
            "Usage: cargo run --example enroll_face -- <id> <name> <aligned_image>"
        );
        std::process::exit(1);
    }

    let id = &args[1];
    let name = &args[2];
    let image_path = &args[3];

    let database_path = "enrollments.json";

    println!("SecureVision - Face Enrollment");
    println!("User: {} ({})", name, id);

    println!("Loading MobileFaceNet...");

    let model = tract_onnx::onnx()
        .model_for_path("models/w600k_mbf.onnx")?
        .with_input_fact(
            0,
            f32::fact([1, 3, 112, 112]).into()
        )?
        .into_optimized()?
        .into_runnable()?;

    println!("Generating face embedding...");

    let embedding = generate_embedding(image_path, &model)?;

    let mut database = FaceDatabase::load(database_path)?;

    database
        .enroll(id.clone(), name.clone(), embedding)
        .map_err(std::io::Error::other)?;

    database.save(database_path)?;

    println!("Enrollment successful!");
    println!("Total enrolled users: {}", database.users.len());

    Ok(())
}
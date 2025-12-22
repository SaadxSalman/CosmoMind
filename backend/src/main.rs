use ax_um::{routing::post, Json, Router};
use crate::agents::inspection::{InspectionAgent, InspectionResult};

#[tokio::main]
async fn main() {
    let app = Router::new()
        .route("/inspect", post(handle_inspection));

    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

async fn handle_inspection(Json(payload): Json<Vec<u8>>) -> Json<InspectionResult> {
    let agent = InspectionAgent::new("Gemma-Vision-Alpha");
    let result = agent.analyze_product(payload).await;
    Json(result)
}
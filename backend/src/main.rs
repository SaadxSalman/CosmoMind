use axum::{
    routing::{get, post},
    Json, Router,
};
use tower_http::cors::{Any, CorsLayer};
use std::net::SocketAddr;

#[tokio::main]
async fn main() {
    // Define the CORS policy
    let cors = CorsLayer::new()
        // For development, you can allow any origin. 
        // In production, use .allow_origin("http://your-domain.com".parse().unwrap())
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    let app = Router::new()
        .route("/status", get(|| async { "Factory Agents Active" }))
        .route("/inspect", post(handle_inspection))
        .layer(cors); // Apply the CORS layer to all routes

    let addr = SocketAddr::from(([127, 0, 0, 1], 3000));
    println!("Backend listening on {}", addr);
    
    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

// Simplified handler for demonstration
async fn handle_inspection(Json(_payload): Json<Vec<u8>>) -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "type": "DefectDetected",
        "confidence": 0.98,
        "detail": "Microscopic fracture detected in upper-left quadrant"
    }))
}
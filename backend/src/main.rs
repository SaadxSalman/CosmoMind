use axum::{routing::get, Router};

#[tokio::main]
async fn main() {
    let app = Router::new().route("/status", get(|| async { "Factory Agents Active" }));

    let listener = tokio::net::TcpListener::bind("127.0.0.1:3000").await.unwrap();
    println!("Backend listening on {}", listener.local_addr().unwrap());
    axum::serve(listener, app).await.unwrap();
}
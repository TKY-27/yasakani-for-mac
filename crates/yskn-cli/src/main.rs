#![forbid(unsafe_code)]
#[tokio::main(flavor = "current_thread")]
async fn main() {
    if let Err(e) = yskn::cli::main(std::env::args().skip(1).collect()).await {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

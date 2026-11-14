#![forbid(unsafe_code)]
#[tokio::main(flavor = "current_thread")]
async fn main() {
    if let Err(e) = yskn::cli::main(std::env::args().skip(1).collect()).await {
        // Paths, arguments and loader errors can hold terminal controls;
        // --raw covers chat text only, so these are always escaped.
        eprintln!("error: {}", yskn::cli::visible(&e.message));
        std::process::exit(1);
    }
}

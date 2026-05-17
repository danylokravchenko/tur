fn main() {
    println!("cargo:rerun-if-changed=kernels/rms_norm_linear.metal");

    let metal_enabled = std::env::var("CARGO_FEATURE_METAL").is_ok();
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();

    if metal_enabled && target_os == "macos" {
        compile_metal_kernels();
    }
}

fn compile_metal_kernels() {
    let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR not set");
    let src = "kernels/rms_norm_linear.metal";
    let air = format!("{out_dir}/rms_norm_linear.air");
    let lib = format!("{out_dir}/rms_norm_linear.metallib");

    let ok = std::process::Command::new("xcrun")
        .args(["-sdk", "macosx", "metal", "-O2", "-c", src, "-o", &air])
        .status()
        .expect("xcrun metal failed — install Xcode Command Line Tools")
        .success();
    assert!(ok, "Metal shader compilation failed: {src}");

    let ok = std::process::Command::new("xcrun")
        .args(["-sdk", "macosx", "metallib", &air, "-o", &lib])
        .status()
        .expect("xcrun metallib failed")
        .success();
    assert!(ok, "metallib link failed");
}

// Build the single Metal translation unit and embed protocol-owned constants.
// Non-Apple targets intentionally compile only the API-compatible unavailable
// backend, so this script performs no toolchain discovery for them.
use std::env;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn main() {
    println!("cargo:rustc-check-cfg=cfg(neo_metal_shaders)");
    println!("cargo:rerun-if-changed=shaders/goldilocks.metal");
    println!("cargo:rerun-if-changed=shaders/seeded_ajtai.metal");
    println!("cargo:rerun-if-changed=shaders/lane_commitments.metal");

    if env::var_os("CARGO_FEATURE_METAL").is_none() {
        return;
    }

    let target = env::var("TARGET").expect("Cargo sets TARGET");
    if !target.contains("apple") {
        return;
    }

    // The target, not the host, selects the SDK so cross-compiles use the
    // correct Metal standard library and deployment surface.
    let sdk = if target.contains("apple-ios-sim") || target.contains("x86_64-apple-ios") {
        "iphonesimulator"
    } else if target.contains("apple-ios") {
        "iphoneos"
    } else {
        "macosx"
    };
    let manifest = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").expect("Cargo sets CARGO_MANIFEST_DIR"));
    let out = PathBuf::from(env::var_os("OUT_DIR").expect("Cargo sets OUT_DIR"));
    // `goldilocks.metal` is the root translation unit and includes the
    // phase-specific shader files at its end.
    let source = manifest.join("shaders/goldilocks.metal");
    let air = out.join("goldilocks.air");
    let library = out.join("nightstream-metal.metallib");
    let constants = out.join("poseidon2.constants");
    write_poseidon2_constants(&constants);
    let developer_dir = developer_dir();

    let compile = run_xcrun(
        developer_dir.as_deref(),
        sdk,
        &["metal", "-std=metal3.0", "-c"],
        &[&source, Path::new("-o"), &air],
    )
    .unwrap_or_else(|error| panic!("run Metal compiler: {error}"));
    if !compile.status.success() {
        panic!(
            "compile Metal shader: {}",
            String::from_utf8_lossy(&compile.stderr).trim()
        );
    }

    let link = run_xcrun(
        developer_dir.as_deref(),
        sdk,
        &["metal"],
        &[&air, Path::new("-o"), &library],
    )
    .unwrap_or_else(|error| panic!("run Metal linker: {error}"));
    if !link.status.success() {
        panic!("link Metal library: {}", String::from_utf8_lossy(&link.stderr).trim());
    }

    println!("cargo:rustc-cfg=neo_metal_shaders");
}

fn write_poseidon2_constants(path: &Path) {
    // Serialize the canonical Rust constants instead of maintaining a second
    // shader-side copy of protocol data.
    let constants = neo_ccs::crypto::poseidon2_goldilocks::round_constants();
    let mut words = Vec::with_capacity(
        constants.initial.len() * 8 + constants.internal.len() + constants.terminal.len() * 8 + constants.diag.len(),
    );
    words.extend(constants.initial.into_iter().flatten());
    words.extend(constants.internal);
    words.extend(constants.terminal.into_iter().flatten());
    words.extend(constants.diag);

    let mut bytes = Vec::with_capacity(words.len() * size_of::<u64>());
    for word in words {
        bytes.extend_from_slice(&word.to_le_bytes());
    }
    std::fs::write(path, bytes).expect("write canonical Poseidon2 constants");
}

fn developer_dir() -> Option<PathBuf> {
    // CommandLineTools lacks the Metal compiler. Honor a full selected Xcode,
    // then try the standard application path without mutating global selection.
    let selected = Command::new("xcode-select").arg("-p").output().ok()?;
    if selected.status.success() {
        let path = PathBuf::from(String::from_utf8_lossy(&selected.stdout).trim());
        if path.join("Platforms").is_dir() {
            return Some(path);
        }
    }
    let standard = PathBuf::from("/Applications/Xcode.app/Contents/Developer");
    standard.is_dir().then_some(standard)
}

fn run_xcrun(developer_dir: Option<&Path>, sdk: &str, prefix: &[&str], paths: &[&Path]) -> std::io::Result<Output> {
    let mut command = Command::new("xcrun");
    if let Some(developer_dir) = developer_dir {
        command.env("DEVELOPER_DIR", developer_dir);
    }
    command.arg("--sdk").arg(sdk).args(prefix);
    for path in paths {
        command.arg(path);
    }
    command.output()
}

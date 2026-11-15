use std::env;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use neo_math::{Fq, D};
use p3_field::{extension::BinomiallyExtendable, PrimeField64};

fn main() {
    println!("cargo:rerun-if-changed=shaders/nightstream.metal");
    println!("cargo:rerun-if-changed=shaders/joint.metal");
    println!("cargo:rerun-if-changed=shaders/assignments.metal");
    println!("cargo:rerun-if-changed=shaders/carried.metal");
    println!("cargo:rerun-if-changed=shaders/relation.metal");
    println!("cargo:rerun-if-changed=shaders/production_commitment.metal");
    println!("cargo:rerun-if-changed=shaders/dec_forms.metal");
    println!("cargo:rerun-if-changed=shaders/decomposition.metal");

    if env::var_os("CARGO_FEATURE_METAL").is_none() {
        return;
    }

    let target = env::var("TARGET").expect("Cargo sets TARGET");
    if !target.contains("apple") {
        return;
    }

    let sdk = if target.contains("apple-ios-sim") || target.contains("x86_64-apple-ios") {
        "iphonesimulator"
    } else if target.contains("apple-ios") {
        "iphoneos"
    } else {
        "macosx"
    };
    let manifest = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").expect("Cargo sets CARGO_MANIFEST_DIR"));
    let out = PathBuf::from(env::var_os("OUT_DIR").expect("Cargo sets OUT_DIR"));
    let source = manifest.join("shaders/nightstream.metal");
    let air = out.join("nightstream.air");
    let library = out.join("nightstream-metal.metallib");
    let developer_dir = developer_dir();
    // The shaders take the field and the ring from `neo-math`, so no shader
    // repeats a profile value.
    let profile = format!(
        "constant ulong FQ_MODULUS = {:#x}ul;\nconstant ulong RING_DEGREE = {D};\nconstant ulong KX_NON_RESIDUE = {};\n",
        Fq::ORDER_U64,
        <Fq as BinomiallyExtendable<2>>::W.as_canonical_u64(),
    );
    std::fs::write(out.join("profile.metal"), profile).expect("write the shader profile header");

    let compile = run_xcrun(
        developer_dir.as_deref(),
        sdk,
        &["metal", "-std=metal3.0", "-c"],
        &[Path::new("-I"), &out, &source, Path::new("-o"), &air],
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
}

fn developer_dir() -> Option<PathBuf> {
    // CommandLineTools lacks the Metal compiler.
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

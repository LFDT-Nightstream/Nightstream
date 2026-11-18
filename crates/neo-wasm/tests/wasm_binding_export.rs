use std::{fs, path::Path, process::Command};

use sha2::{Digest, Sha256};

const BINDINGS: [(&str, &str); 4] = [
    (
        "wasm-binding-micro-v1.json",
        "8e22efcf3b24efc439df4372d6cb4b30460d9f40584d9f19a56170b2808ac48b",
    ),
    (
        "wasm-binding-small-v1.json",
        "c0b63e6a3fd26ffadb76ae7aa94c850adc7f32a7c6bd96f7f41167434b3d40ed",
    ),
    (
        "wasm-binding-medium-v1.json",
        "eef9c7ad8cdbd5fe41a3f3a718c963f9e4f24d59ffa0c4f00c53c1c2f71e4e25",
    ),
    (
        "wasm-binding-large-v1.json",
        "70a6d117206da811051eacce370bdab6bba1ca1ed14f73bbd3753cb3a2c1c74d",
    ),
];

#[test]
fn binding_exports_match_their_pinned_digests() {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("wasm-binding-export");
    let _ = fs::remove_dir_all(&dir);
    let output = Command::new(env!("CARGO_BIN_EXE_export_wasm_rocq"))
        .arg("--binding")
        .arg(&dir)
        .output()
        .expect("run export_wasm_rocq");
    assert!(
        output.status.success(),
        "binding export failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    for (name, digest) in BINDINGS {
        let contents = fs::read(dir.join(name)).expect("read binding export");
        assert_eq!(format!("{:x}", Sha256::digest(&contents)), digest, "{name}");
    }
}

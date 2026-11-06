//! The embedded zstd copies are the Lean outputs, byte for byte. After a Lean
//! regeneration, `artifacts/README.md` says how to make the copies again.

use std::path::Path;

fn lean_output(name: &str) -> Vec<u8> {
    std::fs::read(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("artifacts")
            .join(name),
    )
    .unwrap()
}

#[test]
fn embedded_packages_are_the_lean_outputs() {
    assert!(
        super::reference_package() == lean_output("nightstream-fprime-stage1-poseidon2-hash-chain-v1.json").as_slice(),
        "the embedded reference package is not the Lean output"
    );
    assert!(
        super::terminal_package() == lean_output("compression-terminal-v1.json").as_slice(),
        "the embedded terminal package is not the Lean output"
    );
}

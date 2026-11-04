//! The embedded zstd copies are the Lean outputs, byte for byte. After a Lean
//! regeneration, `artifacts/README.md` says how to make the copies again.

use std::path::Path;

fn lean_output(name: &str) -> Vec<u8> {
    std::fs::read(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../formal/nightstream-fprime/artifacts")
            .join(name),
    )
    .unwrap()
}

#[test]
fn embedded_packages_are_the_lean_outputs() {
    let references = super::references();
    for (embedded, name) in [
        (references.light, "nightstream-fprime-option-b-light-v1.json"),
        (references.refresh, "nightstream-fprime-option-b-refresh-v1.json"),
        (references.finish, "nightstream-fprime-option-b-finish-v1.json"),
    ] {
        assert!(
            embedded == lean_output(name).as_slice(),
            "the embedded {name} is not the Lean output"
        );
    }
}

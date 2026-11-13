//! The embedded zstd copies are the Lean outputs, byte for byte, and each
//! hashes to its pinned identity, which `embedded` takes without a hash.
//! After a Lean regeneration, `artifacts/README.md` says how to make the
//! copies again.

use std::path::Path;

use nightstream_fprime::{load_bounded_depth_packages, BOUNDED_DEPTH_PACKAGES_IDENTITY};

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
    let [light, refresh] = super::copies();
    for (embedded, name) in [
        (light, "nightstream-fprime-bounded-depth-light-v1.json"),
        (refresh, "nightstream-fprime-bounded-depth-refresh-v1.json"),
    ] {
        assert!(
            embedded == lean_output(name),
            "the embedded {name} is not the Lean output"
        );
    }
}

#[test]
fn embedded_packages_have_their_pinned_identities() {
    let [light, refresh] = super::copies();
    let checked = load_bounded_depth_packages(&light, &refresh, BOUNDED_DEPTH_PACKAGES_IDENTITY).unwrap();
    assert_eq!(super::embedded().packages.identity(), checked.identity());
}

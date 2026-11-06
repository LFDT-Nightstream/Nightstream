//! The embedded zstd copies are the Lean outputs, byte for byte, and each
//! hashes to its pinned identity, which `embedded` takes without a hash.
//! After a Lean regeneration, `artifacts/README.md` says how to make the
//! copies again.

use std::path::Path;

use nightstream_fprime::{
    load_option_b_finish, load_option_b_packages, OPTION_B_FINISH_IDENTITY, OPTION_B_PACKAGES_IDENTITY,
};

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
    let [light, refresh, finish] = super::copies();
    for (embedded, name) in [
        (light, "nightstream-fprime-option-b-light-v1.json"),
        (refresh, "nightstream-fprime-option-b-refresh-v1.json"),
        (finish, "nightstream-fprime-option-b-finish-v1.json"),
    ] {
        assert!(
            embedded == lean_output(name),
            "the embedded {name} is not the Lean output"
        );
    }
}

#[test]
fn embedded_packages_have_their_pinned_identities() {
    let [light, refresh, finish] = super::copies();
    let checked = load_option_b_packages(&light, &refresh, OPTION_B_PACKAGES_IDENTITY).unwrap();
    load_option_b_finish(&finish, OPTION_B_FINISH_IDENTITY).unwrap();
    assert_eq!(super::embedded().packages.identity(), checked.identity());
}

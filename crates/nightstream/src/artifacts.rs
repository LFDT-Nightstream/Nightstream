//! The three Lean packages that every circuit needs, embedded as zstd copies
//! of the Lean outputs (`artifacts/README.md`), so that a dependency needs no
//! file of the repository. They are decoded once per process. Their
//! identities are the pins without a hash at run time, because the bytes are
//! part of the binary: `embedded_packages_have_their_pinned_identities`
//! checks that each copy hashes to its pin.

use std::io::Read;
use std::sync::OnceLock;

use nightstream_fprime::{
    load_pinned_option_b_finish, load_pinned_option_b_packages, OptionBPackages, StepPackage,
    OPTION_B_PACKAGES_IDENTITY,
};

/// The decoded light and refresh packages of the reference application, whose
/// application every `Circuit::compile` replaces, and the finish package.
pub(crate) struct Embedded {
    pub(crate) packages: OptionBPackages,
    pub(crate) finish: StepPackage,
}

/// The embedded packages, decoded on first use.
pub(crate) fn embedded() -> &'static Embedded {
    static EMBEDDED: OnceLock<Embedded> = OnceLock::new();
    EMBEDDED.get_or_init(|| {
        let [light, refresh, finish] = copies();
        Embedded {
            packages: load_pinned_option_b_packages(&light, &refresh, OPTION_B_PACKAGES_IDENTITY)
                .expect("the embedded step packages decode"),
            finish: load_pinned_option_b_finish(&finish).expect("the embedded finish package decodes"),
        }
    })
}

/// The unpacked light, refresh and finish packages
/// (`nightstream-fprime-option-b-{light,refresh,finish}-v1.json`).
fn copies() -> [Vec<u8>; 3] {
    [
        unpack(include_bytes!(
            "../artifacts/nightstream-fprime-option-b-light-v1.json.zst"
        )),
        unpack(include_bytes!(
            "../artifacts/nightstream-fprime-option-b-refresh-v1.json.zst"
        )),
        unpack(include_bytes!(
            "../artifacts/nightstream-fprime-option-b-finish-v1.json.zst"
        )),
    ]
}

/// The build embeds a valid zstd frame, so a failure here is a build defect
/// (test `embedded_packages_are_the_lean_outputs`).
fn unpack(mut compressed: &[u8]) -> Vec<u8> {
    let mut bytes = Vec::new();
    ruzstd::decoding::StreamingDecoder::new(&mut compressed)
        .expect("an embedded zstd frame")
        .read_to_end(&mut bytes)
        .expect("an embedded zstd frame");
    bytes
}

#[cfg(test)]
#[path = "../tests/artifacts/embedded.rs"]
mod tests;

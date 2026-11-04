//! The three Lean packages that every circuit needs, embedded as zstd copies
//! of the Lean outputs (`artifacts/README.md`), so that a dependency needs no
//! file of the repository. Each is unpacked once. `Circuit::compile` checks
//! their identity pins, so a wrong copy fails there.

use std::io::Read;
use std::sync::OnceLock;

use crate::circuit::References;

/// The light, refresh and finish packages of the reference application
/// (`nightstream-fprime-option-b-{light,refresh,finish}-v1.json`), whose
/// application every `Circuit::compile` replaces.
pub(crate) fn references() -> References<'static> {
    static LIGHT: OnceLock<Vec<u8>> = OnceLock::new();
    static REFRESH: OnceLock<Vec<u8>> = OnceLock::new();
    static FINISH: OnceLock<Vec<u8>> = OnceLock::new();
    References {
        light: LIGHT.get_or_init(|| {
            unpack(include_bytes!(
                "../artifacts/nightstream-fprime-option-b-light-v1.json.zst"
            ))
        }),
        refresh: REFRESH.get_or_init(|| {
            unpack(include_bytes!(
                "../artifacts/nightstream-fprime-option-b-refresh-v1.json.zst"
            ))
        }),
        finish: FINISH.get_or_init(|| {
            unpack(include_bytes!(
                "../artifacts/nightstream-fprime-option-b-finish-v1.json.zst"
            ))
        }),
    }
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

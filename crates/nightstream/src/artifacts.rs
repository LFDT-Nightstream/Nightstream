//! The two Lean packages that every circuit needs, embedded as zstd copies of
//! the Lean outputs (`artifacts/README.md`), so that a dependency needs no
//! Git LFS file. Each is unpacked once. Its user checks its identity pins, so
//! a wrong copy fails there.

use std::io::Read;
use std::sync::OnceLock;

/// The Stage 1 reference package, whose shared F′ verifier every
/// `Circuit::compile` keeps (`nightstream-fprime-stage1-poseidon2-hash-chain-v1.json`).
pub(crate) fn reference_package() -> &'static [u8] {
    static BYTES: OnceLock<Vec<u8>> = OnceLock::new();
    BYTES.get_or_init(|| {
        unpack(include_bytes!(
            "../artifacts/nightstream-fprime-stage1-poseidon2-hash-chain-v1.json.zst"
        ))
    })
}

/// The compression terminal package (`compression-terminal-v1.json`).
pub(crate) fn terminal_package() -> &'static [u8] {
    static BYTES: OnceLock<Vec<u8>> = OnceLock::new();
    BYTES.get_or_init(|| unpack(include_bytes!("../artifacts/compression-terminal-v1.json.zst")))
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

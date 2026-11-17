use std::io::Read;
use std::sync::OnceLock;

use nightstream_fprime::{load_pinned_bounded_depth_packages, BoundedDepthPackages, BOUNDED_DEPTH_PACKAGES_IDENTITY};

pub(crate) struct Embedded {
    pub(crate) packages: BoundedDepthPackages,
}

pub(crate) fn embedded() -> &'static Embedded {
    static EMBEDDED: OnceLock<Embedded> = OnceLock::new();
    EMBEDDED.get_or_init(|| {
        let [light, refresh] = copies();
        Embedded {
            packages: load_pinned_bounded_depth_packages(&light, &refresh, BOUNDED_DEPTH_PACKAGES_IDENTITY)
                .expect("the embedded step packages decode"),
        }
    })
}

fn copies() -> [Vec<u8>; 2] {
    [
        unpack(include_bytes!(
            "../artifacts/nightstream-fprime-bounded-depth-light-v1.json.zst"
        )),
        unpack(include_bytes!(
            "../artifacts/nightstream-fprime-bounded-depth-refresh-v1.json.zst"
        )),
    ]
}

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

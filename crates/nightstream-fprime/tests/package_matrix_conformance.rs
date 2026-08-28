use std::path::PathBuf;

#[path = "../src/bin/check_package_conformance/support.rs"]
mod support;

#[test]
fn final_rust_matrices_equal_the_lean_padded_rows_entry_for_entry() {
    let artifact_directory =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../formal/nightstream-fprime/artifacts");
    support::run(
        &artifact_directory.join("nightstream-fprime-stage1-v1.json"),
        &artifact_directory.join("nightstream-fprime-stage1-v1-expanded.json"),
        &artifact_directory.join("nightstream-fprime-stage1-piccs-parity-v1.json"),
        &artifact_directory.join("nightstream-fprime-stage1-pidec-parity-v1.json"),
        [
            12_756_407_480_944_487_176,
            17_097_603_764_386_178_571,
            11_791_428_871_054_057_896,
            14_346_937_702_828_624_285,
        ],
    );
}

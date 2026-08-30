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
            5_326_948_389_888_638_380,
            15_945_253_772_729_055_182,
            12_038_831_075_978_321_435,
            4_066_786_242_110_063_495,
        ],
    );
}

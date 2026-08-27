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
        [
            2_880_828_118_570_533_443,
            12_363_340_834_605_518_522,
            17_891_354_081_046_714_225,
            8_467_327_743_520_570_474,
        ],
    );
}

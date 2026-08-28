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
            18_090_610_635_114_842_464,
            5_494_511_358_918_718_774,
            14_026_867_434_695_270_642,
            8_861_486_951_490_451_735,
        ],
    );
}

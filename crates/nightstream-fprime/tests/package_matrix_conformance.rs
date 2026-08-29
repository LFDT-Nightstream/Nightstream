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
            3_355_019_049_079_043_662,
            4_920_201_927_044_277_974,
            5_339_237_732_450_517_664,
            894_111_819_037_169_888,
        ],
    );
}

use std::io::Read;

use super::*;

/// The driver supplies the two Lean export paths. This exercises production
/// row, witness, ownership and source-bound checks, then every matrix row.
#[test]
#[ignore = "run tools/recursive-constraint-minimizer/experiments/check_wide_physical_package.sh with the Lean export"]
fn wide_physical_package_passes_production_validation() {
    let mut paths = String::new();
    std::io::stdin().read_to_string(&mut paths).unwrap();
    let paths = paths.lines().collect::<Vec<_>>();
    assert_eq!(paths.len(), 2, "physical and matrix export paths");
    let bytes = std::fs::read(paths[0]).unwrap();
    let raw: RawPackage = serde_json::from_slice(&bytes).expect("Lean physical package");
    let package = validate_per_application_package_schema(raw, [0; 4], 8)
        .expect("wide physical package uses the production loader");
    assert_eq!(package.permutation_invocation_count(), 7_638);
    assert_eq!(package.compact_template_count(), 108);
    assert_eq!(package.compact_invocation_count(), PHI81_INVOCATIONS);
    let matrix_value: Value = serde_json::from_slice(&std::fs::read(paths[1]).unwrap()).unwrap();
    let matrix = matrix_program::MatrixProgram::decode(&matrix_value).expect("wide matrix program");
    matrix
        .validate(package.row_count())
        .expect("new source-row bounds");
    assert_eq!(matrix.row_count().unwrap(), package.ccs_relation().row_count());
    assert_eq!(package.ccs_relation().row_count(), 3_248_956);
    assert_eq!(package.ccs_relation().column_count(), 137_341_846);
    let nonzeros = matrix
        .validate_all_rows(package.ccs_relation().column_count(), &|row| package.source_row(row))
        .expect("all matrix rows use the new physical source archive");
    assert_eq!(
        nonzeros,
        [
            16_904_205,
            3_144_304,
            264_072_386,
            25_752_159,
            800_725_610,
            1_496_903_449,
            0,
            104_652,
            0,
            0,
            0,
            0,
            0,
        ]
    );
    println!(
        "wide_physical_rows={} columns={} witness_instructions={} assertions={}",
        package.row_count(),
        package.total_column_count(),
        package.witness_instruction_count(),
        package.assertion_row_count()
    );
    println!("wide_logical_nonzeros={}", nonzeros.iter().sum::<u64>());
}

const PHI81_INVOCATIONS: usize = 52_326;

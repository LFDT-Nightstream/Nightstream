use std::io::Read;

use super::*;
use p3_field::PrimeCharacteristicRing;
use rayon::prelude::*;

fn rows_hold(package: &LoadedPerApplicationPackage, values: &[i8]) -> Result<(), PackageError> {
    let rows = package.row_count();
    let workers = rayon::current_num_threads();
    let chunk = rows.div_ceil(workers);
    (0..workers).into_par_iter().try_for_each(|worker| {
        let start = (worker * chunk).min(rows);
        let end = (start + chunk).min(rows);
        package.matrix_program.visit_rows(
            package.logical_column_count(),
            start,
            end,
            &|index| package.circuit.source_row(index),
            |index, forms| {
                let mut evaluated = [Goldilocks::ZERO; 14];
                for (port, form) in forms.iter().enumerate() {
                    for entry in form.entries() {
                        match values.get(entry.column) {
                            Some(1) => evaluated[port] += entry.coefficient,
                            Some(-1) => evaluated[port] -= entry.coefficient,
                            Some(0) => (),
                            _ => return Err(PackageError::Invalid("wide logical coordinate")),
                        }
                    }
                }
                let mut residual = Goldilocks::ZERO;
                for term in package.ccs_relation().terms() {
                    let mut value = Goldilocks::from_u64(term.coefficient());
                    for (port, exponent) in term.exponents().iter().enumerate() {
                        for _ in 0..*exponent {
                            value *= evaluated[port];
                        }
                    }
                    residual += value;
                }
                if residual != Goldilocks::ZERO {
                    return Err(PackageError::UnsatisfiedAssertionRow { row: index });
                }
                Ok(())
            },
        )
    })
}

#[test]
#[ignore = "run tools/recursive-constraint-minimizer/experiments/check_wide_assignment.sh with the Lean export"]
fn wide_sealed_package_constructs_a_complete_assignment() {
    let mut paths = String::new();
    std::io::stdin().read_to_string(&mut paths).unwrap();
    let paths = paths.lines().collect::<Vec<_>>();
    assert_eq!(paths.len(), 2, "sealed package and Lean base fixture");
    let value = serde_json::from_slice(&std::fs::read(paths[0]).unwrap()).unwrap();
    // This check covers the actual sealed decoder and witness engine. The
    // identity and verifier-context pins are checked at production selection.
    let package = decode_per_application_value(value, [0; 4]).expect("wide sealed package");
    let fixture: Value = serde_json::from_slice(&std::fs::read(paths[1]).unwrap()).unwrap();
    let private: Vec<u64> = serde_json::from_value(fixture[1][2].clone()).unwrap();
    let public: Vec<u64> = serde_json::from_value(fixture[1][3].clone()).unwrap();
    let mut physical = package
        .execute_witness(&private, &public)
        .expect("wide physical witness");
    let logical = package
        .execute_logical_assignment(&physical)
        .expect("wide committed witness");
    assert_eq!(logical.len(), 137_341_846);
    let expected_public: Vec<u64> = serde_json::from_value(fixture[1][4][2].clone()).unwrap();
    for (column, expected) in expected_public.into_iter().enumerate() {
        assert_eq!(logical.value(column).unwrap(), expected, "public column {column}");
    }
    rows_hold(&package, logical.balanced_values()).expect("every wide CCS row holds");
    println!(
        "wide_assignment_rows={} coordinates={}",
        package.row_count(),
        logical.len()
    );

    let mut unused: Vec<[usize; 2]> = serde_json::from_value(fixture[2].clone()).unwrap();
    unused.push(serde_json::from_value(fixture[3].clone()).unwrap());
    for [start, count] in unused {
        for value in &mut physical.private_values[start..start + count] {
            *value = if *value == 0 { 1 } else { 0 };
        }
    }
    let reconstructed = package
        .execute_logical_assignment(&physical)
        .expect("temporary values are not read");
    assert_eq!(logical.balanced_values(), reconstructed.balanced_values());
    drop(reconstructed);
    let bit = fixture[4].as_u64().unwrap() as usize;
    physical.private_values[bit] = 1 - physical.private_values[bit];
    let altered = package
        .execute_logical_assignment(&physical)
        .expect("Boolean mutation remains encodable");
    assert!(
        rows_hold(&package, altered.balanced_values()).is_err(),
        "a changed checked sampler bit must fail"
    );
}

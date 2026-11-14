//! The fold gate of `Params::for_ccs_shape` against Lean's bound
//! (`Lifecycle.BoundedDepth.RandomOracle.FoldGate`), on the light and the
//! refresh relation of each package that the tests load: today's fixture,
//! `nebula-v1-` and `nebula-n1-`. Lean emits the production counts, the
//! census numerator, `bits` and the exact error `count/denominator`. The
//! statistical term of either fold with `Q` queries is at most
//! `(Q + 1)/2^bits` (`FoldGate.refresh_statisticalError_le_bits`,
//! `FoldGate.light_statisticalError_le_bits`). On each relation, the gate
//! must read Lean's counts, charge Lean's error, pass `bits` and refuse
//! `bits + 1`.

use neo_params::{nightstream_p64_k16::CHALLENGE_ALPHABET, NeoParams, ParamsError};
use nightstream_fprime::load_bounded_depth_packages;
use serde_json::Value;

use super::super::{Relation, RelationError};
use super::support::{artifact, packages};
use crate::nebula::{test_packages::bounded_depth_packages, Package};

fn number(value: &Value) -> u64 {
    value.as_u64().expect("a natural number")
}

fn words(value: &Value) -> Vec<u64> {
    value
        .as_array()
        .expect("little-endian words")
        .iter()
        .map(number)
        .collect()
}

#[test]
fn the_gate_charges_the_lean_error_on_each_step_relation() {
    let lean: Value =
        serde_json::from_slice(&artifact("nightstream-fprime-fold-gate-parity-v1.json")).expect("Lean fold gate file");
    let [schema, cube_variables, width, matrices, numerator, bits, count, denominator] =
        <[Value; 8]>::try_from(lean.as_array().expect("schema-1 array").clone()).expect("eight values");
    assert_eq!(number(&schema), 1, "Lean fold gate schema");
    let bits = u32::try_from(number(&bits)).expect("bits");
    let nebula = |package: Package, stem| {
        let [light, refresh] = bounded_depth_packages(stem);
        load_bounded_depth_packages(&light, &refresh, package.bounded_depth_identity()).expect("Nebula packages")
    };
    for (name, packages) in [
        ("fixture", packages()),
        ("nebula-v1-", nebula(Package::V1, "nebula-v1-")),
        ("nebula-n1-", nebula(Package::N1, "nebula-n1-")),
    ] {
        for package in [packages.light(), packages.refresh()] {
            let context = format!("{name} circuit {}", package.circuit());
            let relation = Relation::new(package, bits).unwrap_or_else(|error| panic!("{context}: {error}"));
            let structure = relation.structure();
            let summary = NeoParams::nightstream_p64_k16()
                .padded_row_security_summary_for_shape(
                    structure.domain_rows(),
                    structure.m(),
                    structure.t(),
                    structure.max_degree(),
                    CHALLENGE_ALPHABET.len() as u32,
                )
                .expect("census");
            assert_eq!(
                [
                    u64::from(summary.cube_variables),
                    u64::from(summary.verifier_degree),
                    structure.t() as u64
                ],
                [&cube_variables, &width, &matrices].map(number),
                "{context}: the gate reads Lean's counts"
            );
            assert_eq!(
                summary.field_factor,
                u128::from(number(&numerator)),
                "{context}: the census numerator is Lean's"
            );
            assert_eq!(
                (
                    summary.fold_error.count.to_u64_digits(),
                    summary.fold_error.denominator.to_u64_digits()
                ),
                (words(&count), words(&denominator)),
                "{context}: the gate charges Lean's error"
            );
            assert_eq!(
                relation.params().inner().lambda(),
                bits,
                "{context}: the gate certifies Lean's bits"
            );
            assert!(
                matches!(
                    Relation::new(package, bits + 1),
                    Err(RelationError::Parameters(ParamsError::InsufficientStatisticalSecurity { required, available }))
                        if required == bits + 1 && available == bits
                ),
                "{context}: the gate refuses 2^-{}",
                bits + 1
            );
        }
    }
}

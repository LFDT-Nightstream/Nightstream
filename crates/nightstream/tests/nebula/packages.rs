//! The bounded-depth packages that the Nebula tests compile and load: an
//! application's light and refresh step packages (`""` for today's fixture,
//! `"nebula-v1-"`, `"nebula-n1-"`). Also the comparison of an application's
//! rows with the row digest that Lean writes.

pub(crate) fn bounded_depth_packages(stem: &str) -> [Vec<u8>; 2] {
    let read = |name: String| {
        std::fs::read(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../formal/nightstream-fprime/artifacts")
                .join(name),
        )
        .expect("Lean bounded-depth package")
    };
    [
        read(format!("nightstream-fprime-bounded-depth-{stem}light-v1.json")),
        read(format!("nightstream-fprime-bounded-depth-{stem}refresh-v1.json")),
    ]
}

pub(crate) fn references(packages: &[Vec<u8>; 2]) -> crate::References<'_> {
    crate::References {
        light: &packages[0],
        refresh: &packages[1],
    }
}

/// FNV-1a over the little-endian bytes of `word`.
fn fnv(hash: u64, word: u64) -> u64 {
    word.to_le_bytes()
        .iter()
        .fold(hash, |hash, &byte| (hash ^ u64::from(byte)).wrapping_mul(0x100000001b3))
}

/// Requires `application` to have the row count and the row digest of
/// `fixture`, which Lean's emitter writes (`EmitMain.applicationDigest`):
/// each side's constant, then its terms by variable, with equal variables
/// merged and zero coefficients dropped.
pub(crate) fn assert_rows(application: &crate::application::ApplicationCircuit, fixture: &[u8]) {
    use std::{collections::BTreeMap, ops::ControlFlow};

    use neo_math::F;
    use p3_field::{PrimeCharacteristicRing, PrimeField64};

    let records = application.records();
    let mut hash = fnv(0xcbf29ce484222325, records.row_count() as u64);
    for row in 0..records.row_count() {
        let header = records.row_header(row).unwrap();
        let mut sides: [BTreeMap<usize, F>; 3] = Default::default();
        let _: ControlFlow<()> = records
            .visit_terms(row, |term| {
                *sides[term.form.index()]
                    .entry(term.variable)
                    .or_insert(F::ZERO) += F::from_u64(term.coefficient);
                Ok(ControlFlow::Continue(()))
            })
            .unwrap();
        for (side, constant) in sides.iter().zip(header.constants) {
            let terms: Vec<_> = side
                .iter()
                .filter(|(_, value)| **value != F::ZERO)
                .collect();
            hash = fnv(fnv(hash, constant), terms.len() as u64);
            for (variable, value) in terms {
                hash = fnv(fnv(hash, *variable as u64), value.as_canonical_u64());
            }
        }
    }
    let expected: serde_json::Value = serde_json::from_slice(fixture).unwrap();
    assert_eq!(Some(records.row_count() as u64), expected["rows"].as_u64(), "rows");
    assert_eq!(Some(hash), expected["digest"].as_u64(), "row digest");
}

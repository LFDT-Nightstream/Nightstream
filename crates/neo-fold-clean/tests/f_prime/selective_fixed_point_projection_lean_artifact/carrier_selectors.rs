//! Exact selector rows for the bounded fixed-point carrier.
//!
//! Owns: projection and validation of all three selector-domain rows and the
//! one selector-total row from the final thirteen-port emitter.
//!
//! Does not own: selector values, active-branch semantics, retained-row
//! coverage, CCS/CE membership, or row removal.

use std::fmt::Write as _;

use neo_fold_clean::frontends::r1cs_f_prime::{
    SelectiveEmittedRowFamily, SelectiveProjectedPort, SelectiveProjectedRowsAudit,
};
use neo_math::F;
use p3_field::PrimeCharacteristicRing;

use super::{selective_matrix_rows::write_raw_row, GeneratedLeanFile};

const GENERATED_PATH: &str =
    "formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/FPrimeSelectiveFixedPoint/Carrier270/Generated/SelectorRows.lean";
const NAMESPACE: &str = "Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.Generated.SelectorRows";
const GENERAL_SELECTOR: usize = 1;
const BIT: usize = 0;
const C: usize = 4;
const SELECTOR_COUNT: usize = 3;

fn is_exact_explicit(port: &SelectiveProjectedPort, terms: &[(usize, F)]) -> bool {
    port.geometric_runs().is_empty()
        && port.explicit().len() == terms.len()
        && port
            .explicit()
            .iter()
            .zip(terms)
            .all(|(actual, &(column, coefficient))| actual.column() == column && actual.coefficient() == coefficient)
}

pub(super) fn render(projected: &SelectiveProjectedRowsAudit) -> GeneratedLeanFile {
    let selectors = projected.selector_columns();
    let domain_rows = projected.selector_domain_row_artifacts();
    let total_row = projected.one_hot_row_artifact();
    assert_eq!(selectors.len(), SELECTOR_COUNT, "selector width");
    assert_eq!(domain_rows.len(), SELECTOR_COUNT, "selector-domain row count");
    assert_eq!(selectors, [270, 271, 272], "fixed-point selector columns");

    for (arm, (row, &selector)) in domain_rows.iter().zip(selectors).enumerate() {
        assert_eq!(row.family(), SelectiveEmittedRowFamily::SelectorDomain);
        assert_eq!(row.arm(), None);
        assert!(is_exact_explicit(&row.ports()[GENERAL_SELECTOR], &[(0, F::ONE)]));
        assert!(is_exact_explicit(&row.ports()[BIT], &[(selector, F::ONE)]));
        for (port_index, port) in row.ports().iter().enumerate() {
            if port_index != GENERAL_SELECTOR && port_index != BIT {
                assert!(port.explicit().is_empty(), "selector {arm} port {port_index}");
                assert!(
                    port.geometric_runs().is_empty(),
                    "selector {arm} geometric port {port_index}"
                );
            }
        }
    }

    assert_eq!(total_row.family(), SelectiveEmittedRowFamily::OneHot);
    assert_eq!(total_row.arm(), None);
    assert!(is_exact_explicit(&total_row.ports()[GENERAL_SELECTOR], &[(0, F::ONE)]));
    assert!(is_exact_explicit(
        &total_row.ports()[C],
        &[
            (0, -F::ONE),
            (selectors[0], F::ONE),
            (selectors[1], F::ONE),
            (selectors[2], F::ONE),
        ]
    ));
    for (port_index, port) in total_row.ports().iter().enumerate() {
        if port_index != GENERAL_SELECTOR && port_index != C {
            assert!(port.explicit().is_empty(), "one-hot port {port_index}");
            assert!(port.geometric_runs().is_empty(), "one-hot geometric port {port_index}");
        }
    }

    let mut contents = String::new();
    writeln!(
        contents,
        "import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.Materialized.Schema\n\n\
/-! Generated file: exact fixed-point selector rows.\n\n\
Owns: three proof-free selector-domain rows and one proof-free selector-total\n\
row projected from the prepared selective emitter.\n\n\
Does not own: decoding, selector values, retained-row coverage, branch\n\
semantics, CCS/CE membership, or row removal. Do not hand-edit.\n\n\
Emits constraints: no.\n\n\
| Artifact field | Exact source | Equation ownership |\n\
|---|---|---|\n\
| `rawRows[0..3]` | selector-domain owner | Boolean selector residuals |\n\
| `rawRows[3]` | one-hot owner | selector sum equals constant one |\n\
-/\n\n\
namespace {NAMESPACE}\n\n\
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.Materialized\n\n\
def rawRows : List RawRow := ["
    )
    .expect("render selector header");
    for (index, row) in domain_rows
        .iter()
        .chain(core::iter::once(total_row))
        .enumerate()
    {
        if index != 0 {
            contents.push_str(",\n");
        }
        write_raw_row(&mut contents, row);
    }
    writeln!(contents, "]\n\nend {NAMESPACE}").expect("render selector footer");
    assert!(contents.lines().count() < 1_500, "selector artifact line limit");
    GeneratedLeanFile {
        relative_path: GENERATED_PATH.to_owned(),
        contents,
    }
}

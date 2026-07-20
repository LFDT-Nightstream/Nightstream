//! Exact private-alignment padding rows for the bounded fixed-point carrier.
//!
//! Owns: bounded projection of all 38 private-alignment rows from the same
//! thirteen-port emitter used by the stabilized fixed-point relation.
//!
//! Does not own: assignment semantics, constant-one authority, branch-private
//! decoding, CCS/CE membership, commitment alignment, or row removal.

use std::fmt::Write as _;

use neo_fold_clean::frontends::r1cs_f_prime::{
    SelectiveEmittedRowFamily, SelectiveProjectedPort, SelectiveProjectedRowsAudit,
};
use neo_math::F;
use p3_field::PrimeCharacteristicRing;

use super::{selective_matrix_rows::write_raw_row, GeneratedLeanFile};

const GENERATED_PATH: &str =
    "formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/FPrimeSelectiveFixedPoint/Carrier270/Generated/PrivatePaddingRows.lean";
const NAMESPACE: &str =
    "Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.Generated.PrivatePaddingRows";
const GENERAL_SELECTOR: usize = 1;
const C: usize = 4;
const PRIVATE_PADDING_COUNT: usize = 38;

fn is_unit_at(port: &SelectiveProjectedPort, column: usize) -> bool {
    port.geometric_runs().is_empty()
        && port.explicit().len() == 1
        && port.explicit()[0].column() == column
        && port.explicit()[0].coefficient() == F::ONE
}

pub(super) fn render(projected: &SelectiveProjectedRowsAudit) -> GeneratedLeanFile {
    let expected_columns = projected
        .compiler_audit()
        .layout()
        .private_alignment_padding_columns();
    let rows = projected.private_padding_row_artifacts();
    assert_eq!(expected_columns.len(), PRIVATE_PADDING_COUNT, "private padding width");
    assert_eq!(
        rows.len(),
        expected_columns.len(),
        "one row per private padding coordinate"
    );

    for (offset, (artifact, &padding_column)) in rows.iter().zip(expected_columns).enumerate() {
        assert_eq!(artifact.family(), SelectiveEmittedRowFamily::PrivatePadding);
        assert_eq!(artifact.arm(), None);
        assert!(is_unit_at(&artifact.ports()[GENERAL_SELECTOR], 0));
        assert!(is_unit_at(&artifact.ports()[C], padding_column));
        for (port_index, port) in artifact.ports().iter().enumerate() {
            if port_index != GENERAL_SELECTOR && port_index != C {
                assert!(port.explicit().is_empty(), "private padding port {port_index}");
                assert!(
                    port.geometric_runs().is_empty(),
                    "private padding geometric port {port_index}"
                );
            }
        }
        if offset != 0 {
            assert_eq!(artifact.emitted_row(), rows[offset - 1].emitted_row() + 1);
        }
    }

    let mut contents = String::new();
    writeln!(
        contents,
        "import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.Materialized.Schema\n\n\
/-! Generated file: exact fixed-point private-alignment padding rows.\n\n\
Owns: all 38 proof-free thirteen-port rows projected from the prepared\n\
selective emitter.\n\n\
Does not own: decoding, row semantics, constant-one authority, private branch\n\
values, CCS/CE membership, commitment alignment, or row removal. Do not\n\
hand-edit.\n\n\
Emits constraints: no.\n\n\
| Artifact field | Exact source | Equation ownership |\n\
|---|---|---|\n\
| `rawRows` | final thirteen-port emitter | `-(z[0] * z[273+i])` for `i < 38` |\n\
-/\n\n\
namespace {NAMESPACE}\n\n\
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.Materialized\n\n\
set_option maxRecDepth 100000 in\n\
def rawRows : List RawRow := ["
    )
    .expect("render private padding header");
    for (offset, row) in rows.iter().enumerate() {
        if offset != 0 {
            contents.push_str(",\n");
        }
        write_raw_row(&mut contents, row);
    }
    writeln!(contents, "]\n\nend {NAMESPACE}").expect("render private padding footer");
    assert!(contents.lines().count() < 1_500, "private padding artifact line limit");
    GeneratedLeanFile {
        relative_path: GENERATED_PATH.to_owned(),
        contents,
    }
}

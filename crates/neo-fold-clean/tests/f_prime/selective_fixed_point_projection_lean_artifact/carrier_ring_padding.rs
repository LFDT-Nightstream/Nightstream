//! Exact final ring-alignment rows for the bounded fixed-point carrier.
//!
//! Owns: bounded projection of all 52 final `D = 64` alignment rows from the
//! same thirteen-port emitter used by the stabilized fixed-point relation.
//!
//! Does not own: the earlier 38-column private-layout alignment, assignment
//! semantics, constant-one authority, CCS/CE membership, or row removal.

use std::fmt::Write as _;

use neo_fold_clean::frontends::r1cs_f_prime::{
    SelectiveEmittedRowFamily, SelectiveProjectedPort, SelectiveProjectedRowsAudit,
};
use neo_math::F;
use p3_field::PrimeCharacteristicRing;

use super::{selective_matrix_rows::write_raw_row, GeneratedLeanFile};

const GENERATED_PATH: &str =
    "formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/FPrimeSelectiveFixedPoint/Carrier270/Generated/RingPaddingRows.lean";
const NAMESPACE: &str =
    "Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.Generated.RingPaddingRows";
const GENERAL_SELECTOR: usize = 1;
const C: usize = 4;
const RING_PADDING_COUNT: usize = 52;

fn is_unit_at(port: &SelectiveProjectedPort, column: usize) -> bool {
    port.geometric_runs().is_empty()
        && port.explicit().len() == 1
        && port.explicit()[0].column() == column
        && port.explicit()[0].coefficient() == F::ONE
}

pub(super) fn render(projected: &SelectiveProjectedRowsAudit) -> GeneratedLeanFile {
    let rows = projected.ring_padding_row_artifacts();
    assert_eq!(rows.len(), RING_PADDING_COUNT, "final ring padding width");
    let first_column = projected.columns() - rows.len();
    let first_emitted_row = rows[0].emitted_row();
    let run_index = rows[0].run_index();

    for (offset, artifact) in rows.iter().enumerate() {
        let padding_column = first_column + offset;
        assert_eq!(artifact.family(), SelectiveEmittedRowFamily::RingPadding);
        assert_eq!(artifact.arm(), None);
        assert_eq!(artifact.run_index(), run_index);
        assert_eq!(artifact.emitted_row(), first_emitted_row + offset);
        assert!(is_unit_at(&artifact.ports()[GENERAL_SELECTOR], 0));
        assert!(is_unit_at(&artifact.ports()[C], padding_column));
        for (port_index, port) in artifact.ports().iter().enumerate() {
            if port_index != GENERAL_SELECTOR && port_index != C {
                assert!(port.explicit().is_empty(), "ring padding port {port_index}");
                assert!(
                    port.geometric_runs().is_empty(),
                    "ring padding geometric port {port_index}"
                );
            }
        }
    }

    let mut contents = String::new();
    writeln!(
        contents,
        "import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.Materialized.Schema\n\n\
/-! Generated file: exact fixed-point final ring-alignment rows.\n\n\
Owns: all 52 proof-free thirteen-port rows emitted after the final selective\n\
column allocation to align the relation width to `D = 64`.\n\n\
Does not own: the earlier private-layout padding, decoding, row semantics,\n\
constant-one authority, CCS/CE membership, commitment alignment, or row\n\
removal. Do not hand-edit.\n\n\
Emits constraints: no.\n\n\
| Artifact field | Exact source | Equation ownership |\n\
|---|---|---|\n\
| `firstEmittedRow` | final emitter row cursor | first final alignment row |\n\
| `runIndex` | compiler ownership ledger | unique ring-padding run |\n\
| `rawRows` | final thirteen-port emitter | `-(z[0] * z[11725454+i])` for `i < 52` |\n\
-/\n\n\
namespace {NAMESPACE}\n\n\
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.Materialized\n\n\
def firstEmittedRow : Nat := {first_emitted_row}\n\n\
def runIndex : Nat := {run_index}\n\n\
set_option maxRecDepth 100000 in\n\
def rawRows : List RawRow := ["
    )
    .expect("render ring padding header");
    for (offset, row) in rows.iter().enumerate() {
        if offset != 0 {
            contents.push_str(",\n");
        }
        write_raw_row(&mut contents, row);
    }
    writeln!(contents, "]\n\nend {NAMESPACE}").expect("render ring padding footer");
    assert!(contents.lines().count() < 1_500, "ring padding artifact line limit");
    GeneratedLeanFile {
        relative_path: GENERATED_PATH.to_owned(),
        contents,
    }
}

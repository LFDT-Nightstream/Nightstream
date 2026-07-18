//! Three-matrix diagnostic owner for the shared PiRLC beta ladder.
//!
//! Owns: unique selection of the `ProjectionLadderAudit` shared by both
//! returned-parent `YZColLimb` identities, exact row/column interval checks,
//! and rendering of the active Lean artifact.
//!
//! Does not own: transcript derivation of beta, semantic parent authority,
//! projection-identity soundness, compact lowering, cost estimates, or row
//! removal.
//!
//! Emits constraints: no.
//!
//! Authority boundary: exact source rows and wire schedules are replayed from
//! production. Stage and role labels are selectors only.
//!
//! | Stage path | Mathematical obligation | Multiplicity | Evidence |
//! |---|---|---:|---|
//! | `nifs.pi_rlc.verify.projection_shared.beta_ladder` | `p[0] = 1`; `p[i+1] = p[i] * beta` | 272 source-R1CS rows | exact trace replay |
//! | returned-parent `YZColLimb` users | both identities consume this exact 55-wire ladder | two consumers | exact column equality |

use std::fmt::Write as _;
use std::fs;
use std::ops::Range;

use neo_fold_clean::engine::r1cs_circuit::builder::ProjectionLadderAudit;
use neo_fold_clean::engine::r1cs_circuit::{ProjectionIdentityRole, R1csEncodingTrace, R1csSnapshot};
use neo_fold_clean::paper::reductions::pi_rlc_circuit::stage as pi_rlc_stage;
use neo_math::ring::D;
use neo_math::F;
use p3_field::PrimeField64;

use super::repo_root;

const LEAN_DATA_PATH: &str = "formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/FPrimeRecursive/PiRlcProjection/Generated/BetaLadderData.lean";
const K_MUL_ROWS: usize = 5;
const POWER_COUNT: usize = D + 1;
const SOURCE_ROW_COUNT: usize = 2 + D * K_MUL_ROWS;

#[derive(Debug, PartialEq, Eq)]
struct BetaLadderOwner {
    row_start: usize,
    row_end: usize,
    allocated_start: usize,
    allocated_end: usize,
    beta_columns: [usize; 2],
    power_columns: Vec<[usize; 2]>,
}

fn exact_stage_interval(
    trace: &R1csEncodingTrace,
    expected_rows: Range<usize>,
    expected_columns: Range<usize>,
) -> (Range<usize>, Range<usize>) {
    let matching = trace
        .stages()
        .windows(2)
        .filter(|pair| {
            pair[0].label == pi_rlc_stage::PROJECTION_SHARED_BETA_LADDER
                && pair[0].row == expected_rows.start
                && pair[1].row == expected_rows.end
                && pair[0].col == expected_columns.start
                && pair[1].col == expected_columns.end
        })
        .collect::<Vec<_>>();
    let [pair] = matching.as_slice() else {
        panic!(
            "expected exactly one `{}` checkpoint interval with rows {expected_rows:?} and columns {expected_columns:?}, found {}",
            pi_rlc_stage::PROJECTION_SHARED_BETA_LADDER,
            matching.len()
        );
    };
    (pair[0].row..pair[1].row, pair[0].col..pair[1].col)
}

fn selected_owner(trace: &R1csEncodingTrace, audits: &[ProjectionLadderAudit]) -> BetaLadderOwner {
    let y_zcol_identities = trace
        .projection_identities()
        .iter()
        .filter_map(|identity| match identity.role {
            ProjectionIdentityRole::YZColLimb { limb } => Some((limb, identity)),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(
        y_zcol_identities
            .iter()
            .map(|(limb, _)| *limb)
            .collect::<Vec<_>>(),
        [0, 1],
        "the diagnostic profile must contain exactly the ordered c0/c1 y_zcol identities"
    );
    assert_eq!(
        y_zcol_identities[0].1.power_columns, y_zcol_identities[1].1.power_columns,
        "both returned-parent y_zcol identities must consume the same complete ladder"
    );

    let matching = audits
        .iter()
        .filter(|audit| {
            audit.power_columns == y_zcol_identities[0].1.power_columns
                && audit.power_columns == y_zcol_identities[1].1.power_columns
        })
        .collect::<Vec<_>>();
    let [audit] = matching.as_slice() else {
        panic!(
            "expected exactly one beta ladder shared by both y_zcol identities, found {}",
            matching.len()
        );
    };
    assert_eq!(audit.power_columns.len(), POWER_COUNT, "complete beta ladder width");
    assert_eq!(
        audit.row_end - audit.row_start,
        SOURCE_ROW_COUNT,
        "exact beta ladder row count"
    );

    let allocated_start = audit.power_columns[0][0];
    let allocated_end = audit.power_columns[POWER_COUNT - 1][1] + 1;
    assert_eq!(
        allocated_end - allocated_start,
        SOURCE_ROW_COUNT,
        "the ladder must allocate one column per source definition"
    );
    assert_eq!(
        audit.power_columns[0],
        [allocated_start, allocated_start + 1],
        "beta^0 must begin the ladder SSA interval"
    );
    for (index, columns) in audit.power_columns.iter().copied().enumerate().skip(1) {
        assert_eq!(
            columns,
            [
                allocated_start + K_MUL_ROWS * index,
                allocated_start + K_MUL_ROWS * index + 1
            ],
            "each beta-power output must terminate its contiguous five-column K-mul block"
        );
    }
    assert!(
        audit
            .beta_columns
            .iter()
            .all(|&column| column < allocated_start),
        "beta inputs must predate the ladder SSA interval"
    );
    let (stage_rows, stage_columns) =
        exact_stage_interval(trace, audit.row_start..audit.row_end, allocated_start..allocated_end);
    assert_eq!(stage_rows, audit.row_start..audit.row_end, "exact ladder stage rows");
    assert_eq!(
        stage_columns,
        allocated_start..allocated_end,
        "exact ladder stage columns"
    );

    BetaLadderOwner {
        row_start: audit.row_start,
        row_end: audit.row_end,
        allocated_start,
        allocated_end,
        beta_columns: audit.beta_columns,
        power_columns: audit.power_columns.clone(),
    }
}

fn lean_k_columns(columns: [usize; 2]) -> String {
    format!("{{ c0 := {}, c1 := {} }}", columns[0], columns[1])
}

fn lean_k_columns_list(values: impl IntoIterator<Item = [usize; 2]>) -> String {
    format!(
        "[{}]",
        values
            .into_iter()
            .map(lean_k_columns)
            .collect::<Vec<_>>()
            .join(", ")
    )
}

fn lean_terms(terms: &[(usize, F)]) -> String {
    format!(
        "[{}]",
        terms
            .iter()
            .map(|&(column, coefficient)| format!("({column}, {})", coefficient.as_canonical_u64()))
            .collect::<Vec<_>>()
            .join(", ")
    )
}

fn lean_source_rows(source: &R1csSnapshot, owner: &BetaLadderOwner) -> String {
    let rows = (owner.row_start..owner.row_end)
        .map(|row| {
            format!(
                "({row}, ⟨{}, {}, {}⟩)",
                lean_terms(source.a_row(row)),
                lean_terms(source.b_row(row)),
                lean_terms(source.c_row(row))
            )
        })
        .collect::<Vec<_>>();
    assert_eq!(rows.len(), SOURCE_ROW_COUNT, "exact beta ladder source rows");
    format!("[{}]", rows.join(",\n   "))
}

fn render(source: &R1csSnapshot, trace: &R1csEncodingTrace, audits: &[ProjectionLadderAudit]) -> String {
    let owner = selected_owner(trace, audits);
    let mut rendered = String::new();
    rendered.push_str(
        "import Nightstream.Implementation.R1CS.Artifacts.FPrimeRecursive.PiRlcProjection.BetaLadderSchema\n\n",
    );
    rendered.push_str(
        "/-! Generated by `active_pi_rlc_projection_artifacts_match_production_trace`; do not hand-edit. -/\n\n",
    );
    rendered.push_str("namespace Nightstream.Implementation.R1CS.FPrimeRecursivePiRlcProjectionBetaLadderData\n\n");
    writeln!(
        rendered,
        "def stagePath : String := \"{}\"",
        pi_rlc_stage::PROJECTION_SHARED_BETA_LADDER
    )
    .expect("render stage path");
    writeln!(rendered, "def rowStart : Nat := {}", owner.row_start).expect("render row start");
    writeln!(rendered, "def rowEnd : Nat := {}", owner.row_end).expect("render row end");
    writeln!(rendered, "def allocatedStart : Nat := {}", owner.allocated_start).expect("render column start");
    writeln!(rendered, "def allocatedEnd : Nat := {}", owner.allocated_end).expect("render column end");
    writeln!(
        rendered,
        "def betaColumns : ProjectionProgram.KColumns := {}",
        lean_k_columns(owner.beta_columns)
    )
    .expect("render beta columns");
    writeln!(
        rendered,
        "def powerColumns : List ProjectionProgram.KColumns := {}\n",
        lean_k_columns_list(owner.power_columns.iter().copied())
    )
    .expect("render power columns");
    writeln!(
        rendered,
        "def sourceRows : List (Nat × Row) :=\n  {}\n",
        lean_source_rows(source, &owner)
    )
    .expect("render exact source rows");
    rendered.push_str("end Nightstream.Implementation.R1CS.FPrimeRecursivePiRlcProjectionBetaLadderData\n");
    rendered
}

pub(super) fn check_generated_artifact(
    source: &R1csSnapshot,
    trace: &R1csEncodingTrace,
    audits: &[ProjectionLadderAudit],
) {
    let rendered = render(source, trace, audits);
    let path = repo_root().join(LEAN_DATA_PATH);
    let committed = fs::read_to_string(&path).unwrap_or_default();
    if committed != rendered {
        let expected = path.with_extension("lean.expected");
        fs::create_dir_all(expected.parent().expect("generated artifact parent"))
            .expect("create generated artifact directory");
        fs::write(&expected, &rendered).expect("write expected active beta-ladder artifact");
    }
    assert_eq!(
        committed, rendered,
        "active beta-ladder artifact drifted; review the generated .expected file"
    );
}

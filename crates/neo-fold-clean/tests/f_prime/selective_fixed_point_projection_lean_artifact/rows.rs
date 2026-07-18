//! Exact 12-way source-row shard partition for the selected projection certificate.
//!
//! Owns: deterministic partition of all 5,724 retained rows into the shared
//! beta shard, three shared-rho shards, three complete source-pair shards per
//! coefficient limb, and one tail shard per limb.
//!
//! Does not own: row semantics, trace reconstruction, serializer bindings,
//! full-relation placement, cost lowering, or row removal.
//!
//! Emits constraints: no.
//!
//! | Shard family | Shards | Rows per shard |
//! |---|---:|---:|
//! | beta ladder | 1 | 272 |
//! | rho evaluations | 3 | 540 |
//! | limb source pairs | 6 | 565 |
//! | limb tails | 2 | 221 |

use std::collections::HashSet;
use std::fmt::Write as _;
use std::ops::Range;

use neo_fold_clean::frontends::r1cs_f_prime::ivc::{PiCcsOutputYZcolProjectionAudit, PiRlcYZcolProjectionRowAudit};
use neo_fold_clean::paper::reductions::pi_rlc_circuit::stage;
use neo_math::F;
use p3_field::PrimeField64;

use super::GeneratedLeanFile;

const GENERATED_ROOT: &str =
    "formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/FPrimeSelectiveFixedPoint/PiRlcProjection/YZcol/Generated/Rows";
const SOURCES_PER_SHARD: usize = 5;
const GENERATED_NAMESPACE: &str =
    "Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Generated.Rows";

struct RowShard<'a> {
    name: String,
    namespace: String,
    stage_paths: Vec<&'static str>,
    owner_scope: String,
    ranges: Vec<Range<usize>>,
    rows: Vec<&'a PiRlcYZcolProjectionRowAudit>,
}

fn select_rows<'a>(
    source_rows: &'a [PiRlcYZcolProjectionRowAudit],
    ranges: &[Range<usize>],
) -> Vec<&'a PiRlcYZcolProjectionRowAudit> {
    source_rows
        .iter()
        .filter(|row| ranges.iter().any(|range| range.contains(&row.index())))
        .collect()
}

fn make_shard<'a>(
    source_rows: &'a [PiRlcYZcolProjectionRowAudit],
    name: impl Into<String>,
    namespace: impl Into<String>,
    stage_paths: Vec<&'static str>,
    owner_scope: impl Into<String>,
    ranges: Vec<Range<usize>>,
    expected_rows: usize,
) -> RowShard<'a> {
    let name = name.into();
    let rows = select_rows(source_rows, &ranges);
    assert_eq!(rows.len(), expected_rows, "exact row count for {name}");
    RowShard {
        name,
        namespace: namespace.into(),
        stage_paths,
        owner_scope: owner_scope.into(),
        ranges,
        rows,
    }
}

fn partition(projection: &PiCcsOutputYZcolProjectionAudit) -> Vec<RowShard<'_>> {
    let identity = projection.identity();
    let source_rows = identity.source_rows();
    let mut shards = Vec::with_capacity(12);

    shards.push(make_shard(
        source_rows,
        "BetaLadder",
        "BetaLadder",
        vec![stage::PROJECTION_SHARED_BETA_LADDER],
        "shared beta powers 0 through 54",
        vec![identity.shared().beta_ladder_rows()],
        272,
    ));

    let rho = identity.shared().rho_evaluations();
    assert_eq!(rho.len(), 15, "active rho evaluation count");
    for shard in 0..3 {
        let ranges = rho[shard * SOURCES_PER_SHARD..(shard + 1) * SOURCES_PER_SHARD]
            .iter()
            .map(|evaluation| evaluation.rows())
            .collect();
        shards.push(make_shard(
            source_rows,
            format!("RhoEvaluations{shard}"),
            format!("RhoEvaluations.Shard{shard}"),
            vec![stage::PROJECTION_SHARED_RHO_EVALUATIONS],
            format!(
                "rho sources {} through {}",
                shard * SOURCES_PER_SHARD,
                (shard + 1) * SOURCES_PER_SHARD - 1
            ),
            ranges,
            5 * 108,
        ));
    }

    for limb in 0..2 {
        let owner = identity.limb(limb);
        assert_eq!(owner.input_evaluations().len(), 15, "active limb input count");
        assert_eq!(owner.rho_products().len(), 15, "active limb product count");
        for shard in 0..3 {
            let mut ranges = Vec::with_capacity(2 * SOURCES_PER_SHARD);
            for source in shard * SOURCES_PER_SHARD..(shard + 1) * SOURCES_PER_SHARD {
                ranges.push(owner.input_evaluations()[source].rows());
                ranges.push(owner.rho_products()[source].rows());
            }
            shards.push(make_shard(
                source_rows,
                format!("Limb{limb}Pairs{shard}"),
                format!("Limb{limb}.Pairs.Shard{shard}"),
                match limb {
                    0 => vec![
                        stage::IDENTITIES_Y_ZCOL_EVALUATIONS_INPUTS_LIMB0,
                        stage::IDENTITIES_Y_ZCOL_K_PRODUCTS_RHO_TIMES_INPUT_LIMB0,
                    ],
                    1 => vec![
                        stage::IDENTITIES_Y_ZCOL_EVALUATIONS_INPUTS_LIMB1,
                        stage::IDENTITIES_Y_ZCOL_K_PRODUCTS_RHO_TIMES_INPUT_LIMB1,
                    ],
                    _ => unreachable!("two-limb artifact"),
                },
                format!(
                    "limb {limb}, sources {} through {}",
                    shard * SOURCES_PER_SHARD,
                    (shard + 1) * SOURCES_PER_SHARD - 1
                ),
                ranges,
                5 * (108 + 5),
            ));
        }

        let tail_ranges = vec![
            owner.parent_evaluation().rows(),
            owner.quotient_evaluation().rows(),
            owner.quotient_phi_product().rows(),
            owner.final_rows(),
        ];
        shards.push(make_shard(
            source_rows,
            format!("Limb{limb}Tail"),
            format!("Limb{limb}.Tail"),
            match limb {
                0 => vec![
                    stage::IDENTITIES_Y_ZCOL_EVALUATIONS_OUTPUT_LIMB0,
                    stage::IDENTITIES_Y_ZCOL_EVALUATIONS_QUOTIENT_LIMB0,
                    stage::IDENTITIES_Y_ZCOL_K_PRODUCTS_QUOTIENT_TIMES_PHI_LIMB0,
                    stage::IDENTITIES_Y_ZCOL_FINAL_LIMB_CHECKS_LIMB0,
                ],
                1 => vec![
                    stage::IDENTITIES_Y_ZCOL_EVALUATIONS_OUTPUT_LIMB1,
                    stage::IDENTITIES_Y_ZCOL_EVALUATIONS_QUOTIENT_LIMB1,
                    stage::IDENTITIES_Y_ZCOL_K_PRODUCTS_QUOTIENT_TIMES_PHI_LIMB1,
                    stage::IDENTITIES_Y_ZCOL_FINAL_LIMB_CHECKS_LIMB1,
                ],
                _ => unreachable!("two-limb artifact"),
            },
            format!("limb {limb} parent, quotient, Phi81 product, and final checks"),
            tail_ranges,
            108 + 106 + 5 + 2,
        ));
    }

    assert_eq!(shards.len(), 12, "exact active row-shard count");
    let selected = shards
        .iter()
        .flat_map(|shard| shard.rows.iter().map(|row| row.index()))
        .collect::<Vec<_>>();
    assert_eq!(selected.len(), 5_724, "all retained projection rows selected");
    assert!(
        selected.windows(2).all(|pair| pair[0] < pair[1]),
        "generated shard concatenation must preserve absolute source-row order"
    );
    assert_eq!(
        selected.iter().copied().collect::<HashSet<_>>().len(),
        selected.len(),
        "row shards must be disjoint"
    );
    assert_eq!(
        selected.iter().copied().collect::<HashSet<_>>(),
        source_rows
            .iter()
            .map(PiRlcYZcolProjectionRowAudit::index)
            .collect(),
        "row shards must cover the complete retained certificate"
    );
    shards
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

fn lean_row(row: &PiRlcYZcolProjectionRowAudit) -> String {
    format!(
        "({}, ⟨{}, {}, {}⟩)",
        row.index(),
        lean_terms(row.a()),
        lean_terms(row.b()),
        lean_terms(row.c())
    )
}

fn render(shard: RowShard<'_>) -> GeneratedLeanFile {
    let mut contents = String::new();
    contents.push_str("import Nightstream.Implementation.R1CS.Artifacts.Projection.IndexedRows\n\n");
    contents.push_str("/-! Generated by `active_selective_fixed_point_projection_artifact_matches_retained_certificate`; do not hand-edit.\n\n");
    contents
        .push_str("Owns: exact indexed normalized A/B/C rows for one selected 15/13 source-row artifact shard.\n\n");
    contents.push_str(
        "Does not own: trace semantics, protocol authority, selective lowering, final costs, or row removal.\n\n",
    );
    contents.push_str("Emits constraints: no.\n\n");
    contents.push_str("| Payload | Mathematical obligation | Evidence |\n|---|---|---|\n");
    writeln!(
        contents,
        "| stage path(s) | `{}` | Rust source-arm physical-stage audit |",
        shard.stage_paths.join("`, `")
    )
    .expect("render stage paths");
    writeln!(
        contents,
        "| owner scope | {} | generator shard spanning the listed stage paths |",
        shard.owner_scope
    )
    .expect("render owner scope");
    writeln!(
        contents,
        "| `sourceRows` | preserve every sparse A/B/C term at its absolute row index | {} retained source-R1CS rows |\n-/\n",
        shard.rows.len()
    )
    .expect("render row header");
    writeln!(contents, "namespace {GENERATED_NAMESPACE}.{}\n", shard.namespace).expect("render row namespace");
    writeln!(contents, "def rangeCount : Nat := {}", shard.ranges.len()).expect("render range count");
    writeln!(contents, "def rowCount : Nat := {}", shard.rows.len()).expect("render row count");
    contents.push_str("set_option maxRecDepth 100000 in\n");
    contents.push_str("def sourceRows : List (Nat × Row) :=\n  [");
    for (index, row) in shard.rows.iter().enumerate() {
        if index != 0 {
            contents.push_str("   ,");
        }
        writeln!(contents, "{}", lean_row(row)).expect("render sparse row");
    }
    contents.push_str("  ]\n\n");
    writeln!(contents, "end {GENERATED_NAMESPACE}.{}", shard.namespace).expect("render row namespace end");

    GeneratedLeanFile {
        relative_path: format!("{GENERATED_ROOT}/{}.lean", shard.name),
        contents,
    }
}

pub(super) fn row_shards(projection: &PiCcsOutputYZcolProjectionAudit) -> Vec<GeneratedLeanFile> {
    partition(projection).into_iter().map(render).collect()
}

pub(super) fn source_row_definitions() -> Vec<String> {
    [
        "BetaLadder",
        "RhoEvaluations.Shard0",
        "RhoEvaluations.Shard1",
        "RhoEvaluations.Shard2",
        "Limb0.Pairs.Shard0",
        "Limb0.Pairs.Shard1",
        "Limb0.Pairs.Shard2",
        "Limb0.Tail",
        "Limb1.Pairs.Shard0",
        "Limb1.Pairs.Shard1",
        "Limb1.Pairs.Shard2",
        "Limb1.Tail",
    ]
    .into_iter()
    .map(|namespace| format!("{GENERATED_NAMESPACE}.{namespace}.sourceRows"))
    .collect()
}

pub(super) fn generated_row_modules() -> [&'static str; 12] {
    [
        "BetaLadder",
        "RhoEvaluations0",
        "RhoEvaluations1",
        "RhoEvaluations2",
        "Limb0Pairs0",
        "Limb0Pairs1",
        "Limb0Pairs2",
        "Limb0Tail",
        "Limb1Pairs0",
        "Limb1Pairs1",
        "Limb1Pairs2",
        "Limb1Tail",
    ]
}

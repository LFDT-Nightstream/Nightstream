//! Radix-four candidate census and lifecycle profiles.

#[path = "../../../neo-fold-clean/tests/support/selective_selector_coverage_lean.rs"]
mod selector_coverage_lean;

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::fs::File;
use std::io::{BufReader, BufWriter, Write};
use std::time::Instant;

use neo_fold_clean::config;
use neo_fold_clean::frontends::r1cs_f_prime::{
    SelectiveArmWidthAudit, SelectiveEmittedRowFamily, SelectiveEmittedRowRunAudit,
    SelectiveFirstAcceptedSelectionAudit, SelectiveLinearDefinitionAudit, SelectiveSelectorGateCoverage,
};
use neo_fold_clean::paper::f_prime::nebula_lane_circuit::delayed_nebula_public_suffix_len;
use neo_fold_clean::paper::f_prime::public_input_link::F_PRIME_PUBLIC_INPUT_LEN;
use neo_fold_clean::paper::params::Params;
use neo_math::D;
#[cfg(all(feature = "metal", target_vendor = "apple"))]
use neo_prover_metal::MetalNifsProver;

use super::{common, ms, performance_profile, PROFILE_WAT};

const RADIX_FOUR_SELECTOR_COVERAGE_ARTIFACT_PATH: &str = "/../../formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/FPrimeFullHistory/Generated/FPrimeFullHistoryRadixFourSelectorCoverage.lean";
const RADIX_FOUR_SOURCE_STAGE_COVERAGE_ARTIFACT_PATH: &str = "/../../formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/FPrimeFullHistory/Generated/FPrimeFullHistoryRadixFourSourceStageCoverage.lean";
const RADIX_FOUR_SELECTION_ARTIFACT_PATH: &str = "/../../formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/FPrimeFullHistory/Generated/FPrimeFullHistoryRadixFourFirstAcceptedSelection.lean";
const RADIX_FOUR_GENERAL_FIELD_WIDTH: usize = 23;
const CANONICAL_TERNARY_FIELD_WIDTH: usize = 41;
const BINARY_FIELD_WIDTH: usize = 64;

#[derive(Clone, Copy)]
struct SourceStageOwner {
    name: &'static str,
    path_prefix: &'static str,
    lean_constructor: &'static str,
}

const WIDTH_OWNERS: [SourceStageOwner; 14] = [
    SourceStageOwner {
        name: "application",
        path_prefix: "fprime.recursive.finalize.application",
        lean_constructor: "application",
    },
    SourceStageOwner {
        name: "prelude",
        path_prefix: "fprime.recursive.step.prelude",
        lean_constructor: "prelude",
    },
    SourceStageOwner {
        name: "transcript",
        path_prefix: "fprime.recursive.step.transcript",
        lean_constructor: "transcript",
    },
    SourceStageOwner {
        name: "pi_ccs",
        path_prefix: "nifs.pi_ccs",
        lean_constructor: "piCcs",
    },
    SourceStageOwner {
        name: "running_parent_pi_dec",
        path_prefix: "nifs.running_parent_pi_dec",
        lean_constructor: "runningParentPiDec",
    },
    SourceStageOwner {
        name: "pi_rlc",
        path_prefix: "nifs.pi_rlc",
        lean_constructor: "piRlc",
    },
    SourceStageOwner {
        name: "pi_dec",
        path_prefix: "nifs.pi_dec",
        lean_constructor: "piDec",
    },
    SourceStageOwner {
        name: "point_binding",
        path_prefix: "nifs.point_binding",
        lean_constructor: "pointBinding",
    },
    SourceStageOwner {
        name: "prior_link",
        path_prefix: "fprime.recursive.step.prior_link",
        lean_constructor: "priorLink",
    },
    SourceStageOwner {
        name: "nebula",
        path_prefix: "fprime.recursive.step.nebula",
        lean_constructor: "nebula",
    },
    SourceStageOwner {
        name: "accumulator",
        path_prefix: "fprime.recursive.step.accumulator",
        lean_constructor: "accumulator",
    },
    SourceStageOwner {
        name: "counters",
        path_prefix: "fprime.recursive.step.counters",
        lean_constructor: "counters",
    },
    SourceStageOwner {
        name: "output",
        path_prefix: "fprime.recursive.step.output",
        lean_constructor: "output",
    },
    SourceStageOwner {
        name: "semantic_links",
        path_prefix: "fprime.recursive.finalize.semantic_links",
        lean_constructor: "semanticLinks",
    },
];

#[derive(Clone, Copy)]
struct SourceStageOwnerCensus {
    owner: SourceStageOwner,
    stages: usize,
    source_fields: usize,
    direct: usize,
    decomposition_alias: usize,
    equality_alias: usize,
    linear_definition: usize,
    trace_eliminated: usize,
    allocated_coordinates: usize,
    source_boolean_coordinates: usize,
    outer_norm_coordinates: usize,
    boolean_domain_rows: usize,
    centered_unit_domain_coordinates: usize,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct StageFamilyTotals {
    occurrences: usize,
    source_rows: usize,
    emitted_rows: usize,
    source_fields: usize,
    direct: usize,
    decomposition_alias: usize,
    equality_alias: usize,
    linear_definition: usize,
    trace_eliminated: usize,
    allocated_coordinates: usize,
    source_boolean_coordinates: usize,
    outer_norm_coordinates: usize,
    boolean_domain_rows: usize,
    centered_unit_domain_coordinates: usize,
}

#[derive(Debug)]
struct SelectionSamplerTemplate {
    arm: usize,
    first_rewrite: usize,
    first_stage: usize,
    first_source_row: usize,
    first_emitted_row: usize,
    first_selector_column: usize,
    accepts: Vec<usize>,
    prefixes: Vec<usize>,
    symbols: Vec<usize>,
}

fn selection_sampler_templates(audits: &[SelectiveFirstAcceptedSelectionAudit]) -> Vec<SelectionSamplerTemplate> {
    assert_eq!(audits.len() % D, 0, "selection audit count must contain whole samplers");
    audits
        .chunks_exact(D)
        .map(|group| {
            let first = &group[0];
            let first_selector_column = first.selectors()[0];
            for (position, audit) in group.iter().enumerate() {
                let selector_start = first_selector_column + 45 * position;
                assert_eq!(audit.arm(), first.arm(), "selection sampler changed arm");
                assert_eq!(audit.position(), position, "selection sampler position drifted");
                assert_eq!(
                    audit.rewrite_id().index(),
                    first.rewrite_id().index() + position,
                    "selection rewrite identifiers are not consecutive"
                );
                assert_eq!(
                    audit.stage_occurrence(),
                    first.stage_occurrence() + 2 * position,
                    "selection physical stages are not affine"
                );
                assert_eq!(
                    audit.source_rows(),
                    first.source_rows().start + 48 * position..first.source_rows().start + 48 * position + 36,
                    "selection source rows are not affine"
                );
                assert_eq!(
                    audit.emitted_rows(),
                    first.emitted_rows().start + 9 * position..first.emitted_rows().start + 9 * (position + 1),
                    "selection emitted rows are not affine"
                );
                assert_eq!(
                    audit.selectors(),
                    (selector_start..selector_start + 11).collect::<Vec<_>>(),
                    "selection selector columns are not affine"
                );
                assert_eq!(
                    audit.symbol_products(),
                    (0..11)
                        .map(|candidate| selector_start + 11 + 3 * candidate)
                        .collect::<Vec<_>>(),
                    "selection symbol-product columns are not affine"
                );
                assert_eq!(
                    audit.accepted_products(),
                    (0..11)
                        .map(|candidate| selector_start + 12 + 3 * candidate)
                        .collect::<Vec<_>>(),
                    "selection accepted-product columns are not affine"
                );
                assert_eq!(
                    audit.prefix_products(),
                    (0..11)
                        .map(|candidate| selector_start + 13 + 3 * candidate)
                        .collect::<Vec<_>>(),
                    "selection prefix-product columns are not affine"
                );
                assert_eq!(
                    audit.output(),
                    selector_start + 44,
                    "selection output column is not affine"
                );
            }

            SelectionSamplerTemplate {
                arm: first.arm(),
                first_rewrite: first.rewrite_id().index(),
                first_stage: first.stage_occurrence(),
                first_source_row: first.source_rows().start,
                first_emitted_row: first.emitted_rows().start,
                first_selector_column,
                accepts: merge_selection_windows(group, SelectiveFirstAcceptedSelectionAudit::accepts),
                prefixes: merge_selection_windows(group, SelectiveFirstAcceptedSelectionAudit::prefixes),
                symbols: merge_selection_windows(group, SelectiveFirstAcceptedSelectionAudit::symbols),
            }
        })
        .collect()
}

fn merge_selection_windows(
    group: &[SelectiveFirstAcceptedSelectionAudit],
    select: for<'a> fn(&'a SelectiveFirstAcceptedSelectionAudit) -> &'a [usize],
) -> Vec<usize> {
    let mut values = select(&group[0]).to_vec();
    for (position, audit) in group.iter().enumerate().skip(1) {
        let window = select(audit);
        assert_eq!(window.len(), 11, "selection source window width drifted");
        assert_eq!(
            &values[position..],
            &window[..10],
            "selection source windows do not overlap exactly"
        );
        values.push(window[10]);
    }
    assert_eq!(values.len(), 64, "selection sampler source width drifted");
    values
}

fn stage_family_census(
    recursive: &SelectiveArmWidthAudit,
    emitted_runs: &[SelectiveEmittedRowRunAudit],
    recursive_arm: usize,
    prefix: &str,
) -> BTreeMap<&'static str, StageFamilyTotals> {
    let mut emitted_rows_by_stage = vec![0usize; recursive.physical_stages.len()];
    for run in emitted_runs
        .iter()
        .filter(|run| run.arm() == Some(recursive_arm))
    {
        let Some(stage) = run.source_stage_occurrence() else {
            continue;
        };
        emitted_rows_by_stage[stage] += run.emitted_rows().len();
    }

    let mut families = BTreeMap::<&'static str, StageFamilyTotals>::new();
    for (stage_index, stage) in recursive.physical_stages.iter().enumerate() {
        if !stage.path.starts_with(prefix) {
            continue;
        }
        let family = families.entry(stage.path).or_default();
        family.occurrences += 1;
        family.source_rows += stage.source_rows.len();
        family.emitted_rows += emitted_rows_by_stage[stage_index];
        family.source_fields += stage.source_column_count;
        family.direct += stage.direct_columns;
        family.decomposition_alias += stage.decomposition_alias_columns;
        family.equality_alias += stage.equality_alias_columns;
        family.linear_definition += stage.linear_definition_columns;
        family.trace_eliminated += stage.trace_eliminated_columns;
        family.allocated_coordinates += stage.allocated_coordinates;
        family.source_boolean_coordinates += stage.source_boolean_coordinates;
        family.outer_norm_coordinates += stage.outer_norm_coordinates;
        family.boolean_domain_rows += stage.boolean_domain_rows;
        family.centered_unit_domain_coordinates += stage.centered_unit_domain_coordinates;
    }
    families
}

fn affine_value_fits(first: usize, second: usize, length: usize, next: usize) -> bool {
    second
        .checked_sub(first)
        .and_then(|stride| first.checked_add(stride.checked_mul(length)?))
        == Some(next)
}

fn definitions_share_affine_run(
    definitions: &[SelectiveLinearDefinitionAudit],
    run_start: usize,
    run_length: usize,
    next: &SelectiveLinearDefinitionAudit,
) -> bool {
    let first = &definitions[run_start];
    if first.constant() != next.constant() || first.terms().len() != next.terms().len() {
        return false;
    }
    if run_length > 1 {
        let second = &definitions[run_start + 1];
        if !affine_value_fits(first.target(), second.target(), run_length, next.target()) {
            return false;
        }
        match (first.source_row(), second.source_row(), next.source_row()) {
            (None, None, None) => {}
            (Some(first), Some(second), Some(next)) if affine_value_fits(first, second, run_length, next) => {}
            _ => return false,
        }
    } else if first.source_row().is_none() != next.source_row().is_none() {
        return false;
    }
    first
        .terms()
        .iter()
        .zip(next.terms())
        .enumerate()
        .all(|(term_index, (first_term, next_term))| {
            if first_term.coefficient() != next_term.coefficient() {
                return false;
            }
            run_length == 1
                || affine_value_fits(
                    first_term.column(),
                    definitions[run_start + 1].terms()[term_index].column(),
                    run_length,
                    next_term.column(),
                )
        })
}

fn affine_definition_run_count(definitions: &[SelectiveLinearDefinitionAudit]) -> usize {
    if definitions.is_empty() {
        return 0;
    }
    let mut runs = 1usize;
    let mut run_start = 0usize;
    let mut run_length = 1usize;
    for next in &definitions[1..] {
        if definitions_share_affine_run(definitions, run_start, run_length, next) {
            run_length += 1;
        } else {
            runs += 1;
            run_start += run_length;
            run_length = 1;
        }
    }
    runs
}

fn render_radix_four_selector_coverage_artifact(
    params: &Params,
    scan_steps: usize,
    coverage: &SelectiveSelectorGateCoverage,
) -> String {
    let mut rendered = String::new();
    writeln!(
        rendered,
        "import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.SelectiveSelectorCoverageSchema\n\n\
/-! Generated file: complete run-compressed selector coverage for the exact\n\
production-width WASM Nebula radix-four candidate.\n\n\
Owns: every exclusive compiler owner interval, every final general/evaluation\n\
selector-port interval, and the ordered selective polynomial read from the\n\
final Rust relation.\n\n\
Does not own: arithmetic-family identity, source-to-final assignment\n\
refinement, recursive or terminal relation semantics, constraint necessity,\n\
security reduction, or permission to remove rows.\n\n\
Emits constraints: no. Rust emits this file only after it reconciles the\n\
complete selector CSC ports with the exclusive owner ledger.\n\n\
| Artifact branch | Exact Rust source | Scope |\n\
|---|---|---|\n\
| owner runs | production selective compiler ledger | all rows |\n\
| gate runs | final selector-port CSC matrices | all rows |\n\
| polynomial | final ordered sparse terms | all 74 terms |\n\
| profile constants | radix-four candidate parameters | provenance |\n\
-/\n\n\
namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryRadixFourSelectorCoverage\n\n\
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.SelectorCoverage.Wire\n\n\
def profileId : String := \"wasm-nebula-radix-four-candidate-v1\"\n\
def normBase : Nat := {}\n\
def decompositionExponent : Nat := {}\n\
def normBound : Nat := {}\n\
def kappa : Nat := {}\n\
def effectiveLambda : Nat := {}\n\
def batchSize : Nat := 3\n\
def scanSteps : Nat := {}\n\
def sourceOwnerRunCount : Nat := {}\n\
def sourceNonemptyOwnerRunCount : Nat := {}\n\
def coalescedRunCount : Nat := {}",
        params.b(),
        params.k_rho(),
        params.big_b(),
        params.kappa(),
        params.lambda(),
        scan_steps,
        coverage.owner_runs().len(),
        coverage
            .owner_runs()
            .iter()
            .filter(|owner| !owner.emitted_rows().is_empty())
            .count(),
        coverage.coalesced_owner_gate_runs().len(),
    )
    .expect("render radix-four selector-coverage header");
    selector_coverage_lean::write_coalesced_raw_coverage(&mut rendered, "rawCoverage", coverage)
        .expect("render radix-four selector coverage");
    writeln!(
        rendered,
        "\nend Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryRadixFourSelectorCoverage"
    )
    .expect("render radix-four selector-coverage footer");
    rendered
}

fn assert_radix_four_selector_coverage_artifact_matches_committed(
    params: &Params,
    scan_steps: usize,
    coverage: &SelectiveSelectorGateCoverage,
) {
    let rendered = render_radix_four_selector_coverage_artifact(params, scan_steps, coverage);
    let path = format!(
        "{}{}",
        env!("CARGO_MANIFEST_DIR"),
        RADIX_FOUR_SELECTOR_COVERAGE_ARTIFACT_PATH
    );
    let committed = std::fs::read_to_string(&path).unwrap_or_default();
    if rendered != committed {
        let expected = format!("{path}.expected");
        std::fs::write(&expected, rendered).expect("write reviewed radix-four selector-coverage artifact");
        panic!("radix-four selector-coverage artifact drifted; wrote {expected}. Inspect and promote it explicitly");
    }
}

fn render_radix_four_source_stage_coverage_artifact(
    source_fields: usize,
    physical_stages: usize,
    unowned_empty_stages: usize,
    dispositions: [usize; 5],
    allocated_coordinates: usize,
    owners: &[SourceStageOwnerCensus],
) -> String {
    let mut rendered = String::new();
    writeln!(
        rendered,
        "import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.SourceStageCoverageSchema\n\n\
/-! Generated file: exact top-level source-stage census for the production-width\n\
WASM Nebula radix-four recursive arm.\n\n\
Owns: the five exclusive source-field dispositions and the exact aggregation\n\
of every nonempty physical stage into fourteen reviewed path prefixes.\n\n\
Does not own: arithmetic-family semantics, path-label authority, individual\n\
source-to-final decoder rules, relation soundness, constraint necessity, or\n\
permission to remove rows or columns.\n\n\
Emits constraints: no. Rust emits this file after the live compiler checks the\n\
exclusive physical-stage partition and decoder dispositions.\n\
-/\n\n\
namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryRadixFourSourceStageCoverage\n\n\
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.SourceStageCoverage\n\n\
def profileId : String := \"wasm-nebula-radix-four-candidate-v1\"\n\
def rawCoverage : RawCoverage where",
    )
    .expect("render radix-four source-stage coverage header");
    for line in [
        "  schemaVersion := 1".to_owned(),
        format!("  physicalStages := {physical_stages}"),
        format!("  unownedEmptyStages := {unowned_empty_stages}"),
        format!("  sourceFields := {source_fields}"),
        format!("  direct := {}", dispositions[0]),
        format!("  decompositionAlias := {}", dispositions[1]),
        format!("  equalityAlias := {}", dispositions[2]),
        format!("  linearDefinition := {}", dispositions[3]),
        format!("  traceEliminated := {}", dispositions[4]),
        format!("  allocatedCoordinates := {allocated_coordinates}"),
        "  owners := [".to_owned(),
    ] {
        writeln!(rendered, "{line}").expect("render radix-four source-stage coverage field");
    }
    for (index, census) in owners.iter().enumerate() {
        let separator = if index == 0 { "   " } else { "  ," };
        writeln!(
            rendered,
            "{separator} {{ owner := .{}, stages := {}, sourceFields := {},",
            census.owner.lean_constructor, census.stages, census.source_fields,
        )
        .expect("render radix-four source-stage owner header");
        writeln!(
            rendered,
            "       direct := {}, decompositionAlias := {}, equalityAlias := {},",
            census.direct, census.decomposition_alias, census.equality_alias,
        )
        .expect("render radix-four source-stage owner dispositions");
        writeln!(
            rendered,
            "       linearDefinition := {}, traceEliminated := {}, allocatedCoordinates := {} }}",
            census.linear_definition, census.trace_eliminated, census.allocated_coordinates,
        )
        .expect("render radix-four source-stage owner");
    }
    writeln!(
        rendered,
        "  ]\n\nend Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryRadixFourSourceStageCoverage"
    )
    .expect("render radix-four source-stage coverage footer");
    rendered
}

fn assert_radix_four_source_stage_coverage_artifact_matches_committed(
    source_fields: usize,
    physical_stages: usize,
    unowned_empty_stages: usize,
    dispositions: [usize; 5],
    allocated_coordinates: usize,
    owners: &[SourceStageOwnerCensus],
) {
    let rendered = render_radix_four_source_stage_coverage_artifact(
        source_fields,
        physical_stages,
        unowned_empty_stages,
        dispositions,
        allocated_coordinates,
        owners,
    );
    let path = format!(
        "{}{}",
        env!("CARGO_MANIFEST_DIR"),
        RADIX_FOUR_SOURCE_STAGE_COVERAGE_ARTIFACT_PATH
    );
    let committed = std::fs::read_to_string(&path).unwrap_or_default();
    if rendered != committed {
        let expected = format!("{path}.expected");
        std::fs::write(&expected, rendered).expect("write reviewed radix-four source-stage coverage artifact");
        panic!(
            "radix-four source-stage coverage artifact drifted; wrote {expected}. Inspect and promote it explicitly"
        );
    }
}

fn render_radix_four_selection_artifact(
    relation_rows: usize,
    relation_columns: usize,
    source_rows: usize,
    source_columns: usize,
    audits: &[SelectiveFirstAcceptedSelectionAudit],
) -> String {
    let samplers = selection_sampler_templates(audits);
    assert_eq!(samplers.len(), 8, "production selection sampler count drifted");
    let mut rendered = String::new();
    writeln!(
        rendered,
        concat!(
            "import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.FirstAcceptedSelectionTemplateSchema\n\n",
            "/-! Generated file: exact compact schedule for every production-width\n",
            "radix-four first-accepted selection rewrite.\n\n",
            "Owns: eight source-column samplers and their affine expansion into all 432\n",
            "source and final rewrite intervals. Rust emits this only after exact local\n",
            "source-row checks and compiler-ledger joins pass.\n\n",
            "Does not own: one-hotness, final low-norm gate semantics, complete PiRLC\n",
            "semantics, or permission to remove another row family.\n",
            "-/\n\n",
            "namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryRadixFourFirstAcceptedSelection\n\n",
            "open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.FirstAcceptedSelection\n\n",
            "def profileId : String := \"wasm-nebula-radix-four-candidate-v1\"\n",
            "def rawCoverage : RawCoverage where\n",
            "  schemaVersion := 1\n",
            "  relationRows := {relation_rows}\n",
            "  relationColumns := {relation_columns}\n",
            "  sourceRows := {source_rows}\n",
            "  sourceColumns := {source_columns}\n",
            "  blockCount := {}\n",
            "  sourceBlockRows := {}\n",
            "  emittedBlockRows := {}\n",
            "  samplers := ["
        ),
        audits.len(),
        audits.iter().map(|audit| audit.source_rows().len()).sum::<usize>(),
        audits.iter().map(|audit| audit.emitted_rows().len()).sum::<usize>(),
        relation_rows = relation_rows,
        relation_columns = relation_columns,
        source_rows = source_rows,
        source_columns = source_columns,
    )
    .expect("render selection artifact header");
    for (index, sampler) in samplers.iter().enumerate() {
        let separator = if index == 0 { "    " } else { "  , " };
        writeln!(
            rendered,
            concat!(
                "{separator}{{ arm := {}, firstRewrite := {}, firstStage := {}, firstSourceRow := {},\n",
                "      firstEmittedRow := {}, firstSelectorColumn := {},\n",
                "      accepts := {:?},\n",
                "      prefixes := {:?},\n",
                "      symbols := {:?} }}"
            ),
            sampler.arm,
            sampler.first_rewrite,
            sampler.first_stage,
            sampler.first_source_row,
            sampler.first_emitted_row,
            sampler.first_selector_column,
            sampler.accepts,
            sampler.prefixes,
            sampler.symbols,
            separator = separator,
        )
        .expect("render selection sampler");
    }
    writeln!(
        rendered,
        "  ]\n\nend Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryRadixFourFirstAcceptedSelection"
    )
    .expect("render selection artifact footer");
    rendered
}

fn assert_radix_four_selection_artifact_matches_committed(
    relation_rows: usize,
    relation_columns: usize,
    source_rows: usize,
    source_columns: usize,
    audits: &[SelectiveFirstAcceptedSelectionAudit],
) {
    let rendered =
        render_radix_four_selection_artifact(relation_rows, relation_columns, source_rows, source_columns, audits);
    let path = format!("{}{}", env!("CARGO_MANIFEST_DIR"), RADIX_FOUR_SELECTION_ARTIFACT_PATH);
    let committed = std::fs::read_to_string(&path).unwrap_or_default();
    if rendered != committed {
        let expected = format!("{path}.expected");
        std::fs::write(&expected, rendered).expect("write reviewed radix-four selection artifact");
        panic!("radix-four selection artifact drifted; wrote {expected}. Inspect and promote it explicitly");
    }
}

fn candidate_params() -> Params {
    let base = config::ccs_params(1 << 25, 1 << 25, 13, 8).expect("production WASM parameters");
    let raw = neo_params::NeoParams::new(
        base.q(),
        base.eta(),
        base.d(),
        base.kappa(),
        base.m(),
        4,
        7,
        base.T(),
        base.extension_degree(),
        114,
    )
    .expect("radix-four production-width candidate");
    Params::test_only_from_neo_params(raw)
}

#[test]
#[ignore = "complete production-width radix-four relation census; run explicitly"]
fn wasm_nebula_radix_four_candidate_census() {
    const TARGET_DOMAIN: usize = 1 << 24;

    let wasm = wat::parse_str(PROFILE_WAT).expect("valid profile WAT");
    let artifacts = neo_wasm::extract_wasm_program_artifacts(&wasm).expect("program artifacts");
    let run = neo_wasm::collect_wasmtime_steps(&wasm, "main", &[]).expect("wasmtime trace");
    let profile = performance_profile(3);
    let scan_steps = profile.memory().steps_per_segment();
    let params = candidate_params();
    let started = Instant::now();
    let prep = neo_wasm::nebula::preprocess_seeded(
        params.clone(),
        profile,
        &artifacts,
        &run.initial_locals,
        common::single_function_entry_pc(&artifacts),
        0x57a5_7044,
    )
    .expect("radix-four production-width preprocessing");
    let relation = prep.inner().relation();
    let structure = relation.structure();
    let relation_receipt = prep
        .inner()
        .relation_artifact_receipt()
        .expect("exact radix-four recursive relation artifact receipt");
    let width = relation.low_norm_width_audit().expect("live width audit");
    let snapshot = relation
        .selective_snapshot()
        .expect("checked production selective snapshot");
    let (recursive_domain_run_index, recursive_domain) = snapshot
        .compiler_audit()
        .rows()
        .emitted_runs()
        .iter()
        .enumerate()
        .find(|(_, run)| run.family() == SelectiveEmittedRowFamily::ArmDomain && run.arm() == Some(1))
        .expect("recursive arm-domain run");
    let selector_column = snapshot.selector_cols()[1];
    let recursive_mapping = &snapshot.compiler_audit().rows().arms()[1];
    let pair_row = recursive_mapping
        .centered_domain_pair_row()
        .expect("recursive domain has a centered pair row");
    let tail_row = recursive_mapping
        .centered_domain_tail_row()
        .expect("recursive domain has an odd centered tail row");
    assert!(recursive_domain.emitted_rows().contains(&pair_row));
    assert!(recursive_domain.emitted_rows().contains(&tail_row));
    super::centered_domain_artifact::assert_artifact_matches_committed(
        snapshot.structure(),
        recursive_domain_run_index,
        pair_row,
        tail_row,
        selector_column,
    );
    let coverage = snapshot
        .selector_gate_coverage()
        .expect("exact production selector coverage");
    let selection_audits = snapshot.compiler_audit().first_accepted_selections();
    assert_eq!(selection_audits.len(), 432, "production selection audit count drifted");
    let source_shape = relation.field_arm_shapes()[1];
    assert_radix_four_selection_artifact_matches_committed(
        structure.n,
        structure.m,
        source_shape.rows,
        source_shape.columns,
        selection_audits,
    );
    for (arm_index, arm) in snapshot.arms().enumerate() {
        let mut direct_words = 0usize;
        let mut direct_runs = 0usize;
        let mut previous_direct = None::<(usize, usize)>;
        for field in 1..arm.field_count() {
            let Some(slot) = arm.slot(field) else {
                continue;
            };
            if slot.len() != 23 || arm.coordinate_alias(field).is_some() || arm.equality_source(field).is_some() {
                continue;
            }
            let extends_run = previous_direct.is_some_and(|(previous_field, previous_start)| {
                previous_field + 1 == field && previous_start + 23 == slot.start()
            });
            if !extends_run {
                direct_runs += 1;
            }
            direct_words += 1;
            previous_direct = Some((field, slot.start()));
        }

        let mut derived_words = 0usize;
        let mut derived_runs = 0usize;
        let mut previous_derived_end = None::<usize>;
        for derived in arm.derived_product_sums() {
            let slot = derived.slot();
            assert_eq!(slot.len(), 23, "radix-four derived field width");
            if previous_derived_end != Some(slot.start()) {
                derived_runs += 1;
            }
            derived_words += 1;
            previous_derived_end = Some(slot.start() + slot.len());
        }
        println!(
            "RADIX_FOUR_SEPTENARY_RUN_JSON={{\"arm\":{arm_index},\"direct_words\":{direct_words},\"direct_runs\":{direct_runs},\"derived_words\":{derived_words},\"derived_runs\":{derived_runs}}}",
        );
    }
    let padded = structure.n.max(structure.m).next_power_of_two();
    params
        .validate_ccs_shape(structure.n, structure.m, structure.t(), structure.max_degree())
        .expect("radix-four candidate security census");
    assert_eq!((params.b(), params.k_rho(), params.big_b()), (4, 7, 16_384));
    assert_eq!(
        (structure.n, structure.m),
        (8_102_331, 12_288_726),
        "radix-four candidate relation drifted"
    );
    assert_eq!((structure.t(), structure.max_degree()), (13, 8));
    assert_eq!(relation_receipt.logical_rows(), structure.n as u64);
    assert_eq!(relation_receipt.assignment_fields(), structure.m as u64);
    assert_eq!(relation_receipt.padded_rows(), TARGET_DOMAIN as u64);
    assert_eq!(relation_receipt.row_variables(), 24);
    let delayed_suffix = delayed_nebula_public_suffix_len(
        prep.inner()
            .prep
            .nebula()
            .expect("radix-four Nebula verifier context")
            .stacks,
    );
    assert_eq!(F_PRIME_PUBLIC_INPUT_LEN, 257, "core F-prime logical carrier drifted");
    assert_eq!(delayed_suffix, 2_169, "delayed Nebula suffix drifted");
    assert_eq!(
        snapshot
            .compiler_audit()
            .layout()
            .logical_public_input_len(),
        F_PRIME_PUBLIC_INPUT_LEN + delayed_suffix,
        "composed field-native F-prime public carrier drifted"
    );
    assert_eq!(
        snapshot.compiler_audit().layout().public_input_len(),
        2_430,
        "low-norm public carrier drifted"
    );
    assert_eq!(relation_receipt.public_field_width(), Some(2_430));
    assert_eq!(relation_receipt.semantic_matrix_count(), 13);
    assert_eq!(relation_receipt.joint_matrix_count(), 14);
    assert_eq!(relation_receipt.polynomial_degree(), 8);
    assert_eq!(width.total_coordinates.div_ceil(D) * D, structure.m);
    assert_eq!((coverage.rows(), coverage.columns()), (structure.n, structure.m));
    assert_eq!(coverage.selector_columns(), snapshot.selector_cols());
    assert_eq!(
        coverage.owner_runs(),
        snapshot.compiler_audit().rows().emitted_runs(),
        "production coverage must preserve the complete compiler row ledger"
    );
    assert_eq!(
        coverage
            .owner_runs()
            .iter()
            .filter(|owner| !owner.emitted_rows().is_empty())
            .count(),
        coverage.gate_runs().len(),
        "every nonempty production owner run must have one exact selector gate"
    );
    assert_eq!(
        (
            coverage.owner_runs().len(),
            coverage.gate_runs().len(),
            coverage.coalesced_owner_gate_runs().len(),
        ),
        (185_526, 180_665, 14),
        "radix-four selector-coverage census drifted"
    );
    let definition_arms = snapshot.compiler_audit().source_arm_linear_definitions();
    assert_eq!(definition_arms.len(), snapshot.arm_count());
    assert_eq!(
        definition_arms.iter().map(Vec::len).collect::<Vec<_>>(),
        [2_071, 86_880],
        "radix-four source-definition census drifted"
    );
    let definition_run_counts = definition_arms
        .iter()
        .map(|definitions| affine_definition_run_count(definitions))
        .collect::<Vec<_>>();
    println!(
        "RADIX_FOUR_DEFINITION_JSON={{\"definitions\":{:?},\"affine_runs\":{:?},\"terms\":{:?}}}",
        definition_arms.iter().map(Vec::len).collect::<Vec<_>>(),
        definition_run_counts,
        definition_arms
            .iter()
            .map(|definitions| definitions
                .iter()
                .map(|definition| definition.terms().len())
                .sum::<usize>())
            .collect::<Vec<_>>(),
    );
    assert!(
        structure.n <= TARGET_DOMAIN,
        "radix-four rows exceed the 2^24 production target: {}",
        structure.n
    );
    assert!(
        structure.m <= TARGET_DOMAIN,
        "radix-four columns exceed the 2^24 production target: {}",
        structure.m
    );
    assert_eq!(
        padded, TARGET_DOMAIN,
        "radix-four candidate no longer selects the intended 2^24 joint domain"
    );
    assert_radix_four_selector_coverage_artifact_matches_committed(&params, scan_steps, &coverage);
    let recursive = width.arms.last().expect("recursive width audit");
    let classified_without_canonical_ternary = recursive.unit_columns
        + recursive.balanced_columns * RADIX_FOUR_GENERAL_FIELD_WIDTH
        + recursive.binary_columns * BINARY_FIELD_WIDTH;
    let canonical_ternary_coordinates = recursive
        .retained_coordinates_before_aliases
        .checked_sub(classified_without_canonical_ternary)
        .expect("known radix-four width classes exceed the retained coordinate census");
    assert_eq!(
        canonical_ternary_coordinates % CANONICAL_TERNARY_FIELD_WIDTH,
        0,
        "radix-four retained coordinates contain an unknown width class"
    );
    let canonical_ternary_columns = canonical_ternary_coordinates / CANONICAL_TERNARY_FIELD_WIDTH;
    assert_eq!(
        recursive.retained_coordinates_before_aliases,
        recursive.unit_columns
            + recursive.balanced_columns * RADIX_FOUR_GENERAL_FIELD_WIDTH
            + canonical_ternary_columns * CANONICAL_TERNARY_FIELD_WIDTH
            + recursive.binary_columns * BINARY_FIELD_WIDTH,
        "radix-four width classes do not reconstruct the retained coordinate census"
    );
    assert_eq!(
        recursive.decomposition_aliases,
        canonical_ternary_columns * CANONICAL_TERNARY_FIELD_WIDTH + recursive.binary_columns * BINARY_FIELD_WIDTH,
        "canonical ternary and binary child columns do not alias their complete source words"
    );
    println!(
        "RADIX_FOUR_WIDTH_JSON={{\"shared\":{},\"branch_source_columns\":{},\"eliminated_columns\":{},\"unit_columns\":{},\"general_septenary_columns\":{},\"canonical_ternary_columns\":{},\"binary_columns\":{},\"retained_before_aliases\":{},\"decomposition_aliases\":{},\"equality_aliases\":{},\"branch_coordinates\":{},\"derived_product_sums\":{},\"derived_coordinates\":{},\"total_branch_coordinates\":{}}}",
        width.shared_private_coordinates,
        recursive.branch_source_columns,
        recursive.eliminated_columns,
        recursive.unit_columns,
        recursive.balanced_columns,
        canonical_ternary_columns,
        recursive.binary_columns,
        recursive.retained_coordinates_before_aliases,
        recursive.decomposition_aliases,
        recursive.equality_aliases,
        recursive.branch_coordinates,
        recursive.derived_product_sums,
        recursive.derived_coordinates,
        recursive.total_branch_coordinates,
    );
    let decoder_dispositions = recursive.physical_stages.iter().fold(
        [0usize; 5],
        |[direct, decomposition_alias, equality_alias, linear_definition, trace_eliminated], stage| {
            [
                direct + stage.direct_columns,
                decomposition_alias + stage.decomposition_alias_columns,
                equality_alias + stage.equality_alias_columns,
                linear_definition + stage.linear_definition_columns,
                trace_eliminated + stage.trace_eliminated_columns,
            ]
        },
    );
    assert_eq!(
        decoder_dispositions.iter().sum::<usize>(),
        recursive.branch_source_columns,
        "radix-four stage decoder census does not cover every recursive source field"
    );
    assert_eq!(
        decoder_dispositions[3],
        definition_arms[1].len(),
        "radix-four stage decoder census does not cover every recursive affine definition"
    );
    assert_eq!(
        decoder_dispositions[3] + decoder_dispositions[4],
        recursive.eliminated_columns,
        "radix-four eliminated-field census disagrees with decoder dispositions"
    );
    println!(
        "RADIX_FOUR_SOURCE_STAGE_JSON={{\"stages\":{},\"source_fields\":{},\"direct\":{},\"decomposition_alias\":{},\"equality_alias\":{},\"linear_definition\":{},\"trace_eliminated\":{}}}",
        recursive.physical_stages.len(),
        recursive.branch_source_columns,
        decoder_dispositions[0],
        decoder_dispositions[1],
        decoder_dispositions[2],
        decoder_dispositions[3],
        decoder_dispositions[4],
    );
    for (arm_index, arm) in width.arms.iter().enumerate() {
        let domain = arm
            .physical_stages
            .iter()
            .fold([0usize; 4], |mut total, stage| {
                total[0] += stage.source_boolean_coordinates;
                total[1] += stage.outer_norm_coordinates;
                total[2] += stage.boolean_domain_rows;
                total[3] += stage.centered_unit_domain_coordinates;
                total
            });
        assert_eq!(
            domain.iter().sum::<usize>(),
            arm.branch_coordinates,
            "radix-four domain owners do not cover arm {arm_index} coordinates"
        );
        let emitted_domain_rows = snapshot
            .compiler_audit()
            .rows()
            .emitted_runs()
            .iter()
            .filter(|run| run.family() == SelectiveEmittedRowFamily::ArmDomain && run.arm() == Some(arm_index))
            .map(|run| run.emitted_rows().len())
            .sum::<usize>();
        let centered_unit_domain_rows = domain[3].div_ceil(2);
        assert_eq!(
            domain[2] + centered_unit_domain_rows,
            emitted_domain_rows,
            "arm domain row census drifted"
        );
        println!(
            "RADIX_FOUR_DOMAIN_JSON={{\"arm\":{arm_index},\"source_boolean_coordinates\":{},\"outer_norm_coordinates\":{},\"boolean_domain_rows\":{},\"centered_unit_domain_coordinates\":{},\"centered_unit_domain_rows\":{centered_unit_domain_rows},\"emitted_domain_rows\":{emitted_domain_rows}}}",
            domain[0], domain[1], domain[2], domain[3],
        );
    }
    let mut unowned_empty_stages = 0usize;
    for stage in &recursive.physical_stages {
        let owner_count = WIDTH_OWNERS
            .iter()
            .filter(|owner| stage.path.starts_with(owner.path_prefix))
            .count();
        assert!(
            owner_count <= 1,
            "radix-four physical stage has overlapping top-level owners: {}",
            stage.path,
        );
        if stage.source_column_count != 0 || stage.allocated_coordinates != 0 {
            assert_eq!(
                owner_count, 1,
                "nonempty radix-four physical stage has no top-level owner: {}",
                stage.path,
            );
        } else if owner_count == 0 {
            unowned_empty_stages += 1;
        }
    }
    let owner_censuses = WIDTH_OWNERS
        .iter()
        .copied()
        .map(|owner| {
            let totals = recursive
                .physical_stages
                .iter()
                .filter(|stage| stage.path.starts_with(owner.path_prefix))
                .fold([0usize; 12], |mut total, stage| {
                    total[0] += 1;
                    total[1] += stage.source_column_count;
                    total[2] += stage.direct_columns;
                    total[3] += stage.decomposition_alias_columns;
                    total[4] += stage.equality_alias_columns;
                    total[5] += stage.linear_definition_columns;
                    total[6] += stage.trace_eliminated_columns;
                    total[7] += stage.allocated_coordinates;
                    total[8] += stage.source_boolean_coordinates;
                    total[9] += stage.outer_norm_coordinates;
                    total[10] += stage.boolean_domain_rows;
                    total[11] += stage.centered_unit_domain_coordinates;
                    total
                });
            SourceStageOwnerCensus {
                owner,
                stages: totals[0],
                source_fields: totals[1],
                direct: totals[2],
                decomposition_alias: totals[3],
                equality_alias: totals[4],
                linear_definition: totals[5],
                trace_eliminated: totals[6],
                allocated_coordinates: totals[7],
                source_boolean_coordinates: totals[8],
                outer_norm_coordinates: totals[9],
                boolean_domain_rows: totals[10],
                centered_unit_domain_coordinates: totals[11],
            }
        })
        .collect::<Vec<_>>();
    let mut classified_coordinates = 0usize;
    let mut classified_source_fields = 0usize;
    for census in &owner_censuses {
        classified_coordinates += census.allocated_coordinates;
        classified_source_fields += census.source_fields;
        let owner = census.owner.name;
        println!(
            "RADIX_FOUR_OWNER_JSON={{\"owner\":\"{owner}\",\"stages\":{},\"source_fields\":{},\"direct\":{},\"decomposition_alias\":{},\"equality_alias\":{},\"linear_definition\":{},\"trace_eliminated\":{},\"allocated_coordinates\":{},\"source_boolean_coordinates\":{},\"outer_norm_coordinates\":{},\"boolean_domain_rows\":{},\"centered_unit_domain_coordinates\":{}}}",
            census.stages,
            census.source_fields,
            census.direct,
            census.decomposition_alias,
            census.equality_alias,
            census.linear_definition,
            census.trace_eliminated,
            census.allocated_coordinates,
            census.source_boolean_coordinates,
            census.outer_norm_coordinates,
            census.boolean_domain_rows,
            census.centered_unit_domain_coordinates,
        );
    }
    let pi_rlc_stage_families = stage_family_census(
        recursive,
        snapshot.compiler_audit().rows().emitted_runs(),
        1,
        "nifs.pi_rlc",
    );
    let pi_rlc_family_totals =
        pi_rlc_stage_families
            .values()
            .fold(StageFamilyTotals::default(), |mut total, family| {
                total.occurrences += family.occurrences;
                total.source_rows += family.source_rows;
                total.emitted_rows += family.emitted_rows;
                total.source_fields += family.source_fields;
                total.direct += family.direct;
                total.decomposition_alias += family.decomposition_alias;
                total.equality_alias += family.equality_alias;
                total.linear_definition += family.linear_definition;
                total.trace_eliminated += family.trace_eliminated;
                total.allocated_coordinates += family.allocated_coordinates;
                total.source_boolean_coordinates += family.source_boolean_coordinates;
                total.outer_norm_coordinates += family.outer_norm_coordinates;
                total.boolean_domain_rows += family.boolean_domain_rows;
                total.centered_unit_domain_coordinates += family.centered_unit_domain_coordinates;
                total
            });
    let pi_rlc_owner = owner_censuses
        .iter()
        .find(|census| census.owner.name == "pi_rlc")
        .expect("PiRLC owner census");
    assert_eq!(
        (
            pi_rlc_family_totals.occurrences,
            pi_rlc_family_totals.source_fields,
            pi_rlc_family_totals.direct,
            pi_rlc_family_totals.decomposition_alias,
            pi_rlc_family_totals.equality_alias,
            pi_rlc_family_totals.linear_definition,
            pi_rlc_family_totals.trace_eliminated,
            pi_rlc_family_totals.allocated_coordinates,
            pi_rlc_family_totals.source_boolean_coordinates,
            pi_rlc_family_totals.outer_norm_coordinates,
            pi_rlc_family_totals.boolean_domain_rows,
            pi_rlc_family_totals.centered_unit_domain_coordinates,
        ),
        (
            pi_rlc_owner.stages,
            pi_rlc_owner.source_fields,
            pi_rlc_owner.direct,
            pi_rlc_owner.decomposition_alias,
            pi_rlc_owner.equality_alias,
            pi_rlc_owner.linear_definition,
            pi_rlc_owner.trace_eliminated,
            pi_rlc_owner.allocated_coordinates,
            pi_rlc_owner.source_boolean_coordinates,
            pi_rlc_owner.outer_norm_coordinates,
            pi_rlc_owner.boolean_domain_rows,
            pi_rlc_owner.centered_unit_domain_coordinates,
        ),
        "PiRLC exact-path families do not reconcile with the top-level owner"
    );
    println!(
        "RADIX_FOUR_PI_RLC_FAMILY_TOTAL_JSON={{\"families\":{},\"occurrences\":{},\"source_rows\":{},\"emitted_rows\":{},\"source_fields\":{},\"direct\":{},\"decomposition_alias\":{},\"equality_alias\":{},\"linear_definition\":{},\"trace_eliminated\":{},\"allocated_coordinates\":{},\"source_boolean_coordinates\":{},\"outer_norm_coordinates\":{},\"boolean_domain_rows\":{},\"centered_unit_domain_coordinates\":{}}}",
        pi_rlc_stage_families.len(),
        pi_rlc_family_totals.occurrences,
        pi_rlc_family_totals.source_rows,
        pi_rlc_family_totals.emitted_rows,
        pi_rlc_family_totals.source_fields,
        pi_rlc_family_totals.direct,
        pi_rlc_family_totals.decomposition_alias,
        pi_rlc_family_totals.equality_alias,
        pi_rlc_family_totals.linear_definition,
        pi_rlc_family_totals.trace_eliminated,
        pi_rlc_family_totals.allocated_coordinates,
        pi_rlc_family_totals.source_boolean_coordinates,
        pi_rlc_family_totals.outer_norm_coordinates,
        pi_rlc_family_totals.boolean_domain_rows,
        pi_rlc_family_totals.centered_unit_domain_coordinates,
    );
    for (path, family) in &pi_rlc_stage_families {
        println!(
            "RADIX_FOUR_PI_RLC_FAMILY_JSON={{\"path\":\"{path}\",\"occurrences\":{},\"source_rows\":{},\"emitted_rows\":{},\"source_fields\":{},\"direct\":{},\"decomposition_alias\":{},\"equality_alias\":{},\"linear_definition\":{},\"trace_eliminated\":{},\"allocated_coordinates\":{},\"source_boolean_coordinates\":{},\"outer_norm_coordinates\":{},\"boolean_domain_rows\":{},\"centered_unit_domain_coordinates\":{}}}",
            family.occurrences,
            family.source_rows,
            family.emitted_rows,
            family.source_fields,
            family.direct,
            family.decomposition_alias,
            family.equality_alias,
            family.linear_definition,
            family.trace_eliminated,
            family.allocated_coordinates,
            family.source_boolean_coordinates,
            family.outer_norm_coordinates,
            family.boolean_domain_rows,
            family.centered_unit_domain_coordinates,
        );
    }
    let pi_ccs_stage_families = stage_family_census(
        recursive,
        snapshot.compiler_audit().rows().emitted_runs(),
        1,
        "nifs.pi_ccs",
    );
    let pi_ccs_centered_rows = pi_ccs_stage_families
        .values()
        .map(|family| family.centered_unit_domain_coordinates)
        .sum::<usize>();
    assert_eq!(
        pi_ccs_centered_rows,
        owner_censuses
            .iter()
            .find(|census| census.owner.name == "pi_ccs")
            .expect("PiCCS owner census")
            .centered_unit_domain_coordinates,
        "PiCCS centered-domain families do not reconcile with the top-level owner"
    );
    for (path, family) in pi_ccs_stage_families
        .iter()
        .filter(|(_, family)| family.centered_unit_domain_coordinates != 0)
    {
        println!(
            "RADIX_FOUR_PI_CCS_CENTERED_FAMILY_JSON={{\"path\":\"{path}\",\"occurrences\":{},\"source_rows\":{},\"emitted_rows\":{},\"direct\":{},\"centered_unit_domain_coordinates\":{}}}",
            family.occurrences,
            family.source_rows,
            family.emitted_rows,
            family.direct,
            family.centered_unit_domain_coordinates,
        );
    }
    assert_eq!(
        classified_source_fields, recursive.branch_source_columns,
        "radix-four source owner prefixes do not cover the complete recursive branch"
    );
    assert_eq!(
        classified_coordinates, recursive.branch_coordinates,
        "radix-four width owner prefixes do not cover the complete recursive branch"
    );
    assert_radix_four_source_stage_coverage_artifact_matches_committed(
        recursive.branch_source_columns,
        recursive.physical_stages.len(),
        unowned_empty_stages,
        decoder_dispositions,
        recursive.branch_coordinates,
        &owner_censuses,
    );
    println!(
        "RADIX_FOUR_CANDIDATE_JSON={{\"rows\":{},\"columns\":{},\"ell\":{},\"kappa\":{},\"lambda\":{},\"preprocess_ms\":{:.3}}}",
        structure.n,
        structure.m,
        padded.ilog2(),
        params.kappa(),
        params.lambda(),
        ms(started.elapsed()),
    );
    println!(
        "RADIX_FOUR_COVERAGE_JSON={{\"owner_runs\":{},\"nonempty_owner_runs\":{},\"gate_runs\":{},\"coalesced_runs\":{},\"polynomial_terms\":{}}}",
        coverage.owner_runs().len(),
        coverage
            .owner_runs()
            .iter()
            .filter(|owner| !owner.emitted_rows().is_empty())
            .count(),
        coverage.gate_runs().len(),
        coverage.coalesced_owner_gate_runs().len(),
        coverage.polynomial_terms().len(),
    );
}

#[test]
#[ignore = "writes and restores the production-width radix-four artifacts"]
fn wasm_nebula_radix_four_artifact_restore_profile() {
    let wasm = wat::parse_str(PROFILE_WAT).expect("valid profile WAT");
    let artifacts = neo_wasm::extract_wasm_program_artifacts(&wasm).expect("program artifacts");
    let run = neo_wasm::collect_wasmtime_steps(&wasm, "main", &[]).expect("wasmtime trace");
    let profile = performance_profile(3);
    let entry_pc = common::single_function_entry_pc(&artifacts);
    let params = candidate_params();
    neo_ajtai::set_global_pp_seeded(D, params.kappa() as usize, 12_288_726usize.div_ceil(D), [0x46; 32])
        .expect("install verifier-owned radix-four Ajtai setup");

    let started = Instant::now();
    let live = neo_wasm::nebula::preprocess(params.clone(), profile, &artifacts, &run.initial_locals, entry_pc)
        .expect("live radix-four preprocessing");
    let preprocess_elapsed = started.elapsed();
    let structure = live.inner().relation().structure();
    let shape = (structure.n, structure.m, structure.t());
    let matrix_digest = live.inner().prep.pi_ccs_header_bundle();
    let relation_receipt = live
        .inner()
        .relation_artifact_receipt()
        .expect("live radix-four recursive relation artifact receipt");

    let cache_path = std::env::temp_dir().join(format!(
        "nightstream-radix-four-superneo-cache-{}.bin",
        std::process::id()
    ));
    let started = Instant::now();
    let mut writer = BufWriter::new(File::create(&cache_path).expect("create radix-four cache artifact"));
    let cache_receipt = live
        .inner()
        .prep
        .optimized_cache()
        .superneo()
        .write_artifact(&mut writer, matrix_digest)
        .expect("write radix-four cache artifact");
    writer.flush().expect("flush radix-four cache artifact");
    drop(writer);
    let cache_write_elapsed = started.elapsed();

    let encoder_path = std::env::temp_dir().join(format!(
        "nightstream-radix-four-fprime-encoder-{}.bin",
        std::process::id()
    ));
    let started = Instant::now();
    let mut writer = BufWriter::new(File::create(&encoder_path).expect("create radix-four encoder artifact"));
    let encoder_receipt = live
        .inner()
        .write_encoder_artifact(&mut writer)
        .expect("write radix-four encoder artifact");
    writer.flush().expect("flush radix-four encoder artifact");
    drop(writer);
    let encoder_write_elapsed = started.elapsed();

    let cache_limits = neo_reductions::superneo_eval::SuperneoCacheArtifactLimits::new(
        cache_receipt.artifact_bytes(),
        shape.0,
        shape.1.div_ceil(D) * D,
        shape.2,
    );
    let encoder_limits = neo_fold_clean::frontends::r1cs_f_prime::LowNormEncoderArtifactLimits::new(
        encoder_receipt.encoder().artifact_bytes(),
        shape.0,
        shape.1,
        shape.2,
        2,
        encoder_receipt
            .encoder()
            .arm_field_counts()
            .iter()
            .copied()
            .max()
            .expect("encoder arm fields") as usize,
        encoder_receipt
            .encoder()
            .arm_derived_counts()
            .iter()
            .copied()
            .max()
            .expect("encoder derived fields") as usize,
    );
    let started = Instant::now();
    let (cache, encoder) = std::thread::scope(|scope| {
        let cache = scope.spawn(|| {
            neo_reductions::superneo_eval::SuperneoEvalCache::read_verified_artifact(
                BufReader::new(File::open(&cache_path).expect("open radix-four cache artifact")),
                &cache_receipt,
                cache_limits,
            )
            .expect("load radix-four cache artifact")
        });
        let encoder = scope.spawn(|| {
            neo_fold_clean::frontends::nebula::f_prime::VerifiedNebulaFPrimeEncoderArtifact::read(
                BufReader::new(File::open(&encoder_path).expect("open radix-four encoder artifact")),
                &encoder_receipt,
                encoder_limits,
            )
            .expect("load radix-four encoder artifact")
        });
        (
            cache.join().expect("radix-four cache loader"),
            encoder.join().expect("radix-four encoder loader"),
        )
    });
    let parallel_load_elapsed = started.elapsed();
    std::fs::remove_file(&cache_path).expect("remove radix-four cache artifact");
    std::fs::remove_file(&encoder_path).expect("remove radix-four encoder artifact");

    let started = Instant::now();
    let prepared = neo_wasm::nebula::prepare_profile_with_artifacts(
        params,
        profile,
        &artifacts,
        &run.initial_locals,
        entry_pc,
        cache,
        encoder,
    )
    .expect("restore radix-four prepared profile");
    let restore_elapsed = started.elapsed();
    assert!(prepared
        .inner()
        .relation()
        .structure()
        .is_verifier_artifact_header());
    assert_eq!(
        (
            prepared.inner().relation().structure().n,
            prepared.inner().relation().structure().m,
            prepared.inner().relation().structure().t(),
        ),
        shape,
    );

    let started = Instant::now();
    let bound = prepared
        .bind_program(&artifacts, &run.initial_locals, entry_pc)
        .expect("bind program to restored radix-four profile");
    let bind_elapsed = started.elapsed();
    assert_eq!(
        bound.inner().prep.pi_ccs_header_bundle(),
        matrix_digest,
        "restored radix-four profile matrix authority"
    );
    assert_eq!(
        bound
            .inner()
            .relation_artifact_receipt()
            .expect("restored radix-four recursive relation artifact receipt"),
        relation_receipt,
        "restored radix-four relation must have the exact live receipt"
    );
    println!(
        "RADIX_FOUR_ARTIFACT_JSON={{\"rows\":{},\"columns\":{},\"matrices\":{},\"cache_bytes\":{},\"encoder_bytes\":{},\"preprocess_ms\":{:.3},\"cache_write_ms\":{:.3},\"encoder_write_ms\":{:.3},\"parallel_load_ms\":{:.3},\"restore_ms\":{:.3},\"cold_profile_ms\":{:.3},\"bind_ms\":{:.3}}}",
        shape.0,
        shape.1,
        shape.2,
        cache_receipt.artifact_bytes(),
        encoder_receipt.encoder().artifact_bytes(),
        ms(preprocess_elapsed),
        ms(cache_write_elapsed),
        ms(encoder_write_elapsed),
        ms(parallel_load_elapsed),
        ms(restore_elapsed),
        ms(parallel_load_elapsed + restore_elapsed),
        ms(bind_elapsed),
    );
}

#[test]
#[cfg(all(feature = "metal", target_vendor = "apple"))]
#[ignore = "radix-four Metal proof covering every F-prime lifecycle branch"]
fn wasm_nebula_radix_four_all_branch_metal_profile() {
    let wall_started = Instant::now();
    let wasm = wat::parse_str(PROFILE_WAT).expect("valid profile WAT");
    let artifacts = neo_wasm::extract_wasm_program_artifacts(&wasm).expect("program artifacts");
    let run = neo_wasm::collect_wasmtime_steps(&wasm, "main", &[]).expect("wasmtime trace");
    let trace = neo_wasm::traces_from_wasmtime_steps(&run.steps).expect("normalized trace");
    let memory = neo_fold_clean::frontends::nebula::layout::NebulaParams::new(11, 11, 64, 1024, 16)
        .expect("four-fold reduced Nebula scan");
    let profile = neo_wasm::WasmNebulaProfile::test_profile_with_schedule(memory, 3);
    assert_eq!(profile.memory().steps_per_segment(), 4);

    let started = Instant::now();
    let prep = neo_wasm::nebula::preprocess_seeded_reduced_memory_test_only(
        candidate_params(),
        profile,
        &artifacts,
        &run.initial_locals,
        common::single_function_entry_pc(&artifacts),
        0x57a5_7045,
    )
    .expect("radix-four all-branch preprocessing");
    let preprocess_elapsed = started.elapsed();
    let structure = prep.inner().relation().structure();

    let mut prover = MetalNifsProver::new().expect("Metal prover");
    let started = Instant::now();
    prover
        .prepare_static(
            &prep.inner().prep.log,
            structure,
            prep.inner().prep.optimized_cache(),
            prep.inner().prep.nebula().map(|config| &config.scheme),
        )
        .expect("prepare radix-four Metal state");
    let metal_prepare_elapsed = started.elapsed();
    prover.session().reset_activity();

    let started = Instant::now();
    let proof = neo_wasm::nebula::prove_with_nifs_adapter(&prep, &mut prover, &trace)
        .expect("radix-four all-branch Metal proof");
    let prove_elapsed = started.elapsed();
    assert_eq!(proof.inner().state.step_count, 4);
    assert_eq!(proof.inner().state.chunk_count, 4);
    assert!(proof.inner().final_fold.is_some());
    assert!(prover.session().activity().dispatches > 0);

    let started = Instant::now();
    neo_wasm::nebula::verify_with_witness_opening_backend(&prep, &proof, common::final_state(&trace), &mut prover)
        .expect("radix-four all-branch Metal verification");
    let verify_elapsed = started.elapsed();
    let padded = structure.n.max(structure.m).next_power_of_two();
    println!(
        "RADIX_FOUR_ALL_BRANCH_JSON={{\"rows\":{},\"columns\":{},\"ell\":{},\"folds\":4,\"preprocess_ms\":{:.3},\"metal_prepare_ms\":{:.3},\"prove_ms\":{:.3},\"verify_ms\":{:.3},\"total_ms\":{:.3}}}",
        structure.n,
        structure.m,
        padded.ilog2(),
        ms(preprocess_elapsed),
        ms(metal_prepare_elapsed),
        ms(prove_elapsed),
        ms(verify_elapsed),
        ms(wall_started.elapsed()),
    );
}

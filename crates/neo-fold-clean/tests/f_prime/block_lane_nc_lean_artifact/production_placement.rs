//! Concrete production placement for the terminal raw-old-block emitter.
//!
//! The placement is captured after the real fixed-profile last-step and
//! terminal-fold call schedule. Capture stops before the projection and Ajtai
//! row loops, so the exporter retains exact absolute columns/ranges without
//! materializing the 24,185,169-row family.

use std::fmt::Write as _;

use neo_fold_clean::engine::decider::__test_isolation::capture_last_step_terminal_prefix;
use neo_fold_clean::engine::r1cs_circuit::builder::RowFamilyRange;
use neo_fold_clean::engine::r1cs_circuit::{
    TerminalPendingProjectionAudit, RAW_OLD_BLOCK_CHILD_COUNT, RAW_OLD_BLOCK_PENDING_JOIN_ID,
};
use neo_fold_clean::frontends::r1cs_f_prime::ivc::{
    R1csIvcBranch, R1csIvcPostPiDecExecutionAudit, R1csIvcPreprocessing, R1csIvcRawOldBlockFieldDecoding,
    R1csIvcRawOldBlockProfile,
};
use neo_reductions::optimized_engine::PiCcsProofVariant;
use p3_field::PrimeField64;
use serde_json::{json, Value};

use super::{generated_header, GeneratedLeanFile, GENERATED_ROOT, NAMESPACE_ROOT};

const PROFILE_TAG: usize = 1;
const RECURSIVE_SELECTOR_TAG: usize = 2;
const RADIX: u32 = 2;
const LOGICAL_WIDTH: usize = 11_437_038;
const ACTIVE_LANES: usize = 54;
const PADDED_LANES: usize = 64;
const BLOCK_VARIABLES: usize = 19;
const TENSOR_VARIABLES: usize = 18;
const BLOCK_COUNT: usize = 211_797;
const FINAL_SCALE_ROWS: usize = 270;
const DERIVED_COLUMNS: usize = 24_185_061;
const TOTAL_ROWS: usize = 24_185_169;

#[derive(Clone)]
pub(crate) struct ProductionPlacementCertificate {
    audit: TerminalPendingProjectionAudit,
    relation_matrix_count: usize,
    terminal_claim_count: usize,
    terminal_evaluation_count: usize,
    prefix_rows: usize,
    prefix_columns: usize,
    prefix_nonzero_entries: usize,
    row_families: Vec<RowFamilyRange>,
    old_block: Vec<[usize; 2]>,
    parent: Vec<[usize; 2]>,
    final_witness_first: Vec<usize>,
    ajtai_witness_first: Vec<usize>,
    profile_tag: usize,
    recursive_selector_tag: usize,
    selector_columns: Vec<usize>,
    selector_values: Vec<usize>,
}

impl ProductionPlacementCertificate {
    pub(crate) fn capture(
        prep: &R1csIvcPreprocessing,
        finalized: &neo_fold_clean::lifecycle::UncompressedAudit,
        execution: &R1csIvcPostPiDecExecutionAudit,
    ) -> Self {
        let (prefix, audit) = capture_last_step_terminal_prefix(&prep.prep, finalized)
            .expect("capture fixed-profile production terminal prefix");
        let running = finalized
            .proof
            .state
            .proof
            .running()
            .expect("selected terminal capture requires a materialized running accumulator");
        let terminal_evaluation_count = running
            .claims
            .first()
            .map(|claim| claim.y_ring.len())
            .expect("selected terminal capture requires at least one CE claim");
        assert!(
            running
                .claims
                .iter()
                .all(|claim| claim.y_ring.len() == terminal_evaluation_count),
            "selected terminal CE claims must use one matrix arity"
        );
        let selector_columns = execution
            .selector_writes()
            .iter()
            .map(|write| write.logical_column())
            .collect();
        let selector_values = execution
            .selector_writes()
            .iter()
            .map(|write| write.value().as_canonical_u64() as usize)
            .collect();
        let certificate = Self {
            relation_matrix_count: prep.prep.structure().t(),
            terminal_claim_count: running.claims.len(),
            terminal_evaluation_count,
            prefix_rows: prefix.builder.rows(),
            prefix_columns: prefix.builder.cols(),
            prefix_nonzero_entries: prefix.builder.nonzero_entries(),
            row_families: prefix.builder.row_family_ranges().to_vec(),
            old_block: audit.pending_old_block_cols.clone(),
            parent: audit.parent_y_zcol_cols.clone(),
            final_witness_first: audit.projection_child_witness_first_columns.clone(),
            ajtai_witness_first: audit.ajtai_child_witness_first_columns.clone(),
            profile_tag: PROFILE_TAG,
            recursive_selector_tag: RECURSIVE_SELECTOR_TAG,
            selector_columns,
            selector_values,
            audit,
        };
        certificate
            .validate_against_execution(prep, execution)
            .expect("exact production terminal placement");
        certificate
    }

    pub(crate) fn diagnostic_json(&self) -> Value {
        json!({
            "artifact_kind": "r1cs/f-prime-selected-terminal-prefix-diagnostic",
            "assurance_tier": "Rust diagnostic",
            "profile": {
                "relation": "fixed-point-selective-thirteen-matrix",
                "fixed_one": true,
                "layout": "plain",
                "carrier_width": 270,
                "relation_matrix_count": self.relation_matrix_count,
                "terminal_ce_claim_count": self.terminal_claim_count,
                "terminal_ce_evaluation_count": self.terminal_evaluation_count,
            },
            "prefix": {
                "rows": self.prefix_rows,
                "columns": self.prefix_columns,
                "retained_nonzero_coefficients": self.prefix_nonzero_entries,
                "row_family_metadata": self.row_families.iter().map(|family| json!({
                    "name": family.name,
                    "row_start": family.row_start,
                    "row_end": family.row_end,
                    "row_count": family.row_end - family.row_start,
                })).collect::<Vec<_>>(),
                "capture_mode": "witness-plus-row-families",
                "note": "The selected placement exporter retains lightweight nested family boundaries but deliberately suppresses matrix coefficients before the large projection loop.",
            },
            "omitted_projection": {
                "row_start": self.audit.row_start,
                "row_end": self.audit.row_end,
                "row_count": self.audit.row_end - self.audit.row_start,
                "first_allocated_column": self.audit.first_allocated_column,
                "note": "Placement is captured from the real terminal path before materializing the large raw-old-block row loop.",
            },
            "semantic_scope": {
                "decoded": [],
                "not_decoded": [
                    "selected terminal NIFS",
                    "running/fresh unary relations",
                    "terminal continuity",
                    "direct terminal CE",
                ],
                "warning": "This capture selects the actual thirteen-matrix path but is not yet a row-decoding or refinement theorem.",
            },
        })
    }

    fn validate_against_execution(
        &self,
        prep: &R1csIvcPreprocessing,
        execution: &R1csIvcPostPiDecExecutionAudit,
    ) -> Result<(), String> {
        self.validate()?;
        let expected_arms = [
            R1csIvcBranch::Base,
            R1csIvcBranch::BootstrapRecursive,
            R1csIvcBranch::Recursive,
        ];
        let selector_columns = prep
            .relation()
            .compilation_audit()
            .layout()
            .selector_columns();
        if execution.branch() != R1csIvcBranch::Recursive
            || execution.combined_nc().proof_variant() != PiCcsProofVariant::BlockLaneNcDelayedV1
            || execution.raw_old_block().profile() != R1csIvcRawOldBlockProfile::ActiveFPrimeCombinedNcDelayedV1
            || execution.raw_old_block().field_decoding() != R1csIvcRawOldBlockFieldDecoding::BaseFieldEmbedding
            || execution.raw_old_block().logical_columns() != LOGICAL_WIDTH
            || execution.selector_writes().len() != expected_arms.len()
            || self.selector_columns != selector_columns
            || execution
                .selector_writes()
                .iter()
                .enumerate()
                .any(|(index, write)| {
                    write.arm() != expected_arms[index]
                        || write.logical_column() != selector_columns[index]
                        || write.packed_coordinate()
                            != (
                                write.logical_column() % ACTIVE_LANES,
                                write.logical_column() / ACTIVE_LANES,
                            )
                        || write.value().as_canonical_u64() as usize != self.selector_values[index]
                        || self.selector_values[index] != usize::from(write.arm() == execution.branch())
                })
        {
            return Err("post-PiDEC profile or selector execution binding drift".into());
        }
        Ok(())
    }

    fn validate(&self) -> Result<(), String> {
        let audit = &self.audit;
        let plan = audit.plan;
        let map = &audit.column_map;
        let witness_entries = plan.active_lanes() * plan.packed_columns();
        if self.relation_matrix_count != 13 {
            return Err(format!(
                "selected relation matrix count is {}, expected 13",
                self.relation_matrix_count
            ));
        }
        if self.terminal_claim_count != RAW_OLD_BLOCK_CHILD_COUNT {
            return Err(format!(
                "selected terminal claim count is {}, expected {RAW_OLD_BLOCK_CHILD_COUNT}",
                self.terminal_claim_count
            ));
        }
        if self.terminal_evaluation_count != self.relation_matrix_count {
            return Err(format!(
                "selected terminal evaluation count is {}, relation matrix count is {}",
                self.terminal_evaluation_count, self.relation_matrix_count
            ));
        }
        if self.prefix_rows != audit.row_start {
            return Err(format!(
                "selected terminal prefix stops at row {}, projection starts at {}",
                self.prefix_rows, audit.row_start
            ));
        }
        if self.row_families.is_empty() {
            return Err("selected terminal prefix has no row-family boundaries".to_string());
        }
        if self.prefix_nonzero_entries != 0 {
            return Err(format!(
                "selected terminal diagnostic retained {} matrix coefficients",
                self.prefix_nonzero_entries
            ));
        }
        if self
            .row_families
            .iter()
            .any(|family| family.row_start > family.row_end || family.row_end > self.prefix_rows)
        {
            return Err("selected terminal row-family boundary exceeds the prefix".to_string());
        }
        for required in [
            "terminal.nifs",
            "terminal.running_link",
            "terminal.parent_link",
            "terminal.latest_link",
            "terminal.accumulator",
            "terminal.total",
        ] {
            if !self
                .row_families
                .iter()
                .any(|family| family.name == required)
            {
                return Err(format!("selected terminal prefix is missing row family {required}"));
            }
        }
        if audit.pending_projection_join_id != RAW_OLD_BLOCK_PENDING_JOIN_ID
            || plan.logical_columns() != LOGICAL_WIDTH
            || plan.child_count() != RAW_OLD_BLOCK_CHILD_COUNT
            || plan.active_lanes() != ACTIVE_LANES
            || PADDED_LANES - ACTIVE_LANES != 10
            || plan.block_variables() != BLOCK_VARIABLES
            || !plan.factor_final_round()
            || plan.tensor_variables() != TENSOR_VARIABLES
            || plan.factored_variable() != Some(TENSOR_VARIABLES)
            || plan.packed_columns() != BLOCK_COUNT
            || plan.final_scale_rows() != FINAL_SCALE_ROWS
            || plan.tensor_rows() + plan.projection_product_rows() + plan.final_scale_rows() != DERIVED_COLUMNS
            || audit.radix != RADIX
            || audit.program.plan() != plan
            || audit.program.row_count() != TOTAL_ROWS
            || audit.row_end - audit.row_start != TOTAL_ROWS
            || audit.tensor_rows != (audit.row_start..audit.row_start + plan.tensor_rows())
            || audit.projection_product_rows
                != (audit.tensor_rows.end..audit.tensor_rows.end + plan.projection_product_rows())
            || audit.final_scale_rows
                != (audit.projection_product_rows.end..audit.projection_product_rows.end + plan.final_scale_rows())
            || audit.terminal_rows != (audit.final_scale_rows.end..audit.final_scale_rows.end + plan.terminal_rows())
            || audit.terminal_rows.end != audit.row_end
            || audit.first_allocated_column != audit.tensor_first_allocated_column
            || audit.projection_product_first_allocated_column
                != audit.tensor_first_allocated_column + plan.tensor_rows()
            || audit.final_scale_first_allocated_column
                != audit.projection_product_first_allocated_column + plan.projection_product_rows()
            || audit.final_scale_first_allocated_column + plan.final_scale_rows() - audit.tensor_first_allocated_column
                != DERIVED_COLUMNS
            || self.old_block != audit.pending_old_block_cols
            || self.parent != audit.parent_y_zcol_cols
            || self.final_witness_first != audit.projection_child_witness_first_columns
            || self.ajtai_witness_first != audit.ajtai_child_witness_first_columns
            || self.final_witness_first != self.ajtai_witness_first
            || self.profile_tag != PROFILE_TAG
            || self.recursive_selector_tag != RECURSIVE_SELECTOR_TAG
            || self.old_block.len() != BLOCK_VARIABLES
            || self.parent.len() != ACTIVE_LANES
            || self.final_witness_first.len() != RAW_OLD_BLOCK_CHILD_COUNT
            || self.selector_columns.len() != 3
            || self.selector_values.len() != 3
            || map.layout() != audit.program.layout()
            || map.actual_old_block() != self.old_block
            || map.actual_parent() != self.parent
            || map.actual_child_witness_first() != self.final_witness_first
            || map.actual_tensor_first() != audit.tensor_first_allocated_column
            || map.actual_product_first() != audit.projection_product_first_allocated_column
            || map.actual_final_scale_first() != audit.final_scale_first_allocated_column
        {
            return Err("production terminal placement header or runtime audit drift".into());
        }
        if self
            .final_witness_first
            .windows(2)
            .any(|pair| pair[1] != pair[0] + witness_entries)
            || self
                .final_witness_first
                .last()
                .copied()
                .map(|first| first + witness_entries)
                != Some(audit.tensor_first_allocated_column)
        {
            return Err("production FinalWitnessWires allocation schedule drift".into());
        }
        Ok(())
    }

    pub(crate) fn assert_mutations_fail(
        &self,
        prep: &R1csIvcPreprocessing,
        execution: &R1csIvcPostPiDecExecutionAudit,
    ) {
        let reject = |mutated: &Self, label: &str| {
            assert!(
                mutated.validate_against_execution(prep, execution).is_err(),
                "{label} must fail closed"
            );
        };
        let mut changed = self.clone();
        changed.audit.row_end += 1;
        reject(&changed, "row-stop mutation");
        let mut changed = self.clone();
        changed.relation_matrix_count = 3;
        reject(&changed, "relation-matrix-count mutation");
        let mut changed = self.clone();
        changed.terminal_evaluation_count = 3;
        reject(&changed, "terminal-evaluation-count mutation");
        let mut changed = self.clone();
        changed.prefix_rows += 1;
        reject(&changed, "terminal-prefix-row mutation");
        let mut changed = self.clone();
        changed.row_families.clear();
        reject(&changed, "terminal-row-family mutation");
        let mut changed = self.clone();
        changed.prefix_nonzero_entries = 1;
        reject(&changed, "terminal-prefix-coefficient mutation");
        let mut changed = self.clone();
        changed.old_block[0][0] += 1;
        reject(&changed, "old-block column mutation");
        let mut changed = self.clone();
        changed.parent[0][0] += 1;
        reject(&changed, "parent column mutation");
        let mut changed = self.clone();
        changed.final_witness_first.swap(0, 1);
        reject(&changed, "FinalWitnessWires child-order mutation");
        let mut changed = self.clone();
        changed.audit.tensor_first_allocated_column += 1;
        reject(&changed, "tensor-first mutation");
        let mut changed = self.clone();
        changed.audit.projection_product_first_allocated_column += 1;
        reject(&changed, "product-first mutation");
        let mut changed = self.clone();
        changed.audit.final_scale_first_allocated_column += 1;
        reject(&changed, "final-scale-first mutation");
        let mut changed = self.clone();
        changed.ajtai_witness_first[0] += 1;
        reject(&changed, "Ajtai allocation-join mutation");
        let mut changed = self.clone();
        changed.selector_values[2] = 0;
        reject(&changed, "recursive-selector mutation");
        let mut changed = self.clone();
        changed.profile_tag += 1;
        reject(&changed, "profile mutation");
        let mut changed = self.clone();
        changed.audit.pending_projection_join_id += 1;
        reject(&changed, "pending-join mutation");
    }
}

fn lean_k_columns(columns: &[[usize; 2]]) -> String {
    format!(
        "[{}]",
        columns
            .iter()
            .map(|[c0, c1]| format!("{{ c0 := {c0}, c1 := {c1} }}"))
            .collect::<Vec<_>>()
            .join(", ")
    )
}

fn lean_nats(values: &[usize]) -> String {
    format!(
        "[{}]",
        values
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(", ")
    )
}

pub(super) fn render(certificate: &ProductionPlacementCertificate) -> GeneratedLeanFile {
    certificate
        .validate()
        .expect("render validated production placement");
    let audit = &certificate.audit;
    let namespace = format!("{NAMESPACE_ROOT}.Execution.RawOldBlockProjectionRowAt");
    let mut contents = generated_header(
        "the exact production terminal row interval, pending-state columns, ordered raw WitnessMat bases, profile selector, and shared Ajtai allocation",
    );
    write!(contents, "import {namespace}\n\nnamespace {namespace}\n\n").expect("production placement header");
    writeln!(
        contents,
        "def productionEmitterLayout : EmitterLayout :=\n  {{ rowFirst := {}\n    rowStop := {}\n    oldBlock := {}\n    parent := {}\n    finalWitnessFirst := {}\n    tensorFirst := {}\n    productFirst := {}\n    finalScaleFirst := {} }}",
        audit.row_start,
        audit.row_end,
        lean_k_columns(&certificate.old_block),
        lean_k_columns(&certificate.parent),
        lean_nats(&certificate.final_witness_first),
        audit.tensor_first_allocated_column,
        audit.projection_product_first_allocated_column,
        audit.final_scale_first_allocated_column,
    )
    .expect("production emitter layout");
    writeln!(
        contents,
        "def productionAjtaiChildWitnessFirst : List Nat := {}",
        lean_nats(&certificate.ajtai_witness_first)
    )
    .expect("Ajtai bases");
    for (name, value) in [
        ("productionProfileTag", certificate.profile_tag),
        ("productionPendingProjectionJoinId", audit.pending_projection_join_id),
        ("productionRecursiveSelectorTag", certificate.recursive_selector_tag),
    ] {
        writeln!(contents, "def {name} : Nat := {value}").expect("production placement tag");
    }
    writeln!(
        contents,
        "def productionRecursiveSelectorColumns : List Nat := {}\ndef productionRecursiveSelectorValues : List Nat := {}\ndef productionEmitterLayoutChecked : Bool :=\n  emitterColumnMapValid productionEmitterLayout\ndef productionAjtaiJoinChecked : Bool :=\n  productionEmitterLayout.finalWitnessFirst == productionAjtaiChildWitnessFirst\ndef productionProfileChecked : Bool :=\n  productionProfileTag == RawOldBlockProjectionPlan.profileTag &&\n  productionPendingProjectionJoinId == RawOldBlockProjectionPlan.pendingProjectionJoinId &&\n  productionRecursiveSelectorTag == RawOldBlockProjectionPlan.recursiveSelectorTag &&\n  RawOldBlockProjectionPlan.factorFinalRound &&\n  RawOldBlockProjectionPlan.tensorVariables == {TENSOR_VARIABLES} &&\n  RawOldBlockProjectionPlan.factoredVariable == some {TENSOR_VARIABLES}\ndef productionSelectorChecked : Bool :=\n  productionRecursiveSelectorColumns == RawOldBlockProjectionPlan.recursiveSelectorColumns &&\n  productionRecursiveSelectorValues == RawOldBlockProjectionPlan.recursiveSelectorValues\n\nend {namespace}",
        lean_nats(&certificate.selector_columns),
        lean_nats(&certificate.selector_values),
    )
    .expect("production placement checks");
    GeneratedLeanFile {
        relative_path: format!("{GENERATED_ROOT}/Execution/ProductionEmitterLayout.lean"),
        contents,
    }
}

//! Bounded physical certificate for the terminal raw-old-block emitter.
//!
//! The active profile remains compact and parametric.  This B=2 fixture is
//! small enough to export every sparse row and assignment value, while using
//! the same Rust emitter, row-index plan, `PendingProjectionWires` shape, and
//! ordered `FinalWitnessWires` allocations as production.

use std::fmt::Write as _;
use std::sync::Arc;

use neo_ajtai::{setup as setup_ajtai, AjtaiSModule};
use neo_ccs::Mat;
use neo_fold_clean::engine::decider::__test_isolation::enforce_terminal_raw_old_block_projection_with_ajtai_against;
use neo_fold_clean::engine::r1cs_circuit::{
    CanonicalSparseRow, R1csSnapshot, RawOldBlockProjectionPlan, RawOldBlockProjectionProgram,
    TerminalPendingProjectionAudit, RAW_OLD_BLOCK_PENDING_JOIN_ID,
};
use neo_math::{D, F, K};
use neo_reductions::block_projection::{radix_recompose_raw_witnesses_at_block_point, BLOCK_PROJECTION_POINT_LEN};
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use rand_chacha::{rand_core::SeedableRng, ChaCha20Rng};

use super::{generated_header, GeneratedLeanFile, GENERATED_ROOT, NAMESPACE_ROOT};

const FIXTURE_PROFILE_TAG: usize = 1;
const FIXTURE_LOGICAL_WIDTH: usize = 2 * D;
const FIXTURE_BLOCK_COUNT: usize = 2;
const FIXTURE_CHILD_COUNT: usize = 14;
const FIXTURE_RADIX: u32 = 2;
const ROW_CHUNK: usize = 200;
const ASSIGNMENT_CHUNK: usize = 224;

#[derive(Clone)]
pub(crate) struct TerminalProjectionFixture {
    snapshot: R1csSnapshot,
    audit: TerminalPendingProjectionAudit,
    projection_column_stop: usize,
    row_major_mapping: bool,
}

impl TerminalProjectionFixture {
    pub(crate) fn capture() -> Self {
        let witnesses = (0..FIXTURE_CHILD_COUNT)
            .map(|child| {
                Mat::from_row_major(
                    D,
                    FIXTURE_BLOCK_COUNT,
                    (0..D)
                        .flat_map(|lane| {
                            (0..FIXTURE_BLOCK_COUNT)
                                .map(move |block| F::from_u64((101 * child + 7 * lane + 3 * block + 1) as u64))
                        })
                        .collect(),
                )
            })
            .collect::<Vec<_>>();
        let old_block = std::array::from_fn(|index| K::from(F::from_u64((index + 2) as u64)));
        let parent = radix_recompose_raw_witnesses_at_block_point(
            &witnesses,
            FIXTURE_LOGICAL_WIDTH,
            &old_block,
            K::from(F::from_u64(FIXTURE_RADIX as u64)),
        )
        .expect("bounded native raw-old-block recomposition");
        let mut rng = ChaCha20Rng::seed_from_u64(0xA17A_10DB_10C0_0001);
        let pp = setup_ajtai(&mut rng, D, 1, FIXTURE_BLOCK_COUNT).expect("bounded Ajtai setup");
        let log = AjtaiSModule::new(Arc::new(pp));
        let builder = enforce_terminal_raw_old_block_projection_with_ajtai_against(
            &log,
            FIXTURE_LOGICAL_WIDTH,
            &old_block,
            &parent,
            &witnesses,
            FIXTURE_RADIX,
        )
        .expect("bounded raw-old-block projection plus Ajtai join");
        assert!(builder.is_satisfied(), "bounded terminal projection fixture");
        let [audit] = builder.terminal_pending_projection_audits() else {
            panic!("bounded terminal fixture must own one projection audit")
        };
        let audit = audit.clone();
        let projection_column_stop = audit.final_scale_first_allocated_column + audit.plan.final_scale_rows();
        let certificate = Self {
            snapshot: builder.snapshot(),
            audit,
            projection_column_stop,
            row_major_mapping: true,
        };
        certificate
            .validate()
            .expect("bounded terminal projection certificate");
        certificate
    }

    fn validate(&self) -> Result<(), String> {
        let audit = &self.audit;
        let plan = RawOldBlockProjectionPlan::new(FIXTURE_LOGICAL_WIDTH, FIXTURE_CHILD_COUNT)
            .map_err(|error| error.to_owned())?;
        let program = RawOldBlockProjectionProgram::new(plan, FIXTURE_RADIX).map_err(str::to_owned)?;
        if audit.pending_projection_join_id != RAW_OLD_BLOCK_PENDING_JOIN_ID
            || audit.plan != plan
            || audit.program != program
            || audit.radix != FIXTURE_RADIX
            || audit.row_end - audit.row_start != plan.total_rows()
            || audit.tensor_rows != (audit.row_start..audit.row_start + plan.tensor_rows())
            || audit.projection_product_rows
                != (audit.tensor_rows.end..audit.tensor_rows.end + plan.projection_product_rows())
            || audit.final_scale_rows
                != (audit.projection_product_rows.end..audit.projection_product_rows.end + plan.final_scale_rows())
            || audit.terminal_rows != (audit.final_scale_rows.end..audit.row_end)
            || audit.pending_old_block_cols.len() != BLOCK_PROJECTION_POINT_LEN
            || audit.parent_y_zcol_cols.len() != D
            || audit.projection_child_witness_first_columns.len() != FIXTURE_CHILD_COUNT
            || audit.ajtai_child_witness_first_columns != audit.projection_child_witness_first_columns
        {
            return Err("bounded terminal audit metadata drift".to_owned());
        }
        if audit.pending_old_block_cols
            != (0..BLOCK_PROJECTION_POINT_LEN)
                .map(|index| [1 + 2 * index, 2 + 2 * index])
                .collect::<Vec<_>>()
            || audit.parent_y_zcol_cols
                != (0..D)
                    .map(|lane| {
                        [
                            1 + 2 * BLOCK_PROJECTION_POINT_LEN + 2 * lane,
                            2 + 2 * BLOCK_PROJECTION_POINT_LEN + 2 * lane,
                        ]
                    })
                    .collect::<Vec<_>>()
        {
            return Err("bounded pending old-block/parent absolute pins drift".to_owned());
        }
        let first_child = 1 + 2 * BLOCK_PROJECTION_POINT_LEN + 2 * D;
        let witness_entries = D * FIXTURE_BLOCK_COUNT;
        let expected_child_bases = (0..FIXTURE_CHILD_COUNT)
            .map(|child| first_child + child * witness_entries)
            .collect::<Vec<_>>();
        if audit.projection_child_witness_first_columns != expected_child_bases
            || audit.tensor_first_allocated_column != first_child + FIXTURE_CHILD_COUNT * witness_entries
            || audit.first_allocated_column != audit.tensor_first_allocated_column
            || audit.projection_product_first_allocated_column
                != audit.tensor_first_allocated_column + plan.tensor_rows()
            || audit.final_scale_first_allocated_column
                != audit.projection_product_first_allocated_column + plan.projection_product_rows()
            || self.projection_column_stop != audit.final_scale_first_allocated_column + plan.final_scale_rows()
        {
            return Err("bounded terminal absolute allocation schedule drift".to_owned());
        }
        if !self.row_major_mapping {
            return Err("bounded witness mapping is not lane-major/block-minor".to_owned());
        }
        let column_map = &audit.column_map;
        if column_map.layout() != program.layout()
            || column_map.actual_old_block() != audit.pending_old_block_cols
            || column_map.actual_parent() != audit.parent_y_zcol_cols
            || column_map.actual_child_witness_first() != audit.projection_child_witness_first_columns
            || column_map.actual_tensor_first() != audit.tensor_first_allocated_column
            || column_map.actual_product_first() != audit.projection_product_first_allocated_column
            || column_map.actual_final_scale_first() != audit.final_scale_first_allocated_column
        {
            return Err("bounded internally constructed column map drift".to_owned());
        }
        for relative_row in 0..program.row_count() {
            let physical_row = audit.row_start + relative_row;
            let actual = CanonicalSparseRow {
                a: self.snapshot.a_row(physical_row).to_vec(),
                b: self.snapshot.b_row(physical_row).to_vec(),
                c: self.snapshot.c_row(physical_row).to_vec(),
            };
            let normalized = column_map
                .normalize_actual_row(&actual)
                .ok_or_else(|| format!("projection row {physical_row} escapes its declared column map"))?;
            let expected = program
                .row_at(relative_row)
                .ok_or_else(|| format!("indexed projection row {relative_row} is missing"))?;
            if normalized != expected {
                return Err(format!(
                    "projection row {physical_row} differs from indexed row-at {relative_row}"
                ));
            }
        }
        Ok(())
    }

    pub(crate) fn assert_mutations_fail(&self) {
        let mut child_order = self.clone();
        child_order
            .audit
            .projection_child_witness_first_columns
            .swap(0, 1);
        assert!(child_order.validate().is_err(), "child-order mutation must fail");

        let mut transpose = self.clone();
        transpose.row_major_mapping = false;
        assert!(transpose.validate().is_err(), "lane/block transpose must fail");

        let mut tensor_operand = self.clone();
        tensor_operand.snapshot.apply_b_row_test_mutation(
            tensor_operand.audit.row_start,
            tensor_operand.audit.pending_old_block_cols[0][0],
            F::ONE,
        );
        assert!(
            tensor_operand.validate().is_err(),
            "tensor-operand row mutation must fail"
        );

        let mut tensor_output = self.clone();
        tensor_output.snapshot.apply_c_row_test_mutation(
            tensor_output.audit.row_start,
            tensor_output.audit.tensor_first_allocated_column,
            F::ONE,
        );
        assert!(
            tensor_output.validate().is_err(),
            "tensor-output row mutation must fail"
        );

        let mut lane_block = self.clone();
        lane_block.snapshot.apply_a_row_test_mutation(
            lane_block.audit.projection_product_rows.start,
            lane_block.audit.projection_child_witness_first_columns[0] + 1,
            F::ONE,
        );
        assert!(
            lane_block.validate().is_err(),
            "lane/block source-row mutation must fail"
        );

        let mut parent = self.clone();
        parent.snapshot.apply_a_row_test_mutation(
            parent.audit.terminal_rows.start,
            parent.audit.parent_y_zcol_cols[0][0],
            F::ONE,
        );
        assert!(parent.validate().is_err(), "terminal-parent row mutation must fail");

        let mut radix = self.clone();
        radix.audit.radix += 1;
        assert!(radix.validate().is_err(), "radix mutation must fail");

        let mut allocation_join = self.clone();
        allocation_join.audit.ajtai_child_witness_first_columns[0] += 1;
        assert!(
            allocation_join.validate().is_err(),
            "Ajtai allocation join mutation must fail"
        );

        let mut boundary = self.clone();
        boundary.audit.tensor_rows.end += 1;
        assert!(boundary.validate().is_err(), "row-boundary mutation must fail");
    }
}

fn lean_terms(terms: &[(usize, F)]) -> String {
    format!(
        "[{}]",
        terms
            .iter()
            .map(|(column, coefficient)| format!("({column}, {})", coefficient.as_canonical_u64()))
            .collect::<Vec<_>>()
            .join(", ")
    )
}

fn lean_k_columns(values: &[[usize; 2]]) -> String {
    format!(
        "[{}]",
        values
            .iter()
            .map(|[c0, c1]| format!("{{ c0 := {c0}, c1 := {c1} }}"))
            .collect::<Vec<_>>()
            .join(", ")
    )
}

fn metadata(certificate: &TerminalProjectionFixture) -> GeneratedLeanFile {
    let audit = &certificate.audit;
    let namespace = format!("{NAMESPACE_ROOT}.Execution.TerminalProjectionFixture.Metadata");
    let mut contents = generated_header(
        "the bounded terminal fixture profile, row ranges, absolute source columns, and shared raw-witness allocation",
    );
    write!(
        contents,
        "import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Schema\n\nnamespace {namespace}\n\n"
    )
    .expect("fixture metadata header");
    for (name, value) in [
        ("pendingProjectionProfileTag", FIXTURE_PROFILE_TAG),
        ("pendingProjectionJoinId", audit.pending_projection_join_id),
        ("logicalWidth", FIXTURE_LOGICAL_WIDTH),
        ("blockCount", FIXTURE_BLOCK_COUNT),
        ("childCount", FIXTURE_CHILD_COUNT),
        ("activeLanes", D),
        ("paddedLanes", 64),
        ("radixBase", audit.radix as usize),
        ("rowFirst", audit.row_start),
        ("rowStop", audit.row_end),
        ("rowCount", audit.row_end - audit.row_start),
        ("rowChunkMaximum", ROW_CHUNK),
        ("tensorRowFirst", audit.tensor_rows.start),
        ("tensorRowStop", audit.tensor_rows.end),
        ("productRowFirst", audit.projection_product_rows.start),
        ("productRowStop", audit.projection_product_rows.end),
        ("finalScaleRowFirst", audit.final_scale_rows.start),
        ("finalScaleRowStop", audit.final_scale_rows.end),
        ("terminalRowFirst", audit.terminal_rows.start),
        ("terminalRowStop", audit.terminal_rows.end),
        ("tensorAbsoluteFirstColumn", audit.tensor_first_allocated_column),
        (
            "productAbsoluteFirstColumn",
            audit.projection_product_first_allocated_column,
        ),
        (
            "finalScaleAbsoluteFirstColumn",
            audit.final_scale_first_allocated_column,
        ),
        ("projectionColumnStop", certificate.projection_column_stop),
        ("assignmentColumnCount", certificate.projection_column_stop),
        ("assignmentChunkMaximum", ASSIGNMENT_CHUNK),
    ] {
        writeln!(contents, "def {name} : Nat := {value}").expect("fixture scalar");
    }
    writeln!(contents, "def selectorAbsoluteColumn : Option Nat := none").expect("selector column");
    writeln!(contents, "def selectorValue : Option Nat := none").expect("selector value");
    writeln!(
        contents,
        "def pendingOldBlockAbsoluteColumnList : List RawKColumns := {}",
        lean_k_columns(&audit.pending_old_block_cols)
    )
    .expect("old-block columns");
    writeln!(
        contents,
        "def pendingParentAbsoluteColumnList : List RawKColumns := {}",
        lean_k_columns(&audit.parent_y_zcol_cols)
    )
    .expect("parent columns");
    writeln!(
        contents,
        "def childWitnessAbsoluteFirstList : List Nat := {:?}",
        audit.projection_child_witness_first_columns
    )
    .expect("projection child bases");
    writeln!(
        contents,
        "def ajtaiChildWitnessAbsoluteFirstList : List Nat := {:?}",
        audit.ajtai_child_witness_first_columns
    )
    .expect("Ajtai child bases");
    contents.push_str(
        "\ndef pendingOldBlockAbsoluteColumns (index : Fin 19) : RawKColumns :=\n\
           pendingOldBlockAbsoluteColumnList.getD index.val default\n\
         def pendingParentAbsoluteColumns (index : Fin 54) : RawKColumns :=\n\
           pendingParentAbsoluteColumnList.getD index.val default\n\
         def childWitnessAbsoluteFirst (index : Fin 14) : Nat :=\n\
           childWitnessAbsoluteFirstList.getD index.val 0\n\
         def childWitnessOffset (lane block : Nat) : Nat := lane * blockCount + block\n\
         def sharedFinalWitnessAllocation : Bool :=\n\
           childWitnessAbsoluteFirstList == ajtaiChildWitnessAbsoluteFirstList\n",
    );
    writeln!(contents, "\nend {namespace}").expect("fixture metadata end");
    GeneratedLeanFile {
        relative_path: format!("{GENERATED_ROOT}/Execution/TerminalProjectionFixture/Metadata.lean"),
        contents,
    }
}

pub(super) fn row_chunks(certificate: &TerminalProjectionFixture) -> Vec<GeneratedLeanFile> {
    let rows = (certificate.audit.row_start..certificate.audit.row_end).collect::<Vec<_>>();
    rows.chunks(ROW_CHUNK)
        .enumerate()
        .map(|(chunk_index, chunk)| {
            let namespace = format!("{NAMESPACE_ROOT}.Execution.TerminalProjectionFixture.Rows.Chunk{chunk_index}");
            let mut contents =
                generated_header("one bounded chunk of exact sparse A/B/C terminal raw-old-block projection rows");
            write!(
                contents,
                "import Nightstream.Implementation.R1CS.Core.Semantics\n\n\
                 namespace {namespace}\n\n\
                 def firstPhysicalRow : Nat := {}\n\
                 def recordCount : Nat := {}\n\
                 set_option maxRecDepth 100000 in\n\
                 def values : List Nightstream.Implementation.R1CS.Row := [\n",
                chunk.first().copied().expect("nonempty row chunk"),
                chunk.len()
            )
            .expect("fixture row header");
            for (offset, row) in chunk.iter().copied().enumerate() {
                if offset != 0 {
                    contents.push_str(",\n");
                }
                write!(
                    contents,
                    "  {{ a := {}, b := {}, c := {} }}",
                    lean_terms(certificate.snapshot.a_row(row)),
                    lean_terms(certificate.snapshot.b_row(row)),
                    lean_terms(certificate.snapshot.c_row(row)),
                )
                .expect("fixture row");
            }
            writeln!(contents, "\n]\n\nend {namespace}").expect("fixture rows end");
            GeneratedLeanFile {
                relative_path: format!(
                    "{GENERATED_ROOT}/Execution/TerminalProjectionFixture/Rows/Chunk{chunk_index}.lean"
                ),
                contents,
            }
        })
        .collect()
}

fn assignment_chunks(certificate: &TerminalProjectionFixture) -> Vec<GeneratedLeanFile> {
    certificate.snapshot.witness()[..certificate.projection_column_stop]
        .chunks(ASSIGNMENT_CHUNK)
        .enumerate()
        .map(|(chunk_index, chunk)| {
            let namespace =
                format!("{NAMESPACE_ROOT}.Execution.TerminalProjectionFixture.Assignment.Chunk{chunk_index}");
            let values = chunk
                .iter()
                .map(|value| value.as_canonical_u64().to_string())
                .collect::<Vec<_>>()
                .join(", ");
            let mut contents =
                generated_header("one bounded proof-free chunk of the exact terminal raw-old-block assignment");
            write!(
                contents,
                "namespace {namespace}\n\n\
                 def firstAbsoluteColumn : Nat := {}\n\
                 def recordCount : Nat := {}\n\
                 def values : List Nat := [{values}]\n\n\
                 end {namespace}\n",
                chunk_index * ASSIGNMENT_CHUNK,
                chunk.len(),
            )
            .expect("fixture assignment header");
            GeneratedLeanFile {
                relative_path: format!(
                    "{GENERATED_ROOT}/Execution/TerminalProjectionFixture/Assignment/Chunk{chunk_index}.lean"
                ),
                contents,
            }
        })
        .collect()
}

fn facade(row_chunks: usize, assignment_chunks: usize) -> GeneratedLeanFile {
    let namespace = format!("{NAMESPACE_ROOT}.Execution.TerminalProjectionFixture");
    let mut contents =
        generated_header("the ordered facade over every bounded terminal raw-old-block row and assignment chunk");
    writeln!(contents, "import {namespace}.Metadata").expect("fixture facade metadata");
    for chunk in 0..row_chunks {
        writeln!(contents, "import {namespace}.Rows.Chunk{chunk}").expect("fixture row import");
    }
    for chunk in 0..assignment_chunks {
        writeln!(contents, "import {namespace}.Assignment.Chunk{chunk}").expect("fixture assignment import");
    }
    writeln!(contents, "\nnamespace {namespace}\n").expect("fixture facade namespace");
    writeln!(
        contents,
        "def rows : List Nightstream.Implementation.R1CS.Row := {}",
        (0..row_chunks)
            .map(|chunk| format!("Rows.Chunk{chunk}.values"))
            .collect::<Vec<_>>()
            .join(" ++ ")
    )
    .expect("fixture rows facade");
    writeln!(
        contents,
        "def assignmentValues : List Nat := {}",
        (0..assignment_chunks)
            .map(|chunk| format!("Assignment.Chunk{chunk}.values"))
            .collect::<Vec<_>>()
            .join(" ++ ")
    )
    .expect("fixture assignment facade");
    contents.push_str(
        "def artifactRow (index : Fin rows.length) : Nightstream.Implementation.R1CS.Row :=\n\
           rows.get index\n\
         def assignment (column : Nat) : Nat := assignmentValues.getD column 0\n",
    );
    writeln!(contents, "\nend {namespace}").expect("fixture facade end");
    GeneratedLeanFile {
        relative_path: format!("{GENERATED_ROOT}/Execution/TerminalProjectionFixture.lean"),
        contents,
    }
}

pub(super) fn render(certificate: &TerminalProjectionFixture) -> Vec<GeneratedLeanFile> {
    let mut files = vec![metadata(certificate)];
    let rows = row_chunks(certificate);
    let assignment = assignment_chunks(certificate);
    let row_count = rows.len();
    let assignment_count = assignment.len();
    files.extend(rows);
    files.extend(assignment);
    files.push(facade(row_count, assignment_count));
    files
}

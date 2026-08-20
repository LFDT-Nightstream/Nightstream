//! Exact source-to-final provenance for the terminal public XOut hash.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::ops::Range;
use std::path::{Path, PathBuf};

use neo_ccs::CcsMatrix;
use neo_fold_clean::frontends::nebula::f_prime::NebulaFPrimeStreamingLifecycleArm;
use neo_fold_clean::frontends::r1cs_f_prime::{
    audit_multi_branch_selective_rows_with_complete_source_provenance_with_alignment, SelectiveProjectedPort,
    SelectiveProjectedPoseidon2SboxStep, SelectiveProjectedRowArtifact, SelectiveProjectedRowsAudit,
    SelectiveProjectedSourceLinearCombination, SelectiveRewriteKind, SelectiveSourceRowDisposition,
};
use neo_math::{D, F};
use p3_field::{PrimeCharacteristicRing, PrimeField64};

use crate::streaming_terminal_fixture::StreamingTerminalAuditFixture;

const RECURSIVE_ARM_INDEX: usize = 1;
const X_OUT_HASH_PERMUTATIONS: usize = 9;
const POSEIDON2_SBOX_ROWS: usize = 86;
const POSEIDON2_SOURCE_COLUMNS: usize = 600;
const SLOT_WIDTH: usize = 41;
const ARTIFACT_PATH: &str = "../../formal/nightstream-lean/Nightstream/Implementation/\
R1CS/Artifacts/FPrimeFullHistory/Generated/FPrimeFullHistoryStreamingTerminalXOutPublicHash.lean";
const FIRST_LEAF_STEPS_ARTIFACT_PATH: &str = "../../formal/nightstream-lean/Nightstream/Implementation/\
R1CS/Artifacts/FPrimeFullHistory/Generated/FPrimeFullHistoryStreamingTerminalXOutPoseidonFirstLeafSteps.lean";
const FIRST_LEAF_ROWS_0_ARTIFACT_PATH: &str = "../../formal/nightstream-lean/Nightstream/Implementation/\
R1CS/Artifacts/FPrimeFullHistory/Generated/FPrimeFullHistoryStreamingTerminalXOutPoseidonFirstLeafRows0.lean";
const FIRST_LEAF_ROWS_1_ARTIFACT_PATH: &str = "../../formal/nightstream-lean/Nightstream/Implementation/\
R1CS/Artifacts/FPrimeFullHistory/Generated/FPrimeFullHistoryStreamingTerminalXOutPoseidonFirstLeafRows1.lean";
const FIRST_LEAF_IMAGES_ARTIFACT_PATH: &str = "../../formal/nightstream-lean/Nightstream/Implementation/\
R1CS/Artifacts/FPrimeFullHistory/Generated/FPrimeFullHistoryStreamingTerminalXOutPoseidonFirstLeafImages.lean";

pub struct StreamingTerminalXOutHashProvenance {
    source_rows: Range<usize>,
    emitted_rows: Vec<usize>,
    projected: SelectiveProjectedRowsAudit,
}

impl StreamingTerminalXOutHashProvenance {
    pub fn source_rows(&self) -> Range<usize> {
        self.source_rows.clone()
    }

    pub fn emitted_rows(&self) -> &[usize] {
        &self.emitted_rows
    }

    pub fn projected(&self) -> &SelectiveProjectedRowsAudit {
        &self.projected
    }
}

fn selected_matrix_terms(matrix: &CcsMatrix<F>, selected_rows: &BTreeSet<usize>) -> BTreeMap<(usize, usize), F> {
    let mut terms = BTreeMap::<(usize, usize), F>::new();
    let mut add = |row: usize, column: usize, coefficient: F| {
        if coefficient != F::ZERO {
            *terms.entry((row, column)).or_insert(F::ZERO) += coefficient;
        }
    };

    match matrix {
        CcsMatrix::Identity { n } => {
            for &row in selected_rows.range(..*n) {
                add(row, row, F::ONE);
            }
        }
        CcsMatrix::Csc(_) | CcsMatrix::CscWithSeededPhi81 { .. } => {
            let csc = matrix.sparse_component().expect("sparse matrix component");
            for column in 0..csc.ncols {
                for entry in csc.column_range(column) {
                    let row = csc.row_index(entry);
                    if selected_rows.contains(&row) {
                        add(row, column, csc.vals[entry]);
                    }
                }
            }
            for block in matrix.seeded_phi81_blocks() {
                for &row in selected_rows.range(block.row_start()..block.row_end()) {
                    block.for_each_row_term::<F, _>(row, |column, coefficient| {
                        add(row, column, coefficient);
                    });
                }
            }
            for run in matrix.geometric_runs() {
                if selected_rows.contains(&run.row()) {
                    run.for_each_term(|row, column, coefficient| {
                        add(row, column, coefficient);
                    });
                }
            }
        }
        CcsMatrix::VerifierArtifact { .. } => panic!("terminal fixture must contain exact matrices"),
    }

    terms.retain(|_, coefficient| *coefficient != F::ZERO);
    terms
}

pub fn audit(fixture: &StreamingTerminalAuditFixture) -> StreamingTerminalXOutHashProvenance {
    let lifecycle_arm = NebulaFPrimeStreamingLifecycleArm::Recursive;
    let hash_audit = fixture.lifecycle.after_x_out_hash_audit(lifecycle_arm);
    let hash = hash_audit.hash();
    assert_eq!(hash.input_cols.len(), 32);
    assert_eq!(hash.rounds.len(), X_OUT_HASH_PERMUTATIONS);
    assert_eq!(hash.output_cols.len(), 4);
    assert_eq!(hash_audit.public_words().len(), 4);

    let common = fixture.relation.scheduled_relation().common_relation();
    let compiler = common
        .selective_compiler_audit()
        .expect("terminal lifecycle selective compiler audit");
    let mut rewrites = compiler
        .rows()
        .rewrites()
        .iter()
        .filter(|rewrite| {
            rewrite.arm() == RECURSIVE_ARM_INDEX
                && rewrite.kind() == SelectiveRewriteKind::Poseidon2
                && rewrite.source_rows().len() == 1
                && hash.row_start <= rewrite.source_rows()[0].start
                && rewrite.source_rows()[0].end <= hash.row_end
        })
        .collect::<Vec<_>>();
    rewrites.sort_unstable_by_key(|rewrite| rewrite.source_rows()[0].start);
    assert_eq!(rewrites.len(), X_OUT_HASH_PERMUTATIONS);
    assert!(rewrites.windows(2).all(|pair| {
        pair[0].source_rows()[0].end <= pair[1].source_rows()[0].start
            && pair[0].emitted_rows().end == pair[1].emitted_rows().start
    }));
    let hash_emitted_rows = rewrites
        .iter()
        .flat_map(|rewrite| rewrite.emitted_rows())
        .collect::<Vec<_>>();
    assert_eq!(
        hash_emitted_rows
            .iter()
            .copied()
            .collect::<BTreeSet<_>>()
            .len(),
        hash_emitted_rows.len()
    );

    let canonical_source_rows = hash_audit
        .public_words()
        .iter()
        .flat_map(|word| word.canonical_rows())
        .collect::<BTreeSet<_>>();
    assert_eq!(canonical_source_rows.len(), 4 * 69);
    let equality_source_rows = hash_audit
        .public_words()
        .iter()
        .flat_map(|word| word.equality_rows().iter().copied())
        .collect::<BTreeSet<_>>();
    assert_eq!(equality_source_rows.len(), 4 * 64);
    let arm_mapping = &compiler.rows().arms()[RECURSIVE_ARM_INDEX];
    let retained_source_rows = canonical_source_rows
        .iter()
        .chain(&equality_source_rows)
        .copied()
        .collect::<BTreeSet<_>>();
    let retained_row_pairs = retained_source_rows
        .iter()
        .map(|&source_row| {
            let matches = arm_mapping
                .source_runs()
                .iter()
                .filter(|run| {
                    run.disposition() == SelectiveSourceRowDisposition::Retained
                        && run.source_rows().contains(&source_row)
                })
                .map(|run| {
                    let rows = run.source_rows();
                    let emitted_start = run.emitted_start().expect("retained run emitted start");
                    (source_row, emitted_start + source_row - rows.start)
                })
                .collect::<Vec<_>>();
            let [pair] = matches.as_slice() else {
                panic!("terminal XOut equality source row {source_row} retained match count != 1");
            };
            *pair
        })
        .collect::<Vec<_>>();
    assert_eq!(retained_row_pairs.len(), 4 * (69 + 64));
    let mut emitted_rows = hash_emitted_rows.clone();
    emitted_rows.extend(
        retained_row_pairs
            .iter()
            .map(|&(_, emitted_row)| emitted_row),
    );
    emitted_rows.sort_unstable();
    emitted_rows.dedup();
    assert_eq!(emitted_rows.len(), hash_emitted_rows.len() + retained_row_pairs.len());

    let source_columns = hash
        .input_cols
        .iter()
        .copied()
        .chain(hash.output_cols)
        .chain(hash.rounds[0].state_before_cols)
        .chain(hash.rounds[0].permutation_input_cols)
        .chain(
            hash.rounds
                .iter()
                .flat_map(|round| round.permutation_input_cols.iter().copied()),
        )
        .chain(hash_audit.public_words().iter().flat_map(|word| {
            [word.field_col()]
                .into_iter()
                .chain(word.canonical_bit_cols().iter().copied())
                .chain(word.public_bit_cols().iter().copied())
        }))
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    let common_arms = [
        fixture
            .lifecycle
            .arm(NebulaFPrimeStreamingLifecycleArm::Base)
            .clone(),
        fixture.lifecycle.arm(lifecycle_arm).clone(),
    ];
    let projected = audit_multi_branch_selective_rows_with_complete_source_provenance_with_alignment(
        &common_arms,
        0,
        D,
        0,
        &emitted_rows,
        RECURSIVE_ARM_INDEX,
        &source_columns,
        &retained_row_pairs,
    )
    .expect("project terminal XOut hash source provenance");
    assert_eq!(projected.row_artifacts().len(), emitted_rows.len());
    assert!(projected
        .row_artifacts()
        .iter()
        .zip(&emitted_rows)
        .all(|(artifact, row)| artifact.emitted_row() == *row));
    let source = projected
        .source_provenance()
        .expect("terminal XOut hash complete source provenance");
    assert_eq!(source.arm(), RECURSIVE_ARM_INDEX);
    assert_eq!(
        source.poseidon2_sbox_steps().len() + source.poseidon2_output_steps().len(),
        hash_emitted_rows.len()
    );
    assert_eq!(
        source
            .poseidon2_sbox_steps()
            .iter()
            .map(|step| step.emitted_row())
            .chain(
                source
                    .poseidon2_output_steps()
                    .iter()
                    .map(|step| step.emitted_row())
            )
            .collect::<BTreeSet<_>>(),
        hash_emitted_rows.iter().copied().collect()
    );
    assert_eq!(
        source
            .poseidon2_sbox_steps()
            .iter()
            .map(|step| step.rewrite_id())
            .chain(
                source
                    .poseidon2_output_steps()
                    .iter()
                    .map(|step| step.rewrite_id())
            )
            .collect::<BTreeSet<_>>()
            .len(),
        X_OUT_HASH_PERMUTATIONS
    );
    assert_eq!(
        source
            .retained_steps()
            .iter()
            .map(|step| (step.source_row(), step.emitted_row()))
            .collect::<Vec<_>>(),
        retained_row_pairs
    );
    assert_eq!(
        source
            .requested_source_images()
            .iter()
            .map(|image| image.column())
            .collect::<Vec<_>>(),
        source_columns
    );
    let selected_rows = emitted_rows.iter().copied().collect::<BTreeSet<_>>();
    let common_structure = common.structure();
    let final_structure = fixture.relation.structure();
    assert_eq!(common_structure.matrices.len(), final_structure.matrices.len());
    for (port, (common_matrix, final_matrix)) in common_structure
        .matrices
        .iter()
        .zip(&final_structure.matrices)
        .enumerate()
    {
        assert_eq!(
            selected_matrix_terms(common_matrix, &selected_rows),
            selected_matrix_terms(final_matrix, &selected_rows),
            "terminal XOut hash final matrix port {port}"
        );
    }

    StreamingTerminalXOutHashProvenance {
        source_rows: hash.row_start..hash.row_end,
        emitted_rows,
        projected,
    }
}

fn lean_list(values: impl IntoIterator<Item = usize>) -> String {
    format!(
        "[{}]",
        values
            .into_iter()
            .map(|value| value.to_string())
            .collect::<Vec<_>>()
            .join(", ")
    )
}

fn lean_range(range: Range<usize>) -> String {
    format!("{{ start := {}, stop := {} }}", range.start, range.end)
}

fn compact_ranges(values: impl IntoIterator<Item = usize>) -> Vec<Range<usize>> {
    let mut values = values.into_iter().collect::<Vec<_>>();
    values.sort_unstable();
    values.dedup();
    let mut ranges = Vec::<Range<usize>>::new();
    for value in values {
        if ranges.last().is_some_and(|range| range.end == value) {
            ranges.last_mut().expect("checked range").end += 1;
        } else {
            ranges.push(value..value + 1);
        }
    }
    ranges
}

fn render_ranges(ranges: &[Range<usize>]) -> String {
    format!(
        "[{}]",
        ranges
            .iter()
            .cloned()
            .map(lean_range)
            .collect::<Vec<_>>()
            .join(", ")
    )
}

fn render_absolute_port(port: &SelectiveProjectedPort) -> String {
    assert!(port.seeded_blocks().is_empty());
    let explicit = port
        .explicit()
        .iter()
        .map(|term| {
            format!(
                "{{ column := {}, coefficient := {} }}",
                term.column(),
                term.coefficient().as_canonical_u64()
            )
        })
        .collect::<Vec<_>>()
        .join(", ");
    let geometric = port
        .geometric_runs()
        .iter()
        .map(|run| {
            format!(
                "{{ columnStart := {}, length := {}, initial := {}, ratio := {} }}",
                run.column_start(),
                run.length(),
                run.initial().as_canonical_u64(),
                run.ratio().as_canonical_u64()
            )
        })
        .collect::<Vec<_>>()
        .join(", ");
    format!("{{ explicit := [{explicit}], geometric := [{geometric}] }}")
}

struct Poseidon2CallPlacement {
    round_index: usize,
    rewrite_id: usize,
    source_rows: Range<usize>,
    final_rows: Range<usize>,
    final_columns: usize,
    selector_column: usize,
    local_slot_start: usize,
    input_source_columns: [usize; 8],
    input_images: Vec<SelectiveProjectedPort>,
}

impl Poseidon2CallPlacement {
    fn render(&self) -> String {
        let images = self
            .input_source_columns
            .iter()
            .zip(&self.input_images)
            .map(|(source_column, port)| {
                format!(
                    "{{ sourceColumn := {source_column}, port := {} }}",
                    render_absolute_port(port)
                )
            })
            .collect::<Vec<_>>()
            .join(", ");
        format!(
            "{{ roundIndex := {}, rewriteId := {}, sourceRows := {}, finalRows := {}, finalColumns := {}, selectorColumn := {}, localSlotStart := {}, slotWidth := {}, localSlotCount := {}, inputSourceColumns := {}, inputImages := [{images}] }}",
            self.round_index,
            self.rewrite_id,
            lean_range(self.source_rows.clone()),
            lean_range(self.final_rows.clone()),
            self.final_columns,
            self.selector_column,
            self.local_slot_start,
            SLOT_WIDTH,
            POSEIDON2_SBOX_ROWS,
            lean_list(self.input_source_columns),
        )
    }
}

fn poseidon2_call_placements(
    fixture: &StreamingTerminalAuditFixture,
    provenance: &StreamingTerminalXOutHashProvenance,
    selector_column: usize,
) -> Vec<Poseidon2CallPlacement> {
    let projected = provenance.projected();
    let source = projected
        .source_provenance()
        .expect("terminal XOut hash complete source provenance");
    let rows_by_emitted = projected
        .row_artifacts()
        .iter()
        .map(|row| (row.emitted_row(), row))
        .collect::<BTreeMap<_, _>>();
    let mut grouped = BTreeMap::<usize, Vec<_>>::new();
    for step in source.poseidon2_sbox_steps() {
        grouped.entry(step.rewrite_id()).or_default().push(step);
    }
    let mut calls = grouped.into_values().collect::<Vec<_>>();
    for steps in &mut calls {
        steps.sort_unstable_by_key(|step| step.emitted_row());
    }
    calls.sort_unstable_by_key(|steps| steps[0].source_rows()[0].0);

    let hash = fixture
        .lifecycle
        .after_x_out_hash_audit(NebulaFPrimeStreamingLifecycleArm::Recursive);
    assert_eq!(calls.len(), hash.hash().rounds.len());
    calls
        .into_iter()
        .zip(&hash.hash().rounds)
        .enumerate()
        .map(|(round_index, (steps, round))| {
            assert_eq!(steps.len(), POSEIDON2_SBOX_ROWS);
            let [(source_start, source_stop)] = steps[0].source_rows() else {
                panic!("terminal Poseidon2 call must own one source row range");
            };
            assert!(steps
                .iter()
                .all(|step| step.source_rows() == [(*source_start, *source_stop)]));
            let final_start = steps[0].emitted_row();
            assert!(steps
                .iter()
                .enumerate()
                .all(|(index, step)| step.emitted_row() == final_start + index));
            let rows = steps
                .iter()
                .map(|step| {
                    *rows_by_emitted
                        .get(&step.emitted_row())
                        .expect("terminal Poseidon2 final row")
                })
                .collect::<Vec<_>>();
            let final_columns = rows[0].columns();
            assert!(rows.iter().all(|row| row.columns() == final_columns));

            let mut slot_starts = rows
                .iter()
                .flat_map(|row| row.ports())
                .flat_map(|port| port.geometric_runs())
                .map(|run| {
                    assert_eq!(run.length(), SLOT_WIDTH);
                    assert_eq!(run.ratio(), F::from_u64(3));
                    run.column_start()
                })
                .collect::<BTreeSet<_>>()
                .into_iter();
            let first = slot_starts.next().expect("terminal Poseidon2 final slot");
            let mut slot_runs = vec![(first, 1usize)];
            for start in slot_starts {
                let (run_start, run_length) = slot_runs.last_mut().expect("terminal Poseidon2 slot run");
                if start == *run_start + *run_length * SLOT_WIDTH {
                    *run_length += 1;
                } else {
                    slot_runs.push((start, 1));
                }
            }
            let long_runs = slot_runs
                .into_iter()
                .filter(|(_, length)| *length >= POSEIDON2_SBOX_ROWS)
                .collect::<Vec<_>>();
            let [(long_start, long_length)] = long_runs.as_slice() else {
                panic!("terminal Poseidon2 local slot run must be unique");
            };
            let input_prefix = *long_length - POSEIDON2_SBOX_ROWS;
            assert!(input_prefix == 0 || input_prefix == 8);
            let local_slot_start = *long_start + input_prefix * SLOT_WIDTH;
            assert!(local_slot_start + POSEIDON2_SBOX_ROWS * SLOT_WIDTH <= final_columns);

            let input_source_columns = round.permutation_input_cols;
            let input_images = input_source_columns
                .iter()
                .map(|column| {
                    source
                        .requested_source_images()
                        .iter()
                        .find(|image| image.column() == *column)
                        .unwrap_or_else(|| panic!("missing terminal Poseidon2 source image for column {column}"))
                        .port()
                        .clone()
                })
                .collect::<Vec<_>>();
            assert_eq!(input_images.len(), 8);
            assert!(input_images
                .iter()
                .all(|port| port.seeded_blocks().is_empty()));
            assert!(rows
                .iter()
                .flat_map(|row| row.ports())
                .flat_map(|port| port.explicit())
                .any(|term| term.column() == selector_column));

            Poseidon2CallPlacement {
                round_index,
                rewrite_id: steps[0].rewrite_id(),
                source_rows: *source_start..*source_stop,
                final_rows: final_start..final_start + POSEIDON2_SBOX_ROWS,
                final_columns,
                selector_column,
                local_slot_start,
                input_source_columns,
                input_images,
            }
        })
        .collect()
}

struct FirstPoseidon2Leaf {
    steps: Vec<SelectiveProjectedPoseidon2SboxStep>,
    rows: Vec<SelectiveProjectedRowArtifact>,
    rewrite_id: usize,
    source_rows: Range<usize>,
    final_rows: Range<usize>,
    final_columns: usize,
    zero_source_column: usize,
    input_source_columns: [usize; 4],
    local_source_columns: Range<usize>,
    selector_column: usize,
    external_slots: Vec<usize>,
    local_slots: Vec<usize>,
    source_images: Vec<SelectiveProjectedPort>,
}

impl FirstPoseidon2Leaf {
    fn source_column(&self, column: usize) -> String {
        if column == self.zero_source_column {
            return ".externalB 0".to_owned();
        }
        if let Some(lane) = self
            .input_source_columns
            .iter()
            .position(|candidate| *candidate == column)
        {
            return format!(".externalA {lane}");
        }
        if self.local_source_columns.contains(&column) {
            return format!(".local {}", column - self.local_source_columns.start);
        }
        panic!("unclassified first terminal Poseidon2 source column {column}");
    }

    fn slot(&self, start: usize) -> String {
        if let Ok(index) = self.external_slots.binary_search(&start) {
            return format!(".externalA {index}");
        }
        if let Ok(index) = self.local_slots.binary_search(&start) {
            return format!(".local {index}");
        }
        panic!("unclassified first terminal Poseidon2 final slot {start}");
    }

    fn source_linear_combination(&self, value: &SelectiveProjectedSourceLinearCombination) -> String {
        let terms = value
            .terms()
            .iter()
            .map(|term| {
                format!(
                    "{{ column := {}, coefficient := {} }}",
                    self.source_column(term.column()),
                    term.coefficient().as_canonical_u64()
                )
            })
            .collect::<Vec<_>>()
            .join(", ");
        format!(
            "{{ constant := {}, terms := [{terms}] }}",
            value.constant().as_canonical_u64()
        )
    }

    fn port(&self, port: &SelectiveProjectedPort) -> String {
        assert!(port.seeded_blocks().is_empty());
        let explicit = port
            .explicit()
            .iter()
            .map(|term| {
                let column = match term.column() {
                    0 => ".one",
                    column if column == self.selector_column => ".selector",
                    column => panic!("unclassified first terminal Poseidon2 explicit column {column}"),
                };
                format!(
                    "{{ column := {column}, coefficient := {} }}",
                    term.coefficient().as_canonical_u64()
                )
            })
            .collect::<Vec<_>>()
            .join(", ");
        let geometric = port
            .geometric_runs()
            .iter()
            .map(|run| {
                assert_eq!(run.length(), SLOT_WIDTH);
                format!(
                    "{{ slot := {}, initial := {}, ratio := {} }}",
                    self.slot(run.column_start()),
                    run.initial().as_canonical_u64(),
                    run.ratio().as_canonical_u64()
                )
            })
            .collect::<Vec<_>>()
            .join(", ");
        format!("{{ explicit := [{explicit}], geometric := [{geometric}] }}")
    }

    fn write_step(&self, rendered: &mut String, index: usize) {
        let step = &self.steps[index];
        writeln!(
            rendered,
            "\ndef rawStep{index:02} : RawStep where\n  rowOffset := {index}\n  input := {}\n  output := {}",
            self.source_linear_combination(step.input()),
            self.source_linear_combination(step.output()),
        )
        .expect("render first terminal Poseidon2 step");
    }

    fn write_row(&self, rendered: &mut String, index: usize) {
        writeln!(
            rendered,
            "\ndef rawRow{index:02} : RawRow where\n  rowOffset := {index}\n  ports := ["
        )
        .expect("render first terminal Poseidon2 row header");
        for (port_index, port) in self.rows[index].ports().iter().enumerate() {
            let separator = if port_index == 0 { "    " } else { "  , " };
            writeln!(rendered, "{separator}{}", self.port(port)).expect("render first terminal Poseidon2 port");
        }
        writeln!(rendered, "  ]").expect("render first terminal Poseidon2 row footer");
    }

    fn render_steps(&self) -> String {
        let mut rendered = String::new();
        writeln!(
            rendered,
            "import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyPoseidonLeafSchema\n\n\
             /-! Generated exact source steps for the first terminal XOut Poseidon2 leaf. -/\n\n\
             set_option autoImplicit false\n\n\
             namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingTerminalXOutPoseidonFirstLeaf\n\n\
             open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyPoseidonLeafSchema"
        )
        .expect("render first terminal Poseidon2 step preamble");
        for index in 0..self.steps.len() {
            self.write_step(&mut rendered, index);
        }
        writeln!(rendered, "\ndef rawSteps : List RawStep := [").expect("render first terminal step list header");
        for index in 0..self.steps.len() {
            let separator = if index == 0 { "  " } else { ", " };
            writeln!(rendered, "{separator}rawStep{index:02}").expect("render first terminal step item");
        }
        writeln!(
            rendered,
            "]\n\nend Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingTerminalXOutPoseidonFirstLeaf"
        )
        .expect("render first terminal step footer");
        rendered
    }

    fn render_rows(&self, start: usize, stop: usize, shard: usize) -> String {
        let mut rendered = String::new();
        writeln!(
            rendered,
            "import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyPoseidonLeafSchema\n\n\
             /-! Generated exact final ports for first terminal XOut Poseidon2 rows {start} through {stop}. -/\n\n\
             set_option autoImplicit false\n\n\
             namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingTerminalXOutPoseidonFirstLeaf\n\n\
             open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyPoseidonLeafSchema"
        )
        .expect("render first terminal Poseidon2 row preamble");
        for index in start..stop {
            self.write_row(&mut rendered, index);
        }
        writeln!(rendered, "\ndef rawRows{shard} : List RawRow := [").expect("render first terminal row list header");
        for index in start..stop {
            let separator = if index == start { "  " } else { ", " };
            writeln!(rendered, "{separator}rawRow{index:02}").expect("render first terminal row item");
        }
        writeln!(
            rendered,
            "]\n\nend Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingTerminalXOutPoseidonFirstLeaf"
        )
        .expect("render first terminal row footer");
        rendered
    }

    fn render_images(&self) -> String {
        let mut rendered = String::new();
        writeln!(
            rendered,
            "import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyPoseidonLeafSchema\n\n\
             /-! Generated exact external source images for the first terminal XOut Poseidon2 leaf. -/\n\n\
             set_option autoImplicit false\n\n\
             namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingTerminalXOutPoseidonFirstLeaf\n\n\
             open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyPoseidonLeafSchema"
        )
        .expect("render first terminal Poseidon2 image preamble");
        for (index, image) in self.source_images.iter().enumerate() {
            writeln!(
                rendered,
                "\ndef rawImage{index} : RawSourceImage where\n  lane := {index}\n  port := {}",
                self.port(image)
            )
            .expect("render first terminal Poseidon2 image");
        }
        writeln!(rendered, "\ndef rawImages : List RawSourceImage := [")
            .expect("render first terminal image list header");
        for index in 0..self.source_images.len() {
            let separator = if index == 0 { "  " } else { ", " };
            writeln!(rendered, "{separator}rawImage{index}").expect("render first terminal image item");
        }
        writeln!(
            rendered,
            "]\n\nend Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingTerminalXOutPoseidonFirstLeaf"
        )
        .expect("render first terminal image footer");
        rendered
    }

    fn placement(&self) -> String {
        format!(
            "{{ rewriteId := {}, sourceRows := {}, finalRows := {}, finalColumns := {}, selectorColumn := {}, externalSlotStarts := {}, localSlotStart := {}, slotWidth := {}, localSlotCount := {} }}",
            self.rewrite_id,
            lean_range(self.source_rows.clone()),
            lean_range(self.final_rows.clone()),
            self.final_columns,
            self.selector_column,
            lean_list(self.external_slots.iter().copied()),
            self.local_slots[0],
            SLOT_WIDTH,
            POSEIDON2_SBOX_ROWS,
        )
    }
}

fn first_poseidon2_leaf(
    fixture: &StreamingTerminalAuditFixture,
    provenance: &StreamingTerminalXOutHashProvenance,
) -> FirstPoseidon2Leaf {
    let projected = provenance.projected();
    let source = projected
        .source_provenance()
        .expect("terminal XOut hash complete source provenance");
    let rewrite_id = source
        .poseidon2_sbox_steps()
        .iter()
        .min_by_key(|step| step.emitted_row())
        .expect("first terminal XOut Poseidon2 step")
        .rewrite_id();
    let mut steps = source
        .poseidon2_sbox_steps()
        .iter()
        .filter(|step| step.rewrite_id() == rewrite_id)
        .cloned()
        .collect::<Vec<_>>();
    steps.sort_unstable_by_key(|step| step.emitted_row());
    assert_eq!(steps.len(), POSEIDON2_SBOX_ROWS);
    let rows_by_emitted = projected
        .row_artifacts()
        .iter()
        .map(|row| (row.emitted_row(), row))
        .collect::<BTreeMap<_, _>>();
    let rows = steps
        .iter()
        .map(|step| {
            (*rows_by_emitted
                .get(&step.emitted_row())
                .expect("first terminal Poseidon2 final row"))
            .clone()
        })
        .collect::<Vec<_>>();
    let (source_row_start, source_row_stop) = match steps[0].source_rows() {
        [(start, stop)] => (*start, *stop),
        _ => panic!("first terminal Poseidon2 leaf must own one source row range"),
    };
    assert!(steps
        .iter()
        .all(|step| step.source_rows() == [(source_row_start, source_row_stop)]));
    let final_row_start = steps[0].emitted_row();
    assert!(steps
        .iter()
        .enumerate()
        .all(|(index, step)| step.emitted_row() == final_row_start + index));
    let final_columns = rows[0].columns();
    assert!(rows.iter().all(|row| row.columns() == final_columns));

    let hash_audit = fixture
        .lifecycle
        .after_x_out_hash_audit(NebulaFPrimeStreamingLifecycleArm::Recursive);
    let first_round = &hash_audit.hash().rounds[0];
    let input_source_columns: [usize; 4] = first_round.permutation_input_cols[..4]
        .try_into()
        .expect("first terminal Poseidon2 rate inputs");
    let zero_source_column = first_round.state_before_cols[0];
    assert!(first_round
        .state_before_cols
        .iter()
        .all(|column| *column == zero_source_column));
    let local_source_columns =
        first_round.first_allocated_column..first_round.first_allocated_column + POSEIDON2_SOURCE_COLUMNS;
    let source_image_columns = [zero_source_column]
        .into_iter()
        .chain(input_source_columns)
        .collect::<Vec<_>>();
    let source_images = source_image_columns
        .iter()
        .map(|column| {
            source
                .requested_source_images()
                .iter()
                .find(|image| image.column() == *column)
                .unwrap_or_else(|| panic!("missing first terminal Poseidon2 source image for column {column}"))
                .port()
                .clone()
        })
        .collect::<Vec<_>>();
    assert_eq!(source_images.len(), 5);
    let all_slot_starts = rows
        .iter()
        .flat_map(|row| row.ports())
        .flat_map(|port| port.geometric_runs())
        .map(|run| {
            assert_eq!(run.length(), SLOT_WIDTH);
            assert_eq!(run.ratio(), F::from_u64(3));
            run.column_start()
        })
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    assert!(all_slot_starts.len() >= POSEIDON2_SBOX_ROWS);
    let local_run_starts = (0..=all_slot_starts.len() - POSEIDON2_SBOX_ROWS)
        .filter(|start| {
            all_slot_starts[*start..*start + POSEIDON2_SBOX_ROWS]
                .windows(2)
                .all(|pair| pair[1] == pair[0] + SLOT_WIDTH)
        })
        .collect::<Vec<_>>();
    let [local_run_start] = local_run_starts.as_slice() else {
        panic!("first terminal Poseidon2 local slot run is not unique");
    };
    let local_slots = all_slot_starts[*local_run_start..*local_run_start + POSEIDON2_SBOX_ROWS].to_vec();
    let local_slot_set = local_slots.iter().copied().collect::<BTreeSet<_>>();
    let external_slots = all_slot_starts
        .into_iter()
        .filter(|start| !local_slot_set.contains(start))
        .collect::<Vec<_>>();
    assert_eq!(external_slots.len(), 3);
    let explicit_columns = rows
        .iter()
        .flat_map(|row| row.ports())
        .flat_map(|port| port.explicit())
        .map(|term| term.column())
        .collect::<BTreeSet<_>>();
    let selector_column = *explicit_columns
        .iter()
        .find(|column| **column != 0)
        .expect("first terminal Poseidon2 selector");
    assert_eq!(explicit_columns, [0, selector_column].into_iter().collect());

    FirstPoseidon2Leaf {
        steps,
        rows,
        rewrite_id,
        source_rows: source_row_start..source_row_stop,
        final_rows: final_row_start..final_row_start + POSEIDON2_SBOX_ROWS,
        final_columns,
        zero_source_column,
        input_source_columns,
        local_source_columns,
        selector_column,
        external_slots,
        local_slots,
        source_images,
    }
}

pub fn first_leaf_artifacts(fixture: &StreamingTerminalAuditFixture) -> Vec<(PathBuf, String)> {
    let provenance = audit(fixture);
    let leaf = first_poseidon2_leaf(fixture, &provenance);
    [
        (FIRST_LEAF_STEPS_ARTIFACT_PATH, leaf.render_steps()),
        (FIRST_LEAF_ROWS_0_ARTIFACT_PATH, leaf.render_rows(0, 43, 0)),
        (
            FIRST_LEAF_ROWS_1_ARTIFACT_PATH,
            leaf.render_rows(43, POSEIDON2_SBOX_ROWS, 1),
        ),
        (FIRST_LEAF_IMAGES_ARTIFACT_PATH, leaf.render_images()),
    ]
    .into_iter()
    .map(|(path, contents)| (Path::new(env!("CARGO_MANIFEST_DIR")).join(path), contents))
    .collect()
}

fn render_round(round: &neo_fold_clean::engine::r1cs_circuit::builder::Poseidon2HashRoundAudit) -> String {
    let kind = match &round.kind {
        neo_fold_clean::engine::r1cs_circuit::builder::Poseidon2HashRoundAuditKind::Absorb { chunk_cols } => {
            format!(".absorb {}", lean_list(chunk_cols.iter().copied()))
        }
        neo_fold_clean::engine::r1cs_circuit::builder::Poseidon2HashRoundAuditKind::Pad => ".pad".to_owned(),
    };
    format!(
        "    {{ kind := {kind}, stateBeforeColumns := {}, permutationInputColumns := {}, permutationOutputColumns := {}, definingRows := {}, call := {{ rowStart := {}, rowEnd := {}, inputColumns := {}, firstAllocatedColumn := {} }} }},",
        lean_list(round.state_before_cols),
        lean_list(round.permutation_input_cols),
        lean_list(round.permutation_output_cols),
        lean_list(round.defining_rows.iter().copied()),
        round.defining_rows.len(),
        round.defining_rows.len() + 600,
        lean_list(round.permutation_input_cols),
        round.first_allocated_column,
    )
}

fn render_source_terms(value: &SelectiveProjectedSourceLinearCombination) -> String {
    format!(
        "[{}]",
        value
            .terms()
            .iter()
            .map(|term| format!(
                "{{ column := {}, coefficient := {} }}",
                term.column(),
                term.coefficient().as_canonical_u64()
            ))
            .collect::<Vec<_>>()
            .join(", ")
    )
}

pub fn render(fixture: &StreamingTerminalAuditFixture) -> String {
    let provenance = audit(fixture);
    let first_leaf = first_poseidon2_leaf(fixture, &provenance);
    let call_placements = poseidon2_call_placements(fixture, &provenance, first_leaf.selector_column);
    let hash = fixture
        .lifecycle
        .after_x_out_hash_audit(NebulaFPrimeStreamingLifecycleArm::Recursive);
    let trace = hash.hash();
    let source = provenance
        .projected()
        .source_provenance()
        .expect("terminal XOut hash complete source provenance");
    let render_source_images = |columns: &[usize]| {
        format!(
            "[{}]",
            columns
                .iter()
                .map(|column| {
                    let image = source
                        .requested_source_images()
                        .iter()
                        .find(|image| image.column() == *column)
                        .unwrap_or_else(|| panic!("missing terminal XOut source image for column {column}"));
                    format!(
                        "{{ sourceColumn := {column}, port := {} }}",
                        render_absolute_port(image.port())
                    )
                })
                .collect::<Vec<_>>()
                .join(", ")
        )
    };
    let x_out_images = render_source_images(&trace.input_cols);
    let output_images = render_source_images(&trace.output_cols);
    let rows_by_emitted = provenance
        .projected()
        .row_artifacts()
        .iter()
        .map(|row| (row.emitted_row(), row))
        .collect::<BTreeMap<_, _>>();
    let mut output_steps = source.poseidon2_output_steps().iter().collect::<Vec<_>>();
    output_steps.sort_unstable_by_key(|step| step.emitted_row());
    assert_eq!(output_steps.len(), trace.output_cols.len());
    let final_rewrite_id = call_placements
        .last()
        .expect("terminal XOut Poseidon2 calls are empty")
        .rewrite_id;
    let output_copies = format!(
        "[{}]",
        output_steps
            .iter()
            .enumerate()
            .map(|(lane, step)| {
                assert_eq!(step.rewrite_id(), final_rewrite_id);
                assert_eq!(step.output().constant(), F::ZERO);
                let [output_term] = step.output().terms() else {
                    panic!("terminal XOut output copy must own one source column");
                };
                assert_eq!(output_term.coefficient(), F::ONE);
                assert_eq!(output_term.column(), trace.output_cols[lane]);
                let row = rows_by_emitted
                    .get(&step.emitted_row())
                    .expect("missing terminal XOut output-copy final row");
                let (source_start, source_stop) = step
                    .source_rows()
                    .first()
                    .copied()
                    .filter(|_| step.source_rows().len() == 1)
                    .expect("terminal XOut output copy must own one source range");
                let final_ports = format!(
                    "[{}]",
                    row.ports()
                        .iter()
                        .map(render_absolute_port)
                        .collect::<Vec<_>>()
                        .join(", ")
                );
                format!(
                    "{{ lane := {lane}, rewriteId := {}, sourceRows := {}, finalRow := {}, finalRows := {}, finalColumns := {}, selectorColumn := {}, outputSourceColumn := {}, linearFormConstant := {}, linearFormTerms := {}, finalPorts := {final_ports} }}",
                    step.rewrite_id(),
                    lean_range(source_start..source_stop),
                    step.emitted_row(),
                    row.rows(),
                    row.columns(),
                    first_leaf.selector_column,
                    output_term.column(),
                    step.linear_form().constant().as_canonical_u64(),
                    render_source_terms(step.linear_form()),
                )
            })
            .collect::<Vec<_>>()
            .join(", ")
    );
    let mut rounds = String::from("[\n");
    for round in &trace.rounds {
        writeln!(rounds, "{}", render_round(round)).expect("render XOut hash round");
    }
    rounds.push_str("  ]");

    let mut words = String::from("[\n");
    for word in hash.public_words() {
        writeln!(
            words,
            "    {{ fieldColumn := {}, canonicalBitColumns := {}, highIsMaxColumn := {}, inverseColumn := {}, publicBitColumns := {}, canonicalRows := {}, equalityRows := {} }},",
            word.field_col(),
            lean_list(word.canonical_bit_cols().iter().copied()),
            word.high_is_max_col(),
            word.inverse_col(),
            lean_list(word.public_bit_cols().iter().copied()),
            lean_range(word.canonical_rows()),
            lean_list(word.equality_rows().iter().copied()),
        )
        .expect("render XOut public word");
    }
    words.push_str("  ]");
    let call_placement_defs = call_placements
        .iter()
        .enumerate()
        .map(|(index, placement)| {
            format!(
                "def callPlacement{index} : PoseidonCallPlacement := {}",
                placement.render()
            )
        })
        .collect::<Vec<_>>()
        .join("\n\n");
    let call_placements = format!(
        "[{}]",
        (0..call_placements.len())
            .map(|index| format!("callPlacement{index}"))
            .collect::<Vec<_>>()
            .join(", ")
    );

    let source_rows = trace.row_start..trace.row_end;
    let all_source_rows = source_rows
        .clone()
        .chain(
            hash.public_words()
                .iter()
                .flat_map(|word| word.canonical_rows()),
        )
        .chain(
            hash.public_words()
                .iter()
                .flat_map(|word| word.equality_rows().iter().copied()),
        );
    let source_row_runs = compact_ranges(all_source_rows);
    let final_row_runs = compact_ranges(provenance.emitted_rows().iter().copied());
    format!(
        "import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingTerminalXOutPublicHashSchema\n\n\
         /-! Generated compact certificate for the exact recursive-terminal XOut public hash.\n\n\
         Emits constraints: no. Rust emits the source rows and their final selective projection.\n\
         -/\n\n\
         set_option autoImplicit false\n\n\
         namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingTerminalXOutPublicHash\n\n\
         open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingTerminalXOutPublicHash.Artifact\n\n\
         def rounds : List Nightstream.Implementation.R1CS.Poseidon2Sponge.Round := {}\n\n\
         def trace : Nightstream.Implementation.R1CS.Poseidon2Sponge.Trace :=\n  {{ inputColumns := {}, zeroColumn := {}, zeroRow := {}, rounds := rounds, outputColumns := {} }}\n\n\
         {}\n\n\
         def callPlacements : List PoseidonCallPlacement := {}\n\n\
         def outputCopies : List OutputCopyPlacement := {}\n\n\
         def xOutImages : List SourceImage := {}\n\n\
         def outputImages : List SourceImage := {}\n\n\
         def publicWords : List PublicWord := {}\n\n\
         def rawArtifact : RawArtifact :=\n  {{ schemaVersion := 2,\n    profileId := \"nightstream/goldilocks/b2-k16/streaming-terminal-x-out-public-hash/v2\",\n    sourceArtifactIdentity := \"rust:nightstream/streaming-terminal-lifecycle/source-rows/v1\",\n    finalArtifactIdentity := \"rust:nightstream/streaming-selective-ccs/final-rows/v1\",\n    lifecycleScope := \"recursive-terminal-arm-435\",\n    rowFamily := \"terminal.streaming.x_out.public_hash\",\n    sourceHashRows := {}, sourceRowRuns := {}, finalRowRuns := {},\n    firstLeafPlacement := {}, callPlacements := callPlacements, outputCopies := outputCopies,\n    xOutImages := xOutImages, outputImages := outputImages,\n    trace := trace, publicWords := publicWords }}\n\n\
         end Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingTerminalXOutPublicHash\n",
        rounds,
        lean_list(trace.input_cols.iter().copied()),
        trace.zero_col,
        trace.zero_row,
        lean_list(trace.output_cols),
        call_placement_defs,
        call_placements,
        output_copies,
        x_out_images,
        output_images,
        words,
        lean_range(source_rows),
        render_ranges(&source_row_runs),
        render_ranges(&final_row_runs),
        first_leaf.placement(),
    )
}

pub fn artifact_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(ARTIFACT_PATH)
}

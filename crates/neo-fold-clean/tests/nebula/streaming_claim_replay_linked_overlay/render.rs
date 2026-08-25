//! Lean rendering primitives for the claim-replay linked-overlay receipt.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;

use neo_fold_clean::frontends::r1cs_f_prime::{
    SelectiveLinearDefinitionAudit, SelectiveProjectedPort, SelectiveProjectedSourceDecoderTemplateInstances,
    SelectiveProjectedSourceDefinition, SelectiveProjectedSourceLinearCombination, SelectiveProjectedSourceProvenance,
    SelectiveProjectedSourceSlot,
};
use neo_math::F;
use p3_field::{PrimeCharacteristicRing, PrimeField64};

pub(super) fn lean_decoder_instances(instances: SelectiveProjectedSourceDecoderTemplateInstances) -> String {
    format!(
        "{{ sourceStart := {}, count := {}, sourceStride := {}, finalStart := {}, finalStride := {}, referenceStart := {}, referenceStride := {}, referenceFinalStart := {}, referenceFinalStride := {} }}",
        instances.source_start(),
        instances.count(),
        instances.source_stride(),
        instances.final_start(),
        instances.final_stride(),
        instances.reference_start(),
        instances.reference_stride(),
        instances.reference_final_start(),
        instances.reference_final_stride(),
    )
}

pub(super) fn lean_field(value: F) -> u64 {
    value.as_canonical_u64()
}

pub(super) fn canonical_poseidon_call(
    steps: &[neo_fold_clean::frontends::r1cs_f_prime::SelectiveProjectedPoseidon2SboxStep],
    rows: &[&neo_fold_clean::frontends::r1cs_f_prime::SelectiveProjectedRowArtifact],
    source_start: usize,
    final_start: usize,
    selector_column: usize,
    swap_direct_groups: bool,
) -> (Vec<String>, Vec<String>) {
    assert_eq!(steps.len(), 86);
    assert_eq!(rows.len(), 86);
    let source_stop = source_start + 600;
    let final_stop = final_start + 86 * 41;
    let mut external_sources = steps
        .iter()
        .flat_map(|step| step.input().terms().iter().chain(step.output().terms()))
        .map(|term| term.column())
        .filter(|column| !(source_start..source_stop).contains(column))
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    assert_eq!(external_sources.len(), 8);
    let mut external_slots = rows
        .iter()
        .flat_map(|row| row.ports())
        .flat_map(|port| port.geometric_runs())
        .map(|run| {
            assert_eq!(run.length(), 41);
            run.column_start()
        })
        .filter(|start| !(final_start..final_stop).contains(start))
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    if swap_direct_groups {
        assert_eq!(external_slots.len(), 8);
        external_sources.rotate_left(4);
        external_slots.rotate_left(4);
    }

    let source_shape = |value: &SelectiveProjectedSourceLinearCombination| {
        let mut terms = value
            .terms()
            .iter()
            .map(|term| {
                let role = if (source_start..source_stop).contains(&term.column()) {
                    (1usize, term.column() - source_start)
                } else {
                    (
                        0,
                        external_sources
                            .iter()
                            .position(|&column| column == term.column())
                            .expect("Poseidon2 source term has a canonical external role"),
                    )
                };
                (role, lean_field(term.coefficient()))
            })
            .collect::<Vec<_>>();
        terms.sort_unstable();
        format!("{}:{terms:?}", lean_field(value.constant()))
    };
    let step_shapes = steps
        .iter()
        .map(|step| format!("{}=>{}", source_shape(step.input()), source_shape(step.output())))
        .collect::<Vec<_>>();

    let row_shapes = rows
        .iter()
        .map(|row| {
            row.ports()
                .iter()
                .map(|port| {
                    assert!(port.seeded_blocks().is_empty());
                    let mut explicit = port
                        .explicit()
                        .iter()
                        .map(|term| {
                            let role = match term.column() {
                                0 => 0usize,
                                column if column == selector_column => 1,
                                column => panic!("unexpected explicit Poseidon2 column {column}"),
                            };
                            (role, lean_field(term.coefficient()))
                        })
                        .collect::<Vec<_>>();
                    explicit.sort_unstable();
                    let mut geometric = port
                        .geometric_runs()
                        .iter()
                        .map(|run| {
                            assert_eq!(run.length(), 41);
                            let role = if (final_start..final_stop).contains(&run.column_start()) {
                                assert_eq!((run.column_start() - final_start) % 41, 0);
                                (1usize, (run.column_start() - final_start) / 41)
                            } else {
                                (
                                    0,
                                    external_slots
                                        .iter()
                                        .position(|&start| start == run.column_start())
                                        .expect("Poseidon2 geometric run has a canonical external role"),
                                )
                            };
                            (role, lean_field(run.initial()), lean_field(run.ratio()))
                        })
                        .collect::<Vec<_>>();
                    geometric.sort_unstable();
                    format!("{explicit:?}:{geometric:?}")
                })
                .collect::<Vec<_>>()
                .join("|")
        })
        .collect::<Vec<_>>();
    (step_shapes, row_shapes)
}

pub(super) fn lean_source_linear_combination(value: &SelectiveProjectedSourceLinearCombination) -> String {
    let terms = value
        .terms()
        .iter()
        .map(|term| {
            format!(
                "{{ column := {}, coefficient := {} }}",
                term.column(),
                lean_field(term.coefficient())
            )
        })
        .collect::<Vec<_>>()
        .join(", ");
    format!(
        "{{ constant := {}, terms := [{}] }}",
        lean_field(value.constant()),
        terms
    )
}

pub(super) fn lean_linear_definition(definition: &SelectiveLinearDefinitionAudit) -> String {
    let terms = definition
        .terms()
        .iter()
        .map(|term| {
            format!(
                "{{ column := {}, coefficient := {} }}",
                term.column(),
                lean_field(term.coefficient())
            )
        })
        .collect::<Vec<_>>()
        .join(", ");
    format!(
        "{{ target := {}, value := {{ constant := {}, terms := [{}] }} }}",
        definition.target(),
        lean_field(definition.constant()),
        terms
    )
}

pub(super) fn lean_compact_port(port: &SelectiveProjectedPort) -> String {
    assert!(
        port.seeded_blocks().is_empty(),
        "retained claim-replay rows must not contain seeded final ports"
    );
    assert!(
        port.explicit()
            .windows(2)
            .all(|terms| terms[0].column() < terms[1].column()),
        "retained explicit terms must have strict canonical column order"
    );

    let explicit = port
        .explicit()
        .iter()
        .map(|term| {
            format!(
                "{{ column := {}, coefficient := {} }}",
                term.column(),
                lean_field(term.coefficient())
            )
        })
        .collect::<Vec<_>>()
        .join(", ");

    let mut geometric_runs = port.geometric_runs().to_vec();
    geometric_runs.sort_by_key(|run| run.column_start());
    let mut next_column = port.explicit().last().map_or(0, |term| term.column() + 1);
    for run in &geometric_runs {
        assert!(run.length() > 0, "retained geometric runs must be nonempty");
        assert!(
            next_column <= run.column_start(),
            "retained compact terms must have strict canonical column order"
        );
        assert!(
            run.initial() != F::ZERO && run.ratio() != F::ZERO,
            "retained geometric runs must expand to nonzero coefficients"
        );
        next_column = run
            .column_start()
            .checked_add(run.length())
            .expect("retained geometric run end must fit usize");
    }
    let geometric = geometric_runs
        .iter()
        .map(|run| {
            format!(
                "{{ columnStart := {}, length := {}, initial := {}, ratio := {} }}",
                run.column_start(),
                run.length(),
                lean_field(run.initial()),
                lean_field(run.ratio())
            )
        })
        .collect::<Vec<_>>()
        .join(", ");
    format!("{{ explicit := [{explicit}], geometric := [{geometric}] }}")
}

pub(super) fn semantic_source_map(
    source: &SelectiveProjectedSourceProvenance,
    semantic_indices: &[usize],
) -> (
    Vec<SelectiveProjectedSourceSlot>,
    Vec<SelectiveProjectedSourceDefinition>,
) {
    let slots_by_column = source
        .retained_slots()
        .iter()
        .map(|slot| (slot.column(), *slot))
        .collect::<BTreeMap<_, _>>();
    let definitions_by_target = source
        .linear_definitions()
        .iter()
        .map(|definition| (definition.target(), definition))
        .collect::<BTreeMap<_, _>>();
    assert_eq!(
        slots_by_column.len(),
        source.retained_slots().len(),
        "retained source slots must have unique source-column ownership"
    );
    assert_eq!(
        definitions_by_target.len(),
        source.linear_definitions().len(),
        "retained source definitions must have unique target ownership"
    );

    let mut needed = BTreeSet::new();
    let mut frontier = Vec::new();
    for &index in semantic_indices {
        let step = source
            .retained_steps()
            .get(index)
            .expect("semantic retained-row index must be in range");
        for value in [step.a(), step.b(), step.c()] {
            for term in value.terms() {
                if needed.insert(term.column()) {
                    frontier.push(term.column());
                }
            }
        }
    }
    while let Some(column) = frontier.pop() {
        let slot = slots_by_column.get(&column);
        let definition = definitions_by_target.get(&column);
        assert!(
            slot.is_some() ^ definition.is_some(),
            "each semantic source column must have exactly one retained slot or definition owner"
        );
        if let Some(definition) = definition {
            for term in definition.terms() {
                if needed.insert(term.column()) {
                    frontier.push(term.column());
                }
            }
        }
    }

    let slots = source
        .retained_slots()
        .iter()
        .copied()
        .filter(|slot| needed.contains(&slot.column()))
        .collect::<Vec<_>>();
    let definitions = source
        .linear_definitions()
        .iter()
        .filter(|definition| needed.contains(&definition.target()))
        .cloned()
        .collect::<Vec<_>>();
    assert_eq!(
        slots.len() + definitions.len(),
        needed.len(),
        "semantic source-map dependency closure must be complete"
    );
    (slots, definitions)
}

pub(super) fn write_source_map(
    rendered: &mut String,
    name: &str,
    slots: &[SelectiveProjectedSourceSlot],
    definitions: &[SelectiveProjectedSourceDefinition],
) {
    writeln!(rendered, "def {name}Slots : List RawSourceSlot :=\n  [").expect("render retained slot header");
    for (index, slot) in slots.iter().enumerate() {
        let separator = if index == 0 { "    " } else { "  , " };
        writeln!(
            rendered,
            "{separator}{{ column := {}, start := {}, width := {} }}",
            slot.column(),
            slot.start(),
            slot.width()
        )
        .expect("render retained slot");
    }
    writeln!(rendered, "  ]\n").expect("render retained slot footer");

    writeln!(rendered, "def {name}Definitions : List RawSourceDefinition :=\n  [")
        .expect("render retained definition header");
    for (index, definition) in definitions.iter().enumerate() {
        let separator = if index == 0 { "    " } else { "  , " };
        let value = format!(
            "{{ constant := {}, terms := [{}] }}",
            lean_field(definition.constant()),
            definition
                .terms()
                .iter()
                .map(|term| format!(
                    "{{ column := {}, coefficient := {} }}",
                    term.column(),
                    lean_field(term.coefficient())
                ))
                .collect::<Vec<_>>()
                .join(", ")
        );
        writeln!(
            rendered,
            "{separator}{{ target := {}, value := {} }}",
            definition.target(),
            value
        )
        .expect("render retained definition");
    }
    writeln!(rendered, "  ]\n").expect("render retained definition footer");
}

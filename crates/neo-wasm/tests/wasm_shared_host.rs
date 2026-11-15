mod common;

use neo_application::{ColumnWidth, GadgetDescriptor};
use neo_math::F;
use neo_wasm::layout::*;
use neo_wasm::nebula::{step_inputs, WasmApplication, WasmSize};
use p3_field::PrimeCharacteristicRing;
use std::num::NonZeroUsize;

fn app(rows: usize) -> WasmApplication {
    let rows = NonZeroUsize::new(rows).unwrap();
    let mut geometry = WasmSize::Small.geometry();
    geometry.b_ops = neo_wasm::nebula::plan_ordered_slots(rows).unwrap().len();
    WasmApplication::new(&geometry, WasmSize::Small.limits().regions().unwrap().0, rows).unwrap()
}

#[test]
fn grouped_host_rows_hold_at_boundaries_and_padded_tails() {
    let setup = common::host_event_fixture::host_event_lifecycle_setup();
    for count in [1, 2, 3, 7, 8, 9, 16] {
        let rows = NonZeroUsize::new(count).unwrap();
        let inputs = step_inputs(&setup.trace, setup.trace.len().div_ceil(count) + 1, rows).unwrap();
        let application = app(count);
        for (index, input) in inputs.iter().enumerate() {
            application
                .check(&input.z)
                .unwrap_or_else(|error| panic!("rows={count}, step={index}: {error}"));
            assert_eq!(
                application.output_words(&input.z),
                neo_wasm::nebula::state_words(input.next)
            );
        }
    }
}

#[test]
fn shared_host_input_output_and_permutation_columns_are_checked() {
    let setup = common::host_event_fixture::host_event_lifecycle_setup();
    let count = NonZeroUsize::new(8).unwrap();
    let inputs = step_inputs(&setup.trace, setup.trace.len().div_ceil(8), count).unwrap();
    let application = app(8);
    let source = neo_wasm::operations::r1cs::operation_circuit()
        .unwrap()
        .relation();
    let width = source.r1cs().column_count();
    let gadget = source
        .r1cs()
        .catalog()
        .gadget_occurrences()
        .iter()
        .find_map(|gadget| {
            if let GadgetDescriptor::EventCommitment {
                output,
                auxiliary_start,
                auxiliary_len,
                ..
            } = gadget.descriptor()
            {
                Some((*output, *auxiliary_start..*auxiliary_start + *auxiliary_len))
            } else {
                None
            }
        })
        .unwrap();
    let (output, auxiliary) = gadget;
    for column in (width..width + 12).chain(output).chain(auxiliary) {
        let mut forged = inputs[0].z.clone();
        let location = application.witness_column(0, column).unwrap();
        assert_eq!(Some(location), application.witness_column(7, column));
        forged[location] += F::ONE;
        assert!(application.check(&forged).is_err(), "changed shared column {column}");
    }
}

#[test]
fn checked_cursor_rows_match_the_lean_model() {
    let source = neo_wasm::operations::r1cs::operation_circuit()
        .unwrap()
        .relation();
    let family = |name: &str| {
        source
            .columns()
            .families()
            .iter()
            .find(|family| family.name == name)
            .unwrap()
    };
    let positions = family("GATHER_WORD_POSITION");
    let kinds = family("GATHER_KIND");
    let absorb = family("ABSORB").start;
    let advice = family("G_ADVICE").start;
    let enter = kinds.start + neo_wasm::WasmHostEventSlotKind::EnterOpaque.index();
    assert_eq!(positions.len, 8);
    for family in [
        positions,
        kinds,
        family("ABSORB"),
        family("G_ADVICE"),
        family("COL_GATHER_ACTIVE"),
    ] {
        assert_eq!(family.width, ColumnWidth::Boolean);
    }
    let last = positions.start + 7;
    let before = COL_HOST_EVENT_SLOT_CURSOR_BEFORE;
    let after = COL_HOST_EVENT_SLOT_CURSOR_AFTER;
    let one = vec![(COL_ONE, F::ONE)];
    let expected = [
        [
            (0..8)
                .map(|i| (positions.start + i, F::ONE))
                .chain([(COL_GATHER_ACTIVE, -F::ONE)])
                .collect(),
            one.clone(),
            vec![],
        ],
        [
            (0..kinds.len)
                .map(|i| (kinds.start + i, F::ONE))
                .chain([(COL_GATHER_ACTIVE, -F::ONE)])
                .collect(),
            one,
            vec![],
        ],
        [
            vec![(COL_GATHER_ACTIVE, F::ONE)],
            (0..8)
                .map(|i| (positions.start + i, F::from_usize(i)))
                .chain([(before, -F::ONE)])
                .collect(),
            vec![],
        ],
        [
            vec![(enter, F::ONE)],
            vec![(before, F::ONE), (COL_ONE, F::ONE)],
            vec![
                (before, F::ONE),
                (COL_GATHER_ACTIVE, F::ONE),
                (last, -F::from_u64(8)),
                (after, -F::ONE),
            ],
        ],
        [
            vec![(last, F::ONE)],
            vec![(COL_ONE, F::ONE), (advice, -F::ONE)],
            vec![(absorb, F::ONE)],
        ],
        [vec![(absorb, F::ONE)], vec![(enter, F::ONE)], vec![]],
    ];
    let normalize = |mut terms: Vec<(usize, F)>| {
        terms.retain(|(_, c)| *c != F::ZERO);
        terms.sort_by_key(|(column, _)| *column);
        terms
    };
    let actual = source
        .r1cs()
        .catalog()
        .rows()
        .iter()
        .map(|row| {
            let row = row.row();
            [
                normalize(row.a_terms().to_vec()),
                normalize(row.b_terms().to_vec()),
                normalize(row.c_terms().to_vec()),
            ]
        })
        .collect::<Vec<_>>();
    for row in expected {
        assert!(actual.contains(&row.map(normalize)), "missing checked cursor row");
    }
    for column in (positions.start..positions.end())
        .chain(kinds.start..kinds.end())
        .chain([COL_GATHER_ACTIVE, absorb, advice])
    {
        let boolean = [
            vec![(column, F::ONE)],
            vec![(column, F::ONE), (COL_ONE, -F::ONE)],
            vec![],
        ];
        assert!(
            actual.contains(&boolean.map(normalize)),
            "missing Boolean row for {column}"
        );
    }

    assert!(neo_wasm::build_wasm_relation_layout()
        .auxiliary
        .continuity
        .links()
        .any(|link| { link.previous_step_column == after && link.next_step_column == before }));
}

#[test]
fn original_host_block_has_only_the_four_gated_output_readers() {
    let source = neo_wasm::operations::r1cs::operation_circuit()
        .unwrap()
        .relation();
    let gadget = source
        .r1cs()
        .catalog()
        .gadget_occurrences()
        .iter()
        .find(|gadget| matches!(gadget.descriptor(), GadgetDescriptor::EventCommitment { .. }))
        .unwrap();
    let GadgetDescriptor::EventCommitment {
        output,
        auxiliary_start,
        auxiliary_len,
        ..
    } = gadget.descriptor()
    else {
        unreachable!()
    };
    let private = |column: usize| {
        output.contains(&column) || (*auxiliary_start..*auxiliary_start + *auxiliary_len).contains(&column)
    };
    let block = &source.r1cs().catalog().rows()[gadget.row_range().clone()];
    let entries = block
        .iter()
        .map(|row| {
            let row = row.row();
            row.a_terms().len() + row.b_terms().len() + row.c_terms().len()
        })
        .sum::<usize>();
    let nonlinear = block
        .iter()
        .filter(|row| {
            let variable = |terms: &[(usize, F)]| {
                terms
                    .iter()
                    .any(|(column, coefficient)| *column != COL_ONE && *coefficient != F::ZERO)
            };
            variable(row.row().a_terms()) && variable(row.row().b_terms())
        })
        .count();
    let readers = source
        .r1cs()
        .catalog()
        .rows()
        .iter()
        .enumerate()
        .filter(|(index, row)| {
            !gadget.row_range().contains(index)
                && row
                    .row()
                    .a_terms()
                    .iter()
                    .chain(row.row().b_terms())
                    .chain(row.row().c_terms())
                    .any(|(column, _)| private(*column))
        })
        .collect::<Vec<_>>();
    assert_eq!(readers.len(), 4);
    let absorb = source
        .columns()
        .families()
        .iter()
        .find(|family| family.name == "ABSORB")
        .unwrap()
        .start;
    for (index, (_, row)) in readers.iter().enumerate() {
        assert_eq!(row.row().a_terms(), [(absorb, F::ONE)]);
        assert_eq!(
            row.row().b_terms(),
            [(COL_COMM_CHAIN_AFTER[index], F::ONE), (output[index], -F::ONE)]
        );
        assert!(row.row().c_terms().is_empty());
    }
    assert_eq!(block.len(), 716);
    assert_eq!(nonlinear, 472);
    eprintln!(
        "shared_host_block_source_rows\t{}\tnonlinear\t{nonlinear}\tentries\t{entries}\tshared_columns\t728",
        block.len()
    );
}

#[test]
fn a_consistent_rehashed_block_still_must_match_each_absorb_input_lane() {
    use neo_application::event_commitment::{commit_block, EventCommitment};
    use nightstream::nebula::R1csStepError;
    let setup = common::host_event_fixture::host_event_lifecycle_setup();
    let source = neo_wasm::operations::r1cs::operation_circuit()
        .unwrap()
        .relation();
    let width = source.r1cs().column_count();
    let flag = source
        .columns()
        .families()
        .iter()
        .find(|family| family.name == "ABSORB")
        .unwrap()
        .start;
    let commitment = source
        .r1cs()
        .catalog()
        .gadget_occurrences()
        .iter()
        .find_map(|gadget| {
            if let GadgetDescriptor::EventCommitment {
                previous,
                block,
                output,
                auxiliary_start,
                ..
            } = gadget.descriptor()
            {
                Some(EventCommitment {
                    previous: *previous,
                    block: *block,
                    output: *output,
                    auxiliary_start: *auxiliary_start,
                })
            } else {
                None
            }
        })
        .unwrap();
    let own = step_inputs(&setup.trace, setup.trace.len(), NonZeroUsize::MIN).unwrap();
    let absorbing = own.iter().position(|row| row.z[flag] == F::ONE).unwrap();
    assert!(absorbing >= 7);
    let input = step_inputs(
        &setup.trace[absorbing - 7..=absorbing],
        1,
        NonZeroUsize::new(8).unwrap(),
    )
    .unwrap()
    .remove(0);
    let application = app(8);
    application.check(&input.z).unwrap();
    let links = neo_wasm::build_wasm_relation_layout()
        .auxiliary
        .continuity
        .links()
        .count();
    let gates = application.relation_rows() - 7 * links - 12;
    let inputs = commitment
        .previous
        .into_iter()
        .chain(commitment.block)
        .collect::<Vec<_>>();
    for lane in 0..12 {
        let mut fake = own[absorbing].z.clone();
        fake[inputs[lane]] += F::ONE;
        commitment.assign_auxiliaries(&mut fake);
        let output = commit_block(
            commitment.previous.map(|column| fake[column]),
            commitment.block.map(|column| fake[column]),
        );
        for (column, value) in commitment.output.into_iter().zip(output) {
            fake[column] = value;
        }
        let mut forged = input.z.clone();
        for (lane, column) in inputs.iter().enumerate() {
            forged[application.witness_column(0, width + lane).unwrap()] = fake[*column];
        }
        for column in commitment.auxiliary_range().chain(commitment.output) {
            forged[application.witness_column(0, column).unwrap()] = fake[column];
        }
        for (column, value) in COL_COMM_CHAIN_AFTER.into_iter().zip(output) {
            forged[application.witness_column(7, column).unwrap()] = value;
        }
        assert!(
            matches!(application.check(&forged), Err(R1csStepError::Relation { row }) if row == gates + lane),
            "rehashed lane {lane} must fail its own input binding"
        );
    }
}

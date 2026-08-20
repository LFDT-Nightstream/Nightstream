//! Compact exact geometry for the full terminal Nebula finalizer family.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use neo_ccs::{LaneCommitments, SeededPhi81LinearBlock};
use neo_fold_clean::engine::r1cs_circuit::builder::{
    Poseidon2HashAudit, Poseidon2PermutationAudit, BALANCED_TERNARY_DIGITS,
};
use neo_fold_clean::engine::r1cs_circuit::field_ext::KVar;
use neo_fold_clean::engine::r1cs_circuit::{Lc, R1csBuilder, R1csSnapshot, Var};
use neo_fold_clean::frontends::r1cs_f_prime::terminal_r1cs::STREAMING_TERMINAL_R1CS_FAMILY_NAMES;
use neo_fold_clean::paper::construction2::nebula_lane::NEBULA_GAMMA_TRANSCRIPT_LABEL;
use neo_fold_clean::paper::f_prime::nebula_lane_circuit::{
    decode_delayed_nebula_public_suffix_circuit, enforce_nebula_advance_circuit,
    enforce_nebula_lane_leaf_digests_circuit, enforce_nebula_maybe_close_circuit, enforce_nebula_maybe_open_circuit,
    NebulaLaneWires, NebulaOpenContextWires,
};
use neo_fold_clean::paper::relations::product_commitment_circuit::{AdvCommitmentDataWires, CommitmentDataWires};
use neo_math::F;
use neo_transcript::{Poseidon2Transcript, Transcript as _};
use p3_field::{PrimeCharacteristicRing, PrimeField64};

use super::streaming_terminal_fixture::StreamingTerminalAuditFixture;
use super::{expected_linear_row, relocated_terms};

const ARTIFACT_PATH: &str = "../../formal/nightstream-lean/Nightstream/Implementation/\
R1CS/Artifacts/FPrimeFullHistory/Generated/FPrimeFullHistoryStreamingTerminalFullFinalizer.lean";

pub(super) fn artifact_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(ARTIFACT_PATH)
}

fn lane_from_fields(fields: [Var; 50]) -> NebulaLaneWires {
    let k = |start: usize| KVar::new(fields[start], fields[start + 1]);
    NebulaLaneWires {
        program_binding_digest: fields[0..4].try_into().expect("four binding fields"),
        open: fields[4],
        seg_idx: fields[5],
        idx: fields[6],
        ts: fields[7],
        gamma: [k(8), k(10)],
        h: [k(12), k(14), k(16), k(18)],
        sp: [fields[20], fields[21]],
        d_pre: [
            fields[22..26].try_into().expect("four ops pre fields"),
            fields[26..30].try_into().expect("four IS pre fields"),
            fields[30..34].try_into().expect("four FS pre fields"),
        ],
        d_seen: [
            fields[34..38].try_into().expect("four ops seen fields"),
            fields[38..42].try_into().expect("four IS seen fields"),
            fields[42..46].try_into().expect("four FS seen fields"),
        ],
        d_mem: fields[46..50].try_into().expect("four memory fields"),
    }
}

fn lane_columns(lane: &NebulaLaneWires) -> [usize; 50] {
    let mut fields = Vec::with_capacity(50);
    fields.extend(lane.program_binding_digest.map(Var::col));
    fields.extend([lane.open, lane.seg_idx, lane.idx, lane.ts].map(Var::col));
    for value in lane.gamma.iter().chain(&lane.h) {
        fields.extend([value.c0.col(), value.c1.col()]);
    }
    fields.extend(lane.sp.map(Var::col));
    fields.extend(lane.d_pre.iter().flatten().map(|wire| wire.col()));
    fields.extend(lane.d_seen.iter().flatten().map(|wire| wire.col()));
    fields.extend(lane.d_mem.map(Var::col));
    fields.try_into().expect("50 Nebula lane fields")
}

fn alloc_from_columns(builder: &mut R1csBuilder, source: &R1csSnapshot, columns: &[usize]) -> Vec<Var> {
    columns
        .iter()
        .map(|&column| builder.alloc(source.witness()[column]))
        .collect()
}

struct Reference {
    source: R1csSnapshot,
    external: BTreeMap<usize, usize>,
    internal_start: usize,
    decode_column_end: usize,
    decode_end: usize,
    open_end: usize,
    leaves_end: usize,
    advance_end: usize,
    close_end: usize,
    opened_lane_columns: [usize; 50],
    advanced_lane_columns: [usize; 50],
    final_lane_columns: [usize; 50],
    closed_column: usize,
    advance_chain_links: [ChainLinkReference; 3],
    poseidon2_hash_audits: Vec<Poseidon2HashAudit>,
    poseidon2_permutation_audits: Vec<Poseidon2PermutationAudit>,
    leaf_sis: [LeafSisReference; 3],
    gamma_mux_selector_column: usize,
    gamma_mux_opened_d_pre_columns: [usize; 12],
    gamma_mux_carried_columns: [usize; 16],
    gamma_mux_output_columns: [usize; 16],
}

struct LeafSeededBlockReference {
    block: SeededPhi81LinearBlock,
    source_columns: Vec<usize>,
    output_columns: Vec<usize>,
    metadata_pin_row_start: usize,
    metadata_columns: [usize; 2],
    metadata_values: [u64; 2],
    opening_row_start: usize,
}

struct LeafSisReference {
    prefix_pin_row_start: usize,
    prefix_constant_columns: Vec<usize>,
    prefix_constant_values: Vec<u64>,
    primary: LeafSeededBlockReference,
    compression: LeafSeededBlockReference,
    envelope_constant_columns: Vec<usize>,
    envelope_constant_values: Vec<u64>,
    digest: Poseidon2HashAudit,
}

struct ChainLinkReference {
    constant_row_start: usize,
    constant_columns: Vec<usize>,
    constant_values: Vec<u64>,
    previous_columns: [usize; 4],
    leaf_columns: [usize; 4],
    digest: Poseidon2HashAudit,
}

const CANONICAL_OPENING_ROWS: usize = 3 * BALANCED_TERNARY_DIGITS + 1;
const CANONICAL_OPENING_COLUMNS: usize = 3 * BALANCED_TERNARY_DIGITS - 1;

fn recover_leaf_block(
    source: &R1csSnapshot,
    block: SeededPhi81LinearBlock,
    openings: &[neo_fold_clean::engine::r1cs_circuit::BalancedTernaryOpeningTraceEntry],
    expected_sources: &[usize],
    phase_row_start: usize,
) -> LeafSeededBlockReference {
    assert_eq!(block.word_starts().len(), expected_sources.len());
    let metadata = [
        constant_pin(source, phase_row_start),
        constant_pin(source, phase_row_start + 1),
    ];
    let metadata_columns = metadata.map(|pin| pin.0);
    let metadata_values = metadata.map(|pin| pin.1);
    assert_eq!(metadata_columns[1], metadata_columns[0] + 1);
    assert_eq!(metadata_values, [54, block.kappa() as u64]);
    let opening_row_start = phase_row_start + metadata.len();
    let mut row = opening_row_start;
    let mut source_columns = Vec::with_capacity(expected_sources.len());
    for (&word_start, &expected_source) in block.word_starts().iter().zip(expected_sources) {
        let opening = openings
            .iter()
            .find(|opening| opening.digit_cols[0] == word_start)
            .unwrap_or_else(|| panic!("seeded word at column {word_start} has no canonical opening"));
        assert_eq!(opening.field_col, expected_source);
        assert_eq!(opening.digit_cols, std::array::from_fn(|index| word_start + index));
        assert_eq!(
            opening.negative_cols,
            std::array::from_fn(|index| word_start + BALANCED_TERNARY_DIGITS + index)
        );
        assert_eq!(
            opening.borrow_cols,
            std::array::from_fn(|index| word_start + 2 * BALANCED_TERNARY_DIGITS + index)
        );
        assert_eq!(opening.digit_rows, row..row + 2 * BALANCED_TERNARY_DIGITS);
        assert_eq!(opening.reconstruction_row, row + 2 * BALANCED_TERNARY_DIGITS);
        assert_eq!(
            opening.transition_rows,
            row + 2 * BALANCED_TERNARY_DIGITS + 1..row + CANONICAL_OPENING_ROWS
        );
        source_columns.push(opening.field_col);
        row += CANONICAL_OPENING_ROWS;
    }
    assert_eq!(block.row_start(), row);
    assert_eq!(
        block.word_starts(),
        (0..expected_sources.len())
            .map(|index| block.word_starts()[0] + CANONICAL_OPENING_COLUMNS * index)
            .collect::<Vec<_>>()
    );

    let output_columns = (block.row_start()..block.row_end())
        .map(|row| {
            assert_eq!(source.b_row(row), &[(0, F::ONE)]);
            let [(column, coefficient)] = source.c_row(row) else {
                panic!("seeded Phi81 row {row} does not have one exact output")
            };
            assert_eq!(*coefficient, F::ONE);
            *column
        })
        .collect::<Vec<_>>();
    assert_eq!(
        output_columns,
        (output_columns[0]..output_columns[0] + output_columns.len()).collect::<Vec<_>>()
    );
    LeafSeededBlockReference {
        block,
        source_columns,
        output_columns,
        metadata_pin_row_start: phase_row_start,
        metadata_columns,
        metadata_values,
        opening_row_start,
    }
}

fn recover_leaf_sis(
    source: &R1csSnapshot,
    openings: &[neo_fold_clean::engine::r1cs_circuit::BalancedTernaryOpeningTraceEntry],
    primary: SeededPhi81LinearBlock,
    compression: SeededPhi81LinearBlock,
    digest: Poseidon2HashAudit,
    expected_data_columns: &[usize],
    expected_digest_columns: [usize; 4],
    leaf_row_start: usize,
) -> LeafSisReference {
    let prefix_count = primary
        .word_starts()
        .len()
        .checked_sub(expected_data_columns.len())
        .expect("Nebula leaf primary input contains its commitment data");
    let primary_sources = primary
        .word_starts()
        .iter()
        .map(|&word_start| {
            openings
                .iter()
                .find(|opening| opening.digit_cols[0] == word_start)
                .unwrap_or_else(|| panic!("primary word at column {word_start} has no opening"))
                .field_col
        })
        .collect::<Vec<_>>();
    assert_eq!(&primary_sources[prefix_count..], expected_data_columns);
    let prefix_constant_columns = primary_sources[..prefix_count].to_vec();
    assert_eq!(
        prefix_constant_columns,
        (prefix_constant_columns[0]..prefix_constant_columns[0] + prefix_count).collect::<Vec<_>>()
    );
    let prefix_constant_values = (0..prefix_count)
        .map(|index| constant_pin(source, leaf_row_start + index).1)
        .collect::<Vec<_>>();
    assert_eq!(
        (0..prefix_count)
            .map(|index| constant_pin(source, leaf_row_start + index).0)
            .collect::<Vec<_>>(),
        prefix_constant_columns
    );

    let primary_phase_start = leaf_row_start + prefix_count;
    let primary = recover_leaf_block(source, primary, openings, &primary_sources, primary_phase_start);
    let compression_phase_start = primary.block.row_end();
    let compression_sources = primary.output_columns.clone();
    let compression = recover_leaf_block(
        source,
        compression,
        openings,
        &compression_sources,
        compression_phase_start,
    );

    let envelope_constant_count = digest
        .input_cols
        .len()
        .checked_sub(compression.output_columns.len())
        .expect("SIS envelope contains the compression output");
    assert_eq!(
        &digest.input_cols[envelope_constant_count..],
        compression.output_columns
    );
    assert_eq!(digest.row_start, compression.block.row_end() + envelope_constant_count);
    let envelope_constant_columns = digest.input_cols[..envelope_constant_count].to_vec();
    assert_eq!(
        envelope_constant_columns,
        (envelope_constant_columns[0]..envelope_constant_columns[0] + envelope_constant_count).collect::<Vec<_>>()
    );
    let envelope_constant_values = (0..envelope_constant_count)
        .map(|index| constant_pin(source, compression.block.row_end() + index).1)
        .collect::<Vec<_>>();
    assert_eq!(
        (0..envelope_constant_count)
            .map(|index| constant_pin(source, compression.block.row_end() + index).0)
            .collect::<Vec<_>>(),
        envelope_constant_columns
    );
    assert_eq!(digest.output_cols, expected_digest_columns);

    LeafSisReference {
        prefix_pin_row_start: leaf_row_start,
        prefix_constant_columns,
        prefix_constant_values,
        primary,
        compression,
        envelope_constant_columns,
        envelope_constant_values,
        digest,
    }
}

fn recover_chain_link(
    source: &R1csSnapshot,
    digest: Poseidon2HashAudit,
    previous_columns: [usize; 4],
    leaf_columns: [usize; 4],
    output_columns: [usize; 4],
) -> ChainLinkReference {
    let suffix = previous_columns
        .into_iter()
        .chain(leaf_columns)
        .collect::<Vec<_>>();
    let constant_count = digest
        .input_cols
        .len()
        .checked_sub(suffix.len())
        .expect("Nebula chain link contains its prior and leaf digests");
    assert_eq!(&digest.input_cols[constant_count..], suffix);
    assert_eq!(digest.output_cols, output_columns);
    let constant_columns = digest.input_cols[..constant_count].to_vec();
    assert!(!constant_columns.is_empty());
    assert_eq!(
        constant_columns,
        (constant_columns[0]..constant_columns[0] + constant_count).collect::<Vec<_>>()
    );
    let constant_row_start = digest
        .row_start
        .checked_sub(constant_count)
        .expect("chain-link constants precede the Poseidon2 trace");
    let constants = (0..constant_count)
        .map(|index| constant_pin(source, constant_row_start + index))
        .collect::<Vec<_>>();
    assert_eq!(constants.iter().map(|pin| pin.0).collect::<Vec<_>>(), constant_columns);
    let constant_values = constants.into_iter().map(|pin| pin.1).collect();
    ChainLinkReference {
        constant_row_start,
        constant_columns,
        constant_values,
        previous_columns,
        leaf_columns,
        digest,
    }
}

fn reference_relation(fixture: &StreamingTerminalAuditFixture, full_source: &R1csSnapshot) -> Reference {
    let full_lane = (fixture.source_binding_decoded_column_start + 32
        ..fixture.source_binding_decoded_column_start + 82)
        .collect::<Vec<_>>();
    let mut builder = R1csBuilder::new();
    let lane_fields: [Var; 50] = alloc_from_columns(&mut builder, full_source, &full_lane)
        .try_into()
        .expect("50 lane inputs");
    let lane = lane_from_fields(lane_fields);
    let payload = alloc_from_columns(&mut builder, full_source, &fixture.delayed_payload_columns);
    let ops_data = alloc_from_columns(&mut builder, full_source, &fixture.fresh_adv_data_columns.ops);
    let is_data = alloc_from_columns(&mut builder, full_source, &fixture.fresh_adv_data_columns.is);
    let fs_data = alloc_from_columns(&mut builder, full_source, &fixture.fresh_adv_data_columns.fs);
    let adv: AdvCommitmentDataWires = LaneCommitments {
        ops: CommitmentDataWires {
            d: fixture.fresh_adv_d,
            kappa: fixture.fresh_adv_kappa,
            data: ops_data,
        },
        is: CommitmentDataWires {
            d: fixture.fresh_adv_d,
            kappa: fixture.fresh_adv_kappa,
            data: is_data,
        },
        fs: CommitmentDataWires {
            d: fixture.fresh_adv_d,
            kappa: fixture.fresh_adv_kappa,
            data: fs_data,
        },
    };
    let vk_fs: [Var; 4] = alloc_from_columns(&mut builder, full_source, &fixture.vk_fs_columns)
        .try_into()
        .expect("four verifier-key fields");
    let boundary: [Var; 4] = alloc_from_columns(&mut builder, full_source, &fixture.boundary_columns)
        .try_into()
        .expect("four boundary fields");
    let accumulator: [Var; 4] = alloc_from_columns(&mut builder, full_source, &fixture.accumulator_columns)
        .try_into()
        .expect("four accumulator fields");
    let context = NebulaOpenContextWires {
        vk_fs,
        z_i: boundary,
        acc_digest: accumulator,
    };

    let internal_start = builder.cols();
    let delayed = decode_delayed_nebula_public_suffix_circuit(&mut builder, &payload, fixture.stacks)
        .expect("decode reference delayed payload");
    let decode_end = builder.rows();
    let decode_column_end = builder.cols();
    let gamma_mux_selector_column = delayed.open.col();
    let gamma_mux_opened_d_pre_columns: [usize; 12] = delayed
        .d_pre
        .iter()
        .flatten()
        .map(|wire| wire.col())
        .collect::<Vec<_>>()
        .try_into()
        .expect("twelve opened d_pre fields");
    let gamma_mux_carried_columns: [usize; 16] = lane
        .gamma
        .iter()
        .flat_map(|value| [value.c0.col(), value.c1.col()])
        .chain(lane.d_pre.iter().flatten().map(|wire| wire.col()))
        .collect::<Vec<_>>()
        .try_into()
        .expect("sixteen carried gamma and d_pre fields");
    let opened = enforce_nebula_maybe_open_circuit(&mut builder, &lane, &delayed, &context, fixture.seg_max);
    let opened_lane_columns = lane_columns(&opened);
    let gamma_mux_output_columns: [usize; 16] = opened
        .gamma
        .iter()
        .flat_map(|value| [value.c0.col(), value.c1.col()])
        .chain(opened.d_pre.iter().flatten().map(|wire| wire.col()))
        .collect::<Vec<_>>()
        .try_into()
        .expect("sixteen opened gamma and d_pre outputs");
    let open_end = builder.rows();
    builder.enable_encoding_trace();
    let leaves = enforce_nebula_lane_leaf_digests_circuit(
        &mut builder,
        adv.ops.d,
        adv.ops.kappa,
        &adv.ops.data,
        &adv.is.data,
        &adv.fs.data,
    );
    let leaves_end = builder.rows();
    let leaf_blocks: [SeededPhi81LinearBlock; 6] = builder
        .seeded_phi81_a_blocks()
        .to_vec()
        .try_into()
        .expect("three ordered primary/compression SIS block pairs");
    let leaf_openings = builder
        .encoding_trace()
        .balanced_ternary_openings()
        .to_vec();
    let leaf_digest_columns = leaves.map(|digest| digest.map(Var::col));
    let advanced = enforce_nebula_advance_circuit(&mut builder, &opened, &delayed.step, leaves);
    let advanced_lane_columns = lane_columns(&advanced);
    let advance_end = builder.rows();
    let transition = enforce_nebula_maybe_close_circuit(&mut builder, &advanced, fixture.steps_per_segment);
    let close_end = builder.rows();
    builder.enforce_eq(&Lc::from_var(transition.closed), &Lc::from_const(F::ONE));

    let mut external = BTreeMap::from([(0, 0)]);
    external.extend(lane_columns(&lane).into_iter().zip(full_lane));
    external.extend(
        payload
            .iter()
            .map(|wire| wire.col())
            .zip(&fixture.delayed_payload_columns)
            .map(|(reference, &full)| (reference, full)),
    );
    for (reference, full) in adv
        .ops
        .data
        .iter()
        .chain(&adv.is.data)
        .chain(&adv.fs.data)
        .map(|wire| wire.col())
        .zip(
            fixture
                .fresh_adv_data_columns
                .ops
                .iter()
                .chain(&fixture.fresh_adv_data_columns.is)
                .chain(&fixture.fresh_adv_data_columns.fs),
        )
    {
        external.insert(reference, *full);
    }
    external.extend(vk_fs.map(Var::col).into_iter().zip(fixture.vk_fs_columns));
    external.extend(
        boundary
            .map(Var::col)
            .into_iter()
            .zip(fixture.boundary_columns),
    );
    external.extend(
        accumulator
            .map(Var::col)
            .into_iter()
            .zip(fixture.accumulator_columns),
    );

    let final_lane_columns = lane_columns(&transition.lane);
    let closed_column = transition.closed.col();
    let poseidon2_hash_audits = builder.poseidon2_hash_audits();
    let poseidon2_permutation_audits = builder.poseidon2_permutation_audits();
    let source = builder.snapshot();
    drop(builder);
    assert!(source.is_satisfied(source.witness()));
    let leaf_hash_audits: [Poseidon2HashAudit; 3] = poseidon2_hash_audits
        .iter()
        .filter(|audit| audit.row_start >= open_end && audit.row_end <= leaves_end)
        .cloned()
        .collect::<Vec<_>>()
        .try_into()
        .expect("three ordered Nebula leaf Poseidon2 hashes");
    let expected_leaf_data = [&adv.ops.data, &adv.is.data, &adv.fs.data];
    let mut next_leaf_row = open_end;
    let leaf_sis: [LeafSisReference; 3] = std::array::from_fn(|index| {
        let primary = leaf_blocks[2 * index].clone();
        let compression = leaf_blocks[2 * index + 1].clone();
        let digest = leaf_hash_audits[index].clone();
        let leaf = recover_leaf_sis(
            &source,
            &leaf_openings,
            primary,
            compression,
            digest,
            &expected_leaf_data[index]
                .iter()
                .map(|wire| wire.col())
                .collect::<Vec<_>>(),
            leaf_digest_columns[index],
            next_leaf_row,
        );
        next_leaf_row = leaf.digest.row_end;
        leaf
    });
    assert_eq!(next_leaf_row, leaves_end);
    let advance_hashes: [Poseidon2HashAudit; 3] = poseidon2_hash_audits
        .iter()
        .filter(|audit| audit.row_start >= leaves_end && audit.row_end <= advance_end)
        .cloned()
        .collect::<Vec<_>>()
        .try_into()
        .expect("three ordered Nebula D_seen chain hashes");
    let advance_chain_links = std::array::from_fn(|index| {
        recover_chain_link(
            &source,
            advance_hashes[index].clone(),
            lane.d_seen[index].map(Var::col),
            leaf_digest_columns[index],
            advanced.d_seen[index].map(Var::col),
        )
    });
    Reference {
        source,
        external,
        internal_start,
        decode_column_end,
        decode_end,
        open_end,
        leaves_end,
        advance_end,
        close_end,
        opened_lane_columns,
        advanced_lane_columns,
        final_lane_columns,
        closed_column,
        advance_chain_links,
        poseidon2_hash_audits,
        poseidon2_permutation_audits,
        leaf_sis,
        gamma_mux_selector_column,
        gamma_mux_opened_d_pre_columns,
        gamma_mux_carried_columns,
        gamma_mux_output_columns,
    }
}

fn constant_pin(source: &R1csSnapshot, row: usize) -> (usize, u64) {
    assert_eq!(source.b_row(row), &[(0, F::ONE)]);
    assert!(source.c_row(row).is_empty());
    let columns = source
        .a_row(row)
        .iter()
        .filter_map(|&(column, coefficient)| (column != 0).then_some((column, coefficient)))
        .collect::<Vec<_>>();
    let [(column, coefficient)] = columns.as_slice() else {
        panic!("gamma transcript row {row} is not one constant binding")
    };
    assert_eq!(*coefficient, F::ONE);
    let value = source.witness()[*column].as_canonical_u64();
    assert_eq!(
        source.a_row(row),
        expected_linear_row(*column, &[(0, F::from_u64(value))])
    );
    (*column, value)
}

#[derive(Clone, Copy)]
enum GammaPiece {
    Pin(usize),
    Call(usize),
}

struct GammaTranscript {
    row_start: usize,
    row_stop: usize,
    pin_rows: Vec<usize>,
    pins: Vec<(usize, u64)>,
    calls: Vec<Poseidon2PermutationAudit>,
    initial_absorbed: usize,
    gamma1_columns: [usize; 2],
    gamma2_columns: [usize; 2],
}

fn gamma_transcript(reference: &Reference, staged_digest_stop: usize) -> GammaTranscript {
    const MUX_ROWS: usize = 16;
    const EXPECTED_PINS: usize = 84;
    const EXPECTED_CALLS: usize = 29;
    let row_start = staged_digest_stop;
    let row_stop = reference
        .open_end
        .checked_sub(MUX_ROWS)
        .expect("open phase contains the final output muxes");
    let calls = reference
        .poseidon2_permutation_audits
        .iter()
        .copied()
        .filter(|call| call.row_start >= row_start && call.row_end <= row_stop)
        .collect::<Vec<_>>();
    assert_eq!(calls.len(), EXPECTED_CALLS);
    for call in &calls {
        assert_eq!(call.row_end - call.row_start, 600);
        assert_eq!(call.allocated_col_count, 600);
        assert_eq!(
            call.output_cols,
            std::array::from_fn(|lane| call.first_allocated_col + 592 + lane)
        );
    }

    let mut pieces = Vec::with_capacity(EXPECTED_PINS + EXPECTED_CALLS);
    let mut pins = Vec::with_capacity(EXPECTED_PINS);
    let mut pin_rows = Vec::with_capacity(EXPECTED_PINS);
    let mut row = row_start;
    let mut next_call = 0;
    while row < row_stop {
        if next_call < calls.len() && calls[next_call].row_start == row {
            pieces.push(GammaPiece::Call(next_call));
            row = calls[next_call].row_end;
            next_call += 1;
        } else {
            let pin = constant_pin(&reference.source, row);
            pieces.push(GammaPiece::Pin(pins.len()));
            pin_rows.push(row);
            pins.push(pin);
            row += 1;
        }
    }
    assert_eq!(row, row_stop);
    assert_eq!(next_call, calls.len());
    assert_eq!(pins.len(), EXPECTED_PINS);
    assert_eq!(row_stop - row_start, EXPECTED_PINS + EXPECTED_CALLS * 600);

    let native = Poseidon2Transcript::new(NEBULA_GAMMA_TRANSCRIPT_LABEL);
    let native_state = native
        .state()
        .map(|value| value.as_canonical_u64())
        .to_vec();
    assert_eq!(pins.iter().take(8).map(|pin| pin.1).collect::<Vec<_>>(), native_state);
    assert!(matches!(pieces.get(0), Some(GammaPiece::Pin(0))));
    assert!(matches!(pieces.get(7), Some(GammaPiece::Pin(7))));

    let challenge_calls = pieces
        .windows(3)
        .filter_map(|window| match window {
            [GammaPiece::Call(call), GammaPiece::Pin(first), GammaPiece::Pin(second)]
                if pins[*first].1 == 0x101 && pins[*second].1 == 2 =>
            {
                Some(*call)
            }
            _ => None,
        })
        .collect::<Vec<_>>();
    let [gamma1_call, gamma2_call] = challenge_calls.as_slice() else {
        panic!("gamma transcript must contain two two-field query calls: {challenge_calls:?}")
    };
    let gamma1_columns = calls[*gamma1_call].output_cols[..2]
        .try_into()
        .expect("two gamma1 fields");
    let gamma2_columns = calls[*gamma2_call].output_cols[..2]
        .try_into()
        .expect("two gamma2 fields");

    GammaTranscript {
        row_start,
        row_stop,
        pin_rows,
        pins,
        calls,
        initial_absorbed: native.absorbed(),
        gamma1_columns,
        gamma2_columns,
    }
}

fn render_poseidon2_calls(
    calls: &[Poseidon2PermutationAudit],
    row_offset: usize,
    relocate: impl Fn(usize) -> usize,
) -> String {
    let entries = calls
        .iter()
        .map(|call| {
            format!(
                "{{ rowStart := {}, rowEnd := {}, inputColumns := {:?}, firstAllocatedColumn := {} }}",
                row_offset + call.row_start,
                row_offset + call.row_end,
                call.input_cols.map(&relocate),
                relocate(call.first_allocated_col),
            )
        })
        .collect::<Vec<_>>()
        .join(",\n      ");
    format!("[{}]", entries)
}

fn lean_seed_rows(rows: &[Vec<[u8; 32]>]) -> String {
    let rows = rows
        .iter()
        .map(|chunks| {
            let chunks = chunks
                .iter()
                .map(|seed| {
                    format!(
                        "[{}]",
                        seed.iter()
                            .map(u8::to_string)
                            .collect::<Vec<_>>()
                            .join(", ")
                    )
                })
                .collect::<Vec<_>>();
            format!("[{}]", chunks.join(", "))
        })
        .collect::<Vec<_>>();
    format!("[{}]", rows.join(", "))
}

fn render_seed_schedule(block: &SeededPhi81LinearBlock) -> String {
    format!(
        "{{ chunkSize := {}, seedsByOutput := {}, rejectionFuel := 16 }}",
        block.chunk_size(),
        lean_seed_rows(block.chunk_seeds_by_row()),
    )
}

fn assert_same_schedule(left: &SeededPhi81LinearBlock, right: &SeededPhi81LinearBlock) {
    assert_eq!(left.word_width(), right.word_width());
    assert_eq!(left.kappa(), right.kappa());
    assert_eq!(left.message_cols(), right.message_cols());
    assert_eq!(left.chunk_size(), right.chunk_size());
    assert_eq!(left.chunk_seeds_by_row(), right.chunk_seeds_by_row());
    assert_eq!(
        left.has_superneo_transformed_columns(),
        right.has_superneo_transformed_columns()
    );
}

fn render_column_runs(columns: impl IntoIterator<Item = usize>) -> String {
    let columns = columns.into_iter().collect::<Vec<_>>();
    if columns.is_empty() {
        return "[]".into();
    }
    let mut runs = Vec::new();
    let mut start = columns[0];
    let mut stop = start + 1;
    for &column in &columns[1..] {
        if column == stop {
            stop += 1;
        } else {
            runs.push((start, stop));
            start = column;
            stop = column + 1;
        }
    }
    runs.push((start, stop));
    runs.into_iter()
        .map(|(start, stop)| {
            if stop == start + 1 {
                format!("[{start}]")
            } else {
                format!("List.range' {start} {}", stop - start)
            }
        })
        .collect::<Vec<_>>()
        .join(" ++ ")
}

fn render_leaf_block(
    reference: &LeafSeededBlockReference,
    schedule: &str,
    row_offset: usize,
    relocate: impl Fn(usize) -> usize,
) -> String {
    let word_starts = reference
        .block
        .word_starts()
        .iter()
        .copied()
        .map(&relocate)
        .collect::<Vec<_>>();
    assert_eq!(
        word_starts,
        (0..word_starts.len())
            .map(|index| word_starts[0] + CANONICAL_OPENING_COLUMNS * index)
            .collect::<Vec<_>>()
    );
    let output_columns = reference
        .output_columns
        .iter()
        .copied()
        .map(&relocate)
        .collect::<Vec<_>>();
    format!(
        "{{ sourceColumns := {}\n        metadataPinRowStart := {}\n        metadataValues := {:?}\n        metadataStartColumn := {}\n        openingRowStart := {}\n        block :=\n          {{ rowStart := {}\n            wordStarts := (List.range {}).map (fun index => {} + {} * index)\n            wordWidth := {}\n            kappa := {}\n            messageCols := {}\n            outputColumns := {}\n            superneoTransformedColumns := {}\n            schedule := {} }} }}",
        render_column_runs(reference.source_columns.iter().copied().map(&relocate)),
        row_offset + reference.metadata_pin_row_start,
        reference.metadata_values,
        relocate(reference.metadata_columns[0]),
        row_offset + reference.opening_row_start,
        row_offset + reference.block.row_start(),
        word_starts.len(),
        word_starts[0],
        CANONICAL_OPENING_COLUMNS,
        reference.block.word_width(),
        reference.block.kappa(),
        reference.block.message_cols(),
        render_column_runs(output_columns),
        reference.block.has_superneo_transformed_columns(),
        schedule,
    )
}

fn render_leaf(reference: &LeafSisReference, row_offset: usize, relocate: impl Fn(usize) -> usize + Copy) -> String {
    format!(
        "{{ prefixPinRowStart := {}\n    prefixConstantValues := {:?}\n    prefixConstantStartColumn := {}\n    primary :=\n      {}\n    compression :=\n      {}\n    envelopeConstantValues := {:?}\n    envelopeConstantStartColumn := {}\n    digestInputColumns := {}\n    digestOutputColumns := {}\n    digestRowStart := {}\n    digestRowStop := {} }}",
        row_offset + reference.prefix_pin_row_start,
        reference.prefix_constant_values,
        relocate(reference.prefix_constant_columns[0]),
        render_leaf_block(&reference.primary, "leafPrimarySchedule", row_offset, relocate),
        render_leaf_block(
            &reference.compression,
            "leafCompressionSchedule",
            row_offset,
            relocate,
        ),
        reference.envelope_constant_values,
        relocate(reference.envelope_constant_columns[0]),
        render_column_runs(reference.digest.input_cols.iter().copied().map(relocate)),
        render_column_runs(reference.digest.output_cols.into_iter().map(relocate)),
        row_offset + reference.digest.row_start,
        row_offset + reference.digest.row_end,
    )
}

fn render_chain_link(
    reference: &ChainLinkReference,
    row_offset: usize,
    relocate: impl Fn(usize) -> usize + Copy,
) -> String {
    format!(
        "{{ constantRowStart := {}\n      traceRowStart := {}\n      traceRowStop := {}\n      recipe :=\n        {{ constantValues := {:?}\n          constantStartColumn := {}\n          localColumns := {}\n          payloadColumns := {}\n          orderedInputColumns := {}\n          outputColumns := {} }} }}",
        row_offset + reference.constant_row_start,
        row_offset + reference.digest.row_start,
        row_offset + reference.digest.row_end,
        reference.constant_values,
        relocate(reference.constant_columns[0]),
        render_column_runs(reference.previous_columns.into_iter().map(relocate)),
        render_column_runs(reference.leaf_columns.into_iter().map(relocate)),
        render_column_runs(reference.digest.input_cols.iter().copied().map(relocate)),
        render_column_runs(reference.digest.output_cols.into_iter().map(relocate)),
    )
}

fn render_terms(terms: &[(usize, F)], relocate: impl Fn(usize) -> usize + Copy) -> String {
    let terms = terms
        .iter()
        .map(|&(column, coefficient)| format!("({}, {})", relocate(column), coefficient.as_canonical_u64()))
        .collect::<Vec<_>>();
    format!("[{}]", terms.join(", "))
}

fn render_owned_rows(
    source: &R1csSnapshot,
    rows: impl IntoIterator<Item = usize>,
    row_offset: usize,
    relocate: impl Fn(usize) -> usize + Copy,
) -> String {
    rows.into_iter()
        .map(|row| {
            format!(
                "({}, ⟨{}, {}, {}⟩)",
                row_offset + row,
                render_terms(source.a_row(row), relocate),
                render_terms(source.b_row(row), relocate),
                render_terms(source.c_row(row), relocate),
            )
        })
        .collect::<Vec<_>>()
        .join(",\n    ")
}

fn assert_bit_row(source: &R1csSnapshot, row: usize, column: usize) {
    assert_eq!(source.a_row(row), &[(column, F::ONE)]);
    assert_eq!(source.b_row(row), &[(0, -F::ONE), (column, F::ONE)]);
    assert!(source.c_row(row).is_empty());
}

fn assert_open_algebra_prefix(
    source: &R1csSnapshot,
    row_start: usize,
    internal_column_start: usize,
    lane_open: usize,
    lane_segment_index: usize,
    lane_step_index: usize,
    input_open: usize,
    segment_maximum: u64,
) -> usize {
    assert_eq!(segment_maximum, 1, "selected terminal profile segment maximum");
    let bit_columns = (internal_column_start..internal_column_start + 16).collect::<Vec<_>>();
    let mut row = row_start;
    for &column in &bit_columns {
        assert_bit_row(source, row, column);
        row += 1;
    }
    let recomposition = bit_columns
        .iter()
        .enumerate()
        .map(|(index, &column)| (column, F::from_u64(1u64 << index)))
        .collect::<Vec<_>>();
    assert_eq!(
        source.a_row(row),
        expected_linear_row(lane_segment_index, &recomposition)
    );
    assert_eq!(source.b_row(row), &[(0, F::ONE)]);
    assert!(source.c_row(row).is_empty());
    row += 1;

    let mut equal = 0;
    let mut next_equal = internal_column_start + 16;
    for index in (0..16).rev() {
        let bit = bit_columns[index];
        let bound_bit = (segment_maximum >> index) & 1;
        if bound_bit == 0 {
            assert_eq!(source.a_row(row), &[(equal, F::ONE)]);
            assert_eq!(source.b_row(row), &[(bit, F::ONE)]);
            assert!(source.c_row(row).is_empty());
            row += 1;
        }
        assert_eq!(source.a_row(row), &[(equal, F::ONE)]);
        if bound_bit == 0 {
            assert_eq!(source.b_row(row), &[(0, F::ONE), (bit, -F::ONE)]);
        } else {
            assert_eq!(source.b_row(row), &[(bit, F::ONE)]);
        }
        assert_eq!(source.c_row(row), &[(next_equal, F::ONE)]);
        equal = next_equal;
        next_equal += 1;
        row += 1;
    }
    assert_eq!(source.a_row(row), expected_linear_row(equal, &[]));
    assert_eq!(source.b_row(row), &[(0, F::ONE)]);
    assert!(source.c_row(row).is_empty());
    row += 1;

    assert_bit_row(source, row, lane_open);
    assert_bit_row(source, row + 1, input_open);
    assert_eq!(
        source.a_row(row + 2),
        &[(0, -F::ONE), (lane_open, F::ONE), (input_open, F::ONE)]
    );
    assert_eq!(source.b_row(row + 2), &[(0, F::ONE)]);
    assert!(source.c_row(row + 2).is_empty());
    assert_eq!(source.a_row(row + 3), &[(input_open, F::ONE)]);
    assert_eq!(source.b_row(row + 3), &[(lane_step_index, F::ONE)]);
    assert!(source.c_row(row + 3).is_empty());
    row + 4
}

fn first_internal_column(source: &R1csSnapshot, rows: std::ops::Range<usize>, external: &BTreeSet<usize>) -> usize {
    rows.flat_map(|row| {
        source
            .a_row(row)
            .iter()
            .chain(source.b_row(row))
            .chain(source.c_row(row))
            .map(|&(column, _)| column)
            .collect::<Vec<_>>()
    })
    .filter(|column| *column != 0 && !external.contains(column))
    .min()
    .expect("finalizer core must allocate internal columns")
}

fn contiguous_start(columns: &[usize], label: &str) -> usize {
    let start = *columns
        .first()
        .unwrap_or_else(|| panic!("{label} must not be empty"));
    assert!(
        columns.iter().copied().eq(start..start + columns.len()),
        "{label} must be contiguous"
    );
    start
}

pub(super) fn render(fixture: StreamingTerminalAuditFixture) -> String {
    let family_name = STREAMING_TERMINAL_R1CS_FAMILY_NAMES[6];
    let ranges = fixture
        .terminal
        .row_family_ranges()
        .iter()
        .filter(|range| range.name == family_name)
        .map(|range| range.row_start..range.row_end)
        .collect::<Vec<_>>();
    let [shape_rows, core_rows] = ranges.as_slice() else {
        panic!("terminal finalizer family must have shape and core ranges: {ranges:?}")
    };
    assert_eq!(shape_rows.len(), 6);
    let shape_columns = fixture
        .fresh_adv_shape_columns
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
    let shape_values = [fixture.fresh_adv_d, fixture.fresh_adv_kappa]
        .into_iter()
        .cycle()
        .take(6)
        .collect::<Vec<_>>();
    let full_source = fixture.terminal.snapshot();
    for (row, (&column, &value)) in shape_rows
        .clone()
        .zip(shape_columns.iter().zip(&shape_values))
    {
        assert_eq!(
            full_source.a_row(row),
            expected_linear_row(column, &[(0, F::from_u64(value as u64))])
        );
        assert_eq!(full_source.b_row(row), &[(0, F::ONE)]);
        assert!(full_source.c_row(row).is_empty());
    }

    let reference = reference_relation(&fixture, &full_source);
    assert_eq!(reference.source.rows(), core_rows.len());
    let full_external = reference
        .external
        .values()
        .copied()
        .collect::<BTreeSet<_>>();
    let full_internal_start = first_internal_column(&full_source, core_rows.clone(), &full_external);
    for (reference_row, full_row) in (0..reference.source.rows()).zip(core_rows.clone()) {
        assert_eq!(
            relocated_terms(
                reference.source.a_row(reference_row),
                &reference.external,
                reference.internal_start,
                full_internal_start,
            ),
            full_source.a_row(full_row),
        );
        assert_eq!(
            relocated_terms(
                reference.source.b_row(reference_row),
                &reference.external,
                reference.internal_start,
                full_internal_start,
            ),
            full_source.b_row(full_row),
        );
        assert_eq!(
            relocated_terms(
                reference.source.c_row(reference_row),
                &reference.external,
                reference.internal_start,
                full_internal_start,
            ),
            full_source.c_row(full_row),
        );
    }

    let relocate_column = |column: usize| {
        reference
            .external
            .get(&column)
            .copied()
            .unwrap_or_else(|| full_internal_start + column - reference.internal_start)
    };
    let opened_lane_columns = reference.opened_lane_columns.map(relocate_column);
    let advanced_lane_columns = reference.advanced_lane_columns.map(relocate_column);
    let final_lane_columns = reference.final_lane_columns.map(relocate_column);
    assert_eq!(final_lane_columns, fixture.final_lane_columns);
    let closed_column = relocate_column(reference.closed_column);
    let lane_start = fixture.source_binding_decoded_column_start + 32;
    let payload_start = contiguous_start(&fixture.delayed_payload_columns, "delayed payload");
    let ops_start = contiguous_start(&fixture.fresh_adv_data_columns.ops, "ops commitment");
    let is_start = contiguous_start(&fixture.fresh_adv_data_columns.is, "IS commitment");
    let fs_start = contiguous_start(&fixture.fresh_adv_data_columns.fs, "FS commitment");
    let open_internal_column_start = full_internal_start + reference.decode_column_end - reference.internal_start;
    let open_algebra_row_stop = assert_open_algebra_prefix(
        &full_source,
        core_rows.start + reference.decode_end,
        open_internal_column_start,
        lane_start + 4,
        lane_start + 5,
        lane_start + 6,
        payload_start + 1400,
        fixture.seg_max,
    ) - core_rows.start;
    let staged_digest = reference
        .poseidon2_hash_audits
        .first()
        .expect("staged Nebula lane digest audit");
    assert_eq!(staged_digest.input_cols.len(), 58);
    assert_eq!(staged_digest.rounds.len(), 16);
    assert_eq!(staged_digest.row_start, staged_digest.zero_row);
    assert_eq!(
        staged_digest.row_start,
        reference.decode_end + 53 + 13,
        "staged digest follows open algebra and 13 constants"
    );
    let staged_constant_start = staged_digest
        .zero_col
        .checked_sub(13)
        .expect("staged digest constant start");
    let staged_constant_values = (staged_constant_start..staged_digest.zero_col)
        .map(|column| reference.source.witness()[column].as_canonical_u64())
        .collect::<Vec<_>>();
    let staged_digest_constant_start_column = relocate_column(staged_constant_start);
    let staged_digest_input_columns = staged_digest
        .input_cols
        .iter()
        .copied()
        .map(relocate_column)
        .collect::<Vec<_>>();
    let staged_digest_output_columns = staged_digest.output_cols.map(relocate_column);
    let staged_digest_row_start = core_rows.start + staged_digest.row_start;
    let staged_digest_row_stop = core_rows.start + staged_digest.row_end;
    let gamma_transcript = gamma_transcript(&reference, staged_digest.row_end);
    let gamma_transcript_pin_rows = gamma_transcript
        .pin_rows
        .iter()
        .map(|row| core_rows.start + row)
        .collect::<Vec<_>>();
    let gamma_transcript_pin_columns = gamma_transcript
        .pins
        .iter()
        .map(|pin| relocate_column(pin.0))
        .collect::<Vec<_>>();
    let gamma_transcript_pin_values = gamma_transcript
        .pins
        .iter()
        .map(|pin| pin.1)
        .collect::<Vec<_>>();
    let gamma_transcript_calls = render_poseidon2_calls(&gamma_transcript.calls, core_rows.start, relocate_column);
    let gamma1_columns = gamma_transcript.gamma1_columns.map(relocate_column);
    let gamma2_columns = gamma_transcript.gamma2_columns.map(relocate_column);
    let gamma_mux_opened_columns: [usize; 16] = gamma_transcript
        .gamma1_columns
        .into_iter()
        .chain(gamma_transcript.gamma2_columns)
        .chain(reference.gamma_mux_opened_d_pre_columns)
        .collect::<Vec<_>>()
        .try_into()
        .expect("sixteen opened mux sources");
    assert_eq!(reference.open_end - gamma_transcript.row_stop, 16);
    for (index, ((&opened, &carried), &output)) in gamma_mux_opened_columns
        .iter()
        .zip(&reference.gamma_mux_carried_columns)
        .zip(&reference.gamma_mux_output_columns)
        .enumerate()
    {
        assert_ne!(opened, carried);
        assert_ne!(output, carried);
        let row = gamma_transcript.row_stop + index;
        assert_eq!(
            reference.source.a_row(row),
            &[(reference.gamma_mux_selector_column, F::ONE)]
        );
        assert_eq!(
            reference.source.b_row(row),
            expected_linear_row(opened, &[(carried, F::ONE)])
        );
        assert_eq!(
            reference.source.c_row(row),
            expected_linear_row(output, &[(carried, F::ONE)])
        );
    }
    let gamma_mux_selector_column = relocate_column(reference.gamma_mux_selector_column);
    let gamma_mux_opened_columns = gamma_mux_opened_columns.map(relocate_column);
    let gamma_mux_carried_columns = reference.gamma_mux_carried_columns.map(relocate_column);
    let gamma_mux_output_columns = reference.gamma_mux_output_columns.map(relocate_column);
    for leaf in &reference.leaf_sis[1..] {
        assert_same_schedule(&reference.leaf_sis[0].primary.block, &leaf.primary.block);
        assert_same_schedule(&reference.leaf_sis[0].compression.block, &leaf.compression.block);
    }
    let advance_chain_links = reference
        .advance_chain_links
        .iter()
        .map(|link| render_chain_link(link, core_rows.start, relocate_column))
        .collect::<Vec<_>>()
        .join(",\n    ");
    let advance_algebra_indices = (reference.leaves_end..reference.advance_end)
        .filter(|&row| {
            !reference
                .advance_chain_links
                .iter()
                .any(|link| (link.constant_row_start..link.digest.row_end).contains(&row))
        })
        .collect::<Vec<_>>();
    assert_eq!(advance_algebra_indices.len(), 19);
    let advance_algebra_rows = render_owned_rows(
        &reference.source,
        advance_algebra_indices,
        core_rows.start,
        relocate_column,
    );
    let close_rows = render_owned_rows(
        &reference.source,
        reference.advance_end..reference.close_end,
        core_rows.start,
        relocate_column,
    );
    let terminal_closed_row = render_owned_rows(
        &reference.source,
        [reference.close_end],
        core_rows.start,
        relocate_column,
    );
    let leaf_definitions = format!(
        "def leafPrimarySchedule : Nightstream.Implementation.R1CS.SeededPhi81.SeedSchedule :=\n  {}\n\n\
         def leafCompressionSchedule : Nightstream.Implementation.R1CS.SeededPhi81.SeedSchedule :=\n  {}\n\n\
         def opsLeaf : LeafHashArtifact :=\n  {}\n\n\
         def isLeaf : LeafHashArtifact :=\n  {}\n\n\
         def fsLeaf : LeafHashArtifact :=\n  {}\n\n\
         def advanceChainLinks : List PoseidonHashArtifact :=\n  [\n    {}\n  ]\n\n\
         def advanceAlgebraRows : List (Nat × Row) :=\n  [\n    {}\n  ]\n\n\
         def closeRows : List (Nat × Row) :=\n  [\n    {}\n  ]\n\n\
         def terminalClosedRow : Nat × Row :=\n  {}\n",
        render_seed_schedule(&reference.leaf_sis[0].primary.block),
        render_seed_schedule(&reference.leaf_sis[0].compression.block),
        render_leaf(&reference.leaf_sis[0], core_rows.start, relocate_column),
        render_leaf(&reference.leaf_sis[1], core_rows.start, relocate_column),
        render_leaf(&reference.leaf_sis[2], core_rows.start, relocate_column),
        advance_chain_links,
        advance_algebra_rows,
        close_rows,
        terminal_closed_row,
    );

    format!(
        "import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingTerminalFinalizerSchema\n\n\
         /-! Generated exact full-layout Rust terminal Nebula-finalizer geometry.\n\n\
         Rust compares every core row with the ordered production phase reference.\n\
         The empty SHA field is legacy diagnostic structure and is not authority.\n\
         Emits constraints: no.\n\
         -/\n\n\
         set_option autoImplicit false\n\n\
         namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingTerminalFullFinalizer\n\n\
         open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingTerminalFinalizer.Artifact\n\n\
         def lifecycleScope : String := \"recursive-terminal-arm-435\"\n\n\
         {}\n\
         def rawArtifact : RawArtifact :=\n  \
         {{ schemaVersion := 3,\n    \
            profileId := \"nightstream/goldilocks/streaming-terminal-full-finalizer/v1\",\n    \
            sourceIdentity := \"rust:nightstream/streaming-terminal-lifecycle/source-rows/v1\",\n    \
            sourceRowsSha256 := \"\", columnCount := {},\n    \
            shapeRowStart := {}, shapeRowStop := {}, shapeColumns := {:?},\n    \
            dimension := {}, kappa := {},\n    \
            stepsPerSegment := {}, segmentMaximum := {}, stackCount := {}, stackPointerBits := {},\n    \
            coreRowStart := {}, coreRowStop := {}, internalColumnStart := {},\n    \
            laneColumns := List.range' {} 50, payloadColumns := List.range' {} {},\n    \
            opsColumns := List.range' {} {}, isColumns := List.range' {} {},\n    \
            fsColumns := List.range' {} {}, vkFsColumns := {:?},\n    \
            boundaryColumns := {:?}, accumulatorColumns := {:?},\n    \
            decodeRowStop := {}, openRowStop := {}, leavesRowStop := {},\n    \
            openAlgebraRowStop := {}, openInternalColumnStart := {},\n    \
            stagedDigestConstantValues := {:?}, stagedDigestConstantStartColumn := {},\n    \
            stagedDigestInputColumns := {:?}, stagedDigestOutputColumns := {:?},\n    \
            stagedDigestRowStart := {}, stagedDigestRowStop := {},\n    \
            gammaTranscriptRowStart := {}, gammaTranscriptRowStop := {},\n    \
            gammaTranscriptPinRows := {:?}, gammaTranscriptPinColumns := {:?},\n    \
            gammaTranscriptPinValues := {:?}, gammaTranscriptInitialAbsorbed := {},\n    \
            gammaTranscriptCalls := {}, gamma1Columns := {:?}, gamma2Columns := {:?},\n    \
            gammaMuxSelectorColumn := {}, gammaMuxOpenedColumns := {:?},\n    \
            gammaMuxCarriedColumns := {:?}, gammaMuxOutputColumns := {:?},\n    \
            opsLeaf := opsLeaf, isLeaf := isLeaf, fsLeaf := fsLeaf,\n    \
            advanceChainLinks := advanceChainLinks,\n    \
            openedLaneColumns := {:?}, advancedLaneColumns := {:?},\n    \
            advanceAlgebraRows := advanceAlgebraRows, closeRows := closeRows,\n    \
            terminalClosedRow := terminalClosedRow,\n    \
            advanceRowStop := {}, closeRowStop := {}, coreRowCount := {},\n    \
            finalLaneColumns := {:?}, closedColumn := {} }}\n\n\
         end Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingTerminalFullFinalizer\n",
        leaf_definitions,
        full_source.cols(),
        shape_rows.start,
        shape_rows.end,
        shape_columns,
        fixture.fresh_adv_d,
        fixture.fresh_adv_kappa,
        fixture.steps_per_segment,
        fixture.seg_max,
        fixture.stacks.count,
        fixture.stacks.sigma,
        core_rows.start,
        core_rows.end,
        full_internal_start,
        lane_start,
        payload_start,
        fixture.delayed_payload_columns.len(),
        ops_start,
        fixture.fresh_adv_data_columns.ops.len(),
        is_start,
        fixture.fresh_adv_data_columns.is.len(),
        fs_start,
        fixture.fresh_adv_data_columns.fs.len(),
        fixture.vk_fs_columns,
        fixture.boundary_columns,
        fixture.accumulator_columns,
        reference.decode_end,
        reference.open_end,
        reference.leaves_end,
        open_algebra_row_stop,
        open_internal_column_start,
        staged_constant_values,
        staged_digest_constant_start_column,
        staged_digest_input_columns,
        staged_digest_output_columns,
        staged_digest_row_start,
        staged_digest_row_stop,
        core_rows.start + gamma_transcript.row_start,
        core_rows.start + gamma_transcript.row_stop,
        gamma_transcript_pin_rows,
        gamma_transcript_pin_columns,
        gamma_transcript_pin_values,
        gamma_transcript.initial_absorbed,
        gamma_transcript_calls,
        gamma1_columns,
        gamma2_columns,
        gamma_mux_selector_column,
        gamma_mux_opened_columns,
        gamma_mux_carried_columns,
        gamma_mux_output_columns,
        opened_lane_columns,
        advanced_lane_columns,
        reference.advance_end,
        reference.close_end,
        reference.source.rows(),
        final_lane_columns,
        closed_column,
    )
}

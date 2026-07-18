//! Three-matrix diagnostic PiRLC rejection-sampler layout artifact.
//!
//! Owns: exact source-row geometry from 15 sampler scalars through their 16
//! canonical lanes, 64 candidates, and first-54 selection tails, plus compact
//! affine row/column metadata for Lean refinement.
//!
//! Does not own: Poseidon2 transcript authority, sampler soundness, projection
//! identities, encoded costs, or permission to remove constraints.
//!
//! Emits constraints: no.
//!
//! | Stage path | Obligation checked | Multiplicity | Evidence tier |
//! |---|---|---:|---|
//! | `challenge.transcript.lane_bit_decomposition` | exact canonical-u64 rows and lane placement | `15 x 16` | artifact-checked |
//! | `challenge.sampler.chunk` | exact acceptance, mod-5, symbol, and prefix geometry | `15 x 64` | artifact-checked |
//! | `challenge.sampler.acceptance_bound` | exact six-row bounded-success tail | `15` | artifact-checked |
//! | `challenge.sampler.selection.initialize` | exact selection-zero row | `15` | artifact-checked |
//! | `challenge.sampler.selection.one_hot` | exact one-hot window equations | `15 x 54 x 12` | artifact-checked |
//! | `challenge.sampler.selection.products` | exact selected-symbol products | `15 x 54 x 33` | artifact-checked |
//! | `challenge.sampler.selection.bindings` | exact accept/prefix/symbol bindings | `15 x 54 x 3` | artifact-checked |
//! | sampler output -> projection rho | physical output-column identity | `15 x 54` | artifact-checked |

use std::collections::{BTreeMap, HashMap};
use std::fmt::Write as _;
use std::fs;

use neo_fold_clean::engine::r1cs_circuit::alphabet_sampling::pi_rlc_challenge_stage;
use neo_fold_clean::engine::r1cs_circuit::{
    AcceptanceTraceEntry, CanonicalU64TraceEntry, FirstAcceptedSelectionTraceEntry, Mod5TraceEntry, R1csEncodingTrace,
    R1csSnapshot, Var,
};
use neo_math::F;
use p3_field::PrimeCharacteristicRing;

use super::{active_rho_challenge_wiring, repo_root};

const LEAN_DATA_PATH: &str = "formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/FPrimeRecursive/PiRlcChallenge/Generated/SamplerLayoutData.lean";
const SCALAR_COUNT: usize = 15;
const BLOCK_COUNT: usize = 4;
const LANES_PER_BLOCK: usize = 4;
const LANE_COUNT: usize = BLOCK_COUNT * LANES_PER_BLOCK;
const CHUNKS_PER_LANE: usize = 4;
const CANDIDATE_COUNT: usize = LANE_COUNT * CHUNKS_PER_LANE;
const OUTPUT_COUNT: usize = 54;
const SELECTION_WIDTH: usize = CANDIDATE_COUNT - OUTPUT_COUNT + 1;
const CANONICAL_ROWS: usize = 69;
const LANE_RESIDUAL_ROWS: usize = 104;
const LANE_ROWS: usize = CANONICAL_ROWS + LANE_RESIDUAL_ROWS;
const TAIL_ROWS: usize = 6 + 1 + OUTPUT_COUNT * (12 + 33 + 3);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Candidate {
    symbol: usize,
    accept: usize,
    prefix: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Affine1 {
    base: usize,
    stride: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Affine3 {
    base: usize,
    rho_stride: usize,
    block_stride: usize,
    lane_stride: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct SamplerLayout {
    initial_count: Affine1,
    initialization_row: Affine1,
    selection_zero: Affine1,
    selection_zero_row: Affine1,
    field: Affine3,
    bit_start: Affine3,
    canonical_row: Affine3,
    tail_first_allocated: Affine1,
    tail_row: Affine1,
    output_offset: usize,
    output_stride: usize,
}

type Row = Vec<(usize, F)>;

fn normalized(terms: impl IntoIterator<Item = (usize, F)>) -> Row {
    let mut combined = BTreeMap::<usize, F>::new();
    for (column, coefficient) in terms {
        *combined.entry(column).or_insert(F::ZERO) += coefficient;
    }
    combined
        .into_iter()
        .filter(|(_, coefficient)| *coefficient != F::ZERO)
        .collect()
}

fn one_row() -> Row {
    vec![(Var::ONE.col(), F::ONE)]
}

fn assert_row(source: &R1csSnapshot, row: usize, a: Row, b: Row, c: Row, label: &str) {
    assert_eq!(source.a_row(row), normalized(a), "{label}: A row {row}");
    assert_eq!(source.b_row(row), normalized(b), "{label}: B row {row}");
    assert_eq!(source.c_row(row), normalized(c), "{label}: C row {row}");
}

fn assert_linear_zero(source: &R1csSnapshot, row: usize, terms: Row, label: &str) {
    assert_row(source, row, terms, one_row(), Vec::new(), label);
}

fn assert_bit(source: &R1csSnapshot, row: usize, column: usize, label: &str) {
    assert_row(
        source,
        row,
        vec![(column, F::ONE)],
        vec![(column, F::ONE), (Var::ONE.col(), -F::ONE)],
        Vec::new(),
        label,
    );
}

fn powers_of_two(columns: impl IntoIterator<Item = usize>, sign: F) -> Row {
    let mut power = F::ONE;
    columns
        .into_iter()
        .map(|column| {
            let term = (column, sign * power);
            power += power;
            term
        })
        .collect()
}

fn selection_start(trace: &R1csEncodingTrace, expected_outputs: &[usize]) -> usize {
    let selections = trace.first_accepted_selections();
    let width = expected_outputs.len();
    assert!(selections.len() >= width, "active selection trace width");
    let matches = (0..=selections.len() - width)
        .filter(|&start| {
            selections[start..start + width]
                .iter()
                .enumerate()
                .all(|(offset, event)| {
                    event.position == offset % OUTPUT_COUNT && event.output.col() == expected_outputs[offset]
                })
        })
        .collect::<Vec<_>>();
    let &[start] = matches.as_slice() else {
        panic!("expected one active 15 x 54 selection window, found {}", matches.len());
    };
    start
}

fn candidates(events: &[FirstAcceptedSelectionTraceEntry]) -> [Candidate; CANDIDATE_COUNT] {
    assert_eq!(events.len(), OUTPUT_COUNT, "one scalar selection output count");
    let mut out = [None; CANDIDATE_COUNT];
    for (position, event) in events.iter().enumerate() {
        assert_eq!(event.position, position, "selection position order");
        assert_eq!(event.one_hot.len(), SELECTION_WIDTH, "selection window width");
        assert_eq!(event.symbols.len(), SELECTION_WIDTH, "selection symbol width");
        assert_eq!(event.accepts.len(), SELECTION_WIDTH, "selection accept width");
        assert_eq!(event.prefixes.len(), SELECTION_WIDTH, "selection prefix width");
        assert_eq!(event.products.len(), SELECTION_WIDTH, "selection product width");
        for offset in 0..SELECTION_WIDTH {
            let candidate = Candidate {
                symbol: event.symbols[offset].col(),
                accept: event.accepts[offset].col(),
                prefix: event.prefixes[offset].col(),
            };
            let slot = &mut out[position + offset];
            if let Some(previous) = slot {
                assert_eq!(*previous, candidate, "overlapping selection windows agree");
            } else {
                *slot = Some(candidate);
            }
        }
    }
    out.map(|candidate| candidate.expect("all 64 candidates covered by selection windows"))
}

fn affine1(values: &[usize], label: &str) -> Affine1 {
    assert_eq!(values.len(), SCALAR_COUNT, "{label} scalar count");
    let base = values[0];
    let stride = values[1]
        .checked_sub(base)
        .unwrap_or_else(|| panic!("{label} increases"));
    for (rho, value) in values.iter().copied().enumerate() {
        assert_eq!(value, base + stride * rho, "{label} affine formula");
    }
    Affine1 { base, stride }
}

fn affine3(values: &[[[usize; LANES_PER_BLOCK]; BLOCK_COUNT]; SCALAR_COUNT], label: &str) -> Affine3 {
    let base = values[0][0][0];
    let rho_stride = values[1][0][0]
        .checked_sub(base)
        .unwrap_or_else(|| panic!("{label} rho increases"));
    let block_stride = values[0][1][0]
        .checked_sub(base)
        .unwrap_or_else(|| panic!("{label} block increases"));
    let lane_stride = values[0][0][1]
        .checked_sub(base)
        .unwrap_or_else(|| panic!("{label} lane increases"));
    for rho in 0..SCALAR_COUNT {
        for block in 0..BLOCK_COUNT {
            for lane in 0..LANES_PER_BLOCK {
                assert_eq!(
                    values[rho][block][lane],
                    base + rho_stride * rho + block_stride * block + lane_stride * lane,
                    "{label} affine formula"
                );
            }
        }
    }
    Affine3 {
        base,
        rho_stride,
        block_stride,
        lane_stride,
    }
}

fn assert_canonical_rows(source: &R1csSnapshot, event: &CanonicalU64TraceEntry) {
    assert_eq!(event.source_rows.len(), CANONICAL_ROWS, "canonical row count");
    let bit_start = event.bits[0].col();
    assert!(
        event
            .bits
            .iter()
            .enumerate()
            .all(|(offset, bit)| bit.col() == bit_start + offset),
        "canonical bit columns are contiguous"
    );
    assert_eq!(event.high_is_max.col(), bit_start + 64, "canonical high flag column");
    assert_eq!(event.inverse.col(), bit_start + 65, "canonical inverse column");
    for (offset, bit) in event.bits.iter().enumerate() {
        assert_bit(source, event.source_rows.start + offset, bit.col(), "canonical bit");
    }
    let mut recomposition = vec![(event.field.col(), F::ONE)];
    recomposition.extend(powers_of_two(event.bits.iter().map(|bit| bit.col()), -F::ONE));
    assert_linear_zero(
        source,
        event.source_rows.start + 64,
        recomposition,
        "canonical recomposition",
    );
    assert_bit(
        source,
        event.source_rows.start + 65,
        event.high_is_max.col(),
        "canonical high flag",
    );
    let high = powers_of_two(event.bits[32..].iter().map(|bit| bit.col()), F::ONE);
    let low = powers_of_two(event.bits[..32].iter().map(|bit| bit.col()), F::ONE);
    let mut difference = high;
    difference.push((Var::ONE.col(), -F::from_u64(0xffff_ffff)));
    assert_row(
        source,
        event.source_rows.start + 66,
        vec![(event.high_is_max.col(), F::ONE)],
        difference.clone(),
        Vec::new(),
        "canonical high equality",
    );
    assert_row(
        source,
        event.source_rows.start + 67,
        difference,
        vec![(event.inverse.col(), F::ONE)],
        vec![(Var::ONE.col(), F::ONE), (event.high_is_max.col(), -F::ONE)],
        "canonical high inequality",
    );
    assert_row(
        source,
        event.source_rows.start + 68,
        vec![(event.high_is_max.col(), F::ONE)],
        low,
        Vec::new(),
        "canonical low gate",
    );
}

fn assert_acceptance_rows(source: &R1csSnapshot, event: &AcceptanceTraceEntry) {
    assert_eq!(event.source_rows.len(), 4, "acceptance row count");
    assert_eq!(event.allocated_columns.len(), 2, "acceptance column count");
    assert_eq!(event.accept.col(), event.allocated_columns.start, "accept column order");
    assert_eq!(
        event.inverse.col(),
        event.allocated_columns.start + 1,
        "inverse column order"
    );
    assert_bit(source, event.source_rows.start, event.accept.col(), "accept bit");
    let mut difference = powers_of_two(event.chunk_bits.iter().map(|bit| bit.col()), F::ONE);
    difference.push((Var::ONE.col(), -F::from_u64(65_535)));
    let one_minus = vec![(Var::ONE.col(), F::ONE), (event.accept.col(), -F::ONE)];
    assert_row(
        source,
        event.source_rows.start + 1,
        one_minus.clone(),
        difference.clone(),
        Vec::new(),
        "accept zero case",
    );
    assert_row(
        source,
        event.source_rows.start + 2,
        difference,
        vec![(event.inverse.col(), F::ONE)],
        vec![(event.accept.col(), F::ONE)],
        "accept inverse",
    );
    assert_row(
        source,
        event.source_rows.start + 3,
        one_minus,
        vec![(event.inverse.col(), F::ONE)],
        Vec::new(),
        "accept inverse canonicalization",
    );
}

fn assert_mod5_rows(source: &R1csSnapshot, event: &Mod5TraceEntry) {
    assert_eq!(event.source_rows.len(), 20, "mod-5 row count");
    assert_eq!(event.allocated_columns.len(), 19, "mod-5 column count");
    let first = event.allocated_columns.start;
    assert_eq!(event.index.col(), first, "mod-5 index column");
    assert_eq!(event.quotient.col(), first + 1, "mod-5 quotient column");
    assert!(
        event
            .index_products
            .iter()
            .enumerate()
            .all(|(i, value)| value.col() == first + 2 + i),
        "mod-5 product columns"
    );
    assert!(
        event
            .quotient_bits
            .iter()
            .enumerate()
            .all(|(i, value)| value.col() == first + 5 + i),
        "mod-5 quotient-bit columns"
    );
    let factors = [
        event.index,
        event.index_products[0],
        event.index_products[1],
        event.index_products[2],
    ];
    for offset in 0..4 {
        let c = (offset < 3)
            .then(|| vec![(event.index_products[offset].col(), F::ONE)])
            .unwrap_or_default();
        assert_row(
            source,
            event.source_rows.start + offset,
            vec![(factors[offset].col(), F::ONE)],
            vec![
                (event.index.col(), F::ONE),
                (Var::ONE.col(), -F::from_u64((offset + 1) as u64)),
            ],
            c,
            "mod-5 index range",
        );
    }
    for (offset, bit) in event.quotient_bits.iter().enumerate() {
        assert_bit(
            source,
            event.source_rows.start + 4 + offset,
            bit.col(),
            "mod-5 quotient bit",
        );
    }
    let mut quotient = vec![(event.quotient.col(), F::ONE)];
    quotient.extend(powers_of_two(event.quotient_bits.iter().map(|bit| bit.col()), -F::ONE));
    assert_linear_zero(source, event.source_rows.start + 18, quotient, "mod-5 quotient");
    let mut decomposition = powers_of_two(event.chunk_bits.iter().map(|bit| bit.col()), F::ONE);
    decomposition.extend([(event.quotient.col(), -F::from_u64(5)), (event.index.col(), -F::ONE)]);
    assert_linear_zero(
        source,
        event.source_rows.start + 19,
        decomposition,
        "mod-5 decomposition",
    );
}

fn assert_selection_rows(source: &R1csSnapshot, event: &FirstAcceptedSelectionTraceEntry) {
    assert_eq!(event.one_hot_rows.len(), SELECTION_WIDTH + 1, "one-hot row count");
    assert_eq!(
        event.product_rows.len(),
        3 * SELECTION_WIDTH,
        "selection product row count"
    );
    assert_eq!(event.bind_rows.len(), 3, "selection bind row count");
    for (offset, selector) in event.one_hot.iter().enumerate() {
        assert_bit(
            source,
            event.one_hot_rows.start + offset,
            selector.col(),
            "selection one-hot bit",
        );
    }
    let mut one_hot_sum = event
        .one_hot
        .iter()
        .map(|selector| (selector.col(), F::ONE))
        .collect::<Row>();
    one_hot_sum.push((Var::ONE.col(), -F::ONE));
    assert_linear_zero(source, event.one_hot_rows.end - 1, one_hot_sum, "selection one-hot sum");
    for index in 0..SELECTION_WIDTH {
        let row = event.product_rows.start + 3 * index;
        let selector = vec![(event.one_hot[index].col(), F::ONE)];
        assert_row(
            source,
            row,
            selector.clone(),
            vec![(event.symbols[index].col(), F::ONE)],
            vec![(event.products[index].symbol.col(), F::ONE)],
            "selection symbol product",
        );
        assert_row(
            source,
            row + 1,
            selector.clone(),
            vec![(event.accepts[index].col(), F::ONE)],
            vec![(event.products[index].accepted.col(), F::ONE)],
            "selection accept product",
        );
        assert_row(
            source,
            row + 2,
            selector,
            vec![(event.prefixes[index].col(), F::ONE)],
            vec![(event.products[index].prefix.col(), F::ONE)],
            "selection prefix product",
        );
    }
    let mut accepted = event
        .products
        .iter()
        .map(|product| (product.accepted.col(), F::ONE))
        .collect::<Row>();
    accepted.push((Var::ONE.col(), -F::ONE));
    assert_linear_zero(source, event.bind_rows.start, accepted, "selection accept bind");
    let mut prefix = event
        .products
        .iter()
        .map(|product| (product.prefix.col(), F::ONE))
        .collect::<Row>();
    prefix.push((Var::ONE.col(), -F::from_u64(event.position as u64)));
    assert_linear_zero(source, event.bind_rows.start + 1, prefix, "selection prefix bind");
    let mut symbol = vec![(event.output.col(), F::ONE)];
    symbol.extend(
        event
            .products
            .iter()
            .map(|product| (product.symbol.col(), -F::ONE)),
    );
    assert_linear_zero(source, event.bind_rows.start + 2, symbol, "selection symbol bind");
}

fn recurrence_prefix(source: &R1csSnapshot, row: usize, cumulative: usize, accept: usize) -> usize {
    assert_eq!(source.b_row(row), one_row(), "candidate recurrence B row");
    assert!(source.c_row(row).is_empty(), "candidate recurrence C row");
    let remaining = source
        .a_row(row)
        .iter()
        .copied()
        .filter(|(column, _)| *column != cumulative && *column != accept)
        .collect::<Vec<_>>();
    assert_eq!(
        source
            .a_row(row)
            .iter()
            .find(|(column, _)| *column == cumulative),
        Some(&(cumulative, F::ONE)),
        "candidate recurrence output"
    );
    assert_eq!(
        source
            .a_row(row)
            .iter()
            .find(|(column, _)| *column == accept),
        Some(&(accept, -F::ONE)),
        "candidate recurrence accept"
    );
    let &[(prefix, coefficient)] = remaining.as_slice() else {
        panic!("candidate recurrence has one prior prefix column");
    };
    assert_eq!(coefficient, -F::ONE, "candidate recurrence prior coefficient");
    prefix
}

fn extract(source: &R1csSnapshot, trace: &R1csEncodingTrace) -> SamplerLayout {
    let projection_outputs = active_rho_challenge_wiring::projection_rho_columns(trace)
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
    assert_eq!(
        projection_outputs.len(),
        SCALAR_COUNT * OUTPUT_COUNT,
        "projection output width"
    );
    let selection_start = selection_start(trace, &projection_outputs);
    let selection_events =
        &trace.first_accepted_selections()[selection_start..selection_start + SCALAR_COUNT * OUTPUT_COUNT];

    let acceptance_by_column = trace
        .acceptance_chunks()
        .iter()
        .enumerate()
        .map(|(index, event)| (event.accept.col(), index))
        .collect::<HashMap<_, _>>();
    let mod5_by_symbol = trace
        .mod5_chunks()
        .iter()
        .enumerate()
        .map(|(index, event)| (event.allocated_columns.end, index))
        .collect::<HashMap<_, _>>();
    let canonical_by_bit_start = trace
        .canonical_u64_decompositions()
        .iter()
        .enumerate()
        .map(|(index, event)| (event.bits[0].col(), index))
        .collect::<HashMap<_, _>>();
    assert_eq!(
        acceptance_by_column.len(),
        trace.acceptance_chunks().len(),
        "unique acceptance outputs"
    );
    assert_eq!(
        mod5_by_symbol.len(),
        trace.mod5_chunks().len(),
        "unique mod-5 successors"
    );
    assert_eq!(
        canonical_by_bit_start.len(),
        trace.canonical_u64_decompositions().len(),
        "unique canonical bit starts"
    );

    let mut initial_counts = Vec::with_capacity(SCALAR_COUNT);
    let mut initialization_rows = Vec::with_capacity(SCALAR_COUNT);
    let mut selection_zeros = Vec::with_capacity(SCALAR_COUNT);
    let mut selection_zero_rows = Vec::with_capacity(SCALAR_COUNT);
    let mut fields = [[[0; LANES_PER_BLOCK]; BLOCK_COUNT]; SCALAR_COUNT];
    let mut bit_starts = fields;
    let mut canonical_rows = fields;
    let mut tail_first_allocated = Vec::with_capacity(SCALAR_COUNT);
    let mut tail_rows = Vec::with_capacity(SCALAR_COUNT);
    let mut output_offset = None;
    let mut output_stride = None;

    for rho in 0..SCALAR_COUNT {
        let events = &selection_events[rho * OUTPUT_COUNT..(rho + 1) * OUTPUT_COUNT];
        let candidates = candidates(events);
        let first_mod5 = &trace.mod5_chunks()[*mod5_by_symbol
            .get(&candidates[0].symbol)
            .expect("first selection symbol has traced mod-5 predecessor")];
        let initial_count = recurrence_prefix(
            source,
            first_mod5.source_rows.end + 1,
            candidates[0].symbol + 1,
            candidates[0].accept,
        );
        let stage_matches = trace
            .stages()
            .iter()
            .filter(|stage| stage.label == pi_rlc_challenge_stage::SAMPLE_INITIALIZE && stage.col == initial_count)
            .collect::<Vec<_>>();
        let [initial_stage] = stage_matches.as_slice() else {
            panic!("rho {rho}: one initialize checkpoint, found {}", stage_matches.len());
        };
        assert_linear_zero(
            source,
            initial_stage.row,
            vec![(initial_count, F::ONE)],
            "sampler initialize",
        );
        initial_counts.push(initial_count);
        initialization_rows.push(initial_stage.row);

        let mut final_cumulative = 0;
        for lane_index in 0..LANE_COUNT {
            let first_candidate = lane_index * CHUNKS_PER_LANE;
            let first_accept = &trace.acceptance_chunks()[*acceptance_by_column
                .get(&candidates[first_candidate].accept)
                .expect("selection accept has traced source")];
            let canonical = &trace.canonical_u64_decompositions()[*canonical_by_bit_start
                .get(&first_accept.chunk_bits[0].col())
                .expect("candidate bits have canonical source")];
            assert_canonical_rows(source, canonical);
            let block = lane_index / LANES_PER_BLOCK;
            let lane = lane_index % LANES_PER_BLOCK;
            fields[rho][block][lane] = canonical.field.col();
            bit_starts[rho][block][lane] = canonical.bits[0].col();
            canonical_rows[rho][block][lane] = canonical.source_rows.start;

            for chunk in 0..CHUNKS_PER_LANE {
                let index = first_candidate + chunk;
                let candidate = candidates[index];
                let acceptance = &trace.acceptance_chunks()[*acceptance_by_column
                    .get(&candidate.accept)
                    .expect("selection accept has traced source")];
                let mod5 = &trace.mod5_chunks()[*mod5_by_symbol
                    .get(&candidate.symbol)
                    .expect("selection symbol has traced mod-5 predecessor")];
                let acceptance_bits = acceptance.chunk_bits.map(|bit| bit.col());
                let canonical_bits = canonical.bits[chunk * 16..(chunk + 1) * 16]
                    .iter()
                    .map(|bit| bit.col())
                    .collect::<Vec<_>>();
                assert_eq!(
                    acceptance_bits.as_slice(),
                    canonical_bits,
                    "candidate bits are the canonical lane slice"
                );
                assert_eq!(
                    mod5.chunk_bits.map(|bit| bit.col()),
                    acceptance.chunk_bits.map(|bit| bit.col()),
                    "acceptance and mod-5 share candidate bits"
                );
                assert_eq!(
                    acceptance.source_rows.start,
                    canonical.source_rows.end + 26 * chunk,
                    "accept row geometry"
                );
                assert_eq!(
                    acceptance.allocated_columns.start,
                    canonical.bits[0].col() + 66 + 23 * chunk,
                    "accept column geometry"
                );
                assert_eq!(
                    mod5.source_rows.start, acceptance.source_rows.end,
                    "mod-5 follows acceptance"
                );
                assert_eq!(
                    mod5.allocated_columns.start, acceptance.allocated_columns.end,
                    "mod-5 follows acceptance columns"
                );
                assert_eq!(
                    candidate.symbol, mod5.allocated_columns.end,
                    "symbol follows mod-5 columns"
                );
                let cumulative = candidate.symbol + 1;
                let expected_recurrence_prefix = if index == 0 { initial_count } else { final_cumulative };
                if index > 0 {
                    assert_eq!(candidate.prefix, expected_recurrence_prefix, "selection prefix chain");
                }
                assert_acceptance_rows(source, acceptance);
                assert_mod5_rows(source, mod5);
                assert_linear_zero(
                    source,
                    mod5.source_rows.end,
                    vec![
                        (candidate.symbol, F::ONE),
                        (mod5.index.col(), -F::ONE),
                        (Var::ONE.col(), F::from_u64(2)),
                    ],
                    "candidate centered symbol",
                );
                assert_linear_zero(
                    source,
                    mod5.source_rows.end + 1,
                    vec![
                        (cumulative, F::ONE),
                        (expected_recurrence_prefix, -F::ONE),
                        (candidate.accept, -F::ONE),
                    ],
                    "candidate prefix recurrence",
                );
                final_cumulative = cumulative;
            }
            assert_eq!(
                canonical.source_rows.end + LANE_RESIDUAL_ROWS,
                canonical.source_rows.start + LANE_ROWS
            );
        }

        let last_canonical_row = canonical_rows[rho][BLOCK_COUNT - 1][LANES_PER_BLOCK - 1];
        let tail_row = last_canonical_row + LANE_ROWS;
        let first_allocated = final_cumulative + 1;
        tail_rows.push(tail_row);
        tail_first_allocated.push(first_allocated);
        assert_eq!(
            candidates[0].prefix,
            first_allocated + 5,
            "selection zero prefix column"
        );
        assert_ne!(
            initial_count, candidates[0].prefix,
            "sampler and selection zero wires stay distinct"
        );
        selection_zeros.push(candidates[0].prefix);
        selection_zero_rows.push(tail_row + 6);

        let slack = first_allocated;
        for bit in 0..4 {
            assert_bit(source, tail_row + bit, slack + 1 + bit, "acceptance-bound slack bit");
        }
        let mut slack_recomposition = vec![(slack, F::ONE)];
        slack_recomposition.extend(powers_of_two(slack + 1..slack + 5, -F::ONE));
        assert_linear_zero(source, tail_row + 4, slack_recomposition, "acceptance-bound slack");
        assert_linear_zero(
            source,
            tail_row + 5,
            vec![
                (final_cumulative, F::ONE),
                (slack, -F::ONE),
                (Var::ONE.col(), -F::from_u64(OUTPUT_COUNT as u64)),
            ],
            "acceptance lower bound",
        );
        assert_linear_zero(
            source,
            tail_row + 6,
            vec![(first_allocated + 5, F::ONE)],
            "selection initialize",
        );

        for (position, event) in events.iter().enumerate() {
            let row_start = tail_row + 7 + position * 48;
            let column_start = first_allocated + 6 + position * 45;
            assert_eq!(event.one_hot_rows, row_start..row_start + 12, "one-hot row geometry");
            assert_eq!(
                event.product_rows,
                row_start + 12..row_start + 45,
                "product row geometry"
            );
            assert_eq!(event.bind_rows, row_start + 45..row_start + 48, "bind row geometry");
            assert!(
                event
                    .one_hot
                    .iter()
                    .enumerate()
                    .all(|(offset, value)| value.col() == column_start + offset),
                "one-hot columns"
            );
            for (offset, products) in event.products.iter().enumerate() {
                assert_eq!(
                    products.symbol.col(),
                    column_start + 11 + 3 * offset,
                    "symbol-product column"
                );
                assert_eq!(
                    products.accepted.col(),
                    column_start + 12 + 3 * offset,
                    "accept-product column"
                );
                assert_eq!(
                    products.prefix.col(),
                    column_start + 13 + 3 * offset,
                    "prefix-product column"
                );
            }
            assert_eq!(event.output.col(), column_start + 44, "selection output column");
            assert_eq!(
                event.output.col(),
                projection_outputs[rho * OUTPUT_COUNT + position],
                "projection aliases sampler output"
            );
            assert_selection_rows(source, event);
        }
        assert_eq!(
            events.last().expect("selection outputs").bind_rows.end,
            tail_row + TAIL_ROWS,
            "tail row extent"
        );
        let this_offset = events[0].output.col() - first_allocated;
        let this_stride = events[1].output.col() - events[0].output.col();
        assert_eq!(
            output_offset.get_or_insert(this_offset),
            &this_offset,
            "output offset shared"
        );
        assert_eq!(
            output_stride.get_or_insert(this_stride),
            &this_stride,
            "output stride shared"
        );
    }

    SamplerLayout {
        initial_count: affine1(&initial_counts, "initial count"),
        initialization_row: affine1(&initialization_rows, "initialization row"),
        selection_zero: affine1(&selection_zeros, "selection zero"),
        selection_zero_row: affine1(&selection_zero_rows, "selection zero row"),
        field: affine3(&fields, "digest field"),
        bit_start: affine3(&bit_starts, "canonical bit start"),
        canonical_row: affine3(&canonical_rows, "canonical row"),
        tail_first_allocated: affine1(&tail_first_allocated, "tail first allocated"),
        tail_row: affine1(&tail_rows, "tail row"),
        output_offset: output_offset.expect("selection output offset"),
        output_stride: output_stride.expect("selection output stride"),
    }
}

fn render(layout: &SamplerLayout) -> String {
    let mut rendered = String::new();
    rendered.push_str(
        "/-! Generated by `active_pi_rlc_projection_artifacts_match_production_trace`; do not hand-edit. -/\n\n",
    );
    rendered.push_str("namespace Nightstream.Implementation.R1CS.FPrimeRecursivePiRlcChallengeSamplerLayoutData\n\n");
    for (name, value) in [
        ("scalarCount", SCALAR_COUNT),
        ("digestBlockCount", BLOCK_COUNT),
        ("laneCount", LANE_COUNT),
        ("lanesPerBlock", LANES_PER_BLOCK),
        ("chunksPerLane", CHUNKS_PER_LANE),
        ("candidateCount", CANDIDATE_COUNT),
        ("outputCount", OUTPUT_COUNT),
        ("canonicalRows", CANONICAL_ROWS),
        ("laneRows", LANE_ROWS),
        ("tailRows", TAIL_ROWS),
        ("initialCountBase", layout.initial_count.base),
        ("initialCountStride", layout.initial_count.stride),
        ("initializationRowBase", layout.initialization_row.base),
        ("initializationRowStride", layout.initialization_row.stride),
        ("selectionZeroBase", layout.selection_zero.base),
        ("selectionZeroStride", layout.selection_zero.stride),
        ("selectionZeroRowBase", layout.selection_zero_row.base),
        ("selectionZeroRowStride", layout.selection_zero_row.stride),
        ("fieldBase", layout.field.base),
        ("fieldRhoStride", layout.field.rho_stride),
        ("fieldBlockStride", layout.field.block_stride),
        ("fieldLaneStride", layout.field.lane_stride),
        ("bitStartBase", layout.bit_start.base),
        ("bitStartRhoStride", layout.bit_start.rho_stride),
        ("bitStartBlockStride", layout.bit_start.block_stride),
        ("bitStartLaneStride", layout.bit_start.lane_stride),
        ("canonicalRowBase", layout.canonical_row.base),
        ("canonicalRowRhoStride", layout.canonical_row.rho_stride),
        ("canonicalRowBlockStride", layout.canonical_row.block_stride),
        ("canonicalRowLaneStride", layout.canonical_row.lane_stride),
        ("tailFirstAllocatedBase", layout.tail_first_allocated.base),
        ("tailFirstAllocatedStride", layout.tail_first_allocated.stride),
        ("tailRowBase", layout.tail_row.base),
        ("tailRowStride", layout.tail_row.stride),
        ("outputOffset", layout.output_offset),
        ("outputStride", layout.output_stride),
    ] {
        writeln!(rendered, "def {name} : Nat := {value}").expect("render sampler metadata");
    }
    rendered.push_str("\nend Nightstream.Implementation.R1CS.FPrimeRecursivePiRlcChallengeSamplerLayoutData\n");
    rendered
}

pub(super) fn check_generated_artifact(source: &R1csSnapshot, trace: &R1csEncodingTrace) {
    let rendered = render(&extract(source, trace));
    let path = repo_root().join(LEAN_DATA_PATH);
    let committed = fs::read_to_string(&path).unwrap_or_default();
    if committed != rendered {
        let expected = path.with_extension("lean.expected");
        fs::create_dir_all(expected.parent().expect("sampler artifact parent"))
            .expect("create sampler artifact directory");
        fs::write(&expected, &rendered).expect("write expected active sampler layout");
    }
    assert_eq!(committed, rendered, "active sampler layout drifted");
}

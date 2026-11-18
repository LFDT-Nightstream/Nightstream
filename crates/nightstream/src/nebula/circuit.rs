//! `Lifecycle/Nebula/MemoryCircuit.lean`.

use neo_ccs::crypto::poseidon2::{RATE, WIDTH};
use neo_math::F;
use p3_field::PrimeCharacteristicRing;
use std::collections::BTreeSet;

use super::app::Application;
use super::expr::{lower_affine, lower_constraint, Expr};
use super::rows::{pack, Sym, View};
use super::transcript::{chain_tag, header, plan_digest, text, Lane, ETA_TAG, STATE_TAG};
use super::words::{self, Layout};
use super::Geometry;
use crate::application::{permutation, Affine, ApplicationBuilder, ApplicationCircuit, ApplicationError, Variable};

type State = [Affine; WIDTH];

fn zero_state() -> State {
    std::array::from_fn(|_| Affine::constant(F::ZERO))
}

/// `transcriptChunks` (`blockE`, `Hash.inputChunks`).
fn transcript_chunks(blocks: &[Vec<Expr>]) -> Vec<Vec<Affine>> {
    blocks
        .iter()
        .flat_map(|block| {
            let mut framed = vec![Affine::constant(F::from_usize(block.len()))];
            framed.extend(
                block
                    .iter()
                    .map(|word| lower_affine(word).expect("transcript words are affine")),
            );
            framed
                .chunks(RATE)
                .map(<[Affine]>::to_vec)
                .collect::<Vec<_>>()
        })
        .collect()
}

/// `Sponge.circuit` (`absorbE`).
fn absorb(
    builder: &mut ApplicationBuilder,
    mut state: State,
    chunks: &[Vec<Affine>],
) -> Result<State, ApplicationError> {
    for chunk in chunks {
        for (lane, value) in state.iter_mut().enumerate() {
            let word = chunk
                .get(lane)
                .cloned()
                .unwrap_or_else(|| Affine::constant(F::ZERO));
            *value = value.clone() + word;
        }
        state = permutation(builder, state)?;
    }
    Ok(state)
}

fn lane(state: &State, k: usize) -> Expr {
    Expr::Var(
        state[k]
            .as_variable()
            .expect("a permutation output lane is one variable"),
    )
}

fn text_words(tag: &str) -> Vec<Expr> {
    text(tag).into_iter().map(Expr::Const).collect()
}

pub(super) struct Phase {
    pub name: &'static str,
    pub rows: usize,
    pub local_words: usize,
}

pub(super) fn memory_circuit<A: Application>(geometry: &Geometry, app: &A) -> Result<ApplicationCircuit, crate::Error> {
    memory_circuit_phases(geometry, app).map(|(circuit, _)| circuit)
}

pub(super) fn memory_circuit_phases<A: Application>(
    geometry: &Geometry,
    app: &A,
) -> Result<(ApplicationCircuit, Vec<Phase>), crate::Error> {
    build(geometry, app, true)
}

#[cfg(test)]
pub(super) fn memory_circuit_unshared<A: Application>(
    geometry: &Geometry,
    app: &A,
) -> Result<ApplicationCircuit, crate::Error> {
    build(geometry, app, false).map(|(circuit, _)| circuit)
}

fn build<A: Application>(
    geometry: &Geometry,
    app: &A,
    share_reads: bool,
) -> Result<(ApplicationCircuit, Vec<Phase>), crate::Error> {
    let layout = Layout::new(geometry, app.width(), app.aux());
    let mut builder = ApplicationBuilder::new(layout.count())?;
    let source_wires: Vec<Variable> = builder.private_inputs().to_vec();
    let digest = plan_digest(geometry);
    let (ops_header, mem_header) = (header(Lane::Ops, &digest), header(Lane::Mem, &digest));
    let headers = std::array::from_fn(|i| if i < 4 { ops_header[i] } else { mem_header[i % 4] });
    let source_sym = Sym::new(geometry, layout, &source_wires, headers);
    let source_view = View::new(&source_sym);
    let mut rows = app.rows(&source_view)?;
    let readable = source_view.variables();
    for row in &rows {
        if let Some(variable) = row.variable_outside(&readable) {
            return Err(ApplicationError::VariableOutOfScope(variable.index()).into());
        }
    }
    let shared = if share_reads {
        super::readonly::ReadOnlyValues::checked(&rows, geometry, &layout, &source_wires)
    } else {
        super::readonly::ReadOnlyValues::default()
    };
    for row in &mut rows {
        shared.rename(row);
    }
    let mut hints = app.poseidon2_hints(&source_view)?;
    for lane in hints.iter_mut().flat_map(|hint| &mut hint.lanes) {
        shared.rename_expr(lane);
    }
    let wires = shared.wires(&source_wires);
    let sym = Sym::new(geometry, layout, &wires, headers);
    let run = |count: usize, index: &dyn Fn(usize) -> usize| (0..count).map(|k| sym.word(index(k))).collect::<Vec<_>>();

    let state_blocks = |word: fn(&Layout, usize) -> usize, carry: usize| {
        vec![
            text_words(STATE_TAG),
            run(layout.width(), &|k| word(&layout, k)),
            run(39, &|k| carry + k),
        ]
    };
    let chain_blocks = |lane: Lane, previous: usize, lanes: Vec<Expr>| {
        let mut seen = vec![sym.word(words::IDX_EFF)];
        seen.extend(run(4, &|k| words::SEEN_PREV + previous + k));
        vec![text_words(chain_tag(lane)), seen, pack(lanes)]
    };
    let ops_lanes = (0..geometry.b_ops).flat_map(|j| sym.op_lane(j)).collect();
    let scan_lanes = |start: &dyn Fn(usize) -> usize| {
        (0..geometry.b_scan)
            .flat_map(|j| sym.scan_lane(start(j)))
            .collect::<Vec<_>>()
    };
    let mut eta_words = run(4, &|k| words::PROPOSAL + k);
    eta_words.extend(run(4, &|k| words::CARRY_IN + 35 + k));
    eta_words.extend(run(4, &|k| words::PROPOSAL + 4 + k));
    let eta_blocks = vec![
        text_words(ETA_TAG),
        digest.iter().copied().map(Expr::Const).collect(),
        vec![sym.c_in(2)],
        eta_words,
    ];

    let absorbing = |builder: &mut ApplicationBuilder, blocks: &[Vec<Expr>]| {
        absorb(builder, zero_state(), &transcript_chunks(blocks))
    };
    let mut phases = Vec::new();
    let mut mark = |builder: &ApplicationBuilder, name: &'static str| {
        let (rows, local_words) = phases.iter().fold((0, 0), |(rows, local), phase: &Phase| {
            (rows + phase.rows, local + phase.local_words)
        });
        phases.push(Phase {
            name,
            rows: builder.row_count() - rows,
            local_words: builder.local_count() - local_words,
        });
    };
    let state_in = absorbing(&mut builder, &state_blocks(Layout::app_in, words::CARRY_IN))?;
    mark(&builder, "State digest sponge (input)");
    let state_out = absorbing(&mut builder, &state_blocks(Layout::app_out, words::CARRY_OUT))?;
    mark(&builder, "State digest sponge (output)");
    let chain_ops = absorbing(&mut builder, &chain_blocks(Lane::Ops, 0, ops_lanes))?;
    mark(&builder, "Operation record chain sponge");
    let chain_initial = absorbing(
        &mut builder,
        &chain_blocks(Lane::Mem, 4, scan_lanes(&|j| sym.initial(j))),
    )?;
    mark(&builder, "Initial-scan record chain sponge");
    let chain_final = absorbing(
        &mut builder,
        &chain_blocks(Lane::Mem, 8, scan_lanes(&|j| sym.final_(j))),
    )?;
    mark(&builder, "Final-scan record chain sponge");
    let eta = absorbing(&mut builder, &eta_blocks)?;
    mark(&builder, "η transcript sponge");
    let squeeze1 = absorb(&mut builder, eta.clone(), &[Vec::new()])?;
    let squeeze2 = absorb(&mut builder, squeeze1.clone(), &[Vec::new()])?;
    let squeeze3 = absorb(&mut builder, squeeze2.clone(), &[Vec::new()])?;
    mark(&builder, "η squeezes");

    let input = builder.input_state();
    let output = builder.output_state();
    for k in 0..4 {
        lower_constraint(&mut builder, &(lane(&state_in, k) - Expr::Var(input[k])))?;
    }
    let mut output_rows = [0; 4];
    for (k, row) in output_rows.iter_mut().enumerate() {
        *row = lower_constraint(&mut builder, &(lane(&state_out, k) - Expr::Var(output[k])))?;
    }
    for (chain, first) in [(&chain_ops, 23), (&chain_initial, 27), (&chain_final, 31)] {
        for k in 0..4 {
            lower_constraint(&mut builder, &(lane(chain, k) - sym.c_out(first + k)))?;
        }
    }
    for (k, state) in [&eta, &squeeze1, &squeeze2, &squeeze3]
        .into_iter()
        .enumerate()
    {
        lower_constraint(&mut builder, &(lane(state, 0) - sym.word(words::ETA_FRESH + k)))?;
    }
    mark(&builder, "Sponge output bindings");
    let mut checked_bits = BTreeSet::new();
    for (name, rows) in sym.memory_row_groups() {
        for row in &rows {
            if !shared.is_empty()
                && ((name == "Bit rows" && super::readonly::repeated_bit(row, &mut checked_bits))
                    || (name == "Operation slot rows" && super::readonly::zero_row(row)))
            {
                continue;
            }
            lower_constraint(&mut builder, row)?;
        }
        mark(&builder, name);
    }
    let view = View::new(&sym);
    let first_app_row = builder.row_count();
    let aliases = super::poseidon::lower(&mut builder, rows, hints, &view, app.aux())?;
    mark(&builder, "Application rows");
    let circuit = builder.finish_with_output_rows(output_rows)?;
    super::poseidon::check_prefix(&circuit, first_app_row, &aliases)?;
    Ok((circuit, phases))
}

#[cfg(test)]
#[path = "../../tests/nebula/parity.rs"]
mod parity;

#[cfg(test)]
#[path = "../../tests/nebula/cost.rs"]
mod cost;

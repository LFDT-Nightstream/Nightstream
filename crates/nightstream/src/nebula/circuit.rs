//! The memory circuit of a geometry and an application, built with
//! `ApplicationBuilder` in the order of `Lifecycle/Nebula/MemoryCircuit.lean`:
//! nine sponge children (the two state digests, the three record chains, the
//! `η` transcript and its three squeezes), then the hash rows, the memory
//! rows, and the application's rows. `tests/nebula/parity.rs` requires the
//! machine's circuit to equal each Lean package's, row for row. It does not
//! own witness values. Its output rows are the `state_out` hash rows, which
//! are not the last rows.

use neo_ccs::crypto::poseidon2_goldilocks::{round_constants, Poseidon2RoundConstants, RATE, WIDTH};
use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::Goldilocks as F;

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

/// `transcriptChunks`: each block is its length, then its words (`blockE`),
/// in rate chunks (`Hash.inputChunks`).
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

/// One sponge child (`Sponge.circuit`): each chunk adds its words to the
/// lanes (`absorbE`) and permutes.
fn absorb(
    builder: &mut ApplicationBuilder,
    constants: &Poseidon2RoundConstants,
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
        state = permutation(builder, state, constants)?;
    }
    Ok(state)
}

/// Lane `k` of a sponge output: a fresh variable of its last permutation.
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

/// The rows and local variables that one named phase of `memory_circuit`
/// adds, in build order.
pub(super) struct Phase {
    pub name: &'static str,
    pub rows: usize,
    pub local_words: usize,
}

/// The Rust-built circuit of a memory geometry and an application. It reads no
/// program, so every program of the geometry has the same circuit.
pub(super) fn memory_circuit<A: Application>(
    geometry: &Geometry,
    app: &A,
) -> Result<ApplicationCircuit, ApplicationError> {
    memory_circuit_phases(geometry, app).map(|(circuit, _)| circuit)
}

/// `memory_circuit` with the rows and local variables of each phase.
pub(super) fn memory_circuit_phases<A: Application>(
    geometry: &Geometry,
    app: &A,
) -> Result<(ApplicationCircuit, Vec<Phase>), ApplicationError> {
    let layout = Layout::new(geometry, app.width(), app.aux());
    let mut builder = ApplicationBuilder::new(layout.count())?;
    let wires: Vec<Variable> = builder.private_inputs().to_vec();
    let digest = plan_digest(geometry);
    let (ops_header, mem_header) = (header(Lane::Ops, &digest), header(Lane::Mem, &digest));
    let headers = std::array::from_fn(|i| if i < 4 { ops_header[i] } else { mem_header[i % 4] });
    let sym = Sym::new(geometry, layout, &wires, headers);
    let constants = round_constants();
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
        absorb(builder, &constants, zero_state(), &transcript_chunks(blocks))
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
    let squeeze1 = absorb(&mut builder, &constants, eta.clone(), &[Vec::new()])?;
    let squeeze2 = absorb(&mut builder, &constants, squeeze1.clone(), &[Vec::new()])?;
    let squeeze3 = absorb(&mut builder, &constants, squeeze2.clone(), &[Vec::new()])?;
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
    for (name, rows) in sym.memory_row_groups() {
        for row in &rows {
            lower_constraint(&mut builder, row)?;
        }
        mark(&builder, name);
    }
    let view = View::new(&sym);
    let readable = view.variables();
    for row in &app.rows(&view) {
        if let Some(variable) = row.variable_outside(&readable) {
            return Err(ApplicationError::VariableOutOfScope(variable.index()));
        }
        lower_constraint(&mut builder, row)?;
    }
    mark(&builder, "Application rows");
    Ok((builder.finish_with_output_rows(output_rows)?, phases))
}

#[cfg(test)]
#[path = "../../tests/nebula/parity.rs"]
mod parity;

#[cfg(test)]
#[path = "../../tests/nebula/cost.rs"]
mod cost;

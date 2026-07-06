//! Ignored perf snapshot for the bit-encoded Fibonacci direct-CCS path.
//!
//! Run with:
//!
//! ```text
//! cargo test -p neo-fold-clean --release --test perf_fibonacci_bits -- --ignored --nocapture
//! ```

use std::time::Instant;

use neo_ccs::matrix::Mat as NeoMat;
use neo_math::{D, F};
use p3_field::PrimeCharacteristicRing;

use neo_fold_clean::config::{EXTENSION_SAFETY_MARGIN_BITS, MIN_EFFECTIVE_LAMBDA, R1CS_PROFILE};
use neo_fold_clean::engine::decider::synthesize_statement_r1cs;
use neo_fold_clean::frontends::direct_ccs::{self, R1cs};
use neo_fold_clean::paper::construction2::{self, FoldProof, ProofState, State};
use neo_fold_clean::paper::digest::{
    digest32_as_fields, initial_boundary_digest, public_trace_seed_digest, state_x_out_digest_with_mode,
    structure_digest, AccumulatorHandle, StateXOutDigestMode,
};
use neo_fold_clean::paper::f_prime::r1cs::{encode_f_prime_public_input, F_PRIME_PUBLIC_INPUT_LEN};
use neo_fold_clean::paper::nifs::NifsProof;
use neo_fold_clean::{
    extend, finish_uncompressed_with_audit, prove, verify_uncompressed_audit, CcsInstance, FoldSchedule,
};

const LIMBS: usize = 12;
const DECIDER_LIMBS: usize = 3;
const VALUE_COUNT: usize = 18;
const DECIDER_VALUE_COUNT: usize = 4;
const DECIDER_VALUE_COUNT_ENV: &str = "NEO_FOLD_FIB_DECIDER_VALUES";
const ROWS_PER_STEP: usize = 1;
const ONE: usize = 0;

#[derive(Default)]
struct ProofShape {
    nofold_steps: usize,
    recursive_steps: usize,
    nifs_folds: usize,
    pi_ccs_outputs_total: usize,
    pi_dec_children_total: usize,
    pi_rlc_parents: usize,
    fe_rounds_total: usize,
    nc_rounds_total: usize,
    max_fe_round_width: usize,
    max_nc_round_width: usize,
}

struct FoldRow {
    label: String,
    pi_ccs_outputs: usize,
    pi_dec_children: usize,
    fe_rounds: usize,
    nc_rounds: usize,
    max_fe_round_width: usize,
    max_nc_round_width: usize,
    /// Wall-clock for the prover-side work that produced this fold.
    /// For step rows: the time of the matching `extend` call. For the
    /// `final` row: the `finish_uncompressed` time (which is where
    /// `prove_final_fold` runs).
    prove_ms: f64,
}

fn a_bit(j: usize) -> usize {
    1 + j
}

fn b_bit(j: usize) -> usize {
    1 + LIMBS + j
}

fn c_bit(j: usize) -> usize {
    1 + 2 * LIMBS + j
}

fn carry_bit(j: usize) -> usize {
    debug_assert!(j < LIMBS - 1);
    1 + 3 * LIMBS + j
}

fn public_width() -> usize {
    1 + 3 * LIMBS
}

fn private_width() -> usize {
    LIMBS - 1
}

fn circuit_width() -> usize {
    public_width() + private_width()
}

fn fibonacci_addition_r1cs() -> R1cs {
    assert!(circuit_width() <= D);

    let rows = 1 + 3 * LIMBS + private_width() + LIMBS;
    let mut a = NeoMat::zero(rows, D, F::default());
    let mut b = NeoMat::zero(rows, D, F::default());
    let mut c = NeoMat::zero(rows, D, F::default());
    let mut row = 0;

    a[(row, ONE)] = F::ONE;
    b[(row, ONE)] = F::ONE;
    c[(row, ONE)] = F::ONE;
    row += 1;

    for idx in public_bit_indices().chain(carry_bit_indices()) {
        add_boolean_constraint(row, idx, &mut a, &mut b);
        row += 1;
    }

    for j in 0..LIMBS {
        a[(row, a_bit(j))] = F::ONE;
        a[(row, b_bit(j))] = F::ONE;
        if j > 0 {
            a[(row, carry_bit(j - 1))] = F::ONE;
        }
        b[(row, ONE)] = F::ONE;
        c[(row, c_bit(j))] = F::ONE;
        if j + 1 < LIMBS {
            c[(row, carry_bit(j))] = F::from_u64(2);
        }
        row += 1;
    }

    debug_assert_eq!(row, rows);
    R1cs {
        a,
        b,
        c,
        m_in: public_width(),
    }
}

fn add_boolean_constraint(row: usize, idx: usize, a: &mut NeoMat<F>, b: &mut NeoMat<F>) {
    a[(row, idx)] = F::ONE;
    b[(row, ONE)] = F::ONE;
    b[(row, idx)] = -F::ONE;
}

fn public_bit_indices() -> impl Iterator<Item = usize> {
    (0..LIMBS)
        .map(a_bit)
        .chain((0..LIMBS).map(b_bit))
        .chain((0..LIMBS).map(c_bit))
}

fn carry_bit_indices() -> impl Iterator<Item = usize> {
    (0..LIMBS - 1).map(carry_bit)
}

fn fibonacci_values(count: usize) -> Vec<u64> {
    assert!(count >= 2);
    let mut out = Vec::with_capacity(count);
    out.push(1);
    out.push(1);
    while out.len() < count {
        let n = out[out.len() - 1] + out[out.len() - 2];
        out.push(n);
    }
    out
}

fn fibonacci_values_mod(count: usize, limbs: usize) -> Vec<u64> {
    assert!(count >= 2);
    let modulus = 1u64 << limbs;
    let mut out = Vec::with_capacity(count);
    out.push(1);
    out.push(1);
    while out.len() < count {
        let n = (out[out.len() - 1] + out[out.len() - 2]) % modulus;
        out.push(n);
    }
    out
}

fn fibonacci_value_count_from_env(default: usize) -> usize {
    match std::env::var(DECIDER_VALUE_COUNT_ENV) {
        Ok(raw) => {
            let count = raw
                .parse::<usize>()
                .unwrap_or_else(|_| panic!("{DECIDER_VALUE_COUNT_ENV} must be a positive integer, got {raw:?}"));
            assert!(
                count >= 4,
                "{DECIDER_VALUE_COUNT_ENV} must be at least 4 to emit one recursive fold"
            );
            count
        }
        Err(std::env::VarError::NotPresent) => default,
        Err(err) => panic!("failed to read {DECIDER_VALUE_COUNT_ENV}: {err}"),
    }
}

fn f_prime_active_assignment_capacity() -> usize {
    F_PRIME_PUBLIC_INPUT_LEN.div_ceil(D) * D
}

fn assert_fibonacci_values_fit_limbs(values: &[u64], limbs: usize) {
    let max = values.last().copied().unwrap_or(0);
    assert!(
        max < (1u64 << limbs),
        "{DECIDER_VALUE_COUNT_ENV}={} produces max Fibonacci value {max}, which does not fit DECIDER_LIMBS={limbs}. \
         Increase DECIDER_LIMBS only if the F'-active witness capacity still holds: \
         F_PRIME_PUBLIC_INPUT_LEN + 4*limbs <= {}.",
        values.len(),
        f_prime_active_assignment_capacity()
    );
}

fn fibonacci_transition_assignment(prev: u64, curr: u64, next: u64) -> Vec<F> {
    assert_eq!(prev + curr, next);
    assert!(next < (1 << LIMBS));

    let mut z = vec![F::ZERO; D];
    z[ONE] = F::ONE;
    for j in 0..LIMBS {
        z[a_bit(j)] = bit(prev, j);
        z[b_bit(j)] = bit(curr, j);
        z[c_bit(j)] = bit(next, j);
    }

    let mut carry = 0;
    for j in 0..LIMBS {
        let sum = bit_u64(prev, j) + bit_u64(curr, j) + carry;
        carry = sum >> 1;
        if j + 1 < LIMBS {
            z[carry_bit(j)] = F::from_u64(carry);
        } else {
            assert_eq!(carry, 0);
        }
    }
    z
}

fn f_prime_fib_a_bit(j: usize) -> usize {
    F_PRIME_PUBLIC_INPUT_LEN + j
}

fn f_prime_fib_b_bit(j: usize) -> usize {
    F_PRIME_PUBLIC_INPUT_LEN + DECIDER_LIMBS + j
}

fn f_prime_fib_c_bit(j: usize) -> usize {
    F_PRIME_PUBLIC_INPUT_LEN + 2 * DECIDER_LIMBS + j
}

fn f_prime_fib_carry_bit(j: usize) -> usize {
    debug_assert!(j < DECIDER_LIMBS);
    F_PRIME_PUBLIC_INPUT_LEN + 3 * DECIDER_LIMBS + j
}

fn f_prime_fib_private_width() -> usize {
    3 * DECIDER_LIMBS + DECIDER_LIMBS
}

fn f_prime_fib_width() -> usize {
    F_PRIME_PUBLIC_INPUT_LEN + f_prime_fib_private_width()
}

fn f_prime_fibonacci_addition_r1cs() -> R1cs {
    let rows = 1 + 3 * DECIDER_LIMBS + DECIDER_LIMBS + DECIDER_LIMBS;
    let cols = f_prime_fib_width();
    assert!(
        cols <= f_prime_active_assignment_capacity(),
        "F'-linked Fibonacci witness width {cols} exceeds active packed capacity {}",
        f_prime_active_assignment_capacity()
    );
    let mut a = NeoMat::zero(rows, cols, F::default());
    let mut b = NeoMat::zero(rows, cols, F::default());
    let mut c = NeoMat::zero(rows, cols, F::default());
    let mut row = 0;

    a[(row, ONE)] = F::ONE;
    b[(row, ONE)] = F::ONE;
    c[(row, ONE)] = F::ONE;
    row += 1;

    for j in 0..DECIDER_LIMBS {
        add_boolean_constraint(row, f_prime_fib_a_bit(j), &mut a, &mut b);
        row += 1;
        add_boolean_constraint(row, f_prime_fib_b_bit(j), &mut a, &mut b);
        row += 1;
        add_boolean_constraint(row, f_prime_fib_c_bit(j), &mut a, &mut b);
        row += 1;
    }
    for j in 0..DECIDER_LIMBS {
        add_boolean_constraint(row, f_prime_fib_carry_bit(j), &mut a, &mut b);
        row += 1;
    }

    for j in 0..DECIDER_LIMBS {
        a[(row, f_prime_fib_a_bit(j))] = F::ONE;
        a[(row, f_prime_fib_b_bit(j))] = F::ONE;
        if j > 0 {
            a[(row, f_prime_fib_carry_bit(j - 1))] = F::ONE;
        }
        b[(row, ONE)] = F::ONE;
        c[(row, f_prime_fib_c_bit(j))] = F::ONE;
        c[(row, f_prime_fib_carry_bit(j))] = F::from_u64(2);
        row += 1;
    }

    debug_assert_eq!(row, rows);
    R1cs {
        a,
        b,
        c,
        m_in: F_PRIME_PUBLIC_INPUT_LEN,
    }
}

fn f_prime_fibonacci_transition_assignment(prev: u64, curr: u64, next: u64, x_out_target: [F; 4]) -> Vec<F> {
    let modulus = 1u64 << DECIDER_LIMBS;
    assert_eq!((prev + curr) % modulus, next);
    assert!(next < (1 << DECIDER_LIMBS));

    let mut z = encode_f_prime_public_input(x_out_target);
    z.resize(f_prime_fib_width(), F::ZERO);

    for j in 0..DECIDER_LIMBS {
        z[f_prime_fib_a_bit(j)] = bit(prev, j);
        z[f_prime_fib_b_bit(j)] = bit(curr, j);
        z[f_prime_fib_c_bit(j)] = bit(next, j);
    }

    let mut carry = 0;
    for j in 0..DECIDER_LIMBS {
        let sum = bit_u64(prev, j) + bit_u64(curr, j) + carry;
        carry = sum >> 1;
        z[f_prime_fib_carry_bit(j)] = F::from_u64(carry);
    }
    z
}

fn f_prime_base_state(prep: &neo_fold_clean::lifecycle::Preprocessing) -> State {
    let structure = structure_digest(prep.structure());
    let z_0 = initial_boundary_digest(&structure, prep.public_input_len);
    let public_trace = public_trace_seed_digest(&structure);
    let acc_digest = AccumulatorHandle::empty().digest();
    State::base(z_0, public_trace, acc_digest, acc_digest)
}

fn f_prime_state_x_out(prep: &neo_fold_clean::lifecycle::Preprocessing, state: &State) -> [F; 4] {
    let mode = match prep.semantic_state_mode() {
        neo_fold_clean::paper::construction2::SemanticStateMode::Stateless => StateXOutDigestMode::Stateless,
        neo_fold_clean::paper::construction2::SemanticStateMode::Stateful => StateXOutDigestMode::Stateful,
    };
    digest32_as_fields(state_x_out_digest_with_mode(
        mode,
        prep.vk.digest(),
        &structure_digest(prep.structure()),
        state.chunk_count,
        state.step_count,
        state.z_0,
        state.z_i,
        state.pc,
        state.acc_digest,
        state.acc_digest,
        state.public_trace,
        None,
    ))
}

fn f_prime_peek_next_state(
    prep: &neo_fold_clean::lifecycle::Preprocessing,
    state: &State,
    batch: CcsInstance,
) -> State {
    let (next, _) = construction2::step(
        &prep.params,
        prep.structure(),
        prep.optimized_cache(),
        prep.structure_digest(),
        &prep.log,
        prep.mix_rhos_commits(),
        prep.combine_b_pows(),
        &prep.vk,
        state.clone(),
        vec![batch],
    )
    .expect("peek next F' state");
    next
}

fn f_prime_fibonacci_instance(
    prep: &neo_fold_clean::lifecycle::Preprocessing,
    r1cs: &R1cs,
    prev: u64,
    curr: u64,
    next: u64,
    x_out_target: [F; 4],
) -> CcsInstance {
    let z = f_prime_fibonacci_transition_assignment(prev, curr, next, x_out_target);
    direct_ccs::build_instance(prep, r1cs, &z).expect("build F'-linked Fibonacci instance")
}

fn bit(value: u64, idx: usize) -> F {
    F::from_u64(bit_u64(value, idx))
}

fn bit_u64(value: u64, idx: usize) -> u64 {
    (value >> idx) & 1
}

fn r1cs_nnz(r1cs: &R1cs) -> usize {
    matrix_nnz(&r1cs.a) + matrix_nnz(&r1cs.b) + matrix_nnz(&r1cs.c)
}

fn matrix_nnz(mat: &NeoMat<F>) -> usize {
    mat.as_slice().iter().filter(|&&v| v != F::ZERO).count()
}

fn ccs_matrix_nnz(r1cs: &R1cs) -> usize {
    let structure = r1cs.to_structure();
    structure
        .matrices
        .iter()
        .map(|matrix| {
            matrix
                .as_csc()
                .map(|csc| csc.vals.len())
                .unwrap_or_else(|| matrix.rows())
        })
        .sum()
}

fn proof_shape(
    audit: &neo_fold_clean::lifecycle::UncompressedAudit,
    step_prove_ms: &[f64],
    finish_ms: f64,
) -> (ProofShape, Vec<FoldRow>) {
    let mut shape = ProofShape::default();
    let mut rows = Vec::new();

    for (idx, step) in audit.steps.iter().enumerate() {
        match &step.fold {
            FoldProof::NoFold => {
                shape.nofold_steps += 1;
            }
            FoldProof::Recursive(nifs) => {
                shape.recursive_steps += 1;
                let pms = step_prove_ms.get(idx).copied().unwrap_or(0.0);
                record_nifs(&mut shape, &mut rows, format!("step[{idx}]"), nifs, pms);
            }
        }
    }
    if let Some(final_fold) = &audit.proof.final_fold {
        record_nifs(&mut shape, &mut rows, "final".to_owned(), &final_fold.nifs, finish_ms);
    }

    (shape, rows)
}

fn record_nifs(shape: &mut ProofShape, rows: &mut Vec<FoldRow>, label: String, nifs: &NifsProof, prove_ms: f64) {
    let fe_rounds = nifs.pi_ccs.sumcheck.sumcheck_rounds.len();
    let nc_rounds = nifs.pi_ccs.sumcheck.sumcheck_rounds_nc.len();
    let max_fe_round_width = nifs
        .pi_ccs
        .sumcheck
        .sumcheck_rounds
        .iter()
        .map(Vec::len)
        .max()
        .unwrap_or(0);
    let max_nc_round_width = nifs
        .pi_ccs
        .sumcheck
        .sumcheck_rounds_nc
        .iter()
        .map(Vec::len)
        .max()
        .unwrap_or(0);

    shape.nifs_folds += 1;
    shape.pi_ccs_outputs_total += nifs.pi_ccs.outputs.len();
    shape.pi_dec_children_total += nifs.pi_dec.children.len();
    shape.pi_rlc_parents += 1;
    shape.fe_rounds_total += fe_rounds;
    shape.nc_rounds_total += nc_rounds;
    shape.max_fe_round_width = shape.max_fe_round_width.max(max_fe_round_width);
    shape.max_nc_round_width = shape.max_nc_round_width.max(max_nc_round_width);

    rows.push(FoldRow {
        label,
        pi_ccs_outputs: nifs.pi_ccs.outputs.len(),
        pi_dec_children: nifs.pi_dec.children.len(),
        fe_rounds,
        nc_rounds,
        max_fe_round_width,
        max_nc_round_width,
        prove_ms,
    });
}

#[test]
#[ignore]
fn fibonacci_bits_perf_snapshot() {
    let wall_start = Instant::now();

    let r1cs_start = Instant::now();
    let r1cs = fibonacci_addition_r1cs();
    let r1cs_ms = r1cs_start.elapsed().as_secs_f64() * 1000.0;

    let preprocess_start = Instant::now();
    let prep = direct_ccs::preprocess_seeded(&r1cs, /* seed = */ 0xF1B0).expect("preprocess");
    let preprocess_ms = preprocess_start.elapsed().as_secs_f64() * 1000.0;

    let values = fibonacci_values(VALUE_COUNT);
    let transitions = values.len() - 2;

    let build_start = Instant::now();
    let instances = values
        .windows(3)
        .map(|w| {
            let z = fibonacci_transition_assignment(w[0], w[1], w[2]);
            direct_ccs::build_instance(&prep, &r1cs, &z).expect("build Fibonacci transition instance")
        })
        .collect::<Vec<_>>();
    let build_instances_ms = build_start.elapsed().as_secs_f64() * 1000.0;

    let partition_start = Instant::now();
    let batches = FoldSchedule::RowsPerStep(ROWS_PER_STEP)
        .partition(instances)
        .expect("partition");
    let partition_ms = partition_start.elapsed().as_secs_f64() * 1000.0;

    // Prove via explicit extend loop so we can time each step individually.
    // `prove(&prep, std::iter::empty())` returns the initial in-flight proof
    // (Construction-2 base case); each batch is folded via one `extend`.
    let prove_start = Instant::now();
    let mut proof = prove(&prep, std::iter::empty::<Vec<CcsInstance>>()).expect("init in-flight proof");
    let mut step_prove_ms: Vec<f64> = Vec::with_capacity(batches.len());
    for batch in batches {
        let step_start = Instant::now();
        proof = extend(&prep, proof, batch).expect("extend");
        step_prove_ms.push(step_start.elapsed().as_secs_f64() * 1000.0);
    }
    let prove_ms = prove_start.elapsed().as_secs_f64() * 1000.0;

    let finish_start = Instant::now();
    let finished = finish_uncompressed_with_audit(&prep, proof).expect("finish_uncompressed_with_audit");
    let finish_ms = finish_start.elapsed().as_secs_f64() * 1000.0;

    let verify_start = Instant::now();
    verify_uncompressed_audit(&prep, &finished).expect("verify_uncompressed_audit");
    let verify_ms = verify_start.elapsed().as_secs_f64() * 1000.0;

    let total_ms = wall_start.elapsed().as_secs_f64() * 1000.0;
    let (shape, fold_rows) = proof_shape(&finished, &step_prove_ms, finish_ms);

    let final_running_claims = finished
        .proof
        .state
        .proof
        .running()
        .map(|running| running.claims.len())
        .unwrap_or(0);
    let final_latest_instances = match &finished.proof.state.proof {
        ProofState::Initial => 0,
        ProofState::Active { latest, .. } => latest.instances.len(),
    };
    let prove_finish_ms = prove_ms + finish_ms;
    let transitions_per_sec = transitions as f64 / (prove_finish_ms / 1000.0).max(f64::EPSILON);
    let folds_per_sec = shape.nifs_folds as f64 / (prove_finish_ms / 1000.0).max(f64::EPSILON);

    print_report(
        &r1cs,
        &prep,
        &finished,
        &shape,
        &fold_rows,
        &values,
        transitions,
        final_running_claims,
        final_latest_instances,
        Timings {
            r1cs_ms,
            preprocess_ms,
            build_instances_ms,
            partition_ms,
            prove_ms,
            finish_ms,
            verify_ms,
            total_ms,
        },
        transitions_per_sec,
        folds_per_sec,
    );
}

#[test]
#[ignore]
fn fibonacci_decider_r1cs_shape_snapshot() {
    // Full-history audit circuit, not the final IVC terminal decider.
    //
    // This intentionally replays every lifecycle/F' step into one R1CS so
    // we can inspect the cost of that fallback shape. A proper IVC
    // terminal decider must prove the folded accumulator and should not
    // grow linearly with the number of historical steps.
    let wall_start = Instant::now();

    let r1cs_start = Instant::now();
    let r1cs = f_prime_fibonacci_addition_r1cs();
    let r1cs_ms = r1cs_start.elapsed().as_secs_f64() * 1000.0;

    let preprocess_start = Instant::now();
    let prep = direct_ccs::preprocess_seeded(&r1cs, /* seed = */ 0xF1D3).expect("preprocess");
    let preprocess_ms = preprocess_start.elapsed().as_secs_f64() * 1000.0;

    let value_count = fibonacci_value_count_from_env(DECIDER_VALUE_COUNT);
    let values = fibonacci_values_mod(value_count, DECIDER_LIMBS);
    assert_fibonacci_values_fit_limbs(&values, DECIDER_LIMBS);
    let transitions = values.len() - 2;

    let prove_start = Instant::now();
    let mut state = f_prime_base_state(&prep);
    let mut steps = Vec::with_capacity(transitions);
    let mut public_batches: Vec<Vec<neo_fold_clean::paper::relations::CcsClaim>> = Vec::with_capacity(transitions);

    for w in values.windows(3) {
        let dummy = f_prime_fibonacci_instance(&prep, &r1cs, w[0], w[1], w[2], [F::ZERO; 4]);
        let predicted = f_prime_peek_next_state(&prep, &state, dummy);
        let target_x_out = f_prime_state_x_out(&prep, &predicted);
        let batch = f_prime_fibonacci_instance(&prep, &r1cs, w[0], w[1], w[2], target_x_out);
        let public_batch = vec![batch.claim.clone()];

        let (next_state, step_proof) = construction2::step(
            &prep.params,
            prep.structure(),
            prep.optimized_cache(),
            prep.structure_digest(),
            &prep.log,
            prep.mix_rhos_commits(),
            prep.combine_b_pows(),
            &prep.vk,
            state,
            vec![batch],
        )
        .expect("F'-linked Fibonacci step");

        steps.push(step_proof);
        public_batches.push(public_batch);
        state = next_state;
    }
    let prove_ms = prove_start.elapsed().as_secs_f64() * 1000.0;

    let finish_start = Instant::now();
    let in_flight = neo_fold_clean::UncompressedAudit {
        proof: neo_fold_clean::Uncompressed {
            state,
            final_fold: None,
        },
        steps,
        public_batches,
    };
    let finished = finish_uncompressed_with_audit(&prep, in_flight).expect("finish_uncompressed_with_audit");
    let finish_ms = finish_start.elapsed().as_secs_f64() * 1000.0;

    let verify_start = Instant::now();
    verify_uncompressed_audit(&prep, &finished).expect("verify_uncompressed_audit");
    let verify_ms = verify_start.elapsed().as_secs_f64() * 1000.0;

    let synth_start = Instant::now();
    let statement = neo_fold_clean::build_decider_statement(&prep, &finished);
    let synth = synthesize_statement_r1cs(&prep, &statement).expect("synthesize decider R1CS");
    let synth_ms = synth_start.elapsed().as_secs_f64() * 1000.0;

    assert!(
        synth.is_self_sufficient_relation(),
        "decider R1CS is not self-sufficient/satisfied (first bad row: {:?})",
        synth.builder.first_unsatisfied_row()
    );

    let total_ms = wall_start.elapsed().as_secs_f64() * 1000.0;

    println!();
    println!("======================================================================");
    println!("  Fibonacci full-history audit R1CS shape snapshot");
    println!("======================================================================");
    section("Fibonacci trace");
    kv("limbs", DECIDER_LIMBS);
    kv("modulus", 1usize << DECIDER_LIMBS);
    kv("env-selected values", value_count);
    kv_with_note(
        "values",
        values.len(),
        &format!("max = {}", values.last().copied().unwrap_or(0)),
    );
    kv("transitions", transitions);

    section("Application R1CS");
    kv("public vars (F' link)", F_PRIME_PUBLIC_INPUT_LEN);
    kv("private Fibonacci vars", f_prime_fib_private_width());
    kv("total vars", f_prime_fib_width());
    kv("constraints", r1cs.n());
    kv("R1CS nnz", r1cs_nnz(&r1cs));
    kv("CCS nnz", ccs_matrix_nnz(&r1cs));

    section("Full-history audit R1CS");
    kv("rows", synth.builder.rows());
    kv("cols", synth.builder.cols());
    kv("witness len", synth.builder.witness().len());
    kv("recursive F' steps", synth.recursive_step_count);
    kv("cross-step links", synth.cross_step_links);
    kv("CE continuity links", synth.accumulator_claim_links);
    kv("terminal latest link", synth.terminal_latest_link as usize);
    kv("terminal fold emitted", synth.terminal_fold_emitted as usize);
    kv("public-image pins", synth.public_image_pins);
    kv("self-sufficient", synth.is_self_sufficient_relation() as usize);

    section("Timings (ms)");
    kv_ms("build app R1CS", r1cs_ms);
    kv_ms("preprocess", preprocess_ms);
    kv_ms("fold/prove steps", prove_ms);
    kv_ms("finish", finish_ms);
    kv_ms("verify uncompressed", verify_ms);
    kv_ms("synthesize decider R1CS", synth_ms);
    kv_ms("total", total_ms);
    println!("======================================================================");
}

#[allow(clippy::too_many_arguments)]
fn print_report(
    r1cs: &R1cs,
    prep: &neo_fold_clean::lifecycle::Preprocessing,
    finished: &neo_fold_clean::lifecycle::UncompressedAudit,
    shape: &ProofShape,
    fold_rows: &[FoldRow],
    values: &[u64],
    transitions: usize,
    final_running_claims: usize,
    final_latest_instances: usize,
    t: Timings,
    transitions_per_sec: f64,
    folds_per_sec: f64,
) {
    const RULE: &str = "======================================================================";
    const THIN: &str = "──────────────────────────────────────────────────────────────────────";

    println!();
    println!("{RULE}");
    println!("  Fibonacci-bits direct-CCS perf snapshot");
    println!("{RULE}");

    section("Fibonacci trace");
    kv("limbs", LIMBS);
    kv_with_note(
        "values",
        values.len(),
        &format!("max = {}", values.last().copied().unwrap_or(0)),
    );
    kv("transitions", transitions);
    kv("rows per step", ROWS_PER_STEP);

    section("Circuit (R1CS → CCS embedding)");
    kv("public vars", public_width());
    kv("private vars", private_width());
    kv_with_note("total vars", circuit_width(), &format!("padded to {}", r1cs.m()));
    kv("constraints", r1cs.n());
    kv("R1CS nnz", r1cs_nnz(r1cs));
    kv("CCS nnz", ccs_matrix_nnz(r1cs));

    section("CCS structure");
    kv("n (rows)", prep.structure().n);
    kv("m (vars)", prep.structure().m);
    kv("t (matrices)", prep.structure().t());
    kv("f degree", prep.structure().max_degree());

    section("Protocol params");
    kv("profile", R1CS_PROFILE);
    kv("lambda", prep.params.lambda());
    kv("min lambda", MIN_EFFECTIVE_LAMBDA);
    kv("safety margin bits", EXTENSION_SAFETY_MARGIN_BITS);
    kv("extension s", prep.params.extension_degree());
    kv("b", prep.params.b());
    kv("k_rho", prep.params.k_rho());
    kv("B = b^k_rho", prep.params.big_b());
    kv("T", prep.params.T());
    kv("kappa", prep.params.kappa());

    section("Proof shape");
    kv("steps", finished.steps.len());
    kv("no-fold steps", shape.nofold_steps);
    kv("recursive folds", shape.recursive_steps);
    kv("final folds", usize::from(finished.proof.final_fold.is_some()));
    kv("NIFS folds (total)", shape.nifs_folds);
    kv("final running claims", final_running_claims);
    kv("final latest claims", final_latest_instances);

    section("NIFS internals");
    kv("Π_RLC parents", shape.pi_rlc_parents);
    kv("Π_CCS outputs", shape.pi_ccs_outputs_total);
    kv("Π_DEC children", shape.pi_dec_children_total);
    kv_with_note(
        "FE rounds",
        shape.fe_rounds_total,
        &format!("max round width {}", shape.max_fe_round_width),
    );
    kv_with_note(
        "NC rounds",
        shape.nc_rounds_total,
        &format!("max round width {}", shape.max_nc_round_width),
    );

    section("Timing (ms)");
    kv_ms("r1cs build", t.r1cs_ms);
    kv_ms("preprocess", t.preprocess_ms);
    kv_ms("build instances", t.build_instances_ms);
    kv_ms("partition", t.partition_ms);
    kv_ms("prove", t.prove_ms);
    kv_ms("finish", t.finish_ms);
    kv_ms("verify", t.verify_ms);
    println!("    {THIN}");
    kv_ms("total", t.total_ms);

    section("Throughput (prove + finish)");
    kv_rate("transitions/s", transitions_per_sec);
    kv_rate("folds/s", folds_per_sec);

    section("Per-fold shape and prover timing");
    println!(
        "    {:>4}  {:<10}  {:>7}  {:>8}  {:>8}  {:>8}  {:>6}  {:>6}  {:>9}",
        "idx", "source", "ccs_out", "dec_chld", "fe_rounds", "nc_rounds", "fe_max", "nc_max", "prove_ms",
    );
    println!(
        "    {:>4}  {:<10}  {:>7}  {:>8}  {:>8}  {:>8}  {:>6}  {:>6}  {:>9}",
        "----", "----------", "-------", "--------", "---------", "---------", "------", "------", "---------",
    );
    let mut prove_ms_sum = 0.0;
    for (idx, row) in fold_rows.iter().enumerate() {
        prove_ms_sum += row.prove_ms;
        println!(
            "    {:>4}  {:<10}  {:>7}  {:>8}  {:>8}  {:>8}  {:>6}  {:>6}  {:>9.3}",
            idx,
            row.label,
            row.pi_ccs_outputs,
            row.pi_dec_children,
            row.fe_rounds,
            row.nc_rounds,
            row.max_fe_round_width,
            row.max_nc_round_width,
            row.prove_ms,
        );
    }
    println!(
        "    {:>4}  {:<10}  {:>7}  {:>8}  {:>8}  {:>8}  {:>6}  {:>6}  {:>9.3}",
        "", "Σ", "", "", "", "", "", "", prove_ms_sum,
    );

    println!();
    println!("{RULE}");
    println!();
}

struct Timings {
    r1cs_ms: f64,
    preprocess_ms: f64,
    build_instances_ms: f64,
    partition_ms: f64,
    prove_ms: f64,
    finish_ms: f64,
    verify_ms: f64,
    total_ms: f64,
}

fn section(title: &str) {
    println!();
    println!("  {title}");
}

fn kv<V: std::fmt::Display>(label: &str, value: V) {
    println!("    {:<22} {:>12}", label, value.to_string());
}

fn kv_with_note<V: std::fmt::Display>(label: &str, value: V, note: &str) {
    println!("    {:<22} {:>12}  ({note})", label, value.to_string());
}

fn kv_ms(label: &str, value: f64) {
    println!("    {:<22} {:>12.3}", label, value);
}

fn kv_rate(label: &str, value: f64) {
    println!("    {:<22} {:>12.2}", label, value);
}

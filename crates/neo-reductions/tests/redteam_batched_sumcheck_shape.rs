use std::panic::{catch_unwind, AssertUnwindSafe};

use neo_math::K;
use neo_reductions::sumcheck::{
    run_batched_sumcheck_prover, run_sumcheck_prover, verify_batched_sumcheck_rounds,
    verify_sumcheck_rounds_poseidon_v3, BatchedClaim, RoundOracle,
};
use neo_transcript::{Poseidon2Transcript, Transcript};
use p3_field::PrimeCharacteristicRing;

struct DegreeZeroConstantOracle {
    rounds: usize,
    value: K,
}

impl RoundOracle for DegreeZeroConstantOracle {
    fn evals_at(&mut self, points: &[K]) -> Vec<K> {
        vec![self.value; points.len()]
    }

    fn num_rounds(&self) -> usize {
        self.rounds
    }

    fn degree_bound(&self) -> usize {
        0
    }

    fn fold(&mut self, _r: K) {}
}

#[test]
fn batched_sumcheck_verifier_rejects_parallel_array_mismatch_without_panicking() {
    let rounds = vec![vec![vec![K::ZERO]]];
    let result = catch_unwind(AssertUnwindSafe(|| {
        let mut transcript = Poseidon2Transcript::new(b"sumcheck/batched/redteam/panic");
        verify_batched_sumcheck_rounds(&mut transcript, &rounds, &[], &[], &[0])
    }));

    let (_, _, accepted) = result.expect("malformed batched sumcheck must return false, not panic");
    assert!(!accepted, "mismatched batched metadata must be rejected");
}

#[test]
fn batched_sumcheck_verifier_rejects_zero_round_claim_without_metadata() {
    let rounds = vec![Vec::new()];
    let mut transcript = Poseidon2Transcript::new(b"sumcheck/batched/redteam/zero-round");
    let (_, final_values, accepted) = verify_batched_sumcheck_rounds(&mut transcript, &rounds, &[], &[], &[]);

    assert!(
        !accepted && final_values.len() == rounds.len(),
        "verifier accepted a declared claim while omitting its sum, label, degree bound, and final value"
    );
}

#[test]
fn sumcheck_verifier_accepts_finite_round_under_max_degree_bound() {
    let rounds = vec![vec![K::ZERO]];
    let mut transcript = Poseidon2Transcript::new(b"redteam/max-degree");
    let (_, _, accepted) = verify_sumcheck_rounds_poseidon_v3(&mut transcript, usize::MAX, K::ZERO, &rounds);

    assert!(
        accepted,
        "a constant polynomial is within every representable usize degree bound"
    );
}

#[test]
fn sumcheck_provers_accept_degree_zero_constant_oracles_without_panicking() {
    let value = K::ONE;
    let initial_sum = value + value;

    let single = catch_unwind(AssertUnwindSafe(|| {
        let mut transcript = Poseidon2Transcript::new(b"redteam/sumcheck/degree-zero/single");
        let mut oracle = DegreeZeroConstantOracle { rounds: 1, value };
        run_sumcheck_prover(&mut transcript, &mut oracle, initial_sum)
    }));

    let batched = catch_unwind(AssertUnwindSafe(|| {
        let mut transcript = Poseidon2Transcript::new(b"redteam/sumcheck/degree-zero/batched");
        let mut oracle = DegreeZeroConstantOracle { rounds: 1, value };
        let mut claims = [BatchedClaim {
            oracle: &mut oracle,
            claimed_sum: initial_sum,
            label: b"degree-zero",
        }];
        run_batched_sumcheck_prover(&mut transcript, &mut claims)
    }));

    assert!(
        single.is_ok() && batched.is_ok(),
        "completeness failure: a valid degree-zero constant sumcheck oracle panicked in the public prover (single={}, batched={})",
        single.is_ok(),
        batched.is_ok()
    );

    let (single_rounds, _) = single
        .expect("single prover panic checked above")
        .expect("degree-zero single sumcheck must prove");
    let (_, batched_results) = batched
        .expect("batched prover panic checked above")
        .expect("degree-zero batched sumcheck must prove");
    assert_eq!(single_rounds, vec![vec![value]]);
    assert_eq!(batched_results[0].round_polys, vec![vec![value]]);
}

//! Retained red-team regression for Bellpepper's implicit constant-one wire.

#[path = "../support/mod.rs"]
mod support;

use bellpepper_core::{Circuit, ConstraintSystem, SynthesisError};
use ff::Field as _;
use neo_fold_clean::engine::transcript::Transcript;
use neo_fold_clean::frontends::bellpepper::{synthesize_to_ccs, BellpepperGoldilocks};
use neo_fold_clean::paper::construction2::RunningInstance;
use neo_fold_clean::paper::nifs;
use neo_fold_clean::{config, preprocess, CcsInstance};
use neo_math::F;
use p3_field::PrimeCharacteristicRing;

/// For application-public `out = 0`, the Bellpepper relation is impossible:
/// `a * ONE = out` forces `a = 0`, while `a * a = ONE` requires `a² = 1`.
/// It becomes satisfiable only if a malicious prover can replace Bellpepper's
/// implicit `ONE` input with zero.
struct ImpossibleAtZeroCircuit;

impl Circuit<BellpepperGoldilocks> for ImpossibleAtZeroCircuit {
    fn synthesize<CS: ConstraintSystem<BellpepperGoldilocks>>(self, cs: &mut CS) -> Result<(), SynthesisError> {
        let out = cs.alloc_input(|| "out", || Ok(BellpepperGoldilocks::ZERO))?;
        let a = cs.alloc(|| "a", || Ok(BellpepperGoldilocks::ZERO))?;
        cs.enforce(|| "a equals out", |lc| lc + a, |lc| lc + CS::one(), |lc| lc + out);
        cs.enforce(|| "a squared equals one", |lc| lc + a, |lc| lc + a, |lc| lc + CS::one());
        Ok(())
    }
}

#[test]
fn nifs_rejects_bellpepper_proof_with_zero_constant_wire() {
    let circuit = synthesize_to_ccs(ImpossibleAtZeroCircuit).expect("synthesize impossible Bellpepper relation");
    assert_eq!(
        circuit.public_inputs(),
        &[F::ONE, F::ZERO],
        "Bellpepper synthesis starts with the canonical implicit ONE"
    );
    assert!(
        circuit
            .sparse_r1cs
            .is_satisfied_by(&circuit.assignment)
            .is_err(),
        "the public statement out=0 is impossible under canonical Bellpepper semantics"
    );
    let malicious_assignment = [F::ZERO, F::ZERO, F::ZERO];
    assert!(
        circuit
            .sparse_r1cs
            .is_satisfied_by(&malicious_assignment)
            .is_ok(),
        "zeroing the implicit constant is exactly what makes the false relation satisfiable"
    );

    let params = config::ccs_params(
        circuit.structure.n,
        circuit.structure.m,
        circuit.structure.t(),
        circuit.structure.max_degree(),
    )
    .expect("shape-specific params");
    support::install_ajtai_module(&params, &circuit.structure);
    let public_input_len = circuit.public_input_len();
    let prep = preprocess(params, circuit.structure, Some(public_input_len)).expect("Bellpepper CCS preprocessing");

    // Keep the application-public output `out = 0` and the private witness
    // `a = 0`; change only Bellpepper's implicit constant wire from one to
    // zero. The converted CCS currently has no row pinning that syntax wire.
    let fresh =
        CcsInstance::from_low_norm_assignment(&prep.params, &prep.log, prep.structure(), &malicious_assignment, 2)
            .expect("malicious low-norm assignment");
    let fresh_claims = vec![fresh.claim.clone()];

    let mut prover_transcript = Transcript::session();
    let (_next_running, proof) = nifs::prove(
        &mut prover_transcript,
        &prep.params,
        prep.structure(),
        prep.optimized_cache(),
        &prep.log,
        None,
        prep.mix_rhos_commits(),
        prep.combine_b_pows(),
        vec![fresh],
        &RunningInstance::default(),
    )
    .expect("current NIFS.P proves the false Bellpepper statement");

    let mut verifier_transcript = Transcript::session();
    let result = nifs::verify(
        &mut verifier_transcript,
        &prep.params,
        prep.structure(),
        prep.optimized_cache(),
        prep.mix_rhos_commits(),
        prep.combine_b_pows(),
        &fresh_claims,
        &RunningInstance::default(),
        &proof,
    );

    assert!(
        result.is_err(),
        "soundness failure: NIFS.V accepted an impossible Bellpepper circuit after the prover changed its implicit constant-one wire to zero"
    );
}

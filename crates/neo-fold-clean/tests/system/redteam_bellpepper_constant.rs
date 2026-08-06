//! Bellpepper adapter regression for its implicit constant-one wire.

#[path = "../support/mod.rs"]
mod support;

use bellpepper_core::{Circuit, ConstraintSystem, SynthesisError};
use ff::Field as _;
use neo_fold_clean::frontends::bellpepper::{synthesize_to_ccs, BellpepperFrontendError, BellpepperGoldilocks};
use neo_fold_clean::{config, preprocess};
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
fn bellpepper_frontend_rejects_zero_constant_wire() {
    let circuit = synthesize_to_ccs(ImpossibleAtZeroCircuit).expect("synthesize impossible Bellpepper relation");
    assert_eq!(
        &circuit.public_inputs()[..2],
        &[F::ONE, F::ZERO],
        "Bellpepper synthesis starts with the canonical implicit ONE"
    );
    assert!(
        circuit.public_inputs()[2..]
            .iter()
            .all(|&value| value == F::ZERO),
        "the rest of the complete public ring must be zero padding"
    );
    assert!(
        circuit
            .sparse_r1cs
            .is_satisfied_by(&circuit.assignment)
            .is_err(),
        "the public statement out=0 is impossible under canonical Bellpepper semantics"
    );
    let mut malicious = circuit.clone();
    malicious.assignment[0] = F::ZERO;
    assert!(
        malicious
            .sparse_r1cs
            .is_satisfied_by(&malicious.assignment)
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
    let prep =
        preprocess(params, circuit.structure.clone(), Some(public_input_len)).expect("Bellpepper CCS preprocessing");

    assert!(
        matches!(
            malicious.build_instance(&prep),
            Err(BellpepperFrontendError::NonCanonicalConstant)
        ),
        "the Bellpepper adapter accepted a zero implicit constant-one wire"
    );

    let mut noncanonical_padding = circuit.clone();
    noncanonical_padding.assignment[circuit.shape.inputs] = F::ONE;
    assert!(
        matches!(
            noncanonical_padding.build_instance(&prep),
            Err(BellpepperFrontendError::NonCanonicalPublicPadding { index })
                if index == circuit.shape.inputs
        ),
        "the Bellpepper adapter accepted nonzero public-ring completion"
    );
}

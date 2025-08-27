use neo_ccs::{mv_poly, CcsStructure};
use neo_commit::{AjtaiCommitter, TOY_PARAMS};
use neo_fields::{from_base, ExtF, F};
use neo_fold::{verify_open, EvalInstance};
use p3_field::PrimeCharacteristicRing;
use p3_matrix::dense::RowMajorMatrix;
use neo_ring::RingElement;
use neo_modint::{ModInt, Coeff};

fn one_by_one_structure_with<FN: Fn(&[ExtF]) -> ExtF + Send + Sync + 'static>(
    f: FN,
    deg: usize,
) -> CcsStructure {
    // Single matrix 1x1 so mats.len() == 1 and inputs.len() == 1 for f
    let mats = vec![RowMajorMatrix::new(vec![F::ONE], 1)];
    let f = mv_poly(f, deg);
    CcsStructure::new(mats, f)
}

#[test]
fn verify_open_accepts_valid_zero_constraint() {
    // f == 0; u=0, e_eval arbitrary → f(ys)=0 == u*e_eval^2
    let structure = one_by_one_structure_with(|_| ExtF::ZERO, 1);
    let committer = AjtaiCommitter::setup_unchecked(TOY_PARAMS);

    let eval = EvalInstance {
        commitment: vec![],
        r: vec![from_base(F::ONE)],      // random point (not used by this check)
        ys: vec![from_base(F::from_u64(7))], // arbitrary value (norm check skipped)
        u: from_base(F::ZERO),
        e_eval: from_base(F::ONE),
        norm_bound: TOY_PARAMS.norm_bound,
        opening_proof: None,
    };

    assert!(
        verify_open(&structure, &committer, &eval, TOY_PARAMS.max_blind_norm),
        "valid zero-constraint opening should pass"
    );
}

#[test]
fn verify_open_rejects_mismatch_between_f_ys_and_rhs() {
    // f(inputs) = inputs[0]; set ys[0]=1, but u=0,e=1 ⇒ RHS=0; must fail.
    let structure = one_by_one_structure_with(|ins| ins[0], 1);
    let committer = AjtaiCommitter::setup_unchecked(TOY_PARAMS);

    let eval = EvalInstance {
        commitment: vec![],
        r: vec![from_base(F::ONE)],
        ys: vec![from_base(F::ONE)],
        u: from_base(F::ZERO),
        e_eval: from_base(F::ONE),
        norm_bound: TOY_PARAMS.norm_bound,
        opening_proof: None,
    };

    assert!(
        !verify_open(&structure, &committer, &eval, TOY_PARAMS.max_blind_norm),
        "binding mismatch must be rejected"
    );
}

#[test]
fn verify_open_rejects_large_norm_e_eval() {
    // f == 0 with huge e_eval should be rejected by norm bound
    let structure = one_by_one_structure_with(|_| ExtF::ZERO, 1);
    let committer = AjtaiCommitter::setup_unchecked(TOY_PARAMS);

    // Make e_eval exceed the toy max_blind_norm (=64)
    let big = from_base(F::from_u64(1000));
    let eval = EvalInstance {
        commitment: vec![],
        r: vec![from_base(F::ZERO)],
        ys: vec![from_base(F::ZERO)],
        u: from_base(F::ZERO),
        e_eval: big,
        norm_bound: TOY_PARAMS.norm_bound,
        opening_proof: None,
    };

    assert!(
        !verify_open(&structure, &committer, &eval, TOY_PARAMS.max_blind_norm),
        "e_eval above max_blind_norm must be rejected"
    );
}

#[test]
fn verify_open_rejects_large_norm_u() {
    // f == 0 with huge u should be rejected by norm bound
    let structure = one_by_one_structure_with(|_| ExtF::ZERO, 1);
    let committer = AjtaiCommitter::setup_unchecked(TOY_PARAMS);

    // Make u exceed the toy max_blind_norm (=64)
    let big = from_base(F::from_u64(1000));
    let eval = EvalInstance {
        commitment: vec![],
        r: vec![from_base(F::ZERO)],
        ys: vec![from_base(F::ZERO)],
        u: big,
        e_eval: from_base(F::ONE),
        norm_bound: TOY_PARAMS.norm_bound,
        opening_proof: None,
    };

    assert!(
        !verify_open(&structure, &committer, &eval, TOY_PARAMS.max_blind_norm),
        "u above max_blind_norm must be rejected"
    );
}

#[test]
fn verify_open_requires_proof_when_commitment_present() {
    // If a commitment is present but no opening proof is carried in EvalInstance,
    // verification must fail (binding to the commitment cannot be established).
    let structure = one_by_one_structure_with(|_| ExtF::ZERO, 1);
    let committer = AjtaiCommitter::setup_unchecked(TOY_PARAMS);

    let mut eval = EvalInstance {
        commitment: vec![],
        r: vec![from_base(F::ZERO)],
        ys: vec![from_base(F::ZERO)],
        u: from_base(F::ZERO),
        e_eval: from_base(F::ZERO),
        norm_bound: TOY_PARAMS.norm_bound,
        opening_proof: None,
    };
    // Make commitment non-empty
    let zero_ring = RingElement::from_scalar(ModInt::zero(), committer.params().n);
    eval.commitment = vec![zero_ring];

    assert!(
        !verify_open(&structure, &committer, &eval, TOY_PARAMS.max_blind_norm),
        "non-empty commitment without an opening proof must be rejected"
    );
}

#[test]
#[ignore = "Non-linear constraints unsupported; protocol assumes multilinear (deg<=1) for soundness - see docs"]
fn verify_open_accepts_nonlinear_constraint() {
    // For non-linear f (deg > 1), verify_open should skip the point-binding check
    // and rely on the sumcheck proof that was already verified
    let structure = one_by_one_structure_with(|ins| ins[0] * ins[0], 2); // Quadratic: degree 2
    let committer = AjtaiCommitter::setup_unchecked(TOY_PARAMS);

    let eval = EvalInstance {
        commitment: vec![],
        r: vec![from_base(F::ONE)],
        ys: vec![from_base(F::from_u64(5))], // f(ys) = 5^2 = 25
        u: from_base(F::ZERO),               // u * e_eval^2 = 0 * 1^2 = 0
        e_eval: from_base(F::ONE),
        norm_bound: TOY_PARAMS.norm_bound,
        opening_proof: None,
    };

    // This should PASS because for non-linear f, we skip the point-binding check
    // The sumcheck proof (which we assume was already verified) is the real security
    assert!(
        verify_open(&structure, &committer, &eval, TOY_PARAMS.max_blind_norm),
        "non-linear constraint should skip point-binding check and pass"
    );
}

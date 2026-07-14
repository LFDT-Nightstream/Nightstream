//! Retained red-team regression for non-canonical Ajtai commitment encodings.

#[path = "../support/mod.rs"]
mod support;

use std::panic::{catch_unwind, AssertUnwindSafe};

use neo_ajtai::Commitment;
use neo_ccs::traits::SModuleHomomorphism;
use neo_ccs::Mat;
use neo_fold_clean::engine::r1cs_circuit::R1csBuilder;
use neo_fold_clean::paper::reductions::pi_dec_circuit::{alloc_dec_inputs, enforce_dec_v_strict};
use neo_fold_clean::paper::reductions::pi_rlc_circuit::{
    alloc_rlc_commitment_inputs, enforce_rlc_commitment_combination,
};
use neo_fold_clean::paper::relations::{CcsClaim, CcsWitness};
use neo_fold_clean::{CcsInstance, CeClaim, Preprocessing};
use neo_math::{D, F};
use neo_transcript::{Poseidon2Transcript, Transcript};
use p3_field::PrimeCharacteristicRing;

fn lifecycle_accepts(prep: &Preprocessing, fresh: CcsInstance) -> bool {
    let Ok(audit) = neo_fold_clean::prove(prep, [vec![fresh]]) else {
        return false;
    };
    let Ok(proof) = neo_fold_clean::finish_uncompressed(prep, audit) else {
        return false;
    };
    neo_fold_clean::verify_uncompressed(prep, &proof).is_ok()
}

/// A commitment declares `(d, kappa)` but can carry either more or fewer than
/// `d * kappa` coordinates. Pi_CCS absorbs the malformed length and data, but
/// the release Ajtai RLC action processes only full declared columns through
/// `zip`. Both wire images must fail before they can become prover-controlled
/// Fiat-Shamir nonces outside the commitment codomain.
#[test]
fn lifecycle_rejects_noncanonical_fresh_commitment_lengths() {
    let prep = support::toy_preprocessing();
    let canonical = support::toy_instance(&prep, 0);
    let canonical_len = canonical.claim.c.d * canonical.claim.c.kappa;
    assert_eq!(canonical.claim.c.data.len(), canonical_len);

    let mut overlong = canonical.clone();
    overlong
        .claim
        .c
        .data
        .extend(std::iter::repeat_n(F::from_u64(7), D));
    let mut short = canonical;
    short.claim.c.data.truncate(canonical_len - D);

    let overlong_accepted = lifecycle_accepts(&prep, overlong);
    let short_accepted = lifecycle_accepts(&prep, short);
    assert!(
        !overlong_accepted && !short_accepted,
        "soundness failure: the public terminal verifier accepted noncanonical fresh commitment lengths (overlong={overlong_accepted}, short={short_accepted})"
    );
}

/// Strict recursive Π_DEC must validate commitment metadata against the
/// verifier's parameter profile with checked arithmetic.  Otherwise a huge
/// `kappa` can wrap `D * kappa` to zero and make an empty commitment look
/// canonically shaped inside the circuit.
#[cfg(target_pointer_width = "64")]
#[test]
fn recursive_pi_dec_rejects_wrapped_empty_commitment_shape() {
    let prep = support::toy_preprocessing();
    let wrapped_kappa = 1usize << 63;
    assert_eq!(D.wrapping_mul(wrapped_kappa), 0);
    assert_ne!(wrapped_kappa, prep.params.kappa() as usize);

    let malformed = CeClaim {
        c: Commitment {
            d: D,
            kappa: wrapped_kappa,
            data: Vec::new(),
        },
        X: Mat::zero(D, 0, F::ZERO),
        r: Vec::new(),
        s_col: Vec::new(),
        y_ring: Vec::new(),
        ct: Vec::new(),
        aux_openings: Vec::new(),
        y_zcol: Vec::new(),
        m_in: 0,
        fold_digest: [0u8; 32],
        c_step_coords: Vec::new(),
        u_offset: 0,
        u_len: 0,
    };
    let children = vec![malformed.clone(); prep.params.k_rho() as usize];
    let mut builder = R1csBuilder::new();
    let wires = alloc_dec_inputs(&mut builder, &malformed, &children);
    let result = enforce_dec_v_strict(&mut builder, &prep.params, &wires);
    let accepted = result.is_ok() && builder.is_satisfied();

    assert!(
        !accepted,
        "recursive-verifier soundness failure: strict Π_DEC accepted an empty commitment declaring kappa={wrapped_kappa}, outside params.kappa={} after D*kappa wrapped to zero",
        prep.params.kappa(),
    );
}

/// The standalone recursive Π_RLC commitment relation must not infer its
/// verifier-owned codomain from the prover's first commitment. In particular,
/// `kappa = 0` turns the complete commitment check into a vacuous zero-row
/// circuit and lies outside every installed production parameter profile.
#[test]
fn recursive_pi_rlc_rejects_empty_zero_kappa_commitment_shape() {
    let empty = Commitment {
        d: D,
        kappa: 0,
        data: Vec::new(),
    };
    let rhos = [[F::ZERO; D]];
    let mut builder = R1csBuilder::new();
    let wires = alloc_rlc_commitment_inputs(&mut builder, &rhos, core::slice::from_ref(&empty), &empty)
        .expect("malformed commitment should receive a circuit verdict");
    enforce_rlc_commitment_combination(&mut builder, &wires);

    assert!(
        !builder.is_satisfied(),
        "recursive-verifier soundness failure: Π_RLC accepted a vacuous kappa=0 commitment relation"
    );
}

/// A logical width-one SuperNeo witness occupies one full `D`-lane ring
/// column. Lanes 1..D are padding outside `z` and must be canonical zero for
/// a fresh CCS opening. Otherwise they are committed but omitted from the
/// fresh claim and from the NC digit table, giving the prover unbounded
/// coordinates that are not in Definition 12's low-norm witness.
#[test]
fn lifecycle_rejects_nonzero_out_of_norm_fresh_witness_padding_lane() {
    let prep = support::toy_preprocessing();
    let mut z = Mat::zero(D, 1, F::ZERO);
    z[(1, 0)] = F::from_u64(prep.params.b() as u64);
    let fresh = CcsInstance {
        claim: CcsClaim {
            c: prep.log.commit(&z),
            x: vec![F::ZERO],
            m_in: 1,
        },
        witness: CcsWitness { w: Vec::new(), Z: z },
    };

    assert!(
        !lifecycle_accepts(&prep, fresh),
        "soundness failure: the public terminal verifier accepted a fresh commitment with an out-of-norm nonzero lane outside logical z"
    );
}

/// `RotRho` is the typed boundary for a Π_RLC challenge sampled from the
/// configured strong set. A valid cyclotomic rotation matrix is not enough:
/// its defining coefficients must also lie in the configured challenge
/// alphabet, otherwise the advertised expansion factor and extraction
/// argument no longer apply.
#[test]
fn rot_rho_rejects_rotation_outside_configured_strong_set() {
    const OUT_OF_SET_COEFFICIENT: u64 = 1_000;

    let params = neo_fold_clean::Params::production();
    assert!(
        OUT_OF_SET_COEFFICIENT > params.T() as u64,
        "fixture must exceed the configured expansion bound"
    );
    let mut rho = Mat::zero(D, D, F::ZERO);
    for index in 0..D {
        rho[(index, index)] = F::from_u64(OUT_OF_SET_COEFFICIENT);
    }

    let checked = neo_reductions::common::RotRho::new_checked(params.inner(), rho.clone());
    let wire = bincode::serialize(&rho).expect("serialize raw rotation matrix");
    let deserialized = bincode::deserialize::<neo_reductions::common::RotRho>(&wire);

    assert!(
        checked.is_err() && deserialized.is_err(),
        "strong-set validation failure: RotRho accepted multiplication by {OUT_OF_SET_COEFFICIENT}, outside the configured [-2,2] alphabet and above T={}, through checked construction={} or derived deserialization={}",
        params.T(),
        checked.is_ok(),
        deserialized.is_ok(),
    );
}

/// The paper's strong sampling set needs actual entropy and distinct alphabet
/// elements. A duplicate alphabet silently collapses every sampled challenge,
/// while a singleton alphabet must be rejected as input rather than reaching
/// the power-of-two sampler's zero-bit division.
#[test]
fn rot_rho_sampler_rejects_degenerate_alphabets_without_panicking() {
    const DUPLICATE_ZERO: &[i8] = &[0, 0];
    const SINGLE_ZERO: &[i8] = &[0];

    let params = neo_fold_clean::Params::production();
    let profile = neo_reductions::common::RotRing::goldilocks();
    let duplicate_ring = neo_reductions::common::RotRing {
        phi_coeffs: profile.phi_coeffs,
        alphabet: DUPLICATE_ZERO,
        binv_floor: profile.binv_floor,
    };
    let mut duplicate_transcript = Poseidon2Transcript::new(b"redteam/duplicate-alphabet");
    let duplicate =
        neo_reductions::common::sample_rot_rhos_n_typed(&mut duplicate_transcript, params.inner(), &duplicate_ring, 1);

    let singleton_ring = neo_reductions::common::RotRing {
        phi_coeffs: profile.phi_coeffs,
        alphabet: SINGLE_ZERO,
        binv_floor: profile.binv_floor,
    };
    let singleton = catch_unwind(AssertUnwindSafe(|| {
        let mut transcript = Poseidon2Transcript::new(b"redteam/singleton-alphabet");
        neo_reductions::common::sample_rot_rhos_n_typed(&mut transcript, params.inner(), &singleton_ring, 1)
    }));

    let duplicate_accepted = duplicate.is_ok();
    let singleton_rejected_without_panic = matches!(singleton, Ok(Err(_)));
    assert!(
        !duplicate_accepted && singleton_rejected_without_panic,
        "strong-set validation failure: duplicate zero alphabet accepted={duplicate_accepted}, singleton zero alphabet rejected without panic={singleton_rejected_without_panic}"
    );
}

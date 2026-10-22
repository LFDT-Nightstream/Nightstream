use neo_ajtai::{commit_row_major, setup as ajtai_setup};
use neo_ccs::{CcsClaim, CcsStructure, CcsWitness, Mat, SparsePoly};
use neo_math::{D, F, K};
use neo_params::NeoParams;
use neo_reductions::{pi_ccs_prove, pi_ccs_verify};
use neo_transcript::Poseidon2Transcript;
use p3_field::PrimeCharacteristicRing;
use rand_chacha::rand_core::SeedableRng;

mod zero_running;

fn identity_left(n: usize, m: usize) -> Mat<F> {
    let mut mat = Mat::zero(n, m, F::ZERO);
    for i in 0..n.min(m) {
        mat.set(i, i, F::ONE);
    }
    mat
}

fn zero_poly(t: usize) -> SparsePoly<F> {
    SparsePoly::new(t, Vec::new())
}

fn build_fixture(
    n: usize,
    m: usize,
) -> (
    NeoParams,
    CcsStructure<F>,
    CcsClaim<neo_ajtai::Commitment, F>,
    CcsWitness<F>,
    Poseidon2Transcript,
) {
    let params = NeoParams::goldilocks_auto_rectangular_r1cs_ccs(n, n).expect("params");
    let s = CcsStructure::new(vec![identity_left(n, m)], zero_poly(1)).expect("ccs");

    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(123);
    let pp = ajtai_setup(&mut rng, D, params.kappa as usize, m / D).expect("ajtai setup");
    let l = pp;

    let z = Mat::from_row_major(D, m / D, vec![F::ZERO; D * (m / D)]);
    let c = commit_row_major(&l, &z);
    let mcs_inst = CcsClaim {
        adv: None,
        c,
        x: vec![],
        m_in: 0,
    };
    let mcs_wit = CcsWitness { Z: z };

    let tr = Poseidon2Transcript::new_v1_2();
    (params, s, mcs_inst, mcs_wit, tr)
}

#[test]
fn padded_row_identity_rejects_an_extra_sumcheck_round() {
    let (params, s, mcs_inst, mcs_wit, mut tr_p) = build_fixture(4, D);

    let (running, running_witnesses) = zero_running::zero_running(&params, &s, 1, mcs_inst.m_in);
    let (out_me, mut proof) = pi_ccs_prove(
        &mut tr_p,
        &params,
        &s,
        core::slice::from_ref(&mcs_inst),
        core::slice::from_ref(&mcs_wit),
        &running,
        &running_witnesses,
    )
    .expect("prove");

    proof.sumcheck_rounds.push(vec![K::ZERO]);

    let mut tr_v = Poseidon2Transcript::new_v1_2();
    let res = pi_ccs_verify(
        &mut tr_v,
        &params,
        &s,
        core::slice::from_ref(&mcs_inst),
        &running,
        &out_me,
        &proof,
    );
    assert!(res.is_err(), "selected verifier accepted an extra SumCheck round");
}

#[test]
fn padded_row_identity_uses_one_joint_row_cube() {
    let (params, s, mcs_inst, mcs_wit, mut tr_p) = build_fixture(4, D);
    let assignment_width = neo_reductions::common::superneo_carrier_width(s.m);
    let variables =
        s.n.max(assignment_width)
            .next_power_of_two()
            .max(2)
            .trailing_zeros() as usize;

    let (running, running_witnesses) = zero_running::zero_running(&params, &s, 1, mcs_inst.m_in);
    let (out_me, proof) = pi_ccs_prove(
        &mut tr_p,
        &params,
        &s,
        core::slice::from_ref(&mcs_inst),
        core::slice::from_ref(&mcs_wit),
        &running,
        &running_witnesses,
    )
    .expect("prove");

    assert_eq!(proof.sumcheck_rounds.len(), variables);
    assert!(!out_me.is_empty());
    assert_eq!(out_me[0].r.len(), variables);

    let mut tr_v = Poseidon2Transcript::new_v1_2();
    let ok = pi_ccs_verify(
        &mut tr_v,
        &params,
        &s,
        core::slice::from_ref(&mcs_inst),
        &running,
        &out_me,
        &proof,
    )
    .expect("verify should not error");
    assert!(ok);
}

#[test]
fn padded_row_identity_verify_rejects_eval_k_mutation() {
    let (params, s, mcs_inst, mcs_wit, mut tr_p) = build_fixture(4, D);

    let (running, running_witnesses) = zero_running::zero_running(&params, &s, 1, mcs_inst.m_in);
    let (mut out_me, proof) = pi_ccs_prove(
        &mut tr_p,
        &params,
        &s,
        core::slice::from_ref(&mcs_inst),
        core::slice::from_ref(&mcs_wit),
        &running,
        &running_witnesses,
    )
    .expect("prove");

    assert!(!out_me.is_empty());
    out_me[0].eval_k[0] += K::ONE;

    let mut tr_v = Poseidon2Transcript::new_v1_2();
    let result = pi_ccs_verify(
        &mut tr_v,
        &params,
        &s,
        core::slice::from_ref(&mcs_inst),
        &running,
        &out_me,
        &proof,
    );
    assert!(!matches!(result, Ok(true)), "mutated Eval_K was accepted");
}

#[test]
fn padded_row_identity_raw_verify_rejects_eval_a_mutation() {
    let (params, s, mcs_inst, mcs_wit, mut tr_p) = build_fixture(4, D);

    let (running, running_witnesses) = zero_running::zero_running(&params, &s, 1, mcs_inst.m_in);
    let (mut outputs, proof) = pi_ccs_prove(
        &mut tr_p,
        &params,
        &s,
        core::slice::from_ref(&mcs_inst),
        core::slice::from_ref(&mcs_wit),
        &running,
        &running_witnesses,
    )
    .expect("prove");

    assert!(!outputs[0].eval_a.is_empty() && !outputs[0].eval_a[0].is_empty());
    outputs[0].eval_a[0][0] += K::ONE;

    let mut tr_v = Poseidon2Transcript::new_v1_2();
    let result = pi_ccs_verify(
        &mut tr_v,
        &params,
        &s,
        core::slice::from_ref(&mcs_inst),
        &running,
        &outputs,
        &proof,
    );

    assert!(
        !matches!(result, Ok(true)),
        "raw PiCCS accepted a mutated Eval_A output"
    );
}

#[test]
fn padded_row_identity_raw_verify_rejects_noncanonical_extra_output_x_column() {
    let (params, s, mut mcs, wit, mut tr_p) = build_fixture(4, D);
    mcs.m_in = D;
    mcs.x = vec![F::ZERO; D];

    let (running, running_witnesses) = zero_running::zero_running(&params, &s, 1, mcs.m_in);
    let (mut outputs, proof) = pi_ccs_prove(
        &mut tr_p,
        &params,
        &s,
        core::slice::from_ref(&mcs),
        core::slice::from_ref(&wit),
        &running,
        &running_witnesses,
    )
    .expect("prove");

    assert_eq!(outputs[0].X.rows(), D);
    assert_eq!(outputs[0].X.cols(), 1);
    outputs[0].X = Mat::zero(D, 2, F::ZERO);
    outputs[0].X[(0, 1)] = F::ONE;

    let mut tr_v = Poseidon2Transcript::new_v1_2();
    let result = pi_ccs_verify(
        &mut tr_v,
        &params,
        &s,
        core::slice::from_ref(&mcs),
        &running,
        &outputs,
        &proof,
    );

    assert!(
        !matches!(result, Ok(true)),
        "raw Pi_CCS accepted a CE claim with a noncanonical extra X column"
    );
}

#[test]
fn raw_pi_ccs_rejects_fresh_count_above_parameter_profile() {
    let (params, s, claim, witness, mut tr_p) = build_fixture(4, D);
    let cap = params.max_fresh_count_from_rlc_guard().expect("guard") as usize;
    let count = cap + 1;
    let (running, running_witnesses) = zero_running::zero_running(&params, &s, 1, claim.m_in);
    let claims = vec![claim; count];
    let witnesses = vec![witness; count];

    let result = pi_ccs_prove(
        &mut tr_p,
        &params,
        &s,
        &claims,
        &witnesses,
        &running,
        &running_witnesses,
    );

    let error = result
        .err()
        .map(|error| error.to_string())
        .unwrap_or_default();
    assert!(
        error.contains(&format!("fresh source count {count} exceeds")),
        "raw Pi_CCS accepted {} fresh claims under a profile capped at {}: {error}",
        count,
        cap
    );
}

#[test]
fn raw_pi_ccs_rejects_running_count_above_parameter_profile() {
    let (params, s, claim, witness, _) = build_fixture(4, D);
    let (seed, seed_witnesses) = zero_running::zero_running(&params, &s, 1, claim.m_in);

    let count = params.k_rho as usize + 1;
    let running = vec![seed[0].clone(); count];
    let running_witnesses = vec![seed_witnesses[0].clone(); count];
    let mut tr_p = Poseidon2Transcript::new_v1_2();
    let result = pi_ccs_prove(
        &mut tr_p,
        &params,
        &s,
        core::slice::from_ref(&claim),
        core::slice::from_ref(&witness),
        &running,
        &running_witnesses,
    );

    assert!(
        result.is_err(),
        "raw Pi_CCS accepted {count} running claims under a profile fixed to k_rho={}",
        params.k_rho
    );
}

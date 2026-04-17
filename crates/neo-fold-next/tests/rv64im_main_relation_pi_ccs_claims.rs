use bellpepper_core::{test_cs::TestConstraintSystem, ConstraintSystem};
use neo_ajtai::Commitment;
use neo_ccs::{CcsStructure, CeClaim, Mat, SparsePoly, Term};
use neo_fold_next::rv64im::main_relation_circuit::claim::alloc_ce_claim;
use neo_fold_next::rv64im::main_relation_circuit::k_field::{alloc_constant_k, enforce_k_eq, KNum};
use neo_fold_next::rv64im::main_relation_circuit::pi_ccs::{
    bind_header_and_instance_digest, bind_me_inputs, sample_challenges,
};
use neo_fold_next::rv64im::main_relation_circuit::transcript::Poseidon2TranscriptCircuit;
use neo_math::F as GoldilocksF;
use neo_params::NeoParams;
use neo_reductions::engines::utils::{
    bind_header_and_instance_digest_with_digest, bind_me_inputs as bind_native_me_inputs, build_dims_and_policy,
    digest_ccs_matrices, sample_beta_m, sample_challenges as sample_native_challenges,
};
use neo_transcript::{Poseidon2Transcript, Transcript};
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use spartan2::provider::goldi::F as SpartanF;

fn toy_claim(seed: u64) -> CeClaim<Commitment, GoldilocksF, neo_math::K> {
    let mut x = Mat::zero(neo_math::D, 2, GoldilocksF::ZERO);
    x[(0, 0)] = GoldilocksF::from_u64(seed + 1);
    x[(1, 1)] = GoldilocksF::from_u64(seed + 3);
    CeClaim {
        c: Commitment {
            d: neo_math::D,
            kappa: 1,
            data: vec![GoldilocksF::from_u64(seed + 5); neo_math::D],
        },
        X: x,
        r: vec![
            neo_math::K::from(GoldilocksF::from_u64(seed + 7)),
            neo_math::K::from(GoldilocksF::from_u64(seed + 11)),
        ],
        s_col: vec![
            neo_math::K::from(GoldilocksF::from_u64(seed + 13)),
            neo_math::K::from(GoldilocksF::from_u64(seed + 17)),
        ],
        y_ring: vec![
            vec![
                neo_math::K::from(GoldilocksF::from_u64(seed + 19)),
                neo_math::K::from(GoldilocksF::from_u64(seed + 23)),
            ],
            vec![
                neo_math::K::from(GoldilocksF::from_u64(seed + 29)),
                neo_math::K::from(GoldilocksF::from_u64(seed + 31)),
            ],
        ],
        ct: vec![
            neo_math::K::from(GoldilocksF::from_u64(seed + 19)),
            neo_math::K::from(GoldilocksF::from_u64(seed + 29)),
        ],
        aux_openings: vec![neo_math::K::from(GoldilocksF::from_u64(seed + 37))],
        y_zcol: vec![neo_math::K::from(GoldilocksF::from_u64(seed + 41)); 64],
        m_in: 2,
        fold_digest: [seed as u8; 32],
        c_step_coords: vec![GoldilocksF::from_u64(seed + 43)],
        u_offset: 1,
        u_len: 2,
    }
}

#[test]
#[ignore = "Spartan-path tests are parked until native NIFS and F' replacement lands"]
fn rv64im_main_relation_pi_ccs_claim_binding_matches_native_challenges() {
    let params = NeoParams::goldilocks_auto_r1cs_ccs(2).expect("auto params");
    let structure = CcsStructure::new(
        vec![Mat::identity(2)],
        SparsePoly::new(
            1,
            vec![
                Term {
                    coeff: GoldilocksF::from_u64(3),
                    exps: vec![1],
                },
                Term {
                    coeff: GoldilocksF::from_u64(5),
                    exps: vec![0],
                },
            ],
        ),
    )
    .expect("simple ccs structure");
    let dims = build_dims_and_policy(&params, &structure).expect("dims");
    let mat_digest = digest_ccs_matrices(&structure);
    let public_instance_digest = [
        GoldilocksF::from_u64(11),
        GoldilocksF::from_u64(13),
        GoldilocksF::from_u64(17),
        GoldilocksF::from_u64(19),
    ];
    let claims = vec![toy_claim(1), toy_claim(101)];

    let mut native = Poseidon2Transcript::new(b"neo.fold.next/test/pi_ccs_claims");
    bind_header_and_instance_digest_with_digest(
        &mut native,
        &params,
        &structure,
        dims,
        &mat_digest,
        &public_instance_digest,
    )
    .expect("native header binding");
    bind_native_me_inputs(&mut native, &claims).expect("native me binding");
    let native_challenges = sample_native_challenges(&mut native, dims.ell_d, dims.ell).expect("native challenges");
    let native_beta_m = sample_beta_m(&mut native, dims.ell_m).expect("native beta_m");

    let mut cs = TestConstraintSystem::<SpartanF>::new();
    let mut transcript =
        Poseidon2TranscriptCircuit::new(cs.namespace(|| "transcript_init"), b"neo.fold.next/test/pi_ccs_claims")
            .expect("circuit transcript");
    let claim_vars = claims
        .iter()
        .enumerate()
        .map(|(idx, claim)| alloc_ce_claim(&mut cs, claim, &format!("claim_{idx}")))
        .collect::<Result<Vec<_>, _>>()
        .expect("alloc claims");

    bind_header_and_instance_digest(
        &mut cs,
        &mut transcript,
        &params,
        structure.n,
        structure.m,
        structure.t(),
        &structure.f,
        dims,
        &mat_digest,
        &public_instance_digest.map(|value| SpartanF::from_canonical_u64(value.as_canonical_u64())),
    )
    .expect("circuit header binding");
    bind_me_inputs(&mut cs, &mut transcript, &claim_vars).expect("circuit me binding");
    let challenges = sample_challenges(&mut cs, &mut transcript, dims).expect("circuit challenges");

    for (idx, value) in native_challenges.alpha.iter().enumerate() {
        let expected = alloc_constant_k(&mut cs, KNum::from_neo_k(*value), &format!("expected_alpha_{idx}"))
            .expect("expected alpha");
        enforce_k_eq(&mut cs, &challenges.alpha[idx], &expected, &format!("alpha_eq_{idx}"));
    }
    for (idx, value) in native_challenges.beta_a.iter().enumerate() {
        let expected = alloc_constant_k(&mut cs, KNum::from_neo_k(*value), &format!("expected_beta_a_{idx}"))
            .expect("expected beta_a");
        enforce_k_eq(&mut cs, &challenges.beta_a[idx], &expected, &format!("beta_a_eq_{idx}"));
    }
    for (idx, value) in native_challenges.beta_r.iter().enumerate() {
        let expected = alloc_constant_k(&mut cs, KNum::from_neo_k(*value), &format!("expected_beta_r_{idx}"))
            .expect("expected beta_r");
        enforce_k_eq(&mut cs, &challenges.beta_r[idx], &expected, &format!("beta_r_eq_{idx}"));
    }
    let expected_gamma =
        alloc_constant_k(&mut cs, KNum::from_neo_k(native_challenges.gamma), "expected_gamma").expect("gamma");
    enforce_k_eq(&mut cs, &challenges.gamma, &expected_gamma, "gamma_eq");
    for (idx, value) in native_beta_m.iter().enumerate() {
        let expected = alloc_constant_k(&mut cs, KNum::from_neo_k(*value), &format!("expected_beta_m_{idx}"))
            .expect("expected beta_m");
        enforce_k_eq(&mut cs, &challenges.beta_m[idx], &expected, &format!("beta_m_eq_{idx}"));
    }

    assert!(cs.is_satisfied(), "{}", cs.which_is_unsatisfied().unwrap_or_default());
}

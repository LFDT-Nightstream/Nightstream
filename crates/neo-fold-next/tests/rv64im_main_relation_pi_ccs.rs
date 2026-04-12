use bellpepper_core::{num::AllocatedNum, test_cs::TestConstraintSystem, ConstraintSystem};
use neo_ccs::{CcsStructure, Mat, SparsePoly, Term};
use neo_fold_next::rv64im::main_relation_circuit::k_field::{alloc_constant_k, enforce_k_eq, KNum};
use neo_fold_next::rv64im::main_relation_circuit::pi_ccs::{
    bind_header_and_instance_digest, bind_me_input_digests, sample_challenges,
};
use neo_fold_next::rv64im::main_relation_circuit::transcript::Poseidon2TranscriptCircuit;
use neo_math::F as GoldilocksF;
use neo_params::NeoParams;
use neo_reductions::engines::utils::{
    bind_header_and_instance_digest_with_digest, build_dims_and_policy, digest_ccs_matrices, sample_beta_m,
    sample_challenges as sample_native_challenges, PI_CCS_ME_COUNT_RAW_TAG, PI_CCS_ME_DIGEST_RAW_TAG,
    PI_CCS_ME_INPUTS_RAW_DOMAIN_TAG,
};
use neo_transcript::{Poseidon2Transcript, Transcript};
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use spartan2::provider::goldi::F as SpartanF;

fn alloc_f_constant(
    cs: &mut TestConstraintSystem<SpartanF>,
    value: SpartanF,
    label: &str,
) -> Result<AllocatedNum<SpartanF>, bellpepper_core::SynthesisError> {
    AllocatedNum::alloc(cs.namespace(|| label.to_string()), || Ok(value))
}

#[test]
fn rv64im_main_relation_pi_ccs_transcript_matches_native_challenge_sampling() {
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
    let me_input_digests = vec![
        [
            GoldilocksF::from_u64(23),
            GoldilocksF::from_u64(29),
            GoldilocksF::from_u64(31),
            GoldilocksF::from_u64(37),
        ],
        [
            GoldilocksF::from_u64(41),
            GoldilocksF::from_u64(43),
            GoldilocksF::from_u64(47),
            GoldilocksF::from_u64(53),
        ],
    ];

    let mut native = Poseidon2Transcript::new(b"neo.fold.next/test/pi_ccs");
    bind_header_and_instance_digest_with_digest(
        &mut native,
        &params,
        &structure,
        dims,
        &mat_digest,
        &public_instance_digest,
    )
    .expect("native header binding");
    native.append_fields_raw(&[GoldilocksF::from_u64(PI_CCS_ME_INPUTS_RAW_DOMAIN_TAG)]);
    native.append_fields_raw(&[
        GoldilocksF::from_u64(PI_CCS_ME_COUNT_RAW_TAG),
        GoldilocksF::from_u64(me_input_digests.len() as u64),
    ]);
    let packed_digests = core::iter::once(GoldilocksF::from_u64(PI_CCS_ME_DIGEST_RAW_TAG))
        .chain(
            me_input_digests
                .iter()
                .flat_map(|digest| digest.iter().copied()),
        )
        .collect::<Vec<_>>();
    native.append_fields_raw(&packed_digests);
    let native_challenges = sample_native_challenges(&mut native, dims.ell_d, dims.ell).expect("native challenges");
    let native_beta_m = sample_beta_m(&mut native, dims.ell_m).expect("native beta_m");

    let mut cs = TestConstraintSystem::<SpartanF>::new();
    let mut transcript =
        Poseidon2TranscriptCircuit::new(cs.namespace(|| "transcript_init"), b"neo.fold.next/test/pi_ccs")
            .expect("circuit transcript");
    let me_digest_vars = me_input_digests
        .iter()
        .enumerate()
        .map(|(digest_idx, digest)| {
            core::array::from_fn(|field_idx| {
                alloc_f_constant(
                    &mut cs,
                    SpartanF::from_canonical_u64(digest[field_idx].as_canonical_u64()),
                    &format!("me_digest_{digest_idx}_{field_idx}"),
                )
                .expect("me digest")
            })
        })
        .collect::<Vec<_>>();

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
    bind_me_input_digests(
        &mut cs,
        &mut transcript,
        &me_digest_vars,
        &me_input_digests
            .iter()
            .map(|digest| digest.map(|value| SpartanF::from_canonical_u64(value.as_canonical_u64())))
            .collect::<Vec<_>>(),
    )
    .expect("circuit me binding");
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
        alloc_constant_k(&mut cs, KNum::from_neo_k(native_challenges.gamma), "expected_gamma").expect("expected gamma");
    enforce_k_eq(&mut cs, &challenges.gamma, &expected_gamma, "gamma_eq");
    for (idx, value) in native_beta_m.iter().enumerate() {
        let expected = alloc_constant_k(&mut cs, KNum::from_neo_k(*value), &format!("expected_beta_m_{idx}"))
            .expect("expected beta_m");
        enforce_k_eq(&mut cs, &challenges.beta_m[idx], &expected, &format!("beta_m_eq_{idx}"));
    }

    assert!(cs.is_satisfied(), "{}", cs.which_is_unsatisfied().unwrap_or_default());
}

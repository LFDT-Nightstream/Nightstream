use neo_ajtai::{
    has_global_pp_for_dims, s_mul_add, scale_commitment_add_inplace, set_global_pp_seeded, AjtaiSModule, Commitment,
};
use neo_ccs::traits::SModuleHomomorphism;
use neo_ccs::{poly::SparsePoly, poly::Term, CcsClaim, CcsStructure, CcsWitness, Mat};
use neo_fold_prototype::core::ivc::build_superneo_ivc_relations_with_initial_carry_accumulator_handle_perf;
use neo_fold_prototype::core::proof::{Carry, FoldSchedule, StepInput};
use neo_fold_prototype::core::prover::CommitmentMixers;
use neo_fold_prototype::direct_ccs::{
    export_latest_direct_ccs_f_prime_verifier_body_r1cs, measure_latest_direct_ccs_f_prime_verifier_body,
    DirectCcsIvcState, DirectCcsProgram,
};
use neo_math::ring::Rq as RqEl;
use neo_math::{D, F};
use neo_params::NeoParams;
use p3_field::PrimeCharacteristicRing;

fn fibonacci_ccs() -> CcsStructure<F> {
    let mut m = Mat::zero(1, D, F::ZERO);
    m[(0, 0)] = F::ONE;
    m[(0, 1)] = F::ONE;
    m[(0, 2)] = -F::ONE;
    let f = SparsePoly::new(
        1,
        vec![Term {
            coeff: F::ONE,
            exps: vec![1],
        }],
    );
    CcsStructure::new(vec![m], f).expect("valid Fibonacci CCS")
}

fn rot_matrix_to_rq(mat: &Mat<F>) -> RqEl {
    use neo_math::ring::cf_inv;

    let mut coeffs = [F::ZERO; D];
    for i in 0..D {
        coeffs[i] = mat[(i, 0)];
    }
    cf_inv(coeffs)
}

fn ajtai_mixers() -> CommitmentMixers<fn(&[Mat<F>], &[Commitment]) -> Commitment, fn(&[Commitment], u32) -> Commitment>
{
    fn mix_rhos_commits(rhos: &[Mat<F>], cs: &[Commitment]) -> Commitment {
        let mut acc = Commitment::zeros(cs[0].d, cs[0].kappa);
        for (rho, c) in rhos.iter().zip(cs.iter()) {
            let rq = rot_matrix_to_rq(rho);
            s_mul_add(&mut acc, &rq, c);
        }
        acc
    }

    fn combine_b_pows(cs: &[Commitment], b: u32) -> Commitment {
        let mut acc = Commitment::zeros(cs[0].d, cs[0].kappa);
        let base = F::from_u64(b as u64);
        let mut pow = F::ONE;
        for c in cs {
            scale_commitment_add_inplace(&mut acc, pow, c);
            pow *= base;
        }
        acc
    }

    CommitmentMixers {
        mix_rhos_commits,
        combine_b_pows,
    }
}

fn make_ajtai_module(params: &NeoParams) -> AjtaiSModule {
    if !has_global_pp_for_dims(D, 1) {
        let mut seed = [0u8; 32];
        seed[..8].copy_from_slice(&0x5355_5045_524e_454f_u64.to_le_bytes());
        match set_global_pp_seeded(D, params.kappa as usize, 1, seed) {
            Ok(()) => {}
            Err(_err) if has_global_pp_for_dims(D, 1) => {}
            Err(err) => panic!("Ajtai global setup: {err}"),
        }
    }
    AjtaiSModule::from_global_for_dims(D, 1).expect("Ajtai global module")
}

fn step(log: &AjtaiSModule, label: &str, a: u64, b: u64, c: u64) -> StepInput {
    let mut z = vec![F::ZERO; D];
    z[0] = F::from_u64(a);
    z[1] = F::from_u64(b);
    z[2] = F::from_u64(c);
    let mut z_mat = Mat::zero(D, 1, F::ZERO);
    for (idx, value) in z.iter().copied().enumerate() {
        z_mat[(idx, 0)] = value;
    }
    let m_in = 3;
    StepInput {
        label: label.to_string(),
        mcs: CcsClaim {
            c: log.commit(&z_mat),
            x: z[..m_in].to_vec(),
            m_in,
        },
        witness: CcsWitness {
            w: z[m_in..].to_vec(),
            Z: z_mat,
        },
    }
}

#[test]
fn direct_ccs_rejects_second_fold_child_tamper() {
    let params = NeoParams::goldilocks_auto_r1cs_ccs(1).expect("valid Fibonacci params");
    let ccs = fibonacci_ccs();
    let log = make_ajtai_module(&params);
    let steps = vec![step(&log, "direct_0", 1, 2, 3), step(&log, "direct_1", 2, 3, 5)];

    let native = build_superneo_ivc_relations_with_initial_carry_accumulator_handle_perf(
        FoldSchedule::RowsPerChunk(1),
        &params,
        &ccs,
        steps,
        Carry::default(),
        &log,
        ajtai_mixers(),
    )
    .expect("native SuperNeo IVC build");
    assert_eq!(native.relations.len(), 2);

    let program = DirectCcsProgram::new_with_public_input_len(&params, &ccs, 3).expect("fixed direct public input");
    let direct = DirectCcsIvcState::new(program).expect("direct CCS state");
    let after_first = direct
        .append_relation(&native.relations[0], &log, ajtai_mixers())
        .expect("append honest first relation");
    let honest_after_second = after_first
        .append_relation(&native.relations[1], &log, ajtai_mixers())
        .expect("append honest second relation");
    let honest_body =
        export_latest_direct_ccs_f_prime_verifier_body_r1cs(&honest_after_second).expect("honest F' body export");
    assert!(honest_body.constraint_count > 0);
    let honest_shape =
        measure_latest_direct_ccs_f_prime_verifier_body(&honest_after_second).expect("honest F' body shape");
    assert!(honest_shape.nifs.pi_dec_constraints > 0);

    let mut tampered_second = native.relations[1].clone();
    tampered_second.state_out.carry.claims[0].X[(0, 0)] += F::ONE;

    let err = match after_first.append_relation(&tampered_second, &log, ajtai_mixers()) {
        Ok(_) => panic!("tampered second-fold CE child must be rejected"),
        Err(err) => err,
    };
    assert!(
        err.to_string()
            .contains("SuperNeo IVC relation state_out does not match verified NIFS.V output"),
        "unexpected tamper rejection error: {err}"
    );
}

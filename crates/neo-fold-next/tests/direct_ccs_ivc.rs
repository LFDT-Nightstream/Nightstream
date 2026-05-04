use neo_ajtai::{
    has_global_pp_for_dims, s_mul_add, scale_commitment_add_inplace, set_global_pp_seeded, AjtaiSModule, Commitment,
};
use neo_ccs::traits::SModuleHomomorphism;
use neo_ccs::{poly::SparsePoly, poly::Term, CcsClaim, CcsMatrix, CcsStructure, CcsWitness, CscMat, Mat};
use neo_fold_next::ivc::build_superneo_ivc_relations_with_initial_carry_accumulator_handle_perf;
use neo_fold_next::proof::{Carry, FoldSchedule, StepInput};
use neo_fold_next::prover::CommitmentMixers;
use neo_fold_next::{
    direct_ccs_program_from_sparse_r1cs_with_public_input_len, direct_ccs_step_from_low_norm_full_witness,
    verify_direct_ccs_ivc_snark, verify_direct_ccs_ivc_snark_public, verify_direct_ccs_recursive_ivc_snark_public,
    verify_direct_ccs_statement, DirectCcsCompactFPrimeImage, DirectCcsFPrimeLowNormSourceR1cs, DirectCcsIvcState,
    DirectCcsNativeFPrimeAdvice, DirectCcsProgram, DirectCcsRecursiveIvcPublicImage, DirectCcsRecursiveIvcState,
    DirectCcsStep, DirectSparseR1csExport,
};
use neo_math::ring::Rq as RqEl;
use neo_math::{D, F};
use neo_params::NeoParams;
use p3_field::PrimeCharacteristicRing;
fn assert_public_verify_rejects(
    vk: &neo_fold_next::DirectCcsIvcSnarkVerifierKey,
    snark: &neo_fold_next::DirectCcsIvcSnark,
    public_image: neo_fold_next::DirectCcsIvcPublicImage,
    label: &str,
) {
    assert!(
        verify_direct_ccs_ivc_snark_public(vk, &public_image, snark.proof()).is_err(),
        "direct CCS public verifier accepted tampered {label}"
    );
}
fn assert_statement_verify_rejects(
    vk: &neo_fold_next::DirectCcsIvcSnarkVerifierKey,
    snark: &neo_fold_next::DirectCcsIvcSnark,
    statement: neo_fold_next::DirectCcsStatement,
    label: &str,
) {
    assert!(
        verify_direct_ccs_statement(vk, &statement, snark.proof()).is_err(),
        "direct CCS statement verifier accepted tampered {label}"
    );
}
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
    make_ajtai_module_for_cols(params, 1)
}

fn make_ajtai_module_for_cols(params: &NeoParams, cols: usize) -> AjtaiSModule {
    if !has_global_pp_for_dims(D, cols) {
        let mut seed = [0u8; 32];
        seed[..8].copy_from_slice(&0x5355_5045_524e_454f_u64.to_le_bytes());
        match set_global_pp_seeded(D, params.kappa as usize, cols, seed) {
            Ok(()) => {}
            Err(_err) if has_global_pp_for_dims(D, cols) => {}
            Err(err) => panic!("Ajtai global setup: {err}"),
        }
    }
    AjtaiSModule::from_global_for_dims(D, cols).expect("Ajtai global module")
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
fn direct_ccs_append_api_matches_native_superneo_carry() {
    let params = NeoParams::goldilocks_auto_r1cs_ccs(1).expect("valid Fibonacci params");
    let ccs = fibonacci_ccs();
    let log = make_ajtai_module(&params);
    let steps = vec![step(&log, "direct_0", 1, 2, 3), step(&log, "direct_1", 2, 3, 5)];

    let native = build_superneo_ivc_relations_with_initial_carry_accumulator_handle_perf(
        FoldSchedule::RowsPerChunk(1),
        &params,
        &ccs,
        steps.clone(),
        Carry::default(),
        &log,
        ajtai_mixers(),
    )
    .expect("native SuperNeo IVC build");

    let program = DirectCcsProgram::new_with_public_input_len(&params, &ccs, 3).expect("fixed direct public input");
    let mut direct = DirectCcsIvcState::new(program).expect("direct CCS state");
    let base_boundary = direct.construction2_public_boundary();
    assert_eq!(base_boundary.commitment_kappa, params.kappa as u64);
    assert_eq!(base_boundary.commitment_data.len(), D * params.kappa as usize);
    assert!(
        base_boundary
            .commitment_data
            .iter()
            .all(|value| *value == F::ZERO),
        "direct CCS base state must carry full-shape canonical Construction-2 u_perp"
    );
    for step in steps {
        direct = direct
            .append_step(DirectCcsStep::new(step), &log, ajtai_mixers())
            .expect("append direct CCS step");
    }

    assert_eq!(direct.final_state().chunk_count, 2);
    assert_eq!(direct.final_state().step_count, 2);
    assert_eq!(direct.final_state().carry.claims, native.final_state.carry.claims);
    assert_eq!(direct.final_state().carry.witnesses, native.final_state.carry.witnesses);
    let construction2_boundary = direct.construction2_public_boundary();
    let latest = direct
        .latest_relation_and_advice()
        .expect("latest direct Construction-2 summary");
    assert_eq!(construction2_boundary.x_i, latest.construction2_x_out);
    assert!(construction2_boundary.commitment_kappa > 1);
    assert!(construction2_boundary.has_canonical_commitment_shape());
    assert_eq!(
        construction2_boundary.commitment_digest,
        construction2_boundary.expected_commitment_digest()
    );
    assert_eq!(
        construction2_boundary.fresh_instance_digest,
        construction2_boundary.expected_fresh_instance_digest()
    );
}

#[test]
fn direct_ccs_program_builds_canonical_zero_carry_shape() {
    let params = NeoParams::goldilocks_auto_r1cs_ccs(1).expect("valid Fibonacci params");
    let ccs = fibonacci_ccs();
    let _log = make_ajtai_module(&params);
    let program = DirectCcsProgram::new_with_public_input_len(&params, &ccs, 3).expect("fixed direct public input");

    let carry = program
        .canonical_zero_carry()
        .expect("canonical direct zero carry");

    assert_eq!(carry.claims.len(), params.k_rho as usize);
    assert_eq!(carry.witnesses.len(), params.k_rho as usize);
    let first = &carry.claims[0];
    assert_eq!(first.c.d, D);
    assert_eq!(first.c.kappa, params.kappa as usize);
    assert_eq!(first.X.rows(), D);
    assert_eq!(first.X.cols(), 3);
    assert_eq!(first.m_in, 3);
    assert_eq!(first.r.len(), 1);
    assert_eq!(first.s_col.len(), 6);
    assert_eq!(first.y_ring.len(), 1);
    assert_eq!(first.y_ring[0].len(), 64);
    assert_eq!(first.ct.len(), 1);
    assert!(first.aux_openings.is_empty());
    assert!(first.c_step_coords.is_empty());
    assert_eq!(carry.witnesses[0].rows(), D);
    assert_eq!(carry.witnesses[0].cols(), ccs.m.div_ceil(D));
}

#[test]
fn direct_ccs_canonical_zero_seed_first_append_has_steady_accumulator_arity() {
    let params = NeoParams::goldilocks_auto_r1cs_ccs(1).expect("valid Fibonacci params");
    let ccs = fibonacci_ccs();
    let log = make_ajtai_module(&params);
    let program = DirectCcsProgram::new_with_public_input_len(&params, &ccs, 3).expect("fixed direct public input");
    let direct = DirectCcsIvcState::new_with_canonical_zero_carry(program).expect("canonical zero-seeded state");
    let base_boundary = direct.construction2_public_boundary();
    assert_eq!(base_boundary.commitment_kappa, params.kappa as u64);
    assert_eq!(base_boundary.commitment_data.len(), D * params.kappa as usize);
    assert!(
        base_boundary
            .commitment_data
            .iter()
            .all(|value| *value == F::ZERO),
        "canonical zero-seeded direct state must use full-shape zero Construction-2 u_perp"
    );

    let direct = direct
        .append_step(
            DirectCcsStep::new(step(&log, "steady_direct_0", 1, 2, 3)),
            &log,
            ajtai_mixers(),
        )
        .expect("append first steady direct CCS step");
    let latest = direct
        .latest_relation_and_advice()
        .expect("latest steady direct Construction-2 summary");

    assert_eq!(latest.incoming_ce_claims, params.k_rho as usize);
    assert_eq!(latest.output_ce_claims, params.k_rho as usize + 1);
    assert_eq!(latest.final_ce_claims, params.k_rho as usize);
}

#[test]
fn direct_ccs_single_step_compression_uses_latest_chunk_and_no_final_ce_digest() {
    let params = NeoParams::goldilocks_auto_r1cs_ccs(1).expect("valid Fibonacci params");
    let ccs = fibonacci_ccs();
    let log = make_ajtai_module(&params);
    let program = DirectCcsProgram::new_with_public_input_len(&params, &ccs, 3).expect("fixed direct public input");
    let mut direct = DirectCcsIvcState::new(program).expect("direct CCS state");
    direct = direct
        .append_step(
            DirectCcsStep::new(step(&log, "direct_0", 1, 2, 3)),
            &log,
            ajtai_mixers(),
        )
        .expect("append first step");
    let mut trace = Vec::new();
    let (snark, vk, perf) = direct
        .compress_snark_with_trace(&mut |message| trace.push(message.to_string()))
        .unwrap_or_else(|err| panic!("compress direct CCS IVC: {err}; trace={trace:?}"));
    snark
        .verify(&vk, snark.public_image())
        .expect("verify compressed direct CCS IVC through public image");
    verify_direct_ccs_ivc_snark_public(&vk, snark.public_image(), snark.proof())
        .expect("verify compressed direct CCS IVC through public function");
    let statement = snark.statement();
    verify_direct_ccs_statement(&vk, &statement, snark.proof())
        .expect("verify compressed direct CCS IVC through compact statement");
    let mut tampered_statement = statement.clone();
    tampered_statement.step_count_out += 1;
    assert!(
        verify_direct_ccs_statement(&vk, &tampered_statement, snark.proof()).is_err(),
        "statement verifier must reject a tampered step counter"
    );
    verify_direct_ccs_ivc_snark(&direct, snark.proof()).expect("verify state-bound compatibility helper");

    let mut tampered_public_image = snark.public_image().clone();
    tampered_public_image.step_count_out += 1;
    assert!(
        verify_direct_ccs_ivc_snark_public(&vk, &tampered_public_image, snark.proof()).is_err(),
        "public verifier must reject a tampered Construction-2 output counter"
    );

    let mut tampered_snark = snark.clone();
    tampered_snark.public_image_mut().pc += 1;
    assert!(
        tampered_snark
            .verify(&vk, tampered_snark.public_image())
            .is_err(),
        "public verifier must reject a tampered direct relation program counter"
    );

    assert_eq!(
        perf.chunk_count, 1,
        "terminal compression must synthesize only the latest F' chunk"
    );
    assert_eq!(perf.final_ce_bundle_digest_constraints, 0);
    assert_eq!(perf.final_ce_bundle_digest_match_constraints, 0);
    assert_eq!(perf.final_ce_bundle_constraints, 0);
    assert_eq!(
        perf.construction2_fold_constraints, 0,
        "plain direct compression has no prior F' accumulator to fold"
    );
    assert_eq!(
        perf.construction2_fold_final_ce_consistency_constraints, 0,
        "folded prior F' authority must not contain terminal final-CE consistency"
    );
    assert_eq!(perf.terminal_unclassified_private_values, 0);
    assert!(perf.terminal_source_u32_values > 0);
    assert!(perf.terminal_source_u64_values > 0);
    assert_eq!(
        perf.terminal_source_values,
        perf.terminal_source_bit_values + perf.terminal_source_u32_values + perf.terminal_source_u64_values,
        "terminal committed source labels must be explicitly classified"
    );
}

#[test]
fn direct_ccs_public_verifier_rejects_authoritative_boundary_tampering() {
    let params = NeoParams::goldilocks_auto_r1cs_ccs(1).expect("valid Fibonacci params");
    let ccs = fibonacci_ccs();
    let log = make_ajtai_module(&params);
    let program = DirectCcsProgram::new_with_public_input_len(&params, &ccs, 3).expect("fixed direct public input");
    let direct = DirectCcsIvcState::new_with_canonical_zero_carry(program)
        .expect("canonical zero-seeded direct state")
        .append_step(
            DirectCcsStep::new(step(&log, "direct_authority_0", 1, 2, 3)),
            &log,
            ajtai_mixers(),
        )
        .expect("append direct CCS step");
    let (snark, vk, perf) = direct
        .compress_snark()
        .expect("compress direct CCS IVC boundary");
    snark
        .verify(&vk, snark.public_image())
        .expect("baseline direct CCS proof verifies");
    assert_eq!(perf.final_ce_bundle_digest_constraints, 0);
    DirectCcsRecursiveIvcPublicImage::from_terminal_and_f_prime_accumulator(
        snark.public_image().clone(),
        snark.public_image().construction2_accumulator_digest,
        params.b,
        0,
        params.k_rho as u64,
    )
    .expect("recursive public image accepts matching terminal/F' accumulator digests");
    let mut mismatched_f_prime_accumulator = snark.public_image().construction2_accumulator_digest;
    mismatched_f_prime_accumulator[0] ^= 1;
    assert!(
        DirectCcsRecursiveIvcPublicImage::from_terminal_and_f_prime_accumulator(
            snark.public_image().clone(),
            mismatched_f_prime_accumulator,
            params.b,
            0,
            params.k_rho as u64,
        )
        .is_err(),
        "recursive public image must reject a terminal x_out digest that is not the proven F' accumulator digest"
    );

    let mut image = snark.public_image().clone();
    image.mat_digest[0] += F::ONE;
    assert_public_verify_rejects(&vk, &snark, image, "relation digest");

    let mut image = snark.public_image().clone();
    image.vk_fs_digest[0] ^= 1;
    assert_public_verify_rejects(&vk, &snark, image, "vk_fs digest");

    let mut image = snark.public_image().clone();
    image.initial_boundary_digest[0] ^= 1;
    assert_public_verify_rejects(&vk, &snark, image, "initial boundary digest");

    let mut image = snark.public_image().clone();
    image.current_boundary_digest[0] ^= 1;
    assert_public_verify_rejects(&vk, &snark, image, "current boundary digest");

    let mut image = snark.public_image().clone();
    image.accumulator_out_digest[0] ^= 1;
    assert_public_verify_rejects(&vk, &snark, image, "final accumulator digest");

    let mut image = snark.public_image().clone();
    image.public_trace_out_digest[0] ^= 1;
    assert_public_verify_rejects(&vk, &snark, image, "public trace digest");

    let mut image = snark.public_image().clone();
    image.construction2_accumulator_digest[0] ^= 1;
    assert_public_verify_rejects(&vk, &snark, image, "Construction-2 accumulator digest");

    let mut image = snark.public_image().clone();
    let mut bad_x = image.x_out.bytes();
    bad_x[0] ^= 1;
    image.construction2_u_i.x_i =
        neo_fold_next::construction2::Construction2EncodedPublicInput::from_digest_bytes(bad_x);
    image.construction2_u_i.fresh_instance_digest = image.construction2_u_i.expected_fresh_instance_digest();
    assert_public_verify_rejects(&vk, &snark, image, "Construction-2 x_i");

    let mut image = snark.public_image().clone();
    image.construction2_u_i.commitment_data[0] += F::ONE;
    assert_public_verify_rejects(&vk, &snark, image, "stale Construction-2 commitment digest");

    let mut image = snark.public_image().clone();
    image.construction2_u_i.commitment_data[0] += F::ONE;
    image.construction2_u_i.commitment_digest = image.construction2_u_i.expected_commitment_digest();
    image.construction2_u_i.fresh_instance_digest = image.construction2_u_i.expected_fresh_instance_digest();
    assert_public_verify_rejects(
        &vk,
        &snark,
        image,
        "coherently re-digested Construction-2 commitment data",
    );

    let mut image = snark.public_image().clone();
    image.construction2_u_i.commitment_kappa = 0;
    image.construction2_u_i.commitment_data.clear();
    image.construction2_u_i.commitment_digest = image.construction2_u_i.expected_commitment_digest();
    image.construction2_u_i.fresh_instance_digest = image.construction2_u_i.expected_fresh_instance_digest();
    assert_public_verify_rejects(&vk, &snark, image, "noncanonical Construction-2 commitment shape");

    let mut statement = snark.statement();
    statement.mat_digest[0] += F::ONE;
    assert_statement_verify_rejects(&vk, &snark, statement, "statement relation digest");

    let mut statement = snark.statement();
    statement.vk_fs_digest[0] ^= 1;
    assert_statement_verify_rejects(&vk, &snark, statement, "statement vk_fs digest");

    let mut statement = snark.statement();
    statement.initial_boundary_digest[0] ^= 1;
    assert_statement_verify_rejects(&vk, &snark, statement, "statement initial boundary digest");

    let mut statement = snark.statement();
    statement.current_boundary_digest[0] ^= 1;
    assert_statement_verify_rejects(&vk, &snark, statement, "statement current boundary digest");

    let mut statement = snark.statement();
    statement.pc += 1;
    assert_statement_verify_rejects(&vk, &snark, statement, "statement pc");

    let mut statement = snark.statement();
    statement.chunk_count_out += 1;
    assert_statement_verify_rejects(&vk, &snark, statement, "statement chunk counter");

    let mut statement = snark.statement();
    statement.accumulator_out_digest[0] ^= 1;
    assert_statement_verify_rejects(&vk, &snark, statement, "statement accumulator digest");

    let mut statement = snark.statement();
    statement.public_trace_out_digest[0] ^= 1;
    assert_statement_verify_rejects(&vk, &snark, statement, "statement trace digest");

    let mut statement = snark.statement();
    statement.construction2_u_i.commitment_data[0] += F::ONE;
    statement.construction2_u_i.commitment_digest = statement.construction2_u_i.expected_commitment_digest();
    statement.construction2_u_i.fresh_instance_digest = statement.construction2_u_i.expected_fresh_instance_digest();
    assert_statement_verify_rejects(
        &vk,
        &snark,
        statement,
        "statement re-digested Construction-2 commitment",
    );
}

#[test]
fn direct_ccs_multi_step_plain_terminal_compression_is_refused() {
    let params = NeoParams::goldilocks_auto_r1cs_ccs(1).expect("valid Fibonacci params");
    let ccs = fibonacci_ccs();
    let log = make_ajtai_module(&params);
    let program = DirectCcsProgram::new_with_public_input_len(&params, &ccs, 3).expect("fixed direct public input");
    let mut direct = DirectCcsIvcState::new(program).expect("direct CCS state");
    direct = direct
        .append_step(
            DirectCcsStep::new(step(&log, "direct_0", 1, 2, 3)),
            &log,
            ajtai_mixers(),
        )
        .expect("append first step");
    direct = direct
        .append_step(
            DirectCcsStep::new(step(&log, "direct_1", 2, 3, 5)),
            &log,
            ajtai_mixers(),
        )
        .expect("append second step");

    let err = match direct.compress_snark() {
        Ok(_) => panic!("plain direct terminal compression must refuse multi-step latest-only proofs"),
        Err(err) => err,
    };
    assert!(
        err.to_string().contains("latest-only") && err.to_string().contains("disabled for multi-step"),
        "unexpected direct multi-step terminal compression error: {err}"
    );
}

#[test]
fn direct_ccs_append_rejects_shape_drift() {
    let params = NeoParams::goldilocks_auto_r1cs_ccs(1).expect("valid Fibonacci params");
    let ccs = fibonacci_ccs();
    let log = make_ajtai_module(&params);
    let program = DirectCcsProgram::new_with_public_input_len(&params, &ccs, 3).expect("fixed direct public input");
    let direct = DirectCcsIvcState::new(program).expect("direct CCS state");

    let mut bad = step(&log, "bad_shape", 1, 2, 3);
    bad.witness.Z = Mat::zero(D, 2, F::ZERO);

    assert!(
        direct
            .append_step(DirectCcsStep::new(bad), &log, ajtai_mixers())
            .is_err(),
        "direct CCS append must reject witness/shape drift before it enters terminal compression"
    );
}

#[test]
fn direct_ccs_append_rejects_public_input_layout_drift() {
    let params = NeoParams::goldilocks_auto_r1cs_ccs(1).expect("valid Fibonacci params");
    let ccs = fibonacci_ccs();
    let log = make_ajtai_module(&params);
    let program = DirectCcsProgram::new_with_public_input_len(&params, &ccs, 3).expect("fixed direct public input");
    let direct = DirectCcsIvcState::new(program).expect("direct CCS state");

    let mut bad = step(&log, "bad_public_layout", 1, 2, 3);
    bad.mcs.m_in = 2;
    bad.mcs.x.truncate(2);
    bad.witness.w = vec![F::ZERO; ccs.m - bad.mcs.m_in];

    let err = match direct.append_step(DirectCcsStep::new(bad), &log, ajtai_mixers()) {
        Ok(_) => panic!("direct CCS append must reject public input layout drift"),
        Err(err) => err,
    };
    assert!(
        err.to_string().contains("fixed program public input len"),
        "unexpected public input drift error: {err}"
    );
}

#[test]
fn direct_sparse_r1cs_adapter_builds_program_and_step() {
    let params = NeoParams::goldilocks_auto_r1cs_ccs(1).expect("valid tiny R1CS params");
    let a = CcsMatrix::Csc(CscMat::from_triplets(vec![(0, 0, F::ONE)], 1, D));
    let b = CcsMatrix::Csc(CscMat::from_triplets(
        vec![(0, 1, F::ONE), (0, 2, F::ONE), (0, 3, -F::ONE)],
        1,
        D,
    ));
    let c = CcsMatrix::Csc(CscMat::from_triplets(Vec::new(), 1, D));
    let export = DirectSparseR1csExport {
        a: a.clone(),
        b: b.clone(),
        c: c.clone(),
        witness: {
            let mut witness = vec![F::ZERO; D];
            witness[0] = F::ONE;
            witness[1] = F::from_u64(2);
            witness[2] = F::from_u64(3);
            witness[3] = F::from_u64(5);
            witness
        },
        public_input_len: 4,
        constraint_count: 1,
        variable_count: D,
    };
    let program_from_export = export
        .to_direct_ccs_program()
        .expect("exported direct CCS program");
    assert_eq!(program_from_export.structure().n, 1);
    assert_eq!(program_from_export.structure().m, D);

    let program = direct_ccs_program_from_sparse_r1cs_with_public_input_len(&params, a, b, c, 4)
        .expect("direct sparse R1CS adapter");
    let log = make_ajtai_module(&params);

    let step = export
        .clone()
        .into_direct_ccs_step(&program, &log, "r1cs_fib_step_export")
        .expect("exported direct sparse R1CS step");
    let step_from_witness =
        direct_ccs_step_from_low_norm_full_witness(&program, &log, "r1cs_fib_step", &export.witness, 4)
            .expect("direct sparse R1CS step from full witness");
    assert_eq!(
        step.clone().into_step_input().mcs.x,
        step_from_witness.into_step_input().mcs.x
    );
    let (_program_again, _step_again) = export
        .into_direct_ccs_program_and_step(&log, "r1cs_fib_step_export_pair")
        .expect("exported direct sparse R1CS program and step pair");
    let direct = DirectCcsIvcState::new(program)
        .expect("direct CCS state")
        .append_step(step, &log, ajtai_mixers())
        .expect("direct sparse R1CS step");

    assert_eq!(direct.final_state().chunk_count, 1);
    assert_eq!(direct.final_state().step_count, 1);
}

#[test]
fn direct_sparse_r1cs_adapter_rejects_non_low_norm_witness() {
    let mut export = tiny_sparse_r1cs_export(D, 4);
    export.witness[3] = F::from_u64(1u64 << 60);
    let program = export.to_direct_ccs_program().expect("direct CCS program");
    let log = make_ajtai_module(program.params());

    let err = match direct_ccs_step_from_low_norm_full_witness(&program, &log, "bad_r1cs_step", &export.witness, 4) {
        Ok(_) => panic!("direct R1CS adapter must reject witnesses outside the SuperNeo low-norm budget"),
        Err(err) => err,
    };
    assert!(
        err.to_string().contains("not SuperNeo low-norm packable"),
        "unexpected direct R1CS low-norm rejection: {err}"
    );

    let err = match export.into_direct_ccs_step(&program, &log, "bad_r1cs_step_export") {
        Ok(_) => panic!("exported direct R1CS adapter must preserve the low-norm rejection"),
        Err(err) => err,
    };
    assert!(
        err.to_string().contains("not SuperNeo low-norm packable"),
        "unexpected exported direct R1CS low-norm rejection: {err}"
    );
}

#[test]
fn direct_recursive_f_prime_authority_is_not_public_or_terminal_source_image_based() {
    let direct_mod = include_str!("../src/direct_ccs/mod.rs");
    let recursive_src = include_str!("../src/direct_ccs/recursive.rs");
    let f_prime_chain_src = include_str!("../src/direct_ccs/f_prime_chain.rs");
    let construction2_fold_src = include_str!("../src/direct_ccs/construction2_fold.rs");
    let r1cs_src = include_str!("../src/direct_ccs/r1cs.rs");
    let crate_root = include_str!("../src/lib.rs");

    assert!(
        !direct_mod.contains("pub use f_prime_chain"),
        "the low-norm F' chain helper must stay internal until the crate owns a real enc(F') builder"
    );
    assert!(
        !crate_root.contains("DirectCcsFPrimeChain"),
        "public direct CCS API must not expose caller-supplied F' authority helpers"
    );
    assert!(
        !recursive_src.contains("append_latest_terminal_export"),
        "recursive direct CCS append must not fold terminal committed/source-image exports"
    );
    assert!(
        !recursive_src.contains("direct_terminal_shape_export")
            && !recursive_src.contains("DirectCcsTerminalCommittedRelation")
            && !f_prime_chain_src.contains("direct_terminal_shape_export")
            && !f_prime_chain_src.contains("DirectCcsTerminalCommittedRelation"),
        "direct F' authority must not be sourced from terminal committed Spartan exports"
    );
    assert!(
        r1cs_src.contains("not SuperNeo low-norm packable"),
        "direct R1CS adapter must keep the low-norm witness boundary explicit"
    );
    assert!(
        !construction2_fold_src.contains("enforce_direct_terminal_final_ce_consistency")
            && !construction2_fold_src.contains("direct_terminal_construction2_fold_final_ce")
            && !construction2_fold_src.contains("final_ce_consistency"),
        "Construction-2 folded F' authority must not include terminal final-CE consistency; that belongs at final compression"
    );
}

#[test]
fn direct_recursive_ivc_state_starts_without_terminal_step() {
    let params = NeoParams::goldilocks_auto_r1cs_ccs(1).expect("valid Fibonacci params");
    let ccs = fibonacci_ccs();
    let _log = make_ajtai_module(&params);
    let program = DirectCcsProgram::new_with_public_input_len(&params, &ccs, 3).expect("fixed direct public input");
    let recursive = DirectCcsRecursiveIvcState::new_with_canonical_zero_carry(program).expect("direct recursive state");
    let summary = recursive.summary();

    assert_eq!(summary.semantic_chunks, 0);
    assert_eq!(summary.semantic_steps, 0);
    assert_eq!(summary.terminal_chunks_synthesized, 0);
    assert_eq!(summary.carried_semantic_ce_claims, params.k_rho as usize);
    assert_eq!(summary.folded_f_prime_r2_steps, 0);
    assert_eq!(summary.carried_f_prime_ce_claims, 0);
    assert!(!summary.native_f_prime_evaluator_available);
    assert!(!summary.f_prime_encoder_required);
    assert!(!summary.f_prime_encoder_available);
    assert_eq!(summary.compact_f_prime_image_digest, None);
    assert!(!summary.low_norm_f_prime_source_available);
    assert_eq!(summary.low_norm_f_prime_source_len, 0);
    assert_eq!(summary.low_norm_f_prime_source_digest, None);
    assert_eq!(summary.low_norm_f_prime_source_r1cs_constraints, 0);
    assert_eq!(summary.low_norm_f_prime_source_r1cs_variables, 0);
    assert_eq!(summary.low_norm_f_prime_source_r1cs_nnz, 0);
    assert_eq!(summary.low_norm_f_prime_source_shell_constraints, 0);
    assert_eq!(summary.low_norm_f_prime_source_authority_constraints, 0);
    assert_eq!(
        summary.low_norm_f_prime_source_poseidon_digest_recomputation_constraints,
        0
    );
    assert_eq!(summary.low_norm_f_prime_source_nifs_v_verifier_constraints, 0);
    assert_eq!(summary.f_prime_encoder_blocker, None);
    assert!(!summary.standalone_proof_authority_ready);
}

#[test]
fn direct_recursive_ivc_compression_requires_appended_step() {
    let params = NeoParams::goldilocks_auto_r1cs_ccs(1).expect("valid Fibonacci params");
    let ccs = fibonacci_ccs();
    let _log = make_ajtai_module(&params);
    let program = DirectCcsProgram::new_with_public_input_len(&params, &ccs, 3).expect("fixed direct public input");
    let recursive = DirectCcsRecursiveIvcState::new_with_canonical_zero_carry(program).expect("direct recursive state");
    let err = match recursive.compress_recursive_snark() {
        Ok(_) => panic!("empty recursive direct IVC compression must be rejected"),
        Err(err) => err,
    };
    assert!(
        err.to_string().contains("at least one appended F' step"),
        "unexpected empty recursive compression error: {err}"
    );
}

#[test]
fn direct_recursive_ivc_append_does_not_fold_terminal_source_image_exports() {
    let params = NeoParams::goldilocks_auto_r1cs_ccs(1).expect("valid Fibonacci params");
    let ccs = fibonacci_ccs();
    let log = make_ajtai_module(&params);
    let program = DirectCcsProgram::new_with_public_input_len(&params, &ccs, 3).expect("fixed direct public input");
    let recursive = DirectCcsRecursiveIvcState::new_with_canonical_zero_carry(program)
        .expect("direct recursive state")
        .append_step(
            DirectCcsStep::new(step(&log, "recursive_direct_0", 1, 2, 3)),
            &log,
            ajtai_mixers(),
        )
        .expect("append first recursive direct step")
        .append_step(
            DirectCcsStep::new(step(&log, "recursive_direct_1", 2, 3, 5)),
            &log,
            ajtai_mixers(),
        )
        .expect("append second recursive direct step")
        .append_step(
            DirectCcsStep::new(step(&log, "recursive_direct_2", 3, 5, 8)),
            &log,
            ajtai_mixers(),
        )
        .expect("append third recursive direct step");
    let summary = recursive.summary();

    assert_eq!(summary.semantic_chunks, 3);
    assert_eq!(summary.semantic_steps, 3);
    assert_eq!(
        summary.folded_f_prime_r2_steps, 0,
        "direct recursive append must not fold the compact source shell before it contains NIFS.V authority"
    );
    assert_eq!(summary.carried_semantic_ce_claims, params.k_rho as usize);
    assert_eq!(
        summary.carried_f_prime_ce_claims, 0,
        "the F' accumulator must stay empty until a proof-authoritative low-norm enc(F') exists"
    );
    assert!(
        summary.f_prime_encoder_required,
        "multi-step direct recursion must require an encoded prior F' relation"
    );
    assert!(
        summary.native_f_prime_evaluator_available,
        "the latest compact direct F' native evaluator should be available before low-norm encoding is implemented"
    );
    assert!(
        !summary.f_prime_encoder_available,
        "the direct path must not report low-norm F' encoder availability until one exists"
    );
    assert!(
        summary.compact_f_prime_image_digest.is_some(),
        "the compact F' image digest can exist, but it is not proof authority by itself"
    );
    assert!(
        summary.low_norm_f_prime_source_available,
        "the compact native F' advice should now export a low-norm source image"
    );
    assert!(
        summary.low_norm_f_prime_source_len > 0,
        "the low-norm F' source image should contain the bits needed by a future enc(F') relation"
    );
    assert!(
        summary.low_norm_f_prime_source_digest.is_some(),
        "the low-norm F' source image digest should be available as a diagnostic handle"
    );
    assert!(summary.low_norm_f_prime_source_r1cs_constraints > 0);
    assert!(summary.low_norm_f_prime_source_r1cs_variables > 0);
    assert!(summary.low_norm_f_prime_source_r1cs_nnz > 0);
    assert_eq!(
        summary.low_norm_f_prime_source_authority_constraints, 0,
        "Poseidon2 digest recomputation must not be counted as recursive proof authority"
    );
    assert!(
        summary.low_norm_f_prime_source_poseidon_digest_recomputation_constraints > 0,
        "Construction-2 digest fields must not remain self-consistent diagnostic data"
    );
    assert_eq!(
        summary.low_norm_f_prime_source_nifs_v_verifier_constraints, 0,
        "the source shell must not claim to verify NIFS.V"
    );
    assert!(
        summary
            .f_prime_encoder_blocker
            .is_some_and(|blocker| blocker.contains("low-norm")),
        "multi-step direct recursion must expose the missing encoded-F' blocker"
    );
    assert!(
        !summary.standalone_proof_authority_ready,
        "multi-step direct compression must not be marked proof-complete until compact F' source rows include NIFS.V authority"
    );
}

#[test]
fn direct_recursive_latest_step_is_not_historical_replay() {
    let params = NeoParams::goldilocks_auto_r1cs_ccs(1).expect("valid Fibonacci params");
    let ccs = fibonacci_ccs();
    let log = make_ajtai_module(&params);
    let program = DirectCcsProgram::new_with_public_input_len(&params, &ccs, 3).expect("fixed direct public input");
    let recursive = DirectCcsRecursiveIvcState::new_with_canonical_zero_carry(program)
        .expect("direct recursive state")
        .append_step(
            DirectCcsStep::new(step(&log, "recursive_direct_0", 1, 2, 3)),
            &log,
            ajtai_mixers(),
        )
        .expect("append first recursive direct step")
        .append_step(
            DirectCcsStep::new(step(&log, "recursive_direct_1", 2, 3, 5)),
            &log,
            ajtai_mixers(),
        )
        .expect("append second recursive direct step")
        .append_step(
            DirectCcsStep::new(step(&log, "recursive_direct_2", 3, 5, 8)),
            &log,
            ajtai_mixers(),
        )
        .expect("append third recursive direct step");

    let latest = recursive
        .direct_state()
        .latest_relation_and_advice()
        .expect("latest direct F' relation");
    assert_eq!(latest.chunk_index, 2);
    assert_eq!(latest.fresh_claims, 1);
    assert_eq!(latest.incoming_ce_claims, params.k_rho as usize);
    assert_eq!(latest.output_ce_claims, params.k_rho as usize + 1);
    assert_eq!(latest.final_ce_claims, params.k_rho as usize);
    assert_eq!(
        recursive.summary().terminal_chunks_synthesized,
        1,
        "terminal compression may synthesize only the latest F' step, never every historical chunk"
    );
}

#[test]
fn direct_compact_f_prime_image_binds_latest_step_without_terminal_material() {
    let params = NeoParams::goldilocks_auto_r1cs_ccs(1).expect("valid Fibonacci params");
    let ccs = fibonacci_ccs();
    let log = make_ajtai_module(&params);
    let program = DirectCcsProgram::new_with_public_input_len(&params, &ccs, 3).expect("fixed direct public input");
    let recursive = DirectCcsRecursiveIvcState::new_with_canonical_zero_carry(program)
        .expect("direct recursive state")
        .append_step(
            DirectCcsStep::new(step(&log, "recursive_direct_0", 1, 2, 3)),
            &log,
            ajtai_mixers(),
        )
        .expect("append first recursive direct step")
        .append_step(
            DirectCcsStep::new(step(&log, "recursive_direct_1", 2, 3, 5)),
            &log,
            ajtai_mixers(),
        )
        .expect("append second recursive direct step")
        .append_step(
            DirectCcsStep::new(step(&log, "recursive_direct_2", 3, 5, 8)),
            &log,
            ajtai_mixers(),
        )
        .expect("append third recursive direct step");

    let image =
        DirectCcsCompactFPrimeImage::from_latest_state(recursive.direct_state()).expect("compact direct F' image");
    let image_digest = image.expected_digest().expect("valid compact F' digest");
    let summary = recursive.summary();

    assert_eq!(image.chunk_count_in, 2);
    assert_eq!(image.chunk_count_out, 3);
    assert_eq!(image.step_count_in, 2);
    assert_eq!(image.step_count_out, 3);
    assert_eq!(image.fresh_claims, 1);
    assert_eq!(image.incoming_ce_claims, params.k_rho as u64);
    assert_eq!(image.output_ce_claims, params.k_rho as u64 + 1);
    assert_eq!(image.final_ce_claims, params.k_rho as u64);
    assert_ne!(image_digest, [0u8; 32]);
    assert!(summary.native_f_prime_evaluator_available);
    assert_eq!(
        summary.compact_f_prime_image_digest,
        Some(image_digest),
        "recursive summary should expose the compact image digest as a diagnostic handle, not as authority"
    );
    assert!(summary.low_norm_f_prime_source_available);
    assert!(summary.low_norm_f_prime_source_len > 0);
    assert!(summary.low_norm_f_prime_source_digest.is_some());
    assert!(summary.f_prime_encoder_required);
    assert!(!summary.f_prime_encoder_available);
    assert_eq!(
        summary.low_norm_f_prime_source_authority_constraints, 0,
        "compact source diagnostics must not count digest recomputation as authority"
    );
    assert!(
        summary
            .f_prime_encoder_blocker
            .is_some_and(|blocker| blocker.contains("low-norm")),
        "multi-step compact F' image must still report the missing encoder"
    );

    let mut bad = image.clone();
    bad.chunk_count_out += 1;
    assert!(
        bad.validate().is_err(),
        "compact direct F' image must bind the output chunk counter"
    );

    let mut bad = image.clone();
    bad.current_boundary_out_digest[0] ^= 1;
    assert!(
        bad.validate().is_err(),
        "compact direct F' image must bind the latest chunk into the current boundary"
    );

    let mut bad = image.clone();
    bad.x_out = image.x_in.clone();
    assert!(
        bad.validate().is_err(),
        "compact direct F' image must bind x_out to output counters and accumulator handles"
    );
}

#[test]
fn direct_native_f_prime_advice_evaluates_compact_image_and_construction2_boundaries() {
    let params = NeoParams::goldilocks_auto_r1cs_ccs(1).expect("valid Fibonacci params");
    let ccs = fibonacci_ccs();
    let log = make_ajtai_module(&params);
    let program = DirectCcsProgram::new_with_public_input_len(&params, &ccs, 3).expect("fixed direct public input");
    let recursive = DirectCcsRecursiveIvcState::new_with_canonical_zero_carry(program)
        .expect("direct recursive state")
        .append_step(
            DirectCcsStep::new(step(&log, "recursive_direct_0", 1, 2, 3)),
            &log,
            ajtai_mixers(),
        )
        .expect("append first recursive direct step")
        .append_step(
            DirectCcsStep::new(step(&log, "recursive_direct_1", 2, 3, 5)),
            &log,
            ajtai_mixers(),
        )
        .expect("append second recursive direct step")
        .append_step(
            DirectCcsStep::new(step(&log, "recursive_direct_2", 3, 5, 8)),
            &log,
            ajtai_mixers(),
        )
        .expect("append third recursive direct step");

    let advice =
        DirectCcsNativeFPrimeAdvice::from_latest_state(recursive.direct_state()).expect("native direct F' advice");
    let step_image = advice.evaluate().expect("native direct F' evaluation");
    let compact_digest = step_image
        .compact_image()
        .expected_digest()
        .expect("compact F' image digest");
    let source = advice
        .low_norm_source_image()
        .expect("native direct F' low-norm source image");
    let summary = recursive.summary();

    assert_eq!(
        summary.compact_f_prime_image_digest,
        Some(compact_digest),
        "native direct F' advice and recursive summary must agree on the compact latest-step image"
    );
    assert_eq!(
        summary.low_norm_f_prime_source_digest,
        Some(source.expected_digest()),
        "recursive summary should expose the native F' source-image digest as diagnostic evidence"
    );
    assert_eq!(summary.low_norm_f_prime_source_len, source.len());
    assert!(
        source
            .values()
            .iter()
            .all(|value| *value == F::ZERO || *value == F::ONE),
        "direct F' source image must be binary low-norm material before it can be encoded as SuperNeo CCS"
    );
    assert_eq!(source.digest_count(), 15);
    assert_eq!(source.encoded_public_input_count(), 3);
    assert_eq!(source.construction2_commitment_fields(), 0);
    assert_eq!(
        source.u64_count(),
        30,
        "direct F' source image should contain compact counters and handles, not terminal commitment data"
    );
    assert_eq!(
        source.len(),
        source.digest_count() * 256 + source.encoded_public_input_count() * 256 + source.u64_count() * 64,
        "direct F' source image length must be mechanically explained by its primitive encodings"
    );
    let source_r1cs =
        DirectCcsFPrimeLowNormSourceR1cs::from_native_advice(&advice, params.kappa as u64, 1, params.k_rho as u64)
            .expect("native F' low-norm source R1CS");
    assert_eq!(source_r1cs.shape.source_len, source.len());
    assert_eq!(source_r1cs.shape.public_input_len, 257);
    assert_eq!(
        source_r1cs.shape.constraint_count,
        source_r1cs.shape.shell_constraints()
            + source_r1cs.shape.digest_binding_constraints()
            + source_r1cs.shape.authority_constraints()
    );
    assert_eq!(source_r1cs.shape.x_out_link_constraints, 256);
    assert_eq!(source_r1cs.shape.construction2_boundary_link_constraints, 256);
    assert_eq!(source_r1cs.shape.construction2_commitment_shape_constraints, 128);
    assert_eq!(source_r1cs.shape.structural_counter_constraints, 768);
    assert_eq!(source_r1cs.shape.structural_counter_carry_bit_constraints, 189);
    assert_eq!(source_r1cs.shape.canonical_field_lane_count, 80);
    assert_eq!(source_r1cs.shape.canonical_field_lane_constraints, 5040);
    assert!(source_r1cs.shape.shell_constraints() < source_r1cs.shape.constraint_count);
    assert!(
        source_r1cs.shape.poseidon_digest_recomputation_constraints > 0,
        "source R1CS must recompute at least one Construction-2 Poseidon2 digest"
    );
    assert_eq!(
        source_r1cs.shape.nifs_v_verifier_constraints, 0,
        "source R1CS must keep NIFS.V missing until the verifier-shaped F' body is encoded"
    );
    assert_eq!(source_r1cs.shape.authority_constraints(), 0);
    assert!(
        !source_r1cs.shape.has_proof_authority(),
        "low-norm source shell alone must not be promoted to recursive authority"
    );
    let expected_vars = source.len()
        + 446
        + source_r1cs.shape.canonical_field_lane_aux_bits
        + source_r1cs.shape.poseidon_digest_recomputation_aux_bits;
    assert_eq!(source_r1cs.shape.variable_count, expected_vars);
    assert_eq!(source_r1cs.witness.len(), source_r1cs.shape.variable_count);
    assert!(
        source_r1cs.is_satisfied(),
        "native F' low-norm source R1CS witness must satisfy the source-link shell; first_bad={:?}",
        source_r1cs.first_unsatisfied_row()
    );
    assert!(DirectCcsFPrimeLowNormSourceR1cs::from_native_advice(
        &advice,
        params.kappa as u64 + 1,
        1,
        params.k_rho as u64
    )
    .is_err());
    let bits = advice.compact_image().x_out.field_image();
    let wrong_kappa_r1cs = DirectCcsFPrimeLowNormSourceR1cs::from_source_image(
        &source,
        &bits,
        params.kappa as u64 + 1,
        1,
        params.k_rho as u64,
    )
    .unwrap();
    assert!(!wrong_kappa_r1cs.is_satisfied());
    let mut tampered = source_r1cs.clone();
    let source_start = tampered.shape.public_input_len;
    tampered.witness[source_start + source.compact_x_out_bit_offset()] += F::ONE;
    assert!(
        !tampered.is_satisfied(),
        "source R1CS must reject tampering with compact x_out bits"
    );
    let mut tampered = source_r1cs.clone();
    tampered.witness[source_start + source.construction2_u_in_x_i_bit_offset()] += F::ONE;
    assert!(
        !tampered.is_satisfied(),
        "source R1CS must reject tampering with Construction-2 input x_i bits"
    );
    let mut tampered = source_r1cs.clone();
    tampered.witness[source_start + source.current_boundary_out_digest_bit_offset()] += F::ONE;
    assert!(
        !tampered.is_satisfied(),
        "source R1CS must reject tampering with the recomputed current boundary output digest"
    );
    let mut tampered = source_r1cs.clone();
    let bit = source_start + source.construction2_u_in_commitment_digest_bit_offset();
    tampered.witness[bit] = F::ONE - tampered.witness[bit];
    assert!(
        !tampered.is_satisfied(),
        "source R1CS must reject binary tampering with the recomputed Construction-2 input fresh digest preimage"
    );
    let mut tampered = source_r1cs.clone();
    tampered.witness[source_start + source.chunk_count_out_bit_offset()] += F::ONE;
    assert!(
        !tampered.is_satisfied(),
        "source R1CS must reject tampering with chunk_count_out = chunk_count_in + 1"
    );
    let mut tampered = source_r1cs.clone();
    tampered.witness[source_start + source.output_ce_claims_bit_offset()] += F::ONE;
    assert!(
        !tampered.is_satisfied(),
        "source R1CS must reject tampering with output_CE = incoming_CE + fresh_CCS"
    );
    let mut tampered = source_r1cs.clone();
    tampered.witness[source_start + source.fresh_claims_bit_offset()] += F::ONE;
    tampered.witness[source_start + source.output_ce_claims_bit_offset()] += F::ONE;
    assert!(
        !tampered.is_satisfied(),
        "source R1CS must reject non-fixed fresh claim arity"
    );
    for offset in [
        source.final_ce_claims_bit_offset(),
        source.nifs_chunk_index_bit_offset(),
        source.nifs_fresh_claims_bit_offset(),
        source.nifs_incoming_ce_claims_bit_offset(),
        source.nifs_pi_ccs_outputs_bit_offset(),
        source.nifs_final_ce_claims_bit_offset(),
    ] {
        let mut tampered = source_r1cs.clone();
        tampered.witness[source_start + offset] += F::ONE;
        assert!(!tampered.is_satisfied(), "tampered F' NIFS payload accepted");
    }
    let mut tampered = source_r1cs.clone();
    let lane = source_start + source.construction2_u_in_commitment_digest_bit_offset();
    for bit in 0..64 {
        tampered.witness[lane + bit] = if bit == 0 || bit >= 32 { F::ONE } else { F::ZERO };
    }
    assert!(
        !tampered.is_satisfied(),
        "source R1CS must reject non-canonical field lanes"
    );
    assert_eq!(
        summary.low_norm_f_prime_source_r1cs_constraints,
        source_r1cs.shape.constraint_count
    );
    assert_eq!(summary.low_norm_f_prime_source_private_bits, source.len());
    assert_eq!(
        summary.low_norm_f_prime_source_structural_counter_constraints,
        source_r1cs.shape.structural_counter_constraints
    );
    assert_eq!(
        summary.low_norm_f_prime_source_authority_constraints,
        source_r1cs.shape.authority_constraints()
    );
    assert_eq!(
        summary.low_norm_f_prime_source_poseidon_digest_recomputation_constraints,
        source_r1cs.shape.poseidon_digest_recomputation_constraints
    );
    assert_eq!(
        summary.low_norm_f_prime_source_nifs_v_verifier_constraints,
        source_r1cs.shape.nifs_v_verifier_constraints
    );
    let program = source_r1cs
        .to_direct_ccs_program(&params)
        .expect("source R1CS converts to direct CCS program");
    let source_log = make_ajtai_module_for_cols(&params, source_r1cs.shape.variable_count.div_ceil(D));
    let step = source_r1cs
        .to_direct_ccs_step(&program, &source_log, "direct_f_prime_source")
        .expect("source R1CS witness is low-norm packable");
    assert_eq!(step.into_step_input().mcs.m_in, source_r1cs.shape.public_input_len);
    assert_eq!(
        &advice.construction2_u_in().x_i,
        &step_image.compact_image().x_in,
        "native direct F' advice must bind the input Construction-2 instance to x_i"
    );
    assert_eq!(
        &step_image.construction2_u_out().x_i,
        &step_image.compact_image().x_out,
        "native direct F' evaluation must bind the output Construction-2 instance to x_out"
    );
    step_image
        .terminal_public_image()
        .validate_final_construction2_public_boundary()
        .expect("native direct F' step image exports a valid terminal public image");
}

#[test]
#[ignore = "Spartan recursive terminal compression is intentionally expensive; run explicitly when measuring the direct recursive proof surface."]
fn direct_recursive_ivc_compresses_terminal_boundary_and_binds_accumulator_digest() {
    let params = NeoParams::goldilocks_auto_r1cs_ccs(1).expect("valid Fibonacci params");
    let ccs = fibonacci_ccs();
    let log = make_ajtai_module(&params);
    let program = DirectCcsProgram::new_with_public_input_len(&params, &ccs, 3).expect("fixed direct public input");
    let recursive = DirectCcsRecursiveIvcState::new_with_canonical_zero_carry(program)
        .expect("direct recursive state")
        .append_step(
            DirectCcsStep::new(step(&log, "recursive_direct_0", 1, 2, 3)),
            &log,
            ajtai_mixers(),
        )
        .expect("append first recursive direct step");
    let summary = recursive.summary();
    assert_eq!(summary.semantic_chunks, 1);
    assert_eq!(summary.semantic_steps, 1);
    assert_eq!(
        summary.terminal_chunks_synthesized, 1,
        "recursive terminal compression must synthesize one latest F' chunk"
    );
    assert_eq!(summary.carried_semantic_ce_claims, params.k_rho as usize);
    assert_eq!(summary.folded_f_prime_r2_steps, 0);
    assert_eq!(
        summary.carried_f_prime_ce_claims, 0,
        "single-step terminal compression has no folded prior F' accumulator"
    );
    assert!(
        summary.standalone_proof_authority_ready,
        "a single-step direct proof is the Construction-2 base case and needs no folded prior F' chain"
    );
    assert!(summary.native_f_prime_evaluator_available);
    assert!(
        !summary.f_prime_encoder_required,
        "base case has no prior F' step to encode"
    );
    assert!(
        !summary.f_prime_encoder_available,
        "base case should not claim a low-norm direct F' encoder exists"
    );
    assert_eq!(summary.f_prime_encoder_blocker, None);
    assert!(
        summary.compact_f_prime_image_digest.is_some(),
        "base case still has a compact latest F' image"
    );
    assert!(summary.low_norm_f_prime_source_available);
    assert!(summary.low_norm_f_prime_source_len > 0);
    assert!(summary.low_norm_f_prime_source_digest.is_some());
    assert_eq!(
        summary.low_norm_f_prime_source_authority_constraints, 0,
        "base case source digest linkage is diagnostic until NIFS.V rows exist"
    );
    assert!(summary.low_norm_f_prime_source_poseidon_digest_recomputation_constraints > 0);
    assert_eq!(summary.low_norm_f_prime_source_nifs_v_verifier_constraints, 0);
    assert_eq!(recursive.direct_state().final_state().chunk_count, 1);

    let (snark, vk, perf) = recursive
        .compress_recursive_snark()
        .expect("recursive direct compression");
    snark
        .verify(&vk, snark.public_image())
        .expect("recursive direct proof verifies");
    verify_direct_ccs_recursive_ivc_snark_public(&vk, snark.public_image(), &snark)
        .expect("recursive direct public verifier accepts the baseline proof");
    snark
        .public_image()
        .expected_digest()
        .expect("recursive public image digest is well formed");
    assert_eq!(
        snark.public_image().proven_accumulator_digest,
        snark
            .public_image()
            .terminal_public_image
            .accumulator_out_digest,
        "recursive public image must bind the same accumulator digest proven by the terminal F' proof"
    );
    assert_eq!(
        snark.public_image().proven_chunk_count,
        snark.public_image().terminal_public_image.chunk_count_out
    );
    assert_eq!(
        snark.public_image().proven_step_count,
        snark.public_image().terminal_public_image.step_count_out
    );
    assert!(
        snark.f_prime_final_claims().is_empty(),
        "base F' authority comes from the verifier-key-bound default accumulator digest, not a carried CE proof"
    );
    assert_eq!(perf.f_prime_final_ce_constraints, 0);
    assert_eq!(perf.f_prime_final_ce_proof_bytes, 0);
    assert_eq!(
        perf.terminal.construction2_fold_constraints, 0,
        "first recursive step starts from the verifier-key-bound default F' accumulator"
    );
    assert!(snark.f_prime_chain_snark().is_none());
    assert_eq!(perf.f_prime_chain_constraints, 0);
    assert_eq!(perf.f_prime_chain_proof_bytes, 0);
    assert_eq!(perf.terminal_proof_bytes, perf.total_proof_bytes);
}

#[test]
fn direct_recursive_ivc_multi_step_compression_refuses_terminal_source_image_authority() {
    let params = NeoParams::goldilocks_auto_r1cs_ccs(1).expect("valid Fibonacci params");
    let ccs = fibonacci_ccs();
    let log = make_ajtai_module(&params);
    let program = DirectCcsProgram::new_with_public_input_len(&params, &ccs, 3).expect("fixed direct public input");
    let recursive = DirectCcsRecursiveIvcState::new_with_canonical_zero_carry(program)
        .expect("direct recursive state")
        .append_step(
            DirectCcsStep::new(step(&log, "recursive_direct_0", 1, 2, 3)),
            &log,
            ajtai_mixers(),
        )
        .expect("append first recursive direct step")
        .append_step(
            DirectCcsStep::new(step(&log, "recursive_direct_1", 2, 3, 5)),
            &log,
            ajtai_mixers(),
        )
        .expect("append second recursive direct step");

    let err = match recursive.compress_recursive_snark() {
        Ok(_) => panic!("multi-step direct recursive compression must refuse terminal source-image authority"),
        Err(err) => err,
    };
    assert!(
        !recursive.summary().standalone_proof_authority_ready,
        "multi-step direct recursive state must not report standalone authority without folded prior F'"
    );
    assert!(recursive.summary().f_prime_encoder_required);
    assert!(recursive.summary().native_f_prime_evaluator_available);
    assert!(recursive.summary().low_norm_f_prime_source_available);
    assert!(recursive.summary().low_norm_f_prime_source_len > 0);
    assert!(!recursive.summary().f_prime_encoder_available);
    assert_eq!(
        recursive
            .summary()
            .low_norm_f_prime_source_authority_constraints,
        0
    );
    assert!(
        recursive
            .summary()
            .low_norm_f_prime_source_poseidon_digest_recomputation_constraints
            > 0
    );
    assert_eq!(
        recursive
            .summary()
            .low_norm_f_prime_source_nifs_v_verifier_constraints,
        0
    );
    assert!(
        recursive
            .summary()
            .f_prime_encoder_blocker
            .is_some_and(|blocker| blocker.contains("low-norm")),
        "multi-step refusal must surface the missing low-norm direct F' encoder"
    );
    assert!(
        err.to_string()
            .contains("must not fold terminal committed/source-image machinery")
            || err
                .to_string()
                .contains("refuses to fold terminal committed/source-image machinery"),
        "unexpected multi-step direct recursive compression error: {err}"
    );
    assert!(
        err.to_string().contains("low-norm enc(F')"),
        "multi-step refusal must name the missing proof authority, got: {err}"
    );
}

#[test]
fn direct_recursive_multi_step_refusal_happens_before_spartan_synthesis() {
    let params = NeoParams::goldilocks_auto_r1cs_ccs(1).expect("valid Fibonacci params");
    let ccs = fibonacci_ccs();
    let log = make_ajtai_module(&params);
    let program = DirectCcsProgram::new_with_public_input_len(&params, &ccs, 3).expect("fixed direct public input");
    let recursive = DirectCcsRecursiveIvcState::new_with_canonical_zero_carry(program)
        .expect("direct recursive state")
        .append_step(
            DirectCcsStep::new(step(&log, "recursive_direct_0", 1, 2, 3)),
            &log,
            ajtai_mixers(),
        )
        .expect("append first recursive direct step")
        .append_step(
            DirectCcsStep::new(step(&log, "recursive_direct_1", 2, 3, 5)),
            &log,
            ajtai_mixers(),
        )
        .expect("append second recursive direct step");

    let mut trace = Vec::new();
    let err = match recursive.compress_recursive_snark_with_trace(&mut |message| trace.push(message.to_owned())) {
        Ok(_) => panic!("multi-step direct recursive compression must refuse before terminal Spartan synthesis"),
        Err(err) => err,
    };
    assert!(
        err.to_string().contains("low-norm enc(F')"),
        "multi-step refusal must name the missing compact F' relation, got: {err}"
    );
    assert!(
        trace.is_empty(),
        "multi-step refusal must happen before Spartan phases are entered, got trace: {trace:?}"
    );
}

#[test]
#[ignore = "Spartan recursive terminal compression is intentionally expensive; run explicitly when measuring the direct recursive proof surface."]
fn direct_recursive_ivc_public_image_rejects_unbound_accumulator_digest() {
    let params = NeoParams::goldilocks_auto_r1cs_ccs(1).expect("valid Fibonacci params");
    let ccs = fibonacci_ccs();
    let log = make_ajtai_module(&params);
    let program = DirectCcsProgram::new_with_public_input_len(&params, &ccs, 3).expect("fixed direct public input");
    let recursive = DirectCcsRecursiveIvcState::new_with_canonical_zero_carry(program)
        .expect("direct recursive state")
        .append_step(
            DirectCcsStep::new(step(&log, "recursive_direct_0", 1, 2, 3)),
            &log,
            ajtai_mixers(),
        )
        .expect("append first recursive direct step");

    let (snark, vk, _) = recursive
        .compress_recursive_snark()
        .expect("recursive direct compression");
    let mut image = snark.public_image().clone();
    image.proven_accumulator_digest[0] ^= 1;
    assert!(
        image.validate_recursive_boundary().is_err(),
        "recursive boundary must reject an accumulator digest not bound to terminal x_out"
    );
    assert!(
        snark.verify(&vk, &image).is_err(),
        "recursive verifier must reject an accumulator digest not bound to terminal x_out"
    );

    let mut image = snark.public_image().clone();
    image.proven_chunk_count += 1;
    assert!(
        image.validate_recursive_boundary().is_err(),
        "recursive boundary must reject a chunk counter not bound to terminal x_out"
    );
    assert!(
        verify_direct_ccs_recursive_ivc_snark_public(&vk, &image, &snark).is_err(),
        "recursive public verifier must reject a chunk counter not bound to terminal x_out"
    );

    let mut image = snark.public_image().clone();
    image.proven_step_count += 1;
    assert!(
        image.validate_recursive_boundary().is_err(),
        "recursive boundary must reject a step counter not bound to terminal x_out"
    );
    assert!(
        verify_direct_ccs_recursive_ivc_snark_public(&vk, &image, &snark).is_err(),
        "recursive public verifier must reject a step counter not bound to terminal x_out"
    );

    let mut image = snark.public_image().clone();
    image.f_prime_final_ce_claims += 1;
    assert!(
        snark.verify(&vk, &image).is_err(),
        "recursive verifier must reject a final F' CE claim-count mismatch"
    );

    let mut image = snark.public_image().clone();
    image.f_prime_accumulator_base += 1;
    assert!(
        snark.verify(&vk, &image).is_err(),
        "recursive verifier must reject a final F' accumulator base mismatch"
    );

    let mut terminal = snark.public_image().terminal_public_image.clone();
    terminal.accumulator_out_digest[0] ^= 1;
    assert!(
        DirectCcsRecursiveIvcPublicImage::from_terminal_and_f_prime_accumulator(
            terminal,
            [0u8; 32],
            2,
            1,
            params.k_rho as u64
        )
        .is_err(),
        "recursive public image constructor must reject terminal images that fail direct boundary validation"
    );
}

fn tiny_sparse_r1cs_export(variable_count: usize, public_input_len: usize) -> DirectSparseR1csExport {
    assert!(variable_count >= public_input_len);
    let a = CcsMatrix::Csc(CscMat::from_triplets(vec![(0, 0, F::ONE)], 1, variable_count));
    let b = CcsMatrix::Csc(CscMat::from_triplets(vec![(0, 1, F::ONE)], 1, variable_count));
    let c = CcsMatrix::Csc(CscMat::from_triplets(vec![(0, 2, F::ONE)], 1, variable_count));
    let mut witness = vec![F::ZERO; variable_count];
    witness[0] = F::ONE;
    witness[1] = F::ONE;
    witness[2] = F::ONE;
    DirectSparseR1csExport {
        a,
        b,
        c,
        witness,
        public_input_len,
        constraint_count: 1,
        variable_count,
    }
}

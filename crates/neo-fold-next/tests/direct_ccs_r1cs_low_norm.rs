use neo_ajtai::{
    has_global_pp_for_dims, s_mul_add, scale_commitment_add_inplace, set_global_pp_seeded, AjtaiSModule, Commitment,
};
use neo_ccs::{CcsMatrix, CscMat, Mat};
use neo_fold_next::core::prover::CommitmentMixers;
use neo_fold_next::direct_ccs::{
    lower_sparse_r1cs_export_to_low_norm, lower_sparse_r1cs_export_to_low_norm_program_and_step,
    verify_direct_ccs_recursive_ivc_snark_public, DirectCcsProgram, DirectCcsRecursiveIvcState, DirectCcsStep,
    DirectLowNormLaneKind, DirectR1csLowNormLayout, DirectSparseR1csExport,
};
use neo_math::ring::Rq as RqEl;
use neo_math::{D, F};
use neo_params::NeoParams;
use p3_field::PrimeCharacteristicRing;

fn product_export(x: u64, y: u64) -> DirectSparseR1csExport {
    let z = x * y;
    DirectSparseR1csExport {
        a: CcsMatrix::Csc(CscMat::from_triplets(vec![(0, 1, F::ONE)], 1, 4)),
        b: CcsMatrix::Csc(CscMat::from_triplets(vec![(0, 2, F::ONE)], 1, 4)),
        c: CcsMatrix::Csc(CscMat::from_triplets(vec![(0, 3, F::ONE)], 1, 4)),
        witness: vec![F::ONE, F::from_u64(x), F::from_u64(y), F::from_u64(z)],
        public_input_len: 1,
        constraint_count: 1,
        variable_count: 4,
    }
}

fn sparse_r1cs_is_satisfied(export: &DirectSparseR1csExport) -> bool {
    let a = matrix_mul(&export.a, &export.witness);
    let b = matrix_mul(&export.b, &export.witness);
    let c = matrix_mul(&export.c, &export.witness);
    a.iter()
        .zip(b.iter())
        .zip(c.iter())
        .all(|((a, b), c)| *a * *b == *c)
}

fn matrix_mul(matrix: &CcsMatrix<F>, witness: &[F]) -> Vec<F> {
    match matrix {
        CcsMatrix::Identity { n } => witness[..*n].to_vec(),
        CcsMatrix::Csc(csc) => {
            let mut out = vec![F::ZERO; csc.nrows];
            for col in 0..csc.ncols {
                let value = witness[col];
                if value == F::ZERO {
                    continue;
                }
                for idx in csc.col_ptr[col]..csc.col_ptr[col + 1] {
                    out[csc.row_idx[idx]] += csc.vals[idx] * value;
                }
            }
            out
        }
    }
}

#[test]
fn r1cs_low_norm_lowering_turns_arbitrary_field_witness_into_binary_direct_shape() {
    let export = product_export(1 << 55, 1);
    let params = NeoParams::goldilocks_auto_r1cs_ccs(export.constraint_count).expect("params");
    assert!(!export.low_norm_report(&params, 4).low_norm_packable);

    let layout = DirectR1csLowNormLayout::conservative_for_export(&export);
    let lowered = lower_sparse_r1cs_export_to_low_norm(&export, &layout).expect("lower arbitrary R1CS");

    assert!(sparse_r1cs_is_satisfied(&lowered));
    assert!(lowered.low_norm_report(&params, 4).low_norm_packable);
    assert_eq!(lowered.public_input_len, 1);
    assert!(lowered.variable_count > export.variable_count);
    assert!(lowered.constraint_count > export.constraint_count);
}

#[test]
fn r1cs_low_norm_lowering_rejects_false_bit_classification() {
    let export = product_export(7, 9);
    let layout = DirectR1csLowNormLayout::new(
        export.public_input_len,
        vec![
            DirectLowNormLaneKind::Bit,
            DirectLowNormLaneKind::Bit,
            DirectLowNormLaneKind::Field,
            DirectLowNormLaneKind::Field,
        ],
    )
    .expect("layout");
    let err = lower_sparse_r1cs_export_to_low_norm(&export, &layout).expect_err("bit lane must reject value 7");
    assert!(
        err.to_string().contains("bit lane at original column 1"),
        "unexpected error: {err}"
    );
}

#[test]
fn r1cs_low_norm_lowering_builds_direct_step_and_appends() {
    let export = product_export(1 << 55, 1);
    let layout = DirectR1csLowNormLayout::conservative_for_export(&export);
    let lowered_preview = lower_sparse_r1cs_export_to_low_norm(&export, &layout).expect("lower arbitrary R1CS");
    let log = make_ajtai_module_for_cols(18, lowered_preview.variable_count.div_ceil(D));
    let (lowered, program, step) =
        lower_sparse_r1cs_export_to_low_norm_program_and_step(&export, &layout, &log, "low_norm_product")
            .expect("lowered R1CS converts to direct CCS");

    assert!(sparse_r1cs_is_satisfied(&lowered));
    assert!(
        lowered
            .low_norm_report(program.params(), 4)
            .low_norm_packable
    );

    let recursive =
        DirectCcsRecursiveIvcState::new_with_canonical_zero_carry(program.clone()).expect("direct recursive state");
    let recursive = recursive
        .append_step(step, &log, ajtai_mixers())
        .expect("append lowered direct R1CS step");
    let summary = recursive.summary_with_verifier_body_measurement();
    assert_eq!(summary.semantic.chunks, 1);
    assert_eq!(summary.semantic.steps, 1);
    assert_eq!(summary.semantic.carried_ce_claims, program.params().k_rho as usize);
    assert_eq!(summary.f_prime.folded_r2_steps, 0);

    assert!(summary.f_prime.verifier_body.measured);
    assert!(!summary.f_prime.verifier_body.measure_skipped);
    assert_eq!(summary.f_prime.verifier_body.final_ce_relation_constraints, 0);
    assert!(summary.f_prime.verifier_body.nifs.constraints > 0);
    assert!(summary.f_prime.verifier_body.nifs.chunk_meta_constraints > 0);
    assert!(summary.f_prime.verifier_body.nifs.pi_ccs_constraints > 0);
    assert!(summary.f_prime.verifier_body.nifs.pi_rlc_constraints > 0);
    assert!(summary.f_prime.verifier_body.nifs.pi_dec_constraints > 0);
    assert_eq!(
        summary.f_prime.verifier_body.nifs.constraints,
        summary.f_prime.verifier_body.nifs.chunk_meta_constraints
            + summary.f_prime.verifier_body.nifs.pi_ccs_constraints
            + summary.f_prime.verifier_body.nifs.pi_rlc_constraints
            + summary.f_prime.verifier_body.nifs.pi_dec_constraints
    );
    assert!(summary.f_prime.verifier_body.public_link_constraints > 0);
}

#[test]
fn r1cs_low_norm_two_step_recursive_state_refuses_missing_f_prime_authority() {
    let (program, log, first_step, second_step) = direct_low_norm_product_two_step_fixture();

    let recursive = DirectCcsRecursiveIvcState::new_with_canonical_zero_carry(program)
        .expect("direct recursive state")
        .append_step(first_step, &log, ajtai_mixers())
        .expect("append first lowered direct R1CS step")
        .append_step(second_step, &log, ajtai_mixers())
        .expect("append second lowered direct R1CS step");

    let summary = recursive.summary_with_verifier_body_measurement();
    assert_eq!(summary.semantic.chunks, 2);
    assert_eq!(summary.semantic.steps, 2);
    assert_eq!(
        summary.f_prime.folded_r2_steps, 0,
        "two-step low-norm append must not claim folded F' authority until the authority relation is real: chunks={} source_r1cs_constraints={} source_r1cs_variables={} verifier_constraints={} row_cap={} blocker={:?}",
        summary.semantic.chunks,
        summary.f_prime.low_norm_source.r1cs.constraints,
        summary.f_prime.low_norm_source.r1cs.variables,
        summary.f_prime.verifier_body.constraints,
        summary.f_prime.exact_encoder_row_cap,
        summary.proof.encoder_blocker
    );
    assert!(
        !summary.proof.standalone_authority_ready,
        "two-step direct recursion must refuse standalone proof authority without folded F'"
    );
    assert!(
        summary.f_prime.carried_ce_claims == 0,
        "missing folded F' authority must not expose digest-only carried CE claims"
    );
    assert!(summary.f_prime.native_evaluator_available);
    assert!(summary.f_prime.low_norm_source.available);
    assert!(summary.f_prime.verifier_body.measured);
    assert!(
        summary.f_prime.verifier_body.constraints > summary.f_prime.exact_encoder_row_cap,
        "current exact verifier-body encoder should fail for a real size reason: constraints={} cap={} blocker={:?}",
        summary.f_prime.verifier_body.constraints,
        summary.f_prime.exact_encoder_row_cap,
        summary.proof.encoder_blocker
    );
    let err = match recursive.compress_recursive_snark() {
        Ok(_) => panic!("multi-step direct recursive compression must reject missing F' authority"),
        Err(err) => err,
    };
    assert!(
        err.to_string()
            .contains("verifier-shaped direct F' body exceeds the exact low-norm encoder size gate"),
        "unexpected compression error: {err}"
    );
}

#[test]
fn r1cs_low_norm_lowered_product_reports_exact_encoder_size_blocker() {
    let (program, log, first_step, second_step) = lowered_product_two_step_fixture();

    let recursive = DirectCcsRecursiveIvcState::new_with_canonical_zero_carry(program)
        .expect("direct recursive state")
        .append_step(first_step, &log, ajtai_mixers())
        .expect("append first lowered direct R1CS step")
        .append_step(second_step, &log, ajtai_mixers())
        .expect("append second lowered direct R1CS step");

    let summary = recursive.summary_with_verifier_body_measurement();
    assert_eq!(summary.semantic.chunks, 2);
    assert_eq!(summary.semantic.steps, 2);
    assert_eq!(
        summary.f_prime.folded_r2_steps, 0,
        "lowered arbitrary-field product should not silently claim folded F' authority while the exact verifier body is over the gate"
    );
    assert!(!summary.proof.standalone_authority_ready);
    assert!(summary.f_prime.encoder_required);
    assert_eq!(
        summary.proof.encoder_blocker,
        Some("verifier-shaped direct F' body exceeds the exact low-norm encoder size gate")
    );
    assert!(
        summary.f_prime.verifier_body.constraints > summary.f_prime.exact_encoder_row_cap,
        "expected the lowered product blocker to be a real size limit: constraints={} cap={}",
        summary.f_prime.verifier_body.constraints,
        summary.f_prime.exact_encoder_row_cap
    );
}

#[test]
#[ignore = "Future target: runs the real recursive Spartan compression path once the compact F' authority relation exists."]
fn r1cs_low_norm_recursive_snark_target_compresses_f_prime_chain_and_rejects_tampering() {
    let (program, log, first_step, second_step) = direct_low_norm_product_two_step_fixture();

    let recursive = DirectCcsRecursiveIvcState::new_with_canonical_zero_carry(program)
        .expect("direct recursive state")
        .append_step(first_step, &log, ajtai_mixers())
        .expect("append first lowered direct R1CS step")
        .append_step(second_step, &log, ajtai_mixers())
        .expect("append second lowered direct R1CS step");
    let summary = recursive.summary();
    assert_eq!(summary.f_prime.folded_r2_steps, 1);
    assert!(summary.proof.standalone_authority_ready);

    let (snark, vk, perf) = recursive
        .compress_recursive_snark()
        .expect("recursive direct compression with folded F' authority");
    snark
        .verify(&vk, snark.public_image())
        .expect("recursive direct proof verifies");
    verify_direct_ccs_recursive_ivc_snark_public(&vk, snark.public_image(), &snark)
        .expect("recursive direct public verifier accepts the honest proof");

    assert!(
        snark.f_prime_chain_snark().is_some(),
        "multi-step compression must include the folded F' chain SNARK"
    );
    assert!(
        !snark.f_prime_final_claims().is_empty(),
        "multi-step compression must carry the final F' CE claims"
    );
    assert!(perf.f_prime_chain.is_some());
    assert!(perf.f_prime_chain_constraints > 0);
    assert!(perf.f_prime_chain_proof_bytes > 0);
    assert!(perf.f_prime_final_ce_constraints > 0);
    assert!(perf.f_prime_final_ce_proof_bytes > 0);

    let mut image = snark.public_image().clone();
    image.f_prime_final_ce_claims += 1;
    assert!(
        image.validate_recursive_boundary().is_ok(),
        "claim-count tampering keeps the structural boundary well formed, so verifier binding must reject it"
    );
    assert!(
        verify_direct_ccs_recursive_ivc_snark_public(&vk, &image, &snark).is_err(),
        "public verifier must reject a mutated final F' CE claim count"
    );

    let mut image = snark.public_image().clone();
    image.proven_f_prime_accumulator_digest[0] ^= 1;
    assert!(
        image.validate_recursive_boundary().is_err(),
        "recursive boundary must reject a mutated folded F' accumulator digest"
    );
    assert!(
        snark.verify(&vk, &image).is_err(),
        "recursive verifier must reject a mutated folded F' accumulator digest"
    );
}

fn direct_low_norm_product_two_step_fixture() -> (DirectCcsProgram, AjtaiSModule, DirectCcsStep, DirectCcsStep) {
    let first_export = product_export(2, 3);
    let second_export = product_export(3, 4);

    assert!(sparse_r1cs_is_satisfied(&first_export));
    assert!(sparse_r1cs_is_satisfied(&second_export));
    assert_eq!(first_export.constraint_count, second_export.constraint_count);
    assert_eq!(first_export.variable_count, second_export.variable_count);
    assert_eq!(first_export.public_input_len, second_export.public_input_len);

    let log = make_ajtai_module_for_cols(18, first_export.variable_count.div_ceil(D));
    let program = first_export
        .to_direct_ccs_program()
        .expect("low-norm product converts to direct CCS program");
    assert!(
        first_export
            .low_norm_report(program.params(), 4)
            .low_norm_packable
    );
    assert!(
        second_export
            .low_norm_report(program.params(), 4)
            .low_norm_packable
    );
    let first_step = first_export
        .into_direct_ccs_step(&program, &log, "direct_low_norm_product_0")
        .expect("first low-norm product converts to direct CCS step");
    let second_step = second_export
        .into_direct_ccs_step(&program, &log, "direct_low_norm_product_1")
        .expect("second low-norm product converts to direct CCS step");

    (program, log, first_step, second_step)
}

fn lowered_product_two_step_fixture() -> (DirectCcsProgram, AjtaiSModule, DirectCcsStep, DirectCcsStep) {
    let first_export = product_export(1 << 55, 1);
    let second_export = product_export((1 << 55) + 1, 1);
    let layout = DirectR1csLowNormLayout::conservative_for_export(&first_export);
    let first_lowered = lower_sparse_r1cs_export_to_low_norm(&first_export, &layout).expect("lower first product R1CS");
    let second_lowered =
        lower_sparse_r1cs_export_to_low_norm(&second_export, &layout).expect("lower second product R1CS");

    assert!(sparse_r1cs_is_satisfied(&first_lowered));
    assert!(sparse_r1cs_is_satisfied(&second_lowered));
    assert_eq!(first_lowered.constraint_count, second_lowered.constraint_count);
    assert_eq!(first_lowered.variable_count, second_lowered.variable_count);
    assert_eq!(first_lowered.public_input_len, second_lowered.public_input_len);

    let log = make_ajtai_module_for_cols(18, first_lowered.variable_count.div_ceil(D));
    let program = first_lowered
        .to_direct_ccs_program()
        .expect("lowered product converts to direct CCS program");
    let first_step = first_lowered
        .into_direct_ccs_step(&program, &log, "low_norm_product_0")
        .expect("first lowered product converts to direct CCS step");
    let second_step = second_lowered
        .into_direct_ccs_step(&program, &log, "low_norm_product_1")
        .expect("second lowered product converts to direct CCS step");

    (program, log, first_step, second_step)
}

fn make_ajtai_module_for_cols(kappa: usize, cols: usize) -> AjtaiSModule {
    if !has_global_pp_for_dims(D, cols) {
        let mut seed = [0u8; 32];
        seed[..8].copy_from_slice(&0x4452_314c_4f57_4e4d_u64.to_le_bytes());
        match set_global_pp_seeded(D, kappa, cols, seed) {
            Ok(()) => {}
            Err(_err) if has_global_pp_for_dims(D, cols) => {}
            Err(err) => panic!("Ajtai global setup: {err}"),
        }
    }
    AjtaiSModule::from_global_for_dims(D, cols).expect("Ajtai global module")
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

fn rot_matrix_to_rq(mat: &Mat<F>) -> RqEl {
    use neo_math::ring::cf_inv;

    let mut coeffs = [F::ZERO; D];
    for i in 0..D {
        coeffs[i] = mat[(i, 0)];
    }
    cf_inv(coeffs)
}

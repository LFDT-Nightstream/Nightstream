use neo_ajtai::{
    has_global_pp_for_dims, s_mul_add, scale_commitment_add_inplace, set_global_pp_seeded, AjtaiSModule, Commitment,
};
use neo_ccs::{CcsMatrix, CscMat, Mat};
use neo_fold_next::prover::CommitmentMixers;
use neo_fold_next::{
    lower_sparse_r1cs_export_to_low_norm, lower_sparse_r1cs_export_to_low_norm_program_and_step,
    DirectCcsRecursiveIvcState, DirectLowNormLaneKind, DirectR1csLowNormLayout, DirectSparseR1csExport,
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
    assert_eq!(summary.semantic_chunks, 1);
    assert_eq!(summary.semantic_steps, 1);
    assert_eq!(summary.carried_semantic_ce_claims, program.params().k_rho as usize);
    assert_eq!(summary.folded_f_prime_r2_steps, 0);

    assert!(summary.f_prime_verifier_body_measured);
    assert!(!summary.f_prime_verifier_body_measure_skipped);
    assert_eq!(summary.f_prime_verifier_body_final_ce_relation_constraints, 0);
    assert!(summary.f_prime_verifier_body_nifs_constraints > 0);
    assert!(summary.f_prime_verifier_body_public_link_constraints > 0);
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

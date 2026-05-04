use bellpepper_core::{num::AllocatedNum, ConstraintSystem, SynthesisError};
use neo_ajtai::{
    has_global_pp_for_dims, s_mul_add, scale_commitment_add_inplace, set_global_pp_seeded, AjtaiSModule, Commitment,
};
use neo_ccs::Mat;
use neo_fold_next::direct_sparse_r1cs_export_from_spartan_circuit;
use neo_fold_next::prover::CommitmentMixers;
use neo_fold_next::DirectCcsRecursiveIvcState;
use neo_math::ring::Rq as RqEl;
use neo_math::{D, F};
use p3_field::PrimeCharacteristicRing;
use spartan2::provider::{goldi::F as SpartanF, GoldilocksP3MerkleMleEngine};
use spartan2::traits::circuit::SpartanCircuit;

#[derive(Clone)]
struct PublicAndCircuit {
    a: u64,
    b: u64,
    c: u64,
}

impl SpartanCircuit<GoldilocksP3MerkleMleEngine> for PublicAndCircuit {
    fn public_values(&self) -> Result<Vec<SpartanF>, SynthesisError> {
        Ok(vec![
            SpartanF::from_canonical_u64(self.a),
            SpartanF::from_canonical_u64(self.b),
            SpartanF::from_canonical_u64(self.c),
        ])
    }

    fn shared<CS: ConstraintSystem<SpartanF>>(
        &self,
        _: &mut CS,
    ) -> Result<Vec<AllocatedNum<SpartanF>>, SynthesisError> {
        Ok(Vec::new())
    }

    fn precommitted<CS: ConstraintSystem<SpartanF>>(
        &self,
        _: &mut CS,
        _: &[AllocatedNum<SpartanF>],
    ) -> Result<Vec<AllocatedNum<SpartanF>>, SynthesisError> {
        Ok(Vec::new())
    }

    fn num_challenges(&self) -> usize {
        0
    }

    fn synthesize<CS: ConstraintSystem<SpartanF>>(
        &self,
        cs: &mut CS,
        _: &[AllocatedNum<SpartanF>],
        _: &[AllocatedNum<SpartanF>],
        _: Option<&[SpartanF]>,
    ) -> Result<(), SynthesisError> {
        let a = AllocatedNum::alloc_input(cs.namespace(|| "a"), || Ok(SpartanF::from_canonical_u64(self.a)))?;
        let b = AllocatedNum::alloc_input(cs.namespace(|| "b"), || Ok(SpartanF::from_canonical_u64(self.b)))?;
        let c = AllocatedNum::alloc_input(cs.namespace(|| "c"), || Ok(SpartanF::from_canonical_u64(self.c)))?;
        cs.enforce(
            || "a_times_b_eq_c",
            |lc| lc + a.get_variable(),
            |lc| lc + b.get_variable(),
            |lc| lc + c.get_variable(),
        );
        for (label, value) in [("a", &a), ("b", &b), ("c", &c)] {
            cs.enforce(
                || format!("{label}_bit"),
                |lc| lc + value.get_variable(),
                |lc| lc + value.get_variable() - CS::one(),
                |lc| lc,
            );
        }
        Ok(())
    }
}

#[derive(Clone)]
struct LargeAuxCircuit {
    value: u64,
}

impl SpartanCircuit<GoldilocksP3MerkleMleEngine> for LargeAuxCircuit {
    fn public_values(&self) -> Result<Vec<SpartanF>, SynthesisError> {
        Ok(Vec::new())
    }

    fn shared<CS: ConstraintSystem<SpartanF>>(
        &self,
        _: &mut CS,
    ) -> Result<Vec<AllocatedNum<SpartanF>>, SynthesisError> {
        Ok(Vec::new())
    }

    fn precommitted<CS: ConstraintSystem<SpartanF>>(
        &self,
        _: &mut CS,
        _: &[AllocatedNum<SpartanF>],
    ) -> Result<Vec<AllocatedNum<SpartanF>>, SynthesisError> {
        Ok(Vec::new())
    }

    fn num_challenges(&self) -> usize {
        0
    }

    fn synthesize<CS: ConstraintSystem<SpartanF>>(
        &self,
        cs: &mut CS,
        _: &[AllocatedNum<SpartanF>],
        _: &[AllocatedNum<SpartanF>],
        _: Option<&[SpartanF]>,
    ) -> Result<(), SynthesisError> {
        let value = AllocatedNum::alloc(cs.namespace(|| "large_aux"), || {
            Ok(SpartanF::from_canonical_u64(self.value))
        })?;
        cs.enforce(
            || "large_aux_identity",
            |lc| lc + value.get_variable(),
            |lc| lc + CS::one(),
            |lc| lc + value.get_variable(),
        );
        Ok(())
    }
}

#[derive(Clone)]
struct ChallengeCircuit;

impl SpartanCircuit<GoldilocksP3MerkleMleEngine> for ChallengeCircuit {
    fn public_values(&self) -> Result<Vec<SpartanF>, SynthesisError> {
        Ok(Vec::new())
    }

    fn shared<CS: ConstraintSystem<SpartanF>>(
        &self,
        _: &mut CS,
    ) -> Result<Vec<AllocatedNum<SpartanF>>, SynthesisError> {
        Ok(Vec::new())
    }

    fn precommitted<CS: ConstraintSystem<SpartanF>>(
        &self,
        _: &mut CS,
        _: &[AllocatedNum<SpartanF>],
    ) -> Result<Vec<AllocatedNum<SpartanF>>, SynthesisError> {
        Ok(Vec::new())
    }

    fn num_challenges(&self) -> usize {
        1
    }

    fn synthesize<CS: ConstraintSystem<SpartanF>>(
        &self,
        _: &mut CS,
        _: &[AllocatedNum<SpartanF>],
        _: &[AllocatedNum<SpartanF>],
        _: Option<&[SpartanF]>,
    ) -> Result<(), SynthesisError> {
        Ok(())
    }
}

#[derive(Clone)]
struct MismatchedPublicValuesCircuit {
    inner: PublicAndCircuit,
}

impl SpartanCircuit<GoldilocksP3MerkleMleEngine> for MismatchedPublicValuesCircuit {
    fn public_values(&self) -> Result<Vec<SpartanF>, SynthesisError> {
        Ok(vec![
            SpartanF::from_canonical_u64(self.inner.a),
            SpartanF::from_canonical_u64(self.inner.b),
            SpartanF::from_canonical_u64(self.inner.c + 1),
        ])
    }

    fn shared<CS: ConstraintSystem<SpartanF>>(
        &self,
        cs: &mut CS,
    ) -> Result<Vec<AllocatedNum<SpartanF>>, SynthesisError> {
        self.inner.shared(cs)
    }

    fn precommitted<CS: ConstraintSystem<SpartanF>>(
        &self,
        cs: &mut CS,
        shared: &[AllocatedNum<SpartanF>],
    ) -> Result<Vec<AllocatedNum<SpartanF>>, SynthesisError> {
        self.inner.precommitted(cs, shared)
    }

    fn num_challenges(&self) -> usize {
        self.inner.num_challenges()
    }

    fn synthesize<CS: ConstraintSystem<SpartanF>>(
        &self,
        cs: &mut CS,
        shared: &[AllocatedNum<SpartanF>],
        precommitted: &[AllocatedNum<SpartanF>],
        challenges: Option<&[SpartanF]>,
    ) -> Result<(), SynthesisError> {
        self.inner.synthesize(cs, shared, precommitted, challenges)
    }
}

#[test]
fn bellpepper_spartan_circuit_exports_to_low_norm_direct_r1cs_step() {
    let export = direct_sparse_r1cs_export_from_spartan_circuit(&PublicAndCircuit { a: 1, b: 1, c: 1 })
        .expect("bit circuit exports");
    assert_eq!(export.public_input_len, 4);
    assert_eq!(export.variable_count, 4);
    assert_eq!(export.constraint_count, 4);
    assert_eq!(export.witness, vec![F::ONE, F::ONE, F::ONE, F::ONE]);

    let program = export.to_direct_ccs_program().expect("direct CCS program");
    let report = export.low_norm_report(program.params(), 4);
    assert!(report.low_norm_packable, "bit-lowered export must be low-norm packable");
    assert_eq!(report.violation_count, 0);
    assert!(report.first_violations.is_empty());

    let log = make_ajtai_module_for_cols(program.params().kappa as usize, export.variable_count.div_ceil(D));
    let step = export
        .into_direct_ccs_step(&program, &log, "bellpepper_bit_and")
        .expect("low-norm R1CS export converts to direct CCS step");
    assert_eq!(step.into_step_input().mcs.m_in, 4);
}

#[test]
fn bellpepper_spartan_circuit_export_keeps_low_norm_boundary_explicit() {
    let export = direct_sparse_r1cs_export_from_spartan_circuit(&LargeAuxCircuit { value: 1u64 << 60 })
        .expect("large aux circuit exports as sparse R1CS");
    assert_eq!(export.public_input_len, 1);
    assert_eq!(export.variable_count, 2);

    let program = export.to_direct_ccs_program().expect("direct CCS program");
    let report = export.low_norm_report(program.params(), 4);
    assert!(
        !report.low_norm_packable,
        "large field-valued aux is not foldable as one low-norm CCS entry"
    );
    assert_eq!(report.violation_count, 1);
    assert_eq!(report.first_violations.len(), 1);
    assert_eq!(report.first_violations[0].index, 1);
    assert!(
        !report.first_violations[0].is_public,
        "large value is allocated as a private R1CS auxiliary"
    );

    let log = make_ajtai_module_for_cols(program.params().kappa as usize, export.variable_count.div_ceil(D));
    let err = match export.into_direct_ccs_step(&program, &log, "large_aux") {
        Ok(_) => panic!("arbitrary field-valued aux must not be accepted as low-norm direct CCS witness"),
        Err(err) => err,
    };
    assert!(
        err.to_string().contains("not SuperNeo low-norm packable"),
        "unexpected low-norm rejection: {err}"
    );
    assert!(
        err.to_string().contains("index 1") && err.to_string().contains("private"),
        "low-norm rejection should identify the first non-packable variable: {err}"
    );
}

#[test]
fn bellpepper_spartan_circuit_export_rejects_challenge_inputs() {
    let err = match direct_sparse_r1cs_export_from_spartan_circuit(&ChallengeCircuit) {
        Ok(_) => panic!("challenge-bearing circuits need an explicit public challenge boundary"),
        Err(err) => err,
    };
    assert!(
        err.to_string().contains("challenge inputs"),
        "unexpected challenge export error: {err}"
    );
}

#[test]
fn bellpepper_spartan_circuit_export_rejects_public_values_mismatch() {
    let err = match direct_sparse_r1cs_export_from_spartan_circuit(&MismatchedPublicValuesCircuit {
        inner: PublicAndCircuit { a: 1, b: 1, c: 1 },
    }) {
        Ok(_) => panic!("direct R1CS export must reject stale public_values"),
        Err(err) => err,
    };
    assert!(
        err.to_string().contains("public_values mismatch"),
        "unexpected public_values mismatch error: {err}"
    );
}

#[test]
fn bellpepper_spartan_circuit_export_rejects_unsatisfied_assignment() {
    let err = match direct_sparse_r1cs_export_from_spartan_circuit(&PublicAndCircuit { a: 1, b: 0, c: 1 }) {
        Ok(_) => panic!("direct R1CS export must reject unsatisfied assignments"),
        Err(err) => err,
    };
    assert!(
        err.to_string().contains("does not satisfy row") || err.to_string().contains("witness synthesis failed"),
        "unexpected unsatisfied assignment error: {err}"
    );
}

#[test]
fn bellpepper_spartan_circuit_exports_append_through_direct_recursive_state() {
    let first = direct_sparse_r1cs_export_from_spartan_circuit(&PublicAndCircuit { a: 1, b: 1, c: 1 })
        .expect("first bit circuit exports");
    let second = direct_sparse_r1cs_export_from_spartan_circuit(&PublicAndCircuit { a: 0, b: 1, c: 0 })
        .expect("second same-shape bit circuit exports");
    assert_eq!(first.constraint_count, second.constraint_count);
    assert_eq!(first.variable_count, second.variable_count);
    assert_eq!(first.public_input_len, second.public_input_len);

    let program = first.to_direct_ccs_program().expect("direct CCS program");
    let log = make_ajtai_module_for_cols(program.params().kappa as usize, first.variable_count.div_ceil(D));
    let mut recursive =
        DirectCcsRecursiveIvcState::new_with_canonical_zero_carry(program.clone()).expect("direct recursive state");
    recursive = recursive
        .append_step(
            first
                .into_direct_ccs_step(&program, &log, "bellpepper_bit_and_0")
                .expect("first low-norm step"),
            &log,
            ajtai_mixers(),
        )
        .expect("append first exported R1CS step");
    recursive = recursive
        .append_step(
            second
                .into_direct_ccs_step(&program, &log, "bellpepper_bit_and_1")
                .expect("second low-norm step"),
            &log,
            ajtai_mixers(),
        )
        .expect("append second exported R1CS step");

    let summary = recursive.summary();
    assert_eq!(summary.semantic_chunks, 2);
    assert_eq!(summary.semantic_steps, 2);
    assert_eq!(summary.carried_semantic_ce_claims, program.params().k_rho as usize);
    assert!(summary.native_f_prime_evaluator_available);
    assert!(summary.f_prime_encoder_required);
    assert!(
        !summary.standalone_proof_authority_ready,
        "multi-step direct R1CS append must still wait for low-norm enc(F') proof authority"
    );
}

#[test]
fn bellpepper_spartan_circuit_exported_shape_drift_is_rejected_before_folding() {
    let first = direct_sparse_r1cs_export_from_spartan_circuit(&PublicAndCircuit { a: 1, b: 1, c: 1 })
        .expect("bit circuit exports");
    let drift = direct_sparse_r1cs_export_from_spartan_circuit(&LargeAuxCircuit { value: 1 })
        .expect("different-shape circuit exports");
    let program = first.to_direct_ccs_program().expect("direct CCS program");
    let first_log = make_ajtai_module_for_cols(program.params().kappa as usize, first.variable_count.div_ceil(D));
    let drift_program = drift
        .to_direct_ccs_program()
        .expect("drift direct CCS program");
    let drift_log = make_ajtai_module_for_cols(drift_program.params().kappa as usize, drift.variable_count.div_ceil(D));
    let recursive =
        DirectCcsRecursiveIvcState::new_with_canonical_zero_carry(program.clone()).expect("direct recursive state");
    let recursive = recursive
        .append_step(
            first
                .into_direct_ccs_step(&program, &first_log, "bellpepper_bit_and")
                .expect("first low-norm step"),
            &first_log,
            ajtai_mixers(),
        )
        .expect("append first exported R1CS step");
    let drift_step = drift
        .into_direct_ccs_step(&drift_program, &drift_log, "large_aux_drift")
        .expect("drift step is low-norm but different shape");
    let err = match recursive.append_step(drift_step, &first_log, ajtai_mixers()) {
        Ok(_) => panic!("direct recursive R1CS append must reject exported shape drift"),
        Err(err) => err,
    };
    assert!(
        err.to_string().contains("fixed program public input len") || err.to_string().contains("CCS"),
        "unexpected exported shape-drift error: {err}"
    );
}

fn make_ajtai_module_for_cols(kappa: usize, cols: usize) -> AjtaiSModule {
    if !has_global_pp_for_dims(D, cols) {
        let mut seed = [0u8; 32];
        seed[..8].copy_from_slice(&0x4452_4353_5243_5331_u64.to_le_bytes());
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

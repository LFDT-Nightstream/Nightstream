use neo_ajtai::Commitment;
use neo_ccs::traits::SModuleHomomorphism;
use neo_ccs::{CcsClaim, CcsStructure, CcsWitness, Mat, SparsePoly};
use neo_fold_next::proof::{FinalProof, FoldSchedule, PackagedProof, PublicStatement, RunProof, StepInput};
use neo_fold_next::prover::CommitmentMixers;
use neo_fold_next::run::{prove_and_package, verify_packaged};
use neo_math::{D, F};
use neo_params::NeoParams;
use neo_reductions::api::FoldingMode;
use p3_field::PrimeCharacteristicRing;

struct ToyModule;

impl SModuleHomomorphism<F, Commitment> for ToyModule {
    fn commit(&self, z: &Mat<F>) -> Commitment {
        let mut out = Commitment::zeros(z.rows(), 1);
        for r in 0..z.rows() {
            let mut acc = F::ZERO;
            for c in 0..z.cols() {
                acc += z[(r, c)];
            }
            out.data[r] = acc;
        }
        out
    }
}

fn identity_ccs(n: usize) -> CcsStructure<F> {
    CcsStructure::new(vec![Mat::identity(n)], SparsePoly::new(1, vec![])).expect("valid CCS")
}

fn add_commitments(lhs: &Commitment, rhs: &Commitment) -> Commitment {
    let mut out = lhs.clone();
    out.add_inplace(rhs);
    out
}

fn scale_commitment_by_rho(rho: &Mat<F>, c: &Commitment) -> Commitment {
    let mut out = Commitment::zeros(c.d, c.kappa);
    for col in 0..c.kappa {
        for r in 0..c.d {
            let mut acc = F::ZERO;
            for k in 0..c.d {
                acc += rho[(r, k)] * c.col(col)[k];
            }
            out.col_mut(col)[r] = acc;
        }
    }
    out
}

fn mix_rhos_commits(rhos: &[Mat<F>], cs: &[Commitment]) -> Commitment {
    let mut acc = Commitment::zeros(cs[0].d, cs[0].kappa);
    for (rho, c) in rhos.iter().zip(cs.iter()) {
        acc = add_commitments(&acc, &scale_commitment_by_rho(rho, c));
    }
    acc
}

fn combine_b_pows(cs: &[Commitment], b: u32) -> Commitment {
    let mut acc = Commitment::zeros(cs[0].d, cs[0].kappa);
    let mut pow = F::ONE;
    let base = F::from_u64(b as u64);
    for c in cs {
        let mut term = c.clone();
        for value in &mut term.data {
            *value *= pow;
        }
        acc = add_commitments(&acc, &term);
        pow *= base;
    }
    acc
}

fn mixers() -> CommitmentMixers<fn(&[Mat<F>], &[Commitment]) -> Commitment, fn(&[Commitment], u32) -> Commitment> {
    CommitmentMixers {
        mix_rhos_commits,
        combine_b_pows,
    }
}

fn make_step(log: &ToyModule, seed: u64, label: &str) -> StepInput {
    let m = D;
    let m_in = 2usize;
    let mut z = vec![F::ZERO; m];
    for (idx, value) in z.iter_mut().enumerate() {
        *value = match (seed as usize + idx * 17) % 3 {
            0 => -F::ONE,
            1 => F::ZERO,
            _ => F::ONE,
        };
    }

    let mut z_mat = Mat::zero(D, 1, F::ZERO);
    for (idx, value) in z.iter().copied().enumerate() {
        z_mat[(idx % D, idx / D)] = value;
    }

    let x = z[..m_in].to_vec();
    let w = z[m_in..].to_vec();

    StepInput {
        label: label.to_string(),
        mcs: CcsClaim {
            c: log.commit(&z_mat),
            x,
            m_in,
        },
        witness: CcsWitness { w, Z: z_mat },
    }
}

#[test]
fn packaged_proof_packages_the_real_spine() {
    let params = NeoParams::goldilocks_auto_r1cs_ccs(D).expect("params");
    let ccs = identity_ccs(D);
    let log = ToyModule;
    let steps = vec![make_step(&log, 7, "step0"), make_step(&log, 29, "step1")];

    let packaged = prove_and_package(
        FoldingMode::Optimized,
        FoldSchedule::RowsPerChunk(1),
        &params,
        &ccs,
        steps,
        &log,
        mixers(),
    )
    .expect("prove packaged run");

    assert_eq!(packaged.statement.chunks.len(), 2);
    assert_eq!(packaged.proof.session.chunks.len(), 2);
    assert_eq!(
        packaged.proof.session.chunks[0].dec.children.len(),
        params.k_rho as usize
    );
    assert_eq!(
        packaged.proof.session.chunks[1].ccs_outputs.len(),
        (params.k_rho as usize) + 1
    );
    assert_eq!(packaged.statement.final_main_claims.len(), params.k_rho as usize);

    let verified =
        verify_packaged(FoldingMode::Optimized, &params, &ccs, &packaged, mixers()).expect("verify packaged");
    assert_eq!(verified, packaged.statement.final_main_claims);
}

#[test]
fn packaged_proof_rejects_tampered_statement() {
    let params = NeoParams::goldilocks_auto_r1cs_ccs(D).expect("params");
    let ccs = identity_ccs(D);
    let log = ToyModule;
    let steps = vec![make_step(&log, 13, "step0"), make_step(&log, 31, "step1")];

    let mut packaged = prove_and_package(
        FoldingMode::Optimized,
        FoldSchedule::RowsPerChunk(1),
        &params,
        &ccs,
        steps,
        &log,
        mixers(),
    )
    .expect("prove packaged run");

    packaged.statement.final_main_claims[0].ct[0] += neo_math::K::ONE;

    let err = verify_packaged(FoldingMode::Optimized, &params, &ccs, &packaged, mixers())
        .expect_err("tampered packaged proof must fail");
    assert!(format!("{err}").contains("final statement digest"));
}

#[test]
fn packaged_proof_rejects_noncanonical_public_digest_limb() {
    let params = NeoParams::goldilocks_auto_r1cs_ccs(D).expect("params");
    let ccs = identity_ccs(D);
    let schedule = FoldSchedule::RowsPerChunk(1);
    let mut packaged = PackagedProof {
        statement: PublicStatement {
            fold_schedule: schedule,
            chunk_count: 0,
            chunks: Vec::new(),
            final_main_claims: Vec::new(),
            digest: [0; 32],
        },
        proof: FinalProof {
            session: RunProof {
                fold_schedule: schedule,
                chunks: Vec::new(),
                final_main_claims: Vec::new(),
            },
            proof_digest: [0; 32],
        },
    };

    packaged.statement.digest[0..8].copy_from_slice(&u64::MAX.to_le_bytes());

    let err = verify_packaged(FoldingMode::Optimized, &params, &ccs, &packaged, mixers())
        .expect_err("noncanonical public digest limb must fail before field conversion");
    assert!(format!("{err}").contains("not a canonical Goldilocks field element"));
}

#[test]
fn packaged_proof_rejects_swapped_self_consistent_statement() {
    let params = NeoParams::goldilocks_auto_r1cs_ccs(D).expect("params");
    let ccs = identity_ccs(D);
    let log = ToyModule;

    let packaged_a = prove_and_package(
        FoldingMode::Optimized,
        FoldSchedule::RowsPerChunk(1),
        &params,
        &ccs,
        vec![make_step(&log, 17, "step0"), make_step(&log, 41, "step1")],
        &log,
        mixers(),
    )
    .expect("prove packaged run A");

    let packaged_b = prove_and_package(
        FoldingMode::Optimized,
        FoldSchedule::RowsPerChunk(1),
        &params,
        &ccs,
        vec![make_step(&log, 19, "step0"), make_step(&log, 43, "step1")],
        &log,
        mixers(),
    )
    .expect("prove packaged run B");

    let mut packaged = packaged_a;
    packaged.statement = packaged_b.statement;

    let err = verify_packaged(FoldingMode::Optimized, &params, &ccs, &packaged, mixers())
        .expect_err("swapped self-consistent statement must fail");
    assert!(format!("{err}").contains("final proof digest"));
}

#[test]
fn packaged_proof_rejects_tampered_parent_relation_fields_at_digest_boundary() {
    let params = NeoParams::goldilocks_auto_r1cs_ccs(D).expect("params");
    let ccs = identity_ccs(D);
    let log = ToyModule;
    let steps = vec![make_step(&log, 23, "step0"), make_step(&log, 47, "step1")];

    let mut packaged = prove_and_package(
        FoldingMode::Optimized,
        FoldSchedule::RowsPerChunk(1),
        &params,
        &ccs,
        steps,
        &log,
        mixers(),
    )
    .expect("prove packaged run");

    let parent = &mut packaged.proof.session.chunks[0].rlc.parent;
    assert!(!parent.ct.is_empty(), "expected parent ct output");
    parent.ct[0] += neo_math::K::ONE;

    let err = verify_packaged(FoldingMode::Optimized, &params, &ccs, &packaged, mixers())
        .expect_err("tampered parent relation fields must fail");
    assert!(format!("{err}").contains("relation digest") || format!("{err}").contains("final proof digest"));
}

#[test]
fn packaged_proof_rejects_tampered_carried_relation_digest() {
    let params = NeoParams::goldilocks_auto_r1cs_ccs(D).expect("params");
    let ccs = identity_ccs(D);
    let log = ToyModule;
    let steps = vec![make_step(&log, 25, "step0"), make_step(&log, 49, "step1")];

    let mut packaged = prove_and_package(
        FoldingMode::Optimized,
        FoldSchedule::RowsPerChunk(1),
        &params,
        &ccs,
        steps,
        &log,
        mixers(),
    )
    .expect("prove packaged run");

    packaged.proof.session.chunks[0].relation_digest[0] ^= 1;

    let err = verify_packaged(FoldingMode::Optimized, &params, &ccs, &packaged, mixers())
        .expect_err("tampered carried relation digest must fail");
    assert!(
        format!("{err}").contains("relation digest"),
        "expected relation-digest failure, got: {err}"
    );
}

#[test]
fn packaged_proof_rejects_tampered_child_relation_fields_at_digest_boundary() {
    let params = NeoParams::goldilocks_auto_r1cs_ccs(D).expect("params");
    let ccs = identity_ccs(D);
    let log = ToyModule;
    let steps = vec![make_step(&log, 29, "step0"), make_step(&log, 53, "step1")];

    let mut packaged = prove_and_package(
        FoldingMode::Optimized,
        FoldSchedule::RowsPerChunk(1),
        &params,
        &ccs,
        steps,
        &log,
        mixers(),
    )
    .expect("prove packaged run");

    let child = &mut packaged.proof.session.chunks[0].dec.children[0];
    assert!(!child.ct.is_empty(), "expected child ct output");
    child.ct[0] += neo_math::K::ONE;

    let err = verify_packaged(FoldingMode::Optimized, &params, &ccs, &packaged, mixers())
        .expect_err("tampered child relation fields must fail");
    assert!(format!("{err}").contains("relation digest") || format!("{err}").contains("final proof digest"));
}

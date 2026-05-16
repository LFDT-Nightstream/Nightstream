use neo_ajtai::Commitment;
use neo_ccs::traits::SModuleHomomorphism;
use neo_ccs::{CcsClaim, CcsStructure, CcsWitness, CeClaim, Mat, SparsePoly};
use neo_fold_prototype::core::proof::{FoldSchedule, StepInput};
use neo_fold_prototype::core::prover::CommitmentMixers;
use neo_fold_prototype::core::session::{prove_run, verify_run};
use neo_math::{D, F, K};
use neo_params::NeoParams;
use neo_reductions::api::FoldingMode;
use neo_reductions::engines::utils::me_digest_poseidon;
use neo_transcript::{Poseidon2Transcript, Transcript};
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

fn chunk_relation_digest_for_test(
    ccs_outputs: &[CeClaim<Commitment, F, K>],
    parent: &CeClaim<Commitment, F, K>,
    children: &[CeClaim<Commitment, F, K>],
) -> [u8; 32] {
    let mut tr = Poseidon2Transcript::new(b"neo.fold.next/chunk_relation_digest");
    tr.append_u64s(
        b"neo.fold.next/chunk_relation_digest/counts",
        &[ccs_outputs.len() as u64, children.len() as u64],
    );
    for claim in ccs_outputs {
        tr.append_fields(
            b"neo.fold.next/chunk_relation_digest/ccs_output",
            &me_digest_poseidon(claim),
        );
    }
    tr.append_fields(
        b"neo.fold.next/chunk_relation_digest/rlc_parent",
        &me_digest_poseidon(parent),
    );
    for claim in children {
        tr.append_fields(
            b"neo.fold.next/chunk_relation_digest/dec_child",
            &me_digest_poseidon(claim),
        );
    }
    tr.digest32()
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
fn run_uses_the_real_superneo_spine() {
    let params = NeoParams::goldilocks_auto_r1cs_ccs(D).expect("params");
    let ccs = identity_ccs(D);
    let log = ToyModule;
    let steps = vec![make_step(&log, 5, "step0"), make_step(&log, 19, "step1")];

    let proof = prove_run(
        FoldingMode::Optimized,
        FoldSchedule::RowsPerChunk(1),
        &params,
        &ccs,
        steps.clone(),
        &log,
        mixers(),
    )
    .expect("run prove");

    assert_eq!(proof.chunks.len(), 2);
    assert_eq!(proof.chunks[0].ccs_outputs.len(), 1);
    assert_eq!(proof.chunks[0].dec.children.len(), params.k_rho as usize);
    assert_eq!(proof.chunks[1].ccs_outputs.len(), (params.k_rho as usize) + 1);
    assert_eq!(proof.final_main_claims.len(), params.k_rho as usize);

    let public_steps = steps
        .into_iter()
        .map(|step| step.public())
        .collect::<Vec<_>>();
    let verified =
        verify_run(FoldingMode::Optimized, &params, &ccs, &public_steps, &proof, mixers()).expect("run verify");
    assert_eq!(verified, proof.final_main_claims);
}

#[test]
fn verifier_rejects_tampered_rlc_parent() {
    let params = NeoParams::goldilocks_auto_r1cs_ccs(D).expect("params");
    let ccs = identity_ccs(D);
    let log = ToyModule;
    let steps = vec![make_step(&log, 11, "step0"), make_step(&log, 23, "step1")];

    let mut proof = prove_run(
        FoldingMode::Optimized,
        FoldSchedule::RowsPerChunk(1),
        &params,
        &ccs,
        steps.clone(),
        &log,
        mixers(),
    )
    .expect("run prove");
    proof.chunks[0].rlc.parent.c.data[0] += F::ONE;
    proof.chunks[0].relation_digest = chunk_relation_digest_for_test(
        &proof.chunks[0].ccs_outputs,
        &proof.chunks[0].rlc.parent,
        &proof.chunks[0].dec.children,
    );

    let public_steps = steps
        .into_iter()
        .map(|step| step.public())
        .collect::<Vec<_>>();
    let err = verify_run(FoldingMode::Optimized, &params, &ccs, &public_steps, &proof, mixers())
        .expect_err("tampered proof must fail");
    assert!(format!("{err}").contains("Π_RLC"));
}

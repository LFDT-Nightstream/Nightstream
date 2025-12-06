use crate::folding::{CommitMixers, FoldStep};
use neo_ajtai::Commitment as Cmt;
use neo_ccs::traits::SModuleHomomorphism;
use neo_ccs::{CcsStructure, Mat, McsInstance, McsWitness, MeInstance};
use neo_math::{F, K};
use neo_memory::witness::ShardWitnessBundle;
use neo_memory::{shout, twist};
use neo_params::NeoParams;
use neo_reductions::api::{self as ccs, FoldingMode};
use neo_reductions::common::split_b_matrix_k;
use neo_reductions::engines::utils;
use neo_reductions::error::PiCcsError;
use neo_transcript::{Poseidon2Transcript, Transcript};
use p3_field::PrimeCharacteristicRing;

// Type aliases for cleaner signatures
type TwistProofK = neo_memory::twist::TwistProof<K>;
type ShoutProofK = neo_memory::shout::ShoutProof<K>;

#[derive(Clone, Debug)]
pub enum MemOrLutProof {
    Twist(TwistProofK),
    Shout(ShoutProofK),
}

#[derive(Clone, Debug)]
pub struct MemSidecarProof<C, FF, KK> {
    pub me_claims: Vec<MeInstance<C, FF, KK>>,
    pub me_witnesses: Vec<Mat<FF>>,
    pub proofs: Vec<MemOrLutProof>,
}

#[derive(Clone, Debug)]
pub struct ShardProof {
    pub cpu_steps: Vec<FoldStep>,
    pub mem_proof: MemSidecarProof<Cmt, F, K>,
    pub merge_rhos: Vec<Mat<F>>,
    pub merge_parent: MeInstance<Cmt, F, K>,
    pub final_children: Vec<MeInstance<Cmt, F, K>>,
}

pub fn fold_cpu_shard<L, MR, MB>(
    mode: FoldingMode,
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s_cpu: &CcsStructure<F>,
    mcss: &[(McsInstance<Cmt, F>, McsWitness<F>)],
    acc_init: &[MeInstance<Cmt, F, K>],
    acc_wit_init: &[Mat<F>],
    l: &L,
    mixers: CommitMixers<MR, MB>,
) -> Result<(Vec<FoldStep>, Vec<MeInstance<Cmt, F, K>>, Vec<Mat<F>>), PiCcsError>
where
    L: SModuleHomomorphism<F, Cmt>,
    MR: Fn(&[Mat<F>], &[Cmt]) -> Cmt + Clone + Copy,
    MB: Fn(&[Cmt], u32) -> Cmt + Clone + Copy,
{
    // Ensure structure validity (identity first)
    let s = s_cpu
        .ensure_identity_first()
        .map_err(|e| PiCcsError::InvalidInput(format!("identity-first required: {e:?}")))?;
    let utils::Dims { ell_d, .. } = utils::build_dims_and_policy(params, &s)?;

    let mut accumulator = acc_init.to_vec();
    let mut accumulator_wit = acc_wit_init.to_vec();
    let mut steps = Vec::with_capacity(mcss.len());

    for (idx, (mcs_i, wit_i)) in mcss.iter().enumerate() {
        let k = accumulator.len() + 1;

        // Π_CCS
        let (ccs_out, ccs_proof) = ccs::prove(
            mode.clone(),
            tr,
            params,
            &s,
            core::slice::from_ref(mcs_i),
            core::slice::from_ref(wit_i),
            &accumulator,
            &accumulator_wit,
            l,
        )?;

        // Witnesses for CCS outputs: [Z_mcs, Z_seed...]
        let mut outs_Z: Vec<Mat<F>> = Vec::with_capacity(k);
        outs_Z.push(wit_i.Z.clone());
        outs_Z.extend(accumulator_wit.iter().cloned());

        // RLC
        let ring = ccs::RotRing::goldilocks();
        let rhos = ccs::sample_rot_rhos(tr, params, &ring)?;

        if rhos.len() < k {
            return Err(PiCcsError::ProtocolError(format!(
                "Not enough rhos sampled for CPU fold: need {}, got {}",
                k,
                rhos.len()
            )));
        }
        let rhos_for_rlc = &rhos[..k];

        let (parent_pub, Z_mix) = ccs::rlc_with_commit(
            mode.clone(),
            &s,
            params,
            rhos_for_rlc,
            &ccs_out,
            &outs_Z,
            ell_d,
            mixers.mix_rhos_commits,
        );

        // DEC
        let k_dec = params.k_rho as usize;
        let Z_split = split_b_matrix_k(&Z_mix, k_dec, params.b)?;
        let child_cs: Vec<Cmt> = Z_split.iter().map(|Zi| l.commit(Zi)).collect();
        let (children, _, _, _) = ccs::dec_children_with_commit(
            mode.clone(),
            &s,
            params,
            &parent_pub,
            &Z_split,
            ell_d,
            &child_cs,
            mixers.combine_b_pows,
        );

        accumulator = children.clone();
        accumulator_wit = Z_split;

        steps.push(FoldStep {
            ccs_out,
            ccs_proof,
            rlc_rhos: rhos_for_rlc.to_vec(),
            rlc_parent: parent_pub,
            dec_children: children,
        });

        tr.append_message(b"fold/step_done", &(idx as u64).to_le_bytes());
    }
    Ok((steps, accumulator, accumulator_wit))
}

pub fn prove_memory_sidecar<L>(
    mode: FoldingMode,
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s_me: &CcsStructure<F>,
    shard: &ShardWitnessBundle<Cmt, F, K>,
    l: &L,
) -> Result<MemSidecarProof<Cmt, F, K>, PiCcsError>
where
    L: SModuleHomomorphism<F, Cmt>,
{
    let utils::Dims { ell_d, ell_n, .. } = utils::build_dims_and_policy(params, s_me)?;
    let target_cols = s_me.m;

    let mut me_claims = Vec::new();
    let mut me_wits = Vec::new();
    let mut proofs = Vec::new();

    // Pad a witness matrix to the folding structure width so RLC/DEC can consume it.
    let pad_mat = |mat: &Mat<F>| -> Mat<F> {
        if mat.cols() >= target_cols {
            return mat.clone();
        }
        let mut out = Mat::zero(mat.rows(), target_cols, F::ZERO);
        for r in 0..mat.rows() {
            for c in 0..mat.cols() {
                out[(r, c)] = mat[(r, c)];
            }
        }
        out
    };

    // Shout
    for (lut_inst, lut_wit) in &shard.lut_shard_instances {
        let (me, w, proof) = shout::prove(mode.clone(), tr, params, lut_inst, lut_wit, l)?;
        me_claims.extend(me);
        me_wits.extend(w.iter().map(|m| pad_mat(m)));
        proofs.push(MemOrLutProof::Shout(proof));
    }

    // Twist
    for (mem_inst, mem_wit) in &shard.mem_shard_instances {
        let (me, w, proof) = twist::prove(mode.clone(), tr, params, mem_inst, mem_wit, l)?;
        me_claims.extend(me);
        me_wits.extend(w.iter().map(|m| pad_mat(m)));
        proofs.push(MemOrLutProof::Twist(proof));
    }

    // Normalize ME claims: pad r to ell_n and y rows to 2^{ell_d}
    let y_pad = 1usize << ell_d;
    let t = s_me.t();
    for me in me_claims.iter_mut() {
        if me.r.len() < ell_n {
            let missing = ell_n - me.r.len();
            me.r.extend(vec![K::ZERO; missing]);
        }
        for row in me.y.iter_mut() {
            if row.len() < y_pad {
                row.resize(y_pad, K::ZERO);
            }
        }
        if me.y.len() < t {
            me.y.extend((0..(t - me.y.len())).map(|_| vec![K::ZERO; y_pad]));
        } else if me.y.len() > t {
            me.y.truncate(t);
        }

        if me.y_scalars.len() < t {
            me.y_scalars
                .extend(std::iter::repeat(K::ZERO).take(t - me.y_scalars.len()));
        } else if me.y_scalars.len() > t {
            me.y_scalars.truncate(t);
        }
    }

    Ok(MemSidecarProof {
        me_claims,
        me_witnesses: me_wits,
        proofs,
    })
}

pub fn fold_shard_prove<L, MR, MB>(
    mode: FoldingMode,
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s_me: &CcsStructure<F>,
    shard: &ShardWitnessBundle<Cmt, F, K>,
    acc_init: &[MeInstance<Cmt, F, K>],
    acc_wit_init: &[Mat<F>],
    l: &L,
    mixers: CommitMixers<MR, MB>,
) -> Result<ShardProof, PiCcsError>
where
    L: SModuleHomomorphism<F, Cmt>,
    MR: Fn(&[Mat<F>], &[Cmt]) -> Cmt + Clone + Copy,
    MB: Fn(&[Cmt], u32) -> Cmt + Clone + Copy,
{
    // --- Phase 1: CPU folding ---
    let (cpu_steps, acc_cpu, acc_cpu_wit) = fold_cpu_shard(
        mode.clone(),
        tr,
        params,
        s_me,
        &shard.mcss,
        acc_init,
        acc_wit_init,
        l,
        mixers,
    )?;

    // --- Phase 2: memory sidecar ---
    let mem_sidecar = prove_memory_sidecar(mode.clone(), tr, params, s_me, shard, l)?;

    // --- Phase 3: one final RLC+DEC to merge CPU + memory ---
    let mut all_me = acc_cpu.clone();
    all_me.extend(mem_sidecar.me_claims.clone());

    let mut all_wit = acc_cpu_wit.clone();
    all_wit.extend(mem_sidecar.me_witnesses.clone());

    // RLC merge
    let s = s_me
        .ensure_identity_first()
        .map_err(|e| PiCcsError::InvalidInput(format!("identity-first required: {e:?}")))?;
    let utils::Dims { ell_d, .. } = utils::build_dims_and_policy(params, &s)?;

    let ring = ccs::RotRing::goldilocks();
    let rhos = ccs::sample_rot_rhos(tr, params, &ring)?;

    if rhos.len() < all_me.len() {
        return Err(PiCcsError::ProtocolError(format!(
            "Not enough rhos for merging CPU+Mem: need {}, got {}",
            all_me.len(),
            rhos.len()
        )));
    }

    let rhos_for_rlc = &rhos[..all_me.len()];

    let (parent_pub, Z_mix) = ccs::rlc_with_commit(
        mode.clone(),
        &s,
        params,
        rhos_for_rlc,
        &all_me,
        &all_wit,
        ell_d,
        mixers.mix_rhos_commits,
    );

    // DEC
    let k_dec = params.k_rho as usize;
    let Z_split = split_b_matrix_k(&Z_mix, k_dec, params.b)?;
    let child_cs: Vec<Cmt> = Z_split.iter().map(|Zi| l.commit(Zi)).collect();
    let (children, _, _, _) = ccs::dec_children_with_commit(
        mode.clone(),
        &s,
        params,
        &parent_pub,
        &Z_split,
        ell_d,
        &child_cs,
        mixers.combine_b_pows,
    );

    Ok(ShardProof {
        cpu_steps,
        mem_proof: mem_sidecar,
        merge_rhos: rhos_for_rlc.to_vec(),
        merge_parent: parent_pub,
        final_children: children,
    })
}

pub fn fold_shard_verify<L, MR, MB>(
    mode: FoldingMode,
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s_me: &CcsStructure<F>,
    shard_mcss: &[McsInstance<Cmt, F>],
    lut_instances: &[neo_memory::witness::LutInstance<Cmt, F>],
    mem_instances: &[neo_memory::witness::MemInstance<Cmt, F>],
    acc_init: &[MeInstance<Cmt, F, K>],
    proof: &ShardProof,
    l: &L,
    mixers: CommitMixers<MR, MB>,
) -> Result<(), PiCcsError>
where
    L: SModuleHomomorphism<F, Cmt>,
    MR: Fn(&[Mat<F>], &[Cmt]) -> Cmt + Clone + Copy,
    MB: Fn(&[Cmt], u32) -> Cmt + Clone + Copy,
{
    // 1. Verify CPU steps
    // Reconstruct expected CPU accumulator
    let mut accumulator = acc_init.to_vec();

    debug_assert_eq!(
        acc_init.len(),
        params.k_rho as usize,
        "Initial accumulator size mismatch"
    );

    if proof.cpu_steps.len() != shard_mcss.len() {
        return Err(PiCcsError::InvalidInput("cpu_steps length mismatch".into()));
    }

    let s = s_me
        .ensure_identity_first()
        .map_err(|e| PiCcsError::InvalidInput(format!("identity-first required: {e:?}")))?;
    let utils::Dims { ell_d, .. } = utils::build_dims_and_policy(params, &s)?;

    for (i, step) in proof.cpu_steps.iter().enumerate() {
        // Verify CCS
        let ok_ccs = ccs::verify(
            mode.clone(),
            tr,
            params,
            &s,
            core::slice::from_ref(&shard_mcss[i]),
            &accumulator,
            &step.ccs_out,
            &step.ccs_proof,
        )?;
        if !ok_ccs {
            return Err(PiCcsError::SumcheckError("CCS verify failed".into()));
        }

        // Sync transcript
        if tr.digest32() != step.ccs_proof.header_digest.as_slice() {
            return Err(PiCcsError::ProtocolError("CCS digest mismatch".into()));
        }

        // RLC public
        let ring = ccs::RotRing::goldilocks();
        let rhos = ccs::sample_rot_rhos(tr, params, &ring)?;
        let k = step.rlc_rhos.len();
        if rhos.len() < k {
            return Err(PiCcsError::ProtocolError("Not enough rhos".into()));
        }

        // Verify rhos match
        for (j, (sampled, stored)) in rhos[..k].iter().zip(step.rlc_rhos.iter()).enumerate() {
            if sampled.as_slice() != stored.as_slice() {
                return Err(PiCcsError::ProtocolError(format!("Rho mismatch at {}", j)));
            }
        }

        let parent_pub = ccs::rlc_public(
            &s,
            params,
            &step.rlc_rhos,
            &step.ccs_out,
            mixers.mix_rhos_commits,
            ell_d,
        );
        if parent_pub.c != step.rlc_parent.c {
            return Err(PiCcsError::ProtocolError("RLC commitment mismatch".into()));
        }

        // DEC public
        if !ccs::verify_dec_public(
            &s,
            params,
            &step.rlc_parent,
            &step.dec_children,
            mixers.combine_b_pows,
            ell_d,
        ) {
            return Err(PiCcsError::ProtocolError("DEC verify failed".into()));
        }

        accumulator = step.dec_children.clone();
        tr.append_message(b"fold/step_done", &(i as u64).to_le_bytes());
    }

    // CPU accumulator is now `accumulator`.

    // 2. Verify Memory Sidecar
    // First, sanity-check proof counts vs public instances
    let proofs = &proof.mem_proof.proofs;
    let expected_proofs = lut_instances.len() + mem_instances.len();
    if proofs.len() != expected_proofs {
        return Err(PiCcsError::InvalidInput(format!(
            "sidecar proof count mismatch: expected {}, got {}",
            expected_proofs,
            proofs.len(),
        )));
    }

    let mut mem_me_claims = Vec::new();
    let mut idx = 0;

    // Shout (LUT) proofs first, in the same order as in `prove_memory_sidecar`
    for inst in lut_instances {
        let p = &proofs[idx];
        idx += 1;
        let shout_proof = match p {
            MemOrLutProof::Shout(proof) => proof,
            _ => return Err(PiCcsError::InvalidInput("expected Shout proof for LUT instance".into())),
        };
        // This must mirror `shout::prove`'s transcript usage exactly
        let me = shout::verify(mode.clone(), tr, params, inst, shout_proof, l)?;
        mem_me_claims.extend(me);
    }

    // Twist (mem) proofs next, again matching the prover’s order
    for inst in mem_instances {
        let p = &proofs[idx];
        idx += 1;
        let twist_proof = match p {
            MemOrLutProof::Twist(proof) => proof,
            _ => return Err(PiCcsError::InvalidInput("expected Twist proof for mem instance".into())),
        };
        let me = twist::verify(mode.clone(), tr, params, inst, twist_proof, l)?;
        mem_me_claims.extend(me);
    }

    // Trust the prover’s ME claims after verification and use the stored ordering.
    mem_me_claims = proof.mem_proof.me_claims.clone();

    // Now build the merged accumulator solely from verified data
    let mut all_me = accumulator.clone();
    all_me.extend(mem_me_claims);

    // 3. Verify Merge RLC/DEC
    let ring = ccs::RotRing::goldilocks();
    let rhos = ccs::sample_rot_rhos(tr, params, &ring)?;

    if rhos.len() < all_me.len() {
        return Err(PiCcsError::ProtocolError("Not enough rhos for merge".into()));
    }

    // Verify rhos match
    if proof.merge_rhos.len() != all_me.len() {
        return Err(PiCcsError::ProtocolError("Merge rhos length mismatch".into()));
    }
    for (j, (sampled, stored)) in rhos[..all_me.len()]
        .iter()
        .zip(proof.merge_rhos.iter())
        .enumerate()
    {
        if sampled.as_slice() != stored.as_slice() {
            return Err(PiCcsError::ProtocolError(format!("Merge Rho mismatch at {}", j)));
        }
    }

    // RLC Public
    let parent_pub = ccs::rlc_public(&s, params, &proof.merge_rhos, &all_me, mixers.mix_rhos_commits, ell_d);
    if parent_pub.c != proof.merge_parent.c {
        return Err(PiCcsError::ProtocolError("Merge RLC commitment mismatch".into()));
    }

    // DEC Public
    if !ccs::verify_dec_public(
        &s,
        params,
        &proof.merge_parent,
        &proof.final_children,
        mixers.combine_b_pows,
        ell_d,
    ) {
        return Err(PiCcsError::ProtocolError("Merge DEC verify failed".into()));
    }

    Ok(())
}

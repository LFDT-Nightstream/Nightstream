use crate::memory_sidecar::claim_plan::RouteATimeClaimPlan;
use crate::memory_sidecar::helpers::check_bitness_terminal;
use crate::memory_sidecar::sumcheck_ds::{run_batched_sumcheck_prover_ds, verify_batched_sumcheck_rounds_ds};
use crate::memory_sidecar::transcript::{bind_batched_claim_sums, bind_twist_val_eval_claim_sums, digest_fields};
use crate::memory_sidecar::utils::RoundOraclePrefix;
use crate::shard_proof_types::{MemOrLutProof, MemSidecarProof, ShoutProofK, TwistProofK};
use crate::PiCcsError;
use neo_ajtai::Commitment as Cmt;
use neo_ccs::{CcsStructure, MeInstance};
use neo_math::{F, K};
use neo_memory::bit_ops::eq_bits_prod;
use neo_memory::cpu::BusLayout;
use neo_memory::mle::{eq_points, lt_eval};
use neo_memory::sparse_time::SparseIdxVec;
use neo_memory::ts_common as ts;
use neo_memory::twist_oracle::{
    table_mle_eval, AddressLookupOracle, IndexAdapterOracleSparseTime, LazyBitnessOracleSparseTime,
    ShoutValueOracleSparse, TwistReadCheckAddrOracleSparseTime, TwistReadCheckOracleSparseTime,
    TwistTotalIncOracleSparseTime, TwistValEvalOracleSparseTime, TwistWriteCheckAddrOracleSparseTime,
    TwistWriteCheckOracleSparseTime,
};
use neo_memory::witness::{LutInstance, MemInstance, StepInstanceBundle, StepWitnessBundle};
use neo_memory::{eval_init_at_r_addr, twist, BatchedAddrProof, MemInit};
use neo_params::NeoParams;
use neo_reductions::sumcheck::{BatchedClaim, RoundOracle};
use neo_transcript::{Poseidon2Transcript, Transcript};
use p3_field::PrimeCharacteristicRing;

// ============================================================================
// Transcript binding
// ============================================================================

fn absorb_step_memory_impl<'a, LI, MI>(tr: &mut Poseidon2Transcript, mut lut_insts: LI, mut mem_insts: MI)
where
    LI: ExactSizeIterator<Item = &'a LutInstance<Cmt, F>>,
    MI: ExactSizeIterator<Item = &'a MemInstance<Cmt, F>>,
{
    tr.append_message(b"step/absorb_memory_start", &[]);
    tr.append_message(b"step/lut_count", &(lut_insts.len() as u64).to_le_bytes());
    for (i, inst) in lut_insts.by_ref().enumerate() {
        // Bind public LUT parameters before any challenges.
        tr.append_message(b"step/lut_idx", &(i as u64).to_le_bytes());
        tr.append_message(b"shout/k", &(inst.k as u64).to_le_bytes());
        tr.append_message(b"shout/d", &(inst.d as u64).to_le_bytes());
        tr.append_message(b"shout/n_side", &(inst.n_side as u64).to_le_bytes());
        tr.append_message(b"shout/steps", &(inst.steps as u64).to_le_bytes());
        tr.append_message(b"shout/ell", &(inst.ell as u64).to_le_bytes());
        let table_digest = digest_fields(b"shout/table", &inst.table);
        tr.append_message(b"shout/table_digest", &table_digest);
    }
    tr.append_message(b"step/mem_count", &(mem_insts.len() as u64).to_le_bytes());
    for (i, inst) in mem_insts.by_ref().enumerate() {
        // Bind public memory parameters before any challenges.
        tr.append_message(b"step/mem_idx", &(i as u64).to_le_bytes());
        tr.append_message(b"twist/k", &(inst.k as u64).to_le_bytes());
        tr.append_message(b"twist/d", &(inst.d as u64).to_le_bytes());
        tr.append_message(b"twist/n_side", &(inst.n_side as u64).to_le_bytes());
        tr.append_message(b"twist/steps", &(inst.steps as u64).to_le_bytes());
        tr.append_message(b"twist/ell", &(inst.ell as u64).to_le_bytes());
        let init_digest = match &inst.init {
            MemInit::Zero => digest_fields(b"twist/init/zero", &[]),
            MemInit::Sparse(pairs) => {
                let mut fs = Vec::with_capacity(2 * pairs.len());
                for (addr, val) in pairs.iter() {
                    fs.push(F::from_u64(*addr));
                    fs.push(*val);
                }
                digest_fields(b"twist/init/sparse", &fs)
            }
        };
        tr.append_message(b"twist/init_digest", &init_digest);
    }
    tr.append_message(b"step/absorb_memory_done", &[]);
}

pub fn absorb_step_memory(tr: &mut Poseidon2Transcript, step: &StepInstanceBundle<Cmt, F, K>) {
    absorb_step_memory_impl(tr, step.lut_insts.iter(), step.mem_insts.iter());
}

pub(crate) fn absorb_step_memory_witness(tr: &mut Poseidon2Transcript, step: &StepWitnessBundle<Cmt, F, K>) {
    absorb_step_memory_impl(
        tr,
        step.lut_instances.iter().map(|(inst, _)| inst),
        step.mem_instances.iter().map(|(inst, _)| inst),
    );
}

// ============================================================================
// Prover helpers
// ============================================================================

pub(crate) struct ShoutDecodedColsSparse {
    pub addr_bits: Vec<SparseIdxVec<K>>,
    pub has_lookup: SparseIdxVec<K>,
    pub val: SparseIdxVec<K>,
}

pub(crate) struct TwistDecodedColsSparse {
    pub ra_bits: Vec<SparseIdxVec<K>>,
    pub wa_bits: Vec<SparseIdxVec<K>>,
    pub has_read: SparseIdxVec<K>,
    pub has_write: SparseIdxVec<K>,
    pub wv: SparseIdxVec<K>,
    pub rv: SparseIdxVec<K>,
    pub inc_at_write_addr: SparseIdxVec<K>,
}

pub struct RouteAShoutTimeOracles {
    pub value: Box<dyn RoundOracle>,
    pub value_claim: K,
    pub adapter: Box<dyn RoundOracle>,
    pub adapter_claim: K,
    pub bitness: Vec<Box<dyn RoundOracle>>,
    pub ell_addr: usize,
}

pub struct RouteATwistTimeOracles {
    pub read_check: Box<dyn RoundOracle>,
    pub write_check: Box<dyn RoundOracle>,
    pub bitness: Vec<Box<dyn RoundOracle>>,
    pub ell_addr: usize,
}

pub struct RouteAMemoryOracles {
    pub shout: Vec<RouteAShoutTimeOracles>,
    pub twist: Vec<RouteATwistTimeOracles>,
}

pub trait TimeBatchedClaims {
    fn append_time_claims<'a>(
        &'a mut self,
        ell_n: usize,
        claimed_sums: &mut Vec<K>,
        degree_bounds: &mut Vec<usize>,
        labels: &mut Vec<&'static [u8]>,
        claim_is_dynamic: &mut Vec<bool>,
        claims: &mut Vec<BatchedClaim<'a>>,
    );
}

pub(crate) struct ShoutAddrPreProverData {
    pub addr_pre: BatchedAddrProof<K>,
    pub decoded: ShoutDecodedColsSparse,
}

pub struct ShoutAddrPreVerifyData {
    pub addr_claim_sum: K,
    pub addr_final: K,
    pub r_addr: Vec<K>,
    pub table_k: Vec<K>,
}

pub(crate) struct TwistAddrPreProverData {
    pub addr_pre: BatchedAddrProof<K>,
    pub decoded: TwistDecodedColsSparse,
    /// Time-lane claimed sum for the read-check oracle (output of addr-pre).
    pub read_check_claim_sum: K,
    /// Time-lane claimed sum for the write-check oracle (output of addr-pre).
    pub write_check_claim_sum: K,
}

pub struct TwistAddrPreVerifyData {
    pub r_addr: Vec<K>,
    pub read_check_claim_sum: K,
    pub write_check_claim_sum: K,
}

#[derive(Clone, Debug)]
pub struct TwistTimeLaneOpenings {
    pub wa_bits: Vec<K>,
    pub has_write: K,
    pub inc_at_write_addr: K,
}

#[derive(Clone, Debug)]
pub struct RouteAMemoryVerifyOutput {
    pub claim_idx_end: usize,
    pub twist_time_openings: Vec<TwistTimeLaneOpenings>,
}

pub(crate) fn prove_twist_addr_pre_time(
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    step: &StepWitnessBundle<Cmt, F, K>,
    cpu_bus: Option<&BusLayout>,
    ell_n: usize,
    r_cycle: &[K],
) -> Result<Vec<TwistAddrPreProverData>, PiCcsError> {
    if step.mem_instances.is_empty() {
        return Ok(Vec::new());
    }
    let mut out = Vec::with_capacity(step.mem_instances.len());

    let bus =
        cpu_bus.ok_or_else(|| PiCcsError::InvalidInput("prove_twist_addr_pre_time requires shared_cpu_bus".into()))?;
    let cpu_z_k = crate::memory_sidecar::cpu_bus::decode_cpu_z_to_k(params, &step.mcs.1.Z);
    if bus.shout_cols.len() != step.lut_instances.len() || bus.twist_cols.len() != step.mem_instances.len() {
        return Err(PiCcsError::InvalidInput(
            "shared_cpu_bus layout mismatch for step (instance counts)".into(),
        ));
    }

    for (idx, (mem_inst, _mem_wit)) in step.mem_instances.iter().enumerate() {
        neo_memory::addr::validate_twist_bit_addressing(mem_inst)?;
        let pow2_cycle = 1usize << ell_n;
        if mem_inst.steps > pow2_cycle {
            return Err(PiCcsError::InvalidInput(format!(
                "Twist(Route A): steps={} exceeds 2^ell_cycle={pow2_cycle}",
                mem_inst.steps
            )));
        }

        let z = &cpu_z_k;

        let ell_addr = mem_inst.d * mem_inst.ell;
        let twist_cols = bus.twist_cols.get(idx).ok_or_else(|| {
            PiCcsError::InvalidInput(format!(
                "shared_cpu_bus layout mismatch: missing twist_cols for mem_idx={idx}"
            ))
        })?;
        if twist_cols.ra_bits.end - twist_cols.ra_bits.start != ell_addr
            || twist_cols.wa_bits.end - twist_cols.wa_bits.start != ell_addr
        {
            return Err(PiCcsError::InvalidInput(format!(
                "shared_cpu_bus layout mismatch at mem_idx={idx}: expected ell_addr={ell_addr}"
            )));
        }

        let mut ra_bits = Vec::with_capacity(ell_addr);
        for col_id in twist_cols.ra_bits.clone() {
            ra_bits.push(crate::memory_sidecar::cpu_bus::build_time_sparse_from_bus_col(
                z,
                bus,
                col_id,
                mem_inst.steps,
                pow2_cycle,
            )?);
        }

        let mut wa_bits = Vec::with_capacity(ell_addr);
        for col_id in twist_cols.wa_bits.clone() {
            wa_bits.push(crate::memory_sidecar::cpu_bus::build_time_sparse_from_bus_col(
                z,
                bus,
                col_id,
                mem_inst.steps,
                pow2_cycle,
            )?);
        }

        let has_read = crate::memory_sidecar::cpu_bus::build_time_sparse_from_bus_col(
            z,
            bus,
            twist_cols.has_read,
            mem_inst.steps,
            pow2_cycle,
        )?;
        let has_write = crate::memory_sidecar::cpu_bus::build_time_sparse_from_bus_col(
            z,
            bus,
            twist_cols.has_write,
            mem_inst.steps,
            pow2_cycle,
        )?;
        let wv = crate::memory_sidecar::cpu_bus::build_time_sparse_from_bus_col(
            z,
            bus,
            twist_cols.wv,
            mem_inst.steps,
            pow2_cycle,
        )?;
        let rv = crate::memory_sidecar::cpu_bus::build_time_sparse_from_bus_col(
            z,
            bus,
            twist_cols.rv,
            mem_inst.steps,
            pow2_cycle,
        )?;
        let inc_at_write_addr = crate::memory_sidecar::cpu_bus::build_time_sparse_from_bus_col(
            z,
            bus,
            twist_cols.inc,
            mem_inst.steps,
            pow2_cycle,
        )?;

        let decoded = TwistDecodedColsSparse {
            ra_bits,
            wa_bits,
            has_read,
            has_write,
            wv,
            rv,
            inc_at_write_addr,
        };

        let init_sparse: Vec<(usize, K)> = match &mem_inst.init {
            MemInit::Zero => Vec::new(),
            MemInit::Sparse(pairs) => pairs
                .iter()
                .map(|(addr, val)| {
                    let addr_usize = usize::try_from(*addr).map_err(|_| {
                        PiCcsError::InvalidInput(format!("Twist: init address doesn't fit usize: addr={addr}"))
                    })?;
                    if addr_usize >= mem_inst.k {
                        return Err(PiCcsError::InvalidInput(format!(
                            "Twist: init address out of range: addr={addr} >= k={}",
                            mem_inst.k
                        )));
                    }
                    Ok((addr_usize, (*val).into()))
                })
                .collect::<Result<_, _>>()?,
        };

        let mut read_addr_oracle = TwistReadCheckAddrOracleSparseTime::new(
            init_sparse.clone(),
            r_cycle,
            decoded.has_read.clone(),
            decoded.rv.clone(),
            &decoded.ra_bits,
            decoded.has_write.clone(),
            &decoded.wa_bits,
            decoded.inc_at_write_addr.clone(),
        );
        let mut write_addr_oracle = TwistWriteCheckAddrOracleSparseTime::new(
            init_sparse,
            r_cycle,
            decoded.has_write.clone(),
            decoded.wv.clone(),
            &decoded.wa_bits,
            decoded.inc_at_write_addr.clone(),
        );

        let labels: [&[u8]; 2] = [b"twist/read_addr_pre".as_slice(), b"twist/write_addr_pre".as_slice()];
        let claimed_sums = vec![K::ZERO, K::ZERO];
        tr.append_message(b"twist/addr_pre_time/claim_idx", &(idx as u64).to_le_bytes());
        bind_batched_claim_sums(tr, b"twist/addr_pre_time/claimed_sums", &claimed_sums, &labels);

        let mut claims = [
            BatchedClaim {
                oracle: &mut read_addr_oracle,
                claimed_sum: K::ZERO,
                label: labels[0],
            },
            BatchedClaim {
                oracle: &mut write_addr_oracle,
                claimed_sum: K::ZERO,
                label: labels[1],
            },
        ];

        let (r_addr, per_claim_results) = run_batched_sumcheck_prover_ds(tr, b"twist/addr_pre_time", idx, &mut claims)?;
        if per_claim_results.len() != 2 {
            return Err(PiCcsError::ProtocolError(format!(
                "twist addr-pre per-claim results len()={}, expected 2",
                per_claim_results.len()
            )));
        }

        out.push(TwistAddrPreProverData {
            addr_pre: BatchedAddrProof {
                claimed_sums,
                round_polys: vec![
                    per_claim_results[0].round_polys.clone(),
                    per_claim_results[1].round_polys.clone(),
                ],
                r_addr: r_addr.clone(),
            },
            decoded,
            read_check_claim_sum: per_claim_results[0].final_value,
            write_check_claim_sum: per_claim_results[1].final_value,
        });
    }

    Ok(out)
}

pub(crate) fn prove_shout_addr_pre_time(
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    step: &StepWitnessBundle<Cmt, F, K>,
    cpu_bus: Option<&BusLayout>,
    ell_n: usize,
    r_cycle: &[K],
) -> Result<Vec<ShoutAddrPreProverData>, PiCcsError> {
    if step.lut_instances.is_empty() {
        return Ok(Vec::new());
    }
    let mut out = Vec::with_capacity(step.lut_instances.len());

    let bus =
        cpu_bus.ok_or_else(|| PiCcsError::InvalidInput("prove_shout_addr_pre_time requires shared_cpu_bus".into()))?;
    let cpu_z_k = crate::memory_sidecar::cpu_bus::decode_cpu_z_to_k(params, &step.mcs.1.Z);
    if bus.shout_cols.len() != step.lut_instances.len() || bus.twist_cols.len() != step.mem_instances.len() {
        return Err(PiCcsError::InvalidInput(
            "shared_cpu_bus layout mismatch for step (instance counts)".into(),
        ));
    }

    for (idx, (lut_inst, _lut_wit)) in step.lut_instances.iter().enumerate() {
        neo_memory::addr::validate_shout_bit_addressing(lut_inst)?;
        let pow2_cycle = 1usize << ell_n;
        if lut_inst.steps > pow2_cycle {
            return Err(PiCcsError::InvalidInput(format!(
                "Shout(Route A): steps={} exceeds 2^ell_cycle={pow2_cycle}",
                lut_inst.steps
            )));
        }

        let z = &cpu_z_k;
        let ell_addr = lut_inst.d * lut_inst.ell;
        let shout_cols = bus.shout_cols.get(idx).ok_or_else(|| {
            PiCcsError::InvalidInput(format!(
                "shared_cpu_bus layout mismatch: missing shout_cols for lut_idx={idx}"
            ))
        })?;
        if shout_cols.addr_bits.end - shout_cols.addr_bits.start != ell_addr {
            return Err(PiCcsError::InvalidInput(format!(
                "shared_cpu_bus layout mismatch at lut_idx={idx}: expected ell_addr={ell_addr}"
            )));
        }

        let mut addr_bits = Vec::with_capacity(ell_addr);
        for col_id in shout_cols.addr_bits.clone() {
            addr_bits.push(crate::memory_sidecar::cpu_bus::build_time_sparse_from_bus_col(
                z,
                bus,
                col_id,
                lut_inst.steps,
                pow2_cycle,
            )?);
        }

        let has_lookup = crate::memory_sidecar::cpu_bus::build_time_sparse_from_bus_col(
            z,
            bus,
            shout_cols.has_lookup,
            lut_inst.steps,
            pow2_cycle,
        )?;
        let val = crate::memory_sidecar::cpu_bus::build_time_sparse_from_bus_col(
            z,
            bus,
            shout_cols.val,
            lut_inst.steps,
            pow2_cycle,
        )?;

        let decoded = ShoutDecodedColsSparse {
            addr_bits,
            has_lookup,
            val,
        };
        let table_k: Vec<K> = lut_inst.table.iter().map(|&v| v.into()).collect();
        let (mut addr_oracle, addr_claim_sum) =
            AddressLookupOracle::new(&decoded.addr_bits, &decoded.has_lookup, &table_k, r_cycle, ell_addr);

        let labels: [&[u8]; 1] = [b"shout/addr_pre".as_slice()];
        let claimed_sums = vec![addr_claim_sum];
        tr.append_message(b"shout/addr_pre_time/claim_idx", &(idx as u64).to_le_bytes());
        bind_batched_claim_sums(tr, b"shout/addr_pre_time/claimed_sums", &claimed_sums, &labels);

        let mut claims = [BatchedClaim {
            oracle: &mut addr_oracle,
            claimed_sum: addr_claim_sum,
            label: labels[0],
        }];

        let (r_addr, per_claim_results) = run_batched_sumcheck_prover_ds(tr, b"shout/addr_pre_time", idx, &mut claims)?;

        if per_claim_results.len() != 1 {
            return Err(PiCcsError::ProtocolError(format!(
                "shout addr pre-time per-claim results len()={}, expected 1",
                per_claim_results.len()
            )));
        }

        out.push(ShoutAddrPreProverData {
            addr_pre: BatchedAddrProof {
                claimed_sums,
                round_polys: vec![per_claim_results[0].round_polys.clone()],
                r_addr,
            },
            decoded,
        });
    }

    Ok(out)
}

pub fn verify_shout_addr_pre_time(
    tr: &mut Poseidon2Transcript,
    step: &StepInstanceBundle<Cmt, F, K>,
    mem_proof: &MemSidecarProof<Cmt, F, K>,
) -> Result<Vec<ShoutAddrPreVerifyData>, PiCcsError> {
    let mut out = Vec::with_capacity(step.lut_insts.len());

    for (idx, lut_inst) in step.lut_insts.iter().enumerate() {
        let table_k: Vec<K> = lut_inst.table.iter().map(|&v| v.into()).collect();
        let proof = match mem_proof.proofs.get(idx) {
            Some(MemOrLutProof::Shout(p)) => p,
            _ => return Err(PiCcsError::InvalidInput("expected Shout proof".into())),
        };

        let ell_addr = lut_inst.d * lut_inst.ell;
        if proof.addr_pre.claimed_sums.len() != 1 {
            return Err(PiCcsError::InvalidInput(format!(
                "shout addr_pre claimed_sums.len()={}, expected 1",
                proof.addr_pre.claimed_sums.len()
            )));
        }
        if proof.addr_pre.round_polys.len() != 1 {
            return Err(PiCcsError::InvalidInput(format!(
                "shout addr_pre round_polys.len()={}, expected 1",
                proof.addr_pre.round_polys.len()
            )));
        }
        if proof.addr_pre.r_addr.len() != ell_addr {
            return Err(PiCcsError::InvalidInput(format!(
                "shout addr_pre r_addr.len()={}, expected ell_addr={}",
                proof.addr_pre.r_addr.len(),
                ell_addr
            )));
        }
        if proof
            .addr_pre
            .round_polys
            .first()
            .map(|rounds| rounds.len())
            .unwrap_or(0)
            != ell_addr
        {
            return Err(PiCcsError::InvalidInput(format!(
                "shout addr_pre round_polys[0].len()={}, expected ell_addr={}",
                proof
                    .addr_pre
                    .round_polys
                    .first()
                    .map(|rounds| rounds.len())
                    .unwrap_or(0),
                ell_addr
            )));
        }

        let labels: [&[u8]; 1] = [b"shout/addr_pre".as_slice()];
        tr.append_message(b"shout/addr_pre_time/claim_idx", &(idx as u64).to_le_bytes());
        bind_batched_claim_sums(
            tr,
            b"shout/addr_pre_time/claimed_sums",
            &proof.addr_pre.claimed_sums,
            &labels,
        );

        let degree_bounds = vec![2usize];
        let (r_addr, finals, ok) = verify_batched_sumcheck_rounds_ds(
            tr,
            b"shout/addr_pre_time",
            idx,
            &proof.addr_pre.round_polys,
            &proof.addr_pre.claimed_sums,
            &labels,
            &degree_bounds,
        );
        if !ok {
            return Err(PiCcsError::SumcheckError(
                "shout addr-pre batched sumcheck invalid".into(),
            ));
        }
        if r_addr != proof.addr_pre.r_addr {
            return Err(PiCcsError::ProtocolError(
                "shout addr_pre r_addr mismatch: transcript-derived vs proof".into(),
            ));
        }
        if finals.len() != 1 {
            return Err(PiCcsError::ProtocolError(format!(
                "shout addr-pre finals.len()={}, expected 1",
                finals.len()
            )));
        }
        let addr_claim_sum = proof.addr_pre.claimed_sums[0];
        let addr_final = finals[0];

        out.push(ShoutAddrPreVerifyData {
            addr_claim_sum,
            addr_final,
            r_addr,
            table_k,
        });
    }

    Ok(out)
}

pub fn verify_twist_addr_pre_time(
    tr: &mut Poseidon2Transcript,
    step: &StepInstanceBundle<Cmt, F, K>,
    mem_proof: &MemSidecarProof<Cmt, F, K>,
) -> Result<Vec<TwistAddrPreVerifyData>, PiCcsError> {
    let mut out = Vec::with_capacity(step.mem_insts.len());
    let proof_offset = step.lut_insts.len();

    for (idx, mem_inst) in step.mem_insts.iter().enumerate() {
        let proof = match mem_proof.proofs.get(proof_offset + idx) {
            Some(MemOrLutProof::Twist(p)) => p,
            _ => return Err(PiCcsError::InvalidInput("expected Twist proof".into())),
        };

        if proof.addr_pre.claimed_sums.len() != 2 {
            return Err(PiCcsError::InvalidInput(format!(
                "twist addr_pre claimed_sums.len()={}, expected 2",
                proof.addr_pre.claimed_sums.len()
            )));
        }
        if proof.addr_pre.round_polys.len() != 2 {
            return Err(PiCcsError::InvalidInput(format!(
                "twist addr_pre round_polys.len()={}, expected 2",
                proof.addr_pre.round_polys.len()
            )));
        }
        if proof.addr_pre.claimed_sums[0] != K::ZERO || proof.addr_pre.claimed_sums[1] != K::ZERO {
            return Err(PiCcsError::ProtocolError(
                "twist addr_pre claimed_sums mismatch (expected both 0)".into(),
            ));
        }

        let labels: [&[u8]; 2] = [b"twist/read_addr_pre".as_slice(), b"twist/write_addr_pre".as_slice()];
        let degree_bounds = vec![2usize, 2usize];
        tr.append_message(b"twist/addr_pre_time/claim_idx", &(idx as u64).to_le_bytes());
        bind_batched_claim_sums(
            tr,
            b"twist/addr_pre_time/claimed_sums",
            &proof.addr_pre.claimed_sums,
            &labels,
        );

        let (r_addr, finals, ok) = verify_batched_sumcheck_rounds_ds(
            tr,
            b"twist/addr_pre_time",
            idx,
            &proof.addr_pre.round_polys,
            &proof.addr_pre.claimed_sums,
            &labels,
            &degree_bounds,
        );
        if !ok {
            return Err(PiCcsError::SumcheckError(
                "twist addr-pre batched sumcheck invalid".into(),
            ));
        }
        if r_addr != proof.addr_pre.r_addr {
            return Err(PiCcsError::ProtocolError(
                "twist addr_pre r_addr mismatch: transcript-derived vs proof".into(),
            ));
        }

        let ell_addr = mem_inst.d * mem_inst.ell;
        if r_addr.len() != ell_addr {
            return Err(PiCcsError::InvalidInput(format!(
                "twist addr_pre r_addr.len()={}, expected ell_addr={}",
                r_addr.len(),
                ell_addr
            )));
        }
        if finals.len() != 2 {
            return Err(PiCcsError::ProtocolError(format!(
                "twist addr-pre finals.len()={}, expected 2",
                finals.len()
            )));
        }

        out.push(TwistAddrPreVerifyData {
            r_addr,
            read_check_claim_sum: finals[0],
            write_check_claim_sum: finals[1],
        });
    }

    Ok(out)
}

pub(crate) fn build_route_a_memory_oracles(
    _params: &NeoParams,
    step: &StepWitnessBundle<Cmt, F, K>,
    _ell_n: usize,
    r_cycle: &[K],
    shout_pre: &[ShoutAddrPreProverData],
    twist_pre: &[TwistAddrPreProverData],
) -> Result<RouteAMemoryOracles, PiCcsError> {
    if shout_pre.len() != step.lut_instances.len() {
        return Err(PiCcsError::InvalidInput(format!(
            "shout pre-time count mismatch (expected {}, got {})",
            step.lut_instances.len(),
            shout_pre.len()
        )));
    }
    if twist_pre.len() != step.mem_instances.len() {
        return Err(PiCcsError::InvalidInput(format!(
            "twist pre-time decoded count mismatch (expected {}, got {})",
            step.mem_instances.len(),
            twist_pre.len()
        )));
    }

    let mut shout_oracles = Vec::with_capacity(step.lut_instances.len());
    for ((lut_inst, _lut_wit), pre) in step.lut_instances.iter().zip(shout_pre.iter()) {
        let ell_addr = lut_inst.d * lut_inst.ell;
        if pre.addr_pre.r_addr.len() != ell_addr {
            return Err(PiCcsError::InvalidInput(format!(
                "Shout(Route A): r_addr.len()={} != ell_addr={}",
                pre.addr_pre.r_addr.len(),
                ell_addr
            )));
        }

        let (value_oracle, value_claim) =
            ShoutValueOracleSparse::new(r_cycle, pre.decoded.has_lookup.clone(), pre.decoded.val.clone());
        let (adapter_oracle, adapter_claim) = IndexAdapterOracleSparseTime::new_with_gate(
            r_cycle,
            pre.decoded.has_lookup.clone(),
            pre.decoded.addr_bits.clone(),
            &pre.addr_pre.r_addr,
        );

        let mut bitness: Vec<Box<dyn RoundOracle>> = Vec::with_capacity(ell_addr + 1);
        for col in pre.decoded.addr_bits.iter().cloned() {
            bitness.push(Box::new(LazyBitnessOracleSparseTime::new_with_cycle(r_cycle, col)));
        }
        bitness.push(Box::new(LazyBitnessOracleSparseTime::new_with_cycle(
            r_cycle,
            pre.decoded.has_lookup.clone(),
        )));

        shout_oracles.push(RouteAShoutTimeOracles {
            value: Box::new(value_oracle),
            value_claim,
            adapter: Box::new(adapter_oracle),
            adapter_claim,
            bitness,
            ell_addr,
        });
    }

    let mut twist_oracles = Vec::with_capacity(step.mem_instances.len());
    for ((mem_inst, _mem_wit), pre) in step.mem_instances.iter().zip(twist_pre.iter()) {
        let init_at_r_addr = eval_init_at_r_addr(&mem_inst.init, mem_inst.k, &pre.addr_pre.r_addr)?;
        let ell_addr = mem_inst.d * mem_inst.ell;
        if pre.addr_pre.r_addr.len() != ell_addr {
            return Err(PiCcsError::InvalidInput(format!(
                "Twist(Route A): r_addr.len()={} != ell_addr={}",
                pre.addr_pre.r_addr.len(),
                ell_addr
            )));
        }

        let read_check = TwistReadCheckOracleSparseTime::new(
            r_cycle,
            pre.decoded.has_read.clone(),
            pre.decoded.rv.clone(),
            pre.decoded.ra_bits.clone(),
            pre.decoded.has_write.clone(),
            pre.decoded.inc_at_write_addr.clone(),
            pre.decoded.wa_bits.clone(),
            &pre.addr_pre.r_addr,
            init_at_r_addr,
        );
        let write_check = TwistWriteCheckOracleSparseTime::new(
            r_cycle,
            pre.decoded.has_write.clone(),
            pre.decoded.wv.clone(),
            pre.decoded.inc_at_write_addr.clone(),
            pre.decoded.wa_bits.clone(),
            &pre.addr_pre.r_addr,
            init_at_r_addr,
        );

        let mut bitness: Vec<Box<dyn RoundOracle>> = Vec::with_capacity(2 * ell_addr + 2);
        for bits in pre
            .decoded
            .ra_bits
            .iter()
            .cloned()
            .chain(pre.decoded.wa_bits.iter().cloned())
        {
            bitness.push(Box::new(LazyBitnessOracleSparseTime::new_with_cycle(r_cycle, bits)));
        }
        bitness.push(Box::new(LazyBitnessOracleSparseTime::new_with_cycle(
            r_cycle,
            pre.decoded.has_read.clone(),
        )));
        bitness.push(Box::new(LazyBitnessOracleSparseTime::new_with_cycle(
            r_cycle,
            pre.decoded.has_write.clone(),
        )));

        twist_oracles.push(RouteATwistTimeOracles {
            read_check: Box::new(read_check),
            write_check: Box::new(write_check),
            bitness,
            ell_addr,
        });
    }

    Ok(RouteAMemoryOracles {
        shout: shout_oracles,
        twist: twist_oracles,
    })
}

pub struct RouteAShoutTimeClaimsGuard<'a> {
    pub value_prefixes: Vec<RoundOraclePrefix<'a>>,
    pub adapter_prefixes: Vec<RoundOraclePrefix<'a>>,
    pub value_claims: Vec<K>,
    pub adapter_claims: Vec<K>,
    pub bitness: Vec<Vec<Box<dyn RoundOracle>>>,
}

pub fn build_route_a_shout_time_claims_guard<'a>(
    shout_oracles: &'a mut [RouteAShoutTimeOracles],
    ell_n: usize,
) -> RouteAShoutTimeClaimsGuard<'a> {
    let mut value_prefixes: Vec<RoundOraclePrefix<'a>> = Vec::with_capacity(shout_oracles.len());
    let mut adapter_prefixes: Vec<RoundOraclePrefix<'a>> = Vec::with_capacity(shout_oracles.len());
    let mut value_claims: Vec<K> = Vec::with_capacity(shout_oracles.len());
    let mut adapter_claims: Vec<K> = Vec::with_capacity(shout_oracles.len());
    let mut bitness: Vec<Vec<Box<dyn RoundOracle>>> = Vec::with_capacity(shout_oracles.len());

    for o in shout_oracles.iter_mut() {
        bitness.push(core::mem::take(&mut o.bitness));
        value_claims.push(o.value_claim);
        adapter_claims.push(o.adapter_claim);
        value_prefixes.push(RoundOraclePrefix::new(o.value.as_mut(), ell_n));
        adapter_prefixes.push(RoundOraclePrefix::new(o.adapter.as_mut(), ell_n));
    }

    RouteAShoutTimeClaimsGuard {
        value_prefixes,
        adapter_prefixes,
        value_claims,
        adapter_claims,
        bitness,
    }
}

pub struct ShoutRouteAProtocol<'a> {
    guard: RouteAShoutTimeClaimsGuard<'a>,
}

impl<'a> ShoutRouteAProtocol<'a> {
    pub fn new(shout_oracles: &'a mut [RouteAShoutTimeOracles], ell_n: usize) -> Self {
        Self {
            guard: build_route_a_shout_time_claims_guard(shout_oracles, ell_n),
        }
    }
}

impl<'o> TimeBatchedClaims for ShoutRouteAProtocol<'o> {
    fn append_time_claims<'a>(
        &'a mut self,
        _ell_n: usize,
        claimed_sums: &mut Vec<K>,
        degree_bounds: &mut Vec<usize>,
        labels: &mut Vec<&'static [u8]>,
        claim_is_dynamic: &mut Vec<bool>,
        claims: &mut Vec<BatchedClaim<'a>>,
    ) {
        append_route_a_shout_time_claims(
            &mut self.guard,
            claimed_sums,
            degree_bounds,
            labels,
            claim_is_dynamic,
            claims,
        );
    }
}

pub fn append_route_a_shout_time_claims<'a>(
    guard: &'a mut RouteAShoutTimeClaimsGuard<'_>,
    claimed_sums: &mut Vec<K>,
    degree_bounds: &mut Vec<usize>,
    labels: &mut Vec<&'static [u8]>,
    claim_is_dynamic: &mut Vec<bool>,
    claims: &mut Vec<BatchedClaim<'a>>,
) {
    for (((value_time, adapter_time), bitness_vec), (value_claim, adapter_claim)) in guard
        .value_prefixes
        .iter_mut()
        .zip(guard.adapter_prefixes.iter_mut())
        .zip(guard.bitness.iter_mut())
        .zip(guard.value_claims.iter().zip(guard.adapter_claims.iter()))
    {
        claimed_sums.push(*value_claim);
        degree_bounds.push(value_time.degree_bound());
        labels.push(b"shout/value");
        claim_is_dynamic.push(true);
        claims.push(BatchedClaim {
            oracle: value_time,
            claimed_sum: *value_claim,
            label: b"shout/value",
        });

        claimed_sums.push(*adapter_claim);
        degree_bounds.push(adapter_time.degree_bound());
        labels.push(b"shout/adapter");
        claim_is_dynamic.push(true);
        claims.push(BatchedClaim {
            oracle: adapter_time,
            claimed_sum: *adapter_claim,
            label: b"shout/adapter",
        });

        for bit_oracle in bitness_vec.iter_mut() {
            claimed_sums.push(K::ZERO);
            degree_bounds.push(bit_oracle.degree_bound());
            labels.push(b"shout/bitness");
            claim_is_dynamic.push(false);
            claims.push(BatchedClaim {
                oracle: bit_oracle.as_mut(),
                claimed_sum: K::ZERO,
                label: b"shout/bitness",
            });
        }
    }
}

pub struct RouteATwistTimeClaimsGuard<'a> {
    pub read_check_prefixes: Vec<RoundOraclePrefix<'a>>,
    pub write_check_prefixes: Vec<RoundOraclePrefix<'a>>,
    pub read_check_claims: Vec<K>,
    pub write_check_claims: Vec<K>,
    pub bitness: Vec<Vec<Box<dyn RoundOracle>>>,
}

pub fn build_route_a_twist_time_claims_guard<'a>(
    twist_oracles: &'a mut [RouteATwistTimeOracles],
    ell_n: usize,
    read_check_claims: Vec<K>,
    write_check_claims: Vec<K>,
) -> RouteATwistTimeClaimsGuard<'a> {
    let mut read_check_prefixes: Vec<RoundOraclePrefix<'a>> = Vec::with_capacity(twist_oracles.len());
    let mut write_check_prefixes: Vec<RoundOraclePrefix<'a>> = Vec::with_capacity(twist_oracles.len());
    let mut bitness: Vec<Vec<Box<dyn RoundOracle>>> = Vec::with_capacity(twist_oracles.len());

    if read_check_claims.len() != twist_oracles.len() {
        panic!(
            "twist read-check claim count mismatch (claims={}, oracles={})",
            read_check_claims.len(),
            twist_oracles.len()
        );
    }
    if write_check_claims.len() != twist_oracles.len() {
        panic!(
            "twist write-check claim count mismatch (claims={}, oracles={})",
            write_check_claims.len(),
            twist_oracles.len()
        );
    }

    for o in twist_oracles.iter_mut() {
        bitness.push(core::mem::take(&mut o.bitness));
        read_check_prefixes.push(RoundOraclePrefix::new(o.read_check.as_mut(), ell_n));
        write_check_prefixes.push(RoundOraclePrefix::new(o.write_check.as_mut(), ell_n));
    }

    RouteATwistTimeClaimsGuard {
        read_check_prefixes,
        write_check_prefixes,
        read_check_claims,
        write_check_claims,
        bitness,
    }
}

pub fn append_route_a_twist_time_claims<'a>(
    guard: &'a mut RouteATwistTimeClaimsGuard<'_>,
    claimed_sums: &mut Vec<K>,
    degree_bounds: &mut Vec<usize>,
    labels: &mut Vec<&'static [u8]>,
    claim_is_dynamic: &mut Vec<bool>,
    claims: &mut Vec<BatchedClaim<'a>>,
) {
    for (((read_check_time, write_check_time), bitness_vec), (read_claim, write_claim)) in guard
        .read_check_prefixes
        .iter_mut()
        .zip(guard.write_check_prefixes.iter_mut())
        .zip(guard.bitness.iter_mut())
        .zip(
            guard
                .read_check_claims
                .iter()
                .zip(guard.write_check_claims.iter()),
        )
    {
        claimed_sums.push(*read_claim);
        degree_bounds.push(read_check_time.degree_bound());
        labels.push(b"twist/read_check");
        claim_is_dynamic.push(true);
        claims.push(BatchedClaim {
            oracle: read_check_time,
            claimed_sum: *read_claim,
            label: b"twist/read_check",
        });

        claimed_sums.push(*write_claim);
        degree_bounds.push(write_check_time.degree_bound());
        labels.push(b"twist/write_check");
        claim_is_dynamic.push(true);
        claims.push(BatchedClaim {
            oracle: write_check_time,
            claimed_sum: *write_claim,
            label: b"twist/write_check",
        });

        for bit_oracle in bitness_vec.iter_mut() {
            claimed_sums.push(K::ZERO);
            degree_bounds.push(bit_oracle.degree_bound());
            labels.push(b"twist/bitness");
            claim_is_dynamic.push(false);
            claims.push(BatchedClaim {
                oracle: bit_oracle.as_mut(),
                claimed_sum: K::ZERO,
                label: b"twist/bitness",
            });
        }
    }
}

pub struct TwistRouteAProtocol<'a> {
    guard: RouteATwistTimeClaimsGuard<'a>,
}

impl<'a> TwistRouteAProtocol<'a> {
    pub fn new(
        twist_oracles: &'a mut [RouteATwistTimeOracles],
        ell_n: usize,
        read_check_claims: Vec<K>,
        write_check_claims: Vec<K>,
    ) -> Self {
        Self {
            guard: build_route_a_twist_time_claims_guard(twist_oracles, ell_n, read_check_claims, write_check_claims),
        }
    }
}

impl<'o> TimeBatchedClaims for TwistRouteAProtocol<'o> {
    fn append_time_claims<'a>(
        &'a mut self,
        _ell_n: usize,
        claimed_sums: &mut Vec<K>,
        degree_bounds: &mut Vec<usize>,
        labels: &mut Vec<&'static [u8]>,
        claim_is_dynamic: &mut Vec<bool>,
        claims: &mut Vec<BatchedClaim<'a>>,
    ) {
        append_route_a_twist_time_claims(
            &mut self.guard,
            claimed_sums,
            degree_bounds,
            labels,
            claim_is_dynamic,
            claims,
        );
    }
}

pub(crate) fn finalize_route_a_memory_prover(
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s: &CcsStructure<F>,
    step: &StepWitnessBundle<Cmt, F, K>,
    prev_step: Option<&StepWitnessBundle<Cmt, F, K>>,
    prev_twist_decoded: Option<&[TwistDecodedColsSparse]>,
    oracles: &mut RouteAMemoryOracles,
    shout_pre: &[ShoutAddrPreProverData],
    twist_pre: &[TwistAddrPreProverData],
    r_time: &[K],
    m_in: usize,
    step_idx: usize,
) -> Result<MemSidecarProof<Cmt, F, K>, PiCcsError> {
    let has_prev = prev_step.is_some();
    if has_prev != prev_twist_decoded.is_some() {
        return Err(PiCcsError::InvalidInput(format!(
            "Twist rollover decoded cache mismatch: prev_step.is_some()={} but prev_twist_decoded.is_some()={}",
            has_prev,
            prev_twist_decoded.is_some()
        )));
    }
    if shout_pre.len() != step.lut_instances.len() {
        return Err(PiCcsError::InvalidInput(format!(
            "shout pre-time count mismatch (expected {}, got {})",
            step.lut_instances.len(),
            shout_pre.len()
        )));
    }
    if twist_pre.len() != step.mem_instances.len() {
        return Err(PiCcsError::InvalidInput(format!(
            "twist pre-time count mismatch (expected {}, got {})",
            step.mem_instances.len(),
            twist_pre.len()
        )));
    }
    if oracles.twist.len() != step.mem_instances.len() {
        return Err(PiCcsError::InvalidInput(format!(
            "twist oracle count mismatch (expected {}, got {})",
            step.mem_instances.len(),
            oracles.twist.len()
        )));
    }

    for (idx, (lut_inst, lut_wit)) in step.lut_instances.iter().enumerate() {
        if !lut_inst.comms.is_empty() || !lut_wit.mats.is_empty() {
            return Err(PiCcsError::InvalidInput(format!(
                "shared CPU bus requires metadata-only Shout instances (comms/mats must be empty, lut_idx={idx})"
            )));
        }
    }
    for (idx, (mem_inst, mem_wit)) in step.mem_instances.iter().enumerate() {
        if !mem_inst.comms.is_empty() || !mem_wit.mats.is_empty() {
            return Err(PiCcsError::InvalidInput(format!(
                "shared CPU bus requires metadata-only Twist instances (comms/mats must be empty, mem_idx={idx})"
            )));
        }
    }
    if let Some(prev) = prev_step {
        if prev.mem_instances.len() != step.mem_instances.len() {
            return Err(PiCcsError::InvalidInput(format!(
                "Twist rollover requires stable mem instance count: prev has {}, current has {}",
                prev.mem_instances.len(),
                step.mem_instances.len()
            )));
        }
        for (idx, (mem_inst, mem_wit)) in prev.mem_instances.iter().enumerate() {
            if !mem_inst.comms.is_empty() || !mem_wit.mats.is_empty() {
                return Err(PiCcsError::InvalidInput(format!(
                    "shared CPU bus requires metadata-only Twist instances (comms/mats must be empty, prev mem_idx={idx})"
                )));
            }
        }
    }
    let mut cpu_me_claims_val: Vec<MeInstance<Cmt, F, K>> = Vec::new();
    let mut proofs: Vec<MemOrLutProof> = Vec::new();

    // --------------------------------------------------------------------
    // Phase 2: Twist val-eval sum-check (batched across mem instances).
    // --------------------------------------------------------------------
    let mut twist_val_eval_proofs: Vec<twist::TwistValEvalProof<K>> = Vec::new();
    let mut r_val: Vec<K> = Vec::new();
    if !step.mem_instances.is_empty() {
        let plan = crate::memory_sidecar::claim_plan::TwistValEvalClaimPlan::build(
            step.mem_instances.iter().map(|(inst, _)| inst),
            has_prev,
        );
        let n_mem = step.mem_instances.len();
        let claims_per_mem = plan.claims_per_mem;
        let claim_count = plan.claim_count;

        let mut val_oracles: Vec<Box<dyn RoundOracle>> = Vec::with_capacity(claim_count);
        let mut bind_claims: Vec<(u8, K)> = Vec::with_capacity(claim_count);
        let mut claimed_sums: Vec<K> = Vec::with_capacity(claim_count);

        let mut claimed_inc_sums_lt: Vec<K> = Vec::with_capacity(n_mem);
        let mut claimed_inc_sums_total: Vec<K> = Vec::with_capacity(n_mem);
        let mut claimed_prev_inc_sums_total: Vec<Option<K>> = Vec::with_capacity(n_mem);

        let mut claim_idx = 0usize;
        for (i_mem, (mem_inst, _mem_wit)) in step.mem_instances.iter().enumerate() {
            let pre = twist_pre
                .get(i_mem)
                .ok_or_else(|| PiCcsError::ProtocolError("missing Twist pre-time data".into()))?;
            let decoded = &pre.decoded;
            let r_addr = &pre.addr_pre.r_addr;
            let (oracle_lt, claimed_inc_sum_lt) = TwistValEvalOracleSparseTime::new(
                decoded.wa_bits.clone(),
                decoded.has_write.clone(),
                decoded.inc_at_write_addr.clone(),
                r_addr,
                r_time,
            );
            let (oracle_total, claimed_inc_sum_total) = TwistTotalIncOracleSparseTime::new(
                decoded.wa_bits.clone(),
                decoded.has_write.clone(),
                decoded.inc_at_write_addr.clone(),
                r_addr,
            );

            val_oracles.push(Box::new(oracle_lt));
            bind_claims.push((plan.bind_tags[claim_idx], claimed_inc_sum_lt));
            claimed_sums.push(claimed_inc_sum_lt);
            claim_idx += 1;

            val_oracles.push(Box::new(oracle_total));
            bind_claims.push((plan.bind_tags[claim_idx], claimed_inc_sum_total));
            claimed_sums.push(claimed_inc_sum_total);
            claim_idx += 1;

            claimed_inc_sums_lt.push(claimed_inc_sum_lt);
            claimed_inc_sums_total.push(claimed_inc_sum_total);

            if let Some(prev) = prev_step {
                let (prev_inst, _prev_wit) = prev
                    .mem_instances
                    .get(i_mem)
                    .ok_or_else(|| PiCcsError::ProtocolError("missing prev mem instance".into()))?;
                if prev_inst.d != mem_inst.d || prev_inst.ell != mem_inst.ell || prev_inst.k != mem_inst.k {
                    return Err(PiCcsError::InvalidInput(format!(
                        "Twist rollover requires stable geometry at mem_idx={}: prev (k={}, d={}, ell={}) vs cur (k={}, d={}, ell={})",
                        i_mem, prev_inst.k, prev_inst.d, prev_inst.ell, mem_inst.k, mem_inst.d, mem_inst.ell
                    )));
                }
                let prev_decoded = prev_twist_decoded
                    .ok_or_else(|| PiCcsError::ProtocolError("missing prev Twist decoded cols".into()))?
                    .get(i_mem)
                    .ok_or_else(|| PiCcsError::ProtocolError("missing prev Twist decoded cols at mem_idx".into()))?;
                let (oracle_prev_total, claimed_prev_total) = TwistTotalIncOracleSparseTime::new(
                    prev_decoded.wa_bits.clone(),
                    prev_decoded.has_write.clone(),
                    prev_decoded.inc_at_write_addr.clone(),
                    r_addr,
                );

                val_oracles.push(Box::new(oracle_prev_total));
                bind_claims.push((plan.bind_tags[claim_idx], claimed_prev_total));
                claimed_sums.push(claimed_prev_total);
                claim_idx += 1;

                claimed_prev_inc_sums_total.push(Some(claimed_prev_total));
            } else {
                claimed_prev_inc_sums_total.push(None);
            }
        }

        tr.append_message(
            b"twist/val_eval/batch_start",
            &(step.mem_instances.len() as u64).to_le_bytes(),
        );
        tr.append_message(b"twist/val_eval/step_idx", &(step_idx as u64).to_le_bytes());
        bind_twist_val_eval_claim_sums(tr, &bind_claims);

        let mut claims: Vec<BatchedClaim<'_>> = val_oracles
            .iter_mut()
            .zip(claimed_sums.iter())
            .zip(plan.labels.iter())
            .map(|((oracle, sum), label)| BatchedClaim {
                oracle: oracle.as_mut(),
                claimed_sum: *sum,
                label: *label,
            })
            .collect();

        let (r_val_out, per_claim_results) =
            run_batched_sumcheck_prover_ds(tr, b"twist/val_eval_batch", step_idx, claims.as_mut_slice())?;

        if per_claim_results.len() != claim_count {
            return Err(PiCcsError::ProtocolError(format!(
                "twist val-eval results count mismatch (expected {}, got {})",
                claim_count,
                per_claim_results.len()
            )));
        }
        if r_val_out.len() != r_time.len() {
            return Err(PiCcsError::ProtocolError(format!(
                "twist val-eval r_val.len()={}, expected ell_n={}",
                r_val_out.len(),
                r_time.len()
            )));
        }
        r_val = r_val_out;

        for i in 0..n_mem {
            let base = claims_per_mem * i;
            twist_val_eval_proofs.push(twist::TwistValEvalProof {
                claimed_inc_sum_lt: claimed_inc_sums_lt[i],
                rounds_lt: per_claim_results[base].round_polys.clone(),
                claimed_inc_sum_total: claimed_inc_sums_total[i],
                rounds_total: per_claim_results[base + 1].round_polys.clone(),
                claimed_prev_inc_sum_total: claimed_prev_inc_sums_total[i],
                rounds_prev_total: has_prev.then(|| per_claim_results[base + 2].round_polys.clone()),
            });
        }

        tr.append_message(b"twist/val_eval/batch_done", &[]);
    }

    for pre in shout_pre {
        let mut proof = ShoutProofK::default();
        proof.addr_pre = pre.addr_pre.clone();
        proofs.push(MemOrLutProof::Shout(proof));
    }

    for idx in 0..step.mem_instances.len() {
        let mut proof = TwistProofK::default();
        proof.addr_pre = twist_pre
            .get(idx)
            .ok_or_else(|| PiCcsError::ProtocolError("missing Twist addr_pre".into()))?
            .addr_pre
            .clone();
        proof.val_eval = twist_val_eval_proofs.get(idx).cloned();

        proofs.push(MemOrLutProof::Twist(proof));
    }

    if !step.mem_instances.is_empty() {
        if r_val.len() != r_time.len() {
            return Err(PiCcsError::ProtocolError(format!(
                "twist val-eval r_val.len()={}, expected ell_n={}",
                r_val.len(),
                r_time.len()
            )));
        }

        // In shared-bus mode, val-lane checks read bus openings from CPU ME claims at r_val.
        // Emit CPU ME at r_val for current step (and previous step for rollover).
        let (mcs_inst, mcs_wit) = &step.mcs;
        let cpu_claims_cur = ts::emit_me_claims_for_mats(
            tr,
            b"cpu_bus/me_digest_val",
            params,
            s,
            core::slice::from_ref(&mcs_inst.c),
            core::slice::from_ref(&mcs_wit.Z),
            &r_val,
            m_in,
        )?;
        if cpu_claims_cur.len() != 1 {
            return Err(PiCcsError::ProtocolError(format!(
                "expected exactly 1 CPU ME claim at r_val, got {}",
                cpu_claims_cur.len()
            )));
        }
        cpu_me_claims_val.extend(cpu_claims_cur);

        if let Some(prev) = prev_step {
            let (prev_mcs_inst, prev_mcs_wit) = &prev.mcs;
            let cpu_claims_prev = ts::emit_me_claims_for_mats(
                tr,
                b"cpu_bus/me_digest_val",
                params,
                s,
                core::slice::from_ref(&prev_mcs_inst.c),
                core::slice::from_ref(&prev_mcs_wit.Z),
                &r_val,
                m_in,
            )?;
            if cpu_claims_prev.len() != 1 {
                return Err(PiCcsError::ProtocolError(format!(
                    "expected exactly 1 prev CPU ME claim at r_val, got {}",
                    cpu_claims_prev.len()
                )));
            }
            cpu_me_claims_val.extend(cpu_claims_prev);
        }
    }

    if step.mem_instances.is_empty() {
        if !twist_val_eval_proofs.is_empty() {
            return Err(PiCcsError::ProtocolError(
                "twist val-eval proofs must be empty when no mem instances are present".into(),
            ));
        }
        if !r_val.is_empty() {
            return Err(PiCcsError::ProtocolError(
                "twist r_val must be empty when no mem instances are present".into(),
            ));
        }
        if !cpu_me_claims_val.is_empty() {
            return Err(PiCcsError::ProtocolError(
                "twist val-lane ME claims must be empty when no mem instances are present".into(),
            ));
        }
    } else if cpu_me_claims_val.is_empty() {
        return Err(PiCcsError::ProtocolError(
            "twist val-eval requires non-empty val-lane ME claims".into(),
        ));
    }

    Ok(MemSidecarProof {
        cpu_me_claims_val,
        proofs,
    })
}

// ============================================================================
// ============================================================================
pub fn verify_route_a_memory_step(
    tr: &mut Poseidon2Transcript,
    cpu_bus: &BusLayout,
    step: &StepInstanceBundle<Cmt, F, K>,
    prev_step: Option<&StepInstanceBundle<Cmt, F, K>>,
    ccs_out0: &MeInstance<Cmt, F, K>,
    r_time: &[K],
    r_cycle: &[K],
    batched_final_values: &[K],
    batched_claimed_sums: &[K],
    claim_idx_start: usize,
    mem_proof: &MemSidecarProof<Cmt, F, K>,
    shout_pre: &[ShoutAddrPreVerifyData],
    twist_pre: &[TwistAddrPreVerifyData],
    step_idx: usize,
) -> Result<RouteAMemoryVerifyOutput, PiCcsError> {
    let chi_cycle_at_r_time = eq_points(r_time, r_cycle);
    if ccs_out0.r.as_slice() != r_time {
        return Err(PiCcsError::ProtocolError(
            "CPU ME output r mismatch (expected shared r_time)".into(),
        ));
    }
    let has_prev = prev_step.is_some();
    if let Some(prev) = prev_step {
        if prev.mem_insts.len() != step.mem_insts.len() {
            return Err(PiCcsError::InvalidInput(format!(
                "Twist rollover requires stable mem instance count: prev has {}, current has {}",
                prev.mem_insts.len(),
                step.mem_insts.len()
            )));
        }
        for (idx, (prev_inst, inst)) in prev.mem_insts.iter().zip(step.mem_insts.iter()).enumerate() {
            if prev_inst.d != inst.d || prev_inst.ell != inst.ell || prev_inst.k != inst.k {
                return Err(PiCcsError::InvalidInput(format!(
                    "Twist rollover requires stable geometry at mem_idx={}: prev (k={}, d={}, ell={}) vs cur (k={}, d={}, ell={})",
                    idx, prev_inst.k, prev_inst.d, prev_inst.ell, inst.k, inst.d, inst.ell
                )));
            }
        }
    }

    for (idx, inst) in step.lut_insts.iter().enumerate() {
        if !inst.comms.is_empty() {
            return Err(PiCcsError::InvalidInput(format!(
                "shared CPU bus requires metadata-only Shout instances (comms must be empty, lut_idx={idx})"
            )));
        }
    }
    for (idx, inst) in step.mem_insts.iter().enumerate() {
        if !inst.comms.is_empty() {
            return Err(PiCcsError::InvalidInput(format!(
                "shared CPU bus requires metadata-only Twist instances (comms must be empty, mem_idx={idx})"
            )));
        }
    }
    if let Some(prev) = prev_step {
        for (idx, inst) in prev.lut_insts.iter().enumerate() {
            if !inst.comms.is_empty() {
                return Err(PiCcsError::InvalidInput(format!(
                    "shared CPU bus requires metadata-only Shout instances (comms must be empty, prev lut_idx={idx})"
                )));
            }
        }
        for (idx, inst) in prev.mem_insts.iter().enumerate() {
            if !inst.comms.is_empty() {
                return Err(PiCcsError::InvalidInput(format!(
                    "shared CPU bus requires metadata-only Twist instances (comms must be empty, prev mem_idx={idx})"
                )));
            }
        }
    }

    let proofs_mem = &mem_proof.proofs;

    if cpu_bus.shout_cols.len() != step.lut_insts.len() || cpu_bus.twist_cols.len() != step.mem_insts.len() {
        return Err(PiCcsError::InvalidInput(
            "shared_cpu_bus layout mismatch for step (instance counts)".into(),
        ));
    }

    let bus_y_base_time = if cpu_bus.bus_cols > 0 {
        ccs_out0
            .y_scalars
            .len()
            .checked_sub(cpu_bus.bus_cols)
            .ok_or_else(|| PiCcsError::InvalidInput("CPU y_scalars too short for bus openings".into()))?
    } else {
        0usize
    };
    let claim_plan = RouteATimeClaimPlan::build(step, claim_idx_start)?;
    if claim_plan.claim_idx_end > batched_final_values.len() {
        return Err(PiCcsError::InvalidInput(format!(
            "batched_final_values too short (need at least {}, have {})",
            claim_plan.claim_idx_end,
            batched_final_values.len()
        )));
    }
    if claim_plan.claim_idx_end > batched_claimed_sums.len() {
        return Err(PiCcsError::InvalidInput(format!(
            "batched_claimed_sums too short (need at least {}, have {})",
            claim_plan.claim_idx_end,
            batched_claimed_sums.len()
        )));
    }

    let expected_proofs = step.lut_insts.len() + step.mem_insts.len();
    if proofs_mem.len() != expected_proofs {
        return Err(PiCcsError::InvalidInput(format!(
            "mem proof count mismatch (expected {}, got {})",
            expected_proofs,
            proofs_mem.len()
        )));
    }
    if shout_pre.len() != step.lut_insts.len() {
        return Err(PiCcsError::InvalidInput(format!(
            "shout pre-time count mismatch (expected {}, got {})",
            step.lut_insts.len(),
            shout_pre.len()
        )));
    }
    if twist_pre.len() != step.mem_insts.len() {
        return Err(PiCcsError::InvalidInput(format!(
            "twist pre-time count mismatch (expected {}, got {})",
            step.mem_insts.len(),
            twist_pre.len()
        )));
    }

    let mut twist_time_openings: Vec<TwistTimeLaneOpenings> = Vec::with_capacity(step.mem_insts.len());

    // Shout instances first.
    for (proof_idx, inst) in step.lut_insts.iter().enumerate() {
        match &proofs_mem[proof_idx] {
            MemOrLutProof::Shout(_proof) => {}
            _ => return Err(PiCcsError::InvalidInput("expected Shout proof".into())),
        }

        let layout = inst.shout_layout();
        let ell_addr = layout.ell_addr;

        let shout_cols = cpu_bus
            .shout_cols
            .get(proof_idx)
            .ok_or_else(|| PiCcsError::InvalidInput("shared_cpu_bus layout mismatch (shout)".into()))?;
        if shout_cols.addr_bits.end - shout_cols.addr_bits.start != ell_addr {
            return Err(PiCcsError::InvalidInput(format!(
                "shared_cpu_bus layout mismatch at lut_idx={proof_idx}: expected ell_addr={ell_addr}"
            )));
        }

        let mut addr_bits_open = Vec::with_capacity(ell_addr);
        for (_j, col_id) in shout_cols.addr_bits.clone().enumerate() {
            addr_bits_open.push(
                ccs_out0
                    .y_scalars
                    .get(cpu_bus.y_scalar_index(bus_y_base_time, col_id))
                    .copied()
                    .ok_or_else(|| PiCcsError::ProtocolError("CPU y_scalars missing Shout addr_bits opening".into()))?,
            );
        }
        let has_lookup_open = ccs_out0
            .y_scalars
            .get(cpu_bus.y_scalar_index(bus_y_base_time, shout_cols.has_lookup))
            .copied()
            .ok_or_else(|| PiCcsError::ProtocolError("CPU y_scalars missing Shout has_lookup opening".into()))?;
        let val_open = ccs_out0
            .y_scalars
            .get(cpu_bus.y_scalar_index(bus_y_base_time, shout_cols.val))
            .copied()
            .ok_or_else(|| PiCcsError::ProtocolError("CPU y_scalars missing Shout val opening".into()))?;

        let pre = shout_pre
            .get(proof_idx)
            .ok_or_else(|| PiCcsError::InvalidInput(format!("missing pre-time Shout data at index {}", proof_idx)))?;

        let shout_claims = claim_plan
            .shout
            .get(proof_idx)
            .ok_or_else(|| PiCcsError::ProtocolError(format!("missing Shout claim schedule at index {}", proof_idx)))?;

        // Route A Shout ordering in batched_time:
        // - value (time rounds only)
        // - adapter (time rounds only)
        // - bitness for addr_bits then has_lookup
        let value_claim = batched_claimed_sums[shout_claims.value];
        let value_final = batched_final_values[shout_claims.value];
        let adapter_claim = batched_claimed_sums[shout_claims.adapter];
        let adapter_final = batched_final_values[shout_claims.adapter];
        for (j, b_open) in addr_bits_open.iter().enumerate() {
            let bitness_idx = shout_claims
                .bitness_addr_bits
                .start
                .checked_add(j)
                .ok_or_else(|| PiCcsError::ProtocolError("bitness index overflow".into()))?;
            check_bitness_terminal(
                chi_cycle_at_r_time,
                *b_open,
                batched_final_values[bitness_idx],
                "shout bitness addr_bits",
            )?;
        }
        check_bitness_terminal(
            chi_cycle_at_r_time,
            has_lookup_open,
            batched_final_values[shout_claims.bitness_has_lookup],
            "shout bitness has_lookup",
        )?;

        let expected_value_final = chi_cycle_at_r_time * has_lookup_open * val_open;
        if expected_value_final != value_final {
            return Err(PiCcsError::ProtocolError("shout value terminal value mismatch".into()));
        }

        let eq_addr = eq_bits_prod(&addr_bits_open, &pre.r_addr)?;
        let expected_adapter_final = chi_cycle_at_r_time * has_lookup_open * eq_addr;
        if expected_adapter_final != adapter_final {
            return Err(PiCcsError::ProtocolError(
                "shout adapter terminal value mismatch".into(),
            ));
        }

        if value_claim != pre.addr_claim_sum {
            return Err(PiCcsError::ProtocolError(
                "shout value claimed sum != addr claimed sum".into(),
            ));
        }

        let table_eval = table_mle_eval(&pre.table_k, &pre.r_addr);
        let expected_addr_final = table_eval * adapter_claim;
        if expected_addr_final != pre.addr_final {
            return Err(PiCcsError::ProtocolError("shout addr terminal value mismatch".into()));
        }
    }

    // Twist instances next.
    let proof_mem_offset = step.lut_insts.len();

    // --------------------------------------------------------------------
    // Twist time checks at addr-pre `r_addr`.
    // --------------------------------------------------------------------
    for (i_mem, inst) in step.mem_insts.iter().enumerate() {
        let twist_proof = match &proofs_mem[proof_mem_offset + i_mem] {
            MemOrLutProof::Twist(proof) => proof,
            _ => return Err(PiCcsError::InvalidInput("expected Twist proof".into())),
        };
        let layout = inst.twist_layout();
        let ell_addr = layout.ell_addr;

        let twist_cols = cpu_bus
            .twist_cols
            .get(i_mem)
            .ok_or_else(|| PiCcsError::InvalidInput("shared_cpu_bus layout mismatch (twist)".into()))?;
        if twist_cols.ra_bits.end - twist_cols.ra_bits.start != ell_addr
            || twist_cols.wa_bits.end - twist_cols.wa_bits.start != ell_addr
        {
            return Err(PiCcsError::InvalidInput(format!(
                "shared_cpu_bus layout mismatch at mem_idx={i_mem}: expected ell_addr={ell_addr}"
            )));
        }

        let mut ra_bits_open = Vec::with_capacity(ell_addr);
        for col_id in twist_cols.ra_bits.clone() {
            ra_bits_open.push(
                ccs_out0
                    .y_scalars
                    .get(cpu_bus.y_scalar_index(bus_y_base_time, col_id))
                    .copied()
                    .ok_or_else(|| PiCcsError::ProtocolError("CPU y_scalars missing Twist ra_bits opening".into()))?,
            );
        }
        let mut wa_bits_open = Vec::with_capacity(ell_addr);
        for col_id in twist_cols.wa_bits.clone() {
            wa_bits_open.push(
                ccs_out0
                    .y_scalars
                    .get(cpu_bus.y_scalar_index(bus_y_base_time, col_id))
                    .copied()
                    .ok_or_else(|| PiCcsError::ProtocolError("CPU y_scalars missing Twist wa_bits opening".into()))?,
            );
        }

        let has_read_open = ccs_out0
            .y_scalars
            .get(cpu_bus.y_scalar_index(bus_y_base_time, twist_cols.has_read))
            .copied()
            .ok_or_else(|| PiCcsError::ProtocolError("CPU y_scalars missing Twist has_read opening".into()))?;
        let has_write_open = ccs_out0
            .y_scalars
            .get(cpu_bus.y_scalar_index(bus_y_base_time, twist_cols.has_write))
            .copied()
            .ok_or_else(|| PiCcsError::ProtocolError("CPU y_scalars missing Twist has_write opening".into()))?;
        let wv_open = ccs_out0
            .y_scalars
            .get(cpu_bus.y_scalar_index(bus_y_base_time, twist_cols.wv))
            .copied()
            .ok_or_else(|| PiCcsError::ProtocolError("CPU y_scalars missing Twist wv opening".into()))?;
        let rv_open = ccs_out0
            .y_scalars
            .get(cpu_bus.y_scalar_index(bus_y_base_time, twist_cols.rv))
            .copied()
            .ok_or_else(|| PiCcsError::ProtocolError("CPU y_scalars missing Twist rv opening".into()))?;
        let inc_write_open = ccs_out0
            .y_scalars
            .get(cpu_bus.y_scalar_index(bus_y_base_time, twist_cols.inc))
            .copied()
            .ok_or_else(|| PiCcsError::ProtocolError("CPU y_scalars missing Twist inc opening".into()))?;

        let pre = twist_pre
            .get(i_mem)
            .ok_or_else(|| PiCcsError::InvalidInput(format!("missing Twist pre-time data at index {}", i_mem)))?;
        let r_addr = &pre.r_addr;
        if r_addr.len() != ell_addr {
            return Err(PiCcsError::InvalidInput(format!(
                "Twist r_addr.len()={}, expected ell_addr={}",
                r_addr.len(),
                ell_addr
            )));
        }

        let twist_claims = claim_plan
            .twist
            .get(i_mem)
            .ok_or_else(|| PiCcsError::ProtocolError(format!("missing Twist claim schedule at index {}", i_mem)))?;

        // Route A Twist ordering in batched_time:
        // - read_check (time rounds only)
        // - write_check (time rounds only)
        // - bitness for ra_bits then wa_bits then has_read then has_write (time-only)
        let read_check_claim = batched_claimed_sums[twist_claims.read_check];
        let read_check_final = batched_final_values[twist_claims.read_check];
        let write_check_claim = batched_claimed_sums[twist_claims.write_check];
        let write_check_final = batched_final_values[twist_claims.write_check];

        if read_check_claim != pre.read_check_claim_sum {
            return Err(PiCcsError::ProtocolError(
                "twist read_check claimed sum != addr-pre final".into(),
            ));
        }
        if write_check_claim != pre.write_check_claim_sum {
            return Err(PiCcsError::ProtocolError(
                "twist write_check claimed sum != addr-pre final".into(),
            ));
        }

        // Bitness terminal checks (ra_bits then wa_bits then has_read then has_write).
        for (j, col_open) in ra_bits_open
            .iter()
            .chain(wa_bits_open.iter())
            .chain([has_read_open, has_write_open].iter())
            .enumerate()
        {
            let bitness_idx = if j < twist_claims.ell_addr {
                twist_claims
                    .bitness_ra_bits
                    .start
                    .checked_add(j)
                    .ok_or_else(|| PiCcsError::ProtocolError("bitness index overflow".into()))?
            } else if j < 2 * twist_claims.ell_addr {
                twist_claims
                    .bitness_wa_bits
                    .start
                    .checked_add(j - twist_claims.ell_addr)
                    .ok_or_else(|| PiCcsError::ProtocolError("bitness index overflow".into()))?
            } else if j == 2 * twist_claims.ell_addr {
                twist_claims.bitness_has_read
            } else {
                twist_claims.bitness_has_write
            };
            check_bitness_terminal(
                chi_cycle_at_r_time,
                *col_open,
                batched_final_values[bitness_idx],
                "twist bitness",
            )?;
        }

        let val_eval = twist_proof
            .val_eval
            .as_ref()
            .ok_or_else(|| PiCcsError::InvalidInput("Twist(Route A): missing val_eval proof".into()))?;

        let init_at_r_addr = eval_init_at_r_addr(&inst.init, inst.k, r_addr)?;
        let claimed_val = init_at_r_addr + val_eval.claimed_inc_sum_lt;

        let read_eq_addr = eq_bits_prod(&ra_bits_open, r_addr)?;
        let write_eq_addr = eq_bits_prod(&wa_bits_open, r_addr)?;

        // Terminal checks for read_check / write_check at (r_time, r_addr).
        let expected_read_check_final = chi_cycle_at_r_time * has_read_open * (claimed_val - rv_open) * read_eq_addr;
        if expected_read_check_final != read_check_final {
            return Err(PiCcsError::ProtocolError(
                "twist/read_check terminal value mismatch".into(),
            ));
        }

        let expected_write_check_final =
            chi_cycle_at_r_time * has_write_open * (wv_open - claimed_val - inc_write_open) * write_eq_addr;
        if expected_write_check_final != write_check_final {
            return Err(PiCcsError::ProtocolError(
                "twist/write_check terminal value mismatch".into(),
            ));
        }

        twist_time_openings.push(TwistTimeLaneOpenings {
            wa_bits: wa_bits_open,
            has_write: has_write_open,
            inc_at_write_addr: inc_write_open,
        });
    }

    // --------------------------------------------------------------------
    // Phase 2: Verify batched Twist val-eval sum-check, deriving shared r_val.
    // --------------------------------------------------------------------
    let mut r_val: Vec<K> = Vec::new();
    let mut val_eval_finals: Vec<K> = Vec::new();
    if !step.mem_insts.is_empty() {
        let plan = crate::memory_sidecar::claim_plan::TwistValEvalClaimPlan::build(step.mem_insts.iter(), has_prev);
        let claim_count = plan.claim_count;

        let mut per_claim_rounds: Vec<Vec<Vec<K>>> = Vec::with_capacity(claim_count);
        let mut per_claim_sums: Vec<K> = Vec::with_capacity(claim_count);
        let mut bind_claims: Vec<(u8, K)> = Vec::with_capacity(claim_count);
        let mut claim_idx = 0usize;

        for (i_mem, _inst) in step.mem_insts.iter().enumerate() {
            let twist_proof = match &proofs_mem[proof_mem_offset + i_mem] {
                MemOrLutProof::Twist(proof) => proof,
                _ => return Err(PiCcsError::InvalidInput("expected Twist proof".into())),
            };
            let val = twist_proof
                .val_eval
                .as_ref()
                .ok_or_else(|| PiCcsError::InvalidInput("Twist(Route A): missing val_eval proof".into()))?;

            per_claim_rounds.push(val.rounds_lt.clone());
            per_claim_sums.push(val.claimed_inc_sum_lt);
            bind_claims.push((plan.bind_tags[claim_idx], val.claimed_inc_sum_lt));
            claim_idx += 1;

            per_claim_rounds.push(val.rounds_total.clone());
            per_claim_sums.push(val.claimed_inc_sum_total);
            bind_claims.push((plan.bind_tags[claim_idx], val.claimed_inc_sum_total));
            claim_idx += 1;

            if has_prev {
                let prev_total = val.claimed_prev_inc_sum_total.ok_or_else(|| {
                    PiCcsError::InvalidInput("Twist(Route A): missing claimed_prev_inc_sum_total".into())
                })?;
                let prev_rounds = val
                    .rounds_prev_total
                    .clone()
                    .ok_or_else(|| PiCcsError::InvalidInput("Twist(Route A): missing rounds_prev_total".into()))?;
                per_claim_rounds.push(prev_rounds);
                per_claim_sums.push(prev_total);
                bind_claims.push((plan.bind_tags[claim_idx], prev_total));
                claim_idx += 1;
            } else if val.claimed_prev_inc_sum_total.is_some() || val.rounds_prev_total.is_some() {
                return Err(PiCcsError::InvalidInput(
                    "Twist(Route A): rollover fields present but prev_step is None".into(),
                ));
            }
        }

        tr.append_message(
            b"twist/val_eval/batch_start",
            &(step.mem_insts.len() as u64).to_le_bytes(),
        );
        tr.append_message(b"twist/val_eval/step_idx", &(step_idx as u64).to_le_bytes());
        bind_twist_val_eval_claim_sums(tr, &bind_claims);

        let (r_val_out, finals_out, ok) = verify_batched_sumcheck_rounds_ds(
            tr,
            b"twist/val_eval_batch",
            step_idx,
            &per_claim_rounds,
            &per_claim_sums,
            &plan.labels,
            &plan.degree_bounds,
        );
        if !ok {
            return Err(PiCcsError::SumcheckError(
                "twist val-eval batched sumcheck invalid".into(),
            ));
        }
        if r_val_out.len() != r_time.len() {
            return Err(PiCcsError::ProtocolError(format!(
                "twist val-eval r_val.len()={}, expected ell_n={}",
                r_val_out.len(),
                r_time.len()
            )));
        }
        if finals_out.len() != claim_count {
            return Err(PiCcsError::ProtocolError(format!(
                "twist val-eval finals.len()={}, expected {}",
                finals_out.len(),
                claim_count
            )));
        }
        r_val = r_val_out;
        val_eval_finals = finals_out;

        tr.append_message(b"twist/val_eval/batch_done", &[]);
    }

    // Verify val-eval terminal identity against CPU ME openings at r_val.
    let lt = if step.mem_insts.is_empty() {
        if !r_val.is_empty() {
            return Err(PiCcsError::ProtocolError(
                "twist val-eval produced r_val but no mem instances are present".into(),
            ));
        }
        K::ZERO
    } else {
        if r_val.len() != r_time.len() {
            return Err(PiCcsError::ProtocolError(format!(
                "twist val-eval r_val.len()={}, expected ell_n={}",
                r_val.len(),
                r_time.len()
            )));
        }
        lt_eval(&r_val, r_time)
    };

    let (cpu_me_val_cur, cpu_me_val_prev, bus_y_base_val) = if step.mem_insts.is_empty() {
        if !mem_proof.cpu_me_claims_val.is_empty() {
            return Err(PiCcsError::InvalidInput(
                "proof contains val-lane CPU ME claims with no Twist instances".into(),
            ));
        }
        (None, None, 0usize)
    } else {
        let expected = 1usize + usize::from(has_prev);
        if mem_proof.cpu_me_claims_val.len() != expected {
            return Err(PiCcsError::InvalidInput(format!(
                "shared bus expects {} CPU ME claim(s) at r_val, got {}",
                expected,
                mem_proof.cpu_me_claims_val.len()
            )));
        }

        let cpu_me_cur = mem_proof
            .cpu_me_claims_val
            .get(0)
            .ok_or_else(|| PiCcsError::ProtocolError("missing CPU ME claim at r_val".into()))?;
        if cpu_me_cur.r.as_slice() != r_val {
            return Err(PiCcsError::ProtocolError(
                "CPU ME(val) r mismatch (expected r_val)".into(),
            ));
        }
        if cpu_me_cur.c != step.mcs_inst.c {
            return Err(PiCcsError::ProtocolError(
                "CPU ME(val) commitment mismatch (current step)".into(),
            ));
        }
        let cpu_me_prev = if has_prev {
            let prev_inst =
                prev_step.ok_or_else(|| PiCcsError::ProtocolError("prev_step missing with has_prev=true".into()))?;
            let cpu_me_prev = mem_proof
                .cpu_me_claims_val
                .get(1)
                .ok_or_else(|| PiCcsError::ProtocolError("missing prev CPU ME claim at r_val".into()))?;
            if cpu_me_prev.r.as_slice() != r_val {
                return Err(PiCcsError::ProtocolError(
                    "CPU ME(val/prev) r mismatch (expected r_val)".into(),
                ));
            }
            if cpu_me_prev.c != prev_inst.mcs_inst.c {
                return Err(PiCcsError::ProtocolError("CPU ME(val/prev) commitment mismatch".into()));
            }
            Some(cpu_me_prev)
        } else {
            None
        };

        let bus_y_base_val = cpu_me_cur
            .y_scalars
            .len()
            .checked_sub(cpu_bus.bus_cols)
            .ok_or_else(|| PiCcsError::InvalidInput("CPU y_scalars too short for bus openings".into()))?;

        (Some(cpu_me_cur), cpu_me_prev, bus_y_base_val)
    };

    for (i_mem, inst) in step.mem_insts.iter().enumerate() {
        let twist_proof = match &proofs_mem[proof_mem_offset + i_mem] {
            MemOrLutProof::Twist(proof) => proof,
            _ => return Err(PiCcsError::InvalidInput("expected Twist proof".into())),
        };
        let val_eval = twist_proof
            .val_eval
            .as_ref()
            .ok_or_else(|| PiCcsError::InvalidInput("Twist(Route A): missing val_eval proof".into()))?;
        let layout = inst.twist_layout();
        let ell_addr = layout.ell_addr;

        let cpu_me_cur =
            cpu_me_val_cur.ok_or_else(|| PiCcsError::ProtocolError("missing CPU ME claim at r_val".into()))?;

        let twist_cols = cpu_bus
            .twist_cols
            .get(i_mem)
            .ok_or_else(|| PiCcsError::InvalidInput("shared_cpu_bus layout mismatch (twist)".into()))?;
        if twist_cols.wa_bits.end - twist_cols.wa_bits.start != ell_addr {
            return Err(PiCcsError::InvalidInput(format!(
                "shared_cpu_bus layout mismatch at mem_idx={i_mem}: expected ell_addr={ell_addr}"
            )));
        }

        let mut wa_bits_val_open = Vec::with_capacity(ell_addr);
        for col_id in twist_cols.wa_bits.clone() {
            wa_bits_val_open.push(
                cpu_me_cur
                    .y_scalars
                    .get(cpu_bus.y_scalar_index(bus_y_base_val, col_id))
                    .copied()
                    .ok_or_else(|| PiCcsError::ProtocolError("CPU y_scalars missing wa_bits(val) opening".into()))?,
            );
        }
        let has_write_val_open = cpu_me_cur
            .y_scalars
            .get(cpu_bus.y_scalar_index(bus_y_base_val, twist_cols.has_write))
            .copied()
            .ok_or_else(|| PiCcsError::ProtocolError("CPU y_scalars missing has_write(val) opening".into()))?;
        let inc_at_write_addr_val_open = cpu_me_cur
            .y_scalars
            .get(cpu_bus.y_scalar_index(bus_y_base_val, twist_cols.inc))
            .copied()
            .ok_or_else(|| PiCcsError::ProtocolError("CPU y_scalars missing inc(val) opening".into()))?;

        let r_addr = twist_pre
            .get(i_mem)
            .ok_or_else(|| PiCcsError::InvalidInput(format!("missing Twist pre-time data at index {}", i_mem)))?
            .r_addr
            .as_slice();
        let eq_wa_val = eq_bits_prod(&wa_bits_val_open, r_addr)?;
        let inc_at_r_addr_val = has_write_val_open * inc_at_write_addr_val_open * eq_wa_val;
        let expected_lt_final = inc_at_r_addr_val * lt;
        let claims_per_mem = if has_prev { 3 } else { 2 };
        let base = claims_per_mem * i_mem;
        if expected_lt_final != val_eval_finals[base] {
            return Err(PiCcsError::ProtocolError(
                "twist/val_eval_lt terminal value mismatch".into(),
            ));
        }
        let expected_total_final = inc_at_r_addr_val;
        if expected_total_final != val_eval_finals[base + 1] {
            return Err(PiCcsError::ProtocolError(
                "twist/val_eval_total terminal value mismatch".into(),
            ));
        }

        if has_prev {
            let prev =
                prev_step.ok_or_else(|| PiCcsError::ProtocolError("prev_step missing with has_prev=true".into()))?;
            let prev_inst = prev
                .mem_insts
                .get(i_mem)
                .ok_or_else(|| PiCcsError::ProtocolError("missing prev mem instance".into()))?;
            let cpu_me_prev = cpu_me_val_prev
                .ok_or_else(|| PiCcsError::ProtocolError("missing prev CPU ME claim at r_val".into()))?;

            // Terminal check for prev-total: uses previous-step openings at current r_val.
            let mut wa_bits_prev_open = Vec::with_capacity(ell_addr);
            for col_id in twist_cols.wa_bits.clone() {
                wa_bits_prev_open.push(
                    cpu_me_prev
                        .y_scalars
                        .get(cpu_bus.y_scalar_index(bus_y_base_val, col_id))
                        .copied()
                        .ok_or_else(|| {
                            PiCcsError::ProtocolError("CPU y_scalars missing wa_bits(prev) opening".into())
                        })?,
                );
            }
            let has_write_prev_open = cpu_me_prev
                .y_scalars
                .get(cpu_bus.y_scalar_index(bus_y_base_val, twist_cols.has_write))
                .copied()
                .ok_or_else(|| PiCcsError::ProtocolError("CPU y_scalars missing has_write(prev) opening".into()))?;
            let inc_prev_open = cpu_me_prev
                .y_scalars
                .get(cpu_bus.y_scalar_index(bus_y_base_val, twist_cols.inc))
                .copied()
                .ok_or_else(|| PiCcsError::ProtocolError("CPU y_scalars missing inc(prev) opening".into()))?;

            let eq_wa_prev = eq_bits_prod(&wa_bits_prev_open, r_addr)?;
            let inc_at_r_addr_prev = has_write_prev_open * inc_prev_open * eq_wa_prev;
            if inc_at_r_addr_prev != val_eval_finals[base + 2] {
                return Err(PiCcsError::ProtocolError(
                    "twist/rollover_prev_total terminal value mismatch".into(),
                ));
            }

            // Enforce rollover equation: Init_i(r_addr) == Init_{i-1}(r_addr) + PrevTotal(i).
            let claimed_prev_total = val_eval
                .claimed_prev_inc_sum_total
                .ok_or_else(|| PiCcsError::ProtocolError("twist rollover missing claimed_prev_inc_sum_total".into()))?;
            let init_prev_at_r_addr = eval_init_at_r_addr(&prev_inst.init, prev_inst.k, r_addr)?;
            let init_cur_at_r_addr = eval_init_at_r_addr(&inst.init, inst.k, r_addr)?;
            if init_cur_at_r_addr != init_prev_at_r_addr + claimed_prev_total {
                return Err(PiCcsError::ProtocolError("twist rollover init check failed".into()));
            }
        }
    }

    Ok(RouteAMemoryVerifyOutput {
        claim_idx_end: claim_plan.claim_idx_end,
        twist_time_openings,
    })
}

use crate::memory_sidecar::claim_plan::RouteATimeClaimPlan;
use crate::memory_sidecar::helpers::{
    check_bitness_terminal, emit_twist_val_lane_openings, me_identity_open, me_identity_opens,
};
use crate::memory_sidecar::sumcheck_ds::{run_batched_sumcheck_prover_ds, verify_batched_sumcheck_rounds_ds};
use crate::memory_sidecar::transcript::{bind_batched_claim_sums, bind_twist_val_eval_claim_sums, digest_fields};
use crate::memory_sidecar::utils::RoundOraclePrefix;
use crate::shard_proof_types::{MemOrLutProof, MemSidecarProof, ShoutProofK, TwistProofK};
use crate::PiCcsError;
use neo_ajtai::Commitment as Cmt;
use neo_ccs::{CcsStructure, Mat, MeInstance};
use neo_math::{F, K};
use neo_memory::bit_ops::eq_bits_prod;
use neo_memory::mle::{eq_points, lt_eval};
use neo_memory::ts_common as ts;
use neo_memory::twist_oracle::{
    table_mle_eval, LazyBitnessOracle, TwistReadCheckAddrOracle, TwistTotalIncOracleSparse, TwistValEvalOracleSparse,
    TwistWriteCheckAddrOracle,
};
use neo_memory::witness::{LutInstance, MemInstance, StepInstanceBundle, StepWitnessBundle};
use neo_memory::{eval_init_at_r_addr, shout, twist, BatchedAddrProof, MemInit};
use neo_params::NeoParams;
use neo_reductions::sumcheck::{BatchedClaim, RoundOracle};
use neo_transcript::{Poseidon2Transcript, Transcript};
use p3_field::PrimeCharacteristicRing;

// ============================================================================
// Transcript binding
// ============================================================================

fn absorb_step_memory_commitments_impl<'a, LI, MI>(tr: &mut Poseidon2Transcript, mut lut_insts: LI, mut mem_insts: MI)
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
        shout::absorb_commitments(tr, inst);
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
        twist::absorb_commitments(tr, inst);
    }
    tr.append_message(b"step/absorb_memory_done", &[]);
}

pub fn absorb_step_memory_commitments(tr: &mut Poseidon2Transcript, step: &StepInstanceBundle<Cmt, F, K>) {
    absorb_step_memory_commitments_impl(tr, step.lut_insts.iter(), step.mem_insts.iter());
}

pub(crate) fn absorb_step_memory_commitments_witness(
    tr: &mut Poseidon2Transcript,
    step: &StepWitnessBundle<Cmt, F, K>,
) {
    absorb_step_memory_commitments_impl(
        tr,
        step.lut_instances.iter().map(|(inst, _)| inst),
        step.mem_instances.iter().map(|(inst, _)| inst),
    );
}

// ============================================================================
// Prover helpers
// ============================================================================

pub struct RouteAMemoryOracles {
    pub shout: Vec<shout::RouteAShoutOracles>,
    pub twist: Vec<twist::RouteATwistOracles>,
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

pub struct ShoutAddrPreProverData {
    pub addr_pre: BatchedAddrProof<K>,
    pub decoded: shout::ShoutDecodedCols,
    pub table_k: Vec<K>,
}

pub struct ShoutAddrPreVerifyData {
    pub addr_claim_sum: K,
    pub addr_final: K,
    pub r_addr: Vec<K>,
    pub table_k: Vec<K>,
}

pub struct TwistAddrPreProverData {
    pub addr_pre: BatchedAddrProof<K>,
    pub decoded: twist::TwistDecodedCols,
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

pub fn prove_twist_addr_pre_time(
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    step: &StepWitnessBundle<Cmt, F, K>,
    ell_n: usize,
    r_cycle: &[K],
) -> Result<Vec<TwistAddrPreProverData>, PiCcsError> {
    let mut out = Vec::with_capacity(step.mem_instances.len());

    for (idx, (mem_inst, mem_wit)) in step.mem_instances.iter().enumerate() {
        let decoded = twist::decode_twist_cols(params, mem_inst, mem_wit, ell_n)?;

        let init_sparse: Vec<(usize, K)> = match &mem_inst.init {
            MemInit::Zero => Vec::new(),
            MemInit::Sparse(pairs) => pairs
                .iter()
                .map(|(addr, val)| {
                    let addr_usize = usize::try_from(*addr).map_err(|_| {
                        PiCcsError::InvalidInput(format!(
                            "Twist: init address doesn't fit usize: addr={addr}"
                        ))
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

        let mut read_addr_oracle = TwistReadCheckAddrOracle::new(
            init_sparse.clone(),
            r_cycle,
            decoded.has_read.clone(),
            decoded.rv.clone(),
            &decoded.ra_bits,
            decoded.has_write.clone(),
            &decoded.wa_bits,
            decoded.inc_at_write_addr.clone(),
        );
        let mut write_addr_oracle = TwistWriteCheckAddrOracle::new(
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

pub fn prove_shout_addr_pre_time(
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    step: &StepWitnessBundle<Cmt, F, K>,
    ell_n: usize,
    r_cycle: &[K],
) -> Result<Vec<ShoutAddrPreProverData>, PiCcsError> {
    let mut out = Vec::with_capacity(step.lut_instances.len());

    for (idx, (lut_inst, lut_wit)) in step.lut_instances.iter().enumerate() {
        let decoded = shout::decode_shout_cols(params, lut_inst, lut_wit, ell_n)?;
        let table_k: Vec<K> = lut_inst.table.iter().map(|&v| v.into()).collect();
        let (mut addr_oracle, addr_claim_sum) = shout::build_shout_addr_oracle(lut_inst, &decoded, r_cycle, &table_k)?;

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
            table_k,
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

pub fn build_route_a_memory_oracles(
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
        shout_oracles.push(shout::build_route_a_shout_oracles(
            lut_inst,
            &pre.decoded,
            r_cycle,
            &pre.addr_pre.r_addr,
        )?);
    }

    let mut twist_oracles = Vec::with_capacity(step.mem_instances.len());
    for ((mem_inst, _mem_wit), pre) in step.mem_instances.iter().zip(twist_pre.iter()) {
        let init_at_r_addr = eval_init_at_r_addr(&mem_inst.init, mem_inst.k, &pre.addr_pre.r_addr)?;
        twist_oracles.push(twist::build_route_a_twist_oracles(
            mem_inst,
            &pre.decoded,
            r_cycle,
            &pre.addr_pre.r_addr,
            init_at_r_addr,
        )?);
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
    pub bitness: Vec<Vec<LazyBitnessOracle>>,
}

pub fn build_route_a_shout_time_claims_guard<'a>(
    shout_oracles: &'a mut [shout::RouteAShoutOracles],
    ell_n: usize,
) -> RouteAShoutTimeClaimsGuard<'a> {
    let mut value_prefixes: Vec<RoundOraclePrefix<'a>> = Vec::with_capacity(shout_oracles.len());
    let mut adapter_prefixes: Vec<RoundOraclePrefix<'a>> = Vec::with_capacity(shout_oracles.len());
    let mut value_claims: Vec<K> = Vec::with_capacity(shout_oracles.len());
    let mut adapter_claims: Vec<K> = Vec::with_capacity(shout_oracles.len());
    let mut bitness: Vec<Vec<LazyBitnessOracle>> = Vec::with_capacity(shout_oracles.len());

    for o in shout_oracles.iter_mut() {
        bitness.push(core::mem::take(&mut o.bitness));
        value_claims.push(o.value_claim);
        adapter_claims.push(o.adapter_claim);
        value_prefixes.push(RoundOraclePrefix::new(&mut o.value, ell_n));
        adapter_prefixes.push(RoundOraclePrefix::new(&mut o.adapter, ell_n));
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
    pub fn new(shout_oracles: &'a mut [shout::RouteAShoutOracles], ell_n: usize) -> Self {
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
            debug_assert_eq!(
                bit_oracle.compute_claim(),
                K::ZERO,
                "lazy shout bitness claim should be 0"
            );
            claimed_sums.push(K::ZERO);
            degree_bounds.push(bit_oracle.degree_bound());
            labels.push(b"shout/bitness");
            claim_is_dynamic.push(false);
            claims.push(BatchedClaim {
                oracle: bit_oracle,
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
    pub bitness: Vec<Vec<LazyBitnessOracle>>,
}

pub fn build_route_a_twist_time_claims_guard<'a>(
    twist_oracles: &'a mut [twist::RouteATwistOracles],
    ell_n: usize,
    read_check_claims: Vec<K>,
    write_check_claims: Vec<K>,
) -> RouteATwistTimeClaimsGuard<'a> {
    let mut read_check_prefixes: Vec<RoundOraclePrefix<'a>> = Vec::with_capacity(twist_oracles.len());
    let mut write_check_prefixes: Vec<RoundOraclePrefix<'a>> = Vec::with_capacity(twist_oracles.len());
    let mut bitness: Vec<Vec<LazyBitnessOracle>> = Vec::with_capacity(twist_oracles.len());

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
        read_check_prefixes.push(RoundOraclePrefix::new(&mut o.read_check, ell_n));
        write_check_prefixes.push(RoundOraclePrefix::new(&mut o.write_check, ell_n));
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
            debug_assert_eq!(
                bit_oracle.compute_claim(),
                K::ZERO,
                "lazy twist bitness claim should be 0"
            );
            claimed_sums.push(K::ZERO);
            degree_bounds.push(bit_oracle.degree_bound());
            labels.push(b"twist/bitness");
            claim_is_dynamic.push(false);
            claims.push(BatchedClaim {
                oracle: bit_oracle,
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
        twist_oracles: &'a mut [twist::RouteATwistOracles],
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

pub struct RouteAMemoryProverOutput {
    pub mem: MemSidecarProof<Cmt, F, K>,
    pub me_wits_time: Vec<Mat<F>>,
    pub me_wits_val: Vec<Mat<F>>,
}

pub fn finalize_route_a_memory_prover(
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s: &CcsStructure<F>,
    step: &StepWitnessBundle<Cmt, F, K>,
    prev_step: Option<&StepWitnessBundle<Cmt, F, K>>,
    prev_twist_decoded: Option<&[twist::TwistDecodedCols]>,
    oracles: &mut RouteAMemoryOracles,
    shout_pre: &[ShoutAddrPreProverData],
    twist_pre: &[TwistAddrPreProverData],
    r_time: &[K],
    m_in: usize,
    step_idx: usize,
) -> Result<RouteAMemoryProverOutput, PiCcsError> {
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
        if lut_inst.comms.len() != lut_wit.mats.len() {
            return Err(PiCcsError::InvalidInput(format!(
                "Shout: comms.len()={} != mats.len()={} (lut_idx={})",
                lut_inst.comms.len(),
                lut_wit.mats.len(),
                idx
            )));
        }
    }
    for (idx, (mem_inst, mem_wit)) in step.mem_instances.iter().enumerate() {
        if mem_inst.comms.len() != mem_wit.mats.len() {
            return Err(PiCcsError::InvalidInput(format!(
                "Twist: comms.len()={} != mats.len()={} (mem_idx={})",
                mem_inst.comms.len(),
                mem_wit.mats.len(),
                idx
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
            if mem_inst.comms.len() != mem_wit.mats.len() {
                return Err(PiCcsError::InvalidInput(format!(
                    "Twist(prev): comms.len()={} != mats.len()={} (mem_idx={})",
                    mem_inst.comms.len(),
                    mem_wit.mats.len(),
                    idx
                )));
            }
        }
    }

    let mut me_claims_time: Vec<MeInstance<Cmt, F, K>> = Vec::new();
    let mut me_wits_time: Vec<Mat<F>> = Vec::new();

    let mut me_claims_val: Vec<MeInstance<Cmt, F, K>> = Vec::new();
    let mut me_wits_val: Vec<Mat<F>> = Vec::new();
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
            let (oracle_lt, claimed_inc_sum_lt) = TwistValEvalOracleSparse::new(
                &decoded.wa_bits,
                decoded.has_write.clone(),
                decoded.inc_at_write_addr.clone(),
                r_addr,
                r_time,
            );
            let (oracle_total, claimed_inc_sum_total) = TwistTotalIncOracleSparse::new(
                &decoded.wa_bits,
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
                let (oracle_prev_total, claimed_prev_total) = TwistTotalIncOracleSparse::new(
                    &prev_decoded.wa_bits,
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

    for ((lut_inst, lut_wit), pre) in step.lut_instances.iter().zip(shout_pre.iter()) {
        let mut proof = ShoutProofK::default();
        proof.addr_pre = pre.addr_pre.clone();

        me_claims_time.extend(ts::emit_me_claims_for_mats(
            tr,
            b"shout/me_digest",
            params,
            s,
            &lut_inst.comms,
            &lut_wit.mats,
            r_time,
            m_in,
        )?);
        me_wits_time.extend(lut_wit.mats.iter().cloned());
        proofs.push(MemOrLutProof::Shout(proof));
    }

    for (idx, (mem_inst, mem_wit)) in step.mem_instances.iter().enumerate() {
        let mut proof = TwistProofK::default();
        proof.addr_pre = twist_pre
            .get(idx)
            .ok_or_else(|| PiCcsError::ProtocolError("missing Twist addr_pre".into()))?
            .addr_pre
            .clone();
        proof.val_eval = twist_val_eval_proofs.get(idx).cloned();

        me_claims_time.extend(ts::emit_me_claims_for_mats(
            tr,
            b"twist/me_digest",
            params,
            s,
            &mem_inst.comms,
            &mem_wit.mats,
            r_time,
            m_in,
        )?);
        me_wits_time.extend(mem_wit.mats.iter().cloned());

        // Emit only the columns needed to verify Twist's val-eval terminal check at r_val:
        // wa_bits, has_write, inc_at_write_addr.
        if r_val.len() != r_time.len() {
            return Err(PiCcsError::ProtocolError(format!(
                "twist val-eval r_val.len()={}, expected ell_n={}",
                r_val.len(),
                r_time.len()
            )));
        }
        emit_twist_val_lane_openings(
            tr,
            params,
            s,
            mem_inst,
            mem_wit,
            &r_val,
            m_in,
            &mut me_claims_val,
            &mut me_wits_val,
        )?;
        if let Some(prev) = prev_step {
            let (prev_inst, prev_wit) = prev
                .mem_instances
                .get(idx)
                .ok_or_else(|| PiCcsError::ProtocolError("missing prev mem instance".into()))?;
            emit_twist_val_lane_openings(
                tr,
                params,
                s,
                prev_inst,
                prev_wit,
                &r_val,
                m_in,
                &mut me_claims_val,
                &mut me_wits_val,
            )?;
        }

        proofs.push(MemOrLutProof::Twist(proof));
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
        if !me_claims_val.is_empty() || !me_wits_val.is_empty() {
            return Err(PiCcsError::ProtocolError(
                "twist val-lane ME claims must be empty when no mem instances are present".into(),
            ));
        }
    } else if me_claims_val.is_empty() || me_wits_val.is_empty() {
        return Err(PiCcsError::ProtocolError(
            "twist val-eval requires non-empty val-lane ME claims".into(),
        ));
    }

    Ok(RouteAMemoryProverOutput {
        mem: MemSidecarProof {
            me_claims_time,
            me_claims_val,
            proofs,
        },
        me_wits_time,
        me_wits_val,
    })
}

// ============================================================================
// Verifier helpers
// ============================================================================

pub struct RouteAMemoryVerifyOutput<C> {
    pub collected_me_time: Vec<MeInstance<C, F, K>>,
    pub collected_me_val: Vec<MeInstance<C, F, K>>,
    pub claim_idx_end: usize,
    pub twist_total_inc_sums: Vec<K>,
}

pub fn verify_route_a_memory_step(
    tr: &mut Poseidon2Transcript,
    _params: &NeoParams,
    step: &StepInstanceBundle<Cmt, F, K>,
    prev_step: Option<&StepInstanceBundle<Cmt, F, K>>,
    r_time: &[K],
    r_cycle: &[K],
    batched_final_values: &[K],
    batched_claimed_sums: &[K],
    claim_idx_start: usize,
    mem_proof: &MemSidecarProof<Cmt, F, K>,
    shout_pre: &[ShoutAddrPreVerifyData],
    twist_pre: &[TwistAddrPreVerifyData],
    step_idx: usize,
) -> Result<RouteAMemoryVerifyOutput<Cmt>, PiCcsError> {
    let chi_cycle_at_r_time = eq_points(r_time, r_cycle);
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

    let proofs_mem = &mem_proof.proofs;
    let mut me_time_offset = 0usize;
    let mut me_val_offset = 0usize;
    let mut collected_me_time = Vec::new();
    let mut collected_me_val = Vec::new();
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

    // Shout instances first.
    for (proof_idx, inst) in step.lut_insts.iter().enumerate() {
        let _shout_proof = match &proofs_mem[proof_idx] {
            MemOrLutProof::Shout(proof) => proof,
            _ => return Err(PiCcsError::InvalidInput("expected Shout proof".into())),
        };

        let shout_me_count = inst.comms.len();
        let me_slice = mem_proof
            .me_claims_time
            .get(me_time_offset..me_time_offset + shout_me_count)
            .ok_or_else(|| {
                PiCcsError::InvalidInput(format!(
                    "Not enough ME claims for Shout: need {} at offset {}, have {}",
                    shout_me_count,
                    me_time_offset,
                    mem_proof.me_claims_time.len()
                ))
            })?;
        me_time_offset += shout_me_count;

        if me_slice.len() != inst.comms.len() {
            return Err(PiCcsError::InvalidInput(format!(
                "Shout: ME claim count {} != comms.len() {}",
                me_slice.len(),
                inst.comms.len()
            )));
        }

        let layout = inst.shout_layout();
        let ell_addr = layout.ell_addr;
        if shout_me_count != ell_addr + 2 {
            return Err(PiCcsError::InvalidInput(format!(
                "Shout comms.len()={}, expected {} (= d*ell + 2)",
                shout_me_count,
                ell_addr + 2
            )));
        }

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

        let addr_bits_open = me_identity_opens(me_slice, ell_addr)?;
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
        let has_lookup_open = me_identity_open(
            me_slice
                .get(layout.has_lookup)
                .ok_or_else(|| PiCcsError::InvalidInput("Shout: missing has_lookup ME claim".into()))?,
        )?;
        check_bitness_terminal(
            chi_cycle_at_r_time,
            has_lookup_open,
            batched_final_values[shout_claims.bitness_has_lookup],
            "shout bitness has_lookup",
        )?;

        let val_open = me_identity_open(
            me_slice
                .get(layout.val)
                .ok_or_else(|| PiCcsError::InvalidInput("Shout: missing val ME claim".into()))?,
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

        for (j, me) in me_slice.iter().enumerate() {
            if me.r != *r_time {
                return Err(PiCcsError::ProtocolError(format!(
                    "Shout ME[{}] r != r_time (Route A requires shared r)",
                    j
                )));
            }
            if me.c != inst.comms[j] {
                return Err(PiCcsError::ProtocolError(format!(
                    "Shout ME[{}] commitment mismatch",
                    j
                )));
            }
        }

        collected_me_time.extend_from_slice(me_slice);
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

        let twist_me_count = inst.comms.len();
        let me_slice = mem_proof
            .me_claims_time
            .get(me_time_offset..me_time_offset + twist_me_count)
            .ok_or_else(|| {
                PiCcsError::InvalidInput(format!(
                    "Not enough ME claims for Twist: need {} at offset {}, have {}",
                    twist_me_count,
                    me_time_offset,
                    mem_proof.me_claims_time.len()
                ))
            })?;
        me_time_offset += twist_me_count;

        if me_slice.len() != inst.comms.len() {
            return Err(PiCcsError::InvalidInput(format!(
                "Twist: ME claim count {} != comms.len() {}",
                me_slice.len(),
                inst.comms.len()
            )));
        }

        let layout = inst.twist_layout();
        let ell_addr = layout.ell_addr;
        if twist_me_count != 2 * ell_addr + 5 {
            return Err(PiCcsError::InvalidInput(format!(
                "Twist comms.len()={}, expected {} (= 2*d*ell + 5)",
                twist_me_count,
                2 * ell_addr + 5
            )));
        }

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

        let ra_bits_open = me_identity_opens(me_slice, ell_addr)?;
        let wa_bits_open = me_identity_opens(
            me_slice
                .get(layout.wa_bits.clone())
                .ok_or_else(|| PiCcsError::InvalidInput("Twist: missing wa_bits ME claims".into()))?,
            ell_addr,
        )?;

        let has_read_open = me_identity_open(
            me_slice
                .get(layout.has_read)
                .ok_or_else(|| PiCcsError::InvalidInput("Twist: missing has_read ME claim".into()))?,
        )?;
        let has_write_open = me_identity_open(
            me_slice
                .get(layout.has_write)
                .ok_or_else(|| PiCcsError::InvalidInput("Twist: missing has_write ME claim".into()))?,
        )?;
        let wv_open = me_identity_open(
            me_slice
                .get(layout.wv)
                .ok_or_else(|| PiCcsError::InvalidInput("Twist: missing wv ME claim".into()))?,
        )?;
        let rv_open = me_identity_open(
            me_slice
                .get(layout.rv)
                .ok_or_else(|| PiCcsError::InvalidInput("Twist: missing rv ME claim".into()))?,
        )?;
        let inc_write_open = me_identity_open(
            me_slice
                .get(layout.inc_at_write_addr)
                .ok_or_else(|| PiCcsError::InvalidInput("Twist: missing inc_at_write_addr ME claim".into()))?,
        )?;

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

        // Enforce r/commitment alignment for the r_time ME claims.
        for (j, me) in me_slice.iter().enumerate() {
            if me.r != *r_time {
                return Err(PiCcsError::ProtocolError(format!(
                    "Twist ME[{}] r != r_time (Route A requires shared r)",
                    j
                )));
            }
            if me.c != inst.comms[j] {
                return Err(PiCcsError::ProtocolError(format!(
                    "Twist ME[{}] commitment mismatch",
                    j
                )));
            }
        }

        collected_me_time.extend_from_slice(me_slice);
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

    // Verify val-eval terminal identity against ME openings at r_val.
    let mut twist_total_inc_sums: Vec<K> = Vec::with_capacity(step.mem_insts.len());
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
        let val_me_count = layout.val_lane_len();
        let me_cur_slice = mem_proof
            .me_claims_val
            .get(me_val_offset..me_val_offset + val_me_count)
            .ok_or_else(|| {
                PiCcsError::InvalidInput(format!(
                    "Not enough ME claims for Twist val lane (current): need {} at offset {}, have {}",
                    val_me_count,
                    me_val_offset,
                    mem_proof.me_claims_val.len()
                ))
            })?;
        me_val_offset += val_me_count;
        let me_prev_slice = if has_prev {
            let slice = mem_proof
                .me_claims_val
                .get(me_val_offset..me_val_offset + val_me_count)
                .ok_or_else(|| {
                    PiCcsError::InvalidInput(format!(
                        "Not enough ME claims for Twist val lane (prev): need {} at offset {}, have {}",
                        val_me_count,
                        me_val_offset,
                        mem_proof.me_claims_val.len()
                    ))
                })?;
            me_val_offset += val_me_count;
            Some(slice)
        } else {
            None
        };

        // Layout: wa_bits (ell_addr), has_write, inc_at_write_addr.
        let wa_bits_val_open = me_identity_opens(me_cur_slice, ell_addr)?;
        let has_write_val_open = me_identity_open(
            me_cur_slice
                .get(ell_addr)
                .ok_or_else(|| PiCcsError::InvalidInput("Twist(val): missing has_write ME claim".into()))?,
        )?;
        let inc_at_write_addr_val_open = me_identity_open(
            me_cur_slice
                .get(ell_addr + 1)
                .ok_or_else(|| PiCcsError::InvalidInput("Twist(val): missing inc_at_write_addr ME claim".into()))?,
        )?;

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

        twist_total_inc_sums.push(val_eval.claimed_inc_sum_total);

        // Enforce r/commitment alignment for the r_val ME claims.
        for (j, me) in me_cur_slice.iter().enumerate() {
            if me.r != r_val {
                return Err(PiCcsError::ProtocolError(format!("Twist(val) ME[{}] r != r_val", j)));
            }
            let want_comm = if j < ell_addr {
                inst.comms
                    .get(layout.wa_bits.start + j)
                    .ok_or_else(|| PiCcsError::InvalidInput("Twist(val): comms too short".into()))?
            } else if j == ell_addr {
                inst.comms
                    .get(layout.has_write)
                    .ok_or_else(|| PiCcsError::InvalidInput("Twist(val): comms too short".into()))?
            } else {
                inst.comms
                    .get(layout.inc_at_write_addr)
                    .ok_or_else(|| PiCcsError::InvalidInput("Twist(val): comms too short".into()))?
            };
            if me.c != *want_comm {
                return Err(PiCcsError::ProtocolError(format!(
                    "Twist(val) ME[{}] commitment mismatch",
                    j
                )));
            }
        }

        collected_me_val.extend_from_slice(me_cur_slice);

        if let Some(prev_slice) = me_prev_slice {
            let prev =
                prev_step.ok_or_else(|| PiCcsError::ProtocolError("prev_step missing with has_prev=true".into()))?;
            let prev_inst = prev
                .mem_insts
                .get(i_mem)
                .ok_or_else(|| PiCcsError::ProtocolError("missing prev mem instance".into()))?;

            // Terminal check for prev-total: uses previous-step openings at current r_val.
            let wa_bits_prev_open = me_identity_opens(prev_slice, ell_addr)?;
            let has_write_prev_open = me_identity_open(
                prev_slice
                    .get(ell_addr)
                    .ok_or_else(|| PiCcsError::InvalidInput("Twist(val/prev): missing has_write ME claim".into()))?,
            )?;
            let inc_prev_open = me_identity_open(prev_slice.get(ell_addr + 1).ok_or_else(|| {
                PiCcsError::InvalidInput("Twist(val/prev): missing inc_at_write_addr ME claim".into())
            })?)?;
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

            // Enforce r/commitment alignment for the previous-step r_val ME claims.
            let layout = prev_inst.twist_layout();
            for (j, me) in prev_slice.iter().enumerate() {
                if me.r != r_val {
                    return Err(PiCcsError::ProtocolError(format!(
                        "Twist(val/prev) ME[{}] r != r_val",
                        j
                    )));
                }
                let want_comm = if j < ell_addr {
                    prev_inst
                        .comms
                        .get(layout.wa_bits.start + j)
                        .ok_or_else(|| PiCcsError::InvalidInput("Twist(val/prev): comms too short".into()))?
                } else if j == ell_addr {
                    prev_inst
                        .comms
                        .get(layout.has_write)
                        .ok_or_else(|| PiCcsError::InvalidInput("Twist(val/prev): comms too short".into()))?
                } else {
                    prev_inst
                        .comms
                        .get(layout.inc_at_write_addr)
                        .ok_or_else(|| PiCcsError::InvalidInput("Twist(val/prev): comms too short".into()))?
                };
                if me.c != *want_comm {
                    return Err(PiCcsError::ProtocolError(format!(
                        "Twist(val/prev) ME[{}] commitment mismatch",
                        j
                    )));
                }
            }

            collected_me_val.extend_from_slice(prev_slice);
        }
    }

    if me_time_offset != mem_proof.me_claims_time.len() {
        return Err(PiCcsError::InvalidInput(format!(
            "unused ME claims: consumed {}, proof has {}",
            me_time_offset,
            mem_proof.me_claims_time.len()
        )));
    }
    if me_val_offset != mem_proof.me_claims_val.len() {
        return Err(PiCcsError::InvalidInput(format!(
            "unused ME claims (val lane): consumed {}, proof has {}",
            me_val_offset,
            mem_proof.me_claims_val.len()
        )));
    }

    Ok(RouteAMemoryVerifyOutput {
        collected_me_time,
        collected_me_val,
        claim_idx_end: claim_plan.claim_idx_end,
        twist_total_inc_sums,
    })
}

use crate::memory_sidecar::utils::RoundOraclePrefix;
use crate::shard_proof_types::{MemOrLutProof, MemSidecarProof, ShoutProofK, TwistProofK};
use crate::PiCcsError;
use neo_ajtai::Commitment as Cmt;
use neo_ccs::{CcsStructure, Mat, MeInstance};
use neo_math::{KExtensions, F, K};
use neo_memory::bit_ops::eq_bits_prod;
use neo_memory::mle::{eq_points, lt_eval};
use neo_memory::ts_common as ts;
use neo_memory::twist_oracle::{table_mle_eval, LazyBitnessOracle, TwistTotalIncOracleSparse, TwistValEvalOracleSparse};
use neo_memory::witness::StepWitnessBundle;
use neo_memory::{shout, twist};
use neo_params::NeoParams;
use neo_reductions::sumcheck::{
    run_batched_sumcheck_prover, run_sumcheck_prover, verify_batched_sumcheck_rounds, verify_sumcheck_rounds,
    BatchedClaim, RoundOracle,
};
use neo_transcript::{Poseidon2Transcript, Transcript};
use p3_field::PrimeCharacteristicRing;

// ============================================================================
// Transcript binding
// ============================================================================

pub fn absorb_step_memory_commitments(tr: &mut Poseidon2Transcript, step: &StepWitnessBundle<Cmt, F, K>) {
    fn digest_fields(label: &'static [u8], fs: &[F]) -> [u8; 32] {
        let mut h = Poseidon2Transcript::new(b"memory/public_digest");
        h.append_message(b"digest/label", label);
        h.append_message(b"digest/len", &(fs.len() as u64).to_le_bytes());
        h.append_fields(b"digest/fields", fs);
        h.digest32()
    }

    tr.append_message(b"step/absorb_memory_start", &[]);
    tr.append_message(b"step/lut_count", &(step.lut_instances.len() as u64).to_le_bytes());
    for (i, (inst, _)) in step.lut_instances.iter().enumerate() {
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
    tr.append_message(b"step/mem_count", &(step.mem_instances.len() as u64).to_le_bytes());
    for (i, (inst, _)) in step.mem_instances.iter().enumerate() {
        // Bind public memory parameters before any challenges.
        tr.append_message(b"step/mem_idx", &(i as u64).to_le_bytes());
        tr.append_message(b"twist/k", &(inst.k as u64).to_le_bytes());
        tr.append_message(b"twist/d", &(inst.d as u64).to_le_bytes());
        tr.append_message(b"twist/n_side", &(inst.n_side as u64).to_le_bytes());
        tr.append_message(b"twist/steps", &(inst.steps as u64).to_le_bytes());
        tr.append_message(b"twist/ell", &(inst.ell as u64).to_le_bytes());
        let init_vals_digest = digest_fields(b"twist/init_vals", &inst.init_vals);
        tr.append_message(b"twist/init_vals_digest", &init_vals_digest);
        twist::absorb_commitments(tr, inst);
    }
    tr.append_message(b"step/absorb_memory_done", &[]);
}

fn bind_batched_claim_sums(
    tr: &mut Poseidon2Transcript,
    prefix: &'static [u8],
    claimed_sums: &[K],
    labels: &[&'static [u8]],
) {
    debug_assert_eq!(claimed_sums.len(), labels.len());
    tr.append_message(prefix, &(claimed_sums.len() as u64).to_le_bytes());
    for (i, (sum, label)) in claimed_sums.iter().zip(labels.iter()).enumerate() {
        tr.append_message(b"addr_batch/label", label);
        tr.append_message(b"addr_batch/idx", &(i as u64).to_le_bytes());
        tr.append_fields(b"addr_batch/claimed_sum", &sum.as_coeffs());
    }
}

fn bind_twist_val_eval_claim_sums(tr: &mut Poseidon2Transcript, claims: &[(u8, K)]) {
    tr.append_message(b"twist/val_eval/claimed_sums_len", &(claims.len() as u64).to_le_bytes());
    for (i, (kind, sum)) in claims.iter().enumerate() {
        tr.append_message(b"twist/val_eval/claim_idx", &(i as u64).to_le_bytes());
        tr.append_message(b"twist/val_eval/claim_kind", &[*kind]);
        tr.append_fields(b"twist/val_eval/claimed_sum", &sum.as_coeffs());
    }
}

// ============================================================================
// Prover helpers
// ============================================================================

pub struct RouteAMemoryOracles {
    pub shout: Vec<shout::RouteAShoutOracles>,
    pub twist: Vec<twist::RouteATwistOraclesV2>,
}

pub struct ShoutAddrPreProverData {
    pub addr_claim_sum: K,
    pub addr_rounds: Vec<Vec<K>>,
    pub r_addr: Vec<K>,
    pub decoded: shout::ShoutDecodedCols,
}

pub struct ShoutAddrPreVerifyData {
    pub addr_claim_sum: K,
    pub addr_final: K,
    pub r_addr: Vec<K>,
}

pub fn decode_twist_pre_time(
    params: &NeoParams,
    step: &StepWitnessBundle<Cmt, F, K>,
    ell_n: usize,
) -> Result<Vec<twist::TwistDecodedCols>, PiCcsError> {
    let mut out = Vec::with_capacity(step.mem_instances.len());
    for (mem_inst, mem_wit) in step.mem_instances.iter() {
        out.push(twist::decode_twist_cols(params, mem_inst, mem_wit, ell_n)?);
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
        let (mut addr_oracle, addr_claim_sum) = shout::build_shout_addr_oracle(lut_inst, &decoded, r_cycle)?;

        // Bind claimed sum before deriving address challenges.
        tr.append_message(b"shout/addr_claim_idx", &(idx as u64).to_le_bytes());
        tr.append_fields(b"shout/addr_claim_sum", &addr_claim_sum.as_coeffs());

        let (addr_rounds, r_addr) = run_sumcheck_prover(tr, &mut addr_oracle, addr_claim_sum)
            .map_err(|e| PiCcsError::SumcheckError(format!("shout addr sumcheck: {e}")))?;

        out.push(ShoutAddrPreProverData {
            addr_claim_sum,
            addr_rounds,
            r_addr,
            decoded,
        });
    }

    Ok(out)
}

pub fn verify_shout_addr_pre_time(
    tr: &mut Poseidon2Transcript,
    step: &StepWitnessBundle<Cmt, F, K>,
    mem_proof: &MemSidecarProof<Cmt, F, K>,
) -> Result<Vec<ShoutAddrPreVerifyData>, PiCcsError> {
    let mut out = Vec::with_capacity(step.lut_instances.len());

    for (idx, (lut_inst, _)) in step.lut_instances.iter().enumerate() {
        let proof = match mem_proof.proofs.get(idx) {
            Some(MemOrLutProof::Shout(p)) => p,
            _ => return Err(PiCcsError::InvalidInput("expected Shout proof".into())),
        };

        let ell_addr = lut_inst.d * lut_inst.ell;
        if proof.addr_rounds.len() != ell_addr {
            return Err(PiCcsError::InvalidInput(format!(
                "shout addr_rounds.len()={}, expected ell_addr={}",
                proof.addr_rounds.len(),
                ell_addr
            )));
        }
        if proof.addr_r.len() != ell_addr {
            return Err(PiCcsError::InvalidInput(format!(
                "shout addr_r.len()={}, expected ell_addr={}",
                proof.addr_r.len(),
                ell_addr
            )));
        }

        tr.append_message(b"shout/addr_claim_idx", &(idx as u64).to_le_bytes());
        tr.append_fields(b"shout/addr_claim_sum", &proof.addr_claim_sum.as_coeffs());

        let (r_addr, addr_final, ok) = verify_sumcheck_rounds(tr, 2, proof.addr_claim_sum, &proof.addr_rounds);
        if !ok {
            return Err(PiCcsError::SumcheckError("shout addr sumcheck invalid".into()));
        }
        if r_addr != proof.addr_r {
            return Err(PiCcsError::ProtocolError(
                "shout addr_r mismatch: transcript-derived vs proof".into(),
            ));
        }

        out.push(ShoutAddrPreVerifyData {
            addr_claim_sum: proof.addr_claim_sum,
            addr_final,
            r_addr,
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
    twist_decoded: &[twist::TwistDecodedCols],
) -> Result<RouteAMemoryOracles, PiCcsError> {
    if shout_pre.len() != step.lut_instances.len() {
        return Err(PiCcsError::InvalidInput(format!(
            "shout pre-time count mismatch (expected {}, got {})",
            step.lut_instances.len(),
            shout_pre.len()
        )));
    }
    if twist_decoded.len() != step.mem_instances.len() {
        return Err(PiCcsError::InvalidInput(format!(
            "twist pre-time decoded count mismatch (expected {}, got {})",
            step.mem_instances.len(),
            twist_decoded.len()
        )));
    }

    let mut shout_oracles = Vec::with_capacity(step.lut_instances.len());
    for ((lut_inst, _lut_wit), pre) in step.lut_instances.iter().zip(shout_pre.iter()) {
        shout_oracles.push(shout::build_route_a_shout_oracles(
            lut_inst,
            &pre.decoded,
            r_cycle,
            &pre.r_addr,
        )?);
    }

    let mut twist_oracles = Vec::with_capacity(step.mem_instances.len());
    for ((mem_inst, _mem_wit), decoded) in step.mem_instances.iter().zip(twist_decoded.iter()) {
        twist_oracles.push(twist::build_route_a_twist_oracles_v2(
            mem_inst,
            decoded,
            r_cycle,
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
    pub read_value_prefixes: Vec<RoundOraclePrefix<'a>>,
    pub read_adapter_prefixes: Vec<RoundOraclePrefix<'a>>,
    pub write_value_prefixes: Vec<RoundOraclePrefix<'a>>,
    pub write_adapter_prefixes: Vec<RoundOraclePrefix<'a>>,
    pub read_value_claims: Vec<K>,
    pub read_adapter_claims: Vec<K>,
    pub write_value_claims: Vec<K>,
    pub write_adapter_claims: Vec<K>,
    pub bitness: Vec<Vec<LazyBitnessOracle>>,
}

pub fn build_route_a_twist_time_claims_guard_v1<'a>(
    twist_oracles: &'a mut [twist::RouteATwistOraclesV1],
    ell_n: usize,
) -> RouteATwistTimeClaimsGuard<'a> {
    let mut read_value_prefixes: Vec<RoundOraclePrefix<'a>> = Vec::with_capacity(twist_oracles.len());
    let mut read_adapter_prefixes: Vec<RoundOraclePrefix<'a>> = Vec::with_capacity(twist_oracles.len());
    let mut write_value_prefixes: Vec<RoundOraclePrefix<'a>> = Vec::with_capacity(twist_oracles.len());
    let mut write_adapter_prefixes: Vec<RoundOraclePrefix<'a>> = Vec::with_capacity(twist_oracles.len());
    let mut read_value_claims: Vec<K> = Vec::with_capacity(twist_oracles.len());
    let mut read_adapter_claims: Vec<K> = Vec::with_capacity(twist_oracles.len());
    let mut write_value_claims: Vec<K> = Vec::with_capacity(twist_oracles.len());
    let mut write_adapter_claims: Vec<K> = Vec::with_capacity(twist_oracles.len());
    let mut bitness: Vec<Vec<LazyBitnessOracle>> = Vec::with_capacity(twist_oracles.len());

    for o in twist_oracles.iter_mut() {
        bitness.push(core::mem::take(&mut o.bitness));
        read_value_claims.push(o.read_value_claim);
        read_adapter_claims.push(o.read_adapter_claim);
        write_value_claims.push(o.write_value_claim);
        write_adapter_claims.push(o.write_adapter_claim);
        read_value_prefixes.push(RoundOraclePrefix::new(&mut o.read_value, ell_n));
        read_adapter_prefixes.push(RoundOraclePrefix::new(&mut o.read_adapter, ell_n));
        write_value_prefixes.push(RoundOraclePrefix::new(&mut o.write_value, ell_n));
        write_adapter_prefixes.push(RoundOraclePrefix::new(&mut o.write_adapter, ell_n));
    }

    RouteATwistTimeClaimsGuard {
        read_value_prefixes,
        read_adapter_prefixes,
        write_value_prefixes,
        write_adapter_prefixes,
        read_value_claims,
        read_adapter_claims,
        write_value_claims,
        write_adapter_claims,
        bitness,
    }
}

pub fn append_route_a_twist_time_claims_v1<'a>(
    guard: &'a mut RouteATwistTimeClaimsGuard<'_>,
    claimed_sums: &mut Vec<K>,
    degree_bounds: &mut Vec<usize>,
    labels: &mut Vec<&'static [u8]>,
    claim_is_dynamic: &mut Vec<bool>,
    claims: &mut Vec<BatchedClaim<'a>>,
) {
    let claim_iter = guard
        .read_value_claims
        .iter()
        .zip(guard.read_adapter_claims.iter())
        .zip(guard.write_value_claims.iter())
        .zip(guard.write_adapter_claims.iter());

    for (
        (((read_value_time, read_adapter_time), write_value_time), write_adapter_time),
        (bitness_vec, (((read_value_claim, read_adapter_claim), write_value_claim), write_adapter_claim)),
    ) in guard
        .read_value_prefixes
        .iter_mut()
        .zip(guard.read_adapter_prefixes.iter_mut())
        .zip(guard.write_value_prefixes.iter_mut())
        .zip(guard.write_adapter_prefixes.iter_mut())
        .zip(guard.bitness.iter_mut().zip(claim_iter))
    {
        claimed_sums.push(*read_value_claim);
        degree_bounds.push(read_value_time.degree_bound());
        labels.push(b"twist/read_value");
        claim_is_dynamic.push(true);
        claims.push(BatchedClaim {
            oracle: read_value_time,
            claimed_sum: *read_value_claim,
            label: b"twist/read_value",
        });

        claimed_sums.push(*read_adapter_claim);
        degree_bounds.push(read_adapter_time.degree_bound());
        labels.push(b"twist/read_adapter");
        claim_is_dynamic.push(true);
        claims.push(BatchedClaim {
            oracle: read_adapter_time,
            claimed_sum: *read_adapter_claim,
            label: b"twist/read_adapter",
        });

        claimed_sums.push(*write_value_claim);
        degree_bounds.push(write_value_time.degree_bound());
        labels.push(b"twist/write_value");
        claim_is_dynamic.push(true);
        claims.push(BatchedClaim {
            oracle: write_value_time,
            claimed_sum: *write_value_claim,
            label: b"twist/write_value",
        });

        claimed_sums.push(*write_adapter_claim);
        degree_bounds.push(write_adapter_time.degree_bound());
        labels.push(b"twist/write_adapter");
        claim_is_dynamic.push(true);
        claims.push(BatchedClaim {
            oracle: write_adapter_time,
            claimed_sum: *write_adapter_claim,
            label: b"twist/write_adapter",
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

pub struct RouteATwistTimeClaimsGuardV2<'a> {
    pub read_check_prefixes: Vec<RoundOraclePrefix<'a>>,
    pub write_check_prefixes: Vec<RoundOraclePrefix<'a>>,
    pub bitness: Vec<Vec<LazyBitnessOracle>>,
}

pub fn build_route_a_twist_time_claims_guard_v2<'a>(
    twist_oracles: &'a mut [twist::RouteATwistOraclesV2],
    ell_n: usize,
) -> RouteATwistTimeClaimsGuardV2<'a> {
    let mut read_check_prefixes: Vec<RoundOraclePrefix<'a>> = Vec::with_capacity(twist_oracles.len());
    let mut write_check_prefixes: Vec<RoundOraclePrefix<'a>> = Vec::with_capacity(twist_oracles.len());
    let mut bitness: Vec<Vec<LazyBitnessOracle>> = Vec::with_capacity(twist_oracles.len());

    for o in twist_oracles.iter_mut() {
        bitness.push(core::mem::take(&mut o.bitness));
        read_check_prefixes.push(RoundOraclePrefix::new(&mut o.read_check, ell_n));
        write_check_prefixes.push(RoundOraclePrefix::new(&mut o.write_check, ell_n));
    }

    RouteATwistTimeClaimsGuardV2 {
        read_check_prefixes,
        write_check_prefixes,
        bitness,
    }
}

pub fn append_route_a_twist_time_claims_v2<'a>(
    guard: &'a mut RouteATwistTimeClaimsGuardV2<'_>,
    claimed_sums: &mut Vec<K>,
    degree_bounds: &mut Vec<usize>,
    labels: &mut Vec<&'static [u8]>,
    claim_is_dynamic: &mut Vec<bool>,
    claims: &mut Vec<BatchedClaim<'a>>,
) {
    for ((read_check_time, write_check_time), bitness_vec) in guard
        .read_check_prefixes
        .iter_mut()
        .zip(guard.write_check_prefixes.iter_mut())
        .zip(guard.bitness.iter_mut())
    {
        // Route A Twist v2: read/write checks are zero-checks, so the claimed sum must be 0.
        claimed_sums.push(K::ZERO);
        degree_bounds.push(read_check_time.degree_bound());
        labels.push(b"twist/read_check");
        claim_is_dynamic.push(false);
        claims.push(BatchedClaim {
            oracle: read_check_time,
            claimed_sum: K::ZERO,
            label: b"twist/read_check",
        });

        claimed_sums.push(K::ZERO);
        degree_bounds.push(write_check_time.degree_bound());
        labels.push(b"twist/write_check");
        claim_is_dynamic.push(false);
        claims.push(BatchedClaim {
            oracle: write_check_time,
            claimed_sum: K::ZERO,
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
    oracles: &mut RouteAMemoryOracles,
    shout_pre: &[ShoutAddrPreProverData],
    twist_decoded: &[twist::TwistDecodedCols],
    r_time: &[K],
    time_final_values: &[K],
    claim_idx_start: usize,
    m_in: usize,
) -> Result<RouteAMemoryProverOutput, PiCcsError> {
    if shout_pre.len() != step.lut_instances.len() {
        return Err(PiCcsError::InvalidInput(format!(
            "shout pre-time count mismatch (expected {}, got {})",
            step.lut_instances.len(),
            shout_pre.len()
        )));
    }
    if twist_decoded.len() != step.mem_instances.len() {
        return Err(PiCcsError::InvalidInput(format!(
            "twist pre-time decoded count mismatch (expected {}, got {})",
            step.mem_instances.len(),
            twist_decoded.len()
        )));
    }
    if oracles.twist.len() != step.mem_instances.len() {
        return Err(PiCcsError::InvalidInput(format!(
            "twist oracle count mismatch (expected {}, got {})",
            step.mem_instances.len(),
            oracles.twist.len()
        )));
    }

    let mut me_claims_time: Vec<MeInstance<Cmt, F, K>> = Vec::new();
    let mut me_wits_time: Vec<Mat<F>> = Vec::new();

    let mut me_claims_val: Vec<MeInstance<Cmt, F, K>> = Vec::new();
    let mut me_wits_val: Vec<Mat<F>> = Vec::new();
    let mut proofs: Vec<MemOrLutProof> = Vec::new();

    // --------------------------------------------------------------------
    // Phase 2: Twist address-round continuation for read/write checks.
    // --------------------------------------------------------------------
    let mut twist_addr_proofs: Vec<twist::BatchedAddrProof<K>> = Vec::with_capacity(step.mem_instances.len());
    {
        // Walk the batched time proof order to locate Twist read/write intermediate sums.
        let mut claim_idx = claim_idx_start;
        for (lut_inst, _lut_wit) in &step.lut_instances {
            let ell_addr = lut_inst.d * lut_inst.ell;
            // shout/value, shout/adapter, shout/bitness × (ell_addr+1)
            claim_idx += 2 + (ell_addr + 1);
        }

        for (i_mem, (mem_inst, _mem_wit)) in step.mem_instances.iter().enumerate() {
            let ell_addr = mem_inst.d * mem_inst.ell;
            let read_sum = *time_final_values.get(claim_idx).ok_or_else(|| {
                PiCcsError::InvalidInput(format!(
                    "missing batched time final value for twist/read_check at claim {}",
                    claim_idx
                ))
            })?;
            claim_idx += 1;
            let write_sum = *time_final_values.get(claim_idx).ok_or_else(|| {
                PiCcsError::InvalidInput(format!(
                    "missing batched time final value for twist/write_check at claim {}",
                    claim_idx
                ))
            })?;
            claim_idx += 1;
            // twist/bitness × (2*ell_addr + 2)
            claim_idx += 2 * ell_addr + 2;

            // After ell_n time folds, the 2D check oracles have exactly ell_addr rounds remaining.
            if oracles.twist[i_mem].read_check.num_rounds() != ell_addr
                || oracles.twist[i_mem].write_check.num_rounds() != ell_addr
            {
                return Err(PiCcsError::ProtocolError(format!(
                    "Twist(Route A): unexpected remaining rounds after time folding (read={}, write={}, expected ell_addr={})",
                    oracles.twist[i_mem].read_check.num_rounds(),
                    oracles.twist[i_mem].write_check.num_rounds(),
                    ell_addr
                )));
            }

            let labels: [&[u8]; 2] = [b"twist/read_check/addr".as_slice(), b"twist/write_check/addr".as_slice()];
            let claimed_sums = vec![read_sum, write_sum];
            tr.append_message(b"twist/check_addr_batch/claim_idx", &(i_mem as u64).to_le_bytes());
            bind_batched_claim_sums(tr, b"twist/check_addr_batch/claimed_sums", &claimed_sums, &labels);

            let (read_check, write_check) = {
                let o = &mut oracles.twist[i_mem];
                (&mut o.read_check, &mut o.write_check)
            };

            let mut claims = [
                BatchedClaim {
                    oracle: read_check,
                    claimed_sum: read_sum,
                    label: labels[0],
                },
                BatchedClaim {
                    oracle: write_check,
                    claimed_sum: write_sum,
                    label: labels[1],
                },
            ];

            let (r_addr, per_claim_results) = run_batched_sumcheck_prover(tr, &mut claims)
                .map_err(|e| PiCcsError::SumcheckError(format!("twist check addr batched sumcheck: {e}")))?;
            if r_addr.len() != ell_addr {
                return Err(PiCcsError::ProtocolError(format!(
                    "twist check addr r_addr.len()={}, expected ell_addr={}",
                    r_addr.len(),
                    ell_addr
                )));
            }
            if per_claim_results.len() != 2 {
                return Err(PiCcsError::ProtocolError(format!(
                    "twist check addr per-claim results len()={}, expected 2",
                    per_claim_results.len()
                )));
            }
            let mut it = per_claim_results.into_iter();
            let read_rounds = it
                .next()
                .ok_or_else(|| PiCcsError::ProtocolError("missing twist/read_check addr rounds".into()))?
                .round_polys;
            let write_rounds = it
                .next()
                .ok_or_else(|| PiCcsError::ProtocolError("missing twist/write_check addr rounds".into()))?
                .round_polys;

            twist_addr_proofs.push(twist::BatchedAddrProof {
                claimed_sums,
                round_polys: vec![read_rounds, write_rounds],
                r_addr,
            });
        }
    }

    // --------------------------------------------------------------------
    // Phase 2: Twist val-eval sum-check (batched across mem instances).
    // --------------------------------------------------------------------
    let mut twist_val_eval_proofs: Vec<twist::TwistValEvalProof<K>> = Vec::new();
    let mut r_val: Vec<K> = Vec::new();
    if !step.mem_instances.is_empty() {
        let n_mem = step.mem_instances.len();
        let claim_count = 2 * n_mem;

        let mut val_oracles: Vec<Box<dyn RoundOracle>> = Vec::with_capacity(claim_count);
        let mut bind_claims: Vec<(u8, K)> = Vec::with_capacity(claim_count);
        let mut claimed_sums: Vec<K> = Vec::with_capacity(claim_count);
        let mut labels: Vec<&'static [u8]> = Vec::with_capacity(claim_count);

        let mut claimed_inc_sums_lt: Vec<K> = Vec::with_capacity(n_mem);
        let mut claimed_inc_sums_total: Vec<K> = Vec::with_capacity(n_mem);
        let mut claimed_vals: Vec<K> = Vec::with_capacity(n_mem);

        if twist_addr_proofs.len() != step.mem_instances.len() {
            return Err(PiCcsError::ProtocolError(format!(
                "twist addr proof count mismatch (expected {}, got {})",
                step.mem_instances.len(),
                twist_addr_proofs.len()
            )));
        }

        for (i_mem, ((mem_inst, _mem_wit), decoded)) in step
            .mem_instances
            .iter()
            .zip(twist_decoded.iter())
            .enumerate()
        {
            let r_addr = &twist_addr_proofs[i_mem].r_addr;
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

            let init_table_k: Vec<K> = mem_inst.init_vals.iter().map(|&v| v.into()).collect();
            let init_at_r_addr = table_mle_eval(&init_table_k, r_addr);
            let claimed_val = init_at_r_addr + claimed_inc_sum_lt;

            val_oracles.push(Box::new(oracle_lt));
            bind_claims.push((0u8, claimed_inc_sum_lt));
            claimed_sums.push(claimed_inc_sum_lt);
            labels.push(b"twist/val_eval_lt");

            val_oracles.push(Box::new(oracle_total));
            bind_claims.push((1u8, claimed_inc_sum_total));
            claimed_sums.push(claimed_inc_sum_total);
            labels.push(b"twist/val_eval_total");

            claimed_inc_sums_lt.push(claimed_inc_sum_lt);
            claimed_inc_sums_total.push(claimed_inc_sum_total);
            claimed_vals.push(claimed_val);
        }

        tr.append_message(
            b"twist/val_eval/batch_start",
            &(step.mem_instances.len() as u64).to_le_bytes(),
        );
        bind_twist_val_eval_claim_sums(tr, &bind_claims);

        let mut claims: Vec<BatchedClaim<'_>> = val_oracles
            .iter_mut()
            .zip(claimed_sums.iter())
            .zip(labels.iter())
            .map(|((oracle, sum), label)| BatchedClaim {
                oracle: oracle.as_mut(),
                claimed_sum: *sum,
                label: *label,
            })
            .collect();

        let (r_val_out, per_claim_results) = run_batched_sumcheck_prover(tr, claims.as_mut_slice())
            .map_err(|e| PiCcsError::SumcheckError(format!("twist val-eval batched sumcheck: {e}")))?;

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
            twist_val_eval_proofs.push(twist::TwistValEvalProof {
                claimed_val: claimed_vals[i],
                claimed_inc_sum_lt: claimed_inc_sums_lt[i],
                rounds_lt: per_claim_results[2 * i]
                    .round_polys
                    .clone(),
                claimed_inc_sum_total: claimed_inc_sums_total[i],
                rounds_total: per_claim_results[2 * i + 1]
                    .round_polys
                    .clone(),
                r_val: r_val.clone(),
            });
        }

        tr.append_message(b"twist/val_eval/batch_done", &[]);
    }

    for ((lut_inst, lut_wit), pre) in step.lut_instances.iter().zip(shout_pre.iter()) {
        let mut proof = ShoutProofK::default();
        proof.me_claim_count = lut_wit.mats.len();
        proof.addr_claim_sum = pre.addr_claim_sum;
        proof.addr_rounds = pre.addr_rounds.clone();
        proof.addr_r = pre.r_addr.clone();

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
        proof.me_claim_count = mem_wit.mats.len();
        proof.addr_batch = twist_addr_proofs
            .get(idx)
            .cloned()
            .ok_or_else(|| PiCcsError::ProtocolError("missing Twist addr continuation proof".into()))?;
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
        if !r_val.is_empty() {
            let parts = twist::split_mem_mats(mem_inst, mem_wit);
            let ell_addr = mem_inst.d * mem_inst.ell;

            for (b, mat) in parts.wa_bit_mats.iter().enumerate() {
                let comm = mem_inst.comms.get(ell_addr + b).ok_or_else(|| {
                    PiCcsError::InvalidInput("Twist: comms shorter than wa_bit_mats".into())
                })?;
                me_claims_val.push(ts::mk_me_opening_with_ccs(
                    tr,
                    b"twist/me_digest_val",
                    params,
                    s,
                    comm,
                    mat,
                    &r_val,
                    m_in,
                )?);
                me_wits_val.push((*mat).clone());
            }

            let comm_has_write = mem_inst.comms.get(2 * ell_addr + 1).ok_or_else(|| {
                PiCcsError::InvalidInput("Twist: comms shorter than has_write".into())
            })?;
            me_claims_val.push(ts::mk_me_opening_with_ccs(
                tr,
                b"twist/me_digest_val",
                params,
                s,
                comm_has_write,
                parts.has_write_mat,
                &r_val,
                m_in,
            )?);
            me_wits_val.push(parts.has_write_mat.clone());

            let comm_inc_at_write_addr = mem_inst.comms.get(2 * ell_addr + 4).ok_or_else(|| {
                PiCcsError::InvalidInput("Twist: comms shorter than inc_at_write_addr".into())
            })?;
            me_claims_val.push(ts::mk_me_opening_with_ccs(
                tr,
                b"twist/me_digest_val",
                params,
                s,
                comm_inc_at_write_addr,
                parts.inc_at_write_addr_mat,
                &r_val,
                m_in,
            )?);
            me_wits_val.push(parts.inc_at_write_addr_mat.clone());
        }

        proofs.push(MemOrLutProof::Twist(proof));
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

pub fn append_expected_batched_time_metadata_for_memory(
    step: &StepWitnessBundle<Cmt, F, K>,
    expected_degree_bounds: &mut Vec<usize>,
    expected_labels: &mut Vec<&[u8]>,
    claim_is_dynamic: &mut Vec<bool>,
) {
    for (lut_inst, _lut_wit) in &step.lut_instances {
        let ell_addr = lut_inst.d * lut_inst.ell;

        expected_degree_bounds.push(3);
        expected_labels.push(b"shout/value");
        claim_is_dynamic.push(true);

        expected_degree_bounds.push(2 + ell_addr);
        expected_labels.push(b"shout/adapter");
        claim_is_dynamic.push(true);

        for _ in 0..(ell_addr + 1) {
            expected_degree_bounds.push(3);
            expected_labels.push(b"shout/bitness");
            claim_is_dynamic.push(false);
        }
    }

    for (mem_inst, _mem_wit) in &step.mem_instances {
        let ell_addr = mem_inst.d * mem_inst.ell;

        expected_degree_bounds.push(3 + ell_addr);
        expected_labels.push(b"twist/read_check");
        claim_is_dynamic.push(false);

        expected_degree_bounds.push(3 + ell_addr);
        expected_labels.push(b"twist/write_check");
        claim_is_dynamic.push(false);

        let bitness_count = 2 * ell_addr + 2;
        for _ in 0..bitness_count {
            expected_degree_bounds.push(3);
            expected_labels.push(b"twist/bitness");
            claim_is_dynamic.push(false);
        }
    }
}

pub struct RouteAMemoryVerifyOutput<C> {
    pub collected_me_time: Vec<MeInstance<C, F, K>>,
    pub collected_me_val: Vec<MeInstance<C, F, K>>,
    pub claim_idx_end: usize,
    pub twist_rollover: Vec<(Vec<K>, K)>,
}

pub fn verify_route_a_memory_step(
    tr: &mut Poseidon2Transcript,
    _params: &NeoParams,
    step: &StepWitnessBundle<Cmt, F, K>,
    r_time: &[K],
    r_cycle: &[K],
    batched_final_values: &[K],
    batched_claimed_sums: &[K],
    claim_idx_start: usize,
    mem_proof: &MemSidecarProof<Cmt, F, K>,
    shout_pre: &[ShoutAddrPreVerifyData],
) -> Result<RouteAMemoryVerifyOutput<Cmt>, PiCcsError> {
    let chi_cycle_at_r_time = eq_points(r_time, r_cycle);

    let proofs_mem = &mem_proof.proofs;
    let mut me_time_offset = 0usize;
    let mut me_val_offset = 0usize;
    let mut collected_me_time = Vec::new();
    let mut collected_me_val = Vec::new();
    let mut claim_idx = claim_idx_start;

    let expected_proofs = step.lut_instances.len() + step.mem_instances.len();
    if proofs_mem.len() != expected_proofs {
        return Err(PiCcsError::InvalidInput(format!(
            "mem proof count mismatch (expected {}, got {})",
            expected_proofs,
            proofs_mem.len()
        )));
    }
    if shout_pre.len() != step.lut_instances.len() {
        return Err(PiCcsError::InvalidInput(format!(
            "shout pre-time count mismatch (expected {}, got {})",
            step.lut_instances.len(),
            shout_pre.len()
        )));
    }

    // Shout instances first.
    for (proof_idx, (inst, _)) in step.lut_instances.iter().enumerate() {
        let shout_proof = match &proofs_mem[proof_idx] {
            MemOrLutProof::Shout(proof) => proof,
            _ => return Err(PiCcsError::InvalidInput("expected Shout proof".into())),
        };

        let shout_me_count = shout_proof.me_claim_count;
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

        let ell_addr = inst.d * inst.ell;
        if shout_me_count < ell_addr + 2 {
            return Err(PiCcsError::InvalidInput(format!(
                "Shout me_claim_count={} too small (need at least {} for addr_bits+has_lookup+val)",
                shout_me_count,
                ell_addr + 2
            )));
        }

        let pre = shout_pre
            .get(proof_idx)
            .ok_or_else(|| PiCcsError::InvalidInput(format!("missing pre-time Shout data at index {}", proof_idx)))?;

        // Route A Shout ordering in batched_time:
        // - value (time rounds only)
        // - adapter (time rounds only)
        // - bitness for addr_bits then has_lookup
        let value_claim = batched_claimed_sums[claim_idx];
        let value_final = batched_final_values[claim_idx];
        claim_idx += 1;
        let adapter_claim = batched_claimed_sums[claim_idx];
        let adapter_final = batched_final_values[claim_idx];
        claim_idx += 1;

        for j in 0..ell_addr {
            let b_open = *me_slice[j]
                .y_scalars
                .get(0)
                .ok_or_else(|| PiCcsError::ProtocolError("ME.y_scalars missing identity-first entry".into()))?;
            let expected_eval = chi_cycle_at_r_time * b_open * (b_open - K::ONE);
            if expected_eval != batched_final_values[claim_idx] {
                return Err(PiCcsError::ProtocolError(format!(
                    "shout bitness final value mismatch at addr_bit {}",
                    j
                )));
            }
            claim_idx += 1;
        }
        let has_lookup_open = *me_slice[ell_addr]
            .y_scalars
            .get(0)
            .ok_or_else(|| PiCcsError::ProtocolError("ME.y_scalars missing identity-first entry".into()))?;
        {
            let expected_eval = chi_cycle_at_r_time * has_lookup_open * (has_lookup_open - K::ONE);
            if expected_eval != batched_final_values[claim_idx] {
                return Err(PiCcsError::ProtocolError(
                    "shout bitness final value mismatch at has_lookup".into(),
                ));
            }
            claim_idx += 1;
        }

        let addr_bits_open: Vec<K> = me_slice[..ell_addr]
            .iter()
            .map(|me| {
                me.y_scalars
                    .get(0)
                    .copied()
                    .ok_or_else(|| PiCcsError::ProtocolError("ME.y_scalars missing identity-first entry".into()))
            })
            .collect::<Result<_, _>>()?;
        let val_open = *me_slice[ell_addr + 1]
            .y_scalars
            .get(0)
            .ok_or_else(|| PiCcsError::ProtocolError("ME.y_scalars missing identity-first entry".into()))?;

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

        let table_k: Vec<K> = inst.table.iter().map(|&v| v.into()).collect();
        let table_eval = table_mle_eval(&table_k, &pre.r_addr);
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
            if j < inst.comms.len() && me.c != inst.comms[j] {
                return Err(PiCcsError::ProtocolError(format!(
                    "Shout ME[{}] commitment mismatch",
                    j
                )));
            }
        }

        collected_me_time.extend_from_slice(me_slice);
    }

    // Twist instances next.
    let proof_mem_offset = step.lut_instances.len();

    // --------------------------------------------------------------------
    // Phase 2: Verify Twist addr-round continuations for read/write checks,
    // deriving per-instance r_addr (must happen before val-eval in transcript).
    // --------------------------------------------------------------------
    let mut twist_r_addrs: Vec<Vec<K>> = Vec::with_capacity(step.mem_instances.len());

    for (i_mem, (inst, _)) in step.mem_instances.iter().enumerate() {
        let twist_proof = match &proofs_mem[proof_mem_offset + i_mem] {
            MemOrLutProof::Twist(proof) => proof,
            _ => return Err(PiCcsError::InvalidInput("expected Twist proof".into())),
        };

        let twist_me_count = twist_proof.me_claim_count;
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

        let ell_addr = inst.d * inst.ell;
        if twist_me_count < 2 * ell_addr + 5 {
            return Err(PiCcsError::InvalidInput(format!(
                "Twist me_claim_count={} too small (need at least {} for current layout)",
                twist_me_count,
                2 * ell_addr + 5
            )));
        }

        // Route A Twist ordering in batched_time:
        // - read_check (time rounds only; final value is the post-time running sum over addr vars)
        // - write_check (time rounds only; final value is the post-time running sum over addr vars)
        // - bitness for ra_bits then wa_bits then has_read then has_write (time-only)
        let read_check_claim = batched_claimed_sums[claim_idx];
        let read_time_sum = batched_final_values[claim_idx];
        claim_idx += 1;
        let write_check_claim = batched_claimed_sums[claim_idx];
        let write_time_sum = batched_final_values[claim_idx];
        claim_idx += 1;

        let ra_bits_open: Vec<K> = me_slice[..ell_addr]
            .iter()
            .map(|me| {
                me.y_scalars
                    .get(0)
                    .copied()
                    .ok_or_else(|| PiCcsError::ProtocolError("ME.y_scalars missing identity-first entry".into()))
            })
            .collect::<Result<_, _>>()?;
        let wa_bits_open: Vec<K> = me_slice[ell_addr..2 * ell_addr]
            .iter()
            .map(|me| {
                me.y_scalars
                    .get(0)
                    .copied()
                    .ok_or_else(|| PiCcsError::ProtocolError("ME.y_scalars missing identity-first entry".into()))
            })
            .collect::<Result<_, _>>()?;

        let has_read_open = *me_slice[2 * ell_addr + 0]
            .y_scalars
            .get(0)
            .ok_or_else(|| PiCcsError::ProtocolError("ME.y_scalars missing identity-first entry".into()))?;
        let has_write_open = *me_slice[2 * ell_addr + 1]
            .y_scalars
            .get(0)
            .ok_or_else(|| PiCcsError::ProtocolError("ME.y_scalars missing identity-first entry".into()))?;
        let wv_open = *me_slice[2 * ell_addr + 2]
            .y_scalars
            .get(0)
            .ok_or_else(|| PiCcsError::ProtocolError("ME.y_scalars missing identity-first entry".into()))?;
        let rv_open = *me_slice[2 * ell_addr + 3]
            .y_scalars
            .get(0)
            .ok_or_else(|| PiCcsError::ProtocolError("ME.y_scalars missing identity-first entry".into()))?;
        let inc_write_open = *me_slice[2 * ell_addr + 4]
            .y_scalars
            .get(0)
            .ok_or_else(|| PiCcsError::ProtocolError("ME.y_scalars missing identity-first entry".into()))?;

        // Bitness terminal checks (ra_bits then wa_bits then has_read then has_write).
        for (j, col_open) in ra_bits_open
            .iter()
            .chain(wa_bits_open.iter())
            .chain([has_read_open, has_write_open].iter())
            .enumerate()
        {
            let expected_eval = chi_cycle_at_r_time * *col_open * (*col_open - K::ONE);
            if expected_eval != batched_final_values[claim_idx] {
                return Err(PiCcsError::ProtocolError(format!(
                    "twist/bitness terminal value mismatch at chunk {}",
                    j
                )));
            }
            claim_idx += 1;
        }

        // Route A Twist v2: read/write checks are zero-checks over time×addr.
        if read_check_claim != K::ZERO {
            return Err(PiCcsError::ProtocolError(
                "twist/read_check claimed sum must be 0".into(),
            ));
        }
        if write_check_claim != K::ZERO {
            return Err(PiCcsError::ProtocolError(
                "twist/write_check claimed sum must be 0".into(),
            ));
        }

        // Verify the address-round continuation sumchecks from the post-time running sums.
        let addr_batch = &twist_proof.addr_batch;
        if addr_batch.claimed_sums.len() != 2 {
            return Err(PiCcsError::InvalidInput(format!(
                "twist addr_batch.claimed_sums.len()={}, expected 2",
                addr_batch.claimed_sums.len()
            )));
        }
        if addr_batch.round_polys.len() != 2 {
            return Err(PiCcsError::InvalidInput(format!(
                "twist addr_batch.round_polys.len()={}, expected 2",
                addr_batch.round_polys.len()
            )));
        }
        if addr_batch.r_addr.len() != ell_addr {
            return Err(PiCcsError::InvalidInput(format!(
                "twist addr_batch.r_addr.len()={}, expected ell_addr={}",
                addr_batch.r_addr.len(),
                ell_addr
            )));
        }
        for (i, rounds) in addr_batch.round_polys.iter().enumerate() {
            if rounds.len() != ell_addr {
                return Err(PiCcsError::InvalidInput(format!(
                    "twist addr_batch.round_polys[{}].len()={}, expected ell_addr={}",
                    i,
                    rounds.len(),
                    ell_addr
                )));
            }
        }
        if addr_batch.claimed_sums[0] != read_time_sum || addr_batch.claimed_sums[1] != write_time_sum {
            return Err(PiCcsError::ProtocolError(
                "twist addr_batch claimed_sums mismatch with time-round finals".into(),
            ));
        }

        let labels: [&[u8]; 2] = [b"twist/read_check/addr".as_slice(), b"twist/write_check/addr".as_slice()];
        tr.append_message(b"twist/check_addr_batch/claim_idx", &(i_mem as u64).to_le_bytes());
        bind_batched_claim_sums(tr, b"twist/check_addr_batch/claimed_sums", &addr_batch.claimed_sums, &labels);

        let degree_bounds = vec![3 + ell_addr, 3 + ell_addr];
        let (r_addr, finals, ok) = verify_batched_sumcheck_rounds(
            tr,
            &addr_batch.round_polys,
            &addr_batch.claimed_sums,
            &labels,
            &degree_bounds,
        );
        if !ok {
            return Err(PiCcsError::SumcheckError(
                "twist check addr batched sumcheck invalid".into(),
            ));
        }
        if r_addr != addr_batch.r_addr {
            return Err(PiCcsError::ProtocolError(
                "twist addr_batch r_addr mismatch: transcript-derived vs proof".into(),
            ));
        }
        if finals.len() != 2 {
            return Err(PiCcsError::ProtocolError(format!(
                "twist addr_batch finals.len()={}, expected 2",
                finals.len()
            )));
        }

        let val_eval = twist_proof
            .val_eval
            .as_ref()
            .ok_or_else(|| PiCcsError::InvalidInput("Twist(Route A): missing val_eval proof".into()))?;

        // Verify claimed_val = init(r_addr) + claimed_inc_sum.
        let init_table_k: Vec<K> = inst.init_vals.iter().map(|&v| v.into()).collect();
        let init_at_r_addr = table_mle_eval(&init_table_k, &r_addr);
        if val_eval.claimed_val != init_at_r_addr + val_eval.claimed_inc_sum_lt {
            return Err(PiCcsError::ProtocolError("twist claimed_val mismatch".into()));
        }

        let read_eq_addr = eq_bits_prod(&ra_bits_open, &r_addr)?;
        let write_eq_addr = eq_bits_prod(&wa_bits_open, &r_addr)?;

        // Terminal checks for read_check / write_check at (r_time, r_addr).
        let expected_read_check_final =
            chi_cycle_at_r_time * has_read_open * (val_eval.claimed_val - rv_open) * read_eq_addr;
        if expected_read_check_final != finals[0] {
            return Err(PiCcsError::ProtocolError(
                "twist/read_check terminal value mismatch".into(),
            ));
        }

        let expected_write_check_final =
            chi_cycle_at_r_time * has_write_open * (wv_open - val_eval.claimed_val - inc_write_open) * write_eq_addr;
        if expected_write_check_final != finals[1] {
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
            if j < inst.comms.len() && me.c != inst.comms[j] {
                return Err(PiCcsError::ProtocolError(format!(
                    "Twist ME[{}] commitment mismatch",
                    j
                )));
            }
        }

        collected_me_time.extend_from_slice(me_slice);
        twist_r_addrs.push(r_addr);
    }

    // --------------------------------------------------------------------
    // Phase 2: Verify batched Twist val-eval sum-check, deriving shared r_val.
    // --------------------------------------------------------------------
    let mut r_val: Vec<K> = Vec::new();
    let mut val_eval_finals: Vec<K> = Vec::new();
    if !step.mem_instances.is_empty() {
        let n_mem = step.mem_instances.len();
        let claim_count = 2 * n_mem;

        let mut per_claim_rounds: Vec<Vec<Vec<K>>> = Vec::with_capacity(claim_count);
        let mut per_claim_sums: Vec<K> = Vec::with_capacity(claim_count);
        let mut bind_claims: Vec<(u8, K)> = Vec::with_capacity(claim_count);
        let mut labels: Vec<&[u8]> = Vec::with_capacity(claim_count);
        let mut degree_bounds: Vec<usize> = Vec::with_capacity(claim_count);

        for (i_mem, (inst, _wit)) in step.mem_instances.iter().enumerate() {
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
            bind_claims.push((0u8, val.claimed_inc_sum_lt));
            labels.push(b"twist/val_eval_lt".as_slice());
            degree_bounds.push((inst.d * inst.ell) + 3);

            per_claim_rounds.push(val.rounds_total.clone());
            per_claim_sums.push(val.claimed_inc_sum_total);
            bind_claims.push((1u8, val.claimed_inc_sum_total));
            labels.push(b"twist/val_eval_total".as_slice());
            degree_bounds.push((inst.d * inst.ell) + 2);
        }

        tr.append_message(
            b"twist/val_eval/batch_start",
            &(step.mem_instances.len() as u64).to_le_bytes(),
        );
        bind_twist_val_eval_claim_sums(tr, &bind_claims);

        let (r_val_out, finals_out, ok) =
            verify_batched_sumcheck_rounds(tr, &per_claim_rounds, &per_claim_sums, &labels, &degree_bounds);
        if !ok {
            return Err(PiCcsError::SumcheckError("twist val-eval batched sumcheck invalid".into()));
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
    let mut twist_rollover: Vec<(Vec<K>, K)> = Vec::with_capacity(step.mem_instances.len());
    let lt = if step.mem_instances.is_empty() {
        K::ZERO
    } else {
        lt_eval(&r_val, r_time)
    };
    for (i_mem, (inst, _)) in step.mem_instances.iter().enumerate() {
        let twist_proof = match &proofs_mem[proof_mem_offset + i_mem] {
            MemOrLutProof::Twist(proof) => proof,
            _ => return Err(PiCcsError::InvalidInput("expected Twist proof".into())),
        };
        let val_eval = twist_proof
            .val_eval
            .as_ref()
            .ok_or_else(|| PiCcsError::InvalidInput("Twist(Route A): missing val_eval proof".into()))?;
        if val_eval.r_val != r_val {
            return Err(PiCcsError::ProtocolError(
                "twist val-eval r_val mismatch: proof vs transcript-derived".into(),
            ));
        }

        let ell_addr = inst.d * inst.ell;
        let val_me_count = ell_addr + 2;
        let me_val_slice = mem_proof
            .me_claims_val
            .get(me_val_offset..me_val_offset + val_me_count)
            .ok_or_else(|| {
                PiCcsError::InvalidInput(format!(
                    "Not enough ME claims for Twist val lane: need {} at offset {}, have {}",
                    val_me_count,
                    me_val_offset,
                    mem_proof.me_claims_val.len()
                ))
            })?;
        me_val_offset += val_me_count;

        // Layout: wa_bits (ell_addr), has_write, inc_at_write_addr.
        let wa_bits_val_open: Vec<K> = me_val_slice[..ell_addr]
            .iter()
            .map(|me| {
                me.y_scalars
                    .get(0)
                    .copied()
                    .ok_or_else(|| PiCcsError::ProtocolError("ME.y_scalars missing identity-first entry".into()))
            })
            .collect::<Result<_, _>>()?;
        let has_write_val_open = *me_val_slice[ell_addr]
            .y_scalars
            .get(0)
            .ok_or_else(|| PiCcsError::ProtocolError("ME.y_scalars missing identity-first entry".into()))?;
        let inc_at_write_addr_val_open = *me_val_slice[ell_addr + 1]
            .y_scalars
            .get(0)
            .ok_or_else(|| PiCcsError::ProtocolError("ME.y_scalars missing identity-first entry".into()))?;

        let eq_wa_val = eq_bits_prod(&wa_bits_val_open, &twist_r_addrs[i_mem])?;
        let inc_at_r_addr_val = has_write_val_open * inc_at_write_addr_val_open * eq_wa_val;
        let expected_lt_final = inc_at_r_addr_val * lt;
        if expected_lt_final != val_eval_finals[2 * i_mem] {
            return Err(PiCcsError::ProtocolError(
                "twist/val_eval_lt terminal value mismatch".into(),
            ));
        }
        let expected_total_final = inc_at_r_addr_val;
        if expected_total_final != val_eval_finals[2 * i_mem + 1] {
            return Err(PiCcsError::ProtocolError(
                "twist/val_eval_total terminal value mismatch".into(),
            ));
        }

        let init_table_k: Vec<K> = inst.init_vals.iter().map(|&v| v.into()).collect();
        let init_at_r_addr = table_mle_eval(&init_table_k, &twist_r_addrs[i_mem]);
        let end_at_r_addr = init_at_r_addr + val_eval.claimed_inc_sum_total;
        twist_rollover.push((twist_r_addrs[i_mem].clone(), end_at_r_addr));

        // Enforce r/commitment alignment for the r_val ME claims.
        for (j, me) in me_val_slice.iter().enumerate() {
            if me.r != r_val {
                return Err(PiCcsError::ProtocolError(format!(
                    "Twist(val) ME[{}] r != r_val",
                    j
                )));
            }
            let want_comm = if j < ell_addr {
                inst.comms
                    .get(ell_addr + j)
                    .ok_or_else(|| PiCcsError::InvalidInput("Twist(val): comms too short".into()))?
            } else if j == ell_addr {
                inst.comms
                    .get(2 * ell_addr + 1)
                    .ok_or_else(|| PiCcsError::InvalidInput("Twist(val): comms too short".into()))?
            } else {
                inst.comms
                    .get(2 * ell_addr + 4)
                    .ok_or_else(|| PiCcsError::InvalidInput("Twist(val): comms too short".into()))?
            };
            if me.c != *want_comm {
                return Err(PiCcsError::ProtocolError(format!(
                    "Twist(val) ME[{}] commitment mismatch",
                    j
                )));
            }
        }

        collected_me_val.extend_from_slice(me_val_slice);
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
        claim_idx_end: claim_idx,
        twist_rollover,
    })
}

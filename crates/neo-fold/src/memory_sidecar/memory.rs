use crate::memory_sidecar::claim_plan::RouteATimeClaimPlan;
use crate::memory_sidecar::sumcheck_ds::{run_batched_sumcheck_prover_ds, verify_batched_sumcheck_rounds_ds};
use crate::memory_sidecar::transcript::{bind_batched_claim_sums, bind_twist_val_eval_claim_sums, digest_fields};
use crate::memory_sidecar::utils::{bitness_weights, RoundOraclePrefix};
use crate::shard_proof_types::{MemOrLutProof, MemSidecarProof, ShoutAddrPreProof, ShoutProofK, TwistProofK};
use crate::PiCcsError;
use neo_ajtai::Commitment as Cmt;
use neo_ccs::{CcsStructure, MeInstance};
use neo_math::{F, K};
use neo_memory::bit_ops::{eq_bit_affine, eq_bits_prod};
use neo_memory::cpu::BusLayout;
use neo_memory::mle::{eq_points, lt_eval};
use neo_memory::riscv::shout_oracle::RiscvAddressLookupOracleSparse;
use neo_memory::sparse_time::SparseIdxVec;
use neo_memory::ts_common as ts;
use neo_memory::twist_oracle::{
    AddressLookupOracle, IndexAdapterOracleSparseTime, LazyWeightedBitnessOracleSparseTime, ShoutValueOracleSparse,
    TwistLaneSparseCols, TwistReadCheckAddrOracleSparseTimeMultiLane, TwistReadCheckOracleSparseTime,
    TwistTotalIncOracleSparseTime, TwistValEvalOracleSparseTime, TwistWriteCheckAddrOracleSparseTimeMultiLane,
    TwistWriteCheckOracleSparseTime,
};
use neo_memory::witness::{LutInstance, LutTableSpec, MemInstance, StepInstanceBundle, StepWitnessBundle};
use neo_memory::{eval_init_at_r_addr, twist, BatchedAddrProof, MemInit};
use neo_params::NeoParams;
use neo_reductions::sumcheck::{BatchedClaim, RoundOracle};
use neo_transcript::{Poseidon2Transcript, Transcript};
use p3_field::PrimeCharacteristicRing;

// ============================================================================
// Transcript binding
// ============================================================================

fn bind_shout_table_spec(tr: &mut Poseidon2Transcript, spec: &Option<LutTableSpec>) {
    let Some(spec) = spec else {
        return;
    };

    tr.append_message(b"shout/table_spec/tag", &[1u8]);
    match spec {
        LutTableSpec::RiscvOpcode { opcode, xlen } => {
            // Stable numeric encoding: align with `RiscvShoutTables::opcode_to_id`.
            let opcode_id: u64 = match opcode {
                neo_memory::riscv::lookups::RiscvOpcode::And => 0,
                neo_memory::riscv::lookups::RiscvOpcode::Xor => 1,
                neo_memory::riscv::lookups::RiscvOpcode::Or => 2,
                neo_memory::riscv::lookups::RiscvOpcode::Add => 3,
                neo_memory::riscv::lookups::RiscvOpcode::Sub => 4,
                neo_memory::riscv::lookups::RiscvOpcode::Slt => 5,
                neo_memory::riscv::lookups::RiscvOpcode::Sltu => 6,
                neo_memory::riscv::lookups::RiscvOpcode::Sll => 7,
                neo_memory::riscv::lookups::RiscvOpcode::Srl => 8,
                neo_memory::riscv::lookups::RiscvOpcode::Sra => 9,
                neo_memory::riscv::lookups::RiscvOpcode::Eq => 10,
                neo_memory::riscv::lookups::RiscvOpcode::Neq => 11,
                neo_memory::riscv::lookups::RiscvOpcode::Mul => 12,
                neo_memory::riscv::lookups::RiscvOpcode::Mulh => 13,
                neo_memory::riscv::lookups::RiscvOpcode::Mulhu => 14,
                neo_memory::riscv::lookups::RiscvOpcode::Mulhsu => 15,
                neo_memory::riscv::lookups::RiscvOpcode::Div => 16,
                neo_memory::riscv::lookups::RiscvOpcode::Divu => 17,
                neo_memory::riscv::lookups::RiscvOpcode::Rem => 18,
                neo_memory::riscv::lookups::RiscvOpcode::Remu => 19,
                neo_memory::riscv::lookups::RiscvOpcode::Addw => 20,
                neo_memory::riscv::lookups::RiscvOpcode::Subw => 21,
                neo_memory::riscv::lookups::RiscvOpcode::Sllw => 22,
                neo_memory::riscv::lookups::RiscvOpcode::Srlw => 23,
                neo_memory::riscv::lookups::RiscvOpcode::Sraw => 24,
                neo_memory::riscv::lookups::RiscvOpcode::Mulw => 25,
                neo_memory::riscv::lookups::RiscvOpcode::Divw => 26,
                neo_memory::riscv::lookups::RiscvOpcode::Divuw => 27,
                neo_memory::riscv::lookups::RiscvOpcode::Remw => 28,
                neo_memory::riscv::lookups::RiscvOpcode::Remuw => 29,
                neo_memory::riscv::lookups::RiscvOpcode::Andn => 30,
            };

            tr.append_message(b"shout/table_spec/riscv/tag", &[1u8]);
            tr.append_message(b"shout/table_spec/riscv/opcode_id", &opcode_id.to_le_bytes());
            tr.append_message(b"shout/table_spec/riscv/xlen", &(*xlen as u64).to_le_bytes());
        }
    }
}

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
        tr.append_message(b"shout/lanes", &(inst.lanes.max(1) as u64).to_le_bytes());
        bind_shout_table_spec(tr, &inst.table_spec);
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
        tr.append_message(b"twist/lanes", &(inst.lanes.max(1) as u64).to_le_bytes());
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
    pub lanes: Vec<ShoutLaneSparseCols>,
}

pub(crate) struct ShoutLaneSparseCols {
    pub addr_bits: Vec<SparseIdxVec<K>>,
    pub has_lookup: SparseIdxVec<K>,
    pub val: SparseIdxVec<K>,
}

pub(crate) struct TwistDecodedColsSparse {
    pub lanes: Vec<TwistLaneSparseCols>,
}

pub(crate) struct SumRoundOracle {
    oracles: Vec<Box<dyn RoundOracle>>,
    num_rounds: usize,
    degree_bound: usize,
}

impl SumRoundOracle {
    pub(crate) fn new(oracles: Vec<Box<dyn RoundOracle>>) -> Self {
        if oracles.is_empty() {
            panic!("SumRoundOracle requires at least one oracle");
        }

        let num_rounds = oracles[0].num_rounds();
        let degree_bound = oracles[0].degree_bound();
        for (idx, o) in oracles.iter().enumerate().skip(1) {
            if o.num_rounds() != num_rounds {
                panic!(
                    "SumRoundOracle num_rounds mismatch at idx={idx} (got {}, expected {num_rounds})",
                    o.num_rounds()
                );
            }
            if o.degree_bound() != degree_bound {
                panic!(
                    "SumRoundOracle degree_bound mismatch at idx={idx} (got {}, expected {degree_bound})",
                    o.degree_bound()
                );
            }
        }

        Self {
            oracles,
            num_rounds,
            degree_bound,
        }
    }
}

impl RoundOracle for SumRoundOracle {
    fn evals_at(&mut self, points: &[K]) -> Vec<K> {
        let mut acc = vec![K::ZERO; points.len()];
        for o in self.oracles.iter_mut() {
            let ys = o.evals_at(points);
            if ys.len() != acc.len() {
                panic!(
                    "SumRoundOracle eval length mismatch (got {}, expected {})",
                    ys.len(),
                    acc.len()
                );
            }
            for (a, y) in acc.iter_mut().zip(ys) {
                *a += y;
            }
        }
        acc
    }

    fn num_rounds(&self) -> usize {
        self.num_rounds
    }

    fn degree_bound(&self) -> usize {
        self.degree_bound
    }

    fn fold(&mut self, r: K) {
        for o in self.oracles.iter_mut() {
            o.fold(r);
        }
        self.num_rounds = self.oracles[0].num_rounds();
    }
}

/// A trivial RoundOracle that evaluates to zero everywhere.
///
/// Used to keep Shout addr-pre proofs fixed-count (one sumcheck per lane) even for inactive lanes.
pub(crate) struct ZeroRoundOracle {
    num_rounds: usize,
    degree_bound: usize,
}

impl ZeroRoundOracle {
    pub(crate) fn new(num_rounds: usize, degree_bound: usize) -> Self {
        Self {
            num_rounds,
            degree_bound,
        }
    }
}

impl RoundOracle for ZeroRoundOracle {
    fn evals_at(&mut self, points: &[K]) -> Vec<K> {
        vec![K::ZERO; points.len()]
    }

    fn num_rounds(&self) -> usize {
        self.num_rounds
    }

    fn degree_bound(&self) -> usize {
        self.degree_bound
    }

    fn fold(&mut self, _r: K) {}
}

fn build_twist_inc_terms_at_r_addr(lanes: &[TwistLaneSparseCols], r_addr: &[K]) -> Vec<(usize, K)> {
    let ell_addr = r_addr.len();
    let mut out: Vec<(usize, K)> = Vec::new();

    for lane in lanes.iter() {
        debug_assert_eq!(lane.wa_bits.len(), ell_addr, "wa_bits len mismatch");
        for &(t, has_w) in lane.has_write.entries() {
            let inc_t = lane.inc_at_write_addr.get(t);
            if has_w == K::ZERO || inc_t == K::ZERO {
                continue;
            }

            let mut eq_addr = K::ONE;
            for (b, col) in lane.wa_bits.iter().enumerate() {
                let bit = col.get(t);
                eq_addr *= eq_bit_affine(bit, r_addr[b]);
            }

            let inc_at_r_addr = has_w * inc_t * eq_addr;
            if inc_at_r_addr != K::ZERO {
                out.push((t, inc_at_r_addr));
            }
        }
    }

    out
}

pub struct RouteAShoutTimeOracles {
    pub lanes: Vec<RouteAShoutTimeLaneOracles>,
    pub bitness: Vec<Box<dyn RoundOracle>>,
    pub ell_addr: usize,
}

pub struct RouteAShoutTimeLaneOracles {
    pub value: Box<dyn RoundOracle>,
    pub value_claim: K,
    pub adapter: Box<dyn RoundOracle>,
    pub adapter_claim: K,
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

pub(crate) struct ShoutAddrPreBatchProverData {
    pub addr_pre: ShoutAddrPreProof<K>,
    pub decoded: Vec<ShoutDecodedColsSparse>,
}

pub struct ShoutAddrPreVerifyData {
    pub addr_claim_sum: K,
    pub addr_final: K,
    pub r_addr: Vec<K>,
    pub table_eval_at_r_addr: K,
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
pub struct TwistTimeLaneOpeningsLane {
    pub wa_bits: Vec<K>,
    pub has_write: K,
    pub inc_at_write_addr: K,
}

#[derive(Clone, Debug)]
pub struct TwistTimeLaneOpenings {
    pub lanes: Vec<TwistTimeLaneOpeningsLane>,
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
        let expected_lanes = mem_inst.lanes.max(1);
        let twist_inst_cols = bus.twist_cols.get(idx).ok_or_else(|| {
            PiCcsError::InvalidInput(format!(
                "shared_cpu_bus layout mismatch: missing twist_cols for mem_idx={idx}"
            ))
        })?;
        if twist_inst_cols.lanes.len() != expected_lanes {
            return Err(PiCcsError::InvalidInput(format!(
                "shared_cpu_bus layout mismatch at mem_idx={idx}: expected lanes={expected_lanes}, got {}",
                twist_inst_cols.lanes.len()
            )));
        }

        let mut lanes: Vec<TwistLaneSparseCols> = Vec::with_capacity(twist_inst_cols.lanes.len());
        for (lane_idx, twist_cols) in twist_inst_cols.lanes.iter().enumerate() {
            if twist_cols.ra_bits.end - twist_cols.ra_bits.start != ell_addr
                || twist_cols.wa_bits.end - twist_cols.wa_bits.start != ell_addr
            {
                return Err(PiCcsError::InvalidInput(format!(
                    "shared_cpu_bus layout mismatch at mem_idx={idx}, lane={lane_idx}: expected ell_addr={ell_addr}"
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

            lanes.push(TwistLaneSparseCols {
                ra_bits,
                wa_bits,
                has_read,
                has_write,
                wv,
                rv,
                inc_at_write_addr,
            });
        }

        let decoded = TwistDecodedColsSparse { lanes };

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

        let mut read_addr_oracle =
            TwistReadCheckAddrOracleSparseTimeMultiLane::new(init_sparse.clone(), r_cycle, &decoded.lanes);
        let mut write_addr_oracle =
            TwistWriteCheckAddrOracleSparseTimeMultiLane::new(init_sparse, r_cycle, &decoded.lanes);

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
    step_idx: usize,
) -> Result<ShoutAddrPreBatchProverData, PiCcsError> {
    if step.lut_instances.is_empty() {
        return Ok(ShoutAddrPreBatchProverData {
            addr_pre: ShoutAddrPreProof::default(),
            decoded: Vec::new(),
        });
    }

    let bus =
        cpu_bus.ok_or_else(|| PiCcsError::InvalidInput("prove_shout_addr_pre_time requires shared_cpu_bus".into()))?;
    let cpu_z_k = crate::memory_sidecar::cpu_bus::decode_cpu_z_to_k(params, &step.mcs.1.Z);
    if bus.shout_cols.len() != step.lut_instances.len() || bus.twist_cols.len() != step.mem_instances.len() {
        return Err(PiCcsError::InvalidInput(
            "shared_cpu_bus layout mismatch for step (instance counts)".into(),
        ));
    }

    let pow2_cycle = 1usize << ell_n;
    let n_lut = step.lut_instances.len();
    let total_lanes: usize = step
        .lut_instances
        .iter()
        .map(|(inst, _)| inst.lanes.max(1))
        .sum();

    let mut decoded_cols: Vec<ShoutDecodedColsSparse> = Vec::with_capacity(n_lut);
    let mut claimed_sums: Vec<K> = vec![K::ZERO; total_lanes];
    let mut addr_oracles: Vec<Box<dyn RoundOracle>> = Vec::with_capacity(total_lanes);

    let mut ell_addr: Option<usize> = None;
    let mut flat_lane_idx: usize = 0;
    for (idx, (lut_inst, _lut_wit)) in step.lut_instances.iter().enumerate() {
        neo_memory::addr::validate_shout_bit_addressing(lut_inst)?;
        if lut_inst.steps > pow2_cycle {
            return Err(PiCcsError::InvalidInput(format!(
                "Shout(Route A): steps={} exceeds 2^ell_cycle={pow2_cycle}",
                lut_inst.steps
            )));
        }

        let z = &cpu_z_k;
        let inst_ell_addr = lut_inst.d * lut_inst.ell;
        if let Some(prev) = ell_addr {
            if prev != inst_ell_addr {
                return Err(PiCcsError::InvalidInput(format!(
                    "Shout(Route A): batched addr-pre requires uniform ell_addr; got {prev} (lut_idx=0) vs {inst_ell_addr} (lut_idx={idx})"
                )));
            }
        } else {
            ell_addr = Some(inst_ell_addr);
        }
        let inst_cols = bus.shout_cols.get(idx).ok_or_else(|| {
            PiCcsError::InvalidInput(format!(
                "shared_cpu_bus layout mismatch: missing shout_cols for lut_idx={idx}"
            ))
        })?;
        let expected_lanes = lut_inst.lanes.max(1);
        if inst_cols.lanes.len() != expected_lanes {
            return Err(PiCcsError::InvalidInput(format!(
                "shared_cpu_bus layout mismatch at lut_idx={idx}: shout lanes={} but instance expects {}",
                inst_cols.lanes.len(),
                expected_lanes
            )));
        }

        let mut lanes: Vec<ShoutLaneSparseCols> = Vec::with_capacity(expected_lanes);

        for (lane_idx, shout_cols) in inst_cols.lanes.iter().enumerate() {
            if shout_cols.addr_bits.end - shout_cols.addr_bits.start != inst_ell_addr {
                return Err(PiCcsError::InvalidInput(format!(
                    "shared_cpu_bus layout mismatch at lut_idx={idx}, lane_idx={lane_idx}: expected ell_addr={inst_ell_addr}"
                )));
            }

            let mut addr_bits = Vec::with_capacity(inst_ell_addr);
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

            let has_any_lookup = has_lookup.entries().iter().any(|&(_t, gate)| gate != K::ZERO);
            let (addr_oracle, lane_sum): (Box<dyn RoundOracle>, K) = if has_any_lookup {
                match &lut_inst.table_spec {
                    None => {
                        let table_k: Vec<K> = lut_inst.table.iter().map(|&v| v.into()).collect();
                        let (o, sum) =
                            AddressLookupOracle::new(&addr_bits, &has_lookup, &table_k, r_cycle, inst_ell_addr);
                        (Box::new(o), sum)
                    }
                    Some(LutTableSpec::RiscvOpcode { opcode, xlen }) => {
                        let (o, sum) =
                            RiscvAddressLookupOracleSparse::new_sparse_time(*opcode, *xlen, &addr_bits, &has_lookup, r_cycle)?;
                        (Box::new(o), sum)
                    }
                }
            } else {
                (Box::new(ZeroRoundOracle::new(inst_ell_addr, 2)), K::ZERO)
            };

            claimed_sums[flat_lane_idx] = lane_sum;
            addr_oracles.push(addr_oracle);

            lanes.push(ShoutLaneSparseCols {
                addr_bits,
                has_lookup,
                val,
            });
            flat_lane_idx += 1;
        }

        let decoded = ShoutDecodedColsSparse { lanes };

        decoded_cols.push(decoded);
    }
    if flat_lane_idx != total_lanes {
        return Err(PiCcsError::ProtocolError(format!(
            "Shout(Route A): flat lane indexing drift (got {flat_lane_idx}, expected {total_lanes})"
        )));
    }

    let labels_all: Vec<&'static [u8]> = vec![b"shout/addr_pre".as_slice(); total_lanes];
    tr.append_message(b"shout/addr_pre_time/step_idx", &(step_idx as u64).to_le_bytes());
    bind_batched_claim_sums(tr, b"shout/addr_pre_time/claimed_sums", &claimed_sums, &labels_all);

    let ell_addr = ell_addr.unwrap_or(0);
    if addr_oracles.len() != total_lanes {
        return Err(PiCcsError::ProtocolError(format!(
            "Shout(Route A): addr-pre oracle count mismatch (got {}, expected total_lanes={})",
            addr_oracles.len(),
            total_lanes
        )));
    }

    let mut claims: Vec<BatchedClaim<'_>> = addr_oracles
        .iter_mut()
        .zip(claimed_sums.iter())
        .zip(labels_all.iter())
        .map(|((oracle, sum), label)| BatchedClaim {
            oracle: oracle.as_mut(),
            claimed_sum: *sum,
            label: *label,
        })
        .collect();
    let (r_addr, per_claim_results) =
        run_batched_sumcheck_prover_ds(tr, b"shout/addr_pre_time", step_idx, claims.as_mut_slice())?;
    let round_polys = per_claim_results
        .iter()
        .map(|r| r.round_polys.clone())
        .collect::<Vec<_>>();
    if r_addr.len() != ell_addr {
        return Err(PiCcsError::ProtocolError(format!(
            "Shout(Route A): addr-pre r_addr.len()={}, expected ell_addr={ell_addr}",
            r_addr.len()
        )));
    }
    if round_polys.len() != total_lanes {
        return Err(PiCcsError::ProtocolError(format!(
            "Shout(Route A): addr-pre round_polys.len()={}, expected total_lanes={total_lanes}",
            round_polys.len()
        )));
    }

    Ok(ShoutAddrPreBatchProverData {
        addr_pre: ShoutAddrPreProof {
            claimed_sums,
            round_polys,
            r_addr,
        },
        decoded: decoded_cols,
    })
}

pub fn verify_shout_addr_pre_time(
    tr: &mut Poseidon2Transcript,
    step: &StepInstanceBundle<Cmt, F, K>,
    mem_proof: &MemSidecarProof<Cmt, F, K>,
    step_idx: usize,
) -> Result<Vec<ShoutAddrPreVerifyData>, PiCcsError> {
    if step.lut_insts.is_empty() {
        if !mem_proof.shout_addr_pre.claimed_sums.is_empty()
            || !mem_proof.shout_addr_pre.round_polys.is_empty()
            || !mem_proof.shout_addr_pre.r_addr.is_empty()
        {
            return Err(PiCcsError::InvalidInput(
                "shout_addr_pre must be empty when there are no Shout instances".into(),
            ));
        }
        return Ok(Vec::new());
    }

    let total_lanes: usize = step.lut_insts.iter().map(|inst| inst.lanes.max(1)).sum();
    let mut out = Vec::with_capacity(total_lanes);

    let mut ell_addr: Option<usize> = None;
    for (idx, lut_inst) in step.lut_insts.iter().enumerate() {
        neo_memory::addr::validate_shout_bit_addressing(lut_inst)?;
        let inst_ell_addr = lut_inst.d * lut_inst.ell;
        if let Some(prev) = ell_addr {
            if prev != inst_ell_addr {
                return Err(PiCcsError::InvalidInput(format!(
                    "Shout(Route A): batched addr-pre requires uniform ell_addr; got {prev} (lut_idx=0) vs {inst_ell_addr} (lut_idx={idx})"
                )));
            }
        } else {
            ell_addr = Some(inst_ell_addr);
        }
    }
    let ell_addr = ell_addr.unwrap_or(0);

    let proof = &mem_proof.shout_addr_pre;
    if proof.claimed_sums.len() != total_lanes {
        return Err(PiCcsError::InvalidInput(format!(
            "shout_addr_pre claimed_sums.len()={}, expected total_lanes={}",
            proof.claimed_sums.len(),
            total_lanes
        )));
    }
    if proof.round_polys.len() != total_lanes {
        return Err(PiCcsError::InvalidInput(format!(
            "shout_addr_pre round_polys.len()={}, expected total_lanes={total_lanes}",
            proof.round_polys.len()
        )));
    }
    if proof.r_addr.len() != ell_addr {
        return Err(PiCcsError::InvalidInput(format!(
            "shout_addr_pre r_addr.len()={}, expected ell_addr={ell_addr}",
            proof.r_addr.len()
        )));
    }

    for (lane_idx, rounds) in proof.round_polys.iter().enumerate() {
        if rounds.len() != ell_addr {
            return Err(PiCcsError::InvalidInput(format!(
                "shout_addr_pre round_polys[{lane_idx}].len()={}, expected ell_addr={ell_addr}",
                rounds.len()
            )));
        }
    }

    let labels_all: Vec<&'static [u8]> = vec![b"shout/addr_pre".as_slice(); total_lanes];
    tr.append_message(b"shout/addr_pre_time/step_idx", &(step_idx as u64).to_le_bytes());
    bind_batched_claim_sums(tr, b"shout/addr_pre_time/claimed_sums", &proof.claimed_sums, &labels_all);

    let degree_bounds = vec![2usize; total_lanes];
    let (r_addr, finals, ok) = verify_batched_sumcheck_rounds_ds(
        tr,
        b"shout/addr_pre_time",
        step_idx,
        &proof.round_polys,
        &proof.claimed_sums,
        &labels_all,
        &degree_bounds,
    );
    if !ok {
        return Err(PiCcsError::SumcheckError(
            "shout addr-pre batched sumcheck invalid".into(),
        ));
    }
    if r_addr != proof.r_addr {
        return Err(PiCcsError::ProtocolError(
            "shout_addr_pre r_addr mismatch: transcript-derived vs proof".into(),
        ));
    }
    if finals.len() != total_lanes {
        return Err(PiCcsError::ProtocolError(format!(
            "shout addr-pre finals.len()={}, expected total_lanes={total_lanes}",
            finals.len()
        )));
    }

    let mut flat_lane_idx = 0usize;
    for lut_inst in step.lut_insts.iter() {
        let table_eval_at_r_addr = match &lut_inst.table_spec {
            None => {
                let pow2 = 1usize
                    .checked_shl(r_addr.len() as u32)
                    .ok_or_else(|| PiCcsError::InvalidInput("Shout: 2^ell_addr overflow".into()))?;
                let mut acc = K::ZERO;
                for (i, &v) in lut_inst.table.iter().enumerate().take(pow2) {
                    let w = neo_memory::mle::chi_at_index(&r_addr, i);
                    acc += K::from(v) * w;
                }
                acc
            }
            Some(LutTableSpec::RiscvOpcode { opcode, xlen }) => {
                neo_memory::riscv::lookups::evaluate_opcode_mle(*opcode, &r_addr, *xlen)
            }
        };

        let expected_lanes = lut_inst.lanes.max(1);
        for _lane_idx in 0..expected_lanes {
            let addr_claim_sum = *proof
                .claimed_sums
                .get(flat_lane_idx)
                .ok_or_else(|| PiCcsError::ProtocolError("shout addr-pre lane index drift".into()))?;
            let addr_final = *finals
                .get(flat_lane_idx)
                .ok_or_else(|| PiCcsError::ProtocolError("shout addr-pre finals index drift".into()))?;
            out.push(ShoutAddrPreVerifyData {
                addr_claim_sum,
                addr_final,
                r_addr: r_addr.clone(),
                table_eval_at_r_addr,
            });
            flat_lane_idx += 1;
        }
    }
    if flat_lane_idx != total_lanes {
        return Err(PiCcsError::ProtocolError("shout addr-pre lane count mismatch".into()));
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
    shout_pre: &ShoutAddrPreBatchProverData,
    twist_pre: &[TwistAddrPreProverData],
) -> Result<RouteAMemoryOracles, PiCcsError> {
    if shout_pre.decoded.len() != step.lut_instances.len() {
        return Err(PiCcsError::InvalidInput(format!(
            "shout pre-time count mismatch (expected {}, got {})",
            step.lut_instances.len(),
            shout_pre.decoded.len()
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
    let r_addr = &shout_pre.addr_pre.r_addr;
    for (lut_idx, ((lut_inst, _lut_wit), decoded)) in step
        .lut_instances
        .iter()
        .zip(shout_pre.decoded.iter())
        .enumerate()
    {
        let ell_addr = lut_inst.d * lut_inst.ell;
        if r_addr.len() != ell_addr {
            return Err(PiCcsError::InvalidInput(format!(
                "Shout(Route A): r_addr.len()={} != ell_addr={}",
                r_addr.len(),
                ell_addr
            )));
        }

        if decoded.lanes.is_empty() {
            return Err(PiCcsError::InvalidInput(format!(
                "Shout(Route A): decoded lanes empty at lut_idx={lut_idx}"
            )));
        }

        let lane_count = decoded.lanes.len();
        let mut lanes: Vec<RouteAShoutTimeLaneOracles> = Vec::with_capacity(lane_count);

        for lane in decoded.lanes.iter() {
            let (value_oracle, value_claim) =
                ShoutValueOracleSparse::new(r_cycle, lane.has_lookup.clone(), lane.val.clone());

            let (adapter_oracle, adapter_claim) = IndexAdapterOracleSparseTime::new_with_gate(
                r_cycle,
                lane.has_lookup.clone(),
                lane.addr_bits.clone(),
                r_addr,
            );

            lanes.push(RouteAShoutTimeLaneOracles {
                value: Box::new(value_oracle),
                value_claim,
                adapter: Box::new(adapter_oracle),
                adapter_claim,
            });
        }

        let mut bit_cols: Vec<SparseIdxVec<K>> = Vec::with_capacity(lane_count * (ell_addr + 1));
        for lane in decoded.lanes.iter() {
            bit_cols.extend(lane.addr_bits.iter().cloned());
            bit_cols.push(lane.has_lookup.clone());
        }
        let weights = bitness_weights(r_cycle, bit_cols.len(), 0x5348_4F55_54u64 + lut_idx as u64);
        let bitness_oracle = LazyWeightedBitnessOracleSparseTime::new_with_cycle(r_cycle, bit_cols, weights);
        let bitness: Vec<Box<dyn RoundOracle>> = vec![Box::new(bitness_oracle)];

        shout_oracles.push(RouteAShoutTimeOracles {
            lanes,
            bitness,
            ell_addr,
        });
    }

    let mut twist_oracles = Vec::with_capacity(step.mem_instances.len());
    for (mem_idx, ((mem_inst, _mem_wit), pre)) in step.mem_instances.iter().zip(twist_pre.iter()).enumerate() {
        let init_at_r_addr = eval_init_at_r_addr(&mem_inst.init, mem_inst.k, &pre.addr_pre.r_addr)?;
        let ell_addr = mem_inst.d * mem_inst.ell;
        if pre.addr_pre.r_addr.len() != ell_addr {
            return Err(PiCcsError::InvalidInput(format!(
                "Twist(Route A): r_addr.len()={} != ell_addr={}",
                pre.addr_pre.r_addr.len(),
                ell_addr
            )));
        }

        if pre.decoded.lanes.is_empty() {
            return Err(PiCcsError::InvalidInput(format!(
                "Twist(Route A): decoded lanes empty at mem_idx={mem_idx}"
            )));
        }

        let inc_terms_at_r_addr = build_twist_inc_terms_at_r_addr(&pre.decoded.lanes, &pre.addr_pre.r_addr);

        let mut read_oracles: Vec<Box<dyn RoundOracle>> = Vec::with_capacity(pre.decoded.lanes.len());
        let mut write_oracles: Vec<Box<dyn RoundOracle>> = Vec::with_capacity(pre.decoded.lanes.len());
        for lane in pre.decoded.lanes.iter() {
            read_oracles.push(Box::new(TwistReadCheckOracleSparseTime::new_with_inc_terms(
                r_cycle,
                lane.has_read.clone(),
                lane.rv.clone(),
                lane.ra_bits.clone(),
                &pre.addr_pre.r_addr,
                init_at_r_addr,
                inc_terms_at_r_addr.clone(),
            )));
            write_oracles.push(Box::new(TwistWriteCheckOracleSparseTime::new_with_inc_terms(
                r_cycle,
                lane.has_write.clone(),
                lane.wv.clone(),
                lane.inc_at_write_addr.clone(),
                lane.wa_bits.clone(),
                &pre.addr_pre.r_addr,
                init_at_r_addr,
                inc_terms_at_r_addr.clone(),
            )));
        }
        let read_check: Box<dyn RoundOracle> = Box::new(SumRoundOracle::new(read_oracles));
        let write_check: Box<dyn RoundOracle> = Box::new(SumRoundOracle::new(write_oracles));

        let lane_count = pre.decoded.lanes.len();
        let mut bit_cols: Vec<SparseIdxVec<K>> = Vec::with_capacity(lane_count * (2 * ell_addr + 2));
        for lane in pre.decoded.lanes.iter() {
            bit_cols.extend(lane.ra_bits.iter().cloned());
            bit_cols.extend(lane.wa_bits.iter().cloned());
            bit_cols.push(lane.has_read.clone());
            bit_cols.push(lane.has_write.clone());
        }
        let weights = bitness_weights(r_cycle, bit_cols.len(), 0x5457_4953_54u64 + mem_idx as u64);
        let bitness_oracle = LazyWeightedBitnessOracleSparseTime::new_with_cycle(r_cycle, bit_cols, weights);
        let bitness: Vec<Box<dyn RoundOracle>> = vec![Box::new(bitness_oracle)];

        twist_oracles.push(RouteATwistTimeOracles {
            read_check,
            write_check,
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
    pub lane_ranges: Vec<core::ops::Range<usize>>,
    pub lanes: Vec<RouteAShoutTimeLaneClaims<'a>>,
    pub bitness: Vec<Vec<Box<dyn RoundOracle>>>,
}

pub struct RouteAShoutTimeLaneClaims<'a> {
    pub value_prefix: RoundOraclePrefix<'a>,
    pub adapter_prefix: RoundOraclePrefix<'a>,
    pub value_claim: K,
    pub adapter_claim: K,
}

pub fn build_route_a_shout_time_claims_guard<'a>(
    shout_oracles: &'a mut [RouteAShoutTimeOracles],
    ell_n: usize,
) -> RouteAShoutTimeClaimsGuard<'a> {
    let mut lane_ranges: Vec<core::ops::Range<usize>> = Vec::with_capacity(shout_oracles.len());
    let mut lanes: Vec<RouteAShoutTimeLaneClaims<'a>> = Vec::new();
    let mut bitness: Vec<Vec<Box<dyn RoundOracle>>> = Vec::with_capacity(shout_oracles.len());

    for o in shout_oracles.iter_mut() {
        bitness.push(core::mem::take(&mut o.bitness));
        let start = lanes.len();
        for lane in o.lanes.iter_mut() {
            lanes.push(RouteAShoutTimeLaneClaims {
                value_prefix: RoundOraclePrefix::new(lane.value.as_mut(), ell_n),
                adapter_prefix: RoundOraclePrefix::new(lane.adapter.as_mut(), ell_n),
                value_claim: lane.value_claim,
                adapter_claim: lane.adapter_claim,
            });
        }
        let end = lanes.len();
        lane_ranges.push(start..end);
    }

    RouteAShoutTimeClaimsGuard {
        lane_ranges,
        lanes,
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
    if guard.lane_ranges.is_empty() {
        return;
    }
    if guard.bitness.len() != guard.lane_ranges.len() {
        panic!("shout bitness count mismatch");
    }

    let mut lane_ranges_iter = guard.lane_ranges.iter();
    let mut next_end = lane_ranges_iter
        .next()
        .expect("non-empty")
        .end;
    let mut bitness_iter = guard.bitness.iter_mut();

    for (lane_idx, lane) in guard.lanes.iter_mut().enumerate() {
        claimed_sums.push(lane.value_claim);
        degree_bounds.push(lane.value_prefix.degree_bound());
        labels.push(b"shout/value");
        claim_is_dynamic.push(true);
        claims.push(BatchedClaim {
            oracle: &mut lane.value_prefix,
            claimed_sum: lane.value_claim,
            label: b"shout/value",
        });

        claimed_sums.push(lane.adapter_claim);
        degree_bounds.push(lane.adapter_prefix.degree_bound());
        labels.push(b"shout/adapter");
        claim_is_dynamic.push(true);
        claims.push(BatchedClaim {
            oracle: &mut lane.adapter_prefix,
            claimed_sum: lane.adapter_claim,
            label: b"shout/adapter",
        });

        if lane_idx + 1 == next_end {
            let bitness_vec = bitness_iter.next().expect("shout bitness idx drift");
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

            next_end = lane_ranges_iter.next().map(|r| r.end).unwrap_or(usize::MAX);
        }
    }

    if bitness_iter.next().is_some() {
        panic!("shout bitness not fully consumed");
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
    cpu_bus: &BusLayout,
    s: &CcsStructure<F>,
    step: &StepWitnessBundle<Cmt, F, K>,
    prev_step: Option<&StepWitnessBundle<Cmt, F, K>>,
    prev_twist_decoded: Option<&[TwistDecodedColsSparse]>,
    oracles: &mut RouteAMemoryOracles,
    shout_addr_pre: &ShoutAddrPreProof<K>,
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
    let total_lanes: usize = step
        .lut_instances
        .iter()
        .map(|(inst, _)| inst.lanes.max(1))
        .sum();
    if shout_addr_pre.claimed_sums.len() != total_lanes {
        return Err(PiCcsError::InvalidInput(format!(
            "shout addr-pre proof count mismatch (expected claimed_sums.len()=total_lanes={}, got {})",
            total_lanes,
            shout_addr_pre.claimed_sums.len(),
        )));
    }
    if shout_addr_pre.round_polys.len() != total_lanes {
        return Err(PiCcsError::InvalidInput(format!(
            "shout addr-pre round_polys.len()={}, expected total_lanes={total_lanes}",
            shout_addr_pre.round_polys.len(),
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
            if decoded.lanes.is_empty() {
                return Err(PiCcsError::InvalidInput(format!(
                    "Twist(Route A): decoded lanes empty at mem_idx={i_mem}"
                )));
            }

            let mut lt_oracles: Vec<Box<dyn RoundOracle>> = Vec::with_capacity(decoded.lanes.len());
            let mut claimed_inc_sum_lt = K::ZERO;
            for lane in decoded.lanes.iter() {
                let (oracle, claim) = TwistValEvalOracleSparseTime::new(
                    lane.wa_bits.clone(),
                    lane.has_write.clone(),
                    lane.inc_at_write_addr.clone(),
                    r_addr,
                    r_time,
                );
                lt_oracles.push(Box::new(oracle));
                claimed_inc_sum_lt += claim;
            }
            let oracle_lt: Box<dyn RoundOracle> = Box::new(SumRoundOracle::new(lt_oracles));

            let mut total_oracles: Vec<Box<dyn RoundOracle>> = Vec::with_capacity(decoded.lanes.len());
            let mut claimed_inc_sum_total = K::ZERO;
            for lane in decoded.lanes.iter() {
                let (oracle, claim) = TwistTotalIncOracleSparseTime::new(
                    lane.wa_bits.clone(),
                    lane.has_write.clone(),
                    lane.inc_at_write_addr.clone(),
                    r_addr,
                );
                total_oracles.push(Box::new(oracle));
                claimed_inc_sum_total += claim;
            }
            let oracle_total: Box<dyn RoundOracle> = Box::new(SumRoundOracle::new(total_oracles));

            val_oracles.push(oracle_lt);
            bind_claims.push((plan.bind_tags[claim_idx], claimed_inc_sum_lt));
            claimed_sums.push(claimed_inc_sum_lt);
            claim_idx += 1;

            val_oracles.push(oracle_total);
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
                if prev_inst.d != mem_inst.d
                    || prev_inst.ell != mem_inst.ell
                    || prev_inst.k != mem_inst.k
                    || prev_inst.lanes != mem_inst.lanes
                {
                    return Err(PiCcsError::InvalidInput(format!(
                        "Twist rollover requires stable geometry at mem_idx={}: prev (k={}, d={}, ell={}, lanes={}) vs cur (k={}, d={}, ell={}, lanes={})",
                        i_mem,
                        prev_inst.k,
                        prev_inst.d,
                        prev_inst.ell,
                        prev_inst.lanes,
                        mem_inst.k,
                        mem_inst.d,
                        mem_inst.ell,
                        mem_inst.lanes
                    )));
                }
                let prev_decoded = prev_twist_decoded
                    .ok_or_else(|| PiCcsError::ProtocolError("missing prev Twist decoded cols".into()))?
                    .get(i_mem)
                    .ok_or_else(|| PiCcsError::ProtocolError("missing prev Twist decoded cols at mem_idx".into()))?;
                if prev_decoded.lanes.is_empty() {
                    return Err(PiCcsError::ProtocolError(
                        "missing prev Twist decoded cols lanes".into(),
                    ));
                }

                let mut prev_total_oracles: Vec<Box<dyn RoundOracle>> = Vec::with_capacity(prev_decoded.lanes.len());
                let mut claimed_prev_total = K::ZERO;
                for lane in prev_decoded.lanes.iter() {
                    let (oracle, claim) = TwistTotalIncOracleSparseTime::new(
                        lane.wa_bits.clone(),
                        lane.has_write.clone(),
                        lane.inc_at_write_addr.clone(),
                        r_addr,
                    );
                    prev_total_oracles.push(Box::new(oracle));
                    claimed_prev_total += claim;
                }
                let oracle_prev_total: Box<dyn RoundOracle> = Box::new(SumRoundOracle::new(prev_total_oracles));

                val_oracles.push(oracle_prev_total);
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

    if step.lut_instances.is_empty() {
        if !shout_addr_pre.claimed_sums.is_empty()
            || !shout_addr_pre.round_polys.is_empty()
            || !shout_addr_pre.r_addr.is_empty()
        {
            return Err(PiCcsError::ProtocolError(
                "shout_addr_pre must be empty when there are no Shout instances".into(),
            ));
        }
    } else {
        let mut ell_addr: Option<usize> = None;
        for (idx, (inst, _wit)) in step.lut_instances.iter().enumerate() {
            let inst_ell_addr = inst.d * inst.ell;
            if let Some(prev) = ell_addr {
                if prev != inst_ell_addr {
                    return Err(PiCcsError::InvalidInput(format!(
                        "Shout(Route A): batched addr-pre requires uniform ell_addr; got {prev} (lut_idx=0) vs {inst_ell_addr} (lut_idx={idx})"
                    )));
                }
            } else {
                ell_addr = Some(inst_ell_addr);
            }
        }
        let ell_addr = ell_addr.unwrap_or(0);
        if shout_addr_pre.r_addr.len() != ell_addr {
            return Err(PiCcsError::InvalidInput(format!(
                "shout_addr_pre r_addr.len()={}, expected ell_addr={ell_addr}",
                shout_addr_pre.r_addr.len()
            )));
        }
    }

    for _ in 0..step.lut_instances.len() {
        proofs.push(MemOrLutProof::Shout(ShoutProofK::default()));
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
        let core_t = s.t();
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
        let mut cpu_claims_cur = cpu_claims_cur;
        crate::memory_sidecar::cpu_bus::append_bus_openings_to_me_instance(
            params,
            cpu_bus,
            core_t,
            &mcs_wit.Z,
            &mut cpu_claims_cur[0],
        )?;
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
            let mut cpu_claims_prev = cpu_claims_prev;
            crate::memory_sidecar::cpu_bus::append_bus_openings_to_me_instance(
                params,
                cpu_bus,
                core_t,
                &prev_mcs_wit.Z,
                &mut cpu_claims_prev[0],
            )?;
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
        shout_addr_pre: shout_addr_pre.clone(),
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
            if prev_inst.d != inst.d || prev_inst.ell != inst.ell || prev_inst.k != inst.k || prev_inst.lanes != inst.lanes
            {
                return Err(PiCcsError::InvalidInput(format!(
                    "Twist rollover requires stable geometry at mem_idx={}: prev (k={}, d={}, ell={}, lanes={}) vs cur (k={}, d={}, ell={}, lanes={})",
                    idx,
                    prev_inst.k,
                    prev_inst.d,
                    prev_inst.ell,
                    prev_inst.lanes,
                    inst.k,
                    inst.d,
                    inst.ell,
                    inst.lanes
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
    let total_shout_lanes: usize = step.lut_insts.iter().map(|inst| inst.lanes.max(1)).sum();
    if shout_pre.len() != total_shout_lanes {
        return Err(PiCcsError::InvalidInput(format!(
            "shout pre-time count mismatch (expected total_lanes={}, got {})",
            total_shout_lanes,
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
    let mut shout_lane_base: usize = 0;
    for (proof_idx, inst) in step.lut_insts.iter().enumerate() {
        match &proofs_mem[proof_idx] {
            MemOrLutProof::Shout(_proof) => {}
            _ => return Err(PiCcsError::InvalidInput("expected Shout proof".into())),
        }

        let ell_addr = inst.d * inst.ell;
        let expected_lanes = inst.lanes.max(1);

        let inst_cols = cpu_bus
            .shout_cols
            .get(proof_idx)
            .ok_or_else(|| PiCcsError::InvalidInput("shared_cpu_bus layout mismatch (shout)".into()))?;
        if inst_cols.lanes.len() != expected_lanes {
            return Err(PiCcsError::InvalidInput(format!(
                "shared_cpu_bus layout mismatch at lut_idx={proof_idx}: bus shout lanes={} but instance expects {expected_lanes}",
                inst_cols.lanes.len()
            )));
        }

        struct ShoutLaneOpen {
            addr_bits: Vec<K>,
            has_lookup: K,
            val: K,
        }
        let mut lane_opens: Vec<ShoutLaneOpen> = Vec::with_capacity(expected_lanes);
        for (lane_idx, shout_cols) in inst_cols.lanes.iter().enumerate() {
            if shout_cols.addr_bits.end - shout_cols.addr_bits.start != ell_addr {
                return Err(PiCcsError::InvalidInput(format!(
                    "shared_cpu_bus layout mismatch at lut_idx={proof_idx}, lane_idx={lane_idx}: expected ell_addr={ell_addr}"
                )));
            }

            let mut addr_bits_open = Vec::with_capacity(ell_addr);
            for (_j, col_id) in shout_cols.addr_bits.clone().enumerate() {
                addr_bits_open.push(
                    ccs_out0
                        .y_scalars
                        .get(cpu_bus.y_scalar_index(bus_y_base_time, col_id))
                        .copied()
                        .ok_or_else(|| {
                            PiCcsError::ProtocolError("CPU y_scalars missing Shout addr_bits opening".into())
                        })?,
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

            lane_opens.push(ShoutLaneOpen {
                addr_bits: addr_bits_open,
                has_lookup: has_lookup_open,
                val: val_open,
            });
        }

        let shout_claims = claim_plan
            .shout
            .get(proof_idx)
            .ok_or_else(|| PiCcsError::ProtocolError(format!("missing Shout claim schedule at index {}", proof_idx)))?;
        if shout_claims.lanes.len() != expected_lanes {
            return Err(PiCcsError::ProtocolError(format!(
                "Shout claim schedule lane count mismatch at lut_idx={proof_idx}: expected {expected_lanes}, got {}",
                shout_claims.lanes.len()
            )));
        }
        if shout_lane_base
            .checked_add(expected_lanes)
            .ok_or_else(|| PiCcsError::ProtocolError("shout lane index overflow".into()))?
            > shout_pre.len()
        {
            return Err(PiCcsError::ProtocolError(
                "Shout pre-time lane indexing drift".into(),
            ));
        }

        // Route A Shout ordering in batched_time:
        // - value (time rounds only) per lane
        // - adapter (time rounds only) per lane
        // - aggregated bitness for (addr_bits, has_lookup)
        {
            let mut opens: Vec<K> = Vec::with_capacity(expected_lanes * (ell_addr + 1));
            for lane in lane_opens.iter() {
                opens.extend_from_slice(&lane.addr_bits);
                opens.push(lane.has_lookup);
            }
            let weights = bitness_weights(r_cycle, opens.len(), 0x5348_4F55_54u64 + proof_idx as u64);
            let mut acc = K::ZERO;
            for (w, b) in weights.iter().zip(opens.iter()) {
                acc += *w * *b * (*b - K::ONE);
            }
            let expected = chi_cycle_at_r_time * acc;
            if expected != batched_final_values[shout_claims.bitness] {
                return Err(PiCcsError::ProtocolError("shout/bitness terminal value mismatch".into()));
            }
        }

        for (lane_idx, lane) in lane_opens.iter().enumerate() {
            let pre = shout_pre.get(shout_lane_base + lane_idx).ok_or_else(|| {
                PiCcsError::InvalidInput(format!(
                    "missing pre-time Shout lane data at index {}",
                    shout_lane_base + lane_idx
                ))
            })?;
            let lane_claims = shout_claims
                .lanes
                .get(lane_idx)
                .ok_or_else(|| PiCcsError::ProtocolError("shout claim schedule lane idx drift".into()))?;

            let value_claim = batched_claimed_sums[lane_claims.value];
            let value_final = batched_final_values[lane_claims.value];
            let adapter_claim = batched_claimed_sums[lane_claims.adapter];
            let adapter_final = batched_final_values[lane_claims.adapter];

            let expected_value_final = chi_cycle_at_r_time * lane.has_lookup * lane.val;
            if expected_value_final != value_final {
                return Err(PiCcsError::ProtocolError(
                    "shout value terminal value mismatch".into(),
                ));
            }

            let eq_addr = eq_bits_prod(&lane.addr_bits, &pre.r_addr)?;
            let expected_adapter_final = chi_cycle_at_r_time * lane.has_lookup * eq_addr;
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

            let expected_addr_final = pre.table_eval_at_r_addr * adapter_claim;
            if expected_addr_final != pre.addr_final {
                return Err(PiCcsError::ProtocolError(
                    "shout addr terminal value mismatch".into(),
                ));
            }
        }

        shout_lane_base += expected_lanes;
    }
    if shout_lane_base != shout_pre.len() {
        return Err(PiCcsError::ProtocolError(
            "shout pre-time lanes not fully consumed".into(),
        ));
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
        let ell_addr = layout
            .lanes
            .get(0)
            .ok_or_else(|| PiCcsError::InvalidInput("TwistWitnessLayout has no lanes".into()))?
            .ell_addr;

        let twist_inst_cols = cpu_bus
            .twist_cols
            .get(i_mem)
            .ok_or_else(|| PiCcsError::InvalidInput("shared_cpu_bus layout mismatch (twist)".into()))?;
        let expected_lanes = inst.lanes.max(1);
        if twist_inst_cols.lanes.len() != expected_lanes {
            return Err(PiCcsError::InvalidInput(format!(
                "shared_cpu_bus layout mismatch at mem_idx={i_mem}: expected lanes={expected_lanes}, got {}",
                twist_inst_cols.lanes.len()
            )));
        }

        struct TwistLaneTimeOpen {
            ra_bits: Vec<K>,
            wa_bits: Vec<K>,
            has_read: K,
            has_write: K,
            wv: K,
            rv: K,
            inc: K,
        }

        let mut lane_opens: Vec<TwistLaneTimeOpen> = Vec::with_capacity(twist_inst_cols.lanes.len());
        for (lane_idx, twist_cols) in twist_inst_cols.lanes.iter().enumerate() {
            if twist_cols.ra_bits.end - twist_cols.ra_bits.start != ell_addr
                || twist_cols.wa_bits.end - twist_cols.wa_bits.start != ell_addr
            {
                return Err(PiCcsError::InvalidInput(format!(
                    "shared_cpu_bus layout mismatch at mem_idx={i_mem}, lane={lane_idx}: expected ell_addr={ell_addr}"
                )));
            }

            let mut ra_bits_open = Vec::with_capacity(ell_addr);
            for col_id in twist_cols.ra_bits.clone() {
                ra_bits_open.push(
                    ccs_out0
                        .y_scalars
                        .get(cpu_bus.y_scalar_index(bus_y_base_time, col_id))
                        .copied()
                        .ok_or_else(|| {
                            PiCcsError::ProtocolError("CPU y_scalars missing Twist ra_bits opening".into())
                        })?,
                );
            }
            let mut wa_bits_open = Vec::with_capacity(ell_addr);
            for col_id in twist_cols.wa_bits.clone() {
                wa_bits_open.push(
                    ccs_out0
                        .y_scalars
                        .get(cpu_bus.y_scalar_index(bus_y_base_time, col_id))
                        .copied()
                        .ok_or_else(|| {
                            PiCcsError::ProtocolError("CPU y_scalars missing Twist wa_bits opening".into())
                        })?,
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

            lane_opens.push(TwistLaneTimeOpen {
                ra_bits: ra_bits_open,
                wa_bits: wa_bits_open,
                has_read: has_read_open,
                has_write: has_write_open,
                wv: wv_open,
                rv: rv_open,
                inc: inc_write_open,
            });
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

        // Aggregated bitness terminal check (ra_bits, wa_bits, has_read, has_write).
        {
            let mut opens: Vec<K> = Vec::with_capacity(expected_lanes * (2 * ell_addr + 2));
            for lane in lane_opens.iter() {
                opens.extend_from_slice(&lane.ra_bits);
                opens.extend_from_slice(&lane.wa_bits);
                opens.push(lane.has_read);
                opens.push(lane.has_write);
            }
            let weights = bitness_weights(r_cycle, opens.len(), 0x5457_4953_54u64 + i_mem as u64);
            let mut acc = K::ZERO;
            for (w, b) in weights.iter().zip(opens.iter()) {
                acc += *w * *b * (*b - K::ONE);
            }
            let expected = chi_cycle_at_r_time * acc;
            if expected != batched_final_values[twist_claims.bitness] {
                return Err(PiCcsError::ProtocolError("twist/bitness terminal value mismatch".into()));
            }
        }

        let val_eval = twist_proof
            .val_eval
            .as_ref()
            .ok_or_else(|| PiCcsError::InvalidInput("Twist(Route A): missing val_eval proof".into()))?;

        let init_at_r_addr = eval_init_at_r_addr(&inst.init, inst.k, r_addr)?;
        let claimed_val = init_at_r_addr + val_eval.claimed_inc_sum_lt;

        // Terminal checks for read_check / write_check at (r_time, r_addr).
        let mut expected_read_check_final = K::ZERO;
        let mut expected_write_check_final = K::ZERO;
        for lane in lane_opens.iter() {
            let read_eq_addr = eq_bits_prod(&lane.ra_bits, r_addr)?;
            expected_read_check_final += chi_cycle_at_r_time * lane.has_read * (claimed_val - lane.rv) * read_eq_addr;

            let write_eq_addr = eq_bits_prod(&lane.wa_bits, r_addr)?;
            expected_write_check_final +=
                chi_cycle_at_r_time * lane.has_write * (lane.wv - claimed_val - lane.inc) * write_eq_addr;
        }
        if expected_read_check_final != read_check_final {
            return Err(PiCcsError::ProtocolError(
                "twist/read_check terminal value mismatch".into(),
            ));
        }

        if expected_write_check_final != write_check_final {
            return Err(PiCcsError::ProtocolError(
                "twist/write_check terminal value mismatch".into(),
            ));
        }

        twist_time_openings.push(TwistTimeLaneOpenings {
            lanes: lane_opens
                .into_iter()
                .map(|lane| TwistTimeLaneOpeningsLane {
                    wa_bits: lane.wa_bits,
                    has_write: lane.has_write,
                    inc_at_write_addr: lane.inc,
                })
                .collect(),
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
        let ell_addr = layout
            .lanes
            .get(0)
            .ok_or_else(|| PiCcsError::InvalidInput("TwistWitnessLayout has no lanes".into()))?
            .ell_addr;

        let cpu_me_cur =
            cpu_me_val_cur.ok_or_else(|| PiCcsError::ProtocolError("missing CPU ME claim at r_val".into()))?;

        let twist_inst_cols = cpu_bus
            .twist_cols
            .get(i_mem)
            .ok_or_else(|| PiCcsError::InvalidInput("shared_cpu_bus layout mismatch (twist)".into()))?;
        let expected_lanes = inst.lanes.max(1);
        if twist_inst_cols.lanes.len() != expected_lanes {
            return Err(PiCcsError::InvalidInput(format!(
                "shared_cpu_bus layout mismatch at mem_idx={i_mem}: expected lanes={expected_lanes}, got {}",
                twist_inst_cols.lanes.len()
            )));
        }

        let r_addr = twist_pre
            .get(i_mem)
            .ok_or_else(|| PiCcsError::InvalidInput(format!("missing Twist pre-time data at index {}", i_mem)))?
            .r_addr
            .as_slice();

        let mut inc_at_r_addr_val = K::ZERO;
        for (lane_idx, twist_cols) in twist_inst_cols.lanes.iter().enumerate() {
            if twist_cols.wa_bits.end - twist_cols.wa_bits.start != ell_addr {
                return Err(PiCcsError::InvalidInput(format!(
                    "shared_cpu_bus layout mismatch at mem_idx={i_mem}, lane={lane_idx}: expected ell_addr={ell_addr}"
                )));
            }

            let mut wa_bits_val_open = Vec::with_capacity(ell_addr);
            for col_id in twist_cols.wa_bits.clone() {
                wa_bits_val_open.push(
                    cpu_me_cur
                        .y_scalars
                        .get(cpu_bus.y_scalar_index(bus_y_base_val, col_id))
                        .copied()
                        .ok_or_else(|| {
                            PiCcsError::ProtocolError("CPU y_scalars missing wa_bits(val) opening".into())
                        })?,
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

            let eq_wa_val = eq_bits_prod(&wa_bits_val_open, r_addr)?;
            inc_at_r_addr_val += has_write_val_open * inc_at_write_addr_val_open * eq_wa_val;
        }

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
            let mut inc_at_r_addr_prev = K::ZERO;
            for (lane_idx, twist_cols) in twist_inst_cols.lanes.iter().enumerate() {
                if twist_cols.wa_bits.end - twist_cols.wa_bits.start != ell_addr {
                    return Err(PiCcsError::InvalidInput(format!(
                        "shared_cpu_bus layout mismatch at mem_idx={i_mem}, lane={lane_idx}: expected ell_addr={ell_addr}"
                    )));
                }

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
                inc_at_r_addr_prev += has_write_prev_open * inc_prev_open * eq_wa_prev;
            }
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

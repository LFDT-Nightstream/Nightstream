use super::*;

// ============================================================================
// Transcript binding
// ============================================================================

pub(crate) fn bind_shout_table_spec(tr: &mut Poseidon2Transcript, spec: &Option<LutTableSpec>) {
    let Some(spec) = spec else {
        return;
    };

    match spec {
        LutTableSpec::RiscvOpcode { opcode, xlen } => {
            let opcode_id = neo_memory::riscv::lookups::RiscvShoutTables::new(*xlen)
                .opcode_to_id(*opcode)
                .0 as u64;
            tr.append_u64s(
                b"shout/table_spec/meta_u64",
                &[1u64 /* riscv */, opcode_id, *xlen as u64, 0u64],
            );
        }
        LutTableSpec::RiscvOpcodePacked { opcode, xlen } => {
            let opcode_id = neo_memory::riscv::lookups::RiscvShoutTables::new(*xlen)
                .opcode_to_id(*opcode)
                .0 as u64;
            tr.append_u64s(
                b"shout/table_spec/meta_u64",
                &[2u64 /* riscv_packed */, opcode_id, *xlen as u64, 0u64],
            );
        }
        LutTableSpec::RiscvOpcodeEventTablePacked {
            opcode,
            xlen,
            time_bits,
        } => {
            let opcode_id = neo_memory::riscv::lookups::RiscvShoutTables::new(*xlen)
                .opcode_to_id(*opcode)
                .0 as u64;
            tr.append_u64s(
                b"shout/table_spec/meta_u64",
                &[
                    3u64, /* riscv_event_table_packed */
                    opcode_id,
                    *xlen as u64,
                    *time_bits as u64,
                ],
            );
        }
        LutTableSpec::IdentityU32 => {
            tr.append_u64s(b"shout/table_spec/meta_u64", &[4u64 /* identity_u32 */, 0, 0, 0]);
        }
    }
}

#[inline]
fn compute_mem_init_digest(init: &MemInit<F>) -> [u8; 32] {
    match init {
        MemInit::Zero => digest_fields(b"twist/init/zero", &[]),
        MemInit::Sparse(pairs) => {
            let mut fs = Vec::with_capacity(2 * pairs.len());
            for (addr, val) in pairs.iter() {
                fs.push(F::from_u64(*addr));
                fs.push(*val);
            }
            digest_fields(b"twist/init/sparse", &fs)
        }
    }
}

pub(crate) fn absorb_step_memory_impl<'a, LI, MI>(tr: &mut Poseidon2Transcript, mut lut_insts: LI, mut mem_insts: MI)
where
    LI: ExactSizeIterator<Item = &'a LutInstance<Cmt, F>>,
    MI: ExactSizeIterator<Item = &'a MemInstance<Cmt, F>>,
{
    tr.append_message(b"step/absorb_memory_start", &[]);
    tr.append_u64s(b"step/lut_count", &[lut_insts.len() as u64]);
    for (i, inst) in lut_insts.by_ref().enumerate() {
        // Bind public LUT parameters before any challenges.
        tr.append_u64s(
            b"step/lut_meta_u64",
            &[
                i as u64,
                inst.table_id as u64,
                inst.k as u64,
                inst.d as u64,
                inst.n_side as u64,
                inst.steps as u64,
                inst.ell as u64,
                inst.lanes.max(1) as u64,
            ],
        );
        bind_shout_table_spec(tr, &inst.table_spec);
        let table_digest = digest_fields(b"shout/table", &inst.table);
        tr.append_bytes_packed(b"shout/table_digest", &table_digest);

        // Bind commitments so Route-A challenges (r_cycle, addr/time points) are sampled after them.
        tr.append_u64s(b"shout/comms_len", &[inst.comms.len() as u64]);
        if !inst.comms.is_empty() {
            let comm_lens: Vec<u64> = inst
                .comms
                .iter()
                .map(|comm| comm.data.len() as u64)
                .collect();
            tr.append_u64s(b"shout/comm_lens", &comm_lens);
            let total_fields = inst.comms.iter().map(|comm| comm.data.len()).sum::<usize>();
            #[cfg(debug_assertions)]
            {
                let lens_sum: usize = comm_lens.iter().map(|&x| x as usize).sum();
                debug_assert_eq!(lens_sum, total_fields, "shout/comm_lens sum mismatch");
            }
            tr.append_fields_iter(
                b"shout/comm_data_flat",
                total_fields,
                inst.comms.iter().flat_map(|comm| comm.data.iter().copied()),
            );
        }
    }
    tr.append_u64s(b"step/mem_count", &[mem_insts.len() as u64]);
    for (i, inst) in mem_insts.by_ref().enumerate() {
        // Bind public memory parameters before any challenges.
        tr.append_u64s(
            b"step/mem_meta_u64",
            &[
                i as u64,
                inst.mem_id as u64,
                inst.k as u64,
                inst.d as u64,
                inst.n_side as u64,
                inst.steps as u64,
                inst.ell as u64,
                inst.lanes.max(1) as u64,
            ],
        );
        let init_digest = compute_mem_init_digest(&inst.init);
        tr.append_bytes_packed(b"twist/init_digest", &init_digest);

        // Bind commitments so Route-A challenges (r_cycle, addr/time points) are sampled after them.
        tr.append_u64s(b"twist/comms_len", &[inst.comms.len() as u64]);
        if !inst.comms.is_empty() {
            let comm_lens: Vec<u64> = inst
                .comms
                .iter()
                .map(|comm| comm.data.len() as u64)
                .collect();
            tr.append_u64s(b"twist/comm_lens", &comm_lens);
            let total_fields = inst.comms.iter().map(|comm| comm.data.len()).sum::<usize>();
            #[cfg(debug_assertions)]
            {
                let lens_sum: usize = comm_lens.iter().map(|&x| x as usize).sum();
                debug_assert_eq!(lens_sum, total_fields, "twist/comm_lens sum mismatch");
            }
            tr.append_fields_iter(
                b"twist/comm_data_flat",
                total_fields,
                inst.comms.iter().flat_map(|comm| comm.data.iter().copied()),
            );
        }
    }
    tr.append_message(b"step/absorb_memory_done", &[]);
}

pub fn absorb_step_memory(tr: &mut Poseidon2Transcript, step: &StepInstanceBundle<Cmt, F, K>) {
    absorb_step_memory_impl(tr, step.lut_insts.iter(), step.mem_insts.iter());
}

pub(crate) fn absorb_step_memory_witness(tr: &mut Poseidon2Transcript, step: &StepWitnessBundle<Cmt, F, K>) {
    tr.append_message(b"step/absorb_memory_start", &[]);
    tr.append_u64s(b"step/lut_count", &[step.lut_instances.len() as u64]);
    for (i, (inst, _)) in step.lut_instances.iter().enumerate() {
        tr.append_u64s(
            b"step/lut_meta_u64",
            &[
                i as u64,
                inst.table_id as u64,
                inst.k as u64,
                inst.d as u64,
                inst.n_side as u64,
                inst.steps as u64,
                inst.ell as u64,
                inst.lanes.max(1) as u64,
            ],
        );
        bind_shout_table_spec(tr, &inst.table_spec);
        let table_digest = match inst.table_digest {
            Some(d) => {
                #[cfg(debug_assertions)]
                {
                    let computed = digest_fields(b"shout/table", &inst.table);
                    debug_assert_eq!(d, computed, "LutInstance.table_digest mismatch");
                }
                d
            }
            None => digest_fields(b"shout/table", &inst.table),
        };
        tr.append_bytes_packed(b"shout/table_digest", &table_digest);

        tr.append_u64s(b"shout/comms_len", &[inst.comms.len() as u64]);
        if !inst.comms.is_empty() {
            let comm_lens: Vec<u64> = inst
                .comms
                .iter()
                .map(|comm| comm.data.len() as u64)
                .collect();
            tr.append_u64s(b"shout/comm_lens", &comm_lens);
            let total_fields = inst.comms.iter().map(|comm| comm.data.len()).sum::<usize>();
            #[cfg(debug_assertions)]
            {
                let lens_sum: usize = comm_lens.iter().map(|&x| x as usize).sum();
                debug_assert_eq!(lens_sum, total_fields, "shout/comm_lens sum mismatch");
            }
            tr.append_fields_iter(
                b"shout/comm_data_flat",
                total_fields,
                inst.comms.iter().flat_map(|comm| comm.data.iter().copied()),
            );
        }
    }

    tr.append_u64s(b"step/mem_count", &[step.mem_instances.len() as u64]);
    for (i, (inst, _)) in step.mem_instances.iter().enumerate() {
        tr.append_u64s(
            b"step/mem_meta_u64",
            &[
                i as u64,
                inst.mem_id as u64,
                inst.k as u64,
                inst.d as u64,
                inst.n_side as u64,
                inst.steps as u64,
                inst.ell as u64,
                inst.lanes.max(1) as u64,
            ],
        );
        let init_digest = match inst.init_digest {
            Some(d) => {
                #[cfg(debug_assertions)]
                {
                    let computed = compute_mem_init_digest(&inst.init);
                    debug_assert_eq!(d, computed, "MemInstance.init_digest mismatch");
                }
                d
            }
            None => compute_mem_init_digest(&inst.init),
        };
        tr.append_bytes_packed(b"twist/init_digest", &init_digest);

        tr.append_u64s(b"twist/comms_len", &[inst.comms.len() as u64]);
        if !inst.comms.is_empty() {
            let comm_lens: Vec<u64> = inst
                .comms
                .iter()
                .map(|comm| comm.data.len() as u64)
                .collect();
            tr.append_u64s(b"twist/comm_lens", &comm_lens);
            let total_fields = inst.comms.iter().map(|comm| comm.data.len()).sum::<usize>();
            #[cfg(debug_assertions)]
            {
                let lens_sum: usize = comm_lens.iter().map(|&x| x as usize).sum();
                debug_assert_eq!(lens_sum, total_fields, "twist/comm_lens sum mismatch");
            }
            tr.append_fields_iter(
                b"twist/comm_data_flat",
                total_fields,
                inst.comms.iter().flat_map(|comm| comm.data.iter().copied()),
            );
        }
    }
    tr.append_message(b"step/absorb_memory_done", &[]);
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Rv32PackedShoutOp {
    And,
    Andn,
    Add,
    Or,
    Sub,
    Xor,
    Eq,
    Neq,
    Slt,
    Sll,
    Srl,
    Sra,
    Sltu,
    Mul,
    Mulh,
    Mulhu,
    Mulhsu,
    Div,
    Divu,
    Rem,
    Remu,
}

pub(crate) fn rv32_packed_shout_layout(
    spec: &Option<LutTableSpec>,
) -> Result<Option<(Rv32PackedShoutOp, usize, usize)>, PiCcsError> {
    let (opcode, xlen, time_bits) = match spec {
        Some(LutTableSpec::RiscvOpcodePacked { opcode, xlen }) => (*opcode, *xlen, 0usize),
        Some(LutTableSpec::RiscvOpcodeEventTablePacked {
            opcode,
            xlen,
            time_bits,
        }) => (*opcode, *xlen, *time_bits),
        _ => return Ok(None),
    };

    if !matches!(xlen, 32 | 64) {
        return Err(PiCcsError::InvalidInput(format!(
            "packed RISC-V Shout requires xlen=32 or 64 in Route A (got xlen={xlen})"
        )));
    }
    if time_bits == 0 {
        // `RiscvOpcodePacked` uses `time_bits=0` (no prefix). Event-table packed must be >= 1.
        if matches!(spec, Some(LutTableSpec::RiscvOpcodeEventTablePacked { .. })) {
            return Err(PiCcsError::InvalidInput(
                "RiscvOpcodeEventTablePacked requires time_bits >= 1".into(),
            ));
        }
    } else if xlen != 32 {
        return Err(PiCcsError::InvalidInput(
            "packed RV64 Shout does not support event-table mode in Route A".into(),
        ));
    }

    let op =
        match opcode {
            neo_memory::riscv::lookups::RiscvOpcode::And => Rv32PackedShoutOp::And,
            neo_memory::riscv::lookups::RiscvOpcode::Andn => Rv32PackedShoutOp::Andn,
            neo_memory::riscv::lookups::RiscvOpcode::Add => Rv32PackedShoutOp::Add,
            neo_memory::riscv::lookups::RiscvOpcode::Or => Rv32PackedShoutOp::Or,
            neo_memory::riscv::lookups::RiscvOpcode::Sub => Rv32PackedShoutOp::Sub,
            neo_memory::riscv::lookups::RiscvOpcode::Xor => Rv32PackedShoutOp::Xor,
            neo_memory::riscv::lookups::RiscvOpcode::Eq => Rv32PackedShoutOp::Eq,
            neo_memory::riscv::lookups::RiscvOpcode::Neq => Rv32PackedShoutOp::Neq,
            neo_memory::riscv::lookups::RiscvOpcode::Slt => Rv32PackedShoutOp::Slt,
            neo_memory::riscv::lookups::RiscvOpcode::Sll => Rv32PackedShoutOp::Sll,
            neo_memory::riscv::lookups::RiscvOpcode::Srl => Rv32PackedShoutOp::Srl,
            neo_memory::riscv::lookups::RiscvOpcode::Sra
            | neo_memory::riscv::lookups::RiscvOpcode::VirtualMovsignWord => Rv32PackedShoutOp::Sra,
            neo_memory::riscv::lookups::RiscvOpcode::Sltu => Rv32PackedShoutOp::Sltu,
            neo_memory::riscv::lookups::RiscvOpcode::Mul | neo_memory::riscv::lookups::RiscvOpcode::VirtualMulWord => {
                Rv32PackedShoutOp::Mul
            }
            neo_memory::riscv::lookups::RiscvOpcode::Mulh => Rv32PackedShoutOp::Mulh,
            neo_memory::riscv::lookups::RiscvOpcode::Mulhu => Rv32PackedShoutOp::Mulhu,
            neo_memory::riscv::lookups::RiscvOpcode::Mulhsu => Rv32PackedShoutOp::Mulhsu,
            neo_memory::riscv::lookups::RiscvOpcode::Div | neo_memory::riscv::lookups::RiscvOpcode::VirtualDivWord => {
                Rv32PackedShoutOp::Div
            }
            neo_memory::riscv::lookups::RiscvOpcode::Divu
            | neo_memory::riscv::lookups::RiscvOpcode::VirtualDivuWord => Rv32PackedShoutOp::Divu,
            neo_memory::riscv::lookups::RiscvOpcode::Rem | neo_memory::riscv::lookups::RiscvOpcode::VirtualRemWord => {
                Rv32PackedShoutOp::Rem
            }
            neo_memory::riscv::lookups::RiscvOpcode::Remu
            | neo_memory::riscv::lookups::RiscvOpcode::VirtualRemuWord => Rv32PackedShoutOp::Remu,
            _ => {
                return Err(PiCcsError::InvalidInput(format!(
                    "packed RISC-V Shout is unsupported in Route A for opcode={opcode:?}, xlen={xlen}"
                )));
            }
        };

    if xlen == 64
        && !matches!(
            opcode,
            neo_memory::riscv::lookups::RiscvOpcode::Mul | neo_memory::riscv::lookups::RiscvOpcode::Mulhu
        )
    {
        return Err(PiCcsError::InvalidInput(format!(
            "packed RV64 Shout is currently only supported for Mul/Mulhu in Route A (got opcode={opcode:?})"
        )));
    }

    Ok(Some((op, xlen, time_bits)))
}

pub(crate) fn rv32_shout_table_id_from_spec(spec: &Option<LutTableSpec>) -> Result<u32, PiCcsError> {
    let (opcode, xlen) = match spec {
        Some(LutTableSpec::RiscvOpcode { opcode, xlen }) => (*opcode, *xlen),
        Some(LutTableSpec::RiscvOpcodePacked { opcode, xlen }) => (*opcode, *xlen),
        Some(LutTableSpec::RiscvOpcodeEventTablePacked { opcode, xlen, .. }) => (*opcode, *xlen),
        Some(LutTableSpec::IdentityU32) => {
            return Err(PiCcsError::InvalidInput(
                "trace linkage expects RISC-V shout table specs (IdentityU32 is unsupported)".into(),
            ));
        }
        None => {
            return Err(PiCcsError::InvalidInput(
                "trace linkage requires LutTableSpec on Shout instances".into(),
            ));
        }
    };

    if !matches!(xlen, 32 | 64) {
        return Err(PiCcsError::InvalidInput(format!(
            "trace linkage expects RISC-V shout specs with xlen=32 or 64 (got xlen={xlen})"
        )));
    }
    Ok(neo_memory::riscv::lookups::RiscvShoutTables::new(xlen)
        .opcode_to_id(opcode)
        .0)
}

pub(crate) fn rv32_trace_link_table_id_from_spec(spec: &Option<LutTableSpec>) -> Result<Option<u32>, PiCcsError> {
    match spec {
        Some(LutTableSpec::RiscvOpcode { .. })
        | Some(LutTableSpec::RiscvOpcodePacked { .. })
        | Some(LutTableSpec::RiscvOpcodeEventTablePacked { .. }) => Ok(Some(rv32_shout_table_id_from_spec(spec)?)),
        Some(LutTableSpec::IdentityU32) | None => Ok(None),
    }
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
    oracles: Vec<Box<dyn RoundOracle + Send>>,
    num_rounds: usize,
    degree_bound: usize,
}

impl SumRoundOracle {
    pub(crate) fn new(oracles: Vec<Box<dyn RoundOracle + Send>>) -> Result<Self, PiCcsError> {
        if oracles.is_empty() {
            return Err(PiCcsError::ProtocolError(
                "SumRoundOracle requires at least one oracle".into(),
            ));
        }

        let num_rounds = oracles[0].num_rounds();
        let degree_bound = oracles[0].degree_bound();
        for (idx, o) in oracles.iter().enumerate().skip(1) {
            if o.num_rounds() != num_rounds {
                return Err(PiCcsError::ProtocolError(format!(
                    "SumRoundOracle num_rounds mismatch at idx={idx} (got {}, expected {num_rounds})",
                    o.num_rounds()
                )));
            }
            if o.degree_bound() != degree_bound {
                return Err(PiCcsError::ProtocolError(format!(
                    "SumRoundOracle degree_bound mismatch at idx={idx} (got {}, expected {degree_bound})",
                    o.degree_bound()
                )));
            }
        }

        Ok(Self {
            oracles,
            num_rounds,
            degree_bound,
        })
    }
}

impl RoundOracle for SumRoundOracle {
    fn evals_at(&mut self, points: &[K]) -> Vec<K> {
        let mut acc = vec![K::ZERO; points.len()];
        for o in self.oracles.iter_mut() {
            let ys = o.evals_at(points);
            if ys.len() != acc.len() {
                // Non-fatal hardening: avoid panics on malformed oracle outputs.
                // Mismatched lengths are treated as sparse truncation/zero-padding.
                let n = core::cmp::min(ys.len(), acc.len());
                for i in 0..n {
                    acc[i] += ys[i];
                }
                continue;
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

#[inline]
pub(crate) fn interp(a0: K, a1: K, x: K) -> K {
    a0 + (a1 - a0) * x
}

pub(crate) fn log2_pow2(n: usize) -> usize {
    if n == 0 {
        return 0;
    }
    debug_assert!(n.is_power_of_two(), "expected power of two, got {n}");
    n.trailing_zeros() as usize
}

pub(crate) fn gather_pairs_from_sparse(entries: &[(usize, K)]) -> Vec<usize> {
    let mut out: Vec<usize> = Vec::with_capacity(entries.len());
    let mut prev: Option<usize> = None;
    for &(idx, _v) in entries {
        let p = idx >> 1;
        if prev != Some(p) {
            out.push(p);
            prev = Some(p);
        }
    }
    out
}

/// Sparse time-domain oracle for event-table RV32 Shout hash linkage:
///   Σ_t has_lookup(t) · (1 + α·val(t) + β·lhs(t) + γ·rhs(t)) · Π_b eq(time_bit_b(t), r_addr_b)
///
/// Intended usage:
/// - `time_bit_b(t)` encodes the original cycle index of event row `t` (little-endian).
/// - `r_addr` is set to `r_cycle` so the claim is an MLE evaluation over cycle indices.
pub(crate) struct ShoutEventTableHashOracleSparseTime {
    degree_bound: usize,
    r_addr: Vec<K>,

    time_bits: Vec<SparseIdxVec<K>>,
    has_lookup: SparseIdxVec<K>,
    val: SparseIdxVec<K>,
    lhs: SparseIdxVec<K>,
    rhs_terms: Vec<(SparseIdxVec<K>, K)>,

    alpha: K,
    beta: K,
    gamma: K,
}

impl ShoutEventTableHashOracleSparseTime {
    pub(crate) fn new(
        r_addr: &[K],
        time_bits: Vec<SparseIdxVec<K>>,
        has_lookup: SparseIdxVec<K>,
        val: SparseIdxVec<K>,
        lhs: SparseIdxVec<K>,
        rhs_terms: Vec<(SparseIdxVec<K>, K)>,
        alpha: K,
        beta: K,
        gamma: K,
    ) -> (Self, K) {
        let ell_n = log2_pow2(has_lookup.len());
        debug_assert_eq!(val.len(), 1usize << ell_n);
        debug_assert_eq!(lhs.len(), 1usize << ell_n);
        for (i, col) in time_bits.iter().enumerate() {
            debug_assert_eq!(col.len(), 1usize << ell_n, "time_bits[{i}] length mismatch");
        }
        for (i, (col, _w)) in rhs_terms.iter().enumerate() {
            debug_assert_eq!(col.len(), 1usize << ell_n, "rhs_terms[{i}] length mismatch");
        }
        debug_assert_eq!(time_bits.len(), r_addr.len(), "time_bits/r_addr length mismatch");

        let mut claim = K::ZERO;
        for &(t, gate) in has_lookup.entries() {
            if gate == K::ZERO {
                continue;
            }

            let v_t = val.get(t);
            let lhs_t = lhs.get(t);
            let mut rhs_t = K::ZERO;
            for (col, w) in rhs_terms.iter() {
                rhs_t += *w * col.get(t);
            }

            let hash_t = K::ONE + alpha * v_t + beta * lhs_t + gamma * rhs_t;
            if hash_t == K::ZERO {
                continue;
            }

            let mut eq_addr = K::ONE;
            for (b, col) in time_bits.iter().enumerate() {
                eq_addr *= eq_bit_affine(col.get(t), r_addr[b]);
            }

            claim += gate * hash_t * eq_addr;
        }

        (
            Self {
                degree_bound: 2 + r_addr.len(),
                r_addr: r_addr.to_vec(),
                time_bits,
                has_lookup,
                val,
                lhs,
                rhs_terms,
                alpha,
                beta,
                gamma,
            },
            claim,
        )
    }
}

impl RoundOracle for ShoutEventTableHashOracleSparseTime {
    fn evals_at(&mut self, points: &[K]) -> Vec<K> {
        if self.has_lookup.len() == 1 {
            let gate = self.has_lookup.singleton_value();
            let v = self.val.singleton_value();
            let lhs = self.lhs.singleton_value();
            let mut rhs = K::ZERO;
            for (col, w) in self.rhs_terms.iter() {
                rhs += *w * col.singleton_value();
            }
            let hash = gate * (K::ONE + self.alpha * v + self.beta * lhs + self.gamma * rhs);

            let mut eq_addr = K::ONE;
            for (b, col) in self.time_bits.iter().enumerate() {
                eq_addr *= eq_bit_affine(col.singleton_value(), self.r_addr[b]);
            }

            let out = hash * eq_addr;
            return vec![out; points.len()];
        }

        let pairs = gather_pairs_from_sparse(self.has_lookup.entries());
        let half = self.has_lookup.len() / 2;
        debug_assert!(pairs.iter().all(|&p| p < half));

        let mut ys = vec![K::ZERO; points.len()];
        for &pair in pairs.iter() {
            let child0 = 2 * pair;
            let child1 = child0 + 1;

            let gate0 = self.has_lookup.get(child0);
            let gate1 = self.has_lookup.get(child1);
            if gate0 == K::ZERO && gate1 == K::ZERO {
                continue;
            }

            let v0 = self.val.get(child0);
            let v1 = self.val.get(child1);
            let lhs0 = self.lhs.get(child0);
            let lhs1 = self.lhs.get(child1);

            let mut rhs0 = K::ZERO;
            let mut rhs1 = K::ZERO;
            for (col, w) in self.rhs_terms.iter() {
                rhs0 += *w * col.get(child0);
                rhs1 += *w * col.get(child1);
            }

            let mut eq0s: Vec<K> = Vec::with_capacity(self.time_bits.len());
            let mut d_eqs: Vec<K> = Vec::with_capacity(self.time_bits.len());
            for (b, col) in self.time_bits.iter().enumerate() {
                let e0 = eq_bit_affine(col.get(child0), self.r_addr[b]);
                let e1 = eq_bit_affine(col.get(child1), self.r_addr[b]);
                eq0s.push(e0);
                d_eqs.push(e1 - e0);
            }

            for (i, &x) in points.iter().enumerate() {
                let gate_x = interp(gate0, gate1, x);
                if gate_x == K::ZERO {
                    continue;
                }
                let v_x = interp(v0, v1, x);
                let lhs_x = interp(lhs0, lhs1, x);
                let rhs_x = interp(rhs0, rhs1, x);

                let mut prod = gate_x * (K::ONE + self.alpha * v_x + self.beta * lhs_x + self.gamma * rhs_x);
                for (e0, de) in eq0s.iter().zip(d_eqs.iter()) {
                    prod *= *e0 + *de * x;
                }
                ys[i] += prod;
            }
        }

        ys
    }

    fn num_rounds(&self) -> usize {
        log2_pow2(self.has_lookup.len())
    }

    fn degree_bound(&self) -> usize {
        self.degree_bound
    }

    fn fold(&mut self, r: K) {
        if self.num_rounds() == 0 {
            return;
        }
        self.has_lookup.fold_round_in_place(r);
        self.val.fold_round_in_place(r);
        self.lhs.fold_round_in_place(r);
        for (col, _w) in self.rhs_terms.iter_mut() {
            col.fold_round_in_place(r);
        }
        for col in self.time_bits.iter_mut() {
            col.fold_round_in_place(r);
        }
    }
}

pub(crate) fn build_twist_inc_terms_at_r_addr(lanes: &[TwistLaneSparseCols], r_addr: &[K]) -> Vec<(usize, K)> {
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
    pub bitness: Vec<Box<dyn RoundOracle + Send>>,
}

pub struct RouteAShoutTimeLaneOracles {
    pub value: Box<dyn RoundOracle + Send>,
    pub value_claim: K,
    pub adapter: Box<dyn RoundOracle + Send>,
    pub adapter_claim: K,
    pub event_table_hash: Option<Box<dyn RoundOracle + Send>>,
    pub event_table_hash_claim: Option<K>,
    pub gamma_group: Option<usize>,
    pub transport_only: bool,
}

pub struct RouteAShoutGammaGroupOracles {
    pub value: Box<dyn RoundOracle + Send>,
    pub value_claim: K,
    pub adapter: Box<dyn RoundOracle + Send>,
    pub adapter_claim: K,
    pub bitness: Box<dyn RoundOracle + Send>,
}

pub struct RouteATwistTimeOracles {
    pub read_check: Box<dyn RoundOracle + Send>,
    pub write_check: Box<dyn RoundOracle + Send>,
    pub bitness: Vec<Box<dyn RoundOracle + Send>>,
    pub virtual_write_domain: Option<Box<dyn RoundOracle + Send>>,
    pub nonvirtual_arch_domain: Option<Box<dyn RoundOracle + Send>>,
}

pub struct RouteAMemoryOracles {
    pub shout: Vec<RouteAShoutTimeOracles>,
    pub shout_gamma_groups: Vec<RouteAShoutGammaGroupOracles>,
    pub shout_event_trace_hash: Option<RouteAShoutEventTraceHashOracle>,
    pub twist: Vec<RouteATwistTimeOracles>,
}

pub struct RouteAShoutEventTraceHashOracle {
    pub oracle: Box<dyn RoundOracle + Send>,
    pub claim: K,
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
    ) -> Result<(), PiCcsError>;
}

pub(crate) struct ShoutAddrPreBatchProverData {
    pub addr_pre: ShoutAddrPreProof<K>,
    pub decoded: Vec<ShoutDecodedColsSparse>,
}

#[derive(Clone, Debug)]
pub struct ShoutAddrPreVerifyData {
    pub is_active: bool,
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
    pub has_read: K,
    pub wa_bits: Vec<K>,
    pub has_write: K,
    pub wv: K,
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

#[derive(Clone, Copy)]
pub(crate) struct TraceCpuLinkOpenings {
    pub(crate) shout_has_lookup: K,
    pub(crate) shout_val: K,
    pub(crate) shout_link_lhs: K,
    pub(crate) shout_link_rhs: K,
    pub(crate) shout_add_sub_key: K,
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct ShoutTraceLinkSums {
    pub(crate) has_lookup: K,
    pub(crate) val: K,
    pub(crate) link_lhs: K,
    pub(crate) link_rhs: K,
    pub(crate) add_sub_key: K,
    pub(crate) table_id: K,
}

#[inline]
pub(crate) fn verify_non_event_trace_shout_linkage(
    cpu: TraceCpuLinkOpenings,
    sums: ShoutTraceLinkSums,
    expected_table_id: Option<K>,
) -> Result<(), PiCcsError> {
    if sums.has_lookup != cpu.shout_has_lookup {
        return Err(PiCcsError::ProtocolError(
            "trace linkage failed: Shout has_lookup mismatch".into(),
        ));
    }
    if sums.val != cpu.shout_val {
        return Err(PiCcsError::ProtocolError(
            "trace linkage failed: Shout val mismatch".into(),
        ));
    }
    if sums.link_lhs != cpu.shout_link_lhs {
        return Err(PiCcsError::ProtocolError(format!(
            "trace linkage failed: Shout lhs mismatch (sums={}, cpu={})",
            sums.link_lhs, cpu.shout_link_lhs
        )));
    }
    if sums.link_rhs != cpu.shout_link_rhs {
        return Err(PiCcsError::ProtocolError(format!(
            "trace linkage failed: Shout rhs mismatch (sums={}, cpu={})",
            sums.link_rhs, cpu.shout_link_rhs
        )));
    }
    if sums.add_sub_key != cpu.shout_add_sub_key {
        return Err(PiCcsError::ProtocolError(format!(
            "trace linkage failed: Shout add/sub key mismatch (sums={}, cpu={})",
            sums.add_sub_key, cpu.shout_add_sub_key
        )));
    }
    if let Some(expected_table_id) = expected_table_id {
        if sums.table_id != expected_table_id {
            return Err(PiCcsError::ProtocolError(
                "trace linkage failed: Shout table_id mismatch".into(),
            ));
        }
    }
    Ok(())
}

#[inline]
pub(crate) fn eq_single_k(a: K, b: K) -> K {
    a * b + (K::ONE - a) * (K::ONE - b)
}

pub(crate) fn chi_cycle_children(r_cycle: &[K], bit_idx: usize, prefix_eq: K, pair_idx: usize) -> (K, K) {
    let mut suffix = K::ONE;
    let mut shift = 0usize;
    for b in (bit_idx + 1)..r_cycle.len() {
        let bit = (pair_idx >> shift) & 1;
        suffix *= if bit == 1 { r_cycle[b] } else { K::ONE - r_cycle[b] };
        shift += 1;
    }

    let r = r_cycle[bit_idx];
    let child0 = prefix_eq * (K::ONE - r) * suffix;
    let child1 = prefix_eq * r * suffix;
    (child0, child1)
}

#[inline]
pub(crate) fn wb_weight_vector(r_cycle: &[K], len: usize) -> Vec<K> {
    bitness_weights(r_cycle, len, 0x5742_5F42_4F4F_4Cu64)
}

#[inline]
pub(crate) fn w2_decode_pack_weight_vector(r_cycle: &[K], len: usize) -> Vec<K> {
    bitness_weights(r_cycle, len, 0x5732_5F50_4143_4Bu64)
}

#[inline]
pub(crate) fn w2_decode_imm_weight_vector(r_cycle: &[K], len: usize) -> Vec<K> {
    bitness_weights(r_cycle, len, 0x5732_5F49_4D4D_214Du64)
}

#[inline]
pub(crate) fn w3_bitness_weight_vector(r_cycle: &[K], len: usize) -> Vec<K> {
    bitness_weights(r_cycle, len, 0x5733_5F42_4954_2144u64)
}

#[inline]
pub(crate) fn w3_quiescence_weight_vector(r_cycle: &[K], len: usize) -> Vec<K> {
    bitness_weights(r_cycle, len, 0x5733_5F51_5549_4553u64)
}

#[inline]
pub(crate) fn w3_load_weight_vector(r_cycle: &[K], len: usize) -> Vec<K> {
    bitness_weights(r_cycle, len, 0x5733_5F4C_4F41_4421u64)
}

#[inline]
pub(crate) fn w3_store_weight_vector(r_cycle: &[K], len: usize) -> Vec<K> {
    bitness_weights(r_cycle, len, 0x5733_5F53_544F_5245u64)
}

#[inline]
pub(crate) fn control_next_pc_linear_weight_vector(r_cycle: &[K], len: usize) -> Vec<K> {
    bitness_weights(r_cycle, len, 0x4354_524C_4E50_434Cu64)
}

#[inline]
pub(crate) fn control_next_pc_control_weight_vector(r_cycle: &[K], len: usize) -> Vec<K> {
    bitness_weights(r_cycle, len, 0x4354_524C_4E50_4343u64)
}

#[inline]
pub(crate) fn control_branch_semantics_weight_vector(r_cycle: &[K], len: usize) -> Vec<K> {
    bitness_weights(r_cycle, len, 0x4354_524C_4252_534Du64)
}

#[inline]
pub(crate) fn control_writeback_weight_vector(r_cycle: &[K], len: usize) -> Vec<K> {
    bitness_weights(r_cycle, len, 0x4354_524C_5752_4255u64)
}

#[inline]
pub(crate) fn wp_weight_vector(r_cycle: &[K], len: usize) -> Vec<K> {
    bitness_weights(r_cycle, len, 0x5750_5F51_5549_4553u64)
}

pub(crate) fn rv32_trace_wb_columns(layout: &Rv32TraceLayout) -> Vec<usize> {
    vec![layout.active, layout.halted, layout.shout_has_lookup]
}

// Selector(8) + bitness(20) + ALU/branch/decomposition(824).
pub(crate) const W2_FIELDS_RESIDUAL_COUNT: usize = 852;
// Virtual DIV/REM stage selectors (up to rem=19) raise decode/fields multiplicative degree.
pub(crate) const W2_FIELDS_DEGREE_BOUND: usize = 64;
pub(crate) const W2_IMM_RESIDUAL_COUNT: usize = 4;

#[inline]
pub(crate) fn w2_bool01(v: K) -> K {
    v * (v - K::ONE)
}

#[inline]
pub(crate) fn w2_reg_addr_from_bits(bits: [K; 5]) -> K {
    bits[0]
        + K::from(F::from_u64(2)) * bits[1]
        + K::from(F::from_u64(4)) * bits[2]
        + K::from(F::from_u64(8)) * bits[3]
        + K::from(F::from_u64(16)) * bits[4]
}

#[inline]
pub(crate) fn w2_decode_selector_residuals(
    active: K,
    decode_opcode: K,
    opcode_flags: [K; 12],
    op_custom: K,
    funct3_is: [K; 8],
    funct3_bits: [K; 3],
    op_amo: K,
) -> [K; 8] {
    let opcode_one_hot = opcode_flags.into_iter().fold(K::ZERO, |acc, v| acc + v) + op_custom - active;
    let funct3_one_hot = funct3_is.into_iter().fold(K::ZERO, |acc, v| acc + v) - active;
    let funct3_bit0_link = (funct3_is[1] + funct3_is[3] + funct3_is[5] + funct3_is[7]) - funct3_bits[0];
    let funct3_bit1_link = (funct3_is[2] + funct3_is[3] + funct3_is[6] + funct3_is[7]) - funct3_bits[1];
    let funct3_bit2_link = (funct3_is[4] + funct3_is[5] + funct3_is[6] + funct3_is[7]) - funct3_bits[2];
    let branch_f3b1_link = (funct3_is[6] + funct3_is[7]) - (funct3_bits[1] * funct3_bits[2]);
    // Tier-2.1 trace mode lock: op_amo must be zero on every row.
    let amo_forbidden = op_amo;
    let opcode_value_link = opcode_flags[0] * (decode_opcode - K::from(F::from_u64(0x37)))
        + opcode_flags[1] * (decode_opcode - K::from(F::from_u64(0x17)))
        + opcode_flags[2] * (decode_opcode - K::from(F::from_u64(0x6f)))
        + opcode_flags[3] * (decode_opcode - K::from(F::from_u64(0x67)))
        + opcode_flags[4] * (decode_opcode - K::from(F::from_u64(0x63)))
        + opcode_flags[5] * (decode_opcode - K::from(F::from_u64(0x03)))
        + opcode_flags[6] * (decode_opcode - K::from(F::from_u64(0x23)))
        + opcode_flags[7] * (decode_opcode - K::from(F::from_u64(0x13))) * (decode_opcode - K::from(F::from_u64(0x1b)))
        + opcode_flags[8] * (decode_opcode - K::from(F::from_u64(0x33))) * (decode_opcode - K::from(F::from_u64(0x3b)))
        + opcode_flags[9] * (decode_opcode - K::from(F::from_u64(0x0f)))
        + opcode_flags[10] * (decode_opcode - K::from(F::from_u64(0x73)))
        + opcode_flags[11] * (decode_opcode - K::from(F::from_u64(0x2f)))
        + op_custom * (decode_opcode - K::from(F::from_u64(0x0b)));

    [
        opcode_one_hot,
        funct3_one_hot,
        funct3_bit0_link,
        funct3_bit1_link,
        funct3_bit2_link,
        branch_f3b1_link,
        amo_forbidden,
        opcode_value_link,
    ]
}

#[inline]
pub(crate) fn w2_decode_bitness_residuals(opcode_flags: [K; 12], funct3_is: [K; 8]) -> [K; 20] {
    [
        w2_bool01(opcode_flags[0]),
        w2_bool01(opcode_flags[1]),
        w2_bool01(opcode_flags[2]),
        w2_bool01(opcode_flags[3]),
        w2_bool01(opcode_flags[4]),
        w2_bool01(opcode_flags[5]),
        w2_bool01(opcode_flags[6]),
        w2_bool01(opcode_flags[7]),
        w2_bool01(opcode_flags[8]),
        w2_bool01(opcode_flags[9]),
        w2_bool01(opcode_flags[10]),
        w2_bool01(opcode_flags[11]),
        w2_bool01(funct3_is[0]),
        w2_bool01(funct3_is[1]),
        w2_bool01(funct3_is[2]),
        w2_bool01(funct3_is[3]),
        w2_bool01(funct3_is[4]),
        w2_bool01(funct3_is[5]),
        w2_bool01(funct3_is[6]),
        w2_bool01(funct3_is[7]),
    ]
}

#[inline]
pub(crate) fn w2_alu_reg_table_delta_from_bits(funct7_bits: [K; 7], funct3_is: [K; 8]) -> K {
    let is_rv32m = funct7_bits[0];

    let base_delta = funct7_bits[5] * (funct3_is[0] + funct3_is[5]);
    let rv32m_delta = K::from(F::from_u64(9)) * funct3_is[0]
        + K::from(F::from_u64(6)) * funct3_is[1]
        + K::from(F::from_u64(10)) * funct3_is[2]
        + K::from(F::from_u64(8)) * funct3_is[3]
        + K::from(F::from_u64(15)) * funct3_is[4]
        + K::from(F::from_u64(9)) * funct3_is[5]
        + K::from(F::from_u64(16)) * funct3_is[6]
        + K::from(F::from_u64(19)) * funct3_is[7];

    (K::ONE - is_rv32m) * base_delta + is_rv32m * rv32m_delta
}

#[inline]
pub(crate) fn w2_decode_immediate_residuals(
    imm_i: K,
    imm_s: K,
    imm_b: K,
    imm_j: K,
    rd_bits: [K; 5],
    funct3_bits: [K; 3],
    rs1_bits: [K; 5],
    rs2_bits: [K; 5],
    funct7_bits: [K; 7],
) -> [K; 4] {
    let signext_imm12 = K::from(F::from_u64((1u64 << 32) - (1u64 << 11)));
    let signext_imm13 = K::from(F::from_u64((1u64 << 32) - (1u64 << 12)));
    let signext_imm21 = K::from(F::from_u64((1u64 << 32) - (1u64 << 20)));

    let imm_i_res = imm_i
        - rs2_bits[0]
        - K::from(F::from_u64(2)) * rs2_bits[1]
        - K::from(F::from_u64(4)) * rs2_bits[2]
        - K::from(F::from_u64(8)) * rs2_bits[3]
        - K::from(F::from_u64(16)) * rs2_bits[4]
        - K::from(F::from_u64(32)) * funct7_bits[0]
        - K::from(F::from_u64(64)) * funct7_bits[1]
        - K::from(F::from_u64(128)) * funct7_bits[2]
        - K::from(F::from_u64(256)) * funct7_bits[3]
        - K::from(F::from_u64(512)) * funct7_bits[4]
        - K::from(F::from_u64(1024)) * funct7_bits[5]
        - signext_imm12 * funct7_bits[6];

    let imm_s_res = imm_s
        - rd_bits[0]
        - K::from(F::from_u64(2)) * rd_bits[1]
        - K::from(F::from_u64(4)) * rd_bits[2]
        - K::from(F::from_u64(8)) * rd_bits[3]
        - K::from(F::from_u64(16)) * rd_bits[4]
        - K::from(F::from_u64(32)) * funct7_bits[0]
        - K::from(F::from_u64(64)) * funct7_bits[1]
        - K::from(F::from_u64(128)) * funct7_bits[2]
        - K::from(F::from_u64(256)) * funct7_bits[3]
        - K::from(F::from_u64(512)) * funct7_bits[4]
        - K::from(F::from_u64(1024)) * funct7_bits[5]
        - signext_imm12 * funct7_bits[6];

    let imm_b_res = imm_b
        - K::from(F::from_u64(2)) * rd_bits[1]
        - K::from(F::from_u64(4)) * rd_bits[2]
        - K::from(F::from_u64(8)) * rd_bits[3]
        - K::from(F::from_u64(16)) * rd_bits[4]
        - K::from(F::from_u64(32)) * funct7_bits[0]
        - K::from(F::from_u64(64)) * funct7_bits[1]
        - K::from(F::from_u64(128)) * funct7_bits[2]
        - K::from(F::from_u64(256)) * funct7_bits[3]
        - K::from(F::from_u64(512)) * funct7_bits[4]
        - K::from(F::from_u64(1024)) * funct7_bits[5]
        - K::from(F::from_u64(2048)) * rd_bits[0]
        - signext_imm13 * funct7_bits[6];

    let imm_j_res = imm_j
        - K::from(F::from_u64(2)) * rs2_bits[1]
        - K::from(F::from_u64(4)) * rs2_bits[2]
        - K::from(F::from_u64(8)) * rs2_bits[3]
        - K::from(F::from_u64(16)) * rs2_bits[4]
        - K::from(F::from_u64(32)) * funct7_bits[0]
        - K::from(F::from_u64(64)) * funct7_bits[1]
        - K::from(F::from_u64(128)) * funct7_bits[2]
        - K::from(F::from_u64(256)) * funct7_bits[3]
        - K::from(F::from_u64(512)) * funct7_bits[4]
        - K::from(F::from_u64(1024)) * funct7_bits[5]
        - K::from(F::from_u64(2048)) * rs2_bits[0]
        - K::from(F::from_u64(4096)) * funct3_bits[0]
        - K::from(F::from_u64(8192)) * funct3_bits[1]
        - K::from(F::from_u64(16384)) * funct3_bits[2]
        - K::from(F::from_u64(32768)) * rs1_bits[0]
        - K::from(F::from_u64(65536)) * rs1_bits[1]
        - K::from(F::from_u64(131072)) * rs1_bits[2]
        - K::from(F::from_u64(262144)) * rs1_bits[3]
        - K::from(F::from_u64(524288)) * rs1_bits[4]
        - signext_imm21 * funct7_bits[6];

    [imm_i_res, imm_s_res, imm_b_res, imm_j_res]
}

#[inline]
pub(crate) fn w3_load_semantics_residuals(
    rd_val: K,
    ram_rv: K,
    rd_has_write: K,
    ram_has_read: K,
    load_flags: [K; 5],
    ram_rv_q16: K,
    ram_rv_low_bits: [K; 16],
) -> [K; 16] {
    let pow2 = |k: usize| K::from(F::from_u64(1u64 << k));
    let two16 = K::from(F::from_u64(1u64 << 16));
    let lb_sign_coeff = K::from(F::from_u64((1u64 << 32) - (1u64 << 7)));
    let lh_sign_coeff = K::from(F::from_u64((1u64 << 32) - (1u64 << 15)));

    let mut ram_rv_low8 = K::ZERO;
    for (k, b) in ram_rv_low_bits.iter().copied().enumerate().take(8) {
        ram_rv_low8 += pow2(k) * b;
    }
    let mut ram_rv_low16 = K::ZERO;
    for (k, b) in ram_rv_low_bits.iter().copied().enumerate() {
        ram_rv_low16 += pow2(k) * b;
    }

    let lb_val = {
        let mut acc = K::ZERO;
        for (k, b) in ram_rv_low_bits.iter().copied().enumerate().take(8) {
            acc += if k == 7 { lb_sign_coeff } else { pow2(k) } * b;
        }
        acc
    };
    let lh_val = {
        let mut acc = K::ZERO;
        for (k, b) in ram_rv_low_bits.iter().copied().enumerate() {
            if k >= 16 {
                break;
            }
            acc += if k == 15 { lh_sign_coeff } else { pow2(k) } * b;
        }
        acc
    };
    let rd_write_gate = rd_has_write;

    [
        load_flags[4] * rd_write_gate * (rd_val - ram_rv),
        load_flags[0] * rd_write_gate * (rd_val - lb_val),
        load_flags[1] * rd_write_gate * (rd_val - ram_rv_low8),
        load_flags[2] * rd_write_gate * (rd_val - lh_val),
        load_flags[3] * rd_write_gate * (rd_val - ram_rv_low16),
        load_flags[0] * rd_has_write * (rd_has_write - K::ONE),
        load_flags[1] * rd_has_write * (rd_has_write - K::ONE),
        load_flags[2] * rd_has_write * (rd_has_write - K::ONE),
        load_flags[3] * rd_has_write * (rd_has_write - K::ONE),
        load_flags[4] * rd_has_write * (rd_has_write - K::ONE),
        load_flags[0] * (ram_has_read - K::ONE),
        load_flags[1] * (ram_has_read - K::ONE),
        load_flags[2] * (ram_has_read - K::ONE),
        load_flags[3] * (ram_has_read - K::ONE),
        load_flags[4] * (ram_has_read - K::ONE),
        ram_has_read * (ram_rv - two16 * ram_rv_q16 - ram_rv_low16),
    ]
}

#[inline]
pub(crate) fn w3_store_semantics_residuals(
    ram_wv: K,
    ram_rv: K,
    rs2_val: K,
    rd_has_write: K,
    ram_has_read: K,
    ram_has_write: K,
    store_flags: [K; 3],
    rs2_q16: K,
    ram_rv_low_bits: [K; 16],
    rs2_low_bits: [K; 16],
) -> [K; 12] {
    let pow2 = |k: usize| K::from(F::from_u64(1u64 << k));
    let two16 = K::from(F::from_u64(1u64 << 16));
    let mut rs2_low16 = K::ZERO;
    let mut sb_patch = K::ZERO;
    let mut sh_patch = K::ZERO;
    for k in 0..16 {
        let coeff = pow2(k);
        rs2_low16 += coeff * rs2_low_bits[k];
        if k < 8 {
            sb_patch += coeff * (ram_rv_low_bits[k] - rs2_low_bits[k]);
        }
        sh_patch += coeff * (ram_rv_low_bits[k] - rs2_low_bits[k]);
    }
    [
        store_flags[2] * (ram_wv - rs2_val),
        store_flags[0] * (ram_wv - ram_rv + sb_patch),
        store_flags[1] * (ram_wv - ram_rv + sh_patch),
        store_flags[0] * rd_has_write,
        store_flags[1] * rd_has_write,
        store_flags[2] * rd_has_write,
        store_flags[0] * (ram_has_read - K::ONE),
        store_flags[1] * (ram_has_read - K::ONE),
        store_flags[0] * (ram_has_write - K::ONE),
        store_flags[1] * (ram_has_write - K::ONE),
        store_flags[2] * (ram_has_write - K::ONE),
        rs2_val - two16 * rs2_q16 - rs2_low16,
    ]
}

#[inline]
pub(crate) fn control_branch_taken_from_bits(shout_val: K, funct3_bit0: K) -> K {
    shout_val + funct3_bit0 - K::from(F::from_u64(2)) * funct3_bit0 * shout_val
}

#[inline]
pub(crate) fn control_imm_u_from_bits(
    funct3_bits: [K; 3],
    rs1_bits: [K; 5],
    rs2_bits: [K; 5],
    funct7_bits: [K; 7],
) -> K {
    let pow2 = |k: u64| K::from(F::from_u64(1u64 << k));
    let mut out = K::ZERO;
    out += pow2(12) * funct3_bits[0];
    out += pow2(13) * funct3_bits[1];
    out += pow2(14) * funct3_bits[2];
    out += pow2(15) * rs1_bits[0];
    out += pow2(16) * rs1_bits[1];
    out += pow2(17) * rs1_bits[2];
    out += pow2(18) * rs1_bits[3];
    out += pow2(19) * rs1_bits[4];
    out += pow2(20) * rs2_bits[0];
    out += pow2(21) * rs2_bits[1];
    out += pow2(22) * rs2_bits[2];
    out += pow2(23) * rs2_bits[3];
    out += pow2(24) * rs2_bits[4];
    out += pow2(25) * funct7_bits[0];
    out += pow2(26) * funct7_bits[1];
    out += pow2(27) * funct7_bits[2];
    out += pow2(28) * funct7_bits[3];
    out += pow2(29) * funct7_bits[4];
    out += pow2(30) * funct7_bits[5];
    out += pow2(31) * funct7_bits[6];
    out
}

#[inline]
pub(crate) fn control_imm_u_value_from_bits(
    funct3_bits: [K; 3],
    rs1_bits: [K; 5],
    rs2_bits: [K; 5],
    funct7_bits: [K; 7],
    machine_xlen: usize,
) -> K {
    let imm_u = control_imm_u_from_bits(funct3_bits, rs1_bits, rs2_bits, funct7_bits);
    if machine_xlen != 64 {
        return imm_u;
    }
    let two32 = K::from(F::from_u64(1u64 << 32));
    let sign_fill_hi32 = (two32 - K::ONE) * funct7_bits[6];
    imm_u + sign_fill_hi32 * two32
}

#[inline]
pub(crate) fn control_next_pc_linear_residual(
    pc_before: K,
    pc_after: K,
    is_virtual: K,
    op_lui: K,
    op_auipc: K,
    op_load: K,
    op_store: K,
    op_alu_imm: K,
    op_alu_reg: K,
    op_misc_mem: K,
    op_system: K,
    op_amo: K,
    op_custom: K,
) -> K {
    let op_linear =
        op_lui + op_auipc + op_load + op_store + op_alu_imm + op_alu_reg + op_misc_mem + op_system + op_amo + op_custom;
    let non_virtual = K::ONE - is_virtual;
    non_virtual * op_linear * (pc_after - pc_before - K::from(F::from_u64(4)))
}

#[inline]
pub(crate) fn control_next_pc_control_residuals(
    active: K,
    pc_before: K,
    pc_after: K,
    _rs1_val: K,
    jalr_drop_bit: K,
    _imm_i: K,
    imm_b: K,
    imm_j: K,
    imm_sign_bit: K,
    op_jal: K,
    op_jalr: K,
    op_branch: K,
    shout_val: K,
    funct3_bit0: K,
) -> [K; 5] {
    let four = K::from(F::from_u64(4));
    let two32 = K::from(F::from_u64(1u64 << 32));
    let imm_b_signed = imm_b - two32 * imm_sign_bit;
    let imm_j_signed = imm_j - two32 * imm_sign_bit;
    let taken = control_branch_taken_from_bits(shout_val, funct3_bit0);
    [
        op_jal * (pc_after - pc_before - imm_j_signed),
        // JALR target uses modular ADD lookup output (`shout_val`) plus alignment drop.
        op_jalr * (pc_after - shout_val + jalr_drop_bit),
        op_branch * (pc_after - pc_before - four - taken * (imm_b_signed - four)),
        op_jalr * jalr_drop_bit * (jalr_drop_bit - K::ONE),
        (active - op_jalr) * jalr_drop_bit,
    ]
}

#[inline]
pub(crate) fn control_branch_semantics_residuals(
    op_branch: K,
    shout_val: K,
    _funct3_bit0: K,
    funct3_bit1: K,
    funct3_bit2: K,
    funct3_is6: K,
    funct3_is7: K,
) -> [K; 2] {
    [
        op_branch * ((funct3_is6 + funct3_is7) - funct3_bit1 * funct3_bit2),
        op_branch * shout_val * (shout_val - K::ONE),
    ]
}

#[inline]
pub(crate) fn control_writeback_residuals(
    rd_val: K,
    pc_before: K,
    imm_u: K,
    op_lui_write: K,
    op_auipc_write: K,
    op_jal_write: K,
    op_jalr_write: K,
) -> [K; 4] {
    let four = K::from(F::from_u64(4));
    let two32 = K::from(F::from_u64(1u64 << 32));
    let auipc_delta = rd_val - pc_before - imm_u;
    let jal_delta = rd_val - pc_before - four;
    [
        op_lui_write * (rd_val - imm_u),
        // RV32 writeback values are modular u32; allow either exact sum (delta=0)
        // or wrapped sum (delta=-2^32).
        op_auipc_write * auipc_delta * (auipc_delta + two32),
        op_jal_write * jal_delta * (jal_delta + two32),
        op_jalr_write * jal_delta * (jal_delta + two32),
    ]
}

pub(crate) fn rv32_trace_wp_columns(layout: &Rv32TraceLayout) -> Vec<usize> {
    vec![
        layout.is_virtual,
        layout.virtual_sequence_remaining,
        layout.virtual_commit_from_prev,
        layout.instr_word,
        layout.rs1_addr,
        layout.rs1_val,
        layout.rs2_addr,
        layout.rs2_val,
        layout.rd_addr,
        layout.rd_val,
        layout.rd_has_write,
        layout.ram_addr,
        layout.ram_rv,
        layout.ram_wv,
        layout.shout_has_lookup,
        layout.shout_table_id,
        layout.shout_val,
        layout.shout_lhs,
        layout.shout_rhs,
        layout.shout_add_sub_key,
        layout.jalr_drop_bit,
    ]
}

#[inline]
pub(crate) fn trace_uses_rv64_exact_words(cpu_cols_len: usize) -> bool {
    cpu_cols_len >= neo_memory::riscv::trace::Rv64TraceLayout::new().cols
}

pub(crate) fn rv64_trace_exact_word_opening_columns() -> Vec<usize> {
    let layout = neo_memory::riscv::trace::Rv64TraceLayout::new();
    vec![
        layout.rs1_val_lo32,
        layout.rs2_val_lo32,
        layout.rd_val_lo32,
        layout.shout_lhs_lo32,
        layout.shout_lhs_hi32,
        layout.shout_rhs_lo32,
        layout.shout_rhs_hi32,
        layout.shout_add_sub_key_lo32,
        layout.shout_add_sub_key_hi32,
    ]
}

pub(crate) fn rv32_trace_wp_opening_columns(layout: &Rv32TraceLayout) -> Vec<usize> {
    let mut out = Vec::with_capacity(1 + layout.cols);
    out.push(layout.active);
    out.extend(rv32_trace_wp_columns(layout));
    out
}

pub(crate) fn rv32_trace_control_extra_opening_columns(layout: &Rv32TraceLayout) -> Vec<usize> {
    vec![layout.pc_before, layout.pc_after]
}

#[inline]
pub(crate) fn named_opening(openings: &BTreeMap<usize, K>, col_id: usize, label: &str) -> Result<K, PiCcsError> {
    openings
        .get(&col_id)
        .copied()
        .ok_or_else(|| PiCcsError::ProtocolError(format!("{label}: missing opening col_id={col_id}")))
}

pub(crate) fn time_openings_for_point(
    openings: &[crate::shard_proof_types::TimePointOpening],
    point: &[K],
    required_col_ids: &[usize],
    label: &str,
) -> Result<Option<BTreeMap<usize, K>>, PiCcsError> {
    if required_col_ids.is_empty() {
        return Ok(Some(BTreeMap::new()));
    }
    let opening = match time_opening_entry_for_point(openings, point, required_col_ids, label)? {
        Some(opening) => opening,
        None => return Ok(None),
    };
    let mut map = BTreeMap::new();
    for (&col_id, &eval) in opening.col_ids.iter().zip(opening.evals.iter()) {
        map.insert(col_id, eval);
    }
    Ok(Some(map))
}

pub(crate) fn time_opening_entry_for_point<'a>(
    openings: &'a [crate::shard_proof_types::TimePointOpening],
    point: &[K],
    required_col_ids: &[usize],
    label: &str,
) -> Result<Option<&'a crate::shard_proof_types::TimePointOpening>, PiCcsError> {
    if required_col_ids.is_empty() {
        return Ok(None);
    }
    let mut required_norm = required_col_ids.to_vec();
    required_norm.sort_unstable();
    if required_norm.windows(2).any(|w| w[0] == w[1]) {
        return Err(PiCcsError::ProtocolError(format!(
            "{label}: required_col_ids contains duplicates"
        )));
    }

    let mut matched: Option<&crate::shard_proof_types::TimePointOpening> = None;
    for opening in openings
        .iter()
        .filter(|opening| opening.point.as_slice() == point)
    {
        if opening.col_ids.len() != opening.evals.len() {
            return Err(PiCcsError::ProtocolError(format!(
                "{label}: malformed time opening (col_ids={}, evals={})",
                opening.col_ids.len(),
                opening.evals.len()
            )));
        }
        let mut opening_norm = opening.col_ids.clone();
        opening_norm.sort_unstable();
        if opening_norm.windows(2).any(|w| w[0] == w[1]) {
            return Err(PiCcsError::ProtocolError(format!(
                "{label}: malformed time opening has duplicate col_ids"
            )));
        }
        if opening_norm != required_norm {
            continue;
        }

        if matched.is_some() {
            return Err(PiCcsError::ProtocolError(format!(
                "{label}: duplicate time openings for the same point/column set"
            )));
        }
        matched = Some(opening);
    }

    Ok(matched)
}

pub(crate) fn require_time_openings_for_point(
    openings: &[crate::shard_proof_types::TimePointOpening],
    point: &[K],
    required_col_ids: &[usize],
    label: &str,
) -> Result<BTreeMap<usize, K>, PiCcsError> {
    time_openings_for_point(openings, point, required_col_ids, label)?.ok_or_else(|| {
        PiCcsError::ProtocolError(format!(
            "{label}: missing required named time opening for point/column set"
        ))
    })
}

pub(crate) fn require_time_opening_entry_for_point<'a>(
    openings: &'a [crate::shard_proof_types::TimePointOpening],
    point: &[K],
    required_col_ids: &[usize],
    label: &str,
) -> Result<&'a crate::shard_proof_types::TimePointOpening, PiCcsError> {
    time_opening_entry_for_point(openings, point, required_col_ids, label)?.ok_or_else(|| {
        PiCcsError::ProtocolError(format!(
            "{label}: missing required named time opening for point/column set"
        ))
    })
}

pub(crate) fn require_time_openings_covering_point<'a>(
    openings: &'a [crate::shard_proof_types::TimePointOpening],
    point: &[K],
    required_col_ids: &[usize],
    label: &str,
) -> Result<(&'a crate::shard_proof_types::TimePointOpening, BTreeMap<usize, K>), PiCcsError> {
    if required_col_ids.is_empty() {
        return Err(PiCcsError::InvalidInput(format!(
            "{label}: required_col_ids must be non-empty"
        )));
    }
    let required: BTreeSet<usize> = required_col_ids.iter().copied().collect();
    if required.len() != required_col_ids.len() {
        return Err(PiCcsError::InvalidInput(format!(
            "{label}: required_col_ids contains duplicates"
        )));
    }

    let mut matched: Option<(&crate::shard_proof_types::TimePointOpening, BTreeMap<usize, K>)> = None;
    for opening in openings
        .iter()
        .filter(|opening| opening.point.as_slice() == point)
    {
        if opening.col_ids.len() != opening.evals.len() {
            return Err(PiCcsError::ProtocolError(format!(
                "{label}: malformed time opening (col_ids={}, evals={})",
                opening.col_ids.len(),
                opening.evals.len()
            )));
        }
        let mut map = BTreeMap::new();
        for (&col_id, &eval) in opening.col_ids.iter().zip(opening.evals.iter()) {
            map.insert(col_id, eval);
        }
        if !required.iter().all(|col_id| map.contains_key(col_id)) {
            continue;
        }
        let mut selected = BTreeMap::new();
        for &col_id in required.iter() {
            let eval = *map.get(&col_id).ok_or_else(|| {
                PiCcsError::ProtocolError(format!("{label}: missing opening col_id={col_id} in matched entry"))
            })?;
            selected.insert(col_id, eval);
        }
        if matched.is_some() {
            return Err(PiCcsError::ProtocolError(format!(
                "{label}: multiple covering time openings found for point"
            )));
        }
        matched = Some((opening, selected));
    }

    matched.ok_or_else(|| {
        PiCcsError::ProtocolError(format!(
            "{label}: missing required named time opening covering the requested columns"
        ))
    })
}

pub(crate) fn bus_logical_col_ids_for_step_instance(
    step: &StepInstanceBundle<Cmt, F, K>,
    cpu_bus: &BusLayout,
    label: &str,
) -> Result<Vec<usize>, PiCcsError> {
    let cpu_cols_len = step.time_columns.cpu_cols.len();
    let mem_cols_len = step.time_columns.mem_cols.len();
    let total_cols = cpu_cols_len
        .checked_add(mem_cols_len)
        .ok_or_else(|| PiCcsError::InvalidInput(format!("{label}: cpu_cols + mem_cols overflow")))?;
    if mem_cols_len != cpu_bus.bus_cols {
        return Err(PiCcsError::ProtocolError(format!(
            "{label}: mem_cols.len()={} must equal bus_cols={}",
            mem_cols_len, cpu_bus.bus_cols
        )));
    }
    if step.time_columns.col_ids.len() != total_cols {
        return Err(PiCcsError::ProtocolError(format!(
            "{label}: logical col_id table mismatch (col_ids={}, expected cpu+mem={})",
            step.time_columns.col_ids.len(),
            total_cols
        )));
    }
    Ok(step.time_columns.col_ids[cpu_cols_len..].to_vec())
}

pub(crate) fn infer_rv32_trace_t_len_for_wb_wp(
    step: &StepWitnessBundle<Cmt, F, K>,
    trace: &Rv32TraceLayout,
) -> Result<usize, PiCcsError> {
    let t_len = step.time_columns.t;
    if t_len == 0 {
        return Err(PiCcsError::InvalidInput(
            "WB/WP requires canonical time columns with t >= 1".into(),
        ));
    }
    if step.time_columns.cpu_cols.len() < trace.cols {
        return Err(PiCcsError::InvalidInput(format!(
            "WB/WP requires canonical RV32 time cpu prefix columns (got {}, expected at least {})",
            step.time_columns.cpu_cols.len(),
            trace.cols
        )));
    }
    Ok(t_len)
}

pub(crate) fn decode_trace_col_values_batch(
    _params: &NeoParams,
    step: &StepWitnessBundle<Cmt, F, K>,
    t_len: usize,
    col_ids: &[usize],
) -> Result<BTreeMap<usize, Vec<K>>, PiCcsError> {
    if step.time_columns.t != t_len || step.time_columns.cpu_cols.is_empty() {
        return Err(PiCcsError::InvalidInput(format!(
            "WB/WP requires canonical time CPU columns (time_t={}, cpu_cols={}, expected_t={t_len})",
            step.time_columns.t,
            step.time_columns.cpu_cols.len()
        )));
    }

    let unique_col_ids: BTreeSet<usize> = col_ids.iter().copied().collect();
    let mut decoded = BTreeMap::<usize, Vec<K>>::new();
    for col_id in unique_col_ids {
        let vals = step.time_columns.cpu_cols.get(col_id).ok_or_else(|| {
            PiCcsError::InvalidInput(format!(
                "WB/WP: trace col_id {} out of range for time_columns.cpu_cols.len()={}",
                col_id,
                step.time_columns.cpu_cols.len()
            ))
        })?;
        if vals.len() != t_len {
            return Err(PiCcsError::InvalidInput(format!(
                "WB/WP: time_columns.cpu_cols[{col_id}].len()={} != t_len={t_len}",
                vals.len()
            )));
        }
        decoded.insert(col_id, vals.iter().copied().map(K::from).collect());
    }

    Ok(decoded)
}

pub(crate) fn decode_lookup_backed_col_values_batch(
    t_len: usize,
    max_cols: usize,
    time_mem_cols: Option<&[Vec<F>]>,
    col_ids: &[usize],
) -> Result<BTreeMap<usize, Vec<K>>, PiCcsError> {
    let mem_cols =
        time_mem_cols.ok_or_else(|| PiCcsError::InvalidInput("W2: canonical time mem columns are required".into()))?;
    if mem_cols.is_empty() {
        return Err(PiCcsError::InvalidInput(
            "W2: canonical time mem columns are required".into(),
        ));
    }

    let unique_col_ids: BTreeSet<usize> = col_ids.iter().copied().collect();
    let mut decoded = BTreeMap::<usize, Vec<K>>::new();
    for col_id in unique_col_ids {
        if col_id >= max_cols {
            return Err(PiCcsError::InvalidInput(format!(
                "W2: decode lookup-backed column out of range (col_id={col_id}, cols={max_cols})"
            )));
        }
        let vals = mem_cols.get(col_id).ok_or_else(|| {
            PiCcsError::InvalidInput(format!(
                "W2: missing time mem column col_id={col_id} (mem_cols={})",
                mem_cols.len()
            ))
        })?;
        if vals.len() != t_len {
            return Err(PiCcsError::InvalidInput(format!(
                "W2: time mem column length mismatch for col_id={col_id} (len={}, t_len={t_len})",
                vals.len()
            )));
        }
        decoded.insert(col_id, vals.iter().copied().map(K::from).collect());
    }
    Ok(decoded)
}

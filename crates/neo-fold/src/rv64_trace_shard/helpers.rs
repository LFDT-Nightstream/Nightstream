use super::*;

pub(super) fn field_from_u64_injective(value: u64, label: &str) -> Result<F, PiCcsError> {
    if value >= <F as PrimeField64>::ORDER_U64 {
        return Err(PiCcsError::InvalidInput(format!(
            "RV64 trace proving slice requires {label} < Goldilocks modulus; got value={value:#x}"
        )));
    }
    Ok(F::from_u64(value))
}

#[inline]
pub(super) fn validate_rv64_reg_index(reg: u64, label: &str) -> Result<(), PiCcsError> {
    if reg >= 32 {
        return Err(PiCcsError::InvalidInput(format!(
            "{label}: register index out of range: reg={reg} (expected 0..32)"
        )));
    }
    Ok(())
}

pub(super) fn validate_rv64_reg_init_words(reg_init: &HashMap<u64, u64>) -> Result<(), PiCcsError> {
    for (&reg, &value) in reg_init {
        validate_rv64_reg_index(reg, "reg_init_u64")?;
        if reg == 0 && value != 0 {
            return Err(PiCcsError::InvalidInput(
                "reg_init_u64: x0 must be 0 (non-zero init is forbidden)".into(),
            ));
        }
    }
    Ok(())
}

pub(super) fn validate_rv64_reg_output_claims(claims: &ProgramIO<F>, label: &str) -> Result<(), PiCcsError> {
    for (reg, _) in claims.claims() {
        validate_rv64_reg_index(reg, label)?;
    }
    Ok(())
}

pub(super) fn validate_rv64_exact_reg_output_words(outputs: &BTreeMap<u64, u64>) -> Result<(), PiCcsError> {
    for &reg in outputs.keys() {
        validate_rv64_reg_index(reg, "reg_output_claim_exact_u64")?;
    }
    Ok(())
}

#[inline]
pub(super) fn field_from_u64_exact_transport(value: u64) -> F {
    let lo = (value as u32) as u64;
    let hi = value >> 32;
    F::from_u64(lo) + F::from_u64(hi) * F::from_u64(1u64 << 32)
}

pub(super) fn rv64_program_requires_ram_sidecar(program: &[RiscvInstruction]) -> bool {
    program
        .iter()
        .any(|instr| matches!(instr, RiscvInstruction::Load { .. } | RiscvInstruction::Store { .. }))
}

pub(super) fn rv64_program_requires_width_lookup(program: &[RiscvInstruction]) -> bool {
    program
        .iter()
        .any(|instr| matches!(instr, RiscvInstruction::Load { .. } | RiscvInstruction::Store { .. }))
}

pub(super) fn required_bits_for_max_addr(max_addr: u64) -> usize {
    if max_addr == 0 {
        return 1;
    }
    (u64::BITS - max_addr.leading_zeros()) as usize
}

pub(super) fn max_consecutive_pc_run(exec: &RiscvExecTable) -> usize {
    let mut best = 1usize;
    let mut cur = 0usize;
    let mut prev_pc: Option<u64> = None;
    for row in exec.rows.iter().filter(|r| r.active) {
        if prev_pc == Some(row.pc_before) {
            cur += 1;
        } else {
            cur = 1;
            prev_pc = Some(row.pc_before);
        }
        best = best.max(cur);
    }
    best
}

pub(super) fn split_exec_into_fixed_chunks(
    exec: &RiscvExecTable,
    chunk_rows: usize,
) -> Result<Vec<RiscvExecTable>, PiCcsError> {
    if chunk_rows == 0 {
        return Err(PiCcsError::InvalidInput("trace chunk_rows must be non-zero".into()));
    }
    if exec.rows.is_empty() {
        return Err(PiCcsError::InvalidInput("trace execution table is empty".into()));
    }
    if exec.rows.len() <= chunk_rows {
        return Ok(vec![exec.clone()]);
    }

    let mut out = Vec::<RiscvExecTable>::new();
    let total = exec.rows.len();
    let mut start = 0usize;
    while start < total {
        let end = (start + chunk_rows).min(total);
        let mut rows = exec.rows[start..end].to_vec();
        if rows.len() < chunk_rows {
            let last = rows
                .last()
                .ok_or_else(|| PiCcsError::InvalidInput("trace chunk unexpectedly empty".into()))?
                .clone();
            let mut cycle = last.cycle;
            let pad_pc = last.pc_after;
            let pad_halted = last.halted;
            while rows.len() < chunk_rows {
                cycle = cycle
                    .checked_add(1)
                    .ok_or_else(|| PiCcsError::InvalidInput("cycle overflow while chunk-padding trace".into()))?;
                rows.push(RiscvExecRow::inactive(cycle, pad_pc, pad_halted));
            }
        }
        out.push(RiscvExecTable { rows });
        start = end;
    }
    Ok(out)
}

pub(super) fn boundary_splits_virtual_sequence(exec: &RiscvExecTable, chunk_rows: usize) -> bool {
    if chunk_rows == 0 || exec.rows.len() <= chunk_rows {
        return false;
    }
    let total = exec.rows.len();
    let mut boundary = chunk_rows;
    while boundary < total {
        let prev = &exec.rows[boundary - 1];
        if prev.active && prev.is_virtual {
            return true;
        }
        boundary = match boundary.checked_add(chunk_rows) {
            Some(next) => next,
            None => break,
        };
    }
    false
}

pub(super) fn rv64_trace_chunk_to_witness(
    layout: Rv64TraceCcsLayout,
) -> Box<dyn Fn(&[StepTrace<u64, u64, u128>]) -> Vec<F> + Send + Sync> {
    Box::new(move |chunk: &[StepTrace<u64, u64, u128>]| {
        rv64_trace_chunk_to_witness_checked(&layout, chunk)
            .unwrap_or_else(|e| panic!("rv64_trace_chunk_to_witness failed for chunk_len={}: {e}", chunk.len()))
    })
}

pub(super) fn rv64_trace_chunk_to_witness_checked(
    layout: &Rv64TraceCcsLayout,
    chunk: &[StepTrace<u64, u64, u128>],
) -> Result<Vec<F>, String> {
    if chunk.is_empty() {
        return Err("trace chunk witness: chunk must contain at least one step".into());
    }
    if chunk.len() > layout.t {
        return Err(format!(
            "trace chunk witness: chunk.len()={} exceeds layout.t={}",
            chunk.len(),
            layout.t
        ));
    }

    let mut rows = Vec::with_capacity(layout.t);
    for step in chunk {
        rows.push(RiscvExecRow::from_step_with_xlen(step, /*machine_xlen=*/ 64)?);
    }

    let mut cycle = rows
        .last()
        .ok_or_else(|| "trace chunk witness: empty rows after conversion".to_string())?
        .cycle;
    let last_row = rows
        .last()
        .ok_or_else(|| "trace chunk witness: empty rows after conversion".to_string())?;
    let pad_pc = last_row.pc_after;
    let pad_halted = last_row.halted;
    while rows.len() < layout.t {
        cycle = cycle
            .checked_add(1)
            .ok_or_else(|| "trace chunk witness: cycle overflow while padding".to_string())?;
        rows.push(RiscvExecRow::inactive(cycle, pad_pc, pad_halted));
    }

    let exec = RiscvExecTable { rows };
    let (x, w) = rv64_trace_ccs_witness_from_exec_table(layout, &exec)?;
    Ok(x.into_iter().chain(w).collect())
}

pub(super) fn validate_trace_opcode_lookup_one_hot(
    trace: &VmTrace<u64, u64, u128>,
    xlen: usize,
) -> Result<(), PiCcsError> {
    let shout = RiscvShoutTables::new(xlen);
    for (step_idx, step) in trace.steps.iter().enumerate() {
        let mut seen_table_id: Option<u32> = None;
        for ev in &step.shout_events {
            if shout.id_to_opcode(ev.shout_id).is_none() {
                continue;
            }
            if let Some(prev) = seen_table_id {
                return Err(PiCcsError::ProtocolError(format!(
                    "instruction-table lookup flags are not one-hot at step {step_idx}: multiple opcode table_ids ({prev}, {})",
                    ev.shout_id.0
                )));
            }
            seen_table_id = Some(ev.shout_id.0);
        }
    }
    Ok(())
}

pub(super) fn validate_rv64_trace_proving_subset(
    program: &[RiscvInstruction],
) -> Result<HashSet<RiscvOpcode>, PiCcsError> {
    let mut ops = HashSet::new();
    for instruction in program {
        match instruction {
            RiscvInstruction::RAlu { op, .. } => {
                if !rv64_trace_supported_opcode(*op) {
                    return Err(PiCcsError::InvalidInput(format!(
                        "RV64 trace proving slice does not yet support opcode {op:?}"
                    )));
                }
                ops.insert(*op);
            }
            RiscvInstruction::IAlu { op, .. } => {
                if !rv64_trace_supported_opcode(*op) {
                    return Err(PiCcsError::InvalidInput(format!(
                        "RV64 trace proving slice does not yet support immediate opcode {op:?}"
                    )));
                }
                ops.insert(*op);
            }
            RiscvInstruction::Branch { cond, .. } => {
                let cmp_op = cond.to_shout_opcode();
                if !rv64_trace_supported_opcode(cmp_op) {
                    return Err(PiCcsError::InvalidInput(format!(
                        "RV64 trace proving slice does not yet support branch compare opcode {cmp_op:?}"
                    )));
                }
                ops.insert(cmp_op);
            }
            RiscvInstruction::Jalr { .. } | RiscvInstruction::Auipc { .. } => {
                ops.insert(RiscvOpcode::Add);
            }
            RiscvInstruction::Lui { .. }
            | RiscvInstruction::Jal { .. }
            | RiscvInstruction::Halt
            | RiscvInstruction::Nop
            | RiscvInstruction::Ecall => {}
            RiscvInstruction::Load { op, .. } => match op {
                RiscvMemOp::Lb
                | RiscvMemOp::Lh
                | RiscvMemOp::Lw
                | RiscvMemOp::Lbu
                | RiscvMemOp::Lhu
                | RiscvMemOp::Lwu
                | RiscvMemOp::Ld => {}
                _ => {
                    return Err(PiCcsError::InvalidInput(format!(
                        "RV64 trace proving slice does not yet support load-width proving for {op:?}"
                    )));
                }
            },
            RiscvInstruction::Store { op, .. } => match op {
                RiscvMemOp::Sb | RiscvMemOp::Sh | RiscvMemOp::Sw | RiscvMemOp::Sd => {}
                _ => {
                    return Err(PiCcsError::InvalidInput(format!(
                        "RV64 trace proving slice does not yet support store-width proving for {op:?}"
                    )));
                }
            },
            RiscvInstruction::RAluw { op, .. } | RiscvInstruction::IAluw { op, .. } => {
                if !rv64_trace_supported_opcode(*op) {
                    return Err(PiCcsError::InvalidInput(format!(
                        "RV64 trace proving slice does not yet support W-suffix ALU proof semantics for {op:?}"
                    )));
                }
                match op {
                    RiscvOpcode::Mulw => {
                        // MULW is helper-owned in the current RV64 slice:
                        // virtual rows use VMULW, then VMOVSIGNW, then a local compose row before the
                        // architectural commit row.
                        ops.insert(RiscvOpcode::VirtualMulWord);
                        ops.insert(RiscvOpcode::VirtualMovsignWord);
                    }
                    RiscvOpcode::Divw => {
                        ops.insert(RiscvOpcode::VirtualDivWord);
                        ops.insert(RiscvOpcode::VirtualMovsignWord);
                    }
                    RiscvOpcode::Divuw => {
                        ops.insert(RiscvOpcode::VirtualDivuWord);
                        ops.insert(RiscvOpcode::VirtualMovsignWord);
                    }
                    RiscvOpcode::Remw => {
                        ops.insert(RiscvOpcode::VirtualRemWord);
                        ops.insert(RiscvOpcode::VirtualMovsignWord);
                    }
                    RiscvOpcode::Remuw => {
                        ops.insert(RiscvOpcode::VirtualRemuWord);
                        ops.insert(RiscvOpcode::VirtualMovsignWord);
                    }
                    _ => {
                        ops.insert(*op);
                    }
                }
            }
            RiscvInstruction::Poseidon2AbsorbElem { .. }
            | RiscvInstruction::Poseidon2Finalize
            | RiscvInstruction::Poseidon2SqueezeWord { .. } => {}
            RiscvInstruction::LoadReserved { .. }
            | RiscvInstruction::StoreConditional { .. }
            | RiscvInstruction::Amo { .. }
            | RiscvInstruction::Ebreak
            | RiscvInstruction::Fence { .. }
            | RiscvInstruction::FenceI => {
                return Err(PiCcsError::InvalidInput(format!(
                    "RV64 trace proving slice does not yet support instruction {instruction:?}"
                )));
            }
        }
    }
    ops.insert(RiscvOpcode::Add);
    Ok(ops)
}

pub(super) fn rv64_trace_supported_opcode(op: RiscvOpcode) -> bool {
    matches!(
        op,
        RiscvOpcode::Add
            | RiscvOpcode::Sub
            | RiscvOpcode::And
            | RiscvOpcode::Or
            | RiscvOpcode::Xor
            | RiscvOpcode::Sll
            | RiscvOpcode::Srl
            | RiscvOpcode::Slt
            | RiscvOpcode::Sltu
            | RiscvOpcode::Eq
            | RiscvOpcode::Neq
            | RiscvOpcode::Mul
            | RiscvOpcode::Mulhu
            | RiscvOpcode::Addw
            | RiscvOpcode::Subw
            | RiscvOpcode::Sllw
            | RiscvOpcode::Srlw
            | RiscvOpcode::Sraw
            | RiscvOpcode::Mulw
            | RiscvOpcode::Divw
            | RiscvOpcode::Divuw
            | RiscvOpcode::Remw
            | RiscvOpcode::Remuw
    )
}

pub(super) fn validate_rv64_trace_field_injectivity(
    exec: &RiscvExecTable,
    prepared: &Rv64PreparedProgram,
) -> Result<(), PiCcsError> {
    let _ = prepared;
    for row in &exec.rows {
        let _ = field_from_u64_injective(row.cycle, "cycle")?;
        let _ = field_from_u64_injective(row.pc_before, "pc_before")?;
        let _ = field_from_u64_injective(row.pc_after, "pc_after")?;
        if let Some(io) = &row.reg_read_lane0 {
            let _ = field_from_u64_injective(io.addr, "rs1_addr")?;
        }
        if let Some(io) = &row.reg_read_lane1 {
            let _ = field_from_u64_injective(io.addr, "rs2_addr")?;
        }
        if let Some(io) = &row.reg_write_lane0 {
            let _ = field_from_u64_injective(io.addr, "rd_addr")?;
        }
        if let Some(read) = &row.prog_read {
            let _ = field_from_u64_injective(read.addr, "prog_addr")?;
            let _ = field_from_u64_injective(read.value, "prog_value")?;
        }
        for event in &row.shout_events {
            let _ = event;
        }
        for event in &row.ram_events {
            let _ = field_from_u64_injective(event.addr, "ram_addr")?;
        }
    }
    Ok(())
}

pub(super) fn rv64_trace_table_specs(shout_ops: &HashSet<RiscvOpcode>) -> HashMap<u32, LutTableSpec> {
    let shout = RiscvShoutTables::new(64);
    let mut table_specs = HashMap::new();
    for &op in shout_ops {
        let spec = match op {
            RiscvOpcode::Mul | RiscvOpcode::Mulhu => LutTableSpec::RiscvOpcodePacked { opcode: op, xlen: 64 },
            RiscvOpcode::VirtualMulWord
            | RiscvOpcode::VirtualMovsignWord
            | RiscvOpcode::VirtualDivWord
            | RiscvOpcode::VirtualDivuWord
            | RiscvOpcode::VirtualRemWord
            | RiscvOpcode::VirtualRemuWord => LutTableSpec::RiscvOpcodePacked { opcode: op, xlen: 32 },
            _ => LutTableSpec::RiscvOpcode { opcode: op, xlen: 64 },
        };
        table_specs.insert(shout.opcode_to_id(op).0, spec);
    }
    table_specs
}

pub(super) fn build_rv64_width_lookup_tables(
    width_layout: &Rv64WidthSidecarLayout,
    exec: &RiscvExecTable,
    trace_steps: usize,
) -> Result<(HashMap<u32, LutTable<F>>, usize), PiCcsError> {
    let max_cycle = exec
        .rows
        .iter()
        .take(trace_steps)
        .map(|r| r.cycle)
        .max()
        .unwrap_or(0);
    let cycle_d = required_bits_for_max_addr(max_cycle).max(2);
    let cycle_k = 1usize
        .checked_shl(cycle_d as u32)
        .ok_or_else(|| PiCcsError::InvalidInput(format!("RV64 width lookup cycle width too large: d={cycle_d}")))?;

    let wit = rv64_width_sidecar_witness_from_exec_table(width_layout, exec);
    let width_cols = rv64_width_lookup_backed_cols(width_layout);
    if width_cols.is_empty() {
        return Ok((HashMap::new(), cycle_d));
    }

    let mut unique_table_ids: Vec<u32> = width_cols
        .iter()
        .map(|&col_id| rv64_width_lookup_table_id_for_col(col_id))
        .collect();
    unique_table_ids.sort_unstable();
    unique_table_ids.dedup();

    let mut out = HashMap::new();
    if unique_table_ids.len() == 1 {
        let table_id = unique_table_ids[0];
        let n_vals = width_cols.len().max(1);
        let content_len = cycle_k
            .checked_mul(n_vals)
            .ok_or_else(|| PiCcsError::InvalidInput("RV64 width lookup content length overflow".into()))?;
        let mut content = vec![F::ZERO; content_len];
        for (i, row) in exec.rows.iter().enumerate().take(trace_steps) {
            let cycle = row.cycle as usize;
            if cycle >= cycle_k {
                return Err(PiCcsError::ProtocolError(format!(
                    "RV64 width lookup cycle out of range at row {i}: cycle={}, k={cycle_k}",
                    row.cycle
                )));
            }
            let base = cycle * n_vals;
            for (val_slot, &col_id) in width_cols.iter().enumerate() {
                content[base + val_slot] = wit.cols[col_id][i];
            }
        }
        out.insert(
            table_id,
            LutTable {
                table_id,
                k: cycle_k,
                d: cycle_d,
                n_side: 2,
                content,
            },
        );
        return Ok((out, cycle_d));
    }

    for &col_id in &width_cols {
        let table_id = rv64_width_lookup_table_id_for_col(col_id);
        let mut content = vec![F::ZERO; cycle_k];
        for (i, row) in exec.rows.iter().enumerate().take(trace_steps) {
            let cycle = row.cycle as usize;
            if cycle >= cycle_k {
                return Err(PiCcsError::ProtocolError(format!(
                    "RV64 width lookup cycle out of range at row {i}: cycle={}, k={cycle_k}",
                    row.cycle
                )));
            }
            content[cycle] = wit.cols[col_id][i];
        }
        out.insert(
            table_id,
            LutTable {
                table_id,
                k: cycle_k,
                d: cycle_d,
                n_side: 2,
                content,
            },
        );
    }
    Ok((out, cycle_d))
}

pub(super) fn build_decode_lookup_tables(
    prog_layout: &PlainMemLayout,
    prog_init_words: &HashMap<(u32, u64), F>,
) -> HashMap<u32, LutTable<F>> {
    let decode_layout = Rv32DecodeSidecarLayout::new();
    let decode_cols = riscv_decode_lookup_transport_cols(&decode_layout);
    if decode_cols.is_empty() {
        return HashMap::new();
    }
    let mut unique_table_ids: Vec<u32> = decode_cols
        .iter()
        .map(|&col_id| riscv_decode_lookup_table_id_for_col(col_id))
        .collect();
    unique_table_ids.sort_unstable();
    unique_table_ids.dedup();
    let mut out = HashMap::new();
    if unique_table_ids.len() == 1 {
        let table_id = unique_table_ids[0];
        let n_vals = decode_cols.len().max(1);
        let mut content = vec![F::ZERO; prog_layout.k * n_vals];
        for addr in 0..prog_layout.k {
            let instr_word = prog_init_words
                .get(&(PROG_ID.0, addr as u64))
                .copied()
                .unwrap_or(F::ZERO)
                .as_canonical_u64() as u32;
            let row = riscv_decode_lookup_backed_row_from_instr_word(&decode_layout, instr_word, true);
            let base = addr * n_vals;
            for (val_slot, &col_id) in decode_cols.iter().enumerate() {
                content[base + val_slot] = row[col_id];
            }
        }
        out.insert(
            table_id,
            LutTable {
                table_id,
                k: prog_layout.k,
                d: prog_layout.d,
                n_side: prog_layout.n_side,
                content,
            },
        );
        return out;
    }

    for &col_id in decode_cols.iter() {
        let table_id = riscv_decode_lookup_table_id_for_col(col_id);
        let mut content = vec![F::ZERO; prog_layout.k];
        for addr in 0..prog_layout.k {
            let instr_word = prog_init_words
                .get(&(PROG_ID.0, addr as u64))
                .copied()
                .unwrap_or(F::ZERO)
                .as_canonical_u64() as u32;
            let row = riscv_decode_lookup_backed_row_from_instr_word(&decode_layout, instr_word, true);
            content[addr] = row[col_id];
        }
        out.insert(
            table_id,
            LutTable {
                table_id,
                k: prog_layout.k,
                d: prog_layout.d,
                n_side: prog_layout.n_side,
                content,
            },
        );
    }
    out
}

pub(super) fn inject_decode_lookup_events_into_trace(
    trace: &mut VmTrace<u64, u64, u128>,
    prog_layout: &PlainMemLayout,
    prog_init_words: &HashMap<(u32, u64), F>,
) -> Result<(), PiCcsError> {
    let decode_layout = Rv32DecodeSidecarLayout::new();
    let decode_cols = riscv_decode_lookup_transport_cols(&decode_layout);
    if decode_cols.is_empty() {
        return Ok(());
    }
    for step in &mut trace.steps {
        let pc = step.pc_before;
        if (pc as usize) >= prog_layout.k {
            return Err(PiCcsError::ProtocolError(format!(
                "decode lookup PC out of program ROM layout range: pc={pc} k={}",
                prog_layout.k
            )));
        }
        let instr_word = prog_init_words
            .get(&(PROG_ID.0, pc))
            .copied()
            .unwrap_or(F::ZERO)
            .as_canonical_u64() as u32;
        let row = riscv_decode_lookup_backed_row_from_instr_word(&decode_layout, instr_word, true);
        for &col_id in &decode_cols {
            step.shout_events.push(neo_vm_trace::ShoutEvent {
                shout_id: ShoutId(riscv_decode_lookup_table_id_for_col(col_id)),
                key: u128::from(pc),
                value: row[col_id].as_canonical_u64(),
            });
        }
    }
    Ok(())
}

pub(super) fn inject_rv64_width_lookup_events_into_trace(
    trace: &mut VmTrace<u64, u64, u128>,
    exec: &RiscvExecTable,
    width_layout: &Rv64WidthSidecarLayout,
) -> Result<(), PiCcsError> {
    if trace.steps.len() > exec.rows.len() {
        return Err(PiCcsError::ProtocolError(format!(
            "RV64 width lookup injection drift: trace steps {} > exec rows {}",
            trace.steps.len(),
            exec.rows.len()
        )));
    }
    let wit = rv64_width_sidecar_witness_from_exec_table(width_layout, exec);
    let width_cols = rv64_width_lookup_backed_cols(width_layout);
    for (i, step) in trace.steps.iter_mut().enumerate() {
        let cycle = exec
            .rows
            .get(i)
            .ok_or_else(|| PiCcsError::ProtocolError("missing exec row while injecting RV64 width lookups".into()))?
            .cycle;
        for &col_id in &width_cols {
            step.shout_events.push(neo_vm_trace::ShoutEvent {
                shout_id: ShoutId(rv64_width_lookup_table_id_for_col(col_id)),
                key: u128::from(cycle),
                value: wit.cols[col_id][i].as_canonical_u64(),
            });
        }
    }
    Ok(())
}

pub(super) fn table_ell_addr_for_shared_bus(
    table_id: u32,
    table_specs: &HashMap<u32, LutTableSpec>,
    lut_tables: &HashMap<u32, LutTable<F>>,
) -> Result<usize, PiCcsError> {
    let (d, n_side) = if let Some(spec) = table_specs.get(&table_id) {
        match spec {
            LutTableSpec::RiscvOpcode { xlen, .. } => (
                xlen.checked_mul(2)
                    .ok_or_else(|| PiCcsError::InvalidInput(format!("2*xlen overflow for table_id={table_id}")))?,
                2usize,
            ),
            LutTableSpec::IdentityU32 => (32usize, 2usize),
            LutTableSpec::RiscvOpcodePacked { opcode, xlen } => {
                (neo_memory::riscv::packed::rv_packed_d(*opcode, *xlen)?, 2usize)
            }
            LutTableSpec::RiscvOpcodeEventTablePacked { .. } => {
                return Err(PiCcsError::InvalidInput(
                    "RV64 trace proving slice does not support packed event-table opcode tables".into(),
                ));
            }
        }
    } else {
        let table = lut_tables
            .get(&table_id)
            .ok_or_else(|| PiCcsError::InvalidInput(format!("missing LutTable metadata for table_id={table_id}")))?;
        (table.d, table.n_side)
    };
    if n_side == 0 || !n_side.is_power_of_two() {
        return Err(PiCcsError::InvalidInput(format!(
            "table_id={table_id} has non power-of-two n_side={n_side}"
        )));
    }
    let ell = n_side.trailing_zeros() as usize;
    d.checked_mul(ell)
        .ok_or_else(|| PiCcsError::InvalidInput("ell_addr overflow".into()))
}

pub(super) fn estimate_route_a_bus_cols(
    chunk_size: usize,
    table_specs: &HashMap<u32, LutTableSpec>,
    lut_tables: &HashMap<u32, LutTable<F>>,
    mem_layouts: &HashMap<u32, PlainMemLayout>,
    lut_lanes: &HashMap<u32, usize>,
) -> Result<usize, PiCcsError> {
    if chunk_size == 0 {
        return Err(PiCcsError::InvalidInput(
            "route-a bus estimator requires chunk_size > 0".into(),
        ));
    }
    let mut table_ids: Vec<u32> = table_specs
        .keys()
        .copied()
        .chain(lut_tables.keys().copied())
        .collect();
    table_ids.sort_unstable();
    table_ids.dedup();

    let mut shout_shapes = Vec::<ShoutInstanceShape>::with_capacity(table_ids.len());
    let mut shout_upper_cols = 0usize;
    for table_id in table_ids {
        let ell_addr = table_ell_addr_for_shared_bus(table_id, table_specs, lut_tables)?;
        let lanes = lut_lanes.get(&table_id).copied().unwrap_or(1).max(1);
        let shout_lane_cols = ell_addr
            .checked_add(1 + riscv_trace_lookup_n_vals_for_table_id(table_id).max(1))
            .ok_or_else(|| PiCcsError::InvalidInput("shout lane width overflow".into()))?;
        shout_upper_cols = shout_upper_cols
            .checked_add(
                shout_lane_cols
                    .checked_mul(lanes)
                    .ok_or_else(|| PiCcsError::InvalidInput("shout width overflow".into()))?,
            )
            .ok_or_else(|| PiCcsError::InvalidInput("bus width overflow".into()))?;
        shout_shapes.push(ShoutInstanceShape {
            ell_addr,
            lanes,
            n_vals: riscv_trace_lookup_n_vals_for_table_id(table_id).max(1),
            addr_group: trace_lookup_addr_group_for_table_shape(table_id, ell_addr),
            selector_group: riscv_trace_lookup_selector_group_for_table_id(table_id).map(|v| v as u64),
        });
    }

    let mut mem_ids: Vec<u32> = mem_layouts.keys().copied().collect();
    mem_ids.sort_unstable();
    let mut twist_shapes = Vec::<(usize, usize)>::with_capacity(mem_ids.len());
    let mut twist_upper_cols = 0usize;
    for mem_id in mem_ids {
        let layout = mem_layouts
            .get(&mem_id)
            .ok_or_else(|| PiCcsError::InvalidInput(format!("missing mem layout for mem_id={mem_id}")))?;
        if layout.n_side == 0 || !layout.n_side.is_power_of_two() {
            return Err(PiCcsError::InvalidInput(format!(
                "mem_id={mem_id} has non power-of-two n_side={}",
                layout.n_side
            )));
        }
        let ell = layout.n_side.trailing_zeros() as usize;
        let ell_addr = layout
            .d
            .checked_mul(ell)
            .ok_or_else(|| PiCcsError::InvalidInput("twist ell_addr overflow".into()))?;
        let twist_lane_cols = ell_addr
            .checked_mul(2)
            .and_then(|v| v.checked_add(5))
            .ok_or_else(|| PiCcsError::InvalidInput("twist lane width overflow".into()))?;
        let lanes = layout.lanes.max(1);
        twist_upper_cols = twist_upper_cols
            .checked_add(
                twist_lane_cols
                    .checked_mul(lanes)
                    .ok_or_else(|| PiCcsError::InvalidInput("twist width overflow".into()))?,
            )
            .ok_or_else(|| PiCcsError::InvalidInput("bus width overflow".into()))?;
        twist_shapes.push((ell_addr, lanes));
    }
    let upper_bus_cols = shout_upper_cols
        .checked_add(twist_upper_cols)
        .ok_or_else(|| PiCcsError::InvalidInput("bus width overflow".into()))?;
    let m_probe = upper_bus_cols
        .checked_mul(chunk_size)
        .ok_or_else(|| PiCcsError::InvalidInput("route-a bus estimator m_probe overflow".into()))?;
    let layout = build_bus_layout_for_instances_with_shout_shapes_and_twist_lanes(
        m_probe,
        0usize,
        chunk_size,
        shout_shapes,
        twist_shapes,
    )
    .map_err(PiCcsError::InvalidInput)?;
    Ok(layout.bus_cols)
}

pub(super) fn trace_lookup_addr_group_for_table_shape(table_id: u32, ell_addr: usize) -> Option<u64> {
    if table_id <= 19 && ell_addr == RV64_OPCODE_ELL_ADDR {
        riscv_trace_lookup_addr_group_for_table_id(table_id).map(|v| v as u64)
    } else if table_id <= 19 && ell_addr == 66 {
        Some(RV64_PACKED_MUL_ADDR_GROUP)
    } else {
        riscv_trace_lookup_addr_group_for_table_id(table_id)
            .filter(|_| ell_addr != RV64_OPCODE_ELL_ADDR && ell_addr != 66)
            .map(|v| v as u64)
    }
}

pub(super) fn inject_exact_reg_writes_into_trace(trace: &mut VmTrace<u64, u64, u128>) -> Result<(), PiCcsError> {
    for step in &mut trace.steps {
        let reg_writes: Vec<(u64, u64)> = step
            .twist_events
            .iter()
            .filter(|event| {
                event.twist_id == REG_ID && event.kind == neo_vm_trace::TwistOpKind::Write && event.addr < 32
            })
            .map(|event| (event.addr, event.value))
            .collect();
        if reg_writes.len() > 1 {
            return Err(PiCcsError::InvalidInput(format!(
                "RV64 exact register output binding supports at most one architectural register write per step (cycle {})",
                step.cycle
            )));
        }
        if let Some((reg, value)) = reg_writes.first().copied() {
            let lo = value & 0xffff_ffff;
            let hi = value >> 32;
            let hi_addr = reg.checked_add(32).ok_or_else(|| {
                PiCcsError::InvalidInput(format!(
                    "RV64 exact register output write address overflow at cycle {}: reg={reg}",
                    step.cycle
                ))
            })?;
            step.twist_events.push(neo_vm_trace::TwistEvent {
                twist_id: REG_EXACT_ID,
                kind: neo_vm_trace::TwistOpKind::Write,
                addr: reg,
                value: lo,
                lane: Some(0),
            });
            step.twist_events.push(neo_vm_trace::TwistEvent {
                twist_id: REG_EXACT_ID,
                kind: neo_vm_trace::TwistOpKind::Write,
                addr: hi_addr,
                value: hi,
                lane: Some(1),
            });
        }
    }
    Ok(())
}

pub(super) fn final_reg_state_dense_injective(
    exec: &RiscvExecTable,
    reg_init: &HashMap<u64, u64>,
    k: usize,
) -> Result<Vec<F>, PiCcsError> {
    let mut regs = vec![0u64; k];
    validate_rv64_reg_init_words(reg_init)?;
    for (&reg, &value) in reg_init {
        regs[reg as usize] = value;
    }
    regs[0] = 0;
    for r in exec.rows.iter().filter(|r| r.active) {
        if let Some(w) = &r.reg_write_lane0 {
            let addr_usize = usize::try_from(w.addr).map_err(|_| {
                PiCcsError::InvalidInput(format!(
                    "trace register write addr does not fit usize at cycle {}: addr={}",
                    r.cycle, w.addr
                ))
            })?;
            if addr_usize >= k {
                return Err(PiCcsError::InvalidInput(format!(
                    "trace register write addr out of range at cycle {}: addr={}",
                    r.cycle, w.addr
                )));
            }
            if w.addr == 0 {
                return Err(PiCcsError::InvalidInput(format!(
                    "trace writes x0 at cycle {} which is invalid",
                    r.cycle
                )));
            }
            regs[addr_usize] = w.value;
            regs[0] = 0;
        }
    }
    regs.into_iter()
        .map(|value| {
            field_from_u64_injective(value, "final_reg").map_err(|e| {
                PiCcsError::InvalidInput(format!(
                    "{e}; for RV64 exact public register outputs use reg_output_claim_exact_u64(...)"
                ))
            })
        })
        .collect()
}

pub(super) fn final_shared_mem_state_dense(
    aux: Option<&neo_memory::builder::ShardWitnessAux>,
    mem_id: u32,
    k: usize,
    num_bits: usize,
) -> Result<Vec<F>, PiCcsError> {
    if num_bits > neo_memory::output_check::OUTPUT_SUMCHECK_MAX_NUM_BITS {
        return Ok(Vec::new());
    }
    let aux =
        aux.ok_or_else(|| PiCcsError::InvalidInput("RV64 RAM output binding requires shared-bus aux state".into()))?;
    let mut out = vec![F::ZERO; k];
    if let Some(state) = aux.final_mem_states.get(&mem_id) {
        for (&addr, &value) in state {
            let addr_usize = usize::try_from(addr).map_err(|_| {
                PiCcsError::InvalidInput(format!(
                    "RV64 RAM output binding address does not fit usize: addr={addr}"
                ))
            })?;
            if addr_usize >= k {
                return Err(PiCcsError::InvalidInput(format!(
                    "RV64 RAM output binding address out of range: addr={addr} >= k={k}"
                )));
            }
            out[addr_usize] = value;
        }
    }
    Ok(out)
}

pub(super) fn profile_err_to_piccs(err: RiscvProofProfileError) -> PiCcsError {
    PiCcsError::InvalidInput(format!("RV64 proof profile validation failed: {err}"))
}

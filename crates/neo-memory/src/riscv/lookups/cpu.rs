use neo_vm_trace::{Shout, Twist};
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use p3_goldilocks::Goldilocks;
use p3_symmetric::Permutation;
use std::collections::{HashMap, VecDeque};

use crate::riscv::instruction::{compute_op, decomposition_sequence_for_instruction, encode_lookup_key, DecomposedOp};

use super::decode::decode_instruction;
use super::encode::encode_instruction;
use super::isa::{BranchCondition, RiscvInstruction, RiscvMemOp, RiscvOpcode};
use super::tables::RiscvShoutTables;

#[derive(Clone, Debug)]
struct PendingDecomposition {
    instr_word: u32,
    ops: VecDeque<DecomposedOp>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PoseidonMode {
    Absorbing,
    Finalized,
}

#[derive(Clone)]
struct PoseidonPrecompileCtx {
    state: [Goldilocks; neo_ccs::crypto::poseidon2_goldilocks::WIDTH],
    absorb_cursor: usize,
    mode: PoseidonMode,
    final_digest_words: [u32; neo_ccs::crypto::poseidon2_goldilocks::DIGEST_LEN * 2],
    perm: &'static p3_goldilocks::Poseidon2Goldilocks<{ neo_ccs::crypto::poseidon2_goldilocks::WIDTH }>,
}

impl PoseidonPrecompileCtx {
    fn new() -> Self {
        Self {
            state: [Goldilocks::ZERO; neo_ccs::crypto::poseidon2_goldilocks::WIDTH],
            absorb_cursor: 0,
            mode: PoseidonMode::Absorbing,
            final_digest_words: [0; neo_ccs::crypto::poseidon2_goldilocks::DIGEST_LEN * 2],
            perm: neo_ccs::crypto::poseidon2_goldilocks::permutation(),
        }
    }

    fn reset_message(&mut self) {
        self.state.fill(Goldilocks::ZERO);
        self.absorb_cursor = 0;
        self.mode = PoseidonMode::Absorbing;
        self.final_digest_words.fill(0);
    }

    fn permute(&mut self) {
        self.state = self.perm.permute(self.state);
    }

    fn absorb_elem(&mut self, elem: Goldilocks) {
        if self.mode == PoseidonMode::Finalized {
            // Starting a new message after a finalized digest.
            self.reset_message();
        }

        self.state[self.absorb_cursor] += elem;
        self.absorb_cursor += 1;

        if self.absorb_cursor == neo_ccs::crypto::poseidon2_goldilocks::RATE {
            self.permute();
            self.absorb_cursor = 0;
        }
    }

    fn finalize(&mut self) -> Result<(), String> {
        if self.mode == PoseidonMode::Finalized {
            return Err("poseidon2 precompile: finalize called in Finalized mode".into());
        }

        // Match poseidon2_hash semantics exactly:
        // - if there is a partial block, permute once
        // - then add 1 to state[0] and permute again
        if self.absorb_cursor > 0 {
            self.permute();
            self.absorb_cursor = 0;
        }

        self.state[0] += Goldilocks::ONE;
        self.permute();

        for i in 0..neo_ccs::crypto::poseidon2_goldilocks::DIGEST_LEN {
            let v = self.state[i].as_canonical_u64();
            self.final_digest_words[2 * i] = v as u32;
            self.final_digest_words[2 * i + 1] = (v >> 32) as u32;
        }
        self.mode = PoseidonMode::Finalized;
        Ok(())
    }

    fn squeeze_word(&self, idx: u8) -> Result<u32, String> {
        if self.mode != PoseidonMode::Finalized {
            return Err("poseidon2 precompile: squeeze called before finalize".into());
        }
        let i = idx as usize;
        if i >= self.final_digest_words.len() {
            return Err(format!(
                "poseidon2 precompile: squeeze idx out of range (idx={}, max={})",
                idx,
                self.final_digest_words.len() - 1
            ));
        }
        Ok(self.final_digest_words[i])
    }
}

/// A RISC-V CPU that can be traced using Neo's VmCpu trait.
///
/// Implements RV32I/RV64I base instruction set.
/// Based on Jolt's CPU implementation (MIT/Apache-2.0 license).
/// Credit: <https://github.com/a16z/jolt>
pub struct RiscvCpu {
    /// Program counter.
    pub pc: u64,
    /// General-purpose registers (x0-x31, where x0 is always 0).
    pub regs: [u64; 32],
    /// Word size in bits (32 or 64).
    pub xlen: usize,
    /// Whether the CPU has halted.
    pub halted: bool,
    /// Program to execute (list of instructions).
    program: Vec<RiscvInstruction>,
    /// Base address of the program.
    program_base: u64,
    /// Enable runtime Jolt-style decomposition rows for selected opcodes.
    enable_runtime_decomposition: bool,
    /// Pending decomposed steps for the currently fetched architectural instruction.
    pending_decomposition: Option<PendingDecomposition>,
    /// Virtual register values for the currently executing decomposition sequence.
    virtual_regs: HashMap<u64, u64>,
    /// Single-slot Poseidon2 precompile state machine.
    poseidon2_ctx: PoseidonPrecompileCtx,
}

impl RiscvCpu {
    /// Create a new CPU with the given word size.
    pub fn new(xlen: usize) -> Self {
        assert!(xlen == 32 || xlen == 64);
        Self {
            pc: 0,
            regs: [0; 32],
            xlen,
            halted: false,
            program: Vec::new(),
            program_base: 0,
            enable_runtime_decomposition: false,
            pending_decomposition: None,
            virtual_regs: HashMap::new(),
            poseidon2_ctx: PoseidonPrecompileCtx::new(),
        }
    }

    /// Load a program starting at the given base address.
    pub fn load_program(&mut self, base: u64, program: Vec<RiscvInstruction>) {
        self.program_base = base;
        self.program = program;
        self.pc = base;
        self.pending_decomposition = None;
        self.virtual_regs.clear();
    }

    /// Opt into runtime decomposition row emission (off by default).
    pub fn set_runtime_decomposition_enabled(&mut self, enabled: bool) {
        self.enable_runtime_decomposition = enabled;
        if !enabled {
            self.pending_decomposition = None;
            self.virtual_regs.clear();
        }
    }

    /// Set a register value (x0 writes are ignored).
    pub fn set_reg(&mut self, reg: u8, value: u64) {
        if reg != 0 {
            self.regs[reg as usize] = self.mask_value(value);
        }
    }

    /// Get a register value.
    pub fn get_reg(&self, reg: u8) -> u64 {
        self.regs[reg as usize]
    }

    /// Mask a value to the word size.
    fn mask_value(&self, value: u64) -> u64 {
        if self.xlen == 32 {
            value as u32 as u64
        } else {
            value
        }
    }

    /// Sign-extend an immediate.
    fn sign_extend_imm(&self, imm: i32) -> u64 {
        if self.xlen == 32 {
            imm as u32 as u64
        } else {
            imm as i64 as u64
        }
    }

    /// Get the current instruction (if any).
    fn current_instruction(&self) -> Option<&RiscvInstruction> {
        let index = (self.pc - self.program_base) / 4;
        self.program.get(index as usize)
    }

    fn handle_ecall(&mut self) {
        self.halted = true;
    }

    fn write_reg<T: Twist<u64, u64>>(&mut self, twist: &mut T, reg: u8, value: u64) {
        self.write_reg_addr(twist, reg as u64, value);
    }

    fn write_reg_addr<T: Twist<u64, u64>>(&mut self, twist: &mut T, addr: u64, value: u64) {
        if addr == 0 {
            return;
        }
        let masked = self.mask_value(value);
        twist.store_lane(super::REG_ID, addr, masked, /*lane=*/ 0);
        if addr < 32 {
            self.regs[addr as usize] = masked;
        } else {
            self.virtual_regs.insert(addr, masked);
        }
        self.regs[0] = 0;
    }

    fn write_reg_addr_checked<T: Twist<u64, u64>>(
        &mut self,
        twist: &mut T,
        addr: u64,
        value: u64,
        is_virtual: bool,
    ) -> Result<(), String> {
        if is_virtual && addr < 32 {
            return Err(format!(
                "virtual decomposition row attempted architectural register write: addr={addr}"
            ));
        }
        self.write_reg_addr(twist, addr, value);
        Ok(())
    }

    fn read_reg_addr<T: Twist<u64, u64>>(&mut self, twist: &mut T, addr: u64, lane: u32) -> u64 {
        let value = self.mask_value(twist.load_lane(super::REG_ID, addr, lane));
        if addr != 0 && addr < 32 {
            self.regs[addr as usize] = value;
        }
        self.regs[0] = 0;
        value
    }

    #[inline]
    fn lookup_key_for_opcode(&self, op: RiscvOpcode, lhs: u64, rhs: u64) -> u64 {
        encode_lookup_key(op, lhs, rhs, self.xlen)
    }

    fn start_decomposition_for_instruction(&mut self, instr_word: u32, instr: &RiscvInstruction) -> Option<()> {
        if self.xlen != 32 {
            return None;
        }

        let seq = decomposition_sequence_for_instruction(instr)?;

        self.virtual_regs.clear();
        self.pending_decomposition = Some(PendingDecomposition {
            instr_word,
            ops: VecDeque::from(seq),
        });
        Some(())
    }

    fn pop_pending_decomposition_step(
        &mut self,
        instr_word: u32,
    ) -> Result<Option<(DecomposedOp, bool, Option<u32>, u64)>, String> {
        let Some(mut pending) = self.pending_decomposition.take() else {
            return Ok(None);
        };

        if pending.instr_word != instr_word {
            return Err(format!(
                "decomposition PC drift: expected instr_word={:#x}, got {:#x}",
                pending.instr_word, instr_word
            ));
        }

        let op = pending
            .ops
            .pop_front()
            .ok_or_else(|| "pending decomposition has no remaining ops".to_string())?;
        let remaining = pending.ops.len() as u32;
        if remaining > 0 {
            self.pending_decomposition = Some(pending);
            Ok(Some((op, true, Some(remaining), self.pc)))
        } else {
            Ok(Some((op, false, None, self.pc.wrapping_add(4))))
        }
    }

    fn finalize_decomposition_step(
        &mut self,
        opcode: u32,
        is_virtual: bool,
        virtual_sequence_remaining: Option<u32>,
        next_pc: u64,
    ) -> Result<neo_vm_trace::StepMeta<u64>, String> {
        let next_pc_masked = self.mask_value(next_pc);
        if (next_pc_masked & 0b11) != 0 {
            return Err(format!(
                "control-flow target not 4-byte aligned (no compressed): next_pc={:#x}",
                next_pc_masked
            ));
        }
        self.pc = next_pc_masked;
        if !is_virtual {
            self.virtual_regs.clear();
        }
        Ok(neo_vm_trace::StepMeta {
            pc_after: self.pc,
            opcode,
            is_virtual,
            virtual_sequence_remaining,
        })
    }

    fn try_execute_one_pending_decomposition_step<T, S>(
        &mut self,
        instr_word: u32,
        instr: &RiscvInstruction,
        twist: &mut T,
        shout: &mut S,
        shout_tables: &RiscvShoutTables,
    ) -> Result<Option<neo_vm_trace::StepMeta<u64>>, String>
    where
        T: Twist<u64, u64>,
        S: Shout<u64>,
    {
        let Some((op, is_virtual, virtual_sequence_remaining, next_pc)) =
            self.pop_pending_decomposition_step(instr_word)?
        else {
            return Ok(None);
        };
        self.execute_decomposed_op(op, is_virtual, instr, twist, shout, shout_tables)?;
        Ok(Some(self.finalize_decomposition_step(
            instr_word,
            is_virtual,
            virtual_sequence_remaining,
            next_pc,
        )?))
    }

    fn exec_lookup_write_decomposed<T, S>(
        &mut self,
        twist: &mut T,
        shout: &mut S,
        shout_tables: &RiscvShoutTables,
        op: RiscvOpcode,
        dst: u64,
        lhs: u64,
        rhs: u64,
        is_virtual: bool,
    ) -> Result<(), String>
    where
        T: Twist<u64, u64>,
        S: Shout<u64>,
    {
        let lhs_val = self.read_reg_addr(twist, lhs, /*lane=*/ 0);
        let rhs_val = self.read_reg_addr(twist, rhs, /*lane=*/ 1);
        let shout_id = shout_tables.opcode_to_id(op);
        let key = self.lookup_key_for_opcode(op, lhs_val, rhs_val);
        let out = shout.lookup(shout_id, key);
        self.write_reg_addr_checked(twist, dst, out, is_virtual)
    }

    fn execute_decomposed_op<T, S>(
        &mut self,
        op: DecomposedOp,
        is_virtual: bool,
        instr: &RiscvInstruction,
        twist: &mut T,
        shout: &mut S,
        shout_tables: &RiscvShoutTables,
    ) -> Result<(), String>
    where
        T: Twist<u64, u64>,
        S: Shout<u64>,
    {
        match op {
            DecomposedOp::Advice { dst } => {
                let _ = self.read_reg_addr(twist, 0, /*lane=*/ 0);
                let _ = self.read_reg_addr(twist, 0, /*lane=*/ 1);
                self.write_reg_addr_checked(twist, dst, 0, is_virtual)?;
            }
            DecomposedOp::AdviceRemainderAbs { dst, dividend, divisor } => {
                let dividend_val = self.read_reg_addr(twist, dividend, /*lane=*/ 0);
                let divisor_val = self.read_reg_addr(twist, divisor, /*lane=*/ 1);
                let rem = compute_op(RiscvOpcode::Rem, dividend_val, divisor_val, self.xlen);
                let rem_abs = if self.xlen == 32 {
                    (rem as u32 as i32).unsigned_abs() as u64
                } else {
                    (rem as i64).unsigned_abs()
                };
                self.write_reg_addr_checked(twist, dst, rem_abs, is_virtual)?;
            }
            DecomposedOp::AdviceQuotient { dst, op, lhs, rhs } => {
                self.exec_lookup_write_decomposed(twist, shout, shout_tables, op, dst, lhs, rhs, is_virtual)?;
            }
            DecomposedOp::MovSign { dst, src } => {
                let x = self.read_reg_addr(twist, src, /*lane=*/ 0);
                let _ = self.read_reg_addr(twist, 0, /*lane=*/ 1);
                let sign_bit = if self.xlen == 32 { 31u64 } else { 63u64 };
                let shout_id = shout_tables.opcode_to_id(RiscvOpcode::Sra);
                let key = self.lookup_key_for_opcode(RiscvOpcode::Sra, x, sign_bit);
                let sign_mask = shout.lookup(shout_id, key);
                self.write_reg_addr_checked(twist, dst, sign_mask, is_virtual)?;
            }
            DecomposedOp::Move { dst, src } => {
                if !is_virtual && src >= 32 {
                    // Commit rows must stay architecturally shaped (rs1/rs2 lane reads + canonical Shout).
                    // The actual writeback value comes from the virtual accumulator.
                    let (op, rs1, rs2) = match instr {
                        RiscvInstruction::RAlu { op, rs1, rs2, .. } => (*op, *rs1 as u64, *rs2 as u64),
                        other => {
                            return Err(format!(
                                "non-virtual virtual-move commit requires RAlu context (got {other:?})"
                            ));
                        }
                    };
                    let lhs_val = self.read_reg_addr(twist, rs1, /*lane=*/ 0);
                    let rhs_val = self.read_reg_addr(twist, rs2, /*lane=*/ 1);
                    let shout_id = shout_tables.opcode_to_id(op);
                    let key = self.lookup_key_for_opcode(op, lhs_val, rhs_val);
                    let shout_out = shout.lookup(shout_id, key);
                    let x = *self
                        .virtual_regs
                        .get(&src)
                        .ok_or_else(|| format!("missing virtual accumulator value for commit src={src}"))?;
                    if x != shout_out {
                        return Err(format!(
                            "virtual commit mismatch for {op:?}: v_acc={x:#x}, shout={shout_out:#x}"
                        ));
                    }
                    self.write_reg_addr_checked(twist, dst, x, is_virtual)?;
                } else {
                    let x = self.read_reg_addr(twist, src, /*lane=*/ 0);
                    let _ = self.read_reg_addr(twist, 0, /*lane=*/ 1);
                    self.write_reg_addr_checked(twist, dst, x, is_virtual)?;
                }
            }
            DecomposedOp::Add { dst, lhs, rhs } => {
                self.exec_lookup_write_decomposed(
                    twist,
                    shout,
                    shout_tables,
                    RiscvOpcode::Add,
                    dst,
                    lhs,
                    rhs,
                    is_virtual,
                )?;
            }
            DecomposedOp::Sub { dst, lhs, rhs } => {
                self.exec_lookup_write_decomposed(
                    twist,
                    shout,
                    shout_tables,
                    RiscvOpcode::Sub,
                    dst,
                    lhs,
                    rhs,
                    is_virtual,
                )?;
            }
            DecomposedOp::Xor { dst, lhs, rhs } => {
                self.exec_lookup_write_decomposed(
                    twist,
                    shout,
                    shout_tables,
                    RiscvOpcode::Xor,
                    dst,
                    lhs,
                    rhs,
                    is_virtual,
                )?;
            }
            DecomposedOp::Mul { dst, lhs, rhs } => {
                self.exec_lookup_write_decomposed(
                    twist,
                    shout,
                    shout_tables,
                    RiscvOpcode::Mul,
                    dst,
                    lhs,
                    rhs,
                    is_virtual,
                )?;
            }
            DecomposedOp::Mulhu { dst, lhs, rhs } => {
                self.exec_lookup_write_decomposed(
                    twist,
                    shout,
                    shout_tables,
                    RiscvOpcode::Mulhu,
                    dst,
                    lhs,
                    rhs,
                    is_virtual,
                )?;
            }
            DecomposedOp::AssertEq { lhs, rhs } => {
                let lhs_val = self.read_reg_addr(twist, lhs, /*lane=*/ 0);
                let rhs_val = self.read_reg_addr(twist, rhs, /*lane=*/ 1);
                let shout_id = shout_tables.opcode_to_id(RiscvOpcode::Eq);
                let key = self.lookup_key_for_opcode(RiscvOpcode::Eq, lhs_val, rhs_val);
                let eq = shout.lookup(shout_id, key);
                if eq != 1 {
                    return Err(format!("virtual assert-eq failed: lhs={lhs_val:#x}, rhs={rhs_val:#x}"));
                }
            }
            DecomposedOp::AssertLtu { lhs, rhs } => {
                let lhs_val = self.read_reg_addr(twist, lhs, /*lane=*/ 0);
                let rhs_val = self.read_reg_addr(twist, rhs, /*lane=*/ 1);
                let shout_id = shout_tables.opcode_to_id(RiscvOpcode::Sltu);
                let key = self.lookup_key_for_opcode(RiscvOpcode::Sltu, lhs_val, rhs_val);
                let lt = shout.lookup(shout_id, key);
                if lt != 1 {
                    return Err(format!("virtual assert-ltu failed: lhs={lhs_val:#x}, rhs={rhs_val:#x}"));
                }
            }
            DecomposedOp::AssertLte { lhs, rhs } => {
                let lhs_val = self.read_reg_addr(twist, lhs, /*lane=*/ 0);
                let rhs_val = self.read_reg_addr(twist, rhs, /*lane=*/ 1);
                let shout_id = shout_tables.opcode_to_id(RiscvOpcode::Sltu);
                let key = self.lookup_key_for_opcode(RiscvOpcode::Sltu, lhs_val, rhs_val);
                let lt = shout.lookup(shout_id, key);
                if lhs_val > rhs_val || (lhs_val < rhs_val && lt != 1) {
                    return Err(format!("virtual assert-lte failed: lhs={lhs_val:#x}, rhs={rhs_val:#x}"));
                }
            }
            DecomposedOp::AssertLtAbs { lhs, rhs } => {
                let lhs_val = self.read_reg_addr(twist, lhs, /*lane=*/ 0);
                let rhs_val = self.read_reg_addr(twist, rhs, /*lane=*/ 1);
                let lhs_abs = if self.xlen == 32 {
                    (lhs_val as u32 as i32).unsigned_abs() as u64
                } else {
                    (lhs_val as i64).unsigned_abs()
                };
                let rhs_abs = if self.xlen == 32 {
                    (rhs_val as u32 as i32).unsigned_abs() as u64
                } else {
                    (rhs_val as i64).unsigned_abs()
                };
                if lhs_abs >= rhs_abs {
                    return Err(format!(
                        "virtual assert-lt-abs failed: |lhs|={lhs_abs:#x}, |rhs|={rhs_abs:#x}"
                    ));
                }
            }
            DecomposedOp::AssertEqSigns { lhs, rhs } => {
                let lhs_val = self.read_reg_addr(twist, lhs, /*lane=*/ 0);
                let rhs_val = self.read_reg_addr(twist, rhs, /*lane=*/ 1);
                let sign_bit = if self.xlen == 32 { 31 } else { 63 };
                let lhs_sign = (lhs_val >> sign_bit) & 1;
                let rhs_sign = (rhs_val >> sign_bit) & 1;
                if lhs_sign != rhs_sign {
                    return Err(format!(
                        "virtual assert-eq-signs failed: lhs_sign={lhs_sign}, rhs_sign={rhs_sign}"
                    ));
                }
            }
            DecomposedOp::AssertValidDiv0 { divisor, quotient } => {
                let divisor_val = self.read_reg_addr(twist, divisor, /*lane=*/ 0);
                let quotient_val = self.read_reg_addr(twist, quotient, /*lane=*/ 1);
                let all_ones = self.mask_value(u64::MAX);
                let shout_id = shout_tables.opcode_to_id(RiscvOpcode::Eq);
                let key = self.lookup_key_for_opcode(RiscvOpcode::Eq, divisor_val, 0);
                let divisor_is_zero = shout.lookup(shout_id, key);
                if divisor_is_zero > 1 {
                    return Err(format!(
                        "virtual assert-valid-div0 failed: expected eq(divisor,0) lookup to be boolean, got={divisor_is_zero}"
                    ));
                }
                if divisor_is_zero == 1 && quotient_val != all_ones {
                    return Err(format!(
                        "virtual assert-valid-div0 failed: divisor=0, quotient={quotient_val:#x}, expected={all_ones:#x}"
                    ));
                }
            }
            DecomposedOp::ChangeDivisor { dst, dividend, divisor } => {
                let dividend_val = self.read_reg_addr(twist, dividend, /*lane=*/ 0);
                let divisor_val = self.read_reg_addr(twist, divisor, /*lane=*/ 1);
                let out = if self.xlen == 32 {
                    let dividend_i = dividend_val as u32 as i32;
                    let divisor_i = divisor_val as u32 as i32;
                    if dividend_i == i32::MIN && divisor_i == -1 {
                        1u64
                    } else {
                        divisor_val
                    }
                } else {
                    let dividend_i = dividend_val as i64;
                    let divisor_i = divisor_val as i64;
                    if dividend_i == i64::MIN && divisor_i == -1 {
                        1u64
                    } else {
                        divisor_val
                    }
                };
                self.write_reg_addr_checked(twist, dst, out, is_virtual)?;
            }
            DecomposedOp::AssertMulUNoOverflow { lhs, rhs } => {
                let lhs_val = self.read_reg_addr(twist, lhs, /*lane=*/ 0);
                let rhs_val = self.read_reg_addr(twist, rhs, /*lane=*/ 1);
                let shout_id = shout_tables.opcode_to_id(RiscvOpcode::Mulhu);
                let key = self.lookup_key_for_opcode(RiscvOpcode::Mulhu, lhs_val, rhs_val);
                let hi = shout.lookup(shout_id, key);
                if hi != 0 {
                    return Err(format!(
                        "virtual assert-mulu-no-overflow failed: lhs={lhs_val:#x}, rhs={rhs_val:#x}, hi={hi:#x}"
                    ));
                }
            }
            DecomposedOp::AssertValidUnsignedRemainder { remainder, divisor } => {
                let remainder_val = self.read_reg_addr(twist, remainder, /*lane=*/ 0);
                let divisor_val = self.read_reg_addr(twist, divisor, /*lane=*/ 1);
                let shout_id = shout_tables.opcode_to_id(RiscvOpcode::Sltu);
                let key = self.lookup_key_for_opcode(RiscvOpcode::Sltu, remainder_val, divisor_val);
                let lt = shout.lookup(shout_id, key);
                if lt > 1 {
                    return Err(format!(
                        "virtual assert-valid-urem failed: expected sltu(remainder,divisor) lookup to be boolean, got={lt}"
                    ));
                }
                if divisor_val != 0 && lt != 1 {
                    return Err(format!(
                        "virtual assert-valid-urem failed: remainder={remainder_val:#x}, divisor={divisor_val:#x}"
                    ));
                }
            }
        }

        Ok(())
    }
}

impl neo_vm_trace::VmCpu<u64, u64> for RiscvCpu {
    type Error = String;

    fn snapshot_regs(&self) -> Vec<u64> {
        self.regs.to_vec()
    }

    fn pc(&self) -> u64 {
        self.pc
    }

    fn halted(&self) -> bool {
        self.halted
    }

    fn step<T, S>(&mut self, twist: &mut T, shout: &mut S) -> Result<neo_vm_trace::StepMeta<u64>, Self::Error>
    where
        T: Twist<u64, u64>,
        S: Shout<u64>,
    {
        if (self.pc & 0b11) != 0 {
            return Err(format!("PC not 4-byte aligned (no compressed): pc={:#x}", self.pc));
        }

        let ram = super::RAM_ID;
        let prog = super::PROG_ID;
        let shout_tables = RiscvShoutTables::new(self.xlen);
        let add_shout_id = shout_tables.opcode_to_id(RiscvOpcode::Add);

        let instr_word = twist.load(prog, self.pc);
        let instr_word_u32 = u32::try_from(instr_word).map_err(|_| {
            format!(
                "Instruction word at PC {:#x} does not fit in 32 bits: {:#x}",
                self.pc, instr_word
            )
        })?;
        if (instr_word_u32 & 0b11) != 0b11 {
            return Err(format!(
                "Compressed instructions not supported (PC {:#x}, word {:#x})",
                self.pc, instr_word_u32
            ));
        }

        if let Some(expected) = self.current_instruction() {
            let expected_word = encode_instruction(expected);
            if expected_word != instr_word_u32 {
                return Err(format!(
                    "Program ROM mismatch at PC {:#x}: expected {:#x}, got {:#x}",
                    self.pc, expected_word, instr_word_u32
                ));
            }
        }

        let instr = decode_instruction(instr_word_u32).map_err(|e| {
            format!(
                "Failed to decode instruction at PC {:#x} (word {:#x}): {e}",
                self.pc, instr_word_u32
            )
        })?;

        if self.enable_runtime_decomposition {
            if let Some(meta) =
                self.try_execute_one_pending_decomposition_step(instr_word_u32, &instr, twist, shout, &shout_tables)?
            {
                return Ok(meta);
            }

            if self
                .start_decomposition_for_instruction(instr_word_u32, &instr)
                .is_some()
            {
                if let Some(meta) = self.try_execute_one_pending_decomposition_step(
                    instr_word_u32,
                    &instr,
                    twist,
                    shout,
                    &shout_tables,
                )? {
                    return Ok(meta);
                }
            }
        }

        // --------------------------------------------------------------------
        // Regfile-as-Twist (REG_ID): always emit two register reads per step.
        //
        // Lane assignment (RV32 trace convention):
        // - lane 0: read rs1_field
        // - lane 1: read rs2_field
        // --------------------------------------------------------------------
        let reg = super::REG_ID;
        let rs1_field = ((instr_word_u32 >> 15) & 0x1f) as u64;
        let rs2_field = ((instr_word_u32 >> 20) & 0x1f) as u64;
        let rs2_addr = rs2_field;

        let rs1_val = self.mask_value(twist.load_lane(reg, rs1_field, /*lane=*/ 0));
        let rs2_val = self.mask_value(twist.load_lane(reg, rs2_addr, /*lane=*/ 1));

        // Keep the CPU's register snapshot mirror consistent with Twist state.
        self.regs[0] = 0;
        if rs1_field != 0 {
            self.regs[rs1_field as usize] = rs1_val;
        }
        if rs2_addr != 0 {
            self.regs[rs2_addr as usize] = rs2_val;
        }

        // Default: advance PC by 4
        let mut next_pc = self.pc.wrapping_add(4);
        let step_opcode: u32 = instr_word_u32;

        match instr {
            RiscvInstruction::RAlu { op, rd, rs1: _, rs2: _ } => {
                let shout_id = shout_tables.opcode_to_id(op);
                let index = self.lookup_key_for_opcode(op, rs1_val, rs2_val);
                let result = shout.lookup(shout_id, index);
                self.write_reg(twist, rd, result);
            }

            RiscvInstruction::IAlu { op, rd, rs1: _, imm } => {
                let imm_val = self.sign_extend_imm(imm);

                // Use Shout for the ALU operation
                let shout_id = shout_tables.opcode_to_id(op);
                let index = self.lookup_key_for_opcode(op, rs1_val, imm_val);
                let result = shout.lookup(shout_id, index);

                self.write_reg(twist, rd, result);
            }

            RiscvInstruction::Load { op, rd, rs1: _, imm } => {
                let base = rs1_val;
                let imm_val = self.sign_extend_imm(imm);
                let index = self.lookup_key_for_opcode(RiscvOpcode::Add, base, imm_val);
                let addr = shout.lookup(add_shout_id, index);

                // Twist RAM semantics (RV32 trace mode):
                // - Memory is byte-addressed, and `addr` is a byte address.
                // - Twist accesses are word-valued (XLEN bits), i.e. a `load/store` at `addr`
                //   reads/writes the little-endian word window starting at `addr`.
                // - Sub-word ops (LB/LH/LBU/LHU) read the word window at `addr` and then mask/sign-extend
                //   the low byte/halfword.
                //
                // NOTE: This matches the proof-layer convention in `riscv::ccs` and keeps the trace at
                // "≤ 1 Twist read per instruction".
                let raw_value = twist.load(ram, addr);

                // Apply width and sign extension
                let width = op.width_bytes();
                let mask = if width >= 8 {
                    u64::MAX
                } else {
                    (1u64 << (width * 8)) - 1
                };
                let value = raw_value & mask;

                // Sign-extend if needed
                let result = if op.is_sign_extend() {
                    match width {
                        1 => (value as u8) as i8 as i64 as u64,
                        2 => (value as u16) as i16 as i64 as u64,
                        4 => (value as u32) as i32 as i64 as u64,
                        _ => value,
                    }
                } else {
                    value
                };

                self.write_reg(twist, rd, result);
            }

            RiscvInstruction::Store {
                op,
                rs1: _,
                rs2: _,
                imm,
            } => {
                let base = rs1_val;
                let imm_val = self.sign_extend_imm(imm);
                let index = self.lookup_key_for_opcode(RiscvOpcode::Add, base, imm_val);
                let addr = shout.lookup(add_shout_id, index);
                let value = rs2_val;

                // Mask value to store width
                let width = op.width_bytes();
                let mask = if width >= 8 {
                    u64::MAX
                } else {
                    (1u64 << (width * 8)) - 1
                };
                let store_value = value & mask;

                // Twist RAM semantics: see the comment in the Load arm above.
                //
                // For SB/SH, implement a read-modify-write on the word window at `addr`,
                // updating only the low 8/16 bits.
                if width < (self.xlen / 8) {
                    let raw_value = twist.load(ram, addr);
                    let merged = (raw_value & !mask) | store_value;
                    twist.store(ram, addr, merged);
                } else {
                    twist.store(ram, addr, store_value);
                }
            }

            RiscvInstruction::Branch {
                cond,
                rs1: _,
                rs2: _,
                imm,
            } => {
                // Use Shout for the comparison
                let cmp_opcode = cond.to_shout_opcode();
                let shout_id = shout_tables.opcode_to_id(cmp_opcode);
                let index = self.lookup_key_for_opcode(cmp_opcode, rs1_val, rs2_val);
                let cmp = shout.lookup(shout_id, index);
                if cmp > 1 {
                    return Err(format!(
                        "branch compare lookup must be 0/1: cond={cond}, rs1={rs1_val:#x}, rs2={rs2_val:#x}, got={cmp}"
                    ));
                }

                let taken = match cond {
                    // EQ/SLT/SLTU return 1 when the predicate holds.
                    BranchCondition::Eq | BranchCondition::Lt | BranchCondition::Ltu => cmp == 1,
                    // Inverted predicates:
                    // - Ne uses Eq lookup: taken = !(rs1 == rs2)
                    // - Ge/Geu use Slt/Sltu lookup: taken = !(rs1 < rs2)
                    BranchCondition::Ne | BranchCondition::Ge | BranchCondition::Geu => cmp == 0,
                };
                if taken {
                    let imm_u = self.sign_extend_imm(imm);
                    next_pc = self.pc.wrapping_add(imm_u);
                }
            }

            RiscvInstruction::Jal { rd, imm } => {
                // rd = pc + 4 (return address)
                self.write_reg(twist, rd, self.pc.wrapping_add(4));
                // pc = pc + imm
                let imm_u = self.sign_extend_imm(imm);
                next_pc = self.pc.wrapping_add(imm_u);
            }

            RiscvInstruction::Jalr { rd, rs1: _, imm } => {
                let return_addr = self.pc.wrapping_add(4);

                // pc = (rs1 + imm) & !3 (MVP: no compressed instructions)
                // Use Shout ADD for modular semantics, then apply the JALR alignment mask.
                let imm_val = self.sign_extend_imm(imm);
                let index = self.lookup_key_for_opcode(RiscvOpcode::Add, rs1_val, imm_val);
                let sum = shout.lookup(add_shout_id, index);
                next_pc = sum & !3u64;

                // rd = return address
                self.write_reg(twist, rd, return_addr);
            }

            RiscvInstruction::Lui { rd, imm } => {
                // rd = imm << 12 (upper 20 bits)
                let value = (imm as i64 as u64) << 12;
                self.write_reg(twist, rd, value);
            }

            RiscvInstruction::Auipc { rd, imm } => {
                // rd = pc + (imm << 12) (via Shout ADD for modular RV32 semantics)
                let imm_u = self.mask_value((imm as i64 as u64) << 12);
                let index = self.lookup_key_for_opcode(RiscvOpcode::Add, self.pc, imm_u);
                let value = shout.lookup(add_shout_id, index);
                self.write_reg(twist, rd, value);
            }

            RiscvInstruction::Halt => {
                // ECALL trap semantics: halt.
                self.handle_ecall();
            }

            RiscvInstruction::Nop => {}

            // === RV64 W-suffix Operations ===
            RiscvInstruction::RAluw { op, rd, rs1: _, rs2: _ } => {
                let shout_id = shout_tables.opcode_to_id(op);
                let index = self.lookup_key_for_opcode(op, rs1_val, rs2_val);
                let result = shout.lookup(shout_id, index);

                self.write_reg(twist, rd, result);
            }

            RiscvInstruction::IAluw { op, rd, rs1: _, imm } => {
                let imm_val = self.sign_extend_imm(imm);

                let shout_id = shout_tables.opcode_to_id(op);
                let index = self.lookup_key_for_opcode(op, rs1_val, imm_val);
                let result = shout.lookup(shout_id, index);

                self.write_reg(twist, rd, result);
            }

            // === A Extension: Atomics ===
            RiscvInstruction::LoadReserved { op, rd, rs1: _ } => {
                let addr = rs1_val;
                let value = twist.load(ram, addr);

                // Apply width and sign extension
                let width = op.width_bytes();
                let mask = if width >= 8 {
                    u64::MAX
                } else {
                    (1u64 << (width * 8)) - 1
                };
                let result = if op.is_sign_extend() {
                    match width {
                        4 => (value as u32) as i32 as i64 as u64,
                        _ => value & mask,
                    }
                } else {
                    value & mask
                };

                self.write_reg(twist, rd, result);
                // Note: In a real implementation, we'd reserve the address here
            }

            RiscvInstruction::StoreConditional { op, rd, rs1: _, rs2: _ } => {
                let addr = rs1_val;
                let value = rs2_val;

                // Mask value to store width
                let width = op.width_bytes();
                let mask = if width >= 8 {
                    u64::MAX
                } else {
                    (1u64 << (width * 8)) - 1
                };
                let store_value = value & mask;

                // Store the value
                twist.store(ram, addr, store_value);

                // SC returns 0 on success (assuming reservation is valid in single-threaded mode)
                self.write_reg(twist, rd, 0);
            }

            RiscvInstruction::Amo { op, rd, rs1: _, rs2: _ } => {
                let addr = rs1_val;
                let src = rs2_val;

                // Load original value
                let original = self.mask_value(twist.load(ram, addr));
                self.write_reg(twist, rd, original);

                // Compute new value based on AMO operation
                let new_val = match op {
                    RiscvMemOp::AmoswapW | RiscvMemOp::AmoswapD => src,
                    // Use Shout for modular semantics (and to emit a ShoutEvent for the prover).
                    RiscvMemOp::AmoaddW | RiscvMemOp::AmoaddD => {
                        let index = self.lookup_key_for_opcode(RiscvOpcode::Add, original, src);
                        shout.lookup(add_shout_id, index)
                    }
                    RiscvMemOp::AmoxorW | RiscvMemOp::AmoxorD => {
                        let shout_id = shout_tables.opcode_to_id(RiscvOpcode::Xor);
                        let index = self.lookup_key_for_opcode(RiscvOpcode::Xor, original, src);
                        shout.lookup(shout_id, index)
                    }
                    RiscvMemOp::AmoandW | RiscvMemOp::AmoandD => {
                        let shout_id = shout_tables.opcode_to_id(RiscvOpcode::And);
                        let index = self.lookup_key_for_opcode(RiscvOpcode::And, original, src);
                        shout.lookup(shout_id, index)
                    }
                    RiscvMemOp::AmoorW | RiscvMemOp::AmoorD => {
                        let shout_id = shout_tables.opcode_to_id(RiscvOpcode::Or);
                        let index = self.lookup_key_for_opcode(RiscvOpcode::Or, original, src);
                        shout.lookup(shout_id, index)
                    }
                    RiscvMemOp::AmominW => {
                        if (original as i32) < (src as i32) {
                            original
                        } else {
                            src
                        }
                    }
                    RiscvMemOp::AmominD => {
                        if (original as i64) < (src as i64) {
                            original
                        } else {
                            src
                        }
                    }
                    RiscvMemOp::AmomaxW => {
                        if (original as i32) > (src as i32) {
                            original
                        } else {
                            src
                        }
                    }
                    RiscvMemOp::AmomaxD => {
                        if (original as i64) > (src as i64) {
                            original
                        } else {
                            src
                        }
                    }
                    RiscvMemOp::AmominuW | RiscvMemOp::AmominuD => {
                        if original < src {
                            original
                        } else {
                            src
                        }
                    }
                    RiscvMemOp::AmomaxuW | RiscvMemOp::AmomaxuD => {
                        if original > src {
                            original
                        } else {
                            src
                        }
                    }
                    _ => src, // Fallback
                };

                // Store new value
                twist.store(ram, addr, self.mask_value(new_val));
            }

            // === Custom Poseidon2 precompile instructions ===
            RiscvInstruction::Poseidon2AbsorbElem { rs1: _, rs2: _ } => {
                if !cfg!(feature = "poseidon-precompile") {
                    return Err(
                        "poseidon2 precompile instruction executed but feature `poseidon-precompile` is disabled"
                            .into(),
                    );
                }
                let lo = rs1_val as u32 as u64;
                let hi = rs2_val as u32 as u64;
                let elem_u64 = lo | (hi << 32);
                self.poseidon2_ctx
                    .absorb_elem(Goldilocks::from_u64(elem_u64));
            }

            RiscvInstruction::Poseidon2Finalize => {
                if !cfg!(feature = "poseidon-precompile") {
                    return Err(
                        "poseidon2 precompile instruction executed but feature `poseidon-precompile` is disabled"
                            .into(),
                    );
                }
                self.poseidon2_ctx.finalize()?;
            }

            RiscvInstruction::Poseidon2SqueezeWord { rd, idx } => {
                if !cfg!(feature = "poseidon-precompile") {
                    return Err(
                        "poseidon2 precompile instruction executed but feature `poseidon-precompile` is disabled"
                            .into(),
                    );
                }
                let word = self.poseidon2_ctx.squeeze_word(idx)?;
                self.write_reg(twist, rd, word as u64);
            }

            // === System Instructions ===
            RiscvInstruction::Ecall => {
                // ECALL - environment call (syscall).
                self.handle_ecall();
            }

            RiscvInstruction::Ebreak => {
                // EBREAK - debugger breakpoint
                // For now, treat as halt
                self.halted = true;
            }

            RiscvInstruction::Fence { pred: _, succ: _ } => {
                // FENCE - memory ordering
                // No-op in single-threaded execution
            }

            RiscvInstruction::FenceI => {
                // FENCE.I - instruction fence
                // No-op in our implementation
            }
        }

        let next_pc_masked = self.mask_value(next_pc);
        if (next_pc_masked & 0b11) != 0 {
            return Err(format!(
                "control-flow target not 4-byte aligned (no compressed): next_pc={:#x}",
                next_pc_masked
            ));
        }
        self.pc = next_pc_masked;

        Ok(neo_vm_trace::StepMeta {
            pc_after: self.pc,
            opcode: step_opcode,
            is_virtual: false,
            virtual_sequence_remaining: None,
        })
    }
}

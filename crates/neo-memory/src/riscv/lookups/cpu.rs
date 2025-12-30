use neo_vm_trace::{Shout, Twist};

use super::bits::interleave_bits;
use super::decode::decode_instruction;
use super::encode::encode_instruction;
use super::isa::{RiscvInstruction, RiscvMemOp, RiscvOpcode};
use super::tables::RiscvShoutTables;

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
        }
    }

    /// Load a program starting at the given base address.
    pub fn load_program(&mut self, base: u64, program: Vec<RiscvInstruction>) {
        self.program_base = base;
        self.program = program;
        self.pc = base;
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

        // Default: advance PC by 4
        let mut next_pc = self.pc.wrapping_add(4);
        let step_opcode: u32 = instr_word_u32;

        match instr {
            RiscvInstruction::RAlu { op, rd, rs1, rs2 } => {
                let rs1_val = self.get_reg(rs1);
                let rs2_val = self.get_reg(rs2);

                // Use Shout for the ALU operation
                let shout_id = shout_tables.opcode_to_id(op);
                let index = interleave_bits(rs1_val, rs2_val) as u64;
                let result = shout.lookup(shout_id, index);

                self.set_reg(rd, result);
            }

            RiscvInstruction::IAlu { op, rd, rs1, imm } => {
                let rs1_val = self.get_reg(rs1);
                let imm_val = self.sign_extend_imm(imm);

                // Use Shout for the ALU operation
                let shout_id = shout_tables.opcode_to_id(op);
                let index = interleave_bits(rs1_val, imm_val) as u64;
                let result = shout.lookup(shout_id, index);

                self.set_reg(rd, result);
            }

            RiscvInstruction::Load { op, rd, rs1, imm } => {
                let base = self.get_reg(rs1);
                let imm_val = self.sign_extend_imm(imm);
                let index = interleave_bits(base, imm_val) as u64;
                let addr = shout.lookup(add_shout_id, index);

                // Use Twist for memory access
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

                self.set_reg(rd, self.mask_value(result));
            }

            RiscvInstruction::Store { op, rs1, rs2, imm } => {
                let base = self.get_reg(rs1);
                let imm_val = self.sign_extend_imm(imm);
                let index = interleave_bits(base, imm_val) as u64;
                let addr = shout.lookup(add_shout_id, index);
                let value = self.get_reg(rs2);

                // Mask value to store width
                let width = op.width_bytes();
                let mask = if width >= 8 {
                    u64::MAX
                } else {
                    (1u64 << (width * 8)) - 1
                };
                let store_value = value & mask;

                // Use Twist for memory access
                twist.store(ram, addr, store_value);
            }

            RiscvInstruction::Branch { cond, rs1, rs2, imm } => {
                let rs1_val = self.get_reg(rs1);
                let rs2_val = self.get_reg(rs2);

                // Use Shout for the comparison
                let shout_id = shout_tables.opcode_to_id(cond.to_shout_opcode());
                let index = interleave_bits(rs1_val, rs2_val) as u64;
                let _comparison_result = shout.lookup(shout_id, index);

                // Evaluate branch condition
                if cond.evaluate(rs1_val, rs2_val, self.xlen) {
                    next_pc = (self.pc as i64 + imm as i64) as u64;
                }
            }

            RiscvInstruction::Jal { rd, imm } => {
                // rd = pc + 4 (return address)
                self.set_reg(rd, self.pc.wrapping_add(4));
                // pc = pc + imm
                next_pc = (self.pc as i64 + imm as i64) as u64;
            }

            RiscvInstruction::Jalr { rd, rs1, imm } => {
                let rs1_val = self.get_reg(rs1);
                let return_addr = self.pc.wrapping_add(4);

                // pc = (rs1 + imm) & ~1
                // Use Shout ADD for modular RV32 semantics, then apply the JALR alignment mask in-circuit.
                let imm_val = self.sign_extend_imm(imm);
                let index = interleave_bits(rs1_val, imm_val) as u64;
                let sum = shout.lookup(add_shout_id, index);
                next_pc = sum & !1;

                // rd = return address
                self.set_reg(rd, return_addr);
            }

            RiscvInstruction::Lui { rd, imm } => {
                // rd = imm << 12 (upper 20 bits)
                let value = (imm as i64 as u64) << 12;
                self.set_reg(rd, self.mask_value(value));
            }

            RiscvInstruction::Auipc { rd, imm } => {
                // rd = pc + (imm << 12) (via Shout ADD for modular RV32 semantics)
                let imm_u = self.mask_value((imm as i64 as u64) << 12);
                let index = interleave_bits(self.pc, imm_u) as u64;
                let value = shout.lookup(add_shout_id, index);
                self.set_reg(rd, value);
            }

            RiscvInstruction::Halt => {
                self.halted = true;
            }

            RiscvInstruction::Nop => {
            }

            // === RV64 W-suffix Operations ===
            RiscvInstruction::RAluw { op, rd, rs1, rs2 } => {
                let rs1_val = self.get_reg(rs1);
                let rs2_val = self.get_reg(rs2);

                let shout_id = RiscvShoutTables::new(self.xlen).opcode_to_id(op);
                let index = interleave_bits(rs1_val, rs2_val) as u64;
                let result = shout.lookup(shout_id, index);

                self.set_reg(rd, result);
            }

            RiscvInstruction::IAluw { op, rd, rs1, imm } => {
                let rs1_val = self.get_reg(rs1);
                let imm_val = self.sign_extend_imm(imm);

                let shout_id = RiscvShoutTables::new(self.xlen).opcode_to_id(op);
                let index = interleave_bits(rs1_val, imm_val) as u64;
                let result = shout.lookup(shout_id, index);

                self.set_reg(rd, result);
            }

            // === A Extension: Atomics ===
            RiscvInstruction::LoadReserved { op, rd, rs1 } => {
                let addr = self.get_reg(rs1);
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

                self.set_reg(rd, self.mask_value(result));
                // Note: In a real implementation, we'd reserve the address here
            }

            RiscvInstruction::StoreConditional { op, rd, rs1, rs2 } => {
                let addr = self.get_reg(rs1);
                let value = self.get_reg(rs2);

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
                self.set_reg(rd, 0);
            }

            RiscvInstruction::Amo { op, rd, rs1, rs2 } => {
                let addr = self.get_reg(rs1);
                let src = self.mask_value(self.get_reg(rs2));

                // Load original value
                let original = self.mask_value(twist.load(ram, addr));
                self.set_reg(rd, original);

                // Compute new value based on AMO operation
                let new_val = match op {
                    RiscvMemOp::AmoswapW | RiscvMemOp::AmoswapD => src,
                    // Use Shout for modular semantics (and to emit a ShoutEvent for the prover).
                    RiscvMemOp::AmoaddW | RiscvMemOp::AmoaddD => {
                        let index = interleave_bits(original, src) as u64;
                        shout.lookup(add_shout_id, index)
                    }
                    RiscvMemOp::AmoxorW | RiscvMemOp::AmoxorD => {
                        let shout_id = shout_tables.opcode_to_id(RiscvOpcode::Xor);
                        let index = interleave_bits(original, src) as u64;
                        shout.lookup(shout_id, index)
                    }
                    RiscvMemOp::AmoandW | RiscvMemOp::AmoandD => {
                        let shout_id = shout_tables.opcode_to_id(RiscvOpcode::And);
                        let index = interleave_bits(original, src) as u64;
                        shout.lookup(shout_id, index)
                    }
                    RiscvMemOp::AmoorW | RiscvMemOp::AmoorD => {
                        let shout_id = shout_tables.opcode_to_id(RiscvOpcode::Or);
                        let index = interleave_bits(original, src) as u64;
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

            // === System Instructions ===
            RiscvInstruction::Ecall => {
                // ECALL - environment call (syscall)
                // In a real implementation, this would trigger a trap
                // For now, check if a0 (x10) == 0 to halt
                if self.get_reg(10) == 0 {
                    self.halted = true;
                }
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

        self.pc = self.mask_value(next_pc);

        Ok(neo_vm_trace::StepMeta {
            pc_after: self.pc,
            opcode: step_opcode,
        })
    }
}

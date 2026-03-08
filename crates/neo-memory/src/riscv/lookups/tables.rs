use neo_vm_trace::{Shout, ShoutId};
use p3_field::Field;

use crate::riscv::instruction::{
    encode_lookup_key, mask_to_xlen, opcode_lookup_xlen, opcode_operand_mode, operand_mode_keys_enabled,
    try_decode_lookup_operands, OperandMode,
};

use super::alu::compute_op;
use super::bits::uninterleave_bits;
use super::isa::RiscvOpcode;
use super::mle::evaluate_opcode_mle;

#[inline]
fn combined_key_lookup_result(op: RiscvOpcode, key: u128, xlen: usize) -> Option<u64> {
    if !operand_mode_keys_enabled() {
        return None;
    }

    match opcode_operand_mode(op) {
        OperandMode::AddOperands | OperandMode::SubtractOperands
            if matches!(op, RiscvOpcode::Add | RiscvOpcode::Sub) =>
        {
            // ADD/SUB combined-key mode: table behaves as low-word identity.
            Some(mask_to_xlen(key as u64, xlen))
        }
        _ => None,
    }
}

#[inline]
fn lookup_result_for_key(op: RiscvOpcode, key: u128, xlen: usize) -> u64 {
    if let Some(result) = combined_key_lookup_result(op, key, xlen) {
        return result;
    }

    let (rs1, rs2) = try_decode_lookup_operands(op, key, operand_mode_keys_enabled(), xlen)
        .unwrap_or_else(|| uninterleave_bits(key));
    compute_op(op, rs1, rs2, xlen)
}

/// A RISC-V instruction lookup table compatible with Neo's Shout protocol.
///
/// This struct encapsulates:
/// - The opcode (which operation to perform)
/// - The word size (xlen)
/// - Methods for table lookup and MLE evaluation
#[derive(Clone, Debug)]
pub struct RiscvLookupTable<F> {
    /// The RISC-V opcode this table implements.
    pub opcode: RiscvOpcode,
    /// Architectural word size in bits.
    pub arch_xlen: usize,
    /// Lookup-key width in bits.
    pub key_xlen: usize,
    /// Precomputed table values (only for small tables).
    /// For large tables, values are computed on-demand.
    pub values: Option<Vec<F>>,
}

impl<F: Field> RiscvLookupTable<F> {
    /// Create a new lookup table for the given opcode and word size.
    ///
    /// For xlen <= 8, precomputes all table entries.
    /// For larger word sizes, entries are computed on-demand.
    pub fn new(opcode: RiscvOpcode, xlen: usize) -> Self {
        let key_xlen = opcode_lookup_xlen(opcode, xlen);
        let values = if key_xlen <= 8 {
            let table_size = 1usize << (2 * key_xlen);
            Some(
                (0..table_size)
                    .map(|idx| {
                        let entry = lookup_result_for_key(opcode, idx as u128, xlen);
                        F::from_u64(entry)
                    })
                    .collect(),
            )
        } else {
            None
        };

        Self {
            opcode,
            arch_xlen: xlen,
            key_xlen,
            values,
        }
    }

    /// Get the table size (K = 2^{2*key_xlen}) when it fits in `usize`.
    pub fn size(&self) -> Option<usize> {
        if 2 * self.key_xlen >= usize::BITS as usize {
            return None;
        }
        Some(1usize << (2 * self.key_xlen))
    }

    /// Look up a value by index.
    pub fn lookup(&self, index: u128) -> F {
        if let Some(ref values) = self.values {
            values[index as usize]
        } else {
            let entry = lookup_result_for_key(self.opcode, index, self.arch_xlen);
            F::from_u64(entry)
        }
    }

    /// Look up a value by operands.
    pub fn lookup_operands(&self, x: u64, y: u64) -> F {
        let key = encode_lookup_key(self.opcode, x, y, self.arch_xlen);
        self.lookup(key)
    }

    /// Evaluate the MLE at a random point.
    pub fn evaluate_mle(&self, r: &[F]) -> F {
        debug_assert_eq!(r.len(), 2 * self.key_xlen);
        evaluate_opcode_mle(self.opcode, r, self.key_xlen)
    }

    /// Get the content as a vector of field elements (for Shout encoding).
    pub fn content(&self) -> Vec<F> {
        if let Some(ref values) = self.values {
            values.clone()
        } else {
            let table_size = self
                .size()
                .expect("lookup table domain too large to enumerate");
            (0..table_size)
                .map(|idx| self.lookup(idx as u128))
                .collect()
        }
    }
}

/// A RISC-V instruction lookup event for the trace.
///
/// Records an instruction execution that will be proven via Shout.
#[derive(Clone, Debug)]
pub struct RiscvLookupEvent {
    /// The opcode executed.
    pub opcode: RiscvOpcode,
    /// First operand (rs1 value).
    pub rs1: u64,
    /// Second operand (rs2 value).
    pub rs2: u64,
    /// The result (rd value).
    pub result: u64,
}

impl RiscvLookupEvent {
    /// Create a new lookup event.
    pub fn new(opcode: RiscvOpcode, rs1: u64, rs2: u64, xlen: usize) -> Self {
        let result = compute_op(opcode, rs1, rs2, xlen);
        Self {
            opcode,
            rs1,
            rs2,
            result,
        }
    }

    /// Get the lookup index for this event.
    pub fn lookup_index(&self, xlen: usize) -> u128 {
        encode_lookup_key(self.opcode, self.rs1, self.rs2, xlen)
    }
}

/// Range Check table for ADD verification.
///
/// Following Jolt's approach: ADD is verified using a range check that ensures
/// the result is in the correct range [0, 2^xlen). The table maps each value
/// to itself: table[i] = i.
///
/// This table is used to decompose the ADD result into verified chunks.
#[derive(Clone, Debug)]
pub struct RangeCheckTable<F> {
    /// Word size in bits.
    pub xlen: usize,
    /// Precomputed table values.
    pub values: Vec<F>,
}

impl<F: Field> RangeCheckTable<F> {
    /// Create a new range check table.
    pub fn new(xlen: usize) -> Self {
        assert!(xlen <= 16, "Range check table too large for xlen > 16");
        let size = 1usize << xlen;
        let values = (0..size).map(|i| F::from_u64(i as u64)).collect();
        Self { xlen, values }
    }

    /// Get the table size.
    pub fn size(&self) -> usize {
        1usize << self.xlen
    }

    /// Look up a value (identity: table[i] = i).
    pub fn lookup(&self, index: u64) -> F {
        self.values[index as usize]
    }

    /// Evaluate the MLE at a random point.
    ///
    /// For the identity table, the MLE is simply the binary expansion:
    /// RangeCheck~(r) = Σ_{i=0}^{xlen-1} 2^i * r_i
    pub fn evaluate_mle(&self, r: &[F]) -> F {
        debug_assert_eq!(r.len(), self.xlen);
        let mut result = F::ZERO;
        for i in 0..self.xlen {
            result += F::from_u64(1u64 << i) * r[i];
        }
        result
    }

    /// Get the content as a vector of field elements.
    pub fn content(&self) -> Vec<F> {
        self.values.clone()
    }
}

/// A collection of RISC-V lookup tables for the Shout protocol.
///
/// This implements the `Shout` trait and provides lookup tables for all
/// RISC-V ALU operations.
pub struct RiscvShoutTables {
    /// Word size in bits (32 or 64).
    pub xlen: usize,
}

impl RiscvShoutTables {
    /// Create a new set of RISC-V Shout tables.
    pub fn new(xlen: usize) -> Self {
        Self { xlen }
    }

    /// Get the opcode for a given ShoutId.
    pub fn id_to_opcode(&self, id: ShoutId) -> Option<RiscvOpcode> {
        match id.0 {
            0 => Some(RiscvOpcode::And),
            1 => Some(RiscvOpcode::Xor),
            2 => Some(RiscvOpcode::Or),
            3 => Some(RiscvOpcode::Add),
            4 => Some(RiscvOpcode::Sub),
            5 => Some(RiscvOpcode::Slt),
            6 => Some(RiscvOpcode::Sltu),
            7 => Some(RiscvOpcode::Sll),
            8 => Some(RiscvOpcode::Srl),
            9 => Some(RiscvOpcode::Sra),
            10 => Some(RiscvOpcode::Eq),
            11 => Some(RiscvOpcode::Neq),
            // M Extension
            12 => Some(RiscvOpcode::Mul),
            13 => Some(RiscvOpcode::Mulh),
            14 => Some(RiscvOpcode::Mulhu),
            15 => Some(RiscvOpcode::Mulhsu),
            16 => Some(RiscvOpcode::Div),
            17 => Some(RiscvOpcode::Divu),
            18 => Some(RiscvOpcode::Rem),
            19 => Some(RiscvOpcode::Remu),
            // RV64 W-suffix
            20 => Some(RiscvOpcode::Addw),
            21 => Some(RiscvOpcode::Subw),
            22 => Some(RiscvOpcode::Sllw),
            23 => Some(RiscvOpcode::Srlw),
            24 => Some(RiscvOpcode::Sraw),
            25 => None,
            26 => None,
            27 => None,
            28 => None,
            29 => None,
            // Bitmanip
            30 => Some(RiscvOpcode::Andn),
            31 => Some(RiscvOpcode::VirtualMulWord),
            32 => Some(RiscvOpcode::VirtualDivuWord),
            33 => Some(RiscvOpcode::VirtualRemuWord),
            34 => Some(RiscvOpcode::VirtualDivWord),
            35 => Some(RiscvOpcode::VirtualRemWord),
            36 => Some(RiscvOpcode::VirtualMovsignWord),
            _ => None,
        }
    }

    pub fn try_opcode_to_id(&self, op: RiscvOpcode) -> Option<ShoutId> {
        match op {
            RiscvOpcode::And => Some(ShoutId(0)),
            RiscvOpcode::Xor => Some(ShoutId(1)),
            RiscvOpcode::Or => Some(ShoutId(2)),
            RiscvOpcode::Add => Some(ShoutId(3)),
            RiscvOpcode::Sub => Some(ShoutId(4)),
            RiscvOpcode::Slt => Some(ShoutId(5)),
            RiscvOpcode::Sltu => Some(ShoutId(6)),
            RiscvOpcode::Sll => Some(ShoutId(7)),
            RiscvOpcode::Srl => Some(ShoutId(8)),
            RiscvOpcode::Sra => Some(ShoutId(9)),
            RiscvOpcode::Eq => Some(ShoutId(10)),
            RiscvOpcode::Neq => Some(ShoutId(11)),
            RiscvOpcode::Mul => Some(ShoutId(12)),
            RiscvOpcode::Mulh => Some(ShoutId(13)),
            RiscvOpcode::Mulhu => Some(ShoutId(14)),
            RiscvOpcode::Mulhsu => Some(ShoutId(15)),
            RiscvOpcode::Div => Some(ShoutId(16)),
            RiscvOpcode::Divu => Some(ShoutId(17)),
            RiscvOpcode::Rem => Some(ShoutId(18)),
            RiscvOpcode::Remu => Some(ShoutId(19)),
            RiscvOpcode::Addw => Some(ShoutId(20)),
            RiscvOpcode::Subw => Some(ShoutId(21)),
            RiscvOpcode::Sllw => Some(ShoutId(22)),
            RiscvOpcode::Srlw => Some(ShoutId(23)),
            RiscvOpcode::Sraw => Some(ShoutId(24)),
            RiscvOpcode::Mulw | RiscvOpcode::Divw | RiscvOpcode::Divuw | RiscvOpcode::Remw | RiscvOpcode::Remuw => None,
            RiscvOpcode::Andn => Some(ShoutId(30)),
            RiscvOpcode::VirtualMulWord => Some(ShoutId(31)),
            RiscvOpcode::VirtualDivuWord => Some(ShoutId(32)),
            RiscvOpcode::VirtualRemuWord => Some(ShoutId(33)),
            RiscvOpcode::VirtualDivWord => Some(ShoutId(34)),
            RiscvOpcode::VirtualRemWord => Some(ShoutId(35)),
            RiscvOpcode::VirtualMovsignWord => Some(ShoutId(36)),
        }
    }

    /// Get the ShoutId for a given opcode.
    pub fn opcode_to_id(&self, op: RiscvOpcode) -> ShoutId {
        self.try_opcode_to_id(op)
            .unwrap_or_else(|| panic!("opcode {op:?} has no direct Shout table; use helper-owned decomposition"))
    }
}

impl Shout<u128, u64> for RiscvShoutTables {
    fn lookup(&mut self, shout_id: ShoutId, key: u128) -> u64 {
        if let Some(op) = self.id_to_opcode(shout_id) {
            lookup_result_for_key(op, key, self.xlen)
        } else {
            0 // Unknown table
        }
    }
}

impl Shout<u64, u64> for RiscvShoutTables {
    fn lookup(&mut self, shout_id: ShoutId, key: u64) -> u64 {
        if let Some(op) = self.id_to_opcode(shout_id) {
            let key_xlen = opcode_lookup_xlen(op, self.xlen);
            let combined = crate::riscv::instruction::opcode_uses_combined_lookup_key(op);
            assert!(
                key_xlen <= 32 || combined,
                "RiscvShoutTables<u64> cannot soundly transport RV{} {:?} lookup keys (key_xlen={key_xlen}); use u128 shout keys",
                self.xlen,
                op,
            );
            lookup_result_for_key(op, key as u128, self.xlen)
        } else {
            0 // Unknown table
        }
    }
}

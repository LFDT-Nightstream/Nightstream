use neo_vm_trace::{Shout, ShoutId};
use p3_field::Field;

use super::alu::{compute_op, lookup_entry};
use super::bits::{interleave_bits, uninterleave_bits};
use super::isa::RiscvOpcode;
use super::mle::evaluate_opcode_mle;

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
    /// Word size in bits (8, 32, or 64).
    pub xlen: usize,
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
        let values = if xlen <= 8 {
            let table_size = 1usize << (2 * xlen);
            Some(
                (0..table_size)
                    .map(|idx| {
                        let entry = lookup_entry(opcode, idx as u128, xlen);
                        F::from_u64(entry)
                    })
                    .collect(),
            )
        } else {
            None
        };

        Self { opcode, xlen, values }
    }

    /// Get the table size (K = 2^{2*xlen}).
    pub fn size(&self) -> usize {
        1usize << (2 * self.xlen)
    }

    /// Look up a value by index.
    pub fn lookup(&self, index: u128) -> F {
        if let Some(ref values) = self.values {
            values[index as usize]
        } else {
            let entry = lookup_entry(self.opcode, index, self.xlen);
            F::from_u64(entry)
        }
    }

    /// Look up a value by operands.
    pub fn lookup_operands(&self, x: u64, y: u64) -> F {
        let index = interleave_bits(x, y);
        // Mask the index to the correct bit width (index is LSB-aligned)
        let mask = (1u128 << (2 * self.xlen)) - 1;
        self.lookup(index & mask)
    }

    /// Evaluate the MLE at a random point.
    pub fn evaluate_mle(&self, r: &[F]) -> F {
        evaluate_opcode_mle(self.opcode, r, self.xlen)
    }

    /// Get the content as a vector of field elements (for Shout encoding).
    pub fn content(&self) -> Vec<F> {
        if let Some(ref values) = self.values {
            values.clone()
        } else {
            let table_size = self.size();
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
        let index = interleave_bits(self.rs1, self.rs2);
        // With LSB-aligned interleaving, the index is at the LSB
        let mask = (1u128 << (2 * xlen)) - 1;
        index & mask
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
            25 => Some(RiscvOpcode::Mulw),
            26 => Some(RiscvOpcode::Divw),
            27 => Some(RiscvOpcode::Divuw),
            28 => Some(RiscvOpcode::Remw),
            29 => Some(RiscvOpcode::Remuw),
            // Bitmanip
            30 => Some(RiscvOpcode::Andn),
            _ => None,
        }
    }

    /// Get the ShoutId for a given opcode.
    pub fn opcode_to_id(&self, op: RiscvOpcode) -> ShoutId {
        match op {
            RiscvOpcode::And => ShoutId(0),
            RiscvOpcode::Xor => ShoutId(1),
            RiscvOpcode::Or => ShoutId(2),
            RiscvOpcode::Add => ShoutId(3),
            RiscvOpcode::Sub => ShoutId(4),
            RiscvOpcode::Slt => ShoutId(5),
            RiscvOpcode::Sltu => ShoutId(6),
            RiscvOpcode::Sll => ShoutId(7),
            RiscvOpcode::Srl => ShoutId(8),
            RiscvOpcode::Sra => ShoutId(9),
            RiscvOpcode::Eq => ShoutId(10),
            RiscvOpcode::Neq => ShoutId(11),
            // M Extension
            RiscvOpcode::Mul => ShoutId(12),
            RiscvOpcode::Mulh => ShoutId(13),
            RiscvOpcode::Mulhu => ShoutId(14),
            RiscvOpcode::Mulhsu => ShoutId(15),
            RiscvOpcode::Div => ShoutId(16),
            RiscvOpcode::Divu => ShoutId(17),
            RiscvOpcode::Rem => ShoutId(18),
            RiscvOpcode::Remu => ShoutId(19),
            // RV64 W-suffix
            RiscvOpcode::Addw => ShoutId(20),
            RiscvOpcode::Subw => ShoutId(21),
            RiscvOpcode::Sllw => ShoutId(22),
            RiscvOpcode::Srlw => ShoutId(23),
            RiscvOpcode::Sraw => ShoutId(24),
            RiscvOpcode::Mulw => ShoutId(25),
            RiscvOpcode::Divw => ShoutId(26),
            RiscvOpcode::Divuw => ShoutId(27),
            RiscvOpcode::Remw => ShoutId(28),
            RiscvOpcode::Remuw => ShoutId(29),
            // Bitmanip
            RiscvOpcode::Andn => ShoutId(30),
        }
    }
}

impl Shout<u64> for RiscvShoutTables {
    fn lookup(&mut self, shout_id: ShoutId, key: u64) -> u64 {
        // The key is an interleaved index containing both operands
        if let Some(op) = self.id_to_opcode(shout_id) {
            let (rs1, rs2) = uninterleave_bits(key as u128);
            compute_op(op, rs1, rs2, self.xlen)
        } else {
            0 // Unknown table
        }
    }
}

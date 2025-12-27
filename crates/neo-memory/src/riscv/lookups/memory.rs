use std::collections::HashMap;

use neo_vm_trace::{Twist, TwistId};

use super::isa::RiscvMemOp;

/// A RISC-V memory operation event for the trace.
///
/// Records a load or store operation that will be proven via Twist.
#[derive(Clone, Debug)]
pub struct RiscvMemoryEvent {
    /// The memory operation type.
    pub op: RiscvMemOp,
    /// The memory address (base + offset).
    pub addr: u64,
    /// The value loaded or stored.
    pub value: u64,
}

impl RiscvMemoryEvent {
    /// Create a new memory event.
    pub fn new(op: RiscvMemOp, addr: u64, value: u64) -> Self {
        Self { op, addr, value }
    }
}

/// RISC-V memory implementation for the Twist protocol.
///
/// Provides byte-addressable memory with support for different access widths.
pub struct RiscvMemory {
    /// Memory contents (sparse representation).
    data: HashMap<u64, u8>,
    /// Word size in bits (32 or 64).
    pub xlen: usize,
}

impl RiscvMemory {
    /// Create a new empty memory.
    pub fn new(xlen: usize) -> Self {
        Self {
            data: HashMap::new(),
            xlen,
        }
    }

    /// Create memory pre-initialized with a program.
    pub fn with_program(xlen: usize, base_addr: u64, program: &[u8]) -> Self {
        let mut mem = Self::new(xlen);
        for (i, &byte) in program.iter().enumerate() {
            mem.data.insert(base_addr + i as u64, byte);
        }
        mem
    }

    /// Read a byte from memory.
    pub fn read_byte(&self, addr: u64) -> u8 {
        self.data.get(&addr).copied().unwrap_or(0)
    }

    /// Write a byte to memory.
    pub fn write_byte(&mut self, addr: u64, value: u8) {
        if value == 0 {
            self.data.remove(&addr);
        } else {
            self.data.insert(addr, value);
        }
    }

    /// Read a value with the given width (in bytes).
    pub fn read(&self, addr: u64, width: usize) -> u64 {
        let mut value = 0u64;
        for i in 0..width {
            value |= (self.read_byte(addr + i as u64) as u64) << (8 * i);
        }
        value
    }

    /// Write a value with the given width (in bytes).
    pub fn write(&mut self, addr: u64, width: usize, value: u64) {
        for i in 0..width {
            self.write_byte(addr + i as u64, (value >> (8 * i)) as u8);
        }
    }

    /// Execute a memory operation and return the value.
    pub fn execute(&mut self, op: RiscvMemOp, addr: u64, store_value: u64) -> u64 {
        let width = op.width_bytes();

        if op.is_load() {
            let raw = self.read(addr, width);
            // Sign-extend if needed
            if op.is_sign_extend() {
                match width {
                    1 => (raw as u8) as i8 as i64 as u64,
                    2 => (raw as u16) as i16 as i64 as u64,
                    4 => (raw as u32) as i32 as i64 as u64,
                    _ => raw,
                }
            } else {
                raw
            }
        } else {
            self.write(addr, width, store_value);
            store_value
        }
    }
}

impl Twist<u64, u64> for RiscvMemory {
    fn load(&mut self, _twist_id: TwistId, addr: u64) -> u64 {
        // Default: word-sized load
        let width = self.xlen / 8;
        self.read(addr, width)
    }

    fn store(&mut self, _twist_id: TwistId, addr: u64, value: u64) {
        // Default: word-sized store
        let width = self.xlen / 8;
        self.write(addr, width, value);
    }
}

//! Typed memory records of spec §6 and their lane encoding and packing.
//! Mirrors `Spec/Nebula/Records.lean` and `Spec/Nebula/Packing.lean`.

use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::Goldilocks as F;

use super::plan::Plan;

/// One port access of spec §10: `(is_write, is_ram, addr, v_r, v_w)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PortAccess {
    pub is_write: bool,
    pub is_ram: bool,
    pub addr: u64,
    pub vr: u64,
    pub vw: u64,
}

impl PortAccess {
    /// `g = addr + is_ram · R` (spec §6.1).
    pub fn global_index(&self, plan: &Plan) -> usize {
        let addr = self.addr as usize;
        if self.is_ram {
            plan.rom_size() + addr
        } else {
            addr
        }
    }
}

/// Spec §6.1 operation slot. A pad slot is all zero.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct OpSlot {
    pub pad: bool,
    pub is_write: bool,
    pub is_ram: bool,
    pub addr: u64,
    pub vr: u64,
    pub vw: u64,
    pub rt: u64,
}

impl OpSlot {
    pub fn inactive() -> Self {
        Self {
            pad: true,
            ..Self::default()
        }
    }

    pub fn active(access: PortAccess, rt: u64) -> Self {
        Self {
            pad: false,
            is_write: access.is_write,
            is_ram: access.is_ram,
            addr: access.addr,
            vr: access.vr,
            vw: access.vw,
            rt,
        }
    }

    /// Spec §6.3 encoding: field order, little-endian bits.
    pub fn bits(&self, plan: &Plan) -> Vec<bool> {
        let mut bits = vec![self.pad, self.is_write, self.is_ram];
        push_bits(&mut bits, self.addr, plan.mu);
        push_bits(&mut bits, self.vr, 32);
        push_bits(&mut bits, self.vw, 32);
        push_bits(&mut bits, self.rt, plan.w_ts);
        bits
    }
}

/// Spec §6.2 scan slot: one cell's value and stamp.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ScanSlot {
    pub value: u64,
    pub stamp: u64,
}

impl ScanSlot {
    pub fn bits(&self, plan: &Plan) -> Vec<bool> {
        let mut bits = Vec::with_capacity(plan.scan_width());
        push_bits(&mut bits, self.value, 32);
        push_bits(&mut bits, self.stamp, plan.w_ts);
        bits
    }
}

pub fn push_bits(bits: &mut Vec<bool>, value: u64, width: usize) {
    bits.extend((0..width).map(|bit| (value >> bit) & 1 == 1));
}

/// The ops lane of a step: slot-major.
pub fn ops_lane(plan: &Plan, ops: &[OpSlot]) -> Vec<bool> {
    ops.iter().flat_map(|slot| slot.bits(plan)).collect()
}

/// An IS or FS lane of a step: slot-major.
pub fn scan_lane(plan: &Plan, scan: &[ScanSlot]) -> Vec<bool> {
    scan.iter().flat_map(|slot| slot.bits(plan)).collect()
}

/// Spec §9.1 packing: chunks of 63 bits, each read little-endian.
pub fn pack(bits: &[bool]) -> Vec<F> {
    bits.chunks(63)
        .map(|chunk| {
            let value = chunk
                .iter()
                .enumerate()
                .fold(0u64, |value, (bit, set)| value | (u64::from(*set) << bit));
            F::from_u64(value)
        })
        .collect()
}

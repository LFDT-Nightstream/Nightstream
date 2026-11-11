//! The reference semantics of the general register row (scratch
//! prototype): what one row of `lower.rs` does, written plainly. The circuit
//! of `row.rs` must accept exactly these steps; `RowInput` is what its
//! witness reads.

use std::collections::BTreeMap;

use super::isa::{Op, W};
use super::lower::{mul, CRow, Fam, Lowered, K, T1};

/// The values of one row that the circuit does not compute itself: the
/// state before, the two words that the slots read, and an advice value.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RowInput {
    pub pc: u32,
    pub fp: u32,
    pub flag: bool,
    pub regs: [u64; K],
    pub word: u32,
    /// The immediate word (zero when the data slot is busy).
    pub immw: u32,
    /// The data word before the access (zero when there is none).
    pub old: u32,
}

#[derive(Debug, PartialEq, Eq)]
pub enum Stop {
    Halt,
    Unaligned,
    Bounds,
    Native,
    Fuel,
}

pub struct Cpu<'l> {
    pub low: &'l Lowered,
    pub pc: u32,
    pub fp: u32,
    pub flag: bool,
    pub regs: [u64; K],
    pub lin: Vec<u32>,
    pub frame: Vec<u32>,
    pub glob: Vec<u32>,
    pub rows: u64,
    /// Rows that `native` rows stand for, by operation.
    pub estimated: BTreeMap<String, (u64, u64)>,
    pub families: BTreeMap<Fam, u64>,
    /// Rows per register-form origin.
    pub origins: BTreeMap<u16, (u64, u64)>,
    pub trace: Option<Vec<RowInput>>,
    /// Refuse `native` rows (a run for the circuit).
    pub strict: bool,
}

fn sext(v: u64, bits: u32) -> u64 {
    let shift = 64 - bits;
    (((v << shift) as i64) >> shift) as u64
}

/// `neg ⊕ (eq·[a = b] + lt·[a < b])` at width `w`.
pub fn condition(m: [bool; 4], w64: bool, a: u64, b: u64) -> bool {
    let (ua, ub) = if w64 {
        (a, b)
    } else {
        (a & 0xffff_ffff, b & 0xffff_ffff)
    };
    let lt = if m[0] {
        if w64 {
            (ua as i64) < (ub as i64)
        } else {
            (ua as u32 as i32) < (ub as u32 as i32)
        }
    } else {
        ua < ub
    };
    let v = (m[1] && ua == ub) || (m[2] && lt);
    v != m[3]
}

impl<'l> Cpu<'l> {
    pub fn new(low: &'l Lowered, lin_words: usize, frame_words: usize) -> Self {
        let p = &low.program;
        let mut lin = vec![0u32; lin_words];
        for (offset, bytes) in &p.data {
            for (i, byte) in bytes.iter().enumerate() {
                let at = *offset as usize + i;
                lin[at / 4] |= u32::from(*byte) << (8 * (at % 4));
            }
        }
        let mut glob = vec![0u32; 2 * p.globals.len().max(1)];
        for (i, &g) in p.globals.iter().enumerate() {
            glob[2 * i] = g as u32;
            glob[2 * i + 1] = (g >> 32) as u32;
        }
        Self {
            low,
            pc: 0,
            fp: 0,
            flag: false,
            regs: [0; K],
            lin,
            frame: vec![0; frame_words],
            glob,
            rows: 0,
            estimated: BTreeMap::new(),
            families: BTreeMap::new(),
            origins: BTreeMap::new(),
            trace: None,
            strict: false,
        }
    }

    pub fn run(&mut self, args: &[u64], fuel: u64) -> Result<Vec<u64>, Stop> {
        for (i, &a) in args.iter().enumerate() {
            self.regs[i] = a;
        }
        loop {
            if self.rows >= fuel {
                return Err(Stop::Fuel);
            }
            if self.low.rows[self.pc as usize].fam() == Fam::Halt && self.low.rows[self.pc as usize].native.is_none() {
                return Ok(self.regs.to_vec());
            }
            self.step()?;
        }
    }

    fn lin_address(&mut self, row: &CRow, bytes: u64) -> Result<usize, Stop> {
        let ea = (self.regs[row.a as usize] & 0xffff_ffff) + u64::from(row.simm);
        if ea % bytes != 0 {
            if self.strict {
                return Err(Stop::Unaligned);
            }
            let e = self.estimated.entry("unaligned".into()).or_default();
            e.0 += 1;
            e.1 += 2;
        }
        if ea / 4 >= self.lin.len() as u64 {
            return Err(Stop::Bounds);
        }
        Ok(ea as usize)
    }

    fn data_word(&mut self, row: &CRow) -> Result<(Option<(u8, usize)>, u32), Stop> {
        let fam = row.fam();
        let space = (row.m[0], row.m[1]);
        let limb = |l: bool| usize::from(l);
        let at = match fam {
            Fam::Load | Fam::Store if space == (false, false) => {
                let bytes = match (row.uimm, row.m[3]) {
                    (false, false) => 4,
                    (true, false) => 1,
                    _ => 2,
                };
                (0, self.lin_address(row, bytes)? / 4)
            }
            Fam::Load | Fam::Store if row.m[0] => (1, 2 * (self.fp as usize + row.simm as usize) + limb(row.uimm)),
            Fam::Load | Fam::Store => (2, 2 * row.simm as usize + limb(row.uimm)),
            Fam::Call => (1, 2 * (self.fp as usize + row.simm as usize - 1)),
            Fam::Ret => (1, 2 * (self.fp as usize - 1)),
            Fam::BrTable => (
                3,
                row.simm as usize + (self.regs[row.a as usize] & 0xffff_ffff) as usize,
            ),
            _ => return Ok((None, 0)),
        };
        let word = match at.0 {
            0 => self.lin[at.1],
            1 => *self.frame.get(at.1).ok_or(Stop::Bounds)?,
            2 => *self.glob.get(at.1).ok_or(Stop::Bounds)?,
            _ => *self.low.table.get(at.1).ok_or(Stop::Bounds)?,
        };
        Ok((Some(at.0), word).map_first(|s| s.map(|s| (s, at.1))))
    }

    fn lin_bytes(&self, ea: u64, bytes: u32) -> u64 {
        (0..bytes as u64)
            .map(|i| {
                let at = (ea + i) as usize;
                u64::from(self.lin[at / 4] >> (8 * (at % 4)) & 0xff) << (8 * i)
            })
            .sum()
    }

    fn write_word(&mut self, space: u8, at: usize, value: u32) {
        match space {
            0 => self.lin[at] = value,
            1 => self.frame[at] = value,
            _ => self.glob[at] = value,
        }
    }

    pub fn step(&mut self) -> Result<(), Stop> {
        let row = self.low.rows[self.pc as usize];
        let fam = row.fam();
        let (slot, old) = self.data_word(&row)?;
        if let Some(trace) = &mut self.trace {
            trace.push(RowInput {
                pc: self.pc,
                fp: self.fp,
                flag: self.flag,
                regs: self.regs,
                word: row.word(),
                immw: if fam.memory() { 0 } else { row.imm },
                old,
            });
        }
        self.rows += 1;
        *self.families.entry(fam).or_default() += 1;
        let o = self.origins.entry(row.origin).or_default();
        o.0 += u64::from(row.first);
        o.1 += 1;
        let a = self.regs[row.a as usize];
        let breg = self.regs[row.b as usize];
        let b = if row.uimm {
            if fam == Fam::Br {
                0
            } else {
                row.imm64()
            }
        } else {
            breg
        };
        let mask = |v: u64| if row.w64 { v } else { v & 0xffff_ffff };
        let (a0, b0) = (a & 0xffff_ffff, b & 0xffff_ffff);
        let mut pc = self.pc + 1;
        let result: u64 = match fam {
            Fam::Halt => {
                let Some((op, w, estimate)) = row.native else {
                    self.regs[row.d as usize] = 0;
                    return Ok(());
                };
                if self.strict {
                    return Err(Stop::Native);
                }
                let e = self.estimated.entry(format!("{op:?}.{w:?}")).or_default();
                e.0 += 1;
                e.1 += u64::from(estimate);
                self.native(op, w, &row, a, b, &mut pc)?
            }
            Fam::Add => {
                let x = if row.m[1] { 0 } else { a };
                let y = if row.m[2] { b0 << 32 } else { b };
                mask(if row.m[0] { x.wrapping_sub(y) } else { x.wrapping_add(y) })
            }
            Fam::Cmp | Fam::Br | Fam::Assert => {
                let v = condition(row.m, row.w64, a, b);
                if fam == Fam::Cmp {
                    self.flag = v;
                    u64::from(v)
                } else {
                    if fam == Fam::Br && v {
                        pc = row.imm;
                    }
                    0
                }
            }
            Fam::And => mask(a & b),
            Fam::Or => mask(a | b),
            Fam::Xor => mask(a ^ b),
            Fam::Mul => {
                let s = b0 as u32;
                assert!(row.kind() < mul::SHL || s < 64, "the lowering masks shift amounts");
                match (row.kind(), row.w64) {
                    (mul::MUL32, _) => a0.wrapping_mul(b0) & 0xffff_ffff,
                    (mul::MULW, _) => a0 * b0,
                    (mul::MULHI32, _) => (a >> 32).wrapping_mul(b0) & 0xffff_ffff,
                    (mul::SHL, false) => u64::from((a0 as u32) << (s % 32)),
                    (mul::SHRU, false) => u64::from((a0 as u32) >> (s % 32)),
                    (mul::ROTL, false) => u64::from((a0 as u32).rotate_left(s % 32)),
                    (mul::ROTR, false) => u64::from((a0 as u32).rotate_right(s % 32)),
                    (mul::SHL, true) => a << (s % 64),
                    (mul::SHRU, true) => a >> (s % 64),
                    (mul::ROTL, true) => a.rotate_left(s % 64),
                    (mul::ROTR, true) => a.rotate_right(s % 64),
                    _ => panic!("mul kind"),
                }
            }
            Fam::Ext => {
                let bits = if row.m[0] {
                    8
                } else if row.m[1] {
                    16
                } else {
                    32
                };
                if row.m[3] {
                    u64::from(mask(a).count_ones())
                } else {
                    mask(sext(a0, bits))
                }
            }
            Fam::Select => {
                if self.flag {
                    a
                } else {
                    b
                }
            }
            Fam::Load => {
                let (space, _) = slot.unwrap();
                if space == 0 {
                    let ea = a0 + u64::from(row.simm);
                    let lane = (ea % 4) as u32;
                    let (bytes, signed) = (
                        match (row.uimm, row.m[3]) {
                            (false, false) => 4,
                            (true, false) => 1,
                            _ => 2,
                        },
                        row.m[2],
                    );
                    let raw = self.lin_bytes(ea, bytes);
                    debug_assert!(
                        ea % 4 + u64::from(bytes) > 4
                            || raw == u64::from(old >> (8 * lane)) & ((1u64 << (8 * bytes)) - 1)
                    );
                    let v = if signed { sext(raw, 8 * bytes) } else { raw };
                    mask(v)
                } else if row.uimm {
                    a0 | u64::from(old) << 32
                } else {
                    u64::from(old)
                }
            }
            Fam::Store => {
                let (space, at) = slot.unwrap();
                let sv = if row.m[2] { breg >> 32 } else { breg & 0xffff_ffff } as u32;
                if space == 0 {
                    let ea = a0 + u64::from(row.simm);
                    let bytes = match (row.uimm, row.m[3]) {
                        (false, false) => 4,
                        (true, false) => 1,
                        _ => 2,
                    };
                    for i in 0..bytes {
                        let at = (ea + i) as usize;
                        let word = &mut self.lin[at / 4];
                        let shift = 8 * (at % 4);
                        *word = (*word & !(0xff << shift)) | ((sv >> (8 * i)) & 0xff) << shift;
                    }
                } else {
                    self.write_word(space, at, sv);
                }
                0
            }
            Fam::Jump => {
                pc = row.imm;
                0
            }
            Fam::Call => {
                let (_, at) = slot.unwrap();
                self.write_word(1, at, self.pc + 1 + (self.fp << 16));
                self.fp += u32::from(row.simm);
                debug_assert!(row.target >= 4096 || row.call_target() == row.target);
                pc = row.target;
                0
            }
            Fam::Ret => {
                pc = old & 0xffff;
                self.fp = old >> 16;
                0
            }
            Fam::BrTable => {
                pc = old;
                0
            }
            Fam::Host | Fam::Advice => 0,
        };
        self.regs[row.d as usize] = if fam.writes() || fam == Fam::Halt { result } else { 0 };
        if !fam.writes() && fam != Fam::Halt {
            debug_assert_eq!(row.d, T1);
        }
        self.pc = pc;
        Ok(())
    }

    fn native(&mut self, op: Op, w: W, row: &CRow, a: u64, b: u64, pc: &mut u32) -> Result<u64, Stop> {
        let bits = if w == W::I64 { 64 } else { 32 };
        let m = |v: u64| if w == W::I64 { v } else { v & 0xffff_ffff };
        let (ua, ub) = (m(a), m(b));
        let s = |v: u64| if w == W::I64 { v as i64 } else { v as u32 as i32 as i64 };
        Ok(match op {
            Op::DivU => m(ua / ub),
            Op::RemU => m(ua % ub),
            Op::DivS => m(s(ua).wrapping_div(s(ub)) as u64),
            Op::RemS => m(s(ua).wrapping_rem(s(ub)) as u64),
            Op::Clz => u64::from(ua.leading_zeros()) - (64 - bits),
            Op::Ctz => {
                if ua == 0 {
                    bits
                } else {
                    u64::from(ua.trailing_zeros())
                }
            }
            Op::MemSize => (self.lin.len() / 16384) as u64,
            Op::MemGrow => {
                let old = (self.lin.len() / 16384) as u64;
                if old + (a & 0xffff_ffff) > 1024 {
                    0xffff_ffff
                } else {
                    self.lin
                        .resize(((old + (a & 0xffff_ffff)) * 16384) as usize, 0);
                    old
                }
            }
            Op::CallIndirect => {
                let p = &self.low.program;
                let f = p
                    .elements
                    .get(a as u32 as usize)
                    .copied()
                    .flatten()
                    .ok_or(Stop::Native)? as usize;
                if p.types[f] != row.imm {
                    return Err(Stop::Native);
                }
                let at = 2 * (self.fp as usize + row.simm as usize - 1);
                self.frame[at] = self.pc + 1 + (self.fp << 16);
                self.fp += u32::from(row.simm);
                *pc = self.low.entry[f];
                0
            }
            _ => return Err(Stop::Native),
        })
    }
}

trait MapFirst<A, B> {
    fn map_first<C>(self, f: impl FnOnce(A) -> C) -> (C, B);
}

impl<A, B> MapFirst<A, B> for (A, B) {
    fn map_first<C>(self, f: impl FnOnce(A) -> C) -> (C, B) {
        (f(self.0), self.1)
    }
}

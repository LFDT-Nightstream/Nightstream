//! The rows of the general register row, their encoding, the lowering of
//! the register form to them, and their reference semantics `Cpu` (scratch
//! prototype). One row is one operation. Its program word (ROM `code` at
//! `pc`) holds:
//!
//! | bits   | field                                                    |
//! |--------|----------------------------------------------------------|
//! | 0..5   | family code (`Fam`)                                      |
//! | 5      | `w64`: 64-bit operation                                  |
//! | 6      | `uimm`: operand B is the immediate                       |
//! | 7..11  | `m0..m3`: family modifiers                               |
//! | 11..23 | registers A, B, D (4 bits each)                          |
//! | 23..32 | `simm`: 9-bit immediate of a memory row                  |
//!
//! A row whose data slot is idle reads its 32-bit immediate from ROM `imm`
//! at `pc` through that slot; its 64-bit form has the high limb
//! `simm₀ · (2³² − 1)`. Registers 14 and 15 are temporaries of the
//! lowering; a row that writes no register writes zero to register 15.

use super::isa::{Cond, Function, Item, Loc, Op, Program, Row, Space, Src, W};

pub const K: usize = 16;
/// The register file that the allocator sees: two scratch registers of its
/// own, and registers 14 and 15 for the lowering.
pub const ALLOC_K: usize = 14;
pub const T0: u8 = 14;
pub const T1: u8 = 15;
pub const SIMM_BITS: u32 = 9;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub enum Fam {
    Halt,
    Add,
    Cmp,
    Br,
    And,
    Or,
    Xor,
    Mul,
    Ext,
    Select,
    Load,
    Store,
    Jump,
    Call,
    Ret,
    BrTable,
    Host,
    Advice,
    Assert,
}

pub const FAMILIES: usize = 19;

impl Fam {
    pub fn from(code: u8) -> Self {
        assert!((code as usize) < FAMILIES);
        unsafe { std::mem::transmute(code) }
    }

    /// Whether the row uses the data slot, so that it takes its immediate
    /// from `simm`.
    pub fn memory(self) -> bool {
        matches!(self, Fam::Load | Fam::Store | Fam::Call | Fam::Ret | Fam::BrTable)
    }

    pub fn writes(self) -> bool {
        matches!(
            self,
            Fam::Add
                | Fam::Cmp
                | Fam::And
                | Fam::Or
                | Fam::Xor
                | Fam::Mul
                | Fam::Ext
                | Fam::Select
                | Fam::Load
                | Fam::Host
                | Fam::Advice
        )
    }
}

/// Kinds of the `Mul` family (`m0 + 2·m1 + 4·m2`).
pub mod mul {
    pub const MUL32: u8 = 0;
    pub const MULW: u8 = 1;
    pub const MULHI32: u8 = 2;
    pub const SHL: u8 = 3;
    pub const SHRU: u8 = 4;
    pub const ROTL: u8 = 5;
    pub const ROTR: u8 = 6;
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CRow {
    pub fam: u8,
    pub w64: bool,
    pub uimm: bool,
    pub m: [bool; 4],
    pub a: u8,
    pub b: u8,
    pub d: u8,
    pub simm: u16,
    pub imm: u32,
    /// The call target; the word holds it when it fits 12 bits.
    pub target: u32,
    /// The register-form row that this row lowers, for the expansion census.
    pub origin: u16,
    pub first: bool,
    /// A row that the circuit does not have yet: the `Cpu` runs it natively
    /// and counts `estimate` rows for it.
    pub native: Option<(Op, W, u8)>,
}

impl CRow {
    pub fn new(fam: Fam) -> Self {
        Self {
            fam: fam as u8,
            d: T1,
            ..Self::default()
        }
    }

    pub fn fam(&self) -> Fam {
        Fam::from(self.fam)
    }

    pub fn word(&self) -> u32 {
        let m = (0..4).map(|i| u32::from(self.m[i]) << (7 + i)).sum::<u32>();
        u32::from(self.fam)
            | u32::from(self.w64) << 5
            | u32::from(self.uimm) << 6
            | m
            | u32::from(self.a) << 11
            | u32::from(self.b) << 15
            | u32::from(self.d) << 19
            | u32::from(self.simm) << 23
    }

    /// The 64-bit immediate of a row whose data slot is idle.
    pub fn imm64(&self) -> u64 {
        let hi = if self.simm & 1 == 1 { 0xffff_ffffu64 << 32 } else { 0 };
        hi | u64::from(self.imm)
    }

    pub fn kind(&self) -> u8 {
        u8::from(self.m[0]) + 2 * u8::from(self.m[1]) + 4 * u8::from(self.m[2])
    }

    pub fn call_target(&self) -> u32 {
        u32::from(self.a) + 16 * u32::from(self.b) + 256 * (self.kind() as u32 + 8 * u32::from(self.m[3]))
    }
}

/// `(signed, eq, lt, neg)` of a condition: `v = neg ⊕ (eq·[a = b] + lt·[a < b])`.
pub fn cond_bits(c: Cond) -> [bool; 4] {
    use Cond::*;
    match c {
        Eq => [false, true, false, false],
        Ne => [false, true, false, true],
        LtU => [false, false, true, false],
        LtS => [true, false, true, false],
        GtU => [false, true, true, true],
        GtS => [true, true, true, true],
        LeU => [false, true, true, false],
        LeS => [true, true, true, false],
        GeU => [false, false, true, true],
        GeS => [true, false, true, true],
    }
}

enum Fix {
    Label(usize),
    Function(usize),
}

pub struct Lowered {
    pub rows: Vec<CRow>,
    pub entry: Vec<u32>,
    pub table: Vec<u32>,
    pub program: Program,
}

struct Lowering<'p> {
    program: &'p Program,
    rows: Vec<CRow>,
    fixes: Vec<(usize, Fix)>,
    table_base: Vec<u32>,
}

fn reg(loc: Option<Loc>) -> u8 {
    match loc {
        Some(Loc::Reg(r)) => r,
        None => T1,
        Some(Loc::Cell(_)) => panic!("cell form row in the lowering"),
    }
}

impl Lowering<'_> {
    fn push(&mut self, row: CRow) -> usize {
        self.rows.push(row);
        self.rows.len() - 1
    }

    fn alu(&mut self, fam: Fam, w: W, d: u8, a: u8, b: Src) -> usize {
        let mut row = CRow::new(fam);
        row.w64 = w == W::I64;
        row.a = a;
        row.d = d;
        match b {
            Src::Loc(loc) => row.b = reg(Some(loc)),
            Src::Imm(v) => {
                let fits = w == W::I32 || (v >= i64::from(i32::MIN) && v <= i64::from(i32::MAX));
                if !fits {
                    self.constant(T1, v as u64);
                    row.b = T1;
                    return self.push(row);
                }
                row.uimm = true;
                row.imm = v as u32;
                row.simm = u16::from(w == W::I64 && v < 0);
            }
        }
        self.push(row)
    }

    /// `d = v` (any 64-bit constant): one row for a sign-extended 32-bit
    /// value, else the low half (32-bit, so the high limb is zero) and then
    /// `d + (imm << 32)` with the high half as the immediate.
    fn constant(&mut self, d: u8, v: u64) {
        let small = (v as i64) >= i64::from(i32::MIN) && (v as i64) <= i64::from(i32::MAX);
        let mut row = CRow::new(Fam::Add);
        row.w64 = true;
        row.m[1] = true;
        row.uimm = true;
        row.d = d;
        row.imm = v as u32;
        if small {
            row.simm = u16::from((v as i64) < 0);
            self.push(row);
            return;
        }
        row.w64 = false;
        self.push(row);
        let mut high = CRow::new(Fam::Add);
        high.w64 = true;
        high.m[2] = true;
        high.uimm = true;
        high.a = d;
        high.d = d;
        high.imm = (v >> 32) as u32;
        self.push(high);
    }

    fn native(&mut self, op: Op, w: W, estimate: u8, d: u8, a: u8, b: Src) {
        let mut row = CRow::new(Fam::Halt);
        row.native = Some((op, w, estimate));
        row.d = d;
        row.a = a;
        match b {
            Src::Loc(loc) => row.b = reg(Some(loc)),
            Src::Imm(v) => {
                row.uimm = true;
                row.imm = v as u32;
                row.simm = u16::from(v < 0);
            }
        }
        self.push(row);
    }

    /// A `Mul` row; it reads only the low limb of operand B.
    fn mul(&mut self, kind: u8, w64: bool, d: u8, a: u8, b: Src) {
        let mut row = CRow::new(Fam::Mul);
        row.w64 = w64;
        row.a = a;
        row.d = d;
        row.m = [kind & 1 == 1, kind & 2 == 2, kind & 4 == 4, false];
        match b {
            Src::Loc(loc) => row.b = reg(Some(loc)),
            Src::Imm(v) => {
                row.uimm = true;
                row.imm = v as u32;
            }
        }
        self.push(row);
    }

    fn memory(
        &mut self,
        load: bool,
        space: Space,
        d: u8,
        a: u8,
        b: u8,
        simm: u32,
        bytes: u8,
        signed: bool,
        limb: bool,
        w64: bool,
    ) {
        let mut row = CRow::new(if load { Fam::Load } else { Fam::Store });
        row.m[0] = space == Space::Frame;
        row.m[1] = space == Space::Global;
        row.a = a;
        row.b = b;
        row.d = if load { d } else { T1 };
        row.w64 = w64;
        assert!(simm < 1 << SIMM_BITS, "offset {simm}");
        row.simm = simm as u16;
        if load {
            row.m[2] = signed;
        } else {
            row.m[2] = limb;
        }
        match space {
            Space::Linear => {
                let code = match bytes {
                    4 => 0,
                    1 => 1,
                    2 => 2,
                    _ => panic!("width"),
                };
                row.uimm = code & 1 == 1;
                row.m[3] = code & 2 == 2;
            }
            _ => row.uimm = limb,
        }
        self.push(row);
    }

    /// A linear-memory base and an offset that fits `simm`.
    fn base(&mut self, a: Option<Loc>, offset: i64, extra: i64) -> (u8, u32) {
        let total = offset + extra;
        match a {
            Some(loc) if total < 1 << SIMM_BITS => (reg(Some(loc)), total as u32),
            Some(loc) => {
                self.alu(Fam::Add, W::I32, T1, reg(Some(loc)), Src::Imm(total));
                (T1, 0)
            }
            None => {
                self.constant(T1, total as u32 as u64);
                (T1, 0)
            }
        }
    }

    fn row(&mut self, f: &Function, row: &Row) {
        let d = reg(row.dst);
        let a = reg(row.a);
        let b = row.b.unwrap_or(Src::Imm(0));
        let w = row.w;
        let bits = if w == W::I64 { 63 } else { 31 };
        match row.op {
            Op::Add if row.a.is_none() && w == W::I64 && matches!(b, Src::Imm(_)) => {
                let Src::Imm(v) = b else { unreachable!() };
                self.constant(d, v as u64);
            }
            Op::Add | Op::Sub => {
                let at = if row.a.is_none() {
                    let at = self.alu(Fam::Add, w, d, T1, b);
                    self.rows[at].m[1] = true;
                    at
                } else {
                    self.alu(Fam::Add, w, d, a, b)
                };
                self.rows[at].m[0] = row.op == Op::Sub;
            }
            Op::And => drop(self.alu(Fam::And, w, d, a, b)),
            Op::Or => drop(self.alu(Fam::Or, w, d, a, b)),
            Op::Xor => drop(self.alu(Fam::Xor, w, d, a, b)),
            Op::Mul if w == W::I32 => self.mul(mul::MUL32, false, d, a, b),
            Op::Mul => {
                match b {
                    Src::Loc(_) => {
                        let br = reg(row
                            .b
                            .and_then(|s| if let Src::Loc(l) = s { Some(l) } else { None }));
                        self.mul(mul::MULHI32, true, T1, a, b);
                        self.mul(mul::MULHI32, true, T0, br, Src::Loc(Loc::Reg(a)));
                    }
                    Src::Imm(v) => {
                        self.mul(mul::MULHI32, true, T1, a, Src::Imm(v as u32 as i64));
                        self.mul(mul::MUL32, false, T0, a, Src::Imm((v as u64 >> 32) as u32 as i64));
                    }
                }
                self.alu(Fam::Add, W::I32, T1, T1, Src::Loc(Loc::Reg(T0)));
                let low = match b {
                    Src::Imm(v) => Src::Imm(v as u32 as i64),
                    other => other,
                };
                self.mul(mul::MULW, true, T0, a, low);
                let at = self.alu(Fam::Add, W::I64, d, T0, Src::Loc(Loc::Reg(T1)));
                self.rows[at].m[2] = true;
            }
            Op::Shl | Op::ShrU | Op::Rotl | Op::Rotr => {
                let kind = match row.op {
                    Op::Shl => mul::SHL,
                    Op::ShrU => mul::SHRU,
                    Op::Rotl => mul::ROTL,
                    _ => mul::ROTR,
                };
                let amount = match b {
                    Src::Imm(v) => Src::Imm(v & bits),
                    Src::Loc(loc) => {
                        self.alu(Fam::And, W::I32, T1, reg(Some(loc)), Src::Imm(bits));
                        Src::Loc(Loc::Reg(T1))
                    }
                };
                self.mul(kind, w == W::I64, d, a, amount);
            }
            Op::ShrS => {
                let amount = match b {
                    Src::Imm(v) => Src::Imm(v & bits),
                    Src::Loc(loc) => {
                        self.alu(Fam::And, W::I32, T0, reg(Some(loc)), Src::Imm(bits));
                        Src::Loc(Loc::Reg(T0))
                    }
                };
                let at = self.alu(Fam::Cmp, w, T1, a, Src::Imm(0));
                self.rows[at].m = cond_bits(Cond::LtS);
                self.mul(mul::MUL32, false, T1, T1, Src::Imm(0xffff_ffff));
                if w == W::I64 {
                    let at = self.alu(Fam::Ext, W::I64, T1, T1, Src::Imm(0));
                    self.rows[at].m[2] = true;
                }
                self.alu(Fam::Xor, w, d, a, Src::Loc(Loc::Reg(T1)));
                self.mul(mul::SHRU, w == W::I64, d, d, amount);
                self.alu(Fam::Xor, w, d, d, Src::Loc(Loc::Reg(T1)));
            }
            Op::DivU | Op::DivS | Op::RemU | Op::RemS => {
                self.native(row.op, w, if w == W::I64 { 14 } else { 8 }, d, a, b)
            }
            Op::Clz | Op::Ctz => self.native(row.op, w, 12, d, a, b),
            Op::MemSize | Op::MemGrow => self.native(row.op, w, 1, d, a, b),
            Op::CallIndirect => {
                self.native(row.op, w, 3, d, a, b);
                let at = self.rows.len() - 1;
                self.rows[at].simm = row.delta as u16;
                self.rows[at].imm = row.imm as u32;
            }
            Op::Cmp(c) => {
                let at = self.alu(Fam::Cmp, w, if row.dst.is_some() { d } else { T1 }, a, b);
                self.rows[at].m = cond_bits(c);
            }
            Op::Br(c) => {
                // The immediate slot holds the target, so a branch compares
                // register A with register B or with zero (`uimm`).
                let mut br = CRow::new(Fam::Br);
                br.w64 = w == W::I64;
                br.a = a;
                br.m = cond_bits(c);
                match b {
                    Src::Loc(loc) => br.b = reg(Some(loc)),
                    Src::Imm(0) => br.uimm = true,
                    Src::Imm(v) => {
                        let at = self.alu(Fam::Cmp, w, T1, a, Src::Imm(v));
                        self.rows[at].m = cond_bits(c);
                        br.w64 = false;
                        br.a = T1;
                        br.m = cond_bits(Cond::Ne);
                        br.uimm = true;
                    }
                }
                let at = self.push(br);
                self.fixes.push((at, Fix::Label(row.label.unwrap())));
            }
            Op::Popcnt => {
                let at = self.alu(Fam::Ext, w, d, a, Src::Imm(0));
                self.rows[at].m[3] = true;
            }
            Op::Extend8S | Op::Extend16S | Op::Extend32S => {
                let at = self.alu(Fam::Ext, w, d, a, Src::Imm(0));
                let i = match row.op {
                    Op::Extend8S => 0,
                    Op::Extend16S => 1,
                    _ => 2,
                };
                self.rows[at].m[i] = true;
            }
            Op::ExtendU32 => drop(self.alu(Fam::Add, W::I32, d, a, Src::Imm(0))),
            Op::Select => drop(self.alu(Fam::Select, W::I64, d, a, b)),
            Op::Load {
                space: Space::Linear,
                bytes,
                signed,
            } => {
                if bytes == 8 {
                    let (base, off) = self.base(row.a, row.imm, 4);
                    self.memory(true, Space::Linear, T0, base, 0, off, 4, false, false, false);
                    let (base, off) = self.base(row.a, row.imm, 0);
                    self.memory(true, Space::Linear, d, base, 0, off, 4, false, false, false);
                    let at = self.alu(Fam::Add, W::I64, d, d, Src::Loc(Loc::Reg(T0)));
                    self.rows[at].m[2] = true;
                } else {
                    let (base, off) = self.base(row.a, row.imm, 0);
                    self.memory(true, Space::Linear, d, base, 0, off, bytes, signed, false, w == W::I64);
                }
            }
            Op::Store {
                space: Space::Linear,
                bytes,
            } => {
                let value = match b {
                    Src::Loc(loc) => reg(Some(loc)),
                    Src::Imm(v) => {
                        self.constant(T0, v as u64);
                        T0
                    }
                };
                if bytes == 8 {
                    for (limb, extra) in [(false, 0), (true, 4)] {
                        let (base, off) = self.base(row.a, row.imm, extra);
                        self.memory(false, Space::Linear, T1, base, value, off, 4, false, limb, false);
                    }
                } else {
                    let (base, off) = self.base(row.a, row.imm, 0);
                    self.memory(false, Space::Linear, T1, base, value, off, bytes, false, false, false);
                }
            }
            Op::Load { space, .. } => {
                let wide = space == Space::Frame || self.program.global_wide[row.imm as usize];
                let cell = row.imm as u32;
                self.memory(true, space, d, a, 0, cell, 4, false, false, false);
                if wide {
                    self.memory(true, space, d, d, 0, cell, 4, false, true, false);
                }
            }
            Op::Store { space, .. } => {
                let wide = space == Space::Frame || self.program.global_wide[row.imm as usize];
                let value = match b {
                    Src::Loc(loc) => reg(Some(loc)),
                    Src::Imm(v) => {
                        self.constant(T0, v as u64);
                        T0
                    }
                };
                let cell = row.imm as u32;
                self.memory(false, space, T1, a, value, cell, 4, false, false, false);
                if wide {
                    self.memory(false, space, T1, a, value, cell, 4, false, true, false);
                }
            }
            Op::Jump => {
                let at = self.push(CRow::new(Fam::Jump));
                self.fixes.push((at, Fix::Label(row.label.unwrap())));
            }
            Op::BrTable => {
                let table = &self.program.tables[row.imm as usize];
                let n = (table.len() - 1) as i64;
                let at = self.alu(Fam::Cmp, W::I32, T1, a, Src::Imm(n));
                self.rows[at].m = cond_bits(Cond::LtU);
                self.alu(Fam::Select, W::I64, T1, a, Src::Imm(n));
                let mut br = CRow::new(Fam::BrTable);
                br.a = T1;
                br.simm = self.table_base[row.imm as usize] as u16;
                self.push(br);
            }
            Op::Call => {
                let mut call = CRow::new(Fam::Call);
                assert!(row.delta < 1 << SIMM_BITS);
                call.simm = row.delta as u16;
                let at = self.push(call);
                self.fixes.push((at, Fix::Function(row.imm as usize)));
            }
            Op::Host if row.delta == u32::MAX => {
                let mut host = CRow::new(Fam::Host);
                host.a = row.a.map_or(T1, |a| reg(Some(a)));
                host.b = match b {
                    Src::Loc(loc) => reg(Some(loc)),
                    Src::Imm(_) => T1,
                };
                host.d = d;
                self.push(host);
            }
            Op::Host => {
                let results = self.program.functions[row.imm as usize].results;
                if results == 0 {
                    self.push(CRow::new(Fam::Host));
                }
                for r in 0..results {
                    let mut host = CRow::new(Fam::Host);
                    host.d = r as u8;
                    self.push(host);
                }
            }
            Op::Return => drop(self.push(CRow::new(Fam::Ret))),
            Op::Trap | Op::Halt => drop(self.push(CRow::new(Fam::Halt))),
        }
        let _ = f;
    }
}

/// The register-form operation names that `CRow::origin` indexes.
pub fn origins() -> &'static std::sync::Mutex<Vec<String>> {
    static NAMES: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());
    &NAMES
}

fn origin_id(row: &Row) -> u16 {
    let name = format!("{}.{:?}", super::machine::family(row), row.tag);
    let mut names = origins().lock().unwrap();
    let id = names.iter().position(|n| *n == name).unwrap_or_else(|| {
        names.push(name);
        names.len() - 1
    });
    id as u16
}

/// Lower a register-form program (allocated with `ALLOC_K`) to rows: a
/// start stub (`call entry; halt`), then each function.
pub fn lower(program: &Program, entry: usize) -> Lowered {
    let mut table = Vec::new();
    let mut table_base = Vec::new();
    for t in &program.tables {
        table_base.push(table.len() as u32);
        table.extend(t.iter().map(|_| 0u32));
    }
    let mut l = Lowering {
        program,
        rows: Vec::new(),
        fixes: Vec::new(),
        table_base,
    };
    let mut call = CRow::new(Fam::Call);
    call.simm = 1;
    let at = l.push(call);
    l.fixes.push((at, Fix::Function(entry)));
    l.push(CRow::new(Fam::Halt));
    let mut starts = vec![0u32; program.functions.len()];
    let mut label_pc: Vec<Vec<u32>> = Vec::new();
    let mut pending: Vec<(usize, usize, usize)> = Vec::new();
    for (i, f) in program.functions.iter().enumerate() {
        starts[i] = l.rows.len() as u32;
        let mut labels = vec![0u32; f.labels.len()];
        for item in &f.items {
            match item {
                Item::Label(label) => labels[*label] = l.rows.len() as u32,
                Item::Mark(_) => {}
                Item::Row(row) => {
                    let before = l.fixes.len();
                    let first = l.rows.len();
                    l.row(f, row);
                    let origin = origin_id(row);
                    for r in &mut l.rows[first..] {
                        r.origin = origin;
                    }
                    if let Some(r) = l.rows.get_mut(first) {
                        r.first = true;
                    }
                    for fix in &mut l.fixes[before..] {
                        if let Fix::Label(label) = fix.1 {
                            pending.push((fix.0, i, label));
                        }
                    }
                }
            }
        }
        label_pc.push(labels);
    }
    for (at, fix) in std::mem::take(&mut l.fixes) {
        match fix {
            Fix::Function(f) => {
                let target = starts[f];
                let row = &mut l.rows[at];
                row.target = target;
                row.a = (target % 16) as u8;
                row.b = (target / 16 % 16) as u8;
                let high = (target / 256) % 16;
                row.m = [high & 1 == 1, high & 2 == 2, high & 4 == 4, high & 8 == 8];
            }
            Fix::Label(_) => {}
        }
    }
    for (at, f, label) in pending {
        l.rows[at].imm = label_pc[f][label];
    }
    for (t, labels) in program.tables.iter().enumerate() {
        let f = program
            .functions
            .iter()
            .position(|f| {
                f.items
                    .iter()
                    .any(|item| matches!(item, Item::Row(r) if r.op == Op::BrTable && r.imm == t as i64))
            })
            .unwrap();
        for (i, &label) in labels.iter().enumerate() {
            table[l.table_base[t] as usize + i] = label_pc[f][label];
        }
    }
    Lowered {
        rows: l.rows,
        entry: starts,
        table,
        program: program.clone(),
    }
}

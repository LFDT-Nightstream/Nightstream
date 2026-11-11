//! An interpreter of the register form (scratch prototype). It runs the
//! cell form (each frame has its own cells, so a call saves nothing) and the
//! register form (one file of `K` registers, spills in the frame), and it
//! counts the rows it runs and the marks (original instructions) it passes.

use std::collections::BTreeMap;

use super::isa::{Cond, Item, Loc, Op, Program, Row, Space, Src, Tag, W};

#[derive(Debug, PartialEq, Eq)]
pub enum Trap {
    Unreachable,
    Memory,
    Divide,
    Table,
    Fuel,
}

/// One executed row, for the witness of the general row.
#[derive(Clone, Debug)]
pub struct Executed {
    pub function: usize,
    pub item: usize,
    pub regs: Vec<u64>,
    pub fp: u64,
    pub flag: bool,
    /// Linear-memory, frame, or global address and the cell before and after.
    pub access: Option<(Space, u64, u64, u64)>,
}

pub struct Machine<'p> {
    pub program: &'p Program,
    pub registers: Option<usize>,
    regs: Vec<u64>,
    frame: Vec<u64>,
    fp: usize,
    flag: bool,
    pub memory: Vec<u8>,
    pub globals: Vec<u64>,
    calls: Vec<(usize, usize, usize)>,
    pub rows: u64,
    pub marks: u64,
    pub row_mix: BTreeMap<String, u64>,
    pub tag_mix: BTreeMap<Tag, u64>,
    pub mark_mix: BTreeMap<&'static str, u64>,
    pub unaligned: u64,
    pub max_depth: usize,
    pub trace: Option<Vec<Executed>>,
}

fn mask(w: W, v: u64) -> u64 {
    match w {
        W::I32 => v & 0xffff_ffff,
        W::I64 => v,
    }
}

fn sext(v: u64, bits: u32) -> u64 {
    let shift = 64 - bits;
    (((v << shift) as i64) >> shift) as u64
}

pub fn family(row: &Row) -> String {
    if row.tag != Tag::Core {
        let constant = row.a.is_none() && matches!(row.b, Some(Src::Imm(_)));
        return format!("glue.{:?}{}", row.tag, if constant { ".const" } else { "" });
    }
    let w = match row.w {
        W::I32 => "32",
        W::I64 => "64",
    };
    let name = match row.op {
        Op::Cmp(c) => format!("cmp.{c:?}"),
        Op::Br(c) => format!("br.{c:?}"),
        Op::Load { space, bytes, .. } => format!("load.{space:?}.{bytes}"),
        Op::Store { space, bytes } => format!("store.{space:?}.{bytes}"),
        op => format!("{op:?}"),
    };
    match row.op {
        Op::Jump | Op::BrTable | Op::Call | Op::CallIndirect | Op::Return | Op::Host | Op::Trap | Op::Halt => name,
        _ => format!("{name}.{w}"),
    }
}

impl<'p> Machine<'p> {
    pub fn new(program: &'p Program, registers: Option<usize>) -> Self {
        let mut memory = vec![0u8; (program.memory_pages as usize) * 65536];
        for (offset, bytes) in &program.data {
            memory[*offset as usize..*offset as usize + bytes.len()].copy_from_slice(bytes);
        }
        Self {
            program,
            registers,
            regs: vec![0; registers.unwrap_or(0)],
            frame: vec![0; 1 << 16],
            fp: 0,
            flag: false,
            memory,
            globals: program.globals.clone(),
            calls: Vec::new(),
            rows: 0,
            marks: 0,
            row_mix: BTreeMap::new(),
            tag_mix: BTreeMap::new(),
            mark_mix: BTreeMap::new(),
            unaligned: 0,
            max_depth: 0,
            trace: None,
        }
    }

    fn read(&self, loc: Loc) -> u64 {
        match loc {
            Loc::Cell(c) => self.frame[self.fp + c as usize],
            Loc::Reg(r) => self.regs[r as usize],
        }
    }

    fn write(&mut self, loc: Loc, v: u64) {
        match loc {
            Loc::Cell(c) => {
                let at = self.fp + c as usize;
                if at >= self.frame.len() {
                    self.frame.resize(at * 2, 0);
                }
                self.frame[at] = v;
            }
            Loc::Reg(r) => self.regs[r as usize] = v,
        }
    }

    fn src(&self, src: Option<Src>) -> u64 {
        match src {
            Some(Src::Loc(loc)) => self.read(loc),
            Some(Src::Imm(v)) => v as u64,
            None => 0,
        }
    }

    /// Run function `f` on `args` until it returns.
    pub fn run(&mut self, f: usize, args: &[u64], fuel: u64) -> Result<Vec<u64>, Trap> {
        let function = &self.program.functions[f];
        for (i, &arg) in args.iter().enumerate() {
            match self.registers {
                None => self.write(Loc::Cell(i as u32), arg),
                Some(_) => self.regs[i] = arg,
            }
        }
        let mut f = f;
        let mut ip = function.labels[function.entry];
        loop {
            let function = &self.program.functions[f];
            let item = &function.items[ip];
            ip += 1;
            let row = match item {
                Item::Mark(name) => {
                    self.marks += 1;
                    *self.mark_mix.entry(name).or_default() += 1;
                    continue;
                }
                Item::Label(_) => continue,
                Item::Row(row) => row,
            };
            self.rows += 1;
            if self.rows > fuel {
                return Err(Trap::Fuel);
            }
            *self.row_mix.entry(family(row)).or_default() += 1;
            *self.tag_mix.entry(row.tag).or_default() += 1;
            if let Some(trace) = &mut self.trace {
                trace.push(Executed {
                    function: f,
                    item: ip - 1,
                    regs: self.regs.clone(),
                    fp: self.fp as u64,
                    flag: self.flag,
                    access: None,
                });
            }
            let a = row.a.map_or(0, |loc| self.read(loc));
            let b = self.src(row.b);
            let w = row.w;
            let bits = if w == W::I32 { 32 } else { 64 };
            let (ua, ub) = (mask(w, a), mask(w, b));
            let sa = if w == W::I32 { sext(ua, 32) as i64 } else { ua as i64 };
            let sb = if w == W::I32 { sext(ub, 32) as i64 } else { ub as i64 };
            let shift = (ub % bits) as u32;
            let result: Option<u64> = match row.op {
                Op::Add => Some(ua.wrapping_add(ub)),
                Op::Sub => Some(ua.wrapping_sub(ub)),
                Op::Mul => Some(ua.wrapping_mul(ub)),
                Op::And => Some(ua & ub),
                Op::Or => Some(ua | ub),
                Op::Xor => Some(ua ^ ub),
                Op::Shl => Some(ua << shift),
                Op::ShrU => Some(ua >> shift),
                Op::ShrS => Some((sa >> shift) as u64),
                Op::Rotl if w == W::I32 => Some((ua as u32).rotate_left(shift) as u64),
                Op::Rotl => Some(ua.rotate_left(shift)),
                Op::Rotr if w == W::I32 => Some((ua as u32).rotate_right(shift) as u64),
                Op::Rotr => Some(ua.rotate_right(shift)),
                Op::DivU | Op::RemU if ub == 0 => return Err(Trap::Divide),
                Op::DivS | Op::RemS if ub == 0 => return Err(Trap::Divide),
                Op::DivU => Some(ua / ub),
                Op::RemU => Some(ua % ub),
                Op::DivS if sa == (if w == W::I32 { i32::MIN as i64 } else { i64::MIN }) && sb == -1 => {
                    return Err(Trap::Divide)
                }
                Op::DivS => Some(sa.wrapping_div(sb) as u64),
                Op::RemS => Some(sa.wrapping_rem(sb) as u64),
                Op::Cmp(c) => {
                    self.flag = c.holds(w, a, b);
                    Some(u64::from(self.flag))
                }
                Op::Clz => Some(u64::from(ua.leading_zeros() - (64 - bits as u32))),
                Op::Ctz => Some(u64::from(if ua == 0 { bits as u32 } else { ua.trailing_zeros() })),
                Op::Popcnt => Some(u64::from(ua.count_ones())),
                Op::Extend8S => Some(sext(ua, 8)),
                Op::Extend16S => Some(sext(ua, 16)),
                Op::Extend32S => Some(sext(ua, 32)),
                Op::ExtendU32 => Some(a & 0xffff_ffff),
                Op::Select => Some(if self.flag { a } else { b }),
                Op::Load { space, bytes, signed } => Some(self.load(space, row, a, bytes, signed)?),
                Op::Store { space, bytes } => {
                    self.store(space, row, a, b, bytes)?;
                    None
                }
                Op::MemSize => Some(self.memory.len() as u64 / 65536),
                Op::MemGrow => {
                    let old = self.memory.len() as u64 / 65536;
                    if old + ua > 1024 {
                        Some(u64::MAX)
                    } else {
                        self.memory.resize(((old + ua) * 65536) as usize, 0);
                        Some(old)
                    }
                }
                Op::Jump => {
                    ip = function.labels[row.label.unwrap()];
                    None
                }
                Op::Br(c) => {
                    if c.holds(w, a, b) {
                        ip = function.labels[row.label.unwrap()];
                    }
                    None
                }
                Op::BrTable => {
                    let table = &self.program.tables[row.imm as usize];
                    let label = table[(a as u32 as usize).min(table.len() - 1)];
                    ip = function.labels[label];
                    None
                }
                Op::Host if row.delta == u32::MAX => Some(0),
                Op::Call | Op::CallIndirect | Op::Host => {
                    let callee = match row.op {
                        Op::CallIndirect => {
                            let entry = self
                                .program
                                .elements
                                .get(a as u32 as usize)
                                .copied()
                                .flatten();
                            let callee = entry.ok_or(Trap::Table)? as usize;
                            if self.program.types[callee] != row.imm as u32 {
                                return Err(Trap::Table);
                            }
                            callee
                        }
                        _ => row.imm as usize,
                    };
                    let target = &self.program.functions[callee];
                    if target.host {
                        for i in 0..target.results {
                            match self.registers {
                                None => self.write(Loc::Cell(row.delta + i), 0),
                                Some(_) => self.regs[i as usize] = 0,
                            }
                        }
                        None
                    } else {
                        self.calls.push((f, ip, self.fp));
                        self.max_depth = self.max_depth.max(self.calls.len());
                        self.fp += row.delta as usize;
                        if self.fp + 4096 > self.frame.len() {
                            self.frame.resize(self.frame.len() * 2, 0);
                        }
                        f = callee;
                        ip = target.labels[target.entry];
                        None
                    }
                }
                Op::Return => {
                    let Some((caller, back, fp)) = self.calls.pop() else {
                        let results = self.program.functions[f].results;
                        return Ok((0..results)
                            .map(|i| match self.registers {
                                None => self.frame[self.fp + i as usize],
                                Some(_) => self.regs[i as usize],
                            })
                            .collect());
                    };
                    f = caller;
                    ip = back;
                    self.fp = fp;
                    None
                }
                Op::Trap => return Err(Trap::Unreachable),
                Op::Halt => return Ok(Vec::new()),
            };
            if let (Some(v), Some(dst)) = (result, row.dst) {
                self.write(dst, mask(w, v));
            }
        }
    }

    fn load(&mut self, space: Space, row: &Row, a: u64, bytes: u8, signed: bool) -> Result<u64, Trap> {
        match space {
            Space::Frame => Ok(self.frame[self.fp + row.imm as usize]),
            Space::Global => Ok(self.globals[row.imm as usize]),
            Space::Linear => {
                let ea = (a & 0xffff_ffff) + row.imm as u64;
                if ea + u64::from(bytes) > self.memory.len() as u64 {
                    return Err(Trap::Memory);
                }
                if ea % u64::from(bytes) != 0 {
                    self.unaligned += 1;
                }
                let mut v = 0u64;
                for i in 0..bytes as usize {
                    v |= u64::from(self.memory[ea as usize + i]) << (8 * i);
                }
                if let Some(Executed { access, .. }) = self.trace.as_mut().and_then(|t| t.last_mut()) {
                    *access = Some((space, ea, v, v));
                }
                Ok(if signed { sext(v, 8 * u32::from(bytes)) } else { v })
            }
        }
    }

    fn store(&mut self, space: Space, row: &Row, a: u64, b: u64, bytes: u8) -> Result<(), Trap> {
        match space {
            Space::Frame => {
                let at = self.fp + row.imm as usize;
                self.frame[at] = b;
            }
            Space::Global => self.globals[row.imm as usize] = b,
            Space::Linear => {
                let ea = (a & 0xffff_ffff) + row.imm as u64;
                if ea + u64::from(bytes) > self.memory.len() as u64 {
                    return Err(Trap::Memory);
                }
                if ea % u64::from(bytes) != 0 {
                    self.unaligned += 1;
                }
                for i in 0..bytes as usize {
                    self.memory[ea as usize + i] = (b >> (8 * i)) as u8;
                }
            }
        }
        Ok(())
    }
}

pub fn holds(c: Cond, w: W, a: u64, b: u64) -> bool {
    c.holds(w, a, b)
}

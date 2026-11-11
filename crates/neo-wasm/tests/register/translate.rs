//! WASM to the cell form of the register machine (scratch prototype, not
//! proved). WASM validation fixes the operand stack height at each
//! instruction, so stack position `h` of a function is the frame cell
//! `params + locals + h`. `local.get`, constants, `local.tee` and `drop`
//! make no row: the translator keeps a lazy operand (a cell or an
//! immediate) on its own stack. A `local.set` after the row that made the
//! value retargets that row. `block`, `loop`, `end` and `nop` are labels or
//! nothing; a compare before `br_if`, `if` or `select` fuses into it.

use std::collections::HashMap;

use wasmparser::{
    BlockType, ConstExpr, DataKind, ElementItems, ElementKind, Operator, Parser, Payload, TypeRef, ValType,
};

use super::isa::{Cond, Function, Item, Loc, Op, Program, Row, Space, Src, Tag, W};

#[derive(Clone, Copy, Debug, PartialEq)]
enum V {
    Cell(u32),
    Imm(i64),
    Slot,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    Block,
    Loop,
    If,
    Func,
}

#[derive(Clone, Debug)]
struct Ctrl {
    kind: Kind,
    /// The branch target: the start of a loop, the end of anything else.
    label: usize,
    end: usize,
    else_label: Option<usize>,
    height: usize,
    params: usize,
    results: usize,
    reachable_entry: bool,
    targeted: bool,
}

/// A module, parsed.
pub struct Module {
    pub types: Vec<(Vec<ValType>, Vec<ValType>)>,
    /// Type index of each function, imports first.
    pub func_types: Vec<u32>,
    pub imports: usize,
    pub exports: HashMap<String, u32>,
    /// The first type id with the same signature as each type id.
    pub canonical: Vec<u32>,
    pub program: Program,
}

fn const_value(expr: &ConstExpr, globals: &[u64]) -> u64 {
    let mut value = 0;
    for op in expr.get_operators_reader() {
        match op.unwrap() {
            Operator::I32Const { value: v } => value = v as u32 as u64,
            Operator::I64Const { value: v } => value = v as u64,
            Operator::GlobalGet { global_index } => value = globals[global_index as usize],
            Operator::End => {}
            other => panic!("unsupported constant expression {other:?}"),
        }
    }
    value
}

pub fn parse(bytes: &[u8]) -> Result<Module, String> {
    let mut types = Vec::new();
    let mut func_types = Vec::new();
    let mut imports = 0;
    let mut exports = HashMap::new();
    let mut program = Program::default();
    let mut bodies = Vec::new();
    for payload in Parser::new(0).parse_all(bytes) {
        match payload.map_err(|e| e.to_string())? {
            Payload::TypeSection(reader) => {
                for ty in reader.into_iter_err_on_gc_types() {
                    let ty = ty.map_err(|e| e.to_string())?;
                    types.push((ty.params().to_vec(), ty.results().to_vec()));
                }
            }
            Payload::ImportSection(reader) => {
                for import in reader {
                    if let TypeRef::Func(t) = import.map_err(|e| e.to_string())?.ty {
                        func_types.push(t);
                        imports += 1;
                    }
                }
            }
            Payload::FunctionSection(reader) => {
                for t in reader {
                    func_types.push(t.map_err(|e| e.to_string())?);
                }
            }
            Payload::MemorySection(reader) => {
                for memory in reader {
                    program.memory_pages = memory.map_err(|e| e.to_string())?.initial;
                }
            }
            Payload::GlobalSection(reader) => {
                for global in reader {
                    let global = global.map_err(|e| e.to_string())?;
                    let value = const_value(&global.init_expr, &program.globals);
                    program.globals.push(value);
                    program
                        .global_wide
                        .push(global.ty.content_type == ValType::I64);
                }
            }
            Payload::ExportSection(reader) => {
                for export in reader {
                    let export = export.map_err(|e| e.to_string())?;
                    if export.kind == wasmparser::ExternalKind::Func {
                        exports.insert(export.name.to_string(), export.index);
                    }
                }
            }
            Payload::ElementSection(reader) => {
                for element in reader {
                    let element = element.map_err(|e| e.to_string())?;
                    let ElementKind::Active { offset_expr, .. } = element.kind else {
                        continue;
                    };
                    let offset = const_value(&offset_expr, &program.globals) as usize;
                    let ElementItems::Functions(functions) = element.items else {
                        return Err("element expressions".into());
                    };
                    for (i, f) in functions.into_iter().enumerate() {
                        let f = f.map_err(|e| e.to_string())?;
                        if program.elements.len() <= offset + i {
                            program.elements.resize(offset + i + 1, None);
                        }
                        program.elements[offset + i] = Some(f);
                    }
                }
            }
            Payload::DataSection(reader) => {
                for data in reader {
                    let data = data.map_err(|e| e.to_string())?;
                    if let DataKind::Active { offset_expr, .. } = data.kind {
                        program
                            .data
                            .push((const_value(&offset_expr, &program.globals), data.data.to_vec()));
                    }
                }
            }
            Payload::CodeSectionEntry(body) => bodies.push(body),
            _ => {}
        }
    }
    for f in 0..imports {
        let (params, results) = &types[func_types[f] as usize];
        program.functions.push(Function {
            params: params.len() as u32,
            results: results.len() as u32,
            cells: params.len().max(results.len()) as u32,
            entry: 0,
            items: Vec::new(),
            labels: Vec::new(),
            host: true,
        });
    }
    let canonical = (0..types.len())
        .map(|t| types.iter().position(|other| *other == types[t]).unwrap() as u32)
        .collect();
    let mut module = Module {
        types,
        func_types,
        imports,
        exports,
        canonical,
        program,
    };
    for (i, body) in bodies.iter().enumerate() {
        let function = module.translate(imports + i, body)?;
        module.program.functions.push(function);
    }
    module.program.types = module
        .func_types
        .iter()
        .map(|t| module.canonical[*t as usize])
        .collect();
    module.program.signatures = module
        .types
        .iter()
        .map(|(p, r)| (p.len() as u32, r.len() as u32))
        .collect();
    Ok(module)
}

struct Builder<'m> {
    module: &'m Module,
    tables: &'m mut Vec<Vec<usize>>,
    items: Vec<Item>,
    labels: Vec<usize>,
    stack: Vec<V>,
    ctrl: Vec<Ctrl>,
    base: u32,
    max: usize,
    /// The item of the last row whose result is the top of the stack.
    fresh: Option<usize>,
    reachable: bool,
}

impl Builder<'_> {
    fn label(&mut self) -> usize {
        self.labels.push(usize::MAX);
        self.labels.len() - 1
    }

    fn place(&mut self, label: usize) {
        self.labels[label] = self.items.len();
        self.items.push(Item::Label(label));
        self.fresh = None;
    }

    fn slot(&self, h: usize) -> Loc {
        Loc::Cell(self.base + h as u32)
    }

    fn emit(&mut self, row: Row) -> usize {
        self.items.push(Item::Row(row));
        self.fresh = None;
        self.items.len() - 1
    }

    fn src(&self, h: usize) -> Src {
        match self.stack[h] {
            V::Cell(c) => Src::Loc(Loc::Cell(c)),
            V::Imm(v) => Src::Imm(v),
            V::Slot => Src::Loc(self.slot(h)),
        }
    }

    fn mov(&mut self, dst: Loc, src: Src, tag: Tag) {
        if src == Src::Loc(dst) {
            return;
        }
        let mut row = Row::new(Op::Add, W::I64);
        row.dst = Some(dst);
        row.b = Some(src);
        row.tag = tag;
        if let Src::Loc(loc) = src {
            row.a = Some(loc);
            row.b = Some(Src::Imm(0));
        }
        self.emit(row);
    }

    /// Put stack entry `h` in its slot.
    fn materialize(&mut self, h: usize) {
        if self.stack[h] != V::Slot {
            let src = self.src(h);
            self.mov(self.slot(h), src, Tag::Move);
            self.stack[h] = V::Slot;
        }
    }

    /// Put every entry under the top in its slot before a branch on the top,
    /// keeping a compare that made the top last, so that it can fuse.
    fn materialize_under_condition(&mut self) {
        let h = self.top(1);
        if (0..h).all(|i| self.stack[i] == V::Slot) {
            return;
        }
        let held = self.fresh_compare().map(|item| self.items.remove(item));
        for i in 0..h {
            self.materialize(i);
        }
        if let Some(row) = held {
            self.items.push(row);
            self.fresh = Some(self.items.len() - 1);
        }
    }

    fn materialize_all(&mut self) {
        for h in 0..self.stack.len() {
            self.materialize(h);
        }
    }

    /// A location for stack entry `h`, putting an immediate in its slot.
    fn loc(&mut self, h: usize) -> Loc {
        if let V::Imm(_) = self.stack[h] {
            self.materialize(h);
        }
        match self.src(h) {
            Src::Loc(loc) => loc,
            Src::Imm(_) => unreachable!(),
        }
    }

    fn push_result(&mut self, item: Option<usize>) {
        self.stack.push(V::Slot);
        self.max = self.max.max(self.stack.len());
        self.fresh = item;
    }

    /// The position of the `k`-th entry from the top (1 is the top).
    fn top(&self, k: usize) -> usize {
        self.stack.len() - k
    }

    fn pop(&mut self) -> usize {
        self.stack.pop();
        self.stack.len()
    }

    fn binary(&mut self, op: Op, w: W, commutative: bool) {
        let (ha, hb) = (self.top(2), self.top(1));
        let mut op = op;
        let (va, vb) = (self.stack[ha], self.stack[hb]);
        if matches!(va, V::Imm(_)) && !matches!(vb, V::Imm(_)) {
            if commutative {
                self.stack.swap(ha, hb);
            } else if let Op::Cmp(cond) = op {
                self.stack.swap(ha, hb);
                op = Op::Cmp(cond.swap());
            }
        }
        let a = self.loc(ha);
        let b = self.src(hb);
        let mut row = Row::new(op, w);
        row.dst = Some(self.slot(ha));
        row.a = Some(a);
        row.b = Some(b);
        self.stack.truncate(ha);
        let item = self.emit(row);
        self.push_result(Some(item));
    }

    fn unary(&mut self, op: Op, w: W) {
        let h = self.top(1);
        let a = self.loc(h);
        let mut row = Row::new(op, w);
        row.dst = Some(self.slot(h));
        row.a = Some(a);
        self.stack.truncate(h);
        let item = self.emit(row);
        self.push_result(Some(item));
    }

    /// The fresh row that wrote the top of the stack, if it is a compare.
    fn fresh_compare(&self) -> Option<usize> {
        let item = self.fresh?;
        match &self.items[item] {
            Item::Row(row) if matches!(row.op, Op::Cmp(_)) => Some(item),
            _ => None,
        }
    }

    /// A branch to `label` when the top of the stack (popped) is non-zero,
    /// or zero when `negate`. A compare that made it fuses into the branch.
    fn branch_on_top(&mut self, label: usize, negate: bool) {
        let fused = self.fresh_compare();
        let h = self.top(1);
        if let Some(item) = fused {
            self.stack.truncate(h);
            let Item::Row(row) = &mut self.items[item] else {
                unreachable!()
            };
            let Op::Cmp(cond) = row.op else { unreachable!() };
            row.op = Op::Br(if negate { cond.negate() } else { cond });
            row.dst = None;
            row.label = Some(label);
            self.fresh = None;
            return;
        }
        if let V::Imm(v) = self.stack[h] {
            self.stack.truncate(h);
            if (v != 0) != negate {
                self.jump(label);
            }
            return;
        }
        let a = self.loc(h);
        self.stack.truncate(h);
        let mut row = Row::new(Op::Br(if negate { Cond::Eq } else { Cond::Ne }), W::I32);
        row.a = Some(a);
        row.b = Some(Src::Imm(0));
        row.label = Some(label);
        self.emit(row);
    }

    /// Move the top `n` values into the slots from height `to`.
    fn moves(&mut self, n: usize, to: usize) {
        let from = self.stack.len() - n;
        for i in 0..n {
            if from + i != to + i || self.stack[from + i] != V::Slot {
                let src = self.src(from + i);
                self.mov(self.slot(to + i), src, Tag::Move);
            }
        }
    }

    fn needs_moves(&self, n: usize, to: usize) -> bool {
        let from = self.stack.len() - n;
        (0..n).any(|i| from + i != to + i || self.stack[from + i] != V::Slot)
    }

    fn arity(&self, depth: u32) -> (usize, usize, usize) {
        let ctrl = &self.ctrl[self.ctrl.len() - 1 - depth as usize];
        let n = if ctrl.kind == Kind::Loop {
            ctrl.params
        } else {
            ctrl.results
        };
        (ctrl.label, n, ctrl.height)
    }

    fn target(&mut self, depth: u32) {
        let index = self.ctrl.len() - 1 - depth as usize;
        if self.ctrl[index].kind != Kind::Loop {
            self.ctrl[index].targeted = true;
        }
    }

    fn jump(&mut self, label: usize) {
        let mut row = Row::new(Op::Jump, W::I32);
        row.label = Some(label);
        self.emit(row);
    }

    /// The return sequence: the results into cells `0..n`, then `Return`.
    fn ret(&mut self, results: usize) {
        let from = self.stack.len() - results;
        if results == 1 {
            let slot = self.slot(from);
            if let (Some(item), V::Slot) = (self.fresh, self.stack[from]) {
                if let Item::Row(row) = &mut self.items[item] {
                    if row.dst == Some(slot) {
                        row.dst = Some(Loc::Cell(0));
                        self.stack[from] = V::Cell(0);
                    }
                }
            }
        }
        if (0..results).any(|i| matches!(self.stack[from + i], V::Cell(c) if c != i as u32 && c < results as u32)) {
            for h in from..from + results {
                self.materialize(h);
            }
        }
        for i in 0..results {
            let src = self.src(from + i);
            self.mov(Loc::Cell(i as u32), src, Tag::Move);
        }
        self.emit(Row::new(Op::Return, W::I32));
    }

    fn block_type(&self, ty: BlockType) -> (usize, usize) {
        match ty {
            BlockType::Empty => (0, 0),
            BlockType::Type(_) => (0, 1),
            BlockType::FuncType(t) => {
                let (p, r) = &self.module.types[t as usize];
                (p.len(), r.len())
            }
        }
    }

    fn enter(&mut self, kind: Kind, ty: BlockType) {
        let (params, results) = self.block_type(ty);
        let reachable = self.reachable;
        if reachable {
            self.materialize_all();
        }
        let end = self.label();
        let label = if kind == Kind::Loop {
            let start = self.label();
            if reachable {
                self.place(start);
            }
            start
        } else {
            end
        };
        let height = self.stack.len().saturating_sub(params);
        self.ctrl.push(Ctrl {
            kind,
            label,
            end,
            else_label: None,
            height,
            params,
            results,
            reachable_entry: reachable,
            targeted: false,
        });
    }

    fn call(&mut self, op: Op, function: u32, params: usize, results: usize, index: Option<usize>) {
        let h = self.stack.len() - params;
        let a = index.map(|i| self.loc(i));
        if let Some(i) = index {
            self.stack.truncate(i);
        }
        for i in h..h + params {
            self.materialize(i);
        }
        let mut row = Row::new(op, W::I32);
        row.imm = i64::from(function);
        row.delta = self.base + h as u32;
        row.a = a;
        self.stack.truncate(h);
        self.emit(row);
        for _ in 0..results {
            self.push_result(None);
        }
    }

    fn op(&mut self, op: Operator) -> Result<(), String> {
        use Operator as O;
        use W::{I32, I64};
        if !self.reachable {
            match op {
                O::Block { .. } | O::Loop { .. } | O::If { .. } => {
                    self.ctrl.push(Ctrl {
                        kind: Kind::Block,
                        label: usize::MAX,
                        end: usize::MAX,
                        else_label: None,
                        height: self.stack.len(),
                        params: 0,
                        results: 0,
                        reachable_entry: false,
                        targeted: false,
                    });
                    return Ok(());
                }
                O::Else | O::End => {}
                _ => return Ok(()),
            }
        }
        if self.reachable {
            self.items.push(Item::Mark(mnemonic(&op)));
        }
        let op_tail = Some(matches!(op, O::ReturnCall { .. }));
        let bin = |b: &mut Self, op, w| b.binary(op, w, false);
        let com = |b: &mut Self, op, w| b.binary(op, w, true);
        let cmp = |b: &mut Self, c, w| b.binary(Op::Cmp(c), w, matches!(c, Cond::Eq | Cond::Ne));
        match op {
            O::Nop => {}
            O::LocalGet { local_index } => self.stack.push(V::Cell(local_index)),
            O::I32Const { value } => self.stack.push(V::Imm(i64::from(value))),
            O::I64Const { value } => self.stack.push(V::Imm(value)),
            O::LocalSet { local_index } | O::LocalTee { local_index } => {
                let tee = matches!(op, O::LocalTee { .. });
                let x = local_index;
                let h = self.stack.len() - 1;
                let clash = (0..h).any(|i| self.stack[i] == V::Cell(x));
                for i in 0..h {
                    if self.stack[i] == V::Cell(x) {
                        self.materialize(i);
                    }
                }
                let slot = self.slot(h);
                let retarget = match (clash, self.fresh, self.stack[h]) {
                    (false, Some(item), V::Slot) => match &mut self.items[item] {
                        Item::Row(row) if row.dst == Some(slot) => {
                            row.dst = Some(Loc::Cell(x));
                            true
                        }
                        _ => false,
                    },
                    _ => false,
                };
                if !retarget {
                    let src = self.src(h);
                    self.mov(Loc::Cell(x), src, Tag::Move);
                }
                self.stack.truncate(h);
                if tee {
                    self.stack.push(V::Cell(x));
                }
                self.fresh = None;
            }
            O::Drop => {
                self.pop();
                self.fresh = None;
            }
            O::Select | O::TypedSelect { .. } => {
                let h = self.top(1);
                if let V::Imm(v) = self.stack[h] {
                    self.stack.truncate(h);
                    if v as u32 != 0 {
                        self.stack.pop();
                    } else {
                        self.stack.remove(h - 2);
                    }
                    self.fresh = None;
                    return Ok(());
                }
                if self.fresh_compare().is_some() {
                    let item = self.fresh.unwrap();
                    if let Item::Row(row) = &mut self.items[item] {
                        row.dst = None;
                    }
                } else {
                    let a = self.loc(h);
                    let mut row = Row::new(Op::Cmp(Cond::Ne), I32);
                    row.a = Some(a);
                    row.b = Some(Src::Imm(0));
                    self.emit(row);
                }
                self.stack.truncate(h);
                self.binary(Op::Select, I64, false);
            }
            O::I32Add => com(self, Op::Add, I32),
            O::I64Add => com(self, Op::Add, I64),
            O::I32Sub => bin(self, Op::Sub, I32),
            O::I64Sub => bin(self, Op::Sub, I64),
            O::I32Mul => com(self, Op::Mul, I32),
            O::I64Mul => com(self, Op::Mul, I64),
            O::I32And => com(self, Op::And, I32),
            O::I64And => com(self, Op::And, I64),
            O::I32Or => com(self, Op::Or, I32),
            O::I64Or => com(self, Op::Or, I64),
            O::I32Xor => com(self, Op::Xor, I32),
            O::I64Xor => com(self, Op::Xor, I64),
            O::I32Shl => bin(self, Op::Shl, I32),
            O::I64Shl => bin(self, Op::Shl, I64),
            O::I32ShrU => bin(self, Op::ShrU, I32),
            O::I64ShrU => bin(self, Op::ShrU, I64),
            O::I32ShrS => bin(self, Op::ShrS, I32),
            O::I64ShrS => bin(self, Op::ShrS, I64),
            O::I32Rotl => bin(self, Op::Rotl, I32),
            O::I64Rotl => bin(self, Op::Rotl, I64),
            O::I32Rotr => bin(self, Op::Rotr, I32),
            O::I64Rotr => bin(self, Op::Rotr, I64),
            O::I32DivU => bin(self, Op::DivU, I32),
            O::I64DivU => bin(self, Op::DivU, I64),
            O::I32DivS => bin(self, Op::DivS, I32),
            O::I64DivS => bin(self, Op::DivS, I64),
            O::I32RemU => bin(self, Op::RemU, I32),
            O::I64RemU => bin(self, Op::RemU, I64),
            O::I32RemS => bin(self, Op::RemS, I32),
            O::I64RemS => bin(self, Op::RemS, I64),
            O::I32Eq => cmp(self, Cond::Eq, I32),
            O::I64Eq => cmp(self, Cond::Eq, I64),
            O::I32Ne => cmp(self, Cond::Ne, I32),
            O::I64Ne => cmp(self, Cond::Ne, I64),
            O::I32LtU => cmp(self, Cond::LtU, I32),
            O::I64LtU => cmp(self, Cond::LtU, I64),
            O::I32LtS => cmp(self, Cond::LtS, I32),
            O::I64LtS => cmp(self, Cond::LtS, I64),
            O::I32GtU => cmp(self, Cond::GtU, I32),
            O::I64GtU => cmp(self, Cond::GtU, I64),
            O::I32GtS => cmp(self, Cond::GtS, I32),
            O::I64GtS => cmp(self, Cond::GtS, I64),
            O::I32LeU => cmp(self, Cond::LeU, I32),
            O::I64LeU => cmp(self, Cond::LeU, I64),
            O::I32LeS => cmp(self, Cond::LeS, I32),
            O::I64LeS => cmp(self, Cond::LeS, I64),
            O::I32GeU => cmp(self, Cond::GeU, I32),
            O::I64GeU => cmp(self, Cond::GeU, I64),
            O::I32GeS => cmp(self, Cond::GeS, I32),
            O::I64GeS => cmp(self, Cond::GeS, I64),
            O::I32Eqz | O::I64Eqz => {
                let w = if matches!(op, O::I32Eqz) { I32 } else { I64 };
                self.stack.push(V::Imm(0));
                cmp(self, Cond::Eq, w);
            }
            O::I32Clz => self.unary(Op::Clz, I32),
            O::I64Clz => self.unary(Op::Clz, I64),
            O::I32Ctz => self.unary(Op::Ctz, I32),
            O::I64Ctz => self.unary(Op::Ctz, I64),
            O::I32Popcnt => self.unary(Op::Popcnt, I32),
            O::I64Popcnt => self.unary(Op::Popcnt, I64),
            O::I32Extend8S => self.unary(Op::Extend8S, I32),
            O::I64Extend8S => self.unary(Op::Extend8S, I64),
            O::I32Extend16S => self.unary(Op::Extend16S, I32),
            O::I64Extend16S => self.unary(Op::Extend16S, I64),
            O::I64Extend32S | O::I64ExtendI32S => self.unary(Op::Extend32S, I64),
            O::I64ExtendI32U => self.unary(Op::ExtendU32, I64),
            O::I32WrapI64 => {}
            O::I32Load { memarg } => self.load(memarg.offset, 4, false, I32),
            O::I64Load { memarg } => self.load(memarg.offset, 8, false, I64),
            O::I32Load8S { memarg } => self.load(memarg.offset, 1, true, I32),
            O::I32Load8U { memarg } => self.load(memarg.offset, 1, false, I32),
            O::I32Load16S { memarg } => self.load(memarg.offset, 2, true, I32),
            O::I32Load16U { memarg } => self.load(memarg.offset, 2, false, I32),
            O::I64Load8S { memarg } => self.load(memarg.offset, 1, true, I64),
            O::I64Load8U { memarg } => self.load(memarg.offset, 1, false, I64),
            O::I64Load16S { memarg } => self.load(memarg.offset, 2, true, I64),
            O::I64Load16U { memarg } => self.load(memarg.offset, 2, false, I64),
            O::I64Load32S { memarg } => self.load(memarg.offset, 4, true, I64),
            O::I64Load32U { memarg } => self.load(memarg.offset, 4, false, I64),
            O::I32Store { memarg } | O::I64Store32 { memarg } => self.store(memarg.offset, 4),
            O::I64Store { memarg } => self.store(memarg.offset, 8),
            O::I32Store8 { memarg } | O::I64Store8 { memarg } => self.store(memarg.offset, 1),
            O::I32Store16 { memarg } | O::I64Store16 { memarg } => self.store(memarg.offset, 2),
            O::GlobalGet { global_index } => {
                let mut row = Row::new(
                    Op::Load {
                        space: Space::Global,
                        bytes: 8,
                        signed: false,
                    },
                    I64,
                );
                row.dst = Some(self.slot(self.stack.len()));
                row.imm = i64::from(global_index);
                let item = self.emit(row);
                self.push_result(Some(item));
            }
            O::GlobalSet { global_index } => {
                let h = self.top(1);
                let b = self.src(h);
                self.stack.truncate(h);
                let mut row = Row::new(
                    Op::Store {
                        space: Space::Global,
                        bytes: 8,
                    },
                    I64,
                );
                row.b = Some(b);
                row.imm = i64::from(global_index);
                self.emit(row);
            }
            O::MemorySize { .. } => {
                let mut row = Row::new(Op::MemSize, I32);
                row.dst = Some(self.slot(self.stack.len()));
                let item = self.emit(row);
                self.push_result(Some(item));
            }
            O::MemoryGrow { .. } => self.unary(Op::MemGrow, I32),
            O::Block { blockty } => self.enter(Kind::Block, blockty),
            O::Loop { blockty } => self.enter(Kind::Loop, blockty),
            O::If { blockty } => {
                let (params, _) = self.block_type(blockty);
                if params != 0 {
                    return Err("if with params".into());
                }
                let else_label = self.label();
                self.materialize_under_condition();
                self.branch_on_top(else_label, true);
                self.enter(Kind::If, blockty);
                self.ctrl.last_mut().unwrap().else_label = Some(else_label);
            }
            O::Else => {
                let ctrl = self.ctrl.last().unwrap().clone();
                if self.reachable {
                    self.moves(ctrl.results, ctrl.height);
                    self.jump(ctrl.end);
                    self.ctrl.last_mut().unwrap().targeted = true;
                }
                self.stack.truncate(ctrl.height);
                self.reachable = ctrl.reachable_entry;
                if let Some(label) = ctrl.else_label {
                    if self.reachable {
                        self.place(label);
                    }
                }
                self.ctrl.last_mut().unwrap().else_label = None;
            }
            O::End => {
                let ctrl = self.ctrl.pop().unwrap();
                if ctrl.label == usize::MAX {
                    return Ok(());
                }
                if self.reachable && ctrl.kind != Kind::Func {
                    self.moves(ctrl.results, ctrl.height);
                }
                if ctrl.kind == Kind::Func {
                    if self.reachable {
                        self.ret(ctrl.results);
                    }
                    if ctrl.targeted {
                        self.place(ctrl.end);
                        self.stack.truncate(ctrl.height);
                        for _ in 0..ctrl.results {
                            self.stack.push(V::Slot);
                        }
                        self.max = self.max.max(self.stack.len());
                        self.ret(ctrl.results);
                    }
                    return Ok(());
                }
                let after = self.reachable || ctrl.targeted || ctrl.else_label.is_some() && ctrl.reachable_entry;
                if let Some(label) = ctrl.else_label {
                    if ctrl.reachable_entry {
                        self.place(label);
                    }
                }
                if after {
                    self.place(ctrl.end);
                }
                self.stack.truncate(ctrl.height);
                for _ in 0..ctrl.results {
                    self.stack.push(V::Slot);
                }
                self.max = self.max.max(self.stack.len());
                self.reachable = after;
                self.fresh = None;
            }
            O::Br { relative_depth } => {
                let (label, n, to) = self.arity(relative_depth);
                self.moves(n, to);
                self.target(relative_depth);
                self.jump(label);
                self.reachable = false;
            }
            O::BrIf { relative_depth } => {
                let (label, n, to) = self.arity(relative_depth);
                self.target(relative_depth);
                let h = self.stack.len() - 1;
                let needs = {
                    let saved = self.stack.pop().unwrap();
                    let needs = self.needs_moves(n, to);
                    self.stack.push(saved);
                    needs
                };
                if !needs {
                    self.branch_on_top(label, false);
                } else {
                    let skip = self.label();
                    self.branch_on_top(skip, true);
                    let _ = h;
                    self.moves(n, to);
                    self.jump(label);
                    self.place(skip);
                }
            }
            O::BrTable { targets } => {
                let mut depths: Vec<u32> = targets
                    .targets()
                    .collect::<Result<_, _>>()
                    .map_err(|e| e.to_string())?;
                depths.push(targets.default());
                let h = self.top(1);
                if let V::Imm(v) = self.stack[h] {
                    self.stack.truncate(h);
                    let depth = depths[(v as u32 as usize).min(depths.len() - 1)];
                    let (label, n, to) = self.arity(depth);
                    self.moves(n, to);
                    self.target(depth);
                    self.jump(label);
                    self.reachable = false;
                    return Ok(());
                }
                let a = self.loc(h);
                self.stack.truncate(h);
                let mut table = Vec::new();
                let mut trampolines = Vec::new();
                for &depth in &depths {
                    let (label, n, to) = self.arity(depth);
                    self.target(depth);
                    if self.needs_moves(n, to) {
                        let t = self.label();
                        trampolines.push((t, label, n, to));
                        table.push(t);
                    } else {
                        table.push(label);
                    }
                }
                let mut row = Row::new(Op::BrTable, I32);
                row.a = Some(a);
                row.imm = self.tables.len() as i64;
                self.tables.push(table);
                self.emit(row);
                for (t, label, n, to) in trampolines {
                    self.place(t);
                    self.moves(n, to);
                    self.jump(label);
                }
                self.reachable = false;
            }
            O::Return => {
                let results = self.ctrl[0].results;
                self.ret(results);
                self.reachable = false;
            }
            O::Unreachable => {
                self.emit(Row::new(Op::Trap, I32));
                self.reachable = false;
            }
            O::Call { function_index } | O::ReturnCall { function_index } => {
                let (p, r) = &self.module.types[self.module.func_types[function_index as usize] as usize];
                let op = if (function_index as usize) < self.module.imports {
                    Op::Host
                } else {
                    Op::Call
                };
                let r = r.len();
                let tail = matches!(op_tail, Some(true));
                self.call(op, function_index, p.len(), r, None);
                if tail {
                    self.ret(r);
                    self.reachable = false;
                }
            }
            O::CallIndirect { type_index, .. } => {
                let (p, r) = &self.module.types[type_index as usize];
                let index = self.stack.len() - 1;
                let (p, r) = (p.len(), r.len());
                let h = index - p;
                let a = self.loc(index);
                self.stack.truncate(index);
                for i in h..h + p {
                    self.materialize(i);
                }
                let mut row = Row::new(Op::CallIndirect, I32);
                row.imm = i64::from(self.module.canonical[type_index as usize]);
                row.delta = self.base + h as u32;
                row.a = Some(a);
                self.stack.truncate(h);
                self.emit(row);
                for _ in 0..r {
                    self.push_result(None);
                }
                self.fresh = None;
            }
            other => return Err(format!("unsupported {other:?}")),
        }
        Ok(())
    }

    fn load(&mut self, offset: u64, bytes: u8, signed: bool, w: W) {
        let h = self.top(1);
        let mut row = Row::new(
            Op::Load {
                space: Space::Linear,
                bytes,
                signed,
            },
            w,
        );
        row.imm = offset as i64;
        match self.stack[h] {
            V::Imm(v) => row.imm += v as u32 as i64,
            _ => row.a = Some(self.loc(h)),
        }
        row.dst = Some(self.slot(h));
        self.stack.truncate(h);
        let item = self.emit(row);
        self.push_result(Some(item));
    }

    fn store(&mut self, offset: u64, bytes: u8) {
        let ha = self.top(2);
        let mut row = Row::new(
            Op::Store {
                space: Space::Linear,
                bytes,
            },
            W::I64,
        );
        row.imm = offset as i64;
        match self.stack[ha] {
            V::Imm(v) => row.imm += v as u32 as i64,
            _ => row.a = Some(self.loc(ha)),
        }
        row.b = Some(self.src(ha + 1));
        self.stack.truncate(ha);
        self.emit(row);
    }
}

impl Module {
    fn translate(&mut self, index: usize, body: &wasmparser::FunctionBody) -> Result<Function, String> {
        let (params, results) = self.types[self.func_types[index] as usize].clone();
        let mut locals = params.len() as u32;
        for local in body.get_locals_reader().map_err(|e| e.to_string())? {
            locals += local.map_err(|e| e.to_string())?.0;
        }
        let mut tables = std::mem::take(&mut self.program.tables);
        let mut b = Builder {
            module: self,
            tables: &mut tables,
            items: Vec::new(),
            labels: Vec::new(),
            stack: Vec::new(),
            ctrl: Vec::new(),
            base: locals,
            max: 0,
            fresh: None,
            reachable: true,
        };
        let entry = b.label();
        b.place(entry);
        let end = b.label();
        b.ctrl.push(Ctrl {
            kind: Kind::Func,
            label: end,
            end,
            else_label: None,
            height: 0,
            params: 0,
            results: results.len(),
            reachable_entry: true,
            targeted: false,
        });
        for op in body.get_operators_reader().map_err(|e| e.to_string())? {
            b.max = b.max.max(b.stack.len());
            b.op(op.map_err(|e| e.to_string())?)?;
            b.max = b.max.max(b.stack.len());
        }
        let function = Function {
            params: params.len() as u32,
            results: results.len() as u32,
            cells: (locals + b.max as u32).max(results.len() as u32),
            entry,
            items: std::mem::take(&mut b.items),
            labels: std::mem::take(&mut b.labels),
            host: false,
        };
        self.program.tables = tables;
        Ok(function)
    }
}

/// The WASM mnemonic of an operator, for the opcode mix.
pub fn mnemonic(op: &Operator) -> &'static str {
    use std::sync::Mutex;
    static NAMES: Mutex<Vec<&'static str>> = Mutex::new(Vec::new());
    let name = format!("{op:?}");
    let name = name.split([' ', '{', '(']).next().unwrap();
    let mut names = NAMES.lock().unwrap();
    if let Some(known) = names.iter().find(|known| **known == name) {
        return known;
    }
    let leaked: &'static str = Box::leak(name.to_string().into_boxed_str());
    names.push(leaked);
    leaked
}

use std::collections::BTreeSet;
use std::fmt::Write;

use crate::types::{subtype, subtypes, val_list, Choices, FuncTy, TypeSpace, Val, I32_CONSTS, I64_CONSTS};

const FUEL: u32 = 128;
const MAX_DEPTH: usize = 5;
const INT_BINOPS: &[&str] = &[
    "add", "sub", "mul", "and", "or", "xor", "shl", "shr_s", "shr_u", "rotl", "rotr", "div_s", "div_u", "rem_s",
    "rem_u",
];
const COMPARES: &[&str] = &[
    "eq", "ne", "lt_s", "lt_u", "gt_s", "gt_u", "le_s", "le_u", "ge_s", "ge_u",
];
const OFFSETS: &[u32] = &[0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 8, 65532, 65535, 65536];
const GROW_DELTAS: &[i32] = &[0, 1, 1, 2, 3, -1, 0x10001];
const SMALL_ADDRESSES: &[i32] = &[0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 64, 1024];

pub const FEATURES: [&str; 9] = [
    "if_params",
    "if_reference_params",
    "block_params",
    "loop_params",
    "multi_value",
    "typed_refs",
    "memory_grow",
    "page_edge",
    "equivalent_call_indirect",
];

pub type Features = BTreeSet<&'static str>;

struct Table {
    elem: Val,
    slots: Vec<Option<usize>>,
}

struct Spec {
    space: TypeSpace,
    func_types: Vec<usize>,
    tables: Vec<Table>,
    memory: Option<(u32, Option<u32>)>,
    globals: Vec<(Val, bool)>,
    careful: bool,
}

impl Spec {
    fn fuel_global(&self) -> usize {
        self.globals.len()
    }

    fn user_types(&self) -> usize {
        self.func_types.len() - 1
    }
}

struct Branched;
type Go<T = ()> = Result<T, Branched>;

#[derive(Clone, Copy, PartialEq, Eq)]
enum LabelKind {
    Function,
    Block,
    If,
    Loop,
}

struct Label {
    kind: LabelKind,
    types: Vec<Val>,
}

struct Body<'s, 'c, 'd> {
    spec: &'s Spec,
    c: &'c mut Choices<'d>,
    func: usize,
    results: Vec<Val>,
    locals: Vec<Val>,
    params: usize,
    counters: Vec<usize>,
    code: Vec<String>,
    labels: Vec<Label>,
    budget: i32,
    depth: usize,
    features: Features,
}

impl Body<'_, '_, '_> {
    fn mark(&mut self, feature: &'static str, on: bool) {
        if on {
            self.features.insert(feature);
        }
    }

    fn emit(&mut self, instruction: impl Into<String>) {
        self.budget -= 1;
        self.code.push(instruction.into());
    }

    fn emit_all<const N: usize>(&mut self, instructions: [String; N]) {
        for instruction in instructions {
            self.emit(instruction);
        }
    }

    fn fuel(&mut self) {
        let fuel = self.spec.fuel_global();
        self.code.extend([
            format!("global.get {fuel}"),
            "i32.eqz".into(),
            "if".into(),
            "unreachable".into(),
            "end".into(),
            format!("global.get {fuel}"),
            "i32.const 1".into(),
            "i32.sub".into(),
            format!("global.set {fuel}"),
        ]);
    }

    fn new_local(&mut self, ty: Val) -> usize {
        self.locals.push(ty.defaultable());
        self.locals.len() - 1
    }

    fn readable_locals(&self, ty: Val) -> Vec<usize> {
        (0..self.locals.len())
            .filter(|&l| subtype(self.locals[l], ty))
            .collect()
    }

    fn writable_local(&mut self, ty: Val) -> usize {
        let found = (0..self.locals.len())
            .filter(|&l| subtype(ty, self.locals[l]) && !self.counters.contains(&l))
            .collect::<Vec<_>>();
        if found.is_empty() || self.c.chance(15) {
            self.new_local(ty)
        } else {
            self.c.pick(&found)
        }
    }

    fn rand_val(&mut self) -> Val {
        match self.c.weighted(&[55, 25, 6, 14]) {
            0 => Val::I32,
            1 => Val::I64,
            2 => Val::Func,
            _ => Val::Ref(self.c.chance(60), self.c.below(self.spec.user_types())),
        }
    }

    fn rand_vals(&mut self, max: usize) -> Vec<Val> {
        (0..self.c.below(max + 1))
            .map(|_| self.rand_val())
            .collect()
    }

    fn i32_const(&mut self) -> i32 {
        if self.c.chance(25) {
            self.c.next() as i32
        } else {
            self.c.pick(I32_CONSTS)
        }
    }

    fn i64_const(&mut self) -> i64 {
        if self.c.chance(25) {
            ((u64::from(self.c.next()) << 32) | u64::from(self.c.next())) as i64
        } else {
            self.c.pick(I64_CONSTS)
        }
    }

    fn ref_func_of(&mut self, ty: usize) -> usize {
        let space = &self.spec.space;
        let types = &self.spec.func_types;
        let exact = (0..types.len())
            .filter(|&f| types[f] == ty)
            .collect::<Vec<_>>();
        let equivalent = (0..types.len())
            .filter(|&f| types[f] != ty && space.equivalent(types[f], ty))
            .collect::<Vec<_>>();
        if !equivalent.is_empty() && self.c.chance(30) {
            self.c.pick(&equivalent)
        } else {
            self.c.pick(&exact)
        }
    }

    fn leaf(&mut self, ty: Val) {
        let locals = self.readable_locals(ty);
        if !locals.is_empty() && self.c.chance(30) {
            let local = self.c.pick(&locals);
            return self.emit(format!("local.get {local}"));
        }
        match ty {
            Val::I32 | Val::I64 => {
                let globals = (0..self.spec.globals.len())
                    .filter(|&g| self.spec.globals[g].0 == ty)
                    .collect::<Vec<_>>();
                if !globals.is_empty() && self.c.chance(15) {
                    let global = self.c.pick(&globals);
                    self.emit(format!("global.get {global}"));
                } else if ty == Val::I32 {
                    let value = self.i32_const();
                    self.emit(format!("i32.const {value}"));
                } else {
                    let value = self.i64_const();
                    self.emit(format!("i64.const {value}"));
                }
            }
            Val::Func => {
                let func = self.c.below(self.spec.func_types.len());
                self.emit(format!("ref.func $f{func}"));
            }
            Val::Ref(_, k) => {
                self.mark("typed_refs", true);
                let func = self.ref_func_of(k);
                self.emit(format!("ref.func $f{func}"));
            }
            Val::Extern => unreachable!("generated code has no externref values"),
        }
    }

    fn expr(&mut self, ty: Val) -> Go {
        self.budget -= 1;
        if self.budget <= 0 || self.depth >= MAX_DEPTH {
            self.leaf(ty);
            return Ok(());
        }
        self.depth += 1;
        let result = match ty {
            Val::I32 | Val::I64 => self.int_expr(ty),
            _ => self.ref_expr(ty),
        };
        self.depth -= 1;
        result
    }

    fn int_expr(&mut self, ty: Val) -> Go {
        let name = ty.int().unwrap();
        let i32_only = u32::from(ty == Val::I32);
        let memory = u32::from(self.spec.memory.is_some());
        match self.c.weighted(&[
            30,
            15,
            6 * i32_only,
            6,
            5,
            12 * memory,
            6 * memory * i32_only,
            i32_only,
            4,
            6,
            6,
            10,
        ]) {
            0 => self.leaf(ty),
            1 => {
                self.expr(ty)?;
                self.expr(ty)?;
                self.binop(name);
            }
            2 => {
                let operand = if self.c.chance(50) { Val::I32 } else { Val::I64 };
                self.expr(operand)?;
                self.expr(operand)?;
                let op = self.c.pick(COMPARES);
                self.emit(format!("{}.{op}", operand.int().unwrap()));
            }
            3 => {
                self.expr(ty)?;
                self.unop(ty);
            }
            4 => {
                let from = if ty == Val::I32 { Val::I64 } else { Val::I32 };
                self.expr(from)?;
                self.convert(from, ty);
            }
            5 => self.load(ty)?,
            6 if self.c.chance(50) => self.emit("memory.size"),
            6 => self.grow(),
            7 => {
                let table = self.c.below(self.spec.tables.len());
                self.emit(format!("table.size {table}"));
            }
            8 => {
                self.expr(ty)?;
                self.expr(ty)?;
                self.expr(Val::I32)?;
                self.emit("select");
            }
            9 => {
                let produced = self.call()?;
                self.finish(produced, &[ty])?;
            }
            10 => {
                let produced = self.call_indirect()?;
                self.finish(produced, &[ty])?;
            }
            _ => {
                let produced = self.fresh_control(vec![ty])?;
                self.finish(produced, &[ty])?;
            }
        }
        Ok(())
    }

    fn binop(&mut self, name: &str) {
        let op = self.c.pick(INT_BINOPS);
        if self.spec.careful && (op.starts_with("div") || op.starts_with("rem")) {
            self.emit(format!("{name}.const 1"));
            self.emit(format!("{name}.or"));
        }
        self.emit(format!("{name}.{op}"));
    }

    fn unop(&mut self, ty: Val) {
        let name = ty.int().unwrap();
        let ops: &[&str] = if ty == Val::I32 {
            &["clz", "ctz", "popcnt", "extend8_s", "extend16_s"]
        } else {
            &["clz", "ctz", "popcnt", "extend8_s", "extend16_s", "extend32_s"]
        };
        let op = self.c.pick(ops);
        self.emit(format!("{name}.{op}"));
    }

    fn convert(&mut self, from: Val, to: Val) {
        let op = match (from, to) {
            (Val::I64, Val::I32) => self.c.pick(&["i64.eqz", "i32.wrap_i64"]),
            (Val::I32, Val::I64) => self.c.pick(&["i64.extend_i32_s", "i64.extend_i32_u"]),
            _ => unreachable!("conversions are between i32 and i64"),
        };
        self.emit(op);
    }

    fn grow(&mut self) {
        self.mark("memory_grow", true);
        let delta = self.c.pick(GROW_DELTAS);
        self.emit(format!("i32.const {delta}"));
        self.emit("memory.grow");
    }

    fn address(&mut self, width: u32) -> Go<bool> {
        let weights: &[u32] = if self.spec.careful {
            &[40, 20, 40, 0, 0]
        } else {
            &[25, 25, 25, 15, 10]
        };
        let back = if self.spec.careful {
            (width + self.c.below(4) as u32) as i64
        } else {
            self.c.below(11) as i64 - 2
        };
        let near_end = match self.c.weighted(weights) {
            0 => {
                let address = self.c.pick(SMALL_ADDRESSES);
                self.emit(format!("i32.const {address}"));
                false
            }
            1 => {
                let page = if self.spec.careful { 1 } else { 1 + self.c.below(3) };
                self.emit(format!("i32.const {}", page as i64 * 65536 - back));
                true
            }
            2 => {
                self.emit_all([
                    "memory.size".into(),
                    "i32.const 16".into(),
                    "i32.shl".into(),
                    format!("i32.const {back}"),
                    "i32.sub".into(),
                ]);
                true
            }
            3 => {
                self.expr(Val::I32)?;
                false
            }
            _ => {
                self.leaf(Val::I32);
                false
            }
        };
        self.mark("page_edge", near_end);
        Ok(near_end)
    }

    fn memarg(&mut self, natural: u32, near_end: bool) -> String {
        let offset = match (self.spec.careful, near_end) {
            (false, _) => self.c.pick(OFFSETS),
            (true, false) => self.c.pick(&OFFSETS[..11]),
            (true, true) => 0,
        };
        let mut out = if offset == 0 {
            String::new()
        } else {
            format!(" offset={offset}")
        };
        if natural > 1 && self.c.chance(20) {
            out += " align=1";
        }
        out
    }

    fn load(&mut self, ty: Val) -> Go {
        let (op, width) = if ty == Val::I32 {
            self.c.pick(&[
                ("i32.load", 4),
                ("i32.load8_s", 1),
                ("i32.load8_u", 1),
                ("i32.load16_s", 2),
                ("i32.load16_u", 2),
            ])
        } else {
            self.c.pick(&[
                ("i64.load", 8),
                ("i64.load8_s", 1),
                ("i64.load8_u", 1),
                ("i64.load16_s", 2),
                ("i64.load16_u", 2),
                ("i64.load32_s", 4),
                ("i64.load32_u", 4),
            ])
        };
        let near_end = self.address(width)?;
        let memarg = self.memarg(width, near_end);
        self.emit(format!("{op}{memarg}"));
        Ok(())
    }

    fn store_op(&mut self, ty: Val) -> (&'static str, u32) {
        if ty == Val::I32 {
            self.c
                .pick(&[("i32.store", 4), ("i32.store8", 1), ("i32.store16", 2)])
        } else {
            self.c.pick(&[
                ("i64.store", 8),
                ("i64.store8", 1),
                ("i64.store16", 2),
                ("i64.store32", 4),
            ])
        }
    }

    fn store(&mut self, op: &str, width: u32, near_end: bool) {
        let memarg = self.memarg(width, near_end);
        self.emit(format!("{op}{memarg}"));
    }

    fn ref_expr(&mut self, ty: Val) -> Go {
        let tables = (0..self.spec.tables.len())
            .filter(|&t| subtype(self.spec.tables[t].elem, ty))
            .collect::<Vec<_>>();
        match self
            .c
            .weighted(&[50, if tables.is_empty() { 0 } else { 15 }, 10, 8, 8, 9])
        {
            0 => self.leaf(ty),
            1 => {
                let table = self.c.pick(&tables);
                self.table_index(table)?;
                self.emit(format!("table.get {table}"));
            }
            2 => {
                self.expr(ty)?;
                self.expr(ty)?;
                self.expr(Val::I32)?;
                self.emit(format!("select (result {})", ty.wat()));
            }
            3 => {
                let produced = self.call()?;
                self.finish(produced, &[ty])?;
            }
            4 => {
                let produced = self.call_indirect()?;
                self.finish(produced, &[ty])?;
            }
            _ => {
                let produced = self.fresh_control(vec![ty])?;
                self.finish(produced, &[ty])?;
            }
        }
        Ok(())
    }

    fn table_index(&mut self, table: usize) -> Go {
        if self.c.chance(80) {
            let size = self.spec.tables[table].slots.len();
            let slot = self.c.below(size + usize::from(!self.spec.careful));
            self.emit(format!("i32.const {slot}"));
            Ok(())
        } else {
            self.expr(Val::I32)
        }
    }

    fn call(&mut self) -> Go<Vec<Val>> {
        let count = self.spec.func_types.len();
        let forward = (self.func + 1..count).collect::<Vec<_>>();
        let func = if forward.is_empty() || (!self.spec.careful && self.c.chance(10)) {
            self.c.below(count)
        } else {
            self.c.pick(&forward)
        };
        let ty = &self.spec.space.types[self.spec.func_types[func]];
        for &param in &ty.params {
            self.expr(param)?;
        }
        self.emit(format!("call $f{func}"));
        self.mark("multi_value", ty.results.len() > 1);
        Ok(ty.results.clone())
    }

    fn indirect_target(&mut self) -> (usize, usize, usize) {
        let table = if self.spec.tables.len() > 1 && self.c.chance(30) {
            1
        } else {
            0
        };
        let slots = &self.spec.tables[table].slots;
        let held_slots = (0..slots.len())
            .filter(|&slot| slots[slot].is_some_and(|f| f > self.func))
            .collect::<Vec<_>>();
        let slot = if self.spec.careful && !held_slots.is_empty() {
            self.c.pick(&held_slots)
        } else {
            self.c.below(slots.len() + 1)
        };
        let held = slots
            .get(slot)
            .copied()
            .flatten()
            .map(|f| self.spec.func_types[f]);
        let ty = match held {
            Some(held) if self.spec.careful || self.c.chance(75) => {
                let space = &self.spec.space;
                let same = (0..space.types.len())
                    .filter(|&k| space.equivalent(k, held))
                    .collect::<Vec<_>>();
                let ty = self.c.pick(&same);
                self.mark("equivalent_call_indirect", ty != held);
                ty
            }
            _ => self.c.below(self.spec.user_types() + 1),
        };
        (table, slot, ty)
    }

    fn call_indirect(&mut self) -> Go<Vec<Val>> {
        let (table, slot, ty) = self.indirect_target();
        let def = self.spec.space.types[ty].clone();
        for &param in &def.params {
            self.expr(param)?;
        }
        if self.spec.careful || self.c.chance(85) {
            self.emit(format!("i32.const {slot}"));
        } else {
            self.expr(Val::I32)?;
        }
        self.emit(format!("call_indirect {table} (type $t{ty})"));
        self.mark("multi_value", def.results.len() > 1);
        Ok(def.results)
    }

    fn finish(&mut self, mut stack: Vec<Val>, target: &[Val]) -> Go {
        let keep = stack
            .iter()
            .zip(target)
            .take_while(|(&a, &b)| subtype(a, b))
            .count();
        while stack.len() > keep {
            let top = stack.pop().unwrap();
            self.consume(top);
        }
        for &ty in &target[keep..] {
            self.expr(ty)?;
        }
        Ok(())
    }

    fn consume(&mut self, top: Val) {
        if self.c.chance(35) {
            let local = self.writable_local(top);
            self.emit(format!("local.set {local}"));
        } else {
            self.emit("drop");
        }
    }

    fn walk(&mut self, mut stack: Vec<Val>, target: &[Val], steps: usize) -> Go {
        for _ in 0..steps {
            if self.budget <= 0 {
                break;
            }
            self.step(&mut stack)?;
        }
        self.finish(stack, target)
    }

    fn step(&mut self, stack: &mut Vec<Val>) -> Go {
        let len = stack.len();
        let top_int = stack.last().and_then(|top| top.int()).is_some();
        let same_ints = top_int && len >= 2 && stack[len - 1] == stack[len - 2];
        let storable =
            !self.spec.careful && self.spec.memory.is_some() && top_int && len >= 2 && stack[len - 2] == Val::I32;
        let selectable = len >= 3 && stack[len - 1] == Val::I32 && stack[len - 2] == stack[len - 3];
        let on = |flag: bool, weight: u32| if flag { weight } else { 0 };
        match self.c.weighted(&[
            30,
            20,
            on(same_ints, 12),
            on(top_int, 8),
            on(len > 0, 6),
            on(len > 0, 3),
            on(storable, 5),
            on(selectable, 3),
            12,
            on(len > 0, 5),
            3,
            5,
        ]) {
            0 => self.stmt()?,
            1 => {
                let ty = self.rand_val();
                self.expr(ty)?;
                stack.push(ty);
            }
            2 => {
                let ty = stack.pop().unwrap();
                stack.pop();
                let name = ty.int().unwrap();
                if self.c.chance(25) {
                    let op = self.c.pick(COMPARES);
                    self.emit(format!("{name}.{op}"));
                    stack.push(Val::I32);
                } else {
                    self.binop(name);
                    stack.push(ty);
                }
            }
            3 => {
                let ty = stack.pop().unwrap();
                if self.c.chance(70) {
                    self.unop(ty);
                    stack.push(ty);
                } else {
                    let to = if ty == Val::I32 { Val::I64 } else { Val::I32 };
                    self.convert(ty, to);
                    stack.push(to);
                }
            }
            4 => {
                let top = stack.pop().unwrap();
                self.consume(top);
            }
            5 => {
                let top = stack.pop().unwrap();
                let local = self.writable_local(top);
                self.emit(format!("local.tee {local}"));
                stack.push(self.locals[local]);
            }
            6 => {
                let ty = stack.pop().unwrap();
                stack.pop();
                let (op, width) = self.store_op(ty);
                self.store(op, width, false);
            }
            7 => {
                stack.pop();
                let ty = stack.pop().unwrap();
                if ty.int().is_some() {
                    self.emit("select");
                } else {
                    self.emit(format!("select (result {})", ty.wat()));
                }
            }
            8 => {
                let count = self.c.below(len.min(2) + 1);
                let params = stack.split_off(len - count);
                let results = self.rand_vals(2);
                let produced = self.control(params, results)?;
                stack.extend(produced);
            }
            9 => self.call_on_stack(stack)?,
            10 => self.terminator()?,
            _ => {
                let types = self.br_if()?;
                stack.extend(types);
            }
        }
        Ok(())
    }

    fn call_on_stack(&mut self, stack: &mut Vec<Val>) -> Go {
        let len = stack.len();
        let callees = (self.func + 1..self.spec.func_types.len())
            .filter(|&f| {
                let params = &self.spec.space.types[self.spec.func_types[f]].params;
                !params.is_empty() && params.len() <= len && subtypes(&stack[len - params.len()..], params)
            })
            .collect::<Vec<_>>();
        if callees.is_empty() {
            return self.stmt();
        }
        let func = self.c.pick(&callees);
        let ty = &self.spec.space.types[self.spec.func_types[func]];
        stack.truncate(len - ty.params.len());
        stack.extend(&ty.results);
        self.mark("multi_value", ty.results.len() > 1);
        self.emit(format!("call $f{func}"));
        Ok(())
    }

    fn stmt(&mut self) -> Go {
        let memory = self.spec.memory.is_some();
        let mutable = (0..self.spec.globals.len())
            .filter(|&g| self.spec.globals[g].1)
            .collect::<Vec<_>>();
        let on = |flag: bool, weight: u32| if flag { weight } else { 0 };
        match self.c.weighted(&[
            20,
            on(!mutable.is_empty(), 5),
            on(memory, 15),
            5,
            4,
            on(memory, 4),
            8,
            8,
            6,
            15,
        ]) {
            0 => {
                let ty = self.rand_val();
                let local = self.writable_local(ty);
                self.expr(self.locals[local])?;
                self.emit(format!("local.set {local}"));
            }
            1 => {
                let global = self.c.pick(&mutable);
                self.expr(self.spec.globals[global].0)?;
                self.emit(format!("global.set {global}"));
            }
            2 => {
                let ty = if self.c.chance(60) { Val::I32 } else { Val::I64 };
                let (op, width) = self.store_op(ty);
                let near_end = self.address(width)?;
                self.expr(ty)?;
                self.store(op, width, near_end);
            }
            3 => {
                let ty = self.rand_val();
                self.expr(ty)?;
                self.emit("drop");
            }
            4 => {
                let table = self.c.below(self.spec.tables.len());
                self.table_index(table)?;
                self.expr(self.spec.tables[table].elem)?;
                self.emit(format!("table.set {table}"));
            }
            5 => {
                self.grow();
                self.emit("drop");
            }
            6 => {
                let produced = self.call()?;
                self.finish(produced, &[])?;
            }
            7 => {
                let produced = self.call_indirect()?;
                self.finish(produced, &[])?;
            }
            8 => {
                let types = self.br_if()?;
                self.finish(types, &[])?;
            }
            _ => {
                let results = self.rand_vals(2);
                let produced = self.fresh_control(results)?;
                self.finish(produced, &[])?;
            }
        }
        Ok(())
    }

    fn forward_labels(&self) -> Vec<(usize, Vec<Val>)> {
        self.labels
            .iter()
            .rev()
            .enumerate()
            .filter(|(_, label)| matches!(label.kind, LabelKind::Block | LabelKind::If))
            .map(|(depth, label)| (depth, label.types.clone()))
            .collect()
    }

    fn br_if(&mut self) -> Go<Vec<Val>> {
        let labels = self.forward_labels();
        if labels.is_empty() {
            self.stmt()?;
            return Ok(Vec::new());
        }
        let (depth, types) = labels[self.c.below(labels.len())].clone();
        for &ty in &types {
            self.expr(ty)?;
        }
        self.expr(Val::I32)?;
        self.emit(format!("br_if {depth}"));
        Ok(types)
    }

    fn terminator(&mut self) -> Go {
        let forward = self.forward_labels();
        let unreachable = if self.spec.careful { 0 } else { 3 };
        match self
            .c
            .weighted(&[30, 25, if forward.is_empty() { 0 } else { 15 }, unreachable, 10, 10])
        {
            0 => {
                let mut labels = forward;
                labels.push((self.labels.len() - 1, self.results.clone()));
                let (depth, types) = labels[self.c.below(labels.len())].clone();
                for &ty in &types {
                    self.expr(ty)?;
                }
                self.emit(format!("br {depth}"));
            }
            1 => {
                for ty in self.results.clone() {
                    self.expr(ty)?;
                }
                self.emit("return");
            }
            2 => {
                let (default, types) = forward[self.c.below(forward.len())].clone();
                let same = forward
                    .iter()
                    .filter(|(_, other)| *other == types)
                    .map(|(depth, _)| *depth)
                    .collect::<Vec<_>>();
                let targets = (0..self.c.below(4))
                    .map(|_| format!("{} ", self.c.pick(&same)))
                    .collect::<String>();
                for &ty in &types {
                    self.expr(ty)?;
                }
                self.expr(Val::I32)?;
                self.emit(format!("br_table {targets}{default}"));
            }
            3 => self.emit("unreachable"),
            4 => {
                let callees = (0..self.spec.func_types.len())
                    .filter(|&f| self.spec.space.types[self.spec.func_types[f]].results == self.results)
                    .collect::<Vec<_>>();
                let Some(&func) = callees.get(self.c.below(callees.len())) else {
                    for ty in self.results.clone() {
                        self.expr(ty)?;
                    }
                    self.emit("return");
                    return Err(Branched);
                };
                for param in self.spec.space.types[self.spec.func_types[func]]
                    .params
                    .clone()
                {
                    self.expr(param)?;
                }
                self.emit(format!("return_call $f{func}"));
            }
            _ => {
                let (table, slot, ty) = self.indirect_target();
                let def = self.spec.space.types[ty].clone();
                if def.results != self.results {
                    for ty in self.results.clone() {
                        self.expr(ty)?;
                    }
                    self.emit("return");
                    return Err(Branched);
                }
                for &param in &def.params {
                    self.expr(param)?;
                }
                self.emit(format!("i32.const {slot}"));
                self.emit(format!("return_call_indirect {table} (type $t{ty})"));
            }
        }
        Err(Branched)
    }

    fn fresh_control(&mut self, results: Vec<Val>) -> Go<Vec<Val>> {
        let params = self.rand_vals(2);
        for &param in &params {
            self.expr(param)?;
        }
        self.control(params, results)
    }

    fn control(&mut self, params: Vec<Val>, results: Vec<Val>) -> Go<Vec<Val>> {
        let blocktype = FuncTy {
            params: params.clone(),
            results: results.clone(),
        }
        .blocktype();
        self.mark("multi_value", results.len() > 1 || params.len() > 1);
        let steps = 1 + self.c.below(4);
        let kind = [LabelKind::Block, LabelKind::If, LabelKind::Loop][self.c.weighted(&[40, 35, 25])];
        if kind == LabelKind::If {
            self.expr(Val::I32)?;
        }
        let label_types = if kind == LabelKind::Loop { &params } else { &results };
        let label = Label {
            kind,
            types: label_types.clone(),
        };
        match kind {
            LabelKind::Block => {
                self.mark("block_params", !params.is_empty());
                self.emit(format!("block{blocktype}"));
                self.labels.push(label);
                let _ = self.walk(params, &results, steps);
            }
            LabelKind::If => {
                self.mark("if_params", !params.is_empty());
                self.mark("if_reference_params", params.iter().any(|param| param.int().is_none()));
                self.emit(format!("if{blocktype}"));
                self.labels.push(label);
                let _ = self.walk(params.clone(), &results, steps);
                let at = self.code.len();
                self.code.push("else".into());
                let steps = self.c.below(4);
                let _ = self.walk(params, &results, steps);
                if self.code.len() == at + 1 {
                    self.code.pop();
                }
            }
            _ => {
                self.mark("loop_params", !params.is_empty());
                let counter = self.new_local(Val::I32);
                self.counters.push(counter);
                let iterations = 1 + self.c.below(3);
                self.emit(format!("i32.const {iterations}"));
                self.emit(format!("local.set {counter}"));
                self.emit(format!("loop{blocktype}"));
                self.labels.push(label);
                self.fuel();
                if self.walk(params.clone(), &params, steps).is_ok() {
                    self.emit_all([
                        format!("local.get {counter}"),
                        "i32.const 1".into(),
                        "i32.sub".into(),
                        format!("local.tee {counter}"),
                        "br_if 0".into(),
                    ]);
                    let steps = self.c.below(3);
                    let _ = self.walk(params, &results, steps);
                }
            }
        }
        self.labels.pop();
        self.emit("end");
        Ok(results)
    }
}

fn function(spec: &Spec, c: &mut Choices, func: usize, features: &mut Features) -> String {
    let ty = spec.func_types[func];
    let def = &spec.space.types[ty];
    let budget = if func == 0 { 40 + c.below(60) } else { 15 + c.below(40) };
    let mut body = Body {
        spec,
        c,
        func,
        results: def.results.clone(),
        locals: def.params.clone(),
        params: def.params.len(),
        counters: Vec::new(),
        code: Vec::new(),
        labels: vec![Label {
            kind: LabelKind::Function,
            types: def.results.clone(),
        }],
        budget: budget as i32,
        depth: 0,
        features: Features::new(),
    };
    body.fuel();
    let steps = 2 + body.c.below(5);
    let _ = body.walk(Vec::new(), &def.results, steps);
    features.extend(body.features);
    let export = if func == 0 { " (export \"main\")" } else { "" };
    format!(
        "(func $f{func}{export} (type $t{ty}){}\n  {})\n",
        val_list("local", &body.locals[body.params..]),
        body.code.join("\n  ")
    )
}

pub struct Generated {
    pub wat: String,
    pub features: Features,
}

fn main_results(c: &mut Choices) -> Vec<Val> {
    match c.weighted(&[40, 20, 15, 10]) {
        0 => vec![Val::I32],
        1 => vec![Val::I64],
        2 => Vec::new(),
        _ => [
            vec![Val::I32, Val::I64],
            vec![Val::I64, Val::I32],
            vec![Val::I32, Val::I32],
        ][c.below(3)]
        .clone(),
    }
}

fn tables(c: &mut Choices, space: &TypeSpace, func_types: &[usize]) -> Vec<Table> {
    let mut slots = (1..func_types.len()).map(Some).collect::<Vec<_>>();
    if c.chance(30) {
        slots.reverse();
    }
    if c.chance(30) {
        slots.push(Some(0));
    }
    if c.chance(40) {
        slots.push(None);
    }
    let mut tables = vec![Table { elem: Val::Func, slots }];
    if c.chance(25) {
        let k = c.below(func_types.len() - 1);
        let holders = (1..func_types.len())
            .filter(|&f| space.equivalent(func_types[f], k))
            .collect::<Vec<_>>();
        let slots = (0..1 + c.below(3))
            .map(|_| c.chance(75).then(|| c.pick(&holders)))
            .collect();
        tables.push(Table {
            elem: Val::Ref(true, k),
            slots,
        });
    }
    tables
}

fn tables_wat(tables: &[Table]) -> String {
    let mut wat = String::new();
    for (index, table) in tables.iter().enumerate() {
        let _ = writeln!(wat, "(table {} {})", table.slots.len(), table.elem.wat());
        let items = table
            .slots
            .iter()
            .map(|slot| match (slot, table.elem) {
                (Some(func), _) => format!(" (item ref.func $f{func})"),
                (None, Val::Ref(_, k)) => format!(" (item ref.null $t{k})"),
                (None, _) => " (item ref.null func)".into(),
            })
            .collect::<String>();
        let _ = writeln!(wat, "(elem (table {index}) (i32.const 0) {}{items})", table.elem.wat());
    }
    wat
}

fn memory_wat(c: &mut Choices, memory: Option<(u32, Option<u32>)>) -> String {
    let Some((initial, maximum)) = memory else {
        return String::new();
    };
    let maximum = maximum.map(|m| format!(" {m}")).unwrap_or_default();
    let mut wat = format!("(memory {initial}{maximum})\n");
    let bytes = u64::from(initial) * 65536;
    let segments = if initial > 0 { c.below(3) } else { 0 };
    for _ in 0..segments {
        let len = 1 + c.below(8) as u64;
        let offset = if c.chance(50) {
            bytes - len
        } else {
            c.pick(&[0, 4, 8, 100])
        };
        let data = (0..len)
            .map(|_| format!("\\{:02x}", c.next() as u8))
            .collect::<String>();
        let _ = writeln!(wat, "(data (i32.const {offset}) \"{data}\")");
    }
    wat
}

fn globals_wat(c: &mut Choices, globals: &[(Val, bool)]) -> String {
    let mut wat = String::new();
    for &(ty, mutable) in globals {
        let init = match ty {
            Val::I32 => format!("i32.const {}", c.pick(I32_CONSTS)),
            _ => format!("i64.const {}", c.pick(I64_CONSTS)),
        };
        let ty = if mutable {
            format!("(mut {})", ty.wat())
        } else {
            ty.wat()
        };
        let _ = writeln!(wat, "(global {ty} ({init}))");
    }
    let _ = writeln!(wat, "(global (mut i32) (i32.const {FUEL}))");
    wat
}

pub fn module(c: &mut Choices) -> Generated {
    let mut space = TypeSpace::default();
    let user_types = 2 + c.below(4);
    for index in 0..user_types {
        let (ty, class) = space.declaration(c, index, false);
        space.push(ty, class);
    }
    let results = main_results(c);
    let main_type = space.push(
        FuncTy {
            params: Vec::new(),
            results: results.clone(),
        },
        None,
    );
    let func_types = std::iter::once(main_type)
        .chain(0..user_types)
        .collect::<Vec<_>>();
    let tables = tables(c, &space, &func_types);
    let careful = c.chance(70);
    let memory = c.chance(85).then(|| {
        let initial = if careful {
            c.pick(&[1, 1, 2])
        } else {
            c.pick(&[1, 1, 1, 0, 2, 2])
        };
        let maximum = match c.weighted(&[30, 20, 30, 20]) {
            0 => None,
            extra => Some(initial + extra as u32 - 1),
        };
        (initial, maximum)
    });
    let globals = (0..c.below(3))
        .map(|_| (if c.chance(60) { Val::I32 } else { Val::I64 }, c.chance(70)))
        .collect();
    let spec = Spec {
        space,
        func_types,
        tables,
        memory,
        globals,
        careful,
    };

    let mut features = Features::new();
    if results.len() > 1 {
        features.insert("multi_result_main");
    }
    let mut wat = format!("(module\n{}", spec.space.wat());
    wat += &tables_wat(&spec.tables);
    wat += &memory_wat(c, spec.memory);
    wat += &globals_wat(c, &spec.globals);
    let declared = (0..spec.func_types.len())
        .map(|f| format!(" $f{f}"))
        .collect::<String>();
    let _ = writeln!(wat, "(elem declare func{declared})");
    for func in 0..spec.func_types.len() {
        wat += &function(&spec, c, func, &mut features);
    }
    wat += ")\n";
    Generated { wat, features }
}

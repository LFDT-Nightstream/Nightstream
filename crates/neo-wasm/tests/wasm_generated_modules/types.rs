use std::fmt::Write;

pub const I32_CONSTS: &[i32] = &[
    0,
    1,
    2,
    3,
    4,
    5,
    7,
    8,
    15,
    16,
    31,
    32,
    42,
    63,
    64,
    255,
    65535,
    65536,
    65532,
    -1,
    -2,
    -8,
    i32::MIN,
    0x7fffffff,
];
pub const I64_CONSTS: &[i64] = &[
    0,
    1,
    2,
    3,
    31,
    32,
    63,
    64,
    -1,
    -2,
    i64::MIN,
    i64::MAX,
    0xffffffff,
    1 << 32,
    -1 << 40,
];

pub struct Choices<'a> {
    data: &'a [u32],
    at: usize,
}

impl<'a> Choices<'a> {
    pub fn new(data: &'a [u32]) -> Self {
        Self { data, at: 0 }
    }

    pub fn next(&mut self) -> u32 {
        let value = self.data.get(self.at).copied().unwrap_or(0);
        self.at += 1;
        value
    }

    pub fn below(&mut self, n: usize) -> usize {
        if n <= 1 {
            0
        } else {
            self.next() as usize % n
        }
    }

    pub fn chance(&mut self, percent: u32) -> bool {
        self.next() % 100 >= 100 - percent
    }

    pub fn pick<T: Copy>(&mut self, items: &[T]) -> T {
        items[self.below(items.len())]
    }

    pub fn weighted(&mut self, weights: &[u32]) -> usize {
        let mut roll = self.next() % weights.iter().sum::<u32>().max(1);
        for (index, &weight) in weights.iter().enumerate() {
            if roll < weight {
                return index;
            }
            roll -= weight;
        }
        0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Val {
    I32,
    I64,
    Func,
    Extern,
    Ref(bool, usize),
}

impl Val {
    pub fn wat(self) -> String {
        match self {
            Val::I32 => "i32".into(),
            Val::I64 => "i64".into(),
            Val::Func => "funcref".into(),
            Val::Extern => "externref".into(),
            Val::Ref(true, ty) => format!("(ref null $t{ty})"),
            Val::Ref(false, ty) => format!("(ref $t{ty})"),
        }
    }

    pub fn int(self) -> Option<&'static str> {
        match self {
            Val::I32 => Some("i32"),
            Val::I64 => Some("i64"),
            _ => None,
        }
    }

    pub fn defaultable(self) -> Val {
        match self {
            Val::Ref(_, ty) => Val::Ref(true, ty),
            other => other,
        }
    }
}

pub fn subtype(a: Val, b: Val) -> bool {
    match (a, b) {
        _ if a == b => true,
        (Val::Ref(..), Val::Func) => true,
        (Val::Ref(false, x), Val::Ref(true, y)) => x == y,
        _ => false,
    }
}

pub fn subtypes(a: &[Val], b: &[Val]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(&a, &b)| subtype(a, b))
}

pub fn val_list(keyword: &str, vals: &[Val]) -> String {
    if vals.is_empty() {
        return String::new();
    }
    let mut out = format!(" ({keyword}");
    for val in vals {
        out += " ";
        out += &val.wat();
    }
    out + ")"
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FuncTy {
    pub params: Vec<Val>,
    pub results: Vec<Val>,
}

impl FuncTy {
    pub fn blocktype(&self) -> String {
        val_list("param", &self.params) + &val_list("result", &self.results)
    }
}

#[derive(Clone, Default)]
pub struct TypeSpace {
    pub types: Vec<FuncTy>,
    pub equivalence_class: Vec<usize>,
}

impl TypeSpace {
    pub fn push(&mut self, ty: FuncTy, class: Option<usize>) -> usize {
        let index = self.types.len();
        self.types.push(ty);
        self.equivalence_class.push(class.unwrap_or(index));
        index
    }

    pub fn equivalent(&self, a: usize, b: usize) -> bool {
        self.equivalence_class[a] == self.equivalence_class[b]
    }

    pub fn val(&self, c: &mut Choices, index: usize, externs: bool) -> Val {
        match c.weighted(&[45, 22, 8, if externs { 6 } else { 0 }, 25]) {
            0 => Val::I32,
            1 => Val::I64,
            2 => Val::Func,
            3 => Val::Extern,
            _ => Val::Ref(c.chance(60), c.below(index + 1)),
        }
    }

    pub fn vals(&self, c: &mut Choices, index: usize, externs: bool, max: usize) -> Vec<Val> {
        (0..c.below(max + 1))
            .map(|_| self.val(c, index, externs))
            .collect()
    }

    fn equivalent_target(&self, c: &mut Choices, target: usize, index: usize) -> usize {
        let same = (0..index)
            .filter(|&k| self.equivalent(k, target))
            .collect::<Vec<_>>();
        if same.is_empty() {
            target
        } else {
            c.pick(&same)
        }
    }

    fn rewrite(&self, c: &mut Choices, vals: &[Val], of: usize, index: usize, keep_self: bool) -> Vec<Val> {
        vals.iter()
            .map(|&val| match val {
                Val::Ref(null, k) if k == of && keep_self => Val::Ref(null, index),
                Val::Ref(null, k) if k == of => Val::Ref(null, of),
                Val::Ref(null, k) => Val::Ref(null, self.equivalent_target(c, k, index)),
                other => other,
            })
            .collect()
    }

    pub fn variant(&self, c: &mut Choices, of: usize, index: usize) -> FuncTy {
        let ty = &self.types[of];
        FuncTy {
            params: self.rewrite(c, &ty.params, of, index, true),
            results: self.rewrite(c, &ty.results, of, index, true),
        }
    }

    pub fn near_miss(&self, c: &mut Choices, of: usize, index: usize, externs: bool) -> FuncTy {
        let ty = &self.types[of];
        let mut params = self.rewrite(c, &ty.params, of, index, false);
        let results = self.rewrite(c, &ty.results, of, index, false);
        let refers_to_self = ty
            .params
            .iter()
            .chain(&ty.results)
            .any(|&val| val == Val::Ref(true, of) || val == Val::Ref(false, of));
        if !refers_to_self || c.chance(30) {
            if params.is_empty() || c.chance(15) {
                params.push(self.val(c, index, externs));
            } else {
                let at = c.below(params.len());
                params[at] = match params[at] {
                    Val::Ref(null, k) if c.chance(50) => Val::Ref(!null, k),
                    Val::Ref(null, _) if c.chance(50) => Val::Ref(null, c.below(index + 1)),
                    Val::I32 => Val::I64,
                    Val::I64 => Val::I32,
                    _ => self.val(c, index, externs),
                };
            }
        }
        FuncTy { params, results }
    }

    pub fn retarget(&self, c: &mut Choices, indices: &[usize]) -> Option<(usize, FuncTy)> {
        let mut sites = Vec::new();
        for &index in indices {
            let ty = &self.types[index];
            for (at, val) in ty.params.iter().chain(&ty.results).enumerate() {
                let &Val::Ref(null, k) = val else { continue };
                let others = (0..index)
                    .filter(|&other| other != k && k != index && self.equivalent(other, k))
                    .collect::<Vec<_>>();
                if !others.is_empty() {
                    sites.push((index, at, Val::Ref(null, c.pick(&others))));
                }
            }
        }
        let (index, at, val) = *sites.get(c.below(sites.len()))?;
        let mut ty = self.types[index].clone();
        let params = ty.params.len();
        if at < params {
            ty.params[at] = val;
        } else {
            ty.results[at - params] = val;
        }
        Some((index, ty))
    }

    pub fn declaration(&self, c: &mut Choices, index: usize, externs: bool) -> (FuncTy, Option<usize>) {
        let mode = if index == 0 { 0 } else { c.weighted(&[40, 35, 25]) };
        let of = c.below(index);
        match mode {
            0 => (
                FuncTy {
                    params: self.vals(c, index, externs, 3),
                    results: self.vals(c, index, externs, 2),
                },
                None,
            ),
            1 => (self.variant(c, of, index), Some(self.equivalence_class[of])),
            _ => (self.near_miss(c, of, index, externs), None),
        }
    }

    pub fn wat(&self) -> String {
        self.types
            .iter()
            .enumerate()
            .map(|(index, ty)| format!("(type $t{index} (func{}))\n", ty.blocktype()))
            .collect()
    }
}

pub fn type_mutation_pair(c: &mut Choices) -> [String; 2] {
    let leaves = 1 + c.below(4);
    let mut space = TypeSpace::default();
    for index in 0..leaves {
        let (ty, class) = space.declaration(c, index, true);
        space.push(ty, class);
    }
    let call_type = leaves;
    let args = (0..1 + c.below(3))
        .map(|_| {
            if c.chance(50) {
                Val::Ref(c.chance(60), c.below(call_type + 1))
            } else {
                space.val(c, call_type, false)
            }
        })
        .collect::<Vec<_>>();
    space.push(
        FuncTy {
            params: args.clone(),
            results: vec![Val::I32],
        },
        None,
    );
    let callee_type = call_type + 1;
    let (callee, class) = callee_declaration(&space, c, call_type, callee_type);
    space.push(callee, class);
    let main = space.push(
        FuncTy {
            params: Vec::new(),
            results: vec![Val::I32],
        },
        None,
    );

    let mut second = space.clone();
    for _ in 0..1 + c.below(2) {
        let mut retargetable = (0..leaves).collect::<Vec<_>>();
        retargetable.push(callee_type);
        if c.chance(50) {
            if let Some((index, ty)) = second.retarget(c, &retargetable) {
                second.types[index] = ty;
                continue;
            }
        }
        let index = if c.chance(60) { callee_type } else { c.below(leaves) };
        second.types[index] = if index == callee_type {
            callee_declaration(&second, c, call_type, callee_type).0
        } else if c.chance(50) {
            second.variant(c, index, index)
        } else {
            second.declaration(c, index, true).0
        };
    }

    let mut locals = Vec::new();
    let mut code = Vec::new();
    for &arg in &args {
        match arg {
            Val::I32 => code.push(format!("i32.const {}", c.pick(I32_CONSTS))),
            Val::I64 => code.push(format!("i64.const {}", c.pick(I64_CONSTS))),
            Val::Ref(true, _) | Val::Func if c.chance(40) => {
                locals.push(arg);
                code.push(format!("local.get {}", locals.len() - 1));
            }
            Val::Func => code.push(format!("ref.func $f{}", c.below(callee_type + 1))),
            Val::Ref(_, k) => code.push(format!("ref.func $f{k}")),
            Val::Extern => unreachable!("the call type has no externref parameter"),
        }
    }
    code.push(format!("i32.const {callee_type}"));
    code.push(format!("call_indirect (type $t{call_type})"));

    [space, second].map(|space| {
        let mut wat = format!("(module\n{}", space.wat());
        let _ = writeln!(wat, "(table (export \"types\") {} funcref)", callee_type + 1);
        let funcs = (0..=callee_type)
            .map(|f| format!(" $f{f}"))
            .collect::<String>();
        let _ = writeln!(wat, "(elem (i32.const 0) func{funcs})");
        for func in 0..=callee_type {
            let body = if func == callee_type {
                "i32.const 42"
            } else {
                "unreachable"
            };
            let _ = writeln!(wat, "(func $f{func} (type $t{func}) {body})");
        }
        let _ = writeln!(
            wat,
            "(func (export \"main\") (type $t{main}){}\n  {})\n)",
            val_list("local", &locals),
            code.join("\n  ")
        );
        wat
    })
}

fn callee_declaration(
    space: &TypeSpace,
    c: &mut Choices,
    call_type: usize,
    callee_type: usize,
) -> (FuncTy, Option<usize>) {
    let (params, class) = match c.weighted(&[45, 40, 15]) {
        0 => (
            space.variant(c, call_type, callee_type).params,
            Some(space.equivalence_class[call_type]),
        ),
        1 => (space.near_miss(c, call_type, callee_type, true).params, None),
        _ => (space.vals(c, callee_type, true, 3), None),
    };
    (
        FuncTy {
            params,
            results: vec![Val::I32],
        },
        class,
    )
}

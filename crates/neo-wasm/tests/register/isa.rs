//! The register form of a WASM program (scratch prototype, not for review).
//! A program is a list of items: rows, marks, and labels. A row is one
//! register operation. A mark stands for one original WASM instruction and
//! costs nothing; the machine counts the marks it passes, so the count of
//! original instructions on a run is exact. A label is a position in the
//! item list; a jump resumes at its label, so marks after a label count on
//! every path into it and marks before it only on the fall-through path.

/// A location: a frame cell (cell form) or a register (register form).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Loc {
    Cell(u32),
    Reg(u8),
}

/// The second operand: a location or an immediate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Src {
    Loc(Loc),
    Imm(i64),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum W {
    I32,
    I64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Cond {
    Eq,
    Ne,
    LtU,
    LtS,
    GtU,
    GtS,
    LeU,
    LeS,
    GeU,
    GeS,
}

impl Cond {
    pub fn negate(self) -> Self {
        use Cond::*;
        match self {
            Eq => Ne,
            Ne => Eq,
            LtU => GeU,
            LtS => GeS,
            GtU => LeU,
            GtS => LeS,
            LeU => GtU,
            LeS => GtS,
            GeU => LtU,
            GeS => LtS,
        }
    }

    /// The condition with its operands swapped: `a c b` is `b c.swap() a`.
    pub fn swap(self) -> Self {
        use Cond::*;
        match self {
            Eq | Ne => self,
            LtU => GtU,
            LtS => GtS,
            GtU => LtU,
            GtS => LtS,
            LeU => GeU,
            LeS => GeS,
            GeU => LeU,
            GeS => LeS,
        }
    }

    pub fn holds(self, w: W, a: u64, b: u64) -> bool {
        let (ua, ub, sa, sb) = match w {
            W::I32 => (
                a as u32 as u64,
                b as u32 as u64,
                a as u32 as i32 as i64,
                b as u32 as i32 as i64,
            ),
            W::I64 => (a, b, a as i64, b as i64),
        };
        use Cond::*;
        match self {
            Eq => ua == ub,
            Ne => ua != ub,
            LtU => ua < ub,
            LtS => sa < sb,
            GtU => ua > ub,
            GtS => sa > sb,
            LeU => ua <= ub,
            LeS => sa <= sb,
            GeU => ua >= ub,
            GeS => sa >= sb,
        }
    }
}

/// Where a load or a store goes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Space {
    /// WASM linear memory: address `a + imm`.
    Linear,
    /// The frame: cell `fp + imm` (register form only: spills and fills).
    Frame,
    /// Global `imm`.
    Global,
}

/// The one operation of a row.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Op {
    Add,
    Sub,
    Mul,
    And,
    Or,
    Xor,
    Shl,
    ShrU,
    ShrS,
    Rotl,
    Rotr,
    DivU,
    DivS,
    RemU,
    RemS,
    /// `dst = a cond b`; it also sets the flag.
    Cmp(Cond),
    Clz,
    Ctz,
    Popcnt,
    Extend8S,
    Extend16S,
    Extend32S,
    /// `i64.extend_i32_u`: clear the high limb.
    ExtendU32,
    /// `dst = flag ? a : b`.
    Select,
    Load {
        space: Space,
        bytes: u8,
        signed: bool,
    },
    Store {
        space: Space,
        bytes: u8,
    },
    MemSize,
    MemGrow,
    /// `pc = label`.
    Jump,
    /// `pc = label` when `a cond b`.
    Br(Cond),
    /// `pc = table[min(a, n)]`; `imm` is the table id.
    BrTable,
    /// Call function `imm`; the callee's frame starts at cell `delta`.
    Call,
    /// Call table entry `a` with type `imm`.
    CallIndirect,
    Return,
    /// An imported function, as a stub.
    Host,
    Trap,
    Halt,
}

/// Why a row exists: a row of the original program, or glue that a finite
/// register file adds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Tag {
    Core,
    /// A move or a constant that an operand stack shape forces.
    Move,
    Spill,
    Fill,
    Save,
    Restore,
    ArgMove,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Row {
    pub op: Op,
    pub w: W,
    pub dst: Option<Loc>,
    pub a: Option<Loc>,
    pub b: Option<Src>,
    /// Immediate: memory offset, global index, function, table id, type.
    pub imm: i64,
    /// Branch target.
    pub label: Option<usize>,
    /// Call: the callee frame starts at this cell of the caller frame.
    pub delta: u32,
    pub tag: Tag,
}

impl Row {
    pub fn new(op: Op, w: W) -> Self {
        Self {
            op,
            w,
            dst: None,
            a: None,
            b: None,
            imm: 0,
            label: None,
            delta: 0,
            tag: Tag::Core,
        }
    }

    /// The locations the row reads.
    pub fn uses(&self) -> impl Iterator<Item = Loc> + '_ {
        let b = match self.b {
            Some(Src::Loc(loc)) => Some(loc),
            _ => None,
        };
        self.a.into_iter().chain(b)
    }
}

#[derive(Clone, Debug)]
pub enum Item {
    Row(Row),
    /// One original WASM instruction, by mnemonic.
    Mark(&'static str),
    Label(usize),
}

/// A function in register form.
#[derive(Clone, Debug)]
pub struct Function {
    pub params: u32,
    pub results: u32,
    /// Frame cells: params, locals, operand stack slots (cell form), or the
    /// spill homes (register form).
    pub cells: u32,
    pub entry: usize,
    pub items: Vec<Item>,
    pub labels: Vec<usize>,
    pub host: bool,
}

/// A whole program: functions, tables, globals, memory.
#[derive(Clone, Debug, Default)]
pub struct Program {
    pub functions: Vec<Function>,
    /// `br_table` tables: labels, the last one the default.
    pub tables: Vec<Vec<usize>>,
    /// The function table: function index per entry (`None` for a hole).
    pub elements: Vec<Option<u32>>,
    /// The type id of each function, for `call_indirect`.
    pub types: Vec<u32>,
    /// `(params, results)` of each type id.
    pub signatures: Vec<(u32, u32)>,
    pub globals: Vec<u64>,
    /// Whether each global is 64 bits wide.
    pub global_wide: Vec<bool>,
    pub memory_pages: u64,
    pub data: Vec<(u64, Vec<u8>)>,
}

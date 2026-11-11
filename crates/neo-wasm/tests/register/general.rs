//! The general register row as a Nebula application (scratch prototype):
//! the relation of `row.rs`, the memory catalog (ROM `code` and `imm` at
//! `pc`, RAM `lin`, `frame` and `glob`, ROM `table`), the continuity links of
//! the row state, the Small geometry with two slots per row, the preload of
//! a lowered program, and the step witnesses of a `Cpu` trace.

use std::num::NonZeroUsize;

use nightstream::nebula::{
    batched_slots, pack_regions, Geometry, R1csApplication, R1csOptions, RowRelation, Shared, Size,
};
use nightstream::relation::{
    ColumnFamilySpec, ColumnRegistry, ColumnWidth, ContinuityCatalog, ContinuityGroup, ContinuityLink, MemoryCatalog,
    MemoryKind, MemoryPortActivation, MemoryPortKind, MemoryPortSpec, MemoryPreload, MemorySpec,
};
use nightstream::Goldilocks as F;

use super::cpu::RowInput;
use super::lower::{Lowered, K};
use super::row::{synthesize, Cs, Layout, LIN_WIDTH};

pub const ROWS: usize = 64;
pub const STEPS: usize = 8;
pub const CODE: u32 = 0;
pub const IMM: u32 = 1;
pub const LIN: u32 = 2;
pub const FRAME: u32 = 3;
pub const GLOB: u32 = 4;
pub const TABLE: u32 = 5;
pub const FRAME_WIDTH: usize = 8;
pub const GLOB_WIDTH: usize = 6;
pub const TABLE_WIDTH: usize = 8;

/// One circuit of the study: the program size it holds, the ROM size, and
/// whether it has the redesign of milestone 5 (continuity by column
/// identity, read stamps derived from `diff`, word bits from the code slot).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Plan {
    pub code_width: usize,
    pub r: usize,
    pub redesign: bool,
}

/// The circuit of milestones 2 to 4: programs of 4,096 rows, ROM `2^14`.
pub const BASELINE: Plan = Plan {
    code_width: 12,
    r: 14,
    redesign: false,
};

/// The redesign at the baseline sizes.
pub const REDESIGN: Plan = Plan {
    code_width: 12,
    r: 14,
    redesign: true,
};

/// The redesign for contracts: programs of 1,024 rows, ROM `2^12` (every
/// Starstream program of the corpus has at most 732 rows).
pub const CONTRACT: Plan = Plan {
    code_width: 10,
    r: 12,
    redesign: true,
};

pub fn zero_input() -> RowInput {
    RowInput {
        pc: 0,
        fp: 0,
        flag: false,
        regs: [0; K],
        word: 0,
        immw: 0,
        old: 0,
    }
}

pub fn cs(plan: Plan, check: bool) -> Cs {
    let mut cs = Cs::new(check);
    cs.slot_word = plan.redesign;
    cs
}

/// The relation of one row and its layout.
pub fn shape(plan: Plan) -> (Cs, Layout) {
    let mut cs = cs(plan, false);
    let layout = synthesize(&mut cs, &zero_input());
    (cs, layout)
}

/// One family per run of field columns and one per activation bit, which
/// the catalog requires to be Boolean.
fn registry(l: &Layout) -> ColumnRegistry {
    let mut bools = vec![l.memrow, l.linact, l.frameact, l.globact, l.tabact];
    bools.sort();
    let mut families = Vec::new();
    let mut start = 0;
    let family = |start: usize, len: usize, width: ColumnWidth| ColumnFamilySpec {
        region: "general register row",
        start,
        len,
        name: format!("columns {start}").into(),
        role: "prototype",
        width,
    };
    for b in bools {
        if b > start {
            families.push(family(start, b - start, ColumnWidth::Field));
        }
        families.push(family(b, 1, ColumnWidth::Boolean));
        start = b + 1;
    }
    families.push(family(start, l.width - start, ColumnWidth::Field));
    ColumnRegistry::new(families).unwrap()
}

pub fn links(l: &Layout) -> Vec<(usize, usize)> {
    let mut out = vec![(l.pc1, l.pc), (l.fp1, l.fp), (l.flag1, l.flag)];
    for kk in 0..K {
        for i in 0..2 {
            out.push((l.regs1[kk][i], l.regs[kk][i]));
        }
    }
    out
}

pub fn catalog(l: &Layout) -> MemoryCatalog<u32> {
    let read = |id, address, value, activation, kind| MemorySpec {
        id,
        kind,
        ports: vec![MemoryPortSpec {
            address_columns: vec![address],
            value_column: value,
            kind: MemoryPortKind::Read,
            activation,
        }],
    };
    let write = |id, address, activation| MemorySpec {
        id,
        kind: MemoryKind::Ram,
        ports: vec![MemoryPortSpec {
            address_columns: vec![address],
            value_column: l.new,
            kind: MemoryPortKind::Write {
                value_before_column: Some(l.old),
            },
            activation,
        }],
    };
    MemoryCatalog::new(
        [
            read(CODE, l.pc, l.word, MemoryPortActivation::Always, MemoryKind::Rom),
            read(
                IMM,
                l.pc,
                l.immw,
                MemoryPortActivation::Unless(l.memrow),
                MemoryKind::Rom,
            ),
            write(LIN, l.laddr, MemoryPortActivation::When(l.linact)),
            write(FRAME, l.faddr, MemoryPortActivation::When(l.frameact)),
            write(GLOB, l.gaddr, MemoryPortActivation::When(l.globact)),
            read(
                TABLE,
                l.taddr,
                l.old,
                MemoryPortActivation::When(l.tabact),
                MemoryKind::Rom,
            ),
        ],
        &registry(&l),
    )
    .unwrap()
}

/// The geometry (r of the plan, μ 10, 8 steps a segment, 32 segments) with
/// two operation slots for each of the 64 rows of a step.
pub fn geometry(plan: Plan) -> Geometry {
    Geometry::new(plan.r, 10, 2 * ROWS, STEPS, 32).unwrap()
}

pub fn application(plan: Plan) -> (R1csApplication<u32>, Layout) {
    let (cs, l) = shape(plan);
    let relation = RowRelation {
        rows: cs.rows,
        width: l.width,
        one: 0,
        shared: Shared::NONE,
    };
    let continuity = ContinuityCatalog::new(
        [ContinuityGroup {
            name: "row state",
            role: "prototype",
            links: links(&l)
                .into_iter()
                .map(|(previous_step_column, next_step_column)| ContinuityLink {
                    previous_step_column,
                    next_step_column,
                })
                .collect(),
        }],
        &registry(&l),
    )
    .unwrap();
    let catalog = catalog(&l);
    let (regions, _) = pack_regions([
        (CODE, MemoryKind::Rom, vec![plan.code_width]),
        (IMM, MemoryKind::Rom, vec![plan.code_width]),
        (TABLE, MemoryKind::Rom, vec![TABLE_WIDTH]),
        (LIN, MemoryKind::Ram, vec![LIN_WIDTH]),
        (FRAME, MemoryKind::Ram, vec![FRAME_WIDTH]),
        (GLOB, MemoryKind::Ram, vec![GLOB_WIDTH]),
    ])
    .unwrap();
    let rows = NonZeroUsize::new(ROWS).unwrap();
    let size = Size {
        geometry: geometry(plan),
        rows,
        regions,
        slots: batched_slots(&catalog, &[vec![0], vec![1, 2, 3, 4, 5]], rows).unwrap(),
    };
    let options = R1csOptions {
        chained: plan.redesign,
        derived_read_stamps: plan.redesign,
        slot_bits: if plan.redesign {
            (0..32).map(|k| (l.wb[k], 0, k)).collect()
        } else {
            Vec::new()
        },
    };
    let app = R1csApplication::with_options(relation, &continuity, &catalog, size, options).unwrap();
    (app, l)
}

/// The preload of a lowered program: its words, immediates, branch table,
/// data segments, and globals.
pub fn preload(low: &Lowered, plan: Plan) -> MemoryPreload<u32> {
    let mut p = MemoryPreload::default();
    assert!(
        low.rows.len() <= 1 << plan.code_width,
        "{} rows of program",
        low.rows.len()
    );
    for (pc, row) in low.rows.iter().enumerate() {
        p.insert(CODE, vec![pc as u32], row.word());
        if !row.fam().memory() && row.imm != 0 {
            p.insert(IMM, vec![pc as u32], row.imm);
        }
    }
    for (i, &t) in low.table.iter().enumerate() {
        p.insert(TABLE, vec![i as u32], t);
    }
    let mut words = vec![0u32; 1 << LIN_WIDTH];
    for (offset, bytes) in &low.program.data {
        for (i, byte) in bytes.iter().enumerate() {
            let at = *offset as usize + i;
            assert!(at / 4 < words.len(), "data at {at} is outside the linear-memory region");
            words[at / 4] |= u32::from(*byte) << (8 * (at % 4));
        }
    }
    for (i, &w) in words.iter().enumerate() {
        if w != 0 {
            p.insert(LIN, vec![i as u32], w);
        }
    }
    for (i, &g) in low.program.globals.iter().enumerate() {
        for (limb, w) in [(0, g as u32), (1, (g >> 32) as u32)] {
            if w != 0 {
                p.insert(GLOB, vec![(2 * i + limb) as u32], w);
            }
        }
    }
    p
}

/// The honest one-row witness of each row of `trace`, checking every row
/// as it is built and that each row ends where the next one starts.
pub fn witnesses(trace: &[RowInput], plan: Plan) -> Vec<Vec<F>> {
    let mut out: Vec<Vec<F>> = Vec::with_capacity(trace.len());
    let mut previous: Option<(Layout, Vec<F>)> = None;
    for input in trace {
        let mut cs = cs(plan, true);
        let l = synthesize(&mut cs, input);
        if let Some((pl, pz)) = &previous {
            for (after, before) in links(pl) {
                let link = links(&l).into_iter().find(|link| link.0 == after).unwrap();
                assert_eq!(
                    pz[after],
                    cs.z[link.1],
                    "row {} starts where row {} ends",
                    out.len(),
                    out.len() - 1
                );
                let _ = before;
            }
        }
        previous = Some((l, cs.z.clone()));
        out.push(cs.z);
    }
    out
}

/// The state words of a row's start (link order).
pub fn state(input: &RowInput) -> Vec<u64> {
    let mut s = vec![u64::from(input.pc), u64::from(input.fp), u64::from(input.flag)];
    for r in input.regs {
        s.extend([r & 0xffff_ffff, r >> 32]);
    }
    s
}

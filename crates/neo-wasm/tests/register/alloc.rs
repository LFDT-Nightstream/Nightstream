//! The register form: a file of `K` registers in the row state, every other
//! frame cell spilled to its home `fp + cell` in the frame memory (scratch
//! prototype). Per function the allocator pins cells `0..max(params,
//! results)` to registers `0..`, gives the other registers to the cells with
//! the most uses (each use weighted 8 per enclosing loop), and keeps the last
//! two registers as scratch for fills and spills. A call saves the
//! registers whose cells are live after it (caller-save), moves the
//! arguments into registers `0..params`, and takes its results from
//! registers `0..results`. The callee frame starts after the caller's homes.

use super::isa::{Function, Item, Loc, Op, Program, Row, Space, Src, Tag, W};

type Set = Vec<u64>;

fn set(n: usize) -> Set {
    vec![0; n.div_ceil(64)]
}

fn has(s: &Set, i: u32) -> bool {
    s[i as usize / 64] >> (i % 64) & 1 == 1
}

fn add(s: &mut Set, i: u32) {
    s[i as usize / 64] |= 1 << (i % 64);
}

fn cell(loc: Loc) -> u32 {
    match loc {
        Loc::Cell(c) => c,
        Loc::Reg(_) => panic!("register form input"),
    }
}

/// `(params, results)` of a call row.
fn arity(program: &Program, row: &Row) -> (u32, u32) {
    match row.op {
        Op::Call | Op::Host => {
            let f = &program.functions[row.imm as usize];
            (f.params, f.results)
        }
        Op::CallIndirect => program.signatures[row.imm as usize],
        _ => (0, 0),
    }
}

fn is_call(row: &Row) -> bool {
    matches!(row.op, Op::Call | Op::CallIndirect | Op::Host)
}

/// The cells a row reads and writes.
fn uses_defs(program: &Program, f: &Function, row: &Row) -> (Vec<u32>, Vec<u32>) {
    let mut uses: Vec<u32> = row.uses().map(cell).collect();
    let mut defs: Vec<u32> = row.dst.map(cell).into_iter().collect();
    if is_call(row) {
        let (p, r) = arity(program, row);
        uses.extend(row.delta..row.delta + p);
        defs.extend(row.delta..row.delta + r);
    }
    if row.op == Op::Return {
        uses.extend(0..f.results);
    }
    (uses, defs)
}

/// The cells live after each row of `f` (rows in item order).
fn liveness(program: &Program, f: &Function) -> (Vec<usize>, Vec<Set>) {
    let rows: Vec<usize> = (0..f.items.len())
        .filter(|&i| matches!(f.items[i], Item::Row(_)))
        .collect();
    let row_at = |item: usize| rows.partition_point(|&r| r < item);
    let n = f.cells as usize;
    let succ: Vec<Vec<usize>> = rows
        .iter()
        .enumerate()
        .map(|(k, &i)| {
            let Item::Row(row) = &f.items[i] else { unreachable!() };
            let to = |label: usize| row_at(f.labels[label]);
            let mut s = Vec::new();
            match row.op {
                Op::Jump => s.push(to(row.label.unwrap())),
                Op::Br(_) => s.extend([to(row.label.unwrap()), k + 1]),
                Op::BrTable => s.extend(program.tables[row.imm as usize].iter().map(|&l| to(l))),
                Op::Return | Op::Trap | Op::Halt => {}
                _ => s.push(k + 1),
            }
            s.retain(|&t| t < rows.len());
            s
        })
        .collect();
    let ud: Vec<(Vec<u32>, Vec<u32>)> = rows
        .iter()
        .map(|&i| {
            let Item::Row(row) = &f.items[i] else { unreachable!() };
            uses_defs(program, f, row)
        })
        .collect();
    let mut live_in = vec![set(n); rows.len()];
    let mut live_out = vec![set(n); rows.len()];
    let mut changed = true;
    while changed {
        changed = false;
        for k in (0..rows.len()).rev() {
            let mut out = set(n);
            for &t in &succ[k] {
                for (o, i) in out.iter_mut().zip(&live_in[t]) {
                    *o |= i;
                }
            }
            let mut inn = out.clone();
            for &d in &ud[k].1 {
                inn[d as usize / 64] &= !(1 << (d % 64));
            }
            for &u in &ud[k].0 {
                add(&mut inn, u);
            }
            if inn != live_in[k] || out != live_out[k] {
                live_in[k] = inn;
                live_out[k] = out;
                changed = true;
            }
        }
    }
    (rows, live_out)
}

/// `dst ← src` between two cells: the row writes `src` unchanged.
fn copy(row: &Row) -> Option<(u32, u32)> {
    match (row.op, row.w, row.a, row.b, row.dst) {
        (Op::Add, W::I64, Some(Loc::Cell(s)), Some(Src::Imm(0)), Some(Loc::Cell(d))) => Some((d, s)),
        _ => None,
    }
}

/// The register of each cell (`None`: spilled), by graph colouring: two
/// cells share a register only when no row defines one while the other is
/// live after it. Cells `0..pinned` keep registers `0..pinned`; the others
/// take, heaviest first, a free register that a copy partner or a call
/// argument position prefers, else the lowest free one.
fn assign(program: &Program, f: &Function, k: usize) -> Result<Vec<Option<u8>>, String> {
    let avail = k - 2;
    if f.results as usize > avail {
        return Err(format!("{} results for {avail} registers", f.results));
    }
    let pinned = pinned(f.params, f.results, k);
    let n = f.cells as usize;
    let (rows, live_out) = liveness(program, f);
    let mut edges = vec![std::collections::BTreeSet::<u32>::new(); n];
    let mut prefer: Vec<Vec<u32>> = vec![Vec::new(); n];
    let mut hint: Vec<Option<u8>> = vec![None; n];
    let link = |a: u32, b: u32, edges: &mut Vec<std::collections::BTreeSet<u32>>| {
        if a != b {
            edges[a as usize].insert(b);
            edges[b as usize].insert(a);
        }
    };
    for (r, &i) in rows.iter().enumerate() {
        let Item::Row(row) = &f.items[i] else { unreachable!() };
        let (_, defs) = uses_defs(program, f, row);
        let source = copy(row).map(|(_, s)| s);
        for &d in &defs {
            for c in 0..n as u32 {
                if has(&live_out[r], c) && Some(c) != source {
                    link(d, c, &mut edges);
                }
            }
        }
        if let Some((d, s)) = copy(row) {
            prefer[d as usize].push(s);
            prefer[s as usize].push(d);
        }
        if is_call(row) && row.op != Op::Host {
            let (p, res) = arity(program, row);
            for i in 0..p.max(res) {
                let c = (row.delta + i) as usize;
                if c < n && i < pinned_count(p, res, k) {
                    hint[c] = Some(i as u8);
                }
            }
        }
    }
    if let Some(&first) = rows.first() {
        let live_in: Vec<u32> = {
            let Item::Row(row) = &f.items[first] else {
                unreachable!()
            };
            let (uses, defs) = uses_defs(program, f, row);
            let mut s: Vec<u32> = (0..n as u32)
                .filter(|&c| has(&live_out[0], c) && !defs.contains(&c))
                .collect();
            s.extend(uses);
            s.extend(0..f.params);
            s.sort();
            s.dedup();
            s
        };
        for (x, &a) in live_in.iter().enumerate() {
            for &b in &live_in[x + 1..] {
                link(a, b, &mut edges);
            }
        }
    }
    let mut weight = vec![0u64; f.cells as usize];
    let mut depth = vec![0u32; f.items.len()];
    for (i, item) in f.items.iter().enumerate() {
        if let Item::Row(row) = item {
            if let Some(label) = row.label {
                let start = f.labels[label];
                if start <= i {
                    for d in &mut depth[start..=i] {
                        *d += 1;
                    }
                }
            }
        }
    }
    for (i, item) in f.items.iter().enumerate() {
        if let Item::Row(row) = item {
            let (uses, defs) = uses_defs(program, f, row);
            for c in uses.into_iter().chain(defs) {
                assert!(c < f.cells, "cell {c} of {} cells in {row:?}", f.cells);
                weight[c as usize] += 8u64.saturating_pow(depth[i].min(6));
            }
        }
    }
    let mut regs: Vec<Option<u8>> = vec![None; f.cells as usize];
    for (c, reg) in regs.iter_mut().enumerate().take(pinned) {
        *reg = Some(c as u8);
    }
    let mut order: Vec<usize> = (pinned..f.cells as usize)
        .filter(|&c| weight[c] > 0)
        .collect();
    order.sort_by_key(|&c| std::cmp::Reverse(weight[c]));
    for c in order {
        let taken: Vec<u8> = edges[c].iter().filter_map(|&o| regs[o as usize]).collect();
        let free = |r: u8| (r as usize) < avail && !taken.contains(&r);
        let wanted = prefer[c]
            .iter()
            .filter_map(|&o| regs[o as usize])
            .chain(hint[c])
            .find(|&r| free(r));
        regs[c] = wanted.or_else(|| (0..avail as u8).find(|&r| free(r)));
    }
    Ok(regs)
}

/// The cells `0..pinned` that sit in registers `0..pinned` at a call and a
/// return; any other parameter is passed in its home in the callee frame.
fn pinned(params: u32, results: u32, k: usize) -> usize {
    (params.max(results) as usize).min(k - 2)
}

fn pinned_count(params: u32, results: u32, k: usize) -> u32 {
    pinned(params, results, k) as u32
}

fn frame_row(load: bool, reg: u8, home: u32, tag: Tag) -> Row {
    let op = if load {
        Op::Load {
            space: Space::Frame,
            bytes: 8,
            signed: false,
        }
    } else {
        Op::Store {
            space: Space::Frame,
            bytes: 8,
        }
    };
    let mut row = Row::new(op, W::I64);
    if load {
        row.dst = Some(Loc::Reg(reg));
    } else {
        row.b = Some(Src::Loc(Loc::Reg(reg)));
    }
    row.imm = i64::from(home);
    row.tag = tag;
    row
}

fn move_row(dst: u8, src: u8, tag: Tag) -> Row {
    let mut row = Row::new(Op::Add, W::I64);
    row.dst = Some(Loc::Reg(dst));
    row.a = Some(Loc::Reg(src));
    row.b = Some(Src::Imm(0));
    row.tag = tag;
    row
}

/// A parallel move of registers or homes into registers, with `scratch`
/// to break a cycle.
fn parallel(out: &mut Vec<Item>, moves: Vec<(u8, Result<u8, u32>)>, scratch: u8, tag: Tag) {
    let mut pending: Vec<(u8, u8)> = Vec::new();
    let mut fills = Vec::new();
    for (dst, src) in moves {
        match src {
            Ok(src) if src != dst => pending.push((dst, src)),
            Ok(_) => {}
            Err(home) => fills.push((dst, home)),
        }
    }
    while !pending.is_empty() {
        if let Some(i) = (0..pending.len()).find(|&i| pending.iter().all(|&(_, s)| s != pending[i].0)) {
            let (dst, src) = pending.remove(i);
            out.push(Item::Row(move_row(dst, src, tag)));
        } else {
            let (_, src) = pending[0];
            out.push(Item::Row(move_row(scratch, src, tag)));
            for m in &mut pending {
                if m.1 == src {
                    m.1 = scratch;
                }
            }
        }
    }
    for (dst, home) in fills {
        out.push(Item::Row(frame_row(true, dst, home, tag)));
    }
}

/// A host call in place: the host clobbers no register, so nothing is
/// saved. Each row absorbs two argument registers (a spilled argument is
/// filled into a scratch register first); the last absorbing row writes
/// result 0 and one more row writes each other result.
fn host(items: &mut Vec<Item>, program: &Program, row: &Row, at: &dyn Fn(u32) -> Result<u8, u32>, (s0, s1): (u8, u8)) {
    let callee = &program.functions[row.imm as usize];
    let (p, r) = (callee.params, callee.results);
    let absorbing = p.div_ceil(2).max(1);
    let write = |items: &mut Vec<Item>, mut out: Row, result: Option<u32>| {
        let spill = result.and_then(|c| at(row.delta + c).err());
        out.dst = result.map(|c| Loc::Reg(at(row.delta + c).unwrap_or(s0)));
        items.push(Item::Row(out));
        if let Some(home) = spill {
            items.push(Item::Row(frame_row(false, s0, home, Tag::Spill)));
        }
    };
    for k in 0..absorbing {
        let mut out = Row::new(Op::Host, W::I64);
        out.imm = row.imm;
        out.delta = u32::MAX;
        for (i, scratch) in [(2 * k, s0), (2 * k + 1, s1)] {
            if i >= p {
                continue;
            }
            let reg = match at(row.delta + i) {
                Ok(reg) => reg,
                Err(home) => {
                    items.push(Item::Row(frame_row(true, scratch, home, Tag::Fill)));
                    scratch
                }
            };
            if i % 2 == 0 {
                out.a = Some(Loc::Reg(reg));
            } else {
                out.b = Some(Src::Loc(Loc::Reg(reg)));
            }
        }
        let last = k + 1 == absorbing;
        write(items, out, (last && r > 0).then_some(0));
    }
    for c in 1..r {
        let mut out = Row::new(Op::Host, W::I64);
        out.imm = row.imm;
        out.delta = u32::MAX;
        write(items, out, Some(c));
    }
}

fn rewrite(program: &Program, f: &Function, k: usize) -> Result<Function, String> {
    let regs = assign(program, f, k)?;
    let (rows, live_out) = liveness(program, f);
    let (s0, s1) = ((k - 2) as u8, (k - 1) as u8);
    let mut items = Vec::with_capacity(f.items.len() * 2);
    let mut labels = f.labels.clone();
    let mut row_index = 0;
    let at = |c: u32| regs[c as usize].ok_or(c);
    let own = pinned(f.params, f.results, k) as u32;
    for item in &f.items {
        let row = match item {
            Item::Row(row) => row,
            Item::Label(l) => {
                labels[*l] = items.len();
                items.push(item.clone());
                if *l == f.entry {
                    for c in own..f.params {
                        if let Some(reg) = regs[c as usize] {
                            items.push(Item::Row(frame_row(true, reg, c, Tag::ArgMove)));
                        }
                    }
                }
                continue;
            }
            Item::Mark(_) => {
                items.push(item.clone());
                continue;
            }
        };
        let live = &live_out[row_index];
        debug_assert_eq!(rows[row_index], rows[row_index]);
        row_index += 1;
        if row.op == Op::Host {
            host(&mut items, program, row, &at, (s0, s1));
            continue;
        }
        if is_call(row) {
            let (p, r) = arity(program, row);
            let results: Vec<u32> = (row.delta..row.delta + r).collect();
            let saved: Vec<u32> = (0..f.cells)
                .filter(|&c| has(live, c) && !results.contains(&c) && regs[c as usize].is_some())
                .collect();
            for &c in &saved {
                items.push(Item::Row(frame_row(false, regs[c as usize].unwrap(), c, Tag::Save)));
            }
            let mut call = row.clone();
            if let Some(a) = row.a {
                match at(cell(a)) {
                    Ok(reg) => items.push(Item::Row(move_row(s1, reg, Tag::ArgMove))),
                    Err(home) => items.push(Item::Row(frame_row(true, s1, home, Tag::ArgMove))),
                }
                call.a = Some(Loc::Reg(s1));
            }
            let callee = pinned(p, r, k) as u32;
            for i in callee..p {
                let home = f.cells + 1 + i;
                match at(row.delta + i) {
                    Ok(reg) => items.push(Item::Row(frame_row(false, reg, home, Tag::ArgMove))),
                    Err(c) => {
                        items.push(Item::Row(frame_row(true, s0, c, Tag::ArgMove)));
                        items.push(Item::Row(frame_row(false, s0, home, Tag::ArgMove)));
                    }
                }
            }
            if r as usize > k - 2 {
                return Err(format!("{r} results"));
            }
            let moves = (0..callee).map(|i| (i as u8, at(row.delta + i))).collect();
            parallel(&mut items, moves, s0, Tag::ArgMove);
            call.delta = f.cells + 1;
            items.push(Item::Row(call));
            let back = results
                .iter()
                .enumerate()
                .filter_map(|(i, &c)| regs[c as usize].map(|reg| (reg, Ok(i as u8))))
                .collect();
            for (i, &c) in results.iter().enumerate() {
                if regs[c as usize].is_none() {
                    items.push(Item::Row(frame_row(false, i as u8, c, Tag::ArgMove)));
                }
            }
            parallel(&mut items, back, s0, Tag::ArgMove);
            for &c in &saved {
                items.push(Item::Row(frame_row(true, regs[c as usize].unwrap(), c, Tag::Restore)));
            }
            continue;
        }
        if let Some((d, s)) = copy(row) {
            if regs[d as usize].is_some() && regs[d as usize] == regs[s as usize] {
                continue;
            }
        }
        let mut out = row.clone();
        if let Some(a) = row.a {
            out.a = Some(match at(cell(a)) {
                Ok(reg) => Loc::Reg(reg),
                Err(home) => {
                    items.push(Item::Row(frame_row(true, s0, home, Tag::Fill)));
                    Loc::Reg(s0)
                }
            });
        }
        if let Some(Src::Loc(b)) = row.b {
            out.b = Some(Src::Loc(match at(cell(b)) {
                Ok(reg) => Loc::Reg(reg),
                Err(home) => {
                    items.push(Item::Row(frame_row(true, s1, home, Tag::Fill)));
                    Loc::Reg(s1)
                }
            }));
        }
        let spill = row.dst.and_then(|d| at(cell(d)).err());
        if let Some(d) = row.dst {
            out.dst = Some(Loc::Reg(regs[cell(d) as usize].unwrap_or(s0)));
        }
        items.push(Item::Row(out));
        if let Some(home) = spill {
            items.push(Item::Row(frame_row(false, s0, home, Tag::Spill)));
        }
    }
    Ok(Function {
        items,
        labels,
        ..f.clone()
    })
}

/// The register form of `program` with `k` registers.
pub fn allocate(program: &Program, k: usize) -> Result<Program, String> {
    let functions = program
        .functions
        .iter()
        .map(|f| if f.host { Ok(f.clone()) } else { rewrite(program, f, k) })
        .collect::<Result<_, _>>()?;
    Ok(Program {
        functions,
        ..program.clone()
    })
}

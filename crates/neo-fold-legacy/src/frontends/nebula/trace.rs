//! Nebula native memory machine — the first pass of the two-pass prover and
//! the test oracle. Never verifier authority.
//!
//! Owns: sequential-consistency semantics — cell state, the global
//! never-resetting timestamp, RS/WS tuple emission per op, IS/FS snapshots
//! per segment, and segment op-capacity/step-chunking bookkeeping.
//!
//! Does not own: bit encodings ([`super::layout`]), fingerprint math
//! ([`super::fingerprint`]), circuits, commitments, or any accept/reject
//! decision of the protocol.
//!
//! ## Semantics
//!
//! Every RAM/ROM op — read or write — touches exactly one cell and emits
//! one RS tuple (the cell's previous `(t, v)`) and one WS tuple (the new
//! ones); stack ops (v3.1) emit exactly one:
//!
//! ```text
//! read(seg, a):      RS += (rt, g, v)    WS += (ts+1, g, v)     cell.t ← ts+1
//! write(seg, a, v'): RS += (rt, g, v)    WS += (ts+1, g, v')    cell ← (v', ts+1)
//! push(s, v):                            WS += (ts+1, g_sp, v)  sp += 1
//! pop(s):            RS += (pt, g_sp, v)                        sp -= 1
//! ```
//!
//! with `ts` incremented per op, globally across all segments. Writes to
//! the ROM namespace are rejected; stacks are segment-local —
//! [`SegmentRun::finish`] rejects a segment that leaves a stack
//! non-empty. For an honest run, `IS ∪ WS = RS ∪ FS` holds exactly (Blum
//! et al. invariant, Nebula Lemma 7; pushes cancel their pops) — tests
//! assert product balance without any soundness slack.

use neo_math::K;
use thiserror::Error;

use crate::frontends::nebula::fingerprint::{self, Gammas, MemTuple};
use crate::frontends::nebula::layout::{CellRecord, MemOpRecord, MemSpace, NebulaParams, TS_BITS};

#[derive(Debug, Error, PartialEq, Eq)]
pub enum TraceError {
    #[error("ROM image must have exactly {want} cells, got {got}")]
    RomImageLen { want: u64, got: usize },
    #[error("RAM image must have exactly {want} cells, got {got}")]
    RamImageLen { want: u64, got: usize },
    #[error("write to public ROM (addr {0})")]
    RomWrite(u64),
    #[error("address {addr} out of range for namespace of {cells} cells")]
    AddrRange { addr: u64, cells: u64 },
    #[error("stack {got} does not exist (plan has {stacks} stacks)")]
    StackIndex { got: u8, stacks: usize },
    #[error("stack {0} is full (capacity 2^σ − 1)")]
    StackOverflow(u8),
    #[error("pop from empty stack {0}")]
    StackUnderflow(u8),
    #[error("segment close with {live} live cells on stack {stack} (stacks are segment-local)")]
    StackNotEmpty { stack: u8, live: usize },
    #[error("segment is full ({0} ops)")]
    SegmentFull(usize),
    #[error("chain exceeds seg_max segments")]
    TooManySegments,
    #[error("global timestamp reached 2^{TS_BITS}")]
    TsExhausted,
}

/// The memory of one chain: ROM image plus RAM, with per-cell last-access
/// timestamps and the global op timestamp. Lives across segments.
#[derive(Clone, Debug)]
pub struct Memory {
    params: NebulaParams,
    /// Indexed by global cell index `g`; ROM occupies `[0, R)`. Scanned
    /// cells only — stacks live in `stacks`, never here.
    cells: Vec<CellRecord>,
    /// Live stack cells, bottom to top: `(value, push time)`. Stacks are
    /// segment-local, so these are empty at every segment boundary.
    stacks: Vec<Vec<(u32, u64)>>,
    /// Global timestamp: total ops applied so far, across all segments.
    ts: u64,
    /// Next segment counter.
    seg_idx: u64,
}

impl Memory {
    /// Fresh chain-start memory with zero-initialized RAM.
    pub fn new(params: NebulaParams, rom_image: &[u32]) -> Result<Self, TraceError> {
        let ram_image = vec![0; params.ram_cells() as usize];
        Self::new_with_initial_ram(params, rom_image, &ram_image)
    }

    /// Fresh chain-start memory: ROM and RAM cells hold the
    /// verifier-owned initial images, stacks are empty, and every timestamp is 0.
    pub fn new_with_initial_ram(
        params: NebulaParams,
        rom_image: &[u32],
        ram_image: &[u32],
    ) -> Result<Self, TraceError> {
        if rom_image.len() as u64 != params.rom_cells() {
            return Err(TraceError::RomImageLen {
                want: params.rom_cells(),
                got: rom_image.len(),
            });
        }
        if ram_image.len() as u64 != params.ram_cells() {
            return Err(TraceError::RamImageLen {
                want: params.ram_cells(),
                got: ram_image.len(),
            });
        }
        let mut cells = Vec::with_capacity(params.scanned_cells() as usize);
        cells.extend(rom_image.iter().map(|&v| CellRecord { v, t: 0 }));
        cells.extend(ram_image.iter().map(|&v| CellRecord { v, t: 0 }));
        Ok(Self {
            params,
            cells,
            stacks: vec![Vec::new(); params.num_stacks],
            ts: 0,
            seg_idx: 0,
        })
    }

    /// Global timestamp (ops applied so far). Never resets.
    pub fn ts(&self) -> u64 {
        self.ts
    }

    /// The plan this memory was built under.
    pub fn params(&self) -> &NebulaParams {
        &self.params
    }

    /// Open the next segment. The IS snapshot is taken here; ops are
    /// applied through the returned handle until [`SegmentRun::finish`].
    pub fn begin_segment(&mut self) -> Result<SegmentRun<'_>, TraceError> {
        if self.seg_idx >= self.params.seg_max {
            return Err(TraceError::TooManySegments);
        }
        let is_cells = self.cells.clone();
        let ts_in = self.ts;
        Ok(SegmentRun {
            mem: self,
            is_cells,
            ts_in,
            ops: Vec::new(),
        })
    }

    fn apply(&mut self, seg: bool, addr: u64, write: Option<u32>) -> Result<MemOpRecord, TraceError> {
        let space = if seg { MemSpace::Ram } else { MemSpace::Rom };
        let g = self
            .params
            .global_index(space, addr)
            .map_err(|_| TraceError::AddrRange {
                addr,
                cells: if seg {
                    self.params.ram_cells()
                } else {
                    self.params.rom_cells()
                },
            })?;
        if !seg && write.is_some() {
            return Err(TraceError::RomWrite(addr));
        }
        let cell = self.cells[g as usize];
        let (v_r, rt) = (cell.v, cell.t);
        let wt = self.tick()?;
        debug_assert!(rt < wt, "previous access is always older than the new op");
        let v_w = write.unwrap_or(v_r);
        self.cells[g as usize] = CellRecord { v: v_w, t: wt };
        Ok(MemOpRecord {
            is_write: write.is_some(),
            space,
            addr,
            v_r,
            v_w,
            rt,
        })
    }

    /// Push `v` onto stack `s` (Coral Fig. 7 `check_push`): the new cell
    /// sits at `addr = sp`, its write time is the ticked global `ts`, and
    /// only a WS tuple exists — the RS-side fields are zero (E14).
    fn apply_push(&mut self, s: u8, v: u32) -> Result<MemOpRecord, TraceError> {
        let stack = self.stack(s)?;
        if stack.len() as u64 >= self.params.stack_cells() - 1 {
            return Err(TraceError::StackOverflow(s));
        }
        let addr = stack.len() as u64;
        let wt = self.tick()?;
        self.stacks[s as usize].push((v, wt));
        Ok(MemOpRecord {
            is_write: true,
            space: MemSpace::Stack(s),
            addr,
            v_r: 0,
            v_w: v,
            rt: 0,
        })
    }

    /// Pop stack `s` (Coral Fig. 7 `check_pop`): reads the top cell at
    /// `addr = sp − 1`, carrying its push time as `rt` — the only tuple
    /// is RS, and it cancels the push's WS tuple exactly.
    fn apply_pop(&mut self, s: u8) -> Result<MemOpRecord, TraceError> {
        let stack = self.stack(s)?;
        let Some(&(v, push_time)) = stack.last() else {
            return Err(TraceError::StackUnderflow(s));
        };
        let addr = stack.len() as u64 - 1;
        self.tick()?;
        self.stacks[s as usize].pop();
        Ok(MemOpRecord {
            is_write: false,
            space: MemSpace::Stack(s),
            addr,
            v_r: v,
            v_w: v,
            rt: push_time,
        })
    }

    fn stack(&self, s: u8) -> Result<&Vec<(u32, u64)>, TraceError> {
        self.stacks.get(s as usize).ok_or(TraceError::StackIndex {
            got: s,
            stacks: self.params.num_stacks,
        })
    }

    /// Advance the global timestamp by one op and return the new value.
    fn tick(&mut self) -> Result<u64, TraceError> {
        if self.ts + 1 >= 1 << TS_BITS {
            return Err(TraceError::TsExhausted);
        }
        self.ts += 1;
        Ok(self.ts)
    }
}

/// One open segment: applies ops against the shared [`Memory`] and records
/// everything the prover and the oracle need.
pub struct SegmentRun<'m> {
    mem: &'m mut Memory,
    is_cells: Vec<CellRecord>,
    ts_in: u64,
    ops: Vec<MemOpRecord>,
}

impl SegmentRun<'_> {
    /// Apply one explicit ROM/RAM access and return the exact tuple record.
    /// Application adapters use this to preserve fixed slot positions while
    /// sharing the same native memory semantics as ordinary segment traces.
    pub fn access(&mut self, space: MemSpace, addr: u64, write: Option<u32>) -> Result<MemOpRecord, TraceError> {
        self.check_capacity()?;
        let seg = match space {
            MemSpace::Rom => false,
            MemSpace::Ram => true,
            MemSpace::Stack(s) => {
                return Err(TraceError::StackIndex {
                    got: s,
                    stacks: self.mem.params.num_stacks,
                })
            }
        };
        let op = self.mem.apply(seg, addr, write)?;
        self.ops.push(op);
        Ok(op)
    }

    /// Read a cell; returns its current value.
    pub fn read(&mut self, seg: bool, addr: u64) -> Result<u32, TraceError> {
        let op = self.access(if seg { MemSpace::Ram } else { MemSpace::Rom }, addr, None)?;
        Ok(op.v_r)
    }

    /// Write a cell (RAM only).
    pub fn write(&mut self, seg: bool, addr: u64, v: u32) -> Result<(), TraceError> {
        self.access(if seg { MemSpace::Ram } else { MemSpace::Rom }, addr, Some(v))?;
        Ok(())
    }

    /// Push `v` onto stack `s` (v3.1).
    pub fn push(&mut self, s: u8, v: u32) -> Result<(), TraceError> {
        self.check_capacity()?;
        let op = self.mem.apply_push(s, v)?;
        self.ops.push(op);
        Ok(())
    }

    /// Pop stack `s` (v3.1); returns the popped value.
    pub fn pop(&mut self, s: u8) -> Result<u32, TraceError> {
        self.check_capacity()?;
        let op = self.mem.apply_pop(s)?;
        self.ops.push(op);
        Ok(op.v_r)
    }

    /// Close the segment: snapshot FS, advance the segment counter, and
    /// return the full trace. Rejects a segment that leaves a stack
    /// non-empty — stacks are segment-local, and an unpopped
    /// push could only fail later, at the product equation.
    pub fn finish(self) -> Result<SegmentTrace, TraceError> {
        for (s, stack) in self.mem.stacks.iter().enumerate() {
            if !stack.is_empty() {
                return Err(TraceError::StackNotEmpty {
                    stack: s as u8,
                    live: stack.len(),
                });
            }
        }
        let trace = SegmentTrace {
            params: self.mem.params,
            seg_idx: self.mem.seg_idx,
            ts_in: self.ts_in,
            ts_out: self.mem.ts,
            ops: self.ops,
            is_cells: self.is_cells,
            fs_cells: self.mem.cells.clone(),
        };
        self.mem.seg_idx += 1;
        Ok(trace)
    }

    fn check_capacity(&self) -> Result<(), TraceError> {
        if self.ops.len() >= self.mem.params.ops_per_segment() {
            return Err(TraceError::SegmentFull(self.ops.len()));
        }
        Ok(())
    }
}

/// Everything one segment contributes to the proof, as plain data: the op
/// records (in application order), and the IS/FS snapshots (in global cell
/// order, canonical by construction). Tests mutate copies of
/// these to model attacks; the tuple views below are derived, never stored
/// twice.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SegmentTrace {
    params: NebulaParams,
    /// Segment counter of this trace.
    pub seg_idx: u64,
    /// Global timestamp at segment open.
    pub ts_in: u64,
    /// Global timestamp at segment close (`ts_in + ops.len()`).
    pub ts_out: u64,
    /// Real ops in application order. Op `j` has write timestamp
    /// `ts_in + j + 1`; step `i` of the segment carries ops
    /// `[i · B_ops, (i+1) · B_ops)` of this list (sequential fill), padding
    /// the rest.
    pub ops: Vec<MemOpRecord>,
    /// Memory at segment open, indexed by global cell index.
    pub is_cells: Vec<CellRecord>,
    /// Memory at segment close, indexed by global cell index.
    pub fs_cells: Vec<CellRecord>,
}

impl SegmentTrace {
    /// The plan this trace was produced under.
    pub fn params(&self) -> &NebulaParams {
        &self.params
    }

    /// The real ops of step `i` (empty slice once ops are exhausted).
    pub fn step_ops(&self, i: usize) -> &[MemOpRecord] {
        let lo = (i * self.params.b_ops).min(self.ops.len());
        let hi = ((i + 1) * self.params.b_ops).min(self.ops.len());
        &self.ops[lo..hi]
    }

    /// RS multiset: one tuple per RS-emitting op (everything but pushes),
    /// carrying the cell's previous timestamp and the value
    /// read.
    pub fn rs_tuples(&self) -> Vec<MemTuple> {
        self.ops
            .iter()
            .filter(|op| !is_push(op))
            .map(|op| MemTuple {
                t: op.rt,
                g: self.g(op),
                v: op.v_r,
            })
            .collect()
    }

    /// WS multiset: one tuple per WS-emitting op (everything but pops),
    /// carrying the write timestamp (`ts_in + j + 1` for op
    /// `j` — pops still tick the clock) and the value written back.
    pub fn ws_tuples(&self) -> Vec<MemTuple> {
        self.ops
            .iter()
            .enumerate()
            .filter(|(_, op)| !is_pop(op))
            .map(|(j, op)| MemTuple {
                t: self.ts_in + j as u64 + 1,
                g: self.g(op),
                v: op.v_w,
            })
            .collect()
    }

    /// IS multiset: the open-snapshot cells at their global indices.
    pub fn is_tuples(&self) -> Vec<MemTuple> {
        Self::cell_tuples(&self.is_cells)
    }

    /// FS multiset: the close-snapshot cells at their global indices.
    pub fn fs_tuples(&self) -> Vec<MemTuple> {
        Self::cell_tuples(&self.fs_cells)
    }

    /// The four running products `[h_rs, h_ws, h_is, h_fs]` (order per
    /// [`H_RS`](crate::frontends::nebula::layout::H_RS)) for challenges γ.
    pub fn products(&self, gammas: &Gammas) -> [K; 4] {
        [
            fingerprint::product(gammas, &self.rs_tuples()),
            fingerprint::product(gammas, &self.ws_tuples()),
            fingerprint::product(gammas, &self.is_tuples()),
            fingerprint::product(gammas, &self.fs_tuples()),
        ]
    }

    /// The Nebula balance check on this segment's four multisets. Exact
    /// (probability 1) for honest traces; false w.h.p. for tampered ones.
    pub fn balanced(&self, gammas: &Gammas) -> bool {
        fingerprint::balanced(&self.products(gammas))
    }

    fn g(&self, op: &MemOpRecord) -> u64 {
        self.params
            .global_index(op.space, op.addr)
            .expect("trace ops were validated on application")
    }

    fn cell_tuples(cells: &[CellRecord]) -> Vec<MemTuple> {
        cells
            .iter()
            .enumerate()
            .map(|(g, c)| MemTuple {
                t: c.t,
                g: g as u64,
                v: c.v,
            })
            .collect()
    }
}

/// A stack push emits WS only.
fn is_push(op: &MemOpRecord) -> bool {
    matches!(op.space, MemSpace::Stack(_)) && op.is_write
}

/// A stack pop emits RS only.
fn is_pop(op: &MemOpRecord) -> bool {
    matches!(op.space, MemSpace::Stack(_)) && !op.is_write
}

//! Nebula native memory machine — the first pass of the two-pass prover
//! (spec §1) and the test oracle. Never verifier authority (spec scope).
//!
//! Owns: sequential-consistency semantics — cell state, the global
//! never-resetting timestamp, RS/WS tuple emission per op, IS/FS snapshots
//! per segment, and segment op-capacity/step-chunking bookkeeping.
//!
//! Does not own: bit encodings ([`super::layout`]), fingerprint math
//! ([`super::fingerprint`]), circuits, commitments, or any accept/reject
//! decision of the protocol.
//!
//! ## Semantics (spec §3.1/§3.2, Nebula §4.2)
//!
//! Every op — read or write — touches exactly one cell and emits one RS
//! tuple (the cell's previous `(t, v)`) and one WS tuple (the new ones):
//!
//! ```text
//! read(seg, a):      RS += (rt, g, v)    WS += (ts+1, g, v)     cell.t ← ts+1
//! write(seg, a, v'): RS += (rt, g, v)    WS += (ts+1, g, v')    cell ← (v', ts+1)
//! ```
//!
//! with `ts` incremented per op, globally across all segments. Writes to
//! the ROM namespace are rejected. For an honest run, `IS ∪ WS = RS ∪ FS`
//! holds exactly (Blum et al. invariant, Nebula Lemma 7) — tests assert
//! product balance without any soundness slack.

use neo_math::K;
use thiserror::Error;

use crate::frontends::nebula::fingerprint::{self, Gammas, MemTuple};
use crate::frontends::nebula::layout::{CellRecord, MemOpRecord, NebulaParams, TS_BITS};

#[derive(Debug, Error, PartialEq, Eq)]
pub enum TraceError {
    #[error("ROM image must have exactly {want} cells, got {got}")]
    RomImageLen { want: u64, got: usize },
    #[error("write to public ROM (addr {0})")]
    RomWrite(u64),
    #[error("address {addr} out of range for namespace of {cells} cells")]
    AddrRange { addr: u64, cells: u64 },
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
    /// Indexed by global cell index `g`; ROM occupies `[0, R)`.
    cells: Vec<CellRecord>,
    /// Global timestamp: total ops applied so far, across all segments.
    ts: u64,
    /// Next segment counter.
    seg_idx: u64,
}

impl Memory {
    /// Fresh chain-start memory (spec §3.1): ROM cells hold `rom_image`,
    /// RAM cells hold 0, every timestamp is 0.
    pub fn new(params: NebulaParams, rom_image: &[u32]) -> Result<Self, TraceError> {
        if rom_image.len() as u64 != params.rom_cells() {
            return Err(TraceError::RomImageLen {
                want: params.rom_cells(),
                got: rom_image.len(),
            });
        }
        let mut cells = Vec::with_capacity(params.total_cells() as usize);
        cells.extend(rom_image.iter().map(|&v| CellRecord { v, t: 0 }));
        cells.resize(params.total_cells() as usize, CellRecord { v: 0, t: 0 });
        Ok(Self {
            params,
            cells,
            ts: 0,
            seg_idx: 0,
        })
    }

    /// Global timestamp (ops applied so far). Never resets (spec §6.3).
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
        let g = self
            .params
            .global_index(seg, addr)
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
        if self.ts + 1 >= 1 << TS_BITS {
            return Err(TraceError::TsExhausted);
        }
        let cell = &mut self.cells[g as usize];
        let (v_r, rt) = (cell.v, cell.t);
        self.ts += 1;
        let wt = self.ts;
        debug_assert!(rt < wt, "previous access is always older than the new op");
        let v_w = write.unwrap_or(v_r);
        *cell = CellRecord { v: v_w, t: wt };
        Ok(MemOpRecord {
            is_write: write.is_some(),
            seg,
            addr,
            v_r,
            v_w,
            rt,
        })
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
    /// Read a cell; returns its current value.
    pub fn read(&mut self, seg: bool, addr: u64) -> Result<u32, TraceError> {
        self.check_capacity()?;
        let op = self.mem.apply(seg, addr, None)?;
        self.ops.push(op);
        Ok(op.v_r)
    }

    /// Write a cell (RAM only).
    pub fn write(&mut self, seg: bool, addr: u64, v: u32) -> Result<(), TraceError> {
        self.check_capacity()?;
        let op = self.mem.apply(seg, addr, Some(v))?;
        self.ops.push(op);
        Ok(())
    }

    /// Close the segment: snapshot FS, advance the segment counter, and
    /// return the full trace.
    pub fn finish(self) -> SegmentTrace {
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
        trace
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
/// order — canonical by construction, spec §3.3). Tests mutate copies of
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

    /// RS multiset: one tuple per op, carrying the cell's previous
    /// timestamp and the value read.
    pub fn rs_tuples(&self) -> Vec<MemTuple> {
        self.ops
            .iter()
            .map(|op| MemTuple {
                t: op.rt,
                g: self.g(op),
                v: op.v_r,
            })
            .collect()
    }

    /// WS multiset: one tuple per op, carrying the write timestamp
    /// (`ts_in + j + 1` for op `j`) and the value written back.
    pub fn ws_tuples(&self) -> Vec<MemTuple> {
        self.ops
            .iter()
            .enumerate()
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
            .global_index(op.seg, op.addr)
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

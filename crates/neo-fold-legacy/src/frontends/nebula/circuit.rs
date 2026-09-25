//! `S_mem` — the uniform Nebula step circuit in the mixed-gate
//! CCS idiom of `engine/ccs_native/poseidon2.rs`.
//!
//! Owns: the CCS structure of one step (rows E1–E9, S1–S3, boundary), the
//! step witness layout (`z` column map), and the witness builder. One
//! structure serves every step of every segment: challenges, counters, and
//! products enter through the public `x` bits, never as matrix constants.
//!
//! Does not own: encodings and parameters ([`super::layout`] — this file
//! consumes them), fingerprint semantics ([`super::fingerprint`] — the
//! witness builder calls it so circuit values and native values cannot
//! drift), folding, commitments, or the F′ carry.
//!
//! ## Witness vector layout
//!
//! ```text
//! z = [ 1 | x bits (X_BITS) | gap→d | ops lane | IS lane | FS lane | aux ]
//!       0   1 .. 1+X_BITS     zeros   aligned    aligned   aligned
//! ```
//!
//! Lane regions sit at multiples of `d = 54` and carry constrained-zero
//! tails (L-ALIGN). The aux region holds, per op slot: `diff`
//! (44 bits), `cnt` (running non-pad count), per stack an E11-pinned `sw`
//! bit and a σ-bit running `sp` word (v3.1), and the running `h_rs`/`h_ws`
//! words (128 bits each); per scan slot: the running `h_is`/`h_fs` words.
//! `m_in` is the unique whole-ring completion of `1 + x_bits`. The completion
//! columns are constrained to zero before the private lane columns begin.
//!
//! ## Gate families (one CCS polynomial, 15 matrices)
//!
//! ```text
//! f  =  (M₀z)² − M₀z                                   bitness
//!    +  (M₁z)∘(M₂z)                                    product-zero
//!    +  (M₃z) − (M₄z)                                  linear equality
//!    −  Oz + Az∘Pz + Az∘Qz∘Fₐz − Az∘Qz∘Gₐz∘Vz          K product update
//!    +  Bz∘Qz∘F_bz − Bz∘Qz∘G_bz∘Vz                     (slots O,A,B,P,Q,
//!                                                       Fa,Fb,Ga,Gb,V)
//! ```
//!
//! The K family evaluates one component of `h_next = h_prev · g` with
//! `g = pad + (1 − pad) · f_γ` inline (degree 4): component 0
//! assigns `A = h_prev,0`, `B = W · h_prev,1` (`W` = the extension's
//! binomial constant, read off `neo_math::K` at build time); component 1
//! swaps `A = h_prev,1`, `B = h_prev,0`. Scan rows reuse the family with
//! `P = 0, Q = 1` (no pad gate). Inactive families have all-zero rows.
//!
//! The witness builder is **total**: forged step data (ROM writes, stale
//! timestamps, out-of-range addresses) still produces a well-formed `z` —
//! whose constraint check then fails. Validity lives in the rows, not in
//! the builder; `layout`'s strict encoders remain the honest-path API.

use neo_ccs::{CcsMatrix, CcsStructure, CscMat, SparsePoly, Term};
use neo_math::field::KExtensions;
use neo_math::{D, F, K};
use p3_field::PrimeCharacteristicRing;
use thiserror::Error;

use crate::engine::r1cs_circuit::boolean::enforce_bit;
use crate::engine::r1cs_circuit::{Lc, R1csBuilder, Var};
use crate::frontends::nebula::fingerprint::{self, Gammas, MemTuple};
use crate::frontends::nebula::layout::{
    x_offsets, CellRecord, LayoutError, MemOpRecord, MemSpace, NebulaParams, StepPublicInput, H_FS, H_IS, H_RS, H_WS,
    K_BITS, MAX_STACKS, STEP_IDX_BITS, TS_BITS, VAL_BITS,
};
use crate::paper::relations::Structure;

// Matrix slots (indices into the 15 CCS matrices).
const M_BIT: usize = 0;
const M_PL: usize = 1;
const M_PR: usize = 2;
const M_LL: usize = 3;
const M_LR: usize = 4;
const M_O: usize = 5;
const M_A: usize = 6;
const M_B: usize = 7;
const M_P: usize = 8;
const M_Q: usize = 9;
const M_FA: usize = 10;
const M_FB: usize = 11;
const M_GA: usize = 12;
const M_GB: usize = 13;
const M_V: usize = 14;
const T_MATRICES: usize = 15;

/// The `S_mem` structure plus everything needed to build step witnesses
/// against it. Build once per plan; γ-independent by construction.
pub struct SMemCircuit {
    params: NebulaParams,
    structure: Structure,
    zmap: ZMap,
    nnz: usize,
}

/// One step's inputs, as the prover knows them after the segment's native
/// pass ([`super::trace`]). `is_cells`/`fs_cells` are this step's scan
/// chunk: cells `[idx · B_scan, (idx+1) · B_scan)` of the segment snapshots.
#[derive(Clone, Copy, Debug)]
pub struct StepData<'a> {
    pub seg_idx: u64,
    pub idx: u64,
    pub ts_in: u64,
    /// Running products entering this step (order per [`H_RS`]).
    pub h_in: [K; 4],
    /// Stack pointers entering this step (v3.1; zeros at segment start
    /// and for stack-less plans). Length is `MAX_STACKS`, written
    /// literally (rustc 1.94 const-in-array-length ICE workaround).
    pub sp_in: [u64; 2],
    /// Real ops of this step (`≤ B_ops`; the rest are pads).
    pub ops: &'a [MemOpRecord],
    pub is_cells: &'a [CellRecord],
    pub fs_cells: &'a [CellRecord],
}

#[derive(Debug, Error)]
pub enum SMemR1csError {
    #[error("S_mem R1CS lowering: assignment length {got} != circuit width {expected}")]
    AssignmentLength { got: usize, expected: usize },
    #[error("S_mem R1CS lowering: assignment constant column is not one")]
    ConstantOne,
}

impl SMemCircuit {
    pub fn new(params: NebulaParams) -> Self {
        let zmap = ZMap::new(&params);
        let mut rb = RowBuilder::new();
        emit_rows(&params, &zmap, &mut rb);
        audit_lane_residency(&params, &zmap, &rb);
        let nnz = rb.nnz();
        let structure = rb.finish(zmap.m);
        Self {
            params,
            structure,
            zmap,
            nnz,
        }
    }

    pub fn structure(&self) -> &Structure {
        &self.structure
    }

    pub fn params(&self) -> &NebulaParams {
        &self.params
    }

    /// Logical public payload length (the constant plus the `x` bits).
    pub fn logical_public_input_len(&self) -> usize {
        1 + self.params.x_bits()
    }

    /// Canonical whole-ring public prefix length of `z`.
    pub fn m_in(&self) -> usize {
        self.logical_public_input_len().div_ceil(D) * D
    }

    /// Constraint rows.
    pub fn rows(&self) -> usize {
        self.structure.n
    }

    /// Witness columns.
    pub fn cols(&self) -> usize {
        self.structure.m
    }

    /// Total nonzero matrix entries across all 15 matrices.
    pub fn nnz(&self) -> usize {
        self.nnz
    }

    /// First column of op slot `j` in `z` (its `pad` bit). Exposed for
    /// red-team tests that tamper with committed lane bits directly.
    pub fn op_slot_column(&self, j: usize) -> usize {
        self.zmap.op_slot(j)
    }

    /// Column of op slot `j`'s `sw_s` aux bit (v3.1). Exposed for the
    /// red-team test that a lying push bit is pinned by E11.
    pub fn op_sw_column(&self, j: usize, s: usize) -> usize {
        self.zmap.op_aux[j].sw[s]
    }

    /// The three lane regions as whole ring-column ranges of the packed
    /// witness — the geometry from which a `LaneScheme` is built.
    /// Single source of truth: the same `ZMap` that laid out the rows.
    pub fn lane_ranges(&self) -> crate::paper::relations::LaneRanges {
        use neo_math::D;
        crate::paper::relations::LaneRanges {
            ops: self.zmap.ops_lane / D..self.zmap.is_lane / D,
            is: self.zmap.is_lane / D..self.zmap.fs_lane / D,
            fs: self.zmap.fs_lane / D..(self.zmap.fs_lane / D) + (self.zmap.fs_lane - self.zmap.is_lane) / D,
        }
    }

    /// Build the full assignment for one step. Returns `z` (length
    /// [`Self::cols`]) and the [`StepPublicInput`] it encodes — with
    /// `ts_out`/`h_out` computed from the data, so callers chain steps by
    /// feeding one step's outputs into the next step's inputs.
    pub fn witness(&self, gammas: &Gammas, data: &StepData<'_>) -> Result<(Vec<F>, StepPublicInput), LayoutError> {
        self.witness_inner(gammas, data, None)
    }

    /// Build a step whose operation slots have verifier-fixed positions.
    /// `None` entries are canonical holes; later slots may still be active.
    pub fn witness_slots(
        &self,
        gammas: &Gammas,
        data: &StepData<'_>,
        op_slots: &[Option<MemOpRecord>],
    ) -> Result<(Vec<F>, StepPublicInput), LayoutError> {
        self.witness_inner(gammas, data, Some(op_slots))
    }

    fn witness_inner(
        &self,
        gammas: &Gammas,
        data: &StepData<'_>,
        op_slots: Option<&[Option<MemOpRecord>]>,
    ) -> Result<(Vec<F>, StepPublicInput), LayoutError> {
        let p = &self.params;
        let supplied = op_slots.map_or(data.ops.len(), <[Option<MemOpRecord>]>::len);
        if supplied > p.b_ops {
            return Err(LayoutError::TooManyOps {
                max: p.b_ops,
                got: supplied,
            });
        }
        if data.is_cells.len() != p.b_scan || data.fs_cells.len() != p.b_scan {
            return Err(LayoutError::ScanLen {
                want: p.b_scan,
                got: data.is_cells.len(),
            });
        }
        let slots = op_slot_values(p, gammas, data, op_slots);
        let scans = scan_slot_values(p, gammas, data);
        let last = slots.last().expect("b_ops >= 1");
        let x = StepPublicInput {
            seg_idx: data.seg_idx,
            idx: data.idx,
            ts_in: data.ts_in,
            ts_out: data.ts_in + last.cnt,
            gamma: [gammas.gamma1, gammas.gamma2],
            h_in: data.h_in,
            h_out: [
                last.h_rs,
                last.h_ws,
                scans.last().expect("b_scan >= 1").h_is,
                scans.last().expect("b_scan >= 1").h_fs,
            ],
            sp_in: data.sp_in,
            sp_out: last.sp,
        };

        let addr_off = 3 + p.num_stacks;
        let mut z = vec![F::ZERO; self.zmap.m];
        z[0] = F::ONE;
        for (i, bit) in x.encode(p.stack_shape())?.into_iter().enumerate() {
            z[1 + i] = bit;
        }
        for (j, slot) in slots.iter().enumerate() {
            let off = self.zmap.op_slot(j);
            write_bits(&mut z, off, slot.pad as u64, 1);
            write_bits(&mut z, off + 1, slot.is_write as u64, 1);
            write_bits(&mut z, off + 2, slot.ram as u64, 1);
            for s in 0..p.num_stacks {
                write_bits(&mut z, off + 3 + s, (slot.stk == Some(s)) as u64, 1);
            }
            write_bits(&mut z, off + addr_off, slot.addr, p.addr_bits());
            write_bits(&mut z, off + addr_off + p.addr_bits(), slot.v_r as u64, VAL_BITS);
            write_bits(
                &mut z,
                off + addr_off + p.addr_bits() + VAL_BITS,
                slot.v_w as u64,
                VAL_BITS,
            );
            write_bits(&mut z, off + addr_off + p.addr_bits() + 2 * VAL_BITS, slot.rt, TS_BITS);
            let aux = &self.zmap.op_aux[j];
            write_bits(&mut z, aux.diff, slot.diff, TS_BITS);
            write_bits(&mut z, aux.cnt, slot.cnt, self.zmap.cnt_bits);
            for s in 0..p.num_stacks {
                write_bits(&mut z, aux.sw[s], slot.sw[s] as u64, 1);
                write_bits(&mut z, aux.sp[s], slot.sp[s], p.sigma as usize);
            }
            write_k(&mut z, aux.h_rs, slot.h_rs);
            write_k(&mut z, aux.h_ws, slot.h_ws);
        }
        for (j, scan) in scans.iter().enumerate() {
            let is_off = self.zmap.is_lane + j * (VAL_BITS + TS_BITS);
            let fs_off = self.zmap.fs_lane + j * (VAL_BITS + TS_BITS);
            write_bits(&mut z, is_off, scan.is_v as u64, VAL_BITS);
            write_bits(&mut z, is_off + VAL_BITS, scan.is_t, TS_BITS);
            write_bits(&mut z, fs_off, scan.fs_v as u64, VAL_BITS);
            write_bits(&mut z, fs_off + VAL_BITS, scan.fs_t, TS_BITS);
            let aux = &self.zmap.scan_aux[j];
            write_k(&mut z, aux.h_is, scan.h_is);
            write_k(&mut z, aux.h_fs, scan.h_fs);
        }
        Ok((z, x))
    }

    /// Emit the exact `S_mem` relation as field-native R1CS rows.
    ///
    /// This is the application-relation arm consumed by authoritative F'.
    /// It reads the same 15 matrices as the native CCS relation and factors
    /// only the degree-four K-product family; no second memory semantics
    /// implementation exists at the call site.
    pub fn enforce_in_r1cs(&self, builder: &mut R1csBuilder, assignment: &[F]) -> Result<Vec<Var>, SMemR1csError> {
        let vars = self.allocate_r1cs_assignment(builder, assignment)?;
        self.enforce_allocated_r1cs(builder, &vars)?;
        Ok(vars)
    }

    pub(crate) fn allocate_r1cs_assignment(
        &self,
        builder: &mut R1csBuilder,
        assignment: &[F],
    ) -> Result<Vec<Var>, SMemR1csError> {
        if assignment.len() != self.structure.m {
            return Err(SMemR1csError::AssignmentLength {
                got: assignment.len(),
                expected: self.structure.m,
            });
        }
        if assignment.first().copied() != Some(F::ONE) {
            return Err(SMemR1csError::ConstantOne);
        }

        let mut vars = Vec::with_capacity(assignment.len());
        vars.push(Var::ONE);
        vars.extend(builder.alloc_vec(&assignment[1..]));
        // `S_mem` is natively a low-norm bit assignment, including layout
        // fillers. State that boundary once here so the generic F' lowering
        // keeps every source coordinate one bit wide and preserves the
        // ring-column alignment used by the lane commitments.
        for &var in vars.iter().skip(1) {
            enforce_bit(builder, var);
        }
        Ok(vars)
    }

    pub(crate) fn enforce_allocated_r1cs(&self, builder: &mut R1csBuilder, vars: &[Var]) -> Result<(), SMemR1csError> {
        if vars.len() != self.structure.m {
            return Err(SMemR1csError::AssignmentLength {
                got: vars.len(),
                expected: self.structure.m,
            });
        }
        let matrix_rows: Vec<Vec<Lc>> = self
            .structure
            .matrices
            .iter()
            .map(|matrix| matrix_row_lcs(matrix, &vars, self.structure.n))
            .collect();

        for row in 0..self.structure.n {
            let at = |matrix: usize| &matrix_rows[matrix][row];
            let bit_active = !lc_is_zero(at(M_BIT));
            let product_active = (M_PL..=M_LR).any(|matrix| !lc_is_zero(at(matrix)));
            let k_active = (M_O..=M_V).any(|matrix| !lc_is_zero(at(matrix)));
            assert!(
                !(bit_active && (product_active || k_active)) && !(product_active && k_active),
                "S_mem row families must stay disjoint when lowered to R1CS"
            );

            if bit_active {
                // Covered by the one global source-assignment pass above.
            } else if product_active {
                let rhs = at(M_LR).clone().add_scaled(at(M_LL), -F::ONE);
                builder.enforce(at(M_PL), at(M_PR), &rhs);
            } else if k_active {
                // -O + A(P + Q(FA - GA*V)) + B*Q(FB - GB*V) = 0.
                let ga_v = r1cs_mul_lc(builder, at(M_GA), at(M_V));
                let gb_v = r1cs_mul_lc(builder, at(M_GB), at(M_V));
                let fa_minus_ga_v = at(M_FA).clone().add_scaled(&ga_v, -F::ONE);
                let fb_minus_gb_v = at(M_FB).clone().add_scaled(&gb_v, -F::ONE);
                let q_a = r1cs_mul_lc(builder, at(M_Q), &fa_minus_ga_v);
                let q_b = r1cs_mul_lc(builder, at(M_Q), &fb_minus_gb_v);
                let p_plus_q_a = at(M_P).clone().add_scaled(&q_a, F::ONE);
                let a_term = r1cs_mul_lc(builder, at(M_A), &p_plus_q_a);
                let b_term = r1cs_mul_lc(builder, at(M_B), &q_b);
                let sum = a_term.add_scaled(&b_term, F::ONE);
                builder.enforce_eq(&sum, at(M_O));
            }
        }
        Ok(())
    }
}

fn matrix_row_lcs(matrix: &CcsMatrix<F>, vars: &[Var], rows: usize) -> Vec<Lc> {
    let mut out = vec![Lc::zero(); rows];
    match matrix {
        CcsMatrix::Identity { n } => {
            for row in 0..(*n).min(rows).min(vars.len()) {
                out[row].add_term(vars[row], F::ONE);
            }
        }
        CcsMatrix::Csc(csc) => {
            assert_eq!(csc.ncols, vars.len(), "S_mem matrix width must match witness width");
            for col in 0..csc.ncols {
                for index in csc.column_range(col) {
                    let row = csc.row_index(index);
                    if row < rows {
                        out[row].add_term(vars[col], csc.vals[index]);
                    }
                }
            }
        }
        CcsMatrix::CscWithSeededPhi81 {
            csc,
            blocks,
            geometric_runs,
        } => {
            for col in 0..csc.ncols.min(vars.len()) {
                for index in csc.column_range(col) {
                    let row = csc.row_index(index);
                    if row < rows {
                        out[row].add_term(vars[col], csc.vals[index]);
                    }
                }
            }
            for block in blocks {
                block.for_each_term::<F, _>(|row, col, coefficient| {
                    if row < rows && col < vars.len() {
                        out[row].add_term(vars[col], coefficient);
                    }
                });
            }
            for run in geometric_runs {
                run.for_each_term(|row, col, coefficient| {
                    if row < rows && col < vars.len() {
                        out[row].add_term(vars[col], coefficient);
                    }
                });
            }
        }
        CcsMatrix::VerifierArtifact { .. } => {
            panic!("S_mem circuit construction requires materialized matrix content")
        }
    }
    out
}

fn lc_is_zero(value: &Lc) -> bool {
    value.terms.is_empty() && value.constant == F::ZERO
}

fn r1cs_mul_lc(builder: &mut R1csBuilder, left: &Lc, right: &Lc) -> Lc {
    if lc_is_zero(left) || lc_is_zero(right) {
        Lc::zero()
    } else {
        Lc::from_var(builder.alloc_mul(left, right))
    }
}

// ── Per-slot value computation (shared by witness and nothing else —
//    the semantics live in `fingerprint`, called here so values cannot
//    drift from what the K rows verify) ─────────────────────────────────

struct OpSlotValues {
    pad: bool,
    is_write: bool,
    ram: bool,
    /// Which stack this op's selector bits name (`None` for ROM/RAM/pad —
    /// and for out-of-plan stack indices, which encode as no selector and
    /// are judged by the rows as ROM ops).
    stk: Option<usize>,
    addr: u64,
    v_r: u32,
    v_w: u32,
    rt: u64,
    cnt: u64,
    diff: u64,
    /// `sw_s = stk_s · is_write` (the E11 aux bits).
    sw: [bool; MAX_STACKS],
    /// Running stack pointers after this slot (the E12 aux words).
    sp: [u64; MAX_STACKS],
    h_rs: K,
    h_ws: K,
}

struct ScanSlotValues {
    is_v: u32,
    is_t: u64,
    fs_v: u32,
    fs_t: u64,
    h_is: K,
    h_fs: K,
}

fn op_slot_values(
    p: &NebulaParams,
    gammas: &Gammas,
    data: &StepData<'_>,
    fixed_slots: Option<&[Option<MemOpRecord>]>,
) -> Vec<OpSlotValues> {
    let ts_mask = (1u64 << TS_BITS) - 1;
    let addr_mask = (1u64 << p.addr_bits()) - 1;
    let sp_mask = if p.sigma == 0 { 0 } else { (1u64 << p.sigma) - 1 };
    let mut cnt = 0u64;
    let mut sp = data.sp_in;
    let mut h_rs = data.h_in[H_RS];
    let mut h_ws = data.h_in[H_WS];
    let mut out = Vec::with_capacity(p.b_ops);
    for j in 0..p.b_ops {
        let op = match fixed_slots {
            Some(slots) => slots.get(j).copied().flatten(),
            None => data.ops.get(j).copied(),
        };
        let slot = match op {
            Some(op) => {
                cnt += 1;
                let ram = op.space == MemSpace::Ram;
                // Out-of-plan stack indices have no selector bit: they
                // encode as ROM ops and the rows judge them as such
                // (total builder — validity lives in the rows).
                let stk = match op.space {
                    MemSpace::Stack(s) if (s as usize) < p.num_stacks => Some(s as usize),
                    _ => None,
                };
                let addr = op.addr & addr_mask;
                let rt = op.rt & ts_mask;
                let wt = data.ts_in + cnt;
                // Mirrors the row-side g_lin: base from the one-hot
                // selectors, exactly as the lane bits will be read.
                let g = addr
                    + if ram { p.rom_cells() } else { 0 }
                    + stk.map_or(0, |s| p.scanned_cells() + s as u64 * p.stack_cells());
                let mut sw = [false; MAX_STACKS];
                if let Some(s) = stk {
                    sw[s] = op.is_write;
                    // E12 arithmetic, masked like `diff`: honest traces
                    // never wrap; forged pops/pushes wrap and the σ-bit
                    // word fails the (unmaskable) linear row.
                    sp[s] = if op.is_write {
                        sp[s].wrapping_add(1) & sp_mask
                    } else {
                        sp[s].wrapping_sub(1) & sp_mask
                    };
                }
                let rs_live = stk.is_none() || !op.is_write;
                let ws_live = stk.is_none() || op.is_write;
                if rs_live {
                    h_rs *= fingerprint::fingerprint(gammas, &MemTuple { t: rt, g, v: op.v_r });
                }
                if ws_live {
                    h_ws *= fingerprint::fingerprint(gammas, &MemTuple { t: wt, g, v: op.v_w });
                }
                OpSlotValues {
                    pad: false,
                    is_write: op.is_write,
                    ram,
                    stk,
                    addr,
                    v_r: op.v_r,
                    v_w: op.v_w,
                    rt,
                    cnt,
                    // Honest ops have rt < wt so this is exact; forged ones
                    // wrap and the (gated) E4 row rejects the masked value.
                    // Pushes are E4-exempt; their diff stays 0.
                    diff: if rs_live {
                        wt.wrapping_sub(rt).wrapping_sub(1) & ts_mask
                    } else {
                        0
                    },
                    sw,
                    sp,
                    h_rs,
                    h_ws,
                }
            }
            None => OpSlotValues {
                pad: true,
                is_write: false,
                ram: false,
                stk: None,
                addr: 0,
                v_r: 0,
                v_w: 0,
                rt: 0,
                cnt,
                diff: 0,
                sw: [false; MAX_STACKS],
                sp,
                h_rs,
                h_ws,
            },
        };
        out.push(slot);
    }
    out
}

fn scan_slot_values(p: &NebulaParams, gammas: &Gammas, data: &StepData<'_>) -> Vec<ScanSlotValues> {
    let ts_mask = (1u64 << TS_BITS) - 1;
    let mut h_is = data.h_in[H_IS];
    let mut h_fs = data.h_in[H_FS];
    let mut out = Vec::with_capacity(p.b_scan);
    for j in 0..p.b_scan {
        let g = data.idx * p.b_scan as u64 + j as u64;
        let (is, fs) = (&data.is_cells[j], &data.fs_cells[j]);
        let (is_t, fs_t) = (is.t & ts_mask, fs.t & ts_mask);
        h_is *= fingerprint::fingerprint(gammas, &MemTuple { t: is_t, g, v: is.v });
        h_fs *= fingerprint::fingerprint(gammas, &MemTuple { t: fs_t, g, v: fs.v });
        out.push(ScanSlotValues {
            is_v: is.v,
            is_t,
            fs_v: fs.v,
            fs_t,
            h_is,
            h_fs,
        });
    }
    out
}

fn write_bits(z: &mut [F], offset: usize, value: u64, nbits: usize) {
    for k in 0..nbits {
        z[offset + k] = if (value >> k) & 1 == 1 { F::ONE } else { F::ZERO };
    }
}

fn write_k(z: &mut [F], offset: usize, value: K) {
    let (c0, c1) = value.to_limbs_u64();
    write_bits(z, offset, c0, 64);
    write_bits(z, offset + 64, c1, 64);
}

// ── Column map ────────────────────────────────────────────────────────────

struct OpAux {
    diff: usize,
    cnt: usize,
    /// One E11-pinned `sw_s` bit per stack (empty for stack-less plans).
    sw: Vec<usize>,
    /// One σ-bit running `sp_s` word per stack.
    sp: Vec<usize>,
    h_rs: usize,
    h_ws: usize,
}

struct ScanAux {
    h_is: usize,
    h_fs: usize,
}

struct ZMap {
    ops_lane: usize,
    is_lane: usize,
    fs_lane: usize,
    op_aux: Vec<OpAux>,
    scan_aux: Vec<ScanAux>,
    /// Columns that rows force to zero: the x→lane gap and lane tails.
    fillers: Vec<usize>,
    cnt_bits: usize,
    op_bits: usize,
    m: usize,
}

impl ZMap {
    fn new(p: &NebulaParams) -> Self {
        let cnt_bits = 64 - (p.b_ops as u64).leading_zeros() as usize;
        let x_end = 1 + p.x_bits();
        let ops_lane = x_end.div_ceil(D) * D;
        let is_lane = ops_lane + p.ops_lane_bits();
        let fs_lane = is_lane + p.scan_lane_bits();
        let aux = fs_lane + p.scan_lane_bits();
        // L-ALIGN: lane regions must start at whole ring
        // columns; a violation breaks the Lemma-1 commutation argument.
        assert!(
            ops_lane % D == 0 && is_lane % D == 0 && fs_lane % D == 0,
            "L-ALIGN violated: lane offsets must be multiples of d"
        );

        let mut fillers: Vec<usize> = (x_end..ops_lane).collect();
        fillers.extend(ops_lane + p.b_ops * p.op_bits()..is_lane);
        fillers.extend(is_lane + p.b_scan * (VAL_BITS + TS_BITS)..fs_lane);
        fillers.extend(fs_lane + p.b_scan * (VAL_BITS + TS_BITS)..aux);

        let mut cursor = aux;
        let mut take = |n: usize| {
            let at = cursor;
            cursor += n;
            at
        };
        let op_aux = (0..p.b_ops)
            .map(|_| OpAux {
                diff: take(TS_BITS),
                cnt: take(cnt_bits),
                sw: (0..p.num_stacks).map(|_| take(1)).collect(),
                sp: (0..p.num_stacks).map(|_| take(p.sigma as usize)).collect(),
                h_rs: take(K_BITS),
                h_ws: take(K_BITS),
            })
            .collect();
        let scan_aux = (0..p.b_scan)
            .map(|_| ScanAux {
                h_is: take(K_BITS),
                h_fs: take(K_BITS),
            })
            .collect();
        Self {
            ops_lane,
            is_lane,
            fs_lane,
            op_aux,
            scan_aux,
            fillers,
            cnt_bits,
            op_bits: p.op_bits(),
            m: cursor,
        }
    }

    fn op_slot(&self, j: usize) -> usize {
        self.ops_lane + j * self.op_bits
    }

    /// Bit column of a named x field (`x_offsets` are relative to the x
    /// region, which starts at column 1).
    fn x(&self, offset: usize) -> usize {
        1 + offset
    }
}

// ── Linear forms and the row builder ──────────────────────────────────────

/// A linear form over `z`: a list of `(column, coefficient)` pairs.
/// Column 0 is the constant-one slot. Callers keep columns distinct.
#[derive(Clone, Default)]
struct Lin(Vec<(usize, F)>);

impl Lin {
    fn con(c: F) -> Self {
        Self(vec![(0, c)])
    }

    fn bit(col: usize) -> Self {
        Self(vec![(col, F::ONE)])
    }

    fn word(start: usize, nbits: usize) -> Self {
        Self::word_scaled(start, nbits, F::ONE)
    }

    fn word_scaled(start: usize, nbits: usize, scale: F) -> Self {
        let mut pow = scale;
        let mut terms = Vec::with_capacity(nbits);
        for k in 0..nbits {
            terms.push((start + k, pow));
            pow *= F::from_u64(2);
        }
        Self(terms)
    }

    fn plus(mut self, rhs: Self) -> Self {
        self.0.extend(rhs.0);
        self
    }

    fn minus(self, rhs: Self) -> Self {
        self.plus(rhs.scaled(-F::ONE))
    }

    fn scaled(mut self, s: F) -> Self {
        for term in &mut self.0 {
            term.1 *= s;
        }
        self
    }
}

struct RowBuilder {
    trips: Vec<Vec<(usize, usize, F)>>,
    rows: usize,
}

impl RowBuilder {
    fn new() -> Self {
        Self {
            trips: (0..T_MATRICES).map(|_| Vec::new()).collect(),
            rows: 0,
        }
    }

    fn put(&mut self, matrix: usize, row: usize, lin: &Lin) {
        for &(col, coeff) in &lin.0 {
            self.trips[matrix].push((row, col, coeff));
        }
    }

    fn next_row(&mut self) -> usize {
        let row = self.rows;
        self.rows += 1;
        row
    }

    /// `b² − b = 0` for one column.
    fn bit_row(&mut self, col: usize) {
        let row = self.next_row();
        self.trips[M_BIT].push((row, col, F::ONE));
    }

    fn bit_rows(&mut self, start: usize, len: usize) {
        for col in start..start + len {
            self.bit_row(col);
        }
    }

    /// `L · R = 0`.
    fn product_row(&mut self, l: &Lin, r: &Lin) {
        let row = self.next_row();
        self.put(M_PL, row, l);
        self.put(M_PR, row, r);
    }

    /// `lhs − rhs = 0`.
    fn linear_row(&mut self, lhs: &Lin, rhs: &Lin) {
        let row = self.next_row();
        self.put(M_LL, row, lhs);
        self.put(M_LR, row, rhs);
    }

    /// `L · R − out = 0` — the product and linear families sharing one
    /// row (E11: pinning an aux bit to a product of lane bits).
    fn product_eq_row(&mut self, l: &Lin, r: &Lin, out: &Lin) {
        let row = self.next_row();
        self.put(M_PL, row, l);
        self.put(M_PR, row, r);
        self.put(M_LR, row, out);
    }

    /// Both components of `h_next = h_prev · (P + Q·(f₀ + u·f₁))` with
    /// `f₀ = fa − ga·v`, `f₁ = fb − gb·v` (module header derivation).
    #[allow(clippy::too_many_arguments)]
    fn k_rows(
        &mut self,
        w_const: F,
        o: [&Lin; 2],
        h_prev: [&Lin; 2],
        p: &Lin,
        q: &Lin,
        f: [&Lin; 2],
        g: [&Lin; 2],
        v: &Lin,
    ) {
        for comp in 0..2 {
            let row = self.next_row();
            let (a, b) = if comp == 0 {
                (h_prev[0].clone(), h_prev[1].clone().scaled(w_const))
            } else {
                (h_prev[1].clone(), h_prev[0].clone())
            };
            self.put(M_O, row, o[comp]);
            self.put(M_A, row, &a);
            self.put(M_B, row, &b);
            self.put(M_P, row, p);
            self.put(M_Q, row, q);
            self.put(M_FA, row, f[0]);
            self.put(M_FB, row, f[1]);
            self.put(M_GA, row, g[0]);
            self.put(M_GB, row, g[1]);
            self.put(M_V, row, v);
        }
    }

    fn nnz(&self) -> usize {
        self.trips.iter().map(Vec::len).sum()
    }

    fn finish(self, m: usize) -> Structure {
        let f = SparsePoly::new(
            T_MATRICES,
            vec![
                term(F::ONE, &[(M_BIT, 2)]),
                term(-F::ONE, &[(M_BIT, 1)]),
                term(F::ONE, &[(M_PL, 1), (M_PR, 1)]),
                term(F::ONE, &[(M_LL, 1)]),
                term(-F::ONE, &[(M_LR, 1)]),
                term(-F::ONE, &[(M_O, 1)]),
                term(F::ONE, &[(M_A, 1), (M_P, 1)]),
                term(F::ONE, &[(M_A, 1), (M_Q, 1), (M_FA, 1)]),
                term(-F::ONE, &[(M_A, 1), (M_Q, 1), (M_GA, 1), (M_V, 1)]),
                term(F::ONE, &[(M_B, 1), (M_Q, 1), (M_FB, 1)]),
                term(-F::ONE, &[(M_B, 1), (M_Q, 1), (M_GB, 1), (M_V, 1)]),
            ],
        );
        let n = self.rows;
        let matrices = self
            .trips
            .into_iter()
            .map(|trips| CcsMatrix::Csc(CscMat::from_triplets(trips, n, m)))
            .collect();
        CcsStructure::new_sparse(matrices, f).expect("S_mem CCS structure")
    }
}

fn term(coeff: F, exps: &[(usize, u32)]) -> Term<F> {
    let mut e = vec![0u32; T_MATRICES];
    for &(m, x) in exps {
        e[m] = x;
    }
    Term { coeff, exps: e }
}

// ── Row emission (structure side; must mirror the witness side above) ────

fn emit_rows(p: &NebulaParams, zm: &ZMap, rb: &mut RowBuilder) {
    let w_const = binomial_w();
    let one = Lin::con(F::ONE);
    let ts_in = Lin::word(zm.x(x_offsets::TS_IN), TS_BITS);
    let gamma_word = |k: usize, comp: usize| Lin::word(zm.x(x_offsets::GAMMA + k * K_BITS + comp * 64), 64);
    let g1 = [gamma_word(0, 0), gamma_word(0, 1)];
    let g2 = [gamma_word(1, 0), gamma_word(1, 1)];
    let two_pow_ts = F::from_u64(1 << TS_BITS);

    // L-ALIGN gap and lane tails are pinned to zero.
    for &col in &zm.fillers {
        rb.linear_row(&Lin::bit(col), &Lin::default());
    }

    // Op slots: E1–E14. With S = 0 every stack form below is
    // empty and the emitted rows are exactly v3's.
    let sigma = p.sigma as usize;
    let addr_off = 3 + p.num_stacks;
    for j in 0..p.b_ops {
        let off = zm.op_slot(j);
        let aux = &zm.op_aux[j];
        let pad = Lin::bit(off);
        let is_write = Lin::bit(off + 1);
        let ram = Lin::bit(off + 2);
        let stks: Vec<Lin> = (0..p.num_stacks).map(|s| Lin::bit(off + 3 + s)).collect();
        let sws: Vec<Lin> = (0..p.num_stacks).map(|s| Lin::bit(aux.sw[s])).collect();
        let addr = Lin::word(off + addr_off, p.addr_bits());
        let v_r = Lin::word(off + addr_off + p.addr_bits(), VAL_BITS);
        let v_w = Lin::word(off + addr_off + p.addr_bits() + VAL_BITS, VAL_BITS);
        let rt = Lin::word(off + addr_off + p.addr_bits() + 2 * VAL_BITS, TS_BITS);
        let diff = Lin::word(aux.diff, TS_BITS);
        let cnt = Lin::word(aux.cnt, zm.cnt_bits);
        let not_pad = one.clone().minus(pad.clone());
        // ROM = no selector set; a bit given bitness + E10 exclusivity.
        let rom = stks
            .iter()
            .fold(one.clone().minus(ram.clone()), |acc, stk| acc.minus(stk.clone()));
        // skip_rs = pad + Σ sw (pushes emit no RS tuple);
        // skip_ws = pad + Σ (stk − sw) (pops emit no WS tuple).
        let skip_rs = sws.iter().fold(pad.clone(), |acc, sw| acc.plus(sw.clone()));
        let skip_ws = stks
            .iter()
            .zip(&sws)
            .fold(pad.clone(), |acc, (stk, sw)| acc.plus(stk.clone()).minus(sw.clone()));
        let sw_sum = sws
            .iter()
            .fold(Lin::default(), |acc, sw| acc.plus(sw.clone()));

        // E1: bitness of the lane slot and its aux bits. `sw` needs no
        // bitness row — E11 pins it to a product of two constrained bits.
        rb.bit_rows(off, zm.op_bits);
        rb.bit_rows(aux.diff, TS_BITS);
        rb.bit_rows(aux.cnt, zm.cnt_bits);
        for s in 0..p.num_stacks {
            rb.bit_rows(aux.sp[s], sigma);
        }
        rb.bit_rows(aux.h_rs, K_BITS);
        rb.bit_rows(aux.h_ws, K_BITS);

        // E2: cnt_j = cnt_{j−1} + (1 − pad_j).
        let prev_cnt = if j == 0 {
            Lin::default()
        } else {
            Lin::word(zm.op_aux[j - 1].cnt, zm.cnt_bits)
        };
        rb.linear_row(&cnt, &prev_cnt.plus(not_pad.clone()));

        // E3: reads write back what they read (pops included).
        rb.product_row(&one.clone().minus(is_write.clone()), &v_w.clone().minus(v_r.clone()));

        // E4: rt < wt on every RS-emitting op (RAM/ROM ops and pops —
        // Coral's `push_time < ts`; pushes are exempt:
        // (1−skip_rs)·(wt − rt − 1 − diff) = 0, wt = ts_in + cnt_j.
        let wt = ts_in.clone().plus(cnt.clone());
        rb.product_row(
            &one.clone().minus(skip_rs.clone()),
            &wt.clone().minus(rt.clone()).minus(one.clone()).minus(diff),
        );

        // E5: no writes to ROM.
        rb.product_row(&is_write, &rom);

        // E6: ROM addresses stay below R.
        for k in p.r as usize..p.addr_bits() {
            rb.product_row(&rom, &Lin::bit(off + addr_off + k));
        }

        // E7: pad slots are all-zero.
        for field in std::iter::once(&is_write)
            .chain(std::iter::once(&ram))
            .chain(&stks)
            .chain([&addr, &v_r, &v_w, &rt])
        {
            rb.product_row(&pad, field);
        }

        // E10: namespace selectors are one-hot (pairwise exclusive).
        for s in 0..p.num_stacks {
            rb.product_row(&ram, &stks[s]);
            for t in s + 1..p.num_stacks {
                rb.product_row(&stks[s], &stks[t]);
            }
        }

        // E11–E13, per stack: pin the push bit, run the pointer, and bind
        // the address to it — push writes at sp, pop reads at sp − 1
        // (Coral Fig. 7). The σ-bit `sp` word is the bounds check: a pop
        // at empty (−1) or push at full (2^σ) is unrepresentable.
        for s in 0..p.num_stacks {
            let sp = Lin::word(aux.sp[s], sigma);
            let sp_prev = if j == 0 {
                Lin::word(zm.x(x_offsets::SP + s * 2 * sigma), sigma)
            } else {
                Lin::word(zm.op_aux[j - 1].sp[s], sigma)
            };
            rb.product_eq_row(&stks[s], &is_write, &sws[s]);
            rb.linear_row(
                &sp,
                &sp_prev
                    .clone()
                    .plus(sws[s].clone().scaled(F::TWO))
                    .minus(stks[s].clone()),
            );
            rb.product_row(
                &stks[s],
                &addr
                    .clone()
                    .minus(sp_prev)
                    .plus(one.clone())
                    .minus(is_write.clone()),
            );
        }

        // E14: a push's dead RS-side fields are pinned to zero.
        if p.num_stacks > 0 {
            rb.product_row(&sw_sum, &rt);
            rb.product_row(&sw_sum, &v_r);
        }

        // E8/E9: RS and WS product updates. The fingerprint's packed prefix
        // is affine in lane bits: packed = t + 2^TS_BITS · g with
        // g = addr + ram·R + Σ stk_s·(R + M + s·2^σ) (one-hot selectors
        // keep it linear).
        let mut g_lin = addr
            .clone()
            .plus(Lin(vec![(off + 2, F::from_u64(p.rom_cells()))]));
        for s in 0..p.num_stacks {
            g_lin = g_lin.plus(Lin(vec![(
                off + 3 + s,
                F::from_u64(p.scanned_cells() + s as u64 * p.stack_cells()),
            )]));
        }
        let h_prev = |aux_off: Option<usize>, x_slot: usize| -> [Lin; 2] {
            match aux_off {
                Some(o) => [Lin::word(o, 64), Lin::word(o + 64, 64)],
                None => {
                    let base = zm.x(x_offsets::H_IN + x_slot * K_BITS);
                    [Lin::word(base, 64), Lin::word(base + 64, 64)]
                }
            }
        };
        let prev_rs = h_prev(if j == 0 { None } else { Some(zm.op_aux[j - 1].h_rs) }, H_RS);
        let prev_ws = h_prev(if j == 0 { None } else { Some(zm.op_aux[j - 1].h_ws) }, H_WS);
        let fa_rs = g2[0]
            .clone()
            .minus(rt.clone())
            .minus(g_lin.clone().scaled(two_pow_ts));
        let fa_ws = g2[0]
            .clone()
            .minus(wt.clone())
            .minus(g_lin.clone().scaled(two_pow_ts));
        rb.k_rows(
            w_const,
            [&Lin::word(aux.h_rs, 64), &Lin::word(aux.h_rs + 64, 64)],
            [&prev_rs[0], &prev_rs[1]],
            &skip_rs,
            &one.clone().minus(skip_rs.clone()),
            [&fa_rs, &g2[1]],
            [&g1[0], &g1[1]],
            &v_r,
        );
        rb.k_rows(
            w_const,
            [&Lin::word(aux.h_ws, 64), &Lin::word(aux.h_ws + 64, 64)],
            [&prev_ws[0], &prev_ws[1]],
            &skip_ws,
            &one.clone().minus(skip_ws.clone()),
            [&fa_ws, &g2[1]],
            [&g1[0], &g1[1]],
            &v_w,
        );
    }

    // Scan slots: S1–S3. No pads (exact cover); the packed
    // prefix uses the structural position g_p = idx·B_scan + j.
    let idx_word = Lin::word(zm.x(x_offsets::IDX), STEP_IDX_BITS);
    for j in 0..p.b_scan {
        for (lane_off, aux_off, x_slot) in [
            (zm.is_lane, zm.scan_aux[j].h_is, H_IS),
            (zm.fs_lane, zm.scan_aux[j].h_fs, H_FS),
        ] {
            let cell = lane_off + j * (VAL_BITS + TS_BITS);
            let v = Lin::word(cell, VAL_BITS);
            let t = Lin::word(cell + VAL_BITS, TS_BITS);
            rb.bit_rows(cell, VAL_BITS + TS_BITS);
            rb.bit_rows(aux_off, K_BITS);

            let g_p = idx_word
                .clone()
                .scaled(F::from_u64(p.b_scan as u64))
                .plus(Lin::con(F::from_u64(j as u64)));
            let fa = g2[0].clone().minus(t).minus(g_p.scaled(two_pow_ts));
            let prev = if j == 0 {
                let base = zm.x(x_offsets::H_IN + x_slot * K_BITS);
                [Lin::word(base, 64), Lin::word(base + 64, 64)]
            } else {
                let o = if x_slot == H_IS {
                    zm.scan_aux[j - 1].h_is
                } else {
                    zm.scan_aux[j - 1].h_fs
                };
                [Lin::word(o, 64), Lin::word(o + 64, 64)]
            };
            rb.k_rows(
                w_const,
                [&Lin::word(aux_off, 64), &Lin::word(aux_off + 64, 64)],
                [&prev[0], &prev[1]],
                &Lin::default(),
                &one,
                [&fa, &g2[1]],
                [&g1[0], &g1[1]],
                &v,
            );
        }
    }

    // Boundary rows: the x outputs equal the last slot values.
    let last_op = &zm.op_aux[p.b_ops - 1];
    let last_scan = &zm.scan_aux[p.b_scan - 1];
    rb.linear_row(
        &Lin::word(zm.x(x_offsets::TS_OUT), TS_BITS),
        &ts_in.plus(Lin::word(last_op.cnt, zm.cnt_bits)),
    );
    for s in 0..p.num_stacks {
        rb.linear_row(
            &Lin::word(zm.x(x_offsets::SP + s * 2 * sigma + sigma), sigma),
            &Lin::word(last_op.sp[s], sigma),
        );
    }
    for (slot, aux_off) in [
        (H_RS, last_op.h_rs),
        (H_WS, last_op.h_ws),
        (H_IS, last_scan.h_is),
        (H_FS, last_scan.h_fs),
    ] {
        let x_base = zm.x(x_offsets::H_OUT + slot * K_BITS);
        rb.linear_row(&Lin::word(x_base, 64), &Lin::word(aux_off, 64));
        rb.linear_row(&Lin::word(x_base + 64, 64), &Lin::word(aux_off + 64, 64));
    }
}

/// Static lane-residency audit: every column read by the fingerprint-input
/// matrices — the slots
/// that determine an op's or cell's multiset contribution — must lie in
/// the public `x` region (verifier-checked), the committed lanes (bound by
/// `c_ops`/`c_is`/`c_fs` before γ), or the deterministically-pinned aux:
/// the E2-constrained `cnt` (a function of lane pads) and the
/// E11-constrained `sw` bits (products of two lane bits, gating the
/// push/pop skip terms — v3.1). Run at construction, so a layout change
/// that leaks a free interpretation bit into the products fails the
/// build — not a review.
fn audit_lane_residency(p: &NebulaParams, zm: &ZMap, rb: &RowBuilder) {
    let in_lane = |col: usize| {
        (zm.ops_lane..zm.ops_lane + p.ops_lane_bits()).contains(&col)
            || (zm.is_lane..zm.is_lane + p.scan_lane_bits()).contains(&col)
            || (zm.fs_lane..zm.fs_lane + p.scan_lane_bits()).contains(&col)
    };
    let in_pinned_aux = |col: usize| {
        zm.op_aux
            .iter()
            .any(|a| (a.cnt..a.cnt + zm.cnt_bits).contains(&col) || a.sw.contains(&col))
    };
    for m in [M_P, M_Q, M_FA, M_FB, M_GA, M_GB, M_V] {
        for &(row, col, _) in &rb.trips[m] {
            assert!(
                col < 1 + p.x_bits() || in_lane(col) || in_pinned_aux(col),
                "lane-residency violation: fingerprint-input matrix {m} row {row} reads column {col}"
            );
        }
    }
}

/// The extension's binomial constant `W` (`u² = W`), read off `neo_math::K`
/// so the circuit can never disagree with native K arithmetic.
fn binomial_w() -> F {
    let u = K::from_coeffs([F::ZERO, F::ONE]);
    let coeffs = (u * u).as_coeffs();
    assert!(
        coeffs[1] == F::ZERO,
        "neo_math::K is expected to be a binomial extension (u² ∈ F)"
    );
    coeffs[0]
}

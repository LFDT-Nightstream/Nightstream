//! Encoding functions for Twist and Shout witnesses.
//!
//! This module implements the **index-bit addressing** strategy from the integration plan:
//! instead of committing O(n_side) one-hot columns per address dimension, we commit
//! O(log n_side) bit columns. This provides a 32× reduction in committed address width
//! for typical memory sizes.
//!
//! ## Witness Layout for MemWitness (Twist)
//!
//! The matrices in `MemWitness.mats` are ordered as (Inc matrix REMOVED for soundness):
//! - `0 .. d*ell`:           Read address bits: ra_bits[dim][bit] for dim in 0..d, bit in 0..ell
//! - `d*ell .. 2*d*ell`:     Write address bits: wa_bits[dim][bit]
//! - `2*d*ell + 0`:          has_read(j) flags
//! - `2*d*ell + 1`:          has_write(j) flags
//! - `2*d*ell + 2`:          wv(j) = write values
//! - `2*d*ell + 3`:          rv(j) = read values
//! - `2*d*ell + 4`:          inc_at_write_addr(j) = Inc(write_addr_j, j) - increment at write address
//!
//! NOTE: The full Inc(k,j) matrix was removed because:
//! 1. It caused X pollution (embedded at offset 0)
//! 2. Width mismatch issues (k*steps may exceed ccs_m)
//! 3. Time alignment problems (different offset than other columns)
//!
//! The ValEval oracle now uses has_write, wa_bits, and inc_at_write_addr to reconstruct
//! the needed Inc(r_addr, t) contribution during the sum-check.
//!
//! ## Witness Layout for LutWitness (Shout)
//!
//! The matrices in `LutWitness.mats` are ordered as:
//! - `0 .. d*ell`:           Lookup address bits (masked by has_lookup)
//! - `d*ell`:                has_lookup(j) flags
//! - `d*ell + 1`:            val(j) = observed lookup values
//!
//! Note: `table_at_addr` is NOT committed in the address-domain architecture.
//! The lookup check uses an address-domain sum-check where Tablẽ(r_addr) is
//! computed directly from the public table by the verifier.

use crate::plain::{LutTable, PlainLutTrace, PlainMemLayout, PlainMemTrace};
#[cfg(debug_assertions)]
use crate::shout::{check_shout_semantics, split_lut_mats};
use crate::witness::{LutInstance, LutWitness, MemInstance, MemWitness};
use neo_ajtai::{decomp_b, DecompStyle};
use neo_ccs::matrix::Mat;
use neo_params::NeoParams;
use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::Goldilocks;
use std::marker::PhantomData;

// ============================================================================
// CCS Width Embedding
// ============================================================================

/// Embed a vector into a larger vector at a specific offset.
///
/// This ensures all committed vectors have exactly `total_cols` elements,
/// with the actual data placed starting at `offset`. The first `offset`
/// elements and any elements after the data are zero-padded.
///
/// # Arguments
/// * `total_cols` - The total width of the output vector (should equal `ccs_m`)
/// * `offset` - The starting column for the data (should be >= `m_in` for memory witnesses)
/// * `v` - The input vector to embed
///
/// # Panics
/// Panics if `offset + v.len() > total_cols`.
///
/// # Why This Matters
/// Neo's ME relation requires `X = L_x(Z)` where `L_x` projects to the first `m_in` columns.
/// For memory/LUT witnesses, we want `X = 0` to avoid contaminating the folded X accumulator.
/// By embedding data at `offset >= m_in`, we ensure the first `m_in` columns are zero.
pub fn embed_vec(total_cols: usize, offset: usize, v: &[Goldilocks]) -> Vec<Goldilocks> {
    assert!(
        offset + v.len() <= total_cols,
        "embed_vec: offset ({}) + v.len() ({}) = {} exceeds total_cols ({})",
        offset,
        v.len(),
        offset + v.len(),
        total_cols
    );
    let mut out = vec![Goldilocks::ZERO; total_cols];
    out[offset..offset + v.len()].copy_from_slice(v);
    out
}

/// Compute ceil(log2(n_side)) - the number of bits needed to represent addresses.
pub fn get_ell(n_side: usize) -> usize {
    if n_side <= 1 {
        1 // Need at least 1 bit
    } else if n_side.is_power_of_two() {
        n_side.trailing_zeros() as usize
    } else {
        (usize::BITS - (n_side - 1).leading_zeros()) as usize
    }
}

/// Compute Inc(addr_t, t) for each step t.
///
/// Returns the increment value applied to the address at step t.
fn compute_inc_at_addresses(inc: &[Vec<Goldilocks>], addrs: &[u64], k: usize, num_steps: usize) -> Vec<Goldilocks> {
    let mut result = Vec::with_capacity(num_steps);

    for j in 0..num_steps {
        let addr = addrs[j] as usize;
        if addr < k {
            result.push(inc[addr][j]);
        } else {
            // Address outside tracked range
            result.push(Goldilocks::ZERO);
        }
    }

    result
}

/// Ajtai-encode a vector using base-b balanced decomposition.
pub fn ajtai_encode_vector(params: &NeoParams, v: &[Goldilocks]) -> Mat<Goldilocks> {
    let d = params.d as usize;
    let m = v.len();

    let z_vec = decomp_b(v, params.b, d, DecompStyle::Balanced);
    // decomp_b returns digits "per element", i.e. column-major for the (d × m) matrix:
    // z_vec[c*d + r] = digit r (row) of value c (column).
    // We convert that into Mat's row-major layout below.
    debug_assert_eq!(z_vec.len(), d * m, "Ajtai encoding dimension mismatch");

    let mut row_major = vec![Goldilocks::ZERO; d * m];
    for c in 0..m {
        for r in 0..d {
            row_major[r * m + c] = z_vec[c * d + r];
        }
    }

    Mat::from_row_major(d, m, row_major)
}

/// Encode memory trace for Twist using **index-bit addressing**.
///
/// Instead of committing d one-hot vectors of length n_side*steps each,
/// we commit d*ell bit vectors of length steps each, where ell = ceil(log2(n_side)).
///
/// This provides O(log n_side) columns instead of O(n_side) columns per dimension.
///
/// # Parameters
/// * `ccs_m` - The CCS witness width (`s.m`). If `None`, uses `steps` (legacy mode - NOT RECOMMENDED).
/// * `m_in` - The number of public input columns. Memory data is embedded at offset `m_in`
///   to ensure `X = L_x(Z) = 0` for memory ME claims.
///
/// # Correctness
/// When `ccs_m` is provided, all committed matrices will have exactly `ccs_m` columns,
/// ensuring proper alignment with Neo's ME relation and RLC/DEC pipeline.
pub fn encode_mem_for_twist<C, L>(
    params: &NeoParams,
    layout: &PlainMemLayout,
    trace: &PlainMemTrace<Goldilocks>,
    commit: &L,
    ccs_m: Option<usize>,
    m_in: usize,
) -> (MemInstance<C, Goldilocks>, MemWitness<Goldilocks>)
where
    L: Fn(&Mat<Goldilocks>) -> C,
{
    let mut comms = Vec::new();
    let mut mats = Vec::new();

    let num_steps = trace.steps;
    let n_side = layout.n_side;
    let dim_d = layout.d;
    let ell = get_ell(n_side);

    // Use CCS width if provided, otherwise fall back to legacy mode (steps width)
    // Legacy mode is NOT RECOMMENDED as it causes width mismatches with the CCS structure
    let target_width = ccs_m.unwrap_or(num_steps);
    let data_offset = m_in; // Embed data starting at m_in to keep X = 0

    // Helper to encode a column at CCS width
    let encode_at_ccs_width = |col: &[Goldilocks]| -> Mat<Goldilocks> {
        if ccs_m.is_some() {
            let embedded = embed_vec(target_width, data_offset, col);
            ajtai_encode_vector(params, &embedded)
        } else {
            // Legacy mode: encode at native width
            ajtai_encode_vector(params, col)
        }
    };

    // Helper: Decompose address column into 'ell' bit columns
    // Returns ell vectors, each of length num_steps
    let build_addr_bits = |addrs: &[u64], dim_idx: usize| -> Vec<Vec<Goldilocks>> {
        let divisor = (n_side as u64)
            .checked_pow(dim_idx as u32)
            .expect("Address dimension overflow");
        let mut cols = vec![vec![Goldilocks::ZERO; num_steps]; ell];

        for (j, &addr) in addrs.iter().enumerate() {
            let comp = (addr / divisor) as usize % n_side;
            for b in 0..ell {
                if (comp >> b) & 1 == 1 {
                    cols[b][j] = Goldilocks::ONE;
                }
            }
        }
        cols
    };

    // 1. Read address bits: d*ell matrices
    for dim in 0..dim_d {
        for col in build_addr_bits(&trace.read_addr, dim) {
            let mat = encode_at_ccs_width(&col);
            comms.push(commit(&mat));
            mats.push(mat);
        }
    }

    // 2. Write address bits: d*ell matrices
    for dim in 0..dim_d {
        for col in build_addr_bits(&trace.write_addr, dim) {
            let mat = encode_at_ccs_width(&col);
            comms.push(commit(&mat));
            mats.push(mat);
        }
    }

    // SECURITY FIX: Removed Inc(k, j) flattened matrix.
    //
    // The full k×steps Inc matrix was:
    // 1. Causing X pollution (embedded at offset 0 instead of m_in)
    // 2. Width mismatches (k*steps may exceed ccs_m)
    // 3. Time alignment issues (different column offset than other data)
    //
    // Instead, we commit only inc_at_write_addr (a single column of length steps)
    // which captures the sparse increment information needed for the Twist protocol.
    // The ValEval oracle reconstructs the needed Inc(r_addr, t) from:
    // - has_write(t): whether step t writes
    // - wa_bits(t): the write address at step t
    // - inc_at_write_addr(t): the increment value at step t (if writing)
    //
    // See TwistValEvalOracleSparse in twist_oracle.rs for the reconstruction.

    // 3. has_read(j) flags
    let hr_mat = encode_at_ccs_width(&trace.has_read);
    comms.push(commit(&hr_mat));
    mats.push(hr_mat);

    // 4. has_write(j) flags
    let hw_mat = encode_at_ccs_width(&trace.has_write);
    comms.push(commit(&hw_mat));
    mats.push(hw_mat);

    // 5. wv(j) = write values
    let wv_mat = encode_at_ccs_width(&trace.write_val);
    comms.push(commit(&wv_mat));
    mats.push(wv_mat);

    // 6. rv(j) = read values
    let rv_mat = encode_at_ccs_width(&trace.read_val);
    comms.push(commit(&rv_mat));
    mats.push(rv_mat);

    // 7. inc_at_write_addr(j) = Inc(write_addr_j, j) - the increment at the write address
    let inc_at_write_addr = compute_inc_at_addresses(&trace.inc, &trace.write_addr, layout.k, num_steps);
    let inc_at_write_addr_mat = encode_at_ccs_width(&inc_at_write_addr);
    comms.push(commit(&inc_at_write_addr_mat));
    mats.push(inc_at_write_addr_mat);

    // Commitment order in MemInstance/MemWitness (Route A layout):
    // 0 .. d*ell:         ra_bits (read address bits)
    // d*ell .. 2*d*ell:   wa_bits (write address bits)
    // 2*d*ell + 0:        has_read(j)
    // 2*d*ell + 1:        has_write(j)
    // 2*d*ell + 2:        wv(j) = write_val
    // 2*d*ell + 3:        rv(j) = read_val
    // 2*d*ell + 4:        inc_at_write_addr(j)

    (
        MemInstance {
            comms,
            k: layout.k,
            d: layout.d,
            n_side: layout.n_side,
            steps: num_steps,
            ell,
            init_vals: trace.init_vals.clone(),
            _phantom: PhantomData,
        },
        MemWitness { mats },
    )
}

/// Encode lookup trace for Shout using **index-bit addressing**.
///
/// Instead of committing d one-hot vectors of length n_side*steps each,
/// we commit d*ell bit vectors of length steps each.
///
/// # Parameters
/// * `ccs_m` - The CCS witness width (`s.m`). If `None`, uses `steps` (legacy mode - NOT RECOMMENDED).
/// * `m_in` - The number of public input columns. LUT data is embedded at offset `m_in`
///   to ensure `X = L_x(Z) = 0` for LUT ME claims.
pub fn encode_lut_for_shout<C, L>(
    params: &NeoParams,
    table: &LutTable<Goldilocks>,
    trace: &PlainLutTrace<Goldilocks>,
    commit: &L,
    ccs_m: Option<usize>,
    m_in: usize,
) -> (LutInstance<C, Goldilocks>, LutWitness<Goldilocks>)
where
    L: Fn(&Mat<Goldilocks>) -> C,
{
    let mut comms = Vec::new();
    let mut mats = Vec::new();

    let num_steps = trace.has_lookup.len();
    let n_side = table.n_side;
    let dim_d = table.d;
    let ell = get_ell(n_side);

    // Use CCS width if provided, otherwise fall back to legacy mode
    let target_width = ccs_m.unwrap_or(num_steps);
    let data_offset = m_in;

    // Helper to encode a column at CCS width
    let encode_at_ccs_width = |col: &[Goldilocks]| -> Mat<Goldilocks> {
        if ccs_m.is_some() {
            let embedded = embed_vec(target_width, data_offset, col);
            ajtai_encode_vector(params, &embedded)
        } else {
            ajtai_encode_vector(params, col)
        }
    };

    // Helper: Decompose address column into 'ell' bit columns (masked by has_lookup)
    // If has_lookup[j] = 0, all bits are 0 for that step.
    let build_masked_addr_bits = |addrs: &[u64], flags: &[Goldilocks], dim_idx: usize| -> Vec<Vec<Goldilocks>> {
        let divisor = (n_side as u64)
            .checked_pow(dim_idx as u32)
            .expect("Address dimension overflow");
        let mut cols = vec![vec![Goldilocks::ZERO; num_steps]; ell];

        for (j, (&addr, &flag)) in addrs.iter().zip(flags.iter()).enumerate() {
            if flag == Goldilocks::ONE {
                let comp = (addr / divisor) as usize % n_side;
                for b in 0..ell {
                    if (comp >> b) & 1 == 1 {
                        cols[b][j] = Goldilocks::ONE;
                    }
                }
            }
        }
        cols
    };

    // Lookup address bits: d*ell matrices
    for dim in 0..dim_d {
        for col in build_masked_addr_bits(&trace.addr, &trace.has_lookup, dim) {
            let mat = encode_at_ccs_width(&col);
            comms.push(commit(&mat));
            mats.push(mat);
        }
    }

    // Commit has_lookup(j) so the Shout argument can properly mask and handle address 0
    let has_lookup_mat = encode_at_ccs_width(&trace.has_lookup);
    comms.push(commit(&has_lookup_mat));
    mats.push(has_lookup_mat);

    // Commit observed lookup value val(j) - this ties the VM's lookup results to the table
    let val_mat = encode_at_ccs_width(&trace.val);
    comms.push(commit(&val_mat));
    mats.push(val_mat);

    // Note: table_at_addr is NOT committed in the address-domain architecture.
    // The verifier computes Tablẽ(r_addr) directly from the public table.

    // Witness layout: [addr_bits (d*ell), has_lookup, val]

    let inst = LutInstance {
        comms,
        k: table.k,
        d: table.d,
        n_side: table.n_side,
        steps: num_steps,
        ell,
        table: table.content.clone(),
        _phantom: PhantomData,
    };
    let wit = LutWitness { mats };

    // Debug-only semantic check: ensure Ajtai-encoded witness matches the plain trace
    #[cfg(debug_assertions)]
    {
        // Note: We create a temporary LutInstance without commitments for the check
        // since commitments require Clone and we're just checking the witness structure
        let check_inst = LutInstance::<(), Goldilocks> {
            comms: vec![(); inst.comms.len()],
            k: inst.k,
            d: inst.d,
            n_side: inst.n_side,
            steps: inst.steps,
            ell: inst.ell,
            table: inst.table.clone(),
            _phantom: PhantomData,
        };
        let _ = split_lut_mats(&check_inst, &wit); // Will panic if layout is wrong
        check_shout_semantics(params, &check_inst, &wit, &trace.val)
            .expect("Shout semantic check failed during encoding");
    }

    (inst, wit)
}

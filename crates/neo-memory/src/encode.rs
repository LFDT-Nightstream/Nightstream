//! Encoding functions for Twist and Shout witnesses.
//!
//! This module implements the **index-bit addressing** strategy from the integration plan:
//! instead of committing O(n_side) one-hot columns per address dimension, we commit
//! O(log n_side) bit columns. This provides a 32× reduction in committed address width
//! for typical memory sizes.
//!
//! ## Witness Layout for MemWitness (Twist)
//!
//! The matrices in `MemWitness.mats` are ordered as:
//! - `0 .. d*ell`:           Read address bits: ra_bits[dim][bit] for dim in 0..d, bit in 0..ell
//! - `d*ell .. 2*d*ell`:     Write address bits: wa_bits[dim][bit]
//! - `2*d*ell`:              Inc(k, j) flattened (k*steps elements)
//! - `2*d*ell + 1`:          has_read(j) flags
//! - `2*d*ell + 2`:          has_write(j) flags
//! - `2*d*ell + 3`:          wv(j) = write values
//! - `2*d*ell + 4`:          rv(j) = read values
//!
//! ## Witness Layout for LutWitness (Shout)
//!
//! The matrices in `LutWitness.mats` are ordered as:
//! - `0 .. d*ell`:           Lookup address bits (masked by has_lookup)
//! - `d*ell`:                has_lookup(j) flags
//! - `d*ell + 1`:            val(j) = observed lookup values

use crate::plain::{LutTable, PlainLutTrace, PlainMemLayout, PlainMemTrace};
use crate::shout::{check_shout_semantics, split_lut_mats};
use crate::witness::{LutInstance, LutWitness, MemInstance, MemWitness};
use neo_ajtai::{decomp_b, DecompStyle};
use neo_ccs::matrix::Mat;
use neo_params::NeoParams;
use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::Goldilocks;
use std::marker::PhantomData;

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
pub fn encode_mem_for_twist<C, L>(
    params: &NeoParams,
    layout: &PlainMemLayout,
    trace: &PlainMemTrace<Goldilocks>,
    commit: &L,
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
            let mat = ajtai_encode_vector(params, &col);
            comms.push(commit(&mat));
            mats.push(mat);
        }
    }

    // 2. Write address bits: d*ell matrices
    for dim in 0..dim_d {
        for col in build_addr_bits(&trace.write_addr, dim) {
            let mat = ajtai_encode_vector(params, &col);
            comms.push(commit(&mat));
            mats.push(mat);
        }
    }

    // 3. Inc(k, j) flattened: k*steps elements
    let mut inc_flat = Vec::with_capacity(layout.k * num_steps);
    for k_idx in 0..layout.k {
        inc_flat.extend_from_slice(&trace.inc[k_idx]);
    }
    let inc_mat = ajtai_encode_vector(params, &inc_flat);
    comms.push(commit(&inc_mat));
    mats.push(inc_mat);

    // 4. has_read(j) flags
    let hr_mat = ajtai_encode_vector(params, &trace.has_read);
    comms.push(commit(&hr_mat));
    mats.push(hr_mat);

    // 5. has_write(j) flags
    let hw_mat = ajtai_encode_vector(params, &trace.has_write);
    comms.push(commit(&hw_mat));
    mats.push(hw_mat);

    // 6. wv(j) = write values
    let wv_mat = ajtai_encode_vector(params, &trace.write_val);
    comms.push(commit(&wv_mat));
    mats.push(wv_mat);

    // 7. rv(j) = read values
    let rv_mat = ajtai_encode_vector(params, &trace.read_val);
    comms.push(commit(&rv_mat));
    mats.push(rv_mat);

    // Commitment order in MemInstance/MemWitness:
    // 0 .. d*ell:         ra_bits (read address bits)
    // d*ell .. 2*d*ell:   wa_bits (write address bits)
    // 2*d*ell:            Inc(k, j) flattened
    // 2*d*ell + 1:        has_read(j)
    // 2*d*ell + 2:        has_write(j)
    // 2*d*ell + 3:        wv(j) = write_val
    // 2*d*ell + 4:        rv(j) = read_val

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
pub fn encode_lut_for_shout<C, L>(
    params: &NeoParams,
    table: &LutTable<Goldilocks>,
    trace: &PlainLutTrace<Goldilocks>,
    commit: &L,
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
            let mat = ajtai_encode_vector(params, &col);
            comms.push(commit(&mat));
            mats.push(mat);
        }
    }

    // Commit has_lookup(j) so the Shout argument can properly mask and handle address 0
    let has_lookup_mat = ajtai_encode_vector(params, &trace.has_lookup);
    comms.push(commit(&has_lookup_mat));
    mats.push(has_lookup_mat);

    // Commit observed lookup value val(j) - this ties the VM's lookup results to the table
    let val_mat = ajtai_encode_vector(params, &trace.val);
    comms.push(commit(&val_mat));
    mats.push(val_mat);

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_ell() {
        assert_eq!(get_ell(1), 1); // 1 element needs 1 bit (special case)
        assert_eq!(get_ell(2), 1); // 2 elements need 1 bit
        assert_eq!(get_ell(3), 2); // 3 elements need 2 bits
        assert_eq!(get_ell(4), 2); // 4 elements need 2 bits
        assert_eq!(get_ell(5), 3); // 5 elements need 3 bits
        assert_eq!(get_ell(8), 3); // 8 elements need 3 bits
        assert_eq!(get_ell(256), 8); // 256 elements need 8 bits
    }
}

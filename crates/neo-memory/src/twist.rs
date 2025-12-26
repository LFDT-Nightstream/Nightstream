//! Twist argument for read/write memory correctness (Route A).
//!
//! This module intentionally supports only Route A integration inside
//! `neo-fold::shard`. The legacy fixed-challenge APIs were removed.
//!
//! In Route A:
//! - Twist first runs an **address-domain** batched sumcheck ("addr-pre") to bind `r_addr`
//!   and produce the **time-lane claimed sums** for the read/write checks.
//! - Then the shard runs a shared-challenge **time-domain** batched sumcheck (shared with CCS/Shout),
//!   ending at `r_time`, which yields the `r_time`-lane ME openings.
//! - Finally, Twist runs a separate val-eval sum-check to obtain `Val(r_addr, r_time)` and a fresh `r_val`,
//!   producing the val-lane ME obligations needed to check the terminal identity.
//!
//! The initial memory state is provided via [`crate::mem_init::MemInit`].

use crate::ajtai::decode_vector as ajtai_decode_vector;
use crate::mem_init::MemInit;
use crate::sumcheck_proof::BatchedAddrProof;
use crate::ts_common as ts;
use crate::witness::{MemInstance, MemWitness};
use neo_ajtai::Commitment as AjtaiCmt;
use neo_ccs::matrix::Mat;
use neo_params::NeoParams;
use neo_reductions::error::PiCcsError;
use neo_transcript::Poseidon2Transcript;
use p3_field::PrimeField;
use serde::{Deserialize, Serialize};

// ============================================================================
// Input validation
// ============================================================================

// ============================================================================
// Transcript binding
// ============================================================================

/// Absorb all Twist commitments into the transcript.
///
/// Must be called before sampling any challenge used to open these commitments.
pub fn absorb_commitments<F>(tr: &mut Poseidon2Transcript, inst: &MemInstance<AjtaiCmt, F>) {
    ts::absorb_ajtai_commitments(tr, b"twist/absorb_commitments", b"twist/comm_idx", &inst.comms);
}

// ============================================================================
// Proof metadata (Route A)
// ============================================================================

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TwistProof<F> {
    /// Address-domain sum-check metadata for Route A (two-claim batch: read/write).
    pub addr_pre: BatchedAddrProof<F>,
    pub val_eval: Option<TwistValEvalProof<F>>,
}

impl<F: Default> Default for TwistProof<F> {
    fn default() -> Self {
        Self {
            addr_pre: BatchedAddrProof::default(),
            val_eval: None,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TwistValEvalProof<F> {
    /// Σ_t Inc(r_addr, t) · LT(t, r_time) (init term excluded).
    pub claimed_inc_sum_lt: F,
    /// Sum-check rounds for the LT-weighted claim (ell_n rounds, cycle/time variables).
    pub rounds_lt: Vec<Vec<F>>,

    /// Σ_t Inc(r_addr, t) (total increment over the whole chunk).
    pub claimed_inc_sum_total: F,
    /// Sum-check rounds for the total-increment claim (ell_n rounds, cycle/time variables).
    pub rounds_total: Vec<Vec<F>>,

    /// Optional rollover claim for linking consecutive chunks (Route A):
    /// Σ_t Inc_prev(r_addr_current, t) (total increment over the *previous* chunk, evaluated at the *current* r_addr).
    ///
    /// Present iff the prover links this step to a previous step.
    pub claimed_prev_inc_sum_total: Option<F>,
    /// Sum-check rounds for the rollover total-increment claim (ell_n rounds).
    pub rounds_prev_total: Option<Vec<Vec<F>>>,
}

// ============================================================================
// Witness layout helpers
// ============================================================================

#[derive(Clone, Debug)]
pub struct TwistWitnessParts<'a, F> {
    pub ra_bit_mats: &'a [Mat<F>],
    pub wa_bit_mats: &'a [Mat<F>],
    pub has_read_mat: &'a Mat<F>,
    pub has_write_mat: &'a Mat<F>,
    pub wv_mat: &'a Mat<F>,
    pub rv_mat: &'a Mat<F>,
    pub inc_at_write_addr_mat: &'a Mat<F>,
}

/// Layout: `[ra_bits (d*ell), wa_bits (d*ell), has_read, has_write, wv, rv, inc_at_write_addr]`.
pub fn split_mem_mats<'a, F: Clone>(
    inst: &MemInstance<impl Clone, F>,
    wit: &'a MemWitness<F>,
) -> TwistWitnessParts<'a, F> {
    let ell_addr = inst.d * inst.ell;
    let expected = 2 * ell_addr + 5;
    assert_eq!(
        wit.mats.len(),
        expected,
        "MemWitness has {} matrices, expected {} (2*d*ell={} + has_read + has_write + wv + rv + inc_at_write_addr)",
        wit.mats.len(),
        expected,
        2 * ell_addr
    );

    TwistWitnessParts {
        ra_bit_mats: &wit.mats[..ell_addr],
        wa_bit_mats: &wit.mats[ell_addr..2 * ell_addr],
        has_read_mat: &wit.mats[2 * ell_addr],
        has_write_mat: &wit.mats[2 * ell_addr + 1],
        wv_mat: &wit.mats[2 * ell_addr + 2],
        rv_mat: &wit.mats[2 * ell_addr + 3],
        inc_at_write_addr_mat: &wit.mats[2 * ell_addr + 4],
    }
}

// ============================================================================
// Semantic checker (debug/tests)
// ============================================================================

pub fn check_twist_semantics<F: PrimeField>(
    params: &NeoParams,
    inst: &MemInstance<impl Clone, F>,
    wit: &MemWitness<F>,
) -> Result<(), PiCcsError> {
    crate::addr::validate_twist_bit_addressing(inst)?;

    let parts = split_mem_mats(inst, wit);
    let steps = inst.steps;
    let k = inst.k;

    let has_read = ajtai_decode_vector(params, parts.has_read_mat);
    let has_write = ajtai_decode_vector(params, parts.has_write_mat);
    let wv = ajtai_decode_vector(params, parts.wv_mat);
    let rv = ajtai_decode_vector(params, parts.rv_mat);
    let inc_at_write_addr = ajtai_decode_vector(params, parts.inc_at_write_addr_mat);

    // Bitness of address bits.
    for mat in parts.ra_bit_mats.iter().chain(parts.wa_bit_mats.iter()) {
        let v = ajtai_decode_vector(params, mat);
        for (j, &x) in v.iter().enumerate() {
            if x != F::ZERO && x != F::ONE {
                return Err(PiCcsError::InvalidInput(format!(
                    "Twist: non-binary value in address bit column at step {j}: {x:?}"
                )));
            }
        }
    }

    // Decode addresses.
    let read_addrs = ts::decode_addrs_from_bits(params, parts.ra_bit_mats, inst.d, inst.ell, inst.n_side, steps);
    let write_addrs = ts::decode_addrs_from_bits(params, parts.wa_bit_mats, inst.d, inst.ell, inst.n_side, steps);

    // Route A: initial state comes from the instance init policy.
    let mut mem: std::collections::HashMap<u64, F> = std::collections::HashMap::new();
    match &inst.init {
        MemInit::Zero => {}
        MemInit::Sparse(pairs) => {
            for (addr, val) in pairs.iter() {
                if *val != F::ZERO {
                    mem.insert(*addr, *val);
                }
            }
        }
    }
    for j in 0..steps {
        if has_read[j] == F::ONE {
            let addr = read_addrs[j] as usize;
            if addr < k {
                let cur = mem.get(&(addr as u64)).copied().unwrap_or(F::ZERO);
                if rv[j] != cur {
                    return Err(PiCcsError::InvalidInput(format!(
                        "Twist: read mismatch at step {j}: rv={:?}, mem[{addr}]={:?}",
                        rv[j], cur
                    )));
                }
            }
        }

        if has_write[j] == F::ONE {
            let addr = write_addrs[j] as usize;
            if addr < k {
                let old = mem.get(&(addr as u64)).copied().unwrap_or(F::ZERO);
                let expected_inc = wv[j] - old;
                if inc_at_write_addr[j] != expected_inc {
                    return Err(PiCcsError::InvalidInput(format!(
                        "Twist: inc mismatch at step {j}, addr {addr}: got {:?}, expected {:?}",
                        inc_at_write_addr[j], expected_inc
                    )));
                }
                if wv[j] == F::ZERO {
                    mem.remove(&(addr as u64));
                } else {
                    mem.insert(addr as u64, wv[j]);
                }
            }
        }
    }

    Ok(())
}

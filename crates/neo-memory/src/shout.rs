//! Shout argument for read-only lookup table correctness (Route A).
//!
//! This module is intentionally minimal: the only supported integration is Route A
//! inside `neo-fold::shard`, which proves Shout constraints via shared-challenge
//! batched sum-check and terminal checks bound to ME openings.

use crate::ajtai::decode_vector as ajtai_decode_vector;
use crate::sumcheck_proof::BatchedAddrProof;
use crate::ts_common as ts;
use crate::twist_oracle::{
    build_eq_table, AddressLookupOracle, IndexAdapterOracle, LazyBitnessOracle, ProductRoundOracle,
};
use crate::witness::{LutInstance, LutWitness};
use neo_ajtai::Commitment as AjtaiCmt;
use neo_ccs::matrix::Mat;
use neo_math::{F as BaseField, K as KElem};
use neo_params::NeoParams;
use neo_reductions::error::PiCcsError;
use neo_transcript::Poseidon2Transcript;
use p3_field::PrimeField;
use serde::{Deserialize, Serialize};

// ============================================================================
// Input validation
// ============================================================================

fn validate_index_bit_addressing<Cmt, F>(inst: &LutInstance<Cmt, F>) -> Result<(), PiCcsError> {
    crate::addr::validate_pow2_bit_addressing("Shout", inst.n_side, inst.d, inst.ell, inst.k)?;
    if inst.table.len() != inst.k {
        return Err(PiCcsError::InvalidInput(format!(
            "Shout: table.len()={} must equal k={} for bit addressing",
            inst.table.len(),
            inst.k
        )));
    }

    Ok(())
}

// ============================================================================
// Transcript binding
// ============================================================================

/// Absorb all Shout commitments into the transcript.
///
/// Must be called before sampling any challenge used to open these commitments.
pub fn absorb_commitments<F>(tr: &mut Poseidon2Transcript, inst: &LutInstance<AjtaiCmt, F>) {
    ts::absorb_ajtai_commitments(tr, b"shout/absorb_commitments", b"shout/comm_idx", &inst.comms);
}

// ============================================================================
// Proof metadata (Route A)
// ============================================================================

/// Route A Shout proof metadata.
///
/// In Route A, the time-domain rounds are carried by the shard’s `BatchedTimeProof`.
/// Shout contributes only the address-domain sum-check metadata.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ShoutProof<F> {
    /// Address-domain sum-check metadata for Route A (single-claim batch).
    pub addr_pre: BatchedAddrProof<F>,
}

impl<F: Default> Default for ShoutProof<F> {
    fn default() -> Self {
        Self {
            addr_pre: BatchedAddrProof::default(),
        }
    }
}

// ============================================================================
// Witness layout helpers
// ============================================================================

#[derive(Clone, Debug)]
pub struct ShoutWitnessParts<'a, F> {
    pub addr_bit_mats: &'a [Mat<F>],
    pub has_lookup_mat: &'a Mat<F>,
    pub val_mat: &'a Mat<F>,
}

/// Layout: `[addr_bits (d*ell), has_lookup, val]`.
pub fn split_lut_mats<'a, F: Clone>(
    inst: &LutInstance<impl Clone, F>,
    wit: &'a LutWitness<F>,
) -> ShoutWitnessParts<'a, F> {
    let ell_addr = inst.d * inst.ell;
    let expected = ell_addr + 2;
    assert_eq!(
        wit.mats.len(),
        expected,
        "LutWitness has {} matrices, expected {} (d*ell={} + has_lookup + val)",
        wit.mats.len(),
        expected,
        ell_addr
    );
    ShoutWitnessParts {
        addr_bit_mats: &wit.mats[..ell_addr],
        has_lookup_mat: &wit.mats[ell_addr],
        val_mat: &wit.mats[ell_addr + 1],
    }
}

// ============================================================================
// Decoded columns (Route A helpers)
// ============================================================================

#[derive(Clone, Debug)]
pub struct ShoutDecodedCols {
    pub addr_bits: Vec<Vec<KElem>>,
    pub has_lookup: Vec<KElem>,
    pub val: Vec<KElem>,
}

pub fn decode_shout_cols<Cmt: Clone>(
    params: &NeoParams,
    inst: &LutInstance<Cmt, BaseField>,
    wit: &LutWitness<BaseField>,
    ell_cycle: usize,
) -> Result<ShoutDecodedCols, PiCcsError> {
    validate_index_bit_addressing(inst)?;
    let parts = split_lut_mats(inst, wit);

    let pow2_cycle = 1usize << ell_cycle;
    if inst.steps > pow2_cycle {
        return Err(PiCcsError::InvalidInput(format!(
            "Shout(Route A): steps={} exceeds 2^ell_cycle={pow2_cycle}",
            inst.steps
        )));
    }

    let addr_bits = ts::decode_mats_to_k_padded(params, parts.addr_bit_mats, pow2_cycle);
    let has_lookup = ts::decode_mat_to_k_padded(params, parts.has_lookup_mat, pow2_cycle);
    let val = ts::decode_mat_to_k_padded(params, parts.val_mat, pow2_cycle);

    Ok(ShoutDecodedCols {
        addr_bits,
        has_lookup,
        val,
    })
}

// ============================================================================
// Semantic checker (debug/tests)
// ============================================================================

pub fn check_shout_semantics<F: PrimeField>(
    params: &NeoParams,
    inst: &LutInstance<impl Clone, F>,
    wit: &LutWitness<F>,
    expected_vals: &[F],
) -> Result<(), PiCcsError> {
    validate_index_bit_addressing(inst)?;

    let parts = split_lut_mats(inst, wit);
    let steps = inst.steps;

    // Bitness: addr bits + has_lookup.
    for mat in parts
        .addr_bit_mats
        .iter()
        .chain(core::iter::once(parts.has_lookup_mat))
    {
        let v = ajtai_decode_vector(params, mat);
        for (j, &x) in v.iter().enumerate() {
            if x != F::ZERO && x != F::ONE {
                return Err(PiCcsError::InvalidInput(format!(
                    "Shout: non-binary value at step {j}: {x:?}"
                )));
            }
        }
    }

    let has_lookup = ajtai_decode_vector(params, parts.has_lookup_mat);
    let val = ajtai_decode_vector(params, parts.val_mat);
    let addrs = ts::decode_addrs_from_bits(params, parts.addr_bit_mats, inst.d, inst.ell, inst.n_side, steps);

    for j in 0..steps {
        if has_lookup[j] == F::ONE {
            let addr = addrs[j] as usize;
            if addr >= inst.table.len() {
                return Err(PiCcsError::InvalidInput(format!(
                    "Shout: out-of-range lookup at step {j}: addr={addr} >= table.len()={}",
                    inst.table.len()
                )));
            }
            let table_val = inst.table[addr];
            if val[j] != table_val {
                return Err(PiCcsError::InvalidInput(format!(
                    "Shout: lookup mismatch at step {j}: Table[{addr}]={table_val:?}, committed val={:?}",
                    val[j]
                )));
            }
            if j < expected_vals.len() && val[j] != expected_vals[j] {
                return Err(PiCcsError::InvalidInput(format!(
                    "Shout: expected value mismatch at step {j}: committed={:?}, expected={:?}",
                    val[j], expected_vals[j]
                )));
            }
        }
    }

    Ok(())
}

// ============================================================================
// Route A oracles
// ============================================================================

/// Route A Shout oracles (time-domain only).
pub struct RouteAShoutOracles {
    /// Value oracle: `χ_{r_cycle}(t) * has_lookup(t) * val(t)`.
    pub value: ProductRoundOracle,
    /// Claimed sum for the value oracle.
    pub value_claim: KElem,
    /// Adapter oracle: `χ_{r_cycle}(t) * has_lookup(t) * eq(addr_bits(t), r_addr)`.
    pub adapter: IndexAdapterOracle,
    /// Claimed sum for the adapter oracle.
    pub adapter_claim: KElem,
    /// Bitness checks for address bits + has_lookup (time-domain), χ_{r_cycle}-weighted.
    pub bitness: Vec<LazyBitnessOracle>,
    /// Number of address bits (`d*ell`).
    pub ell_addr: usize,
}

pub fn build_shout_addr_oracle<Cmt: Clone>(
    inst: &LutInstance<Cmt, BaseField>,
    decoded: &ShoutDecodedCols,
    r_cycle: &[KElem],
    table_k: &[KElem],
) -> Result<(AddressLookupOracle, KElem), PiCcsError> {
    let ell_addr = inst.d * inst.ell;
    if table_k.len() != inst.table.len() {
        return Err(PiCcsError::InvalidInput(format!(
            "Shout: table_k.len()={} != inst.table.len()={}",
            table_k.len(),
            inst.table.len()
        )));
    }
    let (oracle, addr_claim_sum) =
        AddressLookupOracle::new(&decoded.addr_bits, &decoded.has_lookup, table_k, r_cycle, ell_addr);
    Ok((oracle, addr_claim_sum))
}

pub fn build_route_a_shout_oracles<Cmt: Clone>(
    inst: &LutInstance<Cmt, BaseField>,
    decoded: &ShoutDecodedCols,
    r_cycle: &[KElem],
    r_addr: &[KElem],
) -> Result<RouteAShoutOracles, PiCcsError> {
    validate_index_bit_addressing(inst)?;
    let ell_addr = inst.d * inst.ell;
    if r_addr.len() != ell_addr {
        return Err(PiCcsError::InvalidInput(format!(
            "Shout(Route A): r_addr.len()={} != ell_addr={}",
            r_addr.len(),
            ell_addr
        )));
    }

    let chi_cycle = build_eq_table(r_cycle);
    if chi_cycle.len() != decoded.has_lookup.len() {
        return Err(PiCcsError::InvalidInput(format!(
            "Shout(Route A): chi_cycle.len()={} != has_lookup.len()={}",
            chi_cycle.len(),
            decoded.has_lookup.len()
        )));
    }

    let value = ProductRoundOracle::new(
        vec![chi_cycle.clone(), decoded.has_lookup.clone(), decoded.val.clone()],
        3,
    );
    let value_claim = value.sum_over_hypercube();

    let adapter = IndexAdapterOracle::new_with_gate(&decoded.addr_bits, &decoded.has_lookup, r_cycle, r_addr);
    let adapter_claim = adapter.compute_claim();

    let mut bitness = Vec::with_capacity(ell_addr + 1);
    for col in decoded.addr_bits.iter().cloned() {
        bitness.push(LazyBitnessOracle::new_with_cycle(r_cycle, col));
    }
    bitness.push(LazyBitnessOracle::new_with_cycle(r_cycle, decoded.has_lookup.clone()));

    Ok(RouteAShoutOracles {
        value,
        value_claim,
        adapter,
        adapter_claim,
        bitness,
        ell_addr,
    })
}

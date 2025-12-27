use crate::ajtai::decode_vector as ajtai_decode_vector;
use crate::riscv_lookups::{compute_op, uninterleave_bits};
use crate::sumcheck_proof::BatchedAddrProof;
use crate::ts_common as ts;
use crate::witness::{LutInstance, LutTableSpec, LutWitness};
use neo_ccs::matrix::Mat;
use neo_params::NeoParams;
use neo_reductions::error::PiCcsError;
use p3_field::PrimeField;
use serde::{Deserialize, Serialize};

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

// ============================================================================
// Semantic checker (debug/tests)
// ============================================================================

pub fn check_shout_semantics<F: PrimeField>(
    params: &NeoParams,
    inst: &LutInstance<impl Clone, F>,
    wit: &LutWitness<F>,
    expected_vals: &[F],
) -> Result<(), PiCcsError> {
    crate::addr::validate_shout_bit_addressing(inst)?;

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
            let table_val = match &inst.table_spec {
                Some(LutTableSpec::RiscvOpcode { opcode, xlen }) => {
                    let (rs1, rs2) = uninterleave_bits(addrs[j] as u128);
                    // NOTE: For RV64 this currently truncates keys to 64 bits at trace time.
                    // This mode is intended for RV32 (xlen=32) until RV64 key encoding is fixed.
                    let out = compute_op(*opcode, rs1, rs2, *xlen);
                    F::from_u64(out)
                }
                None => {
                    if addr >= inst.table.len() {
                        return Err(PiCcsError::InvalidInput(format!(
                            "Shout: out-of-range lookup at step {j}: addr={addr} >= table.len()={}",
                            inst.table.len()
                        )));
                    }
                    inst.table[addr]
                }
            };
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

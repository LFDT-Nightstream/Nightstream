//! Shout argument for read-only lookup table correctness.
//!
//! This module implements the Shout protocol with **index-bit addressing**:
//! instead of committing O(n_side) one-hot columns per dimension, we commit
//! O(log n_side) bit columns and prove consistency via the IDX→OH adapter.
//!
//! ## Protocol Overview
//!
//! 1. Sample r_addr and r_cycle
//! 2. **Index Adapter**: Prove bit columns encode valid addresses
//! 3. **Lookup Check**: Prove lookup values match table at those addresses
//! 4. **Bitness Check**: Prove all address bit columns are binary
//! 5. Generate ME claims for folding
//!
//! ## Witness Layout (Index-Bit Addressing)
//!
//! The matrices in `LutWitness.mats` are ordered as:
//! - `0 .. d*ell`: Lookup address bits (masked by has_lookup)

use crate::twist::ajtai_decode_vector;
use crate::witness::{LutInstance, LutWitness};
use neo_ccs::matrix::Mat;
use neo_ccs::relations::MeInstance;
use neo_params::NeoParams;
use neo_reductions::api::FoldingMode;
use neo_reductions::error::PiCcsError;
use neo_transcript::{Poseidon2Transcript, Transcript};
use p3_field::{PrimeCharacteristicRing, PrimeField};
use serde::{Deserialize, Serialize};

use crate::mle::{build_chi_table, mle_eval};
use crate::twist_oracle::{BitnessOracle, IndexAdapterOracle, ShoutLookupOracle};
use neo_ccs::traits::SModuleHomomorphism;
use neo_math::{from_complex, K as KElem};
use neo_reductions::sumcheck::run_sumcheck_prover;

/// Proof for the Shout lookup argument.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ShoutProof<F> {
    /// Sum-check round messages for index adapter
    pub adapter_rounds: Vec<Vec<F>>,
    /// Sum-check round messages for lookup check
    pub lookup_check_rounds: Vec<Vec<F>>,
    /// Sum-check round messages for bitness checks
    pub bitness_rounds: Vec<Vec<F>>,
    /// Claimed sum for adapter sum-check (for verifier)
    pub adapter_claim: F,
    /// Claimed sum for lookup sum-check (for verifier)
    pub lookup_claim: F,
}

impl<F: Default> Default for ShoutProof<F> {
    fn default() -> Self {
        Self {
            adapter_rounds: Vec::new(),
            lookup_check_rounds: Vec::new(),
            bitness_rounds: Vec::new(),
            adapter_claim: F::default(),
            lookup_claim: F::default(),
        }
    }
}

/// Decomposed witness matrices from LutWitness.
#[derive(Clone, Debug)]
pub struct ShoutWitnessParts<'a, F> {
    /// Address bit matrices: d*ell matrices
    pub addr_bit_mats: &'a [Mat<F>],
    /// has_lookup(j) flag column
    pub has_lookup_mat: &'a Mat<F>,
    /// Observed lookup return value val(j)
    pub val_mat: &'a Mat<F>,
}

/// Split the LutWitness matrices into named parts.
///
/// Layout: [addr_bits (d*ell), has_lookup, val]
pub fn split_lut_mats<'a, F: Clone>(
    inst: &LutInstance<impl Clone, F>,
    wit: &'a LutWitness<F>,
) -> ShoutWitnessParts<'a, F> {
    let d = inst.d;
    let ell = inst.ell;
    let addr_bits_count = d * ell;
    let expected = addr_bits_count + 2; // + has_lookup + val

    assert_eq!(
        wit.mats.len(),
        expected,
        "LutWitness has {} matrices, expected {} (d*ell={} + has_lookup + val)",
        wit.mats.len(),
        expected,
        addr_bits_count
    );

    ShoutWitnessParts {
        addr_bit_mats: &wit.mats[..addr_bits_count],
        has_lookup_mat: &wit.mats[addr_bits_count],
        val_mat: &wit.mats[addr_bits_count + 1],
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

fn sample_ext_point(tr: &mut Poseidon2Transcript, label: &'static [u8], len: usize) -> Vec<KElem> {
    let mut out = Vec::with_capacity(len);
    for i in 0..len {
        tr.append_message(label, &i.to_le_bytes());
        let c0 = tr.challenge_field(b"shout/coord/0");
        let c1 = tr.challenge_field(b"shout/coord/1");
        out.push(from_complex(c0, c1));
    }
    out
}

/// Decode has_lookup flags from address bits.
/// A step has a lookup if any bit in any dimension is set.
///
/// Note: This is kept for backward compatibility but the preferred approach
/// is to use the committed has_lookup column directly.
#[allow(dead_code)]
fn decode_has_lookup<F: PrimeField>(params: &NeoParams, addr_bit_mats: &[Mat<F>], steps: usize) -> Vec<F> {
    let mut has_lookup = vec![F::ZERO; steps];

    for mat in addr_bit_mats {
        let bits = ajtai_decode_vector(params, mat);
        for (j, &b) in bits.iter().enumerate().take(steps) {
            if b != F::ZERO {
                has_lookup[j] = F::ONE;
            }
        }
    }

    has_lookup
}

/// Decode addresses from bit columns.
fn decode_addrs_from_bits<F: PrimeField>(
    params: &NeoParams,
    addr_bit_mats: &[Mat<F>],
    d: usize,
    ell: usize,
    n_side: usize,
    steps: usize,
) -> Vec<u64> {
    let mut addrs = vec![0u64; steps];

    for dim in 0..d {
        let base = dim * ell;
        for j in 0..steps {
            let mut dim_val = 0u64;
            for b in 0..ell {
                if base + b < addr_bit_mats.len() {
                    let bit_col = ajtai_decode_vector(params, &addr_bit_mats[base + b]);
                    if j < bit_col.len() && bit_col[j] == F::ONE {
                        dim_val |= 1u64 << b;
                    }
                }
            }
            addrs[j] += dim_val * (n_side as u64).pow(dim as u32);
        }
    }

    addrs
}

// ============================================================================
// Semantic Checker (Debug)
// ============================================================================

/// Check Shout semantics without cryptographic proofs.
///
/// This validates that the committed witness is consistent:
/// - Address bit columns are binary
/// - has_lookup column is binary
/// - When has_lookup=1, the committed val matches Table[addr]
pub fn check_shout_semantics<F: PrimeField>(
    params: &NeoParams,
    inst: &LutInstance<impl Clone, F>,
    wit: &LutWitness<F>,
    expected_vals: &[F],
) -> Result<(), PiCcsError> {
    let parts = split_lut_mats(inst, wit);
    let steps = inst.steps;
    let d = inst.d;
    let ell = inst.ell;
    let n_side = inst.n_side;

    // Check bitness of address columns
    for mat in parts.addr_bit_mats {
        let v = ajtai_decode_vector(params, mat);
        for (j, &x) in v.iter().enumerate() {
            if x != F::ZERO && x != F::ONE {
                return Err(PiCcsError::InvalidInput(format!(
                    "Non-binary value in address bit column at step {}: {:?}",
                    j, x
                )));
            }
        }
    }

    // Decode has_lookup from committed column
    let has_lookup_vec = ajtai_decode_vector(params, parts.has_lookup_mat);
    for (j, &x) in has_lookup_vec.iter().enumerate() {
        if x != F::ZERO && x != F::ONE {
            return Err(PiCcsError::InvalidInput(format!(
                "Non-binary value in has_lookup at step {}: {:?}",
                j, x
            )));
        }
    }

    // Decode val from committed column
    let val_vec = ajtai_decode_vector(params, parts.val_mat);

    // Decode addresses
    let addrs = decode_addrs_from_bits(params, parts.addr_bit_mats, d, ell, n_side, steps);

    // Check lookup correctness
    for j in 0..steps {
        if has_lookup_vec[j] == F::ONE {
            let addr = addrs[j] as usize;

            if addr >= inst.table.len() {
                return Err(PiCcsError::InvalidInput(format!(
                    "Lookup at step {} has out-of-range address: {} >= {}",
                    j,
                    addr,
                    inst.table.len()
                )));
            }

            let table_val = inst.table[addr];
            let committed_val = val_vec[j];

            // Check committed val matches table
            if table_val != committed_val {
                return Err(PiCcsError::InvalidInput(format!(
                    "Lookup mismatch at step {}: Table[{}] = {:?}, but committed val = {:?}",
                    j, addr, table_val, committed_val
                )));
            }

            // Also check against expected_vals if provided
            if j < expected_vals.len() && committed_val != expected_vals[j] {
                return Err(PiCcsError::InvalidInput(format!(
                    "Lookup value mismatch at step {}: committed {:?}, expected {:?}",
                    j, committed_val, expected_vals[j]
                )));
            }
        }
    }

    Ok(())
}

// ============================================================================
// Prover
// ============================================================================

/// Prove lookup correctness using the Shout argument with index-bit addressing.
pub fn prove<L, Cmt, F, K>(
    _mode: FoldingMode,
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    inst: &LutInstance<Cmt, F>,
    wit: &LutWitness<F>,
    _l: &L,
) -> Result<(Vec<MeInstance<Cmt, F, K>>, Vec<Mat<F>>, ShoutProof<KElem>), PiCcsError>
where
    F: PrimeField + Into<KElem> + Copy,
    K: From<KElem> + Clone,
    Cmt: Clone,
{
    let parts = split_lut_mats(inst, wit);

    let d = inst.d;
    let ell = inst.ell;
    let _n_side = inst.n_side;
    let steps = inst.steps;
    let total_addr_bits = d * ell;

    let pow2_cycle = steps.next_power_of_two().max(1);
    let ell_cycle = pow2_cycle.trailing_zeros() as usize;

    // =========================================================================
    // Phase 1: Sample random points
    // =========================================================================
    let r_cycle = sample_ext_point(tr, b"shout/r_cycle", ell_cycle);
    let r_addr = sample_ext_point(tr, b"shout/r_addr", total_addr_bits);

    // =========================================================================
    // Phase 2: Decode witness from committed columns
    // =========================================================================

    // Decode bit columns to K
    let addr_bits: Vec<Vec<KElem>> = parts
        .addr_bit_mats
        .iter()
        .map(|m| {
            let v = ajtai_decode_vector(params, m);
            let mut out: Vec<KElem> = v.iter().map(|&x| x.into()).collect();
            out.resize(pow2_cycle, KElem::ZERO);
            out
        })
        .collect();

    // Decode has_lookup from committed column (not inferred from bits)
    let has_lookup_vec_f = ajtai_decode_vector(params, parts.has_lookup_mat);
    let has_lookup: Vec<KElem> = {
        let mut v: Vec<KElem> = has_lookup_vec_f.iter().map(|&x| x.into()).collect();
        v.resize(pow2_cycle, KElem::ZERO);
        v
    };

    // Decode val from committed column (the VM's observed lookup values)
    let val_vec_f = ajtai_decode_vector(params, parts.val_mat);
    let val_vec: Vec<KElem> = {
        let mut v: Vec<KElem> = val_vec_f.iter().map(|&x| x.into()).collect();
        v.resize(pow2_cycle, KElem::ZERO);
        v
    };

    // =========================================================================
    // Phase 3: Index Adapter Sum-Check
    // =========================================================================
    // Proves: Σ_t eq(r_cycle, t) * eq(addr_bits_t, r_addr) = adapter_claim

    let mut adapter_oracle = IndexAdapterOracle::new(&addr_bits, &r_cycle, &r_addr);
    // The expected sum is the MLE evaluation of the conceptual one-hot matrix
    // at (r_cycle, r_addr)
    let chi_cycle = build_chi_table(&r_cycle);
    let eq_addr_at_bits = crate::twist_oracle::compute_eq_from_bits(&addr_bits, &r_addr);
    let adapter_claim: KElem = chi_cycle
        .iter()
        .zip(eq_addr_at_bits.iter())
        .map(|(c, e)| *c * *e)
        .sum();

    let (adapter_rounds, adapter_chals) = run_sumcheck_prover(tr, &mut adapter_oracle, adapter_claim)
        .map_err(|e| PiCcsError::SumcheckError(format!("shout adapter: {e}")))?;

    // =========================================================================
    // Phase 4: Lookup Check Sum-Check
    // =========================================================================
    // Proves: Σ_t eq(r_cycle, t) * has_lookup(t) * eq(addr_bits_t, r_addr) * (val(t) - Table(r_addr)) = 0
    // This enforces that whenever has_lookup=1, the committed val equals Table[r_addr]

    // Compute Table(r_addr) by MLE evaluation (scalar)
    let table_at_r_addr: KElem = {
        let table_k: Vec<KElem> = inst.table.iter().map(|&x| x.into()).collect();
        let table_size = inst.table.len().next_power_of_two();
        let mut padded = table_k;
        padded.resize(table_size, KElem::ZERO);

        let ell_table = table_size.trailing_zeros() as usize;
        if r_addr.len() >= ell_table {
            mle_eval(&padded, &r_addr[..ell_table])
        } else {
            KElem::ZERO
        }
    };

    let mut lookup_oracle = ShoutLookupOracle::new(
        &addr_bits,
        has_lookup.clone(),
        val_vec.clone(),
        table_at_r_addr,
        &r_cycle,
        &r_addr,
    );

    // Expected sum: Σ_t χ_cycle(t)·has_lookup(t)·eq(addr_bits_t, r_addr)·(val(t)−Table(r_addr))
    let eq_addr_at_bits = crate::twist_oracle::compute_eq_from_bits(&addr_bits, &r_addr);
    let lookup_claim: KElem = chi_cycle
        .iter()
        .zip(has_lookup.iter())
        .zip(val_vec.iter())
        .zip(eq_addr_at_bits.iter())
        .map(|(((chi, hl), v), eq_addr)| *chi * *hl * (*v - table_at_r_addr) * *eq_addr)
        .sum();

    let (lookup_check_rounds, lookup_chals) = run_sumcheck_prover(tr, &mut lookup_oracle, lookup_claim)
        .map_err(|e| PiCcsError::SumcheckError(format!("shout lookup: {e}")))?;

    // =========================================================================
    // Phase 5: Bitness Checks
    // =========================================================================
    let r_bitness = sample_ext_point(tr, b"shout/bitness", ell_cycle);
    let mut bitness_rounds = Vec::new();

    for bits in &addr_bits {
        let mut oracle = BitnessOracle::new(bits.clone(), &r_bitness);
        let (rounds, _) = run_sumcheck_prover(tr, &mut oracle, KElem::ZERO)
            .map_err(|e| PiCcsError::SumcheckError(format!("shout bitness: {e}")))?;
        bitness_rounds.extend(rounds);
    }

    // =========================================================================
    // Phase 6: Generate ME Claims
    // =========================================================================
    let mut me_instances: Vec<MeInstance<Cmt, F, K>> = Vec::new();
    let mut me_witnesses: Vec<Mat<F>> = Vec::new();

    let d = params.d as usize;
    let y_pad = d.next_power_of_two();

    // Helper to create ME instance
    let create_me = |comm: &Cmt, decoded: &[KElem], r: &[KElem]| -> MeInstance<Cmt, F, K> {
        let mut padded = decoded.to_vec();
        padded.resize(pow2_cycle, KElem::ZERO);
        let y_eval: KElem = mle_eval(&padded, r);

        // y vector: first row holds y_eval, padded to the expected d-domain length
        let mut y_vec = vec![K::from(y_eval); 1];
        y_vec.resize(y_pad, K::from(KElem::ZERO));

        let fold_digest = {
            let mut fork = tr.fork(b"shout/me_digest");
            fork.digest32()
        };

        MeInstance {
            c: comm.clone(),
            X: Mat::from_row_major(params.d as usize, 0, vec![]),
            r: r.iter().map(|&x| K::from(x)).collect(),
            y: vec![y_vec],
            y_scalars: vec![K::from(y_eval)],
            m_in: 0,
            fold_digest,
            c_step_coords: vec![],
            u_offset: 0,
            u_len: 0,
        }
    };

    // ME claims for address bits
    for (i, bits) in addr_bits.iter().enumerate() {
        me_instances.push(create_me(&inst.comms[i], bits, &adapter_chals));
        me_witnesses.push(parts.addr_bit_mats[i].clone());
    }

    // ME claim for has_lookup
    let has_lookup_idx = total_addr_bits;
    me_instances.push(create_me(&inst.comms[has_lookup_idx], &has_lookup, &adapter_chals));
    me_witnesses.push(parts.has_lookup_mat.clone());

    // ME claim for val
    let val_idx = total_addr_bits + 1;
    me_instances.push(create_me(&inst.comms[val_idx], &val_vec, &adapter_chals));
    me_witnesses.push(parts.val_mat.clone());

    let _ = lookup_chals; // Suppress warning

    let proof = ShoutProof {
        adapter_rounds,
        lookup_check_rounds,
        bitness_rounds,
        adapter_claim,
        lookup_claim,
    };

    Ok((me_instances, me_witnesses, proof))
}

// ============================================================================
// Verifier
// ============================================================================

/// Verify lookup correctness using the Shout argument.
pub fn verify<L, Cmt, F, K>(
    _mode: FoldingMode,
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    inst: &LutInstance<Cmt, F>,
    proof: &ShoutProof<KElem>,
    _l: &L,
) -> Result<Vec<MeInstance<Cmt, F, K>>, PiCcsError>
where
    F: PrimeField,
    K: From<KElem> + Clone,
    L: SModuleHomomorphism<F, Cmt>,
    Cmt: Clone,
{
    let d = inst.d;
    let ell = inst.ell;
    let steps = inst.steps;
    let total_addr_bits = d * ell;

    let pow2_cycle = steps.next_power_of_two().max(1);
    let ell_cycle = pow2_cycle.trailing_zeros() as usize;

    // Sample the same random points
    let r_cycle = sample_ext_point(tr, b"shout/r_cycle", ell_cycle);
    let r_addr = sample_ext_point(tr, b"shout/r_addr", total_addr_bits);

    // Verify adapter sum-check using claimed sum from proof
    let adapter_degree = 1 + total_addr_bits;
    let (adapter_chals, adapter_final, adapter_ok) = neo_reductions::sumcheck::verify_sumcheck_rounds(
        tr,
        adapter_degree,
        proof.adapter_claim,
        &proof.adapter_rounds,
    );
    if !adapter_ok {
        return Err(PiCcsError::SumcheckError("shout adapter verification failed".into()));
    }

    // Verify lookup sum-check using claimed sum from proof
    let lookup_degree = 2 + total_addr_bits;
    let (lookup_chals, lookup_final, lookup_ok) = neo_reductions::sumcheck::verify_sumcheck_rounds(
        tr,
        lookup_degree,
        proof.lookup_claim,
        &proof.lookup_check_rounds,
    );
    if !lookup_ok {
        return Err(PiCcsError::SumcheckError("shout lookup verification failed".into()));
    }

    // Verify bitness checks
    let r_bitness = sample_ext_point(tr, b"shout/bitness", ell_cycle);
    for chunk in proof.bitness_rounds.chunks(ell_cycle) {
        if chunk.len() != ell_cycle {
            continue;
        }
        let (_, _, ok) = neo_reductions::sumcheck::verify_sumcheck_rounds(tr, 3, KElem::ZERO, chunk);
        if !ok {
            return Err(PiCcsError::SumcheckError("shout bitness verification failed".into()));
        }
    }

    // Reconstruct ME instances
    let mut me_instances: Vec<MeInstance<Cmt, F, K>> = Vec::new();
    let eval_point: Vec<K> = adapter_chals.iter().map(|&x| K::from(x)).collect();

    for comm in &inst.comms {
        me_instances.push(MeInstance {
            c: comm.clone(),
            X: Mat::from_row_major(params.d as usize, 0, vec![]),
            r: eval_point.clone(),
            y: vec![vec![K::from(KElem::ZERO)]],
            y_scalars: vec![K::from(KElem::ZERO)],
            m_in: 0,
            fold_digest: [0u8; 32],
            c_step_coords: vec![],
            u_offset: 0,
            u_len: 0,
        });
    }

    let _ = (
        r_cycle,
        r_addr,
        r_bitness,
        adapter_final,
        lookup_final,
        lookup_chals,
        params,
    );

    Ok(me_instances)
}

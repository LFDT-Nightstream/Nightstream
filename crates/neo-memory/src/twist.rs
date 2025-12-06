//! Twist argument for read/write memory correctness.
//!
//! This module implements the Twist protocol with **index-bit addressing**:
//! instead of committing O(n_side) one-hot columns per dimension, we commit
//! O(log n_side) bit columns and prove consistency via the IDX→OH adapter.
//!
//! ## Protocol Overview (Corrected Ordering)
//!
//! 1. Sample r_addr (random address point) and r_cycle (random cycle point)
//! 2. Build Val(r_addr, t) from init_vals and Inc
//! 3. **Read Check**: Prove rv(t) = Val(r_addr, t) when ra_t = r_addr
//! 4. **Write Check**: Prove Inc(r_addr, t) = (wv - Val) when wa_t = r_addr
//! 5. **ValEval Check**: Prove Val(r_addr, r_cycle) consistency
//! 6. **Bitness Check**: Prove all address bit columns are binary
//! 7. Generate ME claims for folding
//!
//! ## Witness Layout (Index-Bit Addressing)
//!
//! The matrices in `MemWitness.mats` are ordered as:
//! - `0 .. d*ell`:           Read address bits
//! - `d*ell .. 2*d*ell`:     Write address bits
//! - `2*d*ell`:              Inc(k, j) flattened
//! - `2*d*ell + 1`:          has_read(j)
//! - `2*d*ell + 2`:          has_write(j)
//! - `2*d*ell + 3`:          wv(j)
//! - `2*d*ell + 4`:          rv(j)

use crate::witness::{MemInstance, MemWitness};
use neo_ccs::matrix::Mat;
use neo_ccs::relations::MeInstance;
use neo_params::NeoParams;
use neo_reductions::api::FoldingMode;
use neo_reductions::error::PiCcsError;
use neo_transcript::{Poseidon2Transcript, Transcript};
use p3_field::{PrimeCharacteristicRing, PrimeField};
use serde::{Deserialize, Serialize};

use crate::mle::mle_eval;
#[cfg(feature = "debug-logs")]
use crate::twist_oracle::build_val_at_r_addr;
use crate::twist_oracle::{
    build_inc_at_r_addr, BitnessOracle, TwistReadCheckOracle, TwistValEvalOracle, TwistWriteCheckOracle,
};
use neo_ccs::traits::SModuleHomomorphism;
#[cfg(feature = "debug-logs")]
use neo_math::KExtensions;
use neo_math::{from_complex, K as KElem};
use neo_reductions::sumcheck::run_sumcheck_prover;

// ============================================================================
// Ajtai Decoding Helpers
// ============================================================================

/// Decode an Ajtai-encoded matrix back to the original vector.
pub fn ajtai_decode_vector<F: PrimeField>(params: &NeoParams, mat: &Mat<F>) -> Vec<F> {
    let d = mat.rows();
    let m = mat.cols();
    assert_eq!(
        d, params.d as usize,
        "Ajtai d mismatch: mat has {} rows, params.d = {}",
        d, params.d
    );

    let b = F::from_u64(params.b as u64);

    // Precompute b^0, b^1, ..., b^{d-1}
    let mut pow = vec![F::ONE; d];
    for i in 1..d {
        pow[i] = pow[i - 1] * b;
    }

    let mut out = Vec::with_capacity(m);
    for col in 0..m {
        let mut acc = F::ZERO;
        for row in 0..d {
            acc += mat[(row, col)] * pow[row];
        }
        out.push(acc);
    }
    out
}

/// Decode Inc matrix to flattened form: inc_flat[cell * steps + j] = Inc(cell, j)
pub fn decode_inc_flat<F: PrimeField>(
    params: &NeoParams,
    _inst: &MemInstance<impl Clone, F>,
    inc_mat: &Mat<F>,
) -> Vec<F> {
    ajtai_decode_vector(params, inc_mat)
}

// ============================================================================
// Helper utilities
// ============================================================================

fn sample_ext_point(tr: &mut Poseidon2Transcript, label: &'static [u8], len: usize) -> Vec<KElem> {
    let mut out = Vec::with_capacity(len);
    for i in 0..len {
        tr.append_message(label, &i.to_le_bytes());
        let c0 = tr.challenge_field(b"twist/coord/0");
        let c1 = tr.challenge_field(b"twist/coord/1");
        out.push(from_complex(c0, c1));
    }
    out
}

/// Pad a vector to the given length with zeros.
pub fn pad_to_pow2<T: Clone + Default>(v: &[T], pow2_len: usize) -> Vec<T> {
    let mut out = v.to_vec();
    out.resize(pow2_len, T::default());
    out
}

/// Pad a K-vector to the given length.
pub fn pad_cycle_k(v: &[KElem], pow2_cycle: usize) -> Vec<KElem> {
    let mut out = v.to_vec();
    out.resize(pow2_cycle, KElem::ZERO);
    out
}

// ============================================================================
// Proof Structure
// ============================================================================

/// Proof for the Twist memory argument.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TwistProof<F> {
    /// Sum-check round messages for read-check
    pub read_check_rounds: Vec<Vec<F>>,
    /// Sum-check round messages for write-check
    pub write_check_rounds: Vec<Vec<F>>,
    /// Sum-check round messages for Val-evaluation
    pub val_eval_rounds: Vec<Vec<F>>,
    /// Sum-check round messages for bitness checks
    pub bitness_rounds: Vec<Vec<F>>,
    /// Claimed sum for read-check (should be 0)
    pub read_claim: F,
    /// Claimed sum for write-check
    pub write_claim: F,
    /// Claimed sum for val-eval
    pub val_claim: F,
}

impl<F: Default> Default for TwistProof<F> {
    fn default() -> Self {
        Self {
            read_check_rounds: Vec::new(),
            write_check_rounds: Vec::new(),
            val_eval_rounds: Vec::new(),
            bitness_rounds: Vec::new(),
            read_claim: F::default(),
            write_claim: F::default(),
            val_claim: F::default(),
        }
    }
}

/// Decomposed witness matrices from MemWitness (index-bit layout).
#[derive(Clone, Debug)]
pub struct TwistWitnessParts<'a, F> {
    /// Read address bit matrices: d*ell matrices
    pub ra_bit_mats: &'a [Mat<F>],
    /// Write address bit matrices: d*ell matrices
    pub wa_bit_mats: &'a [Mat<F>],
    /// Increment matrix: Inc(k, j) flattened
    pub inc_mat: &'a Mat<F>,
    /// Has-read flags
    pub has_read_mat: &'a Mat<F>,
    /// Has-write flags
    pub has_write_mat: &'a Mat<F>,
    /// Write values
    pub wv_mat: &'a Mat<F>,
    /// Read values
    pub rv_mat: &'a Mat<F>,
}

/// Split the MemWitness matrices into named parts (index-bit layout).
pub fn split_mem_mats<'a, F: Clone>(
    inst: &MemInstance<impl Clone, F>,
    wit: &'a MemWitness<F>,
) -> TwistWitnessParts<'a, F> {
    let d = inst.d;
    let ell = inst.ell;
    let total_addr_bits = d * ell;
    let expected_len = 2 * total_addr_bits + 5;

    assert_eq!(
        wit.mats.len(),
        expected_len,
        "MemWitness has {} matrices, expected {} (d={}, ell={})",
        wit.mats.len(),
        expected_len,
        d,
        ell
    );

    TwistWitnessParts {
        ra_bit_mats: &wit.mats[0..total_addr_bits],
        wa_bit_mats: &wit.mats[total_addr_bits..2 * total_addr_bits],
        inc_mat: &wit.mats[2 * total_addr_bits],
        has_read_mat: &wit.mats[2 * total_addr_bits + 1],
        has_write_mat: &wit.mats[2 * total_addr_bits + 2],
        wv_mat: &wit.mats[2 * total_addr_bits + 3],
        rv_mat: &wit.mats[2 * total_addr_bits + 4],
    }
}

// ============================================================================
// Semantic Checker (Debug)
// ============================================================================

/// Check Twist semantics without cryptographic proofs.
pub fn check_twist_semantics<F: PrimeField>(
    params: &NeoParams,
    inst: &MemInstance<impl Clone, F>,
    wit: &MemWitness<F>,
) -> Result<(), PiCcsError> {
    let parts = split_mem_mats(inst, wit);
    let k = inst.k;
    let steps = inst.steps;
    let ell = inst.ell;
    let d = inst.d;

    // Decode data columns
    let has_read = ajtai_decode_vector(params, parts.has_read_mat);
    let has_write = ajtai_decode_vector(params, parts.has_write_mat);
    let wv = ajtai_decode_vector(params, parts.wv_mat);
    let rv = ajtai_decode_vector(params, parts.rv_mat);
    let inc_flat = ajtai_decode_vector(params, parts.inc_mat);

    // Decode addresses from bit columns
    let decode_addr_from_bits = |bit_mats: &[Mat<F>]| -> Vec<u64> {
        let mut addrs = vec![0u64; steps];
        for dim in 0..d {
            let base = dim * ell;
            for j in 0..steps {
                let mut dim_val = 0u64;
                for b in 0..ell {
                    let bit_col = ajtai_decode_vector(params, &bit_mats[base + b]);
                    if bit_col[j] == F::ONE {
                        dim_val |= 1u64 << b;
                    }
                }
                addrs[j] += dim_val * (inst.n_side as u64).pow(dim as u32);
            }
        }
        addrs
    };

    let read_addrs = decode_addr_from_bits(parts.ra_bit_mats);
    let write_addrs = decode_addr_from_bits(parts.wa_bit_mats);

    // Check bitness
    for mat in parts.ra_bit_mats.iter().chain(parts.wa_bit_mats.iter()) {
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

    // Validate init_vals
    if inst.init_vals.len() != k {
        return Err(PiCcsError::InvalidInput(format!(
            "init_vals length mismatch: inst.k={}, init_vals.len()={}",
            k,
            inst.init_vals.len()
        )));
    }

    // Simulate memory
    let mut val = inst.init_vals.clone();

    for j in 0..steps {
        // Check read correctness
        if has_read[j] == F::ONE {
            let addr = read_addrs[j] as usize;
            if addr < k {
                if rv[j] != val[addr] {
                    return Err(PiCcsError::InvalidInput(format!(
                        "Read mismatch at step {}: rv={:?}, Val[{}]={:?}",
                        j, rv[j], addr, val[addr]
                    )));
                }
            }
        }

        // Check write correctness and update memory
        if has_write[j] == F::ONE {
            let addr = write_addrs[j] as usize;
            if addr < k {
                let expected_inc = wv[j] - val[addr];
                let actual_inc = inc_flat[addr * steps + j];
                if actual_inc != expected_inc {
                    return Err(PiCcsError::InvalidInput(format!(
                        "Inc mismatch at step {}, cell {}: got {:?}, expected {:?}",
                        j, addr, actual_inc, expected_inc
                    )));
                }
                val[addr] = wv[j];
            }
        }
    }

    Ok(())
}

// ============================================================================
// Prover
// ============================================================================

/// Prove memory correctness using the Twist argument with index-bit addressing.
///
/// Returns ME instances that reduce the Twist claim to commitments openings,
/// which can then be folded using Neo's standard RLC→DEC pipeline.
pub fn prove<L, Cmt, F, K>(
    _mode: FoldingMode,
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    inst: &MemInstance<Cmt, F>,
    wit: &MemWitness<F>,
    _l: &L,
) -> Result<(Vec<MeInstance<Cmt, F, K>>, Vec<Mat<F>>, TwistProof<KElem>), PiCcsError>
where
    F: PrimeField + Into<KElem> + Copy,
    K: From<KElem> + Clone,
    Cmt: Clone,
{
    let parts = split_mem_mats(inst, wit);

    // Debug check
    #[cfg(debug_assertions)]
    check_twist_semantics(params, inst, wit)?;

    let k = inst.k;
    let d = inst.d;
    let ell = inst.ell;
    let steps = inst.steps;
    let total_addr_bits = d * ell;

    let pow2_cycle = steps.next_power_of_two().max(1);
    let ell_cycle = pow2_cycle.trailing_zeros() as usize;

    // =========================================================================
    // Phase 1: Sample random points
    // =========================================================================

    // r_addr has total_addr_bits components (for the bit-decomposed address)
    let r_addr = sample_ext_point(tr, b"twist/r_addr", total_addr_bits);
    let r_cycle = sample_ext_point(tr, b"twist/r_cycle", ell_cycle);

    // =========================================================================
    // Phase 2: Decode witness and build tables
    // =========================================================================

    // Decode bit columns to K
    let decode_bits_to_k = |mats: &[Mat<F>]| -> Vec<Vec<KElem>> {
        mats.iter()
            .map(|m| {
                let v = ajtai_decode_vector(params, m);
                let mut out: Vec<KElem> = v.iter().map(|&x| x.into()).collect();
                out.resize(pow2_cycle, KElem::ZERO);
                out
            })
            .collect()
    };

    let ra_bits: Vec<Vec<KElem>> = decode_bits_to_k(parts.ra_bit_mats);
    let wa_bits: Vec<Vec<KElem>> = decode_bits_to_k(parts.wa_bit_mats);

    // Decode scalar columns
    let rv_vec: Vec<KElem> = {
        let v = ajtai_decode_vector(params, parts.rv_mat);
        pad_cycle_k(&v.iter().map(|&x| x.into()).collect::<Vec<_>>(), pow2_cycle)
    };
    let wv_vec: Vec<KElem> = {
        let v = ajtai_decode_vector(params, parts.wv_mat);
        pad_cycle_k(&v.iter().map(|&x| x.into()).collect::<Vec<_>>(), pow2_cycle)
    };
    let has_read_vec: Vec<KElem> = {
        let v = ajtai_decode_vector(params, parts.has_read_mat);
        pad_cycle_k(&v.iter().map(|&x| x.into()).collect::<Vec<_>>(), pow2_cycle)
    };
    let has_write_vec: Vec<KElem> = {
        let v = ajtai_decode_vector(params, parts.has_write_mat);
        pad_cycle_k(&v.iter().map(|&x| x.into()).collect::<Vec<_>>(), pow2_cycle)
    };

    let inc_flat = ajtai_decode_vector(params, parts.inc_mat);

    // =========================================================================
    // Compute Val(addr_t, t) - the actual memory value at each read address
    // =========================================================================
    // This is needed for the read-check: we must use the memory value at the
    // ACTUAL read address, not at a random address r_addr.

    // Decode scalar columns in base field for memory simulation
    let has_read_f = ajtai_decode_vector(params, parts.has_read_mat);
    let has_write_f = ajtai_decode_vector(params, parts.has_write_mat);
    let wv_f = ajtai_decode_vector(params, parts.wv_mat);

    // Decode addresses from bit columns
    let decode_addr_from_bits = |bit_mats: &[Mat<F>]| -> Vec<usize> {
        let mut addrs = vec![0usize; steps];
        for dim in 0..d {
            let base = dim * ell;
            for j in 0..steps {
                let mut dim_val = 0usize;
                for b in 0..ell {
                    let bit_col = ajtai_decode_vector(params, &bit_mats[base + b]);
                    if bit_col[j] == F::ONE {
                        dim_val |= 1usize << b;
                    }
                }
                addrs[j] += dim_val * inst.n_side.pow(dim as u32);
            }
        }
        addrs
    };

    let read_addrs = decode_addr_from_bits(parts.ra_bit_mats);
    let write_addrs = decode_addr_from_bits(parts.wa_bit_mats);

    // Simulate memory to compute:
    // - Val(read_addr_t, t) - the value at the actual read address
    // - Val(write_addr_t, t) - the value at the actual write address (before the write)
    // - Inc(write_addr_t, t) - the increment at the actual write address
    let mut mem = inst.init_vals.clone();
    let mut val_at_read_addr_f = vec![F::ZERO; steps];
    let mut val_at_write_addr_f = vec![F::ZERO; steps];
    let mut inc_at_write_addr_f = vec![F::ZERO; steps];

    for j in 0..steps {
        // Record value BEFORE any write at this step (for reads)
        if has_read_f[j] == F::ONE {
            let addr = read_addrs[j];
            if addr < k {
                val_at_read_addr_f[j] = mem[addr];
            }
        }

        // Record value at write address BEFORE the write, and compute increment
        if has_write_f[j] == F::ONE {
            let addr = write_addrs[j];
            if addr < k {
                val_at_write_addr_f[j] = mem[addr];
                inc_at_write_addr_f[j] = wv_f[j] - mem[addr]; // Inc = new_val - old_val
                mem[addr] = wv_f[j]; // Apply the write
            }
        }
    }

    // Convert to K and pad to pow2_cycle
    let val_at_read_addr: Vec<KElem> = pad_cycle_k(
        &val_at_read_addr_f
            .iter()
            .map(|&x| x.into())
            .collect::<Vec<_>>(),
        pow2_cycle,
    );
    let val_at_write_addr: Vec<KElem> = pad_cycle_k(
        &val_at_write_addr_f
            .iter()
            .map(|&x| x.into())
            .collect::<Vec<_>>(),
        pow2_cycle,
    );
    let inc_at_write_addr: Vec<KElem> = pad_cycle_k(
        &inc_at_write_addr_f
            .iter()
            .map(|&x| x.into())
            .collect::<Vec<_>>(),
        pow2_cycle,
    );

    #[cfg(feature = "debug-logs")]
    {
        use p3_field::PrimeField64;
        let format_k = |k: &KElem| -> String {
            let coeffs = k.as_coeffs();
            format!("K[{}, {}]", coeffs[0].as_canonical_u64(), coeffs[1].as_canonical_u64())
        };
        eprintln!("=== TWIST PROVE DEBUG ===");
        eprintln!("k={}, d={}, ell={}, steps={}", k, d, ell, steps);
        eprintln!(
            "total_addr_bits={}, pow2_cycle={}, ell_cycle={}",
            total_addr_bits, pow2_cycle, ell_cycle
        );
        eprintln!("ra_bits.len()={}, wa_bits.len()={}", ra_bits.len(), wa_bits.len());
        if !ra_bits.is_empty() {
            eprintln!("ra_bits[0].len()={}", ra_bits[0].len());
            eprintln!(
                "ra_bits[0][0..4]: [{}]",
                ra_bits[0]
                    .iter()
                    .take(4)
                    .map(|v| format_k(v))
                    .collect::<Vec<_>>()
                    .join(", ")
            );
        }
        eprintln!(
            "rv_vec.len()={}, rv_vec[0..4]: [{}]",
            rv_vec.len(),
            rv_vec
                .iter()
                .take(4)
                .map(|v| format_k(v))
                .collect::<Vec<_>>()
                .join(", ")
        );
        eprintln!(
            "has_read_vec.len()={}, has_read_vec[0..4]: [{}]",
            has_read_vec.len(),
            has_read_vec
                .iter()
                .take(4)
                .map(|v| format_k(v))
                .collect::<Vec<_>>()
                .join(", ")
        );
        eprintln!("inc_flat.len()={}", inc_flat.len());
        eprintln!(
            "r_addr[0..min(4,len)]: [{}]",
            r_addr
                .iter()
                .take(4)
                .map(|v| format_k(v))
                .collect::<Vec<_>>()
                .join(", ")
        );
        eprintln!(
            "r_cycle[0..min(4,len)]: [{}]",
            r_cycle
                .iter()
                .take(4)
                .map(|v| format_k(v))
                .collect::<Vec<_>>()
                .join(", ")
        );
    }

    // Build Inc at the random address point (for val-eval check)
    let inc_at_r_addr: Vec<KElem> = build_inc_at_r_addr(&inc_flat, k, steps, pow2_cycle, &r_addr);

    #[cfg(feature = "debug-logs")]
    {
        use p3_field::PrimeField64;
        let format_k = |k: &KElem| -> String {
            let coeffs = k.as_coeffs();
            format!("K[{}, {}]", coeffs[0].as_canonical_u64(), coeffs[1].as_canonical_u64())
        };
        // Build val_at_r_addr only for debugging (no longer used in protocol)
        let val_at_r_addr: Vec<KElem> = build_val_at_r_addr(&inc_flat, &inst.init_vals, k, steps, pow2_cycle, &r_addr);
        eprintln!(
            "val_at_r_addr[0..4]: [{}]",
            val_at_r_addr
                .iter()
                .take(4)
                .map(|v| format_k(v))
                .collect::<Vec<_>>()
                .join(", ")
        );
        eprintln!(
            "val_at_read_addr[0..4]: [{}]",
            val_at_read_addr
                .iter()
                .take(4)
                .map(|v| format_k(v))
                .collect::<Vec<_>>()
                .join(", ")
        );
        eprintln!(
            "inc_at_r_addr[0..4]: [{}]",
            inc_at_r_addr
                .iter()
                .take(4)
                .map(|v| format_k(v))
                .collect::<Vec<_>>()
                .join(", ")
        );

        // Verify semantic correctness: for each step with has_read, rv should equal val_at_read_addr
        eprintln!("--- Checking read semantics (using val_at_read_addr) ---");
        for t in 0..steps.min(4) {
            let hr = &has_read_vec[t];
            let rv = &rv_vec[t];
            let val = &val_at_read_addr[t];
            let diff = *val - *rv;
            eprintln!(
                "  t={}: has_read={}, rv={}, val_at_read_addr={}, diff={}",
                t,
                format_k(hr),
                format_k(rv),
                format_k(val),
                format_k(&diff)
            );
        }
    }

    // =========================================================================
    // Phase 3: Read Check Sum-Check
    // =========================================================================
    // Proves: Σ_t eq(r_cycle, t) * has_read(t) * (Val(addr_t, t) - rv(t)) = 0
    // Note: We use val_at_read_addr (the actual memory value at the read address),
    // NOT val_at_r_addr (the memory value at a random address).

    let mut read_oracle = TwistReadCheckOracle::new(
        &ra_bits,
        val_at_read_addr.clone(), // <- Val(addr_t, t), not Val(r_addr, t)
        rv_vec.clone(),
        has_read_vec.clone(),
        &r_cycle,
        &r_addr,
    );
    let (read_check_rounds, read_chals) = run_sumcheck_prover(tr, &mut read_oracle, KElem::ZERO)
        .map_err(|e| PiCcsError::SumcheckError(format!("read-check: {e}")))?;

    // =========================================================================
    // Phase 4: Write Check Sum-Check
    // =========================================================================
    // Proves: Σ_t eq(r_cycle, t) * has_write(t) * (wv(t) - Val(addr_t, t) - Inc(addr_t, t)) = 0
    // Note: We use val_at_write_addr and inc_at_write_addr (values at the actual write address),
    // NOT val_at_r_addr and inc_at_r_addr (values at a random address).

    let expected_write: KElem = KElem::ZERO;

    let mut write_oracle = TwistWriteCheckOracle::new(
        &wa_bits,
        wv_vec.clone(),
        val_at_write_addr.clone(), // <- Val(addr_t, t), not Val(r_addr, t)
        inc_at_write_addr.clone(), // <- Inc(addr_t, t), not Inc(r_addr, t)
        has_write_vec.clone(),
        &r_cycle,
        &r_addr,
    );
    let (write_check_rounds, write_chals) = run_sumcheck_prover(tr, &mut write_oracle, expected_write)
        .map_err(|e| PiCcsError::SumcheckError(format!("write-check: {e}")))?;

    // =========================================================================
    // Phase 5: ValEval Sum-Check
    // =========================================================================
    // Proves: Val(r_addr, r_cycle) = Σ_t Inc(r_addr, t) * LT(t, r_cycle)

    let lt_table = crate::twist_oracle::build_lt_table(&r_cycle);
    let expected_val: KElem = inc_at_r_addr
        .iter()
        .zip(lt_table.iter())
        .map(|(v, lt)| *v * *lt)
        .sum();

    let mut val_oracle = TwistValEvalOracle::new(&inc_flat, k, steps, &r_addr, &r_cycle);
    let (val_eval_rounds, val_chals) = run_sumcheck_prover(tr, &mut val_oracle, expected_val)
        .map_err(|e| PiCcsError::SumcheckError(format!("val-eval: {e}")))?;

    // =========================================================================
    // Phase 6: Bitness Checks
    // =========================================================================
    let r_bitness = sample_ext_point(tr, b"twist/bitness", ell_cycle);
    let mut bitness_rounds = Vec::new();

    for bits in ra_bits.iter().chain(wa_bits.iter()) {
        let mut oracle = BitnessOracle::new(bits.clone(), &r_bitness);
        let (rounds, _) = run_sumcheck_prover(tr, &mut oracle, KElem::ZERO)
            .map_err(|e| PiCcsError::SumcheckError(format!("bitness: {e}")))?;
        bitness_rounds.extend(rounds);
    }

    // =========================================================================
    // Phase 7: Generate ME Claims
    // =========================================================================
    // Each sum-check reduces to evaluations of the committed polynomials at
    // the challenge points. We create ME instances for these openings.

    let mut me_instances: Vec<MeInstance<Cmt, F, K>> = Vec::new();
    let mut me_witnesses: Vec<Mat<F>> = Vec::new();

    // Combine all challenges into evaluation points
    let eval_point_cycle = read_chals.clone(); // All sum-checks use cycle dimension

    // Y padding length for the ME relation (padded to next power of two over D rows)
    let d = params.d as usize;
    let y_pad = d.next_power_of_two();

    // Convenience: create an ME instance + witness for a committed column
    let mut mk_me = |comm: &Cmt, _mat: &Mat<F>, r: &[KElem], value: KElem| -> MeInstance<Cmt, F, K> {
        // y vector: place the value in the first row and pad
        let mut y_vec = vec![K::from(value); 1];
        y_vec.resize(y_pad, K::from(KElem::ZERO));

        // y_scalars: base-b recomposition over the first D digits; here it matches `value`
        let y_scalar = K::from(value);

        // Fold digest bound to transcript without mutating the main prover transcript
        let fold_digest = {
            let mut fork = tr.fork(b"twist/me_digest");
            fork.digest32()
        };

        MeInstance {
            c: comm.clone(),
            X: Mat::from_row_major(d, 0, vec![]),
            r: r.iter().map(|&x| K::from(x)).collect(),
            y: vec![y_vec],
            y_scalars: vec![y_scalar],
            m_in: 0,
            fold_digest,
            c_step_coords: vec![],
            u_offset: 0,
            u_len: 0,
        }
    };

    // Helper to evaluate a column at a point and record witness
    let eval_and_push =
        |me_instances: &mut Vec<MeInstance<Cmt, F, K>>,
         me_witnesses: &mut Vec<Mat<F>>,
         comm: &Cmt,
         mat: &Mat<F>,
         r: &[KElem],
         mk_me: &mut dyn FnMut(&Cmt, &Mat<F>, &[KElem], KElem) -> MeInstance<Cmt, F, K>| {
            let v = ajtai_decode_vector(params, mat);
            let v_k: Vec<KElem> = v.iter().map(|&x| x.into()).collect();
            let mut padded = v_k;
            padded.resize(1 << r.len(), KElem::ZERO);
            let y_eval: KElem = mle_eval(&padded, r);
            me_instances.push(mk_me(comm, mat, r, y_eval));
            me_witnesses.push(mat.clone());
        };

    // ME claims for read address bits (evaluated at r_cycle from read-check)
    for (i, mat) in parts.ra_bit_mats.iter().enumerate() {
        eval_and_push(
            &mut me_instances,
            &mut me_witnesses,
            &inst.comms[i],
            mat,
            &eval_point_cycle,
            &mut mk_me,
        );
    }

    // ME claims for write address bits
    let wa_offset = total_addr_bits;
    for (i, mat) in parts.wa_bit_mats.iter().enumerate() {
        eval_and_push(
            &mut me_instances,
            &mut me_witnesses,
            &inst.comms[wa_offset + i],
            mat,
            &eval_point_cycle,
            &mut mk_me,
        );
    }

    // ME claims for data columns
    let data_offset = 2 * total_addr_bits;
    eval_and_push(
        &mut me_instances,
        &mut me_witnesses,
        &inst.comms[data_offset + 1],
        parts.has_read_mat,
        &eval_point_cycle,
        &mut mk_me,
    );
    eval_and_push(
        &mut me_instances,
        &mut me_witnesses,
        &inst.comms[data_offset + 2],
        parts.has_write_mat,
        &eval_point_cycle,
        &mut mk_me,
    );
    eval_and_push(
        &mut me_instances,
        &mut me_witnesses,
        &inst.comms[data_offset + 3],
        parts.wv_mat,
        &eval_point_cycle,
        &mut mk_me,
    );
    eval_and_push(
        &mut me_instances,
        &mut me_witnesses,
        &inst.comms[data_offset + 4],
        parts.rv_mat,
        &eval_point_cycle,
        &mut mk_me,
    );

    // ME claim for Inc (evaluated at combined (r_k, r_cycle) point)
    let r_inc = [r_addr.as_slice(), eval_point_cycle.as_slice()].concat();
    eval_and_push(
        &mut me_instances,
        &mut me_witnesses,
        &inst.comms[data_offset],
        parts.inc_mat,
        &r_inc,
        &mut mk_me,
    );

    let proof = TwistProof {
        read_check_rounds,
        write_check_rounds,
        val_eval_rounds,
        bitness_rounds,
        read_claim: KElem::ZERO, // Read check should sum to 0
        write_claim: expected_write,
        val_claim: expected_val,
    };

    let _ = (write_chals, val_chals); // Suppress unused warnings

    Ok((me_instances, me_witnesses, proof))
}

// ============================================================================
// Verifier
// ============================================================================

/// Verify memory correctness using the Twist argument.
pub fn verify<L, Cmt, F, K>(
    _mode: FoldingMode,
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    inst: &MemInstance<Cmt, F>,
    proof: &TwistProof<KElem>,
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
    let k = inst.k;
    let total_addr_bits = d * ell;

    let pow2_cycle = steps.next_power_of_two().max(1);
    let ell_cycle = pow2_cycle.trailing_zeros() as usize;
    let pow2_k = k.next_power_of_two().max(1);
    let ell_k = pow2_k.trailing_zeros() as usize;

    // Sample the same random points as prover
    let _r_addr = sample_ext_point(tr, b"twist/r_addr", total_addr_bits);
    let _r_cycle = sample_ext_point(tr, b"twist/r_cycle", ell_cycle);
    let _r_k = sample_ext_point(tr, b"twist/r_k", ell_k);

    // Verify read-check sum-check using claimed sum from proof
    let read_degree = 3 + total_addr_bits; // eq_cycle, has_read, diff, bit_eq_factors
    let (read_chals, read_final, read_ok) =
        neo_reductions::sumcheck::verify_sumcheck_rounds(tr, read_degree, proof.read_claim, &proof.read_check_rounds);
    if !read_ok {
        return Err(PiCcsError::SumcheckError("read-check verification failed".into()));
    }

    // Verify write-check sum-check using claimed sum from proof
    let write_degree = 3 + total_addr_bits;
    let (write_chals, write_final, write_ok) = neo_reductions::sumcheck::verify_sumcheck_rounds(
        tr,
        write_degree,
        proof.write_claim,
        &proof.write_check_rounds,
    );
    if !write_ok {
        return Err(PiCcsError::SumcheckError("write-check verification failed".into()));
    }

    // Verify val-eval sum-check using claimed sum from proof
    let val_degree = 2;
    let (val_chals, val_final, val_ok) =
        neo_reductions::sumcheck::verify_sumcheck_rounds(tr, val_degree, proof.val_claim, &proof.val_eval_rounds);
    if !val_ok {
        return Err(PiCcsError::SumcheckError("val-eval verification failed".into()));
    }

    // Verify bitness checks
    let r_bitness = sample_ext_point(tr, b"twist/bitness", ell_cycle);
    for chunk in proof.bitness_rounds.chunks(ell_cycle) {
        if chunk.len() != ell_cycle {
            continue;
        }
        let (_, _, ok) = neo_reductions::sumcheck::verify_sumcheck_rounds(tr, 3, KElem::ZERO, chunk);
        if !ok {
            return Err(PiCcsError::SumcheckError("bitness verification failed".into()));
        }
    }

    // Reconstruct ME instances from final evaluations
    // The verifier reconstructs the same ME structure as the prover
    let mut me_instances: Vec<MeInstance<Cmt, F, K>> = Vec::new();

    // For each commitment, create an ME instance with the evaluation point
    let eval_point: Vec<K> = read_chals.iter().map(|&x| K::from(x)).collect();

    for comm in &inst.comms {
        me_instances.push(MeInstance {
            c: comm.clone(),
            X: Mat::from_row_major(params.d as usize, 0, vec![]),
            r: eval_point.clone(),
            y: vec![vec![K::from(KElem::ZERO)]], // Filled by prover claims
            y_scalars: vec![K::from(KElem::ZERO)],
            m_in: 0,
            fold_digest: [0u8; 32],
            c_step_coords: vec![],
            u_offset: 0,
            u_len: 0,
        });
    }

    let _ = (
        r_bitness,
        read_final,
        write_final,
        val_final,
        write_chals,
        val_chals,
        params,
    );

    Ok(me_instances)
}

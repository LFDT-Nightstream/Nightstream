//! Π_RLC verifier (Random Linear Combination) per Neo §4.5.
//! Verifies c, X, and each y_j are the strong‑set linear combination of inputs.
//!
//! This version **fully checks y-vectors** (Vec<K>) with a zero-initialized
//! accumulator to avoid relying on `Vec<K>: Add + Default`.

use crate::strong_set::{StrongSamplingSet, VerificationError, ds};
use crate::transcript::FoldTranscript;
use neo_math::K;
use p3_field::PrimeCharacteristicRing;

/// Verify Π_RLC (Random Linear Combination) relations per Neo §4.5.
///
/// * `c_out` ?= Σ ρ_i · c_i
/// * `x_out` ?= Σ ρ_i · X_i
/// * For each `j`: `y_out[j]` ?= Σ ρ_i · y_in[i][j]
///
/// Uses explicit closures for zero/add operations to avoid trait bound issues.
pub fn verify_linear_rlc<C, XM, S, FLeftC, FLeftX, FLeftY, FZeroC, FAddC, FZeroX, FAddX>(
    transcript: &mut FoldTranscript,
    strong_set: &StrongSamplingSet<S>,
    // Output
    c_out: &C,
    x_out: &XM,
    ys_out: &[Vec<K>],
    // Inputs
    c_in: &[C],
    x_in: &[XM],
    ys_in: &[Vec<Vec<K>>], // shape: [k+1][t]
    // Left actions
    left_c: FLeftC,
    left_x: FLeftX,
    left_y: FLeftY,
    // Zero and add operations (avoids problematic trait bounds)
    zero_c: FZeroC,
    add_c: FAddC,
    zero_x: FZeroX,
    add_x: FAddX,
) -> Result<(), VerificationError>
where
    S: Clone,
    C: Clone + PartialEq,
    XM: Clone + PartialEq,
    FLeftC: Fn(&S, &C) -> C,
    FLeftX: Fn(&S, &XM) -> XM,
    FLeftY: Fn(&S, &Vec<K>) -> Vec<K>,
    FZeroC: Fn() -> C,
    FAddC: Fn(C, C) -> C,
    FZeroX: Fn() -> XM,
    FAddX: Fn(XM, XM) -> XM,
{
    let k1 = c_in.len();
    if k1 == 0 || x_in.len() != k1 || ys_in.len() != k1 {
        return Err(VerificationError::LinearCommit(0));
    }

    // Derive ρ_i via FS (domain-separated)
    let rhos: Vec<S> = strong_set.sample_k(transcript, ds::RLC, k1);

    // Commitments
    let mut c_acc = zero_c();
    for (rho, ci) in rhos.iter().zip(c_in.iter()) {
        c_acc = add_c(c_acc, left_c(rho, ci));
    }
    if &c_acc != c_out {
        return Err(VerificationError::LinearCommit(0));
    }

    // X matrices
    let mut x_acc = zero_x();
    for (rho, xi) in rhos.iter().zip(x_in.iter()) {
        x_acc = add_x(x_acc, left_x(rho, xi));
    }
    if &x_acc != x_out {
        return Err(VerificationError::LinearX(0, 0));
    }

    // y-vectors (per j) 
    let t = ys_out.len();
    // SECURITY: Verify all input y vectors have consistent shape before processing
    if ys_in.iter().any(|ys| ys.len() != t) {
        return Err(VerificationError::LinearY(0, 0));
    }
    for j in 0..t {
        let out_len = ys_out[j].len();
        let mut acc = vec![K::ZERO; out_len];
        for (rho, ys_i) in rhos.iter().zip(ys_in.iter()) {
            let rotated = left_y(rho, &ys_i[j]);
            if rotated.len() != out_len {
                return Err(VerificationError::LinearY(j, out_len));
            }
            for idx in 0..out_len {
                acc[idx] += rotated[idx];
            }
        }
        if acc != ys_out[j] {
            let bad_idx = acc
                .iter()
                .zip(&ys_out[j])
                .position(|(a, b)| a != b)
                .unwrap_or(0);
            return Err(VerificationError::LinearY(j, bad_idx));
        }
    }

    Ok(())
}

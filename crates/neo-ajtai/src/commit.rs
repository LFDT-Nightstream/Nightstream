use p3_goldilocks::Goldilocks as Fq;
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use rand::{RngCore, CryptoRng};
use crate::types::{PP, Commitment};
use crate::error::{AjtaiError, AjtaiResult};


/// Bring in ring & S-action APIs from neo-math.
use neo_math::ring::{Rq as RqEl, cf_inv as cf_unmap, cf};
use neo_math::s_action::SAction;

/// Sample a uniform element from F_q using rejection sampling to avoid bias.
#[inline]
fn sample_uniform_fq<R: RngCore + CryptoRng>(rng: &mut R) -> Fq {
    // Rejection sampling: draw u64; accept if < q; otherwise redraw.
    const Q: u64 = <Fq as PrimeField64>::ORDER_U64; // 2^64 - 2^32 + 1
    loop {
        let x = rng.next_u64();
        if x < Q { return Fq::from_u64(x); }
    }
}

/// MUST: Setup(κ,m) → sample M ← R_q^{κ×m} uniformly (Def. 9).
pub fn setup<R: RngCore + CryptoRng>(rng: &mut R, d: usize, kappa: usize, m: usize) -> AjtaiResult<PP<RqEl>> {
    // Ensure d matches the fixed ring dimension from neo-math
    if d != neo_math::ring::D {
        return Err(AjtaiError::InvalidDimensions("d parameter must match ring dimension D"));
    }
    let mut rows = Vec::with_capacity(kappa);
    for _ in 0..kappa {
        let mut row = Vec::with_capacity(m);
        for _ in 0..m {
            // sample ring element uniformly by sampling d random coefficients in F_q and mapping via cf^{-1}
            let coeffs_vec: Vec<Fq> = (0..neo_math::ring::D)
                .map(|_| sample_uniform_fq(rng))
                .collect();
            let coeffs: [Fq; neo_math::ring::D] = coeffs_vec
                .try_into()
                .map_err(|_| AjtaiError::InvalidDimensions("Failed to create coefficient array"))?;
            row.push(cf_unmap(coeffs));
        }
        rows.push(row);
    }
    Ok(PP { kappa, m, d, m_rows: rows })
}

/// Check if ALL digits are exactly in {-1, 0, 1} for safe pay-per-bit optimization
/// This is a strict precondition to avoid the correctness bug where non-{-1,0,1} digits
/// would be incorrectly treated as -1.
#[allow(non_snake_case, dead_code)]
fn all_digits_pm1_or_zero(Z: &[Fq]) -> bool {
    let m1 = Fq::ZERO - Fq::ONE;
    Z.iter().all(|&d| d == Fq::ZERO || d == Fq::ONE || d == m1)
}

// Helper functions for pay-per-bit optimization
#[inline]
#[allow(dead_code)]
fn add_col(acc: &mut [Fq], col: &[Fq]) {
    for (a, &x) in acc.iter_mut().zip(col) { *a += x; }
}

#[inline] 
#[allow(dead_code)]
fn sub_col(acc: &mut [Fq], col: &[Fq]) {
    for (a, &x) in acc.iter_mut().zip(col) { *a -= x; }
}

/// Pay-per-bit optimized commit for digits strictly in {-1, 0, 1}
/// Uses the rotation matrix identity: cf(a⋅b) = rot(a)⋅cf(b) = ∑_t z_t ⋅ (column t of rot(a))
/// Only adds/subtracts pre-rotated columns when z_t ≠ 0
/// PRECONDITION: ALL digits in Z must be in {-1, 0, 1}
#[allow(non_snake_case, dead_code)]
fn commit_pay_per_bit_pm1(pp: &PP<RqEl>, Z: &[Fq]) -> Commitment {
    let d = pp.d; let m = pp.m; let kappa = pp.kappa;
    let mut c = Commitment::zeros(d, kappa);

    // For each Ajtai row i and message column j
    for i in 0..kappa {
        let acc_i = c.col_mut(i);
        for j in 0..m {
            let v = &Z[j*d..(j+1)*d];  // digits for column j
            
            let m1 = Fq::ZERO - Fq::ONE;
            
            // t = 0: only compute cf(a_ij) if needed
            if v[0] != Fq::ZERO {
                let col0 = cf(pp.m_rows[i][j]);
                if v[0] == Fq::ONE {
                    add_col(acc_i, &col0);
                } else if v[0] == m1 {
                    sub_col(acc_i, &col0);  
                } else {
                    debug_assert!(false, "PRECONDITION VIOLATED: digits must be in {{-1,0,1}}");
                }
            }
            
            // t = 1..d-1: JUMP DIRECTLY to non-zero positions (true pay-per-bit!)
            // This achieves O(#nonzeros·d) instead of O(d²) complexity
            for t in 1..d {
                let vt = v[t];
                if vt != Fq::ZERO {
                    // Jump directly to X^t without iterating through intermediate powers
                    let col = cf(pp.m_rows[i][j].mul_by_monomial(t));
                    if vt == Fq::ONE {
                        add_col(acc_i, &col);
                    } else if vt == m1 {
                        sub_col(acc_i, &col);
                    } else {
                        debug_assert!(false, "PRECONDITION VIOLATED: digits must be in {{-1,0,1}}");
                    }
                }
                // CRITICAL: When vt == 0, we do NO WORK - this is the pay-per-bit savings!
            }
        }
    }
    c
}

/// MUST: Commit(pp, Z) = cf(M · cf^{-1}(Z)) as c ∈ F_q^{d×κ}.  S-homomorphic over S by construction.
/// Automatically chooses between dense O(d²) and sparse O(#nonzeros × d) algorithms based on digit sparsity.
#[allow(non_snake_case)]
pub fn commit(pp: &PP<RqEl>, Z: &[Fq]) -> Commitment {
    // Z is d×m (column-major by (col*d + row)), output c is d×kappa (column-major)
    let d = pp.d; let m = pp.m; let _kappa = pp.kappa;
    assert_eq!(Z.len(), d*m, "Z must be d×m");

    // Choose algorithm: sparse path ONLY if ALL digits are in {-1, 0, 1}
    // This prevents the correctness bug where digits like 2, 3, etc. would be treated as -1
    #[cfg(feature = "variable_time_commit")]
    {
        if all_digits_pm1_or_zero(Z) {
            return commit_pay_per_bit_pm1(pp, Z);
        }
    }
    commit_dense(pp, Z)
}

/// Dense commit implementation (original algorithm)
/// Kept as fallback when digits are not sparse
#[allow(non_snake_case)]
fn commit_dense(pp: &PP<RqEl>, Z: &[Fq]) -> Commitment {
    let d = pp.d; let m = pp.m; let kappa = pp.kappa;

    // Pre-extract columns of Z (digits per column)
    let cols: Vec<&[Fq]> = (0..m).map(|j| &Z[j*d .. (j+1)*d]).collect();

    let mut c = Commitment::zeros(d, kappa);

    // For each Ajtai row i, compute Σ_j S-action(a_ij) · Z_col_j into c_col_i.
    for i in 0..kappa {
        let acc = c.col_mut(i);
        #[allow(clippy::needless_range_loop)]
        for j in 0..m {
            let a_ij = &pp.m_rows[i][j];           // R_q element
            let s_action = SAction::from_ring(*a_ij);  // Create S-action from ring element
            let v: [Fq; neo_math::ring::D] = cols[j].try_into().expect("column length should be d"); // digits for column j

            // Apply S-action to the coefficient vector
            let result = s_action.apply_vec(&v);
            
            // Constant-time accumulation (no secret-dependent branching).
            for (a, &r) in acc.iter_mut().zip(&result) {
                *a += r;
            }
        }
    }
    c
}

/// MUST: Verify opening by recomputing commitment (binding implies uniqueness).
#[allow(non_snake_case)]
pub fn verify_open(pp: &PP<RqEl>, c: &Commitment, Z: &[Fq]) -> bool {
    &commit(pp, Z) == c
}

/// MUST: Verify split opening: c == Σ b^{i-1} c_i and Z == Σ b^{i-1} Z_i, with ||Z_i||_∞<b (range assertions done by caller).
#[allow(non_snake_case)]
pub fn verify_split_open(pp: &PP<RqEl>, c: &Commitment, b: u32, c_is: &[Commitment], Z_is: &[Vec<Fq>]) -> bool {
    let k = c_is.len();
    if k != Z_is.len() { return false; }
    // Check shapes
    for ci in c_is { if ci.d != c.d || ci.kappa != c.kappa { return false; } }
    // Recompose commitment
    let mut acc = Commitment::zeros(c.d, c.kappa);
    let mut pow = Fq::ONE;
    let b_f = Fq::from_u64(b as u64);
    #[allow(clippy::needless_range_loop)]
    for i in 0..k {
        for (a, &x) in acc.data.iter_mut().zip(&c_is[i].data) { *a += x * pow; }
        pow *= b_f;
    }
    if &acc != c { return false; }
    // Recompose Z and check commit again
    let d = pp.d; let m = pp.m;
    let mut Z = vec![Fq::ZERO; d*m];
    let mut pow = Fq::ONE;
    for Zi in Z_is {
        if Zi.len() != d*m { return false; }
        for (a, &x) in Z.iter_mut().zip(Zi) { *a += x * pow; }
        pow *= b_f;
    }
    &commit(pp, &Z) == c
}

/// MUST: S-homomorphism: ρ·L(Z) = L(ρ·Z).  We expose helpers for left-multiplying commitments.
/// Since we don't have direct access to SMatrix, we use SAction to operate on the commitment data.
pub fn s_mul(rho_ring: &RqEl, c: &Commitment) -> Commitment {
    let d = c.d; let kappa = c.kappa;
    let mut out = Commitment::zeros(d, kappa);
    let s_action = SAction::from_ring(*rho_ring);
    
    for col in 0..kappa {
        let src: [Fq; neo_math::ring::D] = c.col(col).try_into().expect("column length should be d");
        let dst_result = s_action.apply_vec(&src);
        let dst = out.col_mut(col);
        dst.copy_from_slice(&dst_result);
    }
    out
}

/// Reference implementation of commit for testing: c = cf(M · cf^{-1}(Z))
/// This implements the specification directly using S-action matrix multiplication
/// Used to validate the optimized commit paths produce identical results
#[allow(non_snake_case, dead_code)]
pub fn commit_spec(pp: &PP<RqEl>, Z: &[Fq]) -> Commitment {
    let d = pp.d; let m = pp.m;
    let mut c = Commitment::zeros(d, pp.kappa);
    
    for i in 0..pp.kappa {
        let acc_i = c.col_mut(i);
        for j in 0..m {
            let a_ij = pp.m_rows[i][j];
            let s = SAction::from_ring(a_ij);
            let v: [Fq; neo_math::ring::D] = Z[j*d..(j+1)*d].try_into().unwrap();
            let w = s.apply_vec(&v);
            for (a, &x) in acc_i.iter_mut().zip(&w) { *a += x; }
        }
    }
    c
}

pub fn s_lincomb(rhos: &[RqEl], cs: &[Commitment]) -> AjtaiResult<Commitment> {
    if rhos.is_empty() {
        return Err(AjtaiError::EmptyInput);
    }
    if rhos.len() != cs.len() {
        return Err(AjtaiError::SizeMismatch { 
            expected: rhos.len(), 
            actual: cs.len() 
        });
    }
    if cs.is_empty() {
        return Err(AjtaiError::EmptyInput);
    }
    
    let mut acc = Commitment::zeros(cs[0].d, cs[0].kappa);
    for (rho, c) in rhos.iter().zip(cs) {
        let term = s_mul(rho, c);
        acc.add_inplace(&term);
    }
    Ok(acc)
}
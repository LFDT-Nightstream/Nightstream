use p3_goldilocks::Goldilocks as Fq;
use p3_field::PrimeCharacteristicRing;
use rand::{RngCore, CryptoRng};
use crate::types::{PP, Commitment};
use crate::util::is_binary_vec;

/// Bring in ring & S-action APIs from neo-math.
use neo_math::ring::{Rq as RqEl, cf_inv as cf_unmap};
use neo_math::s_action::SAction;

/// MUST: Setup(κ,m) → sample M ← R_q^{κ×m} uniformly (Def. 9).
pub fn setup<R: RngCore + CryptoRng>(rng: &mut R, d: usize, kappa: usize, m: usize) -> PP<RqEl> {
    let mut rows = Vec::with_capacity(kappa);
    for _ in 0..kappa {
        let mut row = Vec::with_capacity(m);
        for _ in 0..m {
            // sample ring element uniformly by sampling d random coefficients in F_q and mapping via cf^{-1}
            let coeffs: [Fq; neo_math::ring::D] = (0..neo_math::ring::D).map(|_| {
                // uniformly random field element
                let r = rng.next_u64();
                Fq::from_u64(r)
            }).collect::<Vec<_>>().try_into().unwrap();
            row.push(cf_unmap(coeffs));
        }
        rows.push(row);
    }
    PP { kappa, m, d, m_rows: rows }
}

/// MUST: Commit(pp, Z) = cf(M · cf^{-1}(Z)) as c ∈ F_q^{d×κ}.  S-homomorphic over S by construction.
#[allow(non_snake_case)]
pub fn commit(pp: &PP<RqEl>, Z: &[Fq]) -> Commitment {
    // Z is d×m (column-major by (col*d + row)), output c is d×kappa (column-major)
    let d = pp.d; let m = pp.m; let kappa = pp.kappa;
    assert_eq!(Z.len(), d*m, "Z must be d×m");

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
            
            // Pay-per-bit / small-digit optimization:
            // If v is binary, we could optimize this further, but for now use the general path
            if is_binary_vec(cols[j]) {
                // For binary vectors, we can optimize but use general path for correctness
                for (a, &r) in acc.iter_mut().zip(&result) {
                    *a += r;
                }
                } else {
                // General path: add the S-action result directly
                for (a, &r) in acc.iter_mut().zip(&result) {
                    *a += r;
                }
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

pub fn s_lincomb(rhos: &[RqEl], cs: &[Commitment]) -> Commitment {
    assert!(!rhos.is_empty() && rhos.len() == cs.len());
    let mut acc = Commitment::zeros(cs[0].d, cs[0].kappa);
    for (rho, c) in rhos.iter().zip(cs) {
        let term = s_mul(rho, c);
        acc.add_inplace(&term);
    }
    acc
}
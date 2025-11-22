//! Test helper functions for Ajtai commitment testing
//! 
//! These functions are useful for testing linear algebra properties and extractors
//! but are not part of the core protocol.

use neo_ajtai::{PP, AjtaiError, AjtaiResult, Commitment};
use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::Goldilocks as Fq;
use neo_math::ring::{Rq as RqEl, cf, D};
use neo_math::s_action::SAction;

/// Rotation step for Φ₈₁(X) = X^54 + X^27 + 1 - exposed for testing
/// 
/// This is the same implementation as in commit.rs but exposed for test usage
#[inline]
#[allow(dead_code)]
pub fn rot_step(cur: &[Fq; D], next: &mut [Fq; D]) {
    let last = cur[D - 1];
    next[0] = Fq::ZERO;
    next[1..D].copy_from_slice(&cur[..(D - 1)]);
    next[0] -= last;        // -1 * last
    next[27] -= last;       // -X^27 * last
}

/// PRODUCTION IMPLEMENTATION — true extraction from Ajtai public parameters.
///
/// Builds the linear map `L ∈ F_q^{(d·κ) × (d·m)}` such that, for **column‑major**
/// encodings of `Z ∈ F_q^{d×m}` and commitment `c ∈ F_q^{d×κ}`:
///
///   `vec(c) = L · vec(Z)`  where  `c = cf(M · cf^{-1}(Z))`.
///
/// **Indexing conventions**
/// - `z_len` must be `d·m`. The caller pads rows externally if the circuit padded `z_digits`
///   to a power of two (we intentionally keep the extractor strict).
/// - `num_coords` ≤ `d·κ`; typical usage requests all `d·κ` rows.
///
/// **Performance**
/// - Deduplicates equal `a_ij` by hashing `cf(a_ij)` (O(1) average).
/// - Precomputes all rotation columns per unique ring element via `rot_step`.
/// - Builds rows in parallel with Rayon.
#[allow(dead_code)]
pub fn rows_for_coords(
    pp: &PP<RqEl>, 
    z_len: usize, 
    num_coords: usize
) -> AjtaiResult<Vec<Vec<Fq>>> {
    use rayon::prelude::*;
    
    let d = pp.d;
    let m = pp.m; 
    let kappa = pp.kappa;
    
    // Validate dimensions to prevent binding bugs
    if D != d {
        return Err(AjtaiError::InvalidInput(
            format!("Ajtai ring dimension mismatch: compile-time D ({}) != runtime pp.d ({})", D, d)
        ));
    }
    if z_len != d * m {
        return Err(AjtaiError::InvalidInput("z_len must equal d*m".to_string()));
    }
    if num_coords > d * kappa {
        return Err(AjtaiError::InvalidInput("num_coords exceeds d*kappa".to_string()));
    }

    // Deduplicate identical ring elements
    use std::collections::HashMap;
    
    let mut unique_map = HashMap::new();
    let mut all_elements: Vec<RqEl> = Vec::new();
    let mut element_indices = vec![vec![0; m]; kappa];
    
    for commit_col in 0..kappa {
        for j in 0..m {
            let a_ij = pp.m_rows[commit_col][j];
            let canonical_key = cf(a_ij);
            
            let idx = *unique_map.entry(canonical_key).or_insert_with(|| {
                let new_idx = all_elements.len();
                all_elements.push(a_ij);
                new_idx
            });
            
            element_indices[commit_col][j] = idx;
        }
    }

    // Precompute all rotation columns for each unique element
    let mut rot_columns: Vec<Box<[[Fq; D]]>> = Vec::with_capacity(all_elements.len());
    rot_columns.par_extend(
        all_elements.par_iter().map(|&a_ij| {
            let mut cols = vec![[Fq::ZERO; D]; D].into_boxed_slice();
            precompute_rot_columns(a_ij, &mut cols);
            cols
        })
    );

    // Parallel row computation
    let rows: Vec<Vec<Fq>> = (0..num_coords)
        .into_par_iter()
        .map(|coord_idx| {
            let mut row = vec![Fq::ZERO; z_len];
            
            let commit_col = coord_idx / d;
            let commit_row = coord_idx % d;
            
            if commit_col >= kappa {
                return row;
            }
            
            for j in 0..m {
                let element_idx = element_indices[commit_col][j];
                let cols = &rot_columns[element_idx];
                
                let base_idx = j * d;
                for input_row in 0..d {
                    let input_idx = base_idx + input_row;
                    row[input_idx] = cols[input_row][commit_row];
                }
            }
            
            row
        })
        .collect();
    
    Ok(rows)
}

/// Compute a single Ajtai binding row on-demand from PP
/// 
/// This is the streaming version of `rows_for_coords` that computes only one row
/// to avoid materializing the entire row matrix in memory.
#[allow(dead_code)]
pub fn compute_single_ajtai_row(
    pp: &PP<RqEl>,
    coord_idx: usize,
    z_len: usize,
    num_coords: usize,
) -> AjtaiResult<Vec<Fq>> {
    let d = pp.d;
    let m = pp.m;
    let kappa = pp.kappa;
    
    // Validation
    if D != d {
        return Err(AjtaiError::InvalidInput(
            format!("Ajtai ring dimension mismatch: compile-time D ({}) != runtime pp.d ({})", D, d)
        ));
    }
    if z_len != d * m {
        return Err(AjtaiError::InvalidInput("z_len must equal d*m".to_string()));
    }
    if num_coords > d * kappa {
        return Err(AjtaiError::InvalidInput("num_coords exceeds d*kappa".to_string()));
    }
    if coord_idx >= num_coords {
        return Err(AjtaiError::InvalidInput("coord_idx out of range".to_string()));
    }
    
    let commit_col = coord_idx / d;
    let commit_row = coord_idx % d;
    
    if commit_col >= kappa {
        return Err(AjtaiError::InvalidInput("coord_idx out of range w.r.t. kappa".to_string()));
    }
    
    let mut row = vec![Fq::ZERO; z_len];
    
    for j in 0..m {
        let a_ij = pp.m_rows[commit_col][j];
        let mut col = cf(a_ij);
        let mut nxt = [Fq::ZERO; D];
        
        for t in 0..d {
            row[j * d + t] = col[commit_row];
            rot_step(&col, &mut nxt);
            core::mem::swap(&mut col, &mut nxt);
        }
    }
    
    Ok(row)
}

/// Fill `cols` with the d rotation columns of rot(a): cols[t] = cf(a * X^t).
#[inline]
#[allow(dead_code)]
fn precompute_rot_columns(a: RqEl, cols: &mut [[Fq; D]]) {
    let mut col = cf(a);
    let mut nxt = [Fq::ZERO; D];
    for t in 0..D {
        cols[t] = col;
        rot_step(&col, &mut nxt);
        core::mem::swap(&mut col, &mut nxt);
    }
}

/// Reference implementation for differential testing against the optimized commit
/// 
/// Implements the specification directly: c = cf(M · cf^{-1}(Z))
/// This verifies the fundamental S-action isomorphism cf(a·b) = rot(a)·cf(b) 
/// at the commitment level - exactly the algebra the construction relies on.
/// 
/// ⚠️  FOR TESTING ONLY - NOT CONSTANT TIME ⚠️
#[allow(non_snake_case)]
#[allow(dead_code)]
pub fn commit_spec(pp: &PP<RqEl>, Z: &[Fq]) -> Commitment {
    let d = pp.d; 
    let m = pp.m;
    let mut c = Commitment::zeros(d, pp.kappa);
    
    for i in 0..pp.kappa {
        let acc_i = c.col_mut(i);
        for j in 0..m {
            let s = SAction::from_ring(pp.m_rows[i][j]);
            let v: [Fq; D] = Z[j*d..(j+1)*d].try_into().unwrap();
            let w = s.apply_vec(&v);
            for (a, &x) in acc_i.iter_mut().zip(&w) { 
                *a += x; 
            }
        }
    }
    c
}


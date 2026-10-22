use neo_ajtai::{AjtaiError, AjtaiResult, Commitment, PP};
use neo_math::ring::{cf, Rq as RqEl, D};
use neo_math::s_action::SAction;
use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::Goldilocks as Fq;

#[inline]
#[allow(dead_code)]
pub(crate) fn rot_step(cur: &[Fq; D], next: &mut [Fq; D]) {
    let last = cur[D - 1];
    next[0] = Fq::ZERO;
    next[1..D].copy_from_slice(&cur[..(D - 1)]);
    next[0] -= last;
    next[27] -= last;
}

#[allow(dead_code)]
pub(crate) fn rows_for_coords(pp: &PP<RqEl>, z_len: usize, num_coords: usize) -> AjtaiResult<Vec<Vec<Fq>>> {
    use rayon::prelude::*;

    let d = pp.d();
    let m = pp.m();
    let kappa = pp.kappa();

    if D != d {
        return Err(AjtaiError::InvalidInput(format!(
            "Ajtai ring dimension mismatch: compile-time D ({}) != runtime pp.d ({})",
            D, d
        )));
    }
    if z_len != d * m {
        return Err(AjtaiError::InvalidInput("z_len must equal d*m".to_string()));
    }
    if num_coords > d * kappa {
        return Err(AjtaiError::InvalidInput("num_coords exceeds d*kappa".to_string()));
    }

    use std::collections::HashMap;

    let mut unique_map = HashMap::new();
    let mut all_elements: Vec<RqEl> = Vec::new();
    let mut element_indices = vec![vec![0; m]; kappa];

    for commit_col in 0..kappa {
        for j in 0..m {
            let a_ij = pp.rows()[commit_col][j];
            let canonical_key = cf(a_ij);

            let idx = *unique_map.entry(canonical_key).or_insert_with(|| {
                let new_idx = all_elements.len();
                all_elements.push(a_ij);
                new_idx
            });

            element_indices[commit_col][j] = idx;
        }
    }

    let mut rot_columns: Vec<Box<[[Fq; D]]>> = Vec::with_capacity(all_elements.len());
    rot_columns.par_extend(all_elements.par_iter().map(|&a_ij| {
        let mut cols = vec![[Fq::ZERO; D]; D].into_boxed_slice();
        precompute_rot_columns(a_ij, &mut cols);
        cols
    }));

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

#[allow(dead_code)]
pub(crate) fn compute_single_ajtai_row(
    pp: &PP<RqEl>,
    coord_idx: usize,
    z_len: usize,
    num_coords: usize,
) -> AjtaiResult<Vec<Fq>> {
    let d = pp.d();
    let m = pp.m();
    let kappa = pp.kappa();

    if D != d {
        return Err(AjtaiError::InvalidInput(format!(
            "Ajtai ring dimension mismatch: compile-time D ({}) != runtime pp.d ({})",
            D, d
        )));
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
        return Err(AjtaiError::InvalidInput(
            "coord_idx out of range w.r.t. kappa".to_string(),
        ));
    }

    let mut row = vec![Fq::ZERO; z_len];

    for j in 0..m {
        let a_ij = pp.rows()[commit_col][j];
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

#[allow(non_snake_case)]
#[allow(dead_code)]
pub(crate) fn commit_spec(pp: &PP<RqEl>, Z: &[Fq]) -> Commitment {
    let d = pp.d();
    let m = pp.m();
    let mut c = Commitment::zeros(d, pp.kappa());

    for i in 0..pp.kappa() {
        let acc_i = c.col_mut(i);
        for j in 0..m {
            let s = SAction::from_ring(pp.rows()[i][j]);
            let v: [Fq; D] = Z[j * d..(j + 1) * d].try_into().unwrap();
            let w = s.apply_vec(&v);
            for (a, &x) in acc_i.iter_mut().zip(&w) {
                *a += x;
            }
        }
    }
    c
}

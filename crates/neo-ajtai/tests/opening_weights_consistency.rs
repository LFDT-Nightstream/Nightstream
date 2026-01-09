use neo_ajtai::{commit_row_major_seeded, compute_opening_weights_for_u_seeded};
use neo_ccs::Mat;
use neo_math::{D, F};
use p3_field::PrimeCharacteristicRing;

fn dot_u_commitment(u_vecs: &[[F; D]], c: &neo_ajtai::Commitment) -> F {
    assert_eq!(c.d, D);
    assert_eq!(c.kappa, u_vecs.len());
    assert_eq!(c.data.len(), c.d * c.kappa);

    let mut acc = F::ZERO;
    for i in 0..c.kappa {
        for r in 0..c.d {
            acc += u_vecs[i][r] * c.data[i * c.d + r];
        }
    }
    acc
}

#[test]
fn opening_weights_match_commitment_projection() {
    let seed = [7u8; 32];
    let kappa = 3usize;
    let m = 32usize;

    let mut z = Mat::zero(D, m, F::ZERO);
    for r in 0..D {
        for c in 0..m {
            let v = ((r * 17 + c * 31) % 5) as u64;
            z[(r, c)] = F::from_u64(v);
        }
    }

    let c = commit_row_major_seeded(seed, D, kappa, m, &z);

    let mut u_vecs = Vec::<[F; D]>::with_capacity(kappa);
    for i in 0..kappa {
        let mut u = [F::ZERO; D];
        for r in 0..D {
            u[r] = F::from_u64((i as u64) * 1000 + (r as u64) * 17 + 9);
        }
        u_vecs.push(u);
    }

    let w_u = compute_opening_weights_for_u_seeded(seed, m, &u_vecs);
    assert_eq!(w_u.len(), D * m);

    let mut ip = F::ZERO;
    for r in 0..D {
        for c in 0..m {
            ip += w_u[r * m + c] * z[(r, c)];
        }
    }

    let projected = dot_u_commitment(&u_vecs, &c);
    assert_eq!(ip, projected);
}


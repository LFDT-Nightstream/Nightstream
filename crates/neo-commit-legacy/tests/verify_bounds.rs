use neo_commit::{AjtaiCommitter, TOY_PARAMS, SECURE_PARAMS};
use neo_modint::ModInt;
use neo_ring::RingElement;

fn zero_ring(n: usize) -> RingElement<ModInt> {
    RingElement::from_scalar(ModInt::from_u64(0), n)
}

fn scalar_ring(x: u64, n: usize) -> RingElement<ModInt> {
    RingElement::from_scalar(ModInt::from_u64(x), n)
}

/// e == e_bound should be accepted
#[test]
fn verify_accepts_e_at_bound() {
    let params = TOY_PARAMS;
    let comm = AjtaiCommitter::setup_unchecked(params);

    let w = vec![zero_ring(params.n); params.d];
    let e = vec![scalar_ring(params.e_bound, params.n); params.k];

    // c = A*w + e
    let mut c = vec![zero_ring(params.n); params.k];
    for i in 0..params.k {
        for (aij, wj) in comm.public_matrix()[i].iter().zip(&w) {
            c[i] = c[i].clone() + aij.clone() * wj.clone();
        }
        c[i] = c[i].clone() + e[i].clone();
    }
    assert!(comm.verify(&c, &w, &e), "Verification should accept e at bound {}", params.e_bound);
}

/// e == e_bound+1 should be rejected
#[test]
fn verify_rejects_e_above_bound() {
    let params = TOY_PARAMS;
    let comm = AjtaiCommitter::setup_unchecked(params);

    let w = vec![zero_ring(params.n); params.d];
    let e = vec![scalar_ring(params.e_bound + 1, params.n); params.k];

    let mut c = vec![zero_ring(params.n); params.k];
    for i in 0..params.k {
        for (aij, wj) in comm.public_matrix()[i].iter().zip(&w) {
            c[i] = c[i].clone() + aij.clone() * wj.clone();
        }
        c[i] = c[i].clone() + e[i].clone();
    }
    assert!(!comm.verify(&c, &w, &e), "Verification should reject e above bound {}", params.e_bound + 1);
}

/// w == (norm_bound + beta) should be accepted
#[test]
fn verify_accepts_w_at_bound() {
    let params = SECURE_PARAMS; // TOY has a huge norm_bound; use SECURE for a crisp check
    let comm = AjtaiCommitter::setup_unchecked(params);

    let w_bound = params.norm_bound + params.beta;
    let mut w = vec![zero_ring(params.n); params.d];
    w[0] = scalar_ring(w_bound, params.n);
    let e = vec![zero_ring(params.n); params.k];

    let mut c = vec![zero_ring(params.n); params.k];
    for i in 0..params.k {
        for (aij, wj) in comm.public_matrix()[i].iter().zip(&w) {
            c[i] = c[i].clone() + aij.clone() * wj.clone();
        }
        c[i] = c[i].clone() + e[i].clone();
    }
    assert!(comm.verify(&c, &w, &e), "Verification should accept w at bound {}", w_bound);
}

/// w == (norm_bound + beta + 1) should be rejected
#[test]
fn verify_rejects_w_above_bound() {
    let params = SECURE_PARAMS;
    let comm = AjtaiCommitter::setup_unchecked(params);

    let too_big = params.norm_bound + params.beta + 1;
    let mut w = vec![zero_ring(params.n); params.d];
    w[0] = scalar_ring(too_big, params.n);
    let e = vec![zero_ring(params.n); params.k];

    let mut c = vec![zero_ring(params.n); params.k];
    for i in 0..params.k {
        for (aij, wj) in comm.public_matrix()[i].iter().zip(&w) {
            c[i] = c[i].clone() + aij.clone() * wj.clone();
        }
        c[i] = c[i].clone() + e[i].clone();
    }
    assert!(!comm.verify(&c, &w, &e), "Verification should reject w above bound {}", too_big);
}

/// Test that verification bounds are consistent with what commit produces
#[test]
fn verify_accepts_real_commit_output() {
    let params = TOY_PARAMS;
    let comm = AjtaiCommitter::setup_unchecked(params);
    
    // Create a simple witness
    let w = vec![zero_ring(params.n); params.d];
    let mut transcript = b"test".to_vec();
    
    // Commit and verify the result should always pass
    let (c, e, blinded_w, _r) = comm.commit(&w, &mut transcript).unwrap();
    assert!(comm.verify(&c, &blinded_w, &e), 
        "Verification must accept output from commit operation");
}

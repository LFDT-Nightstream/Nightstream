//! Tests for the modular oracle implementation

use crate::optimized_engine::oracle::gate::{gate_pair, pair_to_full_indices, fold_partial_in_place, PairGate};
use neo_math::K;
use p3_goldilocks::Goldilocks as F;
use p3_field::PrimeCharacteristicRing;

#[test]
fn test_gate_pair_evaluation() {
    let weights = vec![K::from(F::from_u64(10)), K::from(F::from_u64(20)), 
                      K::from(F::from_u64(30)), K::from(F::from_u64(40))];
    
    // Test pair 0: (10, 20)
    let result = gate_pair(&weights, 0, K::ZERO);
    assert_eq!(result, K::from(F::from_u64(10))); // (1-0)*10 + 0*20 = 10
    
    let result = gate_pair(&weights, 0, K::ONE);
    assert_eq!(result, K::from(F::from_u64(20))); // (1-1)*10 + 1*20 = 20
    
    // Test pair 1: (30, 40)
    let result = gate_pair(&weights, 1, K::from(F::from_u64(2)));
    assert_eq!(result, K::from(F::from_u64(50))); // (1-2)*30 + 2*40 = -30 + 80 = 50
}

#[test]
fn test_pair_to_full_indices() {
    // Round 0: stride = 1, pairs are consecutive
    assert_eq!(pair_to_full_indices(0, 0), (0, 1));
    assert_eq!(pair_to_full_indices(1, 0), (2, 3));
    assert_eq!(pair_to_full_indices(2, 0), (4, 5));
    
    // Round 1: stride = 2, pairs skip by 2
    assert_eq!(pair_to_full_indices(0, 1), (0, 2));
    assert_eq!(pair_to_full_indices(1, 1), (1, 3));
    assert_eq!(pair_to_full_indices(2, 1), (4, 6));
    
    // Round 2: stride = 4, pairs skip by 4
    assert_eq!(pair_to_full_indices(0, 2), (0, 4));
    assert_eq!(pair_to_full_indices(1, 2), (1, 5));
}

#[test]
fn test_fold_partial_in_place() {
    let mut v = vec![K::from(F::from_u64(10)), K::from(F::from_u64(20)), 
                     K::from(F::from_u64(30)), K::from(F::from_u64(40))];
    let r = K::from(F::from_u64(3));
    
    fold_partial_in_place(&mut v, r);
    
    // v[0] = (1-3)*10 + 3*20 = -20 + 60 = 40
    // v[1] = (1-3)*30 + 3*40 = -60 + 120 = 60
    assert_eq!(v[0], K::from(F::from_u64(40)));
    assert_eq!(v[1], K::from(F::from_u64(60)));
    
    // Manual truncation needed
    v.truncate(2);
    assert_eq!(v.len(), 2);
}

#[test]
fn test_pair_gate_view() {
    let weights = vec![K::from(F::from_u64(10)), K::from(F::from_u64(20)), 
                      K::from(F::from_u64(30)), K::from(F::from_u64(40))];
    let gate = PairGate::new(&weights);
    
    assert_eq!(gate.half, 2);
    assert_eq!(gate.pair(0), (K::from(F::from_u64(10)), K::from(F::from_u64(20))));
    assert_eq!(gate.pair(1), (K::from(F::from_u64(30)), K::from(F::from_u64(40))));
    
    assert_eq!(gate.eval(0, K::ZERO), K::from(F::from_u64(10)));
    assert_eq!(gate.eval(0, K::ONE), K::from(F::from_u64(20)));
}

#[test]
fn test_eval_ajtai_weights() {
    // Test that Eval weights match paper formula: γ^{i-1 + j·k}
    // Using small parameters: ell_d=1, ell_n=1, k_total=3
    
    let gamma = K::from(F::from_u64(7)); // arbitrary prime
    let k_total = 3;
    let me_offset = 1; // ME witnesses start at instance 2
    let me_count = 2;  // instances 2 and 3
    let t = 2;         // number of matrices
    
    // Precompute γ^k
    let mut gamma_to_k = K::ONE;
    for _ in 0..k_total {
        gamma_to_k *= gamma;
    }
    assert_eq!(gamma_to_k, gamma * gamma * gamma); // γ^3
    
    // Precompute γ^{i_abs-1} for ME witnesses
    let mut gamma_pow_i_abs = vec![K::ONE; me_count];
    {
        let mut g = K::ONE;
        for _ in 0..me_offset { g *= gamma; } // g = γ^1
        for i_off in 0..me_count {
            gamma_pow_i_abs[i_off] = g;
            g *= gamma;
        }
    }
    
    // Check γ^{i_abs-1} values
    assert_eq!(gamma_pow_i_abs[0], gamma);         // γ^1 for instance 2
    assert_eq!(gamma_pow_i_abs[1], gamma * gamma); // γ^2 for instance 3
    
    // Test weights for each (i,j) pair
    for i_off in 0..me_count {
        let i_abs = me_offset + 1 + i_off; // 1-based instance index (2, 3, ...)
        
        for j in 0..t {
            // Expected: γ^{i_abs-1 + j·k}
            let mut expected = K::ONE;
            for _ in 0..(i_abs-1 + j*k_total) {
                expected = expected * gamma;
            }
            
            // Computed: γ^{i_abs-1} * (γ^k)^j
            let mut computed = gamma_pow_i_abs[i_off];
            for _ in 0..j {
                computed *= gamma_to_k;
            }
            
            assert_eq!(computed, expected, 
                "Weight mismatch for (i={}, j={}): computed={:?}, expected={:?}", 
                i_abs, j, computed, expected);
        }
    }
}
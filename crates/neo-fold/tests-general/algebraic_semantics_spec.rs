use p3_goldilocks::Goldilocks as F;
use p3_field::PrimeCharacteristicRing;

/// χ_r over {0,1}^2 in row-major order: (00, 10, 01, 11)
fn chi_r_2d(r0: F, r1: F) -> [F; 4] {
    let one = F::ONE;
    [
        (one - r0) * (one - r1), // 00
        r0 * (one - r1),         // 10
        (one - r0) * r1,         // 01
        r0 * r1,                 // 11
    ]
}

fn dot(a: &[F; 4], b: &[F; 4]) -> F {
    a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3]
}

#[test]
fn residual_inner_product_is_not_f_of_mle_product() {
    // Think of A z, B z, C z as the three row-vectors in R1CS form.
    // Pick values that create cross-terms when multiplying MLE sums.
    let az = [F::from_u64(1), F::from_u64(2), F::from_u64(0), F::from_u64(0)];
    let bz = [F::from_u64(3), F::from_u64(0), F::from_u64(4), F::from_u64(0)];
    let cz = [F::ZERO, F::ZERO, F::ZERO, F::ZERO];

    // Any non-degenerate r works; choose simple field elements.
    let r0 = F::from_u64(2);
    let r1 = F::from_u64(3);
    let chi = chi_r_2d(r0, r1);

    // Correct quantity for Π_CCS terminal check at r:
    //   y0 = ⟨A z, χ_r⟩, y1 = ⟨B z, χ_r⟩, y2 = ⟨C z, χ_r⟩
    //   f(Y(r)) = y0 * y1 - y2
    let y0 = dot(&az, &chi);
    let y1 = dot(&bz, &chi);
    let y2 = dot(&cz, &chi);
    let f_y_r = y0 * y1 - y2;

    // Wrong residual-based substitute:
    //   g(r) = ⟨(A z ∘ B z - C z), χ_r⟩
    let residual = [
        az[0]*bz[0] - cz[0],
        az[1]*bz[1] - cz[1],
        az[2]*bz[2] - cz[2],
        az[3]*bz[3] - cz[3],
    ];
    let g_r = dot(&residual, &chi);

    // They are *not* equal in general (here: 36 != 6)
    assert_ne!(f_y_r, g_r, "f(Y(r)) must not be replaced by ⟨Az∘Bz−Cz, χ_r⟩");
    
    // Verify the specific values to ensure the test is meaningful
    assert_eq!(f_y_r, F::from_u64(36), "Expected f(Y(r)) = 36");
    assert_eq!(g_r, F::from_u64(6), "Expected g(r) = 6");
}

#[test]
fn base_b_recombination_can_stay_constant_while_mle_changes() {
    // Simulate one y-vector of length 4 (two sum-check variables).
    // Choose base b=2 like your codepath.
    let b = F::from_u64(2);

    // Pow-b vector [1, b, b^2, b^3]
    let pow_b = [F::ONE, b, b*b, b*b*b];

    // Pick an r so χ_r is well-defined and not degenerate.
    let r0 = F::from_u64(2);
    let r1 = F::from_u64(3);
    let chi = chi_r_2d(r0, r1);

    // Start from y = 0 and add a delta with zero base-b recomposition:
    //   Δ = [b, -1, 0, 0]  ->  ⟨Δ, pow_b⟩ = b*1 + (-1)*b = 0
    let delta = [b, -F::ONE, F::ZERO, F::ZERO];

    // Base-b recombination doesn't change:
    let base_b_before = F::ZERO;
    let base_b_after  = dot(&delta, &pow_b);
    assert_eq!(base_b_after, base_b_before, "base-b recombination stayed constant");

    // But the MLE evaluation at r *does* change:
    //   ⟨Δ, χ_r⟩ = b*χ00 - 1*χ10
    let mle_delta = dot(&delta, &chi);
    assert_ne!(mle_delta, F::ZERO, "MLE at r changed while base-b recombination did not");
}

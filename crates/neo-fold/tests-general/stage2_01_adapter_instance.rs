//! Stage 2: Adapter-only tests — instance carryover

use neo_ajtai::Commitment;
use neo_ccs::{Mat, relations::MeInstance as ModernMeInstance};
use neo_math::{F, K, KExtensions};
use p3_field::PrimeCharacteristicRing;

#[test]
fn modern_to_legacy_instance_copies_core_fields() {
    let params = neo_params::NeoParams::goldilocks_small_circuits();

    // Build a tiny modern ME instance
    let d = 2usize; let kappa = 2usize; let m_in = 1usize;
    let mut c = Commitment::zeros(d, kappa);
    c.data = vec![F::from_u64(11), F::from_u64(22), F::from_u64(33), F::from_u64(44)];

    // X is d×m_in, row-major
    let x = Mat::from_row_major(d, m_in, vec![F::from_u64(7), F::from_u64(8)]);

    // r in K^ell (pick 2 limbs)
    let r: Vec<K> = vec![K::from_coeffs([F::from_u64(5), F::from_u64(6)])];

    // y has t entries, each length d (use zeros for simplicity)
    let y: Vec<Vec<K>> = vec![vec![K::from_coeffs([F::ZERO, F::ZERO]); d]];
    let y_scalars: Vec<K> = vec![K::from_coeffs([F::ZERO, F::ZERO])];

    let modern = ModernMeInstance {
        c,
        X: x,
        r,
        y,
        y_scalars,
        m_in,
        fold_digest: [0u8;32],
        c_step_coords: vec![],
        u_offset: 0,
        u_len: 0,
    };

    let legacy = neo_fold::bridge_adapter::modern_to_legacy_instance(&modern, &params);

    // c.data copied to c_coords
    let expected_c_coords: Vec<F> = vec![11,22,33,44].into_iter().map(F::from_u64).collect();
    #[allow(deprecated)]
    {
        assert_eq!(legacy.c_coords, expected_c_coords, "c_coords copy mismatch");
    }

    // r converted to base limbs (2 limbs per element)
    #[allow(deprecated)]
    {
        assert_eq!(legacy.r_point.len(), modern.r.len() * 2, "r_point limb count");
    }

    // y converted to base limbs (2 limbs per K entry)
    let expected_y_limbs = modern.y.iter().map(|v| v.len()*2).sum::<usize>();
    #[allow(deprecated)]
    {
        assert_eq!(legacy.y_outputs.len(), expected_y_limbs, "y limbs count");
    }

    // base_b carried from params
    #[allow(deprecated)]
    {
        assert_eq!(legacy.base_b as u32, params.b, "base_b mismatch");
    }
}


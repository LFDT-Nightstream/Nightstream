//! IVC EV (public-ρ) tests: fold enforced in-circuit, ρ is public and recomputed
use neo::F;
use neo::ivc::{
    ev_full_ccs_public_rho, build_ev_full_witness, create_step_digest, rho_from_transcript,
    Accumulator,
};
use neo_ccs::check_ccs_rowwise_zero;
use p3_field::PrimeCharacteristicRing;

fn fe(v: u64) -> F { F::from_u64(v) }

#[test]
fn ev_public_rho_happy_path() {
    let y_prev = vec![fe(3), fe(5), fe(8)];
    let y_step = vec![fe(2), fe(1), fe(4)];

    let prev_acc = Accumulator {
        c_z_digest: [0u8; 32],
        c_coords: vec![],
        y_compact: y_prev.clone(),
        step: 9,
    };

    // Bind transcript to known public data (step, |y_prev|, y_prev)
    let mut step_data = vec![fe(prev_acc.step), fe(y_prev.len() as u64)];
    step_data.extend_from_slice(&y_prev);

    let step_digest = create_step_digest(&step_data);
    let (rho, _dig) = rho_from_transcript(&prev_acc, step_digest);

    let ccs = ev_full_ccs_public_rho(y_prev.len());
    let (witness, y_next) = build_ev_full_witness(rho, &y_prev, &y_step);

    let mut public = vec![rho];
    public.extend_from_slice(&y_prev);
    public.extend_from_slice(&y_next);

    assert!(check_ccs_rowwise_zero(&ccs, &public, &witness).is_ok());
}

#[test]
fn ev_public_rho_rejects_tampering_rho_and_y_next() {
    let y_prev = vec![fe(7), fe(1)];
    let y_step = vec![fe(3), fe(5)];
    let prev_acc = Accumulator { c_z_digest: [0;32], c_coords: vec![], y_compact: y_prev.clone(), step: 1 };

    let step_digest = create_step_digest(&[fe(1), fe(2), fe(3)]);
    let (rho, _) = rho_from_transcript(&prev_acc, step_digest);

    let ccs = ev_full_ccs_public_rho(y_prev.len());
    let (witness, mut y_next) = build_ev_full_witness(rho, &y_prev, &y_step);

    // base public input
    let mut public = vec![rho];
    public.extend_from_slice(&y_prev);
    public.extend_from_slice(&y_next);
    assert!(check_ccs_rowwise_zero(&ccs, &public, &witness).is_ok());

    // tamper rho
    let mut public_rho_bad = public.clone();
    public_rho_bad[0] = public_rho_bad[0] + fe(1);
    assert!(check_ccs_rowwise_zero(&ccs, &public_rho_bad, &witness).is_err());

    // tamper y_next
    y_next[0] = y_next[0] + fe(1);
    let mut public_y_bad = vec![rho];
    public_y_bad.extend_from_slice(&y_prev);
    public_y_bad.extend_from_slice(&y_next);
    assert!(check_ccs_rowwise_zero(&ccs, &public_y_bad, &witness).is_err());
}

#[test]
fn rho_changes_if_accumulator_or_step_digest_changes() {
    let y_prev = vec![fe(10), fe(20)];
    let acc1 = Accumulator { c_z_digest: [0;32], c_coords: vec![], y_compact: y_prev.clone(), step: 5 };
    let acc2 = Accumulator { step: 6, ..acc1.clone() }; // different step

    let sd1 = create_step_digest(&[fe(123)]);
    let sd2 = create_step_digest(&[fe(124)]); // different digest

    let (rho_1a, _) = rho_from_transcript(&acc1, sd1);
    let (rho_1b, _) = rho_from_transcript(&acc1, sd1);
    assert_eq!(rho_1a, rho_1b, "determinism failed");

    let (rho_2, _) = rho_from_transcript(&acc2, sd1);
    assert_ne!(rho_1a, rho_2, "changing accumulator.step must change ρ");

    let (rho_3, _) = rho_from_transcript(&acc1, sd2);
    assert_ne!(rho_1a, rho_3, "changing step_digest must change ρ");
}

// Property-based test with proptest (commented out - add proptest feature if needed)
/*
#[cfg(feature = "proptest")]
mod property_tests {
    use super::*;
    use proptest::prelude::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    use rand::RngCore;

    proptest! {
      #[test]
      fn ev_public_rho_holds_and_tamper_fails(y_len in 1usize..5, seed in 0u64..10000) {
          let mut rng = ChaCha8Rng::seed_from_u64(seed);
          let y_prev: Vec<F> = (0..y_len).map(|_| F::from_u64(rng.next_u64())).collect();
          let y_step: Vec<F> = (0..y_len).map(|_| F::from_u64(rng.next_u64())).collect();

          let acc = Accumulator { c_z_digest: [0;32], c_coords: vec![], y_compact: y_prev.clone(), step: (seed % 1000) as u64 };
          let step_digest = create_step_digest(&y_prev);
          let (rho, _) = rho_from_transcript(&acc, step_digest);

          let ccs = ev_full_ccs_public_rho(y_len);
          let (witness, y_next) = build_ev_full_witness(rho, &y_prev, &y_step);

          let mut public = vec![rho];
          public.extend_from_slice(&y_prev);
          public.extend_from_slice(&y_next);

          prop_assert!(check_ccs_rowwise_zero(&ccs, &public, &witness).is_ok());

          // flip rho
          let mut public_bad = public.clone();
          public_bad[0] = public_bad[0] + F::ONE;
          prop_assert!(check_ccs_rowwise_zero(&ccs, &public_bad, &witness).is_err());
      }
    }
}
*/

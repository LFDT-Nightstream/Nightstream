//! Π_RLC commitment combination, projection-checked — Road A Unit 1
//! (encoding.md candidate E; security-note Lemma 5).
//!
//! The projection variant must accept exactly the combined commitment
//! the trusted Toom-3 D²-materialization variant accepts, reject any
//! tampered output or quotient, and beat it decisively on committed
//! wires. β is allocated directly here; the integration site owns the
//! Lemma 5 transcript schedule (absorb inputs → ρ → absorb combined +
//! quotients → squeeze β).

use neo_ajtai::Commitment;
use neo_fold_clean::engine::r1cs_circuit::field_ext::KVar;
use neo_fold_clean::engine::r1cs_circuit::ring_action::{enforce_beta_ladder, projection_quotient};
use neo_fold_clean::engine::r1cs_circuit::R1csBuilder;
use neo_fold_clean::paper::reductions::pi_rlc_circuit::{
    alloc_rlc_commitment_inputs, enforce_rlc_commitment_combination, enforce_rlc_commitment_combination_projection,
    rlc_projection_quotients,
};
use neo_math::ring::D;
use neo_math::F;
use p3_field::PrimeCharacteristicRing;

const KAPPA: usize = 2;
const PAIRS: usize = 3;

/// Deterministic SplitMix64 — reproducible operands, zero deps.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    fn rho(&mut self) -> [F; D] {
        std::array::from_fn(|_| {
            let v = (self.next() % 5) as i64 - 2;
            if v >= 0 {
                F::from_u64(v as u64)
            } else {
                F::ZERO - F::from_u64((-v) as u64)
            }
        })
    }

    fn commitment(&mut self) -> Commitment {
        Commitment {
            d: D,
            kappa: KAPPA,
            data: (0..(KAPPA * D) as u64)
                .map(|_| F::from_u64(self.next()))
                .collect(),
        }
    }
}

struct Fixture {
    rhos: Vec<[F; D]>,
    inputs: Vec<Commitment>,
    combined: Commitment,
}

/// Honest operands with the combined commitment computed per lane by
/// the native division helper (`out = Σ ρ_i·c_i mod Φ`).
fn fixture(seed: u64) -> Fixture {
    let mut rng = Rng(seed);
    let rhos: Vec<[F; D]> = (0..PAIRS).map(|_| rng.rho()).collect();
    let inputs: Vec<Commitment> = (0..PAIRS).map(|_| rng.commitment()).collect();

    let mut data = Vec::with_capacity(KAPPA * D);
    for lane in 0..KAPPA {
        let pairs: Vec<([F; D], [F; D])> = rhos
            .iter()
            .zip(inputs.iter())
            .map(|(rho, c)| {
                let mut lane_coeffs = [F::ZERO; D];
                lane_coeffs.copy_from_slice(&c.data[lane * D..(lane + 1) * D]);
                (*rho, lane_coeffs)
            })
            .collect();
        let (out, _) = projection_quotient(&pairs);
        data.extend_from_slice(&out);
    }
    let combined = Commitment {
        d: D,
        kappa: KAPPA,
        data,
    };
    Fixture { rhos, inputs, combined }
}

/// The trusted Toom-3 D² variant accepts the fixture's combined
/// commitment (independent validation of the native division helper),
/// and the projection variant accepts the very same wires.
#[test]
fn projection_variant_matches_the_toom3_variant() {
    let f = fixture(11);

    // Trusted reference: D² materialization.
    let mut b = R1csBuilder::new();
    let wires = alloc_rlc_commitment_inputs(&mut b, &f.rhos, &f.inputs, &f.combined).expect("wires");
    enforce_rlc_commitment_combination(&mut b, &wires);
    assert!(b.is_satisfied(), "Toom-3 variant accepts the honest mix");

    // Projection variant on identical inputs.
    let quotients = rlc_projection_quotients(&f.rhos, &f.inputs).expect("quotients");
    let mut b = R1csBuilder::new();
    let wires = alloc_rlc_commitment_inputs(&mut b, &f.rhos, &f.inputs, &f.combined).expect("wires");
    let beta = KVar::alloc(&mut b, F::from_u64(0xBE7A), F::from_u64(0xCAFE));
    let powers = enforce_beta_ladder(&mut b, beta, D);
    let _q_wires =
        enforce_rlc_commitment_combination_projection(&mut b, &powers, &wires, &quotients).expect("projection");
    assert!(b.is_satisfied(), "projection variant accepts the honest mix");
}

/// Tampering the combined output or a quotient breaks a row —
/// restore re-satisfies, so each rejection is attributable.
#[test]
fn projection_variant_rejects_forged_mix_and_quotient() {
    let f = fixture(23);
    let quotients = rlc_projection_quotients(&f.rhos, &f.inputs).expect("quotients");

    let mut b = R1csBuilder::new();
    let wires = alloc_rlc_commitment_inputs(&mut b, &f.rhos, &f.inputs, &f.combined).expect("wires");
    let beta = KVar::alloc(&mut b, F::from_u64(0x5EED), F::from_u64(0xF00D));
    let powers = enforce_beta_ladder(&mut b, beta, D);
    let q_wires =
        enforce_rlc_commitment_combination_projection(&mut b, &powers, &wires, &quotients).expect("projection");
    assert!(b.is_satisfied(), "honest baseline");

    // Forged combined-commitment coefficient — the attack the check
    // exists for.
    let col = wires.combined_c_data[D + 5].col();
    let honest = f.combined.data[D + 5];
    b.tamper_witness(col, honest + F::ONE);
    assert!(!b.is_satisfied(), "a forged combined coefficient must be rejected");
    b.tamper_witness(col, honest);
    assert!(b.is_satisfied(), "restore");

    // Forged quotient wire.
    let col = q_wires[1][7].col();
    let honest = quotients[1][7];
    b.tamper_witness(col, honest + F::ONE);
    assert!(!b.is_satisfied(), "a forged quotient must be rejected");
}

/// Head-to-head committed-wire cost, same operands. Pinned so a
/// regression reopens the design (the module header's ~450k-row cost
/// note is what this replaces at production shape).
#[test]
fn projection_variant_cost_beats_toom3() {
    let f = fixture(42);

    let mut b_ref = R1csBuilder::new();
    let wires = alloc_rlc_commitment_inputs(&mut b_ref, &f.rhos, &f.inputs, &f.combined).expect("wires");
    enforce_rlc_commitment_combination(&mut b_ref, &wires);
    let (ref_cols, ref_rows) = (b_ref.cols(), b_ref.rows());

    let quotients = rlc_projection_quotients(&f.rhos, &f.inputs).expect("quotients");
    let mut b = R1csBuilder::new();
    let wires = alloc_rlc_commitment_inputs(&mut b, &f.rhos, &f.inputs, &f.combined).expect("wires");
    let beta = KVar::alloc(&mut b, F::from_u64(3), F::from_u64(5));
    let powers = enforce_beta_ladder(&mut b, beta, D);
    let _ = enforce_rlc_commitment_combination_projection(&mut b, &powers, &wires, &quotients).expect("projection");
    let (proj_cols, proj_rows) = (b.cols(), b.rows());

    println!("== Π_RLC commitment mix: Toom-3 D² vs projection (κ = {KAPPA}, pairs = {PAIRS}) ==");
    println!("Toom-3      cols {ref_cols:>7}  rows {ref_rows:>7}");
    println!("projection  cols {proj_cols:>7}  rows {proj_rows:>7}");
    println!(
        "ratio       {:.1}× cols, {:.1}× rows",
        ref_cols as f64 / proj_cols as f64,
        ref_rows as f64 / proj_rows as f64
    );

    assert!(
        proj_cols * 2 <= ref_cols,
        "projection must stay ≥ 2× cheaper in committed wires ({proj_cols} vs {ref_cols})"
    );
}

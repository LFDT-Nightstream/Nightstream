//! Π_RLC commitment combination, projection-checked.
//!
//! The projection variant must accept exactly the combined commitment
//! the trusted Toom-3 D²-materialization variant accepts, reject any
//! tampered output or quotient, and beat it decisively on committed
//! wires. In the gadget tests β is allocated directly; native `pi_rlc`
//! owns the Lemma 5 transcript schedule (absorb inputs → ρ → absorb
//! combined + quotients → squeeze β), driven on a real fold by the
//! schedule tests at the bottom of this file.

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
    let quotients: Vec<_> = rlc_projection_quotients(&f.rhos, &f.inputs)
        .expect("quotients")
        .into_iter()
        .map(|lane| lane.q)
        .collect();
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
    let quotients: Vec<_> = rlc_projection_quotients(&f.rhos, &f.inputs)
        .expect("quotients")
        .into_iter()
        .map(|lane| lane.q)
        .collect();

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

/// The native quotient helper polices input shapes instead of slicing
/// blind — each malformed input must land on its exact error variant,
/// not merely "some error".
#[test]
fn quotient_helper_rejects_malformed_inputs() {
    use neo_fold_clean::paper::reductions::pi_rlc_circuit::Error;

    let f = fixture(31);
    let shape_mismatch = |r: Result<_, Error>| {
        matches!(
            r,
            Err(Error::ShapeMismatch {
                what: "projection input commitment shape",
                ..
            })
        )
    };

    let mut wrong_kappa = f.inputs.clone();
    wrong_kappa[1] = Commitment {
        d: D,
        kappa: KAPPA + 1,
        data: vec![F::ZERO; (KAPPA + 1) * D],
    };
    assert!(shape_mismatch(rlc_projection_quotients(&f.rhos, &wrong_kappa)));

    let mut short_data = f.inputs.clone();
    short_data[2].data.truncate(D);
    assert!(shape_mismatch(rlc_projection_quotients(&f.rhos, &short_data)));

    assert!(matches!(
        rlc_projection_quotients(&f.rhos[..2], &f.inputs),
        Err(Error::PairCountMismatch { rhos: 2, inputs: 3 })
    ));
    assert!(matches!(rlc_projection_quotients(&f.rhos, &[]), Err(Error::Empty)));
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

    let quotients: Vec<_> = rlc_projection_quotients(&f.rhos, &f.inputs)
        .expect("quotients")
        .into_iter()
        .map(|lane| lane.q)
        .collect();
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

// ── Lemma 5 β schedule on the native fold path ───────────────────────────

use neo_ccs::Mat;
use neo_fold_clean::engine::transcript::Transcript;
use neo_fold_clean::frontends::direct_ccs::{self, R1cs};
use neo_fold_clean::paper::construction2::RunningInstance;
use neo_fold_clean::paper::f_prime::projection_trace::{
    encode_projection_identity, encode_projection_pair, encode_projection_shared,
};
use neo_fold_clean::paper::relations::ajtai_rlc_mixer;
use neo_fold_clean::paper::{nifs, pi_rlc};
use neo_fold_clean::{CeClaim, Preprocessing};
use neo_math::K;

fn three_term_addition() -> R1cs {
    let m = D;
    let mut a = Mat::zero(1, m, F::ZERO);
    a.set(0, 1, F::ONE);
    a.set(0, 2, F::ONE);
    let mut b = Mat::zero(1, m, F::ZERO);
    b.set(0, 0, F::ONE);
    let mut c = Mat::zero(1, m, F::ZERO);
    c.set(0, 3, F::ONE);
    R1cs { a, b, c, m_in: D }
}

fn assignment(a: u64, b: u64) -> Vec<F> {
    let mut z = vec![F::ZERO; D];
    z[0] = F::ONE;
    z[1] = F::from_u64(a);
    z[2] = F::from_u64(b);
    z[3] = F::from_u64(a + b);
    z
}

/// Two real folds; returns the second fold's Π_CCS output claims (K+k,
/// non-zero commitments) and their witnesses, ready for a standalone
/// Π_RLC run.
fn real_fold_rlc_fixture() -> (Preprocessing, Vec<CeClaim>, Vec<Mat<F>>) {
    let r1cs = three_term_addition();
    let prep = direct_ccs::preprocess_seeded(&r1cs, 61).expect("preprocess");

    let first = direct_ccs::build_instance(&prep, &r1cs, &assignment(1, 0)).expect("first instance");
    let mut tr = Transcript::session();
    let (running, _first_proof) = nifs::prove(
        &mut tr,
        &prep.params,
        prep.structure(),
        prep.optimized_cache(),
        &prep.log,
        None,
        prep.mix_rhos_commits(),
        prep.combine_b_pows(),
        vec![first],
        &RunningInstance::default(),
    )
    .expect("first NIFS.P");

    let second = direct_ccs::build_instance(&prep, &r1cs, &assignment(0, 1)).expect("second instance");
    let second_z = second.witness.Z.clone();
    let mut tr = Transcript::session();
    let (_next, proof) = nifs::prove(
        &mut tr,
        &prep.params,
        prep.structure(),
        prep.optimized_cache(),
        &prep.log,
        None,
        prep.mix_rhos_commits(),
        prep.combine_b_pows(),
        vec![second],
        &running,
    )
    .expect("second NIFS.P");

    let mut witnesses = Vec::with_capacity(1 + running.witnesses.len());
    witnesses.push(second_z);
    witnesses.extend(running.witnesses.iter().cloned());
    (prep, proof.pi_ccs.outputs.clone(), witnesses)
}

/// The β schedule runs identically on the prove and verify paths: after
/// Π_RLC both transcripts are in lockstep (same downstream challenges),
/// with one quotient per κ lane behind the squeezed β.
#[test]
fn projection_schedule_keeps_prover_and_verifier_in_lockstep() {
    let (prep, claims, witnesses) = real_fold_rlc_fixture();

    let mut tr_p = Transcript::session();
    let (out, proof) = pi_rlc::prove(
        &mut tr_p,
        &prep.params,
        prep.structure(),
        prep.mix_rhos_commits(),
        &claims,
        &witnesses,
    )
    .expect("Π_RLC.P");

    let mut tr_v = Transcript::session();
    let combined = pi_rlc::verify(
        &mut tr_v,
        &prep.params,
        prep.structure(),
        prep.mix_rhos_commits(),
        &claims,
        &proof,
    )
    .expect("Π_RLC.V");

    assert_eq!(
        tr_p.challenge_field(b"post_rlc_probe"),
        tr_v.challenge_field(b"post_rlc_probe"),
        "prove/verify transcripts must stay in lockstep through the β schedule"
    );
    assert_eq!(
        out.projection.q_lanes.len(),
        combined.c.kappa,
        "one division quotient per κ lane"
    );
    assert_eq!(out.projection.rhos.len(), claims.len(), "one ρ per folded claim");
}

fn drifted_mixer(rhos: &[Mat<F>], cs: &[Commitment]) -> Commitment {
    let mut c = ajtai_rlc_mixer(rhos, cs);
    c.data[0] += F::ONE;
    c
}

/// Wire-identity fail-closed: a mixer whose output is not Σρ_i·c_i —
/// even off by one coefficient — must be rejected before anything is
/// absorbed, because the projection identity would then describe a
/// commitment the fold never produced.
#[test]
fn projection_schedule_rejects_mixer_that_is_not_the_ring_action() {
    let (prep, claims, witnesses) = real_fold_rlc_fixture();
    let mut tr = Transcript::session();
    let err = pi_rlc::prove(
        &mut tr,
        &prep.params,
        prep.structure(),
        drifted_mixer,
        &claims,
        &witnesses,
    )
    .expect_err("a non-ring-action mixer must fail the wire-identity check");
    assert!(
        matches!(err, pi_rlc::Error::ProjectionMixDrift { lane: 0 }),
        "expected ProjectionMixDrift at lane 0, got {err:?}"
    );
}

/// Wire-identity bridge (Lemma 5 audit item 1): a real fold's recorded
/// schedule, pushed through the F' projection-region encoders, satisfies
/// the batched identity with zero residual — per κ lane the pairs are
/// (ρ_i, c_i lane), the encoder's quotient is exactly the
/// transcript-absorbed q_lane, and the identity output is exactly the
/// combined commitment's lane.
#[test]
fn projection_schedule_bridges_to_the_f_prime_encoders() {
    let (prep, claims, witnesses) = real_fold_rlc_fixture();
    let mut tr = Transcript::session();
    let (out, _proof) = pi_rlc::prove(
        &mut tr,
        &prep.params,
        prep.structure(),
        prep.mix_rhos_commits(),
        &claims,
        &witnesses,
    )
    .expect("Π_RLC.P");

    let schedule = &out.projection;
    let kappa = out.claim.c.kappa;
    let (_shared_lanes, powers) = encode_projection_shared(schedule.beta);

    for lane in 0..kappa {
        let pairs: Vec<([F; D], [F; D])> = schedule
            .rhos
            .iter()
            .zip(claims.iter())
            .map(|(rho, claim)| {
                let mut c_lane = [F::ZERO; D];
                c_lane.copy_from_slice(&claim.c.data[lane * D..(lane + 1) * D]);
                (*rho, c_lane)
            })
            .collect();

        let mut terms = Vec::new();
        for (rho, c) in &pairs {
            let (_lanes, term) = encode_projection_pair(rho, c, &powers);
            terms.push(term);
        }
        let (_identity_lanes, residual) = encode_projection_identity(&pairs, &powers, &terms);
        assert_eq!(
            residual,
            K::ZERO,
            "lane {lane}: real-fold fill must satisfy the projection identity"
        );

        let (mix_out, q) = projection_quotient(&pairs);
        assert_eq!(
            q, schedule.q_lanes[lane],
            "lane {lane}: the transcript-absorbed q must be the encoder\'s q"
        );
        assert_eq!(
            &mix_out[..],
            &out.claim.c.data[lane * D..(lane + 1) * D],
            "lane {lane}: the identity output must be the combined commitment lane"
        );
    }
}

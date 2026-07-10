//! Π_CCS.V SplitNcV1 in-circuit verifier — parity tests against
//! `neo_reductions::engines::optimized_engine` internals.
//!
//! Each sub-step of `paper/reductions/pi_ccs_split_nc_circuit.rs` ships with
//! a parity test in this file. Once the whole verifier is composed, the
//! final test runs a real `nifs::prove` and feeds its proof to the
//! in-circuit verifier — that's the hard gate for "F' embeds NIFS.V".
//!
//! Layout:
//!   - sub-step B: K-batch challenge sampling (this file).
//!   - sub-step C onwards: header/instance binding, ME inputs, χ_α-driven
//!     terminals, FE+NC sumchecks, full verifier.

use neo_ajtai::Commitment;
use neo_ccs::{CeClaim as NeoCeClaim, Mat};
use neo_fold_clean::engine::r1cs_circuit::builder::Var;
use neo_fold_clean::engine::r1cs_circuit::field_ext::KVar;
use neo_fold_clean::engine::r1cs_circuit::{Lc, R1csBuilder, TranscriptGadget};
use neo_fold_clean::paper::digest::{
    accumulator_ce_claim_digest, accumulator_digest_from_running_parts, ccs_claim_digest, ce_claim_digest,
    digest32_as_fields, pi_ccs_instance_digest, pi_ccs_instance_digest_parent_authority, pi_ccs_outputs_digest,
};
use neo_fold_clean::paper::reductions::accumulator_digest_circuit::enforce_accumulator_digest_from_running_circuit;
use neo_fold_clean::paper::reductions::pi_ccs_split_nc_circuit::{
    absorb_engine_header_bundle_and_instance_digest, absorb_engine_me_inputs_accumulator_handle,
    enforce_accumulator_ce_claim_digest, enforce_ccs_claim_digest, enforce_ce_claim_digest, enforce_fe_claimed_initial,
    enforce_header_digest_catch_up, enforce_pi_ccs_instance_digest, enforce_pi_ccs_instance_digest_parent_authority,
    enforce_pi_ccs_outputs_digest, header_digest_bytes_to_fields, sample_engine_beta_m, sample_engine_challenges,
    AccumulatorCeClaimDigestInputs, CeClaimDigestInputs, FeClaimedInitialInputs, PiCcsOutputClaimDigestInputs,
};
use neo_math::ring::D;
use neo_math::{from_complex, KExtensions, F, K};
use neo_reductions::engines::utils::{
    bind_me_inputs_accumulator_handle, sample_beta_m, sample_challenges, PI_CCS_HEADER_BUNDLE_RAW_TAG,
    PI_CCS_INSTANCE_DIGEST_RAW_TAG,
};
use neo_transcript::{Poseidon2Transcript, Transcript};
use p3_field::{PrimeCharacteristicRing, PrimeField64};

type CeClaim = NeoCeClaim<Commitment, F, K>;

const APP: &[u8] = b"neo.test.pi_ccs.split_nc/v1";

fn k_value(b: &R1csBuilder, v: KVar) -> K {
    let c0 = b.witness()[v.c0.col()];
    let c1 = b.witness()[v.c1.col()];
    K::from_coeffs([c0, c1])
}

fn pin_native(b: &mut R1csBuilder, var: KVar, native: K) {
    let [c0, c1] = native.as_coeffs();
    b.enforce_eq(&Lc::from_var(var.c0), &Lc::from_const(c0));
    b.enforce_eq(&Lc::from_var(var.c1), &Lc::from_const(c1));
}

fn alloc_witness_var(b: &mut R1csBuilder, v: F) -> Var {
    b.alloc(v)
}

fn alloc_witness_k(b: &mut R1csBuilder, v: K) -> KVar {
    let [c0, c1] = v.as_coeffs();
    KVar::alloc(b, c0, c1)
}

/// Construct a deterministic CE claim for digest parity testing. `m_in` and
/// `(t, d)` shapes are chosen by the caller. The X matrix is `D × m_in` per
/// the native `me_input_projection_digest_poseidon_into` contract; values
/// are filled with a counter so failures point at specific offsets.
fn build_test_ce_claim(seed: u64, m_in: usize, t: usize, d: usize, kappa: usize, r_len: usize) -> CeClaim {
    let mut s = seed;
    let mut next_f = || -> F {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        F::from_u64(s & 0xFFFF)
    };

    let c_data: Vec<F> = (0..(D * kappa)).map(|_| next_f()).collect();
    let c = Commitment {
        d: D,
        kappa,
        data: c_data,
    };

    // X: D × m_in matrix. SuperNeo `project_x_from_witness_mat` populates
    // only the first `ceil(m_in / D)` ring columns; the rest are structural
    // zeros. Mirror that here so both native `ce_claim_digest` (which
    // hashes only active cols) and the circuit version (which enforces
    // inactive cols are zero) agree on this synthetic shape.
    let active_cols = neo_fold_clean::paper::relations::superneo_public_x_cols(m_in);
    let mut x = Mat::zero(D, m_in, F::ZERO);
    for col in 0..active_cols {
        for row in 0..D {
            x.set(row, col, next_f());
        }
    }

    let r: Vec<K> = (0..r_len)
        .map(|_| K::from_coeffs([next_f(), next_f()]))
        .collect();
    let y_ring: Vec<Vec<K>> = (0..t)
        .map(|_| {
            (0..d)
                .map(|_| K::from_coeffs([next_f(), next_f()]))
                .collect()
        })
        .collect();

    CeClaim {
        adv: None,
        c,
        X: x,
        r,
        s_col: Vec::new(),
        y_ring,
        ct: Vec::new(),
        aux_openings: Vec::new(),
        y_zcol: Vec::new(),
        m_in,
        fold_digest: [0u8; 32],
        c_step_coords: Vec::new(),
        u_offset: 0,
        u_len: 0,
    }
}

fn build_test_accumulator_ce_claim(seed: u64) -> CeClaim {
    let mut claim = build_test_ce_claim(
        seed, /*m_in*/ 4, /*t*/ 3, /*d*/ 4, /*kappa*/ 2, /*r_len*/ 3,
    );
    let mut s = seed ^ 0xACC0_5EED;
    let mut next_f = || -> F {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        F::from_u64(s & 0xFFFF)
    };

    claim.s_col = (0..2)
        .map(|_| K::from_coeffs([next_f(), next_f()]))
        .collect();
    claim.ct = claim.y_ring.iter().map(|row| row[0]).collect();
    claim.y_zcol = (0..4)
        .map(|_| K::from_coeffs([next_f(), next_f()]))
        .collect();
    for (idx, byte) in claim.fold_digest.iter_mut().enumerate() {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *byte = ((s >> 8) as u8).wrapping_add(idx as u8);
    }
    claim
}

fn enforce_accumulator_claim_digest_for_test(b: &mut R1csBuilder, claim: &CeClaim) -> [Var; 4] {
    let c_data_vars: Vec<Var> = claim
        .c
        .data
        .iter()
        .map(|&v| alloc_witness_var(b, v))
        .collect();
    let mut x_flat_vars: Vec<Var> = Vec::with_capacity(claim.X.rows() * claim.X.cols());
    for r in 0..claim.X.rows() {
        for c in 0..claim.X.cols() {
            x_flat_vars.push(alloc_witness_var(b, claim.X[(r, c)]));
        }
    }
    let r_vars: Vec<KVar> = claim
        .r
        .iter()
        .copied()
        .map(|v| alloc_witness_k(b, v))
        .collect();
    let s_col_vars: Vec<KVar> = claim
        .s_col
        .iter()
        .copied()
        .map(|v| alloc_witness_k(b, v))
        .collect();
    let y_ring_vars: Vec<Vec<KVar>> = claim
        .y_ring
        .iter()
        .map(|row| row.iter().copied().map(|v| alloc_witness_k(b, v)).collect())
        .collect();
    let ct_vars: Vec<KVar> = claim
        .ct
        .iter()
        .copied()
        .map(|v| alloc_witness_k(b, v))
        .collect();
    let fold_lanes = digest32_as_fields(claim.fold_digest);
    let fold_digest_wires: [Var; 4] = std::array::from_fn(|i| alloc_witness_var(b, fold_lanes[i]));

    enforce_accumulator_ce_claim_digest(
        b,
        &AccumulatorCeClaimDigestInputs {
            c_d: claim.c.d,
            c_kappa: claim.c.kappa,
            c_data: &c_data_vars,
            x_rows: claim.X.rows(),
            x_cols: claim.X.cols(),
            x_flat_row_major: &x_flat_vars,
            r: &r_vars,
            s_col: &s_col_vars,
            y_ring: &y_ring_vars,
            ct: &ct_vars,
            m_in: claim.m_in,
            fold_digest_fields: fold_digest_wires,
            adv: None,
        },
    )
    .expect("accumulator CE claim digest")
}

// ── sub-step B: K-batch challenge sampling ───────────────────────────────

#[test]
fn sample_engine_challenges_matches_native_for_small_dims() {
    // Exercise a handful of `(ell_d, ell_n)` combos. The native
    // `sample_challenges` always splits the batch as
    // `[α(ell_d) | β_a(ell_d) | β_r(ell_n) | γ(1)]`.
    for &(ell_d, ell_n) in &[(1usize, 1usize), (2, 3), (3, 4), (4, 6), (6, 6)] {
        let mut native_tr = Poseidon2Transcript::new(APP);
        let ch = sample_challenges(&mut native_tr, ell_d, ell_n + ell_d).expect("native sample_challenges");
        assert_eq!(ch.alpha.len(), ell_d, "(ell_d={ell_d}, ell_n={ell_n}) α len");
        assert_eq!(ch.beta_a.len(), ell_d, "(ell_d={ell_d}, ell_n={ell_n}) β_a len");
        assert_eq!(ch.beta_r.len(), ell_n, "(ell_d={ell_d}, ell_n={ell_n}) β_r len");

        let mut b = R1csBuilder::new();
        let mut tr = TranscriptGadget::new(&mut b, APP);
        let gadget = sample_engine_challenges(&mut b, &mut tr, ell_d, ell_n);

        // Length parity.
        assert_eq!(gadget.alpha.len(), ell_d);
        assert_eq!(gadget.beta_a.len(), ell_d);
        assert_eq!(gadget.beta_r.len(), ell_n);

        // Value parity for each lane.
        for (i, var) in gadget.alpha.iter().enumerate() {
            assert_eq!(k_value(&b, *var), ch.alpha[i], "α[{i}]");
            pin_native(&mut b, *var, ch.alpha[i]);
        }
        for (i, var) in gadget.beta_a.iter().enumerate() {
            assert_eq!(k_value(&b, *var), ch.beta_a[i], "β_a[{i}]");
            pin_native(&mut b, *var, ch.beta_a[i]);
        }
        for (i, var) in gadget.beta_r.iter().enumerate() {
            assert_eq!(k_value(&b, *var), ch.beta_r[i], "β_r[{i}]");
            pin_native(&mut b, *var, ch.beta_r[i]);
        }
        assert_eq!(k_value(&b, gadget.gamma), ch.gamma, "γ");
        pin_native(&mut b, gadget.gamma, ch.gamma);

        assert!(b.is_satisfied(), "(ell_d={ell_d}, ell_n={ell_n}) parity");
    }
}

// ── sub-step C: header-bundle + instance-digest absorbs ─────────────────

#[test]
fn absorb_engine_header_bundle_and_instance_digest_matches_native_raw_absorbs() {
    // Mirror `bind_header_and_instance_digest_with_digest` at the raw-absorb
    // level: the native function performs exactly
    //     tr.append_fields_raw(&[11, hb[0..4]])
    //     tr.append_fields_raw(&[12, id[0..4]])
    // using `PI_CCS_HEADER_BUNDLE_RAW_TAG = 11` and
    // `PI_CCS_INSTANCE_DIGEST_RAW_TAG = 12`. The full helper goes through
    // `pi_ccs_header_bundle_digest_fields(params, s, dims, mat_digest)` to
    // produce `hb`, but the wire-side absorb pattern is identical. We
    // exercise that pattern here with arbitrary `(hb, id)` values; the
    // integration test in sub-step L will pin them to real native values.
    use neo_math::F;
    use p3_field::PrimeCharacteristicRing;

    let hb = [
        F::from_u64(11_111),
        F::from_u64(22_222),
        F::from_u64(33_333),
        F::from_u64(44_444),
    ];
    let id = [
        F::from_u64(0xDEAD),
        F::from_u64(0xBEEF),
        F::from_u64(0xCAFE),
        F::from_u64(0xBABE),
    ];

    // Native: drive a transcript through the same two raw absorbs that
    // `bind_header_and_instance_digest_with_digest` performs.
    let mut native_tr = Poseidon2Transcript::new(APP);
    native_tr.append_fields_raw(&[F::from_u64(PI_CCS_HEADER_BUNDLE_RAW_TAG), hb[0], hb[1], hb[2], hb[3]]);
    native_tr.append_fields_raw(&[F::from_u64(PI_CCS_INSTANCE_DIGEST_RAW_TAG), id[0], id[1], id[2], id[3]]);
    // Run a downstream challenge sample to expose any transcript-state drift.
    let native_post_ch = sample_challenges(&mut native_tr, 3, 3 + 4).unwrap();

    // Circuit: same two absorbs via the gadget, then mirror the downstream
    // sample. The instance-digest wires are passed as witness `Var`s that
    // we pin to the native values up front (the F' caller would normally
    // pin them via a digest gadget; here we do it directly).
    let mut b = R1csBuilder::new();
    let mut tr = TranscriptGadget::new(&mut b, APP);
    let id_vars: [_; 4] = std::array::from_fn(|i| {
        let v = b.alloc(id[i]);
        use neo_fold_clean::engine::r1cs_circuit::Lc;
        b.enforce_eq(&Lc::from_var(v), &Lc::from_const(id[i]));
        v
    });
    absorb_engine_header_bundle_and_instance_digest(&mut b, &mut tr, hb, id_vars);
    let gadget_post = sample_engine_challenges(&mut b, &mut tr, 3, 4);

    // Each squeezed challenge must match the native one — that's the
    // transcript-state parity check.
    for (i, var) in gadget_post.alpha.iter().enumerate() {
        assert_eq!(k_value(&b, *var), native_post_ch.alpha[i], "α[{i}] after hb+id");
        pin_native(&mut b, *var, native_post_ch.alpha[i]);
    }
    for (i, var) in gadget_post.beta_a.iter().enumerate() {
        pin_native(&mut b, *var, native_post_ch.beta_a[i]);
    }
    for (i, var) in gadget_post.beta_r.iter().enumerate() {
        pin_native(&mut b, *var, native_post_ch.beta_r[i]);
    }
    pin_native(&mut b, gadget_post.gamma, native_post_ch.gamma);

    assert!(
        b.is_satisfied(),
        "absorb_engine_header_bundle_and_instance_digest must produce native transcript state"
    );
}

#[test]
fn header_bundle_tag_mismatch_breaks_downstream_challenge_pin() {
    // The header-bundle / instance-digest tags (11 and 12) are part of
    // domain separation. If a future refactor swaps them, the transcript
    // state diverges and downstream challenges diverge. The const-bound
    // wires for the tags must enforce this — tampering one of those wires
    // must break some equality.
    use neo_math::F;
    use p3_field::PrimeCharacteristicRing;

    let hb = [F::from_u64(7), F::from_u64(9), F::from_u64(11), F::from_u64(13)];
    let id = [F::from_u64(101), F::from_u64(103), F::from_u64(107), F::from_u64(109)];

    let mut native_tr = Poseidon2Transcript::new(APP);
    native_tr.append_fields_raw(&[F::from_u64(PI_CCS_HEADER_BUNDLE_RAW_TAG), hb[0], hb[1], hb[2], hb[3]]);
    native_tr.append_fields_raw(&[F::from_u64(PI_CCS_INSTANCE_DIGEST_RAW_TAG), id[0], id[1], id[2], id[3]]);
    let native_ch = sample_challenges(&mut native_tr, 2, 2 + 2).unwrap();

    let mut b = R1csBuilder::new();
    let mut tr = TranscriptGadget::new(&mut b, APP);
    let id_vars: [_; 4] = std::array::from_fn(|i| {
        let v = b.alloc(id[i]);
        use neo_fold_clean::engine::r1cs_circuit::Lc;
        b.enforce_eq(&Lc::from_var(v), &Lc::from_const(id[i]));
        v
    });
    absorb_engine_header_bundle_and_instance_digest(&mut b, &mut tr, hb, id_vars);
    let gadget = sample_engine_challenges(&mut b, &mut tr, 2, 2);
    pin_native(&mut b, gadget.alpha[0], native_ch.alpha[0]);

    assert!(b.is_satisfied(), "baseline parity");

    // Locate the const-bound wire for the header-bundle tag (= 11). The
    // tag wire is one of the first allocations made by
    // `absorb_engine_header_bundle_and_instance_digest`. We don't know its
    // exact column without parsing the trace, so we sweep witness columns
    // and tamper the first one whose value equals the header-bundle tag and
    // that sits below the challenge wires.
    let target_value = F::from_u64(PI_CCS_HEADER_BUNDLE_RAW_TAG);
    let challenge_col = gadget.alpha[0].c0.col();
    let mut target_col: Option<usize> = None;
    for col in 1..challenge_col {
        if b.witness()[col] == target_value {
            target_col = Some(col);
            break;
        }
    }
    let target_col = target_col.expect("could not locate header-bundle tag wire");
    let tampered = target_value + F::from_u64(1);
    b.tamper_witness(target_col, tampered);

    assert!(
        !b.is_satisfied(),
        "tampering the header-bundle tag wire must break some constraint"
    );
}

// ── sub-step D: ME-input accumulator-handle absorb ───────────────────────

#[test]
fn absorb_engine_me_inputs_accumulator_handle_matches_native_bind() {
    // Council-required parity test: after the handle absorb, native
    // `bind_me_inputs_accumulator_handle` and circuit
    // `absorb_engine_me_inputs_accumulator_handle` must leave the
    // transcript in the same state. We verify by running a downstream
    // challenge sample on each and comparing limb-for-limb. This pins
    // the entire transcript schedule for the new mode without
    // depending on the full SplitNc Π_CCS verifier path.
    let me_input_count = 14usize;
    let handle = [
        F::from_u64(0xA0A0_A0A0),
        F::from_u64(0xB1B1_B1B1),
        F::from_u64(0xC2C2_C2C2),
        F::from_u64(0xD3D3_D3D3),
    ];

    // Native: domain + count + accumulator-handle absorb, then sample.
    let mut native_tr = Poseidon2Transcript::new(APP);
    bind_me_inputs_accumulator_handle(&mut native_tr, me_input_count, &handle)
        .expect("native bind_me_inputs_accumulator_handle");
    let native_post = sample_challenges(&mut native_tr, 3, 3 + 4).expect("native sample after handle");

    // Circuit: same three raw absorbs via the gadget, then matching
    // downstream sample. Handle wires are pinned to the native values
    // up front (the SplitNc caller normally pins them via the
    // accumulator-digest gadget; here we do it directly).
    let mut b = R1csBuilder::new();
    let mut tr = TranscriptGadget::new(&mut b, APP);
    let handle_vars: [Var; 4] = std::array::from_fn(|i| {
        let v = b.alloc(handle[i]);
        b.enforce_eq(&Lc::from_var(v), &Lc::from_const(handle[i]));
        v
    });
    absorb_engine_me_inputs_accumulator_handle(&mut b, &mut tr, me_input_count, handle_vars);
    let gadget_post = sample_engine_challenges(&mut b, &mut tr, 3, 4);

    for (i, var) in gadget_post.alpha.iter().enumerate() {
        assert_eq!(
            k_value(&b, *var),
            native_post.alpha[i],
            "α[{i}] after accumulator-handle absorb"
        );
        pin_native(&mut b, *var, native_post.alpha[i]);
    }
    for (i, var) in gadget_post.beta_a.iter().enumerate() {
        pin_native(&mut b, *var, native_post.beta_a[i]);
    }
    for (i, var) in gadget_post.beta_r.iter().enumerate() {
        pin_native(&mut b, *var, native_post.beta_r[i]);
    }
    pin_native(&mut b, gadget_post.gamma, native_post.gamma);

    assert!(
        b.is_satisfied(),
        "absorb_engine_me_inputs_accumulator_handle must reproduce native transcript state"
    );
}

#[test]
fn sample_engine_beta_m_matches_native_for_various_ell_m() {
    for &ell_m in &[1usize, 2, 4, 5, 6, 10] {
        let mut native_tr = Poseidon2Transcript::new(APP);
        let native_beta_m: Vec<K> = sample_beta_m(&mut native_tr, ell_m).expect("native sample_beta_m");
        assert_eq!(native_beta_m.len(), ell_m);

        let mut b = R1csBuilder::new();
        let mut tr = TranscriptGadget::new(&mut b, APP);
        let gadget = sample_engine_beta_m(&mut b, &mut tr, ell_m);
        assert_eq!(gadget.len(), ell_m);
        for (i, var) in gadget.iter().enumerate() {
            assert_eq!(k_value(&b, *var), native_beta_m[i], "ell_m={ell_m} β_m[{i}]");
            pin_native(&mut b, *var, native_beta_m[i]);
        }
        assert!(b.is_satisfied(), "ell_m={ell_m} parity");
    }
}

#[test]
fn sample_engine_challenges_matches_native_after_nonempty_prior_absorbs() {
    // In production, `sample_challenges` runs AFTER `bind_header_and_instance_digest`
    // and `bind_me_inputs_accumulator_handle`. The sponge state at that point is mid-rate with
    // some `absorbed > 0`, so the `[2]` domain tag may or may not trigger a
    // permute before the squeeze loop. This test fakes a "prior absorbs"
    // prefix of varying length and shapes to catch rate-boundary bugs.
    use neo_fold_clean::engine::r1cs_circuit::Lc;
    use neo_math::F;

    // Each entry picks an absorb prefix length in fields. Lengths around
    // RATE (4), 2·RATE (8), 3·RATE (12) and other irregular sizes exercise
    // both freshly-permuted-and-empty rate and mid-rate cursors.
    use p3_field::PrimeCharacteristicRing;
    for &prefix_len in &[0usize, 1, 3, 4, 5, 7, 8, 9, 11, 12, 13, 16, 20] {
        let prefix: Vec<F> = (0..prefix_len)
            .map(|i| F::from_u64((i as u64).wrapping_mul(0x9E37_79B9) + 7))
            .collect();

        // Native side: absorb the prefix, then sample_challenges.
        let mut native_tr = Poseidon2Transcript::new(APP);
        native_tr.append_fields_raw(&prefix);
        let ch = sample_challenges(&mut native_tr, /*ell_d*/ 3, /*ell*/ 3 + 5).unwrap();

        // Circuit side: mirror — absorb the prefix as witness vars, then sample.
        let mut b = R1csBuilder::new();
        let mut tr = TranscriptGadget::new(&mut b, APP);
        let prefix_vars: Vec<_> = prefix.iter().map(|&v| b.alloc(v)).collect();
        tr.append_fields_raw_vars(&mut b, &prefix_vars);
        // Pin each prefix wire to its const value so an adversary can't
        // tamper its way back into a happy-path sponge state.
        for (i, var) in prefix_vars.iter().enumerate() {
            b.enforce_eq(&Lc::from_var(*var), &Lc::from_const(prefix[i]));
        }
        let gadget = sample_engine_challenges(&mut b, &mut tr, 3, 5);

        for (i, var) in gadget.alpha.iter().enumerate() {
            assert_eq!(k_value(&b, *var), ch.alpha[i], "prefix_len={prefix_len} α[{i}]");
            pin_native(&mut b, *var, ch.alpha[i]);
        }
        for (i, var) in gadget.beta_a.iter().enumerate() {
            pin_native(&mut b, *var, ch.beta_a[i]);
        }
        for (i, var) in gadget.beta_r.iter().enumerate() {
            pin_native(&mut b, *var, ch.beta_r[i]);
        }
        pin_native(&mut b, gadget.gamma, ch.gamma);

        assert!(
            b.is_satisfied(),
            "prefix_len={prefix_len}: challenges must match native after prior absorbs"
        );
    }
}

#[test]
fn challenges_and_beta_m_compose_with_full_native_sequence() {
    // The native verifier runs `sample_challenges(ell_d, ell)` *then*
    // `sample_beta_m(ell_m)` back-to-back. Both share the same transcript
    // state — so any divergence in either step would corrupt the other.
    let ell_d = 4;
    let ell_n = 6;
    let ell_m = 5;

    let mut native_tr = Poseidon2Transcript::new(APP);
    let ch = sample_challenges(&mut native_tr, ell_d, ell_n + ell_d).unwrap();
    let beta_m = sample_beta_m(&mut native_tr, ell_m).unwrap();

    let mut b = R1csBuilder::new();
    let mut tr = TranscriptGadget::new(&mut b, APP);
    let gch = sample_engine_challenges(&mut b, &mut tr, ell_d, ell_n);
    let gbm = sample_engine_beta_m(&mut b, &mut tr, ell_m);

    for (i, v) in gch.alpha.iter().enumerate() {
        pin_native(&mut b, *v, ch.alpha[i]);
    }
    for (i, v) in gch.beta_a.iter().enumerate() {
        pin_native(&mut b, *v, ch.beta_a[i]);
    }
    for (i, v) in gch.beta_r.iter().enumerate() {
        pin_native(&mut b, *v, ch.beta_r[i]);
    }
    pin_native(&mut b, gch.gamma, ch.gamma);
    for (i, v) in gbm.iter().enumerate() {
        pin_native(&mut b, *v, beta_m[i]);
    }

    assert!(b.is_satisfied(), "challenges + β_m compose with native sequence");

    // Sanity: also confirm the squeezed lanes correspond to K via from_complex.
    // (Belt-and-suspenders against any future encoding drift.)
    let alpha_check: Vec<K> = ch.alpha.clone();
    for (i, v) in gch.alpha.iter().enumerate() {
        let c0 = b.witness()[v.c0.col()];
        let c1 = b.witness()[v.c1.col()];
        assert_eq!(from_complex(c0, c1), alpha_check[i]);
    }
}

// ── sub-step D-rev: paper-layer per-claim digest gadgets ─────────────────

#[test]
fn enforce_ccs_claim_digest_matches_native_paper_layer() {
    // Native paper-layer `ccs_claim_digest(claim)` pushes:
    //   pack_bytes(b"neo.fold.clean/ccs_claim_digest/v1")
    //   | c.d | c.kappa | c.data.len | c.data | x.len | x | m_in
    // The in-circuit mirror must produce byte-identical preimage and
    // therefore the same Poseidon2 digest.
    use neo_ajtai::Commitment;
    use neo_ccs::CcsClaim;

    let mut s: u64 = 0xC11C5;
    let mut next_f = || -> F {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        F::from_u64(s & 0xFFFF)
    };

    let c_d = D;
    let c_kappa = 2usize;
    let c_data: Vec<F> = (0..(c_d * c_kappa)).map(|_| next_f()).collect();
    let x: Vec<F> = (0..3).map(|_| next_f()).collect();
    let m_in = 3usize;

    let native_claim = CcsClaim::<Commitment, F> {
        adv: None,
        c: Commitment {
            d: c_d,
            kappa: c_kappa,
            data: c_data.clone(),
        },
        x: x.clone(),
        m_in,
    };
    let native_digest = ccs_claim_digest(&native_claim);

    let mut b = R1csBuilder::new();
    let c_data_vars: Vec<Var> = c_data
        .iter()
        .map(|&v| alloc_witness_var(&mut b, v))
        .collect();
    let x_vars: Vec<Var> = x.iter().map(|&v| alloc_witness_var(&mut b, v)).collect();
    let digest_wires = enforce_ccs_claim_digest(&mut b, c_d, c_kappa, &c_data_vars, &x_vars, m_in, None);

    for (i, var) in digest_wires.iter().enumerate() {
        assert_eq!(b.witness()[var.col()], native_digest[i], "lane {i}");
        b.enforce_eq(&Lc::from_var(*var), &Lc::from_const(native_digest[i]));
    }
    assert!(
        b.is_satisfied(),
        "ccs_claim_digest parity (first bad row: {:?})",
        b.first_unsatisfied_row()
    );
}

#[test]
fn enforce_ce_claim_digest_matches_native_paper_layer() {
    // Native paper-layer `ce_claim_digest(claim)` pushes:
    //   pack_bytes(b"neo.fold.clean/ce_claim_digest/v2")
    //   | c.d | c.kappa | c.data.len | c.data
    //   | X.rows | X.cols | active_x_cols | X_active(rows × active_x_cols entries)
    //   | r.len | r flat (c0,c1,c0,c1,…)
    //   | y_ring.len | for each row: row.len | row flat (c0,c1,…)
    //   | m_in | digest32_as_fields(fold_digest)
    // Active cols only; inactive cols (>= ceil(m_in / D)) are required to
    // be zero — see `superneo_inactive_x_zero`.
    // The in-circuit mirror must produce byte-identical preimage.
    use neo_fold_clean::paper::digest::digest32_as_fields;

    let claim = build_test_ce_claim(
        0xCEDE, /*m_in*/ 4, /*t*/ 3, /*d*/ 4, /*kappa*/ 1, /*r_len*/ 3,
    );
    let native_digest = ce_claim_digest(&claim);

    let mut b = R1csBuilder::new();

    // Witness wires for the CE claim fields.
    let c_data_vars: Vec<Var> = claim
        .c
        .data
        .iter()
        .map(|&v| alloc_witness_var(&mut b, v))
        .collect();
    // X flattened row-major (native loop order: for r in rows: for c in cols: push).
    let mut x_flat_vars: Vec<Var> = Vec::with_capacity(claim.X.rows() * claim.X.cols());
    for r in 0..claim.X.rows() {
        for c in 0..claim.X.cols() {
            x_flat_vars.push(alloc_witness_var(&mut b, claim.X[(r, c)]));
        }
    }
    let r_vars: Vec<KVar> = claim
        .r
        .iter()
        .copied()
        .map(|v| alloc_witness_k(&mut b, v))
        .collect();
    let y_ring_vars: Vec<Vec<KVar>> = claim
        .y_ring
        .iter()
        .map(|row| {
            row.iter()
                .copied()
                .map(|v| alloc_witness_k(&mut b, v))
                .collect()
        })
        .collect();
    // fold_digest is 32 bytes → 4 F lanes via the same formula the native
    // uses (`digest32_as_fields`). Pin those lanes as witness wires.
    let fold_lanes = digest32_as_fields(claim.fold_digest);
    let fold_digest_wires: [Var; 4] = std::array::from_fn(|i| alloc_witness_var(&mut b, fold_lanes[i]));

    let inputs = CeClaimDigestInputs {
        c_d: claim.c.d,
        c_kappa: claim.c.kappa,
        c_data: &c_data_vars,
        x_rows: claim.X.rows(),
        x_cols: claim.X.cols(),
        x_flat_row_major: &x_flat_vars,
        r: &r_vars,
        y_ring: &y_ring_vars,
        m_in: claim.m_in,
        fold_digest_fields: fold_digest_wires,
        adv: None,
    };
    let digest = enforce_ce_claim_digest(&mut b, &inputs).expect("CE claim digest");

    for (i, var) in digest.iter().enumerate() {
        assert_eq!(b.witness()[var.col()], native_digest[i], "lane {i}");
        b.enforce_eq(&Lc::from_var(*var), &Lc::from_const(native_digest[i]));
    }
    assert!(
        b.is_satisfied(),
        "ce_claim_digest parity (first bad row: {:?})",
        b.first_unsatisfied_row()
    );
}

#[test]
fn enforce_ce_claim_digest_rejects_nonzero_inactive_x() {
    // v2 ce_claim_digest skips X cols >= ceil(m_in / D); the circuit
    // enforces those cols are zero so a prover can't smuggle data into
    // them. Use a wide X (m_in such that inactive cols exist) and tamper
    // one inactive slot — circuit must reject.
    use neo_fold_clean::paper::digest::digest32_as_fields;

    // m_in chosen so active_cols < x_cols: with D = 54 and m_in = 2,
    // active = ceil(2/54) = 1, so col 1 is inactive.
    let claim = build_test_ce_claim(
        0xCEFA, /*m_in*/ 2, /*t*/ 2, /*d*/ 4, /*kappa*/ 1, /*r_len*/ 2,
    );

    let mut b = R1csBuilder::new();
    let c_data_vars: Vec<Var> = claim
        .c
        .data
        .iter()
        .map(|&v| alloc_witness_var(&mut b, v))
        .collect();
    let mut x_flat_vars: Vec<Var> = Vec::with_capacity(claim.X.rows() * claim.X.cols());
    for r in 0..claim.X.rows() {
        for c in 0..claim.X.cols() {
            x_flat_vars.push(alloc_witness_var(&mut b, claim.X[(r, c)]));
        }
    }
    let r_vars: Vec<KVar> = claim
        .r
        .iter()
        .copied()
        .map(|v| alloc_witness_k(&mut b, v))
        .collect();
    let y_ring_vars: Vec<Vec<KVar>> = claim
        .y_ring
        .iter()
        .map(|row| {
            row.iter()
                .copied()
                .map(|v| alloc_witness_k(&mut b, v))
                .collect()
        })
        .collect();
    let fold_lanes = digest32_as_fields(claim.fold_digest);
    let fold_digest_wires: [Var; 4] = std::array::from_fn(|i| alloc_witness_var(&mut b, fold_lanes[i]));

    let inputs = CeClaimDigestInputs {
        c_d: claim.c.d,
        c_kappa: claim.c.kappa,
        c_data: &c_data_vars,
        x_rows: claim.X.rows(),
        x_cols: claim.X.cols(),
        x_flat_row_major: &x_flat_vars,
        r: &r_vars,
        y_ring: &y_ring_vars,
        m_in: claim.m_in,
        fold_digest_fields: fold_digest_wires,
        adv: None,
    };
    enforce_ce_claim_digest(&mut b, &inputs).expect("emit");
    assert!(b.is_satisfied(), "baseline (honest, all-zero inactive) must satisfy");

    // Tamper one inactive slot (row 0, col = active_cols).
    let active_cols = neo_fold_clean::paper::relations::superneo_public_x_cols(claim.m_in);
    assert!(
        active_cols < claim.X.cols(),
        "test setup: expected at least one inactive col"
    );
    let target_col = x_flat_vars[0 * claim.X.cols() + active_cols].col();
    b.tamper_witness(target_col, F::ONE);
    assert!(
        !b.is_satisfied(),
        "ce_claim_digest circuit must reject non-zero inactive X"
    );
}

#[test]
fn enforce_accumulator_ce_claim_digest_matches_native_authority_fields() {
    // This is the HyperNova U_i handle's per-claim building block. Unlike
    // the paper-layer `ce_claim_digest`, it must bind implementation-carried
    // authority fields too: s_col, ct, and fold_digest. It deliberately
    // omits y_zcol because Π_DEC children do not prove a verifier-checkable
    // radix-b y_zcol recomposition equation.
    let claim = build_test_accumulator_ce_claim(0xACCE_551);
    let native_digest = accumulator_ce_claim_digest(&claim);

    let mut b = R1csBuilder::new();
    let digest = enforce_accumulator_claim_digest_for_test(&mut b, &claim);

    for (i, var) in digest.iter().enumerate() {
        assert_eq!(b.witness()[var.col()], native_digest[i], "lane {i}");
        b.enforce_eq(&Lc::from_var(*var), &Lc::from_const(native_digest[i]));
    }
    assert!(
        b.is_satisfied(),
        "accumulator_ce_claim_digest parity (first bad row: {:?})",
        b.first_unsatisfied_row()
    );
}

#[test]
fn enforce_pi_ccs_outputs_digest_matches_native_new_messages_only() {
    let claim = build_test_accumulator_ce_claim(0x0A77_005);
    let native_digest = pi_ccs_outputs_digest(std::slice::from_ref(&claim));
    let mut builder = R1csBuilder::new();
    let y_ring: Vec<Vec<KVar>> = claim
        .y_ring
        .iter()
        .map(|row| {
            row.iter()
                .copied()
                .map(|value| alloc_witness_k(&mut builder, value))
                .collect()
        })
        .collect();
    let y_zcol: Vec<KVar> = claim
        .y_zcol
        .iter()
        .copied()
        .map(|value| alloc_witness_k(&mut builder, value))
        .collect();
    let digest = enforce_pi_ccs_outputs_digest(
        &mut builder,
        &[PiCcsOutputClaimDigestInputs {
            y_ring: &y_ring,
            y_zcol: &y_zcol,
        }],
    )
    .expect("Pi_CCS output digest");

    for (lane, wire) in digest.into_iter().enumerate() {
        assert_eq!(builder.witness()[wire.col()], native_digest[lane], "lane {lane}");
        builder.enforce_eq(&Lc::from_var(wire), &Lc::from_const(native_digest[lane]));
    }
    assert!(builder.is_satisfied());

    let mut changed = claim;
    changed.y_zcol[0] += K::ONE;
    assert_ne!(
        pi_ccs_outputs_digest(std::slice::from_ref(&changed)),
        native_digest,
        "the pre-rho digest must bind newly sent y_zcol"
    );
}

#[test]
fn accumulator_ce_claim_digest_ignores_y_zcol_non_authority() {
    // y_zcol is consumed by the same-step NC/RLC equations, but child
    // y_zcol is not verifier-checkably recomposed by Π_DEC. If this digest
    // absorbed it, the prover would get an unconstrained Fiat-Shamir salt in
    // the recursive accumulator handle.
    let claim = build_test_accumulator_ce_claim(0xA11C_E5A17);
    assert!(!claim.y_zcol.is_empty(), "fixture must expose y_zcol");

    let digest = accumulator_ce_claim_digest(&claim);

    let mut c0_tampered = claim.clone();
    c0_tampered.y_zcol[0] += K::ONE;
    assert_eq!(
        accumulator_ce_claim_digest(&c0_tampered),
        digest,
        "accumulator digest must not absorb c0 of non-authority y_zcol"
    );

    let mut c1_tampered = claim.clone();
    c1_tampered.y_zcol[0] += K::from_coeffs([F::ZERO, F::ONE]);
    assert_eq!(
        accumulator_ce_claim_digest(&c1_tampered),
        digest,
        "accumulator digest must not absorb c1 of non-authority y_zcol"
    );

    let mut y_ring_tampered = claim.clone();
    y_ring_tampered.y_ring[0][0] += K::ONE;
    assert_ne!(
        accumulator_ce_claim_digest(&y_ring_tampered),
        digest,
        "contrast check: authority y_ring must still be absorbed"
    );

    let mut s_col_tampered = claim.clone();
    s_col_tampered.s_col[0] += K::ONE;
    assert_ne!(
        accumulator_ce_claim_digest(&s_col_tampered),
        digest,
        "contrast check: authority s_col must still be absorbed"
    );
}

#[test]
fn enforce_full_running_accumulator_digest_matches_native_with_parent() {
    // The running-accumulator handle is the in-circuit replacement for
    // hashing HyperNova's U_i in `state_x_out`: all authority-bearing child
    // fields plus the Π_RLC parent authority. This test pins the exact
    // native/circuit Poseidon2 preimage, independently of the larger SplitNc
    // verifier.
    let child_a = build_test_accumulator_ce_claim(0xA11CE);
    let child_b = build_test_accumulator_ce_claim(0xB0B);
    let parent = build_test_accumulator_ce_claim(0xFA12_EA7E_u64);
    let native_digest = digest32_as_fields(accumulator_digest_from_running_parts(
        &[child_a.clone(), child_b.clone()],
        Some(&parent),
    ));

    let mut b = R1csBuilder::new();
    let child_a_digest = enforce_accumulator_claim_digest_for_test(&mut b, &child_a);
    let child_b_digest = enforce_accumulator_claim_digest_for_test(&mut b, &child_b);
    let parent_digest = enforce_accumulator_claim_digest_for_test(&mut b, &parent);
    let digest =
        enforce_accumulator_digest_from_running_circuit(&mut b, &[child_a_digest, child_b_digest], Some(parent_digest));

    for (i, var) in digest.iter().enumerate() {
        assert_eq!(b.witness()[var.col()], native_digest[i], "lane {i}");
        b.enforce_eq(&Lc::from_var(*var), &Lc::from_const(native_digest[i]));
    }
    assert!(
        b.is_satisfied(),
        "running-accumulator authority digest parity (first bad row: {:?})",
        b.first_unsatisfied_row()
    );
}

#[test]
fn enforce_pi_ccs_instance_digest_matches_native_for_one_fresh_one_running() {
    // Native `pi_ccs_instance_digest(fresh, running)` hashes per-claim
    // digests under the `neo.fold.clean/pi_ccs_instance_digest/v1` domain.
    // We construct one fresh CCS claim + one running CE claim, compute
    // native and circuit digests, and assert parity.
    use neo_ajtai::Commitment;
    use neo_ccs::CcsClaim;
    use neo_fold_clean::paper::digest::digest32_as_fields;

    let mut s: u64 = 0xA110;
    let mut next_f = || -> F {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        F::from_u64(s & 0xFFFF)
    };

    let c_d = D;
    let c_kappa = 1usize;
    let fresh_c_data: Vec<F> = (0..(c_d * c_kappa)).map(|_| next_f()).collect();
    let fresh_x: Vec<F> = (0..2).map(|_| next_f()).collect();
    let fresh_m_in = 2usize;
    let fresh = CcsClaim::<Commitment, F> {
        adv: None,
        c: Commitment {
            d: c_d,
            kappa: c_kappa,
            data: fresh_c_data.clone(),
        },
        x: fresh_x.clone(),
        m_in: fresh_m_in,
    };

    let running = build_test_ce_claim(0xACCB, 3, 3, 4, 1, 2);

    let native_digest = pi_ccs_instance_digest(&[fresh.clone()], &[running.clone()]);

    // Circuit: build per-claim digests with our gadgets, then call
    // enforce_pi_ccs_instance_digest.
    let mut b = R1csBuilder::new();

    // Fresh CCS digest.
    let fresh_c_data_vars: Vec<Var> = fresh_c_data
        .iter()
        .map(|&v| alloc_witness_var(&mut b, v))
        .collect();
    let fresh_x_vars: Vec<Var> = fresh_x
        .iter()
        .map(|&v| alloc_witness_var(&mut b, v))
        .collect();
    let fresh_digest = enforce_ccs_claim_digest(
        &mut b,
        c_d,
        c_kappa,
        &fresh_c_data_vars,
        &fresh_x_vars,
        fresh_m_in,
        None,
    );

    // Running CE digest.
    let running_c_data_vars: Vec<Var> = running
        .c
        .data
        .iter()
        .map(|&v| alloc_witness_var(&mut b, v))
        .collect();
    let mut running_x_flat: Vec<Var> = Vec::with_capacity(running.X.rows() * running.X.cols());
    for r in 0..running.X.rows() {
        for c in 0..running.X.cols() {
            running_x_flat.push(alloc_witness_var(&mut b, running.X[(r, c)]));
        }
    }
    let running_r: Vec<KVar> = running
        .r
        .iter()
        .copied()
        .map(|v| alloc_witness_k(&mut b, v))
        .collect();
    let running_y: Vec<Vec<KVar>> = running
        .y_ring
        .iter()
        .map(|row| {
            row.iter()
                .copied()
                .map(|v| alloc_witness_k(&mut b, v))
                .collect()
        })
        .collect();
    let fold_lanes = digest32_as_fields(running.fold_digest);
    let fold_wires: [Var; 4] = std::array::from_fn(|i| alloc_witness_var(&mut b, fold_lanes[i]));
    let running_digest = enforce_ce_claim_digest(
        &mut b,
        &CeClaimDigestInputs {
            c_d: running.c.d,
            c_kappa: running.c.kappa,
            c_data: &running_c_data_vars,
            x_rows: running.X.rows(),
            x_cols: running.X.cols(),
            x_flat_row_major: &running_x_flat,
            r: &running_r,
            y_ring: &running_y,
            m_in: running.m_in,
            fold_digest_fields: fold_wires,
            adv: None,
        },
    )
    .expect("running CE digest");

    let instance_digest = enforce_pi_ccs_instance_digest(&mut b, &[fresh_digest], &[running_digest]);

    for (i, var) in instance_digest.iter().enumerate() {
        assert_eq!(b.witness()[var.col()], native_digest[i], "lane {i}");
        b.enforce_eq(&Lc::from_var(*var), &Lc::from_const(native_digest[i]));
    }
    assert!(
        b.is_satisfied(),
        "pi_ccs_instance_digest parity (first bad row: {:?})",
        b.first_unsatisfied_row()
    );
}

#[test]
fn enforce_pi_ccs_parent_authority_instance_digest_matches_native_missing_parent_sentinel() {
    // The composed SplitNc verifier rejects `running_count > 0` with no
    // parent authority before this helper is reached. Still, the helper is
    // documented as a byte-for-byte mirror of native
    // `pi_ccs_instance_digest_parent_authority`, whose malformed branch
    // absorbs `u64::MAX` rather than the empty-running `0` marker. Pin the
    // sentinel branch here so future standalone uses cannot drift from the
    // verifier transcript layout.
    use neo_ccs::CcsClaim;

    let c_d = D;
    let c_kappa = 1usize;
    let c_data = (0..(c_d * c_kappa))
        .map(|i| F::from_u64(0xD16E57_u64 + i as u64))
        .collect::<Vec<_>>();
    let x = vec![F::ONE, F::from_u64(2)];
    let m_in = x.len();
    let fresh = CcsClaim::<Commitment, F> {
        adv: None,
        c: Commitment {
            d: c_d,
            kappa: c_kappa,
            data: c_data.clone(),
        },
        x: x.clone(),
        m_in,
    };
    let running_count = 1usize;
    let native_digest = pi_ccs_instance_digest_parent_authority(&[fresh], running_count, None);

    let mut b = R1csBuilder::new();
    let c_data_vars = c_data
        .iter()
        .map(|&v| alloc_witness_var(&mut b, v))
        .collect::<Vec<_>>();
    let x_vars = x
        .iter()
        .map(|&v| alloc_witness_var(&mut b, v))
        .collect::<Vec<_>>();
    let fresh_digest = enforce_ccs_claim_digest(&mut b, c_d, c_kappa, &c_data_vars, &x_vars, m_in, None);
    let digest = enforce_pi_ccs_instance_digest_parent_authority(&mut b, &[fresh_digest], running_count, None);

    for (i, var) in digest.iter().enumerate() {
        assert_eq!(b.witness()[var.col()], native_digest[i], "lane {i}");
        b.enforce_eq(&Lc::from_var(*var), &Lc::from_const(native_digest[i]));
    }
    assert!(
        b.is_satisfied(),
        "pi_ccs parent-authority malformed-sentinel digest parity (first bad row: {:?})",
        b.first_unsatisfied_row()
    );
}

// ── sub-step E: FE claimed_initial sum ────────────────────────────────────

/// Native mirror of
/// `claimed_initial_sum_from_inputs_with_k_mcs(s, ch, k_mcs, me_inputs)`.
/// Kept local so the test doesn't need to construct a full `CcsStructure`
/// and `Challenges` just to exercise the formula.
fn native_fe_claimed_initial(t: usize, ell_d: usize, gamma: K, alpha: &[K], k_mcs: usize, y_ring: &[Vec<Vec<K>>]) -> K {
    let k_total = k_mcs + y_ring.len();
    if k_total < 2 {
        return K::ZERO;
    }

    // χ_α table.
    let d_sz = 1usize << ell_d;
    let mut chi_a = vec![K::ZERO; d_sz];
    for rho in 0..d_sz {
        let mut w = K::ONE;
        for (bit, &a) in alpha.iter().enumerate() {
            let is_one = ((rho >> bit) & 1) == 1;
            w *= if is_one { a } else { K::ONE - a };
        }
        chi_a[rho] = w;
    }

    // γ^{k_total}.
    let mut gamma_to_k = K::ONE;
    for _ in 0..k_total {
        gamma_to_k *= gamma;
    }

    let mut inner = K::ZERO;
    for j in 0..t {
        for (idx, yj_outer) in y_ring.iter().enumerate() {
            let y_row = &yj_outer[j];
            let mut y_eval = K::ZERO;
            for rho in 0..d_sz {
                y_eval += y_row[rho] * chi_a[rho];
            }
            // weight = γ^{k_mcs + idx + j·k_total}.
            let mut weight = K::ONE;
            for _ in 0..(k_mcs + idx + j * k_total) {
                weight *= gamma;
            }
            inner += weight * y_eval;
        }
    }
    gamma_to_k * inner
}

#[test]
fn enforce_fe_claimed_initial_matches_native_formula_small() {
    // Small but non-trivial: ell_d = 3, t = 3, |me| = 2, k_mcs = 1.
    // k_total = 3, exponents range [k_mcs, k_mcs + |me| - 1] = [1, 2] per j,
    // multiplied by γ^{j·k_total} for j ∈ {0, 1, 2}.
    let ell_d = 3usize;
    let t = 3usize;
    let me_len = 2usize;
    let k_mcs = 1usize;

    // Deterministic K-value generator.
    let mut s: u64 = 0xFE_0001;
    let mut next_f = || -> F {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        F::from_u64(s & 0xFFFF)
    };
    let mut next_k = || -> K { K::from_coeffs([next_f(), next_f()]) };

    let gamma = next_k();
    let alpha: Vec<K> = (0..ell_d).map(|_| next_k()).collect();
    let d_sz = 1usize << ell_d;
    let y_ring: Vec<Vec<Vec<K>>> = (0..me_len)
        .map(|_| {
            (0..t)
                .map(|_| (0..d_sz).map(|_| next_k()).collect())
                .collect()
        })
        .collect();

    let expected = native_fe_claimed_initial(t, ell_d, gamma, &alpha, k_mcs, &y_ring);

    // Circuit side.
    let mut b = R1csBuilder::new();
    let gamma_var = alloc_witness_k(&mut b, gamma);
    let alpha_vars: Vec<KVar> = alpha
        .iter()
        .copied()
        .map(|a| alloc_witness_k(&mut b, a))
        .collect();
    let y_vars: Vec<Vec<Vec<KVar>>> = y_ring
        .iter()
        .map(|outer| {
            outer
                .iter()
                .map(|row| {
                    row.iter()
                        .copied()
                        .map(|v| alloc_witness_k(&mut b, v))
                        .collect()
                })
                .collect()
        })
        .collect();

    let inputs = FeClaimedInitialInputs {
        k_mcs,
        t,
        ell_d,
        gamma: gamma_var,
        alpha: &alpha_vars,
        running_y_ring: &y_vars,
    };
    let result = enforce_fe_claimed_initial(&mut b, &inputs).expect("FE claimed_initial");
    let result_k = k_value(&b, result);
    assert_eq!(result_k, expected, "FE claimed_initial mismatch");

    // Pin and check satisfaction.
    pin_native(&mut b, result, expected);
    assert!(
        b.is_satisfied(),
        "FE claimed_initial circuit unsatisfied (first bad row: {:?})",
        b.first_unsatisfied_row()
    );
}

#[test]
fn enforce_fe_claimed_initial_returns_zero_when_k_total_lt_2() {
    // Native returns K::ZERO if k_total < 2. Three cases:
    //   (k_mcs=0, |me|=0) → k_total = 0 → ZERO
    //   (k_mcs=1, |me|=0) → k_total = 1 → ZERO
    //   (k_mcs=0, |me|=1) → k_total = 1 → ZERO
    let ell_d = 2usize;
    let t = 2usize;

    let mut s: u64 = 0xFE_0002;
    let mut next_f = || -> F {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        F::from_u64(s & 0xFFFF)
    };
    let mut next_k = || -> K { K::from_coeffs([next_f(), next_f()]) };

    let gamma = next_k();
    let alpha: Vec<K> = (0..ell_d).map(|_| next_k()).collect();
    let d_sz = 1usize << ell_d;

    for (k_mcs, me_len) in [(0usize, 0usize), (1, 0), (0, 1)] {
        let y_ring: Vec<Vec<Vec<K>>> = (0..me_len)
            .map(|_| {
                (0..t)
                    .map(|_| (0..d_sz).map(|_| next_k()).collect())
                    .collect()
            })
            .collect();

        let mut b = R1csBuilder::new();
        let gamma_var = alloc_witness_k(&mut b, gamma);
        let alpha_vars: Vec<KVar> = alpha
            .iter()
            .copied()
            .map(|a| alloc_witness_k(&mut b, a))
            .collect();
        let y_vars: Vec<Vec<Vec<KVar>>> = y_ring
            .iter()
            .map(|outer| {
                outer
                    .iter()
                    .map(|row| {
                        row.iter()
                            .copied()
                            .map(|v| alloc_witness_k(&mut b, v))
                            .collect()
                    })
                    .collect()
            })
            .collect();

        let inputs = FeClaimedInitialInputs {
            k_mcs,
            t,
            ell_d,
            gamma: gamma_var,
            alpha: &alpha_vars,
            running_y_ring: &y_vars,
        };
        let result = enforce_fe_claimed_initial(&mut b, &inputs).expect("FE claimed_initial");
        assert_eq!(
            k_value(&b, result),
            K::ZERO,
            "k_mcs={k_mcs} me_len={me_len}: expected K::ZERO"
        );
        pin_native(&mut b, result, K::ZERO);
        assert!(b.is_satisfied(), "k_mcs={k_mcs} me_len={me_len}: zero-case unsatisfied");
    }
}

#[test]
fn enforce_fe_claimed_initial_rejects_alpha_length_mismatch() {
    // alpha.len must equal ell_d. Mismatch must surface as Err, not panic.
    let ell_d = 3usize;
    let t = 2usize;
    let mut b = R1csBuilder::new();
    let gamma_var = alloc_witness_k(&mut b, K::from_u64(7));
    let alpha_vars: Vec<KVar> = (0..2)
        .map(|_| alloc_witness_k(&mut b, K::from_u64(11)))
        .collect();
    let y_vars: Vec<Vec<Vec<KVar>>> = vec![vec![vec![alloc_witness_k(&mut b, K::ZERO); 1 << ell_d]; t]; 2];
    let inputs = FeClaimedInitialInputs {
        k_mcs: 1,
        t,
        ell_d,
        gamma: gamma_var,
        alpha: &alpha_vars, // length 2, not ell_d=3
        running_y_ring: &y_vars,
    };
    assert!(enforce_fe_claimed_initial(&mut b, &inputs).is_err());
}

#[test]
fn enforce_fe_claimed_initial_rejects_tampered_y_ring_entry() {
    // After the gadget runs, tampering any y_ring witness must propagate
    // through the dot product → weight·y_eval → inner → T pin and break.
    let ell_d = 3usize;
    let t = 2usize;
    let me_len = 2usize;
    let k_mcs = 1usize;

    let mut s: u64 = 0xFE_0003;
    let mut next_f = || -> F {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        F::from_u64(s & 0xFFFF)
    };
    let mut next_k = || -> K { K::from_coeffs([next_f(), next_f()]) };

    let gamma = next_k();
    let alpha: Vec<K> = (0..ell_d).map(|_| next_k()).collect();
    let d_sz = 1usize << ell_d;
    let y_ring: Vec<Vec<Vec<K>>> = (0..me_len)
        .map(|_| {
            (0..t)
                .map(|_| (0..d_sz).map(|_| next_k()).collect())
                .collect()
        })
        .collect();
    let expected = native_fe_claimed_initial(t, ell_d, gamma, &alpha, k_mcs, &y_ring);

    let mut b = R1csBuilder::new();
    let gamma_var = alloc_witness_k(&mut b, gamma);
    let alpha_vars: Vec<KVar> = alpha
        .iter()
        .copied()
        .map(|a| alloc_witness_k(&mut b, a))
        .collect();
    let y_vars: Vec<Vec<Vec<KVar>>> = y_ring
        .iter()
        .map(|outer| {
            outer
                .iter()
                .map(|row| {
                    row.iter()
                        .copied()
                        .map(|v| alloc_witness_k(&mut b, v))
                        .collect()
                })
                .collect()
        })
        .collect();
    let inputs = FeClaimedInitialInputs {
        k_mcs,
        t,
        ell_d,
        gamma: gamma_var,
        alpha: &alpha_vars,
        running_y_ring: &y_vars,
    };
    let result = enforce_fe_claimed_initial(&mut b, &inputs).expect("FE claimed_initial");
    pin_native(&mut b, result, expected);
    assert!(b.is_satisfied(), "baseline parity");

    // Tamper y_ring[0][1][2].c0 — one ρ-lane of one (idx, j) row.
    let target = y_vars[0][1][2].c0.col();
    let tampered = b.witness()[target] + F::from_u64(1);
    b.tamper_witness(target, tampered);
    assert!(!b.is_satisfied(), "tampered y_ring must break FE claimed_initial pin");
}

// ── sub-step I: header_digest catch-up squeeze ────────────────────────────

#[test]
fn header_digest_catch_up_matches_native_digest32() {
    // Native `engine::optimized::verify_pi_ccs` calls `tr.digest32()` after
    // the SplitNc engine verifier returns. Drive the same prefix on both
    // sides and confirm the in-circuit catch-up squeeze (a) advances the
    // transcript identically and (b) the four observed lanes equal native.
    let mut native = Poseidon2Transcript::new(APP);
    native.append_fields_raw(&[
        F::from_u64(PI_CCS_HEADER_BUNDLE_RAW_TAG),
        F::from_u64(1),
        F::from_u64(2),
        F::from_u64(3),
        F::from_u64(4),
    ]);
    native.append_fields_raw(&[
        F::from_u64(PI_CCS_INSTANCE_DIGEST_RAW_TAG),
        F::from_u64(5),
        F::from_u64(6),
        F::from_u64(7),
        F::from_u64(8),
    ]);
    let header_bytes = native.digest32();
    let header_fields = header_digest_bytes_to_fields(&header_bytes).expect("decode 32-byte digest");

    let mut b = R1csBuilder::new();
    let mut tr = TranscriptGadget::new(&mut b, APP);
    tr.append_fields_raw_const(
        &mut b,
        &[
            F::from_u64(PI_CCS_HEADER_BUNDLE_RAW_TAG),
            F::from_u64(1),
            F::from_u64(2),
            F::from_u64(3),
            F::from_u64(4),
        ],
    );
    tr.append_fields_raw_const(
        &mut b,
        &[
            F::from_u64(PI_CCS_INSTANCE_DIGEST_RAW_TAG),
            F::from_u64(5),
            F::from_u64(6),
            F::from_u64(7),
            F::from_u64(8),
        ],
    );
    enforce_header_digest_catch_up(&mut b, &mut tr, header_fields);

    assert!(
        b.is_satisfied(),
        "header_digest catch-up must match native digest32 (first bad row: {:?})",
        b.first_unsatisfied_row()
    );
}

#[test]
fn header_digest_catch_up_rejects_tampered_digest() {
    // Compute the honest header digest, flip one byte, and check the
    // in-circuit catch-up rejects it. Soundness: any prover-side tamper of
    // `proof.header_digest` must break the digest pin.
    let mut native = Poseidon2Transcript::new(APP);
    native.append_fields_raw(&[F::from_u64(PI_CCS_HEADER_BUNDLE_RAW_TAG), F::from_u64(42)]);
    let mut header_bytes = native.digest32();
    header_bytes[0] ^= 1;
    let bad_fields = header_digest_bytes_to_fields(&header_bytes).expect("decode 32-byte digest");

    let mut b = R1csBuilder::new();
    let mut tr = TranscriptGadget::new(&mut b, APP);
    tr.append_fields_raw_const(&mut b, &[F::from_u64(PI_CCS_HEADER_BUNDLE_RAW_TAG), F::from_u64(42)]);
    enforce_header_digest_catch_up(&mut b, &mut tr, bad_fields);

    assert!(
        !b.is_satisfied(),
        "tampered header_digest must be rejected by catch-up squeeze"
    );
}

#[test]
fn header_digest_bytes_rejects_wrong_length() {
    // The native `digest32()` always emits 32 bytes. Decoder must refuse
    // anything else rather than silently truncate / zero-pad.
    assert!(header_digest_bytes_to_fields(&[0u8; 31]).is_err());
    assert!(header_digest_bytes_to_fields(&[0u8; 33]).is_err());
    assert!(header_digest_bytes_to_fields(&[]).is_err());
    assert!(header_digest_bytes_to_fields(&[0u8; 32]).is_ok());
}

#[test]
fn header_digest_bytes_rejects_noncanonical_field_limb_alias() {
    // Native `digest32()` serializes canonical Goldilocks field elements. The
    // circuit verifier must not accept a different byte string whose u64 limb
    // aliases to the same field element via `F::from_u64`; otherwise the
    // in-circuit proof verifier accepts a Pi_CCS proof object the native
    // verifier rejects byte-for-byte.
    let mut bytes = [0u8; 32];
    bytes[..8].copy_from_slice(&F::ORDER_U64.to_le_bytes());

    assert!(
        header_digest_bytes_to_fields(&bytes).is_err(),
        "noncanonical digest limb p aliases to zero and must be rejected"
    );
}

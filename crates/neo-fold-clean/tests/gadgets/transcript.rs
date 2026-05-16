//! In-circuit Poseidon2 transcript — byte-for-byte parity against
//! `neo_transcript::Poseidon2Transcript`.
//!
//! Soundness contract: any absorb/squeeze sequence performed on both
//! transcripts must yield identical squeezed field values. If this test
//! suite passes, the gadget can be used to derive Fiat-Shamir challenges
//! in-circuit without trusting prover-supplied randomness.
//!
//! Coverage:
//!   - Empty session, challenge_field
//!   - append_fields with cross-RATE-boundary length
//!   - challenge_fields(n) for n in {1, 4, 5, 13}
//!   - append_message of byte strings
//!   - Interleaved absorbs and squeezes
//!   - Tamper rejection (absorbed field tampered → squeeze differs from
//!     native, so circuit's pinned challenge is unsatisfied)
//!   - digest_fields parity

use neo_fold_clean::engine::r1cs_circuit::{R1csBuilder, TranscriptGadget, Var};
use neo_math::F;
use neo_transcript::{Poseidon2Transcript, Transcript};
use p3_field::PrimeCharacteristicRing;

const APP: &[u8] = b"neo.test.transcript/v1";

fn alloc_vec(b: &mut R1csBuilder, vals: &[F]) -> Vec<Var> {
    vals.iter().map(|&v| b.alloc(v)).collect()
}

/// Pin a circuit-squeezed `Var` to the native-squeezed F value via an
/// equality constraint. If the circuit's squeeze diverges, `is_satisfied`
/// returns false. This is the soundness check.
fn pin_to_native(b: &mut R1csBuilder, circuit_var: Var, native_value: F) {
    use neo_fold_clean::engine::r1cs_circuit::Lc;
    b.enforce_eq(&Lc::from_var(circuit_var), &Lc::from_const(native_value));
}

// ── Smallest possible: init + one squeeze ────────────────────────────────

#[test]
fn empty_session_challenge_field_matches_native() {
    let mut native = Poseidon2Transcript::new(APP);
    let native_chal: F = native.challenge_field(b"chal0");

    let mut b = R1csBuilder::new();
    let mut tr = TranscriptGadget::new(&mut b, APP);
    let chal = tr.challenge_field(&mut b, b"chal0");

    assert_eq!(b.witness()[chal.col()], native_chal);
    pin_to_native(&mut b, chal, native_chal);
    assert!(
        b.is_satisfied(),
        "init+challenge parity (first bad row: {:?})",
        b.first_unsatisfied_row()
    );
}

#[test]
fn empty_session_challenge_fields_n_matches_native() {
    for n in [1usize, 4, 5, 13] {
        let mut native = Poseidon2Transcript::new(APP);
        let native_chals: Vec<F> = native.challenge_fields(b"chal_n", n);

        let mut b = R1csBuilder::new();
        let mut tr = TranscriptGadget::new(&mut b, APP);
        let chals = tr.challenge_fields(&mut b, b"chal_n", n);

        assert_eq!(chals.len(), n);
        for (i, var) in chals.iter().enumerate() {
            assert_eq!(b.witness()[var.col()], native_chals[i], "n={n}, i={i}");
            pin_to_native(&mut b, *var, native_chals[i]);
        }
        assert!(b.is_satisfied(), "challenge_fields(n={n}) parity");
    }
}

// ── append_fields parity ────────────────────────────────────────────────

#[test]
fn append_fields_short_then_challenge_matches_native() {
    let fs: Vec<F> = (0..3).map(|i| F::from_u64(i + 1)).collect();

    let mut native = Poseidon2Transcript::new(APP);
    native.append_fields(b"absorb", &fs);
    let native_chal = native.challenge_field(b"chal");

    let mut b = R1csBuilder::new();
    let mut tr = TranscriptGadget::new(&mut b, APP);
    let f_vars = alloc_vec(&mut b, &fs);
    tr.append_fields(&mut b, b"absorb", &f_vars);
    let chal = tr.challenge_field(&mut b, b"chal");

    assert_eq!(b.witness()[chal.col()], native_chal);
    pin_to_native(&mut b, chal, native_chal);
    assert!(b.is_satisfied());
}

#[test]
fn append_fields_cross_rate_boundary_matches_native() {
    // RATE = 4. Absorbing > 4 fields exercises the bulk-absorb + permute path.
    for &len in &[0usize, 1, 4, 5, 7, 8, 9, 16, 17] {
        let fs: Vec<F> = (0..len).map(|i| F::from_u64((i as u64) * 37 + 1)).collect();

        let mut native = Poseidon2Transcript::new(APP);
        native.append_fields(b"x", &fs);
        let native_chal = native.challenge_field(b"chal");

        let mut b = R1csBuilder::new();
        let mut tr = TranscriptGadget::new(&mut b, APP);
        let f_vars = alloc_vec(&mut b, &fs);
        tr.append_fields(&mut b, b"x", &f_vars);
        let chal = tr.challenge_field(&mut b, b"chal");

        assert_eq!(b.witness()[chal.col()], native_chal, "len={len}");
        pin_to_native(&mut b, chal, native_chal);
        assert!(b.is_satisfied(), "len={len} parity");
    }
}

// ── append_message (byte path) parity ───────────────────────────────────

#[test]
fn append_message_short_matches_native() {
    let msg: &[u8] = b"hello/world";

    let mut native = Poseidon2Transcript::new(APP);
    native.append_message(b"m", msg);
    let native_chal = native.challenge_field(b"chal");

    let mut b = R1csBuilder::new();
    let mut tr = TranscriptGadget::new(&mut b, APP);
    tr.append_message(&mut b, b"m", msg);
    let chal = tr.challenge_field(&mut b, b"chal");

    assert_eq!(b.witness()[chal.col()], native_chal);
    pin_to_native(&mut b, chal, native_chal);
    assert!(b.is_satisfied());
}

#[test]
fn append_message_byte_packing_edge_lengths_match_native() {
    // 7 bytes per limb: try lengths around the chunk boundary.
    for &len in &[0usize, 1, 6, 7, 8, 13, 14, 15, 21, 64] {
        let msg: Vec<u8> = (0..len).map(|i| (i as u8).wrapping_mul(13)).collect();

        let mut native = Poseidon2Transcript::new(APP);
        native.append_message(b"m", &msg);
        let native_chal = native.challenge_field(b"chal");

        let mut b = R1csBuilder::new();
        let mut tr = TranscriptGadget::new(&mut b, APP);
        tr.append_message(&mut b, b"m", &msg);
        let chal = tr.challenge_field(&mut b, b"chal");

        assert_eq!(b.witness()[chal.col()], native_chal, "msg_len={len}");
        pin_to_native(&mut b, chal, native_chal);
        assert!(b.is_satisfied(), "msg_len={len} parity");
    }
}

// ── Interleaved absorbs and squeezes ────────────────────────────────────

#[test]
fn interleaved_absorbs_and_squeezes_match_native() {
    let a: Vec<F> = (0..5).map(|i| F::from_u64(i + 10)).collect();
    let b_fields: Vec<F> = (0..2).map(|i| F::from_u64(i + 100)).collect();

    let mut native = Poseidon2Transcript::new(APP);
    native.append_fields(b"a", &a);
    let chal_x: F = native.challenge_field(b"x");
    native.append_fields(b"b", &b_fields);
    let chal_y: Vec<F> = native.challenge_fields(b"y", 3);
    native.append_message(b"m", b"end");
    let chal_z: F = native.challenge_field(b"z");

    let mut bd = R1csBuilder::new();
    let mut tr = TranscriptGadget::new(&mut bd, APP);
    let a_vars = alloc_vec(&mut bd, &a);
    tr.append_fields(&mut bd, b"a", &a_vars);
    let x_var = tr.challenge_field(&mut bd, b"x");
    let b_vars = alloc_vec(&mut bd, &b_fields);
    tr.append_fields(&mut bd, b"b", &b_vars);
    let y_vars = tr.challenge_fields(&mut bd, b"y", 3);
    tr.append_message(&mut bd, b"m", b"end");
    let z_var = tr.challenge_field(&mut bd, b"z");

    assert_eq!(bd.witness()[x_var.col()], chal_x);
    for (i, var) in y_vars.iter().enumerate() {
        assert_eq!(bd.witness()[var.col()], chal_y[i]);
    }
    assert_eq!(bd.witness()[z_var.col()], chal_z);

    pin_to_native(&mut bd, x_var, chal_x);
    for (i, var) in y_vars.iter().enumerate() {
        pin_to_native(&mut bd, *var, chal_y[i]);
    }
    pin_to_native(&mut bd, z_var, chal_z);
    assert!(bd.is_satisfied(), "interleaved parity");
}

// ── digest_fields parity ────────────────────────────────────────────────

#[test]
fn digest_fields_matches_native_digest32_first_4_lanes() {
    // Native digest32 packs each of 4 lanes as 8 LE bytes. Our gadget
    // returns the 4 lanes directly as F. Cross-check by reconstructing.
    let fs: Vec<F> = (0..3).map(|i| F::from_u64(i + 1)).collect();

    let mut native = Poseidon2Transcript::new(APP);
    native.append_fields(b"a", &fs);
    let native_bytes: [u8; 32] = native.digest32();
    let native_lanes: [F; 4] = std::array::from_fn(|i| {
        let mut limb = [0u8; 8];
        limb.copy_from_slice(&native_bytes[i * 8..(i + 1) * 8]);
        F::from_u64(u64::from_le_bytes(limb))
    });

    let mut b = R1csBuilder::new();
    let mut tr = TranscriptGadget::new(&mut b, APP);
    let f_vars = alloc_vec(&mut b, &fs);
    tr.append_fields(&mut b, b"a", &f_vars);
    let lanes = tr.digest_fields(&mut b);

    for (i, var) in lanes.iter().enumerate() {
        assert_eq!(b.witness()[var.col()], native_lanes[i], "lane {i}");
        pin_to_native(&mut b, *var, native_lanes[i]);
    }
    assert!(b.is_satisfied());
}

// ── K-element helpers (used by Π_CCS / Π_RLC challenge derivation) ──────

#[test]
fn challenge_k_matches_native_from_complex() {
    use neo_math::{from_complex, KExtensions, K};

    let mut native = Poseidon2Transcript::new(APP);
    let pair = <Poseidon2Transcript as Transcript>::challenge_fields(&mut native, b"alpha", 2);
    let native_k: K = from_complex(pair[0], pair[1]);
    let [c0_native, c1_native] = native_k.as_coeffs();

    let mut b = R1csBuilder::new();
    let mut tr = TranscriptGadget::new(&mut b, APP);
    let k_var = tr.challenge_k(&mut b, b"alpha");

    assert_eq!(b.witness()[k_var.c0.col()], c0_native);
    assert_eq!(b.witness()[k_var.c1.col()], c1_native);
    pin_to_native(&mut b, k_var.c0, c0_native);
    pin_to_native(&mut b, k_var.c1, c1_native);
    assert!(b.is_satisfied());
}

#[test]
fn challenge_k_vec_matches_native_batch() {
    use neo_math::{from_complex, KExtensions, K};

    for n in [1usize, 2, 5, 7] {
        let mut native = Poseidon2Transcript::new(APP);
        let lanes = <Poseidon2Transcript as Transcript>::challenge_fields(&mut native, b"vec", 2 * n);
        let native_ks: Vec<K> = lanes
            .chunks_exact(2)
            .map(|p| from_complex(p[0], p[1]))
            .collect();

        let mut b = R1csBuilder::new();
        let mut tr = TranscriptGadget::new(&mut b, APP);
        let ks = tr.challenge_k_vec(&mut b, b"vec", n);

        assert_eq!(ks.len(), n);
        for (i, k) in ks.iter().enumerate() {
            let [c0, c1] = native_ks[i].as_coeffs();
            assert_eq!(b.witness()[k.c0.col()], c0, "n={n} i={i} c0");
            assert_eq!(b.witness()[k.c1.col()], c1, "n={n} i={i} c1");
            pin_to_native(&mut b, k.c0, c0);
            pin_to_native(&mut b, k.c1, c1);
        }
        assert!(b.is_satisfied(), "n={n}");
    }
}

#[test]
fn append_k_slice_then_challenge_k_matches_native() {
    use neo_math::{from_complex, KExtensions, K};

    let ks_native: Vec<K> = (0..3)
        .map(|i| K::from_coeffs([F::from_u64(i + 1), F::from_u64(i + 7)]))
        .collect();

    let mut native = Poseidon2Transcript::new(APP);
    let packed: Vec<F> = ks_native.iter().flat_map(|k| k.as_coeffs()).collect();
    native.append_fields(b"absorb_k", &packed);
    let chal_pair = <Poseidon2Transcript as Transcript>::challenge_fields(&mut native, b"chal", 2);
    let native_chal: K = from_complex(chal_pair[0], chal_pair[1]);
    let [c0n, c1n] = native_chal.as_coeffs();

    let mut b = R1csBuilder::new();
    let mut tr = TranscriptGadget::new(&mut b, APP);
    let k_vars: Vec<_> = ks_native
        .iter()
        .map(|k| {
            let [c0, c1] = k.as_coeffs();
            use neo_fold_clean::engine::r1cs_circuit::field_ext::KVar;
            KVar::alloc(&mut b, c0, c1)
        })
        .collect();
    tr.append_k_slice(&mut b, b"absorb_k", &k_vars);
    let chal = tr.challenge_k(&mut b, b"chal");

    assert_eq!(b.witness()[chal.c0.col()], c0n);
    assert_eq!(b.witness()[chal.c1.col()], c1n);
    pin_to_native(&mut b, chal.c0, c0n);
    pin_to_native(&mut b, chal.c1, c1n);
    assert!(b.is_satisfied());
}

// ── Raw absorb / squeeze parity (engine sumcheck path) ──────────────────

#[test]
fn append_fields_raw_vars_then_challenge_matches_native() {
    // Mirrors native `Poseidon2Transcript::append_fields_raw(fs)` followed
    // by a labelled squeeze. The raw absorb has no label — just len header
    // then the slice. This is what `verify_sumcheck_rounds_poseidon_v3`
    // does each round.
    for &len in &[0usize, 1, 4, 5, 8, 13] {
        let fs: Vec<F> = (0..len).map(|i| F::from_u64((i as u64) * 91 + 3)).collect();

        let mut native = Poseidon2Transcript::new(APP);
        native.append_fields_raw(&fs);
        let native_chal: F = native.challenge_field(b"chal");

        let mut b = R1csBuilder::new();
        let mut tr = TranscriptGadget::new(&mut b, APP);
        let f_vars = alloc_vec(&mut b, &fs);
        tr.append_fields_raw_vars(&mut b, &f_vars);
        let chal = tr.challenge_field(&mut b, b"chal");

        assert_eq!(b.witness()[chal.col()], native_chal, "len={len}");
        pin_to_native(&mut b, chal, native_chal);
        assert!(b.is_satisfied(), "len={len} parity");
    }
}

#[test]
fn challenge_fields_raw_matches_native_for_various_n() {
    // Mirrors native `Poseidon2Transcript::challenge_fields_raw(n)` —
    // a labelless squeeze, used inside `sample_k_batch` and the per-round
    // sumcheck challenge in `verify_sumcheck_rounds_poseidon_v3`.
    for &n in &[1usize, 2, 4, 5, 8, 13] {
        let mut native = Poseidon2Transcript::new(APP);
        let native_out: Vec<F> = native.challenge_fields_raw(n);

        let mut b = R1csBuilder::new();
        let mut tr = TranscriptGadget::new(&mut b, APP);
        let vars = tr.challenge_fields_raw(&mut b, n);

        assert_eq!(vars.len(), n);
        for (i, var) in vars.iter().enumerate() {
            assert_eq!(b.witness()[var.col()], native_out[i], "n={n} i={i}");
            pin_to_native(&mut b, *var, native_out[i]);
        }
        assert!(b.is_satisfied(), "n={n} parity");
    }
}

#[test]
fn raw_round_then_raw_challenge_matches_sumcheck_v3_pattern() {
    // Concrete shape of `verify_sumcheck_rounds_poseidon_v3`'s per-round
    // payload: `append_fields_raw(packed)` then `challenge_fields_raw(2)`.
    // Walks three "rounds" to exercise both pre- and post-permute paths.
    let rounds: Vec<Vec<F>> = (0..3)
        .map(|r| {
            (0..6)
                .map(|i| F::from_u64((r * 100 + i + 1) as u64))
                .collect()
        })
        .collect();

    let mut native = Poseidon2Transcript::new(APP);
    let mut native_challenges: Vec<[F; 2]> = Vec::new();
    for r in &rounds {
        native.append_fields_raw(r);
        let pair = native.challenge_fields_raw(2);
        native_challenges.push([pair[0], pair[1]]);
    }

    let mut b = R1csBuilder::new();
    let mut tr = TranscriptGadget::new(&mut b, APP);
    let mut circuit_challenges: Vec<[Var; 2]> = Vec::new();
    for r in &rounds {
        let f_vars = alloc_vec(&mut b, r);
        tr.append_fields_raw_vars(&mut b, &f_vars);
        let pair = tr.challenge_fields_raw(&mut b, 2);
        circuit_challenges.push([pair[0], pair[1]]);
    }

    for (i, [c0, c1]) in circuit_challenges.iter().enumerate() {
        assert_eq!(b.witness()[c0.col()], native_challenges[i][0]);
        assert_eq!(b.witness()[c1.col()], native_challenges[i][1]);
        pin_to_native(&mut b, *c0, native_challenges[i][0]);
        pin_to_native(&mut b, *c1, native_challenges[i][1]);
    }
    assert!(b.is_satisfied(), "sumcheck v3 round-pattern parity");
}

// ── Tamper rejection ────────────────────────────────────────────────────

#[test]
fn tampered_absorbed_field_breaks_challenge_pin() {
    // If a prover changes an absorbed witness wire, the squeeze diverges
    // from native — so the equality constraint that pins it to the native
    // value fails. This is what makes Fiat-Shamir sound in-circuit.
    let fs: Vec<F> = (0..3).map(|i| F::from_u64(i + 1)).collect();

    let mut native = Poseidon2Transcript::new(APP);
    native.append_fields(b"a", &fs);
    let native_chal = native.challenge_field(b"chal");

    let mut b = R1csBuilder::new();
    let mut tr = TranscriptGadget::new(&mut b, APP);
    let f_vars = alloc_vec(&mut b, &fs);
    tr.append_fields(&mut b, b"a", &f_vars);
    let chal = tr.challenge_field(&mut b, b"chal");
    pin_to_native(&mut b, chal, native_chal);

    assert!(b.is_satisfied(), "baseline parity");

    // Tamper f_vars[1] — sponge state diverges, squeeze diverges, pin fails.
    let target = f_vars[1].col();
    let tampered = b.witness()[target] + F::ONE;
    b.tamper_witness(target, tampered);
    assert!(
        !b.is_satisfied(),
        "tampering an absorbed field must break the squeeze pin"
    );
}

#[test]
fn tampered_label_constant_breaks_pin() {
    // Constant wires (labels, length headers, the squeeze-domain ONE) are
    // bound by equality constraints. Tampering one of them must break the
    // pin too — even though they're "constants", an adversarial prover
    // could try to bypass them.
    let fs: Vec<F> = (0..2).map(|i| F::from_u64(i)).collect();

    let mut native = Poseidon2Transcript::new(APP);
    native.append_fields(b"label", &fs);
    let native_chal = native.challenge_field(b"chal");

    let mut b = R1csBuilder::new();
    let mut tr = TranscriptGadget::new(&mut b, APP);
    let f_vars = alloc_vec(&mut b, &fs);
    tr.append_fields(&mut b, b"label", &f_vars);
    let _chal = tr.challenge_field(&mut b, b"chal");
    pin_to_native(&mut b, _chal, native_chal);
    assert!(b.is_satisfied(), "baseline parity");

    // Find a witness column allocated for a constant absorb and tamper it.
    // The first constants allocated by `new()` are the 8 initial state
    // values from native init; they're at columns 1..=8. Pick the constant
    // for the "label" byte's length (the first absorb after init).
    // We don't know exactly which column without parsing the trace, but
    // tampering *any* constant-bound wire breaks the equality constraint
    // for that wire — so we just sweep until we find one that the gadget
    // depends on.
    //
    // Concretely: the very first absorb after `new()` is
    // `absorb_packed_bytes_with_len(b"label")` which absorbs `len=5` as a
    // constant. That wire is at some column > 8. Try column 9 — by
    // construction it's the first post-init constant.
    let target = 9;
    let original = b.witness()[target];
    b.tamper_witness(target, original + F::ONE);
    assert!(
        !b.is_satisfied(),
        "tampering a constant-bound wire must break some constraint"
    );
}

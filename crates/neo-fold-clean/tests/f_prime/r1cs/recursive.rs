use super::*;

#[test]
fn f_prime_recursive_step_accepts_real_native_nifs_proof() {
    let fixture = build_fixture();
    let cfg = make_step_config(&fixture.prep);
    let source = recursive_source_image(&fixture);
    let inputs = FPrimeRecursiveInputs {
        state: fixture.state.clone(),
        chunk_digest: fixture.chunk_digest,
        semantic_state_digest_out: recursive_acc_digest(&fixture),
        acc_digest_out: recursive_acc_digest(&fixture),
        nifs_msg: msg_from_fixture(&fixture),
        rows_in_chunk: 1,
        source_image: &source.image,
        chunk_count_in_word: source.chunk_count_in_word,
        step_count_in_word: source.step_count_in_word,
        pc_word: source.pc_word,
        prior_x_out_bits: source.prior_x_out_bits,
        public_x_out_bits: source.public_x_out_bits,
    };

    let mut b = R1csBuilder::new();
    let rows_before = b.rows();
    let out = enforce_f_prime_recursive_step_circuit(&mut b, fixture.prep.params(), &cfg, &inputs).expect("emit");
    let rows_added = b.rows() - rows_before;
    assert!(
        rows_added > 100_000,
        "F' recursive step should emit >100k rows; got {rows_added}"
    );
    assert_eq!(out.x_out.len(), 4);
    assert_eq!(out.x_out_bits.len(), F_PRIME_ENC_INST_BITS);
    assert!(
        b.is_satisfied(),
        "real recursive F' step must satisfy (first bad row: {:?})",
        b.first_unsatisfied_row()
    );
    let unconstrained = b.unconstrained_columns();
    assert!(
        unconstrained.is_empty(),
        "recursive F' step left unconstrained columns: {unconstrained:?}"
    );
}

#[test]
fn f_prime_recursive_rejects_chunk_counter_field_modulus_wrap() {
    let fixture = build_fixture();
    let mut state = fixture.state.clone();
    state.chunk_count_in = F::ORDER_U64 - 1;
    state.step_count_in = 1;
    let fixture = rebuild_recursive_fixture_for_state(fixture, state);
    let cfg = make_step_config(&fixture.prep);

    let acc = recursive_acc_digest(&fixture);
    let wrapped_x_out = native_x_out(
        &fixture.state,
        fixture.chunk_digest,
        acc,
        acc,
        0,
        fixture.state.step_count_in + 1,
    );
    let source = recursive_source_image_with_public_x_out(&fixture, wrapped_x_out);
    let inputs = FPrimeRecursiveInputs {
        state: fixture.state.clone(),
        chunk_digest: fixture.chunk_digest,
        semantic_state_digest_out: acc,
        acc_digest_out: acc,
        nifs_msg: msg_from_fixture(&fixture),
        rows_in_chunk: 1,
        source_image: &source.image,
        chunk_count_in_word: source.chunk_count_in_word,
        step_count_in_word: source.step_count_in_word,
        pc_word: source.pc_word,
        prior_x_out_bits: source.prior_x_out_bits,
        public_x_out_bits: source.public_x_out_bits,
    };

    let mut b = R1csBuilder::new();
    enforce_f_prime_recursive_step_circuit(&mut b, fixture.prep.params(), &cfg, &inputs).expect("emit");
    assert!(
        !b.is_satisfied(),
        "F' must reject chunk_count transition p - 1 -> 0; Construction 2 counters are integers, not field elements"
    );
}

#[test]
fn f_prime_recursive_rejects_step_counter_field_modulus_wrap() {
    let fixture = build_fixture();
    let mut state = fixture.state.clone();
    state.chunk_count_in = 1;
    state.step_count_in = F::ORDER_U64 - 1;
    let fixture = rebuild_recursive_fixture_for_state(fixture, state);
    let cfg = make_step_config(&fixture.prep);

    let acc = recursive_acc_digest(&fixture);
    let wrapped_x_out = native_x_out(
        &fixture.state,
        fixture.chunk_digest,
        acc,
        acc,
        fixture.state.chunk_count_in + 1,
        0,
    );
    let source = recursive_source_image_with_public_x_out(&fixture, wrapped_x_out);
    let inputs = FPrimeRecursiveInputs {
        state: fixture.state.clone(),
        chunk_digest: fixture.chunk_digest,
        semantic_state_digest_out: acc,
        acc_digest_out: acc,
        nifs_msg: msg_from_fixture(&fixture),
        rows_in_chunk: 1,
        source_image: &source.image,
        chunk_count_in_word: source.chunk_count_in_word,
        step_count_in_word: source.step_count_in_word,
        pc_word: source.pc_word,
        prior_x_out_bits: source.prior_x_out_bits,
        public_x_out_bits: source.public_x_out_bits,
    };

    let mut b = R1csBuilder::new();
    enforce_f_prime_recursive_step_circuit(&mut b, fixture.prep.params(), &cfg, &inputs).expect("emit");
    assert!(
        !b.is_satisfied(),
        "F' must reject step_count transition p - 1 -> 0; Construction 2 counters are integers, not field elements"
    );
}

#[test]
fn f_prime_recursive_rejects_chunk_count_in_zero() {
    let fixture = build_fixture();
    let cfg = make_step_config(&fixture.prep);
    let mut state = fixture.state.clone();
    state.chunk_count_in = 0;
    let source = recursive_source_image(&fixture);
    let inputs = FPrimeRecursiveInputs {
        semantic_state_digest_out: recursive_acc_digest(&fixture),
        acc_digest_out: recursive_acc_digest(&fixture),
        state,
        chunk_digest: fixture.chunk_digest,
        nifs_msg: msg_from_fixture(&fixture),
        rows_in_chunk: 1,
        source_image: &source.image,
        chunk_count_in_word: source.chunk_count_in_word,
        step_count_in_word: source.step_count_in_word,
        pc_word: source.pc_word,
        prior_x_out_bits: source.prior_x_out_bits,
        public_x_out_bits: source.public_x_out_bits,
    };

    let mut b = R1csBuilder::new();
    let result = enforce_f_prime_recursive_step_circuit(&mut b, fixture.prep.params(), &cfg, &inputs);
    assert!(result.is_err(), "recursive must reject chunk_count_in == 0");
}

#[test]
fn f_prime_recursive_rejects_empty_fresh_batch() {
    let fixture = build_fixture();
    let cfg = make_step_config(&fixture.prep);
    let empty: Vec<CcsClaim> = Vec::new();
    let source = recursive_source_image(&fixture);
    let inputs = FPrimeRecursiveInputs {
        state: fixture.state.clone(),
        chunk_digest: fixture.chunk_digest,
        semantic_state_digest_out: recursive_acc_digest(&fixture),
        acc_digest_out: recursive_acc_digest(&fixture),
        nifs_msg: NifsVCircuitMessages {
            fresh: &empty,
            running: &fixture.running.claims,
            running_parent_authority: fixture.running.parent_authority.as_ref(),
            pi_ccs: &fixture.proof.pi_ccs,
            combined: &fixture.combined,
            children: &fixture.children,
        },
        rows_in_chunk: 1,
        source_image: &source.image,
        chunk_count_in_word: source.chunk_count_in_word,
        step_count_in_word: source.step_count_in_word,
        pc_word: source.pc_word,
        prior_x_out_bits: source.prior_x_out_bits,
        public_x_out_bits: source.public_x_out_bits,
    };

    let mut b = R1csBuilder::new();
    let result = enforce_f_prime_recursive_step_circuit(&mut b, fixture.prep.params(), &cfg, &inputs);
    assert!(
        result.is_err(),
        "recursive must reject empty fresh batch (SuperNeo K \u{2265} 1)"
    );
}

/// Multi-fresh chunk: a real NIFS proof with K=3 fresh CCS instances
/// (the whole batch rooted at one prior Construction-2 state) must
/// satisfy the F' R1CS recursive verifier — proving the in-circuit
/// public-link gate runs over **every** fresh `u_i`, not just `fresh[0]`.
#[test]
fn recursive_step_accepts_multi_fresh_batch() {
    let fixture = build_fixture_with_k_fresh(3);
    assert_eq!(fixture.fresh_claims.len(), 3);
    let cfg = make_step_config(&fixture.prep);
    let source = recursive_source_image(&fixture);
    let inputs = FPrimeRecursiveInputs {
        state: fixture.state.clone(),
        chunk_digest: fixture.chunk_digest,
        semantic_state_digest_out: recursive_acc_digest(&fixture),
        acc_digest_out: recursive_acc_digest(&fixture),
        nifs_msg: msg_from_fixture(&fixture),
        rows_in_chunk: 3,
        source_image: &source.image,
        chunk_count_in_word: source.chunk_count_in_word,
        step_count_in_word: source.step_count_in_word,
        pc_word: source.pc_word,
        prior_x_out_bits: source.prior_x_out_bits,
        public_x_out_bits: source.public_x_out_bits,
    };

    let mut b = R1csBuilder::new();
    enforce_f_prime_recursive_step_circuit(&mut b, fixture.prep.params(), &cfg, &inputs).expect("emit");
    assert!(
        b.is_satisfied(),
        "multi-fresh recursive F' step must satisfy when every fresh.x is linked to the shared prior x_out (first bad row: {:?})",
        b.first_unsatisfied_row()
    );
}

/// Negative companion to [`recursive_step_accepts_multi_fresh_batch`]:
/// the K=3 NIFS proof is generated honestly over a batch where
/// `fresh[2].x` encodes a **different** `enc_inst` than the honest prior
/// `x_out`. NIFS.V on its own accepts (the proof matches the actual
/// fresh handed in). The F' R1CS recursive-link gate must still reject,
/// because `source_image[prior_x_out_bits]` is the *honest* enc_inst and
/// only the per-fresh loop catches the divergent `fresh[2].x[1..]`.
///
/// This isolates the per-fresh recursive-link gate from NIFS.V's own
/// sumcheck/header binding: if the loop bailed after `fresh[0]`, this
/// test would (wrongly) pass.
#[test]
fn f_prime_recursive_rejects_multi_fresh_with_divergent_non_first_link() {
    let fixture = build_fixture_with_divergent_fresh(2);
    assert_eq!(fixture.fresh_claims.len(), 3);
    let cfg = make_step_config(&fixture.prep);
    let source = recursive_source_image(&fixture);
    let inputs = FPrimeRecursiveInputs {
        state: fixture.state.clone(),
        chunk_digest: fixture.chunk_digest,
        semantic_state_digest_out: recursive_acc_digest(&fixture),
        acc_digest_out: recursive_acc_digest(&fixture),
        nifs_msg: msg_from_fixture(&fixture),
        rows_in_chunk: 3,
        source_image: &source.image,
        chunk_count_in_word: source.chunk_count_in_word,
        step_count_in_word: source.step_count_in_word,
        pc_word: source.pc_word,
        prior_x_out_bits: source.prior_x_out_bits,
        public_x_out_bits: source.public_x_out_bits,
    };

    let mut b = R1csBuilder::new();
    enforce_f_prime_recursive_step_circuit(&mut b, fixture.prep.params(), &cfg, &inputs).expect("emit");
    assert!(
        !b.is_satisfied(),
        "F' R1CS must run the per-fresh recursive-link gate at every index — fresh[2].x diverged from \
         source_image[prior_x_out_bits] (honest), so the loop at i=2 must fail"
    );
}

#[test]
fn f_prime_recursive_rejects_fresh_m_in_mismatch() {
    let fixture = build_fixture();
    let cfg = make_step_config(&fixture.prep);
    // Hand-craft a fresh claim with `m_in` unequal to the physical carrier.
    let mut bad_fresh = fixture.fresh_claims.clone();
    bad_fresh[0].m_in = 4;
    bad_fresh[0].x = vec![F::ZERO; 4];
    let source = recursive_source_image(&fixture);
    let inputs = FPrimeRecursiveInputs {
        state: fixture.state.clone(),
        chunk_digest: fixture.chunk_digest,
        semantic_state_digest_out: recursive_acc_digest(&fixture),
        acc_digest_out: recursive_acc_digest(&fixture),
        nifs_msg: NifsVCircuitMessages {
            fresh: &bad_fresh,
            running: &fixture.running.claims,
            running_parent_authority: fixture.running.parent_authority.as_ref(),
            pi_ccs: &fixture.proof.pi_ccs,
            combined: &fixture.combined,
            children: &fixture.children,
        },
        rows_in_chunk: 1,
        source_image: &source.image,
        chunk_count_in_word: source.chunk_count_in_word,
        step_count_in_word: source.step_count_in_word,
        pc_word: source.pc_word,
        prior_x_out_bits: source.prior_x_out_bits,
        public_x_out_bits: source.public_x_out_bits,
    };

    let mut b = R1csBuilder::new();
    let result = enforce_f_prime_recursive_step_circuit(&mut b, fixture.prep.params(), &cfg, &inputs);
    assert!(
        result.is_err(),
        "recursive must reject fresh m_in != F_PRIME_SUPERNEO_PUBLIC_INPUT_LEN"
    );
}

// ── Recursive-step tamper tests ──────────────────────────────────────────
//
// Each tamper test flips ONE F'-side input field and asserts the circuit
// stops satisfying. NIFS.V-internal tampers (sumcheck round, header
// digest, and combined.y_ring) are already covered by the
// L-gate in `tests/reductions/nifs_v.rs`.

#[test]
fn f_prime_recursive_rejects_tampered_public_x_out_bits() {
    let fixture = build_fixture();
    let cfg = make_step_config(&fixture.prep);
    let mut source = recursive_source_image(&fixture);
    // Flip the first bit of the enc_inst body in the source image itself —
    // SourceImageWires::alloc will pick up the tampered value.
    let idx = source.public_x_out_bits.start();
    let original = source.image.values()[idx];
    source.image.set_bit(idx, original == F::ZERO);
    let inputs = FPrimeRecursiveInputs {
        state: fixture.state.clone(),
        chunk_digest: fixture.chunk_digest,
        semantic_state_digest_out: recursive_acc_digest(&fixture),
        acc_digest_out: recursive_acc_digest(&fixture),
        nifs_msg: msg_from_fixture(&fixture),
        rows_in_chunk: 1,
        source_image: &source.image,
        chunk_count_in_word: source.chunk_count_in_word,
        step_count_in_word: source.step_count_in_word,
        pc_word: source.pc_word,
        prior_x_out_bits: source.prior_x_out_bits,
        public_x_out_bits: source.public_x_out_bits,
    };

    let mut b = R1csBuilder::new();
    enforce_f_prime_recursive_step_circuit(&mut b, fixture.prep.params(), &cfg, &inputs).expect("emit");
    assert!(
        !b.is_satisfied(),
        "F' recursive must reject tampered enc_inst(x_out) public output bits"
    );
}

#[test]
fn f_prime_recursive_rejects_tampered_acc_digest_in() {
    let fixture = build_fixture();
    let cfg = make_step_config(&fixture.prep);
    let mut state = fixture.state.clone();
    state.acc_digest_in[0] += F::ONE;
    let source = recursive_source_image(&fixture);
    let inputs = FPrimeRecursiveInputs {
        semantic_state_digest_out: recursive_acc_digest(&fixture),
        acc_digest_out: recursive_acc_digest(&fixture),
        state,
        chunk_digest: fixture.chunk_digest,
        nifs_msg: msg_from_fixture(&fixture),
        rows_in_chunk: 1,
        source_image: &source.image,
        chunk_count_in_word: source.chunk_count_in_word,
        step_count_in_word: source.step_count_in_word,
        pc_word: source.pc_word,
        prior_x_out_bits: source.prior_x_out_bits,
        public_x_out_bits: source.public_x_out_bits,
    };

    let mut b = R1csBuilder::new();
    enforce_f_prime_recursive_step_circuit(&mut b, fixture.prep.params(), &cfg, &inputs).expect("emit");
    assert!(
        !b.is_satisfied(),
        "F' recursive must reject acc_digest_in that doesn't match digest(running)"
    );
}

#[test]
fn f_prime_recursive_rejects_tampered_chunk_digest() {
    let fixture = build_fixture();
    let cfg = make_step_config(&fixture.prep);
    let mut bad_chunk_digest = fixture.chunk_digest;
    bad_chunk_digest[0] += F::ONE;
    let source = recursive_source_image(&fixture);
    let inputs = FPrimeRecursiveInputs {
        state: fixture.state.clone(),
        chunk_digest: bad_chunk_digest,
        semantic_state_digest_out: recursive_acc_digest(&fixture),
        acc_digest_out: recursive_acc_digest(&fixture),
        nifs_msg: msg_from_fixture(&fixture),
        rows_in_chunk: 1,
        source_image: &source.image,
        chunk_count_in_word: source.chunk_count_in_word,
        step_count_in_word: source.step_count_in_word,
        pc_word: source.pc_word,
        prior_x_out_bits: source.prior_x_out_bits,
        public_x_out_bits: source.public_x_out_bits,
    };

    let mut b = R1csBuilder::new();
    enforce_f_prime_recursive_step_circuit(&mut b, fixture.prep.params(), &cfg, &inputs).expect("emit");
    assert!(
        !b.is_satisfied(),
        "F' recursive must reject chunk_digest that diverges from native pre-NIFS transcript absorb"
    );
}

#[test]
fn f_prime_recursive_rejects_tampered_fresh_x_bit() {
    let fixture = build_fixture();
    let cfg = make_step_config(&fixture.prep);
    let mut bad_fresh = fixture.fresh_claims.clone();
    // Flip one enc_inst bit of fresh.x (the first body bit, index 1 —
    // index 0 is the CCS constant-one slot). enc_inst(prior_x_out) check
    // breaks immediately; sumcheck/header challenges diverge as well.
    let bit_idx = 1; // F_PRIME_ENC_INST_OFFSET
    let v = bad_fresh[0].x[bit_idx];
    bad_fresh[0].x[bit_idx] = if v == F::ZERO { F::ONE } else { F::ZERO };
    let source = recursive_source_image(&fixture);
    let inputs = FPrimeRecursiveInputs {
        state: fixture.state.clone(),
        chunk_digest: fixture.chunk_digest,
        semantic_state_digest_out: recursive_acc_digest(&fixture),
        acc_digest_out: recursive_acc_digest(&fixture),
        nifs_msg: NifsVCircuitMessages {
            fresh: &bad_fresh,
            running: &fixture.running.claims,
            running_parent_authority: fixture.running.parent_authority.as_ref(),
            pi_ccs: &fixture.proof.pi_ccs,
            combined: &fixture.combined,
            children: &fixture.children,
        },
        rows_in_chunk: 1,
        source_image: &source.image,
        chunk_count_in_word: source.chunk_count_in_word,
        step_count_in_word: source.step_count_in_word,
        pc_word: source.pc_word,
        prior_x_out_bits: source.prior_x_out_bits,
        public_x_out_bits: source.public_x_out_bits,
    };

    let mut b = R1csBuilder::new();
    enforce_f_prime_recursive_step_circuit(&mut b, fixture.prep.params(), &cfg, &inputs).expect("emit");
    assert!(
        !b.is_satisfied(),
        "F' recursive must reject fresh.x that doesn't encode prior_x_out"
    );
}

#[test]
fn f_prime_recursive_rejects_nonzero_fresh_carrier_padding_when_nifs_proof_matches() {
    // The weak carrier relation accepts any low-norm public vector, so build
    // NIFS.P honestly over a fresh claim whose first ring-completion lane is
    // one. NIFS.V therefore agrees with the claim; only F′'s verifier-owned
    // 270-coordinate carrier contract is responsible for rejecting it.
    let fixture = build_fixture_with_k_fresh_and_public_carrier(1, F::ONE, Some(F::ONE));
    assert_eq!(fixture.fresh_claims[0].x[F_PRIME_PUBLIC_INPUT_LEN], F::ONE);
    let cfg = make_step_config(&fixture.prep);
    let source = recursive_source_image(&fixture);
    let inputs = FPrimeRecursiveInputs {
        state: fixture.state.clone(),
        chunk_digest: fixture.chunk_digest,
        semantic_state_digest_out: recursive_acc_digest(&fixture),
        acc_digest_out: recursive_acc_digest(&fixture),
        nifs_msg: msg_from_fixture(&fixture),
        rows_in_chunk: 1,
        source_image: &source.image,
        chunk_count_in_word: source.chunk_count_in_word,
        step_count_in_word: source.step_count_in_word,
        pc_word: source.pc_word,
        prior_x_out_bits: source.prior_x_out_bits,
        public_x_out_bits: source.public_x_out_bits,
    };

    let mut builder = R1csBuilder::new();
    enforce_f_prime_recursive_step_circuit(&mut builder, fixture.prep.params(), &cfg, &inputs).expect("emit");
    assert!(
        !builder.is_satisfied(),
        "F′ accepted nonzero fresh carrier padding even though that padding is verifier-fixed"
    );
}

#[test]
fn f_prime_recursive_rejects_fresh_public_one_slot_not_one_even_when_nifs_proof_matches() {
    // Build the NIFS proof honestly for a fresh public input whose body is
    // the correct enc_inst(prior_x_out), but whose CCS constant-one slot is
    // zero. NIFS.V itself is consistent with that fresh claim; the F'
    // recursive-link shell must reject via the explicit x[0] == 1 row.
    let fixture = build_fixture_with_k_fresh_and_public_one(1, F::ZERO);
    let cfg = make_step_config(&fixture.prep);
    let source = recursive_source_image(&fixture);
    let inputs = FPrimeRecursiveInputs {
        state: fixture.state.clone(),
        chunk_digest: fixture.chunk_digest,
        semantic_state_digest_out: recursive_acc_digest(&fixture),
        acc_digest_out: recursive_acc_digest(&fixture),
        nifs_msg: msg_from_fixture(&fixture),
        rows_in_chunk: 1,
        source_image: &source.image,
        chunk_count_in_word: source.chunk_count_in_word,
        step_count_in_word: source.step_count_in_word,
        pc_word: source.pc_word,
        prior_x_out_bits: source.prior_x_out_bits,
        public_x_out_bits: source.public_x_out_bits,
    };

    let mut b = R1csBuilder::new();
    enforce_f_prime_recursive_step_circuit(&mut b, fixture.prep.params(), &cfg, &inputs).expect("emit");
    assert!(
        !b.is_satisfied(),
        "F' recursive accepted a fresh public input with x[0] != 1 even though the NIFS proof matched it"
    );
}

#[test]
fn f_prime_recursive_rejects_nonbinary_source_image_public_bit() {
    // SourceImageWires::alloc enforces bitness on every source-image
    // coordinate. Tamper one of the public-x_out source-image bits to a
    // non-{0,1} value and verify F' rejects it — independent of the
    // enc_inst(x_out) algebraic check.
    let fixture = build_fixture();
    let cfg = make_step_config(&fixture.prep);
    let mut source = recursive_source_image(&fixture);
    source
        .image
        .set_raw(source.public_x_out_bits.start(), F::from_u64(2));
    let inputs = FPrimeRecursiveInputs {
        state: fixture.state.clone(),
        chunk_digest: fixture.chunk_digest,
        semantic_state_digest_out: recursive_acc_digest(&fixture),
        acc_digest_out: recursive_acc_digest(&fixture),
        nifs_msg: msg_from_fixture(&fixture),
        rows_in_chunk: 1,
        source_image: &source.image,
        chunk_count_in_word: source.chunk_count_in_word,
        step_count_in_word: source.step_count_in_word,
        pc_word: source.pc_word,
        prior_x_out_bits: source.prior_x_out_bits,
        public_x_out_bits: source.public_x_out_bits,
    };

    let mut b = R1csBuilder::new();
    enforce_f_prime_recursive_step_circuit(&mut b, fixture.prep.params(), &cfg, &inputs).expect("emit");
    assert!(
        !b.is_satisfied(),
        "non-binary source-image bit must be rejected by bitness constraint"
    );
}

// ── Step 3: input-link source-image tampers ──────────────────────────────

#[test]
fn f_prime_recursive_rejects_tampered_prior_source_image_bit() {
    // F' constrains `source_image[prior_x_out_bits] == enc_inst(prior_x_out)`.
    // Flipping one prior-image bit while leaving prior_x_out (computed
    // in-circuit from honest state-in) untouched must fail.
    let fixture = build_fixture();
    let cfg = make_step_config(&fixture.prep);
    let mut source = recursive_source_image(&fixture);
    let idx = source.prior_x_out_bits.start();
    let original = source.image.values()[idx];
    source.image.set_bit(idx, original == F::ZERO);
    let inputs = FPrimeRecursiveInputs {
        state: fixture.state.clone(),
        chunk_digest: fixture.chunk_digest,
        semantic_state_digest_out: recursive_acc_digest(&fixture),
        acc_digest_out: recursive_acc_digest(&fixture),
        nifs_msg: msg_from_fixture(&fixture),
        rows_in_chunk: 1,
        source_image: &source.image,
        chunk_count_in_word: source.chunk_count_in_word,
        step_count_in_word: source.step_count_in_word,
        pc_word: source.pc_word,
        prior_x_out_bits: source.prior_x_out_bits,
        public_x_out_bits: source.public_x_out_bits,
    };

    let mut b = R1csBuilder::new();
    enforce_f_prime_recursive_step_circuit(&mut b, fixture.prep.params(), &cfg, &inputs).expect("emit");
    assert!(
        !b.is_satisfied(),
        "tampered prior source-image bit must break recursive input link"
    );
}

#[test]
fn f_prime_recursive_rejects_fresh_x_not_matching_prior_source_image() {
    // F' wire-to-wire-equates `fresh[0].x[1..] == source_image[prior_x_out_bits]`.
    // Tampering fresh.x[1] inside the NIFS proof — but leaving the
    // source image honest — must fail at that equality (the NIFS algebraic
    // checks already reject it too, but this test exercises the
    // source-image binding specifically).
    let fixture = build_fixture();
    let cfg = make_step_config(&fixture.prep);
    let source = recursive_source_image(&fixture);
    let mut bad_fresh = fixture.fresh_claims.clone();
    let bit_idx = 1; // F_PRIME_ENC_INST_OFFSET
    let v = bad_fresh[0].x[bit_idx];
    bad_fresh[0].x[bit_idx] = if v == F::ZERO { F::ONE } else { F::ZERO };
    let inputs = FPrimeRecursiveInputs {
        state: fixture.state.clone(),
        chunk_digest: fixture.chunk_digest,
        semantic_state_digest_out: recursive_acc_digest(&fixture),
        acc_digest_out: recursive_acc_digest(&fixture),
        nifs_msg: NifsVCircuitMessages {
            fresh: &bad_fresh,
            running: &fixture.running.claims,
            running_parent_authority: fixture.running.parent_authority.as_ref(),
            pi_ccs: &fixture.proof.pi_ccs,
            combined: &fixture.combined,
            children: &fixture.children,
        },
        rows_in_chunk: 1,
        source_image: &source.image,
        chunk_count_in_word: source.chunk_count_in_word,
        step_count_in_word: source.step_count_in_word,
        pc_word: source.pc_word,
        prior_x_out_bits: source.prior_x_out_bits,
        public_x_out_bits: source.public_x_out_bits,
    };

    let mut b = R1csBuilder::new();
    enforce_f_prime_recursive_step_circuit(&mut b, fixture.prep.params(), &cfg, &inputs).expect("emit");
    assert!(!b.is_satisfied(), "fresh.x must match source-image prior enc_inst bits");
}

// ── Step 4: counter source-image tampers ─────────────────────────────────

#[test]
fn f_prime_recursive_rejects_source_image_chunk_count_mismatch() {
    // F' constrains `sw.chunk_count_in == decode(source_image.chunk_count_in_word)`.
    // Flip one bit of the chunk_count word in the source image while the
    // `state.chunk_count_in` field stays honest — the wire-to-LC equality
    // must fail.
    let fixture = build_fixture();
    let cfg = make_step_config(&fixture.prep);
    let mut source = recursive_source_image(&fixture);
    let idx = source.chunk_count_in_word.bits().start();
    source
        .image
        .set_bit(idx, source.image.values()[idx] == F::ZERO);
    let inputs = FPrimeRecursiveInputs {
        state: fixture.state.clone(),
        chunk_digest: fixture.chunk_digest,
        semantic_state_digest_out: recursive_acc_digest(&fixture),
        acc_digest_out: recursive_acc_digest(&fixture),
        nifs_msg: msg_from_fixture(&fixture),
        rows_in_chunk: 1,
        source_image: &source.image,
        chunk_count_in_word: source.chunk_count_in_word,
        step_count_in_word: source.step_count_in_word,
        pc_word: source.pc_word,
        prior_x_out_bits: source.prior_x_out_bits,
        public_x_out_bits: source.public_x_out_bits,
    };

    let mut b = R1csBuilder::new();
    enforce_f_prime_recursive_step_circuit(&mut b, fixture.prep.params(), &cfg, &inputs).expect("emit");
    assert!(
        !b.is_satisfied(),
        "source-image chunk_count word must match in-circuit state var"
    );
}

#[test]
fn f_prime_recursive_rejects_source_image_step_count_mismatch() {
    // Same Step-4 source-image binding as chunk_count, but for
    // `step_count_in`. HyperNova's state hash absorbs the step counter;
    // a prover must not be able to route a stale or forged source-image
    // word while keeping the in-circuit state field honest.
    let fixture = build_fixture();
    let cfg = make_step_config(&fixture.prep);
    let mut source = recursive_source_image(&fixture);
    let idx = source.step_count_in_word.bits().start();
    source
        .image
        .set_bit(idx, source.image.values()[idx] == F::ZERO);
    let inputs = FPrimeRecursiveInputs {
        state: fixture.state.clone(),
        chunk_digest: fixture.chunk_digest,
        semantic_state_digest_out: recursive_acc_digest(&fixture),
        acc_digest_out: recursive_acc_digest(&fixture),
        nifs_msg: msg_from_fixture(&fixture),
        rows_in_chunk: 1,
        source_image: &source.image,
        chunk_count_in_word: source.chunk_count_in_word,
        step_count_in_word: source.step_count_in_word,
        pc_word: source.pc_word,
        prior_x_out_bits: source.prior_x_out_bits,
        public_x_out_bits: source.public_x_out_bits,
    };

    let mut b = R1csBuilder::new();
    enforce_f_prime_recursive_step_circuit(&mut b, fixture.prep.params(), &cfg, &inputs).expect("emit");
    assert!(
        !b.is_satisfied(),
        "source-image step_count word must match in-circuit state var"
    );
}

#[test]
fn f_prime_recursive_rejects_noncanonical_source_image_pc_word() {
    // Overwrite pc's source-image word with `p + 1`. This is noncanonical
    // but field-decodes to the honest single-program pc (`TRIVIAL_PC = 1`).
    // Without the Step 4 canonicality row, the pc equality row alone would
    // accept this source image.
    let fixture = build_fixture();
    let cfg = make_step_config(&fixture.prep);
    let mut source = recursive_source_image(&fixture);
    let start = source.pc_word.bits().start();
    let noncanonical: u64 = 0xFFFF_FFFF_0000_0002;
    for i in 0..64 {
        source
            .image
            .set_bit(start + i, ((noncanonical >> i) & 1) == 1);
    }
    let inputs = FPrimeRecursiveInputs {
        state: fixture.state.clone(),
        chunk_digest: fixture.chunk_digest,
        semantic_state_digest_out: recursive_acc_digest(&fixture),
        acc_digest_out: recursive_acc_digest(&fixture),
        nifs_msg: msg_from_fixture(&fixture),
        rows_in_chunk: 1,
        source_image: &source.image,
        chunk_count_in_word: source.chunk_count_in_word,
        step_count_in_word: source.step_count_in_word,
        pc_word: source.pc_word,
        prior_x_out_bits: source.prior_x_out_bits,
        public_x_out_bits: source.public_x_out_bits,
    };

    let mut b = R1csBuilder::new();
    enforce_f_prime_recursive_step_circuit(&mut b, fixture.prep.params(), &cfg, &inputs).expect("emit");
    assert!(
        !b.is_satisfied(),
        "source-image pc word must be canonical Goldilocks (< p)"
    );
}

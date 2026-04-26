use std::fs;
use std::path::PathBuf;

fn crate_path(path: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(path)
}

fn workspace_path(path: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join(path)
}

#[test]
fn rv64im_ivc_native_module_does_not_reference_spartan2() {
    let source = fs::read_to_string(crate_path("src/rv64im/ivc.rs")).expect("read native IVC module");
    assert!(
        !source.contains("spartan2"),
        "native RV64IM IVC ownership must stay Spartan-free"
    );
}

#[test]
fn rv64im_compressed_main_proof_is_not_a_verifier() {
    let main_proof =
        fs::read_to_string(crate_path("src/rv64im/main_proof.rs")).expect("read compact main proof module");
    let forbidden_wrapper_verify = ["pub fn verify", "(&self, ivc_recursion"].concat();
    assert!(
        main_proof.contains("pub fn expected_ivc_public_image") && !main_proof.contains(&forbidden_wrapper_verify),
        "Rv64imCompressedMainProof may derive the public image, but proof acceptance must call Rv64imIvcSnark::verify"
    );

    let nightstream_verify = fs::read_to_string(crate_path("src/nightstream/rv64im/verify_perf.rs"))
        .expect("read Nightstream verify timing module");
    assert!(
        nightstream_verify.contains(".ivc_snark()") && nightstream_verify.contains(".verify(ivc_recursion_snark_vk"),
        "Nightstream main-proof verification must route through Rv64imIvcSnark::verify"
    );
}

#[test]
fn rv64im_nightstream_verifier_context_binds_actual_ivc_snark_vk() {
    let nightstream = fs::read_to_string(crate_path("src/nightstream/rv64im.rs")).expect("read Nightstream RV64IM");
    let verify_perf =
        fs::read_to_string(crate_path("src/nightstream/rv64im/verify_perf.rs")).expect("read Nightstream verifier");
    assert!(
        nightstream.contains("ivc_recursion_snark_vk: &Rv64imIvcSnarkVerifierKey")
            && nightstream.contains("ivc_recursion_snark_vk.expected_digest()?")
            && nightstream.contains("fn rv64im_verifier_context_digest_from_key_digest")
            && !nightstream.contains("pub fn rv64im_verifier_context_digest_from_key_digest")
            && nightstream.contains("neo.fold.next/nightstream/rv64im/verifier_context/version\", b\"v3\"")
            && nightstream.contains("neo.fold.next/nightstream/rv64im/verifier_context/ivc_recursion_snark_vk"),
        "Nightstream verifier context must bind the actual RV64IM IVC SNARK verifier key, not a caller-supplied digest"
    );
    assert!(
        verify_perf.contains("rv64im_verifier_context_digest(")
            && verify_perf.contains("ivc_recursion_snark_vk")
            && verify_perf.contains("trusted root parameters and IVC verifier key"),
        "Nightstream verification errors and checks must make the IVC verifier-key binding explicit"
    );
}

#[test]
fn rv64im_ivc_compression_module_owns_explicit_compress_boundary() {
    let source = fs::read_to_string(crate_path("src/rv64im/ivc_snark.rs")).expect("read IVC compression module");
    assert!(
        source.contains("impl Rv64imIvcState")
            && source.contains("pub fn compress(&self)")
            && source.contains("verify_rv64im_final_construction2_boundary")
            && source.contains("verify_rv64im_final_ce_bundle")
            && source.contains("setup_rv64im_terminal_f_prime_committed_step_relation")
            && source.contains("prove_rv64im_terminal_f_prime_committed_step_relation")
            && source.contains("verify_rv64im_terminal_f_prime_committed_step_relation")
            && source.contains("final_ce_shape")
            && !source.contains("prove_rv64im_chunk_step_ivc_spartan"),
        "RV64IM IVC compression must stay on the final Construction-2 boundary and cache all sub-proof shapes, not terminal chunk-step Spartan"
    );
}

#[test]
fn rv64im_ivc_compression_has_no_replay_or_terminal_step_acceptance_fallback() {
    for path in [
        "src/rv64im/ivc.rs",
        "src/rv64im/ivc_snark.rs",
        "src/rv64im/main_proof.rs",
        "src/nightstream/rv64im/verify_perf.rs",
    ] {
        let source = fs::read_to_string(crate_path(path)).expect("read RV64IM verifier source");
        assert!(
            !source.contains("verify_against_final")
                && !source.contains("validate_standalone_rv64im_ivc_snark_scope")
                && !source.contains("build_terminal_relation")
                && !source.contains("Rv64imIvcState::verify")
                && !source.contains("validate_replayed_chain_matches"),
            "{path} must not expose full-chain replay or terminal chunk-step proof acceptance"
        );
    }
}

#[test]
fn rv64im_verifiers_do_not_accept_precomputed_public_digests() {
    for path in [
        "src/verifier.rs",
        "src/run.rs",
        "src/finalize.rs",
        "src/rv64im/kernel/simple.rs",
        "src/rv64im/kernel/proof/staged_verify.rs",
    ] {
        let source = fs::read_to_string(crate_path(path)).expect("read verifier source");
        assert!(
            !source.contains("precomputed_digest")
                && !source.contains("precomputed_chunk")
                && !source.contains("verify_chunk_with_precomputed"),
            "{path} must recompute theorem-facing public digests from authoritative public inputs"
        );
    }
}

#[test]
fn rv64im_chunk_step_ivc_uses_named_authoritative_ce_projection() {
    let source = fs::read_to_string(crate_path("src/rv64im/chunk_step_ivc.rs")).expect("read chunk-step IVC module");
    assert!(
        source.contains("struct Rv64imCarriedCeProjection")
            && source.contains("commitment_data: Vec<F>")
            && source.contains("compact_x: Vec<F>")
            && source.contains("Rv64imCarriedCeProjection::from_claim")
            && source.contains("claim.X.rows() != neo_math::D || claim.X.cols() != claim.m_in")
            && source.contains("rv64im_ce_claims_match_projection")
            && source.contains("(Some(lhs), Some(rhs)) => lhs == rhs")
            && !source.contains("(0..lhs.m_in).all(|col|"),
        "chunk-step IVC carry comparison must use a named authoritative CE projection, not an inline partial equality"
    );
}

#[test]
fn rv64im_recursive_step_binds_current_construction2_input_hash_image() {
    let source = fs::read_to_string(crate_path("src/rv64im/main_relation_spartan/recursive_step.rs"))
        .expect("read recursive-step circuit");
    let public_boundary = fs::read_to_string(crate_path(
        "src/rv64im/main_relation_spartan/recursive_step/construction2_public.rs",
    ))
    .expect("read Construction-2 public boundary circuit");
    assert!(
        source.contains("live_folded_accumulator_in_digest")
            && source.contains("recursive_accumulator_instance_digest_circuit_from_claims")
            && source.contains("&chunk_replay.state_in_claims")
            && source.contains("current_x_i_digest")
            && source.contains("current_construction2_u_i_x_i_eq")
            && source.contains("enforce_digest_eq_when_non_base"),
        "terminal F' must bind prior u_i.x_i to the live CE accumulator hash image for non-base steps"
    );
    assert!(
        source.contains("&statement_chunk_index_halves,\n        \"current_construction2_u_i_x_i_eq\"")
            && public_boundary.contains("chunk_count_in_halves")
            && !public_boundary.contains("- CS::one()"),
        "current u_i.x_i hash-image equality must be gated by chunk_count_in so HyperNova's base case i=0 is skipped"
    );
}

#[test]
fn rv64im_recursive_step_enforces_canonical_base_u_perp() {
    let source = fs::read_to_string(crate_path(
        "src/rv64im/main_relation_spartan/recursive_step/construction2_witness.rs",
    ))
    .expect("read recursive-step Construction-2 witness circuit");
    assert!(
        source.contains("enforce_current_input_u_perp_when_base")
            && source.contains("base_current_input_u_perp")
            && source.contains("build_rv64im_main_recursion_construction2_default_fresh_instance")
            && source.contains("chunk_count_is_zero")
            && source.contains("base_current_input_commitment_")
            && source.contains("base_current_input_x_i_bit_"),
        "recursive-step circuit must enforce canonical u_perp for the private current Construction-2 input at i=0"
    );
}

#[test]
fn rv64im_terminal_f_prime_committed_step_uses_superneo_committed_ccs_authority() {
    let recursive_step = fs::read_to_string(crate_path("src/rv64im/main_relation_spartan/recursive_step.rs"))
        .expect("read recursive-step circuit");
    let public_target = fs::read_to_string(crate_path(
        "src/rv64im/main_relation_spartan/recursive_step/public_target.rs",
    ))
    .expect("read recursive-step public target");
    let recursive_witness = fs::read_to_string(crate_path(
        "src/rv64im/main_relation_spartan/recursive_step/construction2_witness.rs",
    ))
    .expect("read recursive-step Construction-2 witness circuit");
    let compression = fs::read_to_string(crate_path("src/rv64im/ivc_snark.rs")).expect("read IVC compression module");
    let terminal_committed_owner = fs::read_to_string(crate_path("src/rv64im/ivc_snark/terminal_f_prime_committed.rs"))
        .expect("read terminal F' committed-step owner");
    let terminal_committed_circuit = fs::read_to_string(crate_path(
        "src/rv64im/ivc_snark/terminal_f_prime_committed/proof_circuit.rs",
    ))
    .expect("read terminal F' committed-step circuit");
    let terminal_committed = format!("{terminal_committed_owner}\n{terminal_committed_circuit}");

    assert!(
        !crate_path("src/rv64im/ivc_snark/construction2_opening.rs").exists()
            && !crate_path("src/rv64im/ivc_snark/terminal_committed_step.rs").exists()
            && !crate_path("src/rv64im/ivc_snark/terminal_f_prime_relation.rs").exists()
            && !compression.contains("mod construction2_opening")
            && !compression.contains("mod terminal_committed_step")
            && !compression.contains("mod terminal_f_prime_relation")
            && !compression.contains("enforce_seeded_ajtai_binary_opening"),
        "legacy bit-opening and extracted public-first terminal relation modules must not be part of RV64IM compressed-proof authority"
    );
    assert!(
        !recursive_step.contains("construction2_witness_digest")
            && !recursive_step.contains("Rv64imMainRecursionStepPublicIoMode")
            && !recursive_witness.contains("construction2_witness_digest"),
        "recursive-step terminal F' proof must not expose the removed witness-digest public IO mode"
    );
    assert!(
        compression.contains("committed_step_circuit")
            && compression.contains("terminal_f_prime_committed_step_proof")
            && compression.contains("verify_rv64im_terminal_f_prime_committed_step_relation")
            && compression.contains("SuperNeo low-norm bound")
            && compression.contains("rv64im_terminal_f_prime_r2_public_values_from_public_image")
            && compression.contains("terminal_f_prime_committed_step_boundary_public_values")
            && compression.contains("terminal_f_prime_r2_public_values")
            && !compression.contains("construction2_witness_digest")
            && !recursive_step.contains("construction2_u_i_commitment_opening")
            && !recursive_witness.contains("enforce_construction2_output_commitment_from_live_witness")
            && !recursive_witness.contains("alloc_logical_field_bits_le")
            && !recursive_witness.contains("packed_commitment_entries_from_logical_bits"),
        "compressed terminal F' proof must not keep the old logical-image Ajtai opener in the recursive circuit"
    );
    assert!(
        terminal_committed.contains("Rv64imTerminalFPrimeR1csCcsRelation")
            && terminal_committed.contains("sparse_r1cs_to_ccs")
            && terminal_committed.contains("build_rv64im_terminal_f_prime_r2_circuit")
            && terminal_committed.contains("enforce_ajtai_commitment_linear_consistency")
            && terminal_committed.contains("enforce_rowwise_terminal_r2")
            && terminal_committed.contains("committed_full_vector")
            && terminal_committed.contains("enforce_committed_superneo_image")
            && terminal_committed.contains("terminal_r2_public_bit_bound")
            && terminal_committed.contains("terminal_r2_witness_bit_bound")
            && terminal_committed.contains("bit_column_terms")
            && terminal_committed.contains("low_norm_bit_values")
            && terminal_committed.contains("UnusedPadding")
            && terminal_committed.contains("padded_private_witness_labels")
            && terminal_committed.contains("num_shared_unpadded")
            && terminal_committed.contains("num_precommitted_unpadded")
            && terminal_committed.contains("num_rest_unpadded")
            && terminal_committed.contains("sparse R1CS matrix references an unused padded witness column")
            && !terminal_committed.contains("return Self::U64;")
            && !terminal_committed.contains("LOW_NORM_LIMB")
            && !terminal_committed.contains("enforce_u32_allocated")
            && terminal_committed.contains("terminal_r2_superneo_padding_zero")
            && terminal_committed.contains("terminal_f_prime_r2_public_value_range")
            && !terminal_committed.contains("build_rv64im_main_recursion_construction2_committed_f_prime_full_vector")
            && terminal_committed.contains("encode_vector_for_full_width")
            && terminal_committed.contains("commit_rv64im_main_recursion_construction2_packed_z")
            && terminal_committed.contains("require_superneo_assignment_commitment")
            && terminal_committed.contains("commitment_matches_public_boundary")
            && terminal_committed.contains("ensure_terminal_f_prime_output_uses_x_only_placeholder")
            && terminal_committed.contains("commitment.kappa != 1")
            && terminal_committed.contains("commitment.data.len() != D")
            && terminal_committed.contains("terminal_f_prime_committed_step_boundary_public_values")
            && terminal_committed.contains("construction2_public_boundary_digest_circuit")
            && terminal_committed.contains("terminal_boundary_x_i_eq")
            && terminal_committed.contains("terminal_boundary_commitment_kappa_matches_data_len")
            && terminal_committed.contains("boundary.commitment_data.len() % D != 0")
            && !terminal_committed.contains("terminal_boundary_commitment_kappa_matches_packed_z_cols")
            && !terminal_committed.contains("FPrimeWitnessImage")
            && !terminal_committed.contains("legacy_binary_full_width"),
        "terminal F' committed-step ownership must commit a low-norm SuperNeo image, prove its Construction-2 public boundary, and use it to recompose the terminal R2 assignment"
    );
    let terminal_r2_public_values = public_target
        .split("pub fn terminal_f_prime_r2_public_values")
        .nth(1)
        .and_then(|tail| {
            tail.split("pub fn terminal_f_prime_r2_public_value_range")
                .next()
        })
        .expect("terminal F' R2 public value method");
    assert!(
        terminal_r2_public_values.contains("x_out.field_image()")
            && terminal_r2_public_values.contains("folded_accumulator_out_digest")
            && terminal_r2_public_values.contains("bridge_handoff_digest")
            && !terminal_r2_public_values.contains("construction2_u_i"),
        "terminal F' R2 public IO must exclude the output u_i.C boundary so the committed Z is not self-referential"
    );
}

#[test]
fn rv64im_ce_circuit_accepts_only_superneo_packed_witness_shape() {
    let witness = fs::read_to_string(crate_path("src/rv64im/main_relation_circuit/witness.rs"))
        .expect("read circuit witness module");
    let ce_consistency = fs::read_to_string(crate_path("src/rv64im/main_relation_circuit/ce_consistency.rs"))
        .expect("read CE consistency module");

    assert!(
        witness.contains("self.cols != expected_m.div_ceil(D)")
            && !witness.contains("self.cols == expected_m")
            && !ce_consistency.contains("return Ok(expected_m)"),
        "RV64IM CE circuit witnesses must reject dense/unpacked compatibility layout and accept only SuperNeo D x ceil(m/D) packing"
    );
}

#[test]
fn rv64im_terminal_f_prime_relation_extraction_legacy_is_removed() {
    let compression = fs::read_to_string(crate_path("src/rv64im/ivc_snark.rs")).expect("read IVC compression module");
    assert!(
        !crate_path("src/rv64im/ivc_snark/terminal_f_prime_relation.rs").exists()
            && !compression.contains("public_first_committed_ccs")
            && !compression.contains("Rv64imTerminalFPrimePublishedCcs")
            && !compression.contains("build_terminal_f_prime_published_r1cs_ccs_relation"),
        "RV64IM compression must not keep the public-first extracted terminal relation compatibility path"
    );
}

#[test]
fn rv64im_construction2_current_input_helper_does_not_keep_witness_digest_cargo() {
    let construction2 = fs::read_to_string(crate_path("src/rv64im/construction2.rs")).expect("read Construction-2");
    let recursive_witness = fs::read_to_string(crate_path(
        "src/rv64im/main_relation_spartan/recursive_step/construction2_witness.rs",
    ))
    .expect("read recursive-step Construction-2 witness digest");
    let nifs_stages =
        fs::read_to_string(crate_path("src/rv64im/main_relation_spartan/nifs_v_stages.rs")).expect("read NIFS stages");
    let pi_ccs = fs::read_to_string(crate_path("src/rv64im/main_relation_circuit/pi_ccs.rs")).expect("read Pi_CCS");
    let compression = fs::read_to_string(crate_path("src/rv64im/ivc_snark.rs")).expect("read IVC compression module");
    let terminal_committed_owner = fs::read_to_string(crate_path("src/rv64im/ivc_snark/terminal_f_prime_committed.rs"))
        .expect("read terminal F' committed-step owner");
    let terminal_committed_circuit = fs::read_to_string(crate_path(
        "src/rv64im/ivc_snark/terminal_f_prime_committed/proof_circuit.rs",
    ))
    .expect("read terminal F' committed-step circuit");
    let terminal_committed = format!("{terminal_committed_owner}\n{terminal_committed_circuit}");

    assert!(
        !construction2.contains("append_state_in_claim_fields")
            && !construction2.contains("build_rv64im_main_recursion_construction2_committed_f_prime_full_vector")
            && !construction2.contains("Rv64imMainRecursionConstruction2FPrimeWitnessImage")
            && !construction2.contains("construction2_f_prime_witness_digest_values")
            && !construction2.contains("digest_layout")
            && !construction2.contains("commit_binary_columns")
            && !construction2.contains("PackedBinaryMatBitSink")
            && !construction2.contains("SeededBinaryColsCommitAudit"),
        "native Construction-2 witness image must not keep the removed witness-digest compatibility path"
    );
    assert!(
        terminal_committed.contains("Rv64imTerminalFPrimeR1csCcsRelation")
            && terminal_committed.contains("sparse_r1cs_to_ccs")
            && terminal_committed.contains("build_rv64im_terminal_f_prime_r2_circuit")
            && terminal_committed.contains("enforce_ajtai_commitment_linear_consistency")
            && terminal_committed.contains("committed_full_vector")
            && terminal_committed.contains("enforce_committed_superneo_image")
            && terminal_committed.contains("terminal_r2_witness_bit_bound")
            && terminal_committed.contains("terminal_r2_superneo_padding_zero")
            && terminal_committed.contains("UnusedPadding")
            && !terminal_committed.contains("return Self::U64;")
            && !terminal_committed.contains("append_f_slice_bits")
            && terminal_committed.contains("encode_vector_for_full_width")
            && terminal_committed.contains("commit_rv64im_main_recursion_construction2_packed_z")
            && terminal_committed.contains("Rv64imTerminalFPrimeR2Assignment")
            && terminal_committed.contains("superneo_pack_status"),
        "terminal F' committed-step compression must commit the low-norm SuperNeo R2 image and must not use the removed logical-image projection as authority"
    );
    assert!(
        compression.contains("committed_step_circuit")
            && compression.contains("setup_rv64im_terminal_f_prime_committed_step_relation")
            && compression.contains("prove_rv64im_terminal_f_prime_committed_step_relation")
            && !compression.contains("build_terminal_f_prime_published_r1cs_ccs_relation")
            && !compression.contains("public_first_committed_ccs"),
        "terminal F' compression must prove the SuperNeo committed-step circuit, not the legacy public-first extracted relation"
    );
    assert!(
        recursive_witness.contains("construction2_current_input_x_from_live_step")
            && !recursive_witness.contains("construction2_witness_digest")
            && !recursive_witness.contains("native_digest_values")
            && !recursive_witness.contains("push_state_in_claims")
            && !recursive_witness.contains("main_projection_digests"),
        "recursive-step current-input helper must not keep the removed witness-digest projection path"
    );
    assert!(
        nifs_stages.contains("me_input_accumulator_handle")
            && nifs_stages.contains("bind_me_inputs_accumulator_handle(")
            && nifs_stages.contains("bind_me_inputs_with_native_claims(")
            && !nifs_stages.contains("logical_me_input_digests")
            && !nifs_stages.contains("bind_me_input_digest_values")
            && !nifs_stages.contains("bind_me_inputs_with_projection_digests"),
        "recursive Pi_CCS production stages must bind the accumulator handle and expose no precomputed ME-digest path"
    );
    assert!(
        !pi_ccs.contains("bind_me_input_digest_values") && !pi_ccs.contains("bind_me_inputs_with_projection_digests"),
        "Pi_CCS must not keep digest-only transcript helpers; diagnostics must measure live CE-claim binding"
    );
    assert!(
        compression.contains("terminal_r2_ccs_rows")
            && compression.contains("terminal_r2_ccs_cols")
            && compression.contains("terminal_r2_ccs_nnz")
            && !compression.contains("terminal_committed_step_logical_shape")
            && !compression.contains("terminal_relation.logical_field_count()"),
        "IVC SNARK setup cache keys must use the terminal R2 CCS shape, not the removed logical witness projection shape"
    );
}

#[test]
fn rv64im_final_construction2_boundary_uses_post_terminal_committed_instance() {
    let native_ivc = fs::read_to_string(crate_path("src/rv64im/ivc.rs")).expect("read native IVC module");
    let f_prime = fs::read_to_string(crate_path("src/rv64im/f_prime.rs")).expect("read F' evaluator");
    let construction2 =
        fs::read_to_string(crate_path("src/rv64im/construction2.rs")).expect("read Construction-2 module");
    let recursive_step = fs::read_to_string(crate_path("src/rv64im/main_relation_spartan/recursive_step.rs"))
        .expect("read recursive-step circuit");
    let terminal_committed = fs::read_to_string(crate_path("src/rv64im/ivc_snark/terminal_f_prime_committed.rs"))
        .expect("read terminal F' committed-step owner");

    assert!(
        f_prime.contains("build_rv64im_main_recursion_construction2_fresh_instance_with_input_and_x_i")
            && f_prime.contains("construction2_u_next.x_i() != &x_out")
            && construction2.contains("rv64im_main_recursion_construction2_x_only_placeholder")
            && construction2.contains("current_input.commitment().commitment()"),
        "native F' must emit only the Construction-2 x_i image; terminal SuperNeo R2 owns the authoritative u_i.C"
    );
    assert!(
        native_ivc.contains("derive_rv64im_terminal_f_prime_committed_fresh_instance")
            && native_ivc.contains("construction2_u_i: committed_construction2_u_next")
            && native_ivc.contains("Rv64imMainRecursionConstruction2PublicBoundary::from_fresh_instance("),
        "the carried public image must publish the committed post-terminal Construction-2 instance"
    );
    assert!(
        recursive_step.contains("canonical_step_image.construction2_u_next().x_i()")
            && recursive_step.contains("backend_relation.construction2_u_next.x_i()"),
        "recursive-step public IO must bind the recomputed Construction-2 x_i while leaving u_i.C to terminal R2"
    );
    assert!(
        terminal_committed.contains("ensure_terminal_f_prime_output_uses_x_only_placeholder")
            && terminal_committed.contains("public_boundary_for_x_i")
            && terminal_committed.contains("require_superneo_assignment_commitment"),
        "the terminal R2 proof must derive and publish the final committed F' instance from the packed SuperNeo witness"
    );
    let setup_from_backend_shape = terminal_committed
        .split("pub(crate) fn from_backend_shape")
        .nth(1)
        .and_then(|tail| tail.split("pub(crate) fn step_cap").next())
        .expect("terminal F' setup from backend shape");
    assert!(
        setup_from_backend_shape.contains("ensure_terminal_f_prime_output_uses_x_only_placeholder"),
        "terminal F' setup/key derivation must reject stale native u_i.C cargo before deriving the committed-step shape"
    );
    assert!(
        setup_from_backend_shape.contains("commit_rv64im_main_recursion_construction2_packed_z")
            && setup_from_backend_shape.contains("Mat::zero(D, packed_cols, F::ZERO)")
            && !setup_from_backend_shape.contains("data: vec![F::ZERO"),
        "terminal F' setup/key derivation must use the same seeded Construction-2 Ajtai commitment context as proving"
    );
}

#[test]
fn rv64im_construction2_witness_image_excludes_public_step_label_cargo() {
    let construction2 = fs::read_to_string(crate_path("src/rv64im/construction2.rs")).expect("read Construction-2");
    let recursive_step = fs::read_to_string(crate_path("src/rv64im/main_relation_spartan/recursive_step.rs"))
        .expect("read recursive-step circuit");

    assert!(
        construction2.contains("does not admit public-step label cargo")
            && construction2.contains("if !step.label.is_empty()"),
        "RV64IM Construction-2 witness image must reject label cargo not covered by the canonical fixed width"
    );
    assert!(
        recursive_step.contains("label: String::new()"),
        "dummy recursive-step shape inputs must match the empty-label RV64IM Construction-2 boundary"
    );
}

#[test]
fn rv64im_recursive_step_chunk_digest_uses_authoritative_public_surfaces() {
    let recursive_step = fs::read_to_string(crate_path("src/rv64im/main_relation_spartan/recursive_step.rs"))
        .expect("read recursive-step circuit");
    let chunk_replay = fs::read_to_string(crate_path(
        "src/rv64im/main_relation_spartan/recursive_step/chunk_replay.rs",
    ))
    .expect("read recursive-step chunk replay");
    let nifs_stages =
        fs::read_to_string(crate_path("src/rv64im/main_relation_spartan/nifs_v_stages.rs")).expect("read NIFS stages");
    let public_chunk =
        fs::read_to_string(crate_path("src/rv64im/main_relation_circuit/public_chunk.rs")).expect("read public chunk");

    assert!(
        public_chunk.contains("pub fn rv64im_public_chunk_digest")
            && nifs_stages.contains("rv64im_public_chunk_digest(")
            && nifs_stages.contains("chunk_relation_digest_circuit_from_vars")
            && nifs_stages.contains("bridge_handoff_digest"),
        "chunk relation digest must hash in-circuit public chunk data and the verifier-bound bridge handoff digest"
    );
    assert!(
        recursive_step.contains("bridge_handoff_digest_input")
            && recursive_step.contains("expected_bridge_handoff_digest")
            && recursive_step.contains("bridge_handoff_digest_eq")
            && recursive_step.contains("terminal_verified_step_statement_digest_eq")
            && recursive_step.contains("chunk_replay.pi_ccs.public_chunk_digest")
            && recursive_step.contains("allocated_digest_field_values(&live_folded_accumulator_out_digest)")
            && !recursive_step.contains(
                "rv64im_recursive_accumulator_instance_digest_from_parts(\n            &payload.state_out_claims"
            ),
        "recursive-step public IO must bind terminal metadata to the live chunk replay surface"
    );
    assert!(
        chunk_replay.contains("replayed_next_claims.effective_claims()")
            && chunk_replay.contains("recursive_accumulator_instance_digest_circuit_from_claims")
            && !chunk_replay.contains("Rv64imChunkNextCarryMode::PreserveIncoming")
            && !chunk_replay.contains("recursive_accumulator_instance_digest_circuit_from_projection_digest_vars")
            && !chunk_replay.contains("state_out_expected_claims"),
        "recursive-step output accumulator digest must be recomputed from live NIFS.V output claims, not carried projection-digest cargo"
    );
}

#[test]
fn rv64im_recursive_state_identity_uses_phi_commitments_not_full_ce_projection_hashes() {
    let recursive_cover = fs::read_to_string(crate_path("src/rv64im/main_relation_spartan/recursive_cover.rs"))
        .expect("read recursive-cover circuit");
    let chunk_step_recursive =
        fs::read_to_string(crate_path("src/rv64im/main_relation_spartan/chunk_step_recursive.rs"))
            .expect("read recursive-step payload builder");
    let final_relation = fs::read_to_string(crate_path("src/rv64im/final_relation.rs")).expect("read final relation");

    let carry_digest = chunk_step_recursive
        .split("pub(crate) fn rv64im_chunk_step_recursive_carry_state_digest")
        .nth(1)
        .and_then(|tail| {
            tail.split("pub(crate) fn build_rv64im_main_recursion_step_spartan_statement")
                .next()
        })
        .expect("carry-state digest function");
    assert!(
        recursive_cover.contains("recursive_accumulator_instance_digest_circuit_from_phi_dec_parent_vars")
            && recursive_cover.contains("main_recursion_recursive_accumulator_phi_dec_parent")
            && recursive_cover.contains("claim.c_data")
            && recursive_cover.contains("params.b")
            && !recursive_cover.contains("me_input_projection_digest_poseidon"),
        "recursive accumulator circuit identity must hash the SuperNeo Π_DEC parent commitment handle, not full CE projection surfaces"
    );
    assert!(
        carry_digest.contains("main_recursion_fixed_step_accumulator_phi_commitments")
            && carry_digest.contains("claim.c.data")
            && carry_digest.contains("rv64im_chunk_fold_transcript_snapshot_digest")
            && !carry_digest.contains("me_input_projection_digest_poseidon")
            && !carry_digest.contains("claim.X")
            && !carry_digest.contains("claim.r")
            && !carry_digest.contains("claim.y_ring"),
        "recursive carry-state digest must use φ(commitments), transcript state, and terminal handle only"
    );
    assert!(
        final_relation.contains("rv64im_recursive_accumulator_instance_digest_from_phi_dec_parent")
            && final_relation.contains("rv64im_recursive_accumulator_phi_dec_parent_commitment")
            && final_relation.contains("scale_commitment_add_inplace")
            && final_relation.contains("rv64im_chunk_fold_carry_recursive_accumulator_digest"),
        "native folded-accumulator identity must share the same SuperNeo Π_DEC parent commitment handle as the circuit"
    );
}

#[test]
fn rv64im_pi_ccs_fs_binding_documents_accumulator_handle_boundary() {
    let pi_ccs =
        fs::read_to_string(crate_path("src/rv64im/main_relation_circuit/pi_ccs.rs")).expect("read Pi_CCS circuit");
    let chunk_replay = fs::read_to_string(crate_path(
        "src/rv64im/main_relation_spartan/recursive_step/chunk_replay.rs",
    ))
    .expect("read recursive chunk replay");
    let recursive_step = fs::read_to_string(crate_path("src/rv64im/main_relation_spartan/recursive_step.rs"))
        .expect("read recursive step");
    assert!(
        pi_ccs.contains("FS binding invariant")
            && pi_ccs.contains("bind_me_inputs_accumulator_handle")
            && pi_ccs.contains("PI_CCS_ME_ACCUMULATOR_HANDLE_RAW_TAG")
            && pi_ccs.contains("final R1 CE proves the omitted fields")
            && pi_ccs.contains("cannot be chosen after challenges")
            && chunk_replay.contains("state_in_var.folded_accumulator_digest")
            && recursive_step.contains("state_in_folded_accumulator_digest_eq_live"),
        "Pi_CCS challenge sampling must use the carried accumulator handle and later constrain it to live φ(commitments)"
    );
}

#[test]
fn rv64im_verified_step_statement_digest_has_one_native_public_formula() {
    let construction2 = fs::read_to_string(crate_path("src/rv64im/construction2.rs")).expect("read Construction-2");
    let terminal_statement = fs::read_to_string(crate_path(
        "src/rv64im/main_relation_spartan/recursive_step/terminal_statement.rs",
    ))
    .expect("read terminal F' statement");
    let expected_digest = construction2
        .split("pub(crate) fn expected_digest(&self) -> [u8; 32]")
        .nth(1)
        .and_then(|tail| {
            tail.split("fn validate_rv64im_main_recursion_construction2_ce_claim_surface")
                .next()
        })
        .expect("verified-step expected digest");

    assert!(
        expected_digest.contains("b\"v2\"")
            && expected_digest.contains("u64::from(self.halted_out)")
            && expected_digest.contains("digest32_as_fields(self.state_in)")
            && expected_digest.contains("digest32_as_fields(self.state_out)")
            && expected_digest.contains("digest32_as_fields(self.public_chunk_digest)")
            && expected_digest.contains("digest32_as_fields(self.chunk_relation_digest)")
            && !expected_digest.contains("append_message(\n            b\"neo.fold.next/rv64im/main_recursion_construction2_verified_step_statement/state_in\""),
        "verified-step digest must use canonical field-limb transcript inputs, matching the terminal F' circuit"
    );
    assert!(
        terminal_statement
            .contains("Ok(build_terminal_f_prime_verified_step_statement(backend_relation)?.expected_digest())",),
        "terminal F' public target must delegate to the same native verified-step digest formula"
    );
}

#[test]
fn rv64im_final_ce_bundle_uses_superneo_ce_projection_digest() {
    let source =
        fs::read_to_string(crate_path("src/rv64im/main_relation_circuit/ce_spartan.rs")).expect("read CE Spartan");
    let compression = fs::read_to_string(crate_path("src/rv64im/ivc_snark.rs")).expect("read IVC compression module");
    assert!(
        source.contains("struct Rv64imCeBundleCircuit")
            && source.contains("me_input_projection_digest_poseidon(")
            && source.contains("me_input_projection_digest_poseidon_values_from_native_claim")
            && source.contains("enforce_paper_ce_claim_consistency")
            && !source.contains("enforce_paper_dec_child_claim_consistency")
            && !source.contains("me_digest_poseidon_values_from_native_claim")
            && !source.contains("native_claim_digest_fields"),
        "final CE bundle proof must expose the SuperNeo CE projection digest and prove the paper CE relation, including X = L_in(Z)"
    );
    assert!(
        compression.contains("canonical_final_ce_claim")
            && compression.contains("ensure_final_ce_claims_are_canonical")
            && compression.contains("non-authoritative transport fields"),
        "the theorem-facing IVC SNARK must strip and reject non-paper CE transport fields instead of treating them as proof authority"
    );
    assert!(
        compression.contains("claim.c.d != neo_math::D")
            && compression.contains("claim.c.kappa == 0")
            && compression.contains("claim.c.data.len() != expected_commitment_words")
            && compression.contains("row.len() != neo_math::D"),
        "the theorem-facing IVC SNARK must reject malformed final CE commitment and y_ring shapes before CE proof verification"
    );
}

#[test]
fn rv64im_final_ce_tamper_coverage_targets_full_superneo_claim_fields() {
    let ce_spartan =
        fs::read_to_string(crate_path("src/rv64im/main_relation_circuit/ce_spartan.rs")).expect("read CE Spartan");
    let compressed_verify = fs::read_to_string(crate_path("src/rv64im/ivc_snark.rs")).expect("read IVC SNARK verifier");
    let decider_test =
        fs::read_to_string(crate_path("tests/rv64im_spartan2_decider.rs")).expect("read Spartan decider tests");

    assert!(
        ce_spartan.contains("verify_rv64im_ce_bundle_relation")
            && ce_spartan.contains("me_input_projection_digest_poseidon_values_from_native_claim")
            && ce_spartan.contains("enforce_paper_ce_claim_consistency")
            && compressed_verify.contains("verify_rv64im_final_ce_bundle")
            && compressed_verify.contains("verify_rv64im_ce_bundle_relation"),
        "final CE fields must be checked by the final R1 CE bundle proof, not by recursive digest authority"
    );
    assert!(
        decider_test.contains("tampered final CE bundle proof bytes")
            && decider_test.contains("tampered final carried CE commitment")
            && decider_test.contains("tampered final carried CE X field")
            && decider_test.contains("tampered final carried CE evaluation point")
            && decider_test.contains("tampered final carried CE y_ring field"),
        "the theorem-facing compressed verifier test must keep final CE proof, c, X, r, and y_ring tamper coverage"
    );
}

#[test]
fn rv64im_terminal_f_prime_exports_sparse_superneo_ccs_shape() {
    let terminal_owner = fs::read_to_string(crate_path("src/rv64im/ivc_snark/terminal_f_prime_committed.rs"))
        .expect("read terminal F' committed relation owner");
    let terminal_circuit = fs::read_to_string(crate_path(
        "src/rv64im/ivc_snark/terminal_f_prime_committed/proof_circuit.rs",
    ))
    .expect("read terminal F' committed relation circuit");
    let terminal = format!("{terminal_owner}\n{terminal_circuit}");
    let r1cs = fs::read_to_string(workspace_path("crates/neo-ccs/src/r1cs.rs")).expect("read neo-ccs R1CS bridge");

    assert!(
        r1cs.contains("pub fn sparse_r1cs_to_ccs")
            && r1cs.contains("CcsStructure::new_sparse")
            && r1cs.contains("f(X0,X1,X2) = X0 * X1 - X2"),
        "R1CS->CCS conversion used by terminal F' must preserve sparse matrices and the row-wise R1CS polynomial"
    );
    assert!(
        terminal.contains("Rv64imTerminalFPrimeR1csCcsRelation")
            && terminal.contains("spartan_sparse_to_superneo_ccs_matrix")
            && terminal.contains("W||1||X layout")
            && terminal.contains("split_terminal_r2_public_values")
            && terminal.contains("terminal_r2_public_value_range")
            && terminal.contains("RV64IM_ENC_INST_BITS")
            && terminal.contains("field_image()")
            && terminal.contains("low_norm_bit_values")
            && terminal.contains("UnusedPadding")
            && terminal.contains("padded_private_witness_labels")
            && terminal.contains("num_shared_unpadded")
            && terminal.contains("num_precommitted_unpadded")
            && terminal.contains("num_rest_unpadded")
            && terminal.contains("sparse R1CS matrix references an unused padded witness column")
            && !terminal.contains("enforce_u32_allocated")
            && !terminal.contains("true_r2_image")
            && !terminal.contains("legacy_binary_full_width")
            && terminal.contains("num_spartan_public")
            && !terminal.contains("num_io + col")
            && terminal.contains("check_ccs_rowwise_zero"),
        "terminal F' export must map the 256-bit Construction-2 x_i image into the SuperNeo R2 public instance without retaining the legacy logical/binary witness image"
    );
}

#[test]
fn rv64im_default_u_perp_does_not_reintroduce_logical_image_width() {
    let default_pair = fs::read_to_string(crate_path("src/rv64im/construction2_default.rs"))
        .expect("read Construction-2 default-pair owner");
    assert!(
        default_pair.contains("Ok(RV64IM_ENC_INST_BITS)")
            && !default_pair.contains("FPrimeLowNormWitnessImage")
            && !default_pair.contains("binary_values")
            && !default_pair.contains("logical_field")
            && !default_pair.contains("* 64"),
        "canonical u_perp must not derive a binary logical-image witness width; terminal SuperNeo R2 owns the committed F' witness"
    );
}

#[test]
fn rv64im_recursive_shape_only_elides_common_zero_rlc_suffix() {
    let source = fs::read_to_string(crate_path("src/rv64im/main_relation_spartan/chunk_step_recursive.rs"))
        .expect("read recursive-step payload builder");
    let nifs =
        fs::read_to_string(crate_path("src/rv64im/main_relation_spartan/nifs_v_stages.rs")).expect("read NIFS stages");
    let pi_rlc =
        fs::read_to_string(crate_path("src/rv64im/main_relation_circuit/pi_rlc.rs")).expect("read Pi_RLC circuit");
    assert!(
        source.contains("common_rlc_zero_commit_suffix_len")
            && source.contains("trailing_zero_rlc_commitment_surface_len")
            && source.contains("claim_has_zero_rlc_commitment_surface")
            && !source.contains("advices.len() == 1\n        && advices[0]"),
        "RLC zero-suffix elision must use the common trailing zero CE c/y_ring surface across the shape, not a one-advice all-zero special case"
    );
    assert!(
        nifs.contains("zero_output_suffix_start")
            && nifs.contains("alloc_ce_claim_x_surface_with_shared_point")
            && nifs.contains("ctx.rlc_zero_commit_suffix_len")
            && pi_rlc.contains("let active_children = &children[..active_children_len]")
            && pi_rlc.contains("ensure_zero_commit_suffix(children, zero_commit_suffix_len)")
            && pi_rlc.contains("if child_idx >= zero_commit_suffix_start")
            && pi_rlc.contains("continue;"),
        "RLC zero-suffix children must allocate only X, prove native zero c/y_ring shape, and stay out of dense c/y_ring folding"
    );
}

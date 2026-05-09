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

fn read_construction2_terminal_helpers() -> String {
    [
        "src/core/construction2/terminal/mod.rs",
        "src/core/construction2/terminal/boundary.rs",
        "src/core/construction2/terminal/commitment.rs",
        "src/core/construction2/terminal/constraints.rs",
        "src/core/construction2/terminal/labels.rs",
        "src/core/construction2/terminal/low_norm.rs",
        "src/core/construction2/terminal/types.rs",
    ]
    .into_iter()
    .map(|path| fs::read_to_string(crate_path(path)).expect("read shared terminal helper"))
    .collect::<Vec<_>>()
    .join("\n")
}

#[test]
fn rv32im_ivc_native_module_does_not_reference_spartan2() {
    let source = fs::read_to_string(crate_path("src/frontends/rv32im/ivc.rs")).expect("read native IVC module");
    assert!(
        !source.contains("spartan2"),
        "native RV32IM IVC ownership must stay Spartan-free"
    );
}

#[test]
fn rv32im_compressed_main_proof_is_not_a_verifier() {
    let main_proof =
        fs::read_to_string(crate_path("src/frontends/rv32im/main_proof.rs")).expect("read compact main proof module");
    let forbidden_wrapper_verify = ["pub fn verify", "(&self, ivc_recursion"].concat();
    assert!(
        main_proof.contains("pub fn expected_ivc_public_image") && !main_proof.contains(&forbidden_wrapper_verify),
        "Rv32imCompressedMainProof may derive the public image, but proof acceptance must call Rv32imIvcSnark::verify"
    );

    let nightstream_verify = fs::read_to_string(crate_path("src/public_proof/rv32im/flow/verify.rs"))
        .expect("read Nightstream verify timing module");
    assert!(
        nightstream_verify.contains(".ivc_snark()") && nightstream_verify.contains(".verify(ivc_recursion_snark_vk"),
        "Nightstream main-proof verification must route through Rv32imIvcSnark::verify"
    );
}

#[test]
fn rv32im_nightstream_verifier_context_binds_actual_ivc_snark_vk() {
    let nightstream =
        fs::read_to_string(crate_path("src/public_proof/rv32im/mod.rs")).expect("read Nightstream RV32IM");
    let statement =
        fs::read_to_string(crate_path("src/public_proof/rv32im/statement.rs")).expect("read Nightstream statement");
    let verify_perf =
        fs::read_to_string(crate_path("src/public_proof/rv32im/flow/verify.rs")).expect("read Nightstream verifier");
    let source = format!("{nightstream}\n{statement}");
    assert!(
        source.contains("ivc_recursion_snark_vk: &Rv32imIvcSnarkVerifierKey")
            && source.contains("ivc_recursion_snark_vk.expected_digest()?")
            && source.contains("fn rv32im_verifier_context_digest_from_key_digest")
            && !source.contains("pub fn rv32im_verifier_context_digest_from_key_digest")
            && source.contains("neo.fold.next/nightstream/rv32im/verifier_context/version\", b\"v3\"")
            && source.contains("neo.fold.next/nightstream/rv32im/verifier_context/ivc_recursion_snark_vk"),
        "Nightstream verifier context must bind the actual RV32IM IVC SNARK verifier key, not a caller-supplied digest"
    );
    assert!(
        verify_perf.contains("rv32im_verifier_context_digest(")
            && verify_perf.contains("ivc_recursion_snark_vk")
            && verify_perf.contains("trusted root parameters and IVC verifier key"),
        "Nightstream verification errors and checks must make the IVC verifier-key binding explicit"
    );
}

#[test]
fn rv32im_ivc_compression_module_owns_explicit_compress_boundary() {
    let source =
        fs::read_to_string(crate_path("src/frontends/rv32im/ivc_snark/mod.rs")).expect("read IVC compression module");
    assert!(
        source.contains("impl Rv32imIvcState")
            && source.contains("pub fn compress(&self)")
            && source.contains("verify_rv32im_final_construction2_boundary")
            && source.contains("verify_rv32im_final_ce_bundle")
            && source.contains("setup_rv32im_terminal_f_prime_committed_step_relation")
            && source.contains("prove_rv32im_terminal_f_prime_committed_step_relation")
            && source.contains("verify_rv32im_terminal_f_prime_committed_step_relation")
            && source.contains("final_ce_shape")
            && !source.contains("prove_rv32im_chunk_step_ivc_spartan"),
        "RV32IM IVC compression must stay on the final Construction-2 boundary and cache all sub-proof shapes, not terminal chunk-step Spartan"
    );
}

#[test]
fn rv32im_ivc_compression_has_no_replay_or_terminal_step_acceptance_fallback() {
    for path in [
        "src/frontends/rv32im/ivc.rs",
        "src/frontends/rv32im/ivc_snark/mod.rs",
        "src/frontends/rv32im/main_proof.rs",
        "src/public_proof/rv32im/flow/verify.rs",
    ] {
        let source = fs::read_to_string(crate_path(path)).expect("read RV32IM verifier source");
        assert!(
            !source.contains("verify_against_final")
                && !source.contains("validate_standalone_rv32im_ivc_snark_scope")
                && !source.contains("build_terminal_relation")
                && !source.contains("Rv32imIvcState::verify")
                && !source.contains("validate_replayed_chain_matches"),
            "{path} must not expose full-chain replay or terminal chunk-step proof acceptance"
        );
    }
}

#[test]
fn rv32im_verifiers_do_not_accept_precomputed_public_digests() {
    for path in [
        "src/core/verifier.rs",
        "src/core/session/mod.rs",
        "src/core/finalize/package.rs",
        "src/core/finalize/verify.rs",
        "src/frontends/rv32im/kernel/simple/mod.rs",
        "src/frontends/rv32im/kernel/proof/staged_verify.rs",
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
fn rv32im_chunk_step_ivc_uses_named_authoritative_ce_projection() {
    let source =
        fs::read_to_string(crate_path("src/frontends/rv32im/chunk/step_ivc.rs")).expect("read chunk-step IVC module");
    assert!(
        source.contains("struct Rv32imCarriedCeProjection")
            && source.contains("commitment_data: Vec<F>")
            && source.contains("compact_x: Vec<F>")
            && source.contains("Rv32imCarriedCeProjection::from_claim")
            && source.contains("claim.X.rows() != neo_math::D || claim.X.cols() != claim.m_in")
            && source.contains("rv32im_ce_claims_match_projection")
            && source.contains("(Some(lhs), Some(rhs)) => lhs == rhs")
            && !source.contains("(0..lhs.m_in).all(|col|"),
        "chunk-step IVC carry comparison must use a named authoritative CE projection, not an inline partial equality"
    );
}

#[test]
fn rv32im_recursive_step_binds_current_construction2_input_hash_image() {
    let source = fs::read_to_string(crate_path(
        "src/frontends/rv32im/main_relation_spartan/recursive_step/mod.rs",
    ))
    .expect("read recursive-step circuit");
    let public_boundary = fs::read_to_string(crate_path(
        "src/frontends/rv32im/main_relation_spartan/recursive_step/construction2_public.rs",
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
fn rv32im_recursive_step_enforces_canonical_base_u_perp() {
    let source = fs::read_to_string(crate_path(
        "src/frontends/rv32im/main_relation_spartan/recursive_step/construction2_witness.rs",
    ))
    .expect("read recursive-step Construction-2 witness circuit");
    assert!(
        source.contains("enforce_current_input_u_perp_when_base")
            && source.contains("base_current_input_u_perp")
            && source.contains("build_rv32im_main_recursion_construction2_default_fresh_instance")
            && source.contains("chunk_count_is_zero")
            && source.contains("base_current_input_commitment_")
            && source.contains("base_current_input_x_i_bit_"),
        "recursive-step circuit must enforce canonical u_perp for the private current Construction-2 input at i=0"
    );
}

#[test]
fn rv32im_terminal_f_prime_committed_step_uses_superneo_committed_ccs_authority() {
    let recursive_step = fs::read_to_string(crate_path(
        "src/frontends/rv32im/main_relation_spartan/recursive_step/mod.rs",
    ))
    .expect("read recursive-step circuit");
    let public_target = fs::read_to_string(crate_path(
        "src/frontends/rv32im/main_relation_spartan/recursive_step/public_target.rs",
    ))
    .expect("read recursive-step public target");
    let recursive_witness = fs::read_to_string(crate_path(
        "src/frontends/rv32im/main_relation_spartan/recursive_step/construction2_witness.rs",
    ))
    .expect("read recursive-step Construction-2 witness circuit");
    let compression =
        fs::read_to_string(crate_path("src/frontends/rv32im/ivc_snark/mod.rs")).expect("read IVC compression module");
    let terminal_committed_owner = fs::read_to_string(crate_path(
        "src/frontends/rv32im/ivc_snark/terminal_f_prime_committed/mod.rs",
    ))
    .expect("read terminal F' committed-step owner");
    let terminal_committed_circuit = fs::read_to_string(crate_path(
        "src/frontends/rv32im/ivc_snark/terminal_f_prime_committed/proof_circuit.rs",
    ))
    .expect("read terminal F' committed-step circuit");
    let construction2_terminal = read_construction2_terminal_helpers();
    let terminal_committed =
        format!("{terminal_committed_owner}\n{terminal_committed_circuit}\n{construction2_terminal}");

    assert!(
        !crate_path("src/frontends/rv32im/ivc_snark/construction2_opening.rs").exists()
            && !crate_path("src/frontends/rv32im/ivc_snark/terminal_committed_step.rs").exists()
            && !crate_path("src/frontends/rv32im/ivc_snark/terminal_f_prime_relation.rs").exists()
            && !compression.contains("mod construction2_opening")
            && !compression.contains("mod terminal_committed_step")
            && !compression.contains("mod terminal_f_prime_relation")
            && !compression.contains("enforce_seeded_ajtai_binary_opening"),
        "legacy bit-opening and extracted public-first terminal relation modules must not be part of RV32IM compressed-proof authority"
    );
    assert!(
        !recursive_step.contains("construction2_witness_digest")
            && !recursive_step.contains("Rv32imMainRecursionStepPublicIoMode")
            && !recursive_witness.contains("construction2_witness_digest"),
        "recursive-step terminal F' proof must not expose the removed witness-digest public IO mode"
    );
    assert!(
        compression.contains("committed_step_circuit")
            && compression.contains("terminal_f_prime_committed_step_proof")
            && compression.contains("verify_rv32im_terminal_f_prime_committed_step_relation")
            && compression.contains("SuperNeo low-norm bound")
            && compression.contains("rv32im_terminal_f_prime_r2_public_values_from_public_image")
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
        terminal_committed.contains("Rv32imTerminalFPrimeR1csCcsRelation")
            && terminal_committed.contains("terminal_source_witness_debug_structure")
            && terminal_committed.contains("build_rv32im_terminal_f_prime_r2_circuit")
            && terminal_committed.contains("enforce_ajtai_commitment_linear_consistency")
            && terminal_committed.contains("synthesize_terminal_f_prime_with_committed_sources")
            && terminal_committed.contains("committed_full_vector")
            && terminal_committed.contains("enforce_committed_superneo_image")
            && terminal_committed.contains("terminal_r2_public_bit_bound")
            && terminal_committed.contains("terminal_r2_witness_bit_bound")
            && terminal_committed.contains("source_lc")
            && terminal_committed.contains("low_norm_bit_values")
            && terminal_committed.contains("UnusedPadding")
            && terminal_committed.contains("padded_private_witness_labels")
            && terminal_committed.contains("num_shared_unpadded")
            && terminal_committed.contains("num_precommitted_unpadded")
            && terminal_committed.contains("num_rest_unpadded")
            && terminal_committed.contains("source low-norm values may be committed")
            && !terminal_committed.contains("return Self::U64;")
            && !terminal_committed.contains("LOW_NORM_LIMB")
            && !terminal_committed.contains("enforce_u32_allocated")
            && terminal_committed.contains("terminal_r2_superneo_padding_zero")
            && terminal_committed.contains("terminal_r2_public_value_range_static")
            && !terminal_committed.contains("build_rv32im_main_recursion_construction2_committed_f_prime_full_vector")
            && terminal_committed.contains("encode_vector_for_full_width")
            && terminal_committed.contains("commit_rv32im_main_recursion_construction2_packed_z")
            && terminal_committed.contains("require_superneo_assignment_commitment")
            && terminal_committed.contains("commitment_matches_public_boundary")
            && terminal_committed.contains("ensure_terminal_f_prime_output_uses_x_only_placeholder")
            && terminal_committed.contains("commitment.kappa != 1")
            && terminal_committed.contains("commitment.data.len() != D")
            && terminal_committed.contains("terminal_f_prime_committed_step_boundary_public_values")
            && terminal_committed.contains("construction2_public_boundary_digest_circuit")
            && terminal_committed.contains("terminal_boundary_x_i_eq")
            && terminal_committed.contains("commitment_kappa_matches_data_len")
            && terminal_committed.contains("boundary.commitment_data.len() % D != 0")
            && !terminal_committed.contains("terminal_boundary_commitment_kappa_matches_packed_z_cols")
            && !terminal_committed.contains("FPrimeWitnessImage")
            && !terminal_committed.contains("legacy_binary_full_width"),
        "terminal F' committed-step ownership must commit a low-norm SuperNeo image, prove its Construction-2 public boundary, and use it to recompose the terminal R2 assignment"
    );
    let terminal_r2_public_values = public_target
        .split("pub(crate) fn terminal_f_prime_r2_public_values_from_parts")
        .nth(1)
        .and_then(|tail| {
            tail.split("pub(crate) fn terminal_f_prime_r2_public_values_from_public_image")
                .next()
        })
        .expect("terminal F' R2 public value method");
    assert!(
        terminal_r2_public_values.contains("field_image()")
            && terminal_r2_public_values.contains("folded_accumulator_out_digest")
            && terminal_r2_public_values.contains("bridge_handoff_digest")
            && !terminal_r2_public_values.contains("construction2_u_i"),
        "terminal F' R2 public IO must exclude the output u_i.C boundary so the committed Z is not self-referential"
    );
}

#[test]
fn direct_ccs_terminal_compressor_uses_folded_ivc_carrier() {
    let source_state =
        fs::read_to_string(crate_path("src/frontends/direct_ccs/state/mod.rs")).expect("read direct CCS state module");
    let source_append = fs::read_to_string(crate_path("src/frontends/direct_ccs/state/append.rs"))
        .expect("read direct CCS append module");
    let source_compress = fs::read_to_string(crate_path("src/frontends/direct_ccs/state/compress.rs"))
        .expect("read direct CCS compression module");
    let source_summary = fs::read_to_string(crate_path("src/frontends/direct_ccs/state/summary.rs"))
        .expect("read direct CCS summary module");
    let source_types =
        fs::read_to_string(crate_path("src/frontends/direct_ccs/state/types.rs")).expect("read direct CCS state types");
    let source_terminal_circuit = fs::read_to_string(crate_path("src/frontends/direct_ccs/terminal/circuit/mod.rs"))
        .expect("read direct CCS terminal F' circuit module");
    let source_prove = fs::read_to_string(crate_path("src/frontends/direct_ccs/terminal/prove.rs"))
        .expect("read direct CCS prove module");
    let source_verify = fs::read_to_string(crate_path("src/frontends/direct_ccs/terminal/verify.rs"))
        .expect("read direct CCS verify module");
    let source_public_image = [
        "src/frontends/direct_ccs/public_image/mod.rs",
        "src/frontends/direct_ccs/public_image/digest.rs",
    ]
    .into_iter()
    .map(|path| fs::read_to_string(crate_path(path)).expect("read direct CCS public-image module"))
    .collect::<Vec<_>>()
    .join("\n");
    let source_accumulator = fs::read_to_string(crate_path("src/frontends/direct_ccs/terminal/gadgets/accumulator.rs"))
        .expect("read direct CCS accumulator digest module");
    let source = format!(
        "{source_state}\n{source_types}\n{source_append}\n{source_compress}\n{source_summary}\n{source_terminal_circuit}\n{source_prove}\n{source_verify}\n{source_public_image}\n{source_accumulator}"
    );
    let direct_mod =
        fs::read_to_string(crate_path("src/frontends/direct_ccs/mod.rs")).expect("read direct CCS owner module");
    let terminal_committed_owner = fs::read_to_string(crate_path("src/frontends/direct_ccs/terminal/committed/mod.rs"))
        .expect("read direct CCS terminal committed owner module");
    let terminal_committed_circuit =
        fs::read_to_string(crate_path("src/frontends/direct_ccs/terminal/committed/circuit.rs"))
            .expect("read direct CCS terminal committed circuit module");
    let terminal_committed_types =
        fs::read_to_string(crate_path("src/frontends/direct_ccs/terminal/committed/types.rs"))
            .expect("read direct CCS terminal committed types module");
    let terminal_committed_source_linking = fs::read_to_string(crate_path(
        "src/frontends/direct_ccs/terminal/committed/source_linking.rs",
    ))
    .expect("read direct CCS terminal committed source-linking module");
    let construction2_terminal = read_construction2_terminal_helpers();
    let terminal_committed = format!(
        "{terminal_committed_owner}\n{terminal_committed_circuit}\n{terminal_committed_types}\n{terminal_committed_source_linking}\n{construction2_terminal}"
    );
    let fibonacci_probe =
        fs::read_to_string(crate_path("src/bin/fibonacci_superneo_probe.rs")).expect("read Fibonacci probe");
    let sha_probe = fs::read_to_string(crate_path("src/bin/sha256_superneo_probe.rs")).expect("read SHA probe");
    let fibonacci_probe_support = fs::read_to_string(crate_path("src/bin/support/fibonacci_superneo_probe_support.rs"))
        .expect("read Fibonacci probe support");
    let sha_probe_support = fs::read_to_string(crate_path("src/bin/support/sha256_superneo_probe_support.rs"))
        .expect("read SHA probe support");

    assert!(
        source.contains("pub struct DirectCcsIvcState")
            && direct_mod.contains("generic non-VM direct CCS/R1CS IVC compression path")
            && source.contains("pub struct DirectCcsProgram")
            && source.contains("pub struct DirectCcsStep")
            && source.contains("pub fn append_relation<L, MR, MB>")
            && source.contains("pub fn append_step<L, MR, MB>")
            && source.contains("pub fn append_all<L, MR, MB>")
            && source.contains("pub fn latest_relation_and_advice")
            && source.contains("pub fn compress_with_trace")
            && source.contains("ensure_terminal_compression_is_proof_complete")
            && source.contains("chunks: vec![last.surface.clone()]")
            && source.contains("initial_claims: last.relation.state_in.carry.claims.clone()")
            && source.contains("initial_transcript: Some(last.relation.state_in.transcript.clone())")
            && source.contains("DirectCcsTerminalFPrimeCircuit")
            && source.contains("Construction2EncodedPublicInput")
            && source.contains("Construction2PublicBoundary")
            && source.contains("direct_accumulator_digest_from_claims")
            && source.contains("direct_state_x_out")
            && source.contains("direct_terminal_accumulator_in_digest_private")
            && source.contains("proof.construction2_u_i.x_i != expected_x_i")
            && source.contains("terminal_circuit.terminal_public_values()")
            && source.contains("enforce_direct_terminal_final_ce_consistency")
            && !source.contains("prove_rv32im_ce_bundle_relation")
            && !source.contains("verify_rv32im_ce_bundle_relation"),
        "direct CCS compression should append verified relations into a carrier, synthesize only the terminal F' step, bind state_in/state_out through folded Construction-2 images, and check final CE privately without a projection digest"
    );
    assert!(
        source.contains("DirectCcsTerminalCommittedRelation::from_terminal_circuit")
            && source.contains("setup_direct_ccs_terminal_committed_relation")
            && source.contains("verify_direct_ccs_terminal_committed_relation")
            && terminal_committed.contains("pub(crate) struct DirectCcsTerminalCommittedRelation")
            && terminal_committed.contains("Construction2PublicBoundary")
            && terminal_committed.contains("enforce_terminal_commitment")
            && terminal_committed.contains("synthesize_body_with_public_inputs")
            && terminal_committed.contains("direct_terminal_r2_source_link")
            && terminal_committed.contains("construction2_x_bit_range")
            && terminal_committed.contains("enforce_terminal_boundary_digests"),
        "direct CCS terminal compression must prove the Construction-2 committed u_i=(C,x_i) boundary, not only a bare latest-step SNARK"
    );
    assert!(
        fibonacci_probe.contains("DirectCcsProgram::new")
            && sha_probe.contains("DirectCcsProgram::new")
            && fibonacci_probe_support.contains("append_relation")
            && sha_probe_support.contains("append_relation")
            && fibonacci_probe.contains("generic direct CCS/R1CS path; no RV32IM VM is used")
            && sha_probe.contains("generic direct CCS/R1CS path; no RV32IM VM is used")
            && fibonacci_probe_support.contains("verifier-shaped F' body")
            && sha_probe_support.contains("verifier-shaped F' body")
            && fibonacci_probe_support.contains("terminal_final_ce=0")
            && sha_probe_support.contains("terminal_final_ce=0")
            && fibonacci_probe_support.contains("multi-step recursive compression is refused")
            && sha_probe_support.contains("multi-step recursive compression is refused")
            && !fibonacci_probe.contains("prove_direct_ccs_f_prime_snark_with_perf")
            && !sha_probe.contains("prove_direct_ccs_f_prime_snark_with_perf")
            && !fibonacci_probe_support.contains("prior chunks are linked through the Construction-2 x_i image")
            && !sha_probe_support.contains("prior chunks are linked through the Construction-2 x_i image"),
        "direct CCS diagnostics must expose the verifier-shaped F' body and refuse multi-step proof output until low-norm enc(F') carries NIFS.V authority"
    );
}

#[test]
fn rv32im_ce_circuit_accepts_only_superneo_packed_witness_shape() {
    let witness =
        fs::read_to_string(crate_path("src/circuit/superneo/witness.rs")).expect("read circuit witness module");
    let ce_consistency = fs::read_to_string(crate_path("src/circuit/superneo/ce_consistency/mod.rs"))
        .expect("read CE consistency module");

    assert!(
        witness.contains("self.cols != expected_m.div_ceil(D)")
            && !witness.contains("self.cols == expected_m")
            && !ce_consistency.contains("return Ok(expected_m)"),
        "RV32IM CE circuit witnesses must reject dense/unpacked compatibility layout and accept only SuperNeo D x ceil(m/D) packing"
    );
}

#[test]
fn rv32im_terminal_f_prime_relation_extraction_legacy_is_removed() {
    let compression =
        fs::read_to_string(crate_path("src/frontends/rv32im/ivc_snark/mod.rs")).expect("read IVC compression module");
    assert!(
        !crate_path("src/frontends/rv32im/ivc_snark/terminal_f_prime_relation.rs").exists()
            && !compression.contains("public_first_committed_ccs")
            && !compression.contains("Rv32imTerminalFPrimePublishedCcs")
            && !compression.contains("build_terminal_f_prime_published_r1cs_ccs_relation"),
        "RV32IM compression must not keep the public-first extracted terminal relation compatibility path"
    );
}

#[test]
fn rv32im_construction2_current_input_helper_does_not_keep_witness_digest_cargo() {
    let construction2 =
        fs::read_to_string(crate_path("src/frontends/rv32im/construction2/mod.rs")).expect("read Construction-2");
    let recursive_witness = fs::read_to_string(crate_path(
        "src/frontends/rv32im/main_relation_spartan/recursive_step/construction2_witness.rs",
    ))
    .expect("read recursive-step Construction-2 witness digest");
    let nifs_stages = fs::read_to_string(crate_path(
        "src/frontends/rv32im/main_relation_spartan/nifs_v_stages.rs",
    ))
    .expect("read NIFS stages");
    let pi_ccs = fs::read_to_string(crate_path("src/circuit/superneo/pi_ccs.rs")).expect("read SuperNeo Pi_CCS");
    let compression =
        fs::read_to_string(crate_path("src/frontends/rv32im/ivc_snark/mod.rs")).expect("read IVC compression module");
    let terminal_committed_owner = fs::read_to_string(crate_path(
        "src/frontends/rv32im/ivc_snark/terminal_f_prime_committed/mod.rs",
    ))
    .expect("read terminal F' committed-step owner");
    let terminal_committed_circuit = fs::read_to_string(crate_path(
        "src/frontends/rv32im/ivc_snark/terminal_f_prime_committed/proof_circuit.rs",
    ))
    .expect("read terminal F' committed-step circuit");
    let construction2_terminal = read_construction2_terminal_helpers();
    let terminal_committed =
        format!("{terminal_committed_owner}\n{terminal_committed_circuit}\n{construction2_terminal}");

    assert!(
        !construction2.contains("append_state_in_claim_fields")
            && !construction2.contains("build_rv32im_main_recursion_construction2_committed_f_prime_full_vector")
            && !construction2.contains("Rv32imMainRecursionConstruction2FPrimeWitnessImage")
            && !construction2.contains("construction2_f_prime_witness_digest_values")
            && !construction2.contains("digest_layout")
            && !construction2.contains("commit_binary_columns")
            && !construction2.contains("PackedBinaryMatBitSink")
            && !construction2.contains("SeededBinaryColsCommitAudit"),
        "native Construction-2 witness image must not keep the removed witness-digest compatibility path"
    );
    assert!(
        terminal_committed.contains("Rv32imTerminalFPrimeR1csCcsRelation")
            && terminal_committed.contains("terminal_source_witness_debug_structure")
            && terminal_committed.contains("build_rv32im_terminal_f_prime_r2_circuit")
            && terminal_committed.contains("enforce_ajtai_commitment_linear_consistency")
            && terminal_committed.contains("committed_full_vector")
            && terminal_committed.contains("enforce_committed_superneo_image")
            && terminal_committed.contains("terminal_r2_witness_bit_bound")
            && terminal_committed.contains("terminal_r2_superneo_padding_zero")
            && terminal_committed.contains("UnusedPadding")
            && !terminal_committed.contains("return Self::U64;")
            && !terminal_committed.contains("append_f_slice_bits")
            && terminal_committed.contains("encode_vector_for_full_width")
            && terminal_committed.contains("commit_rv32im_main_recursion_construction2_packed_z")
            && terminal_committed.contains("Rv32imTerminalFPrimeR2Assignment")
            && terminal_committed.contains("superneo_pack_status"),
        "terminal F' committed-step compression must commit the low-norm SuperNeo R2 image and must not use the removed logical-image projection as authority"
    );
    assert!(
        compression.contains("committed_step_circuit")
            && compression.contains("setup_rv32im_terminal_f_prime_committed_step_relation")
            && compression.contains("prove_rv32im_terminal_f_prime_committed_step_relation")
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
        compression.contains("terminal_r2_source_ccs_rows")
            && compression.contains("terminal_r2_source_ccs_cols")
            && compression.contains("terminal_r2_source_ccs_nnz")
            && compression.contains("terminal_f_prime_r1cs_nnz")
            && !compression.contains("terminal_committed_step_logical_shape")
            && !compression.contains("terminal_relation.logical_field_count()"),
        "IVC SNARK setup cache keys must separate the terminal R2 source CCS shape from the terminal F' R1CS sparsity"
    );
}

#[test]
fn rv32im_final_construction2_boundary_uses_post_terminal_committed_instance() {
    let native_ivc = fs::read_to_string(crate_path("src/frontends/rv32im/ivc.rs")).expect("read native IVC module");
    let ivc_snark =
        fs::read_to_string(crate_path("src/frontends/rv32im/ivc_snark/mod.rs")).expect("read IVC SNARK module");
    let f_prime = fs::read_to_string(crate_path("src/frontends/rv32im/f_prime/mod.rs")).expect("read F' evaluator");
    let construction2 = fs::read_to_string(crate_path("src/frontends/rv32im/construction2/mod.rs"))
        .expect("read Construction-2 module");
    let recursive_step = fs::read_to_string(crate_path(
        "src/frontends/rv32im/main_relation_spartan/recursive_step/mod.rs",
    ))
    .expect("read recursive-step circuit");
    let terminal_committed = fs::read_to_string(crate_path(
        "src/frontends/rv32im/ivc_snark/terminal_f_prime_committed/mod.rs",
    ))
    .expect("read terminal F' committed-step owner");

    assert!(
        f_prime.contains("build_rv32im_main_recursion_construction2_fresh_instance_with_input_and_x_i")
            && f_prime.contains("construction2_u_next.x_i() != &x_out")
            && construction2.contains("rv32im_main_recursion_construction2_x_only_placeholder")
            && construction2.contains("Commitment::zeros(D, 1)"),
        "native F' must emit only the Construction-2 x_i image; terminal SuperNeo R2 owns the authoritative u_i.C"
    );
    assert!(
        native_ivc.contains("construction2_u_i: step_image.construction2_u_next")
            && ivc_snark.contains("public_image_with_terminal_r2_boundary_from_relation")
            && ivc_snark.contains("terminal_committed_step_inputs_from_backend")
            && native_ivc.contains("Rv32imMainRecursionConstruction2PublicBoundary::from_fresh_instance("),
        "native append must stay on x-only Construction-2 state while compression publishes the committed terminal R2 instance"
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
        setup_from_backend_shape.contains("commit_rv32im_main_recursion_construction2_packed_z")
            && setup_from_backend_shape.contains("Mat::zero(D, packed_cols, F::ZERO)")
            && !setup_from_backend_shape.contains("data: vec![F::ZERO"),
        "terminal F' setup/key derivation must use the same seeded Construction-2 Ajtai commitment context as proving"
    );
}

#[test]
fn rv32im_construction2_witness_image_excludes_public_step_label_cargo() {
    let construction2 =
        fs::read_to_string(crate_path("src/frontends/rv32im/construction2/mod.rs")).expect("read Construction-2");
    let recursive_step = fs::read_to_string(crate_path(
        "src/frontends/rv32im/main_relation_spartan/recursive_step/mod.rs",
    ))
    .expect("read recursive-step circuit");

    assert!(
        construction2.contains("does not admit public-step label cargo")
            && construction2.contains("if !step.label.is_empty()"),
        "RV32IM Construction-2 witness image must reject label cargo not covered by the canonical fixed width"
    );
    assert!(
        recursive_step.contains("label: String::new()"),
        "dummy recursive-step shape inputs must match the empty-label RV32IM Construction-2 boundary"
    );
}

#[test]
fn rv32im_recursive_step_chunk_digest_uses_authoritative_public_surfaces() {
    let recursive_step = fs::read_to_string(crate_path(
        "src/frontends/rv32im/main_relation_spartan/recursive_step/mod.rs",
    ))
    .expect("read recursive-step circuit");
    let chunk_replay = fs::read_to_string(crate_path(
        "src/frontends/rv32im/main_relation_spartan/recursive_step/chunk_replay.rs",
    ))
    .expect("read recursive-step chunk replay");
    let nifs_stages = fs::read_to_string(crate_path(
        "src/frontends/rv32im/main_relation_spartan/nifs_v_stages.rs",
    ))
    .expect("read NIFS stages");
    let public_chunk =
        fs::read_to_string(crate_path("src/circuit/superneo/public_chunk.rs")).expect("read public chunk");

    assert!(
        public_chunk.contains("pub fn rv32im_public_chunk_digest")
            && nifs_stages.contains("rv32im_public_chunk_digest(")
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
                "rv32im_recursive_accumulator_instance_digest_from_parts(\n            &payload.state_out_claims"
            ),
        "recursive-step public IO must bind terminal metadata to the live chunk replay surface"
    );
    assert!(
        chunk_replay.contains("replayed_next_claims.effective_claims()")
            && chunk_replay.contains("recursive_accumulator_instance_digest_circuit_from_claims")
            && !chunk_replay.contains("Rv32imChunkNextCarryMode::PreserveIncoming")
            && !chunk_replay.contains("recursive_accumulator_instance_digest_circuit_from_projection_digest_vars")
            && !chunk_replay.contains("state_out_expected_claims"),
        "recursive-step output accumulator digest must be recomputed from live NIFS.V output claims, not carried projection-digest cargo"
    );
}

#[test]
fn rv32im_recursive_state_identity_uses_phi_commitments_not_full_ce_projection_hashes() {
    let recursive_cover = fs::read_to_string(crate_path(
        "src/frontends/rv32im/main_relation_spartan/recursive_cover.rs",
    ))
    .expect("read recursive-cover circuit");
    let chunk_step_recursive = fs::read_to_string(crate_path(
        "src/frontends/rv32im/main_relation_spartan/chunk_step/recursive/mod.rs",
    ))
    .expect("read recursive-step payload builder");
    let final_relation =
        fs::read_to_string(crate_path("src/frontends/rv32im/final_relation.rs")).expect("read final relation");

    let carry_digest = chunk_step_recursive
        .split("pub(crate) fn rv32im_chunk_step_recursive_carry_state_digest")
        .nth(1)
        .and_then(|tail| tail.split("fn pad_matrix_to_shape").next())
        .expect("carry-state digest function");
    assert!(
        recursive_cover.contains("recursive_accumulator_instance_digest_circuit_from_phi_dec_parent_vars")
            && recursive_cover.contains("main_recursion_recursive_accumulator_phi_dec_parent")
            && recursive_cover.contains("claim.commitment.data")
            && recursive_cover.contains("params.b")
            && !recursive_cover.contains("me_input_projection_digest_poseidon"),
        "recursive accumulator circuit identity must hash the SuperNeo Π_DEC parent commitment handle, not full CE projection surfaces"
    );
    assert!(
        carry_digest.contains("main_recursion_fixed_step_accumulator_phi_commitments")
            && carry_digest.contains("claim.c.data")
            && carry_digest.contains("rv32im_chunk_fold_transcript_snapshot_digest")
            && !carry_digest.contains("me_input_projection_digest_poseidon")
            && !carry_digest.contains("claim.X")
            && !carry_digest.contains("claim.r")
            && !carry_digest.contains("claim.y_ring"),
        "recursive carry-state digest must use φ(commitments), transcript state, and terminal handle only"
    );
    assert!(
        final_relation.contains("rv32im_recursive_accumulator_instance_digest_from_phi_dec_parent")
            && final_relation.contains("rv32im_recursive_accumulator_phi_dec_parent_commitment")
            && final_relation.contains("scale_commitment_add_inplace")
            && final_relation.contains("rv32im_chunk_fold_carry_recursive_accumulator_digest"),
        "native folded-accumulator identity must share the same SuperNeo Π_DEC parent commitment handle as the circuit"
    );
}

#[test]
fn rv32im_pi_ccs_fs_binding_documents_accumulator_handle_boundary() {
    let pi_ccs =
        fs::read_to_string(crate_path("src/circuit/superneo/pi_ccs.rs")).expect("read SuperNeo Pi_CCS circuit");
    let chunk_replay = fs::read_to_string(crate_path(
        "src/frontends/rv32im/main_relation_spartan/recursive_step/chunk_replay.rs",
    ))
    .expect("read recursive chunk replay");
    let recursive_step = fs::read_to_string(crate_path(
        "src/frontends/rv32im/main_relation_spartan/recursive_step/mod.rs",
    ))
    .expect("read recursive step");
    assert!(
        pi_ccs.contains("FS binding invariant")
            && pi_ccs.contains("bind_me_inputs_accumulator_handle")
            && pi_ccs.contains("PI_CCS_ME_ACCUMULATOR_HANDLE_RAW_TAG")
            && pi_ccs.contains("final R1 CE proves")
            && pi_ccs.contains("cannot be chosen after challenges")
            && chunk_replay.contains("state_in_var.folded_accumulator_digest")
            && recursive_step.contains("state_in_folded_accumulator_digest_eq_live"),
        "Pi_CCS challenge sampling must use the carried accumulator handle and later constrain it to live φ(commitments)"
    );
}

#[test]
fn rv32im_verified_step_statement_digest_has_one_native_public_formula() {
    let construction2 =
        fs::read_to_string(crate_path("src/frontends/rv32im/construction2/mod.rs")).expect("read Construction-2");
    let terminal_statement = fs::read_to_string(crate_path(
        "src/frontends/rv32im/main_relation_spartan/recursive_step/terminal_statement.rs",
    ))
    .expect("read terminal F' statement");
    let expected_digest = construction2
        .split("pub(crate) fn expected_digest(&self) -> [u8; 32]")
        .nth(1)
        .and_then(|tail| {
            tail.split("fn validate_rv32im_main_recursion_construction2_ce_claim_surface")
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
            && !expected_digest.contains("append_message(\n            b\"neo.fold.next/rv32im/main_recursion_construction2_verified_step_statement/state_in\""),
        "verified-step digest must use canonical field-limb transcript inputs, matching the terminal F' circuit"
    );
    assert!(
        terminal_statement
            .contains("Ok(build_terminal_f_prime_verified_step_statement(backend_relation)?.expected_digest())",),
        "terminal F' public target must delegate to the same native verified-step digest formula"
    );
}

#[test]
fn rv32im_final_ce_bundle_uses_superneo_ce_projection_digest() {
    let source =
        fs::read_to_string(crate_path("src/circuit/superneo/ce_spartan.rs")).expect("read SuperNeo CE Spartan");
    let compression =
        fs::read_to_string(crate_path("src/frontends/rv32im/ivc_snark/mod.rs")).expect("read IVC compression module");
    assert!(
        source.contains("struct Rv32imCeBundleCircuit")
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
fn rv32im_final_ce_tamper_coverage_targets_full_superneo_claim_fields() {
    let ce_spartan =
        fs::read_to_string(crate_path("src/circuit/superneo/ce_spartan.rs")).expect("read SuperNeo CE Spartan");
    let compressed_verify =
        fs::read_to_string(crate_path("src/frontends/rv32im/ivc_snark/mod.rs")).expect("read IVC SNARK verifier");
    let decider_test =
        fs::read_to_string(crate_path("tests/rv32im_spartan2_decider.rs")).expect("read Spartan decider tests");

    assert!(
        ce_spartan.contains("verify_rv32im_ce_bundle_relation")
            && ce_spartan.contains("me_input_projection_digest_poseidon_values_from_native_claim")
            && ce_spartan.contains("enforce_paper_ce_claim_consistency")
            && compressed_verify.contains("verify_rv32im_final_ce_bundle")
            && compressed_verify.contains("verify_rv32im_ce_bundle_relation"),
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
fn rv32im_terminal_f_prime_exports_sparse_superneo_ccs_shape() {
    let terminal_owner = fs::read_to_string(crate_path(
        "src/frontends/rv32im/ivc_snark/terminal_f_prime_committed/mod.rs",
    ))
    .expect("read terminal F' committed relation owner");
    let terminal_circuit = fs::read_to_string(crate_path(
        "src/frontends/rv32im/ivc_snark/terminal_f_prime_committed/proof_circuit.rs",
    ))
    .expect("read terminal F' committed relation circuit");
    let construction2_terminal = read_construction2_terminal_helpers();
    let terminal = format!("{terminal_owner}\n{terminal_circuit}\n{construction2_terminal}");
    let r1cs = fs::read_to_string(workspace_path("crates/neo-ccs/src/r1cs.rs")).expect("read neo-ccs R1CS bridge");

    assert!(
        r1cs.contains("pub fn sparse_r1cs_to_ccs")
            && r1cs.contains("CcsStructure::new_sparse")
            && r1cs.contains("f(X0,X1,X2) = X0 * X1 - X2"),
        "R1CS->CCS conversion used by terminal F' must preserve sparse matrices and the row-wise R1CS polynomial"
    );
    assert!(
        terminal.contains("Rv32imTerminalFPrimeR1csCcsRelation")
            && terminal.contains("terminal_source_witness_debug_structure")
            && terminal.contains("committed_full_vector")
            && terminal.contains("split_terminal_r2_public_values")
            && terminal.contains("terminal_r2_public_value_range")
            && terminal.contains("RV32IM_ENC_INST_BITS")
            && terminal.contains("r2_public_values")
            && terminal.contains("low_norm_bit_values")
            && terminal.contains("UnusedPadding")
            && terminal.contains("padded_private_witness_labels")
            && terminal.contains("num_shared_unpadded")
            && terminal.contains("num_precommitted_unpadded")
            && terminal.contains("num_rest_unpadded")
            && terminal.contains("source low-norm values may be committed")
            && !terminal.contains("enforce_u32_allocated")
            && !terminal.contains("true_r2_image")
            && !terminal.contains("legacy_binary_full_width")
            && terminal.contains("num_spartan_public")
            && !terminal.contains("num_io + col")
            && terminal.contains("source_lc"),
        "terminal F' export must map the 256-bit Construction-2 x_i image into the SuperNeo R2 public instance without retaining the legacy logical/binary witness image"
    );
}

#[test]
fn rv32im_terminal_f_prime_rejects_post_hoc_full_field_bit_expansion() {
    let terminal_owner = fs::read_to_string(crate_path(
        "src/frontends/rv32im/ivc_snark/terminal_f_prime_committed/mod.rs",
    ))
    .expect("read terminal F' committed relation owner");

    assert!(
        terminal_owner.contains("source low-norm values may be committed as SuperNeo digits")
            && terminal_owner.contains("committed_full_vector")
            && terminal_owner.contains("committed_packed_witness")
            && !terminal_owner.contains("post-hoc bit-expand")
            && !terminal_owner.contains("spartan_sparse_to_superneo_ccs_matrix"),
        "terminal F' R2 export must fail before constructing a 64x low-norm expansion over full-field auxiliaries"
    );
}

#[test]
fn rv32im_default_u_perp_does_not_reintroduce_logical_image_width() {
    let default_pair = fs::read_to_string(crate_path("src/frontends/rv32im/construction2/default.rs"))
        .expect("read Construction-2 default-pair owner");
    assert!(
        default_pair.contains("Ok(RV32IM_ENC_INST_BITS)")
            && !default_pair.contains("FPrimeLowNormWitnessImage")
            && !default_pair.contains("binary_values")
            && !default_pair.contains("logical_field")
            && !default_pair.contains("* 64"),
        "canonical u_perp must not derive a binary logical-image witness width; terminal SuperNeo R2 owns the committed F' witness"
    );
}

#[test]
fn rv32im_recursive_shape_only_elides_common_zero_rlc_suffix() {
    let source = fs::read_to_string(crate_path(
        "src/frontends/rv32im/main_relation_spartan/chunk_step/recursive/mod.rs",
    ))
    .expect("read recursive-step payload builder");
    let nifs = fs::read_to_string(crate_path(
        "src/frontends/rv32im/main_relation_spartan/nifs_v_stages.rs",
    ))
    .expect("read NIFS stages");
    let pi_rlc_public =
        fs::read_to_string(crate_path("src/circuit/superneo/pi_rlc/public.rs")).expect("read Pi_RLC public circuit");
    let pi_rlc_constraints = fs::read_to_string(crate_path("src/circuit/superneo/pi_rlc/constraints.rs"))
        .expect("read Pi_RLC constraint helpers");
    let pi_rlc_rho =
        fs::read_to_string(crate_path("src/circuit/superneo/pi_rlc/rho_action.rs")).expect("read Pi_RLC rho action");
    let pi_rlc = [pi_rlc_public, pi_rlc_constraints, pi_rlc_rho].join("\n");
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

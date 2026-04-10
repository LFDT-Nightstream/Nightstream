use neo_fold_next::decider::spartan2::{
    prove_spartan2_backend_binding_shell, prove_spartan2_public_relation_shell, prove_spartan2_public_target_shell,
    setup_spartan2_backend_binding_shell, setup_spartan2_public_relation_shell, setup_spartan2_public_target_shell,
    Spartan2BackendBindingShellSnark, Spartan2DeciderProof, Spartan2PublicRelationShellSnark,
    Spartan2PublicTargetShellSnark,
};
use neo_fold_next::nightstream::rv64im::{
    audit::build_rv64im_hybrid_side_bridge_public_target, build_rv64im_nightstream_from_public_proof,
    Rv64imHybridSideBridgeBackendProof,
};
use neo_fold_next::rv64im::final_relation::prove_rv64im_final_statement_from_accepted;
use neo_fold_next::rv64im::{
    build_rv64im_accepted_proof_artifact, build_rv64im_spartan2_decider_target, parity_source_cases,
    prove_rv64im_public_proof, prove_rv64im_spartan2_decider, setup_rv64im_spartan2_decider, Rv64imProofInput,
};
use spartan2::spartan::R1CSSNarkSerializedSizeBreakdown;
use std::fmt::Write as _;

fn source_case(name: &str) -> neo_fold_next::rv64im::Rv64imParitySourceCase {
    parity_source_cases()
        .into_iter()
        .find(|case| case.manifest.name == name)
        .unwrap_or_else(|| panic!("missing parity source case {name}"))
}

fn proof_input(name: &str) -> Rv64imProofInput {
    let source = source_case(name);
    let max_steps = source.program_words.len();
    Rv64imProofInput { source, max_steps }
}

fn format_breakdown(label: &str, breakdown: R1CSSNarkSerializedSizeBreakdown) -> String {
    let mut out = String::new();
    let _ = write!(&mut out, "{label}: total={}", breakdown.total);
    for (component, bytes) in breakdown.measured_components() {
        let _ = write!(&mut out, " {component}={bytes}");
    }
    out
}

fn decode_decider_proof(proof: &Spartan2DeciderProof) -> Spartan2BackendBindingShellSnark {
    bincode::deserialize(&proof.snark_data).expect("decode backend-binding shell snark")
}

#[test]
fn rv64im_spartan2_shell_size_snapshot() {
    let input = proof_input("control_flow_jal_skip_ecall");
    let proof = prove_rv64im_public_proof(&input).expect("prove rv64im public proof");
    let artifact = build_rv64im_accepted_proof_artifact(&proof).expect("build accepted artifact");
    let (nightstream_statement, nightstream_proof) =
        build_rv64im_nightstream_from_public_proof(&proof).expect("build rv64im nightstream");
    let (statement, final_proof) =
        prove_rv64im_final_statement_from_accepted(&artifact).expect("prove rv64im final statement");
    let target = build_rv64im_spartan2_decider_target(&statement, &final_proof).expect("build rv64im decider target");
    let hybrid_side_bridge_target =
        build_rv64im_hybrid_side_bridge_public_target(&nightstream_statement, &nightstream_proof, &proof.statement)
            .expect("build hybrid-side-bridge public target");

    let shape = target.shape();
    let (public_pk, _) = setup_spartan2_public_target_shell(&shape).expect("setup public-target shell");
    let public_shell = prove_spartan2_public_target_shell(&public_pk, &target).expect("prove public-target shell");
    let (hybrid_side_bridge_pk, _) = setup_spartan2_public_target_shell(&hybrid_side_bridge_target.shape())
        .expect("setup hybrid-side-bridge public-target shell");
    let hybrid_side_bridge_shell =
        prove_spartan2_public_target_shell(&hybrid_side_bridge_pk, &hybrid_side_bridge_target)
            .expect("prove hybrid-side-bridge public-target shell");
    let (hybrid_side_bridge_relation_pk, _) = setup_spartan2_public_relation_shell(&hybrid_side_bridge_target.shape())
        .expect("setup hybrid-side-bridge public-relation shell");
    let hybrid_side_bridge_relation_shell =
        prove_spartan2_public_relation_shell(&hybrid_side_bridge_relation_pk, &hybrid_side_bridge_target)
            .expect("prove hybrid-side-bridge public-relation shell");
    let hybrid_side_bridge_backend_relation = hybrid_side_bridge_target.backend_relation();
    let (hybrid_side_bridge_backend_pk, _) =
        setup_spartan2_backend_binding_shell(&hybrid_side_bridge_backend_relation.shape())
            .expect("setup hybrid-side-bridge backend-binding shell");
    let hybrid_side_bridge_backend_shell =
        prove_spartan2_backend_binding_shell(&hybrid_side_bridge_backend_pk, &hybrid_side_bridge_backend_relation)
            .expect("prove hybrid-side-bridge backend-binding shell");
    let hybrid_side_bridge_backend_proof = Rv64imHybridSideBridgeBackendProof {
        snark_data: hybrid_side_bridge_backend_shell.snark_data.clone(),
    };

    let (decider_pk, _) = setup_rv64im_spartan2_decider(&statement, &final_proof).expect("setup rv64im decider");
    let decider_proof =
        prove_rv64im_spartan2_decider(&decider_pk, &statement, &final_proof).expect("prove rv64im decider");

    println!(
        "rv64im spartan2 shell size snapshot: base_components={} chunk_transitions={} public_io={} backend_public_io={} main_public_target_shell={} main_backend_binding_decider={} hybrid_side_bridge_base_components={} hybrid_side_bridge_chunk_transitions={} hybrid_side_bridge_public_io={} hybrid_side_bridge_backend_public_io={} hybrid_side_bridge_public_target_shell={} hybrid_side_bridge_public_relation_shell={} hybrid_side_bridge_backend_binding_shell={} hybrid_side_bridge_backend_proof={}",
        shape.base_component_count,
        shape.chunk_transition_count,
        shape.public_io_len(),
        shape.backend_public_io_len(),
        public_shell.snark_bytes_len(),
        decider_proof.snark_bytes_len(),
        hybrid_side_bridge_target.shape().base_component_count,
        hybrid_side_bridge_target.shape().chunk_transition_count,
        hybrid_side_bridge_target.shape().public_io_len(),
        hybrid_side_bridge_target.shape().backend_public_io_len(),
        hybrid_side_bridge_shell.snark_bytes_len(),
        hybrid_side_bridge_relation_shell.snark_bytes_len(),
        hybrid_side_bridge_backend_shell.snark_bytes_len(),
        bincode::serialize(&hybrid_side_bridge_backend_proof)
            .expect("serialize hybrid-side-bridge backend proof")
            .len(),
    );

    let public_target_snark: Spartan2PublicTargetShellSnark =
        bincode::deserialize(&public_shell.snark_data).expect("decode public-target shell snark");
    let decider_snark = decode_decider_proof(&decider_proof);
    println!(
        "{}",
        format_breakdown(
            "rv64im spartan2 public-target shell breakdown",
            public_target_snark
                .serialized_size_breakdown()
                .expect("measure public-target shell"),
        )
    );
    println!(
        "{}",
        format_breakdown(
            "rv64im spartan2 backend-binding decider breakdown",
            decider_snark
                .serialized_size_breakdown()
                .expect("measure decider shell"),
        )
    );
    let hybrid_side_bridge_snark: Spartan2PublicTargetShellSnark =
        bincode::deserialize(&hybrid_side_bridge_shell.snark_data)
            .expect("decode hybrid-side-bridge public-target shell snark");
    println!(
        "{}",
        format_breakdown(
            "rv64im hybrid-side-bridge spartan2 public-target shell breakdown",
            hybrid_side_bridge_snark
                .serialized_size_breakdown()
                .expect("measure hybrid-side-bridge public-target shell"),
        )
    );
    let hybrid_side_bridge_relation_snark: Spartan2PublicRelationShellSnark =
        bincode::deserialize(&hybrid_side_bridge_relation_shell.snark_data)
            .expect("decode hybrid-side-bridge public-relation shell snark");
    println!(
        "{}",
        format_breakdown(
            "rv64im hybrid-side-bridge spartan2 public-relation shell breakdown",
            hybrid_side_bridge_relation_snark
                .serialized_size_breakdown()
                .expect("measure hybrid-side-bridge public-relation shell"),
        )
    );
    let hybrid_side_bridge_backend_snark: Spartan2BackendBindingShellSnark =
        bincode::deserialize(&hybrid_side_bridge_backend_shell.snark_data)
            .expect("decode hybrid-side-bridge backend-binding shell snark");
    println!(
        "{}",
        format_breakdown(
            "rv64im hybrid-side-bridge spartan2 backend-binding shell breakdown",
            hybrid_side_bridge_backend_snark
                .serialized_size_breakdown()
                .expect("measure hybrid-side-bridge backend-binding shell"),
        )
    );

    assert!(public_shell.snark_bytes_len() > 0);
    assert!(decider_proof.snark_bytes_len() > 0);
    assert!(hybrid_side_bridge_shell.snark_bytes_len() > 0);
    assert!(hybrid_side_bridge_relation_shell.snark_bytes_len() > 0);
    assert!(hybrid_side_bridge_backend_shell.snark_bytes_len() > 0);
    assert!(hybrid_side_bridge_backend_proof.snark_bytes_len() > 0);
}

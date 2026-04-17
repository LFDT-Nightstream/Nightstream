//! Owns compact digest reconstructions for RV64IM Nightstream theorem-facing surfaces.

use neo_transcript::{Poseidon2Transcript, Transcript};

use crate::finalize::digest32_as_fields;

pub(super) fn kernel_claim_summary_digest_from_surfaces(
    prepared_step_bindings_digest: [u8; 32],
    root0_digest: [u8; 32],
    execution_digest: [u8; 32],
    final_state_digest: [u8; 32],
    transcript_final_digest: [u8; 32],
    final_pc: u64,
    halted: bool,
) -> [u8; 32] {
    let mut terminal = Poseidon2Transcript::new(b"neo.fold.next/rv64im/kernel_claim_terminal_bundle");
    terminal.append_message(b"rv64im/kernel_claim_terminal_bundle/root0_digest", &root0_digest);
    terminal.append_message(
        b"rv64im/kernel_claim_terminal_bundle/execution_digest",
        &execution_digest,
    );
    terminal.append_message(
        b"rv64im/kernel_claim_terminal_bundle/final_state_digest",
        &final_state_digest,
    );
    terminal.append_message(
        b"rv64im/kernel_claim_terminal_bundle/transcript_final_digest",
        &transcript_final_digest,
    );
    terminal.append_u64s(b"rv64im/kernel_claim_terminal_bundle/meta", &[final_pc, halted as u64]);
    let terminal_digest = terminal.digest32();

    let mut summary = Poseidon2Transcript::new(b"neo.fold.next/rv64im/kernel_claim_summary_bundle");
    summary.append_message(
        b"rv64im/kernel_claim_summary_bundle/prepared_step_bindings_digest",
        &prepared_step_bindings_digest,
    );
    summary.append_message(b"rv64im/kernel_claim_summary_bundle/terminal_digest", &terminal_digest);
    summary.digest32()
}

pub(super) fn stage_package_bundle_digest_from_surfaces(
    stage1_packaged_digest: [u8; 32],
    stage2_packaged_digest: [u8; 32],
    stage3_packaged_digest: [u8; 32],
) -> [u8; 32] {
    let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv64im/stage_package_bundle");
    tr.append_message(b"rv64im/stage_package_bundle/stage1", &stage1_packaged_digest);
    tr.append_message(b"rv64im/stage_package_bundle/stage2", &stage2_packaged_digest);
    tr.append_message(b"rv64im/stage_package_bundle/stage3", &stage3_packaged_digest);
    tr.digest32()
}

pub(super) fn stage_package_proof_bundle_digest_from_surfaces(
    stage1_packaged_digest: [u8; 32],
    stage2_packaged_digest: [u8; 32],
    stage3_packaged_digest: [u8; 32],
) -> [u8; 32] {
    let package_bundle_digest = stage_package_bundle_digest_from_surfaces(
        stage1_packaged_digest,
        stage2_packaged_digest,
        stage3_packaged_digest,
    );
    let mut summary = Poseidon2Transcript::new(b"neo.fold.next/rv64im/stage_package_digest_bundle");
    summary.append_message(
        b"rv64im/stage_package_digest_bundle/package_bundle_digest",
        &package_bundle_digest,
    );
    summary.append_message(
        b"rv64im/stage_package_digest_bundle/stage1_digest",
        &stage1_packaged_digest,
    );
    summary.append_message(
        b"rv64im/stage_package_digest_bundle/stage2_digest",
        &stage2_packaged_digest,
    );
    summary.append_message(
        b"rv64im/stage_package_digest_bundle/stage3_digest",
        &stage3_packaged_digest,
    );
    let summary_digest = summary.digest32();

    let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv64im/stage_package_proof_bundle");
    tr.append_message(b"rv64im/stage_package_proof_bundle/summary", &summary_digest);
    tr.digest32()
}

pub(super) fn packaged_claim_proof_digest_from_surfaces(
    label: &'static [u8],
    summary_digest: [u8; 32],
    statement_digest: [u8; 32],
    proof_digest: [u8; 32],
) -> [u8; 32] {
    let mut tr = Poseidon2Transcript::new(label);
    tr.append_message(b"summary_digest", &summary_digest);
    tr.append_message(b"statement_digest", &statement_digest);
    tr.append_message(b"proof_digest", &proof_digest);
    tr.digest32()
}

pub(super) fn packaged_opening_proof_digest_from_surfaces(
    claim_digest: [u8; 32],
    statement_digest: [u8; 32],
    proof_digest: [u8; 32],
) -> [u8; 32] {
    let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv64im/stage_packaged_opening_claim_proof");
    tr.append_fields(
        b"rv64im/stage_packaged_opening_claim_proof/claim_digest",
        &digest32_as_fields(claim_digest),
    );
    tr.append_fields(
        b"rv64im/stage_packaged_opening_claim_proof/statement_digest",
        &digest32_as_fields(statement_digest),
    );
    tr.append_fields(
        b"rv64im/stage_packaged_opening_claim_proof/proof_digest",
        &digest32_as_fields(proof_digest),
    );
    tr.digest32()
}

pub(super) fn kernel_opening_bundle_digest_from_surfaces(
    claim_digest: [u8; 32],
    bindings_opening_digest: [u8; 32],
    prepared_steps_opening_digest: [u8; 32],
) -> [u8; 32] {
    let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv64im/kernel_opening_bundle");
    tr.append_message(b"rv64im/kernel_opening_bundle/claim_digest", &claim_digest);
    tr.append_message(b"rv64im/kernel_opening_bundle/bindings", &bindings_opening_digest);
    tr.append_message(
        b"rv64im/kernel_opening_bundle/prepared_steps",
        &prepared_steps_opening_digest,
    );
    tr.digest32()
}

pub(super) fn kernel_opening_binding_bundle_digest_from_surfaces(
    claim_digest: [u8; 32],
    bindings_opening_digest: [u8; 32],
    prepared_steps_opening_digest: [u8; 32],
) -> [u8; 32] {
    let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv64im/kernel_opening_binding_bundle");
    tr.append_message(b"rv64im/kernel_opening_binding_bundle/claim_digest", &claim_digest);
    tr.append_message(
        b"rv64im/kernel_opening_binding_bundle/bindings_digest",
        &bindings_opening_digest,
    );
    tr.append_message(
        b"rv64im/kernel_opening_binding_bundle/prepared_steps_digest",
        &prepared_steps_opening_digest,
    );
    tr.digest32()
}

pub(super) fn kernel_opening_proof_bundle_digest_from_surfaces(
    opening_bundle_digest: [u8; 32],
    binding_bundle_digest: [u8; 32],
) -> [u8; 32] {
    let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv64im/kernel_opening_proof_bundle");
    tr.append_message(
        b"rv64im/kernel_opening_proof_bundle/opening_digest",
        &opening_bundle_digest,
    );
    tr.append_message(b"rv64im/kernel_opening_proof_bundle/bindings", &binding_bundle_digest);
    tr.digest32()
}

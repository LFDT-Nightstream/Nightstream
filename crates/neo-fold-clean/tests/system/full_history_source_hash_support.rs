use serde_json::Value;

use super::full_history_manifest_identity_support::source_hash;
use super::*;

/// Review metadata for every source that controls the generated M4 relation.
/// These hashes are drift sentinels only; none is used as semantic authority.
pub fn full_history_source_hashes() -> Vec<Value> {
    const SOURCES: &[&str] = &[
        "Cargo.lock",
        "crates/neo-fold-clean/src/engine/r1cs_circuit/builder.rs",
        "crates/neo-ccs/src/seeded_phi81.rs",
        "crates/neo-fold-clean/src/engine/r1cs_circuit/ring_action.rs",
        "crates/neo-fold-clean/src/engine/decider.rs",
        "crates/neo-fold-clean/src/engine/decider/public_image.rs",
        "crates/neo-fold-clean/src/paper/f_prime/r1cs.rs",
        "crates/neo-fold-clean/src/paper/nifs/circuit/mod.rs",
        "crates/neo-fold-clean/src/paper/nifs/circuit/pi_rlc/consistency.rs",
        "crates/neo-fold-clean/src/paper/nifs/circuit/pi_rlc/fold_wires.rs",
        "crates/neo-fold-clean/src/paper/nifs/circuit/pi_rlc/mod.rs",
        "crates/neo-fold-clean/src/paper/nifs/circuit/pi_rlc/padding.rs",
        "crates/neo-fold-clean/src/paper/nifs/circuit/pi_rlc/projection/binding.rs",
        "crates/neo-fold-clean/src/paper/nifs/circuit/pi_rlc/projection/identities.rs",
        "crates/neo-fold-clean/src/paper/nifs/circuit/pi_rlc/projection/mod.rs",
        "crates/neo-fold-clean/src/paper/nifs/circuit/pi_rlc/projection/shared.rs",
        "crates/neo-fold-clean/src/paper/reductions/pi_ccs_split_nc_circuit/verifier.rs",
        "crates/neo-fold-clean/src/paper/reductions/pi_ccs_split_nc_circuit/digests.rs",
        "crates/neo-fold-clean/src/paper/reductions/accumulator_sis_circuit.rs",
        "crates/neo-fold-clean/src/paper/decider_ce_relation/mod.rs",
        "crates/neo-fold-clean/src/paper/decider_ce_relation/commitment.rs",
        "crates/neo-fold-clean/src/paper/decider_ce_relation/evaluation.rs",
        "crates/neo-fold-clean/src/paper/decider_ce_relation/witness.rs",
        "crates/neo-fold-clean/tests/gadgets/checked_program_artifact_support.rs",
        "crates/neo-fold-clean/tests/gadgets/lean_artifact_support.rs",
        "crates/neo-fold-clean/tests/gadgets/seeded_phi81_lean_artifact.rs",
        "crates/neo-fold-clean/tests/system/full_history_equality_artifact_support.rs",
        "crates/neo-fold-clean/tests/system/full_history_encoding_artifact_support.rs",
        "crates/neo-fold-clean/tests/system/full_history_counter_artifact_support.rs",
        "crates/neo-fold-clean/tests/system/full_history_affine_artifact_support.rs",
        "crates/neo-fold-clean/tests/system/full_history_manifest_identity_support.rs",
        "crates/neo-fold-clean/tests/system/full_history_nested_manifest_support.rs",
        "crates/neo-fold-clean/tests/system/full_history_pi_dec_artifact_support.rs",
        "crates/neo-fold-clean/tests/system/full_history_projection_role_support.rs",
        "crates/neo-fold-clean/tests/system/full_history_public_pins_artifact_support.rs",
        "crates/neo-fold-clean/tests/system/full_history_recursive_output_artifact_support.rs",
        "crates/neo-fold-clean/tests/system/full_history_seeded_phi81_artifact_support.rs",
        "crates/neo-fold-clean/tests/system/full_history_source_hash_support.rs",
        "crates/neo-fold-clean/tests/system/full_history_terminal_ce_artifact_support.rs",
        "crates/neo-fold-clean/tests/system/full_history_terminal_accumulator_artifact_support.rs",
        "crates/neo-fold-clean/tests/system/decider_r1cs_manifest.rs",
        "formal/nightstream-lean/Nightstream/Implementation/R1CS/Core/ChaCha8.lean",
        "formal/nightstream-lean/Nightstream/Implementation/R1CS/Core/SeededPhi81.lean",
        "formal/nightstream-lean/Nightstream/Implementation/R1CS/Core/EqualityPins.lean",
        "formal/nightstream-lean/Nightstream/Implementation/R1CS/Ownership/FPrime/FPrimeEncodingArtifact.lean",
        "formal/nightstream-lean/Nightstream/Implementation/R1CS/Core/Relabel.lean",
        "formal/nightstream-lean/Nightstream/Implementation/R1CS/Core/ProjectionProgram.lean",
        "formal/nightstream-lean/Nightstream/Implementation/R1CS/Core/PackedProgram.lean",
        "formal/nightstream-lean/Nightstream/Implementation/R1CS/Core/LinearOutputs.lean",
        "formal/nightstream-lean/Nightstream/Implementation/R1CS/Core/AffinePins.lean",
        "formal/nightstream-lean/Nightstream/Implementation/R1CS/Correspondence/FPrimeFullHistory/FPrimeFullHistoryAffineSound.lean",
        "formal/nightstream-lean/Nightstream/Implementation/R1CS/Correspondence/Gadgets/TerminalCeCompiler.lean",
        "formal/nightstream-lean/Nightstream/Implementation/R1CS/Correspondence/Gadgets/TerminalCeSound.lean",
        "formal/nightstream-lean/Nightstream/Implementation/R1CS/Correspondence/FPrimeFullHistory/FPrimeFullHistoryTerminalCeSound.lean",
    ];

    SOURCES
        .iter()
        .map(|relative| source_hash(&formal_repo_root(), relative))
        .collect()
}

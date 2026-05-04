//! Owns internal conversion between simple-kernel exports and the public RV32IM proof surface.

use crate::proof::PackagedProof;

use super::proof_api::{
    Rv32imAcceptedProofClaim, Rv32imAcceptedProofMainLaneBinding, Rv32imAcceptedProofStatementBinding,
    Rv32imAcceptedProofTerminalBinding, Rv32imJointOpeningClaim, Rv32imJointOpeningClaimBinding,
    Rv32imKernelClaimBundle, Rv32imKernelOpeningClaim, Rv32imKernelOpeningStageClaimBinding,
    Rv32imKernelOpeningTerminalClaimBinding, Rv32imKernelProofBundle, Rv32imMainLaneClaim, Rv32imMainLaneClaimBinding,
    Rv32imMainLaneProofBinding, Rv32imMainLaneProofBundle, Rv32imProof, Rv32imProofStatement, Rv32imRoot0Claim,
    Rv32imRoot0StageClaimBinding, Rv32imRoot0TerminalClaimBinding,
};
use super::proof_witness::{Rv32imProofWitnessBundle, Rv32imStageWitnessProjectionBundle, Rv32imTraceProjectionBundle};
use super::simple::PublicSimpleKernelOutput;
use super::{build_main_lane_surface, SimpleKernelMainLaneArtifact};

fn joint_opening_claim_from_claims(
    statement: &Rv32imProofStatement,
    root_params_id: [u8; 32],
    main_lane: &Rv32imMainLaneClaim,
    opening: &Rv32imKernelOpeningClaim,
) -> Rv32imJointOpeningClaim {
    let binding = Rv32imJointOpeningClaimBinding {
        proof_statement_digest: statement.digest,
        main_lane_claim_digest: main_lane.digest,
        kernel_opening_claim_digest: opening.digest,
        digest: [0; 32],
    };
    let binding = Rv32imJointOpeningClaimBinding {
        digest: binding.expected_digest(),
        ..binding
    };
    let claim = Rv32imJointOpeningClaim {
        root_params_id,
        binding,
        digest: [0; 32],
    };
    Rv32imJointOpeningClaim {
        digest: claim.expected_digest(),
        ..claim
    }
}

pub(super) fn main_lane_proof_bundle_from_artifact(
    main_lane: &SimpleKernelMainLaneArtifact,
    packaged: PackagedProof,
) -> Rv32imMainLaneProofBundle {
    let binding = Rv32imMainLaneProofBinding {
        root_lane_columns_digest: main_lane.binding.root_lane_columns_digest,
        root_lane_commitment_digest: main_lane.binding.root_lane_commitment_digest,
        fold_schedule: main_lane.binding.fold_schedule,
        chunk_count: main_lane.binding.chunk_count,
        public_step_count: main_lane.binding.public_step_count,
        digest: [0; 32],
    };
    let binding = Rv32imMainLaneProofBinding {
        digest: binding.expected_digest(),
        ..binding
    };
    let bundle = Rv32imMainLaneProofBundle {
        binding,
        packaged,
        digest: [0; 32],
    };
    Rv32imMainLaneProofBundle {
        digest: bundle.expected_digest(),
        ..bundle
    }
}

pub(crate) fn accepted_proof_claim_from_statement_and_public_kernel(
    root_params_id: [u8; 32],
    statement: &Rv32imProofStatement,
    kernel_opening: &super::proof_witness::Rv32imKernelOpeningProofBundle,
    kernel_claims: &super::proof_witness::Rv32imKernelClaimProofBundle,
    main_lane: &Rv32imMainLaneProofBundle,
) -> Rv32imAcceptedProofClaim {
    let statement_binding = Rv32imAcceptedProofStatementBinding {
        proof_statement_digest: statement.digest,
        kernel_opening_digest: kernel_opening.digest,
        digest: [0; 32],
    };
    let statement_binding = Rv32imAcceptedProofStatementBinding {
        digest: statement_binding.expected_digest(),
        ..statement_binding
    };
    let main_lane = Rv32imAcceptedProofMainLaneBinding {
        main_lane_bundle_digest: main_lane.digest,
        digest: [0; 32],
    };
    let main_lane = Rv32imAcceptedProofMainLaneBinding {
        digest: main_lane.expected_digest(),
        ..main_lane
    };
    let terminal = Rv32imAcceptedProofTerminalBinding {
        final_state_digest: kernel_claims.final_state_digest(),
        final_pc: statement.final_pc,
        halted: statement.halted,
        digest: [0; 32],
    };
    let terminal = Rv32imAcceptedProofTerminalBinding {
        digest: terminal.expected_digest(),
        ..terminal
    };
    let claim = Rv32imAcceptedProofClaim {
        root_params_id,
        statement: statement_binding,
        main_lane,
        terminal,
        digest: [0; 32],
    };
    Rv32imAcceptedProofClaim {
        digest: claim.expected_digest(),
        ..claim
    }
}

fn main_lane_claim_from_public_kernel(
    root_params_id: [u8; 32],
    main_lane: &Rv32imMainLaneProofBundle,
) -> Rv32imMainLaneClaim {
    let binding = Rv32imMainLaneClaimBinding {
        main_lane_bundle_digest: main_lane.digest,
        digest: [0; 32],
    };
    let binding = Rv32imMainLaneClaimBinding {
        digest: binding.expected_digest(),
        ..binding
    };
    let claim = Rv32imMainLaneClaim {
        root_params_id,
        binding,
        digest: [0; 32],
    };
    Rv32imMainLaneClaim {
        digest: claim.expected_digest(),
        ..claim
    }
}

fn main_lane_claim_from_bundle_digest(
    root_params_id: [u8; 32],
    main_lane_bundle_digest: [u8; 32],
) -> Rv32imMainLaneClaim {
    let binding = Rv32imMainLaneClaimBinding {
        main_lane_bundle_digest,
        digest: [0; 32],
    };
    let binding = Rv32imMainLaneClaimBinding {
        digest: binding.expected_digest(),
        ..binding
    };
    let claim = Rv32imMainLaneClaim {
        root_params_id,
        binding,
        digest: [0; 32],
    };
    Rv32imMainLaneClaim {
        digest: claim.expected_digest(),
        ..claim
    }
}

fn kernel_opening_claim_from_public_kernel(
    root_params_id: [u8; 32],
    stage_claims: &super::proof_witness::Rv32imStageClaimProofBundle,
    stage_packages: &super::proof_witness::Rv32imStagePackageProofBundle,
    kernel_opening: &super::proof_witness::Rv32imKernelOpeningProofBundle,
    kernel_claims: &super::proof_witness::Rv32imKernelClaimProofBundle,
) -> Rv32imKernelOpeningClaim {
    let stages = Rv32imKernelOpeningStageClaimBinding {
        stage_claims_digest: stage_claims.digest,
        stage_packages_digest: stage_packages.digest,
        kernel_opening_digest: kernel_opening.digest,
        digest: [0; 32],
    };
    let stages = Rv32imKernelOpeningStageClaimBinding {
        digest: stages.expected_digest(),
        ..stages
    };
    let terminal = Rv32imKernelOpeningTerminalClaimBinding {
        prepared_step_bindings_digest: kernel_claims.prepared_step_bindings_digest(),
        execution_digest: kernel_claims.execution_digest(),
        transcript_final_digest: kernel_claims.transcript_final_digest(),
        digest: [0; 32],
    };
    let terminal = Rv32imKernelOpeningTerminalClaimBinding {
        digest: terminal.expected_digest(),
        ..terminal
    };
    let claim = Rv32imKernelOpeningClaim {
        root_params_id,
        stages,
        terminal,
        digest: [0; 32],
    };
    Rv32imKernelOpeningClaim {
        digest: claim.expected_digest(),
        ..claim
    }
}

fn root0_claim_from_public_kernel(
    root_params_id: [u8; 32],
    kernel_claims: &super::proof_witness::Rv32imKernelClaimProofBundle,
) -> Rv32imRoot0Claim {
    let summary = &kernel_claims.claims.kernel;
    let stages = Rv32imRoot0StageClaimBinding {
        stage1_digest: summary.stage1_digest,
        stage2_digest: summary.stage2_digest,
        stage3_digest: summary.stage3_digest,
        digest: [0; 32],
    };
    let stages = Rv32imRoot0StageClaimBinding {
        digest: stages.expected_digest(),
        ..stages
    };
    let terminal = Rv32imRoot0TerminalClaimBinding {
        root0_digest: kernel_claims.root0_digest(),
        execution_digest: kernel_claims.execution_digest(),
        final_state_digest: kernel_claims.final_state_digest(),
        transcript_final_digest: kernel_claims.transcript_final_digest(),
        digest: [0; 32],
    };
    let terminal = Rv32imRoot0TerminalClaimBinding {
        digest: terminal.expected_digest(),
        ..terminal
    };
    let claim = Rv32imRoot0Claim {
        root_params_id,
        stages,
        terminal,
        digest: [0; 32],
    };
    Rv32imRoot0Claim {
        digest: claim.expected_digest(),
        ..claim
    }
}

fn root0_claim_from_compact_surfaces(
    statement: &Rv32imProofStatement,
    stage1_digest: [u8; 32],
    stage2_digest: [u8; 32],
    stage3_digest: [u8; 32],
    root0_digest: [u8; 32],
) -> Rv32imRoot0Claim {
    let stages = Rv32imRoot0StageClaimBinding {
        stage1_digest,
        stage2_digest,
        stage3_digest,
        digest: [0; 32],
    };
    let stages = Rv32imRoot0StageClaimBinding {
        digest: stages.expected_digest(),
        ..stages
    };
    let terminal = Rv32imRoot0TerminalClaimBinding {
        root0_digest,
        execution_digest: statement.execution_digest,
        final_state_digest: statement.final_state_digest,
        transcript_final_digest: statement.transcript_final_digest,
        digest: [0; 32],
    };
    let terminal = Rv32imRoot0TerminalClaimBinding {
        digest: terminal.expected_digest(),
        ..terminal
    };
    let claim = Rv32imRoot0Claim {
        root_params_id: statement.root_params_id,
        stages,
        terminal,
        digest: [0; 32],
    };
    Rv32imRoot0Claim {
        digest: claim.expected_digest(),
        ..claim
    }
}

fn accepted_proof_claim_from_compact_surfaces(
    statement: &Rv32imProofStatement,
    main_lane_bundle_digest: [u8; 32],
) -> Rv32imAcceptedProofClaim {
    let statement_binding = Rv32imAcceptedProofStatementBinding {
        proof_statement_digest: statement.digest,
        kernel_opening_digest: statement.kernel_opening_digest,
        digest: [0; 32],
    };
    let statement_binding = Rv32imAcceptedProofStatementBinding {
        digest: statement_binding.expected_digest(),
        ..statement_binding
    };
    let main_lane = Rv32imAcceptedProofMainLaneBinding {
        main_lane_bundle_digest,
        digest: [0; 32],
    };
    let main_lane = Rv32imAcceptedProofMainLaneBinding {
        digest: main_lane.expected_digest(),
        ..main_lane
    };
    let terminal = Rv32imAcceptedProofTerminalBinding {
        final_state_digest: statement.final_state_digest,
        final_pc: statement.final_pc,
        halted: statement.halted,
        digest: [0; 32],
    };
    let terminal = Rv32imAcceptedProofTerminalBinding {
        digest: terminal.expected_digest(),
        ..terminal
    };
    let claim = Rv32imAcceptedProofClaim {
        root_params_id: statement.root_params_id,
        statement: statement_binding,
        main_lane,
        terminal,
        digest: [0; 32],
    };
    Rv32imAcceptedProofClaim {
        digest: claim.expected_digest(),
        ..claim
    }
}

fn kernel_opening_claim_from_statement(statement: &Rv32imProofStatement) -> Rv32imKernelOpeningClaim {
    let stages = Rv32imKernelOpeningStageClaimBinding {
        stage_claims_digest: statement.stage_claims_digest,
        stage_packages_digest: statement.stage_packages_digest,
        kernel_opening_digest: statement.kernel_opening_digest,
        digest: [0; 32],
    };
    let stages = Rv32imKernelOpeningStageClaimBinding {
        digest: stages.expected_digest(),
        ..stages
    };
    let terminal = Rv32imKernelOpeningTerminalClaimBinding {
        prepared_step_bindings_digest: statement.prepared_step_bindings_digest,
        execution_digest: statement.execution_digest,
        transcript_final_digest: statement.transcript_final_digest,
        digest: [0; 32],
    };
    let terminal = Rv32imKernelOpeningTerminalClaimBinding {
        digest: terminal.expected_digest(),
        ..terminal
    };
    let claim = Rv32imKernelOpeningClaim {
        root_params_id: statement.root_params_id,
        stages,
        terminal,
        digest: [0; 32],
    };
    Rv32imKernelOpeningClaim {
        digest: claim.expected_digest(),
        ..claim
    }
}

pub(crate) fn kernel_claim_bundle_from_statement_and_compact_surfaces(
    statement: &Rv32imProofStatement,
    main_lane_bundle_digest: [u8; 32],
    stage1_digest: [u8; 32],
    stage2_digest: [u8; 32],
    stage3_digest: [u8; 32],
    root0_digest: [u8; 32],
) -> Rv32imKernelClaimBundle {
    let accepted = accepted_proof_claim_from_compact_surfaces(statement, main_lane_bundle_digest);
    let main_lane = main_lane_claim_from_bundle_digest(statement.root_params_id, main_lane_bundle_digest);
    let opening = kernel_opening_claim_from_statement(statement);
    let claim = Rv32imKernelClaimBundle {
        accepted,
        main_lane: main_lane.clone(),
        opening: opening.clone(),
        joint_opening: joint_opening_claim_from_claims(statement, statement.root_params_id, &main_lane, &opening),
        root0: root0_claim_from_compact_surfaces(statement, stage1_digest, stage2_digest, stage3_digest, root0_digest),
        digest: [0; 32],
    };
    Rv32imKernelClaimBundle {
        digest: claim.expected_digest(),
        ..claim
    }
}

pub(crate) fn kernel_claim_bundle_from_statement_and_public_kernel(
    statement: &Rv32imProofStatement,
    main_lane_bundle: &Rv32imMainLaneProofBundle,
    stage_claims: &super::proof_witness::Rv32imStageClaimProofBundle,
    stage_packages: &super::proof_witness::Rv32imStagePackageProofBundle,
    kernel_opening: &super::proof_witness::Rv32imKernelOpeningProofBundle,
    kernel_claims: &super::proof_witness::Rv32imKernelClaimProofBundle,
) -> Rv32imKernelClaimBundle {
    let accepted = accepted_proof_claim_from_statement_and_public_kernel(
        statement.root_params_id,
        statement,
        kernel_opening,
        kernel_claims,
        main_lane_bundle,
    );
    let main_lane = main_lane_claim_from_public_kernel(statement.root_params_id, main_lane_bundle);
    let opening = kernel_opening_claim_from_public_kernel(
        statement.root_params_id,
        stage_claims,
        stage_packages,
        kernel_opening,
        kernel_claims,
    );
    let claim = Rv32imKernelClaimBundle {
        accepted,
        main_lane: main_lane.clone(),
        opening: opening.clone(),
        joint_opening: joint_opening_claim_from_claims(statement, statement.root_params_id, &main_lane, &opening),
        root0: root0_claim_from_public_kernel(statement.root_params_id, kernel_claims),
        digest: [0; 32],
    };
    Rv32imKernelClaimBundle {
        digest: claim.expected_digest(),
        ..claim
    }
}

fn kernel_proof_bundle_from_public_kernel(
    root_params_id: [u8; 32],
    kernel: &PublicSimpleKernelOutput,
    trace: Rv32imTraceProjectionBundle,
    stages: Rv32imStageWitnessProjectionBundle,
    stage_claims: super::proof_witness::Rv32imStageClaimProofBundle,
    stage_packages: super::proof_witness::Rv32imStagePackageProofBundle,
    kernel_opening: super::proof_witness::Rv32imKernelOpeningProofBundle,
    kernel_claims: super::proof_witness::Rv32imKernelClaimProofBundle,
    main_lane: Rv32imMainLaneProofBundle,
) -> Rv32imKernelProofBundle {
    let bundle = Rv32imKernelProofBundle {
        root_params_id,
        trace,
        stages,
        stage_claims,
        stage_packages,
        kernel_opening,
        kernel_claims,
        root_lane_columns: kernel.root_lane_columns.clone(),
        root_lane_commitment: kernel.root_lane_commitment.clone(),
        main_lane,
        digest: [0; 32],
    };
    Rv32imKernelProofBundle {
        digest: bundle.expected_digest(),
        ..bundle
    }
}

pub(super) fn proof_from_public_kernel_and_main_lane_bundle(
    root_params_id: [u8; 32],
    kernel: &PublicSimpleKernelOutput,
    main_lane: Rv32imMainLaneProofBundle,
    witness: Rv32imProofWitnessBundle,
) -> Result<Rv32imProof, super::simple::SimpleKernelError> {
    let main_lane_surface = build_main_lane_surface(&kernel.root_lane_columns);
    let trace = witness.trace.projection();
    let stages = witness.stages.projection_bundle();
    let stage_claims = witness.stage_claims.clone();
    let stage_packages = witness.stage_packages.clone();
    let kernel_opening = witness.kernel_opening.clone();
    let kernel_claims = witness.kernel_claims.clone();
    let initial_pc = witness
        .trace
        .trace
        .execution_rows
        .first()
        .map(|row| u64::from(row.pc))
        .unwrap_or(kernel_claims.final_pc());
    let statement = Rv32imProofStatement {
        root_params_id,
        fold_schedule: main_lane.binding.fold_schedule,
        chunk_count: main_lane.binding.chunk_count,
        stage_claims_digest: stage_claims.digest,
        stage_packages_digest: stage_packages.digest,
        kernel_opening_digest: kernel_opening.digest,
        prepared_step_bindings_digest: kernel_claims.prepared_step_bindings_digest(),
        execution_digest: trace.execution_digest(),
        final_state_digest: kernel_claims.final_state_digest(),
        transcript_final_digest: kernel_claims.transcript_final_digest(),
        main_lane_surface_digest: main_lane_surface.digest,
        root_lane_columns_digest: kernel.root_lane_columns.digest,
        public_step_count: kernel.root_lane_columns.time_len,
        initial_pc,
        final_pc: kernel_claims.final_pc(),
        halted: kernel_claims.halted(),
        digest: [0; 32],
    };
    let statement = Rv32imProofStatement {
        digest: statement.expected_digest(),
        ..statement
    };
    let claim = kernel_claim_bundle_from_statement_and_public_kernel(
        &statement,
        &main_lane,
        &stage_claims,
        &stage_packages,
        &kernel_opening,
        &kernel_claims,
    );
    let kernel = kernel_proof_bundle_from_public_kernel(
        root_params_id,
        kernel,
        trace,
        stages,
        stage_claims,
        stage_packages,
        kernel_opening,
        kernel_claims,
        main_lane,
    );
    Ok(Rv32imProof {
        claim,
        statement,
        kernel,
        witness,
    })
}

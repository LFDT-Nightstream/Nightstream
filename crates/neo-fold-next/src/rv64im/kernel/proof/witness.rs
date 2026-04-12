//! Owns RV64IM-owned witness-facing wrappers above the simple-kernel export.

use neo_transcript::{Poseidon2Transcript, Transcript};
use serde::{Deserialize, Serialize};

use super::simple::PublicSimpleKernelOutput;
use super::{
    rv64im_simple_root_context_id,
    stage_artifacts::{build_claim_packaged_proof, verify_claim_packaged_proof},
    RootLaneColumns, RootLaneCommitmentSummaryArtifact, Rv64imParityCaseManifest, SimpleKernelError,
    SimpleKernelKernelClaimBundle, SimpleKernelOpeningBundle, SimpleKernelStageClaimBundle,
    SimpleKernelStagePackageBundle, SimpleKernelStageWitnessBundle, SimpleKernelTraceWitness,
};
use crate::proof::{PackagedProof, PublicStep};
use crate::rv64im::lower::Rv64ExpandedRow;
use crate::rv64im::stage1::Stage1Summary;
use crate::rv64im::stage2::Stage2Summary;
use crate::rv64im::stage3::Stage3Summary;
use crate::rv64im::tables::Rv64FamilyTag;
use crate::rv64im::TranscriptRecord;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Rv64imProofWitnessBundle {
    pub root_params_id: [u8; 32],
    pub trace: Rv64imTraceProofBundle,
    pub stages: Rv64imStageWitnessProofBundle,
    pub stage_claims: Rv64imStageClaimProofBundle,
    pub stage_packages: Rv64imStagePackageProofBundle,
    pub kernel_opening: Rv64imKernelOpeningProofBundle,
    pub kernel_claims: Rv64imKernelClaimProofBundle,
    pub root_lane_columns: RootLaneColumns,
    pub root_lane_commitment: RootLaneCommitmentSummaryArtifact,
    pub digest: [u8; 32],
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Rv64imKernelOpeningBindingBundle {
    pub claim_digest: [u8; 32],
    pub bindings_digest: [u8; 32],
    pub prepared_steps_digest: [u8; 32],
    pub digest: [u8; 32],
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Rv64imKernelOpeningProofBundle {
    pub opening_digest: [u8; 32],
    pub bindings: Rv64imKernelOpeningBindingBundle,
    pub digest: [u8; 32],
    pub opening: SimpleKernelOpeningBundle,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Rv64imKernelOpeningSummaryBundle {
    pub opening_digest: [u8; 32],
    pub bindings: Rv64imKernelOpeningBindingBundle,
    pub digest: [u8; 32],
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Rv64imKernelClaimTerminalBundle {
    pub root0_digest: [u8; 32],
    pub execution_digest: [u8; 32],
    pub final_state_digest: [u8; 32],
    pub transcript_final_digest: [u8; 32],
    pub final_pc: u64,
    pub halted: bool,
    pub digest: [u8; 32],
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Rv64imKernelClaimSummaryBundle {
    pub prepared_step_bindings_digest: [u8; 32],
    pub terminal: Rv64imKernelClaimTerminalBundle,
    pub digest: [u8; 32],
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Rv64imKernelClaimSummaryProofBundle {
    pub summary: Rv64imKernelClaimSummaryBundle,
    pub digest: [u8; 32],
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Rv64imKernelExportClaimProof {
    pub final_state_digest: [u8; 32],
    pub packaged: PackagedProof,
    pub digest: [u8; 32],
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Rv64imKernelClaimProofBundle {
    pub summary: Rv64imKernelClaimSummaryBundle,
    pub packaged: PackagedProof,
    pub digest: [u8; 32],
    pub claims: SimpleKernelKernelClaimBundle,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Rv64imStageClaimDigestBundle {
    pub claim_bundle_digest: [u8; 32],
    pub stage1_digest: [u8; 32],
    pub stage2_digest: [u8; 32],
    pub stage3_digest: [u8; 32],
    pub transcript_digest: [u8; 32],
    pub execution_digest: [u8; 32],
    pub digest: [u8; 32],
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Rv64imStageClaimSummaryProofBundle {
    pub summary: Rv64imStageClaimDigestBundle,
    pub digest: [u8; 32],
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Rv64imStageClaimProofBundle {
    pub summary: Rv64imStageClaimDigestBundle,
    pub packaged: PackagedProof,
    pub digest: [u8; 32],
    pub claims: SimpleKernelStageClaimBundle,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Rv64imStagePackageDigestBundle {
    pub package_bundle_digest: [u8; 32],
    pub stage1_digest: [u8; 32],
    pub stage2_digest: [u8; 32],
    pub stage3_digest: [u8; 32],
    pub digest: [u8; 32],
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Rv64imStagePackageSummaryProofBundle {
    pub summary: Rv64imStagePackageDigestBundle,
    pub digest: [u8; 32],
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Rv64imStagePackageProofBundle {
    pub summary: Rv64imStagePackageDigestBundle,
    pub digest: [u8; 32],
    pub packages: SimpleKernelStagePackageBundle,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Rv64imTraceShapeBundle {
    pub execution_row_count: u64,
    pub real_row_count: u64,
    pub effect_row_count: u64,
    pub commit_row_count: u64,
    pub digest: [u8; 32],
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Rv64imTraceProjectionBundle {
    pub manifest: Rv64imParityCaseManifest,
    pub execution_digest: [u8; 32],
    pub shape: Rv64imTraceShapeBundle,
    pub digest: [u8; 32],
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Rv64imTraceProofBundle {
    pub manifest: Rv64imParityCaseManifest,
    pub execution_digest: [u8; 32],
    pub shape: Rv64imTraceShapeBundle,
    pub digest: [u8; 32],
    pub trace: SimpleKernelTraceWitness,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Rv64imStageWitnessSummaryBundle {
    pub stage1_row_count: u64,
    pub stage2_register_read_count: u64,
    pub stage2_register_write_count: u64,
    pub stage2_ram_event_count: u64,
    pub stage2_twist_link_count: u64,
    pub stage3_continuity_count: u64,
    pub stage3_halted: bool,
    pub transcript_event_count: u64,
    pub digest: [u8; 32],
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Rv64imStageWitnessProjectionBundle {
    pub summary: Rv64imStageWitnessSummaryBundle,
    pub digest: [u8; 32],
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Rv64imStageWitnessProofBundle {
    pub summary: Rv64imStageWitnessSummaryBundle,
    pub digest: [u8; 32],
    pub stages: SimpleKernelStageWitnessBundle,
}

fn digest_to_words(digest: [u8; 32]) -> [u64; 4] {
    let mut words = [0u64; 4];
    for (dst, chunk) in words.iter_mut().zip(digest.chunks_exact(8)) {
        let mut bytes = [0u8; 8];
        bytes.copy_from_slice(chunk);
        *dst = u64::from_le_bytes(bytes);
    }
    words
}

pub(crate) fn stage_claim_bundle_words(claims: &SimpleKernelStageClaimBundle) -> Vec<u64> {
    let mut words = Vec::with_capacity(62);
    words.extend(digest_to_words(claims.digest));
    words.extend(digest_to_words(claims.execution_digest));
    words.extend(digest_to_words(claims.stage1.rows.digest));
    words.extend([
        claims.stage1.claim.row_count as u64,
        claims.stage1.claim.effect_row_count as u64,
        claims.stage1.claim.commit_row_count as u64,
        claims.stage1.claim.real_row_count as u64,
        claims.stage1.claim.preserves_x0_count as u64,
        claims.stage1.claim.mix,
    ]);
    words.extend(digest_to_words(claims.stage2.families.digest));
    words.extend([
        claims.stage2.claim.register_read_count as u64,
        claims.stage2.claim.register_write_count as u64,
        claims.stage2.claim.ram_event_count as u64,
        claims.stage2.claim.twist_link_count as u64,
        claims.stage2.claim.ram_read_count as u64,
        claims.stage2.claim.ram_write_count as u64,
        claims.stage2.claim.reg_mix,
        claims.stage2.claim.ram_mix,
    ]);
    words.extend(digest_to_words(claims.stage3.continuity.digest));
    words.extend([
        claims.stage3.claim.continuity_count as u64,
        claims.stage3.claim.final_step_count as u64,
        claims.stage3.claim.halted as u64,
        claims.stage3.claim.all_continuity_hold as u64,
        claims.stage3.claim.continuity_mix,
    ]);
    words.extend(digest_to_words(claims.transcript.commitment.digest));
    words.extend(digest_to_words(claims.transcript.claim.final_digest));
    words.extend([
        claims.transcript.claim.event_count as u64,
        claims.transcript.claim.kernel_final_mix,
    ]);
    words
}

pub(crate) fn kernel_claim_words_from_components(
    prepared_step_bindings_digest: [u8; 32],
    binding_count: u64,
    first_binding_digest: Option<[u8; 32]>,
    last_binding_digest: Option<[u8; 32]>,
    root0_digest: [u8; 32],
    execution_digest: [u8; 32],
    final_state_digest: [u8; 32],
    transcript_final_digest: [u8; 32],
    final_pc: u64,
    halted: bool,
) -> Vec<u64> {
    let mut words = Vec::with_capacity(28);
    words.extend(digest_to_words(prepared_step_bindings_digest));
    words.extend([binding_count, first_binding_digest.is_some() as u64]);
    if let Some(first) = first_binding_digest {
        words.extend(digest_to_words(first));
    } else {
        words.extend([0; 4]);
    }
    words.extend([last_binding_digest.is_some() as u64]);
    if let Some(last) = last_binding_digest {
        words.extend(digest_to_words(last));
    } else {
        words.extend([0; 4]);
    }
    words.extend(digest_to_words(root0_digest));
    words.extend(digest_to_words(execution_digest));
    words.extend(digest_to_words(final_state_digest));
    words.extend(digest_to_words(transcript_final_digest));
    words.extend([final_pc, halted as u64]);
    words
}

fn kernel_export_claim_words_from_components(final_state_digest: [u8; 32]) -> Vec<u64> {
    let mut words = Vec::with_capacity(4);
    words.extend(digest_to_words(final_state_digest));
    words
}

fn kernel_claim_bundle_words(claims: &SimpleKernelKernelClaimBundle) -> Vec<u64> {
    kernel_claim_words_from_components(
        claims.prepared_step_bindings.digest,
        claims.prepared_step_bindings.binding_count,
        claims.prepared_step_bindings.first_binding_digest,
        claims.prepared_step_bindings.last_binding_digest,
        claims.kernel.root0_digest,
        claims.kernel.execution_digest,
        claims.kernel.final_state_digest,
        claims.kernel.transcript_final_digest,
        claims.kernel.final_pc,
        claims.kernel.halted,
    )
}

fn packaged_claim_proof_digest(label: &'static [u8], summary_digest: [u8; 32], packaged: &PackagedProof) -> [u8; 32] {
    let mut tr = Poseidon2Transcript::new(label);
    tr.append_message(b"summary_digest", &summary_digest);
    tr.append_message(b"statement_digest", &packaged.statement.digest);
    tr.append_message(b"proof_digest", &packaged.proof.proof_digest);
    tr.digest32()
}

pub(crate) fn build_stage_claim_packaged_proof(
    claims: &SimpleKernelStageClaimBundle,
) -> Result<PackagedProof, SimpleKernelError> {
    build_claim_packaged_proof("rv64im/stage_claim_bundle", &stage_claim_bundle_words(claims))
}

pub(crate) fn verify_stage_claim_packaged_proof(
    claims: &SimpleKernelStageClaimBundle,
    packaged: &PackagedProof,
) -> Result<(), SimpleKernelError> {
    verify_claim_packaged_proof("rv64im/stage_claim_bundle", &stage_claim_bundle_words(claims), packaged)
}

pub(crate) fn build_stage_claim_packaged_public_step(
    claims: &SimpleKernelStageClaimBundle,
) -> Result<PublicStep, SimpleKernelError> {
    super::stage_artifacts::build_claim_packaged_public_step(
        "rv64im/stage_claim_bundle",
        &stage_claim_bundle_words(claims),
    )
}

pub(crate) fn build_kernel_claim_packaged_proof(
    claims: &SimpleKernelKernelClaimBundle,
) -> Result<PackagedProof, SimpleKernelError> {
    build_claim_packaged_proof("rv64im/kernel_claim_bundle", &kernel_claim_bundle_words(claims))
}

pub(crate) fn verify_kernel_claim_packaged_proof(
    claims: &SimpleKernelKernelClaimBundle,
    packaged: &PackagedProof,
) -> Result<(), SimpleKernelError> {
    verify_claim_packaged_proof(
        "rv64im/kernel_claim_bundle",
        &kernel_claim_bundle_words(claims),
        packaged,
    )
}

pub(crate) fn build_kernel_claim_packaged_public_step_from_compact_surfaces(
    prepared_step_bindings_digest: [u8; 32],
    binding_count: u64,
    first_binding_digest: Option<[u8; 32]>,
    last_binding_digest: Option<[u8; 32]>,
    root0_digest: [u8; 32],
    execution_digest: [u8; 32],
    final_state_digest: [u8; 32],
    transcript_final_digest: [u8; 32],
    final_pc: u64,
    halted: bool,
) -> Result<PublicStep, SimpleKernelError> {
    super::stage_artifacts::build_claim_packaged_public_step(
        "rv64im/kernel_claim_bundle",
        &kernel_claim_words_from_components(
            prepared_step_bindings_digest,
            binding_count,
            first_binding_digest,
            last_binding_digest,
            root0_digest,
            execution_digest,
            final_state_digest,
            transcript_final_digest,
            final_pc,
            halted,
        ),
    )
}

pub(crate) fn verify_kernel_export_claim_packaged_proof(
    claims: &Rv64imKernelExportClaimProof,
    packaged: &PackagedProof,
) -> Result<(), SimpleKernelError> {
    verify_claim_packaged_proof(
        "rv64im/kernel_export_claim_bundle",
        &kernel_export_claim_words_from_components(claims.final_state_digest()),
        packaged,
    )
}

impl Rv64imProofWitnessBundle {
    pub(crate) fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv64im/proof_witness_bundle");
        tr.append_message(b"rv64im/proof_witness_bundle/root_params_id", &self.root_params_id);
        tr.append_message(b"rv64im/proof_witness_bundle/trace_digest", &self.trace.digest);
        tr.append_message(b"rv64im/proof_witness_bundle/stages_digest", &self.stages.digest);
        tr.append_message(
            b"rv64im/proof_witness_bundle/stage_claims_digest",
            &self.stage_claims.digest,
        );
        tr.append_message(
            b"rv64im/proof_witness_bundle/stage_packages_digest",
            &self.stage_packages.digest,
        );
        tr.append_message(
            b"rv64im/proof_witness_bundle/kernel_opening_digest",
            &self.kernel_opening.digest,
        );
        tr.append_message(
            b"rv64im/proof_witness_bundle/kernel_claims_digest",
            &self.kernel_claims.digest,
        );
        tr.append_message(
            b"rv64im/proof_witness_bundle/root_lane_columns_digest",
            &self.root_lane_columns.digest,
        );
        tr.append_message(
            b"rv64im/proof_witness_bundle/root_lane_commitment_digest",
            &self.root_lane_commitment.digest,
        );
        tr.digest32()
    }
}

impl Rv64imKernelOpeningBindingBundle {
    pub(crate) fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv64im/kernel_opening_binding_bundle");
        tr.append_message(b"rv64im/kernel_opening_binding_bundle/claim_digest", &self.claim_digest);
        tr.append_message(
            b"rv64im/kernel_opening_binding_bundle/bindings_digest",
            &self.bindings_digest,
        );
        tr.append_message(
            b"rv64im/kernel_opening_binding_bundle/prepared_steps_digest",
            &self.prepared_steps_digest,
        );
        tr.digest32()
    }
}

impl Rv64imKernelOpeningProofBundle {
    pub(crate) fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv64im/kernel_opening_proof_bundle");
        tr.append_message(
            b"rv64im/kernel_opening_proof_bundle/opening_digest",
            &self.opening_digest,
        );
        tr.append_message(b"rv64im/kernel_opening_proof_bundle/bindings", &self.bindings.digest);
        tr.digest32()
    }

    pub fn claim_digest(&self) -> [u8; 32] {
        self.bindings.claim_digest
    }

    pub fn opening_digest(&self) -> [u8; 32] {
        self.opening_digest
    }

    pub fn bindings_digest(&self) -> [u8; 32] {
        self.bindings.bindings_digest
    }

    pub fn prepared_steps_digest(&self) -> [u8; 32] {
        self.bindings.prepared_steps_digest
    }

    pub fn summary(&self) -> Rv64imKernelOpeningSummaryBundle {
        let summary = Rv64imKernelOpeningSummaryBundle {
            opening_digest: self.opening_digest,
            bindings: self.bindings.clone(),
            digest: [0; 32],
        };
        Rv64imKernelOpeningSummaryBundle {
            digest: summary.expected_digest(),
            ..summary
        }
    }
}

impl Rv64imKernelOpeningSummaryBundle {
    pub(crate) fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv64im/kernel_opening_summary_bundle");
        tr.append_message(
            b"rv64im/kernel_opening_summary_bundle/opening_digest",
            &self.opening_digest,
        );
        tr.append_message(
            b"rv64im/kernel_opening_summary_bundle/bindings_digest",
            &self.bindings.digest,
        );
        tr.digest32()
    }
}

impl Rv64imKernelClaimTerminalBundle {
    pub(crate) fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv64im/kernel_claim_terminal_bundle");
        tr.append_message(b"rv64im/kernel_claim_terminal_bundle/root0_digest", &self.root0_digest);
        tr.append_message(
            b"rv64im/kernel_claim_terminal_bundle/execution_digest",
            &self.execution_digest,
        );
        tr.append_message(
            b"rv64im/kernel_claim_terminal_bundle/final_state_digest",
            &self.final_state_digest,
        );
        tr.append_message(
            b"rv64im/kernel_claim_terminal_bundle/transcript_final_digest",
            &self.transcript_final_digest,
        );
        tr.append_u64s(
            b"rv64im/kernel_claim_terminal_bundle/meta",
            &[self.final_pc, self.halted as u64],
        );
        tr.digest32()
    }
}

impl Rv64imKernelClaimSummaryBundle {
    pub(crate) fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv64im/kernel_claim_summary_bundle");
        tr.append_message(
            b"rv64im/kernel_claim_summary_bundle/prepared_step_bindings_digest",
            &self.prepared_step_bindings_digest,
        );
        tr.append_message(
            b"rv64im/kernel_claim_summary_bundle/terminal_digest",
            &self.terminal.digest,
        );
        tr.digest32()
    }
}

impl Rv64imKernelClaimSummaryProofBundle {
    pub(crate) fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv64im/kernel_claim_summary_proof_bundle");
        tr.append_message(
            b"rv64im/kernel_claim_summary_proof_bundle/summary",
            &self.summary.digest,
        );
        tr.digest32()
    }

    pub fn prepared_step_bindings_digest(&self) -> [u8; 32] {
        self.summary.prepared_step_bindings_digest
    }

    pub fn root0_digest(&self) -> [u8; 32] {
        self.summary.terminal.root0_digest
    }

    pub fn final_state_digest(&self) -> [u8; 32] {
        self.summary.terminal.final_state_digest
    }

    pub fn execution_digest(&self) -> [u8; 32] {
        self.summary.terminal.execution_digest
    }

    pub fn transcript_final_digest(&self) -> [u8; 32] {
        self.summary.terminal.transcript_final_digest
    }

    pub fn final_pc(&self) -> u64 {
        self.summary.terminal.final_pc
    }

    pub fn halted(&self) -> bool {
        self.summary.terminal.halted
    }
}

impl Rv64imKernelClaimProofBundle {
    pub(crate) fn expected_digest(&self) -> [u8; 32] {
        packaged_claim_proof_digest(
            b"neo.fold.next/rv64im/kernel_claim_proof_bundle",
            self.summary.digest,
            &self.packaged,
        )
    }

    pub fn prepared_step_bindings_digest(&self) -> [u8; 32] {
        self.summary.prepared_step_bindings_digest
    }

    pub fn root0_digest(&self) -> [u8; 32] {
        self.summary.terminal.root0_digest
    }

    pub fn final_state_digest(&self) -> [u8; 32] {
        self.summary.terminal.final_state_digest
    }

    pub fn execution_digest(&self) -> [u8; 32] {
        self.summary.terminal.execution_digest
    }

    pub fn transcript_final_digest(&self) -> [u8; 32] {
        self.summary.terminal.transcript_final_digest
    }

    pub fn final_pc(&self) -> u64 {
        self.summary.terminal.final_pc
    }

    pub fn halted(&self) -> bool {
        self.summary.terminal.halted
    }

    pub fn terminal_digest(&self) -> [u8; 32] {
        self.summary.terminal.digest
    }

    pub fn summary_bundle(&self) -> Rv64imKernelClaimSummaryProofBundle {
        let summary = Rv64imKernelClaimSummaryProofBundle {
            summary: self.summary.clone(),
            digest: [0; 32],
        };
        Rv64imKernelClaimSummaryProofBundle {
            digest: summary.expected_digest(),
            ..summary
        }
    }
}

impl Rv64imKernelExportClaimProof {
    pub(crate) fn summary_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv64im/kernel_export_claim_terminal");
        tr.append_message(
            b"rv64im/kernel_export_claim_terminal/final_state_digest",
            &self.final_state_digest,
        );
        tr.digest32()
    }

    pub(crate) fn expected_digest(&self) -> [u8; 32] {
        packaged_claim_proof_digest(
            b"neo.fold.next/rv64im/kernel_export_claim_proof",
            self.summary_digest(),
            &self.packaged,
        )
    }

    pub fn final_state_digest(&self) -> [u8; 32] {
        self.final_state_digest
    }
}

pub(crate) fn kernel_export_claim_proof_from_bundle(
    bundle: &Rv64imKernelClaimProofBundle,
) -> Result<Rv64imKernelExportClaimProof, SimpleKernelError> {
    let packaged = build_claim_packaged_proof(
        "rv64im/kernel_export_claim_bundle",
        &kernel_export_claim_words_from_components(bundle.final_state_digest()),
    )?;
    let proof = Rv64imKernelExportClaimProof {
        final_state_digest: bundle.final_state_digest(),
        packaged,
        digest: [0; 32],
    };
    Ok(Rv64imKernelExportClaimProof {
        digest: proof.expected_digest(),
        ..proof
    })
}

impl Rv64imStageClaimDigestBundle {
    pub(crate) fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv64im/stage_claim_digest_bundle");
        tr.append_message(
            b"rv64im/stage_claim_digest_bundle/claim_bundle_digest",
            &self.claim_bundle_digest,
        );
        tr.append_message(b"rv64im/stage_claim_digest_bundle/stage1_digest", &self.stage1_digest);
        tr.append_message(b"rv64im/stage_claim_digest_bundle/stage2_digest", &self.stage2_digest);
        tr.append_message(b"rv64im/stage_claim_digest_bundle/stage3_digest", &self.stage3_digest);
        tr.append_message(
            b"rv64im/stage_claim_digest_bundle/transcript_digest",
            &self.transcript_digest,
        );
        tr.append_message(
            b"rv64im/stage_claim_digest_bundle/execution_digest",
            &self.execution_digest,
        );
        tr.digest32()
    }
}

impl Rv64imStageClaimSummaryProofBundle {
    pub(crate) fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv64im/stage_claim_summary_proof_bundle");
        tr.append_message(b"rv64im/stage_claim_summary_proof_bundle/summary", &self.summary.digest);
        tr.digest32()
    }

    pub fn stage1_digest(&self) -> [u8; 32] {
        self.summary.stage1_digest
    }

    pub fn claim_bundle_digest(&self) -> [u8; 32] {
        self.summary.claim_bundle_digest
    }

    pub fn stage2_digest(&self) -> [u8; 32] {
        self.summary.stage2_digest
    }

    pub fn stage3_digest(&self) -> [u8; 32] {
        self.summary.stage3_digest
    }
}

impl Rv64imStageClaimProofBundle {
    pub(crate) fn expected_digest(&self) -> [u8; 32] {
        packaged_claim_proof_digest(
            b"neo.fold.next/rv64im/stage_claim_proof_bundle",
            self.summary.digest,
            &self.packaged,
        )
    }

    pub fn stage1_digest(&self) -> [u8; 32] {
        self.summary.stage1_digest
    }

    pub fn claim_bundle_digest(&self) -> [u8; 32] {
        self.summary.claim_bundle_digest
    }

    pub fn stage2_digest(&self) -> [u8; 32] {
        self.summary.stage2_digest
    }

    pub fn stage3_digest(&self) -> [u8; 32] {
        self.summary.stage3_digest
    }

    pub fn summary_bundle(&self) -> Rv64imStageClaimSummaryProofBundle {
        let summary = Rv64imStageClaimSummaryProofBundle {
            summary: self.summary.clone(),
            digest: [0; 32],
        };
        Rv64imStageClaimSummaryProofBundle {
            digest: summary.expected_digest(),
            ..summary
        }
    }
}

impl Rv64imStagePackageDigestBundle {
    pub(crate) fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv64im/stage_package_digest_bundle");
        tr.append_message(
            b"rv64im/stage_package_digest_bundle/package_bundle_digest",
            &self.package_bundle_digest,
        );
        tr.append_message(b"rv64im/stage_package_digest_bundle/stage1_digest", &self.stage1_digest);
        tr.append_message(b"rv64im/stage_package_digest_bundle/stage2_digest", &self.stage2_digest);
        tr.append_message(b"rv64im/stage_package_digest_bundle/stage3_digest", &self.stage3_digest);
        tr.digest32()
    }
}

impl Rv64imStagePackageSummaryProofBundle {
    pub(crate) fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv64im/stage_package_summary_proof_bundle");
        tr.append_message(
            b"rv64im/stage_package_summary_proof_bundle/summary",
            &self.summary.digest,
        );
        tr.digest32()
    }

    pub fn stage1_digest(&self) -> [u8; 32] {
        self.summary.stage1_digest
    }

    pub fn package_bundle_digest(&self) -> [u8; 32] {
        self.summary.package_bundle_digest
    }

    pub fn stage2_digest(&self) -> [u8; 32] {
        self.summary.stage2_digest
    }

    pub fn stage3_digest(&self) -> [u8; 32] {
        self.summary.stage3_digest
    }
}

impl Rv64imStagePackageProofBundle {
    pub(crate) fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv64im/stage_package_proof_bundle");
        tr.append_message(b"rv64im/stage_package_proof_bundle/summary", &self.summary.digest);
        tr.digest32()
    }

    pub fn stage1_digest(&self) -> [u8; 32] {
        self.summary.stage1_digest
    }

    pub fn package_bundle_digest(&self) -> [u8; 32] {
        self.summary.package_bundle_digest
    }

    pub fn stage2_digest(&self) -> [u8; 32] {
        self.summary.stage2_digest
    }

    pub fn stage3_digest(&self) -> [u8; 32] {
        self.summary.stage3_digest
    }

    pub fn summary_bundle(&self) -> Rv64imStagePackageSummaryProofBundle {
        let summary = Rv64imStagePackageSummaryProofBundle {
            summary: self.summary.clone(),
            digest: [0; 32],
        };
        Rv64imStagePackageSummaryProofBundle {
            digest: summary.expected_digest(),
            ..summary
        }
    }
}

fn family_word(family: Rv64FamilyTag) -> u64 {
    match family {
        Rv64FamilyTag::NativeAlu => 0,
        Rv64FamilyTag::AlignedMemory => 1,
        Rv64FamilyTag::ControlFlow => 2,
        Rv64FamilyTag::NarrowMemory => 3,
        Rv64FamilyTag::Multiply => 4,
        Rv64FamilyTag::UnsignedDivRem => 5,
        Rv64FamilyTag::SignedDivRem => 6,
    }
}

impl Rv64imTraceShapeBundle {
    pub(crate) fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv64im/trace_shape_bundle");
        tr.append_u64s(
            b"rv64im/trace_shape_bundle/meta",
            &[
                self.execution_row_count,
                self.real_row_count,
                self.effect_row_count,
                self.commit_row_count,
            ],
        );
        tr.digest32()
    }
}

impl Rv64imTraceProjectionBundle {
    pub(crate) fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv64im/trace_summary_proof_bundle");
        tr.append_message(b"rv64im/trace_summary_proof_bundle/name", self.manifest.name.as_bytes());
        tr.append_message(
            b"rv64im/trace_summary_proof_bundle/fixture_id",
            self.manifest.fixture_id.as_bytes(),
        );
        tr.append_u64s(
            b"rv64im/trace_summary_proof_bundle/meta",
            &[self.manifest.protocol_version_id, self.manifest.lowering_version_id],
        );
        tr.append_u64s(
            b"rv64im/trace_summary_proof_bundle/family_tag_len",
            &[self.manifest.family_tags.len() as u64],
        );
        for family in &self.manifest.family_tags {
            tr.append_u64s(b"rv64im/trace_summary_proof_bundle/family_tag", &[family_word(*family)]);
        }
        tr.append_message(
            b"rv64im/trace_summary_proof_bundle/execution_digest",
            &self.execution_digest,
        );
        tr.append_message(b"rv64im/trace_summary_proof_bundle/shape_digest", &self.shape.digest);
        tr.digest32()
    }

    pub fn execution_row_count(&self) -> u64 {
        self.shape.execution_row_count
    }

    pub fn execution_digest(&self) -> [u8; 32] {
        self.execution_digest
    }
}

impl Rv64imTraceProofBundle {
    pub(crate) fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv64im/trace_proof_bundle");
        tr.append_message(b"rv64im/trace_proof_bundle/name", self.manifest.name.as_bytes());
        tr.append_message(
            b"rv64im/trace_proof_bundle/fixture_id",
            self.manifest.fixture_id.as_bytes(),
        );
        tr.append_u64s(
            b"rv64im/trace_proof_bundle/meta",
            &[self.manifest.protocol_version_id, self.manifest.lowering_version_id],
        );
        tr.append_u64s(
            b"rv64im/trace_proof_bundle/family_tag_len",
            &[self.manifest.family_tags.len() as u64],
        );
        for family in &self.manifest.family_tags {
            tr.append_u64s(b"rv64im/trace_proof_bundle/family_tag", &[family_word(*family)]);
        }
        tr.append_message(b"rv64im/trace_proof_bundle/execution_digest", &self.execution_digest);
        tr.append_message(b"rv64im/trace_proof_bundle/shape_digest", &self.shape.digest);
        tr.digest32()
    }

    pub fn execution_row_count(&self) -> u64 {
        self.shape.execution_row_count
    }

    pub fn execution_digest(&self) -> [u8; 32] {
        self.execution_digest
    }

    pub fn real_row_count(&self) -> u64 {
        self.shape.real_row_count
    }

    pub fn effect_row_count(&self) -> u64 {
        self.shape.effect_row_count
    }

    pub fn commit_row_count(&self) -> u64 {
        self.shape.commit_row_count
    }

    pub fn projection(&self) -> Rv64imTraceProjectionBundle {
        let summary = Rv64imTraceProjectionBundle {
            manifest: self.manifest.clone(),
            execution_digest: self.execution_digest,
            shape: self.shape.clone(),
            digest: [0; 32],
        };
        Rv64imTraceProjectionBundle {
            digest: summary.expected_digest(),
            ..summary
        }
    }
}

impl Rv64imStageWitnessSummaryBundle {
    pub(crate) fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv64im/stage_witness_summary_bundle");
        tr.append_u64s(
            b"rv64im/stage_witness_summary_bundle/meta",
            &[
                self.stage1_row_count,
                self.stage2_register_read_count,
                self.stage2_register_write_count,
                self.stage2_ram_event_count,
                self.stage2_twist_link_count,
                self.stage3_continuity_count,
                self.stage3_halted as u64,
                self.transcript_event_count,
            ],
        );
        tr.digest32()
    }
}

impl Rv64imStageWitnessProjectionBundle {
    pub(crate) fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv64im/stage_witness_summary_proof_bundle");
        tr.append_message(
            b"rv64im/stage_witness_summary_proof_bundle/summary",
            &self.summary.digest,
        );
        tr.digest32()
    }

    pub fn stage1_row_count(&self) -> u64 {
        self.summary.stage1_row_count
    }

    pub fn stage3_continuity_count(&self) -> u64 {
        self.summary.stage3_continuity_count
    }

    pub fn stage3_halted(&self) -> bool {
        self.summary.stage3_halted
    }
}

impl Rv64imStageWitnessProofBundle {
    pub(crate) fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv64im/stage_witness_proof_bundle");
        tr.append_message(b"rv64im/stage_witness_proof_bundle/summary", &self.summary.digest);
        tr.digest32()
    }

    pub fn stage1_row_count(&self) -> u64 {
        self.summary.stage1_row_count
    }

    pub fn stage3_continuity_count(&self) -> u64 {
        self.summary.stage3_continuity_count
    }

    pub fn stage3_halted(&self) -> bool {
        self.summary.stage3_halted
    }

    pub fn projection_bundle(&self) -> Rv64imStageWitnessProjectionBundle {
        let summary = Rv64imStageWitnessProjectionBundle {
            summary: self.summary.clone(),
            digest: [0; 32],
        };
        Rv64imStageWitnessProjectionBundle {
            digest: summary.expected_digest(),
            ..summary
        }
    }
}

pub(crate) fn trace_proof_bundle_from_trace(
    trace: &SimpleKernelTraceWitness,
    execution_digest: [u8; 32],
) -> Rv64imTraceProofBundle {
    let summary = trace_projection_bundle_from_rows(&trace.manifest, &trace.execution_rows, execution_digest);
    let shape = Rv64imTraceShapeBundle {
        ..summary.shape.clone()
    };
    let bundle = Rv64imTraceProofBundle {
        manifest: trace.manifest.clone(),
        execution_digest,
        shape,
        digest: [0; 32],
        trace: trace.clone(),
    };
    Rv64imTraceProofBundle {
        digest: bundle.expected_digest(),
        ..bundle
    }
}

pub(crate) fn trace_projection_bundle_from_rows(
    manifest: &Rv64imParityCaseManifest,
    rows: &[Rv64ExpandedRow],
    execution_digest: [u8; 32],
) -> Rv64imTraceProjectionBundle {
    let shape = Rv64imTraceShapeBundle {
        execution_row_count: rows.len() as u64,
        real_row_count: rows.iter().filter(|row| row.is_real).count() as u64,
        effect_row_count: rows.iter().filter(|row| row.is_effect_row).count() as u64,
        commit_row_count: rows.iter().filter(|row| row.is_commit_row).count() as u64,
        digest: [0; 32],
    };
    let shape = Rv64imTraceShapeBundle {
        digest: shape.expected_digest(),
        ..shape
    };
    let bundle = Rv64imTraceProjectionBundle {
        manifest: manifest.clone(),
        execution_digest,
        shape,
        digest: [0; 32],
    };
    Rv64imTraceProjectionBundle {
        digest: bundle.expected_digest(),
        ..bundle
    }
}

pub(crate) fn stage_witness_proof_bundle_from_stages(
    stages: &SimpleKernelStageWitnessBundle,
) -> Rv64imStageWitnessProofBundle {
    let summary = stage_witness_projection_bundle_from_summaries(
        &stages.stage1,
        &stages.stage2,
        &stages.stage3,
        &stages.transcript,
    )
    .summary;
    let bundle = Rv64imStageWitnessProofBundle {
        summary,
        digest: [0; 32],
        stages: stages.clone(),
    };
    Rv64imStageWitnessProofBundle {
        digest: bundle.expected_digest(),
        ..bundle
    }
}

pub(crate) fn stage_witness_projection_bundle_from_summaries(
    stage1: &Stage1Summary,
    stage2: &Stage2Summary,
    stage3: &Stage3Summary,
    transcript: &TranscriptRecord,
) -> Rv64imStageWitnessProjectionBundle {
    let summary = Rv64imStageWitnessSummaryBundle {
        stage1_row_count: stage1.rows.len() as u64,
        stage2_register_read_count: stage2.register_reads.len() as u64,
        stage2_register_write_count: stage2.register_writes.len() as u64,
        stage2_ram_event_count: stage2.ram_events.len() as u64,
        stage2_twist_link_count: stage2.twist_links.len() as u64,
        stage3_continuity_count: stage3.continuity.len() as u64,
        stage3_halted: stage3.halted,
        transcript_event_count: transcript.events.len() as u64,
        digest: [0; 32],
    };
    let summary = Rv64imStageWitnessSummaryBundle {
        digest: summary.expected_digest(),
        ..summary
    };
    let bundle = Rv64imStageWitnessProjectionBundle {
        summary,
        digest: [0; 32],
    };
    Rv64imStageWitnessProjectionBundle {
        digest: bundle.expected_digest(),
        ..bundle
    }
}

pub(crate) fn stage_claim_summary_bundle_from_claims(
    claims: &SimpleKernelStageClaimBundle,
) -> Rv64imStageClaimSummaryProofBundle {
    let summary = Rv64imStageClaimDigestBundle {
        claim_bundle_digest: claims.digest,
        stage1_digest: claims.stage1.rows.digest,
        stage2_digest: claims.stage2.families.digest,
        stage3_digest: claims.stage3.continuity.digest,
        transcript_digest: claims.transcript.commitment.digest,
        execution_digest: claims.execution_digest,
        digest: [0; 32],
    };
    let summary = Rv64imStageClaimDigestBundle {
        digest: summary.expected_digest(),
        ..summary
    };
    let bundle = Rv64imStageClaimSummaryProofBundle {
        summary,
        digest: [0; 32],
    };
    Rv64imStageClaimSummaryProofBundle {
        digest: bundle.expected_digest(),
        ..bundle
    }
}

pub(crate) fn stage_claim_proof_bundle_from_claims(
    claims: &SimpleKernelStageClaimBundle,
) -> Result<Rv64imStageClaimProofBundle, SimpleKernelError> {
    let summary = stage_claim_summary_bundle_from_claims(claims).summary;
    let packaged = build_stage_claim_packaged_proof(claims)?;
    let bundle = Rv64imStageClaimProofBundle {
        summary,
        packaged,
        digest: [0; 32],
        claims: claims.clone(),
    };
    Ok(Rv64imStageClaimProofBundle {
        digest: bundle.expected_digest(),
        ..bundle
    })
}

pub(crate) fn stage_package_proof_bundle_from_packages(
    packages: &SimpleKernelStagePackageBundle,
) -> Rv64imStagePackageProofBundle {
    let summary = stage_package_summary_bundle_from_packages(packages);
    let bundle = Rv64imStagePackageProofBundle {
        summary,
        digest: [0; 32],
        packages: packages.clone(),
    };
    Rv64imStagePackageProofBundle {
        digest: bundle.expected_digest(),
        ..bundle
    }
}

pub(crate) fn stage_package_summary_bundle_from_packages(
    packages: &SimpleKernelStagePackageBundle,
) -> Rv64imStagePackageDigestBundle {
    let summary = Rv64imStagePackageDigestBundle {
        package_bundle_digest: packages.digest,
        stage1_digest: packages.stage1.digest,
        stage2_digest: packages.stage2.digest,
        stage3_digest: packages.stage3.digest,
        digest: [0; 32],
    };
    Rv64imStagePackageDigestBundle {
        digest: summary.expected_digest(),
        ..summary
    }
}

pub(crate) fn kernel_opening_proof_bundle_from_opening(
    opening: &SimpleKernelOpeningBundle,
) -> Rv64imKernelOpeningProofBundle {
    let bindings = kernel_opening_binding_bundle_from_opening(opening);
    let bundle = Rv64imKernelOpeningProofBundle {
        opening_digest: opening.digest,
        bindings,
        digest: [0; 32],
        opening: opening.clone(),
    };
    Rv64imKernelOpeningProofBundle {
        digest: bundle.expected_digest(),
        ..bundle
    }
}

pub(crate) fn kernel_opening_binding_bundle_from_opening(
    opening: &SimpleKernelOpeningBundle,
) -> Rv64imKernelOpeningBindingBundle {
    let bindings = Rv64imKernelOpeningBindingBundle {
        claim_digest: opening.claim.digest,
        bindings_digest: opening.bindings.digest,
        prepared_steps_digest: opening.prepared_steps.digest,
        digest: [0; 32],
    };
    Rv64imKernelOpeningBindingBundle {
        digest: bindings.expected_digest(),
        ..bindings
    }
}

pub(crate) fn kernel_claim_summary_bundle_from_claims(
    claims: &SimpleKernelKernelClaimBundle,
) -> Rv64imKernelClaimSummaryProofBundle {
    let summary = &claims.kernel;
    let terminal = Rv64imKernelClaimTerminalBundle {
        root0_digest: summary.root0_digest,
        execution_digest: summary.execution_digest,
        final_state_digest: summary.final_state_digest,
        transcript_final_digest: summary.transcript_final_digest,
        final_pc: summary.final_pc,
        halted: summary.halted,
        digest: [0; 32],
    };
    let terminal = Rv64imKernelClaimTerminalBundle {
        digest: terminal.expected_digest(),
        ..terminal
    };
    let summary = Rv64imKernelClaimSummaryBundle {
        prepared_step_bindings_digest: claims.prepared_step_bindings.digest,
        terminal,
        digest: [0; 32],
    };
    let summary = Rv64imKernelClaimSummaryBundle {
        digest: summary.expected_digest(),
        ..summary
    };
    let bundle = Rv64imKernelClaimSummaryProofBundle {
        summary,
        digest: [0; 32],
    };
    Rv64imKernelClaimSummaryProofBundle {
        digest: bundle.expected_digest(),
        ..bundle
    }
}

pub(crate) fn kernel_claim_proof_bundle_from_claims(
    claims: &SimpleKernelKernelClaimBundle,
) -> Result<Rv64imKernelClaimProofBundle, SimpleKernelError> {
    let summary = kernel_claim_summary_bundle_from_claims(claims).summary;
    let packaged = build_kernel_claim_packaged_proof(claims)?;
    let bundle = Rv64imKernelClaimProofBundle {
        summary,
        packaged,
        digest: [0; 32],
        claims: claims.clone(),
    };
    Ok(Rv64imKernelClaimProofBundle {
        digest: bundle.expected_digest(),
        ..bundle
    })
}

pub(crate) fn proof_witness_bundle_from_public_kernel_and_trace_stages(
    public: &PublicSimpleKernelOutput,
    trace_witness: &SimpleKernelTraceWitness,
    stage_witness: &SimpleKernelStageWitnessBundle,
) -> Result<Rv64imProofWitnessBundle, SimpleKernelError> {
    let trace = trace_proof_bundle_from_trace(trace_witness, public.kernel_claims.kernel.execution_digest);
    let stages = stage_witness_proof_bundle_from_stages(stage_witness);
    let stage_claims = stage_claim_proof_bundle_from_claims(&public.stage_claims)?;
    let stage_packages = stage_package_proof_bundle_from_packages(&public.stage_packages);
    let kernel_opening = kernel_opening_proof_bundle_from_opening(&public.kernel_opening);
    let kernel_claims = kernel_claim_proof_bundle_from_claims(&public.kernel_claims)?;
    let bundle = Rv64imProofWitnessBundle {
        root_params_id: rv64im_simple_root_context_id(),
        trace,
        stages,
        stage_claims,
        stage_packages,
        kernel_opening,
        kernel_claims,
        root_lane_columns: public.root_lane_columns.clone(),
        root_lane_commitment: public.root_lane_commitment.clone(),
        digest: [0; 32],
    };
    Ok(Rv64imProofWitnessBundle {
        digest: bundle.expected_digest(),
        ..bundle
    })
}

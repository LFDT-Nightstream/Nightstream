//! Test-only decider row-family isolation helpers.
//!
//! This module is public only so integration tests can hit individual
//! in-circuit row families without disabling the decider preflight.

use super::{emit_terminal_fold, Preprocessing};
use crate::engine::r1cs_circuit::field_ext::KVar;
use crate::engine::r1cs_circuit::{Lc, R1csBuilder, Var};
use crate::paper::construction2::EncInst;
use crate::paper::decider::PublicImage;
use crate::paper::decider_ce_relation::{
    alloc_final_witness, enforce_final_ce_relations, enforce_final_dec_children_relations,
    enforce_raw_old_block_projection,
};
use crate::paper::digest::{digest32_as_fields, initial_boundary_digest, public_trace_seed_digest, AccumulatorHandle};
use crate::paper::f_prime::r1cs::{FPrimeStateWires, FPrimeStepOutput, F_PRIME_ENC_INST_BITS};
use crate::paper::reductions::pi_ccs_split_nc_circuit::SplitNcPiCcsOutputWires;
use crate::paper::reductions::pi_dec_circuit::{alloc_ce_claim, alloc_dec_child_claim};
use crate::paper::relations::product_commitment_circuit::alloc_adv;
use crate::paper::relations::{CcsClaim, CeClaim, WitnessMat};
use crate::paper::terminal_ce::circuit::{
    enforce_public_from_children, enforce_verify_from_children, TerminalCeVerifierContext,
};
use crate::paper::terminal_ce::{TerminalCeProof, TerminalCePublic, TerminalCeVerifyError};
use crate::paper::{construction2::RunningInstance, nifs::NifsProof};
use neo_ajtai::AjtaiSModule;
use neo_ccs::traits::SModuleHomomorphism;
use neo_math::{KExtensions, F, K};
use p3_field::PrimeCharacteristicRing;

pub struct CeRelationIsolationOutput {
    pub builder: R1csBuilder,
    pub fold_digest_fields: Vec<[Var; 4]>,
}

pub struct TerminalPendingProjectionIsolationOutput {
    pub builder: R1csBuilder,
    pub raw_witness_probe: usize,
    pub parent_c0_probe: usize,
    pub final_scale_c0_probe: Option<usize>,
}

/// Capture the exact fixed-profile terminal projection placement after the
/// normal last-step and terminal-fold call schedule, stopping before the
/// 24.2M-row projection and terminal Ajtai/low-norm loops.
pub fn capture_last_step_terminal_projection_placement(
    prep: &Preprocessing,
    audit: &crate::lifecycle::UncompressedAudit,
) -> Result<crate::engine::r1cs_circuit::TerminalPendingProjectionAudit, String> {
    super::terminal::capture_last_step_terminal_projection_placement(prep, audit).map_err(|error| error.to_string())
}

/// Capture the selected last-step/terminal prefix together with the placement
/// where the omitted large raw-old-block program would begin.
///
/// This is diagnostic-only: it returns rows already emitted by the ordinary
/// terminal path and does not add, remove, or replace any constraint.
pub fn capture_last_step_terminal_prefix(
    prep: &Preprocessing,
    audit: &crate::lifecycle::UncompressedAudit,
) -> Result<
    (
        super::LastStepTerminalSynthesis,
        crate::engine::r1cs_circuit::TerminalPendingProjectionAudit,
    ),
    String,
> {
    let (synthesis, placement) =
        super::terminal::capture_last_step_terminal_prefix(prep, audit).map_err(|error| error.to_string())?;
    let placement = placement.ok_or_else(|| "selected terminal prefix omitted its pending projection".to_string())?;
    Ok((synthesis, placement))
}

/// Emit the direct terminal projection over the same raw witness allocations
/// that the terminal CE relation uses for Ajtai openings.
pub fn enforce_terminal_raw_old_block_projection_against(
    logical_columns: usize,
    old_block: &[K; neo_reductions::block_projection::BLOCK_PROJECTION_POINT_LEN],
    parent_y_zcol: &[K; neo_math::D],
    witnesses: &[WitnessMat],
    radix: u32,
) -> Result<TerminalPendingProjectionIsolationOutput, String> {
    let mut builder = R1csBuilder::new();
    let old_block = alloc_k_vec(&mut builder, old_block);
    let parent = parent_y_zcol
        .iter()
        .map(|value| {
            let [c0, c1] = value.as_coeffs();
            KVar::alloc(&mut builder, c0, c1)
        })
        .collect::<Vec<_>>();
    let witness_wires = witnesses
        .iter()
        .enumerate()
        .map(|(index, witness)| {
            alloc_final_witness(&mut builder, witness, logical_columns).map_err(|error| {
                format!(
                    "raw witness {index} {}: expected {}, got {}",
                    error.what(),
                    error.expected(),
                    error.got()
                )
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let raw_witness_probe = witness_wires
        .first()
        .and_then(|witness| witness.values.first())
        .ok_or_else(|| "terminal pending isolation requires at least one raw witness entry".to_string())?
        .col();
    let parent_c0_probe = parent[0].c0.col();
    enforce_raw_old_block_projection(
        &mut builder,
        logical_columns,
        &old_block,
        &parent,
        &witness_wires,
        radix,
    )
    .map_err(|error| {
        format!(
            "raw old-block projection {}: expected {}, got {}",
            error.what(),
            error.expected(),
            error.got()
        )
    })?;
    let final_scale_c0_probe = builder
        .terminal_pending_projection_audits()
        .last()
        .filter(|audit| audit.plan.factor_final_round())
        .map(|audit| audit.final_scale_first_allocated_column + 3);
    Ok(TerminalPendingProjectionIsolationOutput {
        builder,
        raw_witness_probe,
        parent_c0_probe,
        final_scale_c0_probe,
    })
}

/// Emit the bounded direct projection and then open the exact same ordered
/// `FinalWitnessWires` allocations through an independently supplied Ajtai
/// map.  Artifact tests use the projection audit range only; the following
/// commitment rows certify the allocation join without turning the
/// commitment into projection authority.
pub fn enforce_terminal_raw_old_block_projection_with_ajtai_against(
    log: &AjtaiSModule,
    logical_columns: usize,
    old_block: &[K; neo_reductions::block_projection::BLOCK_PROJECTION_POINT_LEN],
    parent_y_zcol: &[K; neo_math::D],
    witnesses: &[WitnessMat],
    radix: u32,
) -> Result<R1csBuilder, String> {
    let mut builder = R1csBuilder::new();
    let old_block = alloc_k_vec(&mut builder, old_block);
    let parent = alloc_k_vec(&mut builder, parent_y_zcol);
    let witness_wires = witnesses
        .iter()
        .enumerate()
        .map(|(index, witness)| {
            alloc_final_witness(&mut builder, witness, logical_columns).map_err(|error| {
                format!(
                    "raw witness {index} {}: expected {}, got {}",
                    error.what(),
                    error.expected(),
                    error.got()
                )
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    enforce_raw_old_block_projection(
        &mut builder,
        logical_columns,
        &old_block,
        &parent,
        &witness_wires,
        radix,
    )
    .map_err(|error| {
        format!(
            "raw old-block projection {}: expected {}, got {}",
            error.what(),
            error.expected(),
            error.got()
        )
    })?;
    for (index, (witness, wires)) in witnesses.iter().zip(&witness_wires).enumerate() {
        let commitment = log.commit(witness);
        let commitment_wires = commitment
            .data
            .iter()
            .copied()
            .map(|value| builder.alloc(value))
            .collect::<Vec<_>>();
        crate::paper::decider_ce_relation::enforce_ajtai_opening(
            &mut builder,
            log,
            wires,
            &commitment_wires,
            commitment.d,
            commitment.kappa,
        )
        .map_err(|error| format!("Ajtai opening {index}: {error:?}"))?;
    }
    let child_witness_first_columns = witness_wires
        .iter()
        .map(|witness| {
            witness
                .values
                .first()
                .expect("validated raw witness has entries")
                .col()
        })
        .collect();
    builder.record_terminal_pending_projection_ajtai_join(
        crate::engine::r1cs_circuit::RAW_OLD_BLOCK_PENDING_JOIN_ID,
        child_witness_first_columns,
    );
    Ok(builder)
}

/// Emit only terminal CE-relation rows for a hand-crafted pair.
pub fn enforce_ce_relations_against(
    prep: &Preprocessing,
    claim: &CeClaim,
    witness: &WitnessMat,
) -> Result<R1csBuilder, String> {
    Ok(enforce_ce_relations_with_wires_against(prep, claim, witness)?.builder)
}

/// Emit only terminal CE-relation rows and return the non-CE metadata wires
/// that this row family deliberately does not consume.
pub fn enforce_ce_relations_with_wires_against(
    prep: &Preprocessing,
    claim: &CeClaim,
    witness: &WitnessMat,
) -> Result<CeRelationIsolationOutput, String> {
    let mut builder = R1csBuilder::new();
    let claim_wires = alloc_ce_claim(&mut builder, claim);
    let fold_digest_fields = vec![claim_wires.fold_digest_fields];
    let claim_wires_slice = [claim_wires];
    let witnesses_slice = [witness.clone()];
    enforce_final_ce_relations(&mut builder, prep, &claim_wires_slice, &witnesses_slice).map_err(|e| e.to_string())?;
    Ok(CeRelationIsolationOutput {
        builder,
        fold_digest_fields,
    })
}

/// Emit terminal CE-relation rows for caller-supplied claim/witness lists.
pub fn enforce_ce_relations_many_against(
    prep: &Preprocessing,
    claims: &[CeClaim],
    witnesses: &[WitnessMat],
) -> Result<R1csBuilder, String> {
    let mut builder = R1csBuilder::new();
    let claim_wires = claims
        .iter()
        .map(|claim| alloc_ce_claim(&mut builder, claim))
        .collect::<Vec<_>>();
    enforce_final_ce_relations(&mut builder, prep, &claim_wires, witnesses).map_err(|e| e.to_string())?;
    Ok(builder)
}

/// Emit the direct pending projection and all terminal CE rows over one shared
/// ordered raw-witness allocation family. Claims enter through the same strict
/// Π_DEC child carrier as production, which deliberately omits `y_zcol`.
pub fn enforce_ce_relations_many_with_raw_pending_against(
    prep: &Preprocessing,
    claims: &[CeClaim],
    witnesses: &[WitnessMat],
    old_block: &[K; neo_reductions::block_projection::BLOCK_PROJECTION_POINT_LEN],
    parent_y_zcol: &[K; neo_math::D],
) -> Result<R1csBuilder, String> {
    let mut builder = R1csBuilder::new();
    let claim_wires = claims
        .iter()
        .map(|claim| alloc_dec_child_claim(&mut builder, claim))
        .collect::<Vec<_>>();
    let pending = crate::paper::reductions::pi_ccs_split_nc_circuit::PendingProjectionWires {
        old_block: alloc_k_vec(&mut builder, old_block),
        parent_y_zcol: alloc_k_vec(&mut builder, parent_y_zcol),
    };
    crate::paper::decider_ce_relation::enforce_final_ce_relations_with_pending_projection(
        &mut builder,
        prep,
        &claim_wires,
        witnesses,
        &pending,
    )
    .map_err(|error| error.to_string())?;
    Ok(builder)
}

pub struct TerminalCePublicIsolationOutput {
    pub builder: R1csBuilder,
    pub relation_digest: [F; 4],
    pub structure_digest: [F; 4],
    pub params_digest: [F; 4],
    pub terminal_children_digest: [F; 4],
    pub public_digest: [F; 4],
    pub claim_count: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct TerminalCePublicTamperProbes {
    pub c_data: usize,
    pub x: usize,
    pub r_c0: usize,
    pub r_c1: usize,
    pub s_col_c0: usize,
    pub s_col_c1: usize,
    pub y_ring_limb: usize,
    pub y_ring_c1: usize,
    pub ct_c0: usize,
    pub ct_c1: usize,
    pub y_zcol_limb: usize,
    pub y_zcol_c1: usize,
    pub fold_digest_field: usize,
}

pub struct TerminalCePinnedPublicIsolationOutput {
    pub builder: R1csBuilder,
    pub probes: TerminalCePublicTamperProbes,
}

/// Emit only the compact terminal-CE public-statement constructor from real
/// CE-claim wires. This keeps integration tests from depending on
/// `pi_dec_circuit::alloc_ce_claim`, which is intentionally crate-private.
pub fn enforce_terminal_ce_public_from_children_against(
    prep: &Preprocessing,
    claims: &[CeClaim],
) -> Result<TerminalCePublicIsolationOutput, String> {
    let mut builder = R1csBuilder::new();
    let claim_wires = claims
        .iter()
        .map(|claim| alloc_ce_claim(&mut builder, claim))
        .collect::<Vec<_>>();
    let context = TerminalCeVerifierContext::from_preprocessing(prep);
    let public = enforce_public_from_children(&mut builder, &context, &claim_wires).map_err(|e| e.to_string())?;
    let relation_digest = eval_digest(&builder, public.relation_digest);
    let structure_digest = eval_digest(&builder, public.structure_digest);
    let params_digest = eval_digest(&builder, public.params_digest);
    let terminal_children_digest = eval_digest(&builder, public.terminal_children_digest);
    let public_digest = eval_digest(&builder, public.public_digest);
    Ok(TerminalCePublicIsolationOutput {
        builder,
        relation_digest,
        structure_digest,
        params_digest,
        terminal_children_digest,
        public_digest,
        claim_count: public.claim_count,
    })
}

/// Emit the compact terminal-CE public-statement constructor and pin its
/// outputs to the caller's expected public statement.
pub fn enforce_terminal_ce_public_pinned_against(
    prep: &Preprocessing,
    claims: &[CeClaim],
    expected: &TerminalCePublic,
) -> Result<TerminalCePinnedPublicIsolationOutput, String> {
    let mut builder = R1csBuilder::new();
    let claim_wires = claims
        .iter()
        .map(|claim| alloc_ce_claim(&mut builder, claim))
        .collect::<Vec<_>>();
    let probes = terminal_ce_tamper_probes(
        claim_wires
            .first()
            .ok_or_else(|| "terminal CE pinned-public isolation requires at least one child".to_string())?,
    )?;
    let context = TerminalCeVerifierContext::from_preprocessing(prep);
    let public = enforce_public_from_children(&mut builder, &context, &claim_wires).map_err(|e| e.to_string())?;
    if public.claim_count != expected.claim_count {
        return Err(format!(
            "terminal CE public claim_count mismatch (computed {}, expected {})",
            public.claim_count, expected.claim_count
        ));
    }
    enforce_digest_eq_const(&mut builder, public.relation_digest, expected.relation_digest);
    enforce_digest_eq_const(&mut builder, public.structure_digest, expected.structure_digest);
    enforce_digest_eq_const(&mut builder, public.params_digest, expected.params_digest);
    enforce_digest_eq_const(
        &mut builder,
        public.terminal_children_digest,
        expected.terminal_children_digest,
    );
    Ok(TerminalCePinnedPublicIsolationOutput { builder, probes })
}

/// Emit the future compact terminal-CE verifier entrypoint from actual
/// terminal-child wires. The verifier still fails closed, but this exercises
/// the production-shaped data flow: children -> public statement -> verifier.
pub fn enforce_terminal_ce_verify_from_children_against(
    prep: &Preprocessing,
    claims: &[CeClaim],
    proof: &TerminalCeProof,
) -> (R1csBuilder, Result<(), TerminalCeVerifyError>) {
    let mut builder = R1csBuilder::new();
    let claim_wires = claims
        .iter()
        .map(|claim| alloc_ce_claim(&mut builder, claim))
        .collect::<Vec<_>>();
    let context = TerminalCeVerifierContext::from_preprocessing(prep);
    let result = enforce_verify_from_children(&mut builder, &context, &claim_wires, proof).map(|_| ());
    (builder, result)
}

fn terminal_ce_tamper_probes(
    claim: &crate::paper::reductions::pi_dec_circuit::CeClaimWires,
) -> Result<TerminalCePublicTamperProbes, String> {
    Ok(TerminalCePublicTamperProbes {
        c_data: claim
            .c_data
            .first()
            .ok_or_else(|| "terminal CE probe requires non-empty c_data".to_string())?
            .col(),
        x: claim
            .x
            .first()
            .ok_or_else(|| "terminal CE probe requires non-empty X".to_string())?
            .col(),
        r_c0: claim
            .r
            .first()
            .ok_or_else(|| "terminal CE probe requires non-empty r".to_string())?
            .c0
            .col(),
        r_c1: claim
            .r
            .first()
            .ok_or_else(|| "terminal CE probe requires non-empty r".to_string())?
            .c1
            .col(),
        s_col_c0: claim
            .s_col
            .first()
            .ok_or_else(|| "terminal CE probe requires non-empty s_col".to_string())?
            .c0
            .col(),
        s_col_c1: claim
            .s_col
            .first()
            .ok_or_else(|| "terminal CE probe requires non-empty s_col".to_string())?
            .c1
            .col(),
        y_ring_limb: claim
            .y_ring
            .first()
            .and_then(|row| row.first())
            .ok_or_else(|| "terminal CE probe requires non-empty y_ring".to_string())?
            .col(),
        y_ring_c1: claim
            .y_ring
            .first()
            .and_then(|row| row.get(1))
            .ok_or_else(|| "terminal CE probe requires y_ring c1 limb".to_string())?
            .col(),
        ct_c0: claim
            .ct
            .first()
            .ok_or_else(|| "terminal CE probe requires non-empty ct".to_string())?
            .c0
            .col(),
        ct_c1: claim
            .ct
            .first()
            .ok_or_else(|| "terminal CE probe requires non-empty ct".to_string())?
            .c1
            .col(),
        y_zcol_limb: claim
            .y_zcol
            .first()
            .ok_or_else(|| "terminal CE probe requires non-empty y_zcol".to_string())?
            .col(),
        y_zcol_c1: claim
            .y_zcol
            .get(1)
            .ok_or_else(|| "terminal CE probe requires y_zcol c1 limb".to_string())?
            .col(),
        fold_digest_field: claim.fold_digest_fields[0].col(),
    })
}

fn enforce_digest_eq_const(builder: &mut R1csBuilder, digest: [Var; 4], expected: [F; 4]) {
    for (wire, value) in digest.into_iter().zip(expected) {
        builder.enforce_eq(&Lc::from_var(wire), &Lc::from_const(value));
    }
}

fn logical_fresh_bits<'a>(latest: &'a CcsClaim, owner: &str) -> Result<&'a [F], String> {
    let logical_len = 1 + F_PRIME_ENC_INST_BITS;
    if latest.x.len() < logical_len {
        return Err(format!(
            "{owner} expected at least {logical_len} fresh.x coordinates, got {}",
            latest.x.len()
        ));
    }
    if latest.x[logical_len..]
        .iter()
        .any(|&value| value != F::ZERO)
    {
        return Err(format!("{owner} requires zero aligned-public padding"));
    }
    Ok(&latest.x[1..logical_len])
}

/// Emit only the terminal-fold consumed-handle check.
pub fn enforce_terminal_fold_against_last_acc_digest(
    prep: &Preprocessing,
    pre_final_running: &RunningInstance,
    trailing_latest: &[CcsClaim],
    final_fold_nifs: &NifsProof,
    last_acc_digest: [u8; 32],
) -> Result<R1csBuilder, String> {
    let mut builder = R1csBuilder::new();
    let latest = trailing_latest
        .first()
        .ok_or_else(|| "terminal-fold isolation requires non-empty trailing latest".to_string())?;
    let fresh_bits = logical_fresh_bits(latest, "terminal-fold isolation")?;

    let zero_digest = [F::ZERO; 4];
    let state = dummy_state_wires(&mut builder, last_acc_digest);
    let last = FPrimeStepOutput {
        x_out: alloc_digest_fields(&mut builder, zero_digest),
        x_out_bits: fresh_bits.iter().map(|&bit| builder.alloc(bit)).collect(),
        prior_link: None,
        state_in: state,
        state_out: state,
        nifs_running: None,
        nifs_running_parent_authority: None,
        nifs_parent: None,
        nifs_children: None,
        pi_dec_canonical_x_receipt: None,
        fresh_public_suffixes: Vec::new(),
        fresh_adv: Vec::new(),
    };

    emit_terminal_fold(
        &mut builder,
        prep,
        &last,
        pre_final_running,
        trailing_latest,
        final_fold_nifs,
    )
    .map_err(|e| e.to_string())?;
    Ok(builder)
}

/// Emit the terminal fold followed by the direct raw pending projection and
/// terminal CE closure. The projection and Ajtai relation consume the same
/// ordered witness allocations; child `y_zcol` sidecars are not its source.
pub fn enforce_terminal_fold_ce_closure_against(
    prep: &Preprocessing,
    pre_final_running: &RunningInstance,
    trailing_latest: &[CcsClaim],
    final_fold_nifs: &NifsProof,
    last_acc_digest: [u8; 32],
    terminal_witnesses: &[WitnessMat],
) -> Result<R1csBuilder, String> {
    let mut builder = R1csBuilder::new();
    let latest = trailing_latest
        .first()
        .ok_or_else(|| "terminal CE-closure isolation requires non-empty trailing latest".to_string())?;
    let fresh_bits = logical_fresh_bits(latest, "terminal CE-closure isolation")?;

    let zero_digest = [F::ZERO; 4];
    let state = dummy_state_wires(&mut builder, last_acc_digest);
    let last = FPrimeStepOutput {
        x_out: alloc_digest_fields(&mut builder, zero_digest),
        x_out_bits: fresh_bits.iter().map(|&bit| builder.alloc(bit)).collect(),
        prior_link: None,
        state_in: state,
        state_out: state,
        nifs_running: None,
        nifs_running_parent_authority: None,
        nifs_parent: None,
        nifs_children: None,
        pi_dec_canonical_x_receipt: None,
        fresh_public_suffixes: Vec::new(),
        fresh_adv: Vec::new(),
    };
    let (_, _, _, _, _, terminal_children, terminal_pending_projection) = emit_terminal_fold(
        &mut builder,
        prep,
        &last,
        pre_final_running,
        trailing_latest,
        final_fold_nifs,
    )
    .map_err(|e| e.to_string())?;
    match terminal_pending_projection.as_ref() {
        Some(pending) => crate::paper::decider_ce_relation::enforce_final_ce_relations_with_pending_projection(
            &mut builder,
            prep,
            &terminal_children,
            terminal_witnesses,
            pending,
        ),
        None => enforce_final_dec_children_relations(&mut builder, prep, &terminal_children, terminal_witnesses),
    }
    .map_err(|e| e.to_string())?;
    Ok(builder)
}

pub struct TerminalParentProbeWires {
    pub last_parent_y_ring_c1: Var,
}

pub struct TerminalChildrenProbeWires {
    pub last_child_y_ring_c1: Var,
}

/// Emit terminal-fold rows with a last-step parent-authority claim.
pub fn enforce_terminal_fold_parent_authority_against_self(
    prep: &Preprocessing,
    pre_final_running: &RunningInstance,
    trailing_latest: &[CcsClaim],
    final_fold_nifs: &NifsProof,
    last_acc_digest: [u8; 32],
) -> Result<(R1csBuilder, TerminalParentProbeWires), String> {
    let parent = pre_final_running
        .parent_authority
        .as_ref()
        .ok_or_else(|| "terminal parent-authority isolation requires a pre-final parent".to_string())?;
    let mut builder = R1csBuilder::new();
    let latest = trailing_latest
        .first()
        .ok_or_else(|| "terminal parent-authority isolation requires non-empty trailing latest".to_string())?;
    let fresh_bits = logical_fresh_bits(latest, "terminal parent-authority isolation")?;

    let zero_digest = [F::ZERO; 4];
    let state = dummy_state_wires(&mut builder, last_acc_digest);
    let last_parent = alloc_ce_claim(&mut builder, parent);
    let probe = last_parent
        .y_ring
        .first()
        .and_then(|row| row.get(1))
        .copied()
        .ok_or_else(|| "terminal parent-authority probe requires a y_ring c1 limb".to_string())?;
    let last = FPrimeStepOutput {
        x_out: alloc_digest_fields(&mut builder, zero_digest),
        x_out_bits: fresh_bits.iter().map(|&bit| builder.alloc(bit)).collect(),
        prior_link: None,
        state_in: state,
        state_out: state,
        nifs_running: None,
        nifs_running_parent_authority: None,
        nifs_parent: Some(last_parent),
        nifs_children: None,
        pi_dec_canonical_x_receipt: None,
        fresh_public_suffixes: Vec::new(),
        fresh_adv: Vec::new(),
    };

    emit_terminal_fold(
        &mut builder,
        prep,
        &last,
        pre_final_running,
        trailing_latest,
        final_fold_nifs,
    )
    .map_err(|e| e.to_string())?;
    Ok((
        builder,
        TerminalParentProbeWires {
            last_parent_y_ring_c1: probe,
        },
    ))
}

/// Emit terminal-fold rows plus the last-step child→terminal-running continuity link.
pub fn enforce_terminal_fold_children_continuity_against_self(
    prep: &Preprocessing,
    pre_final_running: &RunningInstance,
    trailing_latest: &[CcsClaim],
    final_fold_nifs: &NifsProof,
    last_acc_digest: [u8; 32],
) -> Result<(R1csBuilder, TerminalChildrenProbeWires), String> {
    if pre_final_running.claims.is_empty() {
        return Err("terminal children continuity isolation requires non-empty pre-final running".to_string());
    }
    let mut builder = R1csBuilder::new();
    let latest = trailing_latest
        .first()
        .ok_or_else(|| "terminal children continuity isolation requires non-empty trailing latest".to_string())?;
    let fresh_bits = logical_fresh_bits(latest, "terminal children continuity isolation")?;

    let zero_digest = [F::ZERO; 4];
    let state = dummy_state_wires(&mut builder, last_acc_digest);
    let last_children = pre_final_running
        .claims
        .iter()
        .map(|claim| alloc_ce_claim(&mut builder, claim))
        .collect::<Vec<_>>();
    let probe = last_children[0]
        .y_ring
        .first()
        .and_then(|row| row.get(1))
        .copied()
        .ok_or_else(|| "terminal children continuity probe requires a y_ring c1 limb".to_string())?;
    let last = FPrimeStepOutput {
        x_out: alloc_digest_fields(&mut builder, zero_digest),
        x_out_bits: fresh_bits.iter().map(|&bit| builder.alloc(bit)).collect(),
        prior_link: None,
        state_in: state,
        state_out: state,
        nifs_running: None,
        nifs_running_parent_authority: None,
        nifs_parent: None,
        nifs_children: Some(last_children),
        pi_dec_canonical_x_receipt: None,
        fresh_public_suffixes: Vec::new(),
        fresh_adv: Vec::new(),
    };

    let (
        _emitted,
        _latest_link,
        _parent_link,
        _final_acc,
        terminal_running,
        _terminal_children,
        _terminal_pending_projection,
    ) = emit_terminal_fold(
        &mut builder,
        prep,
        &last,
        pre_final_running,
        trailing_latest,
        final_fold_nifs,
    )
    .map_err(|e| e.to_string())?;
    let prev_children = last
        .nifs_children
        .as_ref()
        .ok_or_else(|| "terminal children continuity isolation missing synthetic previous children".to_string())?;
    super::enforce_child_core_equal_running(&mut builder, prev_children, &terminal_running)?;
    Ok((
        builder,
        TerminalChildrenProbeWires {
            last_child_y_ring_c1: probe,
        },
    ))
}

/// Probe CE-continuity rows for one child/running claim pair.
pub struct CeContinuityProbeWires {
    pub c_data0: Var,
    pub x0: Var,
    pub c_d: Var,
    pub c_kappa: Var,
    pub x_rows: Var,
    pub x_cols: Var,
    pub m_in: Var,
    pub r_c0: Var,
    pub r_c1: Var,
    pub s_col_c0: Var,
    pub s_col_c1: Var,
    pub ct_c1: Var,
    pub y_ring_c1: Var,
    pub fold_digest0: Var,
}

pub fn enforce_ce_continuity_against_self(claim: &CeClaim) -> Result<(R1csBuilder, CeContinuityProbeWires), String> {
    enforce_ce_continuity_between(claim, claim)
}

/// Emit CE-core continuity for separately supplied child and running claims.
/// Their `y_zcol` sidecars are deliberately not allocated.
pub fn enforce_ce_continuity_between(
    child_claim: &CeClaim,
    running_claim: &CeClaim,
) -> Result<(R1csBuilder, CeContinuityProbeWires), String> {
    let mut builder = R1csBuilder::new();
    let child = alloc_dec_child_claim(&mut builder, child_claim);
    let running = alloc_running_claim(&mut builder, running_claim);
    let probes = CeContinuityProbeWires {
        c_data0: running.c_data[0],
        x0: running.x[0],
        c_d: running.c_d_var,
        c_kappa: running.c_kappa_var,
        x_rows: running.x_rows_var,
        x_cols: running.x_cols_var,
        m_in: running.m_in_var,
        r_c0: running
            .r
            .first()
            .ok_or_else(|| "CE-continuity probe requires non-empty r".to_string())?
            .c0,
        r_c1: running
            .r
            .first()
            .ok_or_else(|| "CE-continuity probe requires non-empty r".to_string())?
            .c1,
        s_col_c0: running
            .s_col
            .first()
            .ok_or_else(|| "CE-continuity probe requires non-empty s_col".to_string())?
            .c0,
        s_col_c1: running
            .s_col
            .first()
            .ok_or_else(|| "CE-continuity probe requires non-empty s_col".to_string())?
            .c1,
        ct_c1: running
            .ct
            .first()
            .ok_or_else(|| "CE-continuity probe requires non-empty ct".to_string())?
            .c1,
        y_ring_c1: running
            .y_ring
            .first()
            .and_then(|row| row.first())
            .ok_or_else(|| "CE-continuity probe requires non-empty y_ring".to_string())?
            .c1,
        fold_digest0: running.fold_digest_fields[0],
    };
    super::enforce_child_core_equal_running(&mut builder, &[child], &[running])?;
    Ok((builder, probes))
}

/// Emit only the terminal latest-link rows.
pub fn enforce_terminal_latest_link_against(
    last_x_out_bits: &[F],
    fresh_public_inputs: &[Vec<F>],
) -> Result<R1csBuilder, String> {
    let mut builder = R1csBuilder::new();
    let last_bits: Vec<Var> = last_x_out_bits
        .iter()
        .map(|&bit| builder.alloc(bit))
        .collect();
    let fresh_x: Vec<Vec<Var>> = fresh_public_inputs
        .iter()
        .map(|x| x.iter().map(|&bit| builder.alloc(bit)).collect())
        .collect();
    super::enforce_terminal_latest_link(
        &mut builder,
        crate::paper::f_prime::r1cs::FPrimePublicInputLayout::plain(),
        &fresh_x,
        &last_bits,
    )
    .map_err(|e| e.to_string())?;
    Ok(builder)
}

pub struct BaseStateProbeWires {
    pub vk_fs0: Var,
    pub structure0: Var,
    pub chunk_count: Var,
    pub step_count: Var,
    pub z_0_0: Var,
    pub z_i_0: Var,
    pub pc: Var,
    pub semantic0: Var,
    pub acc0: Var,
    pub public_trace0: Var,
}

/// Probe base-state seed pins.
pub fn enforce_base_state_constants_against(
    prep: &Preprocessing,
    initial_semantic_state_digest: [u8; 32],
) -> (R1csBuilder, BaseStateProbeWires) {
    let mut builder = R1csBuilder::new();
    let structure = *prep.structure_digest();
    let z_0 = initial_boundary_digest(&structure, prep.public_input_len);
    let public_trace = public_trace_seed_digest(&structure);
    let empty_acc = AccumulatorHandle::empty().digest();
    let state = FPrimeStateWires {
        vk_fs_digest: alloc_digest32(&mut builder, prep.vk.digest()),
        pi_ccs_header_bundle: alloc_digest_fields(&mut builder, prep.pi_ccs_header_bundle()),
        chunk_count: builder.alloc(F::ZERO),
        step_count: builder.alloc(F::ZERO),
        z_0: alloc_digest32(&mut builder, z_0),
        z_i: alloc_digest32(&mut builder, z_0),
        pc: builder.alloc(F::ONE),
        semantic_state_digest: alloc_digest32(&mut builder, initial_semantic_state_digest),
        acc_digest: alloc_digest32(&mut builder, empty_acc),
        public_trace: alloc_digest32(&mut builder, public_trace),
        nebula: None,
    };
    let base = FPrimeStepOutput {
        x_out: alloc_digest_fields(&mut builder, [F::ZERO; 4]),
        x_out_bits: Vec::new(),
        prior_link: None,
        state_in: state,
        state_out: state,
        nifs_running: None,
        nifs_running_parent_authority: None,
        nifs_parent: None,
        nifs_children: None,
        pi_dec_canonical_x_receipt: None,
        fresh_public_suffixes: Vec::new(),
        fresh_adv: Vec::new(),
    };
    let public = PublicImage {
        vk_fs_digest: prep.vk.digest(),
        chunk_count: 0,
        step_count: 0,
        z_0,
        z_i: z_0,
        pc: 1,
        initial_semantic_state_digest,
        semantic_state_digest: initial_semantic_state_digest,
        acc_digest: empty_acc,
        public_trace,
        x_out: EncInst::from_digest([0u8; 32]),
    };
    let probes = BaseStateProbeWires {
        vk_fs0: base.state_in.vk_fs_digest[0],
        structure0: base.state_in.pi_ccs_header_bundle[0],
        chunk_count: base.state_in.chunk_count,
        step_count: base.state_in.step_count,
        z_0_0: base.state_in.z_0[0],
        z_i_0: base.state_in.z_i[0],
        pc: base.state_in.pc,
        semantic0: base.state_in.semantic_state_digest[0],
        acc0: base.state_in.acc_digest[0],
        public_trace0: base.state_in.public_trace[0],
    };
    super::enforce_base_state_constants(&mut builder, prep, &public, &base);
    (builder, probes)
}

/// Probe terminal public-image pins against caller-supplied public values.
pub fn enforce_public_image_pins_against(prep: &Preprocessing, public: &PublicImage) -> R1csBuilder {
    enforce_public_image_pins_against_chain(prep, public, public)
}

/// Probe terminal public-image pins with independent chain-derived values.
pub fn enforce_public_image_pins_against_chain(
    prep: &Preprocessing,
    chain: &PublicImage,
    public: &PublicImage,
) -> R1csBuilder {
    let mut builder = R1csBuilder::new();
    let state = FPrimeStateWires {
        vk_fs_digest: alloc_digest32(&mut builder, chain.vk_fs_digest),
        pi_ccs_header_bundle: alloc_digest_fields(&mut builder, prep.pi_ccs_header_bundle()),
        chunk_count: builder.alloc(F::from_u64(chain.chunk_count)),
        step_count: builder.alloc(F::from_u64(chain.step_count)),
        z_0: alloc_digest32(&mut builder, chain.z_0),
        z_i: alloc_digest32(&mut builder, chain.z_i),
        pc: builder.alloc(F::from_u64(chain.pc)),
        semantic_state_digest: alloc_digest32(&mut builder, chain.semantic_state_digest),
        acc_digest: alloc_digest32(&mut builder, chain.acc_digest),
        public_trace: alloc_digest32(&mut builder, chain.public_trace),
        nebula: None,
    };
    let last = FPrimeStepOutput {
        x_out: alloc_digest32(&mut builder, chain.x_out.digest_bytes),
        x_out_bits: Vec::new(),
        prior_link: None,
        state_in: state,
        state_out: state,
        nifs_running: None,
        nifs_running_parent_authority: None,
        nifs_parent: None,
        nifs_children: None,
        pi_dec_canonical_x_receipt: None,
        fresh_public_suffixes: Vec::new(),
        fresh_adv: Vec::new(),
    };
    let final_acc_digest = alloc_digest32(&mut builder, chain.acc_digest);
    super::pin_public_image(&mut builder, public, prep, &last, &final_acc_digest);
    builder
}

pub struct StateLinkProbeWires {
    pub vk_fs0: Var,
    pub structure0: Var,
    pub chunk_count: Var,
    pub step_count: Var,
    pub z_0_0: Var,
    pub z_i_0: Var,
    pub pc: Var,
    pub semantic0: Var,
    pub acc0: Var,
    pub public_trace0: Var,
}

/// Probe `prev.state_out == next.state_in` rows.
pub fn enforce_state_link_against_self() -> (R1csBuilder, StateLinkProbeWires) {
    let mut builder = R1csBuilder::new();
    let acc_digest = [0u8; 32];
    let a = dummy_state_wires(&mut builder, acc_digest);
    let b = dummy_state_wires(&mut builder, acc_digest);
    let probes = StateLinkProbeWires {
        vk_fs0: b.vk_fs_digest[0],
        structure0: b.pi_ccs_header_bundle[0],
        chunk_count: b.chunk_count,
        step_count: b.step_count,
        z_0_0: b.z_0[0],
        z_i_0: b.z_i[0],
        pc: b.pc,
        semantic0: b.semantic_state_digest[0],
        acc0: b.acc_digest[0],
        public_trace0: b.public_trace[0],
    };
    super::enforce_state_link(&mut builder, &a, &b);
    (builder, probes)
}

fn alloc_digest_fields(builder: &mut R1csBuilder, digest: [F; 4]) -> [Var; 4] {
    digest.map(|lane| builder.alloc(lane))
}

fn eval_digest(builder: &R1csBuilder, digest: [Var; 4]) -> [F; 4] {
    digest.map(|lane| builder.witness()[lane.col()])
}

fn alloc_digest32(builder: &mut R1csBuilder, digest: [u8; 32]) -> [Var; 4] {
    alloc_digest_fields(builder, digest32_as_fields(digest))
}

fn dummy_state_wires(builder: &mut R1csBuilder, acc_digest: [u8; 32]) -> FPrimeStateWires {
    let zero = [F::ZERO; 4];
    FPrimeStateWires {
        vk_fs_digest: alloc_digest_fields(builder, zero),
        pi_ccs_header_bundle: alloc_digest_fields(builder, zero),
        chunk_count: builder.alloc(F::ZERO),
        step_count: builder.alloc(F::ZERO),
        z_0: alloc_digest_fields(builder, zero),
        z_i: alloc_digest_fields(builder, zero),
        pc: builder.alloc(F::ONE),
        semantic_state_digest: alloc_digest_fields(builder, zero),
        acc_digest: alloc_digest32(builder, acc_digest),
        public_trace: alloc_digest_fields(builder, zero),
        nebula: None,
    }
}

fn alloc_running_claim(builder: &mut R1csBuilder, claim: &CeClaim) -> SplitNcPiCcsOutputWires {
    let mut x = Vec::with_capacity(claim.X.rows() * claim.X.cols());
    for r in 0..claim.X.rows() {
        for c in 0..claim.X.cols() {
            x.push(builder.alloc(claim.X[(r, c)]));
        }
    }
    SplitNcPiCcsOutputWires {
        c_d: claim.c.d,
        c_d_var: builder.alloc(F::from_u64(claim.c.d as u64)),
        c_kappa: claim.c.kappa,
        c_kappa_var: builder.alloc(F::from_u64(claim.c.kappa as u64)),
        c_data: builder.alloc_vec(&claim.c.data),
        adv: alloc_adv(builder, claim.adv.as_ref()),
        x,
        x_rows: claim.X.rows(),
        x_rows_var: builder.alloc(F::from_u64(claim.X.rows() as u64)),
        x_cols: claim.X.cols(),
        x_cols_var: builder.alloc(F::from_u64(claim.X.cols() as u64)),
        m_in: claim.m_in,
        m_in_var: builder.alloc(F::from_u64(claim.m_in as u64)),
        r: alloc_k_vec(builder, &claim.r),
        s_col: alloc_k_vec(builder, &claim.s_col),
        y_ring: claim
            .y_ring
            .iter()
            .map(|row| alloc_k_vec(builder, row))
            .collect(),
        ct: alloc_k_vec(builder, &claim.ct),
        y_zcol: Vec::new(),
        fold_digest_fields: alloc_digest32(builder, claim.fold_digest),
    }
}

fn alloc_k_vec(builder: &mut R1csBuilder, values: &[K]) -> Vec<KVar> {
    values
        .iter()
        .map(|k| {
            let [c0, c1] = k.as_coeffs();
            KVar::alloc(builder, c0, c1)
        })
        .collect()
}

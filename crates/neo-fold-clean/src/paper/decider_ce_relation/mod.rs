//! Terminal CE-relation gadgets — the in-circuit closure of the
//! HyperNova/SuperNeo §7 decider obligation.
//!
//! ## Ownership
//!
//! Owns the R1CS rows that bind the **final running accumulator's
//! opened witnesses `Z`** to each claim's `(c, X, r, y_ring, ct)` plus
//! the optional implementation-side NC channel `(s_col, y_zcol)` when
//! present — the same SuperNeo CE relation obligations that
//! `lifecycle::verify` executes natively. The Rust function and the
//! circuit verify the same equations; both sides must stay in sync.
//!
//! For each `(claim, witness Z)` in `statement.witness.final_state.proof.running`:
//!
//! 0. **Program public-input shape**: `claim.m_in` must not exceed the
//!    CCS width `structure.m`. If preprocessing fixed
//!    `public_input_len`, then `claim.m_in` must equal it. The prover
//!    cannot relabel the CE claim under a smaller projection or into
//!    packed padding beyond the structure.
//! 1. **Commitment opening**: `commit_Ajtai(Z) == claim.c.data`
//!    (linear in `Z`, coefficients from preprocessing's verifier-owned
//!    Ajtai setup).
//! 2. **Public-input projection**: `L_in(Z) == claim.X` for the
//!    active `m_in` columns; inactive columns are zero (already pinned
//!    by `pi_ccs_split_nc_circuit::validate_inactive_x_zero`).
//! 3. **Low-norm / NC-bound alphabet**: every entry of `Z` lies in
//!    `{-(b-1), …, +(b-1)}`, matching native
//!    `neo_math::balanced::within_nc_bound`. Ensures the Ajtai binding
//!    property's norm bound. (See [`witness::enforce_balanced_alphabet`].)
//! 4. **CE evaluation closure**: for every CCS matrix `M_j`,
//!    `claim.y_ring[j] == multilinear_eval(M_j · Z, claim.r)`.
//!    Closes the SuperNeo §7 obligation against the opened witness —
//!    see [`evaluation::enforce_y_ring_from_z_at_r`].
//! 5. **ct closure**: `claim.ct[j] == claim.y_ring[j][lane=0]` per
//!    SuperNeo Theorem 5 — the constant-term lane of `y_ring[j]` is
//!    the field-level `M̄_j z(r)`. Implementation-consistency
//!    obligation that makes the circuit's CE-relation contract match
//!    the native `verify_uncompressed`'s. See
//!    [`evaluation::enforce_ct_from_y_ring`].
//! 6. **Legacy optional NC-channel consistency**: if `s_col/y_zcol` are
//!    carried by a non-production claim, `claim.y_zcol == Z ·
//!    chi(claim.s_col)`. Production delayed-projection authority does not
//!    consume this sidecar; [`enforce_final_ce_relations_with_pending_projection`]
//!    projects the ordered raw witnesses directly at the verifier's pending
//!    old block.
//! 7. **Unsupported sidecar rejection**: `aux_openings`, Pattern-A
//!    coordinates, and `u_offset/u_len` must be absent. The clean
//!    SplitNc/NIFS circuit does not implement those fields, so they
//!    cannot be accumulator-digested authority.
//!
//! ## Why this lives here, not in `lifecycle::verify`
//!
//! `verify_uncompressed` checks the same SuperNeo CE relation
//! natively. A consumer that runs the Rust verifier sees the
//! obligation closed there; a SNARK consumer that verifies the
//! decider R1CS sees it closed here. Both paths must enforce the
//! same authority set — drifting one without the other is a
//! soundness regression. The two regressions tests
//! (`final_witness_authority_rejects_y_ring_inconsistent_with_m_z_at_r`
//! native and `decider_ce_isolation_rejects_*` in-circuit) check
//! parity per-obligation.
//!
//! ## Composition
//!
//! [`enforce_final_dec_children_relations`] or
//! [`enforce_final_ce_relations_with_pending_projection`] is called from
//! `crate::engine::decider::synthesize_statement_r1cs` after the
//! chain replay and terminal-fold NIFS.V emission. The wires it
//! receives:
//!
//! - `final_claims_wires`: the `CeClaimWires` for each final running
//!   claim, already allocated and constrained by the terminal NIFS's
//!   Π_DEC children block.
//! - `final_witnesses`: the prover-supplied `WitnessMat` for each claim. On
//!   the pending-production path this module allocates each witness exactly
//!   once, projects those wires at `pending.old_block`, and then passes the
//!   identical allocations to Ajtai and the remaining CE checks.
//!
//! No new public types are exposed at the crate root. The orchestrator
//! is the only surface and it is `pub(crate)`.
//!
//! ## Prover-vs-verifier drift — read this before treating the gadget
//!    as production-ready
//!
//! This module turns the decider circuit into the **CE prover/checker**:
//! it allocates the witness `Z` directly inside the circuit and emits
//! the constraints
//!
//!   `Commit(Z) == c`, `X == L_in(Z)`, `y_ring == M·Z(r)`, `Z low-norm`.
//!
//! That is a sound but architecturally wrong shape for a ledger-facing
//! verifier. The production split is:
//!
//! ```text
//! current shape (this module):
//!   decider circuit proves the CE relation directly
//!     ↳ Z is a wire inside the circuit
//!     ↳ M_j · Z is recomputed in-circuit, row by row
//!
//! production shape (NOT IN THIS BRANCH):
//!   off-circuit prover produces a compact proof of the CE relation
//!   decider circuit only VERIFIES that proof
//!     ↳ Z stays out-of-circuit
//!     ↳ constraints are the in-circuit verifier rows for the chosen
//!       proof backend
//! ```
//!
//! The constraint count under the current shape is roughly
//! `O(n · t · m)` — fine for `(n, m)` in the thousands (lifecycle
//! tests), catastrophic for real F'-image sizes (`n, m` ~ 10⁶+ for
//! non-trivial application R1CS, where the constraint count blows past
//! anything that can land in one downstream ledger-facing proof).
//!
//! **Layering note.** SuperNeo's folding leaves a terminal obligation —
//! "these CE claims open to some Z". Discharging that obligation is a
//! SEPARATE proof layer. This module does not choose, prepare, or wire a
//! compact backend; it only defines the direct reference relation that a
//! future compact proof must prove. See the
//! `engine::decider::LastStepTerminalSynthesis` docs for the current
//! inductive picture and scope limits.
//!
//! ## Why this still lives in the tree
//!
//! 1. **Soundness fallback.** A scalability-wrong-but-sound CE closure
//!    is better than no CE closure: the F'-image's `acc_digest` chain
//!    commits to the terminal CE claims, but it does not prove that the
//!    opened witness Z satisfies those claims. Without these rows the
//!    SNARK consumer could accept a final witness Z that is not a real
//!    opening of the terminal accumulator.
//! 2. **Correctness reference.** When the compact terminal CE proof
//!    verifier lands, its outputs must agree byte-for-byte with what
//!    this gadget enforces. The gadget defines the relation; the
//!    production backend will be its proof, not its replacement.
//!
//! ## What still needs designing (out of scope for this module)
//!
//! - **Terminal-CE decider implementation.** Whatever backend lands, the
//!   binding rule is a Poseidon2 digest over the full `terminal_children`
//!   set (every `CeClaim` field — not just commitment data), so the
//!   compact proof can't be replayed against a different set of children.
//! - **In-circuit verifier rows** for the chosen backend.
//! - **Wiring** to terminal NIFS children, so the proof's claimed CE
//!   values are pinned to the verifier-derived `terminal_children`
//!   from `emit_terminal_fold`.
//!
//! Until that lands, the strict-child and pending-projection entrypoints in
//! this module are the soundness contract this code stands on. Treat them as
//! a reference: if you are
//! sizing the decider circuit for production, the rows this gadget
//! emits are NOT the rows that will be there.

mod commitment;
mod evaluation;
mod old_block_projection;
mod witness;

pub(crate) use commitment::{enforce_ajtai_opening, enforce_ajtai_slice_opening, enforce_x_projection};
pub(crate) use evaluation::{enforce_ct_from_y_ring, enforce_y_ring_from_z_at_r, enforce_y_zcol_from_z_at_s_col};
pub(crate) use old_block_projection::enforce_raw_old_block_projection;
pub(crate) use witness::alloc_final_witness;

use thiserror::Error;

use crate::engine::r1cs_circuit::builder::TerminalCeClaimAudit;
use crate::engine::r1cs_circuit::{R1csBuilder, RawOldBlockProjectionPlan, RAW_OLD_BLOCK_CHILD_COUNT};
use crate::lifecycle::Preprocessing;
use crate::paper::reductions::pi_dec_circuit::CeClaimWires;
use crate::paper::relations::WitnessMat;

#[derive(Clone, Copy)]
enum NcChannelClosure {
    LegacyClaimSidecar,
    StrictPiDecChild,
}

#[derive(Debug, Error)]
pub(crate) enum CeRelationError {
    #[error("decider_ce_relation: claim/witness count mismatch (claims={claims}, witnesses={witnesses})")]
    CountMismatch { claims: usize, witnesses: usize },
    #[error("decider_ce_relation: claim {index} {what} shape mismatch (expected {expected}, got {got})")]
    ShapeMismatch {
        index: usize,
        what: &'static str,
        expected: usize,
        got: usize,
    },
    #[error("decider_ce_relation: raw old-block {what} shape mismatch (expected {expected}, got {got})")]
    RawOldBlockProjectionShape {
        what: &'static str,
        expected: usize,
        got: usize,
    },
    #[error("decider_ce_relation: Ajtai global setup unavailable for d={d}, cols={cols}")]
    AjtaiSetupMissing { d: usize, cols: usize },
    #[error(
        "decider_ce_relation: balanced-alphabet gadget is undefined for b={b} (NeoParams::new \
         rejects b < 2; this only fires from a hand-crafted Preprocessing)"
    )]
    InvalidNormBound { b: u32 },
    #[error("decider_ce_relation: claim {index} Nebula adv presence does not match preprocessing")]
    NebulaAdvPresence { index: usize },
}

/// Emit the terminal CE-relation constraint rows.
///
/// `final_claims_wires` MUST be the terminal NIFS output children's wires
/// (`nifs_outputs.children` from `emit_terminal_fold`), whose `c_data`,
/// `X`, `r`, `y_ring` are *constrained* by the upstream Π_CCS / Π_RLC /
/// Π_DEC circuit verifiers — they're the authoritative end of the
/// terminal accumulator. Passing a freshly-allocated copy of
/// `statement.witness.final_state.proof.running.claims` would be a
/// soundness bug: that's prover-supplied private data, not bound to the
/// circuit. Only the witnesses `Z` come from `final_state` — they're the
/// new prover-private payload this gadget closes.
pub(crate) fn enforce_final_ce_relations(
    builder: &mut R1csBuilder,
    prep: &Preprocessing,
    final_claims_wires: &[CeClaimWires],
    final_witnesses: &[WitnessMat],
) -> Result<(), CeRelationError> {
    enforce_final_ce_relations_with_nc_channel(
        builder,
        prep,
        final_claims_wires,
        final_witnesses,
        NcChannelClosure::LegacyClaimSidecar,
    )
}

/// Close the paper-level fields of strict Π_DEC children.
///
/// Strict child allocation deliberately omits `y_zcol`; that value is not a
/// proved Π_DEC child output. The ordinary terminal profile therefore closes
/// only the child CE tuple, while the delayed profile additionally calls
/// [`enforce_final_ce_relations_with_pending_projection`] to bind its
/// separately carried projection.
pub(crate) fn enforce_final_dec_children_relations(
    builder: &mut R1csBuilder,
    prep: &Preprocessing,
    final_claims_wires: &[CeClaimWires],
    final_witnesses: &[WitnessMat],
) -> Result<(), CeRelationError> {
    enforce_final_ce_relations_with_nc_channel(
        builder,
        prep,
        final_claims_wires,
        final_witnesses,
        NcChannelClosure::StrictPiDecChild,
    )
}

fn enforce_final_ce_relations_with_nc_channel(
    builder: &mut R1csBuilder,
    prep: &Preprocessing,
    final_claims_wires: &[CeClaimWires],
    final_witnesses: &[WitnessMat],
    nc_channel_closure: NcChannelClosure,
) -> Result<(), CeRelationError> {
    validate_count(final_claims_wires, final_witnesses)?;
    let expected_m = prep.structure().m;
    for (index, (claim_wires, witness)) in final_claims_wires.iter().zip(final_witnesses).enumerate() {
        let witness_wires =
            alloc_final_witness(builder, witness, expected_m).map_err(|err| witness_shape_err(index, err))?;
        enforce_one_final_ce_relation(builder, prep, index, claim_wires, &witness_wires, nc_channel_closure)?;
    }
    Ok(())
}

/// Emit the production terminal closure with direct delayed-projection rows.
///
/// The witness allocations are made once and consumed twice: first by the
/// raw-old-block projection and then by their corresponding Ajtai openings.
/// This shared allocation is the concrete authority join; child sidecars and
/// digests are not inputs to the projection.
pub(crate) fn enforce_final_ce_relations_with_pending_projection(
    builder: &mut R1csBuilder,
    prep: &Preprocessing,
    final_claims_wires: &[CeClaimWires],
    final_witnesses: &[WitnessMat],
    pending: &crate::paper::reductions::pi_ccs_split_nc_circuit::PendingProjectionWires,
) -> Result<(), CeRelationError> {
    validate_count(final_claims_wires, final_witnesses)?;
    let expected_m = prep.structure().m;
    validate_pending_projection_inputs(prep, final_claims_wires, final_witnesses, pending)?;
    let witness_wires = alloc_final_witness_family(builder, final_witnesses, expected_m)?;

    old_block_projection::enforce_raw_old_block_projection(
        builder,
        expected_m,
        &pending.old_block,
        &pending.parent_y_zcol,
        &witness_wires,
        prep.params.b(),
    )
    .map_err(raw_old_block_projection_err)?;

    for (index, (claim_wires, witness_wires)) in final_claims_wires.iter().zip(&witness_wires).enumerate() {
        enforce_one_final_ce_relation(
            builder,
            prep,
            index,
            claim_wires,
            witness_wires,
            NcChannelClosure::StrictPiDecChild,
        )?;
    }
    let child_witness_first_columns = witness_wires
        .iter()
        .map(|witness| {
            witness
                .values
                .first()
                .expect("validated terminal witness has at least one entry")
                .col()
        })
        .collect();
    builder.record_terminal_pending_projection_ajtai_join(
        crate::engine::r1cs_circuit::RAW_OLD_BLOCK_PENDING_JOIN_ID,
        child_witness_first_columns,
    );
    Ok(())
}

/// Follow the production terminal-closure allocation schedule through the
/// exact final-witness family, then return its compact raw-projection
/// placement without emitting projection or Ajtai/low-norm rows.
///
/// The returned Ajtai bases are not guessed: the normal closure passes this
/// same `witness_wires` vector to every subsequent Ajtai opening.
pub(crate) fn capture_final_ce_pending_projection_placement(
    builder: &mut R1csBuilder,
    prep: &Preprocessing,
    final_claims_wires: &[CeClaimWires],
    final_witnesses: &[WitnessMat],
    pending: &crate::paper::reductions::pi_ccs_split_nc_circuit::PendingProjectionWires,
) -> Result<crate::engine::r1cs_circuit::TerminalPendingProjectionAudit, CeRelationError> {
    validate_count(final_claims_wires, final_witnesses)?;
    let expected_m = prep.structure().m;
    validate_pending_projection_inputs(prep, final_claims_wires, final_witnesses, pending)?;
    let witness_wires = alloc_final_witness_family(builder, final_witnesses, expected_m)?;
    let mut placement = old_block_projection::production_placement(
        builder,
        expected_m,
        &pending.old_block,
        &pending.parent_y_zcol,
        &witness_wires,
        prep.params.b(),
    )
    .map_err(raw_old_block_projection_err)?;
    placement.ajtai_child_witness_first_columns = witness_wires
        .iter()
        .map(|witness| {
            witness
                .values
                .first()
                .expect("validated terminal witness has at least one entry")
                .col()
        })
        .collect();
    if placement.ajtai_child_witness_first_columns != placement.projection_child_witness_first_columns {
        return Err(CeRelationError::RawOldBlockProjectionShape {
            what: "projection/Ajtai FinalWitnessWires allocation join",
            expected: placement.projection_child_witness_first_columns.len(),
            got: placement.ajtai_child_witness_first_columns.len(),
        });
    }
    Ok(placement)
}

fn alloc_final_witness_family(
    builder: &mut R1csBuilder,
    final_witnesses: &[WitnessMat],
    expected_m: usize,
) -> Result<Vec<witness::FinalWitnessWires>, CeRelationError> {
    final_witnesses
        .iter()
        .enumerate()
        .map(|(index, witness)| {
            alloc_final_witness(builder, witness, expected_m).map_err(|err| witness_shape_err(index, err))
        })
        .collect()
}

fn enforce_one_final_ce_relation(
    builder: &mut R1csBuilder,
    prep: &Preprocessing,
    index: usize,
    claim_wires: &CeClaimWires,
    witness_wires: &witness::FinalWitnessWires,
    nc_channel_closure: NcChannelClosure,
) -> Result<(), CeRelationError> {
    let structure = prep.structure();
    let b = prep.params.b();
    let claim_start = builder.rows();
    let claim_first_column = builder.cols();
    validate_claim_shape(prep, index, claim_wires)?;

    let phase_start = builder.rows();
    enforce_ajtai_opening(
        builder,
        &prep.log,
        &witness_wires,
        &claim_wires.c_data,
        claim_wires.c_d,
        claim_wires.c_kappa,
    )
    .map_err(|err| ajtai_setup_err(index, err))?;

    match (prep.nebula(), claim_wires.adv.as_ref()) {
        (None, None) => {}
        (Some(nebula), Some(adv)) => {
            let ops_pp = nebula.scheme.ops_module().verification_pp().map_err(|_| {
                let (d, cols) = nebula.scheme.ops_module().dims();
                CeRelationError::AjtaiSetupMissing { d, cols }
            })?;
            let mem_pp = nebula.scheme.mem_module().verification_pp().map_err(|_| {
                let (d, cols) = nebula.scheme.mem_module().dims();
                CeRelationError::AjtaiSetupMissing { d, cols }
            })?;
            let ranges = nebula.scheme.ranges();
            for (commitment, columns, pp) in [
                (&adv.ops, ranges.ops.clone(), ops_pp.as_ref()),
                (&adv.is, ranges.is.clone(), mem_pp.as_ref()),
                (&adv.fs, ranges.fs.clone(), mem_pp.as_ref()),
            ] {
                enforce_ajtai_slice_opening(
                    builder,
                    &witness_wires,
                    &commitment.data,
                    commitment.d,
                    commitment.kappa,
                    columns,
                    pp,
                )
                .map_err(|err| ajtai_setup_err(index, err))?;
            }
        }
        _ => return Err(CeRelationError::NebulaAdvPresence { index }),
    }
    builder.record_row_family("terminal_ce.claim.commitment", phase_start);

    let phase_start = builder.rows();
    enforce_x_projection(builder, &witness_wires, claim_wires).map_err(|err| projection_err(index, err))?;
    builder.record_row_family("terminal_ce.claim.public_input", phase_start);

    // Low-norm: enforce every entry of `Z` lies in the SuperNeo
    // NC-bound alphabet `{-(b-1), …, +(b-1)}`, matching the native
    // `neo_math::balanced::within_nc_bound`'s `|x| < b` predicate.
    // Implemented as `Π_{a ∈ alphabet} (Z[i,j] - a) = 0`. See
    // [`witness::enforce_balanced_alphabet`].
    let phase_start = builder.rows();
    let norm_first_allocated_column = builder.cols();
    witness::enforce_balanced_alphabet(builder, &witness_wires, b)
        .map_err(|err| CeRelationError::InvalidNormBound { b: err.b })?;
    builder.record_row_family("terminal_ce.claim.norm", phase_start);

    // `enforce_y_ring_from_z_at_r` owns the exact `claim.r` length
    // guard (`|r| == log2(next_pow2(n).max(2))`), so there's no weaker
    // pre-check here.
    let phase_start = builder.rows();
    enforce_y_ring_from_z_at_r(builder, prep, &witness_wires, claim_wires).map_err(|err| y_ring_err(index, err))?;
    builder.record_row_family("terminal_ce.claim.evaluations", phase_start);
    // ct[j] is the constant-term lane of y_ring[j] (Paper Theorem 5).
    // Wire-equality binding `ct[j] == y_ring[j][lane=0]` closes the
    // CE-relation contract so the circuit matches the native
    // `verify_uncompressed` verifier's full obligation set.
    let phase_start = builder.rows();
    enforce_ct_from_y_ring(builder, claim_wires).map_err(|err| y_ring_err(index, err))?;
    builder.record_row_family("terminal_ce.claim.constant_term", phase_start);
    // A legacy full claim carries `s_col/y_zcol` together, so bind that
    // sidecar to this witness. Strict Π_DEC children deliberately omit
    // `y_zcol`; the production delayed path instead binds the separately
    // carried parent value to this same ordered witness family before entering
    // this function.
    let phase_start = builder.rows();
    match nc_channel_closure {
        NcChannelClosure::LegacyClaimSidecar => {
            enforce_y_zcol_from_z_at_s_col(builder, prep, &witness_wires, claim_wires)
                .map_err(|err| y_ring_err(index, err))?;
        }
        NcChannelClosure::StrictPiDecChild => {
            if !claim_wires.y_zcol.is_empty() || claim_wires.y_zcol_lanes != 0 {
                return Err(CeRelationError::ShapeMismatch {
                    index,
                    what: "strict PiDEC child y_zcol carrier",
                    expected: 0,
                    got: claim_wires.y_zcol.len().max(claim_wires.y_zcol_lanes),
                });
            }
        }
    }
    builder.record_row_family("terminal_ce.claim.nc_channel", phase_start);
    builder.record_terminal_ce_claim(TerminalCeClaimAudit {
        row_start: claim_start,
        row_end: builder.rows(),
        first_allocated_column: claim_first_column,
        norm_bound: b,
        expected_public_width: prep.public_input_len,
        structure_rows: structure.n,
        structure_columns: structure.m,
        witness_rows: witness_wires.rows,
        witness_columns: witness_wires.cols,
        witness_cols: witness_wires.values.iter().map(|wire| wire.col()).collect(),
        norm_first_allocated_column,
        commitment_cols: claim_wires.c_data.iter().map(|wire| wire.col()).collect(),
        commitment_d: claim_wires.c_d,
        commitment_kappa: claim_wires.c_kappa,
        public_cols: claim_wires.x.iter().map(|wire| wire.col()).collect(),
        public_rows: claim_wires.x_rows,
        public_width: claim_wires.x_cols,
        public_input_len: claim_wires.m_in,
        point_cols: claim_wires
            .r
            .iter()
            .map(|value| [value.c0.col(), value.c1.col()])
            .collect(),
        evaluation_cols: claim_wires
            .y_ring
            .iter()
            .map(|row| row.iter().map(|wire| wire.col()).collect())
            .collect(),
        constant_term_cols: claim_wires
            .ct
            .iter()
            .map(|value| [value.c0.col(), value.c1.col()])
            .collect(),
        nc_point_cols: claim_wires
            .s_col
            .iter()
            .map(|value| [value.c0.col(), value.c1.col()])
            .collect(),
        nc_evaluation_cols: claim_wires.y_zcol.iter().map(|wire| wire.col()).collect(),
        nc_evaluation_lanes: claim_wires.y_zcol_lanes,
    });
    builder.record_program_range("terminal_ce.claim", claim_start, claim_first_column);
    Ok(())
}

fn validate_count(final_claims_wires: &[CeClaimWires], final_witnesses: &[WitnessMat]) -> Result<(), CeRelationError> {
    if final_claims_wires.len() != final_witnesses.len() {
        return Err(CeRelationError::CountMismatch {
            claims: final_claims_wires.len(),
            witnesses: final_witnesses.len(),
        });
    }
    Ok(())
}

fn validate_pending_projection_inputs(
    prep: &Preprocessing,
    claims: &[CeClaimWires],
    witnesses: &[WitnessMat],
    pending: &crate::paper::reductions::pi_ccs_split_nc_circuit::PendingProjectionWires,
) -> Result<(), CeRelationError> {
    if witnesses.len() != RAW_OLD_BLOCK_CHILD_COUNT {
        return Err(CeRelationError::RawOldBlockProjectionShape {
            what: "ordered raw witness child count",
            expected: RAW_OLD_BLOCK_CHILD_COUNT,
            got: witnesses.len(),
        });
    }
    let logical_columns = prep.structure().m;
    let plan = RawOldBlockProjectionPlan::new(logical_columns, witnesses.len()).map_err(|_| {
        CeRelationError::RawOldBlockProjectionShape {
            what: "logical columns within the fixed block domain",
            expected: neo_reductions::block_projection::BLOCK_PROJECTION_DOMAIN_SIZE * neo_math::D,
            got: logical_columns,
        }
    })?;
    if pending.old_block.len() != plan.block_variables() {
        return Err(CeRelationError::RawOldBlockProjectionShape {
            what: "pending old-block coordinates",
            expected: plan.block_variables(),
            got: pending.old_block.len(),
        });
    }
    if pending.parent_y_zcol.len() != plan.active_lanes() {
        return Err(CeRelationError::RawOldBlockProjectionShape {
            what: "pending parent active lanes",
            expected: plan.active_lanes(),
            got: pending.parent_y_zcol.len(),
        });
    }
    for (index, witness) in witnesses.iter().enumerate() {
        if witness.rows() != plan.packed_rows() || witness.cols() != plan.packed_columns() {
            return Err(CeRelationError::ShapeMismatch {
                index,
                what: "raw witness packed entries",
                expected: plan.packed_rows() * plan.packed_columns(),
                got: witness.rows() * witness.cols(),
            });
        }
    }
    for (index, claim) in claims.iter().enumerate() {
        validate_claim_shape(prep, index, claim)?;
    }
    Ok(())
}

fn validate_claim_shape(prep: &Preprocessing, index: usize, claim_wires: &CeClaimWires) -> Result<(), CeRelationError> {
    let expected_m = prep.structure().m;
    if claim_wires.m_in > expected_m {
        return Err(CeRelationError::ShapeMismatch {
            index,
            what: "m_in vs structure.m",
            expected: expected_m,
            got: claim_wires.m_in,
        });
    }
    if let Some(expected) = prep.public_input_len {
        if claim_wires.m_in != expected {
            return Err(CeRelationError::ShapeMismatch {
                index,
                what: "m_in vs prep.public_input_len",
                expected,
                got: claim_wires.m_in,
            });
        }
    }
    reject_unsupported_sidecars(index, claim_wires)
}

fn raw_old_block_projection_err(err: old_block_projection::RawOldBlockProjectionError) -> CeRelationError {
    CeRelationError::RawOldBlockProjectionShape {
        what: err.what(),
        expected: err.expected(),
        got: err.got(),
    }
}

fn witness_shape_err(index: usize, err: witness::AllocError) -> CeRelationError {
    CeRelationError::ShapeMismatch {
        index,
        what: err.what(),
        expected: err.expected(),
        got: err.got(),
    }
}

fn ajtai_setup_err(index: usize, err: commitment::AjtaiOpeningError) -> CeRelationError {
    match err {
        commitment::AjtaiOpeningError::SetupMissing { d, cols } => CeRelationError::AjtaiSetupMissing { d, cols },
        commitment::AjtaiOpeningError::Shape { what, expected, got } => CeRelationError::ShapeMismatch {
            index,
            what,
            expected,
            got,
        },
    }
}

fn projection_err(index: usize, err: commitment::XProjectionError) -> CeRelationError {
    CeRelationError::ShapeMismatch {
        index,
        what: err.what(),
        expected: err.expected(),
        got: err.got(),
    }
}

fn y_ring_err(index: usize, err: evaluation::YRingError) -> CeRelationError {
    CeRelationError::ShapeMismatch {
        index,
        what: err.what(),
        expected: err.expected(),
        got: err.got(),
    }
}

fn reject_unsupported_sidecars(index: usize, claim: &CeClaimWires) -> Result<(), CeRelationError> {
    if claim.aux_openings_len != 0 {
        return Err(CeRelationError::ShapeMismatch {
            index,
            what: "aux_openings",
            expected: 0,
            got: claim.aux_openings_len,
        });
    }
    if claim.c_step_coords_len != 0 {
        return Err(CeRelationError::ShapeMismatch {
            index,
            what: "c_step_coords",
            expected: 0,
            got: claim.c_step_coords_len,
        });
    }
    if claim.u_offset != 0 {
        return Err(CeRelationError::ShapeMismatch {
            index,
            what: "u_offset",
            expected: 0,
            got: claim.u_offset,
        });
    }
    if claim.u_len != 0 {
        return Err(CeRelationError::ShapeMismatch {
            index,
            what: "u_len",
            expected: 0,
            got: claim.u_len,
        });
    }
    Ok(())
}

use crate::engine::r1cs_circuit::field_ext::KVar;
use crate::engine::r1cs_circuit::poseidon2::enforce_poseidon2_hash;
use crate::engine::r1cs_circuit::R1csBuilder;
use crate::engine::r1cs_circuit::{Lc, Var};
use crate::lifecycle::Preprocessing;
use crate::paper::digest::{
    pack_bytes_as_fields, params_digest, terminal_ce_relation_digest, NEBULA_ADV_PRESENT_MARKER, NEBULA_LEAF_MEM_TAG,
    NEBULA_LEAF_OPS_TAG,
};
use crate::paper::f_prime::nebula_lane_circuit::enforce_nebula_leaf_digest_circuit;
use crate::paper::params::Params;
use crate::paper::reductions::pi_dec_circuit::CeClaimWires;
use crate::paper::relations::product_commitment_circuit::validate_adv_shape;
use crate::paper::relations::{superneo_public_x_cols, Structure};
use crate::paper::terminal_ce::{TerminalCeProof, TerminalCeVerifyError};
use neo_math::{D, F};
use p3_field::PrimeCharacteristicRing;
use thiserror::Error;

const TERMINAL_CHILDREN_DIGEST_DOMAIN: &[u8] = b"neo.fold.clean/terminal_children_digest/v1";
const TERMINAL_CE_CLAIM_DIGEST_DOMAIN: &[u8] = b"neo.fold.clean/terminal_ce_claim_digest/v1";

#[derive(Clone, Debug)]
pub struct TerminalCePublicWires {
    pub relation_digest: [Var; 4],
    pub structure_digest: [Var; 4],
    pub params_digest: [Var; 4],
    pub terminal_children_digest: [Var; 4],
    pub public_digest: [Var; 4],
    pub claim_count: usize,
}

/// Verifier-owned context for the future compact terminal CE proof verifier.
///
/// The proof's public statement must be pinned to this context. It is not
/// enough for a prover to supply self-consistent public digest wires.
#[derive(Clone, Copy)]
pub struct TerminalCeVerifierContext<'a> {
    pub params: &'a Params,
    pub structure: &'a Structure,
    pub structure_digest: [F; 4],
}

impl<'a> TerminalCeVerifierContext<'a> {
    pub fn from_preprocessing(prep: &'a Preprocessing) -> Self {
        Self {
            params: &prep.params,
            structure: prep.structure(),
            structure_digest: *prep.structure_digest(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum TerminalCeCircuitError {
    #[error("terminal CE public circuit: child {index} X.rows ({got}) must equal D ({expected})")]
    XRows {
        index: usize,
        expected: usize,
        got: usize,
    },
    #[error("terminal CE public circuit: child {index} commitment d ({got}) must equal D ({expected})")]
    CommitmentD {
        index: usize,
        expected: usize,
        got: usize,
    },
    #[error("terminal CE public circuit: child {index} commitment kappa ({got}) must equal params.kappa ({expected})")]
    CommitmentKappa {
        index: usize,
        expected: usize,
        got: usize,
    },
    #[error(
        "terminal CE public circuit: child {index} commitment data length ({got}) must equal d*kappa ({expected})"
    )]
    CommitmentDataLen {
        index: usize,
        expected: usize,
        got: usize,
    },
    #[error("terminal CE public circuit: child {index} X flat length ({got}) must equal rows*cols ({expected})")]
    XFlatLen {
        index: usize,
        expected: usize,
        got: usize,
    },
    #[error("terminal CE public circuit: child {index} X.cols ({got}) must equal m_in ({expected})")]
    XCols {
        index: usize,
        expected: usize,
        got: usize,
    },
    #[error("terminal CE public circuit: child {index} m_in ({got}) must not exceed structure.m ({expected})")]
    MInExceedsStructureM {
        index: usize,
        expected: usize,
        got: usize,
    },
    #[error("terminal CE public circuit: child {index} active X cols ({active_cols}) exceed X.cols ({cols})")]
    ActiveXCols {
        index: usize,
        active_cols: usize,
        cols: usize,
    },
    #[error("terminal CE public circuit: child {index} r length ({got}) must equal row-domain length ({expected})")]
    RLen {
        index: usize,
        expected: usize,
        got: usize,
    },
    #[error(
        "terminal CE public circuit: child {index} s_col length ({got}) must equal column-domain length ({expected})"
    )]
    SColLen {
        index: usize,
        expected: usize,
        got: usize,
    },
    #[error("terminal CE public circuit: child {index} y_ring length ({got}) must equal structure.t ({expected})")]
    YRingCount {
        index: usize,
        expected: usize,
        got: usize,
    },
    #[error("terminal CE public circuit: child {index} y_ring row has {got} limbs, expected {expected}")]
    YRingFlatLen {
        index: usize,
        expected: usize,
        got: usize,
    },
    #[error("terminal CE public circuit: child {index} ct length ({got}) must equal y_ring length ({expected})")]
    CtLen {
        index: usize,
        expected: usize,
        got: usize,
    },
    #[error("terminal CE public circuit: child {index} y_zcol lanes ({got}) must equal padded D ({expected})")]
    YZcolLanes {
        index: usize,
        expected: usize,
        got: usize,
    },
    #[error("terminal CE public circuit: child {index} y_zcol has {got} limbs, expected {expected}")]
    YZcolFlatLen {
        index: usize,
        expected: usize,
        got: usize,
    },
    #[error("terminal CE public circuit: child {index} carries unsupported {field} (expected empty/zero)")]
    UnsupportedSidecar { index: usize, field: &'static str },
    #[error("terminal CE public circuit: child {index} invalid product commitment: {detail}")]
    ProductCommitment { index: usize, detail: String },
}

/// Recompute the compact terminal CE public statement from terminal child
/// wires produced by the NIFS/Π_DEC verifier.
///
/// This is the public-input constructor the eventual compact verifier must
/// use. It deliberately rejects the same unsupported sidecars as the direct CE
/// rows; no digest-only authority is introduced here.
pub fn enforce_public_from_children(
    builder: &mut R1csBuilder,
    context: &TerminalCeVerifierContext<'_>,
    terminal_children: &[CeClaimWires],
) -> Result<TerminalCePublicWires, TerminalCeCircuitError> {
    let relation_digest = alloc_const_array(builder, terminal_ce_relation_digest());
    let structure_digest = alloc_const_array(builder, context.structure_digest);
    let params_digest = alloc_const_array(builder, params_digest(context.params.inner()));
    let terminal_children_digest = enforce_terminal_children_digest(builder, context, terminal_children)?;
    let claim_count = terminal_children.len();
    let public_digest = enforce_terminal_ce_public_digest(
        builder,
        relation_digest,
        structure_digest,
        params_digest,
        terminal_children_digest,
        claim_count,
    );
    Ok(TerminalCePublicWires {
        relation_digest,
        structure_digest,
        params_digest,
        terminal_children_digest,
        public_digest,
        claim_count,
    })
}

/// Recompute the terminal-CE public statement from NIFS-output child wires,
/// then verify a compact terminal CE proof inside the decider circuit.
///
/// Fail-closed until the real proof-system verifier is implemented. The direct
/// `paper::decider_ce_relation` rows remain the production soundness contract.
pub fn enforce_verify_from_children(
    builder: &mut R1csBuilder,
    context: &TerminalCeVerifierContext<'_>,
    terminal_children: &[CeClaimWires],
    proof: &TerminalCeProof,
) -> Result<TerminalCePublicWires, TerminalCeVerifyError> {
    let public = enforce_public_from_children(builder, context, terminal_children)
        .map_err(|e| TerminalCeVerifyError::PublicStatement(e.to_string()))?;
    enforce_verify_public_statement(builder, context, &public, proof)?;
    Ok(public)
}

fn enforce_verify_public_statement(
    builder: &mut R1csBuilder,
    context: &TerminalCeVerifierContext<'_>,
    public: &TerminalCePublicWires,
    proof: &TerminalCeProof,
) -> Result<(), TerminalCeVerifyError> {
    enforce_digest_eq_const(builder, public.relation_digest, terminal_ce_relation_digest());
    enforce_digest_eq_const(builder, public.structure_digest, context.structure_digest);
    enforce_digest_eq_const(builder, public.params_digest, params_digest(context.params.inner()));
    enforce_digest_eq_const(builder, public.public_digest, proof.public_digest());
    Err(TerminalCeVerifyError::Unsupported)
}

fn enforce_terminal_children_digest(
    builder: &mut R1csBuilder,
    context: &TerminalCeVerifierContext<'_>,
    terminal_children: &[CeClaimWires],
) -> Result<[Var; 4], TerminalCeCircuitError> {
    let mut preimage = Vec::new();
    extend_packed_bytes_as_fields_wires(builder, &mut preimage, TERMINAL_CHILDREN_DIGEST_DOMAIN);
    preimage.push(alloc_const(builder, F::from_u64(terminal_children.len() as u64)));
    for (index, claim) in terminal_children.iter().enumerate() {
        preimage.extend_from_slice(&enforce_terminal_ce_claim_digest(builder, context, index, claim)?);
    }
    Ok(enforce_poseidon2_hash(builder, &preimage))
}

fn enforce_terminal_ce_public_digest(
    builder: &mut R1csBuilder,
    relation_digest: [Var; 4],
    structure_digest: [Var; 4],
    params_digest: [Var; 4],
    terminal_children_digest: [Var; 4],
    claim_count: usize,
) -> [Var; 4] {
    let mut preimage = Vec::new();
    extend_packed_bytes_as_fields_wires(builder, &mut preimage, b"neo.fold.clean/terminal_ce_public/v1");
    preimage.extend_from_slice(&relation_digest);
    preimage.extend_from_slice(&structure_digest);
    preimage.extend_from_slice(&params_digest);
    preimage.extend_from_slice(&terminal_children_digest);
    preimage.push(alloc_const(builder, F::from_u64(claim_count as u64)));
    enforce_poseidon2_hash(builder, &preimage)
}

fn enforce_terminal_ce_claim_digest(
    builder: &mut R1csBuilder,
    context: &TerminalCeVerifierContext<'_>,
    index: usize,
    claim: &CeClaimWires,
) -> Result<[Var; 4], TerminalCeCircuitError> {
    validate_terminal_child_wires(builder, context, index, claim)?;

    let active_x_cols = superneo_public_x_cols(claim.m_in);
    let mut preimage = Vec::new();
    extend_packed_bytes_as_fields_wires(builder, &mut preimage, TERMINAL_CE_CLAIM_DIGEST_DOMAIN);

    preimage.push(alloc_const(builder, F::from_u64(claim.c_d as u64)));
    preimage.push(alloc_const(builder, F::from_u64(claim.c_kappa as u64)));
    extend_f_slice_wires(builder, &mut preimage, &claim.c_data);

    preimage.push(alloc_const(builder, F::from_u64(claim.x_rows as u64)));
    preimage.push(alloc_const(builder, F::from_u64(claim.x_cols as u64)));
    preimage.push(alloc_const(builder, F::from_u64(active_x_cols as u64)));
    for r in 0..claim.x_rows {
        for c in 0..active_x_cols {
            preimage.push(claim.x[r * claim.x_cols + c]);
        }
    }

    extend_kvar_slice(builder, &mut preimage, &claim.r);
    extend_kvar_slice(builder, &mut preimage, &claim.s_col);
    preimage.push(alloc_const(builder, F::from_u64(claim.y_ring.len() as u64)));
    for row in &claim.y_ring {
        preimage.push(alloc_const(builder, F::from_u64(claim.y_ring_lanes as u64)));
        preimage.extend_from_slice(row);
    }
    extend_kvar_slice(builder, &mut preimage, &claim.ct);
    extend_flat_k_limb_slice(builder, &mut preimage, claim.y_zcol_lanes, &claim.y_zcol);
    // aux_openings are unsupported in the current compact/public circuit path.
    preimage.push(alloc_const(builder, F::ZERO));
    preimage.push(alloc_const(builder, F::from_u64(claim.m_in as u64)));
    preimage.extend_from_slice(&claim.fold_digest_fields);
    // c_step_coords.len, u_offset, u_len are unsupported and validated zero.
    preimage.push(alloc_const(builder, F::ZERO));
    preimage.push(alloc_const(builder, F::ZERO));
    preimage.push(alloc_const(builder, F::ZERO));
    if let Some(adv) = &claim.adv {
        preimage.push(alloc_const(builder, F::from_u64(NEBULA_ADV_PRESENT_MARKER)));
        for digest in [
            enforce_nebula_leaf_digest_circuit(builder, NEBULA_LEAF_OPS_TAG, adv.ops.d, adv.ops.kappa, &adv.ops.data),
            enforce_nebula_leaf_digest_circuit(builder, NEBULA_LEAF_MEM_TAG, adv.is.d, adv.is.kappa, &adv.is.data),
            enforce_nebula_leaf_digest_circuit(builder, NEBULA_LEAF_MEM_TAG, adv.fs.d, adv.fs.kappa, &adv.fs.data),
        ] {
            preimage.extend_from_slice(&digest);
        }
    }

    Ok(enforce_poseidon2_hash(builder, &preimage))
}

fn validate_terminal_child_wires(
    builder: &mut R1csBuilder,
    context: &TerminalCeVerifierContext<'_>,
    index: usize,
    claim: &CeClaimWires,
) -> Result<(), TerminalCeCircuitError> {
    if claim.c_d != D {
        return Err(TerminalCeCircuitError::CommitmentD {
            index,
            expected: D,
            got: claim.c_d,
        });
    }
    let expected_kappa = context.params.kappa() as usize;
    if claim.c_kappa != expected_kappa {
        return Err(TerminalCeCircuitError::CommitmentKappa {
            index,
            expected: expected_kappa,
            got: claim.c_kappa,
        });
    }
    let expected_c_len = claim.c_d * claim.c_kappa;
    if claim.c_data.len() != expected_c_len {
        return Err(TerminalCeCircuitError::CommitmentDataLen {
            index,
            expected: expected_c_len,
            got: claim.c_data.len(),
        });
    }
    validate_adv_shape(claim.adv.as_ref(), claim.c_d, claim.c_kappa, "terminal child")
        .map_err(|detail| TerminalCeCircuitError::ProductCommitment { index, detail })?;
    if claim.x_rows != D {
        return Err(TerminalCeCircuitError::XRows {
            index,
            expected: D,
            got: claim.x_rows,
        });
    }
    let expected_x_len = claim.x_rows * claim.x_cols;
    if claim.x.len() != expected_x_len {
        return Err(TerminalCeCircuitError::XFlatLen {
            index,
            expected: expected_x_len,
            got: claim.x.len(),
        });
    }
    if claim.x_cols != claim.m_in {
        return Err(TerminalCeCircuitError::XCols {
            index,
            expected: claim.m_in,
            got: claim.x_cols,
        });
    }
    if claim.m_in > context.structure.m {
        return Err(TerminalCeCircuitError::MInExceedsStructureM {
            index,
            expected: context.structure.m,
            got: claim.m_in,
        });
    }
    let active_x_cols = superneo_public_x_cols(claim.m_in);
    if active_x_cols > claim.x_cols {
        return Err(TerminalCeCircuitError::ActiveXCols {
            index,
            active_cols: active_x_cols,
            cols: claim.x_cols,
        });
    }
    let expected_r_len = context
        .structure
        .n
        .next_power_of_two()
        .max(2)
        .trailing_zeros() as usize;
    if claim.r.len() != expected_r_len {
        return Err(TerminalCeCircuitError::RLen {
            index,
            expected: expected_r_len,
            got: claim.r.len(),
        });
    }
    let expected_s_col_len = context
        .structure
        .m
        .next_power_of_two()
        .max(2)
        .trailing_zeros() as usize;
    if claim.s_col.len() != expected_s_col_len {
        return Err(TerminalCeCircuitError::SColLen {
            index,
            expected: expected_s_col_len,
            got: claim.s_col.len(),
        });
    }
    for r in 0..claim.x_rows {
        for c in active_x_cols..claim.x_cols {
            builder.enforce_eq(&Lc::from_var(claim.x[r * claim.x_cols + c]), &Lc::zero());
        }
    }
    let expected_t = context.structure.t();
    if claim.y_ring.len() != expected_t {
        return Err(TerminalCeCircuitError::YRingCount {
            index,
            expected: expected_t,
            got: claim.y_ring.len(),
        });
    }
    for row in &claim.y_ring {
        let expected = claim.y_ring_lanes * 2;
        if row.len() != expected {
            return Err(TerminalCeCircuitError::YRingFlatLen {
                index,
                expected,
                got: row.len(),
            });
        }
    }
    let expected_y_ring_limbs = D.next_power_of_two() * 2;
    for row in &claim.y_ring {
        if row.len() != expected_y_ring_limbs {
            return Err(TerminalCeCircuitError::YRingFlatLen {
                index,
                expected: expected_y_ring_limbs,
                got: row.len(),
            });
        }
    }
    if claim.ct.len() != claim.y_ring.len() {
        return Err(TerminalCeCircuitError::CtLen {
            index,
            expected: expected_t,
            got: claim.ct.len(),
        });
    }
    for (ct, row) in claim.ct.iter().zip(claim.y_ring.iter()) {
        builder.enforce_eq(&Lc::from_var(ct.c0), &Lc::from_var(row[0]));
        builder.enforce_eq(&Lc::from_var(ct.c1), &Lc::from_var(row[1]));
        for limb in row.iter().skip(D * 2) {
            builder.enforce_eq(&Lc::from_var(*limb), &Lc::zero());
        }
    }
    if claim.y_zcol_lanes != D.next_power_of_two() {
        return Err(TerminalCeCircuitError::YZcolLanes {
            index,
            expected: D.next_power_of_two(),
            got: claim.y_zcol_lanes,
        });
    }
    let expected_y_zcol_len = claim.y_zcol_lanes * 2;
    if claim.y_zcol.len() != expected_y_zcol_len {
        return Err(TerminalCeCircuitError::YZcolFlatLen {
            index,
            expected: expected_y_zcol_len,
            got: claim.y_zcol.len(),
        });
    }
    for limb in claim.y_zcol.iter().skip(D * 2) {
        builder.enforce_eq(&Lc::from_var(*limb), &Lc::zero());
    }
    if claim.aux_openings_len != 0 {
        return Err(TerminalCeCircuitError::UnsupportedSidecar {
            index,
            field: "aux_openings",
        });
    }
    if claim.c_step_coords_len != 0 {
        return Err(TerminalCeCircuitError::UnsupportedSidecar {
            index,
            field: "c_step_coords",
        });
    }
    if claim.u_offset != 0 {
        return Err(TerminalCeCircuitError::UnsupportedSidecar {
            index,
            field: "u_offset",
        });
    }
    if claim.u_len != 0 {
        return Err(TerminalCeCircuitError::UnsupportedSidecar { index, field: "u_len" });
    }
    Ok(())
}

fn extend_packed_bytes_as_fields_wires(builder: &mut R1csBuilder, out: &mut Vec<Var>, bytes: &[u8]) {
    for value in pack_bytes_as_fields(bytes) {
        out.push(alloc_const(builder, value));
    }
}

fn extend_f_slice_wires(builder: &mut R1csBuilder, out: &mut Vec<Var>, values: &[Var]) {
    out.push(alloc_const(builder, F::from_u64(values.len() as u64)));
    out.extend_from_slice(values);
}

fn extend_kvar_slice(builder: &mut R1csBuilder, out: &mut Vec<Var>, values: &[KVar]) {
    out.push(alloc_const(builder, F::from_u64(values.len() as u64)));
    for value in values {
        out.push(value.c0);
        out.push(value.c1);
    }
}

fn extend_flat_k_limb_slice(builder: &mut R1csBuilder, out: &mut Vec<Var>, lanes: usize, limbs: &[Var]) {
    out.push(alloc_const(builder, F::from_u64(lanes as u64)));
    out.extend_from_slice(limbs);
}

fn alloc_const_array(builder: &mut R1csBuilder, values: [F; 4]) -> [Var; 4] {
    [
        alloc_const(builder, values[0]),
        alloc_const(builder, values[1]),
        alloc_const(builder, values[2]),
        alloc_const(builder, values[3]),
    ]
}

fn alloc_const(builder: &mut R1csBuilder, value: F) -> Var {
    let var = builder.alloc(value);
    builder.enforce_eq(&Lc::from_var(var), &Lc::from_const(value));
    var
}

fn enforce_digest_eq_const(builder: &mut R1csBuilder, digest: [Var; 4], expected: [F; 4]) {
    for (wire, value) in digest.into_iter().zip(expected) {
        builder.enforce_eq(&Lc::from_var(wire), &Lc::from_const(value));
    }
}

//! Construction of Pi_CCS terminal claim objects from backend-owned surfaces.

use neo_ajtai::Commitment as Cmt;
use neo_ccs::{CcsClaim, CcsStructure, CcsWitness, CeClaim};
use neo_math::{F, K};
use neo_params::NeoParams;

use super::backend::PiCcsTerminalOutputSurfaces;
use crate::error::PiCcsError;

#[allow(clippy::too_many_arguments)]
pub(super) fn build_me_outputs_from_terminal_surfaces(
    params: &NeoParams,
    s: &CcsStructure<F>,
    mcs_list: &[CcsClaim<Cmt, F>],
    mcs_witnesses: &[CcsWitness<F>],
    me_inputs: &[CeClaim<Cmt, F, K>],
    row_chals: &[K],
    s_col: &[K],
    surfaces: PiCcsTerminalOutputSurfaces,
    fold_digest: [u8; 32],
) -> Result<Vec<CeClaim<Cmt, F, K>>, PiCcsError> {
    if mcs_list.len() != mcs_witnesses.len() {
        return Err(PiCcsError::InvalidInput(
            "backend terminal output MCS claim/witness count mismatch".into(),
        ));
    }
    let expected = mcs_list.len() + me_inputs.len();
    if surfaces.y_ring.len() != expected {
        return Err(PiCcsError::InvalidInput(
            "backend terminal y_ring surface count mismatch".into(),
        ));
    }
    if let Some(y_zcol) = surfaces.y_zcol.as_ref() {
        if y_zcol.len() != expected {
            return Err(PiCcsError::InvalidInput(
                "backend terminal y_zcol surface count mismatch".into(),
            ));
        }
    } else if !s_col.is_empty() {
        return Err(PiCcsError::InvalidInput(
            "backend terminal outputs omitted active y_zcol surface".into(),
        ));
    }

    let mut out = Vec::with_capacity(expected);
    for (idx, (claim, witness)) in mcs_list.iter().zip(mcs_witnesses.iter()).enumerate() {
        let y_ring = surfaces.y_ring[idx].clone();
        let ct = crate::common::ct_from_y_ring_for_ccs_m(&y_ring, params, s.m);
        let x = crate::common::project_x_from_witness_mat(&witness.Z, s.m, claim.m_in)?;
        out.push(CeClaim {
            adv: None,
            c_step_coords: vec![],
            u_offset: 0,
            u_len: 0,
            c: claim.c.clone(),
            X: x,
            r: row_chals.to_vec(),
            s_col: s_col.to_vec(),
            y_ring,
            ct,
            aux_openings: Vec::new(),
            y_zcol: surfaces
                .y_zcol
                .as_ref()
                .map(|rows| rows[idx].clone())
                .unwrap_or_default(),
            m_in: claim.m_in,
            fold_digest,
        });
    }
    let offset = mcs_list.len();
    for (idx, claim) in me_inputs.iter().enumerate() {
        let surface_idx = offset + idx;
        let y_ring = surfaces.y_ring[surface_idx].clone();
        let ct = crate::common::ct_from_y_ring_for_ccs_m(&y_ring, params, s.m);
        out.push(CeClaim {
            adv: None,
            c_step_coords: vec![],
            u_offset: 0,
            u_len: 0,
            c: claim.c.clone(),
            X: claim.X.clone(),
            r: row_chals.to_vec(),
            s_col: s_col.to_vec(),
            y_ring,
            ct,
            aux_openings: Vec::new(),
            y_zcol: surfaces
                .y_zcol
                .as_ref()
                .map(|rows| rows[surface_idx].clone())
                .unwrap_or_default(),
            m_in: claim.m_in,
            fold_digest,
        });
    }
    Ok(out)
}

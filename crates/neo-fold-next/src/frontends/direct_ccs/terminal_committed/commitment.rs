use neo_ajtai::{
    get_global_pp_seeded_params_for_dims, has_global_pp_for_dims, set_global_pp_seeded, AjtaiSModule, Commitment,
};
use neo_ccs::{traits::SModuleHomomorphism, Mat};
use neo_math::{D, F};
use neo_params::NeoParams;
use neo_transcript::{Poseidon2Transcript, Transcript};

use super::types::SimpleKernelError;
use crate::construction2::Construction2PublicBoundary;
use crate::construction2_terminal::{terminal_boundary_public_values, Construction2TerminalBoundaryView};
use crate::spartan_backend::SpartanF;
use crate::witness_layout::commit_cols_for_full_width;

pub(super) fn terminal_committed_boundary_public_values(boundary: &Construction2PublicBoundary) -> Vec<SpartanF> {
    terminal_boundary_public_values(&direct_terminal_boundary_view(boundary))
}

pub(super) fn direct_terminal_boundary_view(
    boundary: &Construction2PublicBoundary,
) -> Construction2TerminalBoundaryView<'_> {
    Construction2TerminalBoundaryView {
        fresh_instance_digest: boundary.fresh_instance_digest,
        commitment_digest: boundary.commitment_digest,
        commitment_d: boundary.commitment_d,
        commitment_kappa: boundary.commitment_kappa,
        commitment_data: &boundary.commitment_data,
        x_i_bytes: boundary.x_i.bytes(),
    }
}

pub(super) fn direct_terminal_commit_packed_z(
    full_width: usize,
    packed_z: &Mat<F>,
) -> Result<Commitment, SimpleKernelError> {
    let params = NeoParams::goldilocks_auto_r1cs_ccs(full_width).map_err(|err| {
        SimpleKernelError::Bridge(format!(
            "direct terminal commitment params failed for width {full_width}: {err}"
        ))
    })?;
    let m = commit_cols_for_full_width(full_width);
    let want_kappa = params.kappa as usize;
    if has_global_pp_for_dims(D, m) {
        let (kappa, _) = get_global_pp_seeded_params_for_dims(D, m).map_err(|err| {
            SimpleKernelError::Bridge(format!("direct terminal commitment PP registry read failed: {err}"))
        })?;
        if kappa != want_kappa {
            return Err(SimpleKernelError::Bridge(format!(
                "direct terminal commitment PP mismatch for (d,m)=({D},{m}): registered kappa={kappa}, want {want_kappa}"
            )));
        }
    } else {
        set_global_pp_seeded(D, want_kappa, m, direct_terminal_commitment_seed(full_width)).map_err(|err| {
            SimpleKernelError::Bridge(format!(
                "direct terminal commitment PP setup failed for (d,m)=({D},{m}): {err}"
            ))
        })?;
    }
    if packed_z.rows() != D || packed_z.cols() != m {
        return Err(SimpleKernelError::Bridge(format!(
            "direct terminal packed Z shape mismatch: got {}x{}, expected {D}x{m}",
            packed_z.rows(),
            packed_z.cols()
        )));
    }
    let log = AjtaiSModule::from_global_for_dims(D, m).map_err(|err| {
        SimpleKernelError::Bridge(format!(
            "direct terminal commitment module failed for (d,m)=({D},{m}): {err}"
        ))
    })?;
    Ok(log.commit(packed_z))
}

fn direct_terminal_commitment_seed(full_width: usize) -> [u8; 32] {
    let mut tr = Poseidon2Transcript::new(b"neo.fold.next/direct_ccs/construction2_commitment_seed");
    tr.append_message(b"neo.fold.next/direct_ccs/construction2_commitment_seed/version", b"v1");
    tr.append_u64s(
        b"neo.fold.next/direct_ccs/construction2_commitment_seed/full_width",
        &[full_width as u64],
    );
    tr.digest32()
}

//! Nebula Pi_RLC projection-binding compatibility bridge.
//!
//! Owns the temporary host materialization needed to replay Nebula's
//! canonical post-rho projection schedule. It does not own the schedule or
//! its SIS digest; those remain defined by `neo-fold-clean`.

use std::sync::Arc;

use neo_fold_clean::engine::transcript::Transcript;
use neo_fold_clean::paper::nifs::Error;
use neo_fold_clean::paper::params::Params;
use neo_fold_clean::paper::pi_rlc;
use neo_fold_clean::paper::reductions::accumulator_sis_circuit::PI_RLC_PROJECTION_SIS_CONFIG;
use neo_fold_clean::paper::relations::{CeClaim, RlcMixer, Structure};

use crate::fold_output::DeviceCommitments;
use crate::reduce::ccs::{DevicePiCcsKSurfaces, DevicePublicX};
use crate::reduce::rlc::DeviceRhos;
use crate::session::{backend_unavailable, DeviceSession};

pub(crate) fn materialize_inputs(
    shells: &[CeClaim],
    commitments: Option<&Arc<DeviceCommitments>>,
    public_x: Option<&DevicePublicX>,
    surfaces: Option<&DevicePiCcsKSurfaces>,
) -> Result<Vec<CeClaim>, Error> {
    match (commitments, public_x, surfaces) {
        (Some(commitments), Some(public_x), Some(surfaces)) => {
            let mut inputs = shells.to_vec();
            surfaces
                .materialize_claims(&mut inputs)
                .map_err(|_| backend_unavailable("Pi_RLC projection input surface download failed"))?;
            public_x
                .materialize_claims(&mut inputs)
                .map_err(|_| backend_unavailable("Pi_RLC projection input X download failed"))?;
            let commitments = commitments.materialize()?;
            if commitments.len() != inputs.len() {
                return Err(backend_unavailable("Pi_RLC projection input commitment count mismatch"));
            }
            for (input, commitment) in inputs.iter_mut().zip(commitments) {
                input.c = commitment;
            }
            Ok(inputs)
        }
        (None, None, _) => Ok(shells.to_vec()),
        _ => Err(backend_unavailable("Pi_RLC projection input authority incomplete")),
    }
}

pub(crate) fn materialize_parent(
    shell: &CeClaim,
    surfaces: Option<&DevicePiCcsKSurfaces>,
    public_x: Option<&DevicePublicX>,
    commitment: Option<&Arc<DeviceCommitments>>,
) -> Result<CeClaim, Error> {
    match (surfaces, public_x, commitment) {
        (Some(surfaces), Some(public_x), Some(commitment)) => {
            let mut claims = vec![shell.clone()];
            surfaces
                .materialize_claims(&mut claims)
                .map_err(|_| backend_unavailable("Pi_RLC projection parent surface download failed"))?;
            public_x
                .materialize_claims(&mut claims)
                .map_err(|_| backend_unavailable("Pi_RLC projection parent X download failed"))?;
            claims[0].c = commitment
                .materialize()?
                .into_iter()
                .next()
                .ok_or_else(|| backend_unavailable("Pi_RLC projection parent commitment missing"))?;
            Ok(claims.pop().expect("one Pi_RLC projection parent"))
        }
        (None, None, None) => Ok(shell.clone()),
        _ => Err(backend_unavailable("Pi_RLC projection parent authority incomplete")),
    }
}

pub(crate) fn bind_schedule(
    tr: &mut Transcript,
    pp: &Params,
    structure: &Structure,
    mix: RlcMixer,
    session: &mut DeviceSession,
    device_rhos: &mut DeviceRhos,
    inputs: &[CeClaim],
    combined: &CeClaim,
) -> Result<(), Error> {
    let rho_mats = device_rhos
        .mats(&session.device, pp)
        .map_err(|_| backend_unavailable("Pi_RLC projection rho materialization failed"))?;
    let rhos = neo_reductions::common::rot_rhos_from_mats(pp.inner(), &rho_mats, "CUDA Pi_RLC projection schedule")
        .map_err(|_| backend_unavailable("Pi_RLC projection rho validation failed"))?;
    pi_rlc::validate_combined(structure, mix, &rhos, inputs, combined)?;
    if combined.adv.is_some() {
        pi_rlc::bind_backend_projection_schedule_with_digest(tr, &rhos, inputs, combined, |preimage| {
            session
                .sis_digest_host(PI_RLC_PROJECTION_SIS_CONFIG, preimage)
                .map_err(|_| pi_rlc::Error::BackendProjectionDigest)
        })?;
    } else {
        // At current eight-chain scale the independent host SIS jobs overlap
        // better than several underfilled device Ajtai maps. Keep the device
        // path for Nebula's larger adv-bearing schedule, where it is already
        // a measured win, and avoid regressing the plain aggregate contract.
        pi_rlc::bind_backend_projection_schedule(tr, &rhos, inputs, combined)?;
    }
    Ok(())
}

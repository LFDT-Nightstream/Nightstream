//! Metal round evaluation and child openings for the selected C/R/D fold.
//! The host keeps transcript order and public checks; device arithmetic also
//! generates the fixed key and computes child commitments.

use crate::folding::{
    self, ajtai_dec_mixer, ajtai_rlc_mixer, pi_ccs, pi_dec, pi_rlc, transcript::Transcript, CcsClaim, CcsInstance,
    CcsWitness, NifsProof, Params, RunningInstance, Structure,
};
use neo_ajtai::{scale_commitment_add_inplace, Commitment};
use neo_math::{D, F};
use neo_prover_metal::MetalRowProver;
use neo_reductions::{optimized_engine::optimized_prove_with_matrix_rows, superneo_eval::MatrixRows};
use p3_field::PrimeCharacteristicRing;

pub(crate) fn prove(
    device: &mut MetalRowProver,
    transcript: &mut Transcript,
    params: &Params,
    structure: &Structure,
    rows: &dyn MatrixRows,
    workspace_bytes: usize,
    fresh: Vec<CcsInstance>,
    running: RunningInstance,
) -> Result<(RunningInstance, NifsProof), folding::Error> {
    let (claims, witnesses): (Vec<_>, Vec<_>) = fresh
        .into_iter()
        .map(|source| (source.claim, source.witness))
        .unzip();
    let c = prove_pi_ccs(
        device,
        transcript,
        params,
        structure,
        rows,
        workspace_bytes,
        &claims,
        &witnesses,
        &running,
    )?;
    let sources: Vec<_> = witnesses
        .iter()
        .map(|witness| &witness.Z)
        .chain(running.witnesses.iter())
        .collect();
    // The parent witness stays on the device; only its PiDEC digits return.
    let (parent, split, r) = {
        let device = &*device;
        pi_rlc::prove_refs_resident(
            transcript,
            params,
            structure,
            ajtai_rlc_mixer,
            &c.outputs,
            &sources,
            |rhos, witnesses| device.split_rlc_witnesses(rhos, witnesses, params.k_rho() as usize, params.b()),
        )?
    };
    drop(sources);
    drop(witnesses);
    drop(running);
    let (digits, flags) = split
        .map_err(folding::kernels::Error::from)
        .map_err(pi_dec::Error::from)?;
    // The device commits children 1..; child 0 follows from the parent by
    // linearity. The terminal verifier still commits every child itself.
    let mut commitments = device
        .commit_production_prefixes(&digits[1..])
        .map_err(folding::kernels::Error::from)
        .map_err(pi_dec::Error::from)?;
    commitments.insert(0, lowest_child_commitment(&parent.c, &commitments, params.b()));
    let openings = device
        .child_openings(rows, workspace_bytes, &digits, &parent.r, structure.m)
        .map_err(folding::kernels::Error::from)
        .map_err(pi_dec::Error::from)?;
    let (children, ok_y, ok_x, ok_c) =
        neo_reductions::api::dec_children_with_commit_superneo_cached_from_trusted_split_digits(
            neo_reductions::api::FoldingMode::Optimized,
            structure,
            params.inner(),
            &parent,
            &digits,
            &flags,
            D.next_power_of_two().trailing_zeros() as usize,
            &commitments,
            ajtai_dec_mixer,
            None,
            None,
            Some(&openings),
        );
    if !(ok_y && ok_x && ok_c) {
        return Err(pi_dec::Error::Engine(folding::kernels::Error::PiDecPublicCheckFailed { ok_y, ok_x, ok_c }).into());
    }
    let d = pi_dec::Proof { children };
    let children = pi_dec::verify(params, structure, ajtai_dec_mixer, &parent, &d)?;
    Ok((
        RunningInstance::new(children, digits, Some(parent)),
        NifsProof {
            pi_ccs: c,
            pi_rlc: r,
            pi_dec: d,
        },
    ))
}

/// One fold up to its parent, for compression: PiCCS with its round oracle
/// on the device, then PiRLC on the host, which returns the parent witness.
/// The same outputs as `folding::prove_parent_with_rows`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn prove_parent(
    device: &mut MetalRowProver,
    transcript: &mut Transcript,
    params: &Params,
    structure: &Structure,
    rows: &dyn MatrixRows,
    workspace_bytes: usize,
    fresh: Vec<CcsInstance>,
    running: RunningInstance,
) -> Result<(pi_ccs::Proof, pi_rlc::Output, pi_rlc::Proof), folding::Error> {
    let (claims, witnesses): (Vec<_>, Vec<_>) = fresh
        .into_iter()
        .map(|source| (source.claim, source.witness))
        .unzip();
    let c = prove_pi_ccs(
        device,
        transcript,
        params,
        structure,
        rows,
        workspace_bytes,
        &claims,
        &witnesses,
        &running,
    )?;
    let sources: Vec<_> = witnesses
        .iter()
        .map(|witness| &witness.Z)
        .chain(running.witnesses.iter())
        .collect();
    let (parent, r) = pi_rlc::prove_refs(transcript, params, structure, ajtai_rlc_mixer, &c.outputs, &sources)?;
    Ok((c, parent, r))
}

/// PiCCS with its round oracle on the device.
#[allow(clippy::too_many_arguments)]
fn prove_pi_ccs(
    device: &mut MetalRowProver,
    transcript: &mut Transcript,
    params: &Params,
    structure: &Structure,
    rows: &dyn MatrixRows,
    workspace_bytes: usize,
    claims: &[CcsClaim],
    witnesses: &[CcsWitness],
    running: &RunningInstance,
) -> Result<pi_ccs::Proof, folding::Error> {
    let (outputs, sumcheck, _, _) = optimized_prove_with_matrix_rows(
        transcript.inner_mut(),
        params.inner(),
        structure,
        claims,
        witnesses,
        &running.claims,
        &running.witnesses,
        rows,
        workspace_bytes,
        Some(device),
    )
    .map_err(folding::kernels::Error::from)
    .map_err(pi_ccs::Error::from)?;
    Ok(pi_ccs::Proof { outputs, sumcheck })
}

/// Child 0's commitment from the parent's and children 1.. . The split gives
/// `Z_parent = Σ b^i Z_i` and the commitment is linear, so
/// `C_0 = C_parent - Σ_{i≥1} b^i C_i` exactly; `ajtai_dec_mixer` then
/// recomposes the parent.
fn lowest_child_commitment(parent: &Commitment, higher: &[Commitment], b: u32) -> Commitment {
    let mut child = parent.clone();
    let base = F::from_u64(b as u64);
    let mut pow = base;
    for c in higher {
        scale_commitment_add_inplace(&mut child, -pow, c);
        pow *= base;
    }
    child
}

#[cfg(test)]
#[path = "../../tests/engine_internal/dec_children.rs"]
mod dec_children_tests;

/// The Metal device of the compression prover: the backend's own device.
pub(crate) struct CompressionDevice(std::sync::Arc<std::sync::Mutex<MetalRowProver>>);

impl CompressionDevice {
    pub(crate) fn new(device: std::sync::Arc<std::sync::Mutex<MetalRowProver>>) -> Self {
        Self(device)
    }
}

impl neo_spartan::Device for CompressionDevice {
    fn merkle_layers(
        &self,
        rows: usize,
        width: usize,
        fill: &(dyn Fn(&mut [u64]) + Sync),
    ) -> Option<Vec<Vec<[u64; 4]>>> {
        let device = self
            .0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        device.merkle_layers(rows, width, fill).ok()
    }

    fn key_weights(&self, powers: &[[u64; 3]], blocks: usize) -> Option<Vec<[u64; 3]>> {
        let device = self
            .0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        device.production_key_weights(powers, blocks).ok()
    }

    fn dft_columns(&self, values: &mut [u64], width: usize, generator: u64) -> bool {
        let device = self
            .0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        match device.dft_columns(values, width, generator) {
            Ok(()) => true,
            // Only execution touches `values`; any earlier failure leaves it
            // unchanged.
            Err(error @ neo_prover_metal::MetalError::Execution(_)) => {
                panic!("the Metal DFT failed after it started: {error}")
            }
            Err(_) => false,
        }
    }

    fn gkr_layer<'a>(
        &self,
        trees: Vec<Vec<&'a mut [u64]>>,
        eq: &'a mut [u64],
    ) -> Option<Box<dyn neo_spartan::GkrLayer + 'a>> {
        let layer = self.lock().gkr_layer(trees, eq).ok()?;
        Some(Box::new(GkrOnDevice {
            device: self.0.clone(),
            layer,
        }))
    }
}

impl CompressionDevice {
    fn lock(&self) -> std::sync::MutexGuard<'_, MetalRowProver> {
        self.0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

/// One GKR layer's rounds on the backend's device.
struct GkrOnDevice<'a> {
    device: std::sync::Arc<std::sync::Mutex<MetalRowProver>>,
    layer: neo_prover_metal::MetalGkrLayer<'a>,
}

impl neo_spartan::GkrLayer for GkrOnDevice<'_> {
    fn message(&mut self, weights: &[[u64; 3]], mu: [u64; 3], degree: usize) -> Vec<[u64; 3]> {
        let device = self
            .device
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        device
            .gkr_message(&self.layer, weights, mu, degree)
            .expect("the Metal GKR message")
    }

    fn fold(&mut self, r: [u64; 3]) {
        let device = self
            .device
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        device
            .gkr_fold(&mut self.layer, r)
            .expect("the Metal GKR fold");
    }
}

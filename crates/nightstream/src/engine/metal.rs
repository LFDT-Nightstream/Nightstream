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
    fresh: CcsInstance,
    running: RunningInstance,
) -> Result<(RunningInstance, NifsProof), folding::Error> {
    let CcsInstance { claim, witness } = fresh;
    let c = prove_pi_ccs(
        device,
        transcript,
        params,
        structure,
        rows,
        workspace_bytes,
        &claim,
        &witness,
        &running,
    )?;
    let sources: Vec<_> = std::iter::once(&witness.Z)
        .chain(running.witnesses.iter())
        .collect();
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
    drop(witness);
    drop(running);
    let split = split
        .map_err(folding::kernels::Error::from)
        .map_err(pi_dec::Error::from)?;
    // The device commits children 1..; child 0 follows from the parent by
    // linearity. The terminal verifier still commits every child itself.
    let mut commitments = device
        .commit_production_prefixes(&split.planes()[1..])
        .map_err(folding::kernels::Error::from)
        .map_err(pi_dec::Error::from)?;
    commitments.insert(0, lowest_child_commitment(&parent.c, &commitments, params.b()));
    let openings = device
        .child_openings(rows, workspace_bytes, split.planes(), &parent.r, structure.m)
        .map_err(folding::kernels::Error::from)
        .map_err(pi_dec::Error::from)?;
    let neo_reductions::api::DecOutcome {
        children,
        ok_y,
        ok_x,
        ok_c,
    } = neo_reductions::api::dec_children_with_commit_superneo_cached_from_trusted_split_digits(
        neo_reductions::api::FoldingMode::Optimized,
        structure,
        params.inner(),
        &parent,
        &split,
        D.next_power_of_two().trailing_zeros() as usize,
        &commitments,
        ajtai_dec_mixer,
        None,
        Some(&openings),
    );
    if !(ok_y && ok_x && ok_c) {
        return Err(pi_dec::Error::Engine(folding::kernels::Error::PiDecPublicCheckFailed { ok_y, ok_x, ok_c }).into());
    }
    let d = pi_dec::Proof { children };
    let children = pi_dec::verify(params, structure, ajtai_dec_mixer, &parent, &d)?;
    Ok((
        RunningInstance::new(children, split.into_planes()),
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
pub(crate) fn prove_parent(
    device: &mut MetalRowProver,
    transcript: &mut Transcript,
    params: &Params,
    structure: &Structure,
    rows: &dyn MatrixRows,
    workspace_bytes: usize,
    fresh: CcsInstance,
    running: RunningInstance,
) -> Result<(pi_ccs::Proof, pi_rlc::Output, pi_rlc::Proof), folding::Error> {
    let CcsInstance { claim, witness } = fresh;
    let c = prove_pi_ccs(
        device,
        transcript,
        params,
        structure,
        rows,
        workspace_bytes,
        &claim,
        &witness,
        &running,
    )?;
    let sources: Vec<_> = std::iter::once(&witness.Z)
        .chain(running.witnesses.iter())
        .collect();
    let (parent, r) = pi_rlc::prove_refs(transcript, params, structure, ajtai_rlc_mixer, &c.outputs, &sources)?;
    Ok((c, parent, r))
}

/// PiCCS with its round oracle on the device.
fn prove_pi_ccs(
    device: &mut MetalRowProver,
    transcript: &mut Transcript,
    params: &Params,
    structure: &Structure,
    rows: &dyn MatrixRows,
    workspace_bytes: usize,
    claim: &CcsClaim,
    witness: &CcsWitness,
    running: &RunningInstance,
) -> Result<pi_ccs::Proof, folding::Error> {
    let (outputs, sumcheck, _, _) = optimized_prove_with_matrix_rows(
        transcript.inner_mut(),
        params.inner(),
        structure,
        std::slice::from_ref(claim),
        std::slice::from_ref(witness),
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
        self.lock()?.merkle_layers(rows, width, fill).ok()
    }

    fn merkle_leaves(&self, values: &[u64], width: usize) -> Option<Vec<[u64; 4]>> {
        self.lock()?.merkle_leaves(values, width).ok()
    }

    fn key_weights(&self, powers: &[[u64; 3]], blocks: usize) -> Option<Vec<[u64; 3]>> {
        self.lock()?.production_key_weights(powers, blocks).ok()
    }

    fn dft_columns(&self, values: &mut [u64], width: usize, generator: u64) -> Result<bool, String> {
        let Some(device) = self.lock() else {
            return Ok(false);
        };
        match device.dft_columns(values, width, generator) {
            Ok(()) => Ok(true),
            // Only execution touches `values`; any earlier failure leaves it
            // unchanged, and the CPU runs the DFT.
            Err(error @ neo_prover_metal::MetalError::Execution(_)) => Err(error.to_string()),
            Err(_) => Ok(false),
        }
    }

    fn gkr_layer<'a>(
        &self,
        trees: Vec<Vec<&'a mut [u64]>>,
        eq: &'a mut [u64],
    ) -> Option<Box<dyn neo_spartan::GkrLayer + 'a>> {
        let layer = self.lock()?.gkr_layer(trees, eq).ok()?;
        Some(Box::new(GkrOnDevice {
            device: self.0.clone(),
            layer,
        }))
    }
}

impl CompressionDevice {
    /// The device session, or `None` once a panic poisoned it: a poisoned
    /// session is never used again, as in `extend`.
    fn lock(&self) -> Option<std::sync::MutexGuard<'_, MetalRowProver>> {
        self.0.lock().ok()
    }
}

/// One GKR layer's rounds on the backend's device.
struct GkrOnDevice<'a> {
    device: std::sync::Arc<std::sync::Mutex<MetalRowProver>>,
    layer: neo_prover_metal::MetalGkrLayer<'a>,
}

impl neo_spartan::GkrLayer for GkrOnDevice<'_> {
    fn message(&mut self, weights: &[[u64; 3]], mu: [u64; 3], degree: usize) -> Result<Vec<[u64; 3]>, String> {
        let device = self.device.lock().map_err(|_| POISONED.to_string())?;
        device
            .gkr_message(&self.layer, weights, mu, degree)
            .map_err(|error| error.to_string())
    }

    fn fold(&mut self, r: [u64; 3]) -> Result<(), String> {
        let device = self.device.lock().map_err(|_| POISONED.to_string())?;
        device
            .gkr_fold(&mut self.layer, r)
            .map_err(|error| error.to_string())
    }
}

const POISONED: &str = "the device session lock was poisoned";

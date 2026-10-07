//! Metal round evaluation and child openings for the selected C/R/D fold.
//! The host keeps transcript order and public checks; device arithmetic also
//! generates the fixed key and computes child commitments.

use crate::folding::{
    self, ajtai_dec_mixer, ajtai_rlc_mixer, pi_ccs, pi_dec, pi_rlc, transcript::Transcript, CcsInstance, NifsProof,
    Params, RunningInstance, Structure,
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
    let (outputs, sumcheck, _, _) = optimized_prove_with_matrix_rows(
        transcript.inner_mut(),
        params.inner(),
        structure,
        &claims,
        &witnesses,
        &running.claims,
        &running.witnesses,
        rows,
        workspace_bytes,
        Some(device),
    )
    .map_err(folding::kernels::Error::from)
    .map_err(pi_ccs::Error::from)?;
    let c = pi_ccs::Proof { outputs, sumcheck };
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

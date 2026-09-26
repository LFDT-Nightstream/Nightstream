//! Independent C/R/D formulas over the circuit's original exported rows.
//! Fixed-key commitments and immutable protocol input types are shared.

use neo_ajtai::nightstream_fprime_setup::commit_production_signed_unit_prefix_matrices;
use neo_ccs::Mat;
use neo_math::{D, F};
use neo_reductions::{
    common::{decode_pi_rlc_wide_coefficients, split_b_matrix_k_with_nonzero_flags, RotRho},
    paper_exact_engine::{
        dec_reduction_paper_exact_with_rows, paper_exact_prove_with_rows, rlc_reduction_paper_exact_with_commit_mix,
        PaperMatrixRows,
    },
    PiCcsError,
};
use nightstream_fprime::LoadedPerApplicationPackage;
use p3_field::PrimeCharacteristicRing;

use crate::folding::{
    self, ajtai_dec_mixer, ajtai_rlc_mixer, pi_ccs, pi_dec, pi_rlc, transcript::Transcript, CcsInstance, NifsProof,
    Params, RunningInstance, Structure,
};

pub(crate) struct PackageRows<'a>(pub &'a LoadedPerApplicationPackage);

impl PaperMatrixRows<F> for PackageRows<'_> {
    fn shape(&self) -> (usize, usize, usize) {
        (
            self.0.row_count(),
            self.0.logical_column_count(),
            self.0.ccs_relation().matrix_sources().len(),
        )
    }

    fn row(&self, matrix: usize, row: usize) -> Vec<(usize, F)> {
        let mut entries = Vec::new();
        self.0
            .visit_matrix_rows(row..row + 1, |_, row| {
                entries.extend(
                    row.matrix(matrix)
                        .expect("validated matrix slot")
                        .iter()
                        .map(|entry| (entry.column(), F::from_u64(entry.coefficient()))),
                );
                Ok(())
            })
            .expect("PaperExact preparation validated every exported matrix row");
        entries
    }
}

pub(crate) fn prove(
    transcript: &mut Transcript,
    params: &Params,
    structure: &Structure,
    rows: &dyn PaperMatrixRows<F>,
    fresh: Vec<CcsInstance>,
    running: RunningInstance,
) -> Result<(RunningInstance, NifsProof), folding::Error> {
    let (claims, witnesses): (Vec<_>, Vec<_>) = fresh
        .into_iter()
        .map(|source| (source.claim, source.witness))
        .unzip();
    let (outputs, sumcheck, _) = paper_exact_prove_with_rows(
        transcript.inner_mut(),
        params.inner(),
        structure,
        &claims,
        &witnesses,
        &running.claims,
        &running.witnesses,
        rows,
    )
    .map_err(folding::kernels::Error::from)
    .map_err(pi_ccs::Error::from)?;
    let c = pi_ccs::Proof { outputs, sumcheck };
    let rhos = sample_rhos(transcript.inner_mut(), params, c.outputs.len())
        .map_err(folding::kernels::Error::from)
        .map_err(pi_rlc::Error::from)?;
    let mut sources: Vec<_> = witnesses.into_iter().map(|witness| witness.Z).collect();
    sources.extend(running.witnesses);
    let (parent, witness) = rlc_reduction_paper_exact_with_commit_mix(
        structure,
        params.inner(),
        &rhos,
        &c.outputs,
        &sources,
        D.next_power_of_two().trailing_zeros() as usize,
        ajtai_rlc_mixer,
    );
    drop(sources);
    let (digits, _) = split_b_matrix_k_with_nonzero_flags(&witness, params.k_rho() as usize, params.b())
        .map_err(folding::kernels::Error::from)
        .map_err(pi_dec::Error::from)?;
    let commitments = commit_production_signed_unit_prefix_matrices(&digits)
        .map_err(|error| pi_dec::Error::from(error.into_error()))?;
    let (children, ok_y, ok_x, ok_c) = dec_reduction_paper_exact_with_rows(
        structure,
        params.inner(),
        &parent,
        &digits,
        D.next_power_of_two().trailing_zeros() as usize,
        &commitments,
        ajtai_dec_mixer,
        rows,
    )
    .map_err(folding::kernels::Error::from)
    .map_err(pi_dec::Error::from)?;
    if !(ok_y && ok_x && ok_c) {
        return Err(pi_dec::Error::Engine(folding::kernels::Error::PiDecPublicCheckFailed { ok_y, ok_x, ok_c }).into());
    }
    Ok((
        RunningInstance::new(children.clone(), digits, Some(parent.clone())),
        NifsProof {
            pi_ccs: c,
            pi_rlc: pi_rlc::Proof { combined: parent },
            pi_dec: pi_dec::Proof { children },
        },
    ))
}

// Direct v1_1 sampler schedule, copied from neo-fold-clean/engine/paper_exact.rs.
fn sample_rhos(
    transcript: &mut neo_transcript::Poseidon2Transcript,
    params: &Params,
    count: usize,
) -> Result<Vec<Mat<F>>, PiCcsError> {
    if count == 0 || transcript.absorbed() != 0 {
        return Err(PiCcsError::InvalidInput(
            "PaperExact PiRLC sampler input or cursor".into(),
        ));
    }
    let mut output = Vec::with_capacity(count);
    for source in 0..count {
        transcript.absorb_v1_1(&[F::from_u64(4), F::from_usize(source)]);
        let digest = transcript.squeeze_digest_v1_1();
        let symbols = decode_pi_rlc_wide_coefficients(&digest);
        let mut column: Vec<F> = symbols.into_iter().map(|value| F::from_i8(value)).collect();
        let mut matrix = Mat::zero(D, D, F::ZERO);
        for index in 0..D {
            for row in 0..D {
                matrix[(row, index)] = column[row];
            }
            let last = column[D - 1];
            let mut next = vec![F::ZERO; D];
            next[0] = -last;
            next[1..].copy_from_slice(&column[..D - 1]);
            next[D / 2] -= last;
            column = next;
        }
        output.push(
            RotRho::new_checked(params.inner(), matrix)?
                .as_mat()
                .clone(),
        );
    }
    Ok(output)
}

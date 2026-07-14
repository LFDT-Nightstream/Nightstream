//! Ajtai `Y_eval` surfaces for the FE tail.
//!
//! Owns the device forms × witness-plane product used after the row point is
//! fixed. The FE driver owns when this surface is needed; this module owns how
//! the surface is materialized and downloaded for parity replay.

use cuda_core::DeviceBuffer;
use neo_ccs::Mat;
use neo_math::{D, F, K};
use neo_reductions::superneo_eval::SuperneoEvalCache;
use p3_field::PrimeCharacteristicRing;

use crate::device::upload_u64_device_buffer;
use crate::field::k_from_device_words;
use crate::kernels::ajtai::{ring_mat_vec_sparse_rows, ring_mat_vec_sparse_rows_into};
use crate::reduce::ccs::{CcsDeviceError, FePhaseWorkspace};
use crate::ring_layout;

use super::fe::{DeviceAjtaiYEval, DeviceFeBackend};

impl DeviceFeBackend<'_> {
    /// `Y_eval[w][j] = form_j(chi_r) . witness_w`, with both the forms
    /// build and the evaluation on device. Returns `None` on shape surprises
    /// so the CPU `precompute_for_r` path can keep ownership.
    pub fn device_ajtai_y_eval_surface(
        &mut self,
        cache: &SuperneoEvalCache,
        chi_r: &[K],
        n_eff: usize,
        witnesses: &[&Mat<F>],
    ) -> Result<Option<DeviceAjtaiYEval>, CcsDeviceError> {
        let t = cache.matrix_caches().len();
        if t == 0 || witnesses.is_empty() {
            return Ok(None);
        }
        self.oracle_plan.ensure_bar_matrices(self.device, cache)?;
        let (forms_dev, blocks) = {
            let bar = self
                .oracle_plan
                .bar_matrices()
                .expect("bar matrices uploaded above");
            (
                bar.build_forms(self.device, &self.kernels.csr, chi_r, n_eff)?,
                bar.blocks(),
            )
        };
        self.device_ajtai_y_eval_surface_from_forms(forms_dev, t, blocks, witnesses)
    }

    /// Build the Ajtai `Y_eval` surface using row challenges already resident
    /// on device, removing the row-challenge D2H dependency from the forms
    /// path.
    pub fn device_ajtai_y_eval_surface_from_device_challenges(
        &mut self,
        cache: &SuperneoEvalCache,
        challenges: &DeviceBuffer<u64>,
        challenge_count: usize,
        n_eff: usize,
        witnesses: &[&Mat<F>],
    ) -> Result<Option<DeviceAjtaiYEval>, CcsDeviceError> {
        let t = cache.matrix_caches().len();
        if t == 0 || witnesses.is_empty() {
            return Ok(None);
        }
        self.oracle_plan.ensure_bar_matrices(self.device, cache)?;
        let (forms_dev, blocks) = {
            let bar = self
                .oracle_plan
                .bar_matrices()
                .expect("bar matrices uploaded above");
            (
                bar.build_forms_from_device_challenges(
                    self.device,
                    &self.kernels.csr,
                    challenges,
                    challenge_count,
                    n_eff,
                )?,
                bar.blocks(),
            )
        };
        self.device_ajtai_y_eval_surface_from_forms(forms_dev, t, blocks, witnesses)
    }

    pub(super) fn prepare_ajtai_y_eval_workspace(
        &mut self,
        cache: &SuperneoEvalCache,
        challenge_count: usize,
        witnesses: &[&Mat<F>],
        workspace: &mut FePhaseWorkspace,
    ) -> Result<Option<(usize, usize)>, CcsDeviceError> {
        let t = cache.matrix_caches().len();
        if t == 0 || witnesses.is_empty() {
            return Ok(None);
        }
        self.oracle_plan.ensure_bar_matrices(self.device, cache)?;
        let (blocks, form_words, sparse_rows, sparse_chunks) = {
            let bar = self
                .oracle_plan
                .bar_matrices()
                .expect("bar matrices uploaded above");
            let sparse = bar.sparse_form_layout();
            (bar.blocks(), bar.form_words(), sparse.rows, sparse.chunks)
        };
        let row_len = blocks * D;
        let chi_len = 1usize
            .checked_shl(challenge_count as u32)
            .ok_or(CcsDeviceError::Shape(
                "challenge count exceeds addressable tensor table",
            ))?;
        workspace.prepare_y_eval_surface(
            self.device.stream(),
            2 * chi_len,
            form_words,
            witnesses.len() * 2 * t * D,
        )?;
        self.ring_scratch
            .prepare_sparse_mat_vec(self.device.stream(), sparse_rows, sparse_chunks, witnesses.len())?;
        let _ = resident_witness_planes(self.oracle_plan.witness_planes(), witnesses.len(), row_len).ok_or(
            CcsDeviceError::Shape("whole-FE Y_eval requires resident witness planes"),
        )?;
        Ok(Some((t, blocks)))
    }

    pub(super) fn device_ajtai_y_eval_surface_from_device_challenges_in_workspace(
        &mut self,
        cache: &SuperneoEvalCache,
        challenge_count: usize,
        n_eff: usize,
        witnesses: &[&Mat<F>],
        workspace: &mut FePhaseWorkspace,
    ) -> Result<Option<DeviceAjtaiYEval>, CcsDeviceError> {
        let Some((t, blocks)) = self.prepare_ajtai_y_eval_workspace(cache, challenge_count, witnesses, workspace)?
        else {
            return Ok(None);
        };
        let row_len = blocks * D;
        {
            let (challenges, chi_dev, forms_dev) = workspace.challenge_and_forms_buffers();
            let bar = self
                .oracle_plan
                .bar_matrices()
                .expect("bar matrices uploaded above");
            bar.build_forms_from_device_challenges_into(
                self.device,
                &self.kernels.csr,
                challenges,
                challenge_count,
                n_eff,
                chi_dev,
                forms_dev,
            )?;
        }

        let planes_dev = resident_witness_planes(self.oracle_plan.witness_planes(), witnesses.len(), row_len).ok_or(
            CcsDeviceError::Shape("whole-FE Y_eval requires resident witness planes"),
        )?;
        let mut out = workspace.take_y_eval_words();
        {
            let (_, forms_dev) = workspace.forms_buffers();
            let sparse = self
                .oracle_plan
                .bar_matrices()
                .expect("bar matrices uploaded above")
                .sparse_form_layout();
            ring_mat_vec_sparse_rows_into(
                &self.kernels.ring,
                self.device.stream(),
                &mut self.ring_scratch,
                forms_dev,
                2 * t,
                blocks,
                &sparse.entry_blocks,
                &sparse.chunk_rows,
                &sparse.chunk_entry_starts,
                &sparse.chunk_entry_lens,
                &sparse.row_chunk_offsets,
                sparse.chunks,
                planes_dev,
                0,
                witnesses.len(),
                row_len,
                &mut out,
            )?;
        }
        Ok(Some(DeviceAjtaiYEval {
            words: out,
            forms: Some(workspace.take_forms()),
            witnesses: witnesses.len(),
            matrices: t,
        }))
    }

    pub(super) fn device_ajtai_y_eval(
        &mut self,
        cache: &SuperneoEvalCache,
        chi_r: &[K],
        n_eff: usize,
        witnesses: &[&Mat<F>],
    ) -> Result<Option<Vec<Vec<[K; D]>>>, CcsDeviceError> {
        if let Some(host) = self.device_ajtai_y_eval_from_retained_challenges(cache, chi_r, n_eff, witnesses)? {
            return Ok(Some(host));
        }

        let Some(surface) = self.device_ajtai_y_eval_surface(cache, chi_r, n_eff, witnesses)? else {
            return Ok(None);
        };
        let host = self.download_ajtai_y_eval(&surface)?;
        self.last_y_eval = Some(surface);
        Ok(Some(host))
    }

    fn device_ajtai_y_eval_from_retained_challenges(
        &mut self,
        cache: &SuperneoEvalCache,
        chi_r: &[K],
        n_eff: usize,
        witnesses: &[&Mat<F>],
    ) -> Result<Option<Vec<Vec<[K; D]>>>, CcsDeviceError> {
        let Some(challenge_count) = challenge_count_for_chi_table(chi_r.len()) else {
            return Ok(None);
        };
        if self
            .phase_workspace
            .as_ref()
            .is_none_or(|workspace| workspace.challenges().len() < 2 * challenge_count)
        {
            return Ok(None);
        }

        let mut workspace = self
            .phase_workspace
            .take()
            .expect("workspace presence checked above");
        let Some(surface) = self.device_ajtai_y_eval_surface_from_device_challenges_in_workspace(
            cache,
            challenge_count,
            n_eff,
            witnesses,
            &mut workspace,
        )?
        else {
            self.phase_workspace = Some(workspace);
            return Ok(None);
        };
        let host = self.download_ajtai_y_eval(&surface)?;
        self.last_y_eval = Some(surface);
        self.phase_workspace = Some(workspace);
        Ok(Some(host))
    }

    /// Download a resident `Y_eval` surface into the CPU oracle layout.
    pub fn download_ajtai_y_eval(&self, y_eval: &DeviceAjtaiYEval) -> Result<Vec<Vec<[K; D]>>, CcsDeviceError> {
        let words = y_eval.words.to_host_vec(self.device.stream())?;
        self.device.sync()?;

        let per_wit = 2 * y_eval.matrices * D;
        Ok((0..y_eval.witnesses)
            .map(|wit| {
                (0..y_eval.matrices)
                    .map(|j| {
                        let re = &words[wit * per_wit + (2 * j) * D..];
                        let im = &words[wit * per_wit + (2 * j + 1) * D..];
                        let mut row = [K::ZERO; D];
                        for (rho, slot) in row.iter_mut().enumerate() {
                            *slot = k_from_device_words(re[rho], im[rho]);
                        }
                        row
                    })
                    .collect()
            })
            .collect())
    }

    fn device_ajtai_y_eval_surface_from_forms(
        &mut self,
        forms_dev: DeviceBuffer<u64>,
        t: usize,
        blocks: usize,
        witnesses: &[&Mat<F>],
    ) -> Result<Option<DeviceAjtaiYEval>, CcsDeviceError> {
        let row_len = blocks * D;
        let own_planes;
        let planes_dev = if let Some(planes) =
            resident_witness_planes(self.oracle_plan.witness_planes(), witnesses.len(), row_len)
        {
            planes
        } else {
            own_planes = self.upload_witness_planes(witnesses, row_len, blocks)?;
            &own_planes
        };
        let sparse = self
            .oracle_plan
            .bar_matrices()
            .expect("bar matrices uploaded above")
            .sparse_form_layout();
        let out = ring_mat_vec_sparse_rows(
            &self.kernels.ring,
            self.device.stream(),
            &mut self.ring_scratch,
            &forms_dev,
            2 * t,
            blocks,
            &sparse.entry_blocks,
            &sparse.chunk_rows,
            &sparse.chunk_entry_starts,
            &sparse.chunk_entry_lens,
            &sparse.row_chunk_offsets,
            sparse.chunks,
            planes_dev,
            0,
            witnesses.len(),
            row_len,
        )?;
        Ok(Some(DeviceAjtaiYEval {
            words: out,
            forms: Some(forms_dev),
            witnesses: witnesses.len(),
            matrices: t,
        }))
    }

    fn upload_witness_planes(
        &self,
        witnesses: &[&Mat<F>],
        row_len: usize,
        blocks: usize,
    ) -> Result<DeviceBuffer<u64>, CcsDeviceError> {
        if witnesses
            .iter()
            .any(|w| w.rows() != D || w.cols() != blocks)
        {
            return Err(CcsDeviceError::Shape("Ajtai Y_eval witness shape mismatch"));
        }
        let mut plane_words = vec![0u64; witnesses.len() * row_len];
        for (wit, witness) in witnesses.iter().enumerate() {
            let words = ring_layout::mat_to_words(witness);
            plane_words[wit * row_len..(wit + 1) * row_len].copy_from_slice(&words);
        }
        Ok(upload_u64_device_buffer(self.device.stream(), &plane_words)?)
    }
}

fn challenge_count_for_chi_table(len: usize) -> Option<usize> {
    len.is_power_of_two()
        .then_some(len.trailing_zeros() as usize)
}

fn resident_witness_planes(
    planes: Option<(&DeviceBuffer<u64>, usize)>,
    witnesses: usize,
    row_len: usize,
) -> Option<&DeviceBuffer<u64>> {
    planes
        .filter(|(planes, count)| *count == witnesses && planes.len() == witnesses * row_len)
        .map(|(planes, _)| planes)
}

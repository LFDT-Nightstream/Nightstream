//! Device-owned Π_CCS oracle preparation plan.
//!
//! Owns the device-resident FE oracle inputs that are prepared while the
//! optimized engine still owns protocol semantics. This is the CUDA-side
//! boundary where independent oracle work can later be scheduled by the real
//! dependency graph instead of by CPU constructor order.

use cuda_core::DeviceBuffer;
use neo_math::{KExtensions, D, K};
use neo_reductions::optimized_engine::legacy_split_nc::{FeEvalTable, FeMcsRowTables};
use neo_reductions::superneo_eval::{weighted_projection_basis_forms, SuperneoEvalCache, SuperneoZBlocks};
use p3_field::PrimeField64;

use crate::device::{upload_u64_device_buffer, zeroed_u64_device_buffer, Device};
use crate::field::k_from_device_words;
use crate::graph::GraphAllocations;
use crate::kernels::csr::launch_plane_lin_comb;
use crate::reduce::ccs::{CcsDeviceError, SumcheckKernels};
use crate::ring_forms::{DeviceBarMatrices, DeviceRowMatrices};

pub(super) struct DeferredMcsRowTables {
    pub(super) mcs_idx: usize,
    pub(super) n_pad: usize,
    pub(super) table_count: usize,
    pub(super) packed: DeviceBuffer<u64>,
    /// Keeps the tiny matrix-index metadata live until the packed table build
    /// has completed on the stream.
    pub(super) _matrix_indices: DeviceBuffer<u64>,
}

pub(super) struct DeferredEvalTable {
    pub(super) n_pad: usize,
    pub(super) table: DeviceBuffer<u64>,
}

pub(super) struct DevicePiCcsOraclePlan<'a> {
    /// Static bar matrices for the Ajtai forms build, uploaded once per
    /// structure and reused across proves.
    pub(super) bar_matrices: Option<DeviceBarMatrices>,
    /// Static row-major orig CSR for f-var row tables, same lifetime.
    row_matrices: Option<DeviceRowMatrices>,
    /// Caller-supplied fold witness planes (`[count][cols * D]`) shared
    /// across the current fold.
    witness_planes: Option<(&'a DeviceBuffer<u64>, usize)>,
    /// Row tables built from resident MCS witness planes before the
    /// row-phase snapshot is handed to the FE backend.
    pub(super) deferred_row_tables: Vec<DeferredMcsRowTables>,
    /// Carried eval table built from resident running witness planes before
    /// the row-phase snapshot is handed to the FE backend.
    pub(super) deferred_eval_table: Option<DeferredEvalTable>,
}

impl<'a> DevicePiCcsOraclePlan<'a> {
    pub(super) fn new() -> Self {
        Self {
            bar_matrices: None,
            row_matrices: None,
            witness_planes: None,
            deferred_row_tables: Vec::new(),
            deferred_eval_table: None,
        }
    }

    pub(super) fn set_witness_planes(&mut self, planes: &'a DeviceBuffer<u64>, count: usize) {
        self.witness_planes = Some((planes, count));
    }

    pub(super) fn take_statics(&mut self) -> (Option<DeviceBarMatrices>, Option<DeviceRowMatrices>) {
        (self.bar_matrices.take(), self.row_matrices.take())
    }

    pub(super) fn set_statics(&mut self, bar: Option<DeviceBarMatrices>, rows: Option<DeviceRowMatrices>) {
        self.bar_matrices = bar;
        self.row_matrices = rows;
    }

    pub(super) fn reset_deferred(&mut self) {
        self.deferred_row_tables.clear();
        self.deferred_eval_table = None;
    }

    pub(super) fn serves_carried_eval_table(&self) -> bool {
        self.witness_planes.is_some()
    }

    pub(super) fn witness_planes(&self) -> Option<(&'a DeviceBuffer<u64>, usize)> {
        self.witness_planes
    }

    pub(super) fn ensure_bar_matrices(
        &mut self,
        device: &Device,
        cache: &SuperneoEvalCache,
    ) -> Result<(), CcsDeviceError> {
        if self.bar_matrices.as_ref().is_none_or(|b| !b.matches(cache)) {
            self.bar_matrices = Some(DeviceBarMatrices::upload(device, cache)?);
        }
        Ok(())
    }

    pub(super) fn bar_matrices(&self) -> Option<&DeviceBarMatrices> {
        self.bar_matrices.as_ref()
    }

    pub(super) fn record_graph_allocations(&self, allocations: &mut GraphAllocations) {
        if let Some(bar) = &self.bar_matrices {
            bar.record_graph_allocations(allocations);
        }
        if let Some((planes, _)) = self.witness_planes {
            allocations.push(planes);
        }
    }

    pub(super) fn mcs_row_tables(
        &mut self,
        device: &Device,
        kernels: &SumcheckKernels,
        cache: &SuperneoEvalCache,
        mcs_idx: usize,
        f_var_indices: &[usize],
        z_blocks: &SuperneoZBlocks,
        n_eff: usize,
        n_pad: usize,
    ) -> Result<Option<FeMcsRowTables>, CcsDeviceError> {
        if !z_blocks.imag_all_zero() {
            return Ok(None);
        }
        let witness_planes = self.witness_planes;
        let mut deferred = None;
        let mut host = None;
        {
            let mats = self.ensure_row_matrices(device, cache)?;
            if f_var_indices.iter().any(|&j| j >= mats.matrix_count()) {
                return Ok(None);
            }

            let plane_len = mats.blocks() * D;
            if let Some((planes, _count)) =
                witness_planes.filter(|(planes, count)| mcs_idx < *count && planes.len() == *count * plane_len)
            {
                let z_offset = mcs_idx * plane_len;
                let (packed, matrix_indices) = mats.build_packed_row_tables_device(
                    device,
                    &kernels.csr,
                    f_var_indices,
                    planes,
                    z_offset,
                    n_eff,
                    n_pad,
                )?;
                deferred = Some(DeferredMcsRowTables {
                    mcs_idx,
                    n_pad,
                    table_count: f_var_indices.len(),
                    packed,
                    _matrix_indices: matrix_indices,
                });
            } else {
                let plane = z_blocks.re_plane_words();
                if plane.len() != plane_len {
                    return Ok(None);
                }

                let stream = device.stream();
                let plane_dev = upload_u64_device_buffer(stream, &plane)?;
                let mut tables = Vec::with_capacity(f_var_indices.len());
                for &j in f_var_indices {
                    let words = mats.build_row_table(device, &kernels.csr, j, &plane_dev, n_eff, n_pad)?;
                    tables.push(
                        (0..n_pad)
                            .map(|row| k_from_device_words(words[2 * row], words[2 * row + 1]))
                            .collect(),
                    );
                }
                host = Some(tables);
            }
        }
        if let Some(deferred) = deferred {
            self.deferred_row_tables.push(deferred);
            return Ok(Some(FeMcsRowTables::Deferred));
        }
        Ok(host.map(FeMcsRowTables::Host))
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn carried_eval_table(
        &mut self,
        device: &Device,
        kernels: &SumcheckKernels,
        cache: &SuperneoEvalCache,
        carried_coeffs: &[K],
        k_mcs: usize,
        weights: &[K; D],
        mat_coeffs: &[K],
        n_eff: usize,
        n_pad: usize,
    ) -> Result<Option<FeEvalTable>, CcsDeviceError> {
        self.deferred_eval_table = None;
        let Some((planes, count)) = self.witness_planes else {
            return Ok(None);
        };
        if count != k_mcs + carried_coeffs.len() {
            return Ok(None);
        };
        let mats = self.ensure_row_matrices(device, cache)?;
        let plane_len = mats.blocks() * D;
        if planes.len() != count * plane_len {
            return Ok(None);
        }

        let stream = device.stream();
        let mut coeff_words = vec![0u64; 2 * carried_coeffs.len()];
        for (i, coeff) in carried_coeffs.iter().enumerate() {
            let (re, im) = coeff.to_limbs_u64();
            coeff_words[2 * i] = re;
            coeff_words[2 * i + 1] = im;
        }
        let coeffs_dev = upload_u64_device_buffer(stream, &coeff_words)?;
        let mut z_re = zeroed_u64_device_buffer(stream, plane_len)?;
        let mut z_im = zeroed_u64_device_buffer(stream, plane_len)?;
        launch_plane_lin_comb(
            &kernels.csr,
            stream,
            planes,
            &coeffs_dev,
            carried_coeffs.len(),
            k_mcs * plane_len,
            plane_len,
            plane_len,
            &mut z_re,
            &mut z_im,
        )?;

        let (basis_re_words, basis_im_words) = basis_words(weights);
        let table = mats.build_weighted_table_device(
            device,
            &kernels.csr,
            &basis_re_words,
            &basis_im_words,
            &z_re,
            &z_im,
            mat_coeffs,
            n_eff,
            n_pad,
        )?;
        self.deferred_eval_table = Some(DeferredEvalTable { n_pad, table });
        Ok(Some(FeEvalTable::Deferred))
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn eval_weighted_row_table(
        &mut self,
        device: &Device,
        kernels: &SumcheckKernels,
        cache: &SuperneoEvalCache,
        z_blocks: &SuperneoZBlocks,
        weights: &[K; D],
        mat_coeffs: &[K],
        n_eff: usize,
        n_pad: usize,
    ) -> Result<Option<Vec<K>>, CcsDeviceError> {
        if z_blocks.imag_all_zero() {
            return Ok(None);
        }
        let mats = self.ensure_row_matrices(device, cache)?;
        let z_re = z_blocks.re_plane_words();
        if z_re.len() != mats.blocks() * D {
            return Ok(None);
        }
        let z_im = z_blocks.im_plane_words();

        let (basis_re_words, basis_im_words) = basis_words(weights);
        let words = mats.build_weighted_table(
            device,
            &kernels.csr,
            &basis_re_words,
            &basis_im_words,
            &z_re,
            &z_im,
            mat_coeffs,
            n_eff,
            n_pad,
        )?;
        Ok(Some(
            (0..n_pad)
                .map(|row| k_from_device_words(words[2 * row], words[2 * row + 1]))
                .collect(),
        ))
    }

    fn ensure_row_matrices(
        &mut self,
        device: &Device,
        cache: &SuperneoEvalCache,
    ) -> Result<&DeviceRowMatrices, CcsDeviceError> {
        let cached = self.row_matrices.as_ref().is_some_and(|m| m.matches(cache));
        if !cached {
            crate::perf_timed!("session.structure", {
                self.row_matrices = Some(DeviceRowMatrices::upload(device, cache)?);
            });
        }
        Ok(self
            .row_matrices
            .as_ref()
            .expect("row matrices uploaded above"))
    }
}

fn basis_words(weights: &[K; D]) -> (Vec<u64>, Vec<u64>) {
    let (basis_re, basis_im) = weighted_projection_basis_forms(weights);
    let mut basis_re_words = vec![0u64; D * D];
    let mut basis_im_words = vec![0u64; D * D];
    for local in 0..D {
        for lane in 0..D {
            basis_re_words[local * D + lane] = basis_re[local].0[lane].as_canonical_u64();
            basis_im_words[local * D + lane] = basis_im[local].0[lane].as_canonical_u64();
        }
    }
    (basis_re_words, basis_im_words)
}

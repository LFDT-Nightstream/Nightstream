//! Device ownership for Nebula's three witness-lane commitments.
//!
//! The lane matrices come directly from `LaneScheme`; this module only
//! commits the corresponding whole-column slices of resident DEC children.

use std::sync::Arc;

use neo_ajtai::{Commitment, PP};
use neo_ccs::LaneCommitments;
use neo_fold_clean::paper::nifs::Error;
use neo_fold_clean::paper::relations::{LaneRanges, LaneScheme};
use neo_math::{Rq, D};

use crate::commit::DeviceAjtai;
use crate::device::Device;
use crate::field::f_from_device_word;
use crate::reduce::rlc::compose_commitment_words_device;
use crate::session::backend_unavailable;

pub(crate) struct LaneDeviceMaterial {
    ops_pp: Arc<PP<Rq>>,
    mem_pp: Arc<PP<Rq>>,
    ranges: LaneRanges,
}

impl LaneDeviceMaterial {
    pub(crate) fn from_scheme(scheme: &LaneScheme) -> Result<Self, Error> {
        Ok(Self {
            ops_pp: scheme
                .ops_verification_pp()
                .map_err(|_| backend_unavailable("Nebula ops lane PP unavailable"))?,
            mem_pp: scheme
                .mem_verification_pp()
                .map_err(|_| backend_unavailable("Nebula memory lane PP unavailable"))?,
            ranges: scheme.lane_ranges().clone(),
        })
    }
}

pub(crate) struct DeviceLaneAjtai {
    ops_source: Arc<PP<Rq>>,
    mem_source: Arc<PP<Rq>>,
    ranges: LaneRanges,
    ops: DeviceAjtai,
    mem: DeviceAjtai,
}

impl DeviceLaneAjtai {
    pub(crate) fn upload(device: &Device, material: LaneDeviceMaterial) -> Result<Self, Error> {
        let ops = DeviceAjtai::upload(device, &material.ops_pp)
            .map_err(|_| backend_unavailable("upload Nebula ops lane PP failed"))?;
        let mem = DeviceAjtai::upload(device, &material.mem_pp)
            .map_err(|_| backend_unavailable("upload Nebula memory lane PP failed"))?;
        Ok(Self {
            ops_source: material.ops_pp,
            mem_source: material.mem_pp,
            ranges: material.ranges,
            ops,
            mem,
        })
    }

    pub(crate) fn matches(&self, material: &LaneDeviceMaterial) -> bool {
        Arc::ptr_eq(&self.ops_source, &material.ops_pp)
            && Arc::ptr_eq(&self.mem_source, &material.mem_pp)
            && self.ranges == material.ranges
    }

    pub(crate) fn commit_children(
        &mut self,
        device: &Device,
        planes: &cuda_core::DeviceBuffer<u64>,
        children: usize,
        plane_stride: usize,
    ) -> Result<Vec<LaneCommitments<Commitment>>, Error> {
        if children == 0 || planes.len() != children * plane_stride {
            return Err(backend_unavailable("Nebula child lane plane shape mismatch"));
        }
        if self.ranges.fs.end * D > plane_stride {
            return Err(backend_unavailable("Nebula lane range exceeds child witness"));
        }

        let ops_words = self
            .ops
            .commit_planes_device_at(device, planes, self.ranges.ops.start * D, children, plane_stride)
            .map_err(|_| backend_unavailable("Nebula ops child commitment failed"))?;
        let is_words = self
            .mem
            .commit_planes_device_at(device, planes, self.ranges.is.start * D, children, plane_stride)
            .map_err(|_| backend_unavailable("Nebula IS child commitment failed"))?;
        let fs_words = self
            .mem
            .commit_planes_device_at(device, planes, self.ranges.fs.start * D, children, plane_stride)
            .map_err(|_| backend_unavailable("Nebula FS child commitment failed"))?;

        let words_per_commitment = self.ops.kappa() * D;
        if self.mem.kappa() != self.ops.kappa()
            || ops_words.len() != children * words_per_commitment
            || is_words.len() != children * words_per_commitment
            || fs_words.len() != children * words_per_commitment
        {
            return Err(backend_unavailable("Nebula child commitment output shape mismatch"));
        }
        let packed = compose_commitment_words_device(
            device,
            self.ops.module(),
            &[&ops_words, &is_words, &fs_words],
            3 * children * words_per_commitment,
        )
        .map_err(|_| backend_unavailable("pack Nebula child commitments failed"))?;
        let words = packed
            .to_host_vec(device.stream())
            .map_err(|_| backend_unavailable("download Nebula child commitments failed"))?;
        device
            .sync()
            .map_err(|_| backend_unavailable("synchronize Nebula child commitments failed"))?;

        let decode = |coordinate: usize, child: usize| {
            let start = (coordinate * children + child) * words_per_commitment;
            let mut commitment = Commitment::zeros(D, self.ops.kappa());
            for (slot, &word) in commitment
                .data
                .iter_mut()
                .zip(&words[start..start + words_per_commitment])
            {
                *slot = f_from_device_word(word);
            }
            commitment
        };
        Ok((0..children)
            .map(|child| LaneCommitments {
                ops: decode(0, child),
                is: decode(1, child),
                fs: decode(2, child),
            })
            .collect())
    }
}

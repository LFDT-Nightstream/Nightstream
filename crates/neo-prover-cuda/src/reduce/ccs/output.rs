//! Device-resident Pi_CCS output surfaces.
//!
//! Owns the K-valued surface bundle that Pi_RLC can consume without
//! re-uploading host `CeClaim` fields: `[claim][surface][d_pad][c0,c1]`,
//! where surfaces are every `y_ring[j]` followed by optional `y_zcol`.

use std::sync::Arc;

use cuda_core::{CudaStream, DeviceBuffer};
use neo_ajtai::Commitment;
use neo_ccs::{CeClaim, Mat};
use neo_math::{KExtensions, D, F, K};
use p3_field::PrimeCharacteristicRing;

use crate::device::{uninit_u64_device_buffer, upload_u64_device_buffer, Device};
use crate::field::k_from_device_words;
use crate::kernels::pi_ccs_output::{launch_ccs_pack_k_surfaces, launch_ccs_running_initial_sum};
use crate::kernels::pi_rlc::{launch_rlc_pack_active_public_x, RlcKernelModule};
use crate::reduce::ccs::{CcsDeviceError, DeviceAjtaiYEval, DeviceNcFinalState, SumcheckKernels};

pub struct DevicePiCcsKSurfaces {
    stream: Arc<CudaStream>,
    words: DeviceBuffer<u64>,
    keepalive: Vec<DeviceBuffer<u64>>,
    claims: usize,
    t_core: usize,
    include_y_zcol: bool,
    d_pad: usize,
}

impl DevicePiCcsKSurfaces {
    pub(crate) fn pack_raw(
        device: &Device,
        kernels: &SumcheckKernels,
        y_words: &DeviceBuffer<u64>,
        y_zcol_words: Option<&DeviceBuffer<u64>>,
        claims: usize,
        t_core: usize,
        d_pad: usize,
    ) -> Result<Self, CcsDeviceError> {
        if claims == 0 || t_core == 0 || d_pad < D || !d_pad.is_power_of_two() {
            return Err(CcsDeviceError::Shape("raw claim surface shape is unsupported"));
        }
        let include_y_zcol = y_zcol_words.is_some();
        let surface_count = t_core + usize::from(include_y_zcol);
        let stream = device.stream();
        let nc_words = y_zcol_words.unwrap_or(y_words);
        let mut words = uninit_u64_device_buffer(stream, claims * surface_count * d_pad * 2)?;
        launch_ccs_pack_k_surfaces(
            &kernels.output,
            stream,
            y_words,
            nc_words,
            claims,
            t_core,
            include_y_zcol,
            d_pad,
            &mut words,
        )?;
        Ok(Self {
            stream: Arc::clone(stream),
            words,
            keepalive: Vec::new(),
            claims,
            t_core,
            include_y_zcol,
            d_pad,
        })
    }

    pub fn pack(
        device: &Device,
        kernels: &SumcheckKernels,
        y_eval: Option<&DeviceAjtaiYEval>,
        nc_final: Option<&DeviceNcFinalState>,
        d_pad: usize,
    ) -> Result<Self, CcsDeviceError> {
        if d_pad < D || !d_pad.is_power_of_two() {
            return Err(CcsDeviceError::Shape(
                "Pi_CCS output surface d_pad must be power-of-two and >= D",
            ));
        }
        let claims = match (y_eval, nc_final) {
            (Some(y), Some(nc)) if y.witnesses() == nc.witnesses() => y.witnesses(),
            (Some(y), None) => y.witnesses(),
            (None, Some(nc)) => nc.witnesses(),
            (Some(_), Some(_)) => {
                return Err(CcsDeviceError::Shape("Pi_CCS y_eval/NC surface claim count mismatch"));
            }
            (None, None) => return Err(CcsDeviceError::Shape("Pi_CCS output surface needs FE or NC data")),
        };
        let t_core = y_eval.map(DeviceAjtaiYEval::matrices).unwrap_or(0);
        let include_y_zcol = nc_final.is_some();
        let surface_count = t_core + usize::from(include_y_zcol);
        if surface_count == 0 {
            return Err(CcsDeviceError::Shape("Pi_CCS output surface has no K surfaces"));
        }

        let stream = device.stream();
        let y_words = y_eval
            .map(DeviceAjtaiYEval::words)
            .or_else(|| nc_final.map(DeviceNcFinalState::words))
            .expect("checked non-empty above");
        let nc_words = nc_final
            .map(DeviceNcFinalState::words)
            .or_else(|| y_eval.map(DeviceAjtaiYEval::words))
            .expect("checked non-empty above");
        // `ccs_pack_k_surfaces` writes every output limb, including zero
        // padding lanes, before Pi_RLC reads this resident handoff.
        let mut words = uninit_u64_device_buffer(stream, claims * surface_count * d_pad * 2)?;
        launch_ccs_pack_k_surfaces(
            &kernels.output,
            stream,
            y_words,
            nc_words,
            claims,
            t_core,
            include_y_zcol,
            d_pad,
            &mut words,
        )?;
        Ok(Self {
            stream: Arc::clone(stream),
            words,
            keepalive: Vec::new(),
            claims,
            t_core,
            include_y_zcol,
            d_pad,
        })
    }

    pub fn words(&self) -> &DeviceBuffer<u64> {
        &self.words
    }

    pub(crate) fn from_packed_words(
        stream: Arc<CudaStream>,
        words: DeviceBuffer<u64>,
        keepalive: Vec<DeviceBuffer<u64>>,
        claims: usize,
        t_core: usize,
        include_y_zcol: bool,
        d_pad: usize,
    ) -> Result<Self, CcsDeviceError> {
        let surface_count = t_core + usize::from(include_y_zcol);
        if claims == 0
            || surface_count == 0
            || d_pad < D
            || !d_pad.is_power_of_two()
            || words.len() != claims * surface_count * d_pad * 2
        {
            return Err(CcsDeviceError::Shape("packed claim surface shape is unsupported"));
        }
        Ok(Self {
            stream,
            words,
            keepalive,
            claims,
            t_core,
            include_y_zcol,
            d_pad,
        })
    }

    pub(crate) fn into_buffers(self) -> (DeviceBuffer<u64>, Vec<DeviceBuffer<u64>>) {
        let Self { words, keepalive, .. } = self;
        (words, keepalive)
    }

    pub fn claims(&self) -> usize {
        self.claims
    }

    pub fn t_core(&self) -> usize {
        self.t_core
    }

    pub fn include_y_zcol(&self) -> bool {
        self.include_y_zcol
    }

    pub fn surface_count(&self) -> usize {
        self.t_core + usize::from(self.include_y_zcol)
    }

    pub fn d_pad(&self) -> usize {
        self.d_pad
    }

    pub(crate) fn materialize_claims(&self, claims: &mut [CeClaim<Commitment, F, K>]) -> Result<(), CcsDeviceError> {
        if claims.len() != self.claims {
            return Err(CcsDeviceError::Shape("Pi_CCS surface claim count mismatch"));
        }
        let words = self.words.to_host_vec(&self.stream)?;
        for (claim_idx, claim) in claims.iter_mut().enumerate() {
            claim.y_ring = (0..self.t_core)
                .map(|surface| self.decode_surface(&words, claim_idx, surface))
                .collect();
            claim.ct = neo_reductions::common::ct_from_y_ring(&claim.y_ring);
            claim.y_zcol = if self.include_y_zcol {
                self.decode_surface(&words, claim_idx, self.t_core)
            } else {
                Vec::new()
            };
        }
        Ok(())
    }

    fn decode_surface(&self, words: &[u64], claim: usize, surface: usize) -> Vec<K> {
        (0..self.d_pad)
            .map(|lane| {
                let at = ((claim * self.surface_count() + surface) * self.d_pad + lane) * 2;
                k_from_device_words(words[at], words[at + 1])
            })
            .collect()
    }

    #[doc(hidden)]
    pub fn claimed_initial_sum(
        &self,
        device: &Device,
        kernels: &SumcheckKernels,
        alpha: &[K],
        gamma: K,
        k_mcs: usize,
    ) -> Result<K, CcsDeviceError> {
        let chi = host_chi(alpha);
        let mut chi_words = Vec::with_capacity(2 * chi.len());
        for value in &chi {
            let (c0, c1) = value.to_limbs_u64();
            chi_words.extend([c0, c1]);
        }
        let k_total = k_mcs + self.claims;
        let gamma_to_k = pow_k(gamma, k_total);
        let mut matrix_weight = gamma_to_k;
        let mut weight_words = Vec::with_capacity(2 * self.claims * self.t_core);
        for _ in 0..self.t_core {
            let mut child_weight = pow_k(gamma, k_mcs) * matrix_weight;
            for _ in 0..self.claims {
                let (c0, c1) = child_weight.to_limbs_u64();
                weight_words.extend([c0, c1]);
                child_weight *= gamma;
            }
            matrix_weight *= gamma_to_k;
        }
        let stream = device.stream();
        let chi_dev = upload_u64_device_buffer(stream, &chi_words)?;
        let weights = upload_u64_device_buffer(stream, &weight_words)?;
        let mut partials = uninit_u64_device_buffer(stream, 2 * self.claims * self.t_core)?;
        let mut out = uninit_u64_device_buffer(stream, 2)?;
        launch_ccs_running_initial_sum(
            &kernels.output,
            stream,
            &self.words,
            self.claims,
            self.surface_count(),
            self.t_core,
            self.d_pad,
            &chi_dev,
            chi.len(),
            &weights,
            &mut partials,
            &mut out,
        )?;
        let words = out.to_host_vec(stream)?;
        device.sync()?;
        Ok(k_from_device_words(words[0], words[1]))
    }

    pub fn download_surfaces(&self, device: &Device) -> Result<Vec<Vec<Vec<K>>>, CcsDeviceError> {
        let words = self.words.to_host_vec(device.stream())?;
        device.sync()?;
        Ok((0..self.claims)
            .map(|claim| {
                (0..self.surface_count())
                    .map(|surface| {
                        (0..self.d_pad)
                            .map(|lane| {
                                let idx = ((claim * self.surface_count() + surface) * self.d_pad + lane) * 2;
                                k_from_device_words(words[idx], words[idx + 1])
                            })
                            .collect()
                    })
                    .collect()
            })
            .collect())
    }
}

/// Active public columns of one or more CE-claim `X` matrices.
///
/// The canonical host matrix is `D x m_in`, but only `ceil(m_in / D)`
/// columns carry values. Keeping just that row-major prefix avoids carrying or
/// downloading structural zeros between folds.
pub(crate) struct DevicePublicX {
    stream: Arc<CudaStream>,
    words: DeviceBuffer<u64>,
    claims: usize,
    m_in: usize,
    active_cols: usize,
}

impl DevicePublicX {
    pub(crate) fn pack_from_planes(
        device: &Device,
        module: &RlcKernelModule,
        planes: &DeviceBuffer<u64>,
        claims: usize,
        plane_stride: usize,
        m_in: usize,
    ) -> Result<Self, CcsDeviceError> {
        let active_cols = m_in.div_ceil(D);
        let stream = device.stream();
        let mut words = uninit_u64_device_buffer(stream, (claims * D * active_cols).max(1))?;
        launch_rlc_pack_active_public_x(module, stream, planes, claims, plane_stride, m_in, &mut words)?;
        Self::new(Arc::clone(stream), words, claims, m_in)
    }

    pub(crate) fn new(
        stream: Arc<CudaStream>,
        words: DeviceBuffer<u64>,
        claims: usize,
        m_in: usize,
    ) -> Result<Self, CcsDeviceError> {
        let active_cols = m_in.div_ceil(D);
        let logical_words = claims * D * active_cols;
        if claims == 0 || words.len() < logical_words.max(1) {
            return Err(CcsDeviceError::Shape("device public X shape mismatch"));
        }
        Ok(Self {
            stream,
            words,
            claims,
            m_in,
            active_cols,
        })
    }

    pub(crate) fn claims(&self) -> usize {
        self.claims
    }

    pub(crate) fn words_per_claim(&self) -> usize {
        D * self.active_cols
    }

    pub(crate) fn materialize_claims(&self, claims: &mut [CeClaim<Commitment, F, K>]) -> Result<(), CcsDeviceError> {
        if claims.len() != self.claims || claims.iter().any(|claim| claim.m_in != self.m_in) {
            return Err(CcsDeviceError::Shape("device public X claim shape mismatch"));
        }
        let matrices = self.materialize()?;
        for (claim, x) in claims.iter_mut().zip(matrices) {
            claim.X = x;
        }
        Ok(())
    }

    pub(crate) fn materialize(&self) -> Result<Vec<Mat<F>>, CcsDeviceError> {
        let words = if self.active_cols == 0 {
            Vec::new()
        } else {
            self.words.to_host_vec(&self.stream)?
        };
        let stride = self.words_per_claim();
        Ok((0..self.claims)
            .map(|claim_idx| {
                let mut x = Mat::zero(D, self.m_in, F::ZERO);
                let base = claim_idx * stride;
                for row in 0..D {
                    for col in 0..self.active_cols {
                        x[(row, col)] = F::from_u64(words[base + row * self.active_cols + col]);
                    }
                }
                x
            })
            .collect())
    }
}

fn host_chi(alpha: &[K]) -> Vec<K> {
    let mut chi = vec![K::ZERO; 1usize << alpha.len()];
    for (lane, slot) in chi.iter_mut().enumerate() {
        let mut weight = K::ONE;
        for (bit, &value) in alpha.iter().enumerate() {
            weight *= if ((lane >> bit) & 1) == 1 {
                value
            } else {
                K::ONE - value
            };
        }
        *slot = weight;
    }
    chi
}

fn pow_k(mut base: K, mut exponent: usize) -> K {
    let mut acc = K::ONE;
    while exponent != 0 {
        if exponent & 1 == 1 {
            acc *= base;
        }
        base *= base;
        exponent >>= 1;
    }
    acc
}

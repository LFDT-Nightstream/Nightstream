//! FE Ajtai-tail device helpers.
//!
//! Owns only the device wrapper around Ajtai-tail coefficient evaluation.
//! The parent FE oracle still owns row tables, transcript scheduling, and
//! proof-state bookkeeping.

use cuda_core::DeviceBuffer;
use neo_math::{KExtensions, K};
use p3_field::PrimeCharacteristicRing;

use crate::device::Device;
use crate::field::k_from_device_words;
use crate::kernels::pi_ccs_tail::{
    launch_ajtai_tail_round_coeffs, launch_ajtai_tail_round_partials_from_challenges,
    launch_ajtai_tail_round_reduce_from_challenges, FeTailKernelModule,
};
use crate::kernels::sumcheck_common::launch_sum_partials;

use super::{CcsDeviceError, DeviceAjtaiYEval, DeviceFeBackend, DeviceFeOracle, DeviceFeTailRound, SumcheckKernels};

impl DeviceFeOracle {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn write_tail_round_coeffs_from_challenges(
        &mut self,
        device: &Device,
        kernels: &SumcheckKernels,
        tail_module: &FeTailKernelModule,
        y_eval: &DeviceAjtaiYEval,
        tail_headers: &DeviceBuffer<u64>,
        header_offset: usize,
        points: &DeviceBuffer<u64>,
        challenges: &DeviceBuffer<u64>,
        partial_count: usize,
        partials: &mut DeviceBuffer<u64>,
        partial_scratch: &mut DeviceBuffer<u64>,
        inner_sums: &mut DeviceBuffer<u64>,
    ) -> Result<(), CcsDeviceError> {
        let stream = device.stream();
        launch_ajtai_tail_round_partials_from_challenges(
            tail_module,
            stream,
            &y_eval.words,
            tail_headers,
            header_offset,
            points,
            challenges,
            partial_count,
            partials,
        )?;
        if partial_count > 0 {
            launch_sum_partials(
                &kernels.common,
                stream,
                partials,
                partial_count,
                4,
                partial_scratch,
                inner_sums,
            )?;
        }
        launch_ajtai_tail_round_reduce_from_challenges(
            tail_module,
            stream,
            &y_eval.words,
            tail_headers,
            header_offset,
            &self.mcs_meta,
            &self.term_meta,
            &self.term_vars,
            points,
            challenges,
            inner_sums,
            &mut self.coeffs_out,
        )?;
        Ok(())
    }

    /// Device coefficients for the current Ajtai-tail round. The row phase
    /// has already folded to `prefix = alpha'_0..alpha'_{j-1}`; this evaluates
    /// the next bit from the resident `Y_eval` surface.
    pub fn ajtai_tail_round_coeffs(
        &mut self,
        device: &Device,
        tail_module: &FeTailKernelModule,
        y_eval: &DeviceAjtaiYEval,
        params: DeviceFeTailRound<'_>,
    ) -> Result<Vec<K>, CcsDeviceError> {
        if params.alpha.len() != params.beta_a.len() {
            return Err(CcsDeviceError::Shape("Ajtai tail alpha/beta length mismatch"));
        }
        let round = params.prefix.len();
        if round >= params.alpha.len() {
            return Err(CcsDeviceError::Shape("Ajtai tail round out of range"));
        }
        if params.k_mcs > y_eval.witnesses {
            return Err(CcsDeviceError::Shape("Ajtai tail MCS count exceeds Y_eval witnesses"));
        }

        let width = self.coeff_width;
        let (eq_beta_re, eq_beta_im) = params.eq_beta_r.to_limbs_u64();
        let (eq_inputs_re, eq_inputs_im) = params.eq_r_inputs.to_limbs_u64();
        let (gamma_re, gamma_im) = params.gamma.to_limbs_u64();
        let mut gamma_to_k = K::ONE;
        for _ in 0..y_eval.witnesses {
            gamma_to_k *= params.gamma;
        }
        let (gamma_k_re, gamma_k_im) = gamma_to_k.to_limbs_u64();
        let header_words = [
            params.k_mcs as u64,
            y_eval.witnesses as u64,
            y_eval.matrices as u64,
            params.alpha.len() as u64,
            round as u64,
            width as u64,
            params.has_inputs as u64,
            eq_beta_re,
            eq_beta_im,
            eq_inputs_re,
            eq_inputs_im,
            gamma_re,
            gamma_im,
            gamma_k_re,
            gamma_k_im,
        ];
        let mut points_words = Vec::with_capacity(2 * (2 * params.alpha.len() + round));
        for value in params.alpha {
            let (re, im) = value.to_limbs_u64();
            points_words.extend([re, im]);
        }
        for value in params.beta_a {
            let (re, im) = value.to_limbs_u64();
            points_words.extend([re, im]);
        }
        for value in params.prefix {
            let (re, im) = value.to_limbs_u64();
            points_words.extend([re, im]);
        }

        let stream = device.stream();
        let header = DeviceBuffer::from_host(stream, &header_words)?;
        let points = DeviceBuffer::from_host(stream, &points_words)?;
        let mut coeffs = DeviceBuffer::zeroed(stream, width * 2)?;
        launch_ajtai_tail_round_coeffs(
            tail_module,
            stream,
            &y_eval.words,
            &header,
            &self.mcs_meta,
            &self.term_meta,
            &self.term_vars,
            &points,
            &mut coeffs,
        )?;
        let words = coeffs.to_host_vec(stream)?;
        Ok((0..width)
            .map(|idx| k_from_device_words(words[2 * idx], words[2 * idx + 1]))
            .collect())
    }
}

impl DeviceFeBackend<'_> {
    pub fn ajtai_tail_round_coeffs(
        &mut self,
        y_eval: &DeviceAjtaiYEval,
        params: DeviceFeTailRound<'_>,
    ) -> Result<Vec<K>, CcsDeviceError> {
        self.oracle
            .as_mut()
            .ok_or(CcsDeviceError::Shape("FE backend used before start"))?
            .ajtai_tail_round_coeffs(self.device, &self.kernels.tail, y_eval, params)
    }
}

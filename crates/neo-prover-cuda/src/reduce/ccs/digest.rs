//! Device-side Pi_CCS output digest construction.
//!
//! Owns the prover-side handoff from resident Pi_CCS K-surfaces to the
//! canonical `pi_ccs_outputs_digest` field digest. It does not define the
//! digest protocol; the field order mirrors `neo-fold-clean::paper::digest`.

use cuda_core::{DeviceBuffer, PinnedHostBuffer};
use neo_ajtai::Commitment;
use neo_ccs::{CeClaim, Mat};
use neo_math::{F, K};
use p3_field::PrimeCharacteristicRing;

use crate::device::{uninit_u64_device_buffer, upload_u64_device_buffer, Device};
use crate::kernels::poseidon2::DIGEST_LEN;
use crate::kernels::sis::launch_pi_ccs_outputs_preimage;
use crate::reduce::ccs::{CcsDeviceError, DevicePiCcsKSurfaces, SumcheckKernels};
use crate::sis::DeviceSisCache;

use neo_fold_clean::paper::reductions::accumulator_sis_circuit::PI_CCS_OUTPUTS_SIS_CONFIG;

const OUTPUTS_DIGEST_DOMAIN: &[u8] = b"neo.fold.clean/pi_ccs_outputs_digest/v2";
const OUTPUT_CLAIM_CHALLENGE_DOMAIN: &[u8] = b"neo.fold.clean/pi_ccs_output_challenge_digest/v2";
const BYTES_PER_PACKED_LIMB: usize = 7;

pub(crate) fn pi_ccs_outputs_digest_field_count(
    claims: usize,
    t_core: usize,
    d_pad: usize,
    include_y_zcol: bool,
) -> usize {
    let output_domain_len = pack_bytes_as_words(OUTPUTS_DIGEST_DOMAIN).len();
    let claim_domain_len = pack_bytes_as_words(OUTPUT_CLAIM_CHALLENGE_DOMAIN).len();
    let surface_span = 1 + 2 * d_pad;
    let claim_len = claim_domain_len + 1 + t_core * surface_span + if include_y_zcol { surface_span } else { 1 };
    output_domain_len + 1 + claims * claim_len
}

/// Four-field Poseidon2 digest produced entirely from device-resident output
/// surfaces plus the public CE claim shell.
pub struct DevicePiCcsOutputsDigest {
    words: DeviceBuffer<u64>,
}

/// Host-visible non-surface fields of one Pi_CCS output claim.
///
/// The expensive K-valued output rows (`y_ring`, `ct`, `y_zcol`) stay in
/// [`DevicePiCcsKSurfaces`]. This shell is the small public/proof metadata
/// needed to build the canonical output digest preimage around those resident
/// device surfaces.
pub struct PiCcsOutputDigestShell<'a> {
    pub c: &'a Commitment,
    pub x: &'a Mat<F>,
    pub r: &'a [K],
    pub s_col: &'a [K],
    pub aux_openings: &'a [K],
    pub m_in: usize,
    pub fold_digest: [u8; 32],
    pub c_step_coords: &'a [F],
    pub u_offset: usize,
    pub u_len: usize,
}

impl<'a> PiCcsOutputDigestShell<'a> {
    pub fn from_claim(claim: &'a CeClaim<Commitment, F, K>) -> Self {
        Self {
            c: &claim.c,
            x: &claim.X,
            r: &claim.r,
            s_col: &claim.s_col,
            aux_openings: &claim.aux_openings,
            m_in: claim.m_in,
            fold_digest: claim.fold_digest,
            c_step_coords: &claim.c_step_coords,
            u_offset: claim.u_offset,
            u_len: claim.u_len,
        }
    }
}

impl DevicePiCcsOutputsDigest {
    pub fn compute(
        device: &Device,
        kernels: &SumcheckKernels,
        claims: &[CeClaim<Commitment, F, K>],
        surfaces: &DevicePiCcsKSurfaces,
    ) -> Result<Self, CcsDeviceError> {
        let plan = OutputDigestPlan::from_claims(claims, surfaces)?;
        Self::compute_from_plan(device, kernels, &mut DeviceSisCache::default(), surfaces, plan)
    }

    pub fn compute_from_shells(
        device: &Device,
        kernels: &SumcheckKernels,
        shells: &[PiCcsOutputDigestShell<'_>],
        surfaces: &DevicePiCcsKSurfaces,
    ) -> Result<Self, CcsDeviceError> {
        let plan = OutputDigestPlan::from_shells(shells, surfaces)?;
        Self::compute_from_plan(device, kernels, &mut DeviceSisCache::default(), surfaces, plan)
    }

    pub(crate) fn compute_from_shells_with_cache(
        device: &Device,
        kernels: &SumcheckKernels,
        sis: &mut DeviceSisCache,
        shells: &[PiCcsOutputDigestShell<'_>],
        surfaces: &DevicePiCcsKSurfaces,
    ) -> Result<Self, CcsDeviceError> {
        let plan = OutputDigestPlan::from_shells(shells, surfaces)?;
        Self::compute_from_plan(device, kernels, sis, surfaces, plan)
    }

    fn compute_from_plan(
        device: &Device,
        kernels: &SumcheckKernels,
        sis: &mut DeviceSisCache,
        surfaces: &DevicePiCcsKSurfaces,
        plan: OutputDigestPlan,
    ) -> Result<Self, CcsDeviceError> {
        validate_v2_surface_count(&plan, surfaces)?;
        let output_domain = pack_bytes_as_words(OUTPUTS_DIGEST_DOMAIN);
        let claim_domain = pack_bytes_as_words(OUTPUT_CLAIM_CHALLENGE_DOMAIN);
        let mut domains = Vec::with_capacity(output_domain.len() + claim_domain.len());
        domains.extend_from_slice(&output_domain);
        domains.extend_from_slice(&claim_domain);
        let domains = upload_u64_device_buffer(device.stream(), &domains)?;
        let field_count = pi_ccs_outputs_digest_field_count(
            surfaces.claims(),
            surfaces.t_core(),
            surfaces.d_pad(),
            surfaces.include_y_zcol(),
        );
        let mut preimage = uninit_u64_device_buffer(device.stream(), field_count)?;
        launch_pi_ccs_outputs_preimage(
            sis.module(device)?,
            device.stream(),
            surfaces.words(),
            surfaces.claims(),
            surfaces.t_core(),
            surfaces.d_pad(),
            surfaces.include_y_zcol(),
            &domains,
            output_domain.len(),
            claim_domain.len(),
            &mut preimage,
        )?;
        let words = sis.digest_device(device, kernels, PI_CCS_OUTPUTS_SIS_CONFIG, &preimage, field_count)?;
        Ok(Self { words })
    }

    pub fn words(&self) -> &DeviceBuffer<u64> {
        &self.words
    }

    pub fn download(&self, device: &Device) -> Result<[F; DIGEST_LEN], CcsDeviceError> {
        let words = self.enqueue_download(device)?;
        device.sync()?;
        Self::decode_download(words.as_slice())
    }

    pub fn enqueue_download(&self, device: &Device) -> Result<PinnedHostBuffer<u64>, CcsDeviceError> {
        let mut words = PinnedHostBuffer::zeroed(device.ctx(), DIGEST_LEN)?;
        // SAFETY: the returned buffer remains owned by the caller until the
        // later transcript-snapshot synchronization completes this copy.
        unsafe {
            self.words
                .copy_to_pinned_host_async(device.stream(), &mut words)?;
        }
        Ok(words)
    }

    pub fn decode_download(words: &[u64]) -> Result<[F; DIGEST_LEN], CcsDeviceError> {
        if words.len() != DIGEST_LEN {
            return Err(CcsDeviceError::Shape("Pi_CCS output digest word count mismatch"));
        }
        Ok(std::array::from_fn(|i| F::from_u64(words[i])))
    }
}

fn validate_v2_surface_count(plan: &OutputDigestPlan, surfaces: &DevicePiCcsKSurfaces) -> Result<(), CcsDeviceError> {
    if plan.claims != surfaces.claims() {
        return Err(CcsDeviceError::Shape("Pi_CCS v2 output authority claim count mismatch"));
    }
    Ok(())
}

struct OutputDigestPlan {
    claims: usize,
}

impl OutputDigestPlan {
    fn from_claims(
        claims: &[CeClaim<Commitment, F, K>],
        surfaces: &DevicePiCcsKSurfaces,
    ) -> Result<Self, CcsDeviceError> {
        if claims.len() != surfaces.claims() {
            return Err(CcsDeviceError::Shape("Pi_CCS output digest claim count mismatch"));
        }
        let shells = claims
            .iter()
            .map(|claim| {
                check_surface_shape(claim, surfaces)?;
                Ok(PiCcsOutputDigestShell::from_claim(claim))
            })
            .collect::<Result<Vec<_>, CcsDeviceError>>()?;
        Ok(Self { claims: shells.len() })
    }

    fn from_shells(
        shells: &[PiCcsOutputDigestShell<'_>],
        surfaces: &DevicePiCcsKSurfaces,
    ) -> Result<Self, CcsDeviceError> {
        if shells.len() != surfaces.claims() {
            return Err(CcsDeviceError::Shape("Pi_CCS output digest shell count mismatch"));
        }
        for shell in shells {
            check_shell_shape(shell)?;
        }
        Ok(Self { claims: shells.len() })
    }
}

fn check_surface_shape(
    claim: &CeClaim<Commitment, F, K>,
    surfaces: &DevicePiCcsKSurfaces,
) -> Result<(), CcsDeviceError> {
    if claim.y_ring.len() != surfaces.t_core() {
        return Err(CcsDeviceError::Shape("Pi_CCS output digest y_ring row count mismatch"));
    }
    if claim.y_ring.iter().any(|row| row.len() != surfaces.d_pad()) {
        return Err(CcsDeviceError::Shape("Pi_CCS output digest y_ring row width mismatch"));
    }
    if claim.ct.len() != surfaces.t_core() {
        return Err(CcsDeviceError::Shape("Pi_CCS output digest ct length mismatch"));
    }
    let y_zcol_len = if surfaces.include_y_zcol() { surfaces.d_pad() } else { 0 };
    if claim.y_zcol.len() != y_zcol_len {
        return Err(CcsDeviceError::Shape("Pi_CCS output digest y_zcol width mismatch"));
    }
    Ok(())
}

fn check_shell_shape(shell: &PiCcsOutputDigestShell<'_>) -> Result<(), CcsDeviceError> {
    let active_x_cols = shell.m_in.div_ceil(neo_math::D);
    if active_x_cols > shell.x.cols() {
        return Err(CcsDeviceError::Shape(
            "Pi_CCS output digest X active cols exceed matrix width",
        ));
    }
    Ok(())
}

pub(super) fn pack_bytes_as_words(bytes: &[u8]) -> Vec<u64> {
    let mut out = Vec::with_capacity(1 + bytes.len().div_ceil(BYTES_PER_PACKED_LIMB));
    out.push(bytes.len() as u64);
    for chunk in bytes.chunks(BYTES_PER_PACKED_LIMB) {
        let mut limb = [0u8; 8];
        limb[..chunk.len()].copy_from_slice(chunk);
        out.push(u64::from_le_bytes(limb));
    }
    out
}

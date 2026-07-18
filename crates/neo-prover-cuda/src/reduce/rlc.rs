//! Device Π_RLC rho sampling and witness mix.
//!
//! Owns the device-side parts of Π_RLC: sampling the rotation-ring
//! coefficients from a Poseidon2 transcript snapshot, and the bulk witness
//! combination `Z_mix = Σ ρ_i · Z_i`. The current proof object is still
//! assembled on the host, so the rho coefficients are downloaded only when a
//! host consumer still needs the canonical rotation-matrix view.

use std::sync::Arc;

use cuda_core::{CudaStream, DeviceBuffer};
use neo_ajtai::Commitment;
use neo_ccs::{CeClaim, Mat};
use neo_fold_clean::engine::transcript::Poseidon2TranscriptSnapshot;
use neo_fold_clean::paper::params::Params;
use neo_fold_clean::paper::pi_rlc::PI_RLC_INPUT_CLAIMS_DIGEST_LABEL;
use neo_math::{KExtensions, D, F, K};
use neo_reductions::common::RotRing;
use p3_field::{PrimeCharacteristicRing, PrimeField64};

use crate::device::{uninit_u64_device_buffer, upload_u64_device_buffer, zeroed_u64_device_buffer, Device};
use crate::field::{f_from_device_word, k_from_device_words};
use crate::ingest::upload_witness_planes;
use crate::kernels::ajtai::{
    launch_plane_copy, launch_rlc_mix, launch_rlc_mix_retained, AjtaiKernelModule, PendingRlcMix,
};
use crate::kernels::pi_rlc::{
    launch_rlc_combine_k_surfaces, launch_rlc_pack_active_public_x, launch_rlc_pack_public_x, RlcKernelModule,
};
use crate::reduce::ccs::{CcsDeviceError, DevicePiCcsKSurfaces, DevicePublicX, SumcheckKernels};
use crate::ring_layout;
use crate::transcript::DeviceTranscript;

pub struct DeviceRhos {
    coeffs: DeviceBuffer<u64>,
    coeff_words: Option<Vec<u64>>,
    first_coeffs: Option<Vec<F>>,
    count: usize,
}

pub struct PendingRlcTranscriptSnapshot {
    transcript: DeviceTranscript,
}

impl PendingRlcTranscriptSnapshot {
    pub fn finish(self, device: &Device) -> Result<Poseidon2TranscriptSnapshot, CcsDeviceError> {
        let words = self.transcript.state_words_to_host(device)?;
        device.sync()?;
        let (state, absorbed) = DeviceTranscript::decode_state_words(&words);
        Ok(Poseidon2TranscriptSnapshot::from_state_and_absorbed(state, absorbed))
    }
}

impl DeviceRhos {
    pub fn coeffs(&self) -> &DeviceBuffer<u64> {
        &self.coeffs
    }

    pub fn count(&self) -> usize {
        self.count
    }

    pub fn first_coeffs(&mut self, device: &Device) -> Result<&[F], CcsDeviceError> {
        self.ensure_coeff_words(device)?;
        Ok(self
            .first_coeffs
            .as_deref()
            .expect("first rho coefficients cached above"))
    }

    pub fn mats(&mut self, device: &Device, pp: &Params) -> Result<Vec<Mat<F>>, CcsDeviceError> {
        let count = self.count;
        let coeff_words = self.ensure_coeff_words(device)?;
        rho_mats_from_coeff_words(pp, coeff_words, count)
    }

    fn ensure_coeff_words(&mut self, device: &Device) -> Result<&[u64], CcsDeviceError> {
        if self.coeff_words.is_none() {
            let coeff_words = self.coeffs.to_host_vec(device.stream())?;
            device.sync()?;
            self.first_coeffs = Some(
                (0..self.count)
                    .map(|rho| f_from_device_word(coeff_words[rho * D]))
                    .collect(),
            );
            self.coeff_words = Some(coeff_words);
        }
        Ok(self
            .coeff_words
            .as_deref()
            .expect("rho coefficient words cached above"))
    }
}

/// `ell_d = log2(next_power_of_two(D))` — the padded ring-degree exponent
/// Π_RLC claims are combined under.
pub fn ell_d() -> usize {
    D.next_power_of_two().trailing_zeros() as usize
}

pub fn sample_rhos_device(
    device: &Device,
    kernels: &SumcheckKernels,
    pp: &Params,
    snapshot: Poseidon2TranscriptSnapshot,
    count: usize,
) -> Result<(DeviceRhos, Poseidon2TranscriptSnapshot), CcsDeviceError> {
    let (rhos, pending) = sample_rhos_device_deferred(device, kernels, pp, snapshot, count)?;
    Ok((rhos, pending.finish(device)?))
}

pub fn sample_rhos_device_deferred(
    device: &Device,
    kernels: &SumcheckKernels,
    pp: &Params,
    snapshot: Poseidon2TranscriptSnapshot,
    count: usize,
) -> Result<(DeviceRhos, PendingRlcTranscriptSnapshot), CcsDeviceError> {
    check_goldilocks_rho_profile(pp)?;
    let mut transcript = DeviceTranscript::from_state_and_absorbed(device, snapshot.state(), snapshot.absorbed())?;
    let mut coeffs = uninit_u64_device_buffer(device.stream(), count * D)?;
    transcript.enqueue_sample_rlc_rhos(device, &kernels.poseidon, &kernels.poseidon_rc, count, &mut coeffs)?;
    Ok((
        DeviceRhos {
            coeffs,
            coeff_words: None,
            first_coeffs: None,
            count,
        },
        PendingRlcTranscriptSnapshot { transcript },
    ))
}

pub fn sample_rhos_from_device_outputs_digest_deferred(
    device: &Device,
    kernels: &SumcheckKernels,
    pp: &Params,
    snapshot_before_bind: Poseidon2TranscriptSnapshot,
    outputs_digest: &DeviceBuffer<u64>,
    count: usize,
) -> Result<(DeviceRhos, PendingRlcTranscriptSnapshot), CcsDeviceError> {
    check_goldilocks_rho_profile(pp)?;
    if outputs_digest.len() != 4 || count == 0 {
        return Err(CcsDeviceError::Shape(
            "Π_RLC: resident output digest/count shape mismatch",
        ));
    }
    let mut transcript = DeviceTranscript::from_state_and_absorbed(
        device,
        snapshot_before_bind.state(),
        snapshot_before_bind.absorbed(),
    )?;
    let mut coeffs = uninit_u64_device_buffer(device.stream(), count * D)?;
    transcript.enqueue_bind_device_fields_sample_rlc_rhos(
        device,
        &kernels.poseidon,
        &kernels.poseidon_rc,
        PI_RLC_INPUT_CLAIMS_DIGEST_LABEL,
        outputs_digest,
        count,
        &mut coeffs,
    )?;
    Ok((
        DeviceRhos {
            coeffs,
            coeff_words: None,
            first_coeffs: None,
            count,
        },
        PendingRlcTranscriptSnapshot { transcript },
    ))
}

/// `Z_mix = Σ ρ_i · Z_i` on device. `rho_mats` are the D×D rotation
/// matrices (only the first column — the ring element's coefficients — is
/// consumed, exactly like the CPU rotation-ring path). Witnesses must share
/// the packed width.
pub fn mix_witnesses(
    device: &Device,
    ring: &AjtaiKernelModule,
    rho_mats: &[Mat<F>],
    witnesses: &[&Mat<F>],
) -> Result<Mat<F>, CcsDeviceError> {
    if witnesses.is_empty() || rho_mats.len() != witnesses.len() {
        return Err(CcsDeviceError::Shape("Π_RLC: |rhos| must equal |witnesses| (≥ 1)"));
    }
    let cols = witnesses[0].cols();
    if witnesses.iter().any(|w| w.rows() != D || w.cols() != cols) {
        return Err(CcsDeviceError::Shape("Π_RLC: witnesses must be D × shared width"));
    }
    if rho_mats
        .iter()
        .any(|rho| rho.rows() != D || rho.cols() != D)
    {
        return Err(CcsDeviceError::Shape("Π_RLC: rhos must be D × D rotation matrices"));
    }
    let planes = upload_witness_planes(device, witnesses)?;
    mix_planes(device, ring, rho_mats, &planes, witnesses.len(), cols)
}

/// [`mix_witnesses`] against planes already on device (`[k1][cols * D]`,
/// the `upload_witness_planes` layout), downloaded to a host witness.
pub fn mix_planes(
    device: &Device,
    ring: &AjtaiKernelModule,
    rho_mats: &[Mat<F>],
    planes: &DeviceBuffer<u64>,
    k1: usize,
    cols: usize,
) -> Result<Mat<F>, CcsDeviceError> {
    let out = mix_planes_device(device, ring, rho_mats, planes, k1, cols)?;
    download_witness(device, &out, cols)
}

/// [`mix_planes`] without the download: the mixed witness stays on device
/// (`cols * D` words, standard column-major layout) for a consumer on the
/// same stream — Π_DEC splits it in place. The planes MUST be the flattened
/// witnesses in ρ order — callers own that contract; the NIFS parity gate
/// enforces it bit-exactly.
pub fn mix_planes_device(
    device: &Device,
    ring: &AjtaiKernelModule,
    rho_mats: &[Mat<F>],
    planes: &DeviceBuffer<u64>,
    k1: usize,
    cols: usize,
) -> Result<DeviceBuffer<u64>, CcsDeviceError> {
    if k1 == 0 || rho_mats.len() != k1 || planes.len() != k1 * cols * D {
        return Err(CcsDeviceError::Shape("Π_RLC: plane shape must match |rhos|"));
    }
    if rho_mats
        .iter()
        .any(|rho| rho.rows() != D || rho.cols() != D)
    {
        return Err(CcsDeviceError::Shape("Π_RLC: rhos must be D × D rotation matrices"));
    }
    let mut rho_words = vec![0u64; k1 * D];
    for (i, rho) in rho_mats.iter().enumerate() {
        for row in 0..D {
            rho_words[i * D + row] = rho[(row, 0)].as_canonical_u64();
        }
    }
    let stream = device.stream();
    let rhos_dev = upload_u64_device_buffer(stream, &rho_words)?;
    Ok(launch_rlc_mix(ring, stream, &rhos_dev, planes, k1, cols)?)
}

pub fn mix_planes_device_with_rho_coeffs(
    device: &Device,
    ring: &AjtaiKernelModule,
    rho_coeffs: &DeviceBuffer<u64>,
    planes: &DeviceBuffer<u64>,
    k1: usize,
    cols: usize,
) -> Result<DeviceBuffer<u64>, CcsDeviceError> {
    if k1 == 0 || rho_coeffs.len() != k1 * D || planes.len() != k1 * cols * D {
        return Err(CcsDeviceError::Shape("Π_RLC: device rho/plane shape mismatch"));
    }
    Ok(mix_planes_device_with_rho_coeffs_retained(device, ring, rho_coeffs, planes, k1, cols)?.into_words())
}

pub struct DeviceMixedWitness {
    mix: PendingRlcMix,
}

impl DeviceMixedWitness {
    pub fn words(&self) -> &DeviceBuffer<u64> {
        self.mix.out()
    }

    pub fn into_words(self) -> DeviceBuffer<u64> {
        self.mix.into_out()
    }
}

pub fn mix_planes_device_with_rho_coeffs_retained(
    device: &Device,
    ring: &AjtaiKernelModule,
    rho_coeffs: &DeviceBuffer<u64>,
    planes: &DeviceBuffer<u64>,
    k1: usize,
    cols: usize,
) -> Result<DeviceMixedWitness, CcsDeviceError> {
    if k1 == 0 || rho_coeffs.len() != k1 * D || planes.len() != k1 * cols * D {
        return Err(CcsDeviceError::Shape("Π_RLC: device rho/plane shape mismatch"));
    }
    let mix = launch_rlc_mix_retained(ring, device.stream(), rho_coeffs, planes, k1, cols)?;
    Ok(DeviceMixedWitness { mix })
}

/// Device-side `Σ ρ_i · c_i` over Ajtai commitments.
///
/// Commitment storage is `[kappa][D]`, the same column-major ring-vector
/// layout as witness planes, so the Π_RLC ring-mix kernel applies directly.
/// Download one device-resident witness (`cols * D` words, standard
/// column-major layout) back to a host `Mat`.
pub fn download_witness(device: &Device, witness: &DeviceBuffer<u64>, cols: usize) -> Result<Mat<F>, CcsDeviceError> {
    let words = witness.to_host_vec(device.stream())?;
    device.sync()?;
    Ok(ring_layout::mat_from_words(&words, cols))
}

/// Device-side `Σ rho_i · c_i` for Ajtai commitments.
///
/// Commitments are stored as `kappa` ring columns of width `D`, exactly the
/// same column-major layout consumed by the Π_RLC witness-mix kernel. This
/// keeps rho coefficients device-resident through the commitment boundary;
/// callers still download the single mixed commitment because it is part of
/// the public proof surface.
pub fn mix_commitments_device_with_rho_coeffs(
    device: &Device,
    ring: &AjtaiKernelModule,
    rho_coeffs: &DeviceBuffer<u64>,
    commitments: &[Commitment],
) -> Result<Commitment, CcsDeviceError> {
    let first = commitments
        .first()
        .ok_or(CcsDeviceError::Shape("Π_RLC: commitment mix needs inputs"))?;
    if first.d != D {
        return Err(CcsDeviceError::Shape("Π_RLC: commitment ring width mismatch"));
    }
    if commitments
        .iter()
        .any(|commitment| commitment.d != D || commitment.kappa != first.kappa)
    {
        return Err(CcsDeviceError::Shape("Π_RLC: commitment shape mismatch"));
    }
    if rho_coeffs.len() != commitments.len() * D {
        return Err(CcsDeviceError::Shape("Π_RLC: commitment rho shape mismatch"));
    }

    let mut words = Vec::with_capacity(commitments.len() * first.kappa * D);
    for commitment in commitments {
        words.extend(commitment.data.iter().map(|value| value.as_canonical_u64()));
    }
    let stream = device.stream();
    let commitments_dev = upload_u64_device_buffer(stream, &words)?;
    mix_commitments_device_words(
        device,
        ring,
        rho_coeffs,
        &commitments_dev,
        commitments.len(),
        first.kappa,
    )
}

/// Compose already-resident commitment segments into the Π_RLC input order.
///
/// The pieces are copied device-to-device into one `[input][kappa][D]`
/// buffer. This is the commitment analogue of fold-plane composition: it
/// preserves the CPU-visible claim order without re-uploading commitments.
pub fn compose_commitment_words_device(
    device: &Device,
    ring: &AjtaiKernelModule,
    pieces: &[&DeviceBuffer<u64>],
    total_words: usize,
) -> Result<DeviceBuffer<u64>, CcsDeviceError> {
    if pieces.is_empty() || total_words == 0 {
        return Err(CcsDeviceError::Shape("Π_RLC: resident commitment pieces missing"));
    }
    if pieces.iter().map(|piece| piece.len()).sum::<usize>() != total_words {
        return Err(CcsDeviceError::Shape(
            "Π_RLC: resident commitment piece length mismatch",
        ));
    }
    let stream = device.stream();
    let mut out = uninit_u64_device_buffer(stream, total_words)?;
    let mut offset = 0usize;
    for piece in pieces {
        launch_plane_copy(ring, stream, piece, offset, &mut out)?;
        offset += piece.len();
    }
    Ok(out)
}

/// Device-side `Σ rho_i · c_i` from resident commitment words.
pub fn mix_commitments_device_words(
    device: &Device,
    ring: &AjtaiKernelModule,
    rho_coeffs: &DeviceBuffer<u64>,
    commitment_words: &DeviceBuffer<u64>,
    count: usize,
    kappa: usize,
) -> Result<Commitment, CcsDeviceError> {
    if count == 0 || kappa == 0 {
        return Err(CcsDeviceError::Shape("Π_RLC: commitment mix needs inputs"));
    }
    if rho_coeffs.len() != count * D || commitment_words.len() != count * kappa * D {
        return Err(CcsDeviceError::Shape("Π_RLC: resident commitment shape mismatch"));
    }
    let pending = enqueue_mix_commitments_device_words(
        Arc::clone(device.stream()),
        ring,
        rho_coeffs,
        commitment_words,
        count,
        kappa,
    )?;
    finish_mixed_commitment(device.stream(), pending)
}

pub struct PendingMixedCommitment {
    stream: Arc<CudaStream>,
    _commitment_words: Option<DeviceBuffer<u64>>,
    mix: PendingRlcMix,
    kappa: usize,
}

pub fn enqueue_mix_commitments_device_words(
    stream: Arc<CudaStream>,
    ring: &AjtaiKernelModule,
    rho_coeffs: &DeviceBuffer<u64>,
    commitment_words: &DeviceBuffer<u64>,
    count: usize,
    kappa: usize,
) -> Result<PendingMixedCommitment, CcsDeviceError> {
    if count == 0 || kappa == 0 {
        return Err(CcsDeviceError::Shape("Π_RLC: commitment mix needs inputs"));
    }
    if rho_coeffs.len() != count * D || commitment_words.len() != count * kappa * D {
        return Err(CcsDeviceError::Shape("Π_RLC: resident commitment shape mismatch"));
    }
    let mix = launch_rlc_mix_retained(ring, &stream, rho_coeffs, commitment_words, count, kappa)?;
    Ok(PendingMixedCommitment {
        stream,
        _commitment_words: None,
        mix,
        kappa,
    })
}

pub fn enqueue_owned_mix_commitments_device_words(
    stream: Arc<CudaStream>,
    ring: &AjtaiKernelModule,
    rho_coeffs: &DeviceBuffer<u64>,
    commitment_words: DeviceBuffer<u64>,
    count: usize,
    kappa: usize,
) -> Result<PendingMixedCommitment, CcsDeviceError> {
    if count == 0 || kappa == 0 {
        return Err(CcsDeviceError::Shape("Π_RLC: commitment mix needs inputs"));
    }
    if rho_coeffs.len() != count * D || commitment_words.len() != count * kappa * D {
        return Err(CcsDeviceError::Shape("Π_RLC: resident commitment shape mismatch"));
    }
    let mix = launch_rlc_mix_retained(ring, &stream, rho_coeffs, &commitment_words, count, kappa)?;
    Ok(PendingMixedCommitment {
        stream,
        _commitment_words: Some(commitment_words),
        mix,
        kappa,
    })
}

pub fn finish_mixed_commitment(
    main_stream: &Arc<CudaStream>,
    pending: PendingMixedCommitment,
) -> Result<Commitment, CcsDeviceError> {
    main_stream.join(&pending.stream)?;
    let mixed_words = pending.mix.out().to_host_vec(main_stream)?;
    main_stream.synchronize()?;
    Ok(Commitment {
        d: D,
        kappa: pending.kappa,
        data: mixed_words.into_iter().map(f_from_device_word).collect(),
    })
}

pub(crate) fn finish_mixed_commitment_device(
    main_stream: &Arc<CudaStream>,
    pending: PendingMixedCommitment,
) -> Result<(DeviceBuffer<u64>, Vec<DeviceBuffer<u64>>, usize), CcsDeviceError> {
    let PendingMixedCommitment {
        stream,
        _commitment_words,
        mix,
        kappa,
    } = pending;
    main_stream.join(&stream)?;
    let (scratch, words) = mix.into_parts();
    let mut keepalive = vec![scratch];
    keepalive.extend(_commitment_words);
    Ok((words, keepalive, kappa))
}

pub fn project_x_from_mixed_witness(
    device: &Device,
    module: &RlcKernelModule,
    z_mix: &DeviceBuffer<u64>,
    z_cols: usize,
    expected_m: usize,
    m_in: usize,
) -> Result<Mat<F>, CcsDeviceError> {
    if m_in > expected_m {
        return Err(CcsDeviceError::Shape("Π_RLC: public input width exceeds CCS width"));
    }
    if z_mix.len() != z_cols * D || m_in.div_ceil(D) > z_cols {
        return Err(CcsDeviceError::Shape("Π_RLC: mixed witness shape cannot project X"));
    }
    if m_in == 0 {
        return Ok(Mat::zero(D, 0, F::ZERO));
    }

    let stream = device.stream();
    let mut x_words = zeroed_u64_device_buffer(stream, D * m_in)?;
    launch_rlc_pack_public_x(module, stream, z_mix, m_in, z_cols, &mut x_words)?;
    let words = x_words.to_host_vec(stream)?;
    device.sync()?;
    Ok(Mat::from_row_major(
        D,
        m_in,
        words.into_iter().map(f_from_device_word).collect(),
    ))
}

pub enum PendingProjectedX {
    Empty,
    Pending {
        stream: Arc<CudaStream>,
        out_dev: DeviceBuffer<u64>,
        m_in: usize,
    },
}

pub fn enqueue_project_x_from_mixed_witness(
    stream: Arc<CudaStream>,
    module: &RlcKernelModule,
    z_mix: &DeviceBuffer<u64>,
    z_cols: usize,
    expected_m: usize,
    m_in: usize,
) -> Result<PendingProjectedX, CcsDeviceError> {
    if m_in > expected_m {
        return Err(CcsDeviceError::Shape("Π_RLC: public input width exceeds CCS width"));
    }
    if z_mix.len() != z_cols * D || m_in.div_ceil(D) > z_cols {
        return Err(CcsDeviceError::Shape("Π_RLC: mixed witness shape cannot project X"));
    }
    if m_in == 0 {
        return Ok(PendingProjectedX::Empty);
    }

    let active_cols = m_in.div_ceil(D);
    let mut out_dev = uninit_u64_device_buffer(&stream, (D * active_cols).max(1))?;
    launch_rlc_pack_active_public_x(module, &stream, z_mix, 1, z_cols * D, m_in, &mut out_dev)?;
    Ok(PendingProjectedX::Pending { stream, out_dev, m_in })
}

pub fn finish_projected_x(main_stream: &Arc<CudaStream>, pending: PendingProjectedX) -> Result<Mat<F>, CcsDeviceError> {
    let device_x = finish_projected_x_device(main_stream, pending)?;
    Ok(device_x
        .materialize()?
        .into_iter()
        .next()
        .expect("one projected X claim"))
}

pub(crate) fn finish_projected_x_device(
    main_stream: &Arc<CudaStream>,
    pending: PendingProjectedX,
) -> Result<DevicePublicX, CcsDeviceError> {
    match pending {
        PendingProjectedX::Empty => {
            let words = zeroed_u64_device_buffer(main_stream, 1)?;
            DevicePublicX::new(Arc::clone(main_stream), words, 1, 0)
        }
        PendingProjectedX::Pending { stream, out_dev, m_in } => {
            main_stream.join(&stream)?;
            DevicePublicX::new(Arc::clone(main_stream), out_dev, 1, m_in)
        }
    }
}

pub fn claim_shell(
    inputs: &[CeClaim<Commitment, F, K>],
    rhos: &[Mat<F>],
    commitment: Commitment,
) -> Result<CeClaim<Commitment, F, K>, CcsDeviceError> {
    let first_coeffs: Vec<F> = rhos.iter().map(|rho| rho[(0, 0)]).collect();
    claim_shell_from_first_coeffs(inputs, &first_coeffs, commitment)
}

pub fn claim_shell_from_first_coeffs(
    inputs: &[CeClaim<Commitment, F, K>],
    rho_first_coeffs: &[F],
    commitment: Commitment,
) -> Result<CeClaim<Commitment, F, K>, CcsDeviceError> {
    let aux_len = validate_claim_shell_inputs(inputs, rho_first_coeffs.len())?;
    let aux_openings = mix_aux_openings(inputs, rho_first_coeffs, aux_len);
    claim_shell_from_aux_openings(inputs, commitment, aux_openings)
}

pub fn claim_shell_from_device_rhos(
    device: &Device,
    inputs: &[CeClaim<Commitment, F, K>],
    rhos: &mut DeviceRhos,
    commitment: Commitment,
) -> Result<CeClaim<Commitment, F, K>, CcsDeviceError> {
    let aux_len = validate_claim_shell_inputs(inputs, rhos.count())?;
    let aux_openings = if aux_len == 0 {
        Vec::new()
    } else {
        let rho_first_coeffs = rhos.first_coeffs(device)?;
        mix_aux_openings(inputs, rho_first_coeffs, aux_len)
    };
    claim_shell_from_aux_openings(inputs, commitment, aux_openings)
}

pub struct ClaimShellMetadata<'a> {
    pub count: usize,
    pub m_in: usize,
    pub r: &'a [K],
    pub s_col: &'a [K],
    pub has_y_zcol: bool,
    pub fold_digest: [u8; 32],
}

pub fn claim_shell_from_metadata(
    metadata: ClaimShellMetadata<'_>,
    rho_count: usize,
    commitment: Commitment,
) -> Result<CeClaim<Commitment, F, K>, CcsDeviceError> {
    if metadata.count == 0 || metadata.count != rho_count {
        return Err(CcsDeviceError::Shape("Π_RLC: claim metadata rho/input mismatch"));
    }
    Ok(CeClaim {
        adv: None,
        c_step_coords: vec![],
        u_offset: 0,
        u_len: 0,
        c: commitment,
        X: Mat::zero(D, metadata.m_in, F::ZERO),
        r: metadata.r.to_vec(),
        s_col: metadata.s_col.to_vec(),
        y_ring: Vec::new(),
        ct: Vec::new(),
        aux_openings: Vec::new(),
        y_zcol: if metadata.has_y_zcol {
            vec![K::ZERO; D.next_power_of_two()]
        } else {
            Vec::new()
        },
        m_in: metadata.m_in,
        fold_digest: metadata.fold_digest,
    })
}

fn validate_claim_shell_inputs(
    inputs: &[CeClaim<Commitment, F, K>],
    rho_count: usize,
) -> Result<usize, CcsDeviceError> {
    let first = inputs
        .first()
        .ok_or(CcsDeviceError::Shape("Π_RLC: claim shell needs inputs"))?;
    if rho_count != inputs.len() {
        return Err(CcsDeviceError::Shape("Π_RLC: claim shell rho/input mismatch"));
    }
    let aux_len = first.aux_openings.len();
    if inputs
        .iter()
        .any(|claim| claim.aux_openings.len() != aux_len)
    {
        return Err(CcsDeviceError::Shape("Π_RLC: aux_openings length mismatch"));
    }
    Ok(aux_len)
}

fn mix_aux_openings(inputs: &[CeClaim<Commitment, F, K>], rho_first_coeffs: &[F], aux_len: usize) -> Vec<K> {
    let mut aux_openings = vec![K::ZERO; aux_len];
    for (&rho0, claim) in rho_first_coeffs.iter().zip(inputs.iter()) {
        let w = K::from(rho0);
        for (dst, src) in aux_openings.iter_mut().zip(claim.aux_openings.iter()) {
            *dst += w * *src;
        }
    }
    aux_openings
}

fn claim_shell_from_aux_openings(
    inputs: &[CeClaim<Commitment, F, K>],
    commitment: Commitment,
    aux_openings: Vec<K>,
) -> Result<CeClaim<Commitment, F, K>, CcsDeviceError> {
    let first = inputs
        .first()
        .ok_or(CcsDeviceError::Shape("Π_RLC: claim shell needs inputs"))?;
    Ok(CeClaim {
        adv: None,
        c_step_coords: vec![],
        u_offset: 0,
        u_len: 0,
        c: commitment,
        X: Mat::zero(D, first.m_in, F::ZERO),
        r: first.r.clone(),
        s_col: first.s_col.clone(),
        y_ring: Vec::new(),
        ct: Vec::new(),
        aux_openings,
        y_zcol: if first.y_zcol.is_empty() {
            Vec::new()
        } else {
            vec![K::ZERO; D.next_power_of_two()]
        },
        m_in: first.m_in,
        fold_digest: first.fold_digest,
    })
}

pub fn combine_y_ring(
    device: &Device,
    module: &RlcKernelModule,
    rho_coeffs: &DeviceBuffer<u64>,
    inputs: &[CeClaim<Commitment, F, K>],
    t_core: usize,
) -> Result<Vec<Vec<K>>, CcsDeviceError> {
    let d_pad = D.next_power_of_two();
    let words = combine_k_surfaces(device, module, rho_coeffs, inputs, t_core, d_pad, |claim, surface| {
        claim.y_ring.get(surface).map(Vec::as_slice)
    })?;
    Ok((0..t_core)
        .map(|surface| {
            (0..d_pad)
                .map(|lane| read_k_word(&words, surface * d_pad + lane))
                .collect()
        })
        .collect())
}

pub fn combine_y_zcol(
    device: &Device,
    module: &RlcKernelModule,
    rho_coeffs: &DeviceBuffer<u64>,
    inputs: &[CeClaim<Commitment, F, K>],
) -> Result<Vec<K>, CcsDeviceError> {
    let d_pad = D.next_power_of_two();
    let words = combine_k_surfaces(device, module, rho_coeffs, inputs, 1, d_pad, |claim, _| {
        Some(claim.y_zcol.as_slice())
    })?;
    Ok((0..d_pad).map(|lane| read_k_word(&words, lane)).collect())
}

pub struct CombinedKSurfaces {
    pub y_ring: Vec<Vec<K>>,
    pub y_zcol: Vec<K>,
}

pub struct PendingKSurfaces {
    stream: Arc<CudaStream>,
    _input_dev: Option<DeviceBuffer<u64>>,
    _device_inputs: Option<DevicePiCcsKSurfaces>,
    out_dev: DeviceBuffer<u64>,
    t_core: usize,
    include_y_zcol: bool,
    d_pad: usize,
}

impl PendingKSurfaces {
    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }
}

/// Combine every K-valued Π_RLC output surface in one device pass.
///
/// `y_ring` and `y_zcol` share the same rho coefficients and input-claim
/// layout. Keeping them in one packed surface batch avoids sending the same
/// rho/input boundary through two separate host-mediated CUDA calls.
pub fn combine_k_output_surfaces(
    device: &Device,
    module: &RlcKernelModule,
    rho_coeffs: &DeviceBuffer<u64>,
    inputs: &[CeClaim<Commitment, F, K>],
    t_core: usize,
    include_y_zcol: bool,
) -> Result<CombinedKSurfaces, CcsDeviceError> {
    let d_pad = D.next_power_of_two();
    let surface_count = t_core + usize::from(include_y_zcol);
    let words = combine_k_surfaces(
        device,
        module,
        rho_coeffs,
        inputs,
        surface_count,
        d_pad,
        |claim, surface| {
            if surface < t_core {
                claim.y_ring.get(surface).map(Vec::as_slice)
            } else {
                Some(claim.y_zcol.as_slice())
            }
        },
    )?;
    Ok(k_surfaces_from_words(&words, t_core, include_y_zcol, d_pad))
}

/// Combine K-valued Pi_CCS output surfaces already resident on device.
///
/// This is the Pi_CCS -> Pi_RLC resident handoff: unlike
/// [`combine_k_output_surfaces`], it does not rebuild a host `input_words`
/// buffer from `CeClaim`s and upload it again.
pub fn combine_k_output_surfaces_from_device(
    device: &Device,
    module: &RlcKernelModule,
    rho_coeffs: &DeviceBuffer<u64>,
    inputs: &DevicePiCcsKSurfaces,
) -> Result<CombinedKSurfaces, CcsDeviceError> {
    if inputs.claims() == 0 || rho_coeffs.len() != inputs.claims() * D {
        return Err(CcsDeviceError::Shape(
            "Π_RLC: device K-surface rho/input shape mismatch",
        ));
    }
    let stream = device.stream();
    let mut out_dev = uninit_u64_device_buffer(stream, inputs.surface_count() * inputs.d_pad() * 2)?;
    launch_rlc_combine_k_surfaces(
        module,
        stream,
        rho_coeffs,
        inputs.words(),
        inputs.claims(),
        inputs.surface_count(),
        inputs.d_pad(),
        &mut out_dev,
    )?;
    let words = out_dev.to_host_vec(stream)?;
    device.sync()?;
    Ok(k_surfaces_from_words(
        &words,
        inputs.t_core(),
        inputs.include_y_zcol(),
        inputs.d_pad(),
    ))
}

pub fn enqueue_k_output_surfaces_from_device(
    stream: Arc<CudaStream>,
    module: &RlcKernelModule,
    rho_coeffs: &DeviceBuffer<u64>,
    inputs: DevicePiCcsKSurfaces,
) -> Result<PendingKSurfaces, CcsDeviceError> {
    if inputs.claims() == 0 || rho_coeffs.len() != inputs.claims() * D {
        return Err(CcsDeviceError::Shape(
            "Π_RLC: device K-surface rho/input shape mismatch",
        ));
    }
    let t_core = inputs.t_core();
    let include_y_zcol = inputs.include_y_zcol();
    let d_pad = inputs.d_pad();
    let mut out_dev = uninit_u64_device_buffer(&stream, inputs.surface_count() * inputs.d_pad() * 2)?;
    launch_rlc_combine_k_surfaces(
        module,
        &stream,
        rho_coeffs,
        inputs.words(),
        inputs.claims(),
        inputs.surface_count(),
        inputs.d_pad(),
        &mut out_dev,
    )?;
    Ok(PendingKSurfaces {
        stream,
        _input_dev: None,
        _device_inputs: Some(inputs),
        out_dev,
        t_core,
        include_y_zcol,
        d_pad,
    })
}

/// Enqueue Π_RLC K-surface combination on a caller-provided stream.
///
/// This lets the adapter overlap `y_ring` / `y_zcol` surface combination
/// with independent work on the main stream while keeping every buffer the
/// forked stream touches alive until [`finish_k_output_surfaces`].
pub fn enqueue_k_output_surfaces(
    stream: Arc<CudaStream>,
    module: &RlcKernelModule,
    rho_coeffs: &DeviceBuffer<u64>,
    inputs: &[CeClaim<Commitment, F, K>],
    t_core: usize,
    include_y_zcol: bool,
) -> Result<PendingKSurfaces, CcsDeviceError> {
    let d_pad = D.next_power_of_two();
    let surface_count = t_core + usize::from(include_y_zcol);
    let (input_dev, out_dev) = enqueue_k_surfaces(
        &stream,
        module,
        rho_coeffs,
        inputs,
        surface_count,
        d_pad,
        |claim, surface| {
            if surface < t_core {
                claim.y_ring.get(surface).map(Vec::as_slice)
            } else {
                Some(claim.y_zcol.as_slice())
            }
        },
    )?;
    Ok(PendingKSurfaces {
        stream,
        _input_dev: Some(input_dev),
        _device_inputs: None,
        out_dev,
        t_core,
        include_y_zcol,
        d_pad,
    })
}

pub fn finish_k_output_surfaces(
    main_stream: &Arc<CudaStream>,
    pending: PendingKSurfaces,
) -> Result<CombinedKSurfaces, CcsDeviceError> {
    main_stream.join(pending.stream())?;
    let words = pending.out_dev.to_host_vec(main_stream)?;
    main_stream.synchronize()?;
    Ok(k_surfaces_from_words(
        &words,
        pending.t_core,
        pending.include_y_zcol,
        pending.d_pad,
    ))
}

pub(crate) fn finish_k_output_surfaces_device(
    main_stream: &Arc<CudaStream>,
    pending: PendingKSurfaces,
) -> Result<DevicePiCcsKSurfaces, CcsDeviceError> {
    let PendingKSurfaces {
        stream,
        _input_dev,
        _device_inputs,
        out_dev,
        t_core,
        include_y_zcol,
        d_pad,
    } = pending;
    main_stream.join(&stream)?;
    let mut keepalive = _input_dev.into_iter().collect::<Vec<_>>();
    if let Some(inputs) = _device_inputs {
        let (words, nested) = inputs.into_buffers();
        keepalive.push(words);
        keepalive.extend(nested);
    }
    DevicePiCcsKSurfaces::from_packed_words(
        Arc::clone(main_stream),
        out_dev,
        keepalive,
        1,
        t_core,
        include_y_zcol,
        d_pad,
    )
}

fn k_surfaces_from_words(words: &[u64], t_core: usize, include_y_zcol: bool, d_pad: usize) -> CombinedKSurfaces {
    let y_ring = (0..t_core)
        .map(|surface| {
            (0..d_pad)
                .map(|lane| read_k_word(&words, surface * d_pad + lane))
                .collect()
        })
        .collect();
    let y_zcol = if include_y_zcol {
        let base = t_core * d_pad;
        (0..d_pad)
            .map(|lane| read_k_word(&words, base + lane))
            .collect()
    } else {
        Vec::new()
    };
    CombinedKSurfaces { y_ring, y_zcol }
}

fn combine_k_surfaces<'a>(
    device: &Device,
    module: &RlcKernelModule,
    rho_coeffs: &DeviceBuffer<u64>,
    inputs: &'a [CeClaim<Commitment, F, K>],
    surface_count: usize,
    d_pad: usize,
    surface: impl Fn(&'a CeClaim<Commitment, F, K>, usize) -> Option<&'a [K]>,
) -> Result<Vec<u64>, CcsDeviceError> {
    let stream = device.stream();
    let (_input_dev, out_dev) = enqueue_k_surfaces(stream, module, rho_coeffs, inputs, surface_count, d_pad, surface)?;
    let out = out_dev.to_host_vec(stream)?;
    device.sync()?;
    Ok(out)
}

fn enqueue_k_surfaces<'a>(
    stream: &Arc<CudaStream>,
    module: &RlcKernelModule,
    rho_coeffs: &DeviceBuffer<u64>,
    inputs: &'a [CeClaim<Commitment, F, K>],
    surface_count: usize,
    d_pad: usize,
    surface: impl Fn(&'a CeClaim<Commitment, F, K>, usize) -> Option<&'a [K]>,
) -> Result<(DeviceBuffer<u64>, DeviceBuffer<u64>), CcsDeviceError> {
    if inputs.is_empty() || rho_coeffs.len() != inputs.len() * D || d_pad < D {
        return Err(CcsDeviceError::Shape("Π_RLC: K-surface rho/input shape mismatch"));
    }
    if surface_count == 0 {
        let input_dev = uninit_u64_device_buffer(stream, 1)?;
        let out_dev = uninit_u64_device_buffer(stream, 1)?;
        return Ok((input_dev, out_dev));
    }

    let mut input_words = vec![0u64; inputs.len() * surface_count * d_pad * 2];
    for (input_idx, claim) in inputs.iter().enumerate() {
        for surface_idx in 0..surface_count {
            let values = surface(claim, surface_idx).ok_or(CcsDeviceError::Shape("Π_RLC: missing K surface"))?;
            if values.len() != d_pad {
                return Err(CcsDeviceError::Shape("Π_RLC: K surface lane count mismatch"));
            }
            let base = (input_idx * surface_count * d_pad + surface_idx * d_pad) * 2;
            for (lane, value) in values.iter().enumerate() {
                let (c0, c1) = value.to_limbs_u64();
                input_words[base + 2 * lane] = c0;
                input_words[base + 2 * lane + 1] = c1;
            }
        }
    }

    let input_dev = upload_u64_device_buffer(stream, &input_words)?;
    let mut out_dev = uninit_u64_device_buffer(stream, surface_count * d_pad * 2)?;
    launch_rlc_combine_k_surfaces(
        module,
        stream,
        rho_coeffs,
        &input_dev,
        inputs.len(),
        surface_count,
        d_pad,
        &mut out_dev,
    )?;
    Ok((input_dev, out_dev))
}

fn read_k_word(words: &[u64], idx: usize) -> K {
    k_from_device_words(words[2 * idx], words[2 * idx + 1])
}

fn check_goldilocks_rho_profile(pp: &Params) -> Result<(), CcsDeviceError> {
    let ring = RotRing::goldilocks();
    if pp.inner().d as usize != D || ring.alphabet != [-2, -1, 0, 1, 2] {
        return Err(CcsDeviceError::Shape("Π_RLC: unsupported rho sampling profile"));
    }
    Ok(())
}

fn rho_mats_from_coeff_words(pp: &Params, coeff_words: &[u64], count: usize) -> Result<Vec<Mat<F>>, CcsDeviceError> {
    if coeff_words.len() != count * D {
        return Err(CcsDeviceError::Shape("Π_RLC: rho coefficient word count mismatch"));
    }
    let phi = neo_reductions::common::phi_coeffs_from_params(pp.inner())
        .map_err(|_| CcsDeviceError::Shape("Π_RLC: unsupported cyclotomic profile"))?;
    Ok((0..count)
        .map(|rho| {
            let coeffs: Vec<F> = coeff_words[rho * D..][..D]
                .iter()
                .map(|&word| f_from_device_word(word))
                .collect();
            rot_from_coeffs_host(&coeffs, phi)
        })
        .collect())
}

fn rot_from_coeffs_host(coeffs: &[F], phi_coeffs: &[i32]) -> Mat<F> {
    debug_assert_eq!(coeffs.len(), D);
    debug_assert_eq!(phi_coeffs.len(), D);
    let neg_c: Vec<F> = phi_coeffs
        .iter()
        .map(|&cr| f_from_i64(-(cr as i64)))
        .collect();
    let mut rho = Mat::zero(D, D, F::ZERO);
    let mut col = coeffs.to_vec();
    for j in 0..D {
        for r in 0..D {
            rho[(r, j)] = col[r];
        }
        let last = col[D - 1];
        let mut next = vec![F::ZERO; D];
        next[0] = last * neg_c[0];
        for r in 1..D {
            next[r] = col[r - 1] + last * neg_c[r];
        }
        col = next;
    }
    rho
}

fn f_from_i64(value: i64) -> F {
    if value >= 0 {
        F::from_u64(value as u64)
    } else {
        F::ZERO - F::from_u64((-value) as u64)
    }
}

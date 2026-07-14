//! Device-side Pi_CCS output digest construction.
//!
//! Owns the prover-side handoff from resident Pi_CCS K-surfaces to the
//! canonical `pi_ccs_outputs_digest` field digest. It does not define the
//! digest protocol; the field order mirrors `neo-fold-clean::paper::digest`.

use cuda_core::{DeviceBuffer, PinnedHostBuffer};
use neo_ajtai::Commitment;
use neo_ccs::{CeClaim, Mat};
use neo_math::{D, F, K};
use p3_field::{BasedVectorSpace, PrimeCharacteristicRing, PrimeField64};

use neo_fold_clean::paper::digest::digest_fields_as_digest32;

use crate::device::{copy_host_to_device, uninit_u64_device_buffer, Device};
use crate::kernels::ajtai::launch_plane_copy_slice;
use crate::kernels::pi_ccs_digest::{
    launch_ccs_build_accumulator_claim_digest_preimages, launch_ccs_build_output_claim_digest_preimages,
    launch_ccs_build_outputs_digest_preimage,
};
use crate::kernels::poseidon2::{launch_hash_contiguous_cooperative, launch_hash_fields_cooperative_plan, DIGEST_LEN};
use crate::reduce::ccs::{CcsDeviceError, DevicePiCcsKSurfaces, DevicePublicX, SumcheckKernels};

const OUTPUTS_DIGEST_DOMAIN: &[u8] = b"neo.fold.clean/pi_ccs_outputs_digest/v1";
const OUTPUT_CLAIM_DIGEST_DOMAIN: &[u8] = b"neo.fold.clean/pi_ccs_output_claim_digest/v1";
const ACCUMULATOR_CLAIM_DIGEST_DOMAIN: &[u8] = b"neo.fold.clean/accumulator_ce_claim_digest/v1";
const CE_CLAIM_DIGEST_DOMAIN: &[u8] = b"neo.fold.clean/ce_claim_digest/v2";
const BYTES_PER_PACKED_LIMB: usize = 7;

/// Four-field Poseidon2 digest produced entirely from device-resident output
/// surfaces plus the public CE claim shell.
pub struct DevicePiCcsOutputsDigest {
    words: DeviceBuffer<u64>,
    _claim_digests: PendingClaimDigests,
    _header_host: Vec<u64>,
    _header_fields: DeviceBuffer<u64>,
    _preimage: DeviceBuffer<u64>,
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
        Self::compute_from_plan(device, kernels, surfaces, plan, None, None)
    }

    pub fn compute_from_shells(
        device: &Device,
        kernels: &SumcheckKernels,
        shells: &[PiCcsOutputDigestShell<'_>],
        surfaces: &DevicePiCcsKSurfaces,
    ) -> Result<Self, CcsDeviceError> {
        let plan = OutputDigestPlan::from_shells(shells, surfaces)?;
        Self::compute_from_plan(device, kernels, surfaces, plan, None, None)
    }

    pub fn compute_from_shells_with_commitments(
        device: &Device,
        kernels: &SumcheckKernels,
        shells: &[PiCcsOutputDigestShell<'_>],
        surfaces: &DevicePiCcsKSurfaces,
        commitment_words: &DeviceBuffer<u64>,
        words_per_commitment: usize,
    ) -> Result<Self, CcsDeviceError> {
        let plan = OutputDigestPlan::from_shells(shells, surfaces)?;
        Self::compute_from_plan(
            device,
            kernels,
            surfaces,
            plan,
            Some((commitment_words, words_per_commitment)),
            None,
        )
    }

    pub(crate) fn compute_from_shells_with_authority(
        device: &Device,
        kernels: &SumcheckKernels,
        shells: &[PiCcsOutputDigestShell<'_>],
        surfaces: &DevicePiCcsKSurfaces,
        commitment_words: &DeviceBuffer<u64>,
        words_per_commitment: usize,
        public_x: &DevicePublicX,
    ) -> Result<Self, CcsDeviceError> {
        let plan = OutputDigestPlan::from_shells(shells, surfaces)?;
        Self::compute_from_plan(
            device,
            kernels,
            surfaces,
            plan,
            Some((commitment_words, words_per_commitment)),
            Some(public_x),
        )
    }

    fn compute_from_plan(
        device: &Device,
        kernels: &SumcheckKernels,
        surfaces: &DevicePiCcsKSurfaces,
        plan: OutputDigestPlan,
        commitments: Option<(&DeviceBuffer<u64>, usize)>,
        public_x: Option<&DevicePublicX>,
    ) -> Result<Self, CcsDeviceError> {
        let stream = device.stream();
        let claim_digests = compute_claim_digests(device, kernels, surfaces, &plan, commitments, public_x)?;

        let header_host = plan.outputs_header.clone();
        let header_fields = uninit_u64_device_buffer(stream, plan.outputs_header.len())?;
        copy_host_to_device(stream, &header_fields, &header_host)?;
        let mut outputs_preimage = uninit_u64_device_buffer(stream, plan.outputs_preimage_words.max(1))?;
        launch_ccs_build_outputs_digest_preimage(
            &kernels.digest,
            stream,
            &header_fields,
            claim_digests.words(),
            plan.claims,
            &mut outputs_preimage,
        )?;

        let mut words = uninit_u64_device_buffer(stream, DIGEST_LEN)?;
        launch_hash_contiguous_cooperative(
            &kernels.poseidon,
            stream,
            &outputs_preimage,
            plan.outputs_preimage_words,
            &mut words,
            &kernels.poseidon_rc,
        )?;
        Ok(Self {
            words,
            _claim_digests: claim_digests,
            _header_host: header_host,
            _header_fields: header_fields,
            _preimage: outputs_preimage,
        })
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

struct PendingClaimDigests {
    words: DeviceBuffer<u64>,
    _preimages: PendingClaimPreimages,
}

struct PendingClaimPreimages {
    words: DeviceBuffer<u64>,
    _plan_host: Vec<u64>,
    _plan_device: DeviceBuffer<u64>,
    offsets_start: usize,
    lengths_start: usize,
}

impl PendingClaimDigests {
    fn words(&self) -> &DeviceBuffer<u64> {
        &self.words
    }
}

fn compute_claim_digests(
    device: &Device,
    kernels: &SumcheckKernels,
    surfaces: &DevicePiCcsKSurfaces,
    plan: &OutputDigestPlan,
    commitments: Option<(&DeviceBuffer<u64>, usize)>,
    public_x: Option<&DevicePublicX>,
) -> Result<PendingClaimDigests, CcsDeviceError> {
    let preimages = prepare_claim_preimages(device, kernels, surfaces, plan, commitments, public_x)?;
    let stream = device.stream();
    let mut claim_digests = uninit_u64_device_buffer(stream, plan.claims.max(1) * DIGEST_LEN)?;
    launch_hash_fields_cooperative_plan(
        &kernels.poseidon,
        stream,
        plan.claims,
        &preimages.words,
        &preimages._plan_device,
        preimages.offsets_start,
        preimages.lengths_start,
        &mut claim_digests,
        &kernels.poseidon_rc,
    )?;
    Ok(PendingClaimDigests {
        words: claim_digests,
        _preimages: preimages,
    })
}

fn prepare_claim_preimages(
    device: &Device,
    kernels: &SumcheckKernels,
    surfaces: &DevicePiCcsKSurfaces,
    plan: &OutputDigestPlan,
    commitments: Option<(&DeviceBuffer<u64>, usize)>,
    public_x: Option<&DevicePublicX>,
) -> Result<PendingClaimPreimages, CcsDeviceError> {
    let stream = device.stream();
    let packed = PackedClaimDigestPlan::from_plan(plan);
    let plan_host = packed.words;
    let packed_dev = uninit_u64_device_buffer(stream, plan_host.len())?;
    copy_host_to_device(stream, &packed_dev, &plan_host)?;
    let (commitment_words, use_device_commitments, commitment_stride) = match commitments {
        Some((words, stride)) => {
            if stride == 0
                || words.len() != plan.claims * stride
                || plan
                    .commitment_lengths
                    .iter()
                    .any(|&len| len as usize != stride)
            {
                return Err(CcsDeviceError::Shape(
                    "Pi_CCS output digest device commitment shape mismatch",
                ));
            }
            (words, true, stride)
        }
        None => (surfaces.words(), false, 0),
    };
    let (public_x_words, use_device_x, public_x_stride) = match public_x {
        Some(x)
            if x.claims() == plan.claims
                && plan
                    .x_lengths
                    .iter()
                    .all(|&len| len as usize == x.words_per_claim()) =>
        {
            (x.words(), x.active_cols() != 0, x.words_per_claim())
        }
        Some(_) => return Err(CcsDeviceError::Shape("claim digest device X shape mismatch")),
        None => (surfaces.words(), false, 0),
    };
    let mut claim_preimages = uninit_u64_device_buffer(stream, plan.claim_preimage_words.max(1))?;
    launch_ccs_build_output_claim_digest_preimages(
        &kernels.digest,
        stream,
        surfaces.words(),
        commitment_words,
        use_device_commitments,
        commitment_stride,
        public_x_words,
        use_device_x,
        public_x_stride,
        plan.claims,
        surfaces.surface_count(),
        surfaces.t_core(),
        surfaces.d_pad(),
        plan.include_y_zcol,
        plan.write_ct_field,
        plan.write_y_zcol_field,
        &packed_dev,
        packed.prefix_fields_start,
        plan.prefix_fields.len(),
        packed.prefix_offsets_start,
        packed.prefix_lengths_start,
        packed.commitment_offsets_start,
        packed.commitment_lengths_start,
        packed.x_offsets_start,
        packed.x_lengths_start,
        packed.suffix_fields_start,
        plan.suffix_fields.len(),
        packed.suffix_offsets_start,
        packed.suffix_lengths_start,
        packed.preimage_offsets_start,
        packed.preimage_lengths_start,
        &mut claim_preimages,
    )?;
    Ok(PendingClaimPreimages {
        words: claim_preimages,
        _plan_host: plan_host,
        _plan_device: packed_dev,
        offsets_start: packed.preimage_offsets_start,
        lengths_start: packed.preimage_lengths_start,
    })
}

struct PackedClaimDigestPlan {
    words: Vec<u64>,
    prefix_fields_start: usize,
    prefix_offsets_start: usize,
    prefix_lengths_start: usize,
    commitment_offsets_start: usize,
    commitment_lengths_start: usize,
    x_offsets_start: usize,
    x_lengths_start: usize,
    suffix_fields_start: usize,
    suffix_offsets_start: usize,
    suffix_lengths_start: usize,
    preimage_offsets_start: usize,
    preimage_lengths_start: usize,
}

impl PackedClaimDigestPlan {
    fn from_plan(plan: &OutputDigestPlan) -> Self {
        let mut words = Vec::with_capacity(
            plan.prefix_fields.len()
                + plan.prefix_offsets.len()
                + plan.prefix_lengths.len()
                + plan.commitment_offsets.len()
                + plan.commitment_lengths.len()
                + plan.x_offsets.len()
                + plan.x_lengths.len()
                + plan.suffix_fields.len()
                + plan.suffix_offsets.len()
                + plan.suffix_lengths.len()
                + plan.preimage_offsets.len()
                + plan.preimage_lengths.len(),
        );
        let prefix_fields_start = append_plan_segment(&mut words, &plan.prefix_fields);
        let prefix_offsets_start = append_plan_segment(&mut words, &plan.prefix_offsets);
        let prefix_lengths_start = append_plan_segment(&mut words, &plan.prefix_lengths);
        let commitment_offsets_start = append_plan_segment(&mut words, &plan.commitment_offsets);
        let commitment_lengths_start = append_plan_segment(&mut words, &plan.commitment_lengths);
        let x_offsets_start = append_plan_segment(&mut words, &plan.x_offsets);
        let x_lengths_start = append_plan_segment(&mut words, &plan.x_lengths);
        let suffix_fields_start = append_plan_segment(&mut words, &plan.suffix_fields);
        let suffix_offsets_start = append_plan_segment(&mut words, &plan.suffix_offsets);
        let suffix_lengths_start = append_plan_segment(&mut words, &plan.suffix_lengths);
        let preimage_offsets_start = append_plan_segment(&mut words, &plan.preimage_offsets);
        let preimage_lengths_start = append_plan_segment(&mut words, &plan.preimage_lengths);
        Self {
            words,
            prefix_fields_start,
            prefix_offsets_start,
            prefix_lengths_start,
            commitment_offsets_start,
            commitment_lengths_start,
            x_offsets_start,
            x_lengths_start,
            suffix_fields_start,
            suffix_offsets_start,
            suffix_lengths_start,
            preimage_offsets_start,
            preimage_lengths_start,
        }
    }
}

fn append_plan_segment(dst: &mut Vec<u64>, segment: &[u64]) -> usize {
    let start = dst.len();
    dst.extend_from_slice(segment);
    start
}

struct OutputDigestPlan {
    claims: usize,
    prefix_fields: Vec<u64>,
    prefix_offsets: Vec<u64>,
    prefix_lengths: Vec<u64>,
    commitment_offsets: Vec<u64>,
    commitment_lengths: Vec<u64>,
    x_offsets: Vec<u64>,
    x_lengths: Vec<u64>,
    suffix_fields: Vec<u64>,
    suffix_offsets: Vec<u64>,
    suffix_lengths: Vec<u64>,
    preimage_offsets: Vec<u64>,
    preimage_lengths: Vec<u64>,
    claim_preimage_words: usize,
    outputs_header: Vec<u64>,
    outputs_preimage_words: usize,
    include_y_zcol: bool,
    write_ct_field: bool,
    write_y_zcol_field: bool,
}

#[derive(Clone, Copy)]
struct ClaimDigestEncoding {
    domain: &'static [u8],
    include_s_col: bool,
    write_ct_field: bool,
    include_y_zcol: bool,
    write_y_zcol_field: bool,
    include_aux_sidecars: bool,
}

impl ClaimDigestEncoding {
    fn output(include_y_zcol: bool) -> Self {
        Self {
            domain: OUTPUT_CLAIM_DIGEST_DOMAIN,
            include_s_col: true,
            write_ct_field: true,
            include_y_zcol,
            write_y_zcol_field: true,
            include_aux_sidecars: true,
        }
    }

    fn accumulator() -> Self {
        Self {
            domain: ACCUMULATOR_CLAIM_DIGEST_DOMAIN,
            include_s_col: true,
            write_ct_field: true,
            include_y_zcol: false,
            write_y_zcol_field: false,
            include_aux_sidecars: true,
        }
    }

    fn ce_claim() -> Self {
        Self {
            domain: CE_CLAIM_DIGEST_DOMAIN,
            include_s_col: false,
            write_ct_field: false,
            include_y_zcol: false,
            write_y_zcol_field: false,
            include_aux_sidecars: false,
        }
    }
}

impl OutputDigestPlan {
    fn combine_accumulator_plans(
        children: &Self,
        parent_accumulator: &Self,
        parent_ce: &Self,
    ) -> Result<Self, CcsDeviceError> {
        if children.include_y_zcol
            || parent_accumulator.include_y_zcol
            || parent_ce.include_y_zcol
            || children.write_y_zcol_field
            || parent_accumulator.write_y_zcol_field
            || parent_ce.write_y_zcol_field
            || !children.write_ct_field
            || !parent_accumulator.write_ct_field
            || parent_ce.write_ct_field
            || parent_accumulator.claims != 1
            || parent_ce.claims != 1
        {
            return Err(CcsDeviceError::Shape("accumulator claim digest plans are incompatible"));
        }
        let claims = children.claims + 2;
        let mut out = Self {
            claims,
            prefix_fields: Vec::new(),
            prefix_offsets: Vec::with_capacity(claims),
            prefix_lengths: Vec::with_capacity(claims),
            commitment_offsets: Vec::with_capacity(claims),
            commitment_lengths: Vec::with_capacity(claims),
            x_offsets: Vec::with_capacity(claims),
            x_lengths: Vec::with_capacity(claims),
            suffix_fields: Vec::new(),
            suffix_offsets: Vec::with_capacity(claims),
            suffix_lengths: Vec::with_capacity(claims),
            preimage_offsets: Vec::with_capacity(claims),
            preimage_lengths: Vec::with_capacity(claims),
            claim_preimage_words: 0,
            outputs_header: Vec::new(),
            outputs_preimage_words: 0,
            include_y_zcol: false,
            write_ct_field: true,
            write_y_zcol_field: false,
        };
        for plan in [children, parent_accumulator, parent_ce] {
            let prefix_base = out.prefix_fields.len() as u64;
            let suffix_base = out.suffix_fields.len() as u64;
            let preimage_base = out.claim_preimage_words as u64;
            out.prefix_fields.extend_from_slice(&plan.prefix_fields);
            out.prefix_offsets.extend(
                plan.prefix_offsets
                    .iter()
                    .map(|offset| prefix_base + offset),
            );
            out.prefix_lengths.extend_from_slice(&plan.prefix_lengths);
            out.commitment_offsets
                .extend_from_slice(&plan.commitment_offsets);
            out.commitment_lengths
                .extend_from_slice(&plan.commitment_lengths);
            out.x_offsets.extend_from_slice(&plan.x_offsets);
            out.x_lengths.extend_from_slice(&plan.x_lengths);
            out.suffix_fields.extend_from_slice(&plan.suffix_fields);
            out.suffix_offsets.extend(
                plan.suffix_offsets
                    .iter()
                    .map(|offset| suffix_base + offset),
            );
            out.suffix_lengths.extend_from_slice(&plan.suffix_lengths);
            out.preimage_offsets.extend(
                plan.preimage_offsets
                    .iter()
                    .map(|offset| preimage_base + offset),
            );
            out.preimage_lengths
                .extend_from_slice(&plan.preimage_lengths);
            out.claim_preimage_words += plan.claim_preimage_words;
        }
        Ok(out)
    }

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
        Self::from_shells(&shells, surfaces)
    }

    fn from_shells(
        shells: &[PiCcsOutputDigestShell<'_>],
        surfaces: &DevicePiCcsKSurfaces,
    ) -> Result<Self, CcsDeviceError> {
        Self::from_shells_with_encoding(shells, surfaces, ClaimDigestEncoding::output(surfaces.include_y_zcol()))
    }

    fn from_shells_with_encoding(
        shells: &[PiCcsOutputDigestShell<'_>],
        surfaces: &DevicePiCcsKSurfaces,
        encoding: ClaimDigestEncoding,
    ) -> Result<Self, CcsDeviceError> {
        if shells.len() != surfaces.claims() {
            return Err(CcsDeviceError::Shape("Pi_CCS output digest shell count mismatch"));
        }
        if encoding.include_y_zcol && !surfaces.include_y_zcol() {
            return Err(CcsDeviceError::Shape("claim digest requested missing y_zcol surface"));
        }
        let mut prefix_fields = Vec::new();
        let mut prefix_offsets = Vec::with_capacity(shells.len());
        let mut prefix_lengths = Vec::with_capacity(shells.len());
        let mut commitment_offsets = Vec::with_capacity(shells.len());
        let mut commitment_lengths = Vec::with_capacity(shells.len());
        let mut x_offsets = Vec::with_capacity(shells.len());
        let mut x_lengths = Vec::with_capacity(shells.len());
        let mut suffix_fields = Vec::new();
        let mut suffix_offsets = Vec::with_capacity(shells.len());
        let mut suffix_lengths = Vec::with_capacity(shells.len());
        let mut preimage_offsets = Vec::with_capacity(shells.len());
        let mut preimage_lengths = Vec::with_capacity(shells.len());
        let mut claim_preimage_words = 0usize;

        for shell in shells {
            check_shell_shape(shell)?;

            prefix_offsets.push(prefix_fields.len() as u64);
            let before = prefix_fields.len();
            let (commitment_offset, commitment_len, x_offset, x_len) =
                append_claim_prefix(&mut prefix_fields, shell, encoding);
            prefix_lengths.push((prefix_fields.len() - before) as u64);
            commitment_offsets.push(commitment_offset as u64);
            commitment_lengths.push(commitment_len as u64);
            x_offsets.push(x_offset as u64);
            x_lengths.push(x_len as u64);

            suffix_offsets.push(suffix_fields.len() as u64);
            let before = suffix_fields.len();
            append_claim_suffix(&mut suffix_fields, shell, encoding.include_aux_sidecars);
            suffix_lengths.push((suffix_fields.len() - before) as u64);

            preimage_offsets.push(claim_preimage_words as u64);
            let surface_words = surface_preimage_words(
                surfaces.t_core(),
                surfaces.d_pad(),
                encoding.write_ct_field,
                encoding.include_y_zcol,
                encoding.write_y_zcol_field,
            );
            let len = prefix_lengths.last().copied().expect("prefix length") as usize
                + surface_words
                + suffix_lengths.last().copied().expect("suffix length") as usize;
            preimage_lengths.push(len as u64);
            claim_preimage_words += len;
        }

        let mut outputs_header = pack_bytes_as_words(OUTPUTS_DIGEST_DOMAIN);
        outputs_header.push(shells.len() as u64);
        let outputs_preimage_words = outputs_header.len() + shells.len() * DIGEST_LEN;

        Ok(Self {
            claims: shells.len(),
            prefix_fields,
            prefix_offsets,
            prefix_lengths,
            commitment_offsets,
            commitment_lengths,
            x_offsets,
            x_lengths,
            suffix_fields,
            suffix_offsets,
            suffix_lengths,
            preimage_offsets,
            preimage_lengths,
            claim_preimage_words,
            outputs_header,
            outputs_preimage_words,
            include_y_zcol: encoding.include_y_zcol,
            write_ct_field: encoding.write_ct_field,
            write_y_zcol_field: encoding.write_y_zcol_field,
        })
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

fn surface_preimage_words(
    t_core: usize,
    d_pad: usize,
    write_ct_field: bool,
    include_y_zcol: bool,
    write_y_zcol_field: bool,
) -> usize {
    1 + t_core * (1 + d_pad * 2)
        + usize::from(write_ct_field) * (1 + t_core * 2)
        + usize::from(write_y_zcol_field) * (1 + usize::from(include_y_zcol) * d_pad * 2)
}

fn append_claim_prefix(
    out: &mut Vec<u64>,
    shell: &PiCcsOutputDigestShell<'_>,
    encoding: ClaimDigestEncoding,
) -> (usize, usize, usize, usize) {
    let claim_start = out.len();
    out.extend(pack_bytes_as_words(encoding.domain));
    out.push(shell.c.d as u64);
    out.push(shell.c.kappa as u64);
    out.push(shell.c.data.len() as u64);
    let commitment_offset = out.len() - claim_start;
    out.extend(shell.c.data.iter().map(field_word));

    let active_x_cols = shell.m_in.div_ceil(neo_math::D);
    out.push(shell.x.rows() as u64);
    out.push(shell.x.cols() as u64);
    out.push(active_x_cols as u64);
    let x_offset = out.len() - claim_start;
    for row in 0..shell.x.rows() {
        for col in 0..active_x_cols {
            out.push(field_word(&shell.x[(row, col)]));
        }
    }

    append_k_slice(out, shell.r);
    if encoding.include_s_col {
        append_k_slice(out, shell.s_col);
    }
    (
        commitment_offset,
        shell.c.data.len(),
        x_offset,
        shell.x.rows() * active_x_cols,
    )
}

fn append_claim_suffix(out: &mut Vec<u64>, shell: &PiCcsOutputDigestShell<'_>, include_aux_sidecars: bool) {
    if include_aux_sidecars {
        append_k_slice(out, shell.aux_openings);
    }
    out.push(shell.m_in as u64);
    append_digest32(out, shell.fold_digest);
    if include_aux_sidecars {
        out.push(shell.c_step_coords.len() as u64);
        out.extend(shell.c_step_coords.iter().map(field_word));
        out.push(shell.u_offset as u64);
        out.push(shell.u_len as u64);
    }
}

fn append_k_slice(out: &mut Vec<u64>, values: &[K]) {
    out.push(values.len() as u64);
    for value in values {
        append_k(out, value);
    }
}

fn append_k(out: &mut Vec<u64>, value: &K) {
    for limb in value.as_basis_coefficients_slice() {
        out.push(field_word(limb));
    }
}

fn append_digest32(out: &mut Vec<u64>, digest: [u8; 32]) {
    for chunk in digest.chunks_exact(8) {
        let value = u64::from_le_bytes(chunk.try_into().expect("digest limb"));
        out.push(F::from_u64(value).as_canonical_u64());
    }
}

fn pack_bytes_as_words(bytes: &[u8]) -> Vec<u64> {
    let mut out = Vec::with_capacity(1 + bytes.len().div_ceil(BYTES_PER_PACKED_LIMB));
    out.push(bytes.len() as u64);
    for chunk in bytes.chunks(BYTES_PER_PACKED_LIMB) {
        let mut limb = [0u8; 8];
        limb[..chunk.len()].copy_from_slice(chunk);
        out.push(u64::from_le_bytes(limb));
    }
    out
}

fn field_word(value: &F) -> u64 {
    value.as_canonical_u64()
}

pub(crate) fn accumulator_digest_from_surfaces(
    device: &Device,
    kernels: &SumcheckKernels,
    claims: &[CeClaim<Commitment, F, K>],
    surfaces: &DevicePiCcsKSurfaces,
    commitments: &crate::fold_output::DeviceCommitments,
    public_x: &DevicePublicX,
    parent: &CeClaim<Commitment, F, K>,
    parent_surfaces: &DevicePiCcsKSurfaces,
    parent_commitment: &crate::fold_output::DeviceCommitments,
    parent_x: &DevicePublicX,
) -> Result<([u8; 32], [F; DIGEST_LEN]), CcsDeviceError> {
    if commitments.count() != claims.len()
        || commitments.d() != D
        || public_x.claims() != claims.len()
        || claims.iter().any(|claim| {
            claim.c.d != commitments.d()
                || claim.c.kappa != commitments.kappa()
                || claim.c.data.len() != commitments.words_per_commitment()
                || claim.m_in != public_x.m_in()
        })
        || parent_surfaces.claims() != 1
        || parent_commitment.count() != 1
        || parent_commitment.d() != D
        || parent_commitment.words_per_commitment() != commitments.words_per_commitment()
        || parent_x.claims() != 1
        || parent_x.words_per_claim() != public_x.words_per_claim()
        || parent_surfaces.t_core() != surfaces.t_core()
        || parent_surfaces.surface_count() != surfaces.surface_count()
        || parent_surfaces.d_pad() != surfaces.d_pad()
        || parent.m_in != parent_x.m_in()
        || parent.c.kappa != parent_commitment.kappa()
        || parent.c.data.len() != parent_commitment.words_per_commitment()
    {
        return Err(CcsDeviceError::Shape(
            "accumulator digest device claim authority shape mismatch",
        ));
    }
    let shells = claims
        .iter()
        .map(PiCcsOutputDigestShell::from_claim)
        .collect::<Vec<_>>();
    let child_plan =
        OutputDigestPlan::from_shells_with_encoding(&shells, surfaces, ClaimDigestEncoding::accumulator())?;
    let parent_shell = [PiCcsOutputDigestShell::from_claim(parent)];
    let parent_acc_plan = OutputDigestPlan::from_shells_with_encoding(
        &parent_shell,
        parent_surfaces,
        ClaimDigestEncoding::accumulator(),
    )?;
    let parent_ce_plan =
        OutputDigestPlan::from_shells_with_encoding(&parent_shell, parent_surfaces, ClaimDigestEncoding::ce_claim())?;
    let combined_plan = OutputDigestPlan::combine_accumulator_plans(&child_plan, &parent_acc_plan, &parent_ce_plan)?;
    let packed = PackedClaimDigestPlan::from_plan(&combined_plan);
    let stream = device.stream();
    let plan_host = packed.words;
    let plan_device = uninit_u64_device_buffer(stream, plan_host.len())?;
    copy_host_to_device(stream, &plan_device, &plan_host)?;
    let mut combined_preimages = uninit_u64_device_buffer(stream, combined_plan.claim_preimage_words.max(1))?;
    launch_ccs_build_accumulator_claim_digest_preimages(
        &kernels.digest,
        stream,
        surfaces.words(),
        parent_surfaces.words(),
        commitments.words(),
        parent_commitment.words(),
        commitments.words_per_commitment(),
        public_x.words(),
        parent_x.words(),
        public_x.words_per_claim(),
        claims.len(),
        surfaces.surface_count(),
        surfaces.t_core(),
        surfaces.d_pad(),
        &plan_device,
        packed.prefix_fields_start,
        combined_plan.prefix_fields.len(),
        packed.prefix_offsets_start,
        packed.prefix_lengths_start,
        packed.commitment_offsets_start,
        packed.commitment_lengths_start,
        packed.x_offsets_start,
        packed.x_lengths_start,
        packed.suffix_fields_start,
        combined_plan.suffix_fields.len(),
        packed.suffix_offsets_start,
        packed.suffix_lengths_start,
        packed.preimage_offsets_start,
        packed.preimage_lengths_start,
        &mut combined_preimages,
    )?;
    let mut combined_digests = uninit_u64_device_buffer(stream, combined_plan.claims * DIGEST_LEN)?;
    launch_hash_fields_cooperative_plan(
        &kernels.poseidon,
        stream,
        combined_plan.claims,
        &combined_preimages,
        &plan_device,
        packed.preimage_offsets_start,
        packed.preimage_lengths_start,
        &mut combined_digests,
        &kernels.poseidon_rc,
    )?;
    let mut parent_acc_digest = uninit_u64_device_buffer(stream, DIGEST_LEN)?;
    let mut parent_ce_digest = uninit_u64_device_buffer(stream, DIGEST_LEN)?;
    launch_plane_copy_slice(
        kernels.ring(),
        stream,
        &combined_digests,
        claims.len() * DIGEST_LEN,
        0,
        DIGEST_LEN,
        &mut parent_acc_digest,
    )?;
    launch_plane_copy_slice(
        kernels.ring(),
        stream,
        &combined_digests,
        (claims.len() + 1) * DIGEST_LEN,
        0,
        DIGEST_LEN,
        &mut parent_ce_digest,
    )?;
    let mut summary = uninit_u64_device_buffer(stream, 2 * DIGEST_LEN)?;
    // Π_DEC validates the resident children against this parent. Recursive
    // state therefore carries the full parent authority digest directly;
    // constructing an outer hash over every child would duplicate that check.
    launch_plane_copy_slice(
        kernels.ring(),
        stream,
        &parent_acc_digest,
        0,
        0,
        DIGEST_LEN,
        &mut summary,
    )?;
    launch_plane_copy_slice(
        kernels.ring(),
        stream,
        &parent_ce_digest,
        0,
        DIGEST_LEN,
        DIGEST_LEN,
        &mut summary,
    )?;
    let summary_words = summary.to_host_vec(stream)?;
    device.sync()?;
    let fields = std::array::from_fn(|i| F::from_u64(summary_words[i]));
    let parent_ce_fields = std::array::from_fn(|i| F::from_u64(summary_words[DIGEST_LEN + i]));
    Ok((digest_fields_as_digest32(fields), parent_ce_fields))
}

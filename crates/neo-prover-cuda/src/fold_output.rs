//! Device-backed output of one recursive CUDA fold.
//!
//! Owns the Pi_DEC child claim surfaces across folds. The online prover sees
//! only a shape-correct claim shell plus the exact accumulator digest; normal
//! `CeClaim`s are reconstructed when proof/audit code crosses the egress
//! boundary.

use std::any::Any;
use std::sync::{Arc, Mutex};

use cuda_core::{CudaStream, DeviceBuffer};
use neo_ajtai::Commitment;
use neo_ccs::Mat;
use neo_fold_clean::paper::digest::{ce_claim_digest, digest32_as_fields, AccumulatorHandle};
use neo_fold_clean::paper::nifs::{DeferredNifsRunningMaterializer, Error, NifsRunningCarrier};
use neo_fold_clean::{CeClaim, RunningInstance};
use neo_math::{D, F};
use p3_field::{PrimeCharacteristicRing, PrimeField64};

use crate::device::{upload_u64_device_buffer, Device};
use crate::reduce::ccs::{accumulator_digest_from_surfaces, DevicePiCcsKSurfaces, DevicePublicX, SumcheckKernels};
use crate::reduce::dec::DeferredDecStatus;
use crate::reduce::rlc as device_rlc;
use crate::session::backend_unavailable;

pub(crate) enum MixedCommitment {
    Ready(Commitment),
    Pending(device_rlc::PendingMixedCommitment),
}

impl MixedCommitment {
    pub(crate) fn claim_shell_commitment(&self, kappa: usize) -> Commitment {
        match self {
            Self::Ready(commitment) => commitment.clone(),
            Self::Pending(_) => Commitment::zeros(D, kappa),
        }
    }

    pub(crate) fn is_pending(&self) -> bool {
        matches!(self, Self::Pending(_))
    }

    pub(crate) fn finish(self, stream: &Arc<CudaStream>) -> Result<Commitment, Error> {
        match self {
            Self::Ready(commitment) => Ok(commitment),
            Self::Pending(pending) => device_rlc::finish_mixed_commitment(stream, pending)
                .map_err(|_| backend_unavailable("finish pending device Pi_RLC commitment mix failed")),
        }
    }

    pub(crate) fn finish_device(self, device: &Device) -> Result<Arc<DeviceCommitments>, Error> {
        let commitments = match self {
            Self::Ready(commitment) => DeviceCommitments::upload_one(device, &commitment)?,
            Self::Pending(pending) => DeviceCommitments::from_pending_mix(device.stream(), pending)?,
        };
        Ok(Arc::new(commitments))
    }
}

pub(crate) struct DeviceFoldOutput {
    child_surfaces: DevicePiCcsKSurfaces,
    child_commitments: Arc<DeviceCommitments>,
    child_public_x: DevicePublicX,
    claim_shells: Vec<CeClaim>,
    parent_authority: DeviceClaimAuthority,
    deferred_dec_status: Arc<DeferredDecStatus>,
    accumulator_digest: [u8; 32],
    materialized_claims: Mutex<Option<Vec<CeClaim>>>,
}

impl DeviceFoldOutput {
    pub(crate) fn new(
        device: &Device,
        kernels: &SumcheckKernels,
        child_surfaces: DevicePiCcsKSurfaces,
        child_commitments: Arc<DeviceCommitments>,
        child_public_x: DevicePublicX,
        claim_shells: Vec<CeClaim>,
        parent_shell: CeClaim,
        parent_surfaces: DevicePiCcsKSurfaces,
        parent_commitment: Arc<DeviceCommitments>,
        parent_public_x: DevicePublicX,
        deferred_dec_status: Arc<DeferredDecStatus>,
    ) -> Result<Self, Error> {
        if child_surfaces.claims() != claim_shells.len() {
            return Err(backend_unavailable("device fold-output child count mismatch"));
        }
        if child_commitments.count() != claim_shells.len() {
            return Err(backend_unavailable("device fold-output commitment count mismatch"));
        }
        if child_public_x.claims() != claim_shells.len() {
            return Err(backend_unavailable("device fold-output public X count mismatch"));
        }
        let (accumulator_digest, parent_ce_digest) = accumulator_digest_from_surfaces(
            device,
            kernels,
            &claim_shells,
            &child_surfaces,
            &child_commitments,
            &child_public_x,
            &parent_shell,
            &parent_surfaces,
            &parent_commitment,
            &parent_public_x,
        )
        .map_err(|_| backend_unavailable("device fold-output accumulator digest failed"))?;
        let parent_authority = DeviceClaimAuthority::new(
            parent_shell,
            parent_surfaces,
            parent_commitment,
            parent_public_x,
            parent_ce_digest,
        )?;
        Ok(Self {
            child_surfaces,
            child_commitments,
            child_public_x,
            claim_shells,
            parent_authority,
            deferred_dec_status,
            accumulator_digest,
            materialized_claims: Mutex::new(None),
        })
    }

    pub(crate) fn child_surfaces(&self) -> &DevicePiCcsKSurfaces {
        &self.child_surfaces
    }

    pub(crate) fn child_commitments(&self) -> &Arc<DeviceCommitments> {
        &self.child_commitments
    }

    pub(crate) fn accumulator_digest(&self) -> [u8; 32] {
        self.accumulator_digest
    }

    pub(crate) fn accumulator_digest_fields(&self) -> [F; 4] {
        digest32_as_fields(self.accumulator_digest)
    }

    pub(crate) fn parent_authority(&self) -> &CeClaim {
        self.parent_authority.shell()
    }

    pub(crate) fn parent_ce_digest_fields(&self) -> [F; 4] {
        self.parent_authority.ce_digest()
    }

    pub(crate) fn child_count(&self) -> usize {
        self.claim_shells.len()
    }

    fn prover_input(&self) -> RunningInstance {
        RunningInstance {
            claims: self.claim_shells.clone(),
            witnesses: resident_witness_placeholders(self.claim_shells.len()),
            parent_authority: Some(self.parent_authority.shell().clone()),
        }
    }

    fn materialize(&self) -> Result<RunningInstance, Error> {
        let running = RunningInstance {
            claims: self.materialize_claims()?,
            witnesses: resident_witness_placeholders(self.claim_shells.len()),
            parent_authority: Some(self.materialize_parent_authority()?),
        };
        if AccumulatorHandle::from_running_parts(&running.claims, running.parent_authority.as_ref()).digest()
            != self.accumulator_digest
        {
            return Err(backend_unavailable("materialized device accumulator digest mismatch"));
        }
        Ok(running)
    }

    pub(crate) fn materialize_claims(&self) -> Result<Vec<CeClaim>, Error> {
        let mut cached = self
            .materialized_claims
            .lock()
            .map_err(|_| backend_unavailable("device fold-output materialization lock poisoned"))?;
        if let Some(claims) = cached.as_ref() {
            return Ok(claims.clone());
        }

        self.deferred_dec_status
            .check()
            .map_err(|_| backend_unavailable("deferred device Pi_DEC split status failed"))?;

        let commitments = self.child_commitments.materialize()?;
        let mut claims = self.claim_shells.clone();
        self.child_public_x
            .materialize_claims(&mut claims)
            .map_err(|_| backend_unavailable("device fold-output public X download failed"))?;
        self.child_surfaces
            .materialize_claims(&mut claims)
            .map_err(|_| backend_unavailable("device fold-output surface download failed"))?;
        for (claim, commitment) in claims.iter_mut().zip(commitments) {
            claim.c = commitment;
        }
        *cached = Some(claims.clone());
        Ok(claims)
    }

    pub(crate) fn materialize_parent_authority(&self) -> Result<CeClaim, Error> {
        self.parent_authority.materialize()
    }
}

struct DeviceClaimAuthority {
    shell: CeClaim,
    surfaces: DevicePiCcsKSurfaces,
    commitment: Arc<DeviceCommitments>,
    public_x: DevicePublicX,
    ce_digest: [F; 4],
    materialized: Mutex<Option<CeClaim>>,
}

impl DeviceClaimAuthority {
    fn new(
        shell: CeClaim,
        surfaces: DevicePiCcsKSurfaces,
        commitment: Arc<DeviceCommitments>,
        public_x: DevicePublicX,
        ce_digest: [F; 4],
    ) -> Result<Self, Error> {
        if surfaces.claims() != 1 || commitment.count() != 1 || public_x.claims() != 1 {
            return Err(backend_unavailable("device parent claim authority count mismatch"));
        }
        Ok(Self {
            shell,
            surfaces,
            commitment,
            public_x,
            ce_digest,
            materialized: Mutex::new(None),
        })
    }

    fn shell(&self) -> &CeClaim {
        &self.shell
    }

    fn ce_digest(&self) -> [F; 4] {
        self.ce_digest
    }

    fn materialize(&self) -> Result<CeClaim, Error> {
        let mut cached = self
            .materialized
            .lock()
            .map_err(|_| backend_unavailable("device parent claim materialization lock poisoned"))?;
        if let Some(claim) = cached.as_ref() {
            return Ok(claim.clone());
        }
        let mut claims = vec![self.shell.clone()];
        self.public_x
            .materialize_claims(&mut claims)
            .map_err(|_| backend_unavailable("device parent public X download failed"))?;
        self.surfaces
            .materialize_claims(&mut claims)
            .map_err(|_| backend_unavailable("device parent K-surface download failed"))?;
        claims[0].c = self
            .commitment
            .materialize()?
            .into_iter()
            .next()
            .ok_or_else(|| backend_unavailable("device parent commitment missing"))?;
        let claim = claims.pop().expect("one device parent claim");
        if ce_claim_digest(&claim) != self.ce_digest {
            return Err(backend_unavailable("materialized device parent digest mismatch"));
        }
        *cached = Some(claim.clone());
        Ok(claim)
    }
}

/// Canonical Ajtai commitments retained in their device output layout.
///
/// The object is shared by the running carrier, proof carrier, and session
/// plane cache. Pointer identity therefore binds a cached witness plane set to
/// the exact commitment words that were produced for it; zero-valued host
/// shells are never used as cache authority.
pub(crate) struct DeviceCommitments {
    stream: Arc<CudaStream>,
    words: DeviceBuffer<u64>,
    _keepalive: Vec<DeviceBuffer<u64>>,
    count: usize,
    d: usize,
    kappa: usize,
    materialized: Mutex<Option<Vec<Commitment>>>,
}

impl DeviceCommitments {
    pub(crate) fn new(
        stream: Arc<CudaStream>,
        words: DeviceBuffer<u64>,
        count: usize,
        d: usize,
        kappa: usize,
    ) -> Result<Self, Error> {
        Self::new_with_keepalive(stream, words, Vec::new(), count, d, kappa)
    }

    fn new_with_keepalive(
        stream: Arc<CudaStream>,
        words: DeviceBuffer<u64>,
        keepalive: Vec<DeviceBuffer<u64>>,
        count: usize,
        d: usize,
        kappa: usize,
    ) -> Result<Self, Error> {
        if d == 0 || kappa == 0 || words.len() != count * d * kappa {
            return Err(backend_unavailable("device commitment shape mismatch"));
        }
        Ok(Self {
            stream,
            words,
            _keepalive: keepalive,
            count,
            d,
            kappa,
            materialized: Mutex::new(None),
        })
    }

    pub(crate) fn from_pending_mix(
        stream: &Arc<CudaStream>,
        pending: crate::reduce::rlc::PendingMixedCommitment,
    ) -> Result<Self, Error> {
        let (words, keepalive, kappa) = crate::reduce::rlc::finish_mixed_commitment_device(stream, pending)
            .map_err(|_| backend_unavailable("retain pending device commitment mix failed"))?;
        Self::new_with_keepalive(Arc::clone(stream), words, keepalive, 1, D, kappa)
    }

    pub(crate) fn upload_one(device: &Device, commitment: &Commitment) -> Result<Self, Error> {
        let words = commitment
            .data
            .iter()
            .map(|value| value.as_canonical_u64())
            .collect::<Vec<_>>();
        let words = upload_u64_device_buffer(device.stream(), &words)
            .map_err(|_| backend_unavailable("upload mixed commitment failed"))?;
        Self::new(Arc::clone(device.stream()), words, 1, commitment.d, commitment.kappa)
    }

    pub(crate) fn words(&self) -> &DeviceBuffer<u64> {
        &self.words
    }

    pub(crate) fn count(&self) -> usize {
        self.count
    }

    pub(crate) fn d(&self) -> usize {
        self.d
    }

    pub(crate) fn kappa(&self) -> usize {
        self.kappa
    }

    pub(crate) fn words_per_commitment(&self) -> usize {
        self.d * self.kappa
    }

    pub(crate) fn materialize(&self) -> Result<Vec<Commitment>, Error> {
        let mut cached = self
            .materialized
            .lock()
            .map_err(|_| backend_unavailable("device commitment materialization lock poisoned"))?;
        if let Some(commitments) = cached.as_ref() {
            return Ok(commitments.clone());
        }
        let words = self
            .words
            .to_host_vec(&self.stream)
            .map_err(|_| backend_unavailable("device commitment download failed"))?;
        let stride = self.words_per_commitment();
        let commitments = (0..self.count)
            .map(|claim| {
                let mut commitment = Commitment::zeros(self.d, self.kappa);
                for (slot, &word) in commitment
                    .data
                    .iter_mut()
                    .zip(&words[claim * stride..(claim + 1) * stride])
                {
                    *slot = F::from_u64(word);
                }
                commitment
            })
            .collect::<Vec<_>>();
        *cached = Some(commitments.clone());
        Ok(commitments)
    }
}

pub(crate) struct CudaRunningCarrier {
    output: Arc<DeviceFoldOutput>,
}

impl CudaRunningCarrier {
    pub(crate) fn new(output: Arc<DeviceFoldOutput>) -> Self {
        Self { output }
    }

    pub(crate) fn output(&self) -> Arc<DeviceFoldOutput> {
        Arc::clone(&self.output)
    }
}

impl DeferredNifsRunningMaterializer for CudaRunningCarrier {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn materialize(&self) -> Result<RunningInstance, Error> {
        self.output.materialize()
    }

    fn materialize_prover_input(&self) -> Result<RunningInstance, Error> {
        Ok(self.output.prover_input())
    }
}

pub(crate) fn device_output_from_carrier(carrier: Option<&NifsRunningCarrier>) -> Option<Arc<DeviceFoldOutput>> {
    let NifsRunningCarrier::Deferred(materializer) = carrier? else {
        return None;
    };
    materializer
        .as_any()
        .downcast_ref::<CudaRunningCarrier>()
        .map(CudaRunningCarrier::output)
}

fn resident_witness_placeholders(count: usize) -> Vec<Mat<F>> {
    (0..count).map(|_| Mat::zero(0, 0, F::ZERO)).collect()
}

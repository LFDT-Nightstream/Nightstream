//! Prover-backend adapter boundary for NIFS.P.
//!
//! Owns: prover requests, deferred proof/running carriers, post-fold
//! summaries, and the adapter interface.
//!
//! Does not own: reduction semantics, backend kernels, or verifier acceptance.
//!
//! Emits constraints: no.
//!
//! Authority boundary: deferred carriers and summaries are prover-side storage
//! optimizations; the ordinary materialized proof and NIFS verifier remain the
//! authority boundary.
//!
//! | Obligation | Local owner | Emits constraints? | Authority source |
//! |---|---|---|---|
//! | Backend request | [`NifsProverRequest`] | no | Caller-validated protocol inputs |
//! | Deferred egress | [`NifsProofCarrier`], [`NifsRunningCarrier`] | no | Materialized ordinary proof/state |
//! | Backend adapter | [`NifsProverAdapter`] | no | Final NIFS verification |

use std::any::Any;
use std::fmt;
use std::sync::Arc;

use neo_ajtai::AjtaiSModule;
use neo_math::{D, F};
use neo_reductions::optimized_engine::OptimizedStructureCache;
use p3_field::PrimeCharacteristicRing;

use crate::engine::transcript::Transcript;
use crate::frontends::f_prime::compiler::FPrimeFoldPostSummary;
use crate::paper::construction2::RunningInstance;
use crate::paper::nifs::{Error, NifsProof};
use crate::paper::params::Params;
use crate::paper::relations::{CcsInstance, DecMixer, LaneScheme, RlcMixer, Structure};

pub struct NifsProverRequest<'a> {
    pub tr: &'a mut Transcript,
    pub pp: &'a Params,
    pub s: &'a Structure,
    pub cache: &'a OptimizedStructureCache,
    pub log: &'a AjtaiSModule,
    pub lanes: Option<&'a LaneScheme>,
    pub mix_rhos_commits: RlcMixer,
    pub combine_b_pows: DecMixer,
    pub fresh: Vec<CcsInstance>,
    pub running_carrier: Option<&'a NifsRunningCarrier>,
    pub running: &'a RunningInstance,
    pub cache_output_for_next_step: bool,
}

/// Deferred materialization of one verifier-visible NIFS proof.
///
/// This is a prover-side egress contract, not verifier authority. A CUDA
/// backend may keep proof surfaces resident during the online fold, but this
/// materializer must reconstruct the ordinary `NifsProof` bytes when parity,
/// audit, or decider code crosses back to the CPU proof boundary.
pub trait DeferredNifsProofMaterializer: Send + Sync + 'static {
    fn materialize(&self) -> Result<NifsProof, Error>;
}

/// Deferred materialization of the post-fold running accumulator.
///
/// This is the running-state companion to [`DeferredNifsProofMaterializer`].
/// A CUDA backend needs one ownership object for both verifier-visible child
/// claims and prover-private child witnesses; otherwise it can defer proof
/// bytes while still being forced to rebuild the fold output on the host.
pub trait DeferredNifsRunningMaterializer: Send + Sync + 'static {
    fn as_any(&self) -> &dyn Any;

    fn materialize(&self) -> Result<RunningInstance, Error>;

    /// Materialize the shape needed to feed the next prover call.
    ///
    /// The default is the full running accumulator. Accelerator backends may
    /// override this with a claim shell when prover-private witnesses remain
    /// resident in backend-owned memory.
    fn materialize_prover_input(&self) -> Result<RunningInstance, Error> {
        self.materialize()
    }
}

/// Proof payload produced by one NIFS.P prover step.
#[derive(Clone)]
pub enum NifsProofCarrier {
    /// Ordinary verifier-visible proof bytes are available now.
    Materialized(NifsProof),
    /// Ordinary proof bytes can be reconstructed from backend-owned state at
    /// proof egress. The verifier still sees a materialized `NifsProof`.
    Deferred(Arc<dyn DeferredNifsProofMaterializer>),
}

impl fmt::Debug for NifsProofCarrier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Storage timing is a prover implementation detail, not part of the
        // audit representation. Parity gates compare the materialized NIFS
        // proofs directly at their proof-consumption boundary.
        f.write_str("NifsProofCarrier(..)")
    }
}

impl NifsProofCarrier {
    pub fn materialized(proof: NifsProof) -> Self {
        Self::Materialized(proof)
    }

    pub fn deferred(materializer: Arc<dyn DeferredNifsProofMaterializer>) -> Self {
        Self::Deferred(materializer)
    }

    pub fn materialize(&self) -> Result<NifsProof, Error> {
        match self {
            Self::Materialized(proof) => Ok(proof.clone()),
            Self::Deferred(materializer) => materializer.materialize(),
        }
    }

    pub fn into_materialized(self) -> Result<NifsProof, Error> {
        match self {
            Self::Materialized(proof) => Ok(proof),
            Self::Deferred(materializer) => materializer.materialize(),
        }
    }
}

/// Post-fold running accumulator produced by one NIFS.P prover step.
#[derive(Clone)]
pub enum NifsRunningCarrier {
    /// Ordinary verifier/prover running state is available now.
    Materialized(RunningInstance),
    /// Ordinary running state can be reconstructed from backend-owned state at
    /// the next CPU proof boundary. Fast CUDA paths may keep child surfaces
    /// resident until that boundary.
    Deferred(Arc<dyn DeferredNifsRunningMaterializer>),
}

impl fmt::Debug for NifsRunningCarrier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Materialized(_) => f.write_str("NifsRunningCarrier::Materialized(..)"),
            Self::Deferred(_) => f.write_str("NifsRunningCarrier::Deferred(..)"),
        }
    }
}

impl NifsRunningCarrier {
    pub fn materialized(running: RunningInstance) -> Self {
        Self::Materialized(running)
    }

    pub fn deferred(materializer: Arc<dyn DeferredNifsRunningMaterializer>) -> Self {
        Self::Deferred(materializer)
    }

    pub fn as_materialized(&self) -> Option<&RunningInstance> {
        match self {
            Self::Materialized(running) => Some(running),
            Self::Deferred(_) => None,
        }
    }

    pub fn as_materialized_mut(&mut self) -> Option<&mut RunningInstance> {
        match self {
            Self::Materialized(running) => Some(running),
            Self::Deferred(_) => None,
        }
    }

    pub fn materialize(&self) -> Result<RunningInstance, Error> {
        match self {
            Self::Materialized(running) => Ok(running.clone()),
            Self::Deferred(materializer) => materializer.materialize(),
        }
    }

    pub fn materialize_prover_input(&self) -> Result<RunningInstance, Error> {
        match self {
            Self::Materialized(running) => Ok(running.clone()),
            Self::Deferred(materializer) => materializer.materialize_prover_input(),
        }
    }

    pub fn into_materialized(self) -> Result<RunningInstance, Error> {
        match self {
            Self::Materialized(running) => Ok(running),
            Self::Deferred(materializer) => materializer.materialize(),
        }
    }
}

impl From<RunningInstance> for NifsRunningCarrier {
    fn from(running: RunningInstance) -> Self {
        Self::materialized(running)
    }
}

/// Verifier-visible summary of the post-fold running accumulator.
///
/// This is metadata for lifecycle state threading, not proof authority. It
/// lets accelerator backends return the digest/shape facts they already own
/// with the fold output, instead of publishing them through side channels.
#[derive(Clone, Debug, Default)]
pub struct NifsPostFoldSummary {
    acc_digest_override: Option<[u8; 32]>,
    f_prime: Option<FPrimeFoldPostSummary>,
}

impl NifsPostFoldSummary {
    pub fn new(acc_digest_override: Option<[u8; 32]>, f_prime: Option<FPrimeFoldPostSummary>) -> Self {
        Self {
            acc_digest_override,
            f_prime,
        }
    }

    pub fn acc_digest_override(&self) -> Option<[u8; 32]> {
        self.acc_digest_override
    }

    pub fn f_prime(&self) -> Option<&FPrimeFoldPostSummary> {
        self.f_prime.as_ref()
    }
}

/// Output of one NIFS.P prover step.
///
/// Keeping the proof payload behind a named carrier gives accelerator backends
/// one protocol-owned seam for future deferred proof materialization.
pub struct NifsProverOutput {
    running: NifsRunningCarrier,
    proof: NifsProofCarrier,
    post_fold_summary: Option<NifsPostFoldSummary>,
}

impl NifsProverOutput {
    pub fn materialized(running: RunningInstance, proof: NifsProof) -> Self {
        Self {
            running: NifsRunningCarrier::materialized(running),
            proof: NifsProofCarrier::materialized(proof),
            post_fold_summary: None,
        }
    }

    pub fn deferred(running: NifsRunningCarrier, proof: NifsProofCarrier) -> Self {
        Self {
            running,
            proof,
            post_fold_summary: None,
        }
    }

    pub fn with_post_fold_summary(mut self, summary: NifsPostFoldSummary) -> Self {
        self.post_fold_summary = Some(summary);
        self
    }

    pub fn into_carriers(self) -> (NifsRunningCarrier, NifsProofCarrier) {
        (self.running, self.proof)
    }

    pub fn into_carriers_with_summary(self) -> (NifsRunningCarrier, NifsProofCarrier, Option<NifsPostFoldSummary>) {
        (self.running, self.proof, self.post_fold_summary)
    }

    pub fn into_parts(self) -> Result<(RunningInstance, NifsProofCarrier), Error> {
        Ok((self.running.into_materialized()?, self.proof))
    }

    pub fn into_parts_with_summary(
        self,
    ) -> Result<(RunningInstance, NifsProofCarrier, Option<NifsPostFoldSummary>), Error> {
        Ok((self.running.into_materialized()?, self.proof, self.post_fold_summary))
    }

    pub fn into_materialized_parts(self) -> Result<(RunningInstance, NifsProof), Error> {
        Ok((self.running.into_materialized()?, self.proof.into_materialized()?))
    }

    pub fn into_materialized_parts_with_summary(
        self,
    ) -> Result<(RunningInstance, NifsProof, Option<NifsPostFoldSummary>), Error> {
        Ok((
            self.running.into_materialized()?,
            self.proof.into_materialized()?,
            self.post_fold_summary,
        ))
    }
}

#[derive(Clone, Copy, Debug)]
pub struct NifsFPrimeStepContext {
    pub vk_fs_digest: [u8; 32],
    pub structure_digest: [F; 4],
    pub chunk_count: u64,
    pub step_count: u64,
    pub z_0: [u8; 32],
    pub z_i: [u8; 32],
    pub pc: u64,
    pub semantic_state_digest: [u8; 32],
    pub acc_digest: [u8; 32],
    pub public_trace: [u8; 32],
    pub chunk_digest: [F; 4],
}

pub struct NifsFreshInstancesRequest<'a> {
    pub pp: &'a Params,
    pub s: &'a Structure,
    pub cache: &'a OptimizedStructureCache,
    pub log: &'a AjtaiSModule,
    pub m_in: usize,
    pub assignments: &'a [&'a [F]],
    pub image_overlay: Option<NifsFreshImageOverlayRequest<'a>>,
    /// Optional Nebula lane map over the same assignments. Accelerators can
    /// derive its sidecars from their already-resident witness representation.
    pub lane_scheme: Option<&'a LaneScheme>,
}

/// Compact signed-unit assignment offered to accelerator fresh-instance
/// builders. Each ring column has one positive and one negative bit mask.
///
/// This is an execution representation of the ordinary low-norm field
/// assignment. Expanding it with [`Self::to_dense`] is exact; it carries no
/// independent proof authority.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NifsFreshSignedUnitAssignment {
    len: usize,
    positive: Vec<u64>,
    negative: Vec<u64>,
}

impl NifsFreshSignedUnitAssignment {
    pub(crate) fn from_masks(len: usize, positive: Vec<u64>, negative: Vec<u64>) -> Self {
        let columns = len.div_ceil(D);
        debug_assert_eq!(positive.len(), columns);
        debug_assert_eq!(negative.len(), columns);
        debug_assert!(positive
            .iter()
            .zip(&negative)
            .all(|(&positive, &negative)| positive & negative == 0));
        Self {
            len,
            positive,
            negative,
        }
    }

    /// Reference constructor used by backend conformance tests and generic
    /// adapters. Production selective lowering writes these masks directly.
    #[doc(hidden)]
    pub fn from_dense(values: &[F]) -> Option<Self> {
        let columns = values.len().div_ceil(D);
        let mut positive = vec![0u64; columns];
        let mut negative = vec![0u64; columns];
        for (index, &value) in values.iter().enumerate() {
            let digit = neo_math::balanced::to_balanced_i128(value);
            let mask = 1u64 << (index % D);
            match digit {
                1 => positive[index / D] |= mask,
                -1 => negative[index / D] |= mask,
                0 => {}
                _ => return None,
            }
        }
        Some(Self::from_masks(values.len(), positive, negative))
    }

    pub fn len(&self) -> usize {
        self.len
    }

    #[doc(hidden)]
    pub fn positive_masks(&self) -> &[u64] {
        &self.positive
    }

    #[doc(hidden)]
    pub fn negative_masks(&self) -> &[u64] {
        &self.negative
    }

    fn value(&self, index: usize) -> Option<F> {
        if index >= self.len {
            return None;
        }
        let mask = 1u64 << (index % D);
        if self.positive[index / D] & mask != 0 {
            Some(F::ONE)
        } else if self.negative[index / D] & mask != 0 {
            Some(-F::ONE)
        } else {
            Some(F::ZERO)
        }
    }

    #[doc(hidden)]
    pub fn to_dense(&self) -> Vec<F> {
        (0..self.len)
            .map(|index| {
                self.value(index)
                    .expect("index is inside packed assignment")
            })
            .collect()
    }

    #[doc(hidden)]
    pub fn public_input(&self, len: usize) -> Option<Vec<F>> {
        (len <= self.len).then(|| {
            (0..len)
                .map(|index| {
                    self.value(index)
                        .expect("index is inside packed assignment")
                })
                .collect()
        })
    }
}

pub struct NifsFreshSignedUnitInstancesRequest<'a> {
    pub pp: &'a Params,
    pub s: &'a Structure,
    pub cache: &'a OptimizedStructureCache,
    pub log: &'a AjtaiSModule,
    pub m_in: usize,
    pub assignments: &'a [NifsFreshSignedUnitAssignment],
    pub lane_scheme: Option<&'a LaneScheme>,
}

#[derive(Clone, Copy, Debug)]
pub struct NifsFreshImageOverlayRequest<'a> {
    pub app_private_offset: usize,
    pub app_private_var_widths: &'a [usize],
    pub source_assignments: &'a [&'a [F]],
    pub compact_lane_offsets: &'a [usize],
    pub regions: &'a [NifsFreshImageRegion],
    pub semantic_state_in: Option<NifsFreshSemanticStateInOverlay<'a>>,
    pub semantic_state_out: Option<NifsFreshSemanticStateOutOverlay<'a>>,
    pub state_x_out: Option<NifsFreshStateXOutOverlay>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NifsFreshImageRegion {
    pub kind: NifsFreshImageRegionKind,
    pub offset: usize,
    pub bits: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NifsFreshImageRegionKind {
    Boundary,
    StateIn,
    StateOut,
    ChunkDigest,
    AppPrivate,
    IsBase,
    NifsPayloads,
    Kmul,
    RingAction,
    Poseidon,
}

#[derive(Clone, Copy, Debug)]
pub struct NifsFreshSemanticStateInOverlay<'a> {
    pub trace_splice: usize,
    pub trace_words_per_assignment: usize,
    pub preimages: &'a [&'a [F]],
    pub assignment_var_indices: Option<&'a [usize]>,
}

#[derive(Clone, Copy, Debug)]
pub struct NifsFreshSemanticStateOutOverlay<'a> {
    pub trace_splice: usize,
    pub trace_words_per_assignment: usize,
    pub preimages: &'a [&'a [F]],
    pub assignment_var_indices: Option<&'a [usize]>,
    pub digest: [F; 4],
}

#[derive(Clone, Copy, Debug)]
pub struct NifsFreshStateXOutOverlay {
    pub image_values_per_assignment: usize,
    pub state_lane_base: usize,
    pub trace_splice: usize,
    pub trace_words_per_assignment: usize,
    pub public_x_out_lane_offsets: [usize; 4],
    pub include_semantic_state: bool,
    pub pc: u64,
}

pub trait NifsProverAdapter {
    fn begin_f_prime_step(&mut self, context: NifsFPrimeStepContext) {
        let _ = context;
    }

    fn prove(&mut self, request: NifsProverRequest<'_>) -> Result<NifsProverOutput, Error>;

    /// Let a backend build fresh CCS instances from low-norm assignments.
    ///
    /// Returning `Ok(None)` keeps the canonical CPU path. Accelerators can
    /// return ordinary `CcsInstance`s, but they must be field-identical to
    /// `CcsInstance::from_low_norm_assignment` for the same inputs.
    fn build_fresh_instances(
        &mut self,
        request: NifsFreshInstancesRequest<'_>,
    ) -> Result<Option<Vec<CcsInstance>>, Error> {
        let _ = request;
        Ok(None)
    }

    /// Build fresh instances directly from canonical signed-unit masks.
    /// Returning `Ok(None)` asks the frontend to materialize the ordinary
    /// dense assignment and use [`Self::build_fresh_instances`] instead.
    fn build_fresh_signed_unit_instances(
        &mut self,
        request: NifsFreshSignedUnitInstancesRequest<'_>,
    ) -> Result<Option<Vec<CcsInstance>>, Error> {
        let _ = request;
        Ok(None)
    }

    /// Whether a caller should immediately replay `NIFS.V` on the CPU
    /// before compiling the next recursive F' image.
    ///
    /// The CPU path keeps this guard. A backend that has just produced
    /// the fold and still emits the same proof into the final audit can
    /// return `false` to avoid a redundant prover-side replay; verifier
    /// semantics do not change.
    fn requires_recursive_compile_reverify(&self) -> bool {
        true
    }
}

/// Marker for complete NIFS adapters that always use optimized reductions.
///
/// Built-in CUDA and Metal provers implement this marker. Their public NIFS
/// selection therefore includes optimized PiCCS, PiRLC, and PiDEC; callers do
/// not select a second reduction mode.
pub trait OptimizedNifsProverAdapter: NifsProverAdapter {}

/// Canonical optimized host implementation of the complete NIFS prover.
#[derive(Clone, Copy, Debug, Default)]
pub struct OptimizedCpuNifsProver;

/// Independent direct implementation of the complete NIFS prover.
#[derive(Clone, Copy, Debug, Default)]
pub struct PaperExactNifsProver;

/// Complete optimized-host versus PaperExact NIFS crosscheck.
#[derive(Clone, Copy, Debug, Default)]
pub struct CrosscheckNifsProver;

/// One optimized accelerator backend checked against optimized CPU NIFS.
pub struct AcceleratorCrosscheckNifsProver<A> {
    accelerator: A,
}

impl<A> AcceleratorCrosscheckNifsProver<A> {
    pub fn new(accelerator: A) -> Self {
        Self { accelerator }
    }

    pub fn accelerator(&self) -> &A {
        &self.accelerator
    }

    pub fn accelerator_mut(&mut self) -> &mut A {
        &mut self.accelerator
    }

    pub fn into_accelerator(self) -> A {
        self.accelerator
    }
}

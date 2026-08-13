//! Prover-backend adapter boundary for NIFS.P.
//!
//! Owns the concrete prover request and result. The NIFS verifier remains the
//! authority boundary.

use neo_ajtai::AjtaiSModule;
use neo_math::{D, F};
use neo_reductions::optimized_engine::OptimizedStructureCache;
use p3_field::PrimeCharacteristicRing;

use crate::engine::transcript::Transcript;
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
    pub running: &'a RunningInstance,
}

pub struct NifsFreshInstancesRequest<'a> {
    pub pp: &'a Params,
    pub s: &'a Structure,
    pub cache: &'a OptimizedStructureCache,
    pub log: &'a AjtaiSModule,
    pub m_in: usize,
    pub assignments: &'a [&'a [F]],
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

pub trait NifsProverAdapter {
    fn prove(&mut self, request: NifsProverRequest<'_>) -> Result<(RunningInstance, NifsProof), Error>;

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
}

/// Marker for complete NIFS adapters that always use optimized reductions.
///
/// Implemented accelerator provers use this marker when their public NIFS
/// selection includes optimized PiCCS, PiRLC, and PiDEC. Callers then do not
/// select a second reduction mode.
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

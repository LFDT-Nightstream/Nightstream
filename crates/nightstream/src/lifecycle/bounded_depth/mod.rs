//! Lean `BoundedDepth.Terminal`, `Schedule.Admissible`; Cyclo's bounded-depth schedule.
mod compression;
mod encoding;
mod relation;
mod words;

use crate::engine::{Backend, CommitError, EngineError};
use crate::folding::bounded_depth::{prove, Kind};
use crate::folding::FoldContext;
use crate::folding::{pi_dec, pi_rlc, CcsInstance, RunningInstance};
pub(crate) use compression::FinalProof;
use neo_ccs::crypto::poseidon2::poseidon2_hash;
use neo_math::F;
use neo_transcript::domain_chunk_v1_2;
use nightstream_fprime::{
    encode_digest, BoundedDepthPackages, BoundedDepthState, ExpandedPackages, PackageError, PUBLIC_WORDS,
};
use p3_field::{PrimeCharacteristicRing, PrimeField64};
pub use relation::RelationError;
use relation::{CheckedFreshRows, FreshMatrixRows};
pub(crate) use relation::{CompiledStep, Holds, Relation};
use std::sync::Arc;

/// `Schedule.productionLightFoldCap`.
const LIGHT_FOLD_CAP: u64 = 64;

/// SPEC.md Algorithm 9; Lean `FinalVerifier.Mode`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Compression {
    Both,
    Light,
}

impl Compression {
    /// Lean `RunCompleteness.light_run_complete`.
    pub(crate) fn for_max_steps(max_steps: u128) -> Self {
        if max_steps <= u128::from(LIGHT_FOLD_CAP) + 1 {
            Self::Light
        } else {
            Self::Both
        }
    }

    /// Lean `FinalVerifier.Mode.slots`.
    fn slots(self) -> usize {
        match self {
            Self::Both => 2,
            Self::Light => 1,
        }
    }
}
const KEY_DOMAIN: &[u8] = b"Nightstream/OptionB/circuit-key/v1";

#[derive(Debug, thiserror::Error)]
pub enum BoundedDepthError {
    #[error(transparent)]
    Relation(#[from] RelationError),
    #[error(transparent)]
    Fold(#[from] crate::folding::Error),
    #[error(transparent)]
    Package(#[from] PackageError),
    #[error("bounded-depth statement: {0}")]
    Statement(&'static str),
    #[error("bounded-depth fresh instance: {0}")]
    Fresh(&'static str),
    #[error("bounded-depth slot {slot}: {reason}")]
    Slot { slot: usize, reason: String },
    #[error(transparent)]
    Engine(#[from] EngineError),
    #[error(transparent)]
    Compression(#[from] ns_compress::Error),
}

impl BoundedDepthError {
    pub(crate) fn engine_failure(&self) -> Option<&EngineError> {
        use crate::folding::{kernels, pi_ccs, pi_dec, pi_rlc, Error as Fold};
        match self {
            Self::Engine(error)
            | Self::Relation(RelationError::Commitment(CommitError::Engine(error)))
            | Self::Fold(Fold::Ccs(pi_ccs::Error::Engine(kernels::Error::Engine(error))))
            | Self::Fold(Fold::Rlc(pi_rlc::Error::Engine(kernels::Error::Engine(error))))
            | Self::Fold(Fold::Dec(pi_dec::Error::Engine(kernels::Error::Engine(error)))) => Some(error),
            _ => None,
        }
    }
}

impl From<Holds> for BoundedDepthError {
    fn from(holds: Holds) -> Self {
        match holds {
            Holds::Fails(reason) => Self::Fresh(reason),
            Holds::Engine(error) => Self::Engine(error),
        }
    }
}

#[derive(Clone)]
struct Slot {
    accumulator: pi_rlc::Output,
    depth: u64,
}

#[derive(Clone)]
pub(crate) struct Envelope {
    iteration: u64,
    z0: [F; 4],
    current: [F; 4],
    pc: u64,
    slots: [Slot; 2],
    fresh: Option<CcsInstance>,
    checked_rows: Option<Arc<CheckedFreshRows>>,
}

impl Envelope {
    pub(crate) fn state(&self) -> crate::State {
        crate::State::new(self.iteration, self.z0, self.current)
    }
}

#[derive(Clone)]
pub(crate) struct CompiledSteps {
    identity: [u64; 4],
    steps: [Arc<CompiledStep>; 2],
}

impl CompiledSteps {
    pub(crate) fn new(packages: BoundedDepthPackages) -> Result<Self, BoundedDepthError> {
        let ExpandedPackages {
            light,
            refresh,
            identity,
        } = packages.expand()?;
        let (light, refresh) = rayon::join(|| CompiledStep::new(light), || CompiledStep::new(refresh));
        Ok(Self {
            identity,
            steps: [Arc::new(light?), Arc::new(refresh?)],
        })
    }

    pub(crate) fn identity(&self) -> [u64; 4] {
        self.identity
    }

    #[cfg(test)]
    pub(crate) fn steps(&self) -> &[Arc<CompiledStep>; 2] {
        &self.steps
    }
}

pub(crate) struct BoundedDepth {
    backend: Backend,
    relations: [Relation; 2],
    keys: [[F; 4]; 2],
    cap: u64,
    minimum_security_bits: u32,
    compression: Compression,
}

pub(crate) struct Folded {
    kind: Kind,
    accumulator: pi_rlc::Output,
    proof: crate::folding::bounded_depth::Proof,
}

/// SPEC.md section 4.4, step 2; Lean `Lifecycle.BoundedDepth.CircuitKeyDigest.digest`.
fn circuit_key(identity: [u64; 4], circuit: u64) -> [F; 4] {
    let mut input = domain_chunk_v1_2(KEY_DOMAIN).to_vec();
    input.extend(identity.map(F::from_u64));
    input.push(F::from_u64(circuit));
    poseidon2_hash(&input)
}

impl BoundedDepth {
    pub(crate) fn new(
        steps: &CompiledSteps,
        minimum_security_bits: u32,
        backend: Backend,
    ) -> Result<Self, BoundedDepthError> {
        let relations = [
            Relation::new(&steps.steps[0], minimum_security_bits)?,
            Relation::new(&steps.steps[1], minimum_security_bits)?,
        ];
        let identity = steps.identity;
        let keys = [0u64, 1].map(|circuit| circuit_key(identity, circuit));
        Ok(Self {
            backend,
            relations,
            keys,
            cap: LIGHT_FOLD_CAP,
            minimum_security_bits,
            compression: Compression::Both,
        })
    }

    pub(crate) fn with_compression(self, compression: Compression) -> Self {
        Self { compression, ..self }
    }

    #[cfg(test)]
    pub(crate) fn relations(&self) -> &[Relation; 2] {
        &self.relations
    }

    /// HyperNova Definition 12; Lean `defaultSlot`.
    fn default_slot(&self, circuit: usize) -> Result<Slot, BoundedDepthError> {
        let relation = &self.relations[circuit];
        let mut zero = RunningInstance::canonical_zero(relation.params(), relation.structure(), PUBLIC_WORDS)
            .map_err(BoundedDepthError::Statement)?;
        Ok(Slot {
            accumulator: pi_rlc::Output {
                claim: zero.claims.swap_remove(0),
                witness: zero.witnesses.swap_remove(0),
            },
            depth: 0,
        })
    }

    pub(crate) fn initial(&self, z0: [F; 4]) -> Result<Envelope, BoundedDepthError> {
        Ok(Envelope {
            iteration: 0,
            z0,
            current: z0,
            pc: 0,
            slots: [self.default_slot(0)?, self.default_slot(1)?],
            fresh: None,
            checked_rows: None,
        })
    }

    fn state(&self, envelope: &Envelope) -> BoundedDepthState {
        BoundedDepthState {
            keys: self.keys,
            iteration: envelope.iteration,
            z0: envelope.z0,
            current: envelope.current,
            pc: envelope.pc,
            slots: envelope
                .slots
                .each_ref()
                .map(|slot| words::slot(&slot.accumulator.claim, slot.depth)),
        }
    }

    /// `Spec.GlobalParams.accumulatorBound`.
    fn accumulator_bound(&self, depth: u64) -> u64 {
        let params = self.relations[0].params();
        (1 + depth + u64::from(params.k_rho())) * u64::from(params.inner().T()) * u64::from(params.b() - 1)
    }

    /// `StepExtend.Honest`; Cyclo Theorem 3 item 1.
    fn slot_within_depth_bound(&self, slot: &Slot) -> Result<(), &'static str> {
        if slot.depth > LIGHT_FOLD_CAP {
            return Err("depth is above the light fold cap");
        }
        let bound = self.accumulator_bound(slot.depth);
        let within = |value: &F| {
            let value = value.as_canonical_u64();
            value.min(F::ORDER_U64 - value) <= bound
        };
        let witness = &slot.accumulator.witness;
        let fits = match witness.virtual_constant_value() {
            Some(value) => within(value),
            None => witness.is_packed_signed_unit() || witness.as_slice().iter().all(within),
        };
        if fits {
            Ok(())
        } else {
            Err("witness norm is above the bound of the slot's depth")
        }
    }

    /// Lean `FreshInstance.built_holds_iff`.
    pub(crate) fn extend(
        &self,
        envelope: &Envelope,
        witness: &[F],
        next: [F; 4],
        locals: Option<&[F]>,
    ) -> Result<Envelope, BoundedDepthError> {
        let folded = self.fold(envelope)?;
        self.step(envelope, folded, witness, next, locals)
    }

    pub(crate) fn fold(&self, envelope: &Envelope) -> Result<Option<Folded>, BoundedDepthError> {
        let Some(fresh) = &envelope.fresh else {
            return Ok(None);
        };
        let selected = (envelope.pc - 1) as usize;
        let slot = &envelope.slots[selected];
        let kind = if slot.depth < self.cap {
            Kind::Light
        } else {
            Kind::Refresh
        };
        let rows = FreshMatrixRows {
            relation: &self.relations[selected],
            prepared: envelope.checked_rows.as_deref(),
            witness: &fresh.witness.Z,
        };
        let (accumulator, proof) = prove(
            kind,
            &self.backend,
            FoldContext {
                rows: &rows,
                ..rows.relation.context()
            },
            fresh.clone(),
            slot.accumulator.clone(),
        )?;
        Ok(Some(Folded {
            kind,
            accumulator,
            proof,
        }))
    }

    pub(crate) fn step(
        &self,
        envelope: &Envelope,
        folded: Option<Folded>,
        witness: &[F],
        next: [F; 4],
        locals: Option<&[F]>,
    ) -> Result<Envelope, BoundedDepthError> {
        let prior = self.state(envelope);
        let mut slots = envelope.slots.clone();
        let (kind, step) = match (&envelope.fresh, folded) {
            (Some(fresh), Some(folded)) => {
                let selected = (envelope.pc - 1) as usize;
                let kind = folded.kind;
                let (step, _) = words::fold_step(
                    &prior,
                    next,
                    witness,
                    kind,
                    &folded.accumulator.claim,
                    fresh.claim.c.data(),
                    &folded.proof,
                );
                slots[selected] = Slot {
                    accumulator: folded.accumulator,
                    depth: words::next_depth(kind, envelope.slots[selected].depth),
                };
                (kind, step)
            }
            _ => (Kind::Light, words::base_step(&prior, next, witness)?.0),
        };
        let relation = &self.relations[words::pc(kind) as usize - 1];
        #[cfg(test)]
        let keeps_faulty = KEEP_FAULTY_PROOF.get();
        #[cfg(not(test))]
        let keeps_faulty = false;
        let reserved = (!keeps_faulty && matches!(self.backend, Backend::Optimized))
            .then(|| relation.reserve_checked_rows())
            .flatten();
        let keep_checked = reserved.is_some();
        let columns = relation.assignment_columns(&step, locals)?;
        #[cfg(test)]
        if keeps_faulty {
            FAILED_ROW.set(relation.first_failing_row(&step)?);
        }
        let (instance, held) = std::thread::scope(|scope| {
            let check = (!keeps_faulty).then(|| {
                scope.spawn(|| {
                    if keep_checked {
                        relation.checked_source_rows(&columns).map(Some)
                    } else {
                        relation.source_rows_hold(&columns).map(|()| None)
                    }
                })
            });
            let instance = relation.instance(&self.backend, &columns);
            let held = check.map_or(Ok(None), |check| check.join().unwrap_or(Err("row check panicked")));
            (instance, held)
        });
        let instance = instance?;
        let checked_rows = held
            .map_err(BoundedDepthError::Fresh)?
            .zip(reserved)
            .map(|(rows, mut slot)| {
                let checked = Arc::new(CheckedFreshRows::new(&instance.witness.Z, rows));
                *slot = Some(Arc::downgrade(&checked));
                checked
            });
        Ok(Envelope {
            iteration: envelope.iteration + 1,
            z0: envelope.z0,
            current: next,
            pc: words::pc(kind),
            slots,
            fresh: Some(instance),
            checked_rows,
        })
    }

    /// `RunCompleteness.Verifies`, `RunCompleteness.verifies_step`.
    pub(crate) fn verify(
        &self,
        iteration: u64,
        z0: [F; 4],
        current: [F; 4],
        envelope: &Envelope,
    ) -> Result<(), BoundedDepthError> {
        if (envelope.iteration, envelope.z0, envelope.current) != (iteration, z0, current) {
            return Err(BoundedDepthError::Statement(
                "the proof's statement differs from the expected one",
            ));
        }
        if iteration >= F::ORDER_U64 {
            return Err(BoundedDepthError::Statement(
                "iteration is not a canonical field counter",
            ));
        }
        let Some(fresh) = &envelope.fresh else {
            return if iteration == 0 && current == z0 {
                Ok(())
            } else {
                Err(BoundedDepthError::Statement("zero steps require equal endpoints"))
            };
        };
        if iteration == 0 || !matches!(envelope.pc, 1 | 2) {
            return Err(BoundedDepthError::Statement(
                "a recursive proof needs a step and a circuit",
            ));
        }
        for (index, slot) in envelope.slots.iter().enumerate() {
            let relation = &self.relations[index];
            relation
                .shape(&slot.accumulator.claim)
                .map_err(|reason| BoundedDepthError::Slot {
                    slot: index,
                    reason: reason.into(),
                })?;
            self.slot_within_depth_bound(slot)
                .map_err(|reason| BoundedDepthError::Slot {
                    slot: index,
                    reason: reason.into(),
                })?;
            pi_dec::prove_with_production_key(
                &self.backend,
                relation.context(),
                &slot.accumulator.claim,
                slot.accumulator.witness.clone(),
            )
            .map_err(|error| match error {
                pi_dec::Error::Engine(crate::folding::kernels::Error::Engine(error)) => {
                    BoundedDepthError::Engine(error)
                }
                error => BoundedDepthError::Slot {
                    slot: index,
                    reason: error.to_string(),
                },
            })?;
        }
        if fresh.claim.x != encode_digest(self.state(envelope).hash()) {
            return Err(BoundedDepthError::Fresh(
                "public input differs from the hash of the state",
            ));
        }
        Ok(self.relations[(envelope.pc - 1) as usize].holds(&self.backend, fresh)?)
    }
}

#[cfg(test)]
thread_local! {
    pub(crate) static KEEP_FAULTY_PROOF: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    pub(crate) static FAILED_ROW: std::cell::Cell<Option<usize>> = const { std::cell::Cell::new(None) };
}

#[cfg(test)]
#[path = "../../../tests/bounded_depth_lifecycle/mod.rs"]
mod tests;

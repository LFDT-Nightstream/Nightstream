//! The bounded-depth scheme's lifecycle (Lean `BoundedDepth.Terminal`, Cyclo's bounded-depth
//! schedule): a light and a refresh step circuit, each with its own relation,
//! and a state with two slots. Slot `s` accumulates the instances of circuit
//! `s`. Each step folds the previous step's instance into the slot of the
//! circuit that made it: a light fold while that slot's depth is below the
//! cap, a refresh fold at the cap (`Schedule.Admissible`). The first period of
//! steps is `(L^(cap+1) R)^(cap+1) R`, and each later period is
//! `(L^(cap+1) R)^cap R`.
mod encoding;
mod relation;
mod words;

use crate::engine::{Backend, CommitError, EngineError};
use crate::folding::bounded_depth::{prove, Kind};
use crate::folding::{pi_dec, pi_rlc, CcsInstance, RunningInstance};
use neo_ccs::crypto::poseidon2_goldilocks::poseidon2_hash;
use neo_math::F;
use neo_transcript::domain_chunk_v1_2;
use nightstream_fprime::{
    encode_digest, BoundedDepthPackages, BoundedDepthState, PackageError, StepPackage, PUBLIC_WORDS,
};
use p3_field::{PrimeCharacteristicRing, PrimeField64};
pub use relation::RelationError;
pub(crate) use relation::{Holds, Relation};

/// `Schedule.productionLightFoldCap`: the light circuit's depth guard admits
/// depths below 64.
const LIGHT_FOLD_CAP: u64 = 64;
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
}

impl BoundedDepthError {
    /// The engine failure inside this error, if the engine failed: a device
    /// failure judged nothing.
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

/// One slot: the accumulator of its circuit's instances and its depth.
#[derive(Clone)]
struct Slot {
    accumulator: pi_rlc::Output,
    depth: u64,
}

/// A proof of `iteration` steps from `z0` to `current`: the circuit that ran
/// last (`pc`, 0 before the first step), both slots, and the last step's
/// instance.
#[derive(Clone)]
pub(crate) struct Envelope {
    iteration: u64,
    z0: [F; 4],
    current: [F; 4],
    pc: u64,
    slots: [Slot; 2],
    fresh: Option<CcsInstance>,
}

impl Envelope {
    pub(crate) fn state(&self) -> crate::State {
        crate::State::new(self.iteration, self.z0, self.current)
    }
}

pub(crate) struct BoundedDepth {
    backend: Backend,
    packages: BoundedDepthPackages,
    relations: [Relation; 2],
    keys: [[F; 4]; 2],
    cap: u64,
}

/// `BoundedDepth::fold`'s result.
pub(crate) struct Folded {
    kind: Kind,
    accumulator: pi_rlc::Output,
    proof: crate::folding::bounded_depth::Proof,
}

impl BoundedDepth {
    pub(crate) fn new(
        packages: BoundedDepthPackages,
        minimum_security_bits: u32,
        backend: Backend,
    ) -> Result<Self, BoundedDepthError> {
        let relations = [
            Relation::new(packages.light(), minimum_security_bits)?,
            Relation::new(packages.refresh(), minimum_security_bits)?,
        ];
        let identity = packages.identity().map(F::from_u64);
        let keys = [0u64, 1].map(|circuit| {
            let mut input = domain_chunk_v1_2(KEY_DOMAIN).to_vec();
            input.extend(identity);
            input.push(F::from_u64(circuit));
            poseidon2_hash(&input)
        });
        Ok(Self {
            backend,
            packages,
            relations,
            keys,
            cap: LIGHT_FOLD_CAP,
        })
    }

    /// The proof of zero steps from `z0`, with both slots at the default
    /// running instance.
    pub(crate) fn initial(&self, z0: [F; 4]) -> Result<Envelope, BoundedDepthError> {
        let slot = |relation: &Relation| -> Result<Slot, BoundedDepthError> {
            let mut zero = RunningInstance::canonical_zero(relation.params(), relation.structure(), PUBLIC_WORDS)
                .map_err(BoundedDepthError::Statement)?;
            Ok(Slot {
                accumulator: pi_rlc::Output {
                    claim: zero.claims.swap_remove(0),
                    witness: zero.witnesses.swap_remove(0),
                },
                depth: 0,
            })
        };
        Ok(Envelope {
            iteration: 0,
            z0,
            current: z0,
            pc: 0,
            slots: [slot(&self.relations[0])?, slot(&self.relations[1])?],
            fresh: None,
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

    /// `Spec.GlobalParams.accumulatorBound`: the largest coefficient of an
    /// honest slot witness `depth` light folds after its refresh.
    fn accumulator_bound(&self, depth: u64) -> u64 {
        let params = self.relations[0].params();
        (1 + depth + u64::from(params.k_rho())) * u64::from(params.inner().T()) * u64::from(params.b() - 1)
    }

    /// The slot checks of `StepExtend.Honest` that the bytes do not fix: the
    /// depth is at most the light cap, and each coefficient of the witness is
    /// within the accumulator bound of that depth (Cyclo Theorem 3 item 1).
    /// The next light fold needs this bound.
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
        // A packed witness has coefficients in {-1, 0, 1}, and each bound is
        // at least 1.
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

    fn package(&self, kind: Kind) -> &StepPackage {
        match kind {
            Kind::Light => self.packages.light(),
            Kind::Refresh => self.packages.refresh(),
        }
    }

    /// The next proof: the step from `envelope.current` to `next` with the
    /// application witness `witness`. The first step is the light circuit's
    /// base step; every later step folds the previous step's instance. Every
    /// row of the circuit's relation, whose rows verify the fold, must hold on
    /// the new assignment before the proof returns. The instance that
    /// `Relation::instance` builds then holds under the relation (Lean
    /// `FreshInstance.built_holds_iff`).
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

    /// The fold of `envelope`'s fresh instance into its selected slot, the
    /// part of `extend` that needs neither the step's witness nor its output
    /// state. `None` at iteration 0, which has no fresh instance.
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
        let (accumulator, proof) = prove(
            kind,
            &self.backend,
            self.relations[selected].context(),
            fresh.clone(),
            slot.accumulator.clone(),
        )?;
        Ok(Some(Folded {
            kind,
            accumulator,
            proof,
        }))
    }

    /// `extend` after `fold`: `folded` is `fold`'s result on `envelope`.
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
        let logical = relation.assignment(self.package(kind), &step, locals)?;
        #[cfg(test)]
        if CHECK_ROWS.get() {
            FAILED_ROW.set(relation.first_failing_row(self.package(kind), &step)?);
        }
        #[cfg(test)]
        let keeps_faulty = CHECK_ROWS.get();
        #[cfg(not(test))]
        let keeps_faulty = false;
        // The row check runs while the engine commits the witness.
        let (instance, held) = std::thread::scope(|scope| {
            let check = (!keeps_faulty).then(|| scope.spawn(|| relation.rows_hold(&logical)));
            let instance = relation.instance(&self.backend, &logical);
            let held = check.map_or(Ok(()), |check| check.join().unwrap_or(Err("row check panicked")));
            (instance, held)
        });
        let instance = instance?;
        held.map_err(BoundedDepthError::Fresh)?;
        Ok(Envelope {
            iteration: envelope.iteration + 1,
            z0: envelope.z0,
            current: next,
            pc: words::pc(kind),
            slots,
            fresh: Some(instance),
        })
    }

    /// `RunCompleteness.Verifies`: at iteration 0 the current state is `z0`;
    /// otherwise the fresh instance's public input is the hash of the state
    /// with both slots, every slot's claim holds under its circuit's relation
    /// within the bound of the slot's depth, and the fresh instance holds
    /// under the relation of the circuit that ran last. So `extend` works on
    /// each proof that this accepts (`RunCompleteness.verifies_step`).
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
                "iteration is not a canonical Goldilocks counter",
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
            // The verifier splits the witness itself: PiDEC's recomposition
            // reproduces the claim's commitment, evaluations and public input.
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
    /// Set by a test that proves an invalid witness: record the first row of
    /// the step relation that the step fails, and return the faulty proof in
    /// place of the self-check's error, so the test can show that `verify`
    /// refuses it.
    pub(crate) static CHECK_ROWS: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// That row, or `None` when every row holds.
    pub(crate) static FAILED_ROW: std::cell::Cell<Option<usize>> = const { std::cell::Cell::new(None) };
}

#[cfg(test)]
#[path = "../../../tests/bounded_depth_lifecycle/mod.rs"]
mod tests;

//! Option B's lifecycle (Lean `OptionB.Terminal`, Cyclo's bounded-depth
//! schedule): a light and a refresh step circuit, each with its own relation,
//! and a state with two slots. Slot `s` accumulates the instances of circuit
//! `s`. Each step folds the previous step's instance into the slot of the
//! circuit that made it: a light fold while that slot's depth is below the
//! cap, a refresh fold at the cap (`Schedule.Admissible`). The steps run
//! `((L^(cap+1) R)^cap R)`.
mod compression;
mod encoding;
mod final_program;
mod relation;
mod words;

use crate::folding::option_b::{prove, Kind};
use crate::folding::{pi_dec, pi_rlc, CcsInstance, RunningInstance};
pub(crate) use compression::{FinalProof, Keys, Setups};
pub(crate) use final_program::Finish;
use neo_ccs::crypto::poseidon2_goldilocks::poseidon2_hash;
use neo_math::F;
use neo_transcript::domain_chunk_v1_2;
use nightstream_fprime::{encode_digest, OptionBPackages, OptionBState, PackageError, StepPackage, PUBLIC_WORDS};
use p3_field::{PrimeCharacteristicRing, PrimeField64};
pub(crate) use relation::Relation;
pub use relation::RelationError;

/// `Schedule.productionLightFoldCap`: the light circuit's depth guard admits
/// depths below 64.
const LIGHT_FOLD_CAP: u64 = 64;
const KEY_DOMAIN: &[u8] = b"Nightstream/OptionB/circuit-key/v1";

#[derive(Debug, thiserror::Error)]
pub enum OptionBError {
    #[error(transparent)]
    Relation(#[from] RelationError),
    #[error(transparent)]
    Fold(#[from] crate::folding::Error),
    #[error(transparent)]
    Package(#[from] PackageError),
    #[error("option-B statement: {0}")]
    Statement(&'static str),
    #[error("option-B fresh instance: {0}")]
    Fresh(&'static str),
    #[error("option-B slot {slot}: {reason}")]
    Slot { slot: usize, reason: String },
    #[error(transparent)]
    Layer1(#[from] neo_spartan::Error),
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

pub(crate) struct OptionB {
    packages: OptionBPackages,
    relations: [Relation; 2],
    keys: [[F; 4]; 2],
    cap: u64,
}

impl OptionB {
    pub(crate) fn new(packages: OptionBPackages, minimum_security_bits: u32) -> Result<Self, OptionBError> {
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
            packages,
            relations,
            keys,
            cap: LIGHT_FOLD_CAP,
        })
    }

    /// The proof of zero steps from `z0`, with both slots at the default
    /// running instance.
    pub(crate) fn initial(&self, z0: [F; 4]) -> Result<Envelope, OptionBError> {
        let slot = |relation: &Relation| -> Result<Slot, OptionBError> {
            let mut zero = RunningInstance::canonical_zero(relation.params(), relation.structure(), PUBLIC_WORDS)
                .map_err(OptionBError::Statement)?;
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

    fn state(&self, envelope: &Envelope) -> OptionBState {
        OptionBState {
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

    fn package(&self, kind: Kind) -> &StepPackage {
        match kind {
            Kind::Light => self.packages.light(),
            Kind::Refresh => self.packages.refresh(),
        }
    }

    /// The next proof: the step from `envelope.current` to `next` with the
    /// application witness `witness`. The first step is the light circuit's
    /// base step; every later step folds the previous step's instance. The new
    /// instance must hold under its circuit's relation (`Relation::holds`),
    /// whose rows verify the fold, before the proof returns.
    pub(crate) fn extend(&self, envelope: &Envelope, witness: &[F], next: [F; 4]) -> Result<Envelope, OptionBError> {
        let prior = self.state(envelope);
        let mut slots = envelope.slots.clone();
        let (kind, step) = match &envelope.fresh {
            None => (Kind::Light, words::base_step(&prior, next, witness)?.0),
            Some(fresh) => {
                let selected = (envelope.pc - 1) as usize;
                let slot = &envelope.slots[selected];
                let kind = if slot.depth < self.cap {
                    Kind::Light
                } else {
                    Kind::Refresh
                };
                let (folded, proof) = prove(
                    kind,
                    self.relations[selected].context(),
                    fresh.clone(),
                    slot.accumulator.clone(),
                )?;
                let (step, _) =
                    words::fold_step(&prior, next, witness, kind, &folded.claim, fresh.claim.c.data(), &proof);
                slots[selected] = Slot {
                    accumulator: folded,
                    depth: words::next_depth(kind, slot.depth),
                };
                (kind, step)
            }
        };
        let relation = &self.relations[words::pc(kind) as usize - 1];
        let logical = relation.assignment(self.package(kind), &step)?;
        #[cfg(test)]
        if CHECK_ROWS.get() {
            FAILED_ROW.set(relation.first_failing_row(self.package(kind), &step)?);
        }
        let instance = relation.instance(&logical)?;
        relation.holds(&instance).map_err(OptionBError::Fresh)?;
        Ok(Envelope {
            iteration: envelope.iteration + 1,
            z0: envelope.z0,
            current: next,
            pc: words::pc(kind),
            slots,
            fresh: Some(instance),
        })
    }

    /// `Terminal.HoldsFor`: at iteration 0 the current state is `z0`;
    /// otherwise the fresh instance's public input is the hash of the state
    /// with both slots, every slot's claim holds under its circuit's relation
    /// below `B`, and the fresh instance holds under the relation of the
    /// circuit that ran last.
    pub(crate) fn verify(
        &self,
        iteration: u64,
        z0: [F; 4],
        current: [F; 4],
        envelope: &Envelope,
    ) -> Result<(), OptionBError> {
        if (envelope.iteration, envelope.z0, envelope.current) != (iteration, z0, current) {
            return Err(OptionBError::Statement(
                "the proof's statement differs from the expected one",
            ));
        }
        if iteration >= F::ORDER_U64 {
            return Err(OptionBError::Statement(
                "iteration is not a canonical Goldilocks counter",
            ));
        }
        let Some(fresh) = &envelope.fresh else {
            return if iteration == 0 && current == z0 {
                Ok(())
            } else {
                Err(OptionBError::Statement("zero steps require equal endpoints"))
            };
        };
        if iteration == 0 || !matches!(envelope.pc, 1 | 2) {
            return Err(OptionBError::Statement("a recursive proof needs a step and a circuit"));
        }
        for (index, slot) in envelope.slots.iter().enumerate() {
            let relation = &self.relations[index];
            relation
                .shape(&slot.accumulator.claim)
                .map_err(|reason| OptionBError::Slot {
                    slot: index,
                    reason: reason.into(),
                })?;
            // The verifier splits the witness itself: PiDEC's digits reject a
            // norm at or above `B`, and its recomposition reproduces the
            // claim's commitment, evaluations and public input.
            pi_dec::prove_with_production_key(
                relation.context(),
                &slot.accumulator.claim,
                slot.accumulator.witness.clone(),
            )
            .map_err(|error| OptionBError::Slot {
                slot: index,
                reason: error.to_string(),
            })?;
        }
        if fresh.claim.x != encode_digest(self.state(envelope).hash()) {
            return Err(OptionBError::Fresh("public input differs from the hash of the state"));
        }
        self.relations[(envelope.pc - 1) as usize]
            .holds(fresh)
            .map_err(OptionBError::Fresh)
    }
}

#[cfg(test)]
thread_local! {
    /// Set by a test that proves an invalid witness: record the first row of
    /// the step relation that the step fails.
    pub(crate) static CHECK_ROWS: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// That row, or `None` when every row holds.
    pub(crate) static FAILED_ROW: std::cell::Cell<Option<usize>> = const { std::cell::Cell::new(None) };
}

#[cfg(test)]
#[path = "../../../tests/option_b_lifecycle/mod.rs"]
mod tests;

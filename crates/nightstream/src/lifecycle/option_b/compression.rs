//! Option B's compressed proof before the shrink layer (CW-6.2). The last
//! fold refreshes the last step's instance into its slot (a refresh is
//! admissible at any depth, `Schedule.Admissible`), so both slots then hold
//! claims below their bounds and nothing is fresh. One layer-1 argument
//! (`neo-spartan`) proves each slot's claim under its circuit's relation.
//!
//! Owns the compressed proof and its native verifier. The keys are derived
//! from the relations; nothing a proof carries is authority.

use std::path::Path;

use neo_math::F;
use neo_transcript::{domain_chunk_v1_2, Poseidon2Transcript};
use nightstream_fprime::{encode_digest, OptionBState};
use p3_field::{PrimeCharacteristicRing, PrimeField64};

use super::{words, Envelope, OptionB, OptionBError, LIGHT_FOLD_CAP};
use crate::folding::option_b::{self as fold, Kind};
use crate::folding::{CcsClaim, CeClaim};

const SEED_DOMAIN: &[u8] = b"Nightstream/OptionB/compression/v1";

/// The verifier's constants: one layer-1 key per circuit's relation, the
/// honest norm bound of a slot and the bits each layer must reach.
#[derive(Clone)]
pub(crate) struct Keys {
    pub(super) layer1: [neo_spartan::Key; 2],
    bound: u32,
    layer_bits: f64,
}

/// The prover's layer-1 setups and their keys.
pub(crate) struct Setups {
    layer1: [neo_spartan::Setup; 2],
    keys: Keys,
}

/// A compressed proof: the statement, the circuit that ran last, both slots'
/// claims and depths before the last fold, the last step's claim, the last
/// fold and its next claim, and one layer-1 proof per slot.
#[derive(Clone)]
pub(crate) struct Compressed {
    iteration: u64,
    z0: [F; 4],
    current: [F; 4],
    pc: u64,
    pub(super) slots: [(CeClaim, u64); 2],
    fresh: CcsClaim,
    pub(super) fold: fold::Proof,
    pub(super) next: CeClaim,
    pub(super) layer1: [neo_spartan::Proof; 2],
}

impl OptionB {
    /// Each slot's honest bound below `B`: a slot at the cap after the
    /// refresh's 17 sources (`Light.accumulatorBound cap`).
    fn compression_bound(&self) -> u32 {
        let params = self.relations[0].params();
        (1 + LIGHT_FOLD_CAP as u32 + params.k_rho()) * params.inner().T() * (params.b() - 1)
    }

    /// Layer 1 of each slot and the shrink layer each take a quarter of the
    /// error that the last fold leaves under the minimum.
    fn compression_layer_bits(&self) -> Result<f64, OptionBError> {
        self.relations
            .iter()
            .map(|relation| relation.params().compression_security_bits())
            .try_fold(f64::INFINITY, |bits, slot| slot.map(|slot| bits.min(slot + 2.0)))
            .ok_or(OptionBError::Statement(
                "the fold error leaves no room under the security minimum",
            ))
    }

    pub(crate) fn compression_keys(&self) -> Result<Keys, OptionBError> {
        Ok(Keys {
            layer1: [
                neo_spartan::Key::derive(&self.relations[0])?,
                neo_spartan::Key::derive(&self.relations[1])?,
            ],
            bound: self.compression_bound(),
            layer_bits: self.compression_layer_bits()?,
        })
    }

    /// Writes both layer-1 setups under `dir` (`light/`, `refresh/`).
    pub(crate) fn compression_setups(&self, dir: &Path) -> Result<Setups, OptionBError> {
        let build = |relation, name| -> Result<neo_spartan::Setup, OptionBError> {
            let dir = dir.join(name);
            std::fs::create_dir_all(&dir).map_err(neo_spartan::Error::from)?;
            Ok(neo_spartan::Setup::build(relation, &dir)?)
        };
        let layer1 = [
            build(&self.relations[0], "light")?,
            build(&self.relations[1], "refresh")?,
        ];
        let keys = Keys {
            layer1: [layer1[0].key().clone(), layer1[1].key().clone()],
            bound: self.compression_bound(),
            layer_bits: self.compression_layer_bits()?,
        };
        Ok(Setups { layer1, keys })
    }

    /// The transcript that layer 1 of `slot` continues: the domain, the hash
    /// of the state before the last fold and the slot.
    fn layer1_seed(state: [F; 4], slot: usize) -> Poseidon2Transcript {
        let mut transcript = Poseidon2Transcript::new_v1_2();
        transcript.absorb_v1_2(&domain_chunk_v1_2(SEED_DOMAIN));
        let mut words = state.to_vec();
        words.push(F::from_u64(slot as u64));
        transcript.absorb_v1_2(&words);
        transcript
    }

    fn layer1_relation<'a>(keys: &'a Keys, slot: usize) -> Result<neo_spartan::Relation<'a>, OptionBError> {
        Ok(neo_spartan::Relation::new(
            &keys.layer1[slot],
            nightstream_fprime::PUBLIC_WORDS / neo_math::D,
            nightstream_fprime::PI_CCS_V1_2_ROUND_COUNT,
            keys.bound,
            keys.layer_bits,
        )?)
    }

    /// Compresses `envelope`, which does not change.
    pub(crate) fn finish(&self, envelope: &Envelope, setups: &Setups) -> Result<Compressed, OptionBError> {
        let Some(fresh) = envelope.fresh.clone() else {
            return Err(OptionBError::Statement("an initial proof has no fold to finish"));
        };
        let selected = (envelope.pc - 1) as usize;
        let state = self.state(envelope).hash();
        let (next, proof) = fold::prove(
            Kind::Refresh,
            self.relations[selected].context(),
            fresh.clone(),
            envelope.slots[selected].accumulator.clone(),
        )?;
        let mut accumulators = [
            envelope.slots[0].accumulator.clone(),
            envelope.slots[1].accumulator.clone(),
        ];
        accumulators[selected] = next.clone();
        let layer1 = [0, 1].map(|slot| {
            let relation = Self::layer1_relation(&setups.keys, slot)?;
            Ok::<_, OptionBError>(neo_spartan::prove(
                &relation,
                &setups.layer1[slot],
                &neo_spartan::Accelerator::cpu(),
                Self::layer1_seed(state, slot),
                &accumulators[slot].claim,
                &accumulators[slot].witness,
            )?)
        });
        let [light, refresh] = layer1;
        Ok(Compressed {
            iteration: envelope.iteration,
            z0: envelope.z0,
            current: envelope.current,
            pc: envelope.pc,
            slots: envelope
                .slots
                .each_ref()
                .map(|slot| (slot.accumulator.claim.clone(), slot.depth)),
            fresh: fresh.claim,
            fold: proof,
            next: next.claim,
            layer1: [light?, refresh?],
        })
    }

    /// The compressed form of `Terminal.HoldsFor`: the fresh public input is
    /// the hash of the state with both slots, the refresh fold of the fresh
    /// claim into its slot gives the next claim, and layer 1 proves the next
    /// claim and the other slot's claim below the bound.
    pub(crate) fn verify_compressed(
        &self,
        keys: &Keys,
        iteration: u64,
        z0: [F; 4],
        current: [F; 4],
        proof: &Compressed,
    ) -> Result<(), OptionBError> {
        if (proof.iteration, proof.z0, proof.current) != (iteration, z0, current) {
            return Err(OptionBError::Statement(
                "the proof's statement differs from the expected one",
            ));
        }
        if iteration == 0 || iteration >= F::ORDER_U64 || !matches!(proof.pc, 1 | 2) {
            return Err(OptionBError::Statement("a compressed proof needs a step and a circuit"));
        }
        for (index, (claim, _)) in proof.slots.iter().enumerate() {
            self.relations[index]
                .shape(claim)
                .map_err(|reason| OptionBError::Slot {
                    slot: index,
                    reason: reason.into(),
                })?;
        }
        let state = OptionBState {
            keys: self.keys,
            iteration,
            z0,
            current,
            pc: proof.pc,
            slots: proof
                .slots
                .each_ref()
                .map(|(claim, depth)| words::slot(claim, *depth)),
        }
        .hash();
        if proof.fresh.x != encode_digest(state) {
            return Err(OptionBError::Fresh("public input differs from the hash of the state"));
        }
        let selected = (proof.pc - 1) as usize;
        let relation = &self.relations[selected];
        fold::verify(
            Kind::Refresh,
            relation.params(),
            relation.structure(),
            &proof.fresh,
            &proof.slots[selected].0,
            &proof.fold,
            &proof.next,
        )?;
        for slot in 0..2 {
            let claim = if slot == selected {
                &proof.next
            } else {
                &proof.slots[slot].0
            };
            neo_spartan::verify(
                &Self::layer1_relation(keys, slot)?,
                Self::layer1_seed(state, slot),
                claim,
                &proof.layer1[slot],
            )
            .map_err(|error| OptionBError::Slot {
                slot,
                reason: error.to_string(),
            })?;
        }
        Ok(())
    }
}

impl Setups {
    pub(crate) fn keys(&self) -> &Keys {
        &self.keys
    }
}

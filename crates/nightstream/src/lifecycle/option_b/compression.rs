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
use p3_field::PrimeCharacteristicRing;

use super::final_program::{FinalProgram, FinalWitness, Finish};
use super::{words, Envelope, OptionB, OptionBError, LIGHT_FOLD_CAP};
use crate::folding::option_b::{self as fold, Kind};
use crate::folding::{CcsClaim, CeClaim};
use neo_spartan::shrink::{Shape, Shrink};

pub(super) const SEED_DOMAIN: &[u8] = b"Nightstream/OptionB/compression/v1";

/// The verifier's constants: one layer-1 key per circuit's relation, the
/// honest norm bound of a slot and the bits each layer must reach.
#[derive(Clone)]
pub(crate) struct Keys {
    pub(super) layer1: [neo_spartan::Key; 2],
    pub(super) bound: u32,
    pub(super) layer_bits: f64,
}

/// The prover's layer-1 setups and their keys.
pub(crate) struct Setups {
    layer1: [neo_spartan::Setup; 2],
    keys: Keys,
}

/// What finishing a proof makes before the shrink layer: the last step's
/// claim, the last fold and its next claim, and one layer-1 proof per slot.
pub(crate) struct Compressed {
    fresh: CcsClaim,
    fold: fold::Proof,
    next: CeClaim,
    layer1: [neo_spartan::Proof; 2],
}

impl OptionB {
    /// An error of a compression prover on the engine's accelerator. A
    /// device failure judged nothing, so it is an engine failure.
    fn accelerator_error(&self, error: neo_spartan::Error) -> OptionBError {
        match error {
            neo_spartan::Error::Device(reason) => OptionBError::Engine(crate::EngineError::Failure {
                engine: self.backend.engine(),
                reason,
            }),
            error => error.into(),
        }
    }

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

    pub(super) fn layer1_relation<'a>(keys: &'a Keys, slot: usize) -> Result<neo_spartan::Relation<'a>, OptionBError> {
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
            &self.backend,
            self.relations[selected].context(),
            fresh.clone(),
            envelope.slots[selected].accumulator.clone(),
        )?;
        self.backend.release_commitment_key()?;
        let accelerator = self.backend.accelerator();
        let mut accumulators = [
            envelope.slots[0].accumulator.clone(),
            envelope.slots[1].accumulator.clone(),
        ];
        accumulators[selected] = next.clone();
        let layer1 = [0, 1].map(|slot| {
            let relation = Self::layer1_relation(&setups.keys, slot)?;
            neo_spartan::prove(
                &relation,
                &setups.layer1[slot],
                &accelerator,
                Self::layer1_seed(state, slot),
                &accumulators[slot].claim,
                &accumulators[slot].witness,
            )
            .map_err(|error| self.accelerator_error(error))
        });
        let [light, refresh] = layer1;
        Ok(Compressed {
            fresh: fresh.claim,
            fold: proof,
            next: next.claim,
            layer1: [light?, refresh?],
        })
    }
}

/// A finished proof: the statement and a shrink proof that the final program
/// accepts it. It carries no witness and cannot be extended.
#[derive(Clone)]
pub(crate) struct FinalProof {
    iteration: u64,
    z0: [F; 4],
    pub(super) current: [F; 4],
    pub(super) shrink: neo_spartan::shrink::ShrinkProof,
}

impl OptionB {
    pub(super) fn final_program<'a>(
        &self,
        finish: &'a Finish,
        relations: [&'a neo_spartan::Relation<'a>; 2],
        statement: (u64, [F; 4], [F; 4]),
        witness: Option<&'a FinalWitness>,
    ) -> FinalProgram<'a> {
        FinalProgram {
            keys: self.keys,
            finish,
            relations,
            statement,
            witness,
        }
    }

    /// What the final program reads for `envelope`: the last fold and both
    /// layer-1 proofs (`finish`), and the finish package's columns on the last
    /// fold's words.
    pub(crate) fn final_witness(
        &self,
        envelope: &Envelope,
        setups: &Setups,
        finish: &Finish,
    ) -> Result<FinalWitness, OptionBError> {
        let compressed = self.finish(envelope, setups)?;
        let prior = self.state(envelope);
        let witness_words = vec![F::ZERO; finish.package().witness_words()];
        let (words, _) = words::fold_step(
            &prior,
            prior.current,
            &witness_words,
            Kind::Refresh,
            &compressed.next,
            compressed.fresh.c.data(),
            &compressed.fold,
        );
        Ok(FinalWitness {
            columns: finish.columns(&words)?,
            layer1: compressed.layer1,
        })
    }

    /// The shrink proof that the final program accepts `witness` for the
    /// statement. The proof passes the verifier of `verify_final` before it
    /// returns, so a prover fault gives an error and not a bad proof.
    pub(crate) fn prove_final(
        &self,
        keys: &Keys,
        finish: &Finish,
        statement: (u64, [F; 4], [F; 4]),
        witness: &FinalWitness,
    ) -> Result<FinalProof, OptionBError> {
        let relations = [Self::layer1_relation(keys, 0)?, Self::layer1_relation(keys, 1)?];
        let shape = Shape::derive(&self.final_program(finish, [&relations[0], &relations[1]], statement, None))?;
        let layer = Shrink::new(shape, keys.layer_bits)?;
        let program = self.final_program(finish, [&relations[0], &relations[1]], statement, Some(witness));
        let shrink = neo_spartan::shrink::prove(&layer, &program, &self.backend.accelerator())
            .map_err(|error| self.accelerator_error(error))?;
        let checked = self.final_program(finish, [&relations[0], &relations[1]], statement, None);
        neo_spartan::shrink::verify(&layer, &checked, &shrink)?;
        let (iteration, z0, current) = statement;
        Ok(FinalProof {
            iteration,
            z0,
            current,
            shrink,
        })
    }

    /// Checks a finished proof against the expected statement and the keys,
    /// which the verifier derives (`compression_keys`). Keys with fewer layer
    /// bits than this verifier's minimum needs, or another norm bound, fail.
    pub(crate) fn verify_final(
        &self,
        keys: &Keys,
        finish: &Finish,
        iteration: u64,
        z0: [F; 4],
        current: [F; 4],
        proof: &FinalProof,
    ) -> Result<(), OptionBError> {
        if keys.layer_bits < self.compression_layer_bits()? || keys.bound != self.compression_bound() {
            return Err(OptionBError::Statement(
                "the key does not meet this verifier's security minimum",
            ));
        }
        if (proof.iteration, proof.z0, proof.current) != (iteration, z0, current) {
            return Err(OptionBError::Statement(
                "the proof's statement differs from the expected one",
            ));
        }
        let relations = [Self::layer1_relation(keys, 0)?, Self::layer1_relation(keys, 1)?];
        let program = self.final_program(finish, [&relations[0], &relations[1]], (iteration, z0, current), None);
        let layer = Shrink::new(Shape::derive(&program)?, keys.layer_bits)?;
        Ok(neo_spartan::shrink::verify(&layer, &program, &proof.shrink)?)
    }
}

impl Setups {
    pub(crate) fn keys(&self) -> &Keys {
        &self.keys
    }
}

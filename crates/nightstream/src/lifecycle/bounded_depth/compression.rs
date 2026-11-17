//! The compressed proof (SPEC.md section 6): the statement, the slot claims
//! of its mode, and an argument over the Akita commitment scheme that the
//! verifier of `BoundedDepth::verify` accepts some witnesses for them. It
//! carries no witness and cannot be extended.
//!
//! The terminal check has one CE claim for each slot of the mode
//! (`Compression`) and one for the fresh instance. Each slot's accumulator
//! is one, with the bound of its depth. In `Compression::Light` the proof
//! carries slot 0 alone, and the verifier builds slot 1 as the default slot
//! at depth 0 (Lean `FinalVerifier.Mode`). The fresh instance becomes a CE
//! claim by a Π_CCS run with no running claim: its sum-check proves each row
//! of the relation and that each coefficient is in {-1, 0, 1}, given the
//! evaluations of the claim. That run binds the Akita commitment of the
//! witnesses before its challenges, so the committed fresh witness is the
//! one that it judges. `ns-compress` proves the claims.
//!
//! The verifier does not expand the Ajtai key. `compression_setup` commits
//! the key rows once as one polynomial, for the blocks of the relations of
//! the claims up to a power of two (the light relation alone in
//! `Compression::Light`), and `verify_final` reads the commitment
//! from its verification key. The key is a function of the public seeds and
//! of that power of two only, so `ns-compress` pins the key of each size,
//! and a test computes each pinned key again from the seeds. The verifier
//! takes the pinned key of its circuit (`compression_key`), and no proof and
//! no prover value carries it.
//!
//! The soundness error is the sum of three parts. The caller's
//! `minimum_security_bits` gates the first two, each alone and never their
//! sum. The public API has no default for it, and no fold reaches 116 bits.
//!
//! 1. The Π_CCS run has the test error of one fold with one source.
//!    `Params::for_ccs_shape` gates the larger error of a fold with 15
//!    sources, about `2^-115.72` for the Poseidon2 hash chain.
//! 2. The argument has the error of its batches, its lookups, its two
//!    sum-checks and the point of `V`, `2^-119.02` for the Poseidon2 hash
//!    chain. `ns_compress::error_bits` bounds it by `2^-119` in integers,
//!    and `compression_layout` gates that bound.
//! 3. The Akita opening has the knowledge error of Akita's schedule row. No
//!    gate here computes or bounds it. The argument also needs the
//!    verification key to hold the commitment of the Ajtai key.

use std::time::{Duration, Instant};

use neo_ajtai::Commitment;
use neo_ccs::Mat;
use neo_math::{KExtensions, D, F, K};
use neo_reductions::optimized_engine::optimized_verify_with_profile;
use neo_reductions::{FoldProfile, NormSources};
use neo_transcript::{domain_chunk_v1_2, Poseidon2Transcript};
use nightstream_fprime::{
    encode_digest, BoundedDepthState, PI_CCS_V1_2_ROUND_COEFFICIENT_COUNT as ROUND_COEFFICIENTS,
    PI_CCS_V1_2_ROUND_COUNT as ROUNDS, PUBLIC_WORDS,
};
use ns_compress::{Layout, Norm, ProverKey, Statement, VerificationKey};
use p3_field::PrimeField64;

use super::super::encoding::{Reader, Shape, Writer};
use super::super::ProofCodecError;
use super::{words, BoundedDepth, BoundedDepthError, Compression, Envelope, LIGHT_FOLD_CAP};
use crate::folding::transcript::Transcript;
use crate::folding::{pi_ccs, CcsClaim, CeClaim, RunningInstance};

const PI_CCS_DOMAIN: &[u8] = b"Nightstream/OptionB/compress/ccs/v1";
const MATRICES: usize = nightstream_fprime::PI_CCS_V1_2_MATRIX_COUNT;

/// A proof of `iteration` steps from `z0` to `current` without witnesses.
#[derive(Clone)]
pub(crate) struct FinalProof {
    pub(super) iteration: u64,
    pub(super) z0: [F; 4],
    pub(super) current: [F; 4],
    /// `None` for the proof of zero steps.
    pub(super) steps: Option<Steps>,
}

/// What a compressed proof of one or more steps carries.
#[derive(Clone)]
pub(super) struct Steps {
    pub(super) pc: u64,
    /// The accumulator claim and depth of each slot of the mode: both
    /// slots, or slot 0 alone in `Compression::Light`.
    pub(super) slots: Vec<(CeClaim, u64)>,
    pub(super) fresh: Fresh,
    pub(super) pi_ccs: pi_ccs::SumcheckProof,
    pub(super) argument: ns_compress::Proof,
}

/// The CE claim that Π_CCS makes of the fresh instance, without its public
/// input, which is the hash of the state. Π_CCS checks the point and the
/// digest against its transcript.
#[derive(Clone)]
pub(super) struct Fresh {
    pub(super) c: Commitment,
    pub(super) r: Vec<K>,
    pub(super) eval_k: Vec<K>,
    pub(super) eval_a: Vec<Vec<K>>,
    pub(super) fold_digest: [u8; 32],
}

impl Fresh {
    fn claim(&self, x: &[F]) -> CeClaim {
        let columns = x.len() / D;
        let cells = (0..D)
            .flat_map(|lane| (0..columns).map(move |block| x[block * D + lane]))
            .collect();
        CeClaim {
            c: self.c.clone(),
            X: Mat::from_row_major(D, columns, cells),
            r: self.r.clone(),
            eval_k: self.eval_k.clone(),
            eval_a: self.eval_a.clone(),
            m_in: x.len(),
            fold_digest: self.fold_digest,
            adv: None,
        }
    }
}

impl FinalProof {
    pub(crate) fn state(&self) -> crate::State {
        crate::State::new(self.iteration, self.z0, self.current)
    }
}

/// Where `compress` spends its time.
#[derive(Clone, Copy, Debug)]
pub(crate) struct CompressTimes {
    pi_ccs: Duration,
    argument: ns_compress::ProverTimes,
}

/// Where `verify_final` spends its time.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct VerifyTimes {
    pi_ccs: Duration,
    argument: ns_compress::VerifierTimes,
}

fn milliseconds(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1e3
}

impl std::fmt::Display for CompressTimes {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let times = &self.argument;
        write!(
            f,
            "commit W {:.0} ms, PiCCS {:.0} ms, commit V {:.0} ms, weights {:.0} ms, sum-check {:.0} ms, \
             key sum-check {:.0} ms, opening {:.0} ms",
            milliseconds(times.commit_witness),
            milliseconds(self.pi_ccs),
            milliseconds(times.commit_inverse),
            milliseconds(times.weights),
            milliseconds(times.sumcheck),
            milliseconds(times.key),
            milliseconds(times.opening),
        )
    }
}

impl std::fmt::Display for VerifyTimes {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "PiCCS {:.1} ms, matrix pass {:.1} ms, sum-checks and other weights {:.1} ms, Akita setup {:.1} ms, \
             Akita opening {:.1} ms",
            milliseconds(self.pi_ccs),
            milliseconds(self.argument.matrices),
            milliseconds(self.argument.rest),
            milliseconds(self.argument.akita_setup),
            milliseconds(self.argument.akita),
        )
    }
}

impl BoundedDepth {
    /// The claims of the terminal check: the slots of the mode, with the
    /// depths `depths`, then the fresh instance of circuit `pc − 1` (Lean
    /// `Spec.Compression.Rust.productionClaims` and `lightClaims`).
    fn compression_claims(&self, pc: u64, depths: &[u64]) -> Vec<(usize, Norm)> {
        let mut claims: Vec<_> = depths
            .iter()
            .enumerate()
            .map(|(slot, depth)| {
                let bound = self.accumulator_bound(*depth);
                (self.relations[slot].blocks(), Norm::Table { bound })
            })
            .collect();
        claims.push((self.relations[(pc - 1) as usize].blocks(), Norm::Caller));
        claims
    }

    /// The prover's key of the compression: the commitment of the Ajtai key
    /// rows that the relations read, with the rows.
    pub(crate) fn compression_setup(&self) -> Result<ProverKey, BoundedDepthError> {
        Ok(ProverKey::new(self.key_blocks())?)
    }

    /// The verification key of this circuit: the pinned key of the size of
    /// `compression_setup`, which is the key of that setup without the rows.
    pub(crate) fn compression_key(&self) -> Result<VerificationKey, BoundedDepthError> {
        Ok(VerificationKey::new(self.key_blocks())?)
    }

    /// Decode the bytes of the verification key of this circuit. Only the
    /// bytes of `compression_key` decode.
    pub(crate) fn decode_compression_key(&self, bytes: &[u8]) -> Result<VerificationKey, BoundedDepthError> {
        let key = VerificationKey::from_bytes(bytes)?;
        self.check_key_size(&key)?;
        Ok(key)
    }

    /// The key of `compression_key` has the size of this circuit.
    fn check_key_size(&self, key: &VerificationKey) -> Result<(), BoundedDepthError> {
        if key.blocks() != ns_compress::key_blocks(self.key_blocks())? {
            return Err(ns_compress::Error::Key("the key is for another circuit size").into());
        }
        Ok(())
    }

    /// The blocks that the key of the compression must hold: those of the
    /// light relation alone in `Compression::Light`, whose claims are both
    /// of that relation.
    pub(super) fn key_blocks(&self) -> usize {
        match self.compression {
            Compression::Both => self.relations[0].blocks().max(self.relations[1].blocks()),
            Compression::Light => self.relations[0].blocks(),
        }
    }

    /// The slot words of the state of a proof whose argument has `slots`:
    /// in `Compression::Light`, slot 1 is the default slot at depth 0, which
    /// the verifier builds from its own relation and never reads from a
    /// proof (Lean `FinalVerifier.ModeChecks`).
    fn state_slots(&self, slots: &[(&CeClaim, u64)]) -> Result<[nightstream_fprime::Slot; 2], BoundedDepthError> {
        Ok(match self.compression {
            Compression::Both => [words::slot(slots[0].0, slots[0].1), words::slot(slots[1].0, slots[1].1)],
            Compression::Light => [
                words::slot(slots[0].0, slots[0].1),
                words::slot(&self.default_slot(1)?.accumulator.claim, 0),
            ],
        })
    }

    /// The prover's checks of `Compression::Light` on `envelope`: the light
    /// circuit ran last, and slot 1 is the default slot at depth 0, which
    /// the verifier builds. A proof that made a refresh fold fails here.
    fn check_light_envelope(&self, envelope: &Envelope) -> Result<(), BoundedDepthError> {
        if envelope.pc != 1 {
            return Err(BoundedDepthError::Statement(
                "a light compression needs the light circuit last",
            ));
        }
        let slot = &envelope.slots[1];
        if words::slot(&slot.accumulator.claim, slot.depth) != nightstream_fprime::Slot::default_slot() {
            return Err(BoundedDepthError::Slot {
                slot: 1,
                reason: "a light compression needs the default slot at depth 0".into(),
            });
        }
        Ok(())
    }

    /// The checks of the mode on a proof's slots and circuit: the proof
    /// carries the slots of the mode, and in `Compression::Light` the light
    /// circuit ran last (Lean `FinalVerifier.ModeChecks`).
    fn check_mode(&self, pc: u64, slots: usize) -> Result<(), BoundedDepthError> {
        if slots != self.compression.slots() {
            return Err(BoundedDepthError::Statement(
                "the proof carries the slots of another compression",
            ));
        }
        if self.compression == Compression::Light && pc != 1 {
            return Err(BoundedDepthError::Statement(
                "a light compression needs the light circuit last",
            ));
        }
        Ok(())
    }

    /// The layout of the claims under a key of `key_blocks` blocks, when
    /// `ns_compress::error_bits` of it is at least the security minimum.
    /// The error of the batches, the lookups, the sum-checks and the point
    /// of `V` is at most `2^-error_bits`, and the comparison is of integers
    /// (`Spec.Compression.Rust.params_run_soundness_bits`). It has no term
    /// for the Π_CCS run or for the Akita opening.
    pub(crate) fn compression_layout(
        &self,
        key_blocks: usize,
        pc: u64,
        depths: &[u64],
    ) -> Result<Layout, BoundedDepthError> {
        let layout = Layout::new(&self.compression_claims(pc, depths), key_blocks)?;
        if ns_compress::error_bits(&layout, PUBLIC_WORDS, MATRICES) < self.minimum_security_bits {
            return Err(BoundedDepthError::Statement(
                "the batch, lookup and sum-check error of the compression argument is above the security minimum",
            ));
        }
        Ok(layout)
    }

    /// The Π_CCS profile of the fresh instance: it binds `commitment`, the
    /// Akita commitment of the witnesses, after the statement.
    pub(super) fn compression_profile(commitment: &[F]) -> FoldProfile<'_> {
        FoldProfile {
            domain: domain_chunk_v1_2(PI_CCS_DOMAIN),
            children: commitment,
            norm: NormSources::Fresh,
        }
    }

    pub(super) fn compression_statements<'a>(
        &'a self,
        pc: u64,
        slots: &[(&'a CeClaim, u64)],
        fresh: &'a CeClaim,
    ) -> Vec<Statement<'a>> {
        let mut statements: Vec<_> = slots
            .iter()
            .enumerate()
            .map(|(slot, (claim, depth))| Statement {
                claim,
                rows: &self.relations[slot],
                norm: Norm::Table {
                    bound: self.accumulator_bound(*depth),
                },
            })
            .collect();
        statements.push(Statement {
            claim: fresh,
            rows: &self.relations[(pc - 1) as usize],
            norm: Norm::Caller,
        });
        statements
    }

    /// Compress `envelope`, which does not change. The proof is not checked
    /// here: an envelope that `verify` rejects gives an error or a proof that
    /// `verify_final` rejects.
    pub(crate) fn compress(
        &self,
        setup: &ProverKey,
        envelope: &Envelope,
    ) -> Result<(FinalProof, CompressTimes), BoundedDepthError> {
        let Envelope {
            iteration,
            z0,
            current,
            pc,
            ..
        } = *envelope;
        let Some(fresh) = &envelope.fresh else {
            let proof = FinalProof {
                iteration,
                z0,
                current,
                steps: None,
            };
            let times = CompressTimes {
                pi_ccs: Duration::ZERO,
                argument: ns_compress::ProverTimes::default(),
            };
            return Ok((proof, times));
        };
        if !matches!(pc, 1 | 2) {
            return Err(BoundedDepthError::Statement(
                "a recursive proof needs a step and a circuit",
            ));
        }
        if self.compression == Compression::Light {
            self.check_light_envelope(envelope)?;
        }
        let slots = &envelope.slots[..self.compression.slots()];
        let depths: Vec<u64> = slots.iter().map(|slot| slot.depth).collect();
        let layout = self.compression_layout(setup.verification_key().blocks(), pc, &depths)?;
        let mut witnesses: Vec<_> = slots.iter().map(|slot| &slot.accumulator.witness).collect();
        witnesses.push(&fresh.witness.Z);
        let committed = ns_compress::commit(setup, layout, &witnesses)?;

        let start = Instant::now();
        let commitment = ns_compress::byte_words(committed.commitment());
        let mut transcript = Transcript::session();
        let mut ccs = pi_ccs::prove_from_parts_with_rows(
            &mut transcript,
            &self.backend,
            self.relations[(pc - 1) as usize].context(),
            &Self::compression_profile(&commitment),
            &fresh.claim,
            &fresh.witness,
            &RunningInstance::default(),
        )
        .map_err(crate::folding::Error::from)?;
        let pi_ccs = start.elapsed();

        let fresh = ccs.outputs.swap_remove(0);
        let claims: Vec<_> = slots
            .iter()
            .map(|slot| (&slot.accumulator.claim, slot.depth))
            .collect();
        let statements = self.compression_statements(pc, &claims, &fresh);
        let (argument, argument_times) = committed.prove(transcript.inner_mut(), &statements)?;
        let proof = FinalProof {
            iteration,
            z0,
            current,
            steps: Some(Steps {
                pc,
                slots: slots
                    .iter()
                    .map(|slot| (slot.accumulator.claim.clone(), slot.depth))
                    .collect(),
                fresh: Fresh {
                    c: fresh.c,
                    r: fresh.r,
                    eval_k: fresh.eval_k,
                    eval_a: fresh.eval_a,
                    fold_digest: fresh.fold_digest,
                },
                pi_ccs: ccs.sumcheck,
                argument,
            }),
        };
        let times = CompressTimes {
            pi_ccs,
            argument: argument_times,
        };
        Ok((proof, times))
    }

    /// Accept `proof` for the expected statement exactly when `verify`
    /// accepts some proof with witnesses for it (Lean
    /// `RunCompleteness.Verifies`), up to the soundness error of the
    /// argument under `key`, which must have the size of `compression_key`.
    /// No proof carries the key.
    pub(crate) fn verify_final(
        &self,
        key: &VerificationKey,
        iteration: u64,
        z0: [F; 4],
        current: [F; 4],
        proof: &FinalProof,
    ) -> Result<VerifyTimes, BoundedDepthError> {
        if (proof.iteration, proof.z0, proof.current) != (iteration, z0, current) {
            return Err(BoundedDepthError::Statement(
                "the proof's statement differs from the expected one",
            ));
        }
        if iteration >= F::ORDER_U64 {
            return Err(BoundedDepthError::Statement(
                "iteration is not a canonical field counter",
            ));
        }
        let Some(steps) = &proof.steps else {
            return if iteration == 0 && current == z0 {
                Ok(VerifyTimes::default())
            } else {
                Err(BoundedDepthError::Statement("zero steps require equal endpoints"))
            };
        };
        if iteration == 0 || !matches!(steps.pc, 1 | 2) {
            return Err(BoundedDepthError::Statement(
                "a recursive proof needs a step and a circuit",
            ));
        }
        self.check_mode(steps.pc, steps.slots.len())?;
        for (index, (claim, depth)) in steps.slots.iter().enumerate() {
            let slot = |reason: &str| BoundedDepthError::Slot {
                slot: index,
                reason: reason.into(),
            };
            self.relations[index].shape(claim).map_err(slot)?;
            if *depth > LIGHT_FOLD_CAP {
                return Err(slot("depth is above the light fold cap"));
            }
        }
        let slots: Vec<_> = steps.slots.iter().map(|(claim, depth)| (claim, *depth)).collect();
        self.check_key_size(key)?;
        let depths: Vec<u64> = slots.iter().map(|slot| slot.1).collect();
        self.compression_layout(key.blocks(), steps.pc, &depths)?;
        let state = BoundedDepthState {
            keys: self.keys,
            iteration,
            z0,
            current,
            pc: steps.pc,
            slots: self.state_slots(&slots)?,
        };
        let relation = &self.relations[(steps.pc - 1) as usize];
        let fresh = CcsClaim {
            c: steps.fresh.c.clone(),
            x: encode_digest(state.hash()),
            m_in: PUBLIC_WORDS,
            adv: None,
        };
        let output = steps.fresh.claim(&fresh.x);

        let start = Instant::now();
        let commitment = ns_compress::byte_words(&steps.argument.witness_commitment);
        let mut transcript = Poseidon2Transcript::new_v1_2();
        let accepted = optimized_verify_with_profile(
            &mut transcript,
            relation.params().inner(),
            relation.structure(),
            std::slice::from_ref(&fresh),
            &[],
            std::slice::from_ref(&output),
            &steps.pi_ccs,
            &Self::compression_profile(&commitment),
        )
        .map_err(|error| BoundedDepthError::Fresh(fresh_failure(error)))?;
        if !accepted {
            return Err(BoundedDepthError::Fresh("the PiCCS proof of the fresh instance fails"));
        }
        let pi_ccs = start.elapsed();

        let statements = self.compression_statements(steps.pc, &slots, &output);
        let argument = ns_compress::verify(&mut transcript, key, &statements, &steps.argument)?;
        Ok(VerifyTimes { pi_ccs, argument })
    }
}

const MAGIC: &[u8; 16] = b"NS-OPTIONB-FIN-2";
/// The tag of a final proof of `Compression::Light`, which carries slot 0
/// alone and `pc = 1`.
const LIGHT_MAGIC: &[u8; 16] = b"NS-OPTIONB-FIN-L";
const SHAPE: Shape = Shape {
    public_width: PUBLIC_WORDS,
    witness_columns: 0,
    matrices: MATRICES,
};
/// No polynomial of a proof has more variables than this.
const MAX_ROUNDS: u64 = 30;

/// A byte string after its length.
fn write_bytes(output: &mut Writer, bytes: &[u8]) {
    output.word(bytes.len() as u64);
    output.0.extend_from_slice(bytes);
}

fn read_bytes(input: &mut Reader<'_>) -> Result<Vec<u8>, ProofCodecError> {
    let length = usize::try_from(input.word()?).map_err(|_| ProofCodecError("byte string length"))?;
    Ok(input.take(length)?.to_vec())
}

fn write_extension(output: &mut Writer, values: &[K]) {
    for value in values {
        output.fields(&value.as_coeffs());
    }
}

fn read_extension(input: &mut Reader<'_>, count: usize) -> Result<Vec<K>, ProofCodecError> {
    (0..count).map(|_| input.extension()).collect()
}

/// The rounds of one sum-check of the argument after their count.
fn read_rounds<const COEFFICIENTS: usize>(input: &mut Reader<'_>) -> Result<Vec<[K; COEFFICIENTS]>, ProofCodecError> {
    let rounds = input.word()?;
    if rounds > MAX_ROUNDS {
        return Err(ProofCodecError("argument round count"));
    }
    (0..rounds)
        .map(|_| {
            Ok(read_extension(input, COEFFICIENTS)?
                .try_into()
                .expect("the coefficients of one round"))
        })
        .collect()
}

impl FinalProof {
    /// The strict bytes of the proof: the tag of its slots, the statement,
    /// then for one or more steps the depth and claim of each slot that it
    /// carries, the fresh claim without its public input, the Π_CCS rounds
    /// and the argument. Field words are below the modulus, and each byte
    /// string of Akita and the rounds of each sum-check follow their count.
    pub(crate) fn to_bytes(&self) -> Result<Vec<u8>, ProofCodecError> {
        let light = match self.steps.as_ref().map(|steps| (steps.slots.len(), steps.pc)) {
            None | Some((2, _)) => false,
            Some((1, 1)) => true,
            Some((1, _)) => return Err(ProofCodecError("a proof of slot 0 alone has pc = 1")),
            Some(_) => return Err(ProofCodecError("slot count")),
        };
        let mut output = Writer(if light { LIGHT_MAGIC } else { MAGIC }.to_vec());
        output.word(self.steps.as_ref().map_or(0, |steps| steps.pc));
        output.word(self.iteration);
        output.fields(&self.z0);
        output.fields(&self.current);
        let Some(steps) = &self.steps else {
            return Ok(output.0);
        };
        for (claim, depth) in &steps.slots {
            output.word(*depth);
            output.claim(claim, &SHAPE)?;
        }
        let fresh = &steps.fresh;
        if fresh.r.len() != ROUNDS || fresh.eval_a.len() != MATRICES {
            return Err(ProofCodecError("fresh claim shape"));
        }
        output.commitment(&fresh.c)?;
        write_extension(&mut output, &fresh.r);
        output.evaluations(&fresh.eval_k);
        for family in &fresh.eval_a {
            output.evaluations(family);
        }
        output.0.extend_from_slice(&fresh.fold_digest);
        let rounds = &steps.pi_ccs.sumcheck_rounds;
        if rounds.len() != ROUNDS || rounds.iter().any(|round| round.len() != ROUND_COEFFICIENTS) {
            return Err(ProofCodecError("PiCCS round shape"));
        }
        for round in rounds {
            write_extension(&mut output, round);
        }
        let argument = &steps.argument;
        write_bytes(&mut output, &argument.witness_commitment);
        write_bytes(&mut output, &argument.inverse_commitment);
        output.word(argument.rounds.len() as u64);
        for round in &argument.rounds {
            write_extension(&mut output, round);
        }
        write_extension(&mut output, &[argument.witness_value]);
        write_extension(&mut output, &argument.inverse_values);
        write_extension(&mut output, &[argument.key_weight]);
        output.word(argument.key_rounds.len() as u64);
        for round in &argument.key_rounds {
            write_extension(&mut output, round);
        }
        write_extension(&mut output, &[argument.key_value]);
        write_bytes(&mut output, &argument.opening);
        Ok(output.0)
    }

    /// Decode untrusted bytes. Acceptance still needs `verify_final`.
    pub(crate) fn from_bytes(bytes: &[u8]) -> Result<Self, ProofCodecError> {
        let mut input = Reader(bytes);
        let light = match input.take(MAGIC.len())? {
            tag if tag == MAGIC => false,
            tag if tag == LIGHT_MAGIC => true,
            _ => return Err(ProofCodecError("format tag")),
        };
        let pc = input.word()?;
        let iteration = input.field()?.as_canonical_u64();
        let (z0, current) = (input.four()?, input.four()?);
        let steps = match (pc, light) {
            (0, false) if iteration == 0 && z0 == current => None,
            (0, false) => return Err(ProofCodecError("an initial proof starts at its initial state")),
            (1, true) | (1 | 2, false) => {
                let mut slot = || -> Result<(CeClaim, u64), ProofCodecError> {
                    let depth = input.field()?.as_canonical_u64();
                    Ok((input.claim(&SHAPE)?, depth))
                };
                let slots = (0..if light { 1 } else { 2 })
                    .map(|_| slot())
                    .collect::<Result<Vec<_>, _>>()?;
                let c = input.commitment()?;
                let r = read_extension(&mut input, ROUNDS)?;
                let eval_k = input.evaluations()?;
                let eval_a = (0..MATRICES)
                    .map(|_| input.evaluations())
                    .collect::<Result<Vec<_>, _>>()?;
                let mut fold_digest = [0u8; 32];
                for lane in fold_digest.chunks_exact_mut(8) {
                    lane.copy_from_slice(&input.field()?.as_canonical_u64().to_le_bytes());
                }
                let pi_ccs = (0..ROUNDS)
                    .map(|_| read_extension(&mut input, ROUND_COEFFICIENTS))
                    .collect::<Result<Vec<_>, _>>()?;
                let witness_commitment = read_bytes(&mut input)?;
                let inverse_commitment = read_bytes(&mut input)?;
                let rounds = read_rounds(&mut input)?;
                let witness_value = input.extension()?;
                let inverse_values = read_extension(&mut input, 4)?
                    .try_into()
                    .expect("four values");
                let key_weight = input.extension()?;
                let key_rounds = read_rounds(&mut input)?;
                let key_value = input.extension()?;
                Some(Steps {
                    pc,
                    slots,
                    fresh: Fresh {
                        c,
                        r,
                        eval_k,
                        eval_a,
                        fold_digest,
                    },
                    pi_ccs: pi_ccs::SumcheckProof::new(pi_ccs),
                    argument: ns_compress::Proof {
                        witness_commitment,
                        inverse_commitment,
                        rounds,
                        witness_value,
                        inverse_values,
                        key_weight,
                        key_rounds,
                        key_value,
                        opening: read_bytes(&mut input)?,
                    },
                })
            }
            _ => return Err(ProofCodecError("program counter")),
        };
        if !input.0.is_empty() {
            return Err(ProofCodecError("bytes after the proof"));
        }
        Ok(Self {
            iteration,
            z0,
            current,
            steps,
        })
    }
}

fn fresh_failure(error: neo_reductions::PiCcsError) -> &'static str {
    match error {
        neo_reductions::PiCcsError::SumcheckError(_) => "a PiCCS round of the fresh instance fails",
        neo_reductions::PiCcsError::ProtocolError(_) => "the PiCCS outputs of the fresh instance differ",
        _ => "the PiCCS proof of the fresh instance has another shape",
    }
}

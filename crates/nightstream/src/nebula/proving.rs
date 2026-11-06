//! The memory prover and the memory verifier of a circuit (SPEC.md
//! Algorithm 9 and Algorithm 10). The prover runs each segment natively,
//! proves its invocations, and keeps only the last proof. The proof carries
//! the final carry, which the final Stage 1 state binds, so the verifier
//! reads every statement integer from it and the caller never moves a carry.

use std::path::Path;

use p3_field::{PrimeCharacteristicRing, PrimeField64};
use p3_goldilocks::Goldilocks as F;

use super::app::Application;
use super::carry::Carry;
use super::geometry::Geometry;
use super::segment::SegmentError;
use super::transcript::Digest;
use super::{memory_security, terminal_states, Program, Run, Statement, TerminalError};
use crate::lifecycle::ProofCodecError;

/// The bytes of the 39 carry words at the start of an encoded proof.
const CARRY_BYTES: usize = 39 * 8;

/// A memory proof: the step proof of every invocation and the final carry.
pub struct Proof {
    stage1: crate::Proof,
    carry: Carry,
}

#[cfg(test)]
impl Proof {
    /// A proof from a step proof that a test made step by step.
    pub(crate) fn from_parts(stage1: crate::Proof, carry: Carry) -> Self {
        Self { stage1, carry }
    }
}

/// A compressed memory proof (`Prover::compress`) and the final carry.
#[derive(Clone)]
pub struct FinalProof {
    stage1: crate::FinalProof,
    carry: Carry,
}

/// What a verified memory proof shows beyond its two states.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Verified {
    /// `S · N`, the step count of the run.
    pub steps: u64,
    /// The memory root after the last segment.
    pub final_memory_root: Digest,
}

#[derive(Debug, thiserror::Error)]
pub enum ProveError {
    #[error("the program is for another geometry")]
    Geometry,
    #[error(transparent)]
    Segment(#[from] SegmentError),
    #[error(transparent)]
    Proof(#[from] crate::Error),
    #[error("a proof needs at least one segment")]
    Empty,
    #[error("an earlier segment failed, so this run has no proof")]
    Failed,
}

/// Why a memory proof does not verify.
#[derive(Debug, thiserror::Error)]
pub enum VerifyError {
    #[error("the program is for another geometry")]
    Geometry,
    #[error(transparent)]
    Terminal(#[from] TerminalError),
    #[error(transparent)]
    Proof(#[from] crate::Error),
}

/// The memory prover of one circuit (`Circuit::prover`). It proves runs of
/// every program of the circuit's geometry.
pub struct Prover<A> {
    geometry: Geometry,
    app: A,
    stage1: crate::Prover,
}

impl<A: Application + Clone> Prover<A> {
    pub(super) fn new(geometry: Geometry, app: A, stage1: crate::Prover) -> Self {
        Self { geometry, app, stage1 }
    }

    /// Load the prover of a circuit that `Circuit::write` saved for
    /// `geometry` and `app`. A verifier compiles its own circuit and never
    /// loads this file.
    pub fn load(
        path: impl AsRef<Path>,
        geometry: Geometry,
        app: A,
        engine: crate::Engine,
        minimum_security_bits: u32,
    ) -> Result<Self, crate::Error> {
        memory_security(&geometry, minimum_security_bits)?;
        Ok(Self::new(
            geometry,
            app,
            crate::Prover::load(path, engine, minimum_security_bits)?,
        ))
    }

    /// Start a run of `program` from `initial`.
    pub fn start(&self, program: &Program, initial: A::State) -> Result<Proving<'_, A>, ProveError> {
        if program.geometry != self.geometry {
            return Err(ProveError::Geometry);
        }
        Ok(Proving {
            prover: self,
            run: Run::new(self.app.clone(), program, initial),
            last: None,
            failed: false,
        })
    }

    /// The carry words as little-endian words, then the circuit's strict
    /// encoding of the Stage 1 proof, for `Verifier::decode_proof`.
    pub fn encode_proof(&self, proof: &Proof) -> Result<Vec<u8>, crate::Error> {
        let mut bytes = carry_bytes(&proof.carry);
        bytes.extend(self.stage1.encode_proof(&proof.stage1)?);
        Ok(bytes)
    }

    /// `crate::Prover::compression_setup` of the circuit.
    pub fn compression_setup(&self, dir: impl AsRef<Path>) -> Result<crate::CompressionSetup, crate::Error> {
        self.stage1.compression_setup(dir)
    }

    /// Compress `proof` (`crate::Prover::compress`). The carry
    /// does not change.
    pub fn compress(&self, proof: &Proof, setup: &crate::CompressionSetup) -> Result<FinalProof, crate::Error> {
        Ok(FinalProof {
            stage1: self.stage1.compress(&proof.stage1, setup)?,
            carry: proof.carry,
        })
    }
}

/// One run in progress: the native run and the last proof.
pub struct Proving<'p, A: Application> {
    prover: &'p Prover<A>,
    run: Run<A>,
    last: Option<crate::Proof>,
    failed: bool,
}

impl<A: Application + Clone> Proving<'_, A> {
    /// Run the next segment of `N` steps natively, then prove each of its
    /// invocations in order. After an error the run has no proof.
    pub fn segment(&mut self, inputs: &[A::Input]) -> Result<(), ProveError> {
        if self.failed {
            return Err(ProveError::Failed);
        }
        self.failed = true;
        let initial = self.run.initial_state();
        for invocation in self.run.segment(inputs)? {
            let previous = match self.last.take() {
                Some(previous) => previous,
                None => self.prover.stage1.initial(initial)?,
            };
            self.last = Some(
                self.prover
                    .stage1
                    .extend_invocation(&previous, &invocation.words, invocation.output)?,
            );
        }
        self.failed = false;
        Ok(())
    }

    /// The application state after the last segment.
    pub fn state(&self) -> &A::State {
        self.run.state()
    }

    /// The proof of every segment so far and the final carry.
    pub fn finish(self) -> Result<Proof, ProveError> {
        if self.failed {
            return Err(ProveError::Failed);
        }
        Ok(Proof {
            stage1: self.last.ok_or(ProveError::Empty)?,
            carry: *self.run.carry(),
        })
    }
}

/// The memory verifier of one circuit (`Circuit::verifier`): the SPEC.md
/// Algorithm 10 terminal checks, then the Stage 1 verifier of the circuit's
/// relation. It verifies proofs of every program of the circuit's geometry.
pub struct Verifier<A> {
    geometry: Geometry,
    app: A,
    stage1: crate::Verifier,
}

impl<A: Application> Verifier<A> {
    pub(super) fn new(geometry: Geometry, app: A, stage1: crate::Verifier) -> Self {
        Self { geometry, app, stage1 }
    }

    /// Decode untrusted proof bytes: the canonical carry words, then
    /// `crate::Verifier::decode_proof`. Acceptance still needs `verify`.
    pub fn decode_proof(&self, bytes: &[u8]) -> Result<Proof, crate::Error> {
        let (carry, rest) = decode_carry(bytes)?;
        Ok(Proof {
            stage1: self.stage1.decode_proof(rest)?,
            carry,
        })
    }

    /// Verify a memory proof of `program` from `initial` to `last` (SPEC.md
    /// Algorithm 10). The statement integers come from the proof's carry, so
    /// the terminal checks on them and the final Stage 1 state bind the
    /// carry. The initial state opens the start carry, which holds the
    /// program's `D_init` (`Program::d_init`).
    pub fn verify(
        &self,
        program: &Program,
        initial: &A::State,
        last: &A::State,
        proof: &Proof,
    ) -> Result<Verified, VerifyError> {
        let state = self.terminal(program, initial, last, &proof.carry)?;
        self.stage1.verify(&state, &proof.stage1)?;
        Ok(verified(&proof.carry, &program.geometry))
    }

    /// The trusted compression key of the circuit (`crate::Verifier::compression_key`).
    pub fn compression_key(&self) -> Result<crate::CompressionKey, crate::Error> {
        self.stage1.compression_key()
    }

    /// Verify a compressed memory proof as `verify` does: the terminal
    /// checks, then `crate::Verifier::verify_final` with `key`.
    pub fn verify_final(
        &self,
        program: &Program,
        initial: &A::State,
        last: &A::State,
        key: &crate::CompressionKey,
        proof: &FinalProof,
    ) -> Result<Verified, VerifyError> {
        let state = self.terminal(program, initial, last, &proof.carry)?;
        self.stage1.verify_final(&state, key, &proof.stage1)?;
        Ok(verified(&proof.carry, &program.geometry))
    }

    /// `verify` of a Stage 1 proof that a test made step by step, with the
    /// statement and the carry that the test gives.
    #[cfg(test)]
    pub(crate) fn verify_parts(
        &self,
        program: &Program,
        statement: &Statement<A::State>,
        carry: &Carry,
        stage1: &crate::Proof,
    ) -> Result<(), VerifyError> {
        if program.geometry != self.geometry {
            return Err(VerifyError::Geometry);
        }
        let (z0, current) = terminal_states(&self.app, program, statement, carry)?;
        self.stage1
            .verify(&crate::State::new(statement.steps, z0, current), stage1)?;
        Ok(())
    }

    /// The SPEC.md Algorithm 10 statement of `carry` and its terminal checks.
    fn terminal(
        &self,
        program: &Program,
        initial: &A::State,
        last: &A::State,
        carry: &Carry,
    ) -> Result<crate::State, VerifyError> {
        if program.geometry != self.geometry {
            return Err(VerifyError::Geometry);
        }
        let statement = Statement {
            steps: carry.seg_idx.saturating_mul(program.geometry.n as u64),
            initial: initial.clone(),
            final_: last.clone(),
            segments: carry.seg_idx,
            final_ts: carry.ts,
            final_root: carry.mem_root,
        };
        let (z0, current) = terminal_states(&self.app, program, &statement, carry)?;
        Ok(crate::State::new(statement.steps, z0, current))
    }
}

fn verified(carry: &Carry, geometry: &Geometry) -> Verified {
    Verified {
        steps: carry.seg_idx * geometry.n as u64,
        final_memory_root: carry.mem_root,
    }
}

fn carry_bytes(carry: &Carry) -> Vec<u8> {
    carry
        .words()
        .iter()
        .flat_map(|word| word.as_canonical_u64().to_le_bytes())
        .collect()
}

/// The carry of the first `CARRY_BYTES` bytes and the bytes after them.
/// Each word must be below `q`.
fn decode_carry(bytes: &[u8]) -> Result<(Carry, &[u8]), crate::Error> {
    if bytes.len() < CARRY_BYTES {
        return Err(ProofCodecError("memory carry length").into());
    }
    let mut words = [F::default(); 39];
    for (word, chunk) in words.iter_mut().zip(bytes[..CARRY_BYTES].chunks_exact(8)) {
        let value = u64::from_le_bytes(chunk.try_into().expect("eight bytes"));
        if value >= F::ORDER_U64 {
            return Err(ProofCodecError("non-canonical memory carry word").into());
        }
        *word = F::from_u64(value);
    }
    Ok((Carry::from_words(&words), &bytes[CARRY_BYTES..]))
}

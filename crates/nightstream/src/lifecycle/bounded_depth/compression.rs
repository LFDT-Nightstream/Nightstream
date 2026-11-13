//! The compressed proof (SPEC.md section 6): the statement, both slot claims,
//! and an argument over the Akita commitment scheme that the verifier of
//! `BoundedDepth::verify` accepts some witnesses for them. It carries no
//! witness and cannot be extended.
//!
//! The terminal check has three CE claims. Each slot's accumulator is one,
//! with the bound of its depth. The fresh instance becomes one by a Π_CCS
//! run with no running claim: its sum-check proves each row of the relation
//! and that each coefficient is in {-1, 0, 1}, given the evaluations of the
//! claim. That run binds the Akita commitment of the witnesses before its
//! challenges, so the committed fresh witness is the one that it judges.
//! `neo-compress` proves the three claims.
//!
//! The soundness error is the sum of three parts. The caller's
//! `minimum_security_bits` gates the first two, each alone and never their
//! sum. The public API has no default for it, and no fold reaches 110 bits.
//!
//! 1. The Π_CCS run has the test error of one fold with one source.
//!    `Params::for_ccs_shape` gates the larger error of a fold with 17
//!    sources, about `2^-109.15` for the Poseidon2 hash chain.
//! 2. The argument has the error of `neo_compress::error_bits`: its batches,
//!    its lookups, its sum-check and the point of `V`, `2^-119.15` for the
//!    Poseidon2 hash chain. `compression_layout` gates it.
//! 3. The two Akita openings have the knowledge error of Akita's
//!    `fp64_dense` schedules. No gate here computes or bounds it.

use std::time::{Duration, Instant};

use neo_ajtai::Commitment;
use neo_ccs::Mat;
use neo_compress::{Layout, Norm, Statement};
use neo_math::{KExtensions, D, F, K};
use neo_reductions::optimized_engine::optimized_verify_with_profile;
use neo_reductions::{FoldProfile, NormSources};
use neo_transcript::{domain_chunk_v1_2, Poseidon2Transcript};
use nightstream_fprime::{
    encode_digest, BoundedDepthState, PI_CCS_V1_2_ROUND_COEFFICIENT_COUNT as ROUND_COEFFICIENTS,
    PI_CCS_V1_2_ROUND_COUNT as ROUNDS, PUBLIC_WORDS,
};
use p3_field::PrimeField64;

use super::super::encoding::{Reader, Shape, Writer};
use super::super::ProofCodecError;
use super::{words, BoundedDepth, BoundedDepthError, Envelope, LIGHT_FOLD_CAP};
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
    /// Each slot's accumulator claim and depth.
    pub(super) slots: [(CeClaim, u64); 2],
    pub(super) fresh: Fresh,
    pub(super) pi_ccs: pi_ccs::SumcheckProof,
    pub(super) argument: neo_compress::Proof,
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
    argument: neo_compress::ProverTimes,
}

/// Where `verify_final` spends its time.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct VerifyTimes {
    pi_ccs: Duration,
    argument: neo_compress::VerifierTimes,
}

fn milliseconds(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1e3
}

impl std::fmt::Display for CompressTimes {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let times = &self.argument;
        write!(
            f,
            "commit W {:.0} ms, PiCCS {:.0} ms, commit V {:.0} ms, weights {:.0} ms, sum-check {:.0} ms, openings {:.0} ms",
            milliseconds(times.commit_witness),
            milliseconds(self.pi_ccs),
            milliseconds(times.commit_inverse),
            milliseconds(times.weights),
            milliseconds(times.sumcheck),
            milliseconds(times.openings),
        )
    }
}

impl std::fmt::Display for VerifyTimes {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "PiCCS {:.0} ms, Akita {:.0} ms, public weights {:.0} ms",
            milliseconds(self.pi_ccs),
            milliseconds(self.argument.akita),
            milliseconds(self.argument.weights),
        )
    }
}

impl BoundedDepth {
    /// The claims of the terminal check: slot 0, slot 1, then the fresh
    /// instance of circuit `pc − 1`.
    fn compression_claims(&self, pc: u64, depths: [u64; 2]) -> [(usize, Norm); 3] {
        let slot = |slot: usize| {
            let bound = self.accumulator_bound(depths[slot]);
            (self.relations[slot].blocks(), Norm::Table { bound })
        };
        [
            slot(0),
            slot(1),
            (self.relations[(pc - 1) as usize].blocks(), Norm::Caller),
        ]
    }

    /// The layout of the claims, when `neo_compress::error_bits` of it is
    /// within the security minimum. That number is the error of the batches,
    /// the lookups, the sum-check and the point of `V`. It has no term for
    /// the Π_CCS run or for the Akita openings.
    pub(crate) fn compression_layout(&self, pc: u64, depths: [u64; 2]) -> Result<Layout, BoundedDepthError> {
        let layout = Layout::new(&self.compression_claims(pc, depths))?;
        if neo_compress::error_bits(&layout, PUBLIC_WORDS, MATRICES) < f64::from(self.minimum_security_bits) {
            return Err(BoundedDepthError::Statement(
                "the batch, lookup and sum-check error of the compression argument is above the security minimum",
            ));
        }
        Ok(layout)
    }

    /// The Π_CCS profile of the fresh instance: it binds `commitment`, the
    /// Akita commitment of the witnesses, after the statement.
    fn compression_profile(commitment: &[F]) -> FoldProfile<'_> {
        FoldProfile {
            domain: domain_chunk_v1_2(PI_CCS_DOMAIN),
            children: commitment,
            norm: NormSources::Fresh,
        }
    }

    fn compression_statements<'a>(
        &'a self,
        pc: u64,
        slots: [(&'a CeClaim, u64); 2],
        fresh: &'a CeClaim,
    ) -> [Statement<'a>; 3] {
        let slot = |slot: usize| Statement {
            claim: slots[slot].0,
            rows: &self.relations[slot],
            norm: Norm::Table {
                bound: self.accumulator_bound(slots[slot].1),
            },
        };
        [
            slot(0),
            slot(1),
            Statement {
                claim: fresh,
                rows: &self.relations[(pc - 1) as usize],
                norm: Norm::Caller,
            },
        ]
    }

    /// Compress `envelope`, which does not change. The proof is not checked
    /// here: an envelope that `verify` rejects gives an error or a proof that
    /// `verify_final` rejects.
    pub(crate) fn compress(&self, envelope: &Envelope) -> Result<(FinalProof, CompressTimes), BoundedDepthError> {
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
                argument: neo_compress::ProverTimes::default(),
            };
            return Ok((proof, times));
        };
        if !matches!(pc, 1 | 2) {
            return Err(BoundedDepthError::Statement(
                "a recursive proof needs a step and a circuit",
            ));
        }
        let slots = envelope.slots.each_ref();
        let layout = self.compression_layout(pc, slots.map(|slot| slot.depth))?;
        let committed = neo_compress::commit(
            layout,
            &[
                &slots[0].accumulator.witness,
                &slots[1].accumulator.witness,
                &fresh.witness.Z,
            ],
        )?;

        let start = Instant::now();
        let commitment = neo_compress::byte_words(committed.commitment());
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
        let statements =
            self.compression_statements(pc, slots.map(|slot| (&slot.accumulator.claim, slot.depth)), &fresh);
        let (argument, argument_times) = committed.prove(transcript.inner_mut(), &statements)?;
        let proof = FinalProof {
            iteration,
            z0,
            current,
            steps: Some(Steps {
                pc,
                slots: slots.map(|slot| (slot.accumulator.claim.clone(), slot.depth)),
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
    /// argument.
    pub(crate) fn verify_final(
        &self,
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
        let slots = steps.slots.each_ref();
        self.compression_layout(steps.pc, slots.map(|slot| slot.1))?;
        let state = BoundedDepthState {
            keys: self.keys,
            iteration,
            z0,
            current,
            pc: steps.pc,
            slots: slots.map(|(claim, depth)| words::slot(claim, *depth)),
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
        let commitment = neo_compress::byte_words(&steps.argument.witness_commitment);
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

        let statements = self.compression_statements(steps.pc, slots.map(|(claim, depth)| (claim, *depth)), &output);
        let argument = neo_compress::verify(&mut transcript, &statements, &steps.argument)?;
        Ok(VerifyTimes { pi_ccs, argument })
    }
}

const MAGIC: &[u8; 16] = b"NS-OPTIONB-FIN-1";
const SHAPE: Shape = Shape {
    public_width: PUBLIC_WORDS,
    witness_columns: 0,
    matrices: MATRICES,
};
/// No layout has more variables than the largest Akita schedule.
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

impl FinalProof {
    /// The strict bytes of the proof: the statement, then for one or more
    /// steps each slot's depth and claim, the fresh claim without its public
    /// input, the Π_CCS rounds and the argument. Field words are below the
    /// modulus, and each byte string of Akita follows its length.
    pub(crate) fn to_bytes(&self) -> Result<Vec<u8>, ProofCodecError> {
        let mut output = Writer(MAGIC.to_vec());
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
        write_bytes(&mut output, &argument.witness_opening);
        write_bytes(&mut output, &argument.inverse_opening);
        Ok(output.0)
    }

    /// Decode untrusted bytes. Acceptance still needs `verify_final`.
    pub(crate) fn from_bytes(bytes: &[u8]) -> Result<Self, ProofCodecError> {
        let mut input = Reader(bytes);
        if input.take(MAGIC.len())? != MAGIC {
            return Err(ProofCodecError("format tag"));
        }
        let pc = input.word()?;
        let iteration = input.field()?.as_canonical_u64();
        let (z0, current) = (input.four()?, input.four()?);
        let steps = match pc {
            0 if iteration == 0 && z0 == current => None,
            0 => return Err(ProofCodecError("an initial proof starts at its initial state")),
            1 | 2 => {
                let mut slot = || -> Result<(CeClaim, u64), ProofCodecError> {
                    let depth = input.field()?.as_canonical_u64();
                    Ok((input.claim(&SHAPE)?, depth))
                };
                let slots = [slot()?, slot()?];
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
                let rounds = input.word()?;
                if rounds > MAX_ROUNDS {
                    return Err(ProofCodecError("argument round count"));
                }
                let rounds = (0..rounds)
                    .map(|_| {
                        Ok(read_extension(&mut input, 4)?
                            .try_into()
                            .expect("four coefficients"))
                    })
                    .collect::<Result<Vec<_>, ProofCodecError>>()?;
                let witness_value = input.extension()?;
                let inverse_values = read_extension(&mut input, 4)?
                    .try_into()
                    .expect("four values");
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
                    argument: neo_compress::Proof {
                        witness_commitment,
                        inverse_commitment,
                        rounds,
                        witness_value,
                        inverse_values,
                        witness_opening: read_bytes(&mut input)?,
                        inverse_opening: read_bytes(&mut input)?,
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

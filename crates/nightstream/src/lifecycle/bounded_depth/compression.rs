//! SPEC.md section 6 and Algorithms 9 to 11; Lean `FinalVerifier.Mode`.

use std::sync::Arc;
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
use p3_field::{PrimeCharacteristicRing, PrimeField64};

use super::super::encoding::{Reader, Shape, Writer};
use super::super::ProofCodecError;
use super::{words, BoundedDepth, BoundedDepthError, Compression, Envelope, FreshMatrixRows, LIGHT_FOLD_CAP};
use crate::folding::transcript::Transcript;
use crate::folding::{pi_ccs, CcsClaim, CeClaim, FoldContext, RunningInstance};

const PI_CCS_DOMAIN: &[u8] = b"Nightstream/OptionB/compress/ccs/v1";
const MATRICES: usize = nightstream_fprime::PI_CCS_V1_2_MATRIX_COUNT;

#[derive(Clone)]
pub(crate) struct FinalProof {
    pub(super) iteration: u64,
    pub(super) z0: [F; 4],
    pub(super) current: [F; 4],
    pub(super) steps: Option<Steps>,
}

#[derive(Clone)]
pub(super) struct Steps {
    pub(super) pc: u64,
    pub(super) slots: Vec<(CeClaim, u64)>,
    pub(super) fresh: Fresh,
    pub(super) pi_ccs: pi_ccs::SumcheckProof,
    pub(super) argument: ns_compress::Proof,
}

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

#[derive(Clone, Copy, Debug)]
pub(crate) struct CompressTimes {
    pi_ccs: Duration,
    argument: ns_compress::ProverTimes,
}

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
    /// Lean `Spec.Compression.Rust.productionClaims`, `lightClaims`.
    fn compression_claims(&self, pc: u64, depths: &[u64]) -> Vec<(usize, Norm)> {
        let mut claims: Vec<_> = depths
            .iter()
            .enumerate()
            .map(|(slot, depth)| {
                let bound = self.accumulator_bound(*depth);
                (self.relations[slot].blocks(), Norm::Table { bound })
            })
            .collect();
        claims.push((self.relations[(pc - 1) as usize].blocks(), Norm::BoundByCaller));
        claims
    }

    pub(crate) fn compression_setup(&self) -> Result<Arc<ProverKey>, BoundedDepthError> {
        Ok(ProverKey::shared(self.key_blocks())?)
    }

    pub(crate) fn compression_key(&self) -> Result<VerificationKey, BoundedDepthError> {
        Ok(VerificationKey::new(self.key_blocks())?)
    }

    pub(crate) fn decode_compression_key(&self, bytes: &[u8]) -> Result<VerificationKey, BoundedDepthError> {
        let key = VerificationKey::from_bytes(bytes)?;
        self.check_key_size(&key)?;
        Ok(key)
    }

    fn check_key_size(&self, key: &VerificationKey) -> Result<(), BoundedDepthError> {
        if key.blocks() != ns_compress::key_blocks(self.key_blocks())? {
            return Err(ns_compress::Error::Key("the key is for another circuit size").into());
        }
        Ok(())
    }

    pub(super) fn key_blocks(&self) -> usize {
        match self.compression {
            Compression::Both => self.relations[0].blocks().max(self.relations[1].blocks()),
            Compression::Light => self.relations[0].blocks(),
        }
    }

    /// SPEC.md Algorithm 10, step 6; Lean `FinalVerifier.ModeChecks`.
    fn state_slots(&self, slots: &[(&CeClaim, u64)]) -> Result<[nightstream_fprime::Slot; 2], BoundedDepthError> {
        Ok(match self.compression {
            Compression::Both => [words::slot(slots[0].0, slots[0].1), words::slot(slots[1].0, slots[1].1)],
            Compression::Light => [
                words::slot(slots[0].0, slots[0].1),
                words::slot(&self.default_slot(1)?.accumulator.claim, 0),
            ],
        })
    }

    /// SPEC.md Algorithm 9, Rejects.
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

    /// SPEC.md Algorithm 10, step 3; Lean `FinalVerifier.ModeChecks`.
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

    /// `Spec.Compression.Rust.params_run_soundness_bits`.
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
            norm: Norm::BoundByCaller,
        });
        statements
    }

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
        let commitment = ns_compress::byte_words(committed.commitment_to_bind());
        let mut transcript = Transcript::session();
        let rows = FreshMatrixRows {
            relation: &self.relations[(pc - 1) as usize],
            prepared: envelope.checked_rows.as_deref(),
            witness: &fresh.witness.Z,
        };
        let mut ccs = pi_ccs::prove_from_parts_with_rows(
            &mut transcript,
            &self.backend,
            FoldContext {
                rows: &rows,
                ..rows.relation.context()
            },
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
        let (argument, argument_times) = committed.prove_with_backend(
            transcript.inner_mut(),
            &statements,
            self.backend.engine() == crate::Engine::Metal,
        )?;
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

    /// Lean `RunCompleteness.Verifies`.
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
        let slots: Vec<_> = steps
            .slots
            .iter()
            .map(|(claim, depth)| (claim, *depth))
            .collect();
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

const MAGIC: &[u8; 16] = b"NS-ONEINV-FIN--1";
/// SPEC.md section 4.5.
const LIGHT_MAGIC: &[u8; 16] = b"NS-ONEINV-FIN--L";
const ZERO_MAGIC: &[u8; 16] = b"NS-ONEINV-FIN--Z";
const SIGNED_MAGIC: &[u8; 16] = b"NS-ONEINV-FIN--S";
const PUBLIC_START: usize = 8 * (1 + nightstream_fprime::COMMITMENT_WORDS);
const PUBLIC_END: usize = PUBLIC_START + 8 * PUBLIC_WORDS;
const SLOT_BYTES: usize = PUBLIC_END + 8 * (2 * ROUNDS + 2 * (MATRICES + 1) * D);
const MAX_POLYNOMIAL_VARIABLES: u64 = 30;
const SHAPE: Shape = Shape {
    public_width: PUBLIC_WORDS,
    witness_columns: 0,
    matrices: MATRICES,
};

fn signed_byte(value: u64) -> Option<u8> {
    match value {
        0..=127 => Some(value as u8),
        x if (F::ORDER_U64 - 128..F::ORDER_U64).contains(&x) => Some((x - (F::ORDER_U64 - 256)) as u8),
        _ => None,
    }
}

fn public_bytes(slot: &[u8]) -> Option<[u8; PUBLIC_WORDS]> {
    let mut bytes = [0; PUBLIC_WORDS];
    for (out, word) in bytes
        .iter_mut()
        .zip(slot[PUBLIC_START..PUBLIC_END].as_chunks::<8>().0)
    {
        *out = signed_byte(u64::from_le_bytes(*word))?;
    }
    Some(bytes)
}

fn compact_light_slot(output: &mut Writer, begin: usize) -> &'static [u8; 16] {
    let slot = &output.0[begin..];
    if slot.iter().all(|&b| b == 0) {
        output.0.truncate(begin);
        ZERO_MAGIC
    } else if let Some(public) = public_bytes(slot) {
        output
            .0
            .splice(begin + PUBLIC_START..begin + PUBLIC_END, public);
        SIGNED_MAGIC
    } else {
        LIGHT_MAGIC
    }
}

fn read_slot(input: &mut Reader<'_>, tag: &[u8]) -> Result<(CeClaim, u64), ProofCodecError> {
    let mut expanded;
    let slot = if tag == ZERO_MAGIC {
        &[0; SLOT_BYTES][..]
    } else if tag == SIGNED_MAGIC {
        expanded = Vec::with_capacity(SLOT_BYTES);
        expanded.extend_from_slice(input.take(PUBLIC_START)?);
        for &byte in input.take(PUBLIC_WORDS)? {
            let word = F::from_i8(byte as i8).as_canonical_u64();
            expanded.extend_from_slice(&word.to_le_bytes());
        }
        expanded.extend_from_slice(input.take(SLOT_BYTES - PUBLIC_END)?);
        if expanded.iter().all(|&b| b == 0) {
            return Err(ProofCodecError("default slot needs the default tag"));
        }
        &expanded
    } else {
        let slot = input.take(SLOT_BYTES)?;
        if tag == LIGHT_MAGIC && public_bytes(slot).is_some() {
            return Err(ProofCodecError("compact slot needs a compact tag"));
        }
        slot
    };
    let mut slot = Reader(slot);
    let depth = slot.field()?.as_canonical_u64();
    Ok((slot.claim(&SHAPE)?, depth))
}

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

fn quadratic(coefficients: &[K]) -> bool {
    coefficients[3..].iter().all(|&value| value == K::ZERO)
}

fn write_rounds<const N: usize>(output: &mut Writer, rounds: &[[K; N]], masked: bool) -> Result<(), ProofCodecError> {
    if rounds.len() > MAX_POLYNOMIAL_VARIABLES as usize {
        return Err(ProofCodecError("argument round count"));
    }
    output.word(rounds.len() as u64);
    let mut mask = 0u32;
    if masked {
        for (i, round) in rounds.iter().enumerate() {
            if quadratic(round) {
                mask |= 1 << i;
            }
        }
        output.0.extend_from_slice(&mask.to_le_bytes());
    }
    for (i, round) in rounds.iter().enumerate() {
        let count = if mask & (1 << i) != 0 { 3 } else { N };
        write_extension(output, &round[..count]);
    }
    Ok(())
}

fn read_rounds<const N: usize>(input: &mut Reader<'_>, masked: bool) -> Result<Vec<[K; N]>, ProofCodecError> {
    let rounds = input.word()?;
    if rounds > MAX_POLYNOMIAL_VARIABLES {
        return Err(ProofCodecError("argument round count"));
    }
    let mask = if masked {
        let bits = u32::from_le_bytes(input.take(4)?.try_into().expect("one u32"));
        if bits >> rounds != 0 {
            return Err(ProofCodecError("surplus round mask bits"));
        }
        bits
    } else {
        0
    };
    (0..rounds)
        .map(|i| {
            let short = mask & (1 << i) != 0;
            let count = if short { 3 } else { N };
            let mut round = [K::ZERO; N];
            for value in &mut round[..count] {
                *value = input.extension()?;
            }
            if masked && !short && quadratic(&round) {
                return Err(ProofCodecError("explicit quadratic round"));
            }
            Ok(round)
        })
        .collect()
}

impl FinalProof {
    pub(crate) fn to_bytes(&self) -> Result<Vec<u8>, ProofCodecError> {
        if self.iteration >= F::ORDER_U64 {
            return Err(ProofCodecError("iteration word"));
        }
        let light = match &self.steps {
            None if self.iteration != 0 || self.z0 != self.current => {
                return Err(ProofCodecError("an initial proof starts at its initial state"))
            }
            None => false,
            Some(steps) => match (steps.slots.len(), steps.pc) {
                (2, 1 | 2) => false,
                (1, 1) => true,
                (1 | 2, _) => return Err(ProofCodecError("program counter")),
                _ => return Err(ProofCodecError("slot count")),
            },
        };
        let mut output = Writer(MAGIC.to_vec());
        output.word(self.steps.as_ref().map_or(0, |steps| steps.pc));
        output.word(self.iteration);
        output.fields(&self.z0);
        output.fields(&self.current);
        let Some(steps) = &self.steps else {
            return Ok(output.0);
        };
        let slot_begin = output.0.len();
        for (claim, depth) in &steps.slots {
            if *depth >= F::ORDER_U64 {
                return Err(ProofCodecError("slot depth word"));
            }
            output.word(*depth);
            output.claim(claim, &SHAPE)?;
        }
        let tag = if light {
            compact_light_slot(&mut output, slot_begin)
        } else {
            MAGIC
        };
        output.0[..MAGIC.len()].copy_from_slice(tag);
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
        write_rounds(&mut output, &argument.rounds, tag == ZERO_MAGIC)?;
        write_extension(&mut output, &[argument.witness_value]);
        write_extension(&mut output, &[argument.shared_inverse]);
        write_extension(&mut output, &[argument.key_part]);
        write_rounds(&mut output, &argument.key_rounds, false)?;
        write_extension(&mut output, &[argument.key_polynomial_value]);
        write_bytes(&mut output, &argument.opening);
        Ok(output.0)
    }

    pub(crate) fn from_bytes(bytes: &[u8]) -> Result<Self, ProofCodecError> {
        let mut input = Reader(bytes);
        let tag = input.take(MAGIC.len())?;
        let light = match tag {
            tag if tag == MAGIC => false,
            tag if tag == LIGHT_MAGIC || tag == ZERO_MAGIC || tag == SIGNED_MAGIC => true,
            _ => return Err(ProofCodecError("format tag")),
        };
        let pc = input.word()?;
        let iteration = input.field()?.as_canonical_u64();
        let (z0, current) = (input.four()?, input.four()?);
        let steps = match (pc, light) {
            (0, false) if iteration == 0 && z0 == current => None,
            (0, false) => return Err(ProofCodecError("an initial proof starts at its initial state")),
            (1, true) | (1 | 2, false) => {
                let slots = (0..if light { 1 } else { 2 })
                    .map(|_| read_slot(&mut input, tag))
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
                let rounds = read_rounds(&mut input, tag == ZERO_MAGIC)?;
                let witness_value = input.extension()?;
                let shared_inverse = input.extension()?;
                let key_part = input.extension()?;
                let key_rounds = read_rounds(&mut input, false)?;
                let key_polynomial_value = input.extension()?;
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
                        shared_inverse,
                        key_part,
                        key_rounds,
                        key_polynomial_value,
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

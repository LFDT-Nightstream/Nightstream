//! Strict bytes for option-B proofs with their witnesses.
//!
//! Owns the byte layout; acceptance still needs `verify`. The relations fix
//! every size, so the decoder checks the exact length before it allocates.
//! A proof has one encoding: field words below the modulus, evaluations with
//! exactly `D` coefficients, the fresh witness as a `+1` and a `-1` lane mask
//! per column (its commitment rejects other values), and each slot's
//! accumulator, whose witness lies outside {-1, 0, 1}, as field words.
//! Frame digests are not encoded; no check reads them.

use neo_ccs::Mat;
use neo_math::D;

use super::super::encoding::{Reader, Shape, Writer};
use super::super::ProofCodecError;
use super::{Envelope, OptionB, Slot};
use crate::folding::{pi_rlc, CcsClaim, CcsInstance, CcsWitness};
use nightstream_fprime::PUBLIC_WORDS;

const MAGIC: &[u8; 16] = b"NS-OPTIONB-PRF-1";
const WORD: usize = 8;
const KAPPA_WORDS: usize = 1188;
const MATRICES: usize = 4;

impl OptionB {
    fn codec_shape(&self, circuit: usize) -> Shape {
        Shape {
            public_width: PUBLIC_WORDS,
            witness_columns: self.relations[circuit].blocks(),
            matrices: MATRICES,
        }
    }

    /// Commitment, public projection, point, `Eval_K`, then `Eval_A`.
    fn claim_bytes() -> usize {
        WORD * (KAPPA_WORDS + PUBLIC_WORDS + 2 * 28 + (MATRICES + 1) * 2 * D)
    }

    /// The length of a proof of `pc` (0 before the first step).
    fn proof_length(&self, pc: u64) -> Option<usize> {
        let header = MAGIC.len() + WORD * 10;
        if pc == 0 {
            return Some(header);
        }
        let fresh = self.codec_shape(*[0, 1].get(pc as usize - 1)?);
        let slots: usize = (0..2)
            .map(|slot| WORD + Self::claim_bytes() + WORD * D * self.relations[slot].blocks())
            .sum();
        Some(header + slots + WORD * (KAPPA_WORDS + PUBLIC_WORDS) + 2 * WORD * fresh.witness_columns)
    }

    pub(crate) fn encode_proof(&self, envelope: &Envelope) -> Result<Vec<u8>, ProofCodecError> {
        let length = self
            .proof_length(envelope.pc)
            .ok_or(ProofCodecError("program counter"))?;
        let mut output = Writer(Vec::with_capacity(length));
        output.0.extend_from_slice(MAGIC);
        output.word(envelope.pc);
        output.word(envelope.iteration);
        output.fields(&envelope.z0);
        output.fields(&envelope.current);
        let Some(fresh) = &envelope.fresh else {
            return Ok(output.0);
        };
        for (index, slot) in envelope.slots.iter().enumerate() {
            output.word(slot.depth);
            output.claim(&slot.accumulator.claim, &self.codec_shape(index))?;
            let witness = &slot.accumulator.witness;
            if witness.rows() != D || witness.cols() != self.relations[index].blocks() {
                return Err(ProofCodecError("slot witness shape"));
            }
            output.fields(&witness.to_dense_vec());
        }
        output.commitment(&fresh.claim.c)?;
        if fresh.claim.x.len() != PUBLIC_WORDS {
            return Err(ProofCodecError("fresh public input shape"));
        }
        output.fields(&fresh.claim.x);
        output.witness(&fresh.witness.Z, &self.codec_shape((envelope.pc - 1) as usize))?;
        debug_assert_eq!(output.0.len(), length);
        Ok(output.0)
    }

    /// Decode untrusted proof bytes for these relations.
    pub(crate) fn decode_proof(&self, bytes: &[u8]) -> Result<Envelope, ProofCodecError> {
        let mut input = Reader(bytes);
        if input.take(MAGIC.len())? != MAGIC {
            return Err(ProofCodecError("format tag"));
        }
        let pc = input.word()?;
        if Some(bytes.len()) != self.proof_length(pc) {
            return Err(ProofCodecError("length differs from these relations' proof size"));
        }
        let iteration = input.word()?;
        let (z0, current) = (input.four()?, input.four()?);
        if pc == 0 {
            if iteration != 0 || z0 != current {
                return Err(ProofCodecError("an initial proof starts at its initial state"));
            }
            return self
                .initial(z0)
                .map_err(|_| ProofCodecError("initial slots"));
        }
        let mut slots = Vec::with_capacity(2);
        for index in 0..2 {
            let depth = input.word()?;
            let claim = input.claim(&self.codec_shape(index))?;
            let blocks = self.relations[index].blocks();
            let witness = Mat::from_row_major(D, blocks, input.fields(D * blocks)?);
            slots.push(Slot {
                accumulator: pi_rlc::Output { claim, witness },
                depth,
            });
        }
        let c = input.commitment()?;
        let x = input.fields(PUBLIC_WORDS)?;
        let witness = input.witness(&self.codec_shape((pc - 1) as usize))?;
        let [first, second]: [Slot; 2] = slots.try_into().map_err(|_| ProofCodecError("slots"))?;
        Ok(Envelope {
            iteration,
            z0,
            current,
            pc,
            slots: [first, second],
            fresh: Some(CcsInstance {
                claim: CcsClaim {
                    c,
                    x,
                    m_in: PUBLIC_WORDS,
                    adv: None,
                },
                witness: CcsWitness { Z: witness },
            }),
        })
    }
}

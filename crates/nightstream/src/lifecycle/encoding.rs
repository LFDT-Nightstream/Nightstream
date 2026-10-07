//! Strict bytes for selected Stage 1 proofs.
//!
//! Owns the proof byte layout. Does not own acceptance: a decoded proof still
//! needs `verify`. The prepared circuit fixes every size, so the decoder
//! checks the exact byte length before it allocates. Each proof has one
//! encoding: field words are below the modulus, evaluations carry exactly
//! `D` coefficients, and each witness column is a pair of disjoint `+1` and
//! `-1` lane masks. The fixed-key commitment already rejects witness values
//! outside `{-1, 0, 1}`, so the mask form keeps every proof that `verify`
//! can accept.

use neo_ajtai::nightstream_fprime_setup::{signed_unit_prefix_blocks, PRODUCTION_VERIFIER_ROWS};
use neo_ajtai::Commitment;
use neo_ccs::Mat;
use neo_math::{KExtensions, D, F, K};
use nightstream_fprime::{
    PI_CCS_V1_1_ROUND_COEFFICIENT_COUNT, PI_CCS_V1_1_ROUND_COUNT, PI_CCS_V1_1_SOURCE_COUNT, PI_DEC_V1_1_CHILD_COUNT,
};
use p3_field::{PrimeCharacteristicRing, PrimeField64};

use super::{FinalProof, PreparedLifecycle, Stage1Envelope, Stage1State};
use crate::folding::{pi_ccs, pi_rlc, CcsClaim, CcsInstance, CcsWitness, CeClaim, RunningInstance};

const MAGIC: &[u8; 16] = b"NS-STAGE1-PROOF1";
const FINAL_MAGIC: &[u8; 16] = b"NS-FINAL-PROOF01";
const INITIAL: u64 = 0;
const ACTIVE: u64 = 1;
const KAPPA: usize = PRODUCTION_VERIFIER_ROWS as usize;
const DIGEST_BYTES: usize = 32;
const WORD: usize = 8;

#[derive(Debug, thiserror::Error)]
#[error("selected proof bytes: {0}")]
pub struct ProofCodecError(&'static str);

/// Sizes that the prepared circuit fixes.
struct Shape {
    public_width: usize,
    witness_columns: usize,
    matrices: usize,
}

impl Shape {
    /// Commitment, public projection, point, `Eval_K`, `Eval_A`, then the digest.
    fn claim_bytes(&self) -> Option<usize> {
        let evaluations = self.matrices.checked_add(1)?.checked_mul(2 * D)?;
        D.checked_mul(KAPPA)?
            .checked_add(self.public_width)?
            .checked_add(2 * PI_CCS_V1_1_ROUND_COUNT)?
            .checked_add(evaluations)?
            .checked_mul(WORD)?
            .checked_add(DIGEST_BYTES)
    }

    fn witness_bytes(&self) -> Option<usize> {
        self.witness_columns.checked_mul(2 * WORD)
    }

    /// The complete length of one proof of `kind`, or `None` for an unknown kind.
    fn length(&self, kind: u64) -> Option<usize> {
        let header = MAGIC.len() + WORD;
        match kind {
            INITIAL => Some(header + 4 * WORD),
            ACTIVE => {
                let claim = self.claim_bytes()?;
                let witness = self.witness_bytes()?;
                let running = claim
                    .checked_mul(PI_DEC_V1_1_CHILD_COUNT + 1)?
                    .checked_add(witness.checked_mul(PI_DEC_V1_1_CHILD_COUNT)?)?;
                let fresh = (D * KAPPA + self.public_width)
                    .checked_mul(WORD)?
                    .checked_add(witness)?;
                (header + 9 * WORD).checked_add(running)?.checked_add(fresh)
            }
            _ => None,
        }
    }
}

impl PreparedLifecycle {
    fn proof_shape(&self) -> Result<Shape, ProofCodecError> {
        let public_width = self.package.logical_public_input_count();
        if public_width % D != 0 {
            return Err(ProofCodecError("public width is not whole ring elements"));
        }
        Ok(Shape {
            public_width,
            witness_columns: self.structure.m.div_ceil(D),
            matrices: self.structure.t(),
        })
    }

    /// Encode a proof of this circuit. A value outside the selected shape is
    /// an error, never a silently changed proof.
    pub(crate) fn encode_proof(&self, envelope: &Stage1Envelope) -> Result<Vec<u8>, ProofCodecError> {
        let shape = self.proof_shape()?;
        let kind = if envelope.is_initial() { INITIAL } else { ACTIVE };
        let length = shape
            .length(kind)
            .ok_or(ProofCodecError("proof size overflows"))?;
        let mut output = Writer(Vec::with_capacity(length));
        output.0.extend_from_slice(MAGIC);
        output.word(kind);
        let state = envelope.state();
        if kind == INITIAL {
            output.fields(&state.z0());
        } else {
            let running = envelope
                .running()
                .ok_or(ProofCodecError("missing running payload"))?;
            let fresh = envelope
                .fresh()
                .ok_or(ProofCodecError("missing fresh payload"))?;
            let parent = running
                .parent_authority
                .as_ref()
                .ok_or(ProofCodecError("missing PiRLC parent"))?;
            if running.claims.len() != PI_DEC_V1_1_CHILD_COUNT || running.witnesses.len() != PI_DEC_V1_1_CHILD_COUNT {
                return Err(ProofCodecError("running claim or witness count"));
            }
            output.word(state.iteration());
            output.fields(&state.z0());
            output.fields(&state.current());
            for claim in running.claims.iter().chain([parent]) {
                output.claim(claim, &shape)?;
            }
            for witness in &running.witnesses {
                output.witness(witness, &shape)?;
            }
            let claim = &fresh.claim;
            if claim.m_in != shape.public_width || claim.x.len() != shape.public_width || claim.adv.is_some() {
                return Err(ProofCodecError("fresh public input shape"));
            }
            if !fresh.witness.w.is_empty() {
                return Err(ProofCodecError("fresh private witness copy"));
            }
            output.commitment(&claim.c)?;
            output.fields(&claim.x);
            output.witness(&fresh.witness.Z, &shape)?;
        }
        debug_assert_eq!(output.0.len(), length);
        Ok(output.0)
    }

    /// Decode untrusted proof bytes for this circuit.
    pub(crate) fn decode_proof(&self, bytes: &[u8]) -> Result<Stage1Envelope, ProofCodecError> {
        let shape = self.proof_shape()?;
        let mut input = Reader(bytes);
        if input.take(MAGIC.len())? != MAGIC {
            return Err(ProofCodecError("format tag"));
        }
        let kind = input.word()?;
        let length = shape
            .length(kind)
            .ok_or(ProofCodecError("unknown proof kind"))?;
        if bytes.len() != length {
            return Err(ProofCodecError("length differs from this circuit's proof size"));
        }
        if kind == INITIAL {
            return Ok(Stage1Envelope::initial(input.four()?));
        }
        let state = Stage1State::new(input.word()?, input.four()?, input.four()?);
        let mut claims = (0..=PI_DEC_V1_1_CHILD_COUNT)
            .map(|_| input.claim(&shape))
            .collect::<Result<Vec<_>, _>>()?;
        let parent = claims.pop();
        let witnesses = (0..PI_DEC_V1_1_CHILD_COUNT)
            .map(|_| input.witness(&shape))
            .collect::<Result<Vec<_>, _>>()?;
        let c = input.commitment()?;
        let x = input.fields(shape.public_width)?;
        let fresh = CcsInstance {
            claim: CcsClaim {
                c,
                x,
                m_in: shape.public_width,
                adv: None,
            },
            witness: CcsWitness {
                w: Vec::new(),
                Z: input.witness(&shape)?,
            },
        };
        Ok(Stage1Envelope::from_parts(
            state,
            RunningInstance::new(claims, witnesses, parent),
            fresh,
        ))
    }
}

impl PreparedLifecycle {
    /// Encode a finished proof: state, the 16 running claims, the fresh
    /// commitment and public input, the PiCCS rounds and outputs, the PiRLC
    /// parent, then the length-prefixed layer-1 proof.
    pub(crate) fn encode_final_proof(&self, proof: &FinalProof) -> Result<Vec<u8>, ProofCodecError> {
        let shape = self.proof_shape()?;
        let mut output = Writer(Vec::new());
        output.0.extend_from_slice(FINAL_MAGIC);
        output.word(proof.state.iteration());
        output.fields(&proof.state.z0());
        output.fields(&proof.state.current());
        if proof.running.len() != PI_DEC_V1_1_CHILD_COUNT {
            return Err(ProofCodecError("running claim count"));
        }
        for claim in &proof.running {
            output.claim(claim, &shape)?;
        }
        let fresh = &proof.fresh;
        if fresh.m_in != shape.public_width || fresh.x.len() != shape.public_width || fresh.adv.is_some() {
            return Err(ProofCodecError("fresh public input shape"));
        }
        output.commitment(&fresh.c)?;
        output.fields(&fresh.x);
        let rounds = &proof.pi_ccs.sumcheck.sumcheck_rounds;
        if rounds.len() != PI_CCS_V1_1_ROUND_COUNT
            || rounds
                .iter()
                .any(|round| round.len() != PI_CCS_V1_1_ROUND_COEFFICIENT_COUNT)
        {
            return Err(ProofCodecError("PiCCS round shape"));
        }
        for value in rounds.iter().flatten() {
            output.fields(&value.as_coeffs());
        }
        if proof.pi_ccs.outputs.len() != PI_CCS_V1_1_SOURCE_COUNT {
            return Err(ProofCodecError("PiCCS output count"));
        }
        for claim in proof.pi_ccs.outputs.iter().chain([&proof.pi_rlc.combined]) {
            output.claim(claim, &shape)?;
        }
        let layer1 = proof.layer1.to_bytes();
        output.word(layer1.len() as u64);
        output.0.extend_from_slice(&layer1);
        Ok(output.0)
    }

    /// Decode untrusted finished-proof bytes for this circuit. Acceptance
    /// still needs `verify_final`.
    pub(crate) fn decode_final_proof(&self, bytes: &[u8]) -> Result<FinalProof, ProofCodecError> {
        let shape = self.proof_shape()?;
        let mut input = Reader(bytes);
        if input.take(FINAL_MAGIC.len())? != FINAL_MAGIC {
            return Err(ProofCodecError("format tag"));
        }
        let state = Stage1State::new(input.word()?, input.four()?, input.four()?);
        let running = (0..PI_DEC_V1_1_CHILD_COUNT)
            .map(|_| input.claim(&shape))
            .collect::<Result<Vec<_>, _>>()?;
        let fresh = CcsClaim {
            c: input.commitment()?,
            x: input.fields(shape.public_width)?,
            m_in: shape.public_width,
            adv: None,
        };
        let rounds = (0..PI_CCS_V1_1_ROUND_COUNT)
            .map(|_| {
                (0..PI_CCS_V1_1_ROUND_COEFFICIENT_COUNT)
                    .map(|_| input.extension())
                    .collect::<Result<Vec<_>, _>>()
            })
            .collect::<Result<Vec<_>, _>>()?;
        let outputs = (0..PI_CCS_V1_1_SOURCE_COUNT)
            .map(|_| input.claim(&shape))
            .collect::<Result<Vec<_>, _>>()?;
        let combined = input.claim(&shape)?;
        let length = input.word()?;
        if input.0.len() as u64 != length {
            return Err(ProofCodecError("layer-1 length differs from the remaining bytes"));
        }
        let layer1 = neo_spartan::Proof::from_bytes(input.0).map_err(|_| ProofCodecError("layer-1 proof"))?;
        Ok(FinalProof {
            state,
            running,
            fresh,
            pi_ccs: pi_ccs::Proof {
                sumcheck: pi_ccs::SumcheckProof::new(rounds),
                outputs,
            },
            pi_rlc: pi_rlc::Proof { combined },
            layer1,
        })
    }
}

struct Writer(Vec<u8>);

impl Writer {
    fn word(&mut self, value: u64) {
        self.0.extend_from_slice(&value.to_le_bytes());
    }

    fn fields(&mut self, values: &[F]) {
        values
            .iter()
            .for_each(|value| self.word(value.as_canonical_u64()));
    }

    /// Exactly `D` coefficients; the selected surplus must be zero.
    fn evaluations(&mut self, values: &[K]) -> Result<(), ProofCodecError> {
        if values.len() != D.next_power_of_two() || values[D..].iter().any(|value| *value != K::ZERO) {
            return Err(ProofCodecError("evaluation shape or nonzero surplus coefficients"));
        }
        for value in &values[..D] {
            self.fields(&value.as_coeffs());
        }
        Ok(())
    }

    fn commitment(&mut self, commitment: &Commitment) -> Result<(), ProofCodecError> {
        if commitment.d != D || commitment.kappa != KAPPA || commitment.data.len() != D * KAPPA {
            return Err(ProofCodecError("commitment shape"));
        }
        self.fields(&commitment.data);
        Ok(())
    }

    fn claim(&mut self, claim: &CeClaim, shape: &Shape) -> Result<(), ProofCodecError> {
        if claim.m_in != shape.public_width
            || claim.X.rows() != D
            || claim.X.cols() != shape.public_width / D
            || claim.r.len() != PI_CCS_V1_1_ROUND_COUNT
            || claim.eval_a.len() != shape.matrices
            || claim.adv.is_some()
        {
            return Err(ProofCodecError("running claim shape"));
        }
        self.commitment(&claim.c)?;
        self.fields(&claim.X.to_dense_vec());
        for value in &claim.r {
            self.fields(&value.as_coeffs());
        }
        self.evaluations(&claim.eval_k)?;
        for values in &claim.eval_a {
            self.evaluations(values)?;
        }
        self.0.extend_from_slice(&claim.fold_digest);
        Ok(())
    }

    /// One `+1` mask and one `-1` mask per column, from the commitment's own check.
    fn witness(&mut self, witness: &Mat<F>, shape: &Shape) -> Result<(), ProofCodecError> {
        if witness.rows() != D || witness.cols() != shape.witness_columns {
            return Err(ProofCodecError("witness shape"));
        }
        let blocks =
            signed_unit_prefix_blocks(witness).map_err(|_| ProofCodecError("witness values are not signed units"))?;
        let mut blocks = blocks.iter().peekable();
        for column in 0..shape.witness_columns as u64 {
            match blocks.next_if(|block| block.index() == column) {
                Some(block) => {
                    self.word(block.positive());
                    self.word(block.negative());
                }
                None => {
                    self.word(0);
                    self.word(0);
                }
            }
        }
        Ok(())
    }
}

struct Reader<'a>(&'a [u8]);

impl<'a> Reader<'a> {
    fn take(&mut self, length: usize) -> Result<&'a [u8], ProofCodecError> {
        if self.0.len() < length {
            return Err(ProofCodecError("truncated proof"));
        }
        let (head, tail) = self.0.split_at(length);
        self.0 = tail;
        Ok(head)
    }

    fn word(&mut self) -> Result<u64, ProofCodecError> {
        Ok(u64::from_le_bytes(self.take(WORD)?.try_into().expect("one word")))
    }

    fn field(&mut self) -> Result<F, ProofCodecError> {
        let word = self.word()?;
        if word >= F::ORDER_U64 {
            return Err(ProofCodecError("noncanonical field word"));
        }
        Ok(F::from_u64(word))
    }

    fn fields(&mut self, count: usize) -> Result<Vec<F>, ProofCodecError> {
        (0..count).map(|_| self.field()).collect()
    }

    fn four(&mut self) -> Result<[F; 4], ProofCodecError> {
        Ok([self.field()?, self.field()?, self.field()?, self.field()?])
    }

    fn extension(&mut self) -> Result<K, ProofCodecError> {
        Ok(K::from_coeffs([self.field()?, self.field()?]))
    }

    /// `D` coefficients, padded with zeros to the selected length.
    fn evaluations(&mut self) -> Result<Vec<K>, ProofCodecError> {
        let mut values = (0..D)
            .map(|_| self.extension())
            .collect::<Result<Vec<_>, _>>()?;
        values.resize(D.next_power_of_two(), K::ZERO);
        Ok(values)
    }

    fn commitment(&mut self) -> Result<Commitment, ProofCodecError> {
        Ok(Commitment {
            d: D,
            kappa: KAPPA,
            data: self.fields(D * KAPPA)?,
        })
    }

    fn claim(&mut self, shape: &Shape) -> Result<CeClaim, ProofCodecError> {
        let c = self.commitment()?;
        let projection = self.fields(shape.public_width)?;
        let r = (0..PI_CCS_V1_1_ROUND_COUNT)
            .map(|_| self.extension())
            .collect::<Result<Vec<_>, _>>()?;
        let eval_k = self.evaluations()?;
        let eval_a = (0..shape.matrices)
            .map(|_| self.evaluations())
            .collect::<Result<Vec<_>, _>>()?;
        let fold_digest = self.take(DIGEST_BYTES)?.try_into().expect("one digest");
        Ok(CeClaim {
            c,
            X: Mat::from_row_major(D, shape.public_width / D, projection),
            r,
            eval_k,
            eval_a,
            m_in: shape.public_width,
            fold_digest,
            adv: None,
        })
    }

    fn witness(&mut self, shape: &Shape) -> Result<Mat<F>, ProofCodecError> {
        let mut positive = Vec::with_capacity(shape.witness_columns);
        let mut negative = Vec::with_capacity(shape.witness_columns);
        for _ in 0..shape.witness_columns {
            positive.push(self.word()?);
            negative.push(self.word()?);
        }
        Mat::compact_signed_unit_from_column_masks(D, shape.witness_columns, &positive, &negative)
            .map_err(|_| ProofCodecError("witness column masks"))
    }
}

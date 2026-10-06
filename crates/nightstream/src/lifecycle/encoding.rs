//! Strict bytes for selected Stage 1 proofs.
//!
//! Owns the proof byte layout. Does not own acceptance: a decoded proof still
//! needs `verify`. The prepared circuit fixes every size, so the decoder
//! checks the exact byte length before it allocates. The bytes carry exactly
//! the values that `verify` reads: no frame digest and no private witness
//! copy `w`, which no reduction reads. Each proof has one
//! encoding: field words are below the modulus, evaluations carry exactly
//! `D` coefficients, and each witness column is a pair of disjoint `+1` and
//! `-1` lane masks. The fixed-key commitment already rejects witness values
//! outside `{-1, 0, 1}`, so the mask form keeps every proof that `verify`
//! can accept.

use neo_ajtai::nightstream_fprime_setup::{signed_unit_prefix_blocks, PRODUCTION_VERIFIER_ROWS};
use neo_ajtai::Commitment;
use neo_ccs::Mat;
use neo_math::{KExtensions, D, F, K};
use nightstream_fprime::{PI_CCS_V1_1_ROUND_COUNT, PI_DEC_V1_1_CHILD_COUNT};
use p3_field::{PrimeCharacteristicRing, PrimeField64};

use super::{PreparedLifecycle, Stage1Envelope, Stage1State};
use crate::folding::{
    has_canonical_evaluations, CcsClaim, CcsInstance, CcsWitness, CeClaim, RunningInstance, EVALUATION_WIDTH,
};

const MAGIC: &[u8; 16] = b"NS-STAGE1-PROOF1";
const INITIAL: u64 = 0;
const ACTIVE: u64 = 1;
const KAPPA: usize = PRODUCTION_VERIFIER_ROWS as usize;
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
    /// Commitment, public projection, point, `Eval_K`, then `Eval_A`.
    fn claim_bytes(&self) -> Option<usize> {
        let evaluations = self.matrices.checked_add(1)?.checked_mul(2 * D)?;
        D.checked_mul(KAPPA)?
            .checked_add(self.public_width)?
            .checked_add(2 * PI_CCS_V1_1_ROUND_COUNT)?
            .checked_add(evaluations)?
            .checked_mul(WORD)
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
                    .checked_mul(PI_DEC_V1_1_CHILD_COUNT)?
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
    /// an error, never a silently changed proof. Values that `verify` does not
    /// read are not encoded.
    pub(crate) fn encode_proof(&self, envelope: &Stage1Envelope) -> Result<Vec<u8>, ProofCodecError> {
        let shape = self.proof_shape()?;
        let active = envelope.active_parts();
        let kind = if active.is_some() { ACTIVE } else { INITIAL };
        let length = shape
            .length(kind)
            .ok_or(ProofCodecError("proof size overflows"))?;
        let mut output = Writer(Vec::with_capacity(length));
        output.0.extend_from_slice(MAGIC);
        output.word(kind);
        let state = envelope.state();
        let Some((running, fresh)) = active else {
            output.fields(&state.z0());
            return Ok(output.0);
        };
        if running.claims.len() != PI_DEC_V1_1_CHILD_COUNT || running.witnesses.len() != PI_DEC_V1_1_CHILD_COUNT {
            return Err(ProofCodecError("running claim or witness count"));
        }
        output.word(state.iteration());
        output.fields(&state.z0());
        output.fields(&state.current());
        for claim in &running.claims {
            output.claim(claim, &shape)?;
        }
        for witness in &running.witnesses {
            output.witness(witness, &shape)?;
        }
        let claim = &fresh.claim;
        if claim.m_in != shape.public_width || claim.x.len() != shape.public_width || claim.adv.is_some() {
            return Err(ProofCodecError("fresh public input shape"));
        }
        output.commitment(&claim.c)?;
        output.fields(&claim.x);
        output.witness(&fresh.witness.Z, &shape)?;
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
        let claims = (0..PI_DEC_V1_1_CHILD_COUNT)
            .map(|_| input.claim(&shape))
            .collect::<Result<Vec<_>, _>>()?;
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
        // The frame digests stay zero: no reduction reads them.
        Ok(Stage1Envelope::from_parts(
            state,
            RunningInstance::new(claims, witnesses),
            fresh,
        ))
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

    /// The `D` coefficients of a canonical evaluation vector.
    fn evaluations(&mut self, values: &[K]) {
        for value in &values[..D] {
            self.fields(&value.as_coeffs());
        }
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
            || claim.adv.is_some()
        {
            return Err(ProofCodecError("running claim shape"));
        }
        if !has_canonical_evaluations(claim, shape.matrices) {
            return Err(ProofCodecError("evaluation shape or nonzero surplus coefficients"));
        }
        self.commitment(&claim.c)?;
        self.fields(&claim.X.to_dense_vec());
        for value in &claim.r {
            self.fields(&value.as_coeffs());
        }
        self.evaluations(&claim.eval_k);
        for values in &claim.eval_a {
            self.evaluations(values);
        }
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
        values.resize(EVALUATION_WIDTH, K::ZERO);
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
        Ok(CeClaim {
            c,
            X: Mat::from_row_major(D, shape.public_width / D, projection),
            r,
            eval_k,
            eval_a,
            m_in: shape.public_width,
            fold_digest: [0; 32],
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

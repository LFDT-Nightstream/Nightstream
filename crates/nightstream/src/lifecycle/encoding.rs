//! Strict bytes of the claims and witnesses in an option-B proof (SPEC.md section 4.5).
//!
//! Owns the proof byte layout. Does not own acceptance: a decoded proof still
//! needs `verify`. The prepared circuit fixes every size, so the decoder
//! checks the exact byte length before it allocates. The bytes carry exactly
//! the values that `verify` reads, with no frame digest, which no check
//! reads. Each proof has one encoding: field words are below the modulus,
//! evaluations carry exactly
//! `D` coefficients, and each witness column is a pair of disjoint `+1` and
//! `-1` lane masks. The fixed-key commitment already rejects witness values
//! outside `{-1, 0, 1}`, so the mask form keeps every proof that `verify`
//! can accept.

use neo_ajtai::nightstream_fprime_setup::{signed_unit_prefix_blocks, PRODUCTION_VERIFIER_ROWS};
use neo_ajtai::Commitment;
use neo_ccs::Mat;
use neo_math::{KExtensions, D, F, K};
use nightstream_fprime::PI_CCS_V1_2_ROUND_COUNT;
use p3_field::{PrimeCharacteristicRing, PrimeField64};

use crate::folding::{has_canonical_evaluations, CeClaim, EVALUATION_WIDTH};

const KAPPA: usize = PRODUCTION_VERIFIER_ROWS as usize;
const WORD: usize = 8;

#[derive(Debug, thiserror::Error)]
#[error("selected proof bytes: {0}")]
pub struct ProofCodecError(pub(crate) &'static str);

/// Sizes that the prepared circuit fixes.
pub(super) struct Shape {
    pub(super) public_width: usize,
    pub(super) witness_columns: usize,
    pub(super) matrices: usize,
}

pub(super) struct Writer(pub(super) Vec<u8>);

impl Writer {
    pub(super) fn word(&mut self, value: u64) {
        self.0.extend_from_slice(&value.to_le_bytes());
    }

    pub(super) fn fields(&mut self, values: &[F]) {
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

    pub(super) fn commitment(&mut self, commitment: &Commitment) -> Result<(), ProofCodecError> {
        if commitment.d() != D || commitment.kappa() != KAPPA || commitment.data().len() != D * KAPPA {
            return Err(ProofCodecError("commitment shape"));
        }
        self.fields(commitment.data());
        Ok(())
    }

    pub(super) fn claim(&mut self, claim: &CeClaim, shape: &Shape) -> Result<(), ProofCodecError> {
        if claim.m_in != shape.public_width
            || claim.X.rows() != D
            || claim.X.cols() != shape.public_width / D
            || claim.r.len() != PI_CCS_V1_2_ROUND_COUNT
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
    pub(super) fn witness(&mut self, witness: &Mat<F>, shape: &Shape) -> Result<(), ProofCodecError> {
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

pub(super) struct Reader<'a>(pub(super) &'a [u8]);

impl<'a> Reader<'a> {
    pub(super) fn take(&mut self, length: usize) -> Result<&'a [u8], ProofCodecError> {
        if self.0.len() < length {
            return Err(ProofCodecError("truncated proof"));
        }
        let (head, tail) = self.0.split_at(length);
        self.0 = tail;
        Ok(head)
    }

    pub(super) fn word(&mut self) -> Result<u64, ProofCodecError> {
        Ok(u64::from_le_bytes(self.take(WORD)?.try_into().expect("one word")))
    }

    pub(super) fn field(&mut self) -> Result<F, ProofCodecError> {
        let word = self.word()?;
        if word >= F::ORDER_U64 {
            return Err(ProofCodecError("noncanonical field word"));
        }
        Ok(F::from_u64(word))
    }

    pub(super) fn fields(&mut self, count: usize) -> Result<Vec<F>, ProofCodecError> {
        (0..count).map(|_| self.field()).collect()
    }

    pub(super) fn four(&mut self) -> Result<[F; 4], ProofCodecError> {
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

    pub(super) fn commitment(&mut self) -> Result<Commitment, ProofCodecError> {
        Ok(Commitment::new(KAPPA, self.fields(D * KAPPA)?).expect("commitment shape"))
    }

    pub(super) fn claim(&mut self, shape: &Shape) -> Result<CeClaim, ProofCodecError> {
        let c = self.commitment()?;
        let projection = self.fields(shape.public_width)?;
        let r = (0..PI_CCS_V1_2_ROUND_COUNT)
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

    pub(super) fn witness(&mut self, shape: &Shape) -> Result<Mat<F>, ProofCodecError> {
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

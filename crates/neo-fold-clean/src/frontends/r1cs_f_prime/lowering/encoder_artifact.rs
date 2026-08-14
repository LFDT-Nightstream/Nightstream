//! Persistent witness-encoder state for an artifact-backed CCS relation.

use std::io::{Read, Write};
use std::sync::Arc;

use neo_math::{D, F};
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use thiserror::Error;

use crate::engine::r1cs_circuit::builder::ProductFactorTrace;
use crate::engine::r1cs_circuit::Lc;
use crate::paper::relations::Structure;

use super::{CompactIndex, CompactSlot, DerivedProductSumEncoding, MultiBranchLowNormR1cs};

const MAGIC: [u8; 8] = *b"NFENC003";
const SCHEMA_VERSION: u32 = 3;
const DIGEST_LEN: usize = 4;

const FIELD_NONE: u8 = 0;
const FIELD_DIRECT: u8 = 1;
const FIELD_DECOMPOSITION_ALIAS: u8 = 2;
const FIELD_EQUALITY_ALIAS: u8 = 3;

/// Failure while encoding or loading one verifier-owned witness encoder.
#[derive(Debug, Error)]
pub enum LowNormEncoderArtifactError {
    #[error("low-norm encoder artifact I/O failed: {0}")]
    Io(#[from] std::io::Error),
    #[error("low-norm encoder artifact is invalid: {0}")]
    Invalid(String),
}

/// Trusted identity of one persistent low-norm encoder artifact.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LowNormEncoderArtifactReceipt {
    artifact_bytes: u64,
    relation_shape: (u64, u64, u32),
    matrix_digest: [u64; DIGEST_LEN],
    encoder_digest: [u64; DIGEST_LEN],
    arm_field_counts: Vec<u64>,
    arm_derived_counts: Vec<u64>,
}

/// Deployment bounds checked before the encoder expands any artifact runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LowNormEncoderArtifactLimits {
    max_bytes: u64,
    max_rows: usize,
    max_cols: usize,
    max_matrices: usize,
    max_arms: usize,
    max_fields_per_arm: usize,
    max_derived_per_arm: usize,
}

impl LowNormEncoderArtifactLimits {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        max_bytes: u64,
        max_rows: usize,
        max_cols: usize,
        max_matrices: usize,
        max_arms: usize,
        max_fields_per_arm: usize,
        max_derived_per_arm: usize,
    ) -> Self {
        Self {
            max_bytes,
            max_rows,
            max_cols,
            max_matrices,
            max_arms,
            max_fields_per_arm,
            max_derived_per_arm,
        }
    }
}

impl LowNormEncoderArtifactReceipt {
    /// Rebuild a receipt from trusted profile metadata.
    pub fn from_parts(
        artifact_bytes: u64,
        relation_shape: (u64, u64, u32),
        matrix_digest: [u64; DIGEST_LEN],
        encoder_digest: [u64; DIGEST_LEN],
        arm_field_counts: Vec<u64>,
        arm_derived_counts: Vec<u64>,
    ) -> Result<Self, LowNormEncoderArtifactError> {
        if artifact_bytes == 0 || relation_shape.0 == 0 || relation_shape.1 == 0 || relation_shape.2 == 0 {
            return Err(invalid("receipt has an empty artifact or relation shape"));
        }
        if arm_field_counts.is_empty() || arm_field_counts.len() != arm_derived_counts.len() {
            return Err(invalid("receipt arm counts are empty or inconsistent"));
        }
        validate_digest(&matrix_digest, "matrix digest")?;
        validate_digest(&encoder_digest, "encoder digest")?;
        Ok(Self {
            artifact_bytes,
            relation_shape,
            matrix_digest,
            encoder_digest,
            arm_field_counts,
            arm_derived_counts,
        })
    }

    pub fn artifact_bytes(&self) -> u64 {
        self.artifact_bytes
    }

    pub fn relation_shape(&self) -> Result<(usize, usize, usize), LowNormEncoderArtifactError> {
        Ok((
            usize::try_from(self.relation_shape.0).map_err(|_| invalid("receipt row count exceeds usize"))?,
            usize::try_from(self.relation_shape.1).map_err(|_| invalid("receipt column count exceeds usize"))?,
            usize::try_from(self.relation_shape.2).map_err(|_| invalid("receipt matrix count exceeds usize"))?,
        ))
    }

    pub fn matrix_digest(&self) -> [u64; DIGEST_LEN] {
        self.matrix_digest
    }

    pub fn encoder_digest(&self) -> [u64; DIGEST_LEN] {
        self.encoder_digest
    }

    pub fn arm_field_counts(&self) -> &[u64] {
        &self.arm_field_counts
    }

    pub fn arm_derived_counts(&self) -> &[u64] {
        &self.arm_derived_counts
    }
}

/// Encoder state accepted under one trusted receipt.
pub struct VerifiedLowNormEncoderArtifact {
    receipt: LowNormEncoderArtifactReceipt,
    decoded: DecodedEncoder,
}

impl VerifiedLowNormEncoderArtifact {
    pub fn receipt(&self) -> &LowNormEncoderArtifactReceipt {
        &self.receipt
    }

    pub(crate) fn into_relation(
        self,
        structure: Structure,
    ) -> Result<MultiBranchLowNormR1cs, LowNormEncoderArtifactError> {
        if (structure.n, structure.m, structure.t()) != self.receipt.relation_shape()? {
            return Err(invalid("relation header differs from the encoder receipt"));
        }
        Ok(MultiBranchLowNormR1cs {
            structure: Arc::new(structure),
            public_input_len: self.decoded.public_input_len,
            selector_cols: self.decoded.selector_cols,
            public_field_count: self.decoded.public_field_count,
            arm_slots: self.decoded.arm_slots,
            arm_aliases: self.decoded.arm_aliases,
            arm_equal_aliases: self.decoded.arm_equal_aliases,
            arm_centered_columns: self.decoded.arm_centered_columns,
            arm_derived_product_sums: self.decoded.arm_derived_product_sums,
            selective_compiler_audit: None,
        })
    }
}

struct DecodedEncoder {
    public_input_len: usize,
    selector_cols: Vec<usize>,
    public_field_count: usize,
    arm_slots: Vec<Vec<CompactSlot>>,
    arm_aliases: Vec<Vec<CompactSlot>>,
    arm_equal_aliases: Vec<Vec<CompactIndex>>,
    arm_centered_columns: Vec<Vec<bool>>,
    arm_derived_product_sums: Vec<Vec<DerivedProductSumEncoding>>,
}

impl MultiBranchLowNormR1cs {
    /// Write the exact assignment encoder without matrix content.
    pub fn write_encoder_artifact(
        &self,
        mut writer: impl Write,
        matrix_digest: [F; DIGEST_LEN],
    ) -> Result<LowNormEncoderArtifactReceipt, LowNormEncoderArtifactError> {
        let matrix_digest = matrix_digest.map(|value| value.as_canonical_u64());
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&MAGIC);
        push_u32(&mut bytes, SCHEMA_VERSION);
        push_usize(&mut bytes, self.structure.n)?;
        push_usize(&mut bytes, self.structure.m)?;
        let matrix_count = u32::try_from(self.structure.t()).map_err(|_| invalid("matrix count exceeds u32"))?;
        push_u32(&mut bytes, matrix_count);
        for word in matrix_digest {
            push_u64(&mut bytes, word);
        }
        push_usize(&mut bytes, self.public_input_len)?;
        push_usize(&mut bytes, self.public_field_count)?;
        push_usize(&mut bytes, self.selector_cols.len())?;
        for &selector in &self.selector_cols {
            push_usize(&mut bytes, selector)?;
        }
        push_usize(&mut bytes, self.arm_slots.len())?;

        let arm_field_counts = self
            .arm_slots
            .iter()
            .map(|arm| arm.len() as u64)
            .collect::<Vec<_>>();
        let arm_derived_counts = self
            .arm_derived_product_sums
            .iter()
            .map(|arm| arm.len() as u64)
            .collect::<Vec<_>>();
        validate_parallel_arms(self)?;
        for arm in 0..self.arm_slots.len() {
            push_compact_usize(&mut bytes, self.arm_slots[arm].len(), "arm field count")?;
            #[cfg(feature = "perf-timers")]
            let fields_started = bytes.len();
            let _field_run_count = encode_field_runs(self, arm, &mut bytes)?;
            #[cfg(feature = "perf-timers")]
            let field_bytes = bytes.len() - fields_started;
            push_compact_usize(
                &mut bytes,
                self.arm_derived_product_sums[arm].len(),
                "derived-value count",
            )?;
            #[cfg(feature = "perf-timers")]
            let derived_started = bytes.len();
            for derived in &self.arm_derived_product_sums[arm] {
                encode_derived(derived, &mut bytes)?;
            }
            #[cfg(feature = "perf-timers")]
            eprintln!(
                "[low-norm-encoder-artifact] arm={arm} fields={} field_runs={} field_bytes={} derived={} derived_bytes={}",
                self.arm_slots[arm].len(),
                _field_run_count,
                field_bytes,
                self.arm_derived_product_sums[arm].len(),
                bytes.len() - derived_started,
            );
        }

        let artifact_bytes = bytes.len() as u64;
        let encoder_digest = digest_bytes(&bytes);
        let receipt = LowNormEncoderArtifactReceipt::from_parts(
            artifact_bytes,
            (self.structure.n as u64, self.structure.m as u64, matrix_count),
            matrix_digest,
            encoder_digest,
            arm_field_counts,
            arm_derived_counts,
        )?;
        writer.write_all(&bytes)?;
        Ok(receipt)
    }

    /// Load exact encoder state after checking a verifier-owned receipt.
    pub fn read_verified_encoder_artifact(
        reader: impl Read,
        receipt: &LowNormEncoderArtifactReceipt,
        limits: LowNormEncoderArtifactLimits,
    ) -> Result<VerifiedLowNormEncoderArtifact, LowNormEncoderArtifactError> {
        if receipt.artifact_bytes > limits.max_bytes {
            return Err(invalid("artifact exceeds the configured byte limit"));
        }
        let (rows, cols, matrices) = receipt.relation_shape()?;
        if rows > limits.max_rows || cols > limits.max_cols || matrices > limits.max_matrices {
            return Err(invalid("artifact relation shape exceeds its configured limits"));
        }
        if receipt.arm_field_counts.len() > limits.max_arms
            || receipt.arm_derived_counts.len() > limits.max_arms
            || receipt
                .arm_field_counts
                .iter()
                .any(|&count| count > limits.max_fields_per_arm as u64)
            || receipt
                .arm_derived_counts
                .iter()
                .any(|&count| count > limits.max_derived_per_arm as u64)
        {
            return Err(invalid("artifact encoder dimensions exceed their configured limits"));
        }
        let capacity = usize::try_from(receipt.artifact_bytes).map_err(|_| invalid("artifact size exceeds usize"))?;
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(capacity)
            .map_err(|_| invalid("artifact allocation failed"))?;
        reader
            .take(receipt.artifact_bytes.saturating_add(1))
            .read_to_end(&mut bytes)?;
        if bytes.len() as u64 != receipt.artifact_bytes {
            return Err(invalid("artifact byte length differs from the verifier receipt"));
        }
        if digest_bytes(&bytes) != receipt.encoder_digest {
            return Err(invalid("artifact content differs from the verifier receipt"));
        }
        let decoded = decode(&bytes, receipt)?;
        Ok(VerifiedLowNormEncoderArtifact {
            receipt: receipt.clone(),
            decoded,
        })
    }
}

fn validate_parallel_arms(relation: &MultiBranchLowNormR1cs) -> Result<(), LowNormEncoderArtifactError> {
    let arms = relation.arm_slots.len();
    if arms == 0
        || relation.arm_aliases.len() != arms
        || relation.arm_equal_aliases.len() != arms
        || relation.arm_centered_columns.len() != arms
        || relation.arm_derived_product_sums.len() != arms
        || relation.selector_cols.len() != arms
    {
        return Err(invalid("encoder arm arrays are empty or inconsistent"));
    }
    for arm in 0..arms {
        let fields = relation.arm_slots[arm].len();
        if relation.arm_aliases[arm].len() != fields
            || relation.arm_equal_aliases[arm].len() != fields
            || relation.arm_centered_columns[arm].len() != fields
        {
            return Err(invalid("encoder field arrays are inconsistent"));
        }
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FieldEncoding {
    None,
    Direct {
        start: usize,
        width: usize,
        centered: bool,
    },
    DecompositionAlias {
        source: usize,
        digit: usize,
        start: usize,
        centered: bool,
    },
    EqualityAlias {
        source: usize,
        start: usize,
        width: usize,
        centered: bool,
    },
}

#[derive(Clone, Copy, Debug)]
enum FieldStride {
    Direct {
        start: usize,
    },
    DecompositionAlias {
        source: usize,
        digit: usize,
        start: usize,
    },
    EqualityAlias {
        source: usize,
        start: usize,
    },
}

#[derive(Clone, Copy, Debug)]
struct FieldRun {
    first: FieldEncoding,
    stride: Option<FieldStride>,
    length: usize,
}

impl FieldRun {
    fn new(first: FieldEncoding) -> Self {
        Self {
            first,
            stride: None,
            length: 1,
        }
    }

    fn try_push(&mut self, next: FieldEncoding) -> bool {
        if self.first == FieldEncoding::None {
            if next == FieldEncoding::None {
                self.length += 1;
                return true;
            }
            return false;
        }
        if self.length == 1 {
            let Some(stride) = FieldStride::between(self.first, next) else {
                return false;
            };
            self.stride = Some(stride);
            self.length = 2;
            return true;
        }
        if self
            .stride
            .is_some_and(|stride| stride.matches(self.first, next, self.length))
        {
            self.length += 1;
            return true;
        }
        false
    }
}

impl FieldStride {
    fn between(first: FieldEncoding, next: FieldEncoding) -> Option<Self> {
        match (first, next) {
            (
                FieldEncoding::Direct { start, width, centered },
                FieldEncoding::Direct {
                    start: next_start,
                    width: next_width,
                    centered: next_centered,
                },
            ) if width == next_width && centered == next_centered => Some(Self::Direct {
                start: next_start.checked_sub(start)?,
            }),
            (
                FieldEncoding::DecompositionAlias {
                    source,
                    digit,
                    start,
                    centered,
                },
                FieldEncoding::DecompositionAlias {
                    source: next_source,
                    digit: next_digit,
                    start: next_start,
                    centered: next_centered,
                },
            ) if centered == next_centered => Some(Self::DecompositionAlias {
                source: next_source.checked_sub(source)?,
                digit: next_digit.checked_sub(digit)?,
                start: next_start.checked_sub(start)?,
            }),
            (
                FieldEncoding::EqualityAlias {
                    source,
                    start,
                    width,
                    centered,
                },
                FieldEncoding::EqualityAlias {
                    source: next_source,
                    start: next_start,
                    width: next_width,
                    centered: next_centered,
                },
            ) if width == next_width && centered == next_centered => Some(Self::EqualityAlias {
                source: next_source.checked_sub(source)?,
                start: next_start.checked_sub(start)?,
            }),
            _ => None,
        }
    }

    fn matches(self, first: FieldEncoding, next: FieldEncoding, offset: usize) -> bool {
        self.encoding_at(first, offset) == Some(next)
    }

    fn encoding_at(self, first: FieldEncoding, offset: usize) -> Option<FieldEncoding> {
        let affine = |value: usize, stride: usize| value.checked_add(stride.checked_mul(offset)?);
        match (first, self) {
            (FieldEncoding::Direct { start, width, centered }, Self::Direct { start: stride }) => {
                Some(FieldEncoding::Direct {
                    start: affine(start, stride)?,
                    width,
                    centered,
                })
            }
            (
                FieldEncoding::DecompositionAlias {
                    source,
                    digit,
                    start,
                    centered,
                },
                Self::DecompositionAlias {
                    source: source_stride,
                    digit: digit_stride,
                    start: start_stride,
                },
            ) => Some(FieldEncoding::DecompositionAlias {
                source: affine(source, source_stride)?,
                digit: affine(digit, digit_stride)?,
                start: affine(start, start_stride)?,
                centered,
            }),
            (
                FieldEncoding::EqualityAlias {
                    source,
                    start,
                    width,
                    centered,
                },
                Self::EqualityAlias {
                    source: source_stride,
                    start: start_stride,
                },
            ) => Some(FieldEncoding::EqualityAlias {
                source: affine(source, source_stride)?,
                start: affine(start, start_stride)?,
                width,
                centered,
            }),
            _ => None,
        }
    }
}

fn encode_field_runs(
    relation: &MultiBranchLowNormR1cs,
    arm: usize,
    bytes: &mut Vec<u8>,
) -> Result<usize, LowNormEncoderArtifactError> {
    let fields = relation.arm_slots[arm].len();
    if fields == 0 {
        return Err(invalid("encoder arm has no fields"));
    }
    let count_offset = bytes.len();
    push_u32(bytes, 0);
    let mut run_count = 0usize;
    let mut current = FieldRun::new(field_encoding(relation, arm, 0)?);
    for column in 1..fields {
        let next = field_encoding(relation, arm, column)?;
        if !current.try_push(next) {
            encode_field_run(&current, bytes)?;
            run_count += 1;
            current = FieldRun::new(next);
        }
    }
    encode_field_run(&current, bytes)?;
    run_count += 1;
    let compact_count = u32::try_from(run_count).map_err(|_| invalid("field-run count exceeds u32"))?;
    bytes[count_offset..count_offset + 4].copy_from_slice(&compact_count.to_le_bytes());
    Ok(run_count)
}

fn field_encoding(
    relation: &MultiBranchLowNormR1cs,
    arm: usize,
    column: usize,
) -> Result<FieldEncoding, LowNormEncoderArtifactError> {
    let slot = relation.arm_slots[arm][column].get();
    let alias = relation.arm_aliases[arm][column].get();
    let equality = relation.arm_equal_aliases[arm][column].get();
    let centered = relation.arm_centered_columns[arm][column];
    if alias.is_some() && equality.is_some() {
        return Err(invalid("one encoder field has two alias owners"));
    }
    Ok(match (slot, alias, equality) {
        (None, None, None) => FieldEncoding::None,
        (Some((start, width)), None, None) => FieldEncoding::Direct { start, width, centered },
        (Some((start, width)), Some((source, digit)), None) => {
            if width != 1 {
                return Err(invalid("decomposition alias does not occupy one coordinate"));
            }
            FieldEncoding::DecompositionAlias {
                source,
                digit,
                start,
                centered,
            }
        }
        (Some((start, width)), None, Some(source)) => FieldEncoding::EqualityAlias {
            source,
            start,
            width,
            centered,
        },
        _ => return Err(invalid("encoder alias omits its final slot")),
    })
}

fn encode_field_run(run: &FieldRun, bytes: &mut Vec<u8>) -> Result<(), LowNormEncoderArtifactError> {
    push_compact_usize(bytes, run.length, "field-run length")?;
    match run.first {
        FieldEncoding::None => bytes.push(FIELD_NONE),
        FieldEncoding::Direct { start, width, centered } => {
            let stride = match run.stride {
                Some(FieldStride::Direct { start }) => start,
                None if run.length == 1 => 0,
                _ => return Err(invalid("direct field run has an invalid stride")),
            };
            bytes.push(FIELD_DIRECT);
            push_compact_usize(bytes, start, "direct field start")?;
            push_compact_usize(bytes, stride, "direct field stride")?;
            push_compact_width(bytes, width, "direct field width")?;
            bytes.push(u8::from(centered));
        }
        FieldEncoding::DecompositionAlias {
            source,
            digit,
            start,
            centered,
        } => {
            let (source_stride, digit_stride, start_stride) = match run.stride {
                Some(FieldStride::DecompositionAlias { source, digit, start }) => (source, digit, start),
                None if run.length == 1 => (0, 0, 0),
                _ => return Err(invalid("decomposition field run has an invalid stride")),
            };
            bytes.push(FIELD_DECOMPOSITION_ALIAS);
            push_compact_usize(bytes, source, "decomposition source")?;
            push_compact_usize(bytes, source_stride, "decomposition source stride")?;
            push_compact_usize(bytes, digit, "decomposition digit")?;
            push_compact_usize(bytes, digit_stride, "decomposition digit stride")?;
            push_compact_usize(bytes, start, "decomposition start")?;
            push_compact_usize(bytes, start_stride, "decomposition start stride")?;
            bytes.push(u8::from(centered));
        }
        FieldEncoding::EqualityAlias {
            source,
            start,
            width,
            centered,
        } => {
            let (source_stride, start_stride) = match run.stride {
                Some(FieldStride::EqualityAlias { source, start }) => (source, start),
                None if run.length == 1 => (0, 0),
                _ => return Err(invalid("equality field run has an invalid stride")),
            };
            bytes.push(FIELD_EQUALITY_ALIAS);
            push_compact_usize(bytes, source, "equality source")?;
            push_compact_usize(bytes, source_stride, "equality source stride")?;
            push_compact_usize(bytes, start, "equality start")?;
            push_compact_usize(bytes, start_stride, "equality start stride")?;
            push_compact_width(bytes, width, "equality width")?;
            bytes.push(u8::from(centered));
        }
    }
    Ok(())
}

fn encode_derived(derived: &DerivedProductSumEncoding, bytes: &mut Vec<u8>) -> Result<(), LowNormEncoderArtifactError> {
    push_compact_usize(bytes, derived.slot.0, "derived slot start")?;
    push_compact_width(bytes, derived.slot.1, "derived slot width")?;
    let previous = derived
        .previous
        .map(|index| {
            u32::try_from(index)
                .map_err(|_| invalid("derived predecessor exceeds u32"))?
                .checked_add(1)
                .ok_or_else(|| invalid("derived predecessor encoding overflows"))
        })
        .transpose()?
        .unwrap_or(0);
    push_u32(bytes, previous);
    push_compact_usize(bytes, derived.factors.len(), "derived factor count")?;
    for factor in &derived.factors {
        push_field(bytes, factor.coefficient);
        encode_lc(bytes, &factor.left)?;
        encode_lc(bytes, &factor.right)?;
    }
    Ok(())
}

fn encode_lc(bytes: &mut Vec<u8>, lc: &Lc) -> Result<(), LowNormEncoderArtifactError> {
    push_field(bytes, lc.constant);
    push_compact_usize(bytes, lc.terms.len(), "linear-combination term count")?;
    for &(column, coefficient) in &lc.terms {
        push_compact_usize(bytes, column, "linear-combination column")?;
        push_field(bytes, coefficient);
    }
    Ok(())
}

fn decode(
    bytes: &[u8],
    receipt: &LowNormEncoderArtifactReceipt,
) -> Result<DecodedEncoder, LowNormEncoderArtifactError> {
    let mut decoder = Decoder::new(bytes);
    if decoder.array::<8>()? != MAGIC {
        return Err(invalid("artifact magic differs"));
    }
    if decoder.u32()? != SCHEMA_VERSION {
        return Err(invalid("artifact schema differs"));
    }
    let relation_shape = (decoder.usize()?, decoder.usize()?, decoder.u32()? as usize);
    if relation_shape != receipt.relation_shape()? {
        return Err(invalid("artifact relation shape differs from the verifier receipt"));
    }
    let mut matrix_digest = [0u64; DIGEST_LEN];
    for word in &mut matrix_digest {
        *word = decoder.u64()?;
    }
    if matrix_digest != receipt.matrix_digest {
        return Err(invalid("artifact matrix digest differs from the verifier receipt"));
    }
    let public_input_len = decoder.usize()?;
    let public_field_count = decoder.usize()?;
    if public_input_len > relation_shape.1 || public_input_len % D != 0 {
        return Err(invalid("artifact public input width is invalid"));
    }
    let selector_count = decoder.len(8)?;
    if selector_count != receipt.arm_field_counts.len() {
        return Err(invalid("selector count differs from the encoder arm count"));
    }
    let mut selector_cols = Vec::with_capacity(selector_count);
    for _ in 0..selector_count {
        let selector = decoder.usize()?;
        if selector >= relation_shape.1 {
            return Err(invalid("selector lies outside the relation assignment"));
        }
        selector_cols.push(selector);
    }
    let arm_count = decoder.usize()?;
    if arm_count != receipt.arm_field_counts.len() || arm_count != receipt.arm_derived_counts.len() {
        return Err(invalid("artifact arm count differs from the verifier receipt"));
    }

    let mut arm_slots = Vec::with_capacity(arm_count);
    let mut arm_aliases = Vec::with_capacity(arm_count);
    let mut arm_equal_aliases = Vec::with_capacity(arm_count);
    let mut arm_centered_columns = Vec::with_capacity(arm_count);
    let mut arm_derived_product_sums = Vec::with_capacity(arm_count);
    for arm in 0..arm_count {
        let fields = decoder.compact_usize()?;
        if fields as u64 != receipt.arm_field_counts[arm] || public_field_count > fields {
            return Err(invalid("artifact field count differs from the verifier receipt"));
        }
        let mut slots = Vec::with_capacity(fields);
        let mut aliases = Vec::with_capacity(fields);
        let mut equal_aliases = Vec::with_capacity(fields);
        let mut centered_columns = Vec::with_capacity(fields);
        let run_count = decoder.compact_len(5)?;
        let mut column = 0usize;
        for _ in 0..run_count {
            let run = decode_field_run(&mut decoder)?;
            let end = column
                .checked_add(run.length)
                .filter(|&end| end <= fields)
                .ok_or_else(|| invalid("field runs exceed the encoder arm"))?;
            for offset in 0..run.length {
                let encoding = match (run.first, run.stride) {
                    (FieldEncoding::None, None) => FieldEncoding::None,
                    (first, Some(stride)) => stride
                        .encoding_at(first, offset)
                        .ok_or_else(|| invalid("field run affine value overflows"))?,
                    (first, None) if run.length == 1 => first,
                    _ => return Err(invalid("field run is missing its affine stride")),
                };
                let field = decode_field_encoding(encoding, column + offset, relation_shape.1, &slots)?;
                slots.push(CompactSlot::from_option(field.slot));
                aliases.push(CompactSlot::from_option(field.alias));
                equal_aliases.push(CompactIndex::from_option(field.equality));
                centered_columns.push(field.centered);
            }
            column = end;
        }
        if column != fields {
            return Err(invalid("field runs do not partition the encoder arm"));
        }
        arm_slots.push(slots);
        arm_aliases.push(aliases);
        arm_equal_aliases.push(equal_aliases);
        arm_centered_columns.push(centered_columns);

        let derived_count = decoder.compact_usize()?;
        if derived_count as u64 != receipt.arm_derived_counts[arm] {
            return Err(invalid(
                "artifact derived-value count differs from the verifier receipt",
            ));
        }
        let mut derived = Vec::with_capacity(derived_count);
        for index in 0..derived_count {
            derived.push(decode_derived(&mut decoder, fields, relation_shape.1, index)?);
        }
        arm_derived_product_sums.push(derived);
    }
    decoder.finish()?;
    Ok(DecodedEncoder {
        public_input_len,
        selector_cols,
        public_field_count,
        arm_slots,
        arm_aliases,
        arm_equal_aliases,
        arm_centered_columns,
        arm_derived_product_sums,
    })
}

struct DecodedField {
    slot: Option<(usize, usize)>,
    alias: Option<(usize, usize)>,
    equality: Option<usize>,
    centered: bool,
}

fn decode_field_run(decoder: &mut Decoder<'_>) -> Result<FieldRun, LowNormEncoderArtifactError> {
    let length = decoder.compact_usize()?;
    if length == 0 {
        return Err(invalid("field run is empty"));
    }
    let tag = decoder.u8()?;
    let (first, stride) = match tag {
        FIELD_NONE => (FieldEncoding::None, None),
        FIELD_DIRECT => {
            let start = decoder.compact_usize()?;
            let start_stride = decoder.compact_usize()?;
            let width = decoder.compact_width()?;
            let centered = decoder.bool()?;
            (
                FieldEncoding::Direct { start, width, centered },
                Some(FieldStride::Direct { start: start_stride }),
            )
        }
        FIELD_DECOMPOSITION_ALIAS => {
            let source = decoder.compact_usize()?;
            let source_stride = decoder.compact_usize()?;
            let digit = decoder.compact_usize()?;
            let digit_stride = decoder.compact_usize()?;
            let start = decoder.compact_usize()?;
            let start_stride = decoder.compact_usize()?;
            let centered = decoder.bool()?;
            (
                FieldEncoding::DecompositionAlias {
                    source,
                    digit,
                    start,
                    centered,
                },
                Some(FieldStride::DecompositionAlias {
                    source: source_stride,
                    digit: digit_stride,
                    start: start_stride,
                }),
            )
        }
        FIELD_EQUALITY_ALIAS => {
            let source = decoder.compact_usize()?;
            let source_stride = decoder.compact_usize()?;
            let start = decoder.compact_usize()?;
            let start_stride = decoder.compact_usize()?;
            let width = decoder.compact_width()?;
            let centered = decoder.bool()?;
            (
                FieldEncoding::EqualityAlias {
                    source,
                    start,
                    width,
                    centered,
                },
                Some(FieldStride::EqualityAlias {
                    source: source_stride,
                    start: start_stride,
                }),
            )
        }
        _ => return Err(invalid("artifact contains an unknown field-run tag")),
    };
    Ok(FieldRun { first, stride, length })
}

fn decode_field_encoding(
    encoding: FieldEncoding,
    column: usize,
    assignment_width: usize,
    prior_slots: &[CompactSlot],
) -> Result<DecodedField, LowNormEncoderArtifactError> {
    let mut field = DecodedField {
        slot: None,
        alias: None,
        equality: None,
        centered: false,
    };
    match encoding {
        FieldEncoding::None => {}
        FieldEncoding::Direct { start, width, centered } => {
            field.slot = Some(validate_slot(start, width, assignment_width)?);
            field.centered = centered;
        }
        FieldEncoding::DecompositionAlias {
            source,
            digit,
            start,
            centered,
        } => {
            if source >= column {
                return Err(invalid("decomposition alias source is not earlier than its field"));
            }
            let source_slot = prior_slots[source]
                .get()
                .ok_or_else(|| invalid("decomposition alias source has no encoder slot"))?;
            if digit >= source_slot.1 || source_slot.0.checked_add(digit) != Some(start) {
                return Err(invalid("decomposition alias does not select its source coordinate"));
            }
            field.slot = Some(validate_slot(start, 1, assignment_width)?);
            field.alias = Some((source, digit));
            field.centered = centered;
        }
        FieldEncoding::EqualityAlias {
            source,
            start,
            width,
            centered,
        } => {
            if source >= column {
                return Err(invalid("equality alias source is not earlier than its field"));
            }
            let slot = validate_slot(start, width, assignment_width)?;
            if prior_slots[source].get() != Some(slot) {
                return Err(invalid("equality alias does not reuse its source slot"));
            }
            field.slot = Some(slot);
            field.equality = Some(source);
            field.centered = centered;
        }
    }
    if field.centered && field.slot.is_some_and(|slot| slot.1 != 1) {
        return Err(invalid("centered field does not occupy one coordinate"));
    }
    Ok(field)
}

fn decode_derived(
    decoder: &mut Decoder<'_>,
    source_fields: usize,
    assignment_width: usize,
    derived_index: usize,
) -> Result<DerivedProductSumEncoding, LowNormEncoderArtifactError> {
    let start = decoder.compact_usize()?;
    let width = decoder.compact_width()?;
    let slot = validate_slot(start, width, assignment_width)?;
    let previous = match decoder.u32()? {
        0 => None,
        value => {
            let index = (value - 1) as usize;
            if index >= derived_index {
                return Err(invalid("derived predecessor is not earlier than its value"));
            }
            Some(index)
        }
    };
    let factor_count = decoder.compact_len(32)?;
    let mut factors = Vec::with_capacity(factor_count);
    for _ in 0..factor_count {
        factors.push(ProductFactorTrace {
            coefficient: decoder.field()?,
            left: decode_lc(decoder, source_fields)?,
            right: decode_lc(decoder, source_fields)?,
        });
    }
    Ok(DerivedProductSumEncoding {
        slot,
        factors,
        previous,
    })
}

fn decode_lc(decoder: &mut Decoder<'_>, source_fields: usize) -> Result<Lc, LowNormEncoderArtifactError> {
    let constant = decoder.field()?;
    let term_count = decoder.compact_len(12)?;
    let mut terms = Vec::with_capacity(term_count);
    for _ in 0..term_count {
        let column = decoder.compact_usize()?;
        if column >= source_fields {
            return Err(invalid("linear-combination column exceeds its source arm"));
        }
        terms.push((column, decoder.field()?));
    }
    Ok(Lc { terms, constant })
}

fn validate_slot(
    start: usize,
    width: usize,
    assignment_width: usize,
) -> Result<(usize, usize), LowNormEncoderArtifactError> {
    if width == 0
        || start
            .checked_add(width)
            .is_none_or(|end| end > assignment_width)
    {
        return Err(invalid("encoder slot lies outside the relation assignment"));
    }
    Ok((start, width))
}

fn digest_bytes(bytes: &[u8]) -> [u64; DIGEST_LEN] {
    neo_ccs::crypto::poseidon2_goldilocks::poseidon2_hash_packed_bytes(bytes).map(|value| value.as_canonical_u64())
}

fn validate_digest(words: &[u64; DIGEST_LEN], label: &str) -> Result<(), LowNormEncoderArtifactError> {
    if words.iter().any(|&word| word >= F::ORDER_U64) {
        return Err(invalid(format!("{label} contains a noncanonical field word")));
    }
    Ok(())
}

fn invalid(message: impl Into<String>) -> LowNormEncoderArtifactError {
    LowNormEncoderArtifactError::Invalid(message.into())
}

fn push_field(bytes: &mut Vec<u8>, value: F) {
    push_u64(bytes, value.as_canonical_u64());
}

fn push_usize(bytes: &mut Vec<u8>, value: usize) -> Result<(), LowNormEncoderArtifactError> {
    push_u64(
        bytes,
        u64::try_from(value).map_err(|_| invalid("encoder value exceeds u64"))?,
    );
    Ok(())
}

fn push_compact_usize(bytes: &mut Vec<u8>, value: usize, label: &str) -> Result<(), LowNormEncoderArtifactError> {
    push_u32(
        bytes,
        u32::try_from(value).map_err(|_| invalid(format!("{label} exceeds u32")))?,
    );
    Ok(())
}

fn push_compact_width(bytes: &mut Vec<u8>, value: usize, label: &str) -> Result<(), LowNormEncoderArtifactError> {
    push_u16(
        bytes,
        u16::try_from(value).map_err(|_| invalid(format!("{label} exceeds u16")))?,
    );
    Ok(())
}

fn push_u64(bytes: &mut Vec<u8>, value: u64) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn push_u32(bytes: &mut Vec<u8>, value: u32) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn push_u16(bytes: &mut Vec<u8>, value: u16) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

struct Decoder<'a> {
    bytes: &'a [u8],
    cursor: usize,
}

impl<'a> Decoder<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, cursor: 0 }
    }

    fn array<const N: usize>(&mut self) -> Result<[u8; N], LowNormEncoderArtifactError> {
        let end = self
            .cursor
            .checked_add(N)
            .filter(|&end| end <= self.bytes.len())
            .ok_or_else(|| invalid("artifact is truncated"))?;
        let value = self.bytes[self.cursor..end]
            .try_into()
            .expect("checked fixed-size slice");
        self.cursor = end;
        Ok(value)
    }

    fn u8(&mut self) -> Result<u8, LowNormEncoderArtifactError> {
        Ok(self.array::<1>()?[0])
    }

    fn bool(&mut self) -> Result<bool, LowNormEncoderArtifactError> {
        match self.u8()? {
            0 => Ok(false),
            1 => Ok(true),
            _ => Err(invalid("artifact boolean is not zero or one")),
        }
    }

    fn u32(&mut self) -> Result<u32, LowNormEncoderArtifactError> {
        Ok(u32::from_le_bytes(self.array()?))
    }

    fn u16(&mut self) -> Result<u16, LowNormEncoderArtifactError> {
        Ok(u16::from_le_bytes(self.array()?))
    }

    fn u64(&mut self) -> Result<u64, LowNormEncoderArtifactError> {
        Ok(u64::from_le_bytes(self.array()?))
    }

    fn usize(&mut self) -> Result<usize, LowNormEncoderArtifactError> {
        usize::try_from(self.u64()?).map_err(|_| invalid("artifact value exceeds usize"))
    }

    fn compact_usize(&mut self) -> Result<usize, LowNormEncoderArtifactError> {
        Ok(self.u32()? as usize)
    }

    fn compact_width(&mut self) -> Result<usize, LowNormEncoderArtifactError> {
        Ok(self.u16()? as usize)
    }

    fn field(&mut self) -> Result<F, LowNormEncoderArtifactError> {
        let value = self.u64()?;
        if value >= F::ORDER_U64 {
            return Err(invalid("artifact field word is not canonical"));
        }
        Ok(F::from_u64(value))
    }

    fn len(&mut self, minimum_item_bytes: usize) -> Result<usize, LowNormEncoderArtifactError> {
        let count = self.usize()?;
        if count > self.remaining() / minimum_item_bytes.max(1) {
            return Err(invalid("artifact count exceeds its remaining bytes"));
        }
        Ok(count)
    }

    fn compact_len(&mut self, minimum_item_bytes: usize) -> Result<usize, LowNormEncoderArtifactError> {
        let count = self.compact_usize()?;
        if count > self.remaining() / minimum_item_bytes.max(1) {
            return Err(invalid("artifact count exceeds its remaining bytes"));
        }
        Ok(count)
    }

    fn remaining(&self) -> usize {
        self.bytes.len() - self.cursor
    }

    fn finish(self) -> Result<(), LowNormEncoderArtifactError> {
        if self.cursor != self.bytes.len() {
            return Err(invalid("artifact has trailing bytes"));
        }
        Ok(())
    }
}

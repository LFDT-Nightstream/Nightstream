//! Gadget provenance needed by the low-norm `enc(F')` compiler.
//!
//! The field R1CS remains the semantic authority. This trace only records
//! which consecutive R1CS rows came from algebraically stronger gadgets, so
//! the encoder can replace their temporary product wires with exact CCS gates.

use std::ops::Range;

use super::builder::{Lc, Var};

/// One Poseidon2 `x -> x^7` expansion.
#[derive(Clone, Debug)]
pub struct Sbox7TraceEntry {
    pub input: Lc,
    pub intermediates: [Var; 3],
    pub output: Var,
    pub source_rows: Range<usize>,
}

/// One complete WIDTH-8 Poseidon2 permutation.
#[derive(Clone, Debug)]
pub struct PoseidonPermutationTraceEntry {
    pub source_rows: Range<usize>,
}

/// One variable-length Poseidon2 hash and its nested permutation range.
#[derive(Clone, Debug)]
pub struct PoseidonHashTraceEntry {
    pub input_len: usize,
    pub permutation_range: Range<usize>,
    pub source_rows: Range<usize>,
}

/// One multiplication in `K = F[X]/(X^2 - W)`.
#[derive(Clone, Debug)]
pub struct KMulTraceEntry {
    pub a: [Lc; 2],
    pub b: [Lc; 2],
    pub intermediates: [Var; 3],
    pub output: [Var; 2],
    pub source_rows: Range<usize>,
}

/// One length-18 schoolbook convolution used by the 3-way ring product.
#[derive(Clone, Debug)]
pub struct Toom3ConvolutionTrace {
    pub lhs: Vec<Lc>,
    pub rhs: Vec<Lc>,
    /// Row-major `lhs[i] * rhs[j]` product wires.
    pub products: Vec<Var>,
}

/// One complete production Toom-3 ring multiplication.
#[derive(Clone, Debug)]
pub struct RingMulToom3TraceEntry {
    pub rho: Vec<Var>,
    pub c: Vec<Var>,
    pub convolutions: Vec<Toom3ConvolutionTrace>,
    /// Reduced output expressions in terms of the product wires.
    pub reduced_output_lcs: Vec<Lc>,
    pub output: Vec<Var>,
    pub source_rows: Range<usize>,
}

/// Start of one named, sequential circuit-emission stage.
///
/// Checkpoints are diagnostic provenance only. The following checkpoint ends
/// the stage; the final `complete` checkpoint closes the last stage.
#[derive(Clone, Debug)]
pub struct R1csStageCheckpoint {
    pub label: &'static str,
    pub row: usize,
    pub col: usize,
}

/// Append-only high-level provenance for one R1CS emission.
#[derive(Clone, Debug, Default)]
pub struct R1csEncodingTrace {
    sbox7: Vec<Sbox7TraceEntry>,
    poseidon_permutations: Vec<PoseidonPermutationTraceEntry>,
    poseidon_hashes: Vec<PoseidonHashTraceEntry>,
    k_muls: Vec<KMulTraceEntry>,
    ring_muls_toom3: Vec<RingMulToom3TraceEntry>,
    stages: Vec<R1csStageCheckpoint>,
}

impl R1csEncodingTrace {
    pub fn sbox7(&self) -> &[Sbox7TraceEntry] {
        &self.sbox7
    }

    pub fn k_muls(&self) -> &[KMulTraceEntry] {
        &self.k_muls
    }

    pub fn poseidon_permutations(&self) -> &[PoseidonPermutationTraceEntry] {
        &self.poseidon_permutations
    }

    pub fn poseidon_hashes(&self) -> &[PoseidonHashTraceEntry] {
        &self.poseidon_hashes
    }

    pub fn ring_muls_toom3(&self) -> &[RingMulToom3TraceEntry] {
        &self.ring_muls_toom3
    }

    pub fn stages(&self) -> &[R1csStageCheckpoint] {
        &self.stages
    }

    pub(crate) fn push_sbox7(&mut self, entry: Sbox7TraceEntry) {
        self.sbox7.push(entry);
    }

    pub(crate) fn push_k_mul(&mut self, entry: KMulTraceEntry) {
        self.k_muls.push(entry);
    }

    pub(crate) fn push_poseidon_permutation(&mut self, entry: PoseidonPermutationTraceEntry) {
        self.poseidon_permutations.push(entry);
    }

    pub(crate) fn push_poseidon_hash(&mut self, entry: PoseidonHashTraceEntry) {
        self.poseidon_hashes.push(entry);
    }

    pub(crate) fn push_ring_mul_toom3(&mut self, entry: RingMulToom3TraceEntry) {
        self.ring_muls_toom3.push(entry);
    }

    pub(crate) fn push_stage(&mut self, checkpoint: R1csStageCheckpoint) {
        self.stages.push(checkpoint);
    }
}

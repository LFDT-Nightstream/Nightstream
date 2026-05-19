//! Bit-backed CCS-native Poseidon2 sponge builder.
//!
//! Emits a sparse `CcsStructure` whose only nontrivial constraint is
//! "the assignment's bit-backed Poseidon2 trace from `input` produces
//! the recorded final state". Every committed coordinate is `{0, 1}`
//! except the constant-one slot at `z[0]`, so the resulting
//! `CcsInstance` is low-norm under `b = 2` without any further
//! encoding pass.
//!
//! The CCS polynomial is the Plonkish-style mixed-gate
//!
//! ```text
//! f(B, X, Y, Lhs, Rhs) = (B² − B) + (X⁷ − Y) + (Lhs − Rhs)
//! ```
//!
//! with five sparse matrices. Per row, exactly one gate block is
//! active because the inactive matrices have all-zero rows there.
//! Matrices 0 / 1, 2 / 3, 4 select bitness / S-box / linear-equality
//! gates respectively.
//!
//! Shape per Poseidon2 permutation (Goldilocks, WIDTH = 8,
//! HALF_FULL_ROUNDS = 4, PARTIAL_ROUNDS = 22):
//!
//! - 334 committed bit-words per permutation (8 pre-external +
//!   4 × 16 initial full + 22 × 9 partial + 4 × 16 terminal full).
//! - 86 S-box rows per permutation.
//! - 248 linear rows per permutation.
//! - 21,376 bitness rows per permutation (1 per committed bit).
//!
//! For a `poseidon2_hash(input)` of input length `n`, the sponge runs
//! `ceil(n / RATE) + 1` permutations (one per absorb chunk plus the
//! padding permutation), with one extra absorb row group per chunk
//! committing the post-absorb state lanes (8 words = 512 bits each).

use neo_ccs::{CcsMatrix, CcsStructure, CscMat, SparsePoly, Term};
use neo_math::F;
use p3_field::{Field, PrimeCharacteristicRing, PrimeField64};
use p3_goldilocks::Goldilocks;
use rand_chacha_p3::rand_core::{Rng as RandCoreRng, SeedableRng};
use rand_chacha_p3::ChaCha8Rng;

use crate::paper::relations::Structure;

/// Goldilocks bit width (canonical decomposition).
pub const POSEIDON2_GOLDILOCKS_BITS: usize = 64;

/// Poseidon2 state width.
pub const POSEIDON2_WIDTH: usize = 8;

/// Half the number of full rounds (initial and terminal halves are
/// each `HALF_FULL_ROUNDS` long).
pub const POSEIDON2_HALF_FULL_ROUNDS: usize = 4;

/// Number of partial rounds.
pub const POSEIDON2_PARTIAL_ROUNDS: usize = 22;

/// Sponge absorb rate. Matches `neo_params::poseidon2_goldilocks::RATE`.
pub const POSEIDON2_RATE: usize = 4;

/// Sponge digest length. Matches
/// `neo_params::poseidon2_goldilocks::DIGEST_LEN`.
pub const POSEIDON2_DIGEST_LEN: usize = 4;

/// Total bit-words allocated per Poseidon2 permutation by this
/// bit-backed builder.
pub const BIT_BACKED_PERMUTATION_WORDS: usize = POSEIDON2_WIDTH
    + POSEIDON2_WIDTH
    + 2 * POSEIDON2_HALF_FULL_ROUNDS * 2 * POSEIDON2_WIDTH
    + POSEIDON2_PARTIAL_ROUNDS * (1 + POSEIDON2_WIDTH);

/// Committed bits per Poseidon2 permutation under the bit-backed builder.
pub const BITS_PER_PERMUTATION: usize = BIT_BACKED_PERMUTATION_WORDS * POSEIDON2_GOLDILOCKS_BITS;

/// CCS polynomial max degree this module emits.
pub const POSEIDON2_CCS_DEGREE: u32 = 7;

/// Output of [`build_bit_backed_poseidon2_hash`].
#[derive(Debug)]
pub struct CcsNativePoseidon2Hash {
    /// CCS structure with degree-7 polynomial encoding the bit-backed
    /// sponge trace.
    pub structure: Structure,
    /// Low-norm assignment `z = [1 || trace bits]`.
    pub z: Vec<F>,
    /// Sponge output: first `DIGEST_LEN` lanes of the final state.
    pub digest: [F; POSEIDON2_DIGEST_LEN],
    /// CCS row indices of each absorb step's `add_linear_state_rows`
    /// emission. Outer Vec is per absorb (one entry per absorb chunk
    /// plus the padding absorb); inner array is the 8 post-absorb
    /// state words in order. These rows bake the absorbed preimage
    /// values as row constants — F'-side consumers replace them with
    /// variable-source absorb rows when binding the preimage to other
    /// committed image regions.
    pub absorb_rows: Vec<[usize; POSEIDON2_WIDTH]>,
}

/// Output of [`build_bit_backed_poseidon2_permutation`].
#[derive(Debug)]
pub struct CcsNativePoseidon2Permutation {
    /// CCS structure with degree-7 polynomial encoding the bit-backed
    /// permutation trace.
    pub structure: Structure,
    /// Low-norm assignment `z = [1 || trace bits]`.
    pub z: Vec<F>,
    /// Post-permutation state.
    pub output_state: [F; POSEIDON2_WIDTH],
}

/// Build a bit-backed CCS-native Poseidon2 sponge hash trace for
/// `input` of arbitrary length. Mirrors
/// `neo_ccs::crypto::poseidon2_goldilocks::poseidon2_hash`:
///
/// ```text
/// state = [0; WIDTH]
/// for chunk in input.chunks(RATE):
///     state[0..chunk.len()] += chunk
///     state = permute(state)
/// state[0] += 1                  // padding
/// state = permute(state)
/// output = state[0..DIGEST_LEN]
/// ```
///
/// The input is baked into the CCS structure as constants — every
/// absorb step becomes a linear constraint
/// `committed_state_word == previous_state + chunk_value (constant)`.
/// This is the right model for digest call sites where the absorbed
/// preimage is fixed by the prover before commit (the surrounding CCS
/// instance separately commits whatever field arrays the preimage
/// references).
///
/// Returns a typed bundle; `digest` equals native
/// `poseidon2_hash(input)` bit-for-bit for any consistent witness.
pub fn build_bit_backed_poseidon2_hash(input: &[F]) -> CcsNativePoseidon2Hash {
    let constants = poseidon2_constants();
    let mut builder = SparsePermutationBuilder::new();

    let mut state_values = [F::ZERO; POSEIDON2_WIDTH];
    let mut prev_state_words: Option<[Word; POSEIDON2_WIDTH]> = None;
    let mut absorb_rows: Vec<[usize; POSEIDON2_WIDTH]> = Vec::new();

    for chunk in input.chunks(POSEIDON2_RATE) {
        let mut additions = [F::ZERO; POSEIDON2_WIDTH];
        let mut mask = [false; POSEIDON2_WIDTH];
        for (lane, &v) in chunk.iter().enumerate() {
            additions[lane] = v;
            mask[lane] = true;
        }
        let rows = absorb_then_permute(
            &mut builder,
            &constants,
            &mut state_values,
            &mut prev_state_words,
            additions,
            mask,
        );
        absorb_rows.push(rows);
    }

    let mut pad_additions = [F::ZERO; POSEIDON2_WIDTH];
    let mut pad_mask = [false; POSEIDON2_WIDTH];
    pad_additions[0] = F::ONE;
    pad_mask[0] = true;
    let pad_rows = absorb_then_permute(
        &mut builder,
        &constants,
        &mut state_values,
        &mut prev_state_words,
        pad_additions,
        pad_mask,
    );
    absorb_rows.push(pad_rows);

    let (structure, z) = builder.finish();
    let digest: [F; POSEIDON2_DIGEST_LEN] = std::array::from_fn(|i| state_values[i]);
    CcsNativePoseidon2Hash {
        structure,
        z,
        digest,
        absorb_rows,
    }
}

/// Build a bit-backed CCS-native Poseidon2 permutation trace for one
/// full-width input. Same gate-mix encoding as
/// [`build_bit_backed_poseidon2_hash`] but without sponge absorb /
/// padding — emitted as a separate entrypoint mostly so callers and
/// tests can size the permutation in isolation.
pub fn build_bit_backed_poseidon2_permutation(input: [F; POSEIDON2_WIDTH]) -> CcsNativePoseidon2Permutation {
    let constants = poseidon2_constants();
    let mut builder = SparsePermutationBuilder::new();
    let state_words: [Word; POSEIDON2_WIDTH] = std::array::from_fn(|i| builder.push_constrained_word(input[i]));
    let (output_state, _final_words) = append_one_permutation(&mut builder, &constants, input, state_words);
    let (structure, z) = builder.finish();
    CcsNativePoseidon2Permutation {
        structure,
        z,
        output_state,
    }
}

/// Goldilocks Poseidon2 S-box `x → x⁷`. Public for parity tests.
pub fn poseidon2_sbox7(x: F) -> F {
    let x2 = x * x;
    let x4 = x2 * x2;
    x4 * x2 * x
}

/// Append `value`'s little-endian 64-bit decomposition to `z`. Public
/// for parity tests that build bit-backed assignments by hand.
pub fn push_goldilocks_bits(z: &mut Vec<F>, value: F) {
    let raw = value.as_canonical_u64();
    for i in 0..POSEIDON2_GOLDILOCKS_BITS {
        z.push(if ((raw >> i) & 1) == 1 { F::ONE } else { F::ZERO });
    }
}

// ── Internal helpers ──────────────────────────────────────────────────────

#[derive(Clone, Copy)]
struct Word {
    start: usize,
}

#[derive(Clone)]
struct Expr {
    constant: F,
    terms: Vec<(Word, F)>,
}

impl Expr {
    fn zero() -> Self {
        Self {
            constant: F::ZERO,
            terms: Vec::new(),
        }
    }

    fn word(word: Word) -> Self {
        Self {
            constant: F::ZERO,
            terms: vec![(word, F::ONE)],
        }
    }

    fn add_scaled(&self, rhs: &Self, scale: F) -> Self {
        let mut out = self.clone();
        out.constant += rhs.constant * scale;
        out.terms
            .extend(rhs.terms.iter().map(|&(word, coeff)| (word, coeff * scale)));
        out
    }

    fn add_constant(&self, value: F) -> Self {
        let mut out = self.clone();
        out.constant += value;
        out
    }
}

struct Poseidon2Constants {
    initial: [[F; POSEIDON2_WIDTH]; POSEIDON2_HALF_FULL_ROUNDS],
    terminal: [[F; POSEIDON2_WIDTH]; POSEIDON2_HALF_FULL_ROUNDS],
    internal: [F; POSEIDON2_PARTIAL_ROUNDS],
}

fn sample_goldilocks(rng: &mut ChaCha8Rng) -> F {
    loop {
        let x = rng.next_u64();
        if x < Goldilocks::ORDER_U64 {
            return F::from_u64(x);
        }
    }
}

fn poseidon2_constants() -> Poseidon2Constants {
    let mut rng = ChaCha8Rng::from_seed(neo_params::poseidon2_goldilocks::SEED);
    let mut initial = [[F::ZERO; POSEIDON2_WIDTH]; POSEIDON2_HALF_FULL_ROUNDS];
    for round in initial.iter_mut() {
        for slot in round.iter_mut() {
            *slot = sample_goldilocks(&mut rng);
        }
    }
    let mut terminal = [[F::ZERO; POSEIDON2_WIDTH]; POSEIDON2_HALF_FULL_ROUNDS];
    for round in terminal.iter_mut() {
        for slot in round.iter_mut() {
            *slot = sample_goldilocks(&mut rng);
        }
    }
    let mut internal = [F::ZERO; POSEIDON2_PARTIAL_ROUNDS];
    for slot in internal.iter_mut() {
        *slot = sample_goldilocks(&mut rng);
    }
    Poseidon2Constants {
        initial,
        terminal,
        internal,
    }
}

fn internal_diag() -> [F; POSEIDON2_WIDTH] {
    let half = F::from_u64(2).inverse();
    [
        -F::from_u64(2),
        F::ONE,
        F::from_u64(2),
        half,
        F::from_u64(3),
        -half,
        -F::from_u64(3),
        -F::from_u64(4),
    ]
}

fn expr_apply_mat4(state: &mut [Expr; 4]) {
    let x0 = state[0].clone();
    let x1 = state[1].clone();
    let x2 = state[2].clone();
    let x3 = state[3].clone();

    let t01 = x0.clone().add_scaled(&x1, F::ONE);
    let t23 = x2.clone().add_scaled(&x3, F::ONE);
    let t0123 = t01.clone().add_scaled(&t23, F::ONE);
    let t01123 = t0123.clone().add_scaled(&x1, F::ONE);
    let t01233 = t0123.add_scaled(&x3, F::ONE);

    state[3] = t01233.clone().add_scaled(&x0, F::from_u64(2));
    state[1] = t01123.clone().add_scaled(&x2, F::from_u64(2));
    state[0] = t01123.add_scaled(&t01, F::ONE);
    state[2] = t01233.add_scaled(&t23, F::ONE);
}

fn expr_external_linear(input: &[Word; POSEIDON2_WIDTH]) -> [Expr; POSEIDON2_WIDTH] {
    let mut lo: [Expr; 4] = std::array::from_fn(|i| Expr::word(input[i]));
    let mut hi: [Expr; 4] = std::array::from_fn(|i| Expr::word(input[4 + i]));
    expr_apply_mat4(&mut lo);
    expr_apply_mat4(&mut hi);

    let sums: [Expr; 4] = [
        lo[0].clone().add_scaled(&hi[0], F::ONE),
        lo[1].clone().add_scaled(&hi[1], F::ONE),
        lo[2].clone().add_scaled(&hi[2], F::ONE),
        lo[3].clone().add_scaled(&hi[3], F::ONE),
    ];

    std::array::from_fn(|i| {
        let block = if i < 4 { &lo[i] } else { &hi[i - 4] };
        block.clone().add_scaled(&sums[i % 4], F::ONE)
    })
}

fn expr_internal_linear(input: &[Word; POSEIDON2_WIDTH]) -> [Expr; POSEIDON2_WIDTH] {
    let diag = internal_diag();
    let mut sum = Expr::zero();
    for word in input {
        sum = sum.add_scaled(&Expr::word(*word), F::ONE);
    }
    std::array::from_fn(|i| sum.clone().add_scaled(&Expr::word(input[i]), diag[i]))
}

fn value_apply_mat4(state: &mut [F; 4]) {
    let x0 = state[0];
    let x1 = state[1];
    let x2 = state[2];
    let x3 = state[3];

    let t01 = x0 + x1;
    let t23 = x2 + x3;
    let t0123 = t01 + t23;
    let t01123 = t0123 + x1;
    let t01233 = t0123 + x3;

    state[3] = t01233 + F::from_u64(2) * x0;
    state[1] = t01123 + F::from_u64(2) * x2;
    state[0] = t01123 + t01;
    state[2] = t01233 + t23;
}

fn value_external_linear(input: [F; POSEIDON2_WIDTH]) -> [F; POSEIDON2_WIDTH] {
    let mut lo = [input[0], input[1], input[2], input[3]];
    let mut hi = [input[4], input[5], input[6], input[7]];
    value_apply_mat4(&mut lo);
    value_apply_mat4(&mut hi);
    let sums = [lo[0] + hi[0], lo[1] + hi[1], lo[2] + hi[2], lo[3] + hi[3]];
    std::array::from_fn(|i| {
        let block = if i < 4 { lo[i] } else { hi[i - 4] };
        block + sums[i % 4]
    })
}

fn value_internal_linear(input: [F; POSEIDON2_WIDTH]) -> [F; POSEIDON2_WIDTH] {
    let diag = internal_diag();
    let sum = input.iter().copied().fold(F::ZERO, |acc, v| acc + v);
    std::array::from_fn(|i| sum + diag[i] * input[i])
}

struct SparsePermutationBuilder {
    z: Vec<F>,
    trips: [Vec<(usize, usize, F)>; 5],
    rows: usize,
}

impl SparsePermutationBuilder {
    fn new() -> Self {
        Self {
            z: vec![F::ONE],
            trips: std::array::from_fn(|_| Vec::new()),
            rows: 0,
        }
    }

    fn push_word(&mut self, value: F) -> Word {
        let start = self.z.len();
        push_goldilocks_bits(&mut self.z, value);
        Word { start }
    }

    fn add_word_to_matrix(&mut self, matrix: usize, row: usize, word: Word, scale: F) {
        let mut pow2 = scale;
        for i in 0..POSEIDON2_GOLDILOCKS_BITS {
            self.trips[matrix].push((row, word.start + i, pow2));
            pow2 *= F::from_u64(2);
        }
    }

    fn add_expr_to_matrix(&mut self, matrix: usize, row: usize, expr: &Expr) {
        if expr.constant != F::ZERO {
            self.trips[matrix].push((row, 0, expr.constant));
        }
        for &(word, scale) in &expr.terms {
            self.add_word_to_matrix(matrix, row, word, scale);
        }
    }

    fn add_bitness_rows(&mut self, word: Word) {
        for i in 0..POSEIDON2_GOLDILOCKS_BITS {
            let row = self.rows;
            self.trips[0].push((row, word.start + i, F::ONE));
            self.rows += 1;
        }
    }

    fn push_constrained_word(&mut self, value: F) -> Word {
        let word = self.push_word(value);
        self.add_bitness_rows(word);
        word
    }

    fn add_sbox_row(&mut self, x: &Expr, y: Word) {
        let row = self.rows;
        self.add_expr_to_matrix(1, row, x);
        self.add_word_to_matrix(2, row, y, F::ONE);
        self.rows += 1;
    }

    fn add_linear_row(&mut self, lhs: Word, rhs: &Expr) -> usize {
        let row = self.rows;
        self.add_word_to_matrix(3, row, lhs, F::ONE);
        self.add_expr_to_matrix(4, row, rhs);
        self.rows += 1;
        row
    }

    fn add_linear_state_rows(
        &mut self,
        next: &[Word; POSEIDON2_WIDTH],
        exprs: &[Expr; POSEIDON2_WIDTH],
    ) -> [usize; POSEIDON2_WIDTH] {
        std::array::from_fn(|i| self.add_linear_row(next[i], &exprs[i]))
    }

    fn finish(self) -> (Structure, Vec<F>) {
        let f = SparsePoly::new(
            5,
            vec![
                Term {
                    coeff: F::ONE,
                    exps: vec![2, 0, 0, 0, 0],
                },
                Term {
                    coeff: -F::ONE,
                    exps: vec![1, 0, 0, 0, 0],
                },
                Term {
                    coeff: F::ONE,
                    exps: vec![0, 7, 0, 0, 0],
                },
                Term {
                    coeff: -F::ONE,
                    exps: vec![0, 0, 1, 0, 0],
                },
                Term {
                    coeff: F::ONE,
                    exps: vec![0, 0, 0, 1, 0],
                },
                Term {
                    coeff: -F::ONE,
                    exps: vec![0, 0, 0, 0, 1],
                },
            ],
        );
        let n = self.rows;
        let m = self.z.len();
        let matrices = self
            .trips
            .into_iter()
            .map(|trips| CcsMatrix::Csc(CscMat::from_triplets(trips, n, m)))
            .collect();
        let structure = CcsStructure::new_sparse(matrices, f).expect("sparse Poseidon2 CCS structure");
        (structure, self.z)
    }
}

fn append_one_permutation(
    builder: &mut SparsePermutationBuilder,
    constants: &Poseidon2Constants,
    mut state_values: [F; POSEIDON2_WIDTH],
    mut state_words: [Word; POSEIDON2_WIDTH],
) -> ([F; POSEIDON2_WIDTH], [Word; POSEIDON2_WIDTH]) {
    let pre_exprs = expr_external_linear(&state_words);
    state_values = value_external_linear(state_values);
    state_words = std::array::from_fn(|i| builder.push_constrained_word(state_values[i]));
    builder.add_linear_state_rows(&state_words, &pre_exprs);

    for round in 0..POSEIDON2_HALF_FULL_ROUNDS {
        let mut sbox_values = [F::ZERO; POSEIDON2_WIDTH];
        let mut sbox_words = [Word { start: 0 }; POSEIDON2_WIDTH];
        for lane in 0..POSEIDON2_WIDTH {
            let x_expr = Expr::word(state_words[lane]).add_constant(constants.initial[round][lane]);
            sbox_values[lane] = poseidon2_sbox7(state_values[lane] + constants.initial[round][lane]);
            sbox_words[lane] = builder.push_constrained_word(sbox_values[lane]);
            builder.add_sbox_row(&x_expr, sbox_words[lane]);
        }
        let next_exprs = expr_external_linear(&sbox_words);
        state_values = value_external_linear(sbox_values);
        state_words = std::array::from_fn(|i| builder.push_constrained_word(state_values[i]));
        builder.add_linear_state_rows(&state_words, &next_exprs);
    }

    for round in 0..POSEIDON2_PARTIAL_ROUNDS {
        let x_expr = Expr::word(state_words[0]).add_constant(constants.internal[round]);
        let mut sbox_input_values = state_values;
        sbox_input_values[0] += constants.internal[round];
        sbox_input_values[0] = poseidon2_sbox7(sbox_input_values[0]);
        let sbox_word = builder.push_constrained_word(sbox_input_values[0]);
        builder.add_sbox_row(&x_expr, sbox_word);

        let linear_input_words: [Word; POSEIDON2_WIDTH] =
            std::array::from_fn(|i| if i == 0 { sbox_word } else { state_words[i] });
        let next_exprs = expr_internal_linear(&linear_input_words);
        state_values = value_internal_linear(sbox_input_values);
        state_words = std::array::from_fn(|i| builder.push_constrained_word(state_values[i]));
        builder.add_linear_state_rows(&state_words, &next_exprs);
    }

    for round in 0..POSEIDON2_HALF_FULL_ROUNDS {
        let mut sbox_values = [F::ZERO; POSEIDON2_WIDTH];
        let mut sbox_words = [Word { start: 0 }; POSEIDON2_WIDTH];
        for lane in 0..POSEIDON2_WIDTH {
            let x_expr = Expr::word(state_words[lane]).add_constant(constants.terminal[round][lane]);
            sbox_values[lane] = poseidon2_sbox7(state_values[lane] + constants.terminal[round][lane]);
            sbox_words[lane] = builder.push_constrained_word(sbox_values[lane]);
            builder.add_sbox_row(&x_expr, sbox_words[lane]);
        }
        let next_exprs = expr_external_linear(&sbox_words);
        state_values = value_external_linear(sbox_values);
        state_words = std::array::from_fn(|i| builder.push_constrained_word(state_values[i]));
        builder.add_linear_state_rows(&state_words, &next_exprs);
    }

    (state_values, state_words)
}

fn absorb_then_permute(
    builder: &mut SparsePermutationBuilder,
    constants: &Poseidon2Constants,
    state_values: &mut [F; POSEIDON2_WIDTH],
    prev_state_words: &mut Option<[Word; POSEIDON2_WIDTH]>,
    additions: [F; POSEIDON2_WIDTH],
    additions_mask: [bool; POSEIDON2_WIDTH],
) -> [usize; POSEIDON2_WIDTH] {
    for lane in 0..POSEIDON2_WIDTH {
        if additions_mask[lane] {
            state_values[lane] += additions[lane];
        }
    }
    let new_words: [Word; POSEIDON2_WIDTH] = std::array::from_fn(|i| builder.push_constrained_word(state_values[i]));
    let exprs: [Expr; POSEIDON2_WIDTH] = std::array::from_fn(|i| {
        let base = match prev_state_words {
            Some(prev) => Expr::word(prev[i]),
            None => Expr::zero(),
        };
        if additions_mask[i] {
            base.add_constant(additions[i])
        } else {
            base
        }
    });
    let absorb_rows = builder.add_linear_state_rows(&new_words, &exprs);
    let (next_values, next_words) = append_one_permutation(builder, constants, *state_values, new_words);
    *state_values = next_values;
    *prev_state_words = Some(next_words);
    absorb_rows
}

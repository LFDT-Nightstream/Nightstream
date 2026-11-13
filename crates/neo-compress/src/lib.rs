//! An argument of knowledge for CE claims (SuperNeo v1.2 Definition 21) over
//! the Akita commitment scheme, ePrint 2026/1983 (SPEC.md section 6).
//!
//! A claim `(c, X, r, Eval_K, Eval_A)` holds for a witness `z` of ring blocks
//! `z_j` when
//!
//! 1. each row `i` of the Ajtai key opens `c`: `c_i = Σ_j A_ij · z_j`,
//! 2. `Eval_K = conjugate(eq(r_lo, ·)) · Σ_j eq(r_hi, j) · z_j`,
//! 3. `Eval_A_m = Σ_j conjugate(w_mj) · z_j`, where `w_m = M_mᵀ · eq(r, ·)`,
//! 4. the first cells of `z` are `X`, and
//! 5. each coefficient of `z` is in `[-bound, bound]`.
//!
//! The prover commits one polynomial `W` that holds each witness and, for
//! each table, the count of each table value. After the base-field challenges
//! `τ_k`, it commits a polynomial `V` that holds the inverses `1/(τ_k − z)`.
//! Items 1 to 3 are ring equations `Σ_j P_j · z_j = Y`. A challenge `ω`
//! batches them and item 4, a point `δ` turns the batch into an inner product
//! of `W` with public weights (`ring`, `weights`), and one sum-check proves
//! it together with the lookup of item 5 (`lookup`, `sumcheck`). The
//! sum-check ends in one evaluation of `W`, one of `V` and the public weights
//! at its point `ρ`.
//!
//! The verifier does not read the Ajtai key. Preprocessing commits the key as
//! one polynomial, and the verifier holds that commitment (`key`, ePrint
//! 2026/1983 section 9). The proof carries the part of the weights at `ρ`
//! that the key gives. A second sum-check reduces that part to one
//! evaluation of the key polynomial. One Akita proof opens the key
//! polynomial, `W` and `V`. The verifier computes the part of the weights
//! that the matrices give in one pass over the rows of each claim
//! (`weights::matrix_part`).
//!
//! # Transcript
//!
//! The transcript is the caller's Poseidon2 transcript, after the caller's
//! messages. Akita derives its own challenges, and its session is the state
//! of this transcript. The prover and the verifier absorb and draw in this
//! order:
//!
//! 1. absorb the domain, then the bytes of the verification key,
//! 2. absorb each claim: its sizes, then its words,
//! 3. absorb the commitment of `W`,
//! 4. draw `τ_0` to `τ_3`,
//! 5. absorb the commitment of `V`,
//! 6. draw `ω`, `δ`, `γ`, `μ`, each `ψ_k`, `ξ` and `λ` of each table, and
//!    the point `β` of the inverse check,
//! 7. for each round of the first sum-check, absorb its four coefficients
//!    and draw one coordinate of `ρ`,
//! 8. absorb `W(ρ)`, the four values of `V` and the key's part of the
//!    weights,
//! 9. draw the point of the columns of `V`,
//! 10. for each round of the key sum-check, absorb its three coefficients
//!     and draw one coordinate of `(λ, β′)`,
//! 11. absorb the value of the key polynomial,
//! 12. give Akita the label of the opening and lanes 0 to 3 of the sponge
//!     state as its session.
//!
//! # Soundness
//!
//! Soundness error, with `|K| = q²`, `q = 2^64 − 59`, `n` variables of `W`,
//! `N` cells of the slice of a witness with a table, `T` values of its
//! table, `m` public cells, `t` matrices, `s` claims and `b` variables of
//! the blocks of the key polynomial:
//!
//! - batch of the equations at `(ω, δ, γ, μ)`:
//!   `(κ + 1 + t + max(63, m − 1) + s − 1) / |K|`,
//! - lookup: `((N + T) / q)^4` for each table,
//! - batch of the inverse checks and the sums at `(ψ, ξ, λ, β)`: `(2 + n) / |K|`,
//! - first sum-check: `3n / |K|`,
//! - one point for the columns of `V`: `2 / |K|`,
//! - key sum-check, degree 2 over 6 lane variables and the block variables:
//!   `2 · (6 + b) / |K|`,
//! - the Akita opening: the knowledge error of Akita's
//!   `fp64_dense_recursive` schedule row, which this crate does not compute.
//!
//! The key sum-check proves the exact sum that the first sum-check needs, so
//! the claims need no new batch. The identity `ω^i = scale · eq(s, i)` of the
//! rows is exact, so the row point adds no error. No proof exists when `ω`
//! has no row point, with probability at most `31 / |K|`.
//!
//! `error_bits` gives the sum without the Akita term. For the two step
//! circuits of the Poseidon2 hash chain (`N = 2^25` and `2^23`, `T ≤ 20737`,
//! `n ≤ 27`, `m = 320`, `t = 4`, `s = 3`, `b = 19`) the lookups give
//! `2^-155.99`, the other terms `2^-119.02`, and the sum is `2^-119.02`.
//!
//! The argument assumes that the commitment of the verification key is the
//! Akita commitment of the key polynomial, and that Akita is binding and
//! knowledge sound for it: the value that Akita opens is the value of that
//! polynomial. The verifier never sees the polynomial. A third party checks
//! a verification key with `VerificationKey::derive`, which computes it
//! again from the public seeds.

// The claims of `tests/support` name this crate as an integration test does.
#[cfg(test)]
extern crate self as neo_compress;
#[cfg(test)]
#[path = "../tests/support/mod.rs"]
mod support;

mod akita;
mod key;
mod layout;
mod lookup;
mod prover;
pub mod ring;
mod sumcheck;
mod transcript;
mod verifier;
mod weights;

use neo_ajtai::Commitment;
use neo_math::{KExtensions, D, F, K};
use neo_reductions::superneo_eval::MatrixRows;
use neo_transcript::{domain_chunk_v1_2, Poseidon2Transcript};
use p3_field::{PrimeCharacteristicRing, PrimeField64};

pub use key::{key_blocks, ProverKey, VerificationKey};
pub use layout::Layout;
pub use prover::{commit, Committed, ProverTimes};
pub use sumcheck::{KeyRound, Round};
pub use transcript::byte_words;
pub use verifier::{verify, VerifierTimes};

pub type CeClaim = neo_ccs::CeClaim<Commitment, F, K>;

const DOMAIN: &[u8] = b"Nightstream/compress/akita/v2";
const SESSION: &[u8] = b"Nightstream/compress/akita/v2/opening";

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("compression layout: {0}")]
    Layout(&'static str),
    #[error("compression statement: {0}")]
    Statement(&'static str),
    #[error("compression proof: {0}")]
    Rejected(&'static str),
    #[error("compression key: {0}")]
    Key(&'static str),
    /// A lookup challenge is a table value, with probability about `2^-46.7`
    /// for each proof of the Poseidon2 hash chain, or `ω` has no row point
    /// (`key::row_point`). No proof exists for this transcript.
    #[error("compression: a challenge has no proof")]
    Challenge,
    #[error("Akita: {0}")]
    Akita(String),
    #[error(transparent)]
    Rows(#[from] neo_reductions::PiCcsError),
}

/// How a claim's witness gets its norm bound.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Norm {
    /// This argument proves that each coefficient is in `[-bound, bound]`.
    Table { bound: u64 },
    /// The caller proves the bound. The caller's proof must bind the
    /// commitment of `W` before its challenges.
    Caller,
}

/// One CE claim as the verifier knows it: the claim, the rows of its
/// matrices and its norm proof.
#[derive(Clone, Copy)]
pub struct Statement<'a> {
    pub claim: &'a CeClaim,
    pub rows: &'a dyn MatrixRows,
    pub norm: Norm,
}

impl Statement<'_> {
    fn blocks(&self) -> usize {
        self.rows.shape().columns / D
    }
}

/// The argument: the commitments of `W` and of `V`, both sum-checks, the
/// values at their points and one Akita opening.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Proof {
    pub witness_commitment: Vec<u8>,
    pub inverse_commitment: Vec<u8>,
    pub rounds: Vec<Round>,
    /// `W` at the point of the sum-check.
    pub witness_value: K,
    /// Each column of `V` at the point of the sum-check.
    pub inverse_values: [K; 1 << layout::LOOKUP_VARS],
    /// The part of the public weights at the point of the sum-check that
    /// the Ajtai key gives (`weights::key_part`).
    pub key_weight: K,
    pub key_rounds: Vec<KeyRound>,
    /// The key polynomial at the point of the key sum-check and the row
    /// point.
    pub key_value: K,
    /// The Akita proof of the values of the key polynomial, `W` and `V`.
    pub opening: Vec<u8>,
}

/// The layout of `statements` under a key polynomial of `key_blocks`
/// blocks, after the shape checks that each later step relies on.
fn layout(statements: &[Statement<'_>], key_blocks: usize) -> Result<Layout, Error> {
    let lane_vars = layout::LANE_VARS;
    for statement in statements {
        let (claim, shape) = (statement.claim, statement.rows.shape());
        let blocks = statement.blocks();
        let variables = claim.r.len();
        let evaluations = |family: &Vec<K>| family.len() >= D && family[D..].iter().all(|&lane| lane == K::ZERO);
        if claim.c.kappa() != weights::KAPPA
            || claim.c.d() != D
            || claim.adv.is_some()
            || claim.X.rows() != D
            || claim.X.cols() * D != claim.m_in
            || claim.m_in > shape.columns
            || shape.columns != blocks * D
            || claim.eval_a.len() != shape.matrices
            || !evaluations(&claim.eval_k)
            || !claim.eval_a.iter().all(evaluations)
            || variables < lane_vars
            || variables >= usize::BITS as usize
            || blocks > 1 << (variables - lane_vars)
            || shape.rows > 1 << variables
        {
            return Err(Error::Statement("claim shape"));
        }
    }
    let claims: Vec<_> = statements
        .iter()
        .map(|statement| (statement.blocks(), statement.norm))
        .collect();
    Layout::new(&claims, key_blocks)
}

/// Bind the verification key and the statements: the bytes of the key,
/// then for each claim its sizes and its words.
fn absorb_statements(transcript: &mut Poseidon2Transcript, key: &VerificationKey, statements: &[Statement<'_>]) {
    transcript.absorb_v1_2(&domain_chunk_v1_2(DOMAIN));
    transcript::absorb_bytes(transcript, &key.to_bytes());
    for statement in statements {
        let claim = statement.claim;
        let bound = match statement.norm {
            Norm::Table { bound } => bound + 1,
            Norm::Caller => 0,
        };
        let sizes = [
            statement.blocks() as u64,
            bound,
            claim.m_in as u64,
            claim.r.len() as u64,
            claim.eval_a.len() as u64,
        ];
        let mut words: Vec<F> = sizes.map(F::from_u64).to_vec();
        words.extend_from_slice(claim.c.data());
        words.extend(claim.X.to_dense_vec());
        let extension = claim
            .r
            .iter()
            .chain(&claim.eval_k[..D])
            .chain(claim.eval_a.iter().flat_map(|family| &family[..D]));
        words.extend(extension.flat_map(|value| value.as_coeffs()));
        transcript.absorb_v1_2(&words);
    }
}

/// The base-field lookup challenges: the coordinates of extension challenges.
fn lookup_challenges(transcript: &mut Poseidon2Transcript) -> [F; layout::LOOKUPS] {
    let coins: Vec<F> = transcript::challenges(transcript, layout::LOOKUPS.div_ceil(2))
        .iter()
        .flat_map(|coin| coin.as_coeffs())
        .collect();
    std::array::from_fn(|k| coins[k])
}

fn challenges(
    transcript: &mut Poseidon2Transcript,
    layout: &Layout,
    taus: [F; layout::LOOKUPS],
) -> weights::Challenges {
    let tables = layout.tables().count();
    let mut next = |count: usize| transcript::challenges(transcript, count);
    weights::Challenges {
        taus,
        omega: next(1)[0],
        delta: next(1)[0],
        gamma: next(1)[0],
        mu: next(1)[0],
        psi: next(layout::LOOKUPS).try_into().expect("one per lookup"),
        xi: next(tables),
        lambda: next(tables),
        beta: next(layout.span_vars),
    }
}

/// The sum that the sum-check proves: the targets of the weights, and
/// `Σ_k ψ_k · Σ_tables ξ` from the 1 of each inverse check.
fn claimed_sum(statements: &[Statement<'_>], challenges: &weights::Challenges) -> K {
    weights::targets(statements, challenges)
        + challenges.psi.iter().copied().sum::<K>() * challenges.xi.iter().copied().sum::<K>()
}

/// The point at which `V` is opened: the point of the sum-check on the cells
/// with a table, then `column`, then zeros.
fn inverse_point(layout: &Layout, rho: &[K], column: &[K]) -> Vec<K> {
    let mut point = rho[..layout.span_vars].to_vec();
    point.extend_from_slice(column);
    point.resize(layout.inverse_vars, K::ZERO);
    point
}

/// The sum of the soundness errors of this argument for `layout`, without
/// the Akita openings, as `-log2`.
pub fn error_bits(layout: &Layout, public_cells: usize, matrices: usize) -> f64 {
    let extension = 2.0 * (F::ORDER_U64 as f64).log2();
    let lookups: f64 = layout
        .tables()
        .map(|(_, witness, bound, _)| {
            let poles = (witness.len() as f64 + 2.0 * bound as f64 + 1.0) / F::ORDER_U64 as f64;
            poles.powi(layout::LOOKUPS as i32)
        })
        .sum();
    let batch = weights::KAPPA + 1 + matrices + 63.max(public_cells.saturating_sub(1)) + layout.cells.len() - 1;
    let key = 2 * (layout::LANE_VARS + layout.key_block_vars());
    let rest = (batch + 2 + layout.witness_vars + 3 * layout.witness_vars + 2 + key) as f64;
    -(lookups + rest * (-extension).exp2()).log2()
}

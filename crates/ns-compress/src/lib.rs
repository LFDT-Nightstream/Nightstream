//! An argument of knowledge for CE claims (SuperNeo v1.2 Definition 21) over
//! the Akita commitment scheme, ePrint 2026/1983 (SPEC.md section 6).

#[cfg(test)]
extern crate self as ns_compress;
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
    #[error("compression: a challenge has no proof")]
    Challenge,
    #[error("Akita: {0}")]
    Akita(String),
    #[error(transparent)]
    Rows(#[from] neo_reductions::PiCcsError),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Norm {
    Table { bound: u64 },
    Caller,
}

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

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Proof {
    pub witness_commitment: Vec<u8>,
    pub inverse_commitment: Vec<u8>,
    pub rounds: Vec<Round>,
    pub witness_value: K,
    pub inverse_values: [K; 1 << layout::LOOKUP_VARS],
    pub key_weight: K,
    pub key_rounds: Vec<KeyRound>,
    pub key_value: K,
    pub opening: Vec<u8>,
}

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

/// SPEC.md section 5.4, row 6; Lean `Export.BoundedDepth.CompressionSponge.Claim.words`.
fn claim_words(statement: &Statement<'_>) -> Vec<F> {
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
    words
}

fn absorb_statements(transcript: &mut Poseidon2Transcript, key: &VerificationKey, statements: &[Statement<'_>]) {
    transcript.absorb_v1_2(&domain_chunk_v1_2(DOMAIN));
    transcript::absorb_bytes(transcript, &key.to_bytes());
    for statement in statements {
        transcript.absorb_v1_2(&claim_words(statement));
    }
}

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

fn claimed_sum(statements: &[Statement<'_>], challenges: &weights::Challenges) -> K {
    weights::targets(statements, challenges)
        + challenges.psi.iter().copied().sum::<K>() * challenges.xi.iter().copied().sum::<K>()
}

fn inverse_point(layout: &Layout, rho: &[K], column: &[K]) -> Vec<K> {
    let mut point = rho[..layout.span_vars].to_vec();
    point.extend_from_slice(column);
    point.resize(layout.inverse_vars, K::ZERO);
    point
}

/// SPEC.md section 3, `Spec.Compression.Rust.runBits`.
pub fn error_bits(layout: &Layout, public_cells: usize, matrices: usize) -> u32 {
    const { assert!(layout::LOOKUPS >= 4) };
    let q = u128::from(F::ORDER_U64);
    let mut count = 0;
    for (_, witness, bound, _) in layout.tables() {
        let poles = witness.len() as u128 + 2 * u128::from(bound) + 1;
        if poles.checked_mul(poles).is_none_or(|square| square > q) {
            return 0;
        }
        count += 1;
    }
    let batch = weights::KAPPA + 1 + matrices + 63.max(public_cells.saturating_sub(1)) + layout.cells.len() - 1;
    let key = 2 * (layout::LANE_VARS + layout.key_block_vars());
    count += (batch + 2 + layout.witness_vars + 3 * layout.witness_vars + 2 + key) as u128;
    (q * q)
        .checked_div(count)
        .and_then(u128::checked_ilog2)
        .unwrap_or(0)
}

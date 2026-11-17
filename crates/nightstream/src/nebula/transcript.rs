//! SPEC.md section 7.4; `Lifecycle/Transcript.lean`, `Lifecycle/Nebula/Framing.lean`.

use neo_ccs::crypto::poseidon2::{permute_state, RATE, WIDTH};
use neo_math::F;
use neo_math::K;
use p3_field::{BasedVectorSpace, PrimeCharacteristicRing};

use super::geometry::Geometry;

pub type Digest = [F; 4];
pub type State = [F; WIDTH];

/// SPEC.md section 7.4.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Lane {
    Ops,
    Mem,
}

pub fn text(tag: &str) -> Vec<F> {
    tag.bytes().map(F::from_u8).collect()
}

fn absorb(mut state: State, words: &[F]) -> State {
    for chunk in words.chunks(RATE) {
        for (lane, word) in chunk.iter().enumerate() {
            state[lane] += *word;
        }
        state = permute_state(state);
    }
    state
}

pub fn absorb_block(state: State, words: &[F]) -> State {
    let mut block = Vec::with_capacity(words.len() + 1);
    block.push(F::from_usize(words.len()));
    block.extend_from_slice(words);
    absorb(state, &block)
}

pub fn absorbed(blocks: &[&[F]]) -> State {
    blocks
        .iter()
        .fold([F::ZERO; WIDTH], |state, block| absorb_block(state, block))
}

pub fn digest(state: &State) -> Digest {
    [state[0], state[1], state[2], state[3]]
}

pub fn squeeze_k(state: State) -> (K, State) {
    let first = permute_state(state);
    let value = K::from_basis_coefficients_slice(&[state[0], first[0]]).expect("two coefficients");
    (value, permute_state(first))
}

fn header_tag(lane: Lane) -> &'static str {
    match lane {
        Lane::Ops => "Nightstream/Nebula/v3/header-ops",
        Lane::Mem => "Nightstream/Nebula/v3/header-mem",
    }
}

pub(super) const STATE_TAG: &str = "Nightstream/Nebula/v3/state";
pub(super) const ETA_TAG: &str = "Nightstream/Nebula/v3/eta";

pub(super) fn chain_tag(lane: Lane) -> &'static str {
    match lane {
        Lane::Ops => "Nightstream/Nebula/v3/chain-ops",
        Lane::Mem => "Nightstream/Nebula/v3/chain-mem",
    }
}

/// SPEC.md section 7.1.
pub fn plan_digest(geometry: &Geometry) -> Digest {
    let fields: Vec<F> = [
        geometry.r,
        geometry.mu,
        geometry.w_ts,
        geometry.b_ops,
        geometry.b_scan,
        geometry.n,
        geometry.s_max,
    ]
    .iter()
    .map(|value| F::from_usize(*value))
    .collect();
    digest(&absorbed(&[&text("Nightstream/Nebula/v3/plan"), &fields]))
}

/// SPEC.md section 7.4 `header_l`.
pub fn header(lane: Lane, plan_digest: &Digest) -> Digest {
    digest(&absorbed(&[&text(header_tag(lane)), plan_digest]))
}

/// SPEC.md section 7.4.
pub fn chain_link(lane: Lane, index: usize, previous: &Digest, packed: &[F]) -> Digest {
    let mut prefix = Vec::with_capacity(5);
    prefix.push(F::from_usize(index));
    prefix.extend_from_slice(previous);
    digest(&absorbed(&[&text(chain_tag(lane)), &prefix, packed]))
}

pub fn chain_root(lane: Lane, plan_digest: &Digest, packed: &[Vec<F>]) -> Digest {
    packed
        .iter()
        .enumerate()
        .fold(header(lane, plan_digest), |previous, (index, words)| {
            chain_link(lane, index, &previous, words)
        })
}

/// SPEC.md section 7.4.
pub fn eta_challenges(
    plan_digest: &Digest,
    ts: F,
    ops_root: &Digest,
    mem_root: &Digest,
    final_root: &Digest,
) -> (K, K) {
    let mut roots = Vec::with_capacity(12);
    roots.extend_from_slice(ops_root);
    roots.extend_from_slice(mem_root);
    roots.extend_from_slice(final_root);
    let state = absorbed(&[&text(ETA_TAG), plan_digest, &[ts], &roots]);
    let (eta1, state) = squeeze_k(state);
    let (eta2, _) = squeeze_k(state);
    (eta1, eta2)
}

/// SPEC.md section 7.5.
pub fn state_words(app: &[F], carry: &[F]) -> Digest {
    digest(&absorbed(&[&text(STATE_TAG), app, carry]))
}

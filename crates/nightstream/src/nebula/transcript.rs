//! Native Poseidon2 transcript of spec §9 over the v1.1 framing: a block is
//! its length, then its words; a block is absorbed in rate chunks, each added
//! to the rate lanes and permuted; a digest is the first four lanes.
//! Mirrors `Lifecycle/Transcript.lean` and `Lifecycle/Nebula/Framing.lean`.

use neo_ccs::crypto::poseidon2_goldilocks::{permute_state, RATE, WIDTH};
use neo_math::K;
use p3_field::{BasedVectorSpace, PrimeCharacteristicRing};
use p3_goldilocks::Goldilocks as F;

use super::plan::Plan;

pub type Digest = [F; 4];
pub type State = [F; WIDTH];

/// The two lanes of spec §9.2.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Lane {
    Ops,
    Mem,
}

/// ASCII text as field words, one per byte.
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

/// `absorb_block_v1_1`: the block's length, then its words.
pub fn absorb_block(state: State, words: &[F]) -> State {
    let mut block = Vec::with_capacity(words.len() + 1);
    block.push(F::from_usize(words.len()));
    block.extend_from_slice(words);
    absorb(state, &block)
}

/// The transcript state after `reset_v1_1` and one block absorb per block.
pub fn absorbed(blocks: &[&[F]]) -> State {
    blocks
        .iter()
        .fold([F::ZERO; WIDTH], |state, block| absorb_block(state, block))
}

pub fn digest(state: &State) -> Digest {
    [state[0], state[1], state[2], state[3]]
}

/// One extension squeeze: lane 0, permute, lane 0, permute.
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

fn chain_tag(lane: Lane) -> &'static str {
    match lane {
        Lane::Ops => "Nightstream/Nebula/v3/chain-ops",
        Lane::Mem => "Nightstream/Nebula/v3/chain-mem",
    }
}

/// Spec §4.3 `plan_digest`.
pub fn plan_digest(plan: &Plan) -> Digest {
    let fields: Vec<F> = [plan.r, plan.mu, plan.w_ts, plan.b_ops, plan.b_scan, plan.n, plan.s_max]
        .iter()
        .map(|value| F::from_usize(*value))
        .collect();
    let rom: Vec<F> = (0..plan.rom_size())
        .map(|a| F::from_u64(plan.rom[a]))
        .collect();
    let ram: Vec<F> = (0..plan.ram_size())
        .map(|a| F::from_u64(plan.ram[a]))
        .collect();
    digest(&absorbed(&[&text("Nightstream/Nebula/v3/plan"), &fields, &rom, &ram]))
}

/// `header_l` of spec §9.2.
pub fn header(lane: Lane, plan_digest: &Digest) -> Digest {
    digest(&absorbed(&[&text(header_tag(lane)), plan_digest]))
}

/// One chain link of spec §9.2 over packed lane words.
pub fn chain_link(lane: Lane, index: usize, previous: &Digest, packed: &[F]) -> Digest {
    let mut prefix = Vec::with_capacity(5);
    prefix.push(F::from_usize(index));
    prefix.extend_from_slice(previous);
    digest(&absorbed(&[&text(chain_tag(lane)), &prefix, packed]))
}

/// The root of a lane chain over its packed lanes.
pub fn chain_root(lane: Lane, plan_digest: &Digest, packed: &[Vec<F>]) -> Digest {
    packed
        .iter()
        .enumerate()
        .fold(header(lane, plan_digest), |previous, (index, words)| {
            chain_link(lane, index, &previous, words)
        })
}

/// Spec §9.3: the two memory challenges of a segment.
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
    let state = absorbed(&[&text("Nightstream/Nebula/v3/eta"), plan_digest, &[ts], &roots]);
    let (eta1, state) = squeeze_k(state);
    let (eta2, _) = squeeze_k(state);
    (eta1, eta2)
}

/// The four Stage 1 application-state words: the spec §11.1 state digest of
/// the application words and the carry words.
pub fn state_words(app: &[F], carry: &[F]) -> Digest {
    digest(&absorbed(&[&text("Nightstream/Nebula/v3/state"), app, carry]))
}

//! Hashing over any backend, as Plonky3 0.8 hashes: the
//! `DuplexChallenger<Gl, Poseidon2Goldilocks<16>, 16, 12>`, the overwrite leaf sponge
//! (`PaddingFreeSponge`), node compression (`TruncatedPermutation`) and Merkle path
//! verification. `neo-spartan` tests pin each one to its Plonky3 counterpart.
//!
//! `formal/compression-lean/CompressionLeanRust/Hash.lean` proves each function, run on the
//! Lean copies' `Build`, equal to its copy in `Ports/Circuit/Hash.lean`.

use crate::backend::{Backend, Rejected};
use crate::circuit::{DIGEST_LEN, RATE, WIDTH};

pub type Digest<B> = [<B as Backend>::F; DIGEST_LEN];

/// The p3 duplex challenger: overwrite absorb, zero fill and a length tag in lane `RATE` when
/// a partial block is duplexed, samples popped from the end.
pub struct Duplex<B: Backend> {
    pub state: [B::F; WIDTH],
    pub input: Vec<B::F>,
    pub output: Vec<B::F>,
}

impl<B: Backend> Duplex<B> {
    pub fn new(b: &mut B) -> Self {
        let zero = b.constant(0);
        Self {
            state: [zero; WIDTH],
            input: Vec::new(),
            output: Vec::new(),
        }
    }

    pub fn observe(&mut self, b: &mut B, value: B::F) {
        self.output = Vec::new();
        self.input.push(value);
        if self.input.len() == RATE {
            self.duplex(b);
        }
    }

    pub fn observe_slice(&mut self, b: &mut B, values: &[B::F]) {
        let mut i = 0;
        while i < values.len() {
            self.observe(b, values[i]);
            i = i.wrapping_add(1);
        }
    }

    /// The basis coefficients in order, as `observe_algebra_element`.
    pub fn observe_ext(&mut self, b: &mut B, value: B::E) {
        let coordinates = b.coordinates(value);
        self.observe(b, coordinates[0]);
        self.observe(b, coordinates[1]);
        self.observe(b, coordinates[2]);
    }

    pub fn sample(&mut self, b: &mut B) -> B::F {
        if !<[B::F]>::is_empty(&self.input) || <[B::F]>::is_empty(&self.output) {
            self.duplex(b);
        }
        let last = self.output.len().wrapping_sub(1);
        let value = self.output[last];
        let mut rest = Vec::new();
        let mut i = 0;
        while i < last {
            rest.push(self.output[i]);
            i = i.wrapping_add(1);
        }
        self.output = rest;
        value
    }

    /// Coefficient `i` is the `i`-th sample, as `sample_algebra_element`.
    pub fn sample_ext(&mut self, b: &mut B) -> B::E {
        let x = self.sample(b);
        let y = self.sample(b);
        let z = self.sample(b);
        b.ext([x, y, z])
    }

    /// The low `bits` bits of the canonical sample, as `sample_bits`.
    pub fn sample_bits(&mut self, b: &mut B, bits: usize) -> Result<Vec<B::F>, Rejected> {
        let sample = self.sample(b);
        let all = b.bits(sample, 64, "sampled bits")?;
        let mut low = Vec::new();
        let mut i = 0;
        while i < bits && i < all.len() {
            low.push(all[i]);
            i = i.wrapping_add(1);
        }
        Ok(low)
    }

    fn duplex(&mut self, b: &mut B) {
        let absorbed = self.input.len();
        let mut lane = 0;
        while lane < absorbed {
            self.state[lane] = self.input[lane];
            lane = lane.wrapping_add(1);
        }
        self.input = Vec::new();
        if absorbed > 0 {
            let zero = b.constant(0);
            let mut lane = absorbed;
            while lane < RATE {
                self.state[lane] = zero;
                lane = lane.wrapping_add(1);
            }
            let tag = b.constant(absorbed as u64);
            self.state[RATE] = b.add(self.state[RATE], tag);
        }
        self.state = b.permute(self.state);
        let mut output = Vec::new();
        let mut lane = 0;
        while lane < RATE {
            output.push(self.state[lane]);
            lane = lane.wrapping_add(1);
        }
        self.output = output;
    }
}

/// The overwrite sponge of p3 `PaddingFreeSponge` over `values`.
pub fn hash_leaf<B: Backend>(b: &mut B, values: &[B::F]) -> Digest<B> {
    let zero = b.constant(0);
    let mut state = [zero; WIDTH];
    let mut start = 0;
    while start < values.len() {
        let mut lane = 0;
        while lane < RATE && lane < values.len().wrapping_sub(start) {
            state[lane] = values[start.wrapping_add(lane)];
            lane = lane.wrapping_add(1);
        }
        state = b.permute(state);
        start = start.wrapping_add(lane);
    }
    [state[0], state[1], state[2], state[3]]
}

/// p3 `TruncatedPermutation`: `[left, right, 0…]`, permute, first four lanes.
pub fn compress<B: Backend>(b: &mut B, left: Digest<B>, right: Digest<B>) -> Digest<B> {
    let zero = b.constant(0);
    let mut state = [zero; WIDTH];
    let mut lane = 0;
    while lane < DIGEST_LEN {
        state[lane] = left[lane];
        state[DIGEST_LEN.wrapping_add(lane)] = right[lane];
        lane = lane.wrapping_add(1);
    }
    let state = b.permute(state);
    [state[0], state[1], state[2], state[3]]
}

/// The root of `leaf` at the index with little-endian `bits` along `path` (bottom first): bit
/// zero puts the node on the left. Reads the shorter of `bits` and `path`.
pub fn merkle_root<B: Backend>(b: &mut B, leaf: Digest<B>, bits: &[B::F], path: &[Digest<B>]) -> Digest<B> {
    let mut node = leaf;
    let mut level = 0;
    while level < bits.len() && level < path.len() {
        let bit = bits[level];
        let sibling = path[level];
        let mut left = node;
        let mut right = node;
        let mut lane = 0;
        while lane < DIGEST_LEN {
            left[lane] = b.select(bit, node[lane], sibling[lane]);
            let both = b.add(node[lane], sibling[lane]);
            right[lane] = b.sub(both, left[lane]);
            lane = lane.wrapping_add(1);
        }
        node = compress(b, left, right);
        level = level.wrapping_add(1);
    }
    node
}

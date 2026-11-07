//! Hashing over any backend, matching Plonky3 0.8 exactly: the
//! `DuplexChallenger<Gl, Poseidon2Goldilocks<16>, 16, 12>`, the overwrite leaf
//! sponge (`PaddingFreeSponge`), node compression (`TruncatedPermutation`) and
//! Merkle path verification. Tests pin each one to its Plonky3 counterpart.

use neo_ccs::crypto::poseidon2_goldilocks::{DIGEST_LEN, RATE, WIDTH};

use super::Backend;
use crate::Error;

pub(crate) type Digest<B> = [<B as Backend>::F; DIGEST_LEN];

/// The p3 duplex challenger: overwrite absorb, zero fill and a length tag in
/// lane `RATE` when a partial block is duplexed, samples popped from the end.
pub(crate) struct Duplex<B: Backend> {
    pub(crate) state: [B::F; WIDTH],
    pub(crate) input: Vec<B::F>,
    pub(crate) output: Vec<B::F>,
    pub(crate) cursor: Option<[B::F; RATE + 1]>,
}

impl<B: Backend> Duplex<B> {
    pub(crate) fn new(b: &mut B) -> Self {
        let zero = b.constant(0);
        Self {
            state: [zero; WIDTH],
            input: Vec::new(),
            output: Vec::new(),
            cursor: None,
        }
    }

    pub(crate) fn observe(&mut self, b: &mut B, value: B::F) {
        self.output.clear();
        self.cursor = None;
        self.input.push(value);
        if self.input.len() == RATE {
            self.duplex(b);
        }
    }

    pub(crate) fn observe_slice(&mut self, b: &mut B, values: &[B::F]) {
        for &value in values {
            self.observe(b, value);
        }
    }

    /// The basis coefficients in order, as `observe_algebra_element`.
    pub(crate) fn observe_ext(&mut self, b: &mut B, value: B::E) {
        for coordinate in b.coordinates(value) {
            self.observe(b, coordinate);
        }
    }

    pub(crate) fn sample(&mut self, b: &mut B) -> B::F {
        if self.cursor.is_some() {
            let one = b.constant(1);
            return self.sample_if(b, one);
        }
        if !self.input.is_empty() || self.output.is_empty() {
            self.duplex(b);
        }
        self.output
            .pop()
            .expect("a duplexed state fills the output")
    }

    /// `CompressionLean.Circuit.Duplex.sampleIf`: a Boolean enable consumes
    /// one buffered sample, with no pending input. Disabled calls return zero.
    pub(crate) fn sample_if(&mut self, b: &mut B, enabled: B::F) -> B::F {
        assert!(self.input.is_empty());
        let zero = b.constant(0);
        let one = b.constant(1);
        let mut cursor = self.cursor.unwrap_or_else(|| {
            let count = self.output.len();
            assert!(count <= RATE);
            self.output.reverse();
            self.output.resize(RATE, zero);
            std::array::from_fn(|i| if i == count { one } else { zero })
        });
        let refill = b.mul(enabled, cursor[0]);
        let next = b.permute(self.state);
        for (lane, value) in self.state.iter_mut().enumerate() {
            *value = b.select(refill, *value, next[lane]);
        }
        for (lane, value) in self.output.iter_mut().enumerate() {
            *value = b.select(refill, *value, next[RATE - 1 - lane]);
        }
        for (i, value) in cursor.iter_mut().enumerate() {
            *value = b.select(refill, *value, if i == RATE { one } else { zero });
        }
        let result = b.select(enabled, zero, self.output[0]);
        for i in 0..RATE {
            let next = self.output.get(i + 1).copied().unwrap_or(zero);
            self.output[i] = b.select(enabled, self.output[i], next);
        }
        for i in 0..=RATE {
            let next = cursor.get(i + 1).copied().unwrap_or(zero);
            cursor[i] = b.select(enabled, cursor[i], next);
        }
        self.input.clear();
        self.cursor = Some(cursor);
        result
    }

    /// Coefficient `i` is the `i`-th sample, as `sample_algebra_element`.
    pub(crate) fn sample_ext(&mut self, b: &mut B) -> B::E {
        let coordinates = std::array::from_fn(|_| self.sample(b));
        b.ext(coordinates)
    }

    /// The low `bits` bits of the canonical sample, as `sample_bits`.
    pub(crate) fn sample_bits(&mut self, b: &mut B, bits: usize) -> Result<Vec<B::F>, Error> {
        let sample = self.sample(b);
        let mut all = b.bits(sample, 64, "sampled bits")?;
        all.truncate(bits);
        Ok(all)
    }

    fn duplex(&mut self, b: &mut B) {
        let absorbed = self.input.len();
        for (lane, value) in self.input.drain(..).enumerate() {
            self.state[lane] = value;
        }
        if absorbed > 0 {
            let zero = b.constant(0);
            self.state[absorbed..RATE].fill(zero);
            let tag = b.constant(absorbed as u64);
            self.state[RATE] = b.add(self.state[RATE], tag);
        }
        self.state = b.permute(self.state);
        self.output.clear();
        self.output.extend_from_slice(&self.state[..RATE]);
        self.cursor = None;
    }
}

/// The overwrite sponge of p3 `PaddingFreeSponge` over `values`.
pub(crate) fn hash_leaf<B: Backend>(b: &mut B, values: &[B::F]) -> Digest<B> {
    let zero = b.constant(0);
    let mut state = [zero; WIDTH];
    for chunk in values.chunks(RATE) {
        state[..chunk.len()].copy_from_slice(chunk);
        state = b.permute(state);
    }
    std::array::from_fn(|lane| state[lane])
}

/// p3 `TruncatedPermutation`: `[left, right, 0…]`, permute, first four lanes.
pub(crate) fn compress<B: Backend>(b: &mut B, left: Digest<B>, right: Digest<B>) -> Digest<B> {
    let zero = b.constant(0);
    let mut state = [zero; WIDTH];
    state[..DIGEST_LEN].copy_from_slice(&left);
    state[DIGEST_LEN..2 * DIGEST_LEN].copy_from_slice(&right);
    let state = b.permute(state);
    std::array::from_fn(|lane| state[lane])
}

/// The root of `leaf` at the index with little-endian `bits` along `path`
/// (bottom first): bit zero puts the node on the left.
pub(crate) fn merkle_root<B: Backend>(b: &mut B, leaf: Digest<B>, bits: &[B::F], path: &[Digest<B>]) -> Digest<B> {
    assert_eq!(bits.len(), path.len());
    let mut node = leaf;
    for (&bit, sibling) in bits.iter().zip(path) {
        let mut left = node;
        let mut right = node;
        for lane in 0..DIGEST_LEN {
            left[lane] = b.select(bit, node[lane], sibling[lane]);
            let both = b.add(node[lane], sibling[lane]);
            right[lane] = b.sub(both, left[lane]);
        }
        node = compress(b, left, right);
    }
    node
}

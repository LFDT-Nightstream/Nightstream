//! Honest setup oracles ("honest fold", docs/reviews/compression-akita/M2-DESIGN.md).
//!
//! Owns: the exact Reed-Solomon encoding of setup oracles, their Merkle tree
//! and on-disk store, the streaming root, and the honest fold: the prover's
//! partial evaluations and fold, and the verifier's leaf check. A root read
//! from disk is never authority; the verifier's key holds the root.
//!
//! Oracle `k` is a table of `2^n` values, low bit first. Its univariate form
//! `Ô(Y) = Σ_x c_x Y^x` uses the monomial coefficients `c` of the multilinear
//! extension, so `Õ(y, y^2, y^4, ...) = Ô(y)`. The codeword is `Ô` on the
//! coset `g·⟨ω⟩` with `|⟨ω⟩| = 2^{n+1}` (rate 1/2) and `g` the multiplicative
//! generator. Leaf `j < 2^{n-2}` holds, for every oracle in order, the eight
//! values at positions `j + v·2^{n-2}`: the coset `y_j·⟨ω_8⟩`, `y_j = g·ω^j`.

use std::fs::File;
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use neo_ccs::crypto::poseidon2_goldilocks::{RATE, WIDTH};
use p3_dft_v08::{Radix2DFTSmallBatch, TwoAdicSubgroupDft};
use p3_field_v08::{Field, PrimeCharacteristicRing, PrimeField64, TwoAdicField};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::circuit::algebra;
use crate::circuit::hash::{hash_leaf, merkle_root};
use crate::circuit::Backend;
use crate::field::{eq_table, Ext, Gl};
use crate::hash::{absorb, compress, squeeze};
use crate::Error;

/// Variables folded per proof; a leaf holds `2^FOLD` values per oracle.
pub(crate) const FOLD: usize = 3;
const LEAF: usize = 1 << FOLD;
const WORD: usize = 8;

pub(crate) type Digest = [Gl; 4];

/// One queried leaf: every oracle's `2^FOLD` coset values and the
/// authentication path, bottom first. A proof field.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub(crate) struct Leaf {
    pub(crate) values: Vec<Gl>,
    pub(crate) path: Vec<Digest>,
}

/// The prover's store of one setup: oracle codewords and the Merkle tree.
pub(crate) struct Store {
    dir: PathBuf,
    variables: usize,
    count: usize,
}

fn leaf_count(variables: usize) -> usize {
    1 << (variables + 1 - FOLD)
}

/// The root of `count` oracles of `2^variables` values; with `dir`, also
/// write the store. Holds one codeword and one sponge state per leaf.
pub(crate) fn build(
    variables: usize,
    count: usize,
    oracle: impl Fn(usize) -> Vec<Gl>,
    dir: Option<&Path>,
) -> Result<Digest, Error> {
    assert!(variables >= FOLD);
    let leaves = leaf_count(variables);
    let mut states = vec![[Gl::ZERO; WIDTH]; leaves];
    let mut at = 0;
    for k in 0..count {
        let leaf_major = encode(oracle(k), variables);
        if let Some(dir) = dir {
            write_words(&dir.join(format!("oracle-{k:02}.bin")), &leaf_major)?;
        }
        let start = at;
        states
            .par_iter_mut()
            .zip(leaf_major.par_chunks(LEAF))
            .for_each(|(state, values)| {
                absorb(state, start, values);
            });
        at = (at + LEAF) % RATE;
    }
    let mut layer: Vec<Digest> = states
        .par_iter_mut()
        .map(|state| squeeze(state, at))
        .collect();
    let mut file = match dir {
        Some(dir) => Some(BufWriter::new(File::create(dir.join("tree.bin"))?)),
        None => None,
    };
    loop {
        if let Some(file) = &mut file {
            for digest in &layer {
                for word in digest {
                    file.write_all(&word.as_canonical_u64().to_le_bytes())?;
                }
            }
        }
        if layer.len() == 1 {
            break;
        }
        layer = layer
            .par_chunks(2)
            .map(|pair| compress(pair[0], pair[1]))
            .collect();
    }
    if let Some(mut file) = file {
        file.flush()?;
    }
    Ok(layer[0])
}

/// The codeword of one oracle in leaf-major order: monomial coefficients
/// (Möbius transform), then the coset DFT at rate 1/2.
fn encode(mut values: Vec<Gl>, variables: usize) -> Vec<Gl> {
    assert_eq!(values.len(), 1 << variables);
    for t in 0..variables {
        values.par_chunks_mut(2 << t).for_each(|chunk| {
            let (low, high) = chunk.split_at_mut(1 << t);
            for (h, l) in high.iter_mut().zip(low.iter()) {
                *h -= *l;
            }
        });
    }
    values.resize(2 << variables, Gl::ZERO);
    let codeword = Radix2DFTSmallBatch::<Gl>::default().coset_dft(values, Gl::GENERATOR);
    let leaves = leaf_count(variables);
    let mut leaf_major = vec![Gl::ZERO; codeword.len()];
    leaf_major
        .par_chunks_mut(LEAF)
        .enumerate()
        .for_each(|(leaf, out)| {
            for (v, value) in out.iter_mut().enumerate() {
                *value = codeword[leaf + v * leaves];
            }
        });
    leaf_major
}

fn write_words(path: &Path, words: &[Gl]) -> Result<(), Error> {
    let mut file = BufWriter::new(File::create(path)?);
    for word in words {
        file.write_all(&word.as_canonical_u64().to_le_bytes())?;
    }
    file.flush()?;
    Ok(())
}

fn read_words(file: &mut File, offset: u64, count: usize) -> Result<Vec<Gl>, Error> {
    file.seek(SeekFrom::Start(offset))?;
    let mut bytes = vec![0u8; count * WORD];
    file.read_exact(&mut bytes)?;
    Ok(bytes
        .chunks_exact(WORD)
        .map(|word| Gl::new(u64::from_le_bytes(word.try_into().expect("one word"))))
        .collect())
}

impl Store {
    /// Open a store built for `root`. The stored root must equal it; this
    /// detects stale files, it does not make the files authoritative.
    pub(crate) fn open(dir: &Path, variables: usize, count: usize, root: &Digest) -> Result<Self, Error> {
        let store = Self {
            dir: dir.to_path_buf(),
            variables,
            count,
        };
        let mut tree = File::open(dir.join("tree.bin"))?;
        let nodes = 2 * leaf_count(variables) - 1;
        let stored = read_words(&mut tree, ((nodes - 1) * 4 * WORD) as u64, 4)?;
        if stored.as_slice() != root.as_slice() {
            return Err(Error::Setup("setup files do not match the key root"));
        }
        Ok(store)
    }

    /// The leaf at `index`: every oracle's coset values and the path.
    pub(crate) fn read(&self, index: usize) -> Result<Leaf, Error> {
        let mut values = Vec::with_capacity(self.count * LEAF);
        for k in 0..self.count {
            let mut file = File::open(self.dir.join(format!("oracle-{k:02}.bin")))?;
            values.extend(read_words(&mut file, (index * LEAF * WORD) as u64, LEAF)?);
        }
        let mut tree = File::open(self.dir.join("tree.bin"))?;
        let mut path = Vec::new();
        let (mut offset, mut width, mut node) = (0, leaf_count(self.variables), index);
        while width > 1 {
            let words = read_words(&mut tree, ((offset + (node ^ 1)) * 4 * WORD) as u64, 4)?;
            path.push(std::array::from_fn(|lane| words[lane]));
            offset += width;
            width /= 2;
            node /= 2;
        }
        Ok(Leaf { values, path })
    }
}

/// The fold point of the leaf with little-endian index `bits` as the power
/// point `(q, q^2, q^4, ...)` of the folded oracle (`variables - FOLD`
/// coordinates): `q = y^8 = g^8·ω_{2^{n-2}}^index`.
pub(crate) fn query_point<B: Backend>(b: &mut B, variables: usize, bits: &[B::F]) -> Vec<B::E> {
    let rotation = algebra::power_from_bits(b, Gl::two_adic_generator(variables + 1 - FOLD), bits);
    let mut q = b.scale(rotation, Gl::GENERATOR.exp_u64(LEAF as u64));
    (0..variables - FOLD)
        .map(|_| {
            let value = algebra::lift(b, q);
            q = b.mul(q, q);
            value
        })
        .collect()
}

/// One queried leaf as backend words, or zeros of its shape.
pub(crate) struct LeafView<B: Backend> {
    values: Vec<B::F>,
    path: Vec<[B::F; 4]>,
}

impl<B: Backend> LeafView<B> {
    pub(crate) fn read(b: &mut B, leaf: Option<&Leaf>, oracles: usize, variables: usize) -> Result<Self, Error> {
        let depth = variables + 1 - FOLD;
        if leaf.is_some_and(|leaf| leaf.values.len() != oracles * LEAF || leaf.path.len() != depth) {
            return Err(Error::Rejected("setup leaf shape"));
        }
        let values = (0..oracles * LEAF)
            .map(|i| b.private(leaf.map_or(Gl::ZERO, |leaf| leaf.values[i])))
            .collect();
        let path = (0..depth)
            .map(|level| std::array::from_fn(|lane| b.private(leaf.map_or(Gl::ZERO, |leaf| leaf.path[level][lane]))))
            .collect();
        Ok(Self { values, path })
    }
}

/// Check `leaf` at the index with little-endian `bits` against `root`. The
/// groups of per-oracle weights cover the oracles in order. For each group,
/// return the folded value `Σ_u α^[u]·Ô_u(q)` of the combined oracle: the
/// value at `q` of its fold by `alpha`.
pub(crate) fn check<B: Backend>(
    b: &mut B,
    root: [B::F; 4],
    variables: usize,
    bits: &[B::F],
    leaf: &LeafView<B>,
    groups: &[Vec<B::E>],
    alpha: &[B::E; FOLD],
) -> Result<Vec<B::E>, Error> {
    assert_eq!(LEAF * groups.iter().map(Vec::len).sum::<usize>(), leaf.values.len());
    let digest = hash_leaf(b, &leaf.values);
    let computed = merkle_root(b, digest, bits, &leaf.path);
    for (computed, root) in computed.into_iter().zip(root) {
        b.assert_equal(computed, root, "setup leaf path")?;
    }
    // Ô_u(q) = 8^{-1}·y^{-u}·Σ_v ω_8^{-uv}·Ô(y·ω_8^v), so the folded value is
    // Σ_v weight[v]·Ô(y·ω_8^v) with weight[v] = Σ_u α^[u]·8^{-1}·y^{-u}·ω_8^{-uv}.
    // y^{-1} = g^{-1}·ω^{-index}, with ω of order 2^{n+1}.
    let rotation = algebra::power_from_bits(b, Gl::two_adic_generator(variables + 1).inverse(), bits);
    let y_inverse = b.scale(rotation, Gl::GENERATOR.inverse());
    let mut monomials = vec![algebra::ext_one(b)];
    for &a in alpha {
        let scaled: Vec<B::E> = monomials.iter().map(|&m| b.ext_mul(m, a)).collect();
        monomials.extend(scaled);
    }
    let mut scale = b.constant(Gl::from_usize(LEAF).inverse());
    let mut coefficients = Vec::with_capacity(LEAF);
    for &monomial in &monomials {
        coefficients.push(b.ext_scale(monomial, scale));
        scale = b.mul(scale, y_inverse);
    }
    let omega = Gl::two_adic_generator(FOLD);
    let weight: Vec<B::E> = (0..LEAF)
        .map(|v| {
            let mut total = algebra::ext_zero(b);
            for (u, &coefficient) in coefficients.iter().enumerate() {
                let term = algebra::ext_scale_constant(b, coefficient, omega.exp_u64(((LEAF - u) * v % LEAF) as u64));
                total = b.ext_add(total, term);
            }
            total
        })
        .collect();
    let mut oracles = leaf.values.chunks_exact(LEAF);
    let mut folded = Vec::with_capacity(groups.len());
    for weights in groups {
        // Σ_k w_k·Ô_k(y·ω_8^v) per v, then the weights above.
        let mut combined = vec![algebra::ext_zero(b); LEAF];
        for &w in weights {
            let values = oracles.next().expect("the groups cover the oracles");
            for (total, &value) in combined.iter_mut().zip(values) {
                let term = b.ext_scale(w, value);
                *total = b.ext_add(*total, term);
            }
        }
        folded.push(algebra::dot(b, &weight, &combined));
    }
    Ok(folded)
}

/// `w_u = T~(u, rest)` for `u < 2^FOLD`, with `T` low bit first.
pub(crate) fn partials(table: &[Ext], rest: &[Ext]) -> [Ext; LEAF] {
    let eq = eq_table(rest);
    assert_eq!(table.len(), LEAF * eq.len());
    std::array::from_fn(|u| {
        eq.par_iter()
            .enumerate()
            .map(|(y, &w)| w * table[u + LEAF * y])
            .sum()
    })
}

/// The fold `g[y] = Σ_u eq(α, u)·T[u + 2^FOLD·y]`, the table of `T~(α, ·)`.
pub(crate) fn fold(table: &[Ext], alpha: &[Ext; FOLD]) -> Vec<Ext> {
    let weights = eq_table(alpha);
    table
        .par_chunks(LEAF)
        .map(|chunk| chunk.iter().zip(&weights).map(|(&v, &w)| v * w).sum())
        .collect()
}

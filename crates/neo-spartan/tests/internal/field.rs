use neo_verified::algebra;
use neo_verified::field as verified;
use neo_verified::native::Native;
use p3_field_v08::{BasedVectorSpace, Field, PrimeCharacteristicRing, PrimeField64};

use super::word;
use crate::field::{eq_table, ext_words, exts_to_verified, from_verified, from_word, to_verified, Ext, Gl};

const P: u64 = verified::MODULUS;
const EPSILON: u64 = 0xffff_ffff;

/// Canonical words at the edges of the reductions.
const EDGES: [u64; 12] = [
    0,
    1,
    2,
    EPSILON - 1,
    EPSILON,
    EPSILON + 1,
    1 << 32,
    1 << 63,
    (1 << 63) + 1,
    P - EPSILON,
    P - 2,
    P - 1,
];

/// Words that the Metal kernels of `dft_columns`, `gkr_layer` and `key_weights` write back
/// (`shaders/nightstream.metal`, `dft.metal`, `gkr.metal`): `gl_from_word(u64::MAX)`,
/// `gl_sub(0, 1)`, `gl_add(p - 1, p - 1)` and `gl_add(p - 1, 1)`, all canonical, and the words in
/// `[p, 2^64)` that a fold leaves in place as the CPU wrote them. The CUDA crate writes none.
const DEVICE_WORDS: [u64; 9] = [
    EPSILON - 1,
    P - 1,
    P - 2,
    0,
    P,
    P + 1,
    P + (1 << 31),
    u64::MAX - 1,
    u64::MAX,
];

fn randoms(seed: u64) -> Vec<u64> {
    (0..200).map(|i| word(seed, i) % P).collect()
}

fn words() -> Vec<u64> {
    let mut words = EDGES.to_vec();
    words.extend(randoms(0x0066_6965_6c64));
    words
}

fn plonky3(w: u64) -> Gl {
    Gl::new(w)
}

fn ext_in(value: Ext) -> verified::Ext {
    let [c0, c1, c2] = ext_words(value).map(to_verified);
    verified::Ext { c0, c1, c2 }
}

fn ext_out(value: verified::Ext) -> Ext {
    let coordinates = [value.c0, value.c1, value.c2];
    Ext::from_basis_coefficients_fn(|i| from_verified(coordinates[i]))
}

fn ext(seed: u64, index: u64) -> Ext {
    Ext::from_basis_coefficients_fn(|i| plonky3(word(seed, 3 * index + i as u64) % P))
}

/// A verified result must hold Plonky3's value as its canonical word: `==` on
/// `verified::Gl` compares the stored words.
fn same(verified: verified::Gl, plonky3: Gl) -> bool {
    verified == to_verified(plonky3) && from_verified(verified) == plonky3
}

fn same_ext(verified: verified::Ext, plonky3: Ext) -> bool {
    verified == ext_in(plonky3) && ext_out(verified) == plonky3
}

#[test]
fn verified_goldilocks_matches_plonky3() {
    let words = words();
    let exponents = [0, 1, 2, 7, 63, 64, P - 2, P - 1, u64::MAX, word(7, 0), word(7, 1)];
    for (i, &x) in words.iter().enumerate() {
        let a = plonky3(x);
        let va = to_verified(a);
        assert!(same(va, a), "conversion of {x}");
        assert!(same(verified::gl_neg(va), -a), "-{x}");
        if x != 0 {
            assert!(same(verified::gl_inv(va), a.inverse()), "{x}^-1");
        }
        for e in exponents {
            assert!(same(verified::gl_pow(va, e), a.exp_u64(e)), "{x}^{e}");
        }
        let partners: Vec<u64> = if i < EDGES.len() {
            words.clone()
        } else {
            vec![words[(i * 7) % words.len()]]
        };
        for y in partners {
            let b = plonky3(y);
            let vb = to_verified(b);
            assert!(same(verified::gl_add(va, vb), a + b), "{x} + {y}");
            assert!(same(verified::gl_sub(va, vb), a - b), "{x} - {y}");
            assert!(same(verified::gl_mul(va, vb), a * b), "{x} * {y}");
        }
    }
    assert!(same(verified::gl_inv(verified::GL_ZERO), Gl::ZERO), "0^-1 is 0");
}

#[test]
fn verified_extension_matches_plonky3() {
    let exponents = [0, 1, 2, 7, 64, P - 2, u64::MAX, word(8, 0)];
    let edge = |i: usize| Ext::from_basis_coefficients_fn(|c| plonky3(EDGES[(i + 5 * c) % EDGES.len()]));
    let values: Vec<Ext> = (0..EDGES.len())
        .map(edge)
        .chain((0..60).map(|i| ext(0x0065_7874, i)))
        .collect();
    for (i, &a) in values.iter().enumerate() {
        let va = ext_in(a);
        assert!(same_ext(va, a), "conversion of value {i}");
        assert!(same_ext(verified::ext_neg(va), -a), "-value {i}");
        let inverse = a.try_inverse().unwrap_or(Ext::ZERO);
        assert!(same_ext(verified::ext_inv(va), inverse), "value {i}^-1");
        for e in exponents {
            assert!(same_ext(verified::ext_pow(va, e), a.exp_u64(e)), "value {i}^{e}");
        }
        let g = plonky3(EDGES[i % EDGES.len()]);
        assert!(
            same_ext(verified::ext_scale(va, to_verified(g)), a * g),
            "value {i} scaled"
        );
        assert!(
            same_ext(verified::ext_of_gl(to_verified(g)), Ext::from(g)),
            "lift of {g}"
        );
        for &b in &values {
            let vb = ext_in(b);
            assert!(same_ext(verified::ext_add(va, vb), a + b), "sum with value {i}");
            assert!(same_ext(verified::ext_sub(va, vb), a - b), "difference with value {i}");
            assert!(same_ext(verified::ext_mul(va, vb), a * b), "product with value {i}");
        }
    }
    assert!(same_ext(verified::EXT_ZERO, Ext::ZERO) && same_ext(verified::EXT_ONE, Ext::ONE));
}

/// A Plonky3 value may hold any word; the conversion reads its canonical word, as
/// `as_canonical_u64` does, and only `canonical` refuses a word of `p` or more.
#[test]
fn conversion_reads_non_canonical_words_by_their_value() {
    let mut words = DEVICE_WORDS.to_vec();
    words.extend(EDGES);
    words.extend((0..200).map(|i| word(0x6e6f_6e63, i)));
    words.extend((0..64).map(|i| P + (word(0x6869_6768, i) % (u64::MAX - P + 1))));
    for w in words {
        let value = plonky3(w);
        let converted = to_verified(value);
        assert_eq!(verified::gl_word(converted), w % P, "word {w}");
        assert!(from_verified(converted) == value, "round trip of {w}");
        assert_eq!(from_verified(converted).as_canonical_u64(), w % P, "word {w}");
        assert!(same(verified::reduce(w), Gl::from_u64(w)), "reduce {w}");
        assert!(
            from_word(w) == value && from_word(w).as_canonical_u64() == w % P,
            "from_word {w}"
        );
        match verified::canonical(w) {
            Some(g) => assert!(w < P && verified::gl_word(g) == w, "canonical {w}"),
            None => assert!(w >= P, "canonical refused {w}"),
        }
    }
}

/// The prover's parallel `eq_table` against the one that the verifier runs, word for word.
#[test]
fn parallel_eq_table_matches_the_verified_one() {
    let edges: Vec<Ext> = EDGES
        .chunks_exact(3)
        .map(|w| Ext::from_basis_coefficients_fn(|i| plonky3(w[i])))
        .collect();
    let mut points: Vec<Vec<Ext>> = (0..=10)
        .map(|len| (0..len).map(|i| ext(0x6571_7461, i)).collect())
        .collect();
    points.push(edges);
    for point in points {
        let parallel = eq_table(&point);
        let verified = algebra::eq_table(&mut Native, &exts_to_verified(&point));
        assert_eq!(parallel.len(), 1 << point.len());
        assert_eq!(verified.len(), parallel.len());
        for (index, (&v, &p)) in verified.iter().zip(&parallel).enumerate() {
            assert!(same_ext(v, p), "point of {} coordinates, index {index}", point.len());
        }
    }
}

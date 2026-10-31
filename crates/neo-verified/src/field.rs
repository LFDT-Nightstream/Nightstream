// Provenance: adapted from dcSpark/ProtogaLattice at commit 96b57a1c801b,
// `crates/pgl-arith/src/fq.rs` (the canonical word type, `from_canonical`, `reduce`, `pow`
// and `inv`) and `crates/pgl-arith/src/e.rs` (the extension, there `u^2 - 7`), Apache-2.0.
// The word arithmetic is not `fq.rs`'s masks: it is `add64`, `sub64` and `mul64` of
// `formal/nightstream-fprime/NightstreamFPrime/Export/NativeGoldilocks.lean`, step for step,
// so the proofs reuse that file's lemmas. The verifier handles public data only.

//! Goldilocks `Gl` (`p = 2^64 - 2^32 + 1`) and `Ext = Gl[w] / (w^3 - w - 1)`, the layer-1
//! fields, with the operations that the `neo-spartan` verifier uses.
//!
//! Every `Gl` holds its canonical word: `canonical` and `reduce` are the only ways in. The
//! Lean translation reads a `Gl` as any word, so each function reduces the words it reads.
//! `formal/compression-lean/CompressionLeanRust/Field.lean` proves each operation equal to its
//! copy in `Ports/Replay/Field.lean`, and to the operation of `ZMod p` or `AdjoinRoot`.

/// The Goldilocks prime.
pub const MODULUS: u64 = 0xffff_ffff_0000_0001;

/// A Goldilocks element, stored as its canonical word, below `MODULUS`.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Gl(u64);

pub const GL_ZERO: Gl = Gl(0);
pub const GL_ONE: Gl = Gl(1);

/// `word` if it is canonical.
pub fn canonical(word: u64) -> Option<Gl> {
    if word < MODULUS {
        Some(Gl(word))
    } else {
        None
    }
}

/// `word mod p`.
pub fn reduce(word: u64) -> Gl {
    Gl(reduce_word(word))
}

/// The canonical word of `a`.
pub fn gl_word(a: Gl) -> u64 {
    reduce_word(a.0)
}

/// `canonicalize64`: `word mod p`, since every word is below `2p`.
fn reduce_word(word: u64) -> u64 {
    if word < MODULUS {
        word
    } else {
        word.wrapping_sub(MODULUS)
    }
}

/// `add64`, for canonical words.
fn add_words(a: u64, b: u64) -> u64 {
    let room = MODULUS.wrapping_sub(b);
    if a < room {
        a.wrapping_add(b)
    } else {
        a.wrapping_sub(room)
    }
}

/// `sub64`, for canonical words.
fn sub_words(a: u64, b: u64) -> u64 {
    if b <= a {
        a.wrapping_sub(b)
    } else {
        MODULUS.wrapping_sub(b.wrapping_sub(a))
    }
}

/// `low64`: the low 32 bits.
fn low_half(word: u64) -> u64 {
    word as u32 as u64
}

/// `high64`: the high 32 bits.
fn high_half(word: u64) -> u64 {
    word.wrapping_shr(32)
}

/// `foldHigh64`: `2^64·word mod p`, as `(2^32 - 1)·low - high`, since `2^64 = 2^32 - 1` and
/// `2^96 = -1` mod p.
fn fold_high(word: u64) -> u64 {
    let low = low_half(word);
    sub_words(low.wrapping_shl(32).wrapping_sub(low), high_half(word))
}

/// `mul64`: the 128-bit product from four 32-bit limb products, low and high word, then
/// `reduceWide64`. Right for any words.
fn mul_words(a: u64, b: u64) -> u64 {
    let a0 = low_half(a);
    let a1 = high_half(a);
    let b0 = low_half(b);
    let b1 = high_half(b);
    let p00 = a0.wrapping_mul(b0);
    let p10 = a1.wrapping_mul(b0);
    let t0 = p10.wrapping_add(high_half(p00));
    let p01 = a0.wrapping_mul(b1);
    let t1 = p01.wrapping_add(low_half(t0));
    let low = low_half(t1).wrapping_shl(32).wrapping_add(low_half(p00));
    let p11 = a1.wrapping_mul(b1);
    let high = p11.wrapping_add(high_half(t0)).wrapping_add(high_half(t1));
    add_words(reduce_word(low), fold_high(high))
}

pub fn gl_add(a: Gl, b: Gl) -> Gl {
    Gl(add_words(reduce_word(a.0), reduce_word(b.0)))
}

pub fn gl_sub(a: Gl, b: Gl) -> Gl {
    Gl(sub_words(reduce_word(a.0), reduce_word(b.0)))
}

pub fn gl_neg(a: Gl) -> Gl {
    gl_sub(GL_ZERO, a)
}

pub fn gl_mul(a: Gl, b: Gl) -> Gl {
    Gl(mul_words(reduce_word(a.0), reduce_word(b.0)))
}

/// `a^e`, halving `e` as `Word.pow` does.
pub fn gl_pow(a: Gl, e: u64) -> Gl {
    if e == 0 {
        GL_ONE
    } else {
        let half = gl_pow(gl_mul(a, a), e.wrapping_shr(1));
        if e.is_multiple_of(2) {
            half
        } else {
            gl_mul(half, a)
        }
    }
}

/// `a^(p - 2)`: the inverse of a nonzero `a`, and zero for zero.
pub fn gl_inv(a: Gl) -> Gl {
    gl_pow(a, MODULUS.wrapping_sub(2))
}

/// Plonky3's `two_adic_generator(bits)`: its generator of order `2^32`, squared `32 - bits`
/// times. Panics for `bits` above 32, as Plonky3 does.
pub fn two_adic_generator(bits: usize) -> Gl {
    let mut squarings = 32usize.checked_sub(bits).unwrap();
    let mut g = Gl(0x1856_29dc_da58_878c);
    while squarings > 0 {
        g = gl_mul(g, g);
        squarings = squarings.wrapping_sub(1);
    }
    g
}

/// `c0 + c1·w + c2·w^2` with `w^3 = w + 1`.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Ext {
    pub c0: Gl,
    pub c1: Gl,
    pub c2: Gl,
}

pub const EXT_ZERO: Ext = Ext {
    c0: GL_ZERO,
    c1: GL_ZERO,
    c2: GL_ZERO,
};

pub const EXT_ONE: Ext = Ext {
    c0: GL_ONE,
    c1: GL_ZERO,
    c2: GL_ZERO,
};

/// `a` with canonical words.
pub fn ext_reduce(a: Ext) -> Ext {
    Ext {
        c0: Gl(reduce_word(a.c0.0)),
        c1: Gl(reduce_word(a.c1.0)),
        c2: Gl(reduce_word(a.c2.0)),
    }
}

pub fn ext_of_gl(a: Gl) -> Ext {
    Ext {
        c0: Gl(reduce_word(a.0)),
        c1: GL_ZERO,
        c2: GL_ZERO,
    }
}

pub fn ext_add(a: Ext, b: Ext) -> Ext {
    Ext {
        c0: gl_add(a.c0, b.c0),
        c1: gl_add(a.c1, b.c1),
        c2: gl_add(a.c2, b.c2),
    }
}

pub fn ext_sub(a: Ext, b: Ext) -> Ext {
    Ext {
        c0: gl_sub(a.c0, b.c0),
        c1: gl_sub(a.c1, b.c1),
        c2: gl_sub(a.c2, b.c2),
    }
}

pub fn ext_neg(a: Ext) -> Ext {
    ext_sub(EXT_ZERO, a)
}

/// The schoolbook product with `w^3 = w + 1` and `w^4 = w^2 + w`.
pub fn ext_mul(a: Ext, b: Ext) -> Ext {
    let d3 = gl_add(gl_mul(a.c1, b.c2), gl_mul(a.c2, b.c1));
    let d4 = gl_mul(a.c2, b.c2);
    Ext {
        c0: gl_add(gl_mul(a.c0, b.c0), d3),
        c1: gl_add(gl_add(gl_add(gl_mul(a.c0, b.c1), gl_mul(a.c1, b.c0)), d3), d4),
        c2: gl_add(
            gl_add(gl_add(gl_mul(a.c0, b.c2), gl_mul(a.c1, b.c1)), gl_mul(a.c2, b.c0)),
            d4,
        ),
    }
}

pub fn ext_scale(a: Ext, by: Gl) -> Ext {
    Ext {
        c0: gl_mul(a.c0, by),
        c1: gl_mul(a.c1, by),
        c2: gl_mul(a.c2, by),
    }
}

/// `a^e`, halving `e` as `Ext.pow` does.
pub fn ext_pow(a: Ext, e: u64) -> Ext {
    if e == 0 {
        EXT_ONE
    } else {
        let half = ext_pow(ext_mul(a, a), e.wrapping_shr(1));
        if e.is_multiple_of(2) {
            half
        } else {
            ext_mul(half, a)
        }
    }
}

/// The inverse of a nonzero `a`, and zero for zero: the first column of the inverse of the
/// matrix of multiplication by `a`, whose columns are `a`, `a·w = (a2, a0 + a2, a1)` and
/// `a·w^2 = (a1, a1 + a2, a0 + a2)`, as cofactors over the determinant.
pub fn ext_inv(a: Ext) -> Ext {
    let s = gl_add(a.c0, a.c2);
    let t = gl_add(a.c1, a.c2);
    let c0 = gl_sub(gl_mul(s, s), gl_mul(t, a.c1));
    let c1 = gl_sub(gl_mul(t, a.c2), gl_mul(a.c1, s));
    let c2 = gl_sub(gl_mul(a.c1, a.c1), gl_mul(s, a.c2));
    let det = gl_add(gl_add(gl_mul(a.c0, c0), gl_mul(a.c2, c1)), gl_mul(a.c1, c2));
    let scale = gl_inv(det);
    Ext {
        c0: gl_mul(c0, scale),
        c1: gl_mul(c1, scale),
        c2: gl_mul(c2, scale),
    }
}

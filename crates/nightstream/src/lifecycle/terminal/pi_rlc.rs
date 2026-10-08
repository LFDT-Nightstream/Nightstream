//! PiRLC over any backend: the challenge draws and the exact parent claim
//! `Σ_i ρ_i ⋆ source_i`.
//!
//! Owns: the sampler of `neo_reductions::common::sample_rot_rhos_n` (absorb
//! `[4, i]`, squeeze four words, take the low 54 base-5 digits of the
//! 256-bit integer `Σ_l d_l·p^l`, minus two) and the ring products of every
//! source vector.
//!
//! The digits are checked as integers in 16-bit limbs:
//! `Σ_j a_j·5^j + q·5^54 = Σ_l d_l·p^l` with `a_j < 5`, 16-bit limbs of `q`,
//! the canonical bits of `d_l`, and signed carries below `2^21` per limb.
//! Every limb coefficient stays below `2^37`, so no step wraps modulo p.

use neo_math::{D, F};
use neo_spartan::{Backend, Error, FoldTranscript};
use p3_field::PrimeField64;

use super::k::Kw;
use super::words::Evaluations;

const LIMB_BITS: usize = 16;
/// Limb positions of the equation: `d_3·p^3` reaches `2^256`.
const POSITIONS: usize = 16;
/// Limbs of the quotient `q < p^4 / 5^54 < 2^131`.
const QUOTIENT_LIMBS: usize = 9;
/// Carries lie in `(-2^21, 2^21)`.
const CARRY_BITS: usize = 22;
const CARRY_OFFSET: u64 = 1 << (CARRY_BITS - 1);

/// `value` in 16-bit limbs, low first.
fn limbs(mut value: u128) -> Vec<u64> {
    let mut limbs = Vec::new();
    while value != 0 {
        limbs.push((value & 0xffff) as u64);
        value >>= LIMB_BITS;
    }
    limbs
}

/// `p^l` in base 2^16 for `l < 4`: `p = X^4 - X^2 + 1` with `X = 2^16`.
fn prime_powers() -> [Vec<i64>; 4] {
    let p = [1i64, 0, -1, 0, 1];
    let mut powers = [vec![1i64], Vec::new(), Vec::new(), Vec::new()];
    for l in 1..4 {
        let mut next = vec![0i64; powers[l - 1].len() + 4];
        for (i, &a) in powers[l - 1].iter().enumerate() {
            for (j, &b) in p.iter().enumerate() {
                next[i + j] += a * b;
            }
        }
        powers[l] = next;
    }
    powers
}

/// The base-5 digits and the quotient of `Σ_l d_l·p^l`, as hint words.
fn decode_hint(digest: &[u64]) -> Vec<u64> {
    const MODULUS: u128 = F::ORDER_U64 as u128;
    let mut integer = [0u64; 4];
    for &lane in digest.iter().rev() {
        let mut carry = u128::from(lane);
        for word in &mut integer {
            let product = u128::from(*word) * MODULUS + carry;
            *word = product as u64;
            carry = product >> 64;
        }
    }
    let mut words = Vec::with_capacity(D + QUOTIENT_LIMBS);
    for _ in 0..D {
        let mut remainder = 0u128;
        for word in integer.iter_mut().rev() {
            let dividend = (remainder << 64) | u128::from(*word);
            *word = (dividend / 5) as u64;
            remainder = dividend % 5;
        }
        words.push(remainder as u64);
    }
    // What remains is the quotient by 5^54.
    words.extend((0..QUOTIENT_LIMBS).map(|k| (integer[k / 4] >> (LIMB_BITS * (k % 4))) & 0xffff));
    words
}

/// `Σ_k value_k·2^{16k}` of bit forms.
fn limb_of<B: Backend>(b: &mut B, bits: &[B::F]) -> B::F {
    let mut total = b.constant(0);
    for (t, &bit) in bits.iter().enumerate() {
        let weighted = b.scale(bit, 1 << t);
        total = b.add(total, weighted);
    }
    total
}

/// `acc[at] += coefficient·term` for a signed integer coefficient.
fn add_term<B: Backend>(b: &mut B, acc: &mut [B::F], at: usize, term: B::F, coefficient: i64) {
    let magnitude = b.scale(term, coefficient.unsigned_abs());
    acc[at] = if coefficient < 0 {
        b.sub(acc[at], magnitude)
    } else {
        b.add(acc[at], magnitude)
    };
}

/// One challenge `ρ` from four squeezed words: its 54 coefficients.
pub(super) fn decode<B: Backend>(b: &mut B, digest: [B::F; 4]) -> Result<[B::F; D], Error> {
    let hinted = b.hint(&digest, D + QUOTIENT_LIMBS, &decode_hint);
    let (digits, quotient) = hinted.split_at(D);
    let mut equation = vec![b.constant(0); POSITIONS];
    // Left side: Σ_j a_j·5^j, each digit below 5.
    let mut five = 1u128;
    for &digit in digits {
        let bits = b.bits(digit, 3, "PiRLC digit")?;
        let low = b.add(bits[0], bits[1]);
        let over = b.mul(bits[2], low);
        b.assert_zero(over, "PiRLC digit below five")?;
        for (k, limb) in limbs(five).into_iter().enumerate() {
            add_term(b, &mut equation, k, digit, limb as i64);
        }
        five *= 5;
    }
    // q·5^54, with 16-bit limbs of q.
    for (l, &limb) in quotient.iter().enumerate() {
        b.bits(limb, LIMB_BITS, "PiRLC quotient limb")?;
        for (k, power) in limbs(five).into_iter().enumerate() {
            add_term(b, &mut equation, l + k, limb, power as i64);
        }
    }
    // Right side: Σ_l d_l·p^l from the canonical bits of each word.
    let powers = prime_powers();
    for (l, &word) in digest.iter().enumerate() {
        let bits = b.bits(word, 64, "PiRLC digest bits")?;
        for (m, chunk) in bits.chunks(LIMB_BITS).enumerate() {
            let limb = limb_of(b, chunk);
            for (i, &coefficient) in powers[l].iter().enumerate() {
                if coefficient != 0 {
                    add_term(b, &mut equation, m + i, limb, -coefficient);
                }
            }
        }
    }
    // Carries: E_k + c_{k-1} = 2^16·c_k, and the last position closes.
    let carries = b.hint(&equation, POSITIONS - 1, &|values| {
        let p = F::ORDER_U64 as i128;
        let signed = |v: u64| if v > F::ORDER_U64 / 2 { v as i128 - p } else { v as i128 };
        let mut carry = 0i128;
        values[..POSITIONS - 1]
            .iter()
            .map(|&v| {
                carry = (signed(v) + carry) >> LIMB_BITS;
                (carry.rem_euclid(p)) as u64
            })
            .collect()
    });
    let mut previous = b.constant(0);
    for k in 0..POSITIONS {
        let total = b.add(equation[k], previous);
        if k + 1 == POSITIONS {
            b.assert_zero(total, "PiRLC decoding")?;
            break;
        }
        let carry = carries[k];
        let shifted = b.scale(carry, 1 << LIMB_BITS);
        b.assert_equal(total, shifted, "PiRLC decoding")?;
        let offset = b.constant(CARRY_OFFSET);
        let biased = b.add(carry, offset);
        b.bits(biased, CARRY_BITS, "PiRLC carry")?;
        previous = carry;
    }
    let two = b.constant(2);
    Ok(std::array::from_fn(|j| b.sub(digits[j], two)))
}

/// One PiRLC source: its commitment rows, public input columns and
/// evaluations.
pub(super) struct Source<'a, B: Backend> {
    pub(super) commitment: &'a [[B::F; D]],
    pub(super) public: Vec<[B::F; D]>,
    pub(super) evaluations: &'a Evaluations<B>,
}

/// The parent claim's words.
pub(super) struct Parent<B: Backend> {
    pub(super) commitment: Vec<[B::F; D]>,
    pub(super) public: Vec<[B::F; D]>,
    pub(super) eval_k: [Kw<B>; D],
    pub(super) eval_a: Vec<[Kw<B>; D]>,
}

/// `Σ_i ρ_i ⋆ v_i` for one vector of every source.
fn mix<B: Backend>(b: &mut B, rhos: &[[B::F; D]], vectors: &[[B::F; D]]) -> [B::F; D] {
    let mut total = [b.constant(0); D];
    for (rho, vector) in rhos.iter().zip(vectors) {
        let product = b.ring_mul(rho, vector);
        for (sum, value) in total.iter_mut().zip(product) {
            *sum = b.add(*sum, value);
        }
    }
    total
}

/// Draw `ρ_i` for every source and combine them into the parent claim.
pub(super) fn parent<B: Backend>(
    b: &mut B,
    transcript: &mut FoldTranscript<B>,
    sources: &[Source<'_, B>],
) -> Result<Parent<B>, Error> {
    let mut rhos = Vec::with_capacity(sources.len());
    for i in 0..sources.len() {
        transcript.absorb_constants(b, &[4, i as u64]);
        let digest = transcript.squeeze_digest(b);
        rhos.push(decode(b, digest)?);
    }
    let rings = |pick: &dyn Fn(&Source<'_, B>) -> [B::F; D]| -> Vec<[B::F; D]> { sources.iter().map(pick).collect() };
    let commitment = (0..sources[0].commitment.len())
        .map(|row| mix(b, &rhos, &rings(&|source| source.commitment[row])))
        .collect();
    let public = (0..sources[0].public.len())
        .map(|column| mix(b, &rhos, &rings(&|source| source.public[column])))
        .collect();
    let mix_k = |b: &mut B, pick: &dyn Fn(&Evaluations<B>) -> [Kw<B>; D]| -> [Kw<B>; D] {
        let parts: [[B::F; D]; 2] = std::array::from_fn(|part| {
            let vectors: Vec<[B::F; D]> = sources
                .iter()
                .map(|source| pick(source.evaluations).map(|value| value[part]))
                .collect();
            mix(b, &rhos, &vectors)
        });
        std::array::from_fn(|lane| [parts[0][lane], parts[1][lane]])
    };
    let eval_k = mix_k(b, &|evaluations| evaluations.eval_k);
    let eval_a = (0..sources[0].evaluations.eval_a.len())
        .map(|matrix| mix_k(b, &|evaluations| evaluations.eval_a[matrix]))
        .collect();
    Ok(Parent {
        commitment,
        public,
        eval_k,
        eval_a,
    })
}

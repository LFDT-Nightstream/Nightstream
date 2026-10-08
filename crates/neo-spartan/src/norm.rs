//! The norm conjunct's public side: the histogram of committed values over
//! `[-H, H]` and its fraction sum `Σ_t m_t / (β - t)`.
//!
//! The histogram covers every cell of the committed cube, padding included.
//! If one cell holds a value outside `[-H, H]`, the leaf side of logUp has a
//! pole there with a nonzero residue (its count is below `p`), and the table
//! side has none; the two sides then agree at `β` only with negligible
//! probability.

use p3_field::PrimeField64;
use p3_field_v08::PrimeCharacteristicRing;

use crate::circuit::{algebra, Backend};
use crate::field::{signed, Ext, Gl};
use crate::Error;

/// Count each value of `z` in `[-bound, bound]`, slot `t + bound`.
pub(crate) fn histogram(z: &[Gl], bound: u32) -> Result<Vec<u32>, Error> {
    let mut counts = vec![0u32; 2 * bound as usize + 1];
    for &value in z {
        let value = p3_field_v08::PrimeField64::as_canonical_u64(&value);
        let half = neo_math::F::ORDER_U64 / 2;
        let centered = if value <= half {
            value as i64
        } else {
            -((neo_math::F::ORDER_U64 - value) as i64)
        };
        if centered.unsigned_abs() > u64::from(bound) {
            return Err(Error::Witness("a witness value exceeds the norm bound"));
        }
        counts[(centered + i64::from(bound)) as usize] += 1;
    }
    Ok(counts)
}

/// `Σ_t m_t / (β - t)` over `t ∈ [-bound, bound]`. Rejects when `β` is in
/// the table, an event of probability `(2·bound + 1) / |Ext|`.
pub(crate) fn table_sum<B: Backend>(b: &mut B, counts: &[B::F], bound: u32, beta: B::E) -> Result<B::E, Error> {
    assert_eq!(counts.len(), 2 * bound as usize + 1);
    let mut total = algebra::ext_zero(b);
    for (slot, &count) in counts.iter().enumerate() {
        let value = b.ext_constant(Ext::from(signed(slot as i64 - i64::from(bound))));
        let denominator = b.ext_sub(beta, value);
        let inverse = b.ext_inverse(denominator, "histogram pole")?;
        let term = b.ext_scale(inverse, count);
        total = b.ext_add(total, term);
    }
    Ok(total)
}

/// The histogram as transcript words.
pub(crate) fn words(counts: &[u32]) -> Vec<Gl> {
    counts.iter().map(|&count| Gl::from_u32(count)).collect()
}

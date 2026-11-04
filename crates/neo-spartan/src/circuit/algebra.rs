//! The field helpers of `neo_verified::algebra`, with layer-1 values converted at the
//! boundary: each helper that takes a field value here takes a Plonky3 0.8 value and passes
//! its canonical words on.

use neo_verified::algebra as verified;
pub(crate) use neo_verified::algebra::{
    dot, eq_at, eq_eval, eq_table, ext_one, ext_zero, horner, interpolate, lift_word, one_minus, Kx, K,
};

use super::Backend;
use crate::field::{ext_to_verified, to_verified, Ext, Gl};

/// A constant from a layer-1 field value.
pub(crate) fn constant<B: Backend>(b: &mut B, value: Gl) -> B::F {
    verified::constant(b, to_verified(value))
}

/// A proof word from a layer-1 field value.
pub(crate) fn private<B: Backend>(b: &mut B, value: Gl) -> B::F {
    verified::private(b, to_verified(value))
}

/// `a·by` for a layer-1 field constant.
pub(crate) fn scale<B: Backend>(b: &mut B, a: B::F, by: Gl) -> B::F {
    verified::scale(b, a, to_verified(by))
}

/// An extension constant from a layer-1 value.
pub(crate) fn ext_constant<B: Backend>(b: &mut B, value: Ext) -> B::E {
    verified::ext_constant(b, ext_to_verified(value))
}

/// An extension proof value, as three proof words.
pub(crate) fn private_ext<B: Backend>(b: &mut B, value: Ext) -> B::E {
    verified::private_ext(b, ext_to_verified(value))
}

pub(crate) fn ext_scale_constant<B: Backend>(b: &mut B, a: B::E, by: Gl) -> B::E {
    verified::ext_scale_constant(b, a, to_verified(by))
}

/// `generator^index` from the little-endian bits of `index`.
pub(crate) fn power_from_bits<B: Backend>(b: &mut B, generator: Gl, bits: &[B::F]) -> B::F {
    verified::power_from_bits(b, to_verified(generator), bits)
}

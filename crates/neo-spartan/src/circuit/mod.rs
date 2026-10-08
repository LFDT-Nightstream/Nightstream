//! One verifier, two readings.
//!
//! Owns: the `Backend` contract that compression verifiers are written in,
//! and `Native`, the reading on plain values. `Recorder` (record.rs) reads
//! the same code as a constraint system. `Backend` and `Native` are public so
//! that a caller can write its own verifier (nightstream's final program);
//! every value enters as a `u64`, so no Plonky3 0.8 type crosses.
//!
//! Rules for code written over `Backend`:
//! - It never branches on a value. Every loop bound and length comes from a
//!   key or a fixed shape. No method returns a value to the caller.
//! - Proof words enter as `private`, statement words as `public`. They have
//!   no authority until a later check uses them.
//! - `hint` outputs are unconstrained; the caller must constrain each one.
//! - `Native` returns the first failed check as `Err`. `Recorder` notes it and
//!   continues, so the rows it records never depend on the values.

pub(crate) mod algebra;
pub(crate) mod block;
pub(crate) mod hash;
// Until the shrink prover (M3 slice 6) records the verifier, only tests do.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) mod poseidon2;
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) mod record;
pub(crate) mod ring_mul;

use neo_ccs::crypto::poseidon2_goldilocks::WIDTH;
use neo_math::D;
use p3_field_v08::{BasedVectorSpace, Field, PrimeCharacteristicRing, PrimeField64};
use p3_symmetric_v08::Permutation;

use crate::field::{Ext, Gl};
use crate::Error;

pub trait Backend {
    /// A Goldilocks value or wire.
    type F: Copy;
    /// A cubic-extension value or wire (three base coordinates).
    type E: Copy;

    /// A constant. Every `u64` input to this trait is taken modulo p.
    fn constant(&mut self, value: u64) -> Self::F;
    /// A proof word.
    fn private(&mut self, value: u64) -> Self::F;
    /// The next statement word.
    fn public(&mut self, value: u64) -> Self::F;

    fn add(&mut self, a: Self::F, b: Self::F) -> Self::F;
    fn sub(&mut self, a: Self::F, b: Self::F) -> Self::F;
    fn scale(&mut self, a: Self::F, by: u64) -> Self::F;
    fn mul(&mut self, a: Self::F, b: Self::F) -> Self::F;

    fn ext(&mut self, coordinates: [Self::F; 3]) -> Self::E;
    fn coordinates(&mut self, a: Self::E) -> [Self::F; 3];
    /// An extension constant from its three coordinates.
    fn ext_constant(&mut self, value: [u64; 3]) -> Self::E;
    fn ext_add(&mut self, a: Self::E, b: Self::E) -> Self::E;
    fn ext_sub(&mut self, a: Self::E, b: Self::E) -> Self::E;
    fn ext_mul(&mut self, a: Self::E, b: Self::E) -> Self::E;
    /// `a·by` for a base-field `by`.
    fn ext_scale(&mut self, a: Self::E, by: Self::F) -> Self::E;
    /// Rejects zero.
    fn ext_inverse(&mut self, a: Self::E, what: &'static str) -> Result<Self::E, Error>;

    fn assert_zero(&mut self, a: Self::F, what: &'static str) -> Result<(), Error>;
    /// `count` little-endian bits of the canonical integer of `a`; the value
    /// must be below `2^count` (and below p for `count = 64`).
    fn bits(&mut self, a: Self::F, count: usize, what: &'static str) -> Result<Vec<Self::F>, Error>;
    /// `bit ? one : zero` for a constrained bit.
    fn select(&mut self, bit: Self::F, zero: Self::F, one: Self::F) -> Self::F;
    /// The workspace Poseidon2 permutation (width 16).
    fn permute(&mut self, state: [Self::F; WIDTH]) -> [Self::F; WIDTH];
    /// `a·b mod Φ81` for two ring elements in coefficient form.
    fn ring_mul(&mut self, a: &[Self::F; D], b: &[Self::F; D]) -> [Self::F; D];
    /// `len` private words computed from the values of `inputs`. A shape run
    /// passes zeros. The caller must constrain every output.
    fn hint(&mut self, inputs: &[Self::F], len: usize, compute: &dyn Fn(&[u64]) -> Vec<u64>) -> Vec<Self::F>;

    fn assert_ext_zero(&mut self, a: Self::E, what: &'static str) -> Result<(), Error> {
        for coordinate in self.coordinates(a) {
            self.assert_zero(coordinate, what)?;
        }
        Ok(())
    }

    fn assert_equal(&mut self, a: Self::F, b: Self::F, what: &'static str) -> Result<(), Error> {
        let difference = self.sub(a, b);
        self.assert_zero(difference, what)
    }

    fn assert_ext_equal(&mut self, a: Self::E, b: Self::E, what: &'static str) -> Result<(), Error> {
        let difference = self.ext_sub(a, b);
        self.assert_ext_zero(difference, what)
    }
}

/// The plain verifier: values only, `Err` on the first failed check. Its
/// words are layer-1 field values; no caller reads them.
pub struct Native;

impl Backend for Native {
    type F = Gl;
    type E = Ext;

    fn constant(&mut self, value: u64) -> Gl {
        Gl::from_u64(value)
    }

    fn private(&mut self, value: u64) -> Gl {
        Gl::from_u64(value)
    }

    fn public(&mut self, value: u64) -> Gl {
        Gl::from_u64(value)
    }

    fn add(&mut self, a: Gl, b: Gl) -> Gl {
        a + b
    }

    fn sub(&mut self, a: Gl, b: Gl) -> Gl {
        a - b
    }

    fn scale(&mut self, a: Gl, by: u64) -> Gl {
        a * Gl::from_u64(by)
    }

    fn mul(&mut self, a: Gl, b: Gl) -> Gl {
        a * b
    }

    fn ext(&mut self, coordinates: [Gl; 3]) -> Ext {
        Ext::from_basis_coefficients_fn(|i| coordinates[i])
    }

    fn coordinates(&mut self, a: Ext) -> [Gl; 3] {
        let slice = <Ext as BasedVectorSpace<Gl>>::as_basis_coefficients_slice(&a);
        [slice[0], slice[1], slice[2]]
    }

    fn ext_constant(&mut self, value: [u64; 3]) -> Ext {
        Ext::from_basis_coefficients_fn(|i| Gl::from_u64(value[i]))
    }

    fn ext_add(&mut self, a: Ext, b: Ext) -> Ext {
        a + b
    }

    fn ext_sub(&mut self, a: Ext, b: Ext) -> Ext {
        a - b
    }

    fn ext_mul(&mut self, a: Ext, b: Ext) -> Ext {
        a * b
    }

    fn ext_scale(&mut self, a: Ext, by: Gl) -> Ext {
        a * by
    }

    fn ext_inverse(&mut self, a: Ext, what: &'static str) -> Result<Ext, Error> {
        a.try_inverse().ok_or(Error::Rejected(what))
    }

    fn assert_zero(&mut self, a: Gl, what: &'static str) -> Result<(), Error> {
        if a == Gl::ZERO {
            Ok(())
        } else {
            Err(Error::Rejected(what))
        }
    }

    fn bits(&mut self, a: Gl, count: usize, what: &'static str) -> Result<Vec<Gl>, Error> {
        let value = a.as_canonical_u64();
        if count < 64 && value >> count != 0 {
            return Err(Error::Rejected(what));
        }
        Ok((0..count).map(|i| Gl::from_u64((value >> i) & 1)).collect())
    }

    fn select(&mut self, bit: Gl, zero: Gl, one: Gl) -> Gl {
        if bit == Gl::ZERO {
            zero
        } else {
            one
        }
    }

    fn permute(&mut self, state: [Gl; WIDTH]) -> [Gl; WIDTH] {
        crate::hash::permutation().permute(state)
    }

    fn ring_mul(&mut self, a: &[Gl; D], b: &[Gl; D]) -> [Gl; D] {
        ring_mul::multiply(a, b)
    }

    fn hint(&mut self, inputs: &[Gl], len: usize, compute: &dyn Fn(&[u64]) -> Vec<u64>) -> Vec<Gl> {
        let inputs: Vec<u64> = inputs.iter().map(Gl::as_canonical_u64).collect();
        let mut values: Vec<Gl> = compute(&inputs).into_iter().map(Gl::from_u64).collect();
        values.resize(len, Gl::ZERO);
        values
    }
}

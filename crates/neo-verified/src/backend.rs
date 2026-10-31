//! One verifier, two readings: the `Backend` contract that compression verifiers are written
//! in, and the hint kinds that a backend computes.
//!
//! `Native` (native.rs) reads it on plain values, `Recorder` (record.rs) as a constraint
//! system, and `formal/compression-lean/CompressionLeanRust` as the Lean copies' `Build`.
//!
//! Rules for code written over `Backend`:
//! - It never branches on a value. Every loop bound and length comes from a key or a fixed
//!   shape. No method returns a value to the caller.
//! - Proof words enter as `private`, statement words as `public`. They have no authority
//!   until a later check uses them.
//! - `hint` outputs are unconstrained; the caller must constrain each one.
//! - `Native` returns the first failed check as `Err`. `Recorder` notes it and continues, so
//!   the rows it records never depend on the values.

use crate::circuit::WIDTH;
use crate::paths::Paths;
use crate::r1cs::R1cs;

/// A failed check, by name.
#[derive(Clone, Copy)]
pub struct Rejected(pub &'static str);

/// `Copy` under another name, the bound of `Backend::E`. Aeneas names each bound of a trait
/// after the bound's trait, so `type E: Copy` beside `type F: Copy` would give the translated
/// trait two fields of one name.
pub trait Value: Copy {}

impl<T: Copy> Value for T {}

/// How a backend computes the words of a `hint` from the values of its inputs.
#[derive(Clone)]
pub enum Hint {
    /// The inverse of the first input, zero for zero or for no input.
    Inverse,
    /// These words, whatever the inputs.
    Words(Vec<u64>),
    /// The authentication path of each input, read as a leaf index, in a pruned multiproof.
    Paths(Paths),
}

pub trait Backend {
    /// A Goldilocks value or wire.
    type F: Copy;
    /// A cubic-extension value or wire (three base coordinates).
    type E: Value;

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
    fn ext_inverse(&mut self, a: Self::E, what: &'static str) -> Result<Self::E, Rejected>;

    fn assert_zero(&mut self, a: Self::F, what: &'static str) -> Result<(), Rejected>;
    /// `count` little-endian bits of the canonical integer of `a`; the value must be below
    /// `2^count` (and below p for `count = 64`).
    fn bits(&mut self, a: Self::F, count: usize, what: &'static str) -> Result<Vec<Self::F>, Rejected>;
    /// `bit ? one : zero` for a constrained bit.
    fn select(&mut self, bit: Self::F, zero: Self::F, one: Self::F) -> Self::F;
    /// The workspace Poseidon2 permutation (width 16).
    fn permute(&mut self, state: [Self::F; WIDTH]) -> [Self::F; WIDTH];
    /// `len` private words that `hint` computes from the values of `inputs`. A shape run
    /// passes zeros. The caller must constrain every output.
    fn hint(&mut self, inputs: &[Self::F], len: usize, hint: &Hint) -> Vec<Self::F>;
    /// Every row of `system`, with `columns[k]` as its column `k`.
    fn r1cs(&mut self, system: &R1cs, columns: &[Self::F], what: &'static str) -> Result<(), Rejected>;

    fn assert_ext_zero(&mut self, a: Self::E, what: &'static str) -> Result<(), Rejected> {
        let coordinates = self.coordinates(a);
        self.assert_zero(coordinates[0], what)?;
        self.assert_zero(coordinates[1], what)?;
        self.assert_zero(coordinates[2], what)
    }

    fn assert_equal(&mut self, a: Self::F, b: Self::F, what: &'static str) -> Result<(), Rejected> {
        let difference = self.sub(a, b);
        self.assert_zero(difference, what)
    }

    fn assert_ext_equal(&mut self, a: Self::E, b: Self::E, what: &'static str) -> Result<(), Rejected> {
        let difference = self.ext_sub(a, b);
        self.assert_ext_zero(difference, what)
    }
}

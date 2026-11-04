//! One verifier, two readings.
//!
//! The `Backend` contract, `Native` (the reading on plain values) and `Recorder` (the
//! reading as a constraint system) are in `neo_verified`, which Aeneas translates to Lean.
//! `Backend` is public so that a caller can write its own verifier (nightstream's final
//! program); every value enters as a `u64`, so no Plonky3 0.8 type crosses.
//!
//! Owns: `Fast`, the prover's reading on Plonky3 0.8 values, which tests compare with
//! `Native`; the conversions of layer-1 values into `neo_verified` words (`algebra`); and the
//! prover's fast paths behind `Sink` (`record::Trace`, `block`).

pub(crate) mod algebra;
pub(crate) mod block;
pub(crate) mod fold;
pub(crate) mod poseidon2;
pub(crate) mod r1cs;
pub(crate) mod record;

use neo_ccs::crypto::poseidon2_goldilocks::WIDTH;
pub use neo_verified::backend::{Backend, Hint, Rejected};
use neo_verified::native::hint_words;
pub use neo_verified::native::Native;
use neo_verified::r1cs::R1cs;
use p3_field_v08::{BasedVectorSpace, Field, PrimeCharacteristicRing, PrimeField64};
use p3_symmetric_v08::Permutation;

use crate::field::{Ext, Gl};

pub(crate) mod hash {
    pub(crate) use neo_verified::hash::{hash_leaf, merkle_root, Duplex};
}

/// The prover's values: Plonky3 0.8 arithmetic and permutation, `Err` on the first failed
/// check. Its words are layer-1 field values that the prover reads.
pub(crate) struct Fast;

impl Backend for Fast {
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

    fn ext_inverse(&mut self, a: Ext, what: &'static str) -> Result<Ext, Rejected> {
        a.try_inverse().ok_or(Rejected(what))
    }

    fn assert_zero(&mut self, a: Gl, what: &'static str) -> Result<(), Rejected> {
        if a == Gl::ZERO {
            Ok(())
        } else {
            Err(Rejected(what))
        }
    }

    fn bits(&mut self, a: Gl, count: usize, what: &'static str) -> Result<Vec<Gl>, Rejected> {
        let value = a.as_canonical_u64();
        if count < 64 && value >> count != 0 {
            return Err(Rejected(what));
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

    fn permute(&mut self, mut state: [Gl; WIDTH]) -> [Gl; WIDTH] {
        crate::hash::permutation().permute_mut(&mut state);
        state
    }

    fn hint(&mut self, inputs: &[Gl], len: usize, hint: &Hint) -> Vec<Gl> {
        let inputs: Vec<u64> = inputs.iter().map(Gl::as_canonical_u64).collect();
        let mut values: Vec<Gl> = hint_words(hint, &inputs)
            .into_iter()
            .map(Gl::from_u64)
            .collect();
        values.resize(len, Gl::ZERO);
        values
    }

    fn r1cs(&mut self, system: &R1cs, columns: &[Gl], what: &'static str) -> Result<(), Rejected> {
        assert_eq!(columns.len(), system.columns(), "one wire per R1CS column");
        if r1cs::fails(system, columns) {
            Err(Rejected(what))
        } else {
            Ok(())
        }
    }
}

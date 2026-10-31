//! `Native`, the reading of `Backend` on plain values, and the hint words that every backend
//! computes.

use crate::backend::{Backend, Hint, Rejected};
use crate::circuit::WIDTH;
use crate::field::{
    ext_add, ext_inv, ext_mul, ext_scale, ext_sub, gl_add, gl_inv, gl_mul, gl_sub, gl_word, reduce, Ext, Gl,
};
use crate::paths::path_words;
use crate::poseidon2::permute;
use crate::r1cs::{fails, R1cs};

/// The plain verifier: values only, `Err` on the first failed check.
pub struct Native;

/// Whether every coordinate of `a` is zero.
pub fn ext_is_zero(a: Ext) -> bool {
    gl_word(a.c0) == 0 && gl_word(a.c1) == 0 && gl_word(a.c2) == 0
}

/// The words of `hint` on the input words `inputs`.
pub fn hint_words(hint: &Hint, inputs: &[u64]) -> Vec<u64> {
    match hint {
        Hint::Inverse => {
            let mut words = Vec::new();
            let word = if inputs.is_empty() {
                0
            } else {
                gl_word(gl_inv(reduce(inputs[0])))
            };
            words.push(word);
            words
        }
        Hint::Words(words) => words.clone(),
        Hint::Paths(proof) => path_words(proof, inputs),
    }
}

impl Backend for Native {
    type F = Gl;
    type E = Ext;

    fn constant(&mut self, value: u64) -> Gl {
        reduce(value)
    }

    fn private(&mut self, value: u64) -> Gl {
        reduce(value)
    }

    fn public(&mut self, value: u64) -> Gl {
        reduce(value)
    }

    fn add(&mut self, a: Gl, b: Gl) -> Gl {
        gl_add(a, b)
    }

    fn sub(&mut self, a: Gl, b: Gl) -> Gl {
        gl_sub(a, b)
    }

    fn scale(&mut self, a: Gl, by: u64) -> Gl {
        gl_mul(a, reduce(by))
    }

    fn mul(&mut self, a: Gl, b: Gl) -> Gl {
        gl_mul(a, b)
    }

    fn ext(&mut self, coordinates: [Gl; 3]) -> Ext {
        Ext {
            c0: coordinates[0],
            c1: coordinates[1],
            c2: coordinates[2],
        }
    }

    fn coordinates(&mut self, a: Ext) -> [Gl; 3] {
        [a.c0, a.c1, a.c2]
    }

    fn ext_constant(&mut self, value: [u64; 3]) -> Ext {
        Ext {
            c0: reduce(value[0]),
            c1: reduce(value[1]),
            c2: reduce(value[2]),
        }
    }

    fn ext_add(&mut self, a: Ext, b: Ext) -> Ext {
        ext_add(a, b)
    }

    fn ext_sub(&mut self, a: Ext, b: Ext) -> Ext {
        ext_sub(a, b)
    }

    fn ext_mul(&mut self, a: Ext, b: Ext) -> Ext {
        ext_mul(a, b)
    }

    fn ext_scale(&mut self, a: Ext, by: Gl) -> Ext {
        ext_scale(a, by)
    }

    fn ext_inverse(&mut self, a: Ext, what: &'static str) -> Result<Ext, Rejected> {
        if ext_is_zero(a) {
            Err(Rejected(what))
        } else {
            Ok(ext_inv(a))
        }
    }

    fn assert_zero(&mut self, a: Gl, what: &'static str) -> Result<(), Rejected> {
        if gl_word(a) == 0 {
            Ok(())
        } else {
            Err(Rejected(what))
        }
    }

    fn bits(&mut self, a: Gl, count: usize, what: &'static str) -> Result<Vec<Gl>, Rejected> {
        let value = gl_word(a);
        if count < 64 && value.wrapping_shr(count as u32) != 0 {
            Err(Rejected(what))
        } else {
            let mut bits = Vec::new();
            let mut i = 0;
            while i < count {
                let bit = if i < 64 { value.wrapping_shr(i as u32) & 1 } else { 0 };
                bits.push(reduce(bit));
                i = i.wrapping_add(1);
            }
            Ok(bits)
        }
    }

    fn select(&mut self, bit: Gl, zero: Gl, one: Gl) -> Gl {
        if gl_word(bit) == 0 {
            zero
        } else {
            one
        }
    }

    fn permute(&mut self, state: [Gl; WIDTH]) -> [Gl; WIDTH] {
        permute(state)
    }

    fn hint(&mut self, inputs: &[Gl], len: usize, hint: &Hint) -> Vec<Gl> {
        let mut values = Vec::new();
        let mut k = 0;
        while k < inputs.len() {
            values.push(gl_word(inputs[k]));
            k = k.wrapping_add(1);
        }
        let words = hint_words(hint, &values);
        let mut outputs = Vec::new();
        let mut k = 0;
        while k < len {
            let word = if k < words.len() { words[k] } else { 0 };
            outputs.push(reduce(word));
            k = k.wrapping_add(1);
        }
        outputs
    }

    fn r1cs(&mut self, system: &R1cs, columns: &[Gl], what: &'static str) -> Result<(), Rejected> {
        assert!(columns.len() == system.columns());
        if fails(system, columns) {
            Err(Rejected(what))
        } else {
            Ok(())
        }
    }
}

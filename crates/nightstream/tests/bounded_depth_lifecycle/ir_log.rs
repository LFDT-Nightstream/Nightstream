//! A `Backend` that writes each call as one line of the instruction format
//! that `formal/compression-lean`'s checks parse (`CompressionLeanExe.Parse`),
//! the format of `neo-spartan`'s `tests/internal/lowering.rs`, and a toy
//! layer-1 relation that stops a final program at layer 1's shape check.

use std::fmt::Write as _;

use neo_ccs::crypto::poseidon2_goldilocks::WIDTH;
use neo_math::{D, F};
use neo_reductions::superneo_eval::{CachedMatrixRows, SuperneoEvalCacheBuilder};
use neo_spartan::{Backend, Error, Key, R1cs, Relation};
use p3_field::{PrimeCharacteristicRing, PrimeField64};

/// Each call as one instruction; value `v` is the `v`-th output of the run.
#[derive(Default)]
pub(crate) struct Log {
    pub(crate) ir: String,
    pub(crate) values: u32,
}

impl Log {
    fn emit(&mut self, instruction: &str, outputs: usize) -> Vec<u32> {
        self.ir.push_str(instruction);
        self.ir.push('\n');
        let first = self.values;
        self.values += outputs as u32;
        (first..self.values).collect()
    }

    fn one(&mut self, instruction: &str) -> u32 {
        self.emit(instruction, 1)[0]
    }
}

fn canonical(value: u64) -> u64 {
    F::from_u64(value).as_canonical_u64()
}

pub(crate) fn handles(values: &[u32]) -> String {
    values.iter().map(|value| format!(" {value}")).collect()
}

impl Backend for Log {
    type F = u32;
    type E = [u32; 3];

    fn constant(&mut self, value: u64) -> u32 {
        self.one(&format!("c {}", canonical(value)))
    }

    fn private(&mut self, _value: u64) -> u32 {
        self.one("p")
    }

    fn public(&mut self, _value: u64) -> u32 {
        self.one("u")
    }

    fn add(&mut self, a: u32, b: u32) -> u32 {
        self.one(&format!("l 2 {a} 1 {b} 1"))
    }

    fn sub(&mut self, a: u32, b: u32) -> u32 {
        self.one(&format!("l 2 {a} 1 {b} {}", (-F::ONE).as_canonical_u64()))
    }

    fn scale(&mut self, a: u32, by: u64) -> u32 {
        self.one(&format!("l 1 {a} {}", canonical(by)))
    }

    fn mul(&mut self, a: u32, b: u32) -> u32 {
        self.one(&format!("m {a} {b}"))
    }

    fn ext(&mut self, coordinates: [u32; 3]) -> [u32; 3] {
        coordinates
    }

    fn coordinates(&mut self, a: [u32; 3]) -> [u32; 3] {
        a
    }

    fn ext_constant(&mut self, value: [u64; 3]) -> [u32; 3] {
        value.map(|word| self.constant(word))
    }

    fn ext_add(&mut self, a: [u32; 3], b: [u32; 3]) -> [u32; 3] {
        std::array::from_fn(|i| self.add(a[i], b[i]))
    }

    fn ext_sub(&mut self, a: [u32; 3], b: [u32; 3]) -> [u32; 3] {
        std::array::from_fn(|i| self.sub(a[i], b[i]))
    }

    fn ext_mul(&mut self, a: [u32; 3], b: [u32; 3]) -> [u32; 3] {
        let product = self.emit(&format!("x{}{}", handles(&a), handles(&b)), 3);
        [product[0], product[1], product[2]]
    }

    fn ext_scale(&mut self, a: [u32; 3], by: u32) -> [u32; 3] {
        std::array::from_fn(|i| self.mul(a[i], by))
    }

    fn ext_inverse(&mut self, a: [u32; 3], _what: &'static str) -> Result<[u32; 3], Error> {
        let inverse = self.emit(&format!("i{}", handles(&a)), 3);
        Ok([inverse[0], inverse[1], inverse[2]])
    }

    fn assert_zero(&mut self, a: u32, _what: &'static str) -> Result<(), Error> {
        self.emit(&format!("z {a}"), 0);
        Ok(())
    }

    fn bits(&mut self, a: u32, count: usize, _what: &'static str) -> Result<Vec<u32>, Error> {
        Ok(self.emit(&format!("b {a} {count}"), count))
    }

    fn select(&mut self, bit: u32, zero: u32, one: u32) -> u32 {
        let difference = self.sub(one, zero);
        let chosen = self.mul(bit, difference);
        self.add(zero, chosen)
    }

    fn permute(&mut self, state: [u32; WIDTH]) -> [u32; WIDTH] {
        let outputs = self.emit(&format!("q{}", handles(&state)), WIDTH);
        std::array::from_fn(|lane| outputs[lane])
    }

    fn hint(&mut self, _inputs: &[u32], len: usize, _compute: &dyn Fn(&[u64]) -> Vec<u64>) -> Vec<u32> {
        self.emit(&format!("h {len}"), len)
    }

    fn r1cs(&mut self, system: &R1cs, columns: &[u32], _what: &'static str) -> Result<(), Error> {
        let mut line = format!("r {} {}{}", system.rows(), columns.len(), handles(columns));
        for row in 0..system.rows() {
            for side in system.row(row) {
                write!(line, " {}", side.len()).unwrap();
                for (column, coefficient) in side {
                    write!(line, " {column} {coefficient}").unwrap();
                }
            }
        }
        self.emit(&line, 0);
        Ok(())
    }
}

/// A key of four blocks, six rows and four matrices.
pub(crate) fn toy_key() -> Key {
    let mut builder = SuperneoEvalCacheBuilder::new(6, 4 * D, 4).unwrap();
    for row in 0..6 {
        for matrix in 0..4 {
            builder
                .push_row(matrix, row, [(row * 31 + matrix, F::ONE)])
                .unwrap();
        }
    }
    Key::derive(&CachedMatrixRows::new(&builder.finish().unwrap()).unwrap()).unwrap()
}

/// One public block: a final program's claim has five, so `verify_with`
/// returns at its shape check, before any backend call.
pub(crate) fn toy_relation(key: &Key) -> Relation<'_> {
    Relation::new(key, 1, 9, 3_672, 100.0).unwrap()
}

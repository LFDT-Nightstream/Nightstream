//! The prover's trace of a permutation block: the template of `neo_verified::block` in
//! Plonky3 0.8 values, built once.
//!
//! Owns: the cached template and `trace`. A block has 182 cells: the 16 inputs, the 150
//! S-box outputs in round order, and the 16 outputs.

use std::sync::OnceLock;

use neo_ccs::crypto::poseidon2_goldilocks::WIDTH;
pub(crate) use neo_verified::block::OUTPUT;
use neo_verified::circuit::CELLS;
use p3_field_v08::PrimeCharacteristicRing;

use crate::field::{from_verified, Gl};

const _: () = assert!(neo_verified::circuit::WIDTH == WIDTH);

/// A linear form over the cells of one block, plus a constant.
pub(crate) struct Linear {
    terms: Vec<(usize, Gl)>,
    constant: Gl,
}

impl Linear {
    fn new(form: &neo_verified::block::Linear) -> Self {
        Self {
            terms: form
                .terms
                .iter()
                .map(|&(cell, coefficient)| (cell, from_verified(coefficient)))
                .collect(),
            constant: from_verified(form.constant),
        }
    }

    fn evaluate(&self, cells: &[Gl]) -> Gl {
        self.terms
            .iter()
            .fold(self.constant, |acc, &(cell, coefficient)| {
                acc + coefficient * cells[cell]
            })
    }
}

/// The S-box inputs (rows `0..150`) and the outputs (rows `150..166`).
struct Template {
    sbox_inputs: Vec<Linear>,
    outputs: Vec<Linear>,
}

fn template() -> &'static Template {
    static TEMPLATE: OnceLock<Template> = OnceLock::new();
    TEMPLATE.get_or_init(|| {
        let template = neo_verified::block::template();
        Template {
            sbox_inputs: template.sbox_inputs.iter().map(Linear::new).collect(),
            outputs: template.outputs.iter().map(Linear::new).collect(),
        }
    })
}

/// The 182 cell values of one permutation of `input`.
pub(crate) fn trace(input: &[Gl; WIDTH]) -> Vec<Gl> {
    let template = template();
    let mut cells = vec![Gl::ZERO; CELLS];
    cells[..WIDTH].copy_from_slice(input);
    for (round, form) in template.sbox_inputs.iter().enumerate() {
        cells[WIDTH + round] = form.evaluate(&cells).exp_const_u64::<7>();
    }
    for (lane, form) in template.outputs.iter().enumerate() {
        cells[OUTPUT + lane] = form.evaluate(&cells);
    }
    cells
}

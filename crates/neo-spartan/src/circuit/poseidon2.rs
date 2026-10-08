//! The workspace Poseidon2 permutation (width 16, x^7, 4 + 22 + 4 rounds) as
//! one constraint block.
//!
//! Owns: the block template and its trace. A block has 182 cells: the 16
//! inputs, the 150 S-box outputs in round order, and the 16 outputs. It has
//! 166 rows: row `r < 150` is `(X·z)^7 = cell(16 + r)`, where `X` is the S-box
//! input as a linear form of earlier cells and a constant; row `150 + i` is
//! `cell(166 + i) = output_i`, a linear form of the last S-box outputs. The
//! constants come from `round_constants()`, which a workspace test pins to
//! the permutation itself.

use std::sync::OnceLock;

use neo_ccs::crypto::poseidon2_goldilocks::{round_constants, WIDTH};
use p3_field_v08::PrimeCharacteristicRing;

use super::block::{Entry, C, X};
use crate::field::Gl;

pub(crate) const SBOXES: usize = 150;
pub(crate) const CELLS: usize = WIDTH + SBOXES + WIDTH;
pub(crate) const ROWS: usize = SBOXES + WIDTH;
/// The first output cell.
pub(crate) const OUTPUT: usize = WIDTH + SBOXES;

/// A linear form over the cells of one block, plus a constant.
#[derive(Clone, Debug, Default)]
pub(crate) struct Linear {
    pub(crate) terms: Vec<(usize, Gl)>,
    pub(crate) constant: Gl,
}

impl Linear {
    fn cell(cell: usize) -> Self {
        Self {
            terms: vec![(cell, Gl::ONE)],
            constant: Gl::ZERO,
        }
    }

    fn combine(parts: &[(&Linear, Gl)]) -> Self {
        let mut terms: Vec<(usize, Gl)> = Vec::new();
        let mut constant = Gl::ZERO;
        for &(part, weight) in parts {
            constant += part.constant * weight;
            for &(cell, coefficient) in &part.terms {
                match terms.iter_mut().find(|(c, _)| *c == cell) {
                    Some((_, existing)) => *existing += coefficient * weight,
                    None => terms.push((cell, coefficient * weight)),
                }
            }
        }
        terms.retain(|&(_, coefficient)| coefficient != Gl::ZERO);
        terms.sort_by_key(|&(cell, _)| cell);
        Self { terms, constant }
    }

    pub(crate) fn evaluate(&self, cells: &[Gl]) -> Gl {
        self.terms
            .iter()
            .fold(self.constant, |acc, &(cell, coefficient)| {
                acc + coefficient * cells[cell]
            })
    }
}

/// The S-box inputs (rows `0..150`) and the outputs (rows `150..166`).
pub(crate) struct Template {
    pub(crate) sbox_inputs: Vec<Linear>,
    pub(crate) outputs: Vec<Linear>,
}

pub(crate) fn template() -> &'static Template {
    static TEMPLATE: OnceLock<Template> = OnceLock::new();
    TEMPLATE.get_or_init(build)
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

/// Every nonzero of the block. S-box row `r`: `X` is the S-box input form,
/// `C` the S-box output cell. Output row `150 + i`: `C` is the output cell
/// minus its form.
pub(crate) fn entries() -> Vec<Entry> {
    let template = template();
    let mut entries = Vec::new();
    let mut push = |matrix, row, form: &Linear, sign: Gl| {
        for &(cell, coefficient) in &form.terms {
            entries.push(Entry {
                matrix,
                row,
                cell: Some(cell),
                coefficient: sign * coefficient,
            });
        }
        if form.constant != Gl::ZERO {
            entries.push(Entry {
                matrix,
                row,
                cell: None,
                coefficient: sign * form.constant,
            });
        }
    };
    for (row, form) in template.sbox_inputs.iter().enumerate() {
        push(X, row, form, Gl::ONE);
    }
    for (lane, form) in template.outputs.iter().enumerate() {
        push(C, SBOXES + lane, form, -Gl::ONE);
    }
    for row in 0..ROWS {
        let cell = if row < SBOXES {
            WIDTH + row
        } else {
            OUTPUT + row - SBOXES
        };
        entries.push(Entry {
            matrix: C,
            row,
            cell: Some(cell),
            coefficient: Gl::ONE,
        });
    }
    entries
}

fn build() -> Template {
    let constants = round_constants();
    let diag: [Gl; WIDTH] = constants.diag.map(Gl::new);
    let mut state: Vec<Linear> = (0..WIDTH).map(Linear::cell).collect();
    let mut sbox_inputs = Vec::with_capacity(SBOXES);
    state = external_linear(&state);
    full_rounds(&mut state, &constants.initial, &mut sbox_inputs);
    for &constant in &constants.internal {
        let mut input = state[0].clone();
        input.constant += Gl::new(constant);
        state[0] = Linear::cell(WIDTH + sbox_inputs.len());
        sbox_inputs.push(input);
        let parts: Vec<(&Linear, Gl)> = state.iter().map(|form| (form, Gl::ONE)).collect();
        let sum = Linear::combine(&parts);
        state = state
            .iter()
            .zip(&diag)
            .map(|(form, &v)| Linear::combine(&[(&sum, Gl::ONE), (form, v)]))
            .collect();
    }
    full_rounds(&mut state, &constants.terminal, &mut sbox_inputs);
    assert_eq!(sbox_inputs.len(), SBOXES);
    Template {
        sbox_inputs,
        outputs: state,
    }
}

/// Full rounds: every lane gets its constant and an S-box row, then `M_E`.
fn full_rounds(state: &mut Vec<Linear>, rounds: &[[u64; WIDTH]], inputs: &mut Vec<Linear>) {
    for round in rounds {
        for (lane, form) in state.iter_mut().enumerate() {
            let mut input = form.clone();
            input.constant += Gl::new(round[lane]);
            *form = Linear::cell(WIDTH + inputs.len());
            inputs.push(input);
        }
        *state = external_linear(state);
    }
}

/// `M_E`: the 4×4 MDS block `[2 3 1 1]` (circulant) on each chunk, then each
/// lane plus the sum of its residue class mod 4.
fn external_linear(state: &[Linear]) -> Vec<Linear> {
    const M4: [[u64; 4]; 4] = [[2, 3, 1, 1], [1, 2, 3, 1], [1, 1, 2, 3], [3, 1, 1, 2]];
    let mixed: Vec<Linear> = (0..WIDTH)
        .map(|lane| {
            let (chunk, row) = (lane / 4, lane % 4);
            let parts: Vec<(&Linear, Gl)> = (0..4)
                .map(|j| (&state[4 * chunk + j], Gl::from_u64(M4[row][j])))
                .collect();
            Linear::combine(&parts)
        })
        .collect();
    (0..WIDTH)
        .map(|lane| {
            let mut parts: Vec<(&Linear, Gl)> = vec![(&mixed[lane], Gl::ONE)];
            parts.extend((0..4).map(|chunk| (&mixed[4 * chunk + lane % 4], Gl::ONE)));
            Linear::combine(&parts)
        })
        .collect()
}

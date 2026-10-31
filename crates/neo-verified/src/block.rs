//! Uniform blocks: a fixed set of rows over the block's own cells.
//!
//! Owns: the inputs and outputs of each block kind, the Poseidon2 block template, and each
//! kind's rows as entries of the eight CCS matrices `X, A_0, B_0, A_1, B_1, A_2, B_2, C`. A row
//! holds when `(X·z)^7 + Σ_i (A_i·z)·(B_i·z) − C·z = 0`. The first `inputs` cells of a block
//! are its inputs; its outputs are the cells from `OUTPUT` to `CELLS`.
//!
//! A permutation block has 182 cells: the 16 inputs, the 150 S-box outputs in round order,
//! and the 16 outputs. It has 166 rows: row `r < 150` is `(X·z)^7 = cell(16 + r)`, where `X`
//! is the S-box input as a linear form of earlier cells and a constant; row `150 + i` is
//! `cell(166 + i) = output_i`, a linear form of the last S-box outputs. The constants are the
//! tables of `poseidon2.rs`.

use crate::circuit::{Kind, ROWS, SBOXES, WIDTH};
use crate::field::{gl_add, gl_mul, gl_neg, reduce, Gl, GL_ONE, GL_ZERO};
use crate::poseidon2::{DIAGONAL, HALF_FULL_ROUNDS, INITIAL, PARTIAL, PARTIAL_ROUNDS, TERMINAL};

pub const X: usize = 0;
pub const C: usize = 7;
/// The first output cell.
pub const OUTPUT: usize = WIDTH + SBOXES;

/// One nonzero of a block row: the matrix, the template row, the block cell (`None` for the
/// constant one) and the coefficient.
#[derive(Clone, Copy)]
pub struct Entry {
    pub matrix: usize,
    pub row: usize,
    pub cell: Option<usize>,
    pub coefficient: Gl,
}

/// A linear form over the cells of one block, plus a constant.
#[derive(Clone)]
pub struct Linear {
    /// In cell order, with no zero coefficient.
    pub terms: Vec<(usize, Gl)>,
    pub constant: Gl,
}

/// The S-box inputs (rows `0..150`) and the outputs (rows `150..166`).
pub struct Template {
    pub sbox_inputs: Vec<Linear>,
    pub outputs: Vec<Linear>,
}

pub fn inputs(kind: Kind) -> usize {
    match kind {
        Kind::Permutation => WIDTH,
    }
}

fn of_cell(cell: usize) -> Linear {
    Linear {
        terms: Vec::from([(cell, GL_ONE)]),
        constant: GL_ZERO,
    }
}

/// `form` plus the constant `word`.
fn shifted(form: &Linear, word: u64) -> Linear {
    let mut terms = Vec::new();
    let mut k = 0;
    while k < form.terms.len() {
        terms.push(form.terms[k]);
        k = k.wrapping_add(1);
    }
    Linear {
        terms,
        constant: gl_add(form.constant, reduce(word)),
    }
}

/// A sum of weighted forms, its terms merged by cell.
struct Sum {
    merged: Vec<(usize, Gl)>,
    constant: Gl,
}

fn empty_sum() -> Sum {
    Sum {
        merged: Vec::new(),
        constant: GL_ZERO,
    }
}

/// `sum + weight·part`.
fn add_part(sum: &mut Sum, part: &Linear, weight: Gl) {
    sum.constant = gl_add(sum.constant, gl_mul(part.constant, weight));
    let mut k = 0;
    while k < part.terms.len() {
        let (cell, coefficient) = part.terms[k];
        let term = gl_mul(coefficient, weight);
        let mut found = sum.merged.len();
        let mut m = 0;
        while m < sum.merged.len() {
            if found == sum.merged.len() && sum.merged[m].0 == cell {
                found = m;
            }
            m = m.wrapping_add(1);
        }
        if found < sum.merged.len() {
            sum.merged[found].1 = gl_add(sum.merged[found].1, term);
        } else {
            sum.merged.push((cell, term));
        }
        k = k.wrapping_add(1);
    }
}

/// The form of `sum`, its terms in cell order without zero coefficients.
fn sum_form(sum: Sum) -> Linear {
    let mut terms: Vec<(usize, Gl)> = Vec::new();
    let mut m = 0;
    while m < sum.merged.len() {
        let term = sum.merged[m];
        if term.1 != GL_ZERO {
            let mut sorted = Vec::new();
            let mut placed = false;
            let mut k = 0;
            while k < terms.len() {
                if !placed && term.0 < terms[k].0 {
                    sorted.push(term);
                    placed = true;
                }
                sorted.push(terms[k]);
                k = k.wrapping_add(1);
            }
            if !placed {
                sorted.push(term);
            }
            terms = sorted;
        }
        m = m.wrapping_add(1);
    }
    Linear {
        terms,
        constant: sum.constant,
    }
}

/// `M_E`: the 4×4 MDS block `[2 3 1 1]` (circulant) on each chunk, then each lane plus the
/// sum of its residue class mod 4.
fn external_linear(state: &[Linear]) -> Vec<Linear> {
    const M4: [[u64; 4]; 4] = [[2, 3, 1, 1], [1, 2, 3, 1], [1, 1, 2, 3], [3, 1, 1, 2]];
    let mut mixed = Vec::new();
    let mut lane: usize = 0;
    while lane < WIDTH {
        let base = lane & !3;
        let row = M4[lane & 3];
        let mut sum = empty_sum();
        add_part(&mut sum, &state[base], reduce(row[0]));
        add_part(&mut sum, &state[base | 1], reduce(row[1]));
        add_part(&mut sum, &state[base | 2], reduce(row[2]));
        add_part(&mut sum, &state[base | 3], reduce(row[3]));
        mixed.push(sum_form(sum));
        lane = lane.wrapping_add(1);
    }
    let mut out = Vec::new();
    let mut lane: usize = 0;
    while lane < WIDTH {
        let class = lane & 3;
        let mut sum = empty_sum();
        add_part(&mut sum, &mixed[lane], GL_ONE);
        add_part(&mut sum, &mixed[class], GL_ONE);
        add_part(&mut sum, &mixed[4 | class], GL_ONE);
        add_part(&mut sum, &mixed[8 | class], GL_ONE);
        add_part(&mut sum, &mixed[12 | class], GL_ONE);
        out.push(sum_form(sum));
        lane = lane.wrapping_add(1);
    }
    out
}

/// Full rounds: every lane gets its constant and an S-box row, then `M_E`.
fn full_rounds(state: &mut Vec<Linear>, rounds: &[[u64; WIDTH]; HALF_FULL_ROUNDS], inputs: &mut Vec<Linear>) {
    let mut round = 0;
    while round < HALF_FULL_ROUNDS {
        let mut lane = 0;
        while lane < WIDTH {
            let input = shifted(&state[lane], rounds[round][lane]);
            state[lane] = of_cell(WIDTH.wrapping_add(inputs.len()));
            inputs.push(input);
            lane = lane.wrapping_add(1);
        }
        *state = external_linear(state);
        round = round.wrapping_add(1);
    }
}

/// The permutation's S-box input forms and output forms over the block cells.
pub fn template() -> Template {
    let mut state = Vec::new();
    let mut lane = 0;
    while lane < WIDTH {
        state.push(of_cell(lane));
        lane = lane.wrapping_add(1);
    }
    let mut sbox_inputs = Vec::new();
    state = external_linear(&state);
    full_rounds(&mut state, &INITIAL, &mut sbox_inputs);
    let mut round = 0;
    while round < PARTIAL_ROUNDS {
        let input = shifted(&state[0], PARTIAL[round]);
        state[0] = of_cell(WIDTH.wrapping_add(sbox_inputs.len()));
        sbox_inputs.push(input);
        let mut all = empty_sum();
        let mut lane = 0;
        while lane < WIDTH {
            add_part(&mut all, &state[lane], GL_ONE);
            lane = lane.wrapping_add(1);
        }
        let total = sum_form(all);
        let mut next = Vec::new();
        let mut lane = 0;
        while lane < WIDTH {
            let mut sum = empty_sum();
            add_part(&mut sum, &total, GL_ONE);
            add_part(&mut sum, &state[lane], reduce(DIAGONAL[lane]));
            next.push(sum_form(sum));
            lane = lane.wrapping_add(1);
        }
        state = next;
        round = round.wrapping_add(1);
    }
    full_rounds(&mut state, &TERMINAL, &mut sbox_inputs);
    Template {
        sbox_inputs,
        outputs: state,
    }
}

/// The entries of `form` in `matrix` and `row`, each coefficient times `sign`.
fn push_form(entries: &mut Vec<Entry>, matrix: usize, row: usize, form: &Linear, sign: Gl) {
    let mut k = 0;
    while k < form.terms.len() {
        let (cell, coefficient) = form.terms[k];
        entries.push(Entry {
            matrix,
            row,
            cell: Some(cell),
            coefficient: gl_mul(sign, coefficient),
        });
        k = k.wrapping_add(1);
    }
    if form.constant != GL_ZERO {
        entries.push(Entry {
            matrix,
            row,
            cell: None,
            coefficient: gl_mul(sign, form.constant),
        });
    }
}

/// Every nonzero of the rows of `kind`. S-box row `r`: `X` is the S-box input form, `C` the
/// S-box output cell. Output row `150 + i`: `C` is the output cell minus its form.
pub fn entries(kind: Kind) -> Vec<Entry> {
    match kind {
        Kind::Permutation => {
            let template = template();
            let mut entries = Vec::new();
            let mut row = 0;
            while row < template.sbox_inputs.len() {
                push_form(&mut entries, X, row, &template.sbox_inputs[row], GL_ONE);
                row = row.wrapping_add(1);
            }
            let mut lane = 0;
            while lane < template.outputs.len() {
                push_form(
                    &mut entries,
                    C,
                    SBOXES.wrapping_add(lane),
                    &template.outputs[lane],
                    gl_neg(GL_ONE),
                );
                lane = lane.wrapping_add(1);
            }
            let mut row = 0;
            while row < ROWS {
                let cell = if row < SBOXES {
                    WIDTH.wrapping_add(row)
                } else {
                    OUTPUT.wrapping_add(row).wrapping_sub(SBOXES)
                };
                entries.push(Entry {
                    matrix: C,
                    row,
                    cell: Some(cell),
                    coefficient: GL_ONE,
                });
                row = row.wrapping_add(1);
            }
            entries
        }
    }
}

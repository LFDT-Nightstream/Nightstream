//! The degrees of the draws that layer 1 and the shrink charge, as integers.
//!
//! `neo-spartan` charges a draw of degree `d` as a Plonky3 `SecurityTerm` of error
//! `d / |Ext|` (layer 1 also over P1's candidates), so the error that a commitment's
//! charges add is the sum of their degrees over `|Ext|`. Each `*_terms` function gives the
//! degrees that `neo-spartan` charges, in its order; `early`, `linear`, `late`, `layer1` and
//! `shrink` add them, for the proofs in `formal/compression-lean/CompressionLeanRust`.

use crate::circuit::MATRICES;
use crate::layout::{cell_variables, row_variables, Shape};

/// Setup oracles after the key rows: run rows, slots, coefficients, slot blocks.
pub const RUN_ORACLES: usize = 4;
/// Variables folded per proof; a leaf holds `2^FOLD` values per oracle.
pub const FOLD: usize = 3;
/// The shrink's outer sum-check degree: `eq` times the S-box term `(X·z)^7`.
pub const DEGREE: usize = 8;
/// The run group's draws after its first, and the honest-fold partials.
pub const FIXED: [usize; 2] = [RUN_ORACLES - 1, 2 * FOLD];

/// What the layer-1 charges read: the relation's counts and its key structure's variables.
pub struct Relation {
    /// The ring degree `D`.
    pub d: usize,
    pub lane_variables: usize,
    pub block_variables: usize,
    pub kappa: usize,
    pub matrices: usize,
    pub public_blocks: usize,
    pub norm_bound: u32,
    pub run_variables: usize,
    pub row_variables: usize,
    pub slot_variables: usize,
    pub matrix_variables: usize,
}

fn add(a: usize, b: usize) -> usize {
    a.checked_add(b).unwrap()
}

fn sub(a: usize, b: usize) -> usize {
    a.checked_sub(b).unwrap()
}

fn mul(a: usize, b: usize) -> usize {
    a.checked_mul(b).unwrap()
}

fn exp2(v: usize) -> usize {
    // `wrapping_shl` masks the shift, so a shift of `usize::BITS` or more panics here.
    let _fits = sub(sub(usize::BITS as usize, 1), v);
    1usize.wrapping_shl(v as u32)
}

/// Ring rows: the commitment rows, Eval_K (2), Eval_A (2 per matrix) and the public blocks.
pub fn row_count(kappa: usize, matrices: usize, public_blocks: usize) -> usize {
    add(add(add(kappa, 2), mul(2, matrices)), public_blocks)
}

/// The draws of GKR trees of `depths`. Layer `k` has `k` rounds of degree at most 4, and a
/// batching draw or `τ` for each tree of depth at least `k`. So the layers of the deepest
/// tree, `L`, add `Σ_{k ≤ L} 4k = 2L(L + 1)`, and tree `t` adds its depth.
pub fn gkr(depths: &[usize]) -> usize {
    let mut deepest = 0;
    let mut sum = 0;
    let mut t = 0;
    while t < depths.len() {
        deepest = deepest.max(depths[t]);
        sum = add(sum, depths[t]);
        t = add(t, 1);
    }
    add(mul(2, mul(deepest, add(deepest, 1))), sum)
}

fn cube_variables(relation: &Relation) -> usize {
    add(relation.lane_variables, relation.block_variables)
}

/// The norm logUp, the row lookup, the slot scatter, the early GKR and the slot-block test.
pub fn early_terms(relation: &Relation) -> [usize; 5] {
    let cube = cube_variables(relation);
    let runs = relation.run_variables;
    let rows = relation.row_variables;
    let pairs = add(relation.slot_variables, relation.matrix_variables);
    [
        add(exp2(cube), add(mul(2, relation.norm_bound as usize), 1)),
        mul(3, add(exp2(runs), exp2(rows))),
        add(add(exp2(runs), exp2(pairs)), 1),
        gkr(&[cube, runs, rows, runs, pairs]),
        // The P0 slot-block copy against the setup block column at the early run point,
        // then `1 - x ≠ 0` on its high coordinates: one draw each.
        runs,
    ]
}

/// The ring-row batching, the quotient at ζ (degree `2D - 2`) and the linear sum-check.
pub fn linear_terms(relation: &Relation) -> [usize; 3] {
    let rows = row_count(relation.kappa, relation.matrices, relation.public_blocks);
    [
        sub(rows, 1),
        mul(2, sub(relation.d, 1)),
        mul(2, cube_variables(relation)),
    ]
}

/// The block scatter and the late GKR.
pub fn late_terms(relation: &Relation) -> [usize; 2] {
    let blocks = add(relation.slot_variables, 1);
    [
        add(exp2(blocks), exp2(relation.block_variables)),
        gkr(&[blocks, relation.block_variables]),
    ]
}

/// The zero-check point, the outer sum-check, the matrix batching and the inner sum-check.
pub fn shrink_terms(shape: &Shape) -> [usize; 4] {
    let rows = row_variables(shape);
    let cells = cell_variables(shape);
    [rows, mul(DEGREE, rows), sub(MATRICES, 1), mul(2, add(cells, 1))]
}

pub fn early(relation: &Relation) -> usize {
    let [norm, lookup, scatter, trees, test] = early_terms(relation);
    add(add(add(add(norm, lookup), scatter), trees), test)
}

pub fn linear(relation: &Relation) -> usize {
    let [batching, quotient, sumcheck] = linear_terms(relation);
    add(add(batching, quotient), sumcheck)
}

pub fn late(relation: &Relation) -> usize {
    let [scatter, trees] = late_terms(relation);
    add(scatter, trees)
}

/// Every layer-1 draw made after P0.
pub fn layer1(relation: &Relation) -> usize {
    let [run_group, fold_partials] = FIXED;
    add(
        add(add(add(early(relation), linear(relation)), run_group), fold_partials),
        late(relation),
    )
}

pub fn shrink(shape: &Shape) -> usize {
    let [point, outer, batching, inner] = shrink_terms(shape);
    add(add(add(point, outer), batching), inner)
}

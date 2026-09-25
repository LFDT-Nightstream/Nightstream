//! Final witness `Z` wire allocation for the terminal CE-relation gadget.
//!
//! Mirrors the prototype's `PackedWitnessVar` (Bellpepper) under the
//! clean's `R1csBuilder`. Layout is the SuperNeo packed
//! `Z ∈ F^{D × ceil(m / D)}`, row-major, with `D` rows and one row per
//! coefficient lane.

use neo_math::{D, F};
use p3_field::PrimeCharacteristicRing;

use crate::engine::r1cs_circuit::builder::{Lc, R1csBuilder, Var};
use crate::paper::relations::WitnessMat;

/// Wires + native values for one final running's witness `Z`.
#[derive(Clone, Debug)]
pub(crate) struct FinalWitnessWires {
    pub(crate) rows: usize,
    pub(crate) cols: usize,
    pub(crate) values: Vec<Var>,
    #[allow(dead_code)]
    pub(crate) native: Vec<F>,
}

#[derive(Debug)]
pub(crate) struct AllocError {
    what: &'static str,
    expected: usize,
    got: usize,
}

impl AllocError {
    pub(crate) fn what(&self) -> &'static str {
        self.what
    }
    pub(crate) fn expected(&self) -> usize {
        self.expected
    }
    pub(crate) fn got(&self) -> usize {
        self.got
    }
}

pub(crate) fn alloc_final_witness(
    builder: &mut R1csBuilder,
    witness: &WitnessMat,
    expected_m: usize,
) -> Result<FinalWitnessWires, AllocError> {
    if witness.rows() != D {
        return Err(AllocError {
            what: "witness rows (expected D)",
            expected: D,
            got: witness.rows(),
        });
    }
    let want_cols = expected_m.div_ceil(D);
    if witness.cols() != want_cols {
        return Err(AllocError {
            what: "witness cols (expected ceil(m / D))",
            expected: want_cols,
            got: witness.cols(),
        });
    }
    let native = witness.to_dense_vec();
    let values = builder.alloc_vec(&native);
    Ok(FinalWitnessWires {
        rows: witness.rows(),
        cols: witness.cols(),
        values,
        native,
    })
}

impl FinalWitnessWires {
    /// `Z[row, col]` wire (row in 0..D, col in 0..cols).
    pub(crate) fn entry(&self, row: usize, col: usize) -> Option<Var> {
        if row >= self.rows || col >= self.cols {
            return None;
        }
        self.values.get(row * self.cols + col).copied()
    }

    /// Coefficient `c` of the complete packed witness ring blocks, including
    /// the final block's lanes beyond the logical CCS width.
    pub(crate) fn packed_entry(&self, packed_col: usize) -> Option<Var> {
        if self.rows != D || packed_col >= self.cols.saturating_mul(D) {
            return None;
        }
        let off = packed_col % D;
        let block = packed_col / D;
        self.values.get(off * self.cols + block).copied()
    }
}

/// Error returned by [`enforce_balanced_alphabet`].
///
/// `NeoParams::new` rejects `b < 2` at construction time, so this
/// error is structurally unreachable on the honest preprocessing
/// path. It exists so the gadget refuses to emit *any* misleading
/// constraint rows when called with a degenerate `b` (rather than
/// silently emitting a satisfiable fallback the caller might trust).
#[derive(Debug)]
pub(crate) struct AlphabetError {
    pub(crate) b: u32,
}

/// Enforce every witness entry lies in the SuperNeo NC-bound alphabet
/// `{-(b-1), …, +(b-1)}` — matching `neo_math::balanced::within_nc_bound`
/// (which the native verifier uses to reject out-of-bound witnesses).
///
/// One polynomial-root constraint per entry: `Π_{a ∈ alphabet} (Z[i,j]
/// - a) = 0`. The alphabet has `2*b - 1` elements; this chain emits
/// `2*b - 2` auxiliary multiplications + 1 final equality per entry.
///
/// Returns `Err` when `b < 2` — the gadget is undefined there because
/// `within_nc_bound` rejects everything, and emitting a single
/// `v == 0` row would be a satisfiable lie. `NeoParams::new` already
/// rejects `b < 2`, so honest preprocessing never triggers this branch.
pub(crate) fn enforce_balanced_alphabet(
    builder: &mut R1csBuilder,
    witness: &FinalWitnessWires,
    b: u32,
) -> Result<(), AlphabetError> {
    if b < 2 {
        return Err(AlphabetError { b });
    }
    for &v in &witness.values {
        enforce_centered_alphabet(builder, v, b);
    }
    Ok(())
}

fn enforce_centered_alphabet(builder: &mut R1csBuilder, v: Var, b: u32) {
    debug_assert!(b >= 2, "enforce_balanced_alphabet gates `b >= 2` before this is called");
    let bound = b as i64 - 1;
    let alphabet: Vec<i64> = (-bound..=bound).collect();
    let mut acc: Option<Lc> = None;
    let total = alphabet.len();
    for (i, a) in alphabet.iter().enumerate() {
        let mut factor = Lc::from_var(v);
        let neg_a = if *a >= 0 {
            -F::from_u64(*a as u64)
        } else {
            F::from_u64((-*a) as u64)
        };
        factor.add_constant(neg_a);
        match acc.take() {
            None => acc = Some(factor),
            Some(prev) => {
                if i + 1 == total {
                    builder.enforce(&prev, &factor, &Lc::zero());
                    return;
                }
                let next = builder.alloc_mul(&prev, &factor);
                acc = Some(Lc::from_var(next));
            }
        }
    }
    if let Some(only) = acc {
        builder.enforce(&only, &Lc::from_const(F::ONE), &Lc::zero());
    }
}

//! The prover's view of `neo_verified::r1cs::R1cs`: each side of a row as slices, and the
//! parallel row check on Plonky3 values. Tests compare `fails` with the verified one.

use neo_verified::r1cs::R1cs;
use rayon::prelude::*;

use crate::field::{from_verified, Gl};

/// The columns and coefficients of one side of a row.
pub(crate) type Side<'a> = (&'a [u32], &'a [neo_verified::field::Gl]);

/// The `a`, `b` and `c` sides of row `row`.
pub(crate) fn sides(system: &R1cs, row: usize) -> [Side<'_>; 3] {
    let (start, [a, b, c]) = system.bounds(row);
    let (columns, coefficients) = (system.term_columns(), system.term_coefficients());
    let side = |from: usize, to: usize| (&columns[from..to], &coefficients[from..to]);
    [side(start, a), side(a, b), side(b, c)]
}

/// Whether some row fails on the column values `values`.
pub(crate) fn fails(system: &R1cs, values: &[Gl]) -> bool {
    let sum = |(columns, coefficients): Side<'_>| -> Gl {
        columns
            .iter()
            .zip(coefficients)
            .map(|(&column, &coefficient)| from_verified(coefficient) * values[column as usize])
            .sum()
    };
    (0..system.rows()).into_par_iter().any(|row| {
        let [a, b, c] = sides(system, row);
        sum(a) * sum(b) != sum(c)
    })
}

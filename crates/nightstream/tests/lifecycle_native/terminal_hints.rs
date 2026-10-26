//! The computed columns of the terminal package, which `Package::apply`
//! reads from one hint. The shrink prover chooses every hint word. Every
//! word that layer 1 reads must be fixed by its rows' terms alone, so that a
//! change of that word alone fails on every witness that the rows accept.
//! Changes of several words together are outside this test; Lean proves
//! what the rows mean.

use neo_math::F;
use nightstream_fprime::TerminalOutputs;
use p3_field::PrimeCharacteristicRing;

use super::{package, Package, Source};

/// Whether a row fixes each column by its terms alone: the column has a
/// nonzero net coefficient on C and is absent from A and B, or has one on A
/// or B while the other product side is a nonzero constant and the column is
/// absent from the rest. A change of the column alone then changes that row
/// by a nonzero amount, so the row fails on every witness that the rows
/// accept.
fn fixed_columns(package: &Package) -> Vec<bool> {
    let constant = |side: &[(usize, u64)]| {
        !side.is_empty()
            && side
                .iter()
                .all(|&(column, _)| matches!(package.sources[column], Source::One))
            && side.iter().map(|&(_, k)| F::from_u64(k)).sum::<F>() != F::ZERO
    };
    let mut fixed = vec![false; package.sources.len()];
    for row in 0..package.system.rows() {
        let [a, b, c] = package.system.row(row);
        let mut fix = |side: &[(usize, u64)], others: [&[(usize, u64)]; 2]| {
            for column in alone(side, others) {
                fixed[column] = true;
            }
        };
        fix(&c, [&a, &b]);
        if constant(&b) {
            fix(&a, [&b, &c]);
        }
        if constant(&a) {
            fix(&b, [&a, &c]);
        }
    }
    fixed
}

/// The columns of `side` with a nonzero net coefficient that `others` do
/// not name.
fn alone(side: &[(usize, u64)], others: [&[(usize, u64)]; 2]) -> Vec<usize> {
    let mut terms = side.to_vec();
    terms.sort_unstable();
    terms
        .chunk_by(|x, y| x.0 == y.0)
        .filter(|run| run.iter().map(|&(_, k)| F::from_u64(k)).sum::<F>() != F::ZERO)
        .map(|run| run[0].0)
        .filter(|&column| {
            others
                .iter()
                .all(|other| other.iter().all(|&(c, _)| c != column))
        })
        .collect()
}

/// Every output column of `apply` (the claim and transcript that layer 1
/// reads) is a hint word that a row fixes by its terms alone. No row names
/// both an output and a hint word that no row fixes, so such a word cannot
/// move an output through a shared row.
#[test]
fn every_output_hint_word_is_fixed_by_its_rows() {
    let package = package().unwrap();
    let TerminalOutputs {
        commitment,
        public_input,
        point,
        eval_k,
        eval_a,
        state,
    } = &package.outputs;
    let fixed = fixed_columns(package);
    let mut output = vec![false; package.sources.len()];
    for &column in [commitment, public_input, point, eval_k, eval_a, state]
        .into_iter()
        .flatten()
    {
        let Source::Computed(word) = package.sources[column] else {
            panic!("output column {column} is not a hint word");
        };
        assert!(fixed[column], "output hint word {word} is not fixed by its rows");
        output[column] = true;
    }
    let unfixed: Vec<bool> = package
        .sources
        .iter()
        .zip(&fixed)
        .map(|(source, &fixed)| matches!(source, Source::Computed(_)) && !fixed)
        .collect();
    for row in 0..package.system.rows() {
        let columns: Vec<usize> = package
            .system
            .row(row)
            .into_iter()
            .flatten()
            .map(|(column, _)| column)
            .collect();
        assert!(
            !(columns.iter().any(|&c| unfixed[c]) && columns.iter().any(|&c| output[c])),
            "row {row} names an output and a hint word that no row fixes"
        );
    }
    eprintln!(
        "package hint: {} words, {} not fixed by their rows' terms alone",
        package.computed.len(),
        unfixed.iter().filter(|&&unfixed| unfixed).count()
    );
}

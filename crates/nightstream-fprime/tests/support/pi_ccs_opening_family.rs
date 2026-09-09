//! Independent Pad or matrix-family evaluation at one complete point.
//! Uses only the raw Lean package decoder and independent opening arithmetic.

use rayon::prelude::*;

use super::{
    opening::{add_rings, evaluate_block, EqualityTensor, Extension, Ring, DEGREE},
    reference::{matrix::MatrixProgram, source::SourcePackage},
};

pub fn evaluate_family(
    bytes: &[u8],
    logical_width: usize,
    row_count: usize,
    carrier: &[u8],
    weights: &EqualityTensor,
    matrix: Option<usize>,
) -> Ring {
    match matrix {
        None => carrier
            .par_chunks_exact(DEGREE)
            .enumerate()
            .map(|(block, values)| {
                if values.iter().all(|&value| value == 0) {
                    return [Extension::ZERO; DEGREE];
                }
                let coefficients = std::array::from_fn(|lane| weights.at(block * DEGREE + lane));
                evaluate_block(&coefficients, values)
            })
            .reduce(|| [Extension::ZERO; DEGREE], add_rings),
        Some(matrix) => {
            let artifact = SourcePackage::decode(bytes).expect("independent canonical Lean decoder");
            assert_eq!(
                (artifact.logical_rows, artifact.logical_columns, artifact.cube_variables),
                (row_count, logical_width, 28)
            );
            let program = MatrixProgram::decode(&artifact.matrix_program, &artifact.sources, logical_width, row_count)
                .expect("independent canonical matrix program");
            let workers = rayon::current_num_threads();
            let rows_per_worker = row_count.div_ceil(workers);
            (0..workers)
                .into_par_iter()
                .map(|worker| {
                    let mut blocks: Vec<Option<Box<Ring>>> = vec![None; carrier.len() / DEGREE];
                    let start = (worker * rows_per_worker).min(row_count);
                    let end = (start + rows_per_worker).min(row_count);
                    let mut covered = 0;
                    program
                        .visit_rows(start, end, &artifact.sources, |row, forms| {
                            assert_eq!(row, start + covered, "canonical row order");
                            let weight = weights.at(row);
                            for entry in forms[matrix].entries() {
                                let block = entry.column / DEGREE;
                                let lane = entry.column % DEGREE;
                                let coefficients =
                                    blocks[block].get_or_insert_with(|| Box::new([Extension::ZERO; DEGREE]));
                                coefficients[lane] += weight.scale(entry.coefficient);
                            }
                            covered += 1;
                            Ok(())
                        })
                        .expect("independent canonical row traversal");
                    assert_eq!(covered, end - start);
                    blocks
                        .into_iter()
                        .enumerate()
                        .filter_map(|(block, coefficients)| {
                            coefficients.map(|coefficients| {
                                evaluate_block(&coefficients, &carrier[block * DEGREE..(block + 1) * DEGREE])
                            })
                        })
                        .fold([Extension::ZERO; DEGREE], add_rings)
                })
                .reduce(|| [Extension::ZERO; DEGREE], add_rings)
        }
    }
}

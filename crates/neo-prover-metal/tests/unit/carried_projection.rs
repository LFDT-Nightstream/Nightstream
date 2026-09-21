use super::*;
use neo_ccs::{CcsStructure, GeometricRowRun, SparsePoly};
use neo_reductions::{engines::pi_ccs_joint::build_joint_dims, superneo_eval::SuperneoEvalCacheBuilder, Challenges};

#[test]
fn carried_projection_matches_cpu_with_all_production_sources_active() {
    carried_projection_case(3, D + 41, &[]);
    carried_projection_case(2 * D + 2, D + 41, &[]);
}

#[test]
fn carried_projection_scratch_does_not_grow_with_carrier_width() {
    let small = carried_projection_case(3, D + 41, &[]);
    let large = carried_projection_case(3, 2 * D + 41, &[]);
    // The output gains one ring block. Matrix-row scratch stays the same size.
    assert_eq!(large - small, (D * size_of::<K>()) as u64);
}

#[test]
fn omitted_zero_sources_keep_their_logical_gamma_indices() {
    let count = neo_params::NeoParams::nightstream_goldilocks_k16().k_rho as usize;
    let zero_sources = (0..=count)
        .filter(|&source| source != 2)
        .collect::<Vec<_>>();
    carried_projection_case(3, D + 41, &zero_sources);
}

fn carried_projection_case(rows: usize, columns: usize, zero_sources: &[usize]) -> u64 {
    let blocks = columns.div_ceil(D);
    let width = blocks * D;
    let table_len = width.max(rows);
    let matrix_count = 2;
    let mut builder = SuperneoEvalCacheBuilder::new(rows, columns, matrix_count).unwrap();
    let mut expanded = SuperneoEvalCacheBuilder::new(rows, columns, matrix_count).unwrap();
    for row in 0..rows {
        for matrix in 0..matrix_count {
            let entries = [(row % D, F::from_usize(row + matrix + 2)), (D, -F::ONE)];
            let run = GeometricRowRun::new(row, D - 2, 41, F::from_usize(row + 1), F::from_u64(3));
            builder
                .push_row_with_runs(matrix, row, entries, [run])
                .unwrap();
            let mut coefficients = vec![F::ZERO; columns];
            for (column, coefficient) in entries {
                coefficients[column] += coefficient;
            }
            for index in 0..41 {
                coefficients[D - 2 + index] += F::from_usize(row + 1) * F::from_u64(3).exp_u64(index as u64);
            }
            expanded
                .push_row(
                    matrix,
                    row,
                    coefficients
                        .into_iter()
                        .enumerate()
                        .filter(|(_, value)| *value != F::ZERO),
                )
                .unwrap();
        }
    }
    let cache = Arc::new(builder.finish().unwrap());
    let expanded = expanded.finish().unwrap();
    let structure =
        CcsStructure::new_verifier_artifact_header(rows, columns, matrix_count, SparsePoly::new(matrix_count, vec![]))
            .unwrap();
    let mut params = neo_params::NeoParams::nightstream_goldilocks_k16();
    let security = params
        .padded_row_security_summary_for_shape(
            rows,
            columns,
            matrix_count,
            structure.max_degree(),
            neo_params::goldilocks_paper_b2::CHALLENGE_ALPHABET.len() as u32,
        )
        .unwrap();
    params.lambda = params.lambda.min(security.security_bits);
    let count = params.k_rho as usize;
    let dims = build_joint_dims(&params, &structure, 1, count).unwrap();
    let gamma = K::from_coeffs([F::from_u64(3), F::from_u64(5)]);
    let mut words = vec![0u64; (count + 1) * 2 * blocks];
    let mut witnesses = Vec::new();
    for source in 0..=count {
        let mut witness = Mat::zero(D, blocks, F::ZERO);
        for column in 0..width {
            let digit = (source + 7 * column) % 3;
            if digit != 0 && !zero_sources.contains(&source) {
                witness[(column % D, column / D)] = if digit == 1 { F::ONE } else { -F::ONE };
                words[2 * (source * blocks + column / D) + usize::from(digit == 2)] |= 1 << (column % D);
            }
        }
        witnesses.push(witness);
    }
    let fresh = [neo_ccs::CcsWitness {
        w: vec![],
        Z: witnesses.remove(0),
    }];
    let combined = (0..width)
        .map(|column| {
            witnesses
                .iter()
                .enumerate()
                .map(|(source, witness)| k_power(gamma, source) * K::from(witness[(column % D, column / D)]))
                .sum::<K>()
        })
        .collect::<Vec<_>>();
    let combined = SuperneoZBlocks::from_z(&combined);
    // The CPU evaluator owns the independent ring transformation. A separate
    // identity matrix also checks the contribution outside application rows.
    let mut identity = SuperneoEvalCacheBuilder::new(width, width, 1).unwrap();
    for row in 0..width {
        identity.push_row(0, row, [(row, F::ONE)]).unwrap();
    }
    let identity = identity.finish().unwrap();
    let weights = std::array::from_fn(|coefficient| k_power(gamma, count * coefficient));
    let mut expected = identity.eval_weighted_row_table(&combined, &weights, &[K::ONE], width, width);
    expected.resize(table_len, K::ZERO);
    let matrix_weights = std::array::from_fn(|coefficient| k_power(gamma, count * matrix_count * coefficient));
    let coefficients = (0..matrix_count)
        .map(|matrix| k_power(gamma, count * D + count * matrix))
        .collect::<Vec<_>>();
    let matrix = expanded.eval_weighted_row_table(&combined, &matrix_weights, &coefficients, rows, rows);
    for (output, matrix) in expected.iter_mut().zip(matrix) {
        *output += matrix;
    }
    let point = vec![K::ZERO; dims.variables];
    let input = PaperJointOracleInput {
        structure: &structure,
        params: &params,
        fresh_witnesses: &fresh,
        running_witnesses: &witnesses,
        challenges: Challenges::new(point.clone(), gamma),
        prior_point: Some(&point),
        dims,
        cache: Arc::clone(&cache),
    };
    let session = MetalSession::new().unwrap();
    let plan = session.prepare_joint_matrix_plan(cache).unwrap();
    let masks = session
        .prepare_witness_digit_masks(&words, count + 1, blocks, 1, columns)
        .unwrap();
    let before = session.activity().allocated_bytes;
    let (output, len) = session
        .build_joint_common_tables(&plan, &input, &masks, true)
        .unwrap();
    let allocated = session.activity().allocated_bytes - before;
    assert_eq!(len, table_len);
    let actual = session
        .read_buffer::<u64>(&output, 2 * len)
        .chunks_exact(2)
        .map(|words| K::from_coeffs([F::from_u64(words[0]), F::from_u64(words[1])]))
        .collect::<Vec<_>>();
    assert_eq!(actual, expected);
    allocated
}

use super::*;

#[test]
fn source_residual_matches_all_ports_when_zero_factors_and_powers_mix() {
    let rows = PlanRows {
        ordinary: 3,
        coefficient_starts: Vec::new(),
        piece_ranges: Vec::new(),
        rings: Vec::new(),
        offsets: vec![0, 0, 0, 1, 2, 4, 4, 5, 5, 5, 7, 8, 8, 9, 10, 11, 12],
        columns: vec![1, 2, 0, 2, 1, 0, 1, 2, 0, 1, 2, 3],
        coefficients: vec![
            Fq::ONE,
            Fq::NEG_ONE,
            Fq::NEG_ONE,
            Fq::TWO,
            Fq::ONE,
            Fq::TWO,
            Fq::ONE,
            Fq::ONE,
            Fq::ONE,
            Fq::ONE,
            Fq::NEG_ONE,
            Fq::ONE,
        ],
    };
    for columns in [
        [Fq::ONE, Fq::TWO, Fq::from_u64(3), Fq::NEG_ONE],
        [Fq::NEG_ONE, Fq::ZERO, Fq::new(Fq::ORDER_U64 / 2), Fq::TWO],
    ] {
        for row in 0..rows.row_count() {
            let [a, b, c, s] = [0, 1, 2, 3].map(|port| rows.evaluate(row, port, &columns));
            assert_eq!(rows.residual(row, &columns), a * b - c + s.exp_u64(SBOX_DEGREE));
        }
    }
}

#[test]
fn sharing_equal_forms_preserves_each_ports_columns_and_coefficients() {
    let offsets = vec![0, 0, 1, 4, 7, 7, 10, 11, 14];
    let coefficients = [1, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 1, 2, 3].map(Fq::from_u64);
    let mut rows = PlanRows {
        ordinary: 2,
        piece_ranges: Vec::new(),
        rings: Vec::new(),
        columns: (0..coefficients.len() as u32).collect(),
        offsets,
        coefficients: coefficients.to_vec(),
        coefficient_starts: Vec::new(),
    };
    rows.share_coefficients();
    assert_eq!(rows.coefficients.len(), 4);
    let entries = rows.entries();
    for (port, bounds) in rows.offsets.windows(2).enumerate() {
        assert_eq!(
            entries.port(port),
            (&rows.columns[bounds[0]..bounds[1]], &coefficients[bounds[0]..bounds[1]])
        );
    }
}

use super::*;

#[test]
fn source_residual_matches_all_ports_when_zero_factors_and_powers_mix() {
    let rows = PlanRows {
        ordinary: 3,
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

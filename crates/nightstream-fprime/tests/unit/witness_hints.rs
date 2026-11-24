use super::*;
use serde_json::json;

#[test]
fn consecutive_hints_read_the_previous_output_after_remapping() {
    let source = json!([3, [0, 2], [0, 1]]);
    let raw = RawWitnessBatch(2, vec![], vec![json!([2, source]), json!([3, source])]);
    let batch = validate_witness_batch(raw, 2, 4, 0, 4).unwrap();
    let shifted = batch.map_columns(&|column| Some(column + 1)).unwrap();
    for batch in [batch, shifted] {
        let start = batch.start;
        let mut columns = vec![Fq::ZERO; batch.end()];
        columns[start - 1] = Fq::from_u64(2);
        columns[start] = Fq::from_u64(9);
        execute_witness_batch(&batch, &mut columns);
        assert_eq!(&columns[start..], &[Fq::from_u64(3), Fq::ONE]);
    }
}

use super::*;

#[test]
fn deferred_remapping_checks_references_and_keeps_only_layout_ready() {
    let batch = WitnessBatch {
        start: 12,
        recipes: vec![WitnessExpr::Add(Box::new([WitnessExpr::Var(1), WitnessExpr::Const(Fq::ONE)]))],
        hints: vec![WitnessHint::Bit(WitnessExpr::Var(12), 0)],
    };
    let source = WitnessProgram::new(vec![batch]);
    let excluded = 2..8;
    let invalid = |column| (column != 1).then_some(column);
    assert!(source.remap(excluded.clone(), invalid, true).is_err());
    let invalid = |column| (column != 13).then_some(column);
    assert!(source.remap(excluded.clone(), invalid, true).is_err());

    let map = |column| Some(if column < 2 { column } else { column + 4 });
    let deferred = source.remap(excluded.clone(), map, true).unwrap();
    let WitnessProgram::Deferred { batches, .. } = &deferred else {
        panic!("expected a deferred witness program");
    };
    assert_eq!(deferred.len(), 1);
    assert_eq!(deferred.bit_columns(), [17]);
    assert!(LazyLock::get(batches.as_ref()).is_none());

    let eager = source.remap(excluded, map, false).unwrap();
    let execute = |program: &WitnessProgram| {
        let mut assignment = [Fq::ZERO; 18];
        assignment[1] = Fq::from_u64(2);
        execute_witness_batch(&program.batches().unwrap()[0], &mut assignment);
        assignment
    };
    let actual = execute(&deferred);
    assert_eq!(actual, execute(&eager));
    assert_eq!(actual[16..], [Fq::from_u64(3), Fq::ONE]);
    assert!(LazyLock::get(batches.as_ref()).is_some());
}

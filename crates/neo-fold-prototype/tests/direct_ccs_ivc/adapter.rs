use super::*;

#[test]
fn direct_sparse_r1cs_adapter_builds_program_and_step() {
    let params = NeoParams::goldilocks_auto_r1cs_ccs(1).expect("valid tiny R1CS params");
    let a = CcsMatrix::Csc(CscMat::from_triplets(vec![(0, 0, F::ONE)], 1, D));
    let b = CcsMatrix::Csc(CscMat::from_triplets(
        vec![(0, 1, F::ONE), (0, 2, F::ONE), (0, 3, -F::ONE)],
        1,
        D,
    ));
    let c = CcsMatrix::Csc(CscMat::from_triplets(Vec::new(), 1, D));
    let export = DirectSparseR1csExport {
        a: a.clone(),
        b: b.clone(),
        c: c.clone(),
        witness: {
            let mut witness = vec![F::ZERO; D];
            witness[0] = F::ONE;
            witness[1] = F::from_u64(2);
            witness[2] = F::from_u64(3);
            witness[3] = F::from_u64(5);
            witness
        },
        public_input_len: 4,
        constraint_count: 1,
        variable_count: D,
    };
    let program_from_export = export
        .to_direct_ccs_program()
        .expect("exported direct CCS program");
    assert_eq!(program_from_export.structure().n, 1);
    assert_eq!(program_from_export.structure().m, D);

    let program = direct_ccs_program_from_sparse_r1cs_with_public_input_len(&params, a, b, c, 4)
        .expect("direct sparse R1CS adapter");
    let log = make_ajtai_module(&params);

    let step = export
        .clone()
        .into_direct_ccs_step(&program, &log, "r1cs_fib_step_export")
        .expect("exported direct sparse R1CS step");
    let step_from_witness =
        direct_ccs_step_from_low_norm_full_witness(&program, &log, "r1cs_fib_step", &export.witness, 4)
            .expect("direct sparse R1CS step from full witness");
    assert_eq!(
        step.clone().into_step_input().mcs.x,
        step_from_witness.into_step_input().mcs.x
    );
    let (_program_again, _step_again) = export
        .into_direct_ccs_program_and_step(&log, "r1cs_fib_step_export_pair")
        .expect("exported direct sparse R1CS program and step pair");
    let direct = DirectCcsIvcState::new(program)
        .expect("direct CCS state")
        .append_step(step, &log, ajtai_mixers())
        .expect("direct sparse R1CS step");

    assert_eq!(direct.final_state().chunk_count, 1);
    assert_eq!(direct.final_state().step_count, 1);
}

#[test]
fn direct_sparse_r1cs_adapter_rejects_non_low_norm_witness() {
    let mut export = tiny_sparse_r1cs_export(D, 4);
    export.witness[3] = F::from_u64(1u64 << 60);
    let program = export.to_direct_ccs_program().expect("direct CCS program");
    let log = make_ajtai_module(program.params());

    let err = match direct_ccs_step_from_low_norm_full_witness(&program, &log, "bad_r1cs_step", &export.witness, 4) {
        Ok(_) => panic!("direct R1CS adapter must reject witnesses outside the SuperNeo low-norm budget"),
        Err(err) => err,
    };
    assert!(
        err.to_string().contains("not SuperNeo low-norm packable"),
        "unexpected direct R1CS low-norm rejection: {err}"
    );

    let err = match export.into_direct_ccs_step(&program, &log, "bad_r1cs_step_export") {
        Ok(_) => panic!("exported direct R1CS adapter must preserve the low-norm rejection"),
        Err(err) => err,
    };
    assert!(
        err.to_string().contains("not SuperNeo low-norm packable"),
        "unexpected exported direct R1CS low-norm rejection: {err}"
    );
}

fn tiny_sparse_r1cs_export(variable_count: usize, public_input_len: usize) -> DirectSparseR1csExport {
    assert!(variable_count >= public_input_len);
    let a = CcsMatrix::Csc(CscMat::from_triplets(vec![(0, 0, F::ONE)], 1, variable_count));
    let b = CcsMatrix::Csc(CscMat::from_triplets(vec![(0, 1, F::ONE)], 1, variable_count));
    let c = CcsMatrix::Csc(CscMat::from_triplets(vec![(0, 2, F::ONE)], 1, variable_count));
    let mut witness = vec![F::ZERO; variable_count];
    witness[0] = F::ONE;
    witness[1] = F::ONE;
    witness[2] = F::ONE;
    DirectSparseR1csExport {
        a,
        b,
        c,
        witness,
        public_input_len,
        constraint_count: 1,
        variable_count,
    }
}

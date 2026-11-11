use super::*;

fn module(value: u32) -> String {
    format!(
        r#"(module
        (memory 1)
        (func (export "main") (result i32)
            i32.const 0 i32.const {value} i32.store
            i32.const 0 i32.load
            if (result i32) i32.const 7 else i32.const 9 end))"#
    )
}

#[test]
#[ignore = "Creates retained compression setup files. Run this test alone."]
fn full_wasm_packed_range_terminal_proof_checks_memory_control_and_program() {
    let per_step = per_step(4);
    let source = [module(42), module(43)];
    let n = source
        .iter()
        .map(|wat| common::checked_main(wat).trace.len())
        .max()
        .unwrap()
        .div_ceil(per_step.get())
        .next_power_of_two();
    let setup = Setup::of(source.iter().map(|wat| traced(wat, n, per_step)).collect(), n, per_step);
    let (app, geometry) = setup.complete();
    eprintln!("packed_range_terminal_geometry\t{geometry:?}");
    let circuit = nightstream::nebula::Circuit::compile(geometry, app.clone()).unwrap();
    let prover = circuit.prover(nightstream::Engine::Optimized, 109).unwrap();
    let verifier = circuit
        .verifier(nightstream::Engine::Optimized, 109)
        .unwrap();
    let programs = setup
        .programs
        .iter()
        .map(|trace| app.program(&trace.preload).unwrap())
        .collect::<Vec<_>>();
    let proofs = setup
        .programs
        .iter()
        .zip(&programs)
        .map(|(trace, program)| {
            let mut run = prover.start(program, trace.initial).unwrap();
            run.segment(&trace.inputs).unwrap();
            let proof = run.finish().unwrap();
            verifier
                .verify(program, &trace.initial, &trace.last, &proof)
                .unwrap();
            proof
        })
        .collect::<Vec<_>>();
    let trace = &setup.programs[0];
    assert!(verifier
        .verify(&programs[0], &trace.initial, &trace.last, &proofs[1])
        .is_err());
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let directory = std::env::temp_dir().join(format!(
        "nightstream-packed-range-compression-{}-{stamp}",
        std::process::id()
    ));
    std::fs::create_dir(&directory).unwrap();
    eprintln!("packed_range_terminal_setup\t{}", directory.display());
    let compression = prover.compression_setup(&directory).unwrap();
    let final_proof = prover.compress(&proofs[0], &compression).unwrap();
    let key = verifier.compression_key().unwrap();
    verifier
        .verify_final(&programs[0], &trace.initial, &trace.last, &key, &final_proof)
        .unwrap();
    assert!(verifier
        .verify_final(&programs[1], &trace.initial, &trace.last, &key, &final_proof)
        .is_err());
    let mut wrong = trace.last;
    wrong.pc += 1;
    assert!(verifier
        .verify_final(&programs[0], &trace.initial, &wrong, &key, &final_proof)
        .is_err());
    eprintln!("packed_range_terminal_pass\t109");
}

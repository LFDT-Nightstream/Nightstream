use std::time::Instant;

use neo_fold_next::rv64im::audit::{
    build_rv64im_chunk_step_ivc_relations, build_rv64im_main_recursion_f_prime_advices,
    build_rv64im_main_recursion_f_prime_backend_relations_with_spartan_shape_from_advices,
    build_rv64im_main_recursion_step_spartan_compressed_chain_shape,
    debug_check_rv64im_main_recursion_step_spartan_compressed_chain_public_io,
    debug_measure_rv64im_main_recursion_step_spartan_circuit_shape,
    debug_measure_rv64im_main_recursion_step_spartan_commitment_key,
    debug_measure_rv64im_main_recursion_step_spartan_compressed_chain_circuit_shape,
    debug_measure_rv64im_main_recursion_step_spartan_shape_synthesis,
    debug_profile_rv64im_main_recursion_step_chunk_replay_stages,
    debug_profile_rv64im_main_recursion_step_spartan_compressed_chain_prove_stages,
    prove_rv64im_main_recursion_step_spartan, setup_rv64im_main_recursion_step_spartan_cached,
};
use neo_fold_next::rv64im::final_relation::prove_rv64im_final_statement_from_accepted;
use neo_fold_next::rv64im::{
    build_mixed_opcode_perf_source_case, build_rv64im_accepted_proof_artifact,
    build_rv64im_kernel_export_source_from_accepted_artifact, prove_rv64im_public_proof_with_perf, Rv64imProofInput,
};

fn elapsed_ms(started: Instant) -> f64 {
    started.elapsed().as_secs_f64() * 1_000.0
}

fn print_probe(label: &str, ms: f64) {
    println!("n2-latency|{label}|{ms:.3}");
}

#[test]
#[ignore = "manual n=2 latency probe; run with --release -- --ignored --nocapture and a 30s timeout"]
fn rv64im_n2_latency_probe() {
    let source = build_mixed_opcode_perf_source_case(2);
    let input = Rv64imProofInput {
        max_steps: source.program_words.len(),
        source,
    };

    println!("n2-latency|opcode_count|{}", input.max_steps);

    println!("n2-latency|start|prove_public");
    let started = Instant::now();
    let (proof, proof_perf) = prove_rv64im_public_proof_with_perf(&input).expect("prove n=2 public proof");
    print_probe("prove_public.wall", elapsed_ms(started));
    print_probe("prove_public.total", proof_perf.total_ms);
    print_probe("prove_public.shared_trace", proof_perf.shared_trace_ms);
    print_probe("prove_public.simple_kernel", proof_perf.simple_kernel.total_ms);
    print_probe("prove_public.root_main_lane", proof_perf.root_main_lane.total_ms);
    print_probe("prove_public.public_export", proof_perf.public_export_ms);

    println!("n2-latency|start|build_accepted_artifact");
    let started = Instant::now();
    let accepted_artifact = build_rv64im_accepted_proof_artifact(&proof).expect("build n=2 accepted artifact");
    print_probe("build_accepted_artifact.wall", elapsed_ms(started));

    println!("n2-latency|start|build_kernel_export_source");
    let started = Instant::now();
    let _kernel_export_source = build_rv64im_kernel_export_source_from_accepted_artifact(&accepted_artifact)
        .expect("build n=2 kernel-export source");
    print_probe("build_kernel_export_source.wall", elapsed_ms(started));

    println!("n2-latency|start|prove_final_statement");
    let started = Instant::now();
    let (final_statement, final_proof) =
        prove_rv64im_final_statement_from_accepted(&accepted_artifact).expect("prove n=2 final statement");
    print_probe("prove_final_statement.wall", elapsed_ms(started));

    println!("n2-latency|start|build_chunk_step_ivc_relations");
    let started = Instant::now();
    let relations = build_rv64im_chunk_step_ivc_relations(&final_statement, &final_proof)
        .expect("build n=2 chunk-step ivc relations");
    print_probe("build_chunk_step_ivc_relations.wall", elapsed_ms(started));

    println!("n2-latency|start|build_f_prime_advices");
    let started = Instant::now();
    let advices = build_rv64im_main_recursion_f_prime_advices(&relations).expect("build n=2 f-prime advices");
    print_probe("build_f_prime_advices.wall", elapsed_ms(started));

    println!("n2-latency|start|build_backend_relations");
    let started = Instant::now();
    let (spartan_shape, backend_relations) =
        build_rv64im_main_recursion_f_prime_backend_relations_with_spartan_shape_from_advices(&relations, &advices)
            .expect("build n=2 recursion backend relations");
    print_probe("build_backend_relations.wall", elapsed_ms(started));
    println!("n2-latency|backend_relations|{}", backend_relations.len());

    let first_relation = backend_relations
        .first()
        .expect("first n=2 backend relation");

    println!("n2-latency|start|profile_first_step_chunk_replay");
    let started = Instant::now();
    debug_profile_rv64im_main_recursion_step_chunk_replay_stages(first_relation)
        .expect("profile n=2 first-step chunk replay stages");
    print_probe("profile_first_step_chunk_replay.wall", elapsed_ms(started));

    println!("n2-latency|start|measure_first_step_shape_synthesis");
    let started = Instant::now();
    let shape_synth = debug_measure_rv64im_main_recursion_step_spartan_shape_synthesis(&spartan_shape, first_relation)
        .expect("measure n=2 first-step shape synthesis");
    print_probe("measure_first_step_shape_synthesis.wall", elapsed_ms(started));
    print_probe("measure_first_step_shape_synthesis.shared", shape_synth.shared_ms);
    print_probe(
        "measure_first_step_shape_synthesis.precommitted",
        shape_synth.precommitted_ms,
    );
    print_probe(
        "measure_first_step_shape_synthesis.synthesize",
        shape_synth.synthesize_ms,
    );
    println!("n2-latency|shape_synth.num_inputs|{}", shape_synth.num_inputs);
    println!("n2-latency|shape_synth.num_aux|{}", shape_synth.num_aux);
    println!("n2-latency|shape_synth.num_constraints|{}", shape_synth.num_constraints);

    println!("n2-latency|start|measure_first_step_circuit_shape");
    let started = Instant::now();
    let first_step_shape =
        debug_measure_rv64im_main_recursion_step_spartan_circuit_shape(&spartan_shape, first_relation)
            .expect("measure n=2 first-step circuit shape");
    print_probe("measure_first_step_circuit_shape.wall", elapsed_ms(started));
    println!("n2-latency|first_step.num_inputs|{}", first_step_shape.num_inputs);
    println!("n2-latency|first_step.num_aux|{}", first_step_shape.num_aux);
    println!(
        "n2-latency|first_step.num_constraints|{}",
        first_step_shape.num_constraints
    );

    println!("n2-latency|start|measure_first_step_commitment_key");
    let commitment_key_ms =
        debug_measure_rv64im_main_recursion_step_spartan_commitment_key(&spartan_shape, first_relation)
            .expect("measure n=2 first-step commitment key");
    print_probe("measure_first_step_commitment_key.wall", commitment_key_ms);

    println!("n2-latency|start|setup_first_step");
    let started = Instant::now();
    let first_step_keys = setup_rv64im_main_recursion_step_spartan_cached(&spartan_shape, first_relation)
        .expect("setup n=2 first recursive step");
    print_probe("setup_first_step.wall", elapsed_ms(started));

    println!("n2-latency|start|prove_first_step");
    let started = Instant::now();
    let (pk, _) = &*first_step_keys;
    let _first_step_proof = prove_rv64im_main_recursion_step_spartan(pk, &spartan_shape, first_relation)
        .expect("prove n=2 first recursive step");
    print_probe("prove_first_step.wall", elapsed_ms(started));

    println!("n2-latency|start|build_compressed_chain_shape");
    let started = Instant::now();
    let chain_shape =
        build_rv64im_main_recursion_step_spartan_compressed_chain_shape(&spartan_shape, &backend_relations)
            .expect("build n=2 compressed-chain shape");
    print_probe("build_compressed_chain_shape.wall", elapsed_ms(started));

    println!("n2-latency|start|build_compressed_chain_circuit");
    let started = Instant::now();
    debug_check_rv64im_main_recursion_step_spartan_compressed_chain_public_io(&chain_shape, &backend_relations)
        .expect("build n=2 compressed-chain circuit");
    print_probe("build_compressed_chain_circuit.wall", elapsed_ms(started));

    println!("n2-latency|start|profile_compressed_chain_prove");
    let started = Instant::now();
    let compressed_chain_metrics = debug_profile_rv64im_main_recursion_step_spartan_compressed_chain_prove_stages(
        &spartan_shape,
        &backend_relations,
    )
    .expect("profile n=2 compressed-chain prove stages");
    print_probe("profile_compressed_chain_prove.wall", elapsed_ms(started));
    print_probe(
        "profile_compressed_chain_prove.setup",
        compressed_chain_metrics.setup_ms,
    );
    print_probe(
        "profile_compressed_chain_prove.prep_prove",
        compressed_chain_metrics.prep_prove_ms,
    );
    print_probe(
        "profile_compressed_chain_prove.prove",
        compressed_chain_metrics.prove_ms,
    );
    print_probe(
        "profile_compressed_chain_prove.serialize",
        compressed_chain_metrics.serialize_ms,
    );
    println!(
        "n2-latency|compressed_chain.snark_bytes|{}",
        compressed_chain_metrics.snark_bytes
    );
}

#[test]
#[ignore = "manual n=2 shape probe; run separately when the main latency path is already narrowed"]
fn rv64im_n2_shape_probe() {
    let source = build_mixed_opcode_perf_source_case(2);
    let input = Rv64imProofInput {
        max_steps: source.program_words.len(),
        source,
    };
    let (proof, _) = prove_rv64im_public_proof_with_perf(&input).expect("prove n=2 public proof");
    let accepted_artifact = build_rv64im_accepted_proof_artifact(&proof).expect("build n=2 accepted artifact");
    let (final_statement, final_proof) =
        prove_rv64im_final_statement_from_accepted(&accepted_artifact).expect("prove n=2 final statement");
    let relations = build_rv64im_chunk_step_ivc_relations(&final_statement, &final_proof)
        .expect("build n=2 chunk-step ivc relations");
    let advices = build_rv64im_main_recursion_f_prime_advices(&relations).expect("build n=2 f-prime advices");
    let (spartan_shape, backend_relations) =
        build_rv64im_main_recursion_f_prime_backend_relations_with_spartan_shape_from_advices(&relations, &advices)
            .expect("build n=2 recursion backend relations");
    let first_relation = backend_relations
        .first()
        .expect("first n=2 backend relation");

    println!("n2-shape|backend_relations|{}", backend_relations.len());

    let started = Instant::now();
    let first_step_shape =
        debug_measure_rv64im_main_recursion_step_spartan_circuit_shape(&spartan_shape, first_relation)
            .expect("measure n=2 first-step circuit shape");
    print_probe("shape.first_step.wall", elapsed_ms(started));
    println!("n2-shape|first_step.num_inputs|{}", first_step_shape.num_inputs);
    println!("n2-shape|first_step.num_aux|{}", first_step_shape.num_aux);
    println!(
        "n2-shape|first_step.num_constraints|{}",
        first_step_shape.num_constraints
    );

    let chain_shape =
        build_rv64im_main_recursion_step_spartan_compressed_chain_shape(&spartan_shape, &backend_relations)
            .expect("build n=2 compressed-chain shape");
    let started = Instant::now();
    let compressed_chain_shape = debug_measure_rv64im_main_recursion_step_spartan_compressed_chain_circuit_shape(
        &chain_shape,
        &backend_relations,
    )
    .expect("measure n=2 compressed-chain circuit shape");
    print_probe("shape.compressed_chain.wall", elapsed_ms(started));
    println!(
        "n2-shape|compressed_chain.num_inputs|{}",
        compressed_chain_shape.num_inputs
    );
    println!("n2-shape|compressed_chain.num_aux|{}", compressed_chain_shape.num_aux);
    println!(
        "n2-shape|compressed_chain.num_constraints|{}",
        compressed_chain_shape.num_constraints
    );
}

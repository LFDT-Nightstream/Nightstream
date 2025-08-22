use neo_arithmetize::fibonacci_ccs;
use neo_ccs::{CcsInstance, CcsWitness, check_satisfiability};
use neo_commit::{AjtaiCommitter, SECURE_PARAMS};
use neo_decomp::decomp_b;
use neo_fields::{embed_base_to_ext, ExtF, F};
use neo_fold::FoldState;
use neo_ring::RingElement;
use neo_modint::ModInt;
use p3_field::PrimeCharacteristicRing;
use std::time::Instant;

fn main() {
    println!("Neo Lattice Demo: Proving Fibonacci Series");

    // Using secure parameters for production-level security
    println!("DEBUG: Setting up committer with SECURE_PARAMS...");
    println!("DEBUG: SECURE_PARAMS - d: {}, b: {}", SECURE_PARAMS.d, SECURE_PARAMS.b);
    let committer = AjtaiCommitter::setup_unchecked(SECURE_PARAMS);
    println!("DEBUG: Committer setup completed!");

    // High-level input: Length of Fibonacci series to prove
    let fib_length = 3;

    // Convert high-level input to CCS (using new module)
    println!("DEBUG: Creating CCS for fibonacci length {}...", fib_length);
    let ccs = fibonacci_ccs(fib_length);
    println!("DEBUG: CCS creation completed!");

    // Generate witness: Fibonacci sequence
    println!("DEBUG: Generating witness...");
    let mut z: Vec<ExtF> = vec![ExtF::ZERO; fib_length];
    z[0] = embed_base_to_ext(F::ZERO);
    z[1] = embed_base_to_ext(F::ONE);
    for i in 2..fib_length {
        z[i] = z[i - 1] + z[i - 2];
    }
    let witness = CcsWitness { z: z.clone() };
    println!("DEBUG: Witness generation completed!");

    // Pack witness into ring elements for commitment
    println!("DEBUG: Starting witness packing...");
    let z_base: Vec<F> = z.iter().map(|e| e.to_array()[0]).collect();  // Project to base (real part)
    println!("DEBUG: Computing decomposition matrix...");
    let decomp_mat = decomp_b(&z_base, committer.params().b, committer.params().d);
    println!("DEBUG: Packing decomposition...");
    let z_packed: Vec<RingElement<ModInt>> = AjtaiCommitter::pack_decomp(&decomp_mat, &committer.params());
    println!("DEBUG: Witness packing completed!");

    // Commit to packed witness
    println!("DEBUG: Creating commitment...");
    println!("DEBUG: z_packed length: {}", z_packed.len());
    let commit_start = Instant::now();
    let (commitment, _, _, _) = committer.commit(&z_packed, &mut vec![]).unwrap();
    let commit_time = commit_start.elapsed();
    println!("DEBUG: Commitment created in {:.2} ms!", commit_time.as_secs_f64() * 1000.0);

    // Create CCS instance
    println!("DEBUG: Creating CCS instance...");
    let instance = CcsInstance {
        commitment,
        public_input: vec![],
        u: F::ZERO,
        e: F::ONE,
    };
    println!("DEBUG: CCS instance created!");

    // Check satisfiability
    println!("DEBUG: Checking satisfiability...");
    if !check_satisfiability(&ccs, &instance, &witness) {
        println!("Error: Fibonacci witness does not satisfy CCS constraints");
        return;
    }
    println!("DEBUG: Satisfiability check passed!");

    // Setup folding state
    println!("DEBUG: Setting up folding state...");
    let mut fold_state = FoldState::new(ccs);
    println!("DEBUG: Folding state setup completed!");

    // Generate proof for the Fibonacci instance (fold with itself for demo)
    println!("\n=== PROOF GENERATION ===");
    println!("DEBUG: Starting proof generation...");
    let proof_start = Instant::now();
    let proof = fold_state.generate_proof((instance.clone(), witness.clone()), (instance, witness), &committer);
    println!("DEBUG: Proof generation completed!");
    let proof_generation_time = proof_start.elapsed();
    let proof_size_bytes = proof.transcript.len();
    println!("Proof generation completed in {:.2} ms", proof_generation_time.as_secs_f64() * 1000.0);
    println!("Proof size: {} bytes", proof_size_bytes);

    // Verify the proof
    println!("\n=== PROOF VERIFICATION ===");
    println!("DEBUG: Starting proof verification...");
    let verify_start = Instant::now();
    let verified = fold_state.verify(&proof.transcript, &committer);
    println!("DEBUG: Proof verification completed!");
    let verification_time = verify_start.elapsed();
    println!("Proof verification: {}", if verified { "SUCCESS" } else { "FAILURE" });
    println!("Verification completed in {:.2} ms", verification_time.as_secs_f64() * 1000.0);

    // Final Performance Summary
    println!("\n==========================================");
    println!("🏁 FINAL PERFORMANCE SUMMARY");
    println!("==========================================");
    println!("Proof Generation Time:    {:>8.2} ms", proof_generation_time.as_secs_f64() * 1000.0);
    println!("Proof Size:               {:>8} bytes", proof_size_bytes);
    println!("Proof Verification Time:  {:>8.2} ms", verification_time.as_secs_f64() * 1000.0);
    println!("Total Time:               {:>8.2} ms", (proof_generation_time + verification_time).as_secs_f64() * 1000.0);
    println!("Verification Result:      {}", if verified { "✅ PASSED" } else { "❌ FAILED" });
    println!("Fibonacci Length:         {:>8}", fib_length);
    println!("==========================================");
}

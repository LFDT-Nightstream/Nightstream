use neo_ajtai::{set_global_pp, AjtaiSModule, Commitment as Cmt};
use neo_ccs::matrix::Mat;
use neo_ccs::r1cs::r1cs_to_ccs;
use neo_ccs::relations::{CcsStructure, McsInstance};
use neo_fold::folding::FoldRun;
use neo_fold::{
    pi_ccs::FoldingMode,
    session::{FoldingSession, NeoStep, StepArtifacts, StepSpec},
};
use neo_math::{D, F};
use neo_params::NeoParams;
use p3_field::PrimeCharacteristicRing;
use rand_chacha::rand_core::SeedableRng as _;
use serde::{Deserialize, Serialize};
use std::io::{self, Write};

#[derive(Serialize, Deserialize)]
struct SavedProof {
    params: NeoParams,
    ccs: CcsStructure<F>,
    mcss_public: Vec<McsInstance<Cmt, F>>,
    run: FoldRun,
}

fn setup_ajtai_for_dims(m: usize) {
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(420);
    let pp = neo_ajtai::setup(&mut rng, D, 4, m).expect("Ajtai setup should succeed");
    let _ = set_global_pp(pp);
}

fn main() {
    println!("====================================================================================================================");
    println!(
        "This program computes a proof for the computation of the sum of an arbitrary sized list of numbers."
    );
    println!("The circuit works on batches of 4 numbers at a time. Since a circuit is finite, there has to be a fixed step. But with folding, it can be extended to lists of arbitrary sizes.");
    println!(
        "If the list size is not a multiple of 4, the last step has to be completed with zeros."
    );
    println!("====================================================================================================================");

    let step_ccs = step_ccs();

    let mut total_sum = 0;
    let mut step_count = 0;

    let params = NeoParams::goldilocks_auto_r1cs_ccs(step_ccs.n)
        .expect("goldilocks_auto_r1cs_ccs should find valid params");

    setup_ajtai_for_dims(step_ccs.m);
    let l = AjtaiSModule::from_global_for_dims(D, step_ccs.m).expect("AjtaiSModule init");

    let mut circuit = StepCircuit {};

    // NOTE: clone params here so we can also store them into SavedProof later.
    let mut session = FoldingSession::new(FoldingMode::Optimized, params.clone(), l.clone());

    loop {
        let numbers = match collect_four_numbers() {
            Ok(nums) => nums,
            Err(should_quit) => {
                if should_quit {
                    println!(
                        "Completed {} steps with final total sum: {}",
                        step_count, total_sum
                    );
                    break;
                }
                continue;
            }
        };

        let f_inputs = [
            F::from_u64(numbers[0]),
            F::from_u64(numbers[1]),
            F::from_u64(numbers[2]),
            F::from_u64(numbers[3]),
        ];

        let start = std::time::Instant::now();

        session
            .prove_step(
                &mut circuit,
                &Input {
                    new_values: f_inputs,
                },
            )
            .expect("prove_step should succeed with optimized");

        step_count += 1;
        let step_sum: u64 = numbers.iter().sum();
        let prev_sum = total_sum;
        total_sum += step_sum;

        println!("Sum of this step ({}): {}", step_count, step_sum);
        println!("Running sum: {} ( {prev_sum} + {step_sum} )", total_sum);
        println!("Computed new step in {} ms", start.elapsed().as_millis());

        let mcss_public = session.mcss_public();
        if let Some(last) = mcss_public.last() {
            println!("Step {} public x: {:?}", step_count, last.x);
            println!("Step {} commitment c: {:?}", step_count, last.c.data);
        }

        // TODO: what can be print here for the "current step proof"?

        println!("--------------------------------------------------------------------------------------------------------------------");

        loop {
            print!("Add another batch of elements? (y/n): ");
            io::stdout().flush().unwrap();

            let mut input = String::new();
            match io::stdin().read_line(&mut input) {
                Ok(_) => {
                    let input = input.trim().to_lowercase();
                    match input.as_str() {
                        "y" | "yes" => break,
                        "n" | "no" | "quit" => {
                            println!("====================================================================================================================");
                            println!(
                                "Completed {} steps with final total sum: {}. Finalizing aggregated proof.",
                                step_count, total_sum
                            );

                            let start = std::time::Instant::now();
                            let run = session
                                .finalize(&step_ccs)
                                .expect("finalize should produce a FoldRun");

                            let finalize_duration = start.elapsed();

                            println!("Final proof took: {} ms", finalize_duration.as_millis());

                            let mcss_public = session.mcss_public();
                            // Sanity-check: final public state equals the running sum.
                            let last = mcss_public.last().expect("at least one step");
                            assert_eq!(last.x[1], F::from_u64(total_sum));

                            // Save proof to disk.
                            let saved_proof = SavedProof {
                                params: params.clone(),
                                ccs: step_ccs.clone(),
                                mcss_public: mcss_public.clone(),
                                run: run.clone(),
                            };

                            let filename = "proof.bin";
                            match bincode::serialize(&saved_proof) {
                                Ok(bytes) => {
                                    if let Err(e) = std::fs::write(filename, bytes) {
                                        eprintln!("Failed to write proof to file: {}", e);
                                    } else {
                                        println!("Proof saved to {}", filename);
                                    }
                                }
                                Err(e) => eprintln!("Failed to serialize proof: {}", e),
                            }

                            // ---------------------------------------------------------------------------------
                            // Verification phase: reload proof from disk and check both:
                            //  1) cryptographic validity of the proof, and
                            //  2) user-supplied claimed output against the final state in the proof.
                            // ---------------------------------------------------------------------------------
                            println!("Reloading proof from '{}' for verification...", filename);

                            let loaded_bytes = match std::fs::read(filename) {
                                Ok(b) => b,
                                Err(e) => {
                                    eprintln!("Failed to read proof from file: {}", e);
                                    return;
                                }
                            };

                            let loaded: SavedProof = match bincode::deserialize(&loaded_bytes) {
                                Ok(p) => p,
                                Err(e) => {
                                    eprintln!("Failed to deserialize proof: {}", e);
                                    return;
                                }
                            };

                            // Create a fresh session for verification using the loaded params.
                            let verify_session = FoldingSession::new(
                                FoldingMode::Optimized,
                                loaded.params.clone(),
                                l.clone(),
                            );

                            let proof_ok = verify_session
                                .verify(&loaded.ccs, &loaded.mcss_public, &loaded.run)
                                .expect("verify should run");

                            if !proof_ok {
                                eprintln!(
                                    "Cryptographic verification of the loaded proof FAILED."
                                );
                                return;
                            }

                            println!(
                                "Cryptographic verification of the loaded proof succeeded."
                            );

                            // Ask the user for a claimed final output and check it against the proof.
                            println!(
                                "Enter a claimed final sum to check against the proof:"
                            );
                            print!("> ");
                            io::stdout().flush().unwrap();

                            let mut line = String::new();
                            if let Err(e) = io::stdin().read_line(&mut line) {
                                eprintln!("Error reading claimed sum: {}", e);
                                return;
                            }
                            let line = line.trim();
                            let claimed_sum: u64 = match line.parse() {
                                Ok(v) => v,
                                Err(_) => {
                                    eprintln!("Invalid number '{}'", line);
                                    return;
                                }
                            };

                            let final_state = loaded
                                .mcss_public
                                .last()
                                .expect("at least one step in loaded proof")
                                .x[1];

                            if final_state == F::from_u64(claimed_sum) {
                                println!(
                                    "Claimed output {} is CONSISTENT with the proof.",
                                    claimed_sum
                                );
                            } else {
                                println!(
                                    "Claimed output {} is NOT consistent with the proof.",
                                    claimed_sum
                                );
                            }

                            return;
                        }
                        _ => {
                            println!("Please enter 'y' for yes or 'n' for no.");
                            continue;
                        }
                    }
                }
                Err(e) => {
                    eprintln!("Error reading input: {}", e);
                    continue;
                }
            }
        }
    }
}

fn collect_four_numbers() -> Result<[u64; 4], bool> {
    let mut numbers = [0u64; 4];

    for i in 0..4 {
        loop {
            print!("Enter number {} of 4: ", i + 1);
            io::stdout().flush().unwrap();

            let mut input = String::new();
            match io::stdin().read_line(&mut input) {
                Ok(_) => {
                    let input = input.trim();

                    if input.to_lowercase() == "quit" {
                        return Err(true); // Signal to quit
                    }

                    match input.parse::<u64>() {
                        Ok(num) => {
                            numbers[i] = num;
                            break;
                        }
                        Err(_) => {
                            println!(
                                "Invalid number. Please enter a valid number or 'quit' to exit."
                            );
                            continue;
                        }
                    }
                }
                Err(e) => {
                    eprintln!("Error reading input: {}", e);
                    return Err(false); // Signal to retry
                }
            }
        }
    }

    println!("....................................................................................................................");

    Ok(numbers)
}

struct StepCircuit {}

#[derive(Clone)]
struct Input {
    new_values: [F; 4],
}

impl NeoStep for StepCircuit {
    type ExternalInputs = Input;

    fn state_len(&self) -> usize {
        1 // single scalar state: running_sum
    }

    fn step_spec(&self) -> StepSpec {
        StepSpec {
            y_len: 1,
            const1_index: 0,
            // y_step_indices[0] = index in z that holds the *next* state y_{i+1}.
            // We'll place y_{i+1} at z[1].
            y_step_indices: vec![1],
            app_input_indices: Some(vec![]),
            // Public inputs are prefix of z: [1, y_next].
            m_in: 2,
        }
    }

    fn synthesize_step(
        &mut self,
        _step_idx: usize,
        y_prev: &[F],
        inputs: &Self::ExternalInputs,
    ) -> StepArtifacts {
        assert_eq!(y_prev.len(), 1);
        let old_sum = y_prev[0];
        let step_sum = inputs.new_values.iter().copied().sum::<F>();
        let new_sum = old_sum + step_sum;

        StepArtifacts {
            ccs: step_ccs(), // updated below
            witness: vec![
                F::from_u64(1),       // z[0] = 1 (const)
                new_sum,              // z[1] = y_{i+1} (next state, public)
                old_sum,              // z[2] = y_i (previous state, private)
                inputs.new_values[0], // z[3..6] = batch inputs
                inputs.new_values[1],
                inputs.new_values[2],
                inputs.new_values[3],
            ],
            public_app_inputs: vec![],
            spec: self.step_spec().clone(),
        }
    }
}

/// z = [1, new, old, v0, v1, v2, v3]
/// Enforce: old + v0 + v1 + v2 + v3 = new
fn step_ccs() -> neo_ccs::relations::CcsStructure<F> {
    let n = 7; // constraints
    let m = 7; // variables

    // A*z = old + v0 + v1 + v2 + v3 = z[2] + z[3] + z[4] + z[5] + z[6]
    let mut a_data = vec![F::ZERO; n * m];
    a_data[0 * m + 2] = F::ONE; // old
    a_data[0 * m + 3] = F::ONE; // v0
    a_data[0 * m + 4] = F::ONE; // v1
    a_data[0 * m + 5] = F::ONE; // v2
    a_data[0 * m + 6] = F::ONE; // v3
    let a = Mat::from_row_major(n, m, a_data);

    // B*z = 1 (just multiply by constant 1)
    let mut b_data = vec![F::ZERO; n * m];
    b_data[0 * m + 0] = F::ONE; // constant 1 at index 0
    let b = Mat::from_row_major(n, m, b_data);

    // C*z = new = z[1]
    let mut c_data = vec![F::ZERO; n * m];
    c_data[0 * m + 1] = F::ONE; // new at index 1
    let c = Mat::from_row_major(n, m, c_data);

    let s0 = r1cs_to_ccs(a, b, c);

    s0.ensure_identity_first()
        .expect("ensure_identity_first should succeed")
}

use neo_ajtai::{set_global_pp, AjtaiSModule};
use neo_ccs::matrix::Mat;
use neo_ccs::r1cs::r1cs_to_ccs;
use neo_fold::{
    pi_ccs::FoldingMode,
    session::{FoldingSession, NeoStep, StepArtifacts, StepSpec},
};
use neo_math::{D, F};
use neo_params::NeoParams;
use p3_field::PrimeCharacteristicRing;
use rand_chacha::rand_core::SeedableRng as _;
use std::io::{self, Write};

fn setup_ajtai_for_dims(m: usize) {
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(420);
    let pp = neo_ajtai::setup(&mut rng, D, 128, m).expect("Ajtai setup should succeed");
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

    let mut session = FoldingSession::new(FoldingMode::Optimized, params, l.clone());

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
                    running_sum: F::from_u64(total_sum),
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
        println!("Computed proof in {} ms", start.elapsed().as_millis());

        // TODO: what can be print here for the "current step proof"?
        println!("....................................................................................................................");

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

                            // TODO: can we serialize the proof, and save it to a file?
                            // or at least print it

                            let mcss_public = session.mcss_public();

                            assert_eq!(mcss_public.last().unwrap().x[2], F::from_u64(total_sum));

                            // TODO: ideally, here we would prompt the user for a number to verify the proof with?
                            //
                            // at least, my expectation is that the proof can only be verified with a constant number

                            let ok = session
                                .verify(&step_ccs, &mcss_public, &run)
                                .expect("verify should run");

                            assert!(ok, "verification failed");

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
    // TODO: this shouldn't be an input, I think
    // but for some reason it's not y_prev
    //
    // I'm also not sure it's even enforced.
    running_sum: F,
    new_values: [F; 4],
}

impl NeoStep for StepCircuit {
    type ExternalInputs = Input;

    fn state_len(&self) -> usize {
        self.step_spec().y_len
    }

    fn step_spec(&self) -> StepSpec {
        StepSpec {
            y_len: 1,
            const1_index: 0,
            y_step_indices: vec![1, 2],
            app_input_indices: Some(vec![]),
            m_in: 3,
        }
    }

    fn synthesize_step(
        &mut self,
        _step_idx: usize,
        _y_prev: &[F],
        inputs: &Self::ExternalInputs,
    ) -> StepArtifacts {
        let new_sum = inputs.running_sum + inputs.new_values.iter().copied().sum::<F>();

        StepArtifacts {
            ccs: step_ccs(),
            witness: vec![
                F::from_u64(1),       // constant 1
                inputs.running_sum,   // x1 (previous step state)
                new_sum,              // x2 (new running sum)
                inputs.new_values[0], // x3 (input 1)
                inputs.new_values[1], // x4 (input 2)
                inputs.new_values[2], // x5 (input 3)
                inputs.new_values[3], // x6 (input 4)
            ],
            public_app_inputs: vec![],
            spec: self.step_spec().clone(),
        }
    }
}

/// Creates R1CS matrices for the constraint: x1 + x3 + x4 + x5 + x6 = x2
/// Variables: [1, x1, x2, x3, x4, x5, x6] where:
/// - x1 = previous step state
/// - x2 = new running sum (output)
/// - x3, x4, x5, x6 = the 4 input numbers
/// Padded to make square matrices (n = m = 7)
fn step_ccs() -> neo_ccs::relations::CcsStructure<F> {
    // 7 variables: [1, x1, x2, x3, x4, x5, x6]
    // 1 real constraint: x1 + x3 + x4 + x5 + x6 - x2 = 0
    // 6 padding constraints: 0 = 0 (to make n = m = 7)
    // R1CS form: A*z ∘ B*z = C*z where ∘ is element-wise product

    let n = 7; // 7 constraints (1 real + 6 padding)
    let m = 7; // 7 variables

    // Matrix A: coefficients for left side of multiplication
    let mut a_data = vec![F::ZERO; n * m];
    // Real constraint row 0: [0, 1, 0, 1, 1, 1, 1] (x1 + x3 + x4 + x5 + x6)
    a_data[0 * m + 1] = F::ONE; // x1 (previous state)
    a_data[0 * m + 3] = F::ONE; // x3 (input 1)
    a_data[0 * m + 4] = F::ONE; // x4 (input 2)
    a_data[0 * m + 5] = F::ONE; // x5 (input 3)
    a_data[0 * m + 6] = F::ONE; // x6 (input 4)
                                // Padding constraints rows 1-6: all zeros (already initialized)
    let a = Mat::from_row_major(n, m, a_data);

    // Matrix B: coefficients for right side of multiplication
    let mut b_data = vec![F::ZERO; n * m];
    // Real constraint row 0: [1, 0, 0, 0, 0, 0, 0] (multiply by 1)
    b_data[0 * m + 0] = F::ONE; // constant 1
                                // Padding constraints rows 1-6: all zeros (already initialized)
    let b = Mat::from_row_major(n, m, b_data);

    // Matrix C: coefficients for result
    let mut c_data = vec![F::ZERO; n * m];
    // Real constraint row 0: [0, 0, 1, 0, 0, 0, 0] (equals x2)
    c_data[0 * m + 2] = F::ONE; // x2 (new running sum)
                                // Padding constraints rows 1-6: all zeros (already initialized)
    let c = Mat::from_row_major(n, m, c_data);

    let s0 = r1cs_to_ccs(a, b, c);

    s0.ensure_identity_first()
        .expect("ensure_identity_first should succeed")
}

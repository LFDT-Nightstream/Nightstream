#![allow(non_snake_case)]

use std::time::Instant;

use bellpepper::gadgets::boolean::{AllocatedBit, Boolean};
use bellpepper_core::{Circuit, Comparable, ConstraintSystem, Index, LinearCombination, SynthesisError};
use ff::PrimeField;
use neo_ajtai::{set_global_pp, setup as ajtai_setup, AjtaiSModule};
use neo_ccs::{r1cs_to_ccs, CcsStructure, Mat};
use neo_fold::{
    pi_ccs::FoldingMode,
    session::{FoldingSession, NeoStep, StepArtifacts, StepSpec},
};
use neo_math::{D, F};
use neo_params::NeoParams;
use p3_field::PrimeCharacteristicRing;
use rand_chacha::rand_core::SeedableRng;
use sha2::{Digest, Sha256};

extern crate ff;

/// Pad witness to match CCS dimensions (adds slack variables if n > m_original)
fn pad_witness_to_m(mut z: Vec<F>, m_target: usize) -> Vec<F> {
    // Pad with zeros to reach m_target
    z.resize(m_target, F::ZERO);
    z
}

fn setup_ajtai_for_dims(m: usize) {
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(42);
    let pp = ajtai_setup(&mut rng, D, 4, m).expect("Ajtai setup should succeed");
    let _ = set_global_pp(pp);
}

fn fp_to_u64(x: &FpGoldilocks) -> u64 {
    let bytes = x.to_repr();
    u64::from_le_bytes(bytes.0[0..8].try_into().expect("repr is at least 8 bytes"))
}

#[derive(Clone)]
struct NoInputs;

struct StepCircuit {
    steps: Vec<Vec<F>>,
    step_spec: StepSpec,
    step_ccs: CcsStructure<F>,
}

impl NeoStep for StepCircuit {
    type ExternalInputs = NoInputs;

    fn state_len(&self) -> usize {
        0
    }

    fn step_spec(&self) -> StepSpec {
        self.step_spec.clone()
    }

    fn synthesize_step(
        &mut self,
        step_idx: usize,
        _y_prev: &[F],
        _inputs: &Self::ExternalInputs,
    ) -> StepArtifacts {
        let z = self.steps[step_idx].clone();
        let z_padded = pad_witness_to_m(z, self.step_ccs.m);
        StepArtifacts {
            ccs: self.step_ccs.clone(),
            witness: z_padded,
            public_app_inputs: vec![],
            spec: self.step_spec.clone(),
        }
    }
}

#[test]
fn test_sha256_circuit_is_satisfied() {
    use bellpepper_core::test_cs::TestConstraintSystem;

    let mut cs = TestConstraintSystem::<FpGoldilocks>::new();
    let preimage = b"abc".to_vec();
    let circuit = Sha256Circuit {
        preimage: preimage.clone(),
    };
    circuit
        .synthesize(&mut cs)
        .expect("Circuit synthesis should succeed");
    assert!(cs.is_satisfied());

    // Verify that the packed public inputs match the SHA256 digest of the preimage.
    let digest = Sha256::digest(&preimage);
    let digest_bits = bellpepper::gadgets::multipack::bytes_to_bits(digest.as_ref());
    let expected_inputs =
        bellpepper::gadgets::multipack::compute_multipacking::<FpGoldilocks>(&digest_bits);
    assert!(cs.verify(&expected_inputs));
}

#[test]
#[ignore = "Very memory-heavy: builds dense n×n matrices for SHA256 R1CS; run manually."]
fn test_sha256_preimage_64_bytes() {
    test_sha256_preimage_len_bytes(64);
}

#[test]
#[ignore = "Very memory-heavy: builds dense n×n matrices for SHA256 R1CS; run manually."]
fn test_sha256_preimage_128_bytes() {
    test_sha256_preimage_len_bytes(128);
}

#[test]
#[ignore = "Very memory-heavy: builds dense n×n matrices for SHA256 R1CS; run manually."]
fn test_sha256_preimage_256_bytes() {
    test_sha256_preimage_len_bytes(256);
}

#[test]
#[ignore = "Very memory-heavy: builds dense n×n matrices for SHA256 R1CS; run manually."]
fn test_sha256_preimage_512_bytes() {
    test_sha256_preimage_len_bytes(512);
}

#[test]
#[ignore = "Very memory-heavy: builds dense n×n matrices for SHA256 R1CS; run manually."]
fn test_sha256_preimage_4kB() {
    test_sha256_preimage_len_bytes(1 << 12);
}

#[test]
#[ignore = "Very memory-heavy: builds dense n×n matrices for SHA256 R1CS; run manually."]
fn test_sha256_preimage_8kB() {
    test_sha256_preimage_len_bytes(1 << 13);
}

fn test_sha256_preimage_len_bytes(preimage_len_bytes: usize) {
    let preimage = vec![0u8; preimage_len_bytes];
    let digest = Sha256::digest(&preimage);
    let digest_bits = bellpepper::gadgets::multipack::bytes_to_bits(digest.as_ref());
    let expected_inputs_fp =
        bellpepper::gadgets::multipack::compute_multipacking::<FpGoldilocks>(&digest_bits);
    let expected_inputs: Vec<F> = expected_inputs_fp
        .iter()
        .map(|x| F::from_u64(fp_to_u64(x)))
        .collect();

    let (step_ccs, witness) = bellpepper_sha256_circuit(preimage_len_bytes);

    let mut params = NeoParams::goldilocks_auto_r1cs_ccs(step_ccs.n)
        .expect("goldilocks_auto_r1cs_ccs should find valid params");

    params.b = 3;

    setup_ajtai_for_dims(step_ccs.m);
    let l = AjtaiSModule::from_global_for_dims(D, step_ccs.m).expect("AjtaiSModule init");

    let m_in = 1 + expected_inputs.len();

    let step_spec = StepSpec {
        y_len: 0,
        const1_index: 0,
        y_step_indices: vec![],
        app_input_indices: Some((1..m_in).collect()),
        m_in,
    };

    let mut circuit = StepCircuit {
        steps: vec![witness; 1],
        step_spec: step_spec.clone(),
        step_ccs: step_ccs.clone(),
    };

    let mut session = FoldingSession::new(FoldingMode::Optimized, params, l.clone());

    for _ in 0..circuit.steps.len() {
        let start = Instant::now();

        session
            .add_step(&mut circuit, &NoInputs)
            .expect("add_step should succeed with optimized");

        println!("Add step duration: {:?}", start.elapsed());
    }

    // The step's public inputs include the packed SHA256 digest, so we can validate it here.
    {
        let mcss_public = session.mcss_public();
        assert_eq!(mcss_public.len(), 1);
        assert_eq!(mcss_public[0].x[0], F::ONE);
        assert_eq!(mcss_public[0].x[1..], expected_inputs);
    }

    let start = Instant::now();
    let run = session
        .fold_and_prove(&step_ccs)
        .expect("fold_and_prove should produce a FoldRun");
    let finalize_duration = start.elapsed();

    println!("Proof generation time (finalize): {:?}", finalize_duration);

    assert_eq!(run.steps.len(), 1, "should have correct number of steps");

    let mcss_public = session.mcss_public();
    let ok = session
        .verify(&step_ccs, &mcss_public, &run)
        .expect("verify should run");
    assert!(ok, "optimized verification should pass");
}

#[derive(PrimeField)]
#[PrimeFieldModulus = "18446744069414584321"]
#[PrimeFieldGenerator = "7"]
#[PrimeFieldReprEndianness = "little"]
struct FpGoldilocks([u64; 2]);

struct Sha256Circuit {
    #[allow(unused)]
    preimage: Vec<u8>,
}

impl Circuit<FpGoldilocks> for Sha256Circuit {
    fn synthesize<CS: ConstraintSystem<FpGoldilocks>>(
        self,
        cs: &mut CS,
    ) -> Result<(), SynthesisError> {
        // SHA256 expects a big-endian bit order within each byte.
        let bit_values: Vec<_> = bellpepper::gadgets::multipack::bytes_to_bits(&self.preimage)
            .into_iter()
            .map(Some)
            .collect();
        assert_eq!(bit_values.len(), self.preimage.len() * 8);

        let preimage_bits = bit_values
            .into_iter()
            .enumerate()
            .map(|(i, b)| AllocatedBit::alloc(cs.namespace(|| format!("preimage bit {i}")), b))
            .map(|b| b.map(Boolean::from))
            .collect::<Result<Vec<_>, _>>()?;

        // TODO: these would have to be added as outputs
        // it doesn't matter right now though
        let hash_bits =
            bellpepper::gadgets::sha256::sha256(cs.namespace(|| "sha256"), &preimage_bits)?;

        // Bind the SHA256 digest as compact public inputs.
        bellpepper::gadgets::multipack::pack_into_inputs(cs.namespace(|| "hash_out"), &hash_bits)?;

        Ok(())
    }
}

fn bellpepper_sha256_circuit(preimage_len_bytes: usize) -> (CcsStructure<F>, Vec<F>) {
    use bellpepper_core::test_cs::TestConstraintSystem;

    let mut cs = TestConstraintSystem::<FpGoldilocks>::new();

    let circuit = Sha256Circuit {
        preimage: vec![0u8; preimage_len_bytes],
    };
    circuit
        .synthesize(&mut cs)
        .expect("Circuit synthesis should succeed");

    assert!(cs.is_satisfied());

    let num_constraints = cs.num_constraints();
    let num_inputs = cs.num_inputs();
    let num_aux = cs.scalar_aux().len();
    let num_variables = num_inputs + num_aux;

    println!(
        "SHA256 CCS: n={}, m={}",
        num_constraints,
        num_inputs + num_aux
    );

    let n = num_constraints.max(num_variables);

    let f = |row: usize,
             mat: &mut Mat<F>,
             lc: &LinearCombination<FpGoldilocks>| {
        for (var, coeff) in lc.iter() {
            let col = match var.0 {
                Index::Input(i) => i,
                Index::Aux(i) => num_inputs + i,
            };

            let value = fp_to_u64(coeff);
            if value == 0 {
                continue;
            }
            mat[(row, col)] = mat[(row, col)] + F::from_u64(value);
        }
    };

    let mut A = Mat::zero(n, n, F::ZERO);
    let mut B = Mat::zero(n, n, F::ZERO);
    let mut C = Mat::zero(n, n, F::ZERO);

    // Extract constraints and convert to matrix form
    for (row, constraint) in cs.constraints().iter().enumerate() {
        f(row, &mut A, &constraint.0);
        f(row, &mut B, &constraint.1);
        f(row, &mut C, &constraint.2);
    }

    let mut witness: Vec<F> = Vec::with_capacity(num_variables);

    for val in cs.scalar_inputs() {
        witness.push(F::from_u64(fp_to_u64(&val)));
    }

    for val in cs.scalar_aux() {
        witness.push(F::from_u64(fp_to_u64(&val)));
    }

    debug_assert_eq!(witness.len(), num_variables);

    let ccs = r1cs_to_ccs(A, B, C);
    (ccs, witness)
}

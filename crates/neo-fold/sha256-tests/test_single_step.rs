#![allow(non_snake_case)]

use std::time::Instant;

use bellpepper::gadgets::boolean::{AllocatedBit, Boolean};
use bellpepper_core::{
    Circuit, Comparable, ConstraintSystem, Index, LinearCombination, SynthesisError,
};
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
use serde::{Deserialize, Serialize};

extern crate ff;

#[derive(Serialize, Deserialize, Clone)]
struct SparseMatrix {
    rows: usize,
    cols: usize,
    entries: Vec<(usize, usize, u64)>,
}

fn sparse_to_dense_mat(sparse: SparseMatrix) -> Mat<F> {
    let mut data = vec![F::ZERO; sparse.rows * sparse.cols];
    for &(row, col, val) in &sparse.entries {
        data[row * sparse.cols + col] = F::from_u64(val);
    }
    Mat::from_row_major(sparse.rows, sparse.cols, data)
}

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
fn test_sha256_preimage_64_bits() {
    test_sha256_batch_size(1 << 6);
}

#[test]
fn test_sha256_preimage_128_bits() {
    test_sha256_batch_size(1 << 7);
}

#[test]
fn test_sha256_preimage_256_bits() {
    test_sha256_batch_size(1 << 8);
}

#[test]
fn test_sha256_preimage_512_bits() {
    test_sha256_batch_size(1 << 9);
}

#[test]
#[ignore]
fn test_sha256_preimage_4kB() {
    test_sha256_batch_size(1 << 15);
}

#[test]
#[ignore]
fn test_sha256_preimage_8kB() {
    test_sha256_batch_size(1 << 16);
}

/// The size is in bytes
/// This generates a hash of [0; 1 << size]
fn test_sha256_batch_size(size: usize) {
    let (step_ccs, witness) = bellpepper_sha256_circuit(size);

    let mut params = NeoParams::goldilocks_auto_r1cs_ccs(step_ccs.n)
        .expect("goldilocks_auto_r1cs_ccs should find valid params");

    params.b = 3;

    setup_ajtai_for_dims(step_ccs.n);
    let l = AjtaiSModule::from_global_for_dims(D, step_ccs.n).expect("AjtaiSModule init");

    let step_spec = StepSpec {
        y_len: 0,
        const1_index: 1,
        y_step_indices: vec![],
        app_input_indices: Some(vec![]),
        m_in: 1,
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
        let bit_values: Vec<_> = self
            .preimage
            .clone()
            .into_iter()
            .flat_map(|byte| (0..8).map(move |i| (byte >> i) & 1u8 == 1u8))
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
        let _hash_bits =
            bellpepper::gadgets::sha256::sha256(cs.namespace(|| "sha256"), &preimage_bits)?;

        Ok(())
    }
}

fn bellpepper_sha256_circuit(size: usize) -> (CcsStructure<F>, Vec<F>) {
    use bellpepper_core::test_cs::TestConstraintSystem;

    let mut cs = TestConstraintSystem::<FpGoldilocks>::new();

    let circuit = Sha256Circuit {
        preimage: vec![0u8; size],
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

    let mut entries_a: Vec<(usize, usize, u64)> = vec![];
    let mut entries_b: Vec<(usize, usize, u64)> = vec![];
    let mut entries_c: Vec<(usize, usize, u64)> = vec![];

    let f = |row: usize,
             entries: &mut Vec<(usize, usize, u64)>,
             lc: &LinearCombination<FpGoldilocks>| {
        for (var, coeff) in lc.iter() {
            let col = match var.0 {
                Index::Input(i) => i,
                Index::Aux(i) => num_inputs + i,
            };

            let bytes = coeff.to_repr();
            let value = u64::from_le_bytes(bytes.0[0..8].try_into().unwrap());

            if value != 0 {
                entries.push((row, col, value));
            }
        }
    };

    // Extract constraints and convert to matrix form
    for (row, constraint) in cs.constraints().iter().enumerate() {
        f(row, &mut entries_a, &constraint.0);
        f(row, &mut entries_b, &constraint.0);
        f(row, &mut entries_c, &constraint.0);
    }

    let a = SparseMatrix {
        rows: n,
        cols: n,
        entries: entries_a,
    };
    let b = SparseMatrix {
        rows: n,
        cols: n,
        entries: entries_b,
    };
    let c = SparseMatrix {
        rows: n,
        cols: n,
        entries: entries_c,
    };

    let mut witness = vec![F::ONE];

    for val in cs.scalar_inputs() {
        let bytes = val.to_repr();
        let value = u64::from_le_bytes(bytes.0[0..8].try_into().unwrap());
        witness.push(F::from_u64(value));
    }

    for val in cs.scalar_aux() {
        let bytes = val.to_repr();
        let value = u64::from_le_bytes(bytes.0[0..8].try_into().unwrap());
        witness.push(F::from_u64(value));
    }

    let s0 = r1cs_to_ccs(
        sparse_to_dense_mat(a),
        sparse_to_dense_mat(b),
        sparse_to_dense_mat(c),
    );
    let ccs = s0
        .ensure_identity_first()
        .expect("ensure_identity_first should succeed");

    (ccs, witness)
}

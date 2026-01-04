#![allow(non_snake_case)]

use std::time::Instant;

use bellpepper::gadgets::boolean::{AllocatedBit, Boolean};
use bellpepper_core::{
    Circuit, ConstraintSystem, Index, LinearCombination, SynthesisError, Variable,
};
use ff::PrimeField;
use neo_ajtai::{set_global_pp_seeded, AjtaiSModule};
use neo_ccs::{CcsMatrix, CcsStructure, CscMat, SparsePoly, Term};
use neo_fold::{pi_ccs::FoldingMode, session::FoldingSession};
use neo_math::{D, F};
use neo_params::NeoParams;
use neo_reductions::engines::optimized_engine::oracle::SparseCache;
use p3_field::PrimeCharacteristicRing;
use sha2::{Digest, Sha256};

extern crate ff;

/// Pad witness to match CCS dimensions (adds slack variables if n > m_original)
fn pad_witness_to_m(mut z: Vec<F>, m_target: usize) -> Vec<F> {
    // Pad with zeros to reach m_target
    z.resize(m_target, F::ZERO);
    z
}

fn setup_ajtai_for_dims(m: usize) {
    // Deterministic, reloadable PP: allows unloading the multi-GB Ajtai matrix between phases.
    let mut seed = [0u8; 32];
    seed[..8].copy_from_slice(&42u64.to_le_bytes());
    set_global_pp_seeded(D, 4, m, seed).expect("set_global_pp_seeded");
}

fn fp_to_u64(x: &FpGoldilocks) -> u64 {
    let bytes = x.to_repr();
    u64::from_le_bytes(bytes.0[0..8].try_into().expect("repr is at least 8 bytes"))
}

// Bellpepper's TestConstraintSystem is optimized for debuggability (names, hashing, storage),
// which is extremely expensive for million-constraint circuits. For performance tests we only need:
// - variable assignments (inputs + aux) to build the witness, and
// - sparse A/B/C matrices (triplets) for the R1CS -> CCS embedding.
//
// This constraint system skips all namespace/annotation closures and streams constraints directly
// into triplet lists without ever storing full constraints.
const AUX_FLAG: u32 = 1 << 31;

struct TripletConstraintSystem {
    inputs: Vec<F>, // includes input[0] = ONE
    aux: Vec<F>,
    num_constraints: u32,
    a_trips: Vec<(u32, u32, F)>,
    b_trips: Vec<(u32, u32, F)>,
    c_trips: Vec<(u32, u32, F)>,
}

impl TripletConstraintSystem {
    fn new() -> Self {
        Self {
            inputs: vec![F::ONE],
            aux: Vec::new(),
            num_constraints: 0,
            a_trips: Vec::new(),
            b_trips: Vec::new(),
            c_trips: Vec::new(),
        }
    }

    fn push_lc_trips(
        row: u32,
        lc: &LinearCombination<FpGoldilocks>,
        trips: &mut Vec<(u32, u32, F)>,
    ) {
        for (var, coeff) in lc.iter() {
            let value = fp_to_u64(coeff);
            if value == 0 {
                continue;
            }
            let col = match var.0 {
                Index::Input(i) => u32::try_from(i).expect("input index should fit in u32"),
                Index::Aux(i) => AUX_FLAG | u32::try_from(i).expect("aux index should fit in u32"),
            };
            trips.push((row, col, F::from_u64(value)));
        }
    }

    fn resolve_triplets(trips: Vec<(u32, u32, F)>, num_inputs: usize) -> Vec<(usize, usize, F)> {
        trips
            .into_iter()
            .map(|(row, col, value)| {
                let row = row as usize;
                if (col & AUX_FLAG) == 0 {
                    (row, col as usize, value)
                } else {
                    let aux_idx = (col & !AUX_FLAG) as usize;
                    (row, num_inputs + aux_idx, value)
                }
            })
            .collect()
    }
}

impl ConstraintSystem<FpGoldilocks> for TripletConstraintSystem {
    type Root = Self;

    fn new() -> Self {
        Self::new()
    }

    fn alloc<FN, A, AR>(&mut self, _annotation: A, f: FN) -> Result<Variable, SynthesisError>
    where
        FN: FnOnce() -> Result<FpGoldilocks, SynthesisError>,
        A: FnOnce() -> AR,
        AR: Into<String>,
    {
        let idx = self.aux.len();
        let value = f()?;
        self.aux.push(F::from_u64(fp_to_u64(&value)));
        Ok(Variable::new_unchecked(Index::Aux(idx)))
    }

    fn alloc_input<FN, A, AR>(&mut self, _annotation: A, f: FN) -> Result<Variable, SynthesisError>
    where
        FN: FnOnce() -> Result<FpGoldilocks, SynthesisError>,
        A: FnOnce() -> AR,
        AR: Into<String>,
    {
        let idx = self.inputs.len();
        let value = f()?;
        self.inputs.push(F::from_u64(fp_to_u64(&value)));
        Ok(Variable::new_unchecked(Index::Input(idx)))
    }

    fn enforce<A, AR, LA, LB, LC>(&mut self, _annotation: A, a: LA, b: LB, c: LC)
    where
        A: FnOnce() -> AR,
        AR: Into<String>,
        LA: FnOnce(LinearCombination<FpGoldilocks>) -> LinearCombination<FpGoldilocks>,
        LB: FnOnce(LinearCombination<FpGoldilocks>) -> LinearCombination<FpGoldilocks>,
        LC: FnOnce(LinearCombination<FpGoldilocks>) -> LinearCombination<FpGoldilocks>,
    {
        let row = self.num_constraints;
        self.num_constraints += 1;

        let a_lc = a(LinearCombination::zero());
        let b_lc = b(LinearCombination::zero());
        let c_lc = c(LinearCombination::zero());

        Self::push_lc_trips(row, &a_lc, &mut self.a_trips);
        Self::push_lc_trips(row, &b_lc, &mut self.b_trips);
        Self::push_lc_trips(row, &c_lc, &mut self.c_trips);
    }

    fn push_namespace<NR, N>(&mut self, _name_fn: N)
    where
        NR: Into<String>,
        N: FnOnce() -> NR,
    {
    }

    fn pop_namespace(&mut self) {}

    fn get_root(&mut self) -> &mut Self::Root {
        self
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
fn test_sha256_preimage_64_bytes() {
    test_sha256_preimage_len_bytes(64);
}

#[test]
#[ignore = "Slow: large SHA256 circuit; run manually."]
fn test_sha256_preimage_128_bytes() {
    test_sha256_preimage_len_bytes(128);
}

#[test]
#[ignore = "Slow: large SHA256 circuit; run manually."]
fn test_sha256_preimage_256_bytes() {
    test_sha256_preimage_len_bytes(256);
}

#[test]
#[ignore = "Slow: large SHA256 circuit; run manually."]
fn test_sha256_preimage_512_bytes() {
    test_sha256_preimage_len_bytes(512);
}

#[test]
#[ignore = "Slow: large SHA256 circuit; run manually."]
fn test_sha256_preimage_4kB() {
    test_sha256_preimage_len_bytes(1 << 12);
}

#[test]
#[ignore = "Slow: large SHA256 circuit; run manually."]
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

    let mut session = FoldingSession::new(FoldingMode::Optimized, params, l.clone());

    // Preload prover/verifier CCS cache once (shared) to avoid rebuilding for both sides.
    let sparse_cache = std::sync::Arc::new(SparseCache::build(&step_ccs));
    session
        .preload_ccs_sparse_cache(&step_ccs, sparse_cache.clone())
        .expect("preload_ccs_sparse_cache should succeed");
    let start = Instant::now();

    let z = pad_witness_to_m(witness, step_ccs.m);
    let public_input = &z[..m_in];
    let witness = &z[m_in..];

    let step_start = Instant::now();
    session
        .add_step_io(&step_ccs, public_input, witness)
        .expect("add_step should succeed with optimized");
    println!("Add step duration: {:?}", step_start.elapsed());

    // The step's public inputs include the packed SHA256 digest, so we can validate it here.
    {
        let mcss_public = session.mcss_public();
        assert_eq!(mcss_public.len(), 1);
        assert_eq!(mcss_public[0].x[0], F::ONE);
        assert_eq!(mcss_public[0].x[1..], expected_inputs);
    }

    let run = session
        .fold_and_prove(&step_ccs)
        .expect("fold_and_prove should produce a FoldRun");
    println!("Proof generation time (finalize): {:?}", start.elapsed());

    assert_eq!(run.steps.len(), 1, "should have correct number of steps");

    let mcss_public = session.mcss_public();
    let verifier_pre = Instant::now();
    session
        .preload_verifier_ccs_sparse_cache(&step_ccs, sparse_cache.clone())
        .expect("preload_verifier_ccs_sparse_cache should succeed");
    println!("Verifier preprocessing time: {:?}", verifier_pre.elapsed());
    let verify_start = Instant::now();
    let ok = session
        .verify(&step_ccs, &mcss_public, &run)
        .expect("verify should run");
    println!("Verification time: {:?}", verify_start.elapsed());
    assert!(ok, "optimized verification should pass");

    // Sanity check: verification must reject a tampered proof (fast verification should still be sound).
    let mut tampered = run.clone();
    tampered.steps[0].fold.ccs_proof.sumcheck_rounds[0][0] += neo_math::K::ONE;
    match session.verify(&step_ccs, &mcss_public, &tampered) {
        Ok(true) => panic!("tampered proof unexpectedly verified"),
        Ok(false) | Err(_) => {}
    }
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

fn bellpepper_sha256_circuit(
    preimage_len_bytes: usize,
) -> (CcsStructure<F>, Vec<F>) {
    let mut cs = TripletConstraintSystem::new();

    let circuit = Sha256Circuit {
        preimage: vec![0u8; preimage_len_bytes],
    };
    circuit
        .synthesize(&mut cs)
        .expect("Circuit synthesis should succeed");

    let TripletConstraintSystem {
        inputs,
        aux,
        num_constraints,
        a_trips,
        b_trips,
        c_trips,
    } = cs;

    let num_constraints = num_constraints as usize;
    let num_inputs = inputs.len();
    let num_aux = aux.len();
    let num_variables = num_inputs + num_aux;

    println!(
        "SHA256 CCS: n={}, m={}",
        num_constraints,
        num_inputs + num_aux
    );

    let n = num_constraints.max(num_variables);

    let a_trips = TripletConstraintSystem::resolve_triplets(a_trips, num_inputs);
    let b_trips = TripletConstraintSystem::resolve_triplets(b_trips, num_inputs);
    let c_trips = TripletConstraintSystem::resolve_triplets(c_trips, num_inputs);

    let mut witness = inputs;
    witness.extend(aux);
    debug_assert_eq!(witness.len(), num_variables);

    // R1CS → CCS embedding with identity-first form: M_0 = I_n, M_1=A, M_2=B, M_3=C.
    let f_base = SparsePoly::new(
        3,
        vec![
            Term {
                coeff: F::ONE,
                exps: vec![1, 1, 0],
            }, // X1 * X2
            Term {
                coeff: -F::ONE,
                exps: vec![0, 0, 1],
            }, // -X3
        ],
    );

    let matrices = vec![
        CcsMatrix::Identity { n },
        CcsMatrix::Csc(CscMat::from_triplets(a_trips, n, n)),
        CcsMatrix::Csc(CscMat::from_triplets(b_trips, n, n)),
        CcsMatrix::Csc(CscMat::from_triplets(c_trips, n, n)),
    ];
    let f = f_base.insert_var_at_front();

    let ccs = CcsStructure::new_sparse(matrices, f).expect("valid R1CS→CCS structure");
    (ccs, witness)
}

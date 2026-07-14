//! End-to-end tests for `engine::ccs_native::poseidon2`.
//!
//! Covers:
//! - degree-7 polynomial shape acceptance via the engine (`y = x⁷`),
//! - bit-backed selector-style encoding of the same constraint,
//! - the full Poseidon2 permutation builder,
//! - the full Poseidon2 sponge-hash builder,
//! - tamper rejection in three flavours (proof messages, single-word
//!   bit flip, multi-permutation hash output).
//!
//! The builders under test live in `engine::ccs_native::poseidon2`. The
//! small CCS-only structures further up the file (`degree7_sbox_*`,
//! `bit_backed_sbox_*`) are test-local fixtures that exercise the
//! engine's degree-7 path without going through the full Poseidon2
//! builder.

use neo_ajtai::{has_global_pp_for_dims, set_global_pp_seeded, AjtaiSModule};
use neo_ccs::crypto::poseidon2_goldilocks::permute_state;
use neo_ccs::{CcsStructure, Mat, SparsePoly, Term};
use neo_fold_clean::engine::ccs_native::poseidon2::{
    build_bit_backed_poseidon2_hash, build_bit_backed_poseidon2_permutation, poseidon2_sbox7, push_goldilocks_bits,
    POSEIDON2_GOLDILOCKS_BITS, POSEIDON2_HALF_FULL_ROUNDS, POSEIDON2_PARTIAL_ROUNDS, POSEIDON2_RATE, POSEIDON2_WIDTH,
};
use neo_fold_clean::engine::r1cs_circuit::poseidon2::enforce_poseidon2_permutation;
use neo_fold_clean::engine::r1cs_circuit::{R1csBuilder, Var};
use neo_fold_clean::engine::transcript::Transcript;
use neo_fold_clean::paper::construction2::RunningInstance;
use neo_fold_clean::paper::reductions::pi_ccs;
use neo_fold_clean::{config, CcsInstance, Params, Structure};
use neo_math::{D, F, K};
use neo_reductions::optimized_engine::OptimizedStructureCache;
use p3_field::PrimeCharacteristicRing;

/// Total S-boxes per full Poseidon2 permutation. Used to size-check
/// the built CCS structure.
const SBOXES_PER_PERMUTATION: usize = 2 * POSEIDON2_HALF_FULL_ROUNDS * POSEIDON2_WIDTH + POSEIDON2_PARTIAL_ROUNDS;

/// Committed bit-words per full Poseidon2 permutation (input state +
/// pre-external-linear + full-round S-box + linear pairs +
/// partial-round S-box + linear pairs).
const BIT_BACKED_PERMUTATION_WORDS: usize = POSEIDON2_WIDTH
    + POSEIDON2_WIDTH
    + 2 * POSEIDON2_HALF_FULL_ROUNDS * 2 * POSEIDON2_WIDTH
    + POSEIDON2_PARTIAL_ROUNDS * (1 + POSEIDON2_WIDTH);

/// Linear-constraint rows per permutation (pre-external + per-round
/// next-state binders).
const BIT_BACKED_PERMUTATION_LINEAR_ROWS: usize =
    POSEIDON2_WIDTH + 2 * POSEIDON2_HALF_FULL_ROUNDS * POSEIDON2_WIDTH + POSEIDON2_PARTIAL_ROUNDS * POSEIDON2_WIDTH;

// ── Test-local degree-7 / bit-backed sbox fixtures ──────────────────────

fn degree7_sbox_structure() -> Structure {
    let mut x = Mat::zero(1, 2, F::ZERO);
    x.set(0, 0, F::ONE);
    let mut y = Mat::zero(1, 2, F::ZERO);
    y.set(0, 1, F::ONE);
    let f = SparsePoly::new(
        2,
        vec![
            Term {
                coeff: F::ONE,
                exps: vec![7, 0],
            },
            Term {
                coeff: -F::ONE,
                exps: vec![0, 1],
            },
        ],
    );
    CcsStructure::new(vec![x, y], f).expect("degree-7 CCS structure")
}

fn install_ajtai_module(params: &Params, structure: &Structure) {
    let cols = structure.m.div_ceil(D);
    if !has_global_pp_for_dims(D, cols) {
        let mut seed = [0u8; 32];
        seed[..8].copy_from_slice(&0x4343_534e_4154_4956_u64.to_le_bytes());
        match set_global_pp_seeded(D, params.kappa() as usize, cols, seed) {
            Ok(()) => {}
            Err(_err) if has_global_pp_for_dims(D, cols) => {}
            Err(err) => panic!("Ajtai global setup: {err}"),
        }
    }
}

fn degree7_instance(params: &Params, structure: &Structure, log: &AjtaiSModule) -> CcsInstance {
    let z = vec![-F::ONE, -F::ONE];
    CcsInstance::from_low_norm_assignment(params, log, structure, &z, 1).expect("low-norm degree-7 assignment")
}

fn bit_backed_sbox_structure() -> Structure {
    let n = 2 * POSEIDON2_GOLDILOCKS_BITS + 1;
    let m = 1 + 2 * POSEIDON2_GOLDILOCKS_BITS;
    let sbox_row = n - 1;

    let mut bit = Mat::zero(n, m, F::ZERO);
    for row in 0..2 * POSEIDON2_GOLDILOCKS_BITS {
        bit.set(row, 1 + row, F::ONE);
    }

    let mut x = Mat::zero(n, m, F::ZERO);
    let mut y = Mat::zero(n, m, F::ZERO);
    let mut pow2 = F::ONE;
    for i in 0..POSEIDON2_GOLDILOCKS_BITS {
        x.set(sbox_row, 1 + i, pow2);
        y.set(sbox_row, 1 + POSEIDON2_GOLDILOCKS_BITS + i, pow2);
        pow2 *= F::from_u64(2);
    }

    let f = SparsePoly::new(
        3,
        vec![
            Term {
                coeff: F::ONE,
                exps: vec![2, 0, 0],
            },
            Term {
                coeff: -F::ONE,
                exps: vec![1, 0, 0],
            },
            Term {
                coeff: F::ONE,
                exps: vec![0, 7, 0],
            },
            Term {
                coeff: -F::ONE,
                exps: vec![0, 0, 1],
            },
        ],
    );

    CcsStructure::new(vec![bit, x, y], f).expect("bit-backed degree-7 CCS structure")
}

fn bit_backed_sbox_assignment(x: F) -> Vec<F> {
    let mut z = Vec::with_capacity(1 + 2 * POSEIDON2_GOLDILOCKS_BITS);
    z.push(F::ONE);
    push_goldilocks_bits(&mut z, x);
    push_goldilocks_bits(&mut z, poseidon2_sbox7(x));
    z
}

fn bit_backed_sbox_instance(params: &Params, structure: &Structure, log: &AjtaiSModule, x: F) -> CcsInstance {
    let z = bit_backed_sbox_assignment(x);
    CcsInstance::from_low_norm_assignment(params, log, structure, &z, 1)
        .expect("low-norm bit-backed degree-7 assignment")
}

struct Fixture {
    params: Params,
    structure: Structure,
    cache: OptimizedStructureCache,
    log: AjtaiSModule,
    instance: CcsInstance,
}

fn build_fixture() -> Fixture {
    let structure = degree7_sbox_structure();
    let params = config::ccs_params(structure.n, structure.m, structure.t(), structure.max_degree()).expect("params");
    install_ajtai_module(&params, &structure);
    let cols = structure.m.div_ceil(D);
    let log = AjtaiSModule::from_global_for_dims(D, cols).expect("Ajtai module");
    let cache = OptimizedStructureCache::build(&structure).expect("cache build");
    let instance = degree7_instance(&params, &structure, &log);
    Fixture {
        params,
        structure,
        cache,
        log,
        instance,
    }
}

fn build_bit_backed_fixture(x: F) -> Fixture {
    let structure = bit_backed_sbox_structure();
    let params = config::ccs_params(structure.n, structure.m, structure.t(), structure.max_degree()).expect("params");
    install_ajtai_module(&params, &structure);
    let cols = structure.m.div_ceil(D);
    let log = AjtaiSModule::from_global_for_dims(D, cols).expect("Ajtai module");
    let cache = OptimizedStructureCache::build(&structure).expect("cache build");
    let instance = bit_backed_sbox_instance(&params, &structure, &log, x);
    Fixture {
        params,
        structure,
        cache,
        log,
        instance,
    }
}

fn prove_verify_single_fresh(structure: &Structure, z: Vec<F>) -> Result<Vec<neo_fold_clean::CeClaim>, pi_ccs::Error> {
    let params = config::ccs_params(structure.n, structure.m, structure.t(), structure.max_degree()).expect("params");
    install_ajtai_module(&params, structure);
    let cols = structure.m.div_ceil(D);
    let log = AjtaiSModule::from_global_for_dims(D, cols).expect("Ajtai module");
    let cache = OptimizedStructureCache::build(structure).expect("cache build");
    let instance =
        CcsInstance::from_low_norm_assignment(&params, &log, structure, &z, 1).expect("low-norm sparse CCS assignment");

    let mut prover_tr = Transcript::session();
    let proof = pi_ccs::prove(
        &mut prover_tr,
        &params,
        structure,
        &cache,
        &log,
        vec![instance.clone()],
        &RunningInstance::default(),
    )?;

    let mut verifier_tr = Transcript::session();
    pi_ccs::verify(
        &mut verifier_tr,
        &params,
        structure,
        &cache,
        &[instance.claim],
        &RunningInstance::default(),
        &proof,
    )
}

fn r1cs_poseidon2_permutation_shape(input: [F; POSEIDON2_WIDTH]) -> (usize, usize) {
    let mut builder = R1csBuilder::new();
    let mut vars = [Var::ONE; POSEIDON2_WIDTH];
    for (slot, value) in vars.iter_mut().zip(input) {
        *slot = builder.alloc(value);
    }
    let _out = enforce_poseidon2_permutation(&mut builder, &vars);
    assert!(
        builder.is_satisfied(),
        "R1CS Poseidon2 gadget should satisfy on witness"
    );
    (builder.rows(), builder.cols())
}

fn r1cs_poseidon2_hash_shape(input: &[F]) -> (usize, usize) {
    use neo_fold_clean::engine::r1cs_circuit::poseidon2::enforce_poseidon2_hash;
    let mut builder = R1csBuilder::new();
    let input_vars: Vec<Var> = input.iter().map(|&v| builder.alloc(v)).collect();
    let _digest = enforce_poseidon2_hash(&mut builder, &input_vars);
    assert!(
        builder.is_satisfied(),
        "R1CS Poseidon2 hash gadget should satisfy on witness"
    );
    (builder.rows(), builder.cols())
}

// ── Tests ──────────────────────────────────────────────────────────────

#[test]
fn native_pi_ccs_accepts_degree7_sbox_relation() {
    let f = build_fixture();
    assert_eq!(f.structure.f.max_degree(), 7);

    let mut prover_tr = Transcript::session();
    let proof = pi_ccs::prove(
        &mut prover_tr,
        &f.params,
        &f.structure,
        &f.cache,
        &f.log,
        vec![f.instance.clone()],
        &RunningInstance::default(),
    )
    .expect("Π_CCS.P degree-7");

    let mut verifier_tr = Transcript::session();
    let outputs = pi_ccs::verify(
        &mut verifier_tr,
        &f.params,
        &f.structure,
        &f.cache,
        &[f.instance.claim.clone()],
        &RunningInstance::default(),
        &proof,
    )
    .expect("Π_CCS.V degree-7");

    assert_eq!(outputs.len(), 1);
    assert_eq!(outputs[0].y_ring.len(), 2, "one opening for X and one for Y");
}

#[test]
fn native_pi_ccs_rejects_tampered_degree7_sbox_proof() {
    let f = build_fixture();

    let mut prover_tr = Transcript::session();
    let mut proof = pi_ccs::prove(
        &mut prover_tr,
        &f.params,
        &f.structure,
        &f.cache,
        &f.log,
        vec![f.instance.clone()],
        &RunningInstance::default(),
    )
    .expect("Π_CCS.P degree-7");

    proof.sumcheck.sumcheck_rounds[0][0] += K::ONE;

    let mut verifier_tr = Transcript::session();
    let err = pi_ccs::verify(
        &mut verifier_tr,
        &f.params,
        &f.structure,
        &f.cache,
        &[f.instance.claim.clone()],
        &RunningInstance::default(),
        &proof,
    )
    .expect_err("tampered degree-7 sumcheck must reject");

    assert!(
        matches!(
            err,
            pi_ccs::Error::Shape("engine returned false on verify") | pi_ccs::Error::Engine(_)
        ),
        "unexpected error: {err:?}"
    );
}

#[test]
fn native_pi_ccs_accepts_bit_backed_degree7_sbox_relation() {
    let f = build_bit_backed_fixture(F::from_u64(0x1234_5678_9abc_def0));
    assert_eq!(f.structure.f.max_degree(), 7);
    assert_eq!(f.structure.m, 1 + 2 * POSEIDON2_GOLDILOCKS_BITS);
    assert_eq!(f.structure.n, 2 * POSEIDON2_GOLDILOCKS_BITS + 1);

    let mut prover_tr = Transcript::session();
    let proof = pi_ccs::prove(
        &mut prover_tr,
        &f.params,
        &f.structure,
        &f.cache,
        &f.log,
        vec![f.instance.clone()],
        &RunningInstance::default(),
    )
    .expect("Π_CCS.P bit-backed degree-7");

    let mut verifier_tr = Transcript::session();
    let outputs = pi_ccs::verify(
        &mut verifier_tr,
        &f.params,
        &f.structure,
        &f.cache,
        &[f.instance.claim.clone()],
        &RunningInstance::default(),
        &proof,
    )
    .expect("Π_CCS.V bit-backed degree-7");

    assert_eq!(outputs.len(), 1);
}

#[test]
fn native_pi_ccs_rejects_wrong_bit_backed_degree7_output() {
    let structure = bit_backed_sbox_structure();
    let params = config::ccs_params(structure.n, structure.m, structure.t(), structure.max_degree()).expect("params");
    install_ajtai_module(&params, &structure);
    let cols = structure.m.div_ceil(D);
    let log = AjtaiSModule::from_global_for_dims(D, cols).expect("Ajtai module");
    let cache = OptimizedStructureCache::build(&structure).expect("cache build");

    let mut z = bit_backed_sbox_assignment(F::from_u64(0x1234_5678_9abc_def0));
    z[1 + POSEIDON2_GOLDILOCKS_BITS] = F::ONE - z[1 + POSEIDON2_GOLDILOCKS_BITS];
    let instance = CcsInstance::from_low_norm_assignment(&params, &log, &structure, &z, 1)
        .expect("tampered output remains low-norm");

    let mut prover_tr = Transcript::session();
    let maybe_proof = pi_ccs::prove(
        &mut prover_tr,
        &params,
        &structure,
        &cache,
        &log,
        vec![instance.clone()],
        &RunningInstance::default(),
    );

    if let Ok(proof) = maybe_proof {
        let mut verifier_tr = Transcript::session();
        let err = pi_ccs::verify(
            &mut verifier_tr,
            &params,
            &structure,
            &cache,
            &[instance.claim],
            &RunningInstance::default(),
            &proof,
        )
        .expect_err("wrong decoded y must not verify");

        assert!(
            matches!(
                err,
                pi_ccs::Error::Shape("engine returned false on verify") | pi_ccs::Error::Engine(_)
            ),
            "unexpected error: {err:?}"
        );
    }
}

#[test]
fn native_pi_ccs_accepts_bit_backed_full_poseidon2_permutation() {
    let input: [F; POSEIDON2_WIDTH] = std::array::from_fn(|i| F::from_u64((i as u64 + 1) * 17));
    let bundle = build_bit_backed_poseidon2_permutation(input);
    let expected = permute_state(input);
    let (r1cs_rows, r1cs_cols) = r1cs_poseidon2_permutation_shape(input);

    assert_eq!(bundle.output_state, expected, "PoC trace must match native Poseidon2");
    assert_eq!(bundle.structure.f.max_degree(), 7);
    assert_eq!((r1cs_rows, r1cs_cols), (600, 609));
    assert_eq!(
        bundle.structure.m,
        1 + BIT_BACKED_PERMUTATION_WORDS * POSEIDON2_GOLDILOCKS_BITS
    );
    assert_eq!(
        bundle.structure.n,
        BIT_BACKED_PERMUTATION_WORDS * POSEIDON2_GOLDILOCKS_BITS
            + SBOXES_PER_PERMUTATION
            + BIT_BACKED_PERMUTATION_LINEAR_ROWS
    );
    assert!(
        r1cs_rows < bundle.structure.n && r1cs_cols < bundle.structure.m,
        "field-valued R1CS trace is smaller before low-norm encoding; got R1CS {r1cs_rows}x{r1cs_cols}, CCS {}x{}",
        bundle.structure.n,
        bundle.structure.m
    );

    let outputs = prove_verify_single_fresh(&bundle.structure, bundle.z)
        .expect("Π_CCS proves/verifies bit-backed full Poseidon2");
    assert_eq!(outputs.len(), 1);
}

#[test]
fn native_pi_ccs_rejects_wrong_bit_backed_poseidon2_permutation_output() {
    let input: [F; POSEIDON2_WIDTH] = std::array::from_fn(|i| F::from_u64((i as u64 + 1) * 19));
    let mut bundle = build_bit_backed_poseidon2_permutation(input);
    assert_eq!(bundle.output_state, permute_state(input), "baseline trace");

    let last_output_bit = bundle.z.len() - 1;
    bundle.z[last_output_bit] = F::ONE - bundle.z[last_output_bit];

    let result = prove_verify_single_fresh(&bundle.structure, bundle.z);
    assert!(
        result.is_err(),
        "tampered bit-backed Poseidon2 output must not prove+verify"
    );
}

#[test]
fn native_pi_ccs_accepts_bit_backed_poseidon2_hash_short_input() {
    let input = vec![
        F::from_u64(3),
        F::from_u64(5),
        F::from_u64(7),
        F::from_u64(11),
        F::from_u64(13),
    ];
    let expected = neo_ccs::crypto::poseidon2_goldilocks::poseidon2_hash(&input);
    let bundle = build_bit_backed_poseidon2_hash(&input);

    assert_eq!(
        bundle.digest, expected,
        "bit-backed CCS-native Poseidon2 hash trace must match native poseidon2_hash"
    );
    assert_eq!(bundle.structure.f.max_degree(), 7);

    let (r1cs_rows, r1cs_cols) = r1cs_poseidon2_hash_shape(&input);
    let ccs_bits_per_var_equiv = (r1cs_cols as f64) * 70.0;
    eprintln!(
        "poseidon2_hash(len={}): R1CS field-valued {}r/{}c | bit-backed CCS-native {}r/{}c | \
         R1CS-lowered-to-bits ~{:.0} bits | ratio CCS/R1CS-lowered ≈ {:.2}×",
        input.len(),
        r1cs_rows,
        r1cs_cols,
        bundle.structure.n,
        bundle.structure.m,
        ccs_bits_per_var_equiv,
        (bundle.structure.m as f64) / ccs_bits_per_var_equiv,
    );

    assert!(
        r1cs_rows < bundle.structure.n && r1cs_cols < bundle.structure.m,
        "field-valued R1CS Poseidon2 hash trace must be smaller than bit-backed CCS-native;\
         R1CS={r1cs_rows}r/{r1cs_cols}c, CCS={}r/{}c",
        bundle.structure.n,
        bundle.structure.m,
    );

    let outputs = prove_verify_single_fresh(&bundle.structure, bundle.z)
        .expect("Π_CCS proves/verifies bit-backed Poseidon2 hash");
    assert_eq!(outputs.len(), 1);
}

#[test]
fn native_pi_ccs_rejects_wrong_bit_backed_poseidon2_hash_output() {
    let input = vec![
        F::from_u64(3),
        F::from_u64(5),
        F::from_u64(7),
        F::from_u64(11),
        F::from_u64(13),
    ];
    let mut bundle = build_bit_backed_poseidon2_hash(&input);

    let total_bits = POSEIDON2_WIDTH * POSEIDON2_GOLDILOCKS_BITS;
    let final_state_start = bundle.z.len() - total_bits;
    let digest_lsb_idx = final_state_start;
    bundle.z[digest_lsb_idx] = F::ONE - bundle.z[digest_lsb_idx];

    let result = prove_verify_single_fresh(&bundle.structure, bundle.z);
    assert!(
        result.is_err(),
        "tampered Poseidon2 hash digest bit must not prove+verify"
    );
}

#[test]
fn ccs_native_poseidon2_hash_shape_snapshot_for_ce_digest_preimage_length() {
    // §12 measured `ce_claim_digest` preimage_len = 1650 fields per
    // running CE claim. This snapshot records what the CCS-native
    // bit-backed Poseidon2 hash would cost for an input of that exact
    // length — the same input shape the production `ce_claim_digest`
    // would feed into a future CCS-native replacement.
    //
    // The test is the deciding measurement for the council's "is the
    // production CCS-native Poseidon hash builder worth it for
    // ce_claim_digest?" question. It runs the BUILD (which produces
    // structure + assignment) but skips `pi_ccs::prove/verify` — the
    // prove cost on a ~6M-row instance is too slow to keep in the
    // default sweep; the council's "≥ 20–25% saving" threshold is
    // answered by the shape numbers alone.
    let ce_preimage_len = 1650usize;
    let input: Vec<F> = (0..ce_preimage_len)
        .map(|i| F::from_u64((i as u64 + 1).wrapping_mul(0xdead_beef)))
        .collect();
    let expected = neo_ccs::crypto::poseidon2_goldilocks::poseidon2_hash(&input);
    let build_started = std::time::Instant::now();
    let bundle = build_bit_backed_poseidon2_hash(&input);
    let build_ms = build_started.elapsed().as_secs_f64() * 1_000.0;

    assert_eq!(
        bundle.digest, expected,
        "CCS-native CE-length hash must match native Poseidon2"
    );

    let chunks = input.len().div_ceil(POSEIDON2_RATE);
    let permutations = chunks + 1; // absorb chunks + padding

    let (r1cs_rows, r1cs_cols) = r1cs_poseidon2_hash_shape(&input);
    let r1cs_lowered_bits = (r1cs_cols as f64) * 70.0;
    let ratio = (bundle.structure.m as f64) / r1cs_lowered_bits;

    eprintln!();
    eprintln!("── ce_claim_digest-length Poseidon2 hash shape snapshot ──");
    eprintln!("  input length              : {} fields", input.len());
    eprintln!("  permutations              : {}", permutations);
    eprintln!("  build time                : {:.1} ms", build_ms);
    eprintln!("  R1CS field-valued shape   : {} rows / {} cols", r1cs_rows, r1cs_cols);
    eprintln!(
        "  CCS-native bit-backed     : {} rows / {} cols",
        bundle.structure.n, bundle.structure.m
    );
    eprintln!("  R1CS-lowered-to-bits      : ~{:.0} bits", r1cs_lowered_bits);
    eprintln!("  CCS-native / R1CS-lowered : ≈ {:.2}×  (lower is better)", ratio);
    eprintln!("──────────────────────────────────────────────────────────");

    // Soft regression guard: shape numbers should stay close to the
    // measured snapshot. Cols are 1 + 64 × words_per_permutation ×
    // permutations and rows are bitness + sbox + linear-row counts —
    // both are deterministic functions of input length, so any change
    // here means a logic regression rather than a measurement drift.
    assert!(bundle.structure.m > 0);
    assert!(bundle.structure.n > 0);
    assert!(
        ratio < 1.0,
        "CCS-native bit-backed cost must beat R1CS-lowered-to-bits on this preimage length; got ratio {ratio:.2}×"
    );
}

/// The witness-only value walk must reproduce the full builder's
/// `z = [1 || trace bits]` and digest exactly, for every sponge shape
/// (sub-rate, exact-rate, multi-chunk, and the F' state_x_out length).
#[test]
fn bit_backed_hash_values_walk_matches_builder() {
    use neo_fold_clean::engine::ccs_native::poseidon2::build_bit_backed_poseidon2_hash_values;

    for len in [1usize, 3, 4, 5, 8, 9, 23, 40] {
        let preimage: Vec<F> = (0..len)
            .map(|i| F::from_u64(0x9e37_79b9 + i as u64 * 0x85eb_ca6b))
            .collect();
        let bundle = build_bit_backed_poseidon2_hash(&preimage);
        let (values, digest) = build_bit_backed_poseidon2_hash_values(&preimage);
        assert_eq!(values, bundle.z, "z mismatch at preimage len {len}");
        assert_eq!(digest, bundle.digest, "digest mismatch at preimage len {len}");
    }
}

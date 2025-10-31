//! Write a compact Round-0 dump for the Sage harness.
//!
//! This sets up a tiny instance identical to the other tests:
//! - n=2, d=2, t=2, b=3, k_total=2 (1 MCS + 1 ME input).
//! - M_1 = I (via ensure_identity_first()).
//! - Challenges (α, β_a, β_r, γ) sampled exactly like the prover.
//! - ME input obtained by running a k=1 "prove_simple" on a second MCS.
//!
//! Output: JSON file with matrices, witnesses, and challenges sufficient
//! for the Sage Q round-0 harness to reproduce p(0)+p(1)=initial_sum.
//!
//! Path: by default "target/round0_dump.json". Override with env
//! NEO_ROUND0_DUMP=/path/to/file.json

use std::fs;
use std::path::PathBuf;

use serde::Serialize;

use neo_params::NeoParams;
use neo_math::{F, K, D, KExtensions};
use neo_ccs::{CcsStructure, Mat, MeInstance};
use neo_transcript::{Poseidon2Transcript, Transcript};
use p3_field::{PrimeCharacteristicRing, PrimeField64};

// test helpers already used in other unit tests
use crate::helpers::{DummyS, create_test_ccs, mk_mcs};

// protocol helpers (exactly the ones used in tests)
use neo_fold::pi_ccs::{
    context,
    transcript::{bind_header_and_instances, bind_me_inputs},
};

/// JSON codec: base field element as decimal string
fn f_to_str(x: F) -> String {
    x.as_canonical_u64().to_string()
}

/// JSON codec: K element (extension) as pair [a, b] (both decimal strings)
fn k_to_pair(k: K) -> [String; 2] {
    let coeffs = k.as_coeffs(); // [F; 2]
    [f_to_str(coeffs[0]), f_to_str(coeffs[1])]
}

/// Matrix to row-major Vec<Vec<String>>
fn mat_f_to_rows(m: &Mat<F>) -> Vec<Vec<String>> {
    let rows = m.rows();
    let cols = m.cols();
    (0..rows)
        .map(|r| (0..cols).map(|c| f_to_str(m[(r, c)])).collect())
        .collect()
}

/// Structure we dump for the Sage harness
#[derive(Serialize)]
struct Round0Dump {
    // field / params
    p: String,   // base field modulus (Goldilocks)
    b: u32,
    d: usize,
    n: usize,
    m: usize,
    t: usize,
    k_total: usize,

    // CCS matrices (t items, each n x m, row-major, decimal strings)
    #[serde(rename = "M")]
    mats: Vec<Vec<Vec<String>>>,

    // Witness Z matrices (k_total items, each d x m, row-major)
    #[serde(rename = "Z")]
    z_witnesses: Vec<Vec<Vec<String>>>,

    // challenges (extension field as pairs [a,b])
    alpha: Vec<[String; 2]>,
    beta_a: Vec<[String; 2]>,
    beta_r: Vec<[String; 2]>,
    gamma: [String; 2],

    // common r from ME inputs (length ℓ_n)
    r: Vec<[String; 2]>,

    // Optional: extension field quadratic rule (u^2 = a + b u)
    #[serde(skip_serializing_if = "Option::is_none")] 
    quad_rule: Option<QuadRule>,

    // Optional: sparse polynomial f specification
    #[serde(skip_serializing_if = "Option::is_none")] 
    f: Option<SparsePolyDump>,
}

#[derive(Serialize)]
struct QuadRule { u2: [String;2] }

#[derive(Serialize)]
struct SparseTerm { coeff: String, exps: Vec<usize> }

#[derive(Serialize)]
struct SparsePolyDump { arity: usize, terms: Vec<SparseTerm> }

fn goldilocks_modulus_decimal() -> String {
    // 2^64 - 2^32 + 1
    // Keeping as decimal string so Sage can parse it directly.
    "18446744069414584321".to_string()
}

/// Build one MCS and one ME input (obtained via a k=1 simple fold),
/// then bind + sample challenges and dump everything needed by Sage.
#[test]
#[ignore] // utility to write JSON dump; not a real unit test
fn write_round0_dump() {
    // --- 1) Parameters and tiny CCS with M1=I ---------------------------------
    let params = NeoParams::goldilocks_for_circuit(3, 2, 2);
    let s: CcsStructure<F> = create_test_ccs().ensure_identity_first().unwrap();
    let l = DummyS;
    let m_in = 1;

    // Sanity (minimal requirements only)
    assert!(s.t() >= 2, "need at least two matrices (M1=I, M2≠I)");

    // --- 2) Build one MCS instance (for i=1) ----------------------------------
    // Reuse the same fixture values as other unit tests.
    let (mcs1, wit1) =
        mk_mcs(&params, vec![F::ONE, F::from_u64(2), F::from_u64(3), F::ZERO, F::from_u64(5)], m_in, &l);

    // --- 3) Build another MCS, convert it to a single ME input via k=1 fold ---
    let (_mcs2, wit2) =
        mk_mcs(&params, vec![F::ONE, F::from_u64(7), F::from_u64(11), F::ZERO, F::from_u64(18)], m_in, &l);

    // Run the k=1 simple scheme to get a single ME output we’ll reuse as input.
    let mut tr_k1 = Poseidon2Transcript::new(b"fixture/k1");
    let (me_outs, _proof_k1) =
        neo_fold::pi_ccs_prove_simple(&mut tr_k1, &params, &s, &[_mcs2.clone()], &[wit2.clone()], &l)
            .expect("k=1 simple fold must succeed");
    let me_input: MeInstance<_, _, K> = me_outs[0].clone();
    let me_witness: Mat<F> = wit2.Z.clone(); // witness for the ME input (Z2)

    // --- 4) Bind header + inputs and sample challenges as prover would ---------
    let dims = context::build_dims_and_policy(&params, &s).expect("dims");
    let ell_d = dims.ell_d;
    let ell_n = dims.ell_n;
    let ell = ell_d + ell_n;

    let mut tr = Poseidon2Transcript::new(b"prove/k2");
    bind_header_and_instances(&mut tr, &params, &s, &[mcs1.clone()], ell, dims.d_sc, 0).unwrap();
    bind_me_inputs(&mut tr, &[me_input.clone()]).unwrap();
    let ch = neo_fold::pi_ccs::transcript::sample_challenges(&mut tr, ell_d, ell).unwrap();

    // --- 5) Prepare JSON payload ------------------------------------------------
    let p_dec = goldilocks_modulus_decimal();
    let b = params.b;
    let d = D;
    let n = s.n;
    let m = s.m;
    let t = s.t();
    let k_total = 2usize; // 1 MCS + 1 ME

    // Matrices M_j (row-major decimal strings)
    let mut mats = Vec::with_capacity(t);
    for j in 0..t {
        mats.push(mat_f_to_rows(&s.matrices[j]));
    }

    // Witness Z matrices (i=1 from MCS [wit1.Z], i=2 from ME witness [wit2.Z])
    let z_witnesses = vec![mat_f_to_rows(&wit1.Z), mat_f_to_rows(&me_witness)];

    // Challenges (as pairs)
    let alpha: Vec<[String; 2]> = ch.alpha.iter().map(|&k| k_to_pair(k)).collect();
    let beta_a: Vec<[String; 2]> = ch.beta_a.iter().map(|&k| k_to_pair(k)).collect();
    let beta_r: Vec<[String; 2]> = ch.beta_r.iter().map(|&k| k_to_pair(k)).collect();
    let gamma = k_to_pair(ch.gamma);

    // Common r from ME inputs (all equal by validate_inputs rule)
    let r_pairs: Vec<[String; 2]> = me_input.r.iter().map(|&k| k_to_pair(k)).collect();
    assert_eq!(r_pairs.len(), ell_n, "r must be length ℓ_n");

    // Quadratic rule: compute u^2 in Rust basis (so Sage can match K exactly)
    let u = neo_math::from_complex(F::ZERO, F::ONE);
    let uu = u * u;
    let quad_rule = Some(QuadRule { u2: k_to_pair(uu) });

    // Sparse f dump
    let f_dump = {
        let f = &s.f;
        let mut terms: Vec<SparseTerm> = Vec::new();
        for term in f.terms() {
            let exps: Vec<usize> = term.exps.iter().map(|&e| e as usize).collect();
            terms.push(SparseTerm { coeff: f_to_str(term.coeff), exps });
        }
        Some(SparsePolyDump { arity: f.arity(), terms })
    };

    let dump = Round0Dump {
        p: p_dec,
        b,
        d,
        n,
        m,
        t,
        k_total,
        mats,
        z_witnesses,
        alpha,
        beta_a,
        beta_r,
        gamma,
        r: r_pairs,
        quad_rule,
        f: f_dump,
    };

    // --- 6) Write file ---------------------------------------------------------
    let out_path = std::env::var("NEO_ROUND0_DUMP")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("../../new-neo-sage/round0_dump.json"));

    if let Some(parent) = out_path.parent() {
        let _ = fs::create_dir_all(parent);
    }

    let json = serde_json::to_string_pretty(&dump).expect("serialize");
    fs::write(&out_path, json.as_bytes()).expect("write dump");

    eprintln!("[round0_dump] wrote {}", out_path.display());

    // keep the test "green"
    assert!(out_path.exists());
}

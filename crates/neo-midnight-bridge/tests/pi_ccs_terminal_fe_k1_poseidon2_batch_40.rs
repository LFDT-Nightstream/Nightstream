mod common;

use blake2b_simd::State as TranscriptHash;
use midnight_proofs::dev::cost_model::circuit_model;
use neo_math::{KExtensions, D};
use neo_midnight_bridge::k_field::KRepr;
use neo_midnight_bridge::relations::{
    PiCcsFeTerminalK1Instance, PiCcsFeTerminalK1Relation, PiCcsFeTerminalK1Witness, SparsePolyRepr, SparsePolyTermRepr,
};
use p3_field::PrimeField64;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use std::fs;
use std::path::PathBuf;

fn k_to_repr(k: &neo_math::K) -> KRepr {
    let (c0, c1) = k.to_limbs_u64();
    KRepr { c0, c1 }
}

#[test]
fn plonk_kzg_pi_ccs_terminal_fe_k1_poseidon2_batch_40_roundtrip() {
    // Prove the **k=1** FE terminal identity for a real Pi-CCS proof (SplitNcV1),
    // inside Midnight's PLONK/KZG verifier stack.
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let json_path = manifest_dir.join("../neo-fold/poseidon2-tests/poseidon2_ic_circuit_batch_40.json");
    let json = fs::read_to_string(&json_path).expect("read poseidon2 batch-40 json");
    let export = neo_fold::test_export::parse_test_export_json(&json).expect("parse test-export json");

    // Repeat the single witness vector to keep the overall workload similar to the other tests.
    // This relation still targets step 0 (k=1, no ME inputs).
    let target_folding_steps: usize = 2;

    let mut session = neo_fold::test_export::TestExportSession::new_from_circuit_json(&json).expect("session init");
    for i in 0..target_folding_steps {
        let z = &export.witness[i % export.witness.len()];
        session.add_step_witness_u64(z).expect("add witness step");
    }

    let (fold_run, _step_ms) = session
        .fold_and_prove_with_step_timings()
        .expect("fold_and_prove");
    assert_eq!(fold_run.steps.len(), target_folding_steps);
    assert!(session.verify(&fold_run).expect("verify"));

    let s = session.ccs();
    let n_pad = s.n.next_power_of_two().max(2);
    let ell_n = n_pad.trailing_zeros() as usize;
    let d_pad = D.next_power_of_two();
    let ell_d = d_pad.trailing_zeros() as usize;
    assert!(ell_n > 0);
    assert!(ell_d > 0);

    // Take the first step (k=1) Pi-CCS proof + its first output ME instance.
    let step0 = &fold_run.steps[0];
    let pi = &step0.fold.ccs_proof;
    assert_eq!(
        step0.fold.ccs_out.len(),
        1,
        "expected k=1 (no initial accumulator) for step 0"
    );
    let out0 = &step0.fold.ccs_out[0];

    // Extract (r', α') from FE sumcheck challenges.
    assert_eq!(
        pi.sumcheck_challenges.len(),
        ell_n + ell_d,
        "expected sumcheck_challenges = r' || α' with lengths (ell_n, ell_d)"
    );
    let (r_prime, alpha_prime) = pi.sumcheck_challenges.split_at(ell_n);
    assert_eq!(alpha_prime.len(), ell_d);

    // Extract β = (β_a, β_r).
    assert_eq!(pi.challenges_public.beta_a.len(), ell_d);
    assert_eq!(pi.challenges_public.beta_r.len(), ell_n);

    // y_scalars are the values used by the FE terminal identity.
    assert_eq!(out0.y_scalars.len(), s.t());

    // Encode the CCS sparse polynomial f as (u64 coeffs, u32 exps).
    let poly_terms: Vec<SparsePolyTermRepr> =
        s.f.terms()
            .iter()
            .map(|t| SparsePolyTermRepr {
                coeff: t.coeff.as_canonical_u64(),
                exps: t.exps.clone(),
            })
            .collect();
    let poly = SparsePolyRepr {
        t: s.t(),
        terms: poly_terms,
    };

    let rel = PiCcsFeTerminalK1Relation { ell_n, ell_d, poly };

    // Public statement = FE sumcheck final running sum.
    let instance = PiCcsFeTerminalK1Instance {
        final_sum: k_to_repr(&pi.sumcheck_final),
    };

    let witness = PiCcsFeTerminalK1Witness {
        r_prime: r_prime.iter().map(k_to_repr).collect(),
        alpha_prime: alpha_prime.iter().map(k_to_repr).collect(),
        beta_a: pi.challenges_public.beta_a.iter().map(k_to_repr).collect(),
        beta_r: pi.challenges_public.beta_r.iter().map(k_to_repr).collect(),
        y_scalars: out0.y_scalars.iter().map(k_to_repr).collect(),
    };

    // Use test-only KZG params. Production must use Midnight's SRS.
    let circuit = midnight_zk_stdlib::MidnightCircuit::from_relation(&rel);
    println!("MidnightCircuit: {circuit:?}");
    let model = circuit_model::<_, 48, 32>(&circuit);
    let k: u32 = model.k;
    println!("Midnight min_k() for PiCcsFeTerminalK1Relation: {k}");
    println!(
        "CircuitModel: rows={} table_rows={} unusable_rows={} max_deg={} advice_cols={} fixed_cols={} lookups={}",
        model.rows,
        model.table_rows,
        model.nb_unusable_rows,
        model.max_deg,
        model.advice_columns,
        model.fixed_columns,
        model.lookups
    );
    let params = common::test_kzg_params(k);

    let vk = midnight_zk_stdlib::setup_vk(&params, &rel);
    let pk = midnight_zk_stdlib::setup_pk(&rel, &vk);
    let proof = midnight_zk_stdlib::prove::<_, TranscriptHash>(
        &params,
        &pk,
        &rel,
        &instance,
        witness,
        ChaCha20Rng::from_seed([42u8; 32]),
    )
    .expect("prove");

    println!("Midnight proof bytes: {}", proof.len());
    assert!(proof.len() < 50 * 1024, "expected < 50KB proof");

    let params_v = params.verifier_params();
    midnight_zk_stdlib::verify::<PiCcsFeTerminalK1Relation, TranscriptHash>(&params_v, &vk, &instance, None, &proof)
        .expect("verify");
}

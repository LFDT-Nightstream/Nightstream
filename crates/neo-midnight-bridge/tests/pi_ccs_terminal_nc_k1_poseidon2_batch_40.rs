mod common;

use blake2b_simd::State as TranscriptHash;
use midnight_proofs::dev::cost_model::circuit_model;
use neo_math::{KExtensions, D};
use neo_midnight_bridge::k_field::KRepr;
use neo_midnight_bridge::relations::{PiCcsNcTerminalK1Instance, PiCcsNcTerminalK1Relation, PiCcsNcTerminalK1Witness};
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use std::fs;
use std::path::PathBuf;

fn k_to_repr(k: &neo_math::K) -> KRepr {
    let (c0, c1) = k.to_limbs_u64();
    KRepr { c0, c1 }
}

#[test]
fn plonk_kzg_pi_ccs_terminal_nc_k1_poseidon2_batch_40_roundtrip() {
    // Prove the **k=1** NC terminal identity for a real Pi-CCS proof (SplitNcV1),
    // inside Midnight's PLONK/KZG verifier stack.
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let json_path = manifest_dir.join("../neo-fold/poseidon2-tests/poseidon2_ic_circuit_batch_40.json");
    let json = fs::read_to_string(&json_path).expect("read poseidon2 batch-40 json");
    let export = neo_fold::test_export::parse_test_export_json(&json).expect("parse test-export json");

    // Repeat the single witness vector to keep the overall workload similar to the other tests.
    // This relation targets step 0 (k=1, no ME inputs).
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
    let m_pad = s.m.next_power_of_two().max(2);
    let ell_m = m_pad.trailing_zeros() as usize;
    let d_pad = D.next_power_of_two();
    let ell_d = d_pad.trailing_zeros() as usize;

    // Take the first step (k=1) Pi-CCS proof + its first output ME instance.
    let step0 = &fold_run.steps[0];
    let pi = &step0.fold.ccs_proof;
    assert_eq!(
        step0.fold.ccs_out.len(),
        1,
        "expected k=1 (no initial accumulator) for step 0"
    );
    let out0 = &step0.fold.ccs_out[0];
    assert!(!out0.s_col.is_empty(), "expected NC channel s_col present");
    assert_eq!(out0.s_col.len(), ell_m);

    let want_nc_chals = ell_m + ell_d;
    assert_eq!(
        pi.sumcheck_challenges_nc.len(),
        want_nc_chals,
        "expected NC challenges = s' || α'_nc with lengths (ell_m, ell_d)"
    );
    let (s_col_prime, alpha_prime_nc) = pi.sumcheck_challenges_nc.split_at(ell_m);
    assert_eq!(alpha_prime_nc.len(), ell_d);

    assert_eq!(pi.challenges_public.beta_a.len(), ell_d);
    assert_eq!(pi.challenges_public.beta_m.len(), ell_m);

    let params_b = session.params().b;

    let rel = PiCcsNcTerminalK1Relation {
        ell_d,
        ell_m,
        b: params_b,
    };
    let instance = PiCcsNcTerminalK1Instance {
        final_sum_nc: k_to_repr(&pi.sumcheck_final_nc),
    };
    let witness = PiCcsNcTerminalK1Witness {
        s_col_prime: s_col_prime.iter().map(k_to_repr).collect(),
        alpha_prime: alpha_prime_nc.iter().map(k_to_repr).collect(),
        beta_a: pi.challenges_public.beta_a.iter().map(k_to_repr).collect(),
        beta_m: pi.challenges_public.beta_m.iter().map(k_to_repr).collect(),
        gamma: k_to_repr(&pi.challenges_public.gamma),
        y_zcol: out0.y_zcol.iter().map(k_to_repr).collect(),
    };
    assert_eq!(
        witness.y_zcol.len(),
        1usize << ell_d,
        "expected y_zcol padded to 2^ell_d"
    );

    // Use test-only KZG params. Production must use Midnight's SRS.
    let circuit = midnight_zk_stdlib::MidnightCircuit::from_relation(&rel);
    println!("MidnightCircuit: {circuit:?}");
    let model = circuit_model::<_, 48, 32>(&circuit);
    let k: u32 = model.k;
    println!("Midnight min_k() for PiCcsNcTerminalK1Relation: {k}");
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
        ChaCha20Rng::from_seed([52u8; 32]),
    )
    .expect("prove");

    println!("Midnight proof bytes: {}", proof.len());
    assert!(proof.len() < 50 * 1024, "expected < 50KB proof");

    let params_v = params.verifier_params();
    midnight_zk_stdlib::verify::<PiCcsNcTerminalK1Relation, TranscriptHash>(&params_v, &vk, &instance, None, &proof)
        .expect("verify");
}

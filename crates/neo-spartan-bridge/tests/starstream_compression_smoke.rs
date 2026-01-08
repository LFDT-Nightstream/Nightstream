#![allow(non_snake_case)]

//! Phase-1 smoke test: compress a small multi-step optimized FoldRun into a Spartan2 proof.

use std::fs;
use std::path::PathBuf;
use std::sync::Arc;

use neo_ajtai::{set_global_pp_seeded, AjtaiSModule};
use neo_ccs::{r1cs_to_ccs, CcsStructure, Mat};
use neo_fold::pi_ccs::FoldingMode;
use neo_fold::session::{FoldingSession, NeoStep, StepArtifacts, StepSpec};
use neo_math::{D, F};
use neo_params::NeoParams;
use neo_spartan_bridge::circuit::FoldRunWitness;
use neo_spartan_bridge::{compute_vm_digest_v1, prove_fold_run, setup_fold_run, verify_fold_run, verify_fold_run_proof_only};
use p3_field::PrimeCharacteristicRing;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct TestExport {
    ivc_params: IvcParams,
    steps: Vec<StepData>,
}

#[derive(Debug, Deserialize)]
struct IvcParams {
    step_spec: JsonStepSpec,
}

#[derive(Debug, Deserialize)]
struct JsonStepSpec {
    y_len: usize,
    const1_index: usize,
    y_step_indices: Vec<usize>,
    y_prev_indices: Vec<usize>,
}

#[derive(Debug, Clone, Deserialize)]
struct StepData {
    witness: WitnessData,
    r1cs: R1csData,
}

#[derive(Debug, Clone, Deserialize)]
struct WitnessData {
    z_full: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct R1csData {
    num_constraints: usize,
    num_variables: usize,
    num_public_inputs: usize,
    a_sparse: Vec<(usize, usize, String)>,
    b_sparse: Vec<(usize, usize, String)>,
    c_sparse: Vec<(usize, usize, String)>,
}

fn parse_field_element(s: &str) -> F {
    if let Some(hex) = s.strip_prefix("0x") {
        F::from_u64(u64::from_str_radix(hex, 16).expect("valid hex"))
    } else {
        F::from_u64(s.parse::<u64>().expect("valid decimal"))
    }
}

fn sparse_to_dense_mat(sparse: &[(usize, usize, String)], rows: usize, cols: usize) -> Mat<F> {
    let mut data = vec![F::ZERO; rows * cols];
    for (row, col, val_str) in sparse {
        data[row * cols + col] = parse_field_element(val_str);
    }
    Mat::from_row_major(rows, cols, data)
}

fn build_step_ccs(r1cs: &R1csData) -> CcsStructure<F> {
    let n = r1cs.num_constraints;
    let m = r1cs.num_variables;
    let m_padded = n.max(m);

    let a = sparse_to_dense_mat(&r1cs.a_sparse, n, m_padded);
    let b = sparse_to_dense_mat(&r1cs.b_sparse, n, m_padded);
    let c = sparse_to_dense_mat(&r1cs.c_sparse, n, m_padded);
    let s0 = r1cs_to_ccs(a, b, c);

    s0.ensure_identity_first_owned()
        .expect("ensure_identity_first_owned should succeed")
}

fn extract_witness(witness_data: &WitnessData) -> Vec<F> {
    witness_data.z_full.iter().map(|s| parse_field_element(s)).collect()
}

fn pad_witness_to_m(mut z: Vec<F>, m_target: usize) -> Vec<F> {
    z.resize(m_target, F::ZERO);
    z
}

fn load_test_export_and_vm_digest() -> (TestExport, [u8; 32]) {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let json_path = manifest_dir.join("../neo-fold/starstream-tests/test_starstream_tx_export_valid.json");
    let json_bytes = fs::read(&json_path).expect("read JSON bytes");
    let vm_digest = compute_vm_digest_v1(&json_bytes);
    let export: TestExport = serde_json::from_slice(&json_bytes).expect("parse JSON");
    (export, vm_digest)
}

#[derive(Clone)]
struct NoInputs;

struct StarstreamStepCircuit {
    steps: Vec<StepData>,
    step_spec: StepSpec,
    step_ccs: Arc<CcsStructure<F>>,
}

impl NeoStep for StarstreamStepCircuit {
    type ExternalInputs = NoInputs;

    fn state_len(&self) -> usize {
        self.step_spec.y_len
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
        let z = extract_witness(&self.steps[step_idx].witness);
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
fn test_starstream_tx_export_spartan_phase1_smoke() {
    let (export, vm_digest) = load_test_export_and_vm_digest();
    let steps_to_run: usize = std::env::var("NEO_SPARTAN_BRIDGE_STARSTREAM_STEPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(export.steps.len())
        .min(export.steps.len());

    let step_spec = StepSpec {
        y_len: export.ivc_params.step_spec.y_len,
        const1_index: export.ivc_params.step_spec.const1_index,
        y_step_indices: export.ivc_params.step_spec.y_step_indices.clone(),
        app_input_indices: Some(export.ivc_params.step_spec.y_prev_indices.clone()),
        m_in: export.steps[0].r1cs.num_public_inputs,
    };

    let step_ccs = Arc::new(build_step_ccs(&export.steps[0].r1cs));
    let mut circuit = StarstreamStepCircuit {
        steps: export.steps.clone(),
        step_spec: step_spec.clone(),
        step_ccs: step_ccs.clone(),
    };

    let m_commit = step_ccs.n.max(step_ccs.m);
    let params = NeoParams::goldilocks_auto_r1cs_ccs(m_commit).expect("params");
    let seed = [42u8; 32];
    set_global_pp_seeded(D, params.kappa as usize, m_commit, seed).expect("set_global_pp_seeded");
    let committer = AjtaiSModule::from_global_for_dims(D, m_commit).expect("committer");
    let mut session = FoldingSession::new(FoldingMode::Optimized, params, committer);
    let inputs = NoInputs;
    session
        .add_steps(&mut circuit, &inputs, steps_to_run)
        .expect("add_steps");

    let run = session
        .prove_and_verify_collected(step_ccs.as_ref())
        .expect("prove_and_verify_collected");

    let steps_len = run.steps.len();
    assert_eq!(steps_len, steps_to_run);

    // Phase 1: no initial accumulator (k=1 seed).
    let initial_accumulator = Vec::new();
    let steps_public = session.steps_public();
    let witness = FoldRunWitness::new(
        run.clone(),
        steps_public.clone(),
        initial_accumulator.clone(),
        vm_digest,
        None,
    );

    let (pk, vk) = setup_fold_run(session.params(), step_ccs.as_ref(), &witness).expect("setup_fold_run");

    let spartan = prove_fold_run(&pk, session.params(), step_ccs.as_ref(), witness).expect("prove_fold_run");

    // Print proof size information
    let proof_bytes = spartan.proof_data.len();
    let serialized = bincode::serialize(&spartan).expect("serialize proof");
    let serialized_bytes = serialized.len();
    println!("Spartan proof size:");
    println!("  proof_data:  {} bytes ({:.2} KB)", proof_bytes, proof_bytes as f64 / 1024.0);
    println!("  serialized:  {} bytes ({:.2} KB)", serialized_bytes, serialized_bytes as f64 / 1024.0);

    let stmt = verify_fold_run_proof_only(&vk, &spartan).expect("verify_fold_run_proof_only");
    assert_eq!(stmt.vm_digest, vm_digest);

    assert!(
        verify_fold_run(&vk, session.params(), step_ccs.as_ref(), &vm_digest, &steps_public, None, &[], &spartan)
            .expect("verify_fold_run"),
        "Spartan proof should verify"
    );
    let wrong_vm_digest = compute_vm_digest_v1(b"wrong");
    assert!(
        verify_fold_run(&vk, session.params(), step_ccs.as_ref(), &wrong_vm_digest, &steps_public, None, &[], &spartan).is_err(),
        "vm_digest mismatch must be rejected"
    );

    assert!(
        verify_fold_run(&vk, session.params(), step_ccs.as_ref(), &vm_digest, &steps_public, None, &[(0, 0)], &spartan)
            .is_err(),
        "mismatched step_linking policy must be rejected"
    );
}

#[test]
fn test_starstream_tx_export_spartan_phase1_tampering_is_rejected() {
    let (export, vm_digest) = load_test_export_and_vm_digest();

    // Keep this test small: one step is enough to exercise FS bindings (Π‑CCS chals, sumcheck chals, Π‑RLC ρ).
    let steps_to_run: usize = 1.min(export.steps.len());

    let step_spec = StepSpec {
        y_len: export.ivc_params.step_spec.y_len,
        const1_index: export.ivc_params.step_spec.const1_index,
        y_step_indices: export.ivc_params.step_spec.y_step_indices.clone(),
        app_input_indices: Some(export.ivc_params.step_spec.y_prev_indices.clone()),
        m_in: export.steps[0].r1cs.num_public_inputs,
    };

    let step_ccs = Arc::new(build_step_ccs(&export.steps[0].r1cs));
    let mut circuit = StarstreamStepCircuit {
        steps: export.steps.clone(),
        step_spec: step_spec.clone(),
        step_ccs: step_ccs.clone(),
    };

    let m_commit = step_ccs.n.max(step_ccs.m);
    let params = NeoParams::goldilocks_auto_r1cs_ccs(m_commit).expect("params");
    let seed = [42u8; 32];
    set_global_pp_seeded(D, params.kappa as usize, m_commit, seed).expect("set_global_pp_seeded");
    let committer = AjtaiSModule::from_global_for_dims(D, m_commit).expect("committer");
    let mut session = FoldingSession::new(FoldingMode::Optimized, params, committer);
    let inputs = NoInputs;
    session
        .add_steps(&mut circuit, &inputs, steps_to_run)
        .expect("add_steps");

    let run = session
        .prove_and_verify_collected(step_ccs.as_ref())
        .expect("prove_and_verify_collected");
    assert_eq!(run.steps.len(), steps_to_run);

    let initial_accumulator = Vec::new();
    let steps_public = session.steps_public();
    let witness = FoldRunWitness::new(run.clone(), steps_public.clone(), initial_accumulator, vm_digest, None);

    let (pk, vk) = setup_fold_run(session.params(), step_ccs.as_ref(), &witness).expect("setup_fold_run");

    let check_rejected = |w: FoldRunWitness| {
        match prove_fold_run(&pk, session.params(), step_ccs.as_ref(), w) {
            Ok(proof) => {
                assert!(
                    verify_fold_run(&vk, session.params(), step_ccs.as_ref(), &vm_digest, &steps_public, None, &[], &proof)
                        .is_err(),
                    "tampered witness must not verify"
                );
            }
            Err(_) => {
                // Also acceptable: proving can fail early due to unsatisfied constraints.
            }
        }
    };

    // 1) Tamper Π‑RLC ρ (FS enforces transcript-derived sampling).
    let mut w_rho = witness.clone();
    {
        use p3_field::PrimeCharacteristicRing;
        let rho0 = &mut w_rho.fold_run.steps[0].fold.rlc_rhos[0];
        rho0[(0, 0)] = rho0[(0, 0)] + F::ONE;
    }
    check_rejected(w_rho);

    // 2) Tamper Π‑CCS challenge (FS enforces transcript-derived α/β/γ).
    let mut w_ccs_chal = witness.clone();
    {
        use p3_field::PrimeCharacteristicRing;
        let proof = &mut w_ccs_chal.fold_run.steps[0].fold.ccs_proof;
        proof.challenges_public.alpha[0] = proof.challenges_public.alpha[0] + neo_math::K::ONE;
    }
    check_rejected(w_ccs_chal);

    // 3) Tamper sumcheck challenges (FS enforces transcript-derived r_time / α′).
    let mut w_sc_chal = witness.clone();
    {
        use p3_field::PrimeCharacteristicRing;
        let proof = &mut w_sc_chal.fold_run.steps[0].fold.ccs_proof;
        proof.sumcheck_challenges[0] = proof.sumcheck_challenges[0] + neo_math::K::ONE;
    }
    check_rejected(w_sc_chal);

    // 4) Tamper Pattern-A fields (Phase 1 requires Pattern-B only).
    let mut w_pattern = witness.clone();
    {
        use p3_field::PrimeCharacteristicRing;
        w_pattern.fold_run.steps[0].fold.ccs_out[0].c_step_coords.push(F::ONE);
        w_pattern.fold_run.steps[0].fold.ccs_out[0].u_len = 1;
    }
    check_rejected(w_pattern);

    // 5) Tamper fold_digest binding on CCS outputs (Phase 1 requires it equals header_digest).
    let mut w_fd = witness.clone();
    w_fd.fold_run.steps[0].fold.ccs_out[0].fold_digest[0] ^= 1;
    check_rejected(w_fd);

    // 6) Tamper header_digest bytes to a non-canonical Goldilocks encoding (should be rejected).
    let mut w_hdr = witness;
    {
        // Set limb0 = 0xFFFF_FFFF_0000_0001 (i.e. p), which is non-canonical as a u64 encoding.
        let bad_limb0 = [1u8, 0, 0, 0, 0xFF, 0xFF, 0xFF, 0xFF];
        let proof = &mut w_hdr.fold_run.steps[0].fold.ccs_proof;
        proof.header_digest[..8].copy_from_slice(&bad_limb0);

        // Keep Phase-1 structural checks happy: also rewrite each ME fold_digest to match.
        let mut fd = [0u8; 32];
        fd.copy_from_slice(&proof.header_digest[..32]);
        for me in w_hdr.fold_run.steps[0].fold.ccs_out.iter_mut() {
            me.fold_digest = fd;
        }
    }
    check_rejected(w_hdr);
}

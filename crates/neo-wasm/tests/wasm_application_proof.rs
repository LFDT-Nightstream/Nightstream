use neo_wasm::{collect_wasmtime_steps, WasmApplicationModule, WasmApplicationProof, WasmApplicationProofSystem};
use serde_json::Value;

const MODULE_MANIFEST: &[u8] = include_bytes!("fixtures/wasm_benchmark_42x6.module.json");
const PROOF_MANIFEST: &[u8] = include_bytes!("fixtures/wasm_benchmark_42x6.proof.json");

fn module() -> WasmApplicationModule {
    WasmApplicationModule::from_json_slice(MODULE_MANIFEST).expect("Lean module manifest must validate")
}

fn assert_manifest_mutation_rejected(mutate: impl FnOnce(&mut Value)) {
    let mut manifest: Value = serde_json::from_slice(PROOF_MANIFEST).unwrap();
    mutate(&mut manifest);
    let bytes = serde_json::to_vec(&manifest).unwrap();
    assert!(WasmApplicationProofSystem::setup(module(), &bytes).is_err());
}

#[test]
fn lean_owned_application_proves_and_verifies_with_spartan_and_whir() {
    let system = WasmApplicationProofSystem::setup(module(), PROOF_MANIFEST).unwrap();
    assert_eq!(system.module().module_id(), "wasm-benchmark-42x6");
    assert_eq!(
        system.stats(),
        neo_wasm::WasmApplicationProofStats {
            rows: 63,
            private_witness_columns: 1,
            public_values: 62,
            r1cs_nonzero_coefficients: 173,
            native_ccs_nonzero_coefficients: 236,
            maximum_r1cs_row_density: 3,
            maximum_native_ccs_row_density: 4,
            poseidon2_calls: 0,
            maximum_live_witness_columns: 1,
        }
    );

    let run = collect_wasmtime_steps(system.module().bytes(), system.module().entrypoint(), &[]).unwrap();
    let runtime_output = run.results[0].parse::<u64>().unwrap();
    assert_eq!(runtime_output, 252);

    let proof = system.prove(&[runtime_output], &[runtime_output]).unwrap();
    system.verify(&proof, &[252]).unwrap();

    let encoded = proof.to_bytes().unwrap();
    let decoded = WasmApplicationProof::from_bytes(&encoded).unwrap();
    system.verify(&decoded, &[252]).unwrap();

    assert!(system.prove(&[251], &[252]).is_err());
    assert!(system.prove(&[252], &[251]).is_err());
    assert!(system.verify(&proof, &[251]).is_err());

    let mut corrupted = encoded;
    let middle = corrupted.len() / 2;
    corrupted[middle] ^= 1;
    match WasmApplicationProof::from_bytes(&corrupted) {
        Err(_) => {}
        Ok(corrupted_proof) => assert!(system.verify(&corrupted_proof, &[252]).is_err()),
    }
}

#[test]
fn proof_manifest_is_fail_closed() {
    assert_manifest_mutation_rejected(|manifest| manifest["module_id"] = "another-module".into());
    assert_manifest_mutation_rejected(|manifest| manifest["module_hex"] = "00".into());
    assert_manifest_mutation_rejected(|manifest| manifest["matrix_count"] = 3.into());
    assert_manifest_mutation_rejected(|manifest| manifest["polynomial"][0]["exponents"][3] = 0.into());
    assert_manifest_mutation_rejected(|manifest| manifest["columns"][0]["role"] = "output".into());
    assert_manifest_mutation_rejected(|manifest| manifest["rows"][0]["selector"] = 1.into());
    assert_manifest_mutation_rejected(|manifest| manifest["rows"][0]["a"][0]["column"] = 2.into());
    assert_manifest_mutation_rejected(|manifest| manifest["cost"]["rows"] = 62.into());
    assert_manifest_mutation_rejected(|manifest| {
        manifest["metrics"]["maximum_r1cs_row_density"] = 2.into();
    });
}

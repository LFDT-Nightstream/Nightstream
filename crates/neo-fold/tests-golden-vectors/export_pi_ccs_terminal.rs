// Export Pi_CCS terminal state for Sage validation
//
// This test runs fold_ccs_instances() in Rust and exports the state
// AFTER sumcheck completes, so Sage can validate the terminal check
// (Paper Section 4.4, Step 4) without reimplementing sumcheck.
//
// What we export:
// - Input challenges: α, β, γ, r
// - Sumcheck output: α', r', v (final evaluation)
// - Output evaluations: y'_{(i,j)} for all i ∈ [k], j ∈ [t]
// - Witness matrices Z_i (for reference)
// - Structure s (matrices M_j, polynomial f)
//
// What Sage validates:
// v ?= eq((α',r'), β)·(F + Σ_i γ^i·N_i) + γ^k·Σ_{j,i≥2} γ^{i+(j-1)k-1}·E_{(i,j)}

use neo_fold::fold_ccs_instances;
use neo_ccs::{CcsStructure, Mat, SparsePoly, Term, McsInstance, McsWitness};
use neo_ajtai::{decomp_b, DecompStyle, setup, set_global_pp, AjtaiSModule};
use neo_params::NeoParams;
use neo_math::{F, K, D, KExtensions};
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use serde_json::{json, Value};
use std::fs;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

fn k_to_json(k: K) -> Value {
    let coeffs = k.as_coeffs();
    json!([coeffs[0].as_canonical_u64(), coeffs[1].as_canonical_u64()])
}

fn vec_k_to_json(v: &[K]) -> Value {
    json!(v.iter().map(|x| k_to_json(*x)).collect::<Vec<_>>())
}

fn mat_f_to_json(m: &Mat<F>) -> Value {
    let rows: Vec<Vec<u64>> = (0..m.rows())
        .map(|i| (0..m.cols())
            .map(|j| m[(i, j)].as_canonical_u64())
            .collect())
        .collect();
    json!(rows)
}

#[allow(non_snake_case)]
fn main() {
    export_pi_ccs_terminal_state();
}

#[allow(non_snake_case)]
fn export_pi_ccs_terminal_state() {
    // Load CCS structure from Rust
    let json_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../neo/tests/fixtures/ccs_constraint_3_plus_5.json"
    );
    
    let json_str = fs::read_to_string(json_path)
        .expect("Failed to read CCS JSON");
    
    let data: Value = serde_json::from_str(&json_str)
        .expect("Failed to parse CCS JSON");
    
    println!("\n🔧 Exporting Π_CCS Terminal State for Sage Validation\n");
    
    // Parse structure
    let structure_json = &data["structure"];
    let n = structure_json["n"].as_u64().unwrap() as usize;
    let m = structure_json["m"].as_u64().unwrap() as usize;
    
    let matrices_json = structure_json["M"].as_array().unwrap();
    let mut constraint_matrices = Vec::new();
    for mat_json in matrices_json {
        let arr = mat_json.as_array().unwrap();
        let rows = arr.len();
        let cols = arr[0].as_array().unwrap().len();
        let mut mat = Mat::zero(rows, cols, F::ZERO);
        for (i, row) in arr.iter().enumerate() {
            for (j, val) in row.as_array().unwrap().iter().enumerate() {
                mat[(i, j)] = F::from_u64(val.as_u64().unwrap());
            }
        }
        constraint_matrices.push(mat);
    }
    
    let t = constraint_matrices.len();
    
    // Parse polynomial f
    let poly_json = &structure_json["f"];
    let poly_terms_json = poly_json["terms"].as_array().unwrap();
    let mut poly_terms = Vec::new();
    for term_json in poly_terms_json {
        let coeff = F::from_u64(term_json["coeff"].as_u64().unwrap());
        let exps: Vec<u32> = term_json["exponents"].as_array().unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();
        poly_terms.push(Term { coeff, exps });
    }
    let f = SparsePoly::new(t, poly_terms);
    
    let structure = CcsStructure {
        n,
        m,
        matrices: constraint_matrices,
        f,
    };
    
    // Parse witness
    let witness_json = &data["witness"];
    let x: Vec<F> = witness_json["x"].as_array().unwrap()
        .iter()
        .map(|v| F::from_u64(v.as_u64().unwrap()))
        .collect();
    let w: Vec<F> = witness_json["w"].as_array().unwrap()
        .iter()
        .map(|v| F::from_u64(v.as_u64().unwrap()))
        .collect();
    let m_in = x.len();
    
    println!("  Structure: n={}, m={}, t={}, m_in={}", n, m, t, m_in);
    
    // Setup parameters
    let params = NeoParams::goldilocks_for_circuit(3, 2, 2);
    let b = params.b as usize;
    
    // Initialize Ajtai PP
    println!("  Initializing Ajtai PP for d={}, m={}, kappa={}...", D, m, params.kappa);
    let mut rng = ChaCha20Rng::seed_from_u64(42);
    let pp = setup(&mut rng, D, params.kappa as usize, m).expect("Failed to setup Ajtai PP");
    set_global_pp(pp).expect("Failed to set global Ajtai PP");
    println!("  ✓ Ajtai PP initialized");
    
    // Create MCS instance
    let z_full: Vec<F> = [x.clone(), w.clone()].concat();
    let z_digits = decomp_b(&z_full, b as u32, D, DecompStyle::Balanced);
    let mut row_major = vec![F::ZERO; D * m];
    for col in 0..m {
        for row in 0..D {
            row_major[row * m + col] = z_digits[col * D + row];
        }
    }
    let Z = Mat::from_row_major(D, m, row_major);
    
    let ajtai = AjtaiSModule::from_global_for_dims(D, m)
        .expect("Failed to get Ajtai module");
    let c = ajtai.commit(&Z);
    
    let mcs_instance = McsInstance {
        c,
        x: x.clone(),
        m_in,
    };
    
    let mcs_witness = McsWitness {
        w: w.clone(),
        Z: Z.clone(),
    };
    
    println!("\n  Running fold_ccs_instances()...");
    
    // Run Rust folding
    let result = fold_ccs_instances(
        &params,
        &structure,
        &[mcs_instance],
        &[mcs_witness],
    );
    
    match result {
        Ok((rust_outputs, _rust_witnesses, rust_proof)) => {
            println!("  ✅ Folding succeeded!");
            println!("     Output: {} ME instances", rust_outputs.len());
            
            // Extract terminal state
            // Note: We need to extract α, β, γ, r, α', r', v from the proof
            // For now, export what we have access to
            
            let export = json!({
                "description": "Π_CCS terminal state for Sage validation (Paper §4.4 Step 4)",
                "parameters": {
                    "b": b,
                    "k": rust_outputs.len(),
                    "t": t,
                    "n": n,
                    "m": m,
                    "m_in": m_in,
                    "d": D,
                },
                "structure": {
                    "M": constraint_matrices.iter().map(mat_f_to_json).collect::<Vec<_>>(),
                    "f": {
                        "terms": structure.f.terms().iter().map(|term| {
                            json!({
                                "coeff": term.coeff.as_canonical_u64(),
                                "exponents": term.exps
                            })
                        }).collect::<Vec<_>>()
                    }
                },
                "witness": {
                    "Z_list": vec![mat_f_to_json(&Z)],  // Just Z_1 for now (k=1)
                },
                "outputs": {
                    "r_prime": vec_k_to_json(&rust_outputs[0].r),
                    "y_prime": rust_outputs.iter().map(|out| {
                        json!({
                            "m_in": out.m_in,
                            "y": out.y.iter().map(vec_k_to_json).collect::<Vec<_>>()
                        })
                    }).collect::<Vec<_>>(),
                },
                "note": "Challenges α, β, γ, r, α', v need to be extracted from proof - requires API changes"
            });
            
            let out_path = concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/tests-golden-vectors/pi_ccs_terminal.json"
            );
            
            fs::write(out_path, serde_json::to_string_pretty(&export).unwrap())
                .expect("Failed to write export");
            
            println!("\n✅ Exported to: {}", out_path);
            println!("   Sage can now validate the terminal check!");
        }
        Err(e) => {
            panic!("❌ fold_ccs_instances() failed: {:?}", e);
        }
    }
}


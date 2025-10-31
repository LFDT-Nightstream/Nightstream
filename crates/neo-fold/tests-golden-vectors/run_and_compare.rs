// Golden vectors test: Run Rust implementation and compare with Sage outputs
//
// NEW APPROACH: Use export_pi_ccs_terminal.rs instead
// - Rust runs full protocol including sumcheck
// - Exports terminal state (after sumcheck completes)
// - Sage validates the terminal check (Paper §4.4 Step 4)
// - This tests Q polynomial construction without reimplementing sumcheck in Sage

use neo_fold::fold_ccs_instances;
use neo_ccs::{CcsStructure, Mat, SparsePoly, Term, McsInstance, McsWitness, SModuleHomomorphism};
use neo_ajtai::{decomp_b, DecompStyle, setup, set_global_pp, AjtaiSModule};
use neo_params::NeoParams;
use neo_math::{F, K, D};
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use serde_json::Value;
use std::fs;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

fn parse_f(val: &Value) -> F {
    let n = if val.is_u64() {
        val.as_u64().unwrap()
    } else if val.is_string() {
        val.as_str().unwrap().parse::<u64>().unwrap()
    } else {
        panic!("Cannot parse field element from {:?}", val);
    };
    F::from_u64(n)
}

fn parse_k(val: &Value) -> K {
    let n = if val.is_u64() {
        val.as_u64().unwrap()
    } else if val.is_string() {
        val.as_str().unwrap().parse::<u64>().unwrap()
    } else {
        panic!("Cannot parse extension field element from {:?}", val);
    };
    K::from_u64(n)
}

fn parse_matrix_f(json: &Value) -> Mat<F> {
    let arr = json.as_array().expect("Expected matrix array");
    let rows = arr.len();
    let cols = if rows > 0 { arr[0].as_array().unwrap().len() } else { 0 };
    
    let mut mat = Mat::zero(rows, cols, F::ZERO);
    for (i, row) in arr.iter().enumerate() {
        for (j, val) in row.as_array().unwrap().iter().enumerate() {
            mat[(i, j)] = parse_f(val);
        }
    }
    mat
}

fn parse_vec_f(json: &Value) -> Vec<F> {
    json.as_array()
        .expect("Expected vector array")
        .iter()
        .map(parse_f)
        .collect()
}

fn parse_vec_k(json: &Value) -> Vec<K> {
    json.as_array()
        .expect("Expected vector array")
        .iter()
        .map(parse_k)
        .collect()
}

#[allow(non_snake_case)]
#[test]
fn test_rust_vs_sage_golden_vectors() {
    let json_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests-golden-vectors/neo_ccs_golden_vectors.json"
    );
    
    let json_str = fs::read_to_string(json_path)
        .expect("Failed to read golden vectors JSON");
    
    let data: Value = serde_json::from_str(&json_str)
        .expect("Failed to parse golden vectors JSON");
    
    println!("\n🔬 Running Rust Implementation Against Sage Golden Vectors\n");
    
    // Extract parameters
    let vecs = &data["vectors"];
    let params_json = &vecs["params"];
    let k = params_json["k"].as_u64().unwrap() as usize;
    let b = params_json["b"].as_u64().unwrap() as usize;
    let t = params_json["t"].as_u64().unwrap() as usize;
    let n = params_json["n"].as_u64().unwrap() as usize;
    let m = params_json["m"].as_u64().unwrap() as usize;
    let m_in = params_json["m_in"].as_u64().unwrap() as usize;
    
    println!("Parameters: k={}, b={}, t={}, n={}, m={}, m_in={}", k, b, t, n, m, m_in);
    
    // Setup Rust parameters
    let params = NeoParams::goldilocks_for_circuit(k as u32, D as u32, 2);
    assert_eq!(params.b as usize, b, "Rust b should match Sage b");
    
    // Initialize Ajtai PP for the dimensions we need (d=D=54, m=8)
    println!("  Initializing Ajtai PP for d={}, m={}, kappa={}...", D, m, params.kappa);
    let mut rng = ChaCha20Rng::seed_from_u64(42); // Deterministic for testing
    let pp = setup(&mut rng, D, params.kappa as usize, m).expect("Failed to setup Ajtai PP");
    set_global_pp(pp).expect("Failed to set global Ajtai PP");
    println!("  ✓ Ajtai PP initialized");
    
    println!("\n📥 Step 1: Parsing Sage inputs...");
    
    // Parse CCS structure
    let structure_json = &vecs["structure"];
    let matrices_json = structure_json["M"].as_array().unwrap();
    
    let mut constraint_matrices = Vec::new();
    for mat_json in matrices_json {
        constraint_matrices.push(parse_matrix_f(mat_json));
    }
    
    // Parse polynomial f
    let poly_json = &structure_json["f"];
    let poly_terms_json = poly_json["terms"].as_array().unwrap();
    
    let mut poly_terms = Vec::new();
    for term_json in poly_terms_json {
        let coeff = parse_f(&term_json["coeff"]);
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
    
    // Parse MCS input
    let mcs_json = &vecs["inputs"]["MCS"];
    let x1 = parse_vec_f(&mcs_json["x1"]);
    let w1 = parse_vec_f(&mcs_json["w1"]);
    let z_full: Vec<F> = [x1.clone(), w1.clone()].concat();
    
    // Decompose using Rust's D (not Sage's d=54)
    let z_digits = decomp_b(&z_full, b as u32, D, DecompStyle::Balanced);
    let mut row_major = vec![F::ZERO; D * m];
    for col in 0..m {
        for row in 0..D {
            row_major[row * m + col] = z_digits[col * D + row];
        }
    }
    let Z = Mat::from_row_major(D, m, row_major);
    
    // Use real Ajtai commitment
    let ajtai = AjtaiSModule::from_global_for_dims(D, m)
        .expect("Failed to get Ajtai module");
    let c = ajtai.commit(&Z);
    
    let mcs_instance = McsInstance {
        c,
        x: x1.clone(),
        m_in,
    };
    
    let mcs_witness = McsWitness {
        w: w1.clone(),
        Z,
    };
    
    println!("  ✓ Parsed CCS structure ({} matrices {}×{})", t, n, m);
    println!("  ✓ Parsed MCS input (x={}, w={}, Z={}×{})", 
        mcs_instance.x.len(), mcs_witness.w.len(), mcs_witness.Z.rows(), mcs_witness.Z.cols());
    
    // Print first few values for debugging
    println!("\n  MCS Input Details:");
    println!("    x = [{}, {}, {}, ...]", 
        x1[0].as_canonical_u64(), 
        x1[1].as_canonical_u64(),
        x1[2].as_canonical_u64());
    println!("    w = [{}, {}, {}, ...]", 
        w1[0].as_canonical_u64(), 
        w1[1].as_canonical_u64(),
        w1[2].as_canonical_u64());
    println!("    commitment c: {}×{}", mcs_instance.c.d, mcs_instance.c.kappa);
    
    println!("\n  ⚠️  Note: The CCS constraints from Sage may not be satisfied");
    println!("     because we're using random Sage-generated data.");
    println!("     The sumcheck is expected to fail if constraints don't hold.");
    
    println!("\n⚙️  Step 2: Running Rust fold_ccs_instances()...");
    
    let result = fold_ccs_instances(
        &params,
        &structure,
        &[mcs_instance],
        &[mcs_witness],
    );
    
    match result {
        Ok((rust_outputs, _rust_witnesses, _rust_proof)) => {
            println!("  ✅ Rust folding succeeded!");
            println!("     Rust output: {} ME instances", rust_outputs.len());
            
            // Parse Sage outputs
            let sage_dec = &vecs["Pi_DEC"];
            let sage_dec_outputs = sage_dec["output_ME_list"].as_array().unwrap();
            
            println!("\n📊 Step 3: Comparing outputs...");
            
            // Compare output counts
            assert_eq!(rust_outputs.len(), k, "Rust should output k={} instances", k);
            assert_eq!(sage_dec_outputs.len(), k, "Sage should output k={} instances", k);
            println!("  ✓ Output counts match: {} instances", k);
            
            // Compare each ME instance
            let mut all_match = true;
            for i in 0..k {
                println!("\n  Comparing ME instance {}:", i);
                
                let rust_me = &rust_outputs[i];
                let sage_me = &sage_dec_outputs[i];
                
                // Compare evaluation vectors y
                let sage_y_arrays = sage_me["y"].as_array().unwrap();
                assert_eq!(rust_me.y.len(), sage_y_arrays.len(),
                    "ME[{}]: evaluation count mismatch", i);
                
                println!("    Evaluations (y): {} vectors", rust_me.y.len());
                
                // Compare y vectors element-wise
                for (j, sage_y_json) in sage_y_arrays.iter().enumerate() {
                    let sage_y = parse_vec_k(sage_y_json);
                    let rust_y = &rust_me.y[j];
                    
                    if sage_y.len() != rust_y.len() {
                        println!("      ❌ y[{}]: length mismatch (Sage={}, Rust={})", 
                            j, sage_y.len(), rust_y.len());
                        all_match = false;
                        continue;
                    }
                    
                    let mut y_matches = true;
                    for (idx, (sage_val, rust_val)) in sage_y.iter().zip(rust_y.iter()).enumerate() {
                        if sage_val != rust_val {
                            if y_matches {
                                println!("      ❌ y[{}]: values differ:", j);
                                y_matches = false;
                            }
                            if idx < 3 {  // Show first 3 mismatches
                                println!("         [{}]: Sage={:?}, Rust={:?}", idx, sage_val, rust_val);
                            }
                            all_match = false;
                        }
                    }
                    
                    if y_matches {
                        println!("      ✓ y[{}]: all {} values match", j, rust_y.len());
                    }
                }
                
                // Compare random point r
                let sage_r = parse_vec_k(&sage_me["r"]);
                if sage_r.len() != rust_me.r.len() {
                    println!("    ❌ r: length mismatch (Sage={}, Rust={})", 
                        sage_r.len(), rust_me.r.len());
                    all_match = false;
                } else {
                    let mut r_matches = true;
                    for (idx, (sage_val, rust_val)) in sage_r.iter().zip(rust_me.r.iter()).enumerate() {
                        if sage_val != rust_val {
                            if r_matches {
                                println!("    ❌ r: values differ:");
                                r_matches = false;
                            }
                            if idx < 3 {
                                println!("       [{}]: Sage={:?}, Rust={:?}", idx, sage_val, rust_val);
                            }
                            all_match = false;
                        }
                    }
                    
                    if r_matches {
                        println!("    ✓ r: all {} values match", rust_me.r.len());
                    }
                }
            }
            
            // Compare transcript challenges from Π_CCS
            println!("\n  Comparing transcript challenges (Π_CCS):");
            let sage_transcript = &vecs["Pi_CCS"]["transcript"];
            
            // Note: We can't directly access the transcript from rust_proof
            // as it's not exposed. We'd need to modify the API to extract it.
            println!("    ⚠️  Transcript challenges not directly accessible from proof");
            println!("       (would need to modify fold_ccs_instances to return transcript state)");
            
            // Show what Sage has
            let sage_alpha = parse_vec_k(&sage_transcript["alpha"]);
            let sage_beta = parse_vec_k(&sage_transcript["beta"]);
            let sage_gamma = parse_k(&sage_transcript["gamma"]);
            let sage_r_p = parse_vec_k(&sage_transcript["r_p"]);
            
            println!("    Sage challenges:");
            println!("      α: {} elements", sage_alpha.len());
            println!("      β: {} elements", sage_beta.len());
            println!("      γ: {:?}", sage_gamma);
            println!("      r': {} elements", sage_r_p.len());
            
            if all_match {
                println!("\n✅ TEST PASSED!");
                println!("   All accessible values match between Rust and Sage");
            } else {
                println!("\n⚠️  TEST PARTIALLY FAILED");
                println!("   Some values don't match - see details above");
                panic!("Value mismatch detected");
            }
        }
        Err(e) => {
            panic!("❌ fold_ccs_instances() failed: {:?}", e);
        }
    }
}

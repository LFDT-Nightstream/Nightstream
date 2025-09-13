//! Demonstrates EV with public ρ (recomputed by verifier) and in-circuit fold.
//! Run: cargo run -p neo --example hypernova_public_rho

use neo::F;
use neo::ivc::{Accumulator, ev_full_ccs_public_rho, build_ev_full_witness, rho_from_transcript, create_step_digest};
use neo_ccs::check_ccs_rowwise_zero;
use p3_field::{PrimeCharacteristicRing, PrimeField64};

fn main() {
    println!("🚀 HyperNova IVC with Public ρ - Production Path Demo");
    println!("=====================================================");

    // Simple y of length 2
    let y_prev = vec![F::from_u64(3), F::from_u64(5)];
    let y_step = vec![F::from_u64(7), F::from_u64(11)];

    // Fake previous accumulator (only fields needed for ρ)
    let prev_acc = Accumulator {
        c_z_digest: [0u8; 32],
        c_coords: vec![],
        y_compact: y_prev.clone(),
        step: 42,
    };

    println!("📊 Initial State:");
    println!("  y_prev = {:?}", y_prev.iter().map(|f| f.as_canonical_u64()).collect::<Vec<_>>());
    println!("  y_step = {:?}", y_step.iter().map(|f| f.as_canonical_u64()).collect::<Vec<_>>());
    println!("  step   = {}", prev_acc.step);

    // Build step_data and ρ from unified Poseidon2 transcript
    let step_data = {
        let mut v = Vec::new();
        v.push(F::from_u64(prev_acc.step));
        v.extend_from_slice(&y_prev);
        v
    };
    
    println!("🔄 Transcript Processing:");
    let step_digest = create_step_digest(&step_data);
    let (rho, _dig) = rho_from_transcript(&prev_acc, step_digest);
    println!("  step_digest = {:02x?}...", &step_digest[..8]);
    println!("  ρ = {}", rho.as_canonical_u64());

    // EV CCS and witness
    println!("🔧 Building EV Circuit:");
    let ccs = ev_full_ccs_public_rho(y_prev.len());
    println!("  EV CCS: {} constraints, {} witness cols", ccs.n, ccs.m);
    
    let (witness, y_next) = build_ev_full_witness(rho, &y_prev, &y_step);
    println!("  witness length: {}", witness.len());

    // Public input: [ρ | y_prev | y_next]  
    let mut pub_in = Vec::new();
    pub_in.push(rho);
    pub_in.extend_from_slice(&y_prev);
    pub_in.extend_from_slice(&y_next);

    println!("🧮 Circuit Verification:");
    println!("  public input length: {}", pub_in.len());
    println!("  ρ = {}", pub_in[0].as_canonical_u64());
    println!("  y_prev = {:?}", pub_in[1..1+y_prev.len()].iter().map(|f| f.as_canonical_u64()).collect::<Vec<_>>());
    println!("  y_next = {:?}", pub_in[1+y_prev.len()..].iter().map(|f| f.as_canonical_u64()).collect::<Vec<_>>());

    // Sanity: check CCS locally
    let _ok = check_ccs_rowwise_zero(&ccs, &pub_in, &witness)
        .map_err(|e| format!("CCS check failed: {:?}", e)).unwrap();
    
    println!("✅ Success!");
    println!("  EV(public-ρ) satisfied");
    println!("  y_next = {:?}", y_next.iter().map(|f| f.as_canonical_u64()).collect::<Vec<_>>());
    println!("  Fold equation: y_next[k] = y_prev[k] + ρ * y_step[k] ✓");
    
    // Verify the fold equation manually 
    for k in 0..y_prev.len() {
        let expected = y_prev[k] + rho * y_step[k];
        assert_eq!(y_next[k], expected, "Fold equation failed at index {}", k);
        println!("  y_next[{}] = {} + {} * {} = {} ✓", 
                 k, 
                 y_prev[k].as_canonical_u64(), 
                 rho.as_canonical_u64(), 
                 y_step[k].as_canonical_u64(), 
                 expected.as_canonical_u64());
    }

    println!("\n🎯 HyperNova Public-ρ EV Demo Complete!");
    println!("   - ρ challenge derived from unified Poseidon2 transcript");
    println!("   - Fold enforced in-circuit with public ρ");
    println!("   - Production-ready cryptographic path");
}

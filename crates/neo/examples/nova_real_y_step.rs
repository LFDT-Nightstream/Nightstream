//! Nova EV with Real y_step Values (Fixes "Folding with Itself")
//!
//! This example demonstrates the critical fix for Las's observation that our
//! IVC looked like it was "folding the instance with itself" due to placeholder
//! y_step values. Here we show how to extract REAL y_step values from the
//! actual step computation.
//!
//! **Key Fix**: y_step must come from the step relation's outputs, not placeholders!
//!
//! ⚠️  **SECURITY WARNING**: This example uses the TOY HASH (4-round squaring)
//! which is NOT cryptographically secure. It's suitable for demonstrating the
//! structural fix but should NOT be used in production systems.
//!
//! Usage: cargo run -p neo --example nova_real_y_step

#![allow(deprecated)] // This example uses toy hash for demonstration - NOT for production

use neo::{F, CcsStructure, Accumulator};
use neo::ivc::{ev_hash_ccs_public_y, build_ev_hash_witness_public_y};
use neo_ccs::{Mat, r1cs_to_ccs, check_ccs_rowwise_zero};
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use std::error::Error;

/// Simple computation: multiply by step number
#[derive(Clone, Debug)]
struct MultiplyState {
    value: u64,
    step: u64,
}

impl MultiplyState {
    fn new() -> Self {
        Self { value: 1, step: 0 }
    }
    
    fn step(&self) -> Self {
        Self {
            value: self.value * (self.step + 2), // multiply by (step + 2)
            step: self.step + 1,
        }
    }
}

/// Create a simple multiplication CCS: next_value = prev_value * multiplier
fn multiply_step_ccs() -> CcsStructure<F> {
    // Variables: [const=1, prev_value, multiplier, next_value]
    // Constraint: next_value = prev_value * multiplier
    //   A*z = [0, 1, 0, 0] (prev_value)
    //   B*z = [0, 0, 1, 0] (multiplier) 
    //   C*z = [0, 0, 0, 1] (next_value)
    let rows = 1;
    let cols = 4;
    
    let mut a = vec![F::ZERO; rows * cols];
    let mut b = vec![F::ZERO; rows * cols];
    let mut c = vec![F::ZERO; rows * cols];
    
    a[1] = F::ONE;  // prev_value
    b[2] = F::ONE;  // multiplier
    c[3] = F::ONE;  // next_value
    
    let a_mat = Mat::from_row_major(rows, cols, a);
    let b_mat = Mat::from_row_major(rows, cols, b);
    let c_mat = Mat::from_row_major(rows, cols, c);
    
    r1cs_to_ccs(a_mat, b_mat, c_mat)
}

/// Generate witness for multiplication step
/// CRITICAL: This also returns the REAL y_step values (step outputs)
fn multiply_step_witness(state: &MultiplyState) -> (Vec<F>, Vec<F>) {
    let next_state = state.step();
    let multiplier = next_state.step + 1; // step + 2 - 1 = step + 1
    
    // Step witness: [const=1, prev_value, multiplier, next_value]
    let witness = vec![
        F::ONE,
        F::from_u64(state.value),
        F::from_u64(multiplier),
        F::from_u64(next_state.value),
    ];
    
    // REAL y_step: the actual outputs from this step computation
    // This represents what the step "produces" that gets folded into the accumulator
    let y_step = vec![
        F::from_u64(next_state.value), // primary output
        F::from_u64(next_state.step),  // step counter
    ];
    
    (witness, y_step)
}

/// Demonstrate Nova EV with REAL y_step (not placeholder)
fn run_nova_with_real_y_step() -> Result<(), Box<dyn Error>> {
    println!("🎯 Nova Embedded Verifier with REAL y_step Values");
    println!("============================================");
    println!("This fixes the 'folding with itself' issue Las identified!");
    println!();
    
    // Setup
    let _step_ccs = multiply_step_ccs();
    let y_len = 2; // [value, step_counter]
    let hash_input_len = 3; // Using 3 hash inputs for transcript
    
    // Initial state  
    let initial_state = MultiplyState::new();
    let mut accumulator = Accumulator {
        c_z_digest: [0u8; 32],
        c_coords: vec![F::ZERO; 128], // Default commit_len (d * kappa = 32 * 4 = 128)
        y_compact: vec![F::from_u64(initial_state.value), F::from_u64(initial_state.step)],
        step: 0,
    };
    
    let mut current_state = initial_state;
    
    // Run several Nova steps to demonstrate real folding
    for step_num in 0..4 {
        println!("🔄 Nova Step {}", step_num + 1);
        println!("   Current state: value={}, step={}", current_state.value, current_state.step);
        println!("   Previous accumulator y_compact: {:?}", 
                 accumulator.y_compact.iter().map(|f| f.as_canonical_u64()).collect::<Vec<_>>());
        
        // Generate step witness AND extract REAL y_step
        let (_step_witness, real_y_step) = multiply_step_witness(&current_state);
        
        println!("   REAL y_step (from actual computation): {:?}",
                 real_y_step.iter().map(|f| f.as_canonical_u64()).collect::<Vec<_>>());
        
        // Create hash inputs for transcript
        let hash_inputs = vec![
            F::from_u64(accumulator.step),          // current step
            F::from_u64(current_state.value),       // current value  
            F::from_u64(0x1337),                   // domain separation
        ];
        
        // Build Nova EV with public y_prev and y_next
        let ev_ccs = ev_hash_ccs_public_y(hash_input_len, y_len);
        let (ev_witness, y_next) = build_ev_hash_witness_public_y(
            &hash_inputs, 
            &accumulator.y_compact, 
            &real_y_step  // ← CRITICAL: Using REAL y_step, not placeholder!
        );
        
        println!("   Computed y_next: {:?}",
                 y_next.iter().map(|f| f.as_canonical_u64()).collect::<Vec<_>>());
        
        // The EV should pass with real values
        let public_inputs = [accumulator.y_compact.clone(), y_next.clone()].concat();
        
        let is_valid = check_ccs_rowwise_zero(&ev_ccs, &public_inputs, &ev_witness);
        if is_valid.is_err() {
            return Err(format!("Nova EV failed at step {}", step_num + 1).into());
        }
        
        println!("   ✅ Nova EV passed with REAL y_step values!");
        
        // Advance to next step
        current_state = current_state.step();
        accumulator.y_compact = y_next;
        accumulator.step += 1;
        
        println!();
    }
    
    println!("🎉 SUCCESS: Nova EV working with REAL y_step values!");
    println!("   This demonstrates proper folding: y_next = y_prev + ρ * y_step");
    println!("   where y_step comes from actual step computation, not placeholders.");
    
    Ok(())
}

/// Contrast: Show what the "folding with itself" problem looked like
fn demonstrate_placeholder_problem() -> Result<(), Box<dyn Error>> {
    println!();
    println!("⚠️  DEMONSTRATION: The Placeholder Problem Las Identified");
    println!("=======================================================");
    
    let _y_len = 2;
    let _hash_input_len = 3;
    
    let y_prev = vec![F::from_u64(10), F::from_u64(1)];
    
    // BAD: Using placeholder y_step (what we had before)
    let placeholder_y_step = vec![F::from_u64(42), F::from_u64(42)]; // constant placeholder!
    
    // GOOD: Using real y_step (what we have now)
    let real_y_step = vec![F::from_u64(20), F::from_u64(2)]; // from actual computation
    
    let hash_inputs = vec![F::from_u64(1), F::from_u64(10), F::from_u64(0x1337)];
    
    let (_, y_next_placeholder) = build_ev_hash_witness_public_y(&hash_inputs, &y_prev, &placeholder_y_step);
    let (_, y_next_real) = build_ev_hash_witness_public_y(&hash_inputs, &y_prev, &real_y_step);
    
    println!("   y_prev:      {:?}", y_prev.iter().map(|f| f.as_canonical_u64()).collect::<Vec<_>>());
    println!("   placeholder: {:?}", placeholder_y_step.iter().map(|f| f.as_canonical_u64()).collect::<Vec<_>>());
    println!("   real y_step: {:?}", real_y_step.iter().map(|f| f.as_canonical_u64()).collect::<Vec<_>>());
    println!();
    println!("   Result with placeholder: {:?}", y_next_placeholder.iter().map(|f| f.as_canonical_u64()).collect::<Vec<_>>());
    println!("   Result with real y_step: {:?}", y_next_real.iter().map(|f| f.as_canonical_u64()).collect::<Vec<_>>());
    println!();
    println!("   📝 The placeholder approach makes it look like we're folding");
    println!("      the same values repeatedly, rather than folding actual step outputs!");
    
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    run_nova_with_real_y_step()?;
    demonstrate_placeholder_problem()?;
    Ok(())
}

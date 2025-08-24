// neo-bench/src/main.rs
use std::time::Duration;

#[derive(Debug, Clone)]
#[allow(dead_code)]
struct BenchRow {
    impl_name: &'static str, // "nova" or "neo"
    k: u32,
    steps: usize,
    build_ms: f64,
    prove_ms: f64,
    verify_ms: f64,
    total_ms: f64,
    ok: bool,
}

fn ms(d: Duration) -> f64 { d.as_secs_f64() * 1000.0 }

fn print_summary_table(mut rows: Vec<BenchRow>) {
    if rows.is_empty() {
        println!("\n=== Summary ===\n(no rows)");
        return;
    }

    rows.sort_by_key(|r| (r.steps, r.impl_name));

    println!("\n=== Summary: Fibonacci (manual) ===");
    println!(
        "{:<6} {:>8} {:>12} {:>12} {:>12} {:>12} {:>6}",
        "impl", "steps", "build(ms)", "prove(ms)", "verify(ms)", "total(ms)", "ok"
    );
    
    let mut prev_steps: Option<usize> = None;
    for r in &rows {
        // Add separator when steps change
        if let Some(last_steps) = prev_steps {
            if r.steps != last_steps {
                println!("{:-<70}", ""); // Print separator line
            }
        }
        
        println!(
            "{:<6} {:>8} {:>12.3} {:>12.3} {:>12.3} {:>12.3} {:>6}",
            r.impl_name, r.steps, r.build_ms, r.prove_ms, r.verify_ms, r.total_ms,
            if r.ok { "yes" } else { "no" }
        );
        
        prev_steps = Some(r.steps);
    }

    // Optional: speedup table (Neo/Nova prove-time ratio for steps where we have both)
    use std::collections::BTreeMap;
    let mut by_steps: BTreeMap<usize, (Option<f64>, Option<f64>)> = BTreeMap::new();
    for r in &rows {
        let entry = by_steps.entry(r.steps).or_insert((None, None));
        match r.impl_name {
            "nova" => entry.0 = Some(r.prove_ms),
            "neo"  => entry.1 = Some(r.prove_ms),
            _ => {}
        }
    }

    let any_pairs = by_steps.values().any(|(a,b)| a.is_some() && b.is_some());
    if any_pairs {
        println!("\nNeo/Nova prove-time ratio (lower is better):");
        println!("{:>8} {:>12}", "steps", "neo/nova");
        for (steps, (nova, neo)) in by_steps {
            if let (Some(nv), Some(ev)) = (nova, neo) {
                println!("{:>8} {:>12.3}", steps, ev / nv);
            }
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut run_nova_fibo = false;
    let mut run_neo_fibo  = false;

    let mut min_pow: u32 = 3;  // 2^3 = 8 steps (fast for development)
    let mut max_pow: u32 = 6;  // 2^6 = 64 steps

    // Fibonacci seeds (z0 = [a0, a1])
    let mut a0_u64: u64 = 0;
    let mut a1_u64: u64 = 1;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--nova-fibo" => run_nova_fibo = true,
            "--neo-fibo"  => run_neo_fibo  = true,
            "--min" if i + 1 < args.len() => { min_pow = args[i+1].parse().unwrap_or(min_pow); i += 1; }
            "--max" if i + 1 < args.len() => { max_pow = args[i+1].parse().unwrap_or(max_pow); i += 1; }
            "--a0" if i + 1 < args.len() => { a0_u64 = args[i+1].parse().unwrap_or(a0_u64); i += 1; }
            "--a1" if i + 1 < args.len() => { a1_u64 = args[i+1].parse().unwrap_or(a1_u64); i += 1; }
            "--help" | "-h" => {
                eprintln!("Usage: cargo run -p neo-bench -- [--nova-fibo] [--neo-fibo]");
                eprintln!("       [--min K] [--max K] [--a0 X] [--a1 Y]");
                eprintln!();
                eprintln!("  --nova-fibo       Nova recursive Fibonacci (T=2^K steps).");
                eprintln!("  --neo-fibo        Neo CCS Fibonacci (same T steps in one CCS).");
                eprintln!("  --a0, --a1        Fibonacci seeds (default 0, 1).");
                eprintln!("  --min/--max       K range (T = 2^K).");
                return;
            }
            _ => {} // Ignore unknown args
        }
        i += 1;
    }

    let mut rows: Vec<BenchRow> = Vec::new();

    if run_nova_fibo {
        #[cfg(feature = "with-nova")]
        {
            rows.extend(nova_fibo::manual_bench(min_pow, max_pow, a0_u64, a1_u64));
        }
        #[cfg(not(feature = "with-nova"))]
        {
            eprintln!("Nova Fibonacci bench requested, but feature `with-nova` is not enabled.");
            eprintln!("Re-run with: cargo run -p neo-bench --features with-nova -- --nova-fibo");
        }
    }

    if run_neo_fibo {
        rows.extend(neo_fibo_ccs::manual_bench(min_pow, max_pow, a0_u64, a1_u64));
    }

    if !run_nova_fibo && !run_neo_fibo {
        eprintln!("Nothing to run. Try one of:");
        eprintln!("  cargo run -p neo-bench -- --nova-fibo --neo-fibo");
        eprintln!("  cargo run -p neo-bench -- --nova-fibo");
        eprintln!("  cargo run -p neo-bench -- --neo-fibo");
        return;
    }

    print_summary_table(rows);
}



// -----------------------------
// NEW: Nova Fibonacci (recursive)
// -----------------------------
#[cfg(feature = "with-nova")]
mod nova_fibo {
    use core::marker::PhantomData;
    use ff::PrimeField;


    use nova_snark::{
        frontend::{
            num::AllocatedNum,
            {ConstraintSystem, SynthesisError, Assignment},
        },
        PublicParams, RecursiveSNARK,
        provider::{Bn256EngineKZG, GrumpkinEngine},
        traits::{circuit::StepCircuit, snark::default_ck_hint, Engine},
    };

    type E1 = Bn256EngineKZG;
    type E2 = GrumpkinEngine;

    #[derive(Clone, Debug)]
    struct FibCircuit<Scalar: PrimeField> { _p: PhantomData<Scalar> }

    impl<Scalar: PrimeField> FibCircuit<Scalar> {
        pub fn new() -> Self { Self { _p: PhantomData } }
    }

    impl<Scalar: PrimeField> StepCircuit<Scalar> for FibCircuit<Scalar> {
        fn arity(&self) -> usize { 2 } // z = [a_i, a_{i+1}]

        fn synthesize<CS: ConstraintSystem<Scalar>>(
            &self,
            cs: &mut CS,
            z: &[AllocatedNum<Scalar>],
        ) -> Result<Vec<AllocatedNum<Scalar>>, SynthesisError> {
            assert!(z.len() == 2, "expected arity-2 state");
            // next = a_i + a_{i+1}
            let next = AllocatedNum::alloc(cs.namespace(|| "fib_next"), || {
                let a0 = *z[0].get_value().get()?;
                let a1 = *z[1].get_value().get()?;
                Ok(a0 + a1)
            })?;

            // Enforce (a_i + a_{i+1}) * 1 = next
            cs.enforce(
                || "fib constraint",
                |lc| lc + z[0].get_variable() + z[1].get_variable(),
                |lc| lc + CS::one(),
                |lc| lc + next.get_variable(),
            );

            // Output state z' = [a_{i+1}, next]
            Ok(vec![z[1].clone(), next])
        }
    }

    // For Nova 0.39, we need a trivial secondary circuit
    #[derive(Clone, Debug)]
    struct TrivialCircuit<Scalar: PrimeField> { _p: PhantomData<Scalar> }
    
    impl<Scalar: PrimeField> TrivialCircuit<Scalar> {
        pub fn new() -> Self { Self { _p: PhantomData } }
    }
    
    impl<Scalar: PrimeField> StepCircuit<Scalar> for TrivialCircuit<Scalar> {
        fn arity(&self) -> usize { 1 }
        
        fn synthesize<CS: ConstraintSystem<Scalar>>(
            &self,
            _cs: &mut CS,
            z: &[AllocatedNum<Scalar>],
        ) -> Result<Vec<AllocatedNum<Scalar>>, SynthesisError> {
            Ok(z.to_vec()) // Identity circuit
        }
    }

    type C1 = FibCircuit<<E1 as Engine>::Scalar>;
    type C2 = TrivialCircuit<<E2 as Engine>::Scalar>;

    pub fn manual_bench(min_pow: u32, max_pow: u32, a0_u64: u64, a1_u64: u64) -> Vec<crate::BenchRow> {
        use std::time::Instant;
        let mut out = Vec::new();

        println!("=== Manual Nova Fibonacci Benchmark ===");
        for k in min_pow..=max_pow {
            let steps = 1usize << k;
            println!("Nova Fibonacci: {} steps (2^{})", steps, k);

            let circuit_primary   = FibCircuit::<<E1 as Engine>::Scalar>::new();
            let circuit_secondary = TrivialCircuit::<<E2 as Engine>::Scalar>::new();

            let setup_start = Instant::now();
            let pp = PublicParams::<E1, E2, C1, C2>::setup(
                &circuit_primary, &circuit_secondary, &*default_ck_hint(), &*default_ck_hint()
            ).expect("PP setup");
            let setup_time = setup_start.elapsed();

            let z0_primary = vec![
                <E1 as Engine>::Scalar::from(a0_u64),
                <E1 as Engine>::Scalar::from(a1_u64),
            ];
            let z0_secondary = vec![<E2 as Engine>::Scalar::from(0u64)];

            let prove_start = Instant::now();
            let mut rs = RecursiveSNARK::new(
                &pp, &circuit_primary, &circuit_secondary, &z0_primary, &z0_secondary
            ).unwrap();
            for _ in 0..steps { 
                rs.prove_step(&pp, &circuit_primary, &circuit_secondary).unwrap(); 
            }
            let prove_time = prove_start.elapsed();

            let verify_start = Instant::now();
            let res = rs.verify(&pp, steps, &z0_primary, &z0_secondary);
            let verify_time = verify_start.elapsed();
            let ok = res.is_ok();

            println!("  Setup:  {:?}", setup_time);
            println!("  Prove:  {:?}", prove_time);
            println!("  Verify: {:?} (result: {:?})", verify_time, ok);
            println!("  Total:  {:?}", setup_time + prove_time + verify_time);
            println!();

            out.push(crate::BenchRow {
                impl_name: "nova",
                k, steps,
                build_ms: crate::ms(setup_time),
                prove_ms: crate::ms(prove_time),
                verify_ms: crate::ms(verify_time),
                total_ms: crate::ms(setup_time + prove_time + verify_time),
                ok,
            });
        }
        out
    }
}

// -----------------------------
// Neo Fibonacci (CCS)
// -----------------------------
mod neo_fibo_ccs {
    use neo_fields::{F, ExtF};
    use neo_ccs::{CcsStructure, CcsInstance, CcsWitness, mv_poly};
    use p3_matrix::dense::RowMajorMatrix;
    use p3_field::PrimeCharacteristicRing;
    use neo_orchestrator;

    fn build_fibo_ccs(steps: usize, a0: F, a1: F) -> (CcsStructure, CcsInstance, CcsWitness) {
        let n_constraints = steps;
        let n_vars = steps + 3; // 1 constant + (steps+2) Fibonacci numbers
        
        println!("Building CCS: {} constraints, {} variables", n_constraints, n_vars);
        
        // Variable indexing: 0=constant(1), 1=a0, 2=a1, 3=a2, ..., steps+2=a_{steps+1}
        let idx_one = 0;
        let idx_a = |i: usize| 1 + i;
        
        // Initialize matrices
        let mut a_mat = vec![F::ZERO; n_constraints * n_vars];
        let mut b_mat = vec![F::ZERO; n_constraints * n_vars];
        let mut c_mat = vec![F::ZERO; n_constraints * n_vars];
        
        // Build constraints: (a_i + a_{i+1}) * 1 - a_{i+2} = 0 for i = 0..steps-1
        for i in 0..steps {
            let r = i; // constraint index
            
            // A selects (a_i + a_{i+1})
            a_mat[r * n_vars + idx_a(i)]     = F::ONE;
            a_mat[r * n_vars + idx_a(i + 1)] = F::ONE;

            // B selects 1
            b_mat[r * n_vars + idx_one]      = F::ONE;

            // C selects a_{i+2}
            c_mat[r * n_vars + idx_a(i + 2)] = F::ONE;
        }
        
        let mats = vec![
            RowMajorMatrix::new(a_mat, n_vars),
            RowMajorMatrix::new(b_mat, n_vars),
            RowMajorMatrix::new(c_mat, n_vars),
        ];
        
        // f(A,B,C) = A - C, arity = 1 (multilinear, since B=1 is constant)
        // This represents: (a_i + a_{i+1}) * 1 - a_{i+2} = (a_i + a_{i+1}) - a_{i+2} = 0
        let f = mv_poly(|inputs: &[ExtF]| {
            debug_assert!(inputs.len() == 3);
            inputs[0] - inputs[2] // A - C, since B is always 1
        }, 1);
        
        let structure = CcsStructure::new(mats, f);
        
        // Build witness (ExtF required)
        let mut z = vec![ExtF::from(F::ZERO); n_vars];
        z[idx_one] = ExtF::from(F::ONE);
        z[idx_a(0)] = ExtF::from(a0);
        z[idx_a(1)] = ExtF::from(a1);
        
        // Compute Fibonacci sequence
        for i in 2..=steps+1 {
            z[idx_a(i)] = z[idx_a(i-2)] + z[idx_a(i-1)];
        }
        
        let witness = CcsWitness { z };
        let instance = CcsInstance { 
            commitment: vec![], 
            public_input: vec![], 
            u: F::ZERO, 
            e: F::ONE 
        };
        
        (structure, instance, witness)
    }

    pub fn manual_bench(min_pow: u32, max_pow: u32, a0_u64: u64, a1_u64: u64) -> Vec<crate::BenchRow> {
        use std::time::Instant;
        let mut out = Vec::new();

        println!("=== Manual Neo Fibonacci Benchmark ===");
        for k in min_pow..=max_pow {
            let steps = 1usize << k;
            println!("Neo Fibonacci: {} steps (2^{})", steps, k);

            let a0 = F::from_u64(a0_u64);
            let a1 = F::from_u64(a1_u64);

            let build_start = Instant::now();
            let (structure, instance, witness) = build_fibo_ccs(steps, a0, a1);
            let build_time = build_start.elapsed();

            let prove_start = Instant::now();
            let proof_result = neo_orchestrator::prove(&structure, &instance, &witness);
            let prove_time = prove_start.elapsed();

            let verify_start = Instant::now();
            let ok = match &proof_result {
                Ok((proof, _metrics)) => neo_orchestrator::verify(&structure, proof),
                Err(_) => false,
            };
            let verify_time = verify_start.elapsed();

            println!("  Build:  {:?}", build_time);
            println!("  Prove:  {:?}", prove_time);
            println!("  Verify: {:?} (result: {})", verify_time, ok);
            println!("  Total:  {:?}", build_time + prove_time + verify_time);
            println!();

            out.push(crate::BenchRow {
                impl_name: "neo",
                k, steps,
                build_ms: crate::ms(build_time),
                prove_ms: crate::ms(prove_time),
                verify_ms: crate::ms(verify_time),
                total_ms: crate::ms(build_time + prove_time + verify_time),
                ok,
            });
        }
        
        out
    }
}
use neo_ccs::{check_relaxed_satisfiability, mv_poly, CcsInstance, CcsStructure, CcsWitness};
use neo_commit::{AjtaiCommitter, SECURE_PARAMS};
use neo_fields::{embed_base_to_ext, from_base, ExtF, F};
use neo_fold::FoldState;
use p3_field::PrimeCharacteristicRing;
use p3_matrix::dense::RowMajorMatrix;
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    let _rng = rand::rng();

    // Simple R1CS-like CCS structure for X0 * X1 = X2
    let m = 3;
    let a_data = vec![F::ONE, F::ZERO, F::ZERO, F::ONE, F::ZERO, F::ZERO];
    let b_data = vec![F::ZERO, F::ONE, F::ZERO, F::ZERO, F::ONE, F::ZERO];
    let c_data = vec![F::ZERO, F::ZERO, F::ONE, F::ZERO, F::ZERO, F::ONE];
    let mats = vec![
        RowMajorMatrix::new(a_data, m),
        RowMajorMatrix::new(b_data, m),
        RowMajorMatrix::new(c_data, m),
    ];
    let f = mv_poly(
        |inputs: &[ExtF]| {
            if inputs.len() == 3 {
                inputs[0] * inputs[1] - inputs[2]
            } else {
                ExtF::ZERO
            }
        },
        2,
    );
    let structure = CcsStructure::new(mats, f);

    // Committer with parameters close to the Neo paper
    let committer = AjtaiCommitter::setup(SECURE_PARAMS);

    // Build 10 identical satisfying instances
    let mut instances = Vec::new();
    let mut witnesses = Vec::new();
    for i in 0..10 {
        let val = F::from_u64(i as u64 + 2);
        let z_base = vec![F::ONE, val, val];
        let z = z_base.iter().copied().map(from_base).collect();
        let witness = CcsWitness { z };
        let z_mat = neo_decomp::decomp_b(&z_base, SECURE_PARAMS.b, SECURE_PARAMS.d);
        let w = AjtaiCommitter::pack_decomp(&z_mat, &SECURE_PARAMS);
        let mut t = Vec::new();
        let (commit, _, _, _) = committer
            .commit(&w, &mut t)
            .map_err(|e| e.to_string())?;
        let instance = CcsInstance {
            commitment: commit,
            public_input: vec![],
            u: F::ZERO,
            e: F::ONE,
        };
        instances.push(instance);
        witnesses.push(witness);
    }

    // Start with the first instance/witness pair
    let mut accumulated_instance = instances[0].clone();
    let mut accumulated_witness = witnesses[0].clone();

    // Fold remaining nine pairs into the accumulator
    for i in 1..10 {
        let next_instance = instances[i].clone();
        let next_witness = witnesses[i].clone();

        let mut fold_state = FoldState::new(structure.clone());
        let proof = fold_state.generate_proof(
            (accumulated_instance.clone(), accumulated_witness.clone()),
            (next_instance.clone(), next_witness.clone()),
            &committer,
        );

        // Extract rho from the fold state
        let rho = fold_state
            .rhos
            .last()
            .copied()
            .unwrap_or(F::ONE);
        let rho_ext = embed_base_to_ext(rho);

        // Combine witnesses directly (no blinding in demo)
        let new_z: Vec<ExtF> = accumulated_witness
            .z
            .iter()
            .zip(&next_witness.z)
            .map(|(&z1, &z2)| z1 + rho_ext * z2)
            .collect();
        accumulated_witness = CcsWitness { z: new_z };

        // Update relaxation scalars
        let new_u = accumulated_instance.u + rho * next_instance.u;
        let new_e = accumulated_instance.e + rho * next_instance.e;

        // Use folded commitment from last eval instance
        accumulated_instance = CcsInstance {
            commitment: fold_state.eval_instances.last().unwrap().commitment.clone(),
            public_input: vec![],
            u: new_u,
            e: new_e,
        };

        // Verify proof for this fold
        if !fold_state.verify(&proof.transcript, &committer) {
            return Err(format!("verification failed at fold {}", i).into());
        }
    }

    // Final relaxed satisfiability check
    if check_relaxed_satisfiability(
        &structure,
        &accumulated_instance,
        &accumulated_witness,
        accumulated_instance.u,
        accumulated_instance.e,
    ) {
        println!("IVC chaining successful: Final instance (after 10 folds) satisfies relaxed CCS.");
        Ok(())
    } else {
        Err("IVC chaining failed".into())
    }
}

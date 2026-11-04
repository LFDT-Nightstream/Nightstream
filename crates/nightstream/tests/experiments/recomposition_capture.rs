//! Capture actual lifecycle witnesses for isolated, reproducible linear-map experiments.
use super::*;
use p3_field::PrimeCharacteristicRing;
use serde::Deserialize;
use std::{
    fs,
    io::{Read, Write},
    path::PathBuf,
    time::Instant,
};

#[derive(Deserialize)]
struct Request {
    directory: PathBuf,
}

#[test]
#[ignore = "three-step lifecycle capture; run with the experiment's 300-second process-group timeout"]
fn capture_recomposition_workload() {
    let mut request = String::new();
    std::io::stdin().read_to_string(&mut request).unwrap();
    let request: Request = serde_json::from_str(&request).unwrap();
    fs::create_dir_all(&request.directory).unwrap();
    let circuit = Circuit::compile(crate::application::poseidon2_hash_chain_v1().unwrap()).unwrap();
    circuit
        .compiled
        .package
        .write_prepared(fs::File::create(request.directory.join("package.bin")).unwrap())
        .unwrap();
    let prover = circuit.prover(Engine::Optimized, 114).unwrap();
    let verifier = circuit.verifier(Engine::Optimized, 114).unwrap();
    let initial = [202, 203, 204, 205].map(F::from_u64);
    let message = [7, 11, 13, 17].map(F::from_u64);
    let start = Instant::now();
    let mut proof = prover.prove(initial, &message).unwrap();
    for step in 1..=3 {
        let (running, fresh) = proof.active_parts().unwrap();
        fs::write(
            request.directory.join(format!("claims-{step}.json")),
            serde_json::to_vec(&(&fresh.claim, &running.claims)).unwrap(),
        )
        .unwrap();
        let mut file =
            std::io::BufWriter::new(fs::File::create(request.directory.join(format!("witnesses-{step}.bin"))).unwrap());
        file.write_all(b"NSRECOMP1").unwrap();
        file.write_all(&((running.witnesses.len() + 1) as u64).to_le_bytes())
            .unwrap();
        let mut nonzero = 0;
        for witness in std::iter::once(&fresh.witness.Z).chain(&running.witnesses) {
            assert_eq!(witness.rows(), neo_math::D);
            file.write_all(&(witness.cols() as u64).to_le_bytes())
                .unwrap();
            if witness
                .virtual_constant_value()
                .is_some_and(|&v| v == F::ZERO)
            {
                file.write_all(&[0]).unwrap();
            } else {
                let (positive, negative) = witness
                    .packed_signed_unit_column_masks()
                    .expect("packed lifecycle witness");
                file.write_all(&[1]).unwrap();
                for words in [positive, negative] {
                    for value in words {
                        file.write_all(&value.to_le_bytes()).unwrap();
                    }
                }
                nonzero += usize::from(positive.iter().zip(negative).any(|(&p, &n)| p | n != 0));
            }
        }
        file.flush().unwrap();
        println!(
            "EXPERIMENT {}",
            serde_json::json!({"event":"capture","step":step,"nonzero_sources":nonzero,
            "elapsed_seconds":start.elapsed().as_secs_f64(),"circuit_identity":circuit.identity()})
        );
        if step != 3 {
            proof = prover.extend(&proof, &message).unwrap();
        }
    }
    verifier.verify(proof.state(), &proof).unwrap();
    println!(
        "EXPERIMENT {}",
        serde_json::json!({"event":"capture_verified","seconds":start.elapsed().as_secs_f64()})
    );
}

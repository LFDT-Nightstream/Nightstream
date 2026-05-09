use super::*;

pub fn setup_spartan2_public_target_shell(
    shape: &Spartan2DeciderShape,
) -> Result<(Spartan2PublicTargetShellProverKey, Spartan2PublicTargetShellVerifierKey), Spartan2PublicTargetShellError>
{
    Spartan2PublicTargetShellSnark::setup(Spartan2PublicTargetShellCircuit::from_shape(shape))
        .map_err(|err| Spartan2PublicTargetShellError::Setup(err.to_string()))
}

pub fn prove_spartan2_public_target_shell_with_perf(
    pk: &Spartan2PublicTargetShellProverKey,
    target: &Spartan2DeciderTarget,
) -> Result<(Spartan2PublicTargetShellProof, Spartan2PublicTargetShellProvePerf), Spartan2PublicTargetShellError> {
    let total_started = std::time::Instant::now();
    validate_spartan2_decider_target_surface(target)
        .map_err(|err| Spartan2PublicTargetShellError::Prove(err.to_string()))?;
    let circuit = Spartan2PublicTargetShellCircuit::from_target(target);
    let started = std::time::Instant::now();
    let prep = Spartan2PublicTargetShellSnark::prep_prove(pk, circuit.clone(), true)
        .map_err(|err| Spartan2PublicTargetShellError::Prepare(err.to_string()))?;
    let prep_ms = started.elapsed().as_secs_f64() * 1_000.0;
    let started = std::time::Instant::now();
    let (proof, snark_perf) = Spartan2PublicTargetShellSnark::prove_with_perf(pk, circuit, &prep, true)
        .map_err(|err| Spartan2PublicTargetShellError::Prove(err.to_string()))?;
    let mut snark_perf = snark_perf;
    snark_perf.total_ms = started.elapsed().as_secs_f64() * 1_000.0;
    let started = std::time::Instant::now();
    let snark_data =
        bincode::serialize(&proof).map_err(|err| Spartan2PublicTargetShellError::Encode(err.to_string()))?;
    let encode_ms = started.elapsed().as_secs_f64() * 1_000.0;
    Ok((
        Spartan2PublicTargetShellProof { snark_data },
        Spartan2PublicTargetShellProvePerf {
            prep_ms,
            snark_perf,
            encode_ms,
            total_ms: total_started.elapsed().as_secs_f64() * 1_000.0,
        },
    ))
}

pub fn verify_spartan2_public_target_shell(
    vk: &Spartan2PublicTargetShellVerifierKey,
    target: &Spartan2DeciderTarget,
    proof: &Spartan2PublicTargetShellProof,
) -> Result<(), Spartan2PublicTargetShellError> {
    validate_spartan2_decider_target_surface(target)
        .map_err(|err| Spartan2PublicTargetShellError::Verify(err.to_string()))?;
    let proof: Spartan2PublicTargetShellSnark = bincode::deserialize(&proof.snark_data)
        .map_err(|err| Spartan2PublicTargetShellError::Decode(err.to_string()))?;
    let public_values = proof
        .verify(vk)
        .map_err(|err| Spartan2PublicTargetShellError::Verify(err.to_string()))?
        .into_iter()
        .map(|value| F::from_u64(value.to_canonical_u64()))
        .collect::<Vec<_>>();
    if public_values != target.public_io() {
        return Err(Spartan2PublicTargetShellError::PublicIoMismatch);
    }
    Ok(())
}

#[derive(Clone, Debug)]
struct Spartan2PublicTargetShellCircuit {
    public_values: Vec<SpartanF>,
}

impl Spartan2PublicTargetShellCircuit {
    fn from_shape(shape: &Spartan2DeciderShape) -> Self {
        Self {
            public_values: vec![SpartanF::from_canonical_u64(0); shape.public_io_len()],
        }
    }

    fn from_target(target: &Spartan2DeciderTarget) -> Self {
        Self {
            public_values: target
                .public_io()
                .into_iter()
                .map(|value| SpartanF::from_canonical_u64(value.as_canonical_u64()))
                .collect(),
        }
    }
}

impl SpartanCircuit<Spartan2PublicTargetShellEngine> for Spartan2PublicTargetShellCircuit {
    fn public_values(&self) -> Result<Vec<SpartanF>, SynthesisError> {
        Ok(self.public_values.clone())
    }

    fn shared<CS: ConstraintSystem<SpartanF>>(
        &self,
        _: &mut CS,
    ) -> Result<Vec<AllocatedNum<SpartanF>>, SynthesisError> {
        Ok(Vec::new())
    }

    fn precommitted<CS: ConstraintSystem<SpartanF>>(
        &self,
        _: &mut CS,
        _: &[AllocatedNum<SpartanF>],
    ) -> Result<Vec<AllocatedNum<SpartanF>>, SynthesisError> {
        Ok(Vec::new())
    }

    fn num_challenges(&self) -> usize {
        0
    }

    fn synthesize<CS: ConstraintSystem<SpartanF>>(
        &self,
        cs: &mut CS,
        _: &[AllocatedNum<SpartanF>],
        _: &[AllocatedNum<SpartanF>],
        _: Option<&[SpartanF]>,
    ) -> Result<(), SynthesisError> {
        for (idx, value) in self.public_values.iter().copied().enumerate() {
            let witness = AllocatedNum::alloc(cs.namespace(|| format!("public_target_witness_{idx}")), || Ok(value))?;
            let public =
                AllocatedNum::alloc_input(cs.namespace(|| format!("public_target_input_{idx}")), || Ok(value))?;
            cs.enforce(
                || format!("public_target_match_{idx}"),
                |lc| lc + witness.get_variable(),
                |lc| lc + CS::one(),
                |lc| lc + public.get_variable(),
            );
        }
        Ok(())
    }
}

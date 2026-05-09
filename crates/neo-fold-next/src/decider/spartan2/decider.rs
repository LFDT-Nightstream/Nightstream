use super::*;

pub fn setup_spartan2_decider(
    shape: &Spartan2DeciderShape,
) -> Result<(Spartan2DeciderProverKey, Spartan2DeciderVerifierKey), Spartan2DeciderError> {
    let (pk, vk) = setup_spartan2_backend_binding_shell(shape)?;
    Ok((
        Spartan2DeciderProverKey {
            shape: shape.clone(),
            backend: pk,
        },
        Spartan2DeciderVerifierKey {
            shape: shape.clone(),
            backend: vk,
        },
    ))
}

pub fn prove_spartan2_decider_with_perf(
    pk: &Spartan2DeciderProverKey,
    target: &Spartan2DeciderTarget,
) -> Result<(Spartan2DeciderProof, Spartan2DeciderProvePerf), Spartan2DeciderError> {
    let total_started = std::time::Instant::now();
    if target.shape() != pk.shape {
        return Err(Spartan2DeciderError::ShapeMismatch);
    }
    validate_spartan2_decider_target_surface(target)?;
    let started = std::time::Instant::now();
    let relation = target
        .relation()
        .map_err(|err| Spartan2DeciderError::RelationSurface(err.to_string()))?;
    validate_spartan2_decider_relation_surface(&relation)
        .map_err(|err| Spartan2DeciderError::RelationSurface(err.to_string()))?;
    let backend_relation = relation.backend_relation();
    validate_spartan2_backend_relation_surface(&backend_relation).map_err(Spartan2DeciderError::Backend)?;
    let relation_surface_ms = started.elapsed().as_secs_f64() * 1_000.0;
    let (backend, shell_perf) = prove_spartan2_backend_binding_shell_with_perf(&pk.backend, &backend_relation)
        .map_err(Spartan2DeciderError::Backend)?;
    Ok((
        Spartan2DeciderProof {
            shape_digest: pk.shape.digest(),
            snark_data: backend.snark_data,
        },
        Spartan2DeciderProvePerf {
            relation_surface_ms,
            shell: shell_perf,
            total_ms: total_started.elapsed().as_secs_f64() * 1_000.0,
        },
    ))
}

pub fn verify_spartan2_decider(
    vk: &Spartan2DeciderVerifierKey,
    target: &Spartan2DeciderTarget,
    proof: &Spartan2DeciderProof,
) -> Result<(), Spartan2DeciderError> {
    if target.shape() != vk.shape {
        return Err(Spartan2DeciderError::ShapeMismatch);
    }
    if proof.shape_digest != vk.shape.digest() {
        return Err(Spartan2DeciderError::ShapeDigestMismatch);
    }
    validate_spartan2_decider_target_surface(target)?;
    let relation = target
        .relation()
        .map_err(|err| Spartan2DeciderError::RelationSurface(err.to_string()))?;
    validate_spartan2_decider_relation_surface(&relation)
        .map_err(|err| Spartan2DeciderError::RelationSurface(err.to_string()))?;
    let backend_relation = relation.backend_relation();
    validate_spartan2_backend_relation_surface(&backend_relation).map_err(Spartan2DeciderError::Backend)?;
    verify_spartan2_backend_binding_shell(
        &vk.backend,
        &backend_relation,
        &Spartan2BackendBindingShellProof {
            snark_data: proof.snark_data.clone(),
        },
    )
    .map_err(Spartan2DeciderError::Backend)?;
    Ok(())
}

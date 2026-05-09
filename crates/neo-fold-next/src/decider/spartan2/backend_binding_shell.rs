use super::*;

pub fn setup_spartan2_backend_binding_shell(
    shape: &Spartan2DeciderShape,
) -> Result<
    (
        Spartan2BackendBindingShellProverKey,
        Spartan2BackendBindingShellVerifierKey,
    ),
    Spartan2BackendBindingShellError,
> {
    Spartan2BackendBindingShellSnark::setup(Spartan2BackendBindingShellCircuit::from_shape(shape))
        .map_err(|err| Spartan2BackendBindingShellError::Setup(err.to_string()))
}

pub fn prove_spartan2_backend_binding_shell_with_perf(
    pk: &Spartan2BackendBindingShellProverKey,
    relation: &Spartan2DeciderBackendRelation,
) -> Result<(Spartan2BackendBindingShellProof, Spartan2BackendBindingShellProvePerf), Spartan2BackendBindingShellError>
{
    let total_started = std::time::Instant::now();
    validate_spartan2_backend_relation_surface(relation)?;
    let circuit = Spartan2BackendBindingShellCircuit::from_relation(relation);
    let started = std::time::Instant::now();
    let prep = Spartan2BackendBindingShellSnark::prep_prove(pk, circuit.clone(), true)
        .map_err(|err| Spartan2BackendBindingShellError::Prepare(err.to_string()))?;
    let prep_ms = started.elapsed().as_secs_f64() * 1_000.0;
    let started = std::time::Instant::now();
    let (proof, snark_perf) = Spartan2BackendBindingShellSnark::prove_with_perf(pk, circuit, &prep, true)
        .map_err(|err| Spartan2BackendBindingShellError::Prove(err.to_string()))?;
    let mut snark_perf = snark_perf;
    snark_perf.total_ms = started.elapsed().as_secs_f64() * 1_000.0;
    let started = std::time::Instant::now();
    let snark_data =
        bincode::serialize(&proof).map_err(|err| Spartan2BackendBindingShellError::Encode(err.to_string()))?;
    let encode_ms = started.elapsed().as_secs_f64() * 1_000.0;
    Ok((
        Spartan2BackendBindingShellProof { snark_data },
        Spartan2BackendBindingShellProvePerf {
            prep_ms,
            snark_perf,
            encode_ms,
            total_ms: total_started.elapsed().as_secs_f64() * 1_000.0,
        },
    ))
}

pub fn verify_spartan2_backend_binding_shell(
    vk: &Spartan2BackendBindingShellVerifierKey,
    relation: &Spartan2DeciderBackendRelation,
    proof: &Spartan2BackendBindingShellProof,
) -> Result<(), Spartan2BackendBindingShellError> {
    validate_spartan2_backend_relation_surface(relation)?;
    let proof: Spartan2BackendBindingShellSnark = bincode::deserialize(&proof.snark_data)
        .map_err(|err| Spartan2BackendBindingShellError::Decode(err.to_string()))?;
    let public_values = proof
        .verify(vk)
        .map_err(|err| Spartan2BackendBindingShellError::Verify(err.to_string()))?
        .into_iter()
        .map(|value| F::from_u64(value.to_canonical_u64()))
        .collect::<Vec<_>>();
    if public_values != relation.public_io() {
        return Err(Spartan2BackendBindingShellError::PublicIoMismatch);
    }
    Ok(())
}

#[derive(Clone, Debug)]
struct Spartan2BackendBindingShellCircuit {
    public_values: Vec<SpartanF>,
    private_values: Vec<SpartanF>,
    public_semantic_offset: usize,
    public_binding_offset: usize,
    expected_base_component_count: u64,
    expected_chunk_transition_count: u64,
}

impl Spartan2BackendBindingShellCircuit {
    fn from_shape(shape: &Spartan2DeciderShape) -> Self {
        Self {
            public_values: vec![SpartanF::from_canonical_u64(0); shape.backend_public_io_len()],
            private_values: vec![SpartanF::from_canonical_u64(0); shape.backend_witness_field_len()],
            public_semantic_offset: shape.statement_public_io_len(),
            public_binding_offset: shape.statement_public_io_len() + POSEIDON2_DIGEST_LEN,
            expected_base_component_count: shape.base_component_count as u64,
            expected_chunk_transition_count: shape.chunk_transition_count as u64,
        }
    }

    fn from_relation(relation: &Spartan2DeciderBackendRelation) -> Self {
        let shape = relation.shape();
        Self {
            public_values: relation
                .public_io()
                .into_iter()
                .map(|value| SpartanF::from_canonical_u64(value.as_canonical_u64()))
                .collect(),
            private_values: relation
                .witness
                .packed_fields()
                .into_iter()
                .map(|value| SpartanF::from_canonical_u64(value.as_canonical_u64()))
                .collect(),
            public_semantic_offset: relation.statement.public_io().len(),
            public_binding_offset: relation.statement.public_io().len() + POSEIDON2_DIGEST_LEN,
            expected_base_component_count: shape.base_component_count as u64,
            expected_chunk_transition_count: shape.chunk_transition_count as u64,
        }
    }
}

impl SpartanCircuit<Spartan2BackendBindingShellEngine> for Spartan2BackendBindingShellCircuit {
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
        let mut public_inputs = Vec::with_capacity(self.public_values.len());
        for (idx, value) in self.public_values.iter().copied().enumerate() {
            public_inputs.push(AllocatedNum::alloc_input(
                cs.namespace(|| format!("backend_public_input_{idx}")),
                || Ok(value),
            )?);
        }

        let mut private_witness = Vec::with_capacity(self.private_values.len());
        for (idx, value) in self.private_values.iter().copied().enumerate() {
            private_witness.push(AllocatedNum::alloc(
                cs.namespace(|| format!("backend_private_witness_{idx}")),
                || Ok(value),
            )?);
        }

        let digest = hash_packed_goldilocks_fields(cs.namespace(|| "backend_witness_digest"), &private_witness)?;
        let private_base_count = &private_witness[0];
        let private_chunk_count = &private_witness[1];
        let packed_digest_len = packed_bytes_field_len(32);
        let base_digest_offset = 2;
        let chunk_binding_offset = base_digest_offset + self.expected_base_component_count as usize * packed_digest_len;
        let relation_digest_offset = packed_digest_len;
        let relation_digest_end = relation_digest_offset + packed_digest_len;
        let initial_handle_offset = 3 * packed_digest_len;
        let initial_handle_end = initial_handle_offset + FIXED_SHAPE_DIGEST_FIELD_LEN;
        let terminal_handle_offset = initial_handle_end;
        let terminal_handle_end = terminal_handle_offset + FIXED_SHAPE_DIGEST_FIELD_LEN;
        let fold_schedule_offset = terminal_handle_end;
        let semantic_step_count_offset = fold_schedule_offset + 2;
        let public_semantic_step_count = &public_inputs[semantic_step_count_offset];
        let summary_offset = semantic_step_count_offset + 1;
        let summary_end = self.public_semantic_offset;
        cs.enforce(
            || "backend_base_component_count_matches_shape",
            |lc| lc + private_base_count.get_variable(),
            |lc| lc + CS::one(),
            |lc| {
                lc + (
                    SpartanF::from_canonical_u64(self.expected_base_component_count),
                    CS::one(),
                )
            },
        );
        cs.enforce(
            || "backend_chunk_transition_count_matches_shape",
            |lc| lc + private_chunk_count.get_variable(),
            |lc| lc + CS::one(),
            |lc| {
                lc + (
                    SpartanF::from_canonical_u64(self.expected_chunk_transition_count),
                    CS::one(),
                )
            },
        );
        let mut current_handle = public_inputs[initial_handle_offset..initial_handle_end].to_vec();
        let summary_len = spartan2_chunk_summary_field_len();
        let chunk_relation_offset = FixedShapeChunkSummary::chunk_relation_digest_field_offset();
        let terminal_relation_digest_offset = spartan2_chunk_summary_terminal_relation_digest_field_offset();
        let private_chunk_relation_offset =
            Spartan2ChunkTransitionBinding::claimed_chunk_relation_digest_field_offset();
        if self.expected_chunk_transition_count == 0 {
            cs.enforce(
                || "backend_semantic_step_count_zero_when_no_chunks",
                |lc| lc + public_semantic_step_count.get_variable(),
                |lc| lc + CS::one(),
                |lc| lc,
            );
        } else {
            cs.enforce(
                || "backend_first_chunk_start_index_zero",
                |lc| lc + public_inputs[summary_offset].get_variable(),
                |lc| lc + CS::one(),
                |lc| lc,
            );
            for chunk_index in 1..self.expected_chunk_transition_count as usize {
                let previous_base = summary_offset + (chunk_index - 1) * summary_len;
                let current_base = summary_offset + chunk_index * summary_len;
                cs.enforce(
                    || format!("backend_chunk_start_contiguous_{chunk_index}"),
                    |lc| lc + public_inputs[current_base].get_variable(),
                    |lc| lc + CS::one(),
                    |lc| {
                        lc + public_inputs[previous_base].get_variable()
                            + public_inputs[previous_base + 1].get_variable()
                    },
                );
            }
            let last_base = summary_offset + (self.expected_chunk_transition_count as usize - 1) * summary_len;
            cs.enforce(
                || "backend_semantic_step_count_matches_coverage",
                |lc| lc + public_semantic_step_count.get_variable(),
                |lc| lc + CS::one(),
                |lc| lc + public_inputs[last_base].get_variable() + public_inputs[last_base + 1].get_variable(),
            );
        }
        for chunk_index in 0..self.expected_chunk_transition_count as usize {
            let chunk_index_num =
                AllocatedNum::alloc(cs.namespace(|| format!("backend_chunk_index_{chunk_index}")), || {
                    Ok(SpartanF::from_canonical_u64(chunk_index as u64))
                })?;
            cs.enforce(
                || format!("backend_chunk_index_matches_shape_{chunk_index}"),
                |lc| lc + chunk_index_num.get_variable(),
                |lc| lc + CS::one(),
                |lc| lc + (SpartanF::from_canonical_u64(chunk_index as u64), CS::one()),
            );
            let summary_base = summary_offset + chunk_index * summary_len;
            let private_binding_base =
                chunk_binding_offset + chunk_index * Spartan2ChunkTransitionBinding::packed_field_len();
            let start_index = public_inputs[summary_base].clone();
            let public_step_count = public_inputs[summary_base + 1].clone();
            for digest_idx in 0..Spartan2ChunkTransitionBinding::packed_digest_field_len() {
                cs.enforce(
                    || format!("backend_chunk_relation_binding_match_{chunk_index}_{digest_idx}"),
                    |lc| {
                        lc + private_witness[private_binding_base + private_chunk_relation_offset + digest_idx]
                            .get_variable()
                    },
                    |lc| lc + CS::one(),
                    |lc| lc + public_inputs[summary_base + chunk_relation_offset + digest_idx].get_variable(),
                );
            }
            let mut handle_preimage =
                Vec::with_capacity(FIXED_SHAPE_DIGEST_FIELD_LEN + 3 + FIXED_SHAPE_DIGEST_FIELD_LEN);
            handle_preimage.extend(current_handle.iter().cloned());
            handle_preimage.push(chunk_index_num);
            handle_preimage.push(start_index);
            handle_preimage.push(public_step_count);
            handle_preimage.extend(
                public_inputs[summary_base + terminal_relation_digest_offset
                    ..summary_base + terminal_relation_digest_offset + FIXED_SHAPE_DIGEST_FIELD_LEN]
                    .iter()
                    .cloned(),
            );
            current_handle = hash_packed_goldilocks_fields(
                cs.namespace(|| format!("backend_terminal_handle_step_{chunk_index}")),
                &handle_preimage,
            )?
            .into_iter()
            .collect();
        }
        for (idx, handle_value) in current_handle.into_iter().enumerate() {
            let public = &public_inputs[terminal_handle_offset + idx];
            cs.enforce(
                || format!("backend_terminal_handle_match_{idx}"),
                |lc| lc + handle_value.get_variable(),
                |lc| lc + CS::one(),
                |lc| lc + public.get_variable(),
            );
        }
        let mut semantic_preimage = Vec::with_capacity(
            (relation_digest_end - relation_digest_offset) + (summary_end - summary_offset) + digest.len(),
        );
        semantic_preimage.extend(
            public_inputs[relation_digest_offset..relation_digest_end]
                .iter()
                .cloned(),
        );
        semantic_preimage.extend(public_inputs[summary_offset..summary_end].iter().cloned());
        semantic_preimage.extend(digest.iter().cloned());
        let semantic_digest =
            hash_packed_goldilocks_fields(cs.namespace(|| "backend_semantic_digest"), &semantic_preimage)?;
        for (idx, digest_value) in semantic_digest.into_iter().enumerate() {
            let public = &public_inputs[self.public_semantic_offset + idx];
            cs.enforce(
                || format!("backend_semantic_match_{idx}"),
                |lc| lc + digest_value.get_variable(),
                |lc| lc + CS::one(),
                |lc| lc + public.get_variable(),
            );
        }
        let mut binding_preimage = Vec::with_capacity(self.public_binding_offset + digest.len());
        binding_preimage.extend(public_inputs[..self.public_binding_offset].iter().cloned());
        binding_preimage.extend(digest);
        let binding_digest =
            hash_packed_goldilocks_fields(cs.namespace(|| "backend_binding_digest"), &binding_preimage)?;
        for (idx, digest_value) in binding_digest.into_iter().enumerate() {
            let public = &public_inputs[self.public_binding_offset + idx];
            cs.enforce(
                || format!("backend_binding_match_{idx}"),
                |lc| lc + digest_value.get_variable(),
                |lc| lc + CS::one(),
                |lc| lc + public.get_variable(),
            );
        }

        Ok(())
    }
}

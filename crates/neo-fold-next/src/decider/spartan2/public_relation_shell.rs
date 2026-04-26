//! Owns the public fixed-shape Spartan shell used by the decider proof.

use super::*;

pub type Spartan2PublicRelationShellEngine = GoldilocksP3MerkleMleEngine;
pub type Spartan2PublicRelationShellSnark = R1CSSNARK<Spartan2PublicRelationShellEngine>;
pub type Spartan2PublicRelationShellProverKey = spartan2::spartan::SpartanProverKey<Spartan2PublicRelationShellEngine>;
pub type Spartan2PublicRelationShellVerifierKey =
    spartan2::spartan::SpartanVerifierKey<Spartan2PublicRelationShellEngine>;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Spartan2PublicRelationShellProof {
    pub snark_data: Vec<u8>,
}

impl Spartan2PublicRelationShellProof {
    pub fn snark_bytes_len(&self) -> usize {
        self.snark_data.len()
    }
}

#[derive(Debug, Error)]
pub enum Spartan2PublicRelationShellError {
    #[error("spartan2 public-relation shell relation surface mismatch: {0}")]
    RelationSurface(String),
    #[error("spartan2 public-relation shell setup failed: {0}")]
    Setup(String),
    #[error("spartan2 public-relation shell prepare failed: {0}")]
    Prepare(String),
    #[error("spartan2 public-relation shell prove failed: {0}")]
    Prove(String),
    #[error("spartan2 public-relation shell verify failed: {0}")]
    Verify(String),
    #[error("spartan2 public-relation shell proof encoding failed: {0}")]
    Encode(String),
    #[error("spartan2 public-relation shell proof decoding failed: {0}")]
    Decode(String),
    #[error("spartan2 public-relation shell public IO mismatch")]
    PublicIoMismatch,
}

#[derive(Clone, Debug)]
struct Spartan2PublicRelationShellCircuit {
    public_values: Vec<SpartanF>,
    statement_public_io_len: usize,
    expected_base_component_count: u64,
    expected_chunk_transition_count: u64,
}

impl Spartan2PublicRelationShellCircuit {
    fn from_shape(shape: &Spartan2DeciderShape) -> Self {
        Self {
            public_values: vec![SpartanF::from_canonical_u64(0); shape.public_io_len()],
            statement_public_io_len: shape.statement_public_io_len(),
            expected_base_component_count: shape.base_component_count as u64,
            expected_chunk_transition_count: shape.chunk_transition_count as u64,
        }
    }

    fn from_target(target: &Spartan2DeciderTarget) -> Self {
        let shape = target.shape();
        Self {
            public_values: target
                .public_io()
                .into_iter()
                .map(|value| SpartanF::from_canonical_u64(value.as_canonical_u64()))
                .collect(),
            statement_public_io_len: shape.statement_public_io_len(),
            expected_base_component_count: shape.base_component_count as u64,
            expected_chunk_transition_count: shape.chunk_transition_count as u64,
        }
    }
}

impl SpartanCircuit<Spartan2PublicRelationShellEngine> for Spartan2PublicRelationShellCircuit {
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
                cs.namespace(|| format!("public_relation_input_{idx}")),
                || Ok(value),
            )?);
        }

        let packed_digest_len = packed_bytes_field_len(32);
        let initial_handle_offset = 3 * packed_digest_len;
        let initial_handle_end = initial_handle_offset + FIXED_SHAPE_DIGEST_FIELD_LEN;
        let terminal_handle_offset = initial_handle_end;
        let terminal_handle_end = terminal_handle_offset + FIXED_SHAPE_DIGEST_FIELD_LEN;
        let fold_schedule_offset = terminal_handle_end;
        let semantic_step_count_offset = fold_schedule_offset + 2;
        let summary_offset = semantic_step_count_offset + 1;
        let summary_len = spartan2_chunk_summary_field_len();
        let witness_base_count_offset = self.statement_public_io_len;
        let witness_chunk_count_offset = witness_base_count_offset + 1;
        let base_digest_offset = witness_chunk_count_offset + 1;
        let chunk_binding_offset = base_digest_offset + self.expected_base_component_count as usize * packed_digest_len;
        let chunk_relation_offset = FixedShapeChunkSummary::chunk_relation_digest_field_offset();
        let terminal_relation_digest_offset = spartan2_chunk_summary_terminal_relation_digest_field_offset();
        let binding_relation_offset = Spartan2ChunkTransitionBinding::claimed_chunk_relation_digest_field_offset();
        let public_semantic_step_count = &public_inputs[semantic_step_count_offset];

        cs.enforce(
            || "public_relation_base_component_count_matches_shape",
            |lc| lc + public_inputs[witness_base_count_offset].get_variable(),
            |lc| lc + CS::one(),
            |lc| {
                lc + (
                    SpartanF::from_canonical_u64(self.expected_base_component_count),
                    CS::one(),
                )
            },
        );
        cs.enforce(
            || "public_relation_chunk_transition_count_matches_shape",
            |lc| lc + public_inputs[witness_chunk_count_offset].get_variable(),
            |lc| lc + CS::one(),
            |lc| {
                lc + (
                    SpartanF::from_canonical_u64(self.expected_chunk_transition_count),
                    CS::one(),
                )
            },
        );
        if self.expected_chunk_transition_count == 0 {
            cs.enforce(
                || "public_relation_semantic_step_count_zero_when_no_chunks",
                |lc| lc + public_semantic_step_count.get_variable(),
                |lc| lc + CS::one(),
                |lc| lc,
            );
        } else {
            cs.enforce(
                || "public_relation_first_chunk_start_index_zero",
                |lc| lc + public_inputs[summary_offset].get_variable(),
                |lc| lc + CS::one(),
                |lc| lc,
            );
            for chunk_index in 1..self.expected_chunk_transition_count as usize {
                let previous_base = summary_offset + (chunk_index - 1) * summary_len;
                let current_base = summary_offset + chunk_index * summary_len;
                cs.enforce(
                    || format!("public_relation_chunk_start_contiguous_{chunk_index}"),
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
                || "public_relation_semantic_step_count_matches_coverage",
                |lc| lc + public_semantic_step_count.get_variable(),
                |lc| lc + CS::one(),
                |lc| lc + public_inputs[last_base].get_variable() + public_inputs[last_base + 1].get_variable(),
            );
        }

        let mut current_handle = public_inputs[initial_handle_offset..initial_handle_end].to_vec();
        for chunk_index in 0..self.expected_chunk_transition_count as usize {
            let chunk_index_num = AllocatedNum::alloc(
                cs.namespace(|| format!("public_relation_chunk_index_{chunk_index}")),
                || Ok(SpartanF::from_canonical_u64(chunk_index as u64)),
            )?;
            cs.enforce(
                || format!("public_relation_chunk_index_matches_shape_{chunk_index}"),
                |lc| lc + chunk_index_num.get_variable(),
                |lc| lc + CS::one(),
                |lc| lc + (SpartanF::from_canonical_u64(chunk_index as u64), CS::one()),
            );

            let summary_base = summary_offset + chunk_index * summary_len;
            let binding_base = chunk_binding_offset + chunk_index * Spartan2ChunkTransitionBinding::packed_field_len();
            let start_index = public_inputs[summary_base].clone();
            let public_step_count = public_inputs[summary_base + 1].clone();
            let public_step_count_inverse = AllocatedNum::alloc(
                cs.namespace(|| format!("public_relation_public_step_count_inverse_{chunk_index}")),
                || {
                    let value = public_step_count
                        .get_value()
                        .ok_or(SynthesisError::AssignmentMissing)?;
                    spartan_inverse(value).ok_or(SynthesisError::Unsatisfiable)
                },
            )?;
            cs.enforce(
                || format!("public_relation_public_step_count_nonzero_{chunk_index}"),
                |lc| lc + public_step_count.get_variable(),
                |lc| lc + public_step_count_inverse.get_variable(),
                |lc| lc + CS::one(),
            );

            for digest_idx in 0..Spartan2ChunkTransitionBinding::packed_digest_field_len() {
                cs.enforce(
                    || format!("public_relation_chunk_binding_match_{chunk_index}_{digest_idx}"),
                    |lc| lc + public_inputs[binding_base + binding_relation_offset + digest_idx].get_variable(),
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
                cs.namespace(|| format!("public_relation_terminal_handle_step_{chunk_index}")),
                &handle_preimage,
            )?
            .into_iter()
            .collect();
        }

        for (idx, handle_value) in current_handle.into_iter().enumerate() {
            let public = &public_inputs[terminal_handle_offset + idx];
            cs.enforce(
                || format!("public_relation_terminal_handle_match_{idx}"),
                |lc| lc + handle_value.get_variable(),
                |lc| lc + CS::one(),
                |lc| lc + public.get_variable(),
            );
        }

        Ok(())
    }
}

pub fn setup_spartan2_public_relation_shell(
    shape: &Spartan2DeciderShape,
) -> Result<
    (
        Spartan2PublicRelationShellProverKey,
        Spartan2PublicRelationShellVerifierKey,
    ),
    Spartan2PublicRelationShellError,
> {
    Spartan2PublicRelationShellSnark::setup(Spartan2PublicRelationShellCircuit::from_shape(shape))
        .map_err(|err| Spartan2PublicRelationShellError::Setup(err.to_string()))
}

pub fn prove_spartan2_public_relation_shell(
    pk: &Spartan2PublicRelationShellProverKey,
    target: &Spartan2DeciderTarget,
) -> Result<Spartan2PublicRelationShellProof, Spartan2PublicRelationShellError> {
    let relation = target
        .relation()
        .map_err(|err| Spartan2PublicRelationShellError::RelationSurface(err.to_string()))?;
    validate_spartan2_decider_relation_surface(&relation)
        .map_err(|err| Spartan2PublicRelationShellError::RelationSurface(err.to_string()))?;
    let circuit = Spartan2PublicRelationShellCircuit::from_target(target);
    let prep = Spartan2PublicRelationShellSnark::prep_prove(pk, circuit.clone(), true)
        .map_err(|err| Spartan2PublicRelationShellError::Prepare(err.to_string()))?;
    let proof = Spartan2PublicRelationShellSnark::prove(pk, circuit, &prep, true)
        .map_err(|err| Spartan2PublicRelationShellError::Prove(err.to_string()))?;
    let snark_data =
        bincode::serialize(&proof).map_err(|err| Spartan2PublicRelationShellError::Encode(err.to_string()))?;
    Ok(Spartan2PublicRelationShellProof { snark_data })
}

pub fn verify_spartan2_public_relation_shell(
    vk: &Spartan2PublicRelationShellVerifierKey,
    target: &Spartan2DeciderTarget,
    proof: &Spartan2PublicRelationShellProof,
) -> Result<(), Spartan2PublicRelationShellError> {
    let relation = target
        .relation()
        .map_err(|err| Spartan2PublicRelationShellError::RelationSurface(err.to_string()))?;
    validate_spartan2_decider_relation_surface(&relation)
        .map_err(|err| Spartan2PublicRelationShellError::RelationSurface(err.to_string()))?;
    let proof: Spartan2PublicRelationShellSnark = bincode::deserialize(&proof.snark_data)
        .map_err(|err| Spartan2PublicRelationShellError::Decode(err.to_string()))?;
    let public_values = proof
        .verify(vk)
        .map_err(|err| Spartan2PublicRelationShellError::Verify(err.to_string()))?
        .into_iter()
        .map(|value| F::from_u64(value.to_canonical_u64()))
        .collect::<Vec<_>>();
    if public_values != target.public_io() {
        return Err(Spartan2PublicRelationShellError::PublicIoMismatch);
    }
    Ok(())
}

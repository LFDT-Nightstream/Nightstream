use super::*;

impl Spartan2ChunkTransitionBinding {
    fn digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/decider/spartan2/chunk_transition_binding");
        tr.append_message(
            b"neo.fold.next/decider/spartan2/chunk_transition_binding/chunk_relation_digest",
            &self.claimed_chunk_relation_digest,
        );
        tr.append_message(
            b"neo.fold.next/decider/spartan2/chunk_transition_binding/transition_witness_digest",
            &self.transition_witness_digest,
        );
        tr.digest32()
    }

    fn packed_fields(&self) -> Vec<F> {
        let mut out = Vec::with_capacity(Self::packed_field_len());
        extend_packed_bytes_as_fields(&mut out, &self.claimed_chunk_relation_digest);
        extend_packed_bytes_as_fields(&mut out, &self.transition_witness_digest);
        out
    }

    pub(crate) const fn packed_digest_field_len() -> usize {
        6
    }

    pub(crate) const fn claimed_chunk_relation_digest_field_offset() -> usize {
        0
    }

    pub(crate) const fn packed_field_len() -> usize {
        2 * Self::packed_digest_field_len()
    }
}

fn build_chunk_transition_bindings(
    chunk_summaries: &[FixedShapeChunkSummary],
    transition_witness_digests: Vec<[u8; 32]>,
) -> Result<Vec<Spartan2ChunkTransitionBinding>, Spartan2DeciderError> {
    if chunk_summaries.len() != transition_witness_digests.len() {
        return Err(Spartan2DeciderError::RelationSurface(
            "chunk summary count does not match carried chunk transition digests".into(),
        ));
    }
    Ok(chunk_summaries
        .iter()
        .zip(transition_witness_digests)
        .map(|(summary, transition_witness_digest)| Spartan2ChunkTransitionBinding {
            claimed_chunk_relation_digest: summary.chunk_relation_digest,
            transition_witness_digest,
        })
        .collect())
}

fn transition_witness_digests(bindings: &[Spartan2ChunkTransitionBinding]) -> Vec<[u8; 32]> {
    bindings
        .iter()
        .map(|binding| binding.transition_witness_digest)
        .collect()
}

fn validate_spartan2_chunk_layout(
    schedule: FoldSchedule,
    semantic_step_count: u64,
    chunk_summaries: &[FixedShapeChunkSummary],
) -> Result<(), String> {
    let active_chunk_count = chunk_summaries
        .iter()
        .position(|summary| summary.public_step_count == 0)
        .unwrap_or(chunk_summaries.len());
    let (active, padded) = chunk_summaries.split_at(active_chunk_count);
    validate_fixed_shape_chunk_layout(schedule, semantic_step_count, active)?;
    for (idx, summary) in padded.iter().enumerate() {
        if summary.public_step_count != 0 {
            return Err(format!(
                "padded chunk summary {} carries {} public steps; padded fixed-shape tails must be zero",
                active_chunk_count + idx,
                summary.public_step_count
            ));
        }
        if summary.start_index != semantic_step_count {
            return Err(format!(
                "padded chunk summary {} start index {} does not match semantic step count {}",
                active_chunk_count + idx,
                summary.start_index,
                semantic_step_count
            ));
        }
        if summary.public_chunk_digest != [0; 32] {
            return Err(format!(
                "padded chunk summary {} public chunk digest must be zero",
                active_chunk_count + idx
            ));
        }
        if summary.chunk_relation_digest != [0; 32] {
            return Err(format!(
                "padded chunk summary {} chunk relation digest must be zero",
                active_chunk_count + idx
            ));
        }
    }
    Ok(())
}

fn validate_chunk_transition_bindings(
    chunk_summaries: &[FixedShapeChunkSummary],
    chunk_transition_bindings: &[Spartan2ChunkTransitionBinding],
) -> Result<(), String> {
    if chunk_summaries.len() != chunk_transition_bindings.len() {
        return Err("chunk summary count does not match carried chunk transition bindings".into());
    }
    for (idx, (summary, binding)) in chunk_summaries
        .iter()
        .zip(chunk_transition_bindings.iter())
        .enumerate()
    {
        if summary.public_step_count == 0 {
            if binding.claimed_chunk_relation_digest != [0; 32] || binding.transition_witness_digest != [0; 32] {
                return Err(format!(
                    "padded chunk transition binding {} must be canonical zero",
                    idx
                ));
            }
            continue;
        }
        if binding.claimed_chunk_relation_digest != summary.chunk_relation_digest {
            return Err(format!(
                "chunk transition binding {} does not match the carried public chunk relation digest",
                idx
            ));
        }
    }
    Ok(())
}

fn backend_semantic_digest_fields(
    relation_digest: &[u8; 32],
    chunk_summaries: &[FixedShapeChunkSummary],
    witness: &Spartan2DeciderBackendWitness,
) -> [F; POSEIDON2_DIGEST_LEN] {
    let mut preimage = Vec::with_capacity(
        packed_bytes_field_len(32) + chunk_summaries.len() * spartan2_chunk_summary_field_len() + POSEIDON2_DIGEST_LEN,
    );
    extend_packed_bytes_as_fields(&mut preimage, relation_digest);
    for summary in chunk_summaries {
        extend_spartan2_chunk_summary_fields(&mut preimage, summary);
    }
    preimage.extend(witness.digest_fields());
    poseidon2_hash(&preimage)
}

impl Spartan2DeciderRelation {
    fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/decider/spartan2/relation");
        tr.append_message(
            b"neo.fold.next/decider/spartan2/relation/public_statement_digest",
            &self.public_statement_digest,
        );
        tr.append_message(
            b"neo.fold.next/decider/spartan2/relation/relation_digest",
            &self.relation_digest,
        );
        tr.append_message(
            b"neo.fold.next/decider/spartan2/relation/final_proof_digest",
            &self.final_proof_digest,
        );
        tr.append_fields_iter(
            b"neo.fold.next/decider/spartan2/relation/initial_handle_digest",
            FIXED_SHAPE_DIGEST_FIELD_LEN,
            self.initial_handle_digest.iter().copied(),
        );
        tr.append_fields_iter(
            b"neo.fold.next/decider/spartan2/relation/terminal_handle_digest",
            FIXED_SHAPE_DIGEST_FIELD_LEN,
            self.terminal_handle_digest.iter().copied(),
        );
        tr.append_u64s(
            b"neo.fold.next/decider/spartan2/relation/fold_schedule",
            &self.fold_schedule.meta_words(),
        );
        tr.append_u64s(
            b"neo.fold.next/decider/spartan2/relation/chunk_count",
            &[self.chunk_summaries.len() as u64],
        );
        tr.append_u64s(
            b"neo.fold.next/decider/spartan2/relation/semantic_step_count",
            &[self.semantic_step_count],
        );
        for summary in &self.chunk_summaries {
            tr.append_message(
                b"neo.fold.next/decider/spartan2/relation/chunk_summary",
                &summary.digest(),
            );
        }
        tr.append_u64s(
            b"neo.fold.next/decider/spartan2/relation/base_component_count",
            &[self.base_component_digests.len() as u64],
        );
        tr.append_u64s(
            b"neo.fold.next/decider/spartan2/relation/chunk_transition_count",
            &[self.chunk_transition_bindings.len() as u64],
        );
        for digest in &self.base_component_digests {
            tr.append_message(b"neo.fold.next/decider/spartan2/relation/base_component_digest", digest);
        }
        for binding in &self.chunk_transition_bindings {
            tr.append_message(
                b"neo.fold.next/decider/spartan2/relation/chunk_transition_binding",
                &binding.digest(),
            );
        }
        tr.digest32()
    }

    pub fn target(&self) -> Spartan2DeciderTarget {
        Spartan2DeciderTarget {
            statement: Spartan2DeciderStatement {
                public_statement_digest: self.public_statement_digest,
                relation_digest: self.relation_digest,
                final_proof_digest: self.final_proof_digest,
                initial_handle_digest: self.initial_handle_digest,
                terminal_handle_digest: self.terminal_handle_digest,
                fold_schedule: self.fold_schedule,
                semantic_step_count: self.semantic_step_count,
                chunk_summaries: self.chunk_summaries.clone(),
            },
            witness: Spartan2DeciderWitness {
                base_component_digests: self.base_component_digests.clone(),
                chunk_transition_bindings: self.chunk_transition_bindings.clone(),
            },
        }
    }

    pub fn backend_shape(&self) -> Spartan2DeciderShape {
        Spartan2DeciderShape {
            base_component_count: self.base_component_digests.len(),
            chunk_transition_count: self.chunk_transition_bindings.len(),
        }
    }

    pub fn backend_relation(&self) -> Spartan2DeciderBackendRelation {
        Spartan2DeciderBackendRelation {
            statement: Spartan2DeciderStatement {
                public_statement_digest: self.public_statement_digest,
                relation_digest: self.relation_digest,
                final_proof_digest: self.final_proof_digest,
                initial_handle_digest: self.initial_handle_digest,
                terminal_handle_digest: self.terminal_handle_digest,
                fold_schedule: self.fold_schedule,
                semantic_step_count: self.semantic_step_count,
                chunk_summaries: self.chunk_summaries.clone(),
            },
            witness: Spartan2DeciderBackendWitness {
                base_component_count: self.base_component_digests.len() as u64,
                chunk_transition_count: self.chunk_transition_bindings.len() as u64,
                base_component_digests: self.base_component_digests.clone(),
                chunk_transition_bindings: self.chunk_transition_bindings.clone(),
            },
        }
    }

    fn expected_final_proof_digest(&self) -> [u8; 32] {
        digest_fixed_shape_final_proof(
            &self.relation_digest,
            self.chunk_summaries.len() as u64,
            &self.chunk_summaries,
            &self.base_component_digests,
            &transition_witness_digests(&self.chunk_transition_bindings),
        )
    }

    fn expected_terminal_handle_digest(&self) -> [F; FIXED_SHAPE_DIGEST_FIELD_LEN] {
        fixed_shape_terminal_handle_digest_fields(
            digest_fields_as_digest32(self.initial_handle_digest),
            &self.chunk_summaries,
        )
    }
}

pub fn build_spartan2_decider_relation(
    public_statement_digest: [u8; 32],
    relation_digest: [u8; 32],
    final_proof_digest: [u8; 32],
    initial_handle_digest: [F; FIXED_SHAPE_DIGEST_FIELD_LEN],
    terminal_handle_digest: [F; FIXED_SHAPE_DIGEST_FIELD_LEN],
    fold_schedule: FoldSchedule,
    semantic_step_count: u64,
    chunk_summaries: Vec<FixedShapeChunkSummary>,
    base_component_digests: Vec<[u8; 32]>,
    chunk_transition_digests: Vec<[u8; 32]>,
) -> Result<Spartan2DeciderRelation, Spartan2DeciderError> {
    let chunk_transition_bindings = build_chunk_transition_bindings(&chunk_summaries, chunk_transition_digests)?;
    let mut relation = Spartan2DeciderRelation {
        public_statement_digest,
        relation_digest,
        final_proof_digest,
        initial_handle_digest,
        terminal_handle_digest,
        fold_schedule,
        semantic_step_count,
        chunk_summaries,
        base_component_digests,
        chunk_transition_bindings,
        digest: [0; 32],
    };
    relation.digest = relation.expected_digest();
    Ok(relation)
}

pub fn build_spartan2_self_bound_decider_relation(
    public_statement_digest: [u8; 32],
    relation_digest: [u8; 32],
    initial_handle_digest: [F; FIXED_SHAPE_DIGEST_FIELD_LEN],
    fold_schedule: FoldSchedule,
    semantic_step_count: u64,
    chunk_summaries: Vec<FixedShapeChunkSummary>,
    base_component_digests: Vec<[u8; 32]>,
    chunk_transition_digests: Vec<[u8; 32]>,
) -> Result<Spartan2DeciderRelation, Spartan2DeciderError> {
    let mut relation = build_spartan2_decider_relation(
        public_statement_digest,
        relation_digest,
        [0; 32],
        initial_handle_digest,
        [F::ZERO; FIXED_SHAPE_DIGEST_FIELD_LEN],
        fold_schedule,
        semantic_step_count,
        chunk_summaries,
        base_component_digests,
        chunk_transition_digests,
    )?;
    relation.terminal_handle_digest = relation.expected_terminal_handle_digest();
    relation.final_proof_digest = relation.expected_final_proof_digest();
    relation.digest = relation.expected_digest();
    Ok(relation)
}

pub fn validate_spartan2_decider_relation_surface(
    relation: &Spartan2DeciderRelation,
) -> Result<(), Spartan2DeciderError> {
    validate_chunk_transition_bindings(&relation.chunk_summaries, &relation.chunk_transition_bindings)
        .map_err(Spartan2DeciderError::RelationSurface)?;
    validate_spartan2_chunk_layout(
        relation.fold_schedule,
        relation.semantic_step_count,
        &relation.chunk_summaries,
    )
    .map_err(Spartan2DeciderError::RelationSurface)?;
    if relation.terminal_handle_digest != relation.expected_terminal_handle_digest() {
        return Err(Spartan2DeciderError::RelationSurface(
            "terminal handle digest does not match the carried chunk summary chain".into(),
        ));
    }
    if relation.digest != relation.expected_digest() {
        return Err(Spartan2DeciderError::RelationDigestMismatch);
    }
    if relation.final_proof_digest != relation.expected_final_proof_digest() {
        return Err(Spartan2DeciderError::FinalProofDigestMismatch);
    }
    Ok(())
}

impl Spartan2DeciderStatement {
    pub fn digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/decider/spartan2/statement");
        tr.append_message(
            b"neo.fold.next/decider/spartan2/statement/public_statement_digest",
            &self.public_statement_digest,
        );
        tr.append_message(
            b"neo.fold.next/decider/spartan2/statement/relation_digest",
            &self.relation_digest,
        );
        tr.append_message(
            b"neo.fold.next/decider/spartan2/statement/final_proof_digest",
            &self.final_proof_digest,
        );
        tr.append_fields_iter(
            b"neo.fold.next/decider/spartan2/statement/initial_handle_digest",
            FIXED_SHAPE_DIGEST_FIELD_LEN,
            self.initial_handle_digest.iter().copied(),
        );
        tr.append_fields_iter(
            b"neo.fold.next/decider/spartan2/statement/terminal_handle_digest",
            FIXED_SHAPE_DIGEST_FIELD_LEN,
            self.terminal_handle_digest.iter().copied(),
        );
        tr.append_u64s(
            b"neo.fold.next/decider/spartan2/statement/fold_schedule",
            &self.fold_schedule.meta_words(),
        );
        tr.append_u64s(
            b"neo.fold.next/decider/spartan2/statement/semantic_step_count",
            &[self.semantic_step_count],
        );
        for summary in &self.chunk_summaries {
            tr.append_message(
                b"neo.fold.next/decider/spartan2/statement/chunk_summary",
                &summary.digest(),
            );
        }
        tr.digest32()
    }

    pub fn public_io(&self) -> Vec<F> {
        let mut out = Vec::with_capacity(
            3 * packed_bytes_field_len(32)
                + 2 * FIXED_SHAPE_DIGEST_FIELD_LEN
                + 3
                + self.chunk_summaries.len() * spartan2_chunk_summary_field_len(),
        );
        extend_packed_bytes_as_fields(&mut out, &self.public_statement_digest);
        extend_packed_bytes_as_fields(&mut out, &self.relation_digest);
        extend_packed_bytes_as_fields(&mut out, &self.final_proof_digest);
        out.extend(self.initial_handle_digest);
        out.extend(self.terminal_handle_digest);
        let fold_schedule_meta = self.fold_schedule.meta_words();
        out.push(F::from_u64(fold_schedule_meta[0]));
        out.push(F::from_u64(fold_schedule_meta[1]));
        out.push(F::from_u64(self.semantic_step_count));
        for summary in &self.chunk_summaries {
            extend_spartan2_chunk_summary_fields(&mut out, summary);
        }
        out
    }

    pub fn expected_terminal_handle_digest(&self) -> [F; FIXED_SHAPE_DIGEST_FIELD_LEN] {
        fixed_shape_terminal_handle_digest_fields(
            digest_fields_as_digest32(self.initial_handle_digest),
            &self.chunk_summaries,
        )
    }
}

impl Spartan2DeciderWitness {
    pub fn digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/decider/spartan2/witness");
        tr.append_u64s(
            b"neo.fold.next/decider/spartan2/witness/base_component_count",
            &[self.base_component_digests.len() as u64],
        );
        tr.append_u64s(
            b"neo.fold.next/decider/spartan2/witness/chunk_transition_count",
            &[self.chunk_transition_bindings.len() as u64],
        );
        for digest in &self.base_component_digests {
            tr.append_message(b"neo.fold.next/decider/spartan2/witness/base_component_digest", digest);
        }
        for binding in &self.chunk_transition_bindings {
            tr.append_message(
                b"neo.fold.next/decider/spartan2/witness/chunk_transition_binding",
                &binding.digest(),
            );
        }
        tr.digest32()
    }

    pub fn public_io(&self) -> Vec<F> {
        let mut out = Vec::with_capacity(
            2 + self.base_component_digests.len() * packed_bytes_field_len(32)
                + self.chunk_transition_bindings.len() * Spartan2ChunkTransitionBinding::packed_field_len(),
        );
        out.push(F::from_u64(self.base_component_digests.len() as u64));
        out.push(F::from_u64(self.chunk_transition_bindings.len() as u64));
        for digest in &self.base_component_digests {
            extend_packed_bytes_as_fields(&mut out, digest);
        }
        for binding in &self.chunk_transition_bindings {
            out.extend(binding.packed_fields());
        }
        out
    }
}

impl Spartan2DeciderShape {
    pub fn digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/decider/spartan2/shape");
        tr.append_u64s(
            b"neo.fold.next/decider/spartan2/shape/base_component_count",
            &[self.base_component_count as u64],
        );
        tr.append_u64s(
            b"neo.fold.next/decider/spartan2/shape/chunk_transition_count",
            &[self.chunk_transition_count as u64],
        );
        tr.append_u64s(
            b"neo.fold.next/decider/spartan2/shape/public_io_len",
            &[self.public_io_len() as u64],
        );
        tr.append_u64s(
            b"neo.fold.next/decider/spartan2/shape/backend_public_io_len",
            &[self.backend_public_io_len() as u64],
        );
        tr.append_u64s(
            b"neo.fold.next/decider/spartan2/shape/backend_witness_field_len",
            &[self.backend_witness_field_len() as u64],
        );
        tr.digest32()
    }

    pub fn statement_public_io_len(&self) -> usize {
        3 * packed_bytes_field_len(32)
            + 2 * FIXED_SHAPE_DIGEST_FIELD_LEN
            + 3
            + self.chunk_transition_count * spartan2_chunk_summary_field_len()
    }

    pub fn witness_public_io_len(&self) -> usize {
        2 + self.base_component_count * packed_bytes_field_len(32)
            + self.chunk_transition_count * Spartan2ChunkTransitionBinding::packed_field_len()
    }

    pub fn public_io_len(&self) -> usize {
        self.statement_public_io_len() + self.witness_public_io_len()
    }

    pub fn backend_public_io_len(&self) -> usize {
        self.statement_public_io_len() + (2 * POSEIDON2_DIGEST_LEN)
    }

    pub fn backend_witness_field_len(&self) -> usize {
        2 + self.base_component_count * packed_bytes_field_len(32)
            + self.chunk_transition_count * Spartan2ChunkTransitionBinding::packed_field_len()
    }
}

impl Spartan2DeciderBackendWitness {
    pub fn digest_fields(&self) -> [F; POSEIDON2_DIGEST_LEN] {
        poseidon2_hash(&self.packed_fields())
    }

    pub fn packed_fields(&self) -> Vec<F> {
        let mut out = Vec::with_capacity(
            2 + self.base_component_digests.len() * packed_bytes_field_len(32)
                + self.chunk_transition_bindings.len() * Spartan2ChunkTransitionBinding::packed_field_len(),
        );
        out.push(F::from_u64(self.base_component_count));
        out.push(F::from_u64(self.chunk_transition_count));
        for digest in &self.base_component_digests {
            extend_packed_bytes_as_fields(&mut out, digest);
        }
        for binding in &self.chunk_transition_bindings {
            out.extend(binding.packed_fields());
        }
        out
    }
}

impl Spartan2DeciderBackendRelation {
    pub fn shape(&self) -> Spartan2DeciderShape {
        Spartan2DeciderShape {
            base_component_count: self.witness.base_component_digests.len(),
            chunk_transition_count: self.witness.chunk_transition_bindings.len(),
        }
    }

    pub fn witness_digest_fields(&self) -> [F; POSEIDON2_DIGEST_LEN] {
        self.witness.digest_fields()
    }

    pub fn binding_digest_fields(&self) -> [F; POSEIDON2_DIGEST_LEN] {
        let mut preimage = self.statement.public_io();
        preimage.extend(self.semantic_digest_fields());
        preimage.extend(self.witness_digest_fields());
        poseidon2_hash(&preimage)
    }

    pub fn public_io(&self) -> Vec<F> {
        let mut out = self.statement.public_io();
        out.extend(self.semantic_digest_fields());
        out.extend(self.binding_digest_fields());
        out
    }

    pub fn expected_final_proof_digest(&self) -> [u8; 32] {
        digest_fixed_shape_final_proof(
            &self.statement.relation_digest,
            self.statement.chunk_summaries.len() as u64,
            &self.statement.chunk_summaries,
            &self.witness.base_component_digests,
            &transition_witness_digests(&self.witness.chunk_transition_bindings),
        )
    }

    pub fn expected_terminal_handle_digest(&self) -> [F; FIXED_SHAPE_DIGEST_FIELD_LEN] {
        self.statement.expected_terminal_handle_digest()
    }

    pub fn semantic_digest_fields(&self) -> [F; POSEIDON2_DIGEST_LEN] {
        backend_semantic_digest_fields(
            &self.statement.relation_digest,
            &self.statement.chunk_summaries,
            &self.witness,
        )
    }
}

pub fn validate_spartan2_backend_relation_surface(
    relation: &Spartan2DeciderBackendRelation,
) -> Result<(), Spartan2BackendBindingShellError> {
    if relation.witness.base_component_count != relation.witness.base_component_digests.len() as u64 {
        return Err(Spartan2BackendBindingShellError::RelationSurface(
            "private base component count does not match carried base component digests".into(),
        ));
    }
    if relation.witness.chunk_transition_count != relation.witness.chunk_transition_bindings.len() as u64 {
        return Err(Spartan2BackendBindingShellError::RelationSurface(
            "private chunk transition count does not match carried chunk transition bindings".into(),
        ));
    }
    if relation.witness.chunk_transition_count != relation.statement.chunk_summaries.len() as u64 {
        return Err(Spartan2BackendBindingShellError::RelationSurface(
            "private chunk transition count does not match carried public chunk summaries".into(),
        ));
    }
    validate_chunk_transition_bindings(
        &relation.statement.chunk_summaries,
        &relation.witness.chunk_transition_bindings,
    )
    .map_err(Spartan2BackendBindingShellError::RelationSurface)?;
    validate_spartan2_chunk_layout(
        relation.statement.fold_schedule,
        relation.statement.semantic_step_count,
        &relation.statement.chunk_summaries,
    )
    .map_err(Spartan2BackendBindingShellError::RelationSurface)?;
    if relation.statement.terminal_handle_digest != relation.expected_terminal_handle_digest() {
        return Err(Spartan2BackendBindingShellError::RelationSurface(
            "public terminal handle digest does not match the carried chunk summary chain".into(),
        ));
    }
    if relation.statement.final_proof_digest != relation.expected_final_proof_digest() {
        return Err(Spartan2BackendBindingShellError::RelationSurface(
            "public final proof digest does not match the carried fixed-shape backend relation".into(),
        ));
    }
    Ok(())
}

pub(super) fn validate_spartan2_decider_target_surface(
    target: &Spartan2DeciderTarget,
) -> Result<(), Spartan2DeciderError> {
    if target.witness.chunk_transition_bindings.len() != target.statement.chunk_summaries.len() {
        return Err(Spartan2DeciderError::RelationSurface(
            "private chunk transition count does not match carried public chunk summaries".into(),
        ));
    }
    validate_chunk_transition_bindings(
        &target.statement.chunk_summaries,
        &target.witness.chunk_transition_bindings,
    )
    .map_err(Spartan2DeciderError::RelationSurface)?;
    validate_spartan2_chunk_layout(
        target.statement.fold_schedule,
        target.statement.semantic_step_count,
        &target.statement.chunk_summaries,
    )
    .map_err(Spartan2DeciderError::RelationSurface)?;
    if target.statement.terminal_handle_digest != target.statement.expected_terminal_handle_digest() {
        return Err(Spartan2DeciderError::RelationSurface(
            "terminal handle digest does not match the carried chunk summary chain".into(),
        ));
    }
    if target.statement.final_proof_digest != target.expected_final_proof_digest() {
        return Err(Spartan2DeciderError::FinalProofDigestMismatch);
    }
    Ok(())
}

impl Spartan2DeciderTarget {
    pub fn shape(&self) -> Spartan2DeciderShape {
        Spartan2DeciderShape {
            base_component_count: self.witness.base_component_digests.len(),
            chunk_transition_count: self.witness.chunk_transition_bindings.len(),
        }
    }

    pub fn digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/decider/spartan2/target");
        tr.append_message(
            b"neo.fold.next/decider/spartan2/target/statement_digest",
            &self.statement.digest(),
        );
        tr.append_message(
            b"neo.fold.next/decider/spartan2/target/witness_digest",
            &self.witness.digest(),
        );
        tr.digest32()
    }

    pub fn backend_witness(&self) -> Spartan2DeciderBackendWitness {
        Spartan2DeciderBackendWitness {
            base_component_count: self.witness.base_component_digests.len() as u64,
            chunk_transition_count: self.witness.chunk_transition_bindings.len() as u64,
            base_component_digests: self.witness.base_component_digests.clone(),
            chunk_transition_bindings: self.witness.chunk_transition_bindings.clone(),
        }
    }

    pub fn backend_relation(&self) -> Spartan2DeciderBackendRelation {
        Spartan2DeciderBackendRelation {
            statement: self.statement.clone(),
            witness: self.backend_witness(),
        }
    }

    pub fn relation(&self) -> Result<Spartan2DeciderRelation, Spartan2DeciderError> {
        build_spartan2_decider_relation(
            self.statement.public_statement_digest,
            self.statement.relation_digest,
            self.statement.final_proof_digest,
            self.statement.initial_handle_digest,
            self.statement.terminal_handle_digest,
            self.statement.fold_schedule,
            self.statement.semantic_step_count,
            self.statement.chunk_summaries.clone(),
            self.witness.base_component_digests.clone(),
            transition_witness_digests(&self.witness.chunk_transition_bindings),
        )
    }

    pub fn public_io(&self) -> Vec<F> {
        let mut out = self.statement.public_io();
        out.extend(self.witness.public_io());
        out
    }

    pub fn backend_public_io(&self) -> Vec<F> {
        self.backend_relation().public_io()
    }

    pub fn backend_semantic_digest_fields(&self) -> [F; POSEIDON2_DIGEST_LEN] {
        self.backend_relation().semantic_digest_fields()
    }

    pub fn backend_witness_digest_fields(&self) -> [F; POSEIDON2_DIGEST_LEN] {
        self.backend_relation().witness_digest_fields()
    }

    pub fn backend_binding_digest_fields(&self) -> [F; POSEIDON2_DIGEST_LEN] {
        self.backend_relation().binding_digest_fields()
    }

    pub fn expected_final_proof_digest(&self) -> [u8; 32] {
        self.backend_relation().expected_final_proof_digest()
    }

    pub fn expected_terminal_handle_digest(&self) -> [F; FIXED_SHAPE_DIGEST_FIELD_LEN] {
        self.statement.expected_terminal_handle_digest()
    }
}

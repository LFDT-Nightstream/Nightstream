use neo_fold_next::nightstream::rv64im::{
    build_rv64im_bound_side_proof_bundle_from_accepted_artifact, build_rv64im_nightstream_linkage_claims,
    build_rv64im_nightstream_statement_from_final, build_rv64im_side_spartan_from_accepted_artifact,
    measure_rv64im_side_spartan_circuit_constraints, rv64im_nightstream_linkage_root, rv64im_verifier_context_digest,
    Rv64imAuthoritativeSidePublicInstance, Rv64imAuthoritativeSideStatement, Rv64imSideProofBundle,
};
use neo_fold_next::nightstream::NightstreamStatement;
use neo_fold_next::rv64im::final_relation::{
    prove_rv64im_final_statement_from_accepted, Rv64imFinalProof, Rv64imFinalStatement,
};
use neo_fold_next::rv64im::{
    build_mixed_opcode_perf_source_case, build_rv64im_accepted_proof_artifact, measure_rv64im_spartan2_decider_circuit,
    prove_rv64im_public_proof, Rv64imAcceptedProofArtifact, Rv64imMainRelationCircuitMetrics, Rv64imProofInput,
    SimpleKernelError,
};

pub struct Rv64imN2Fixture {
    pub accepted_artifact: Rv64imAcceptedProofArtifact,
    pub final_statement: Rv64imFinalStatement,
    pub final_proof: Rv64imFinalProof,
    pub nightstream_statement: NightstreamStatement,
    pub side_bundle: Rv64imSideProofBundle,
}

impl Rv64imN2Fixture {
    pub fn measure_main_relation(&self) -> Result<Rv64imMainRelationCircuitMetrics, SimpleKernelError> {
        measure_rv64im_spartan2_decider_circuit(&self.final_statement, &self.final_proof)
    }

    pub fn real_rows(&self) -> usize {
        self.accepted_artifact.root_execution.execution_rows.len()
    }

    pub fn packaged_final_main_claims_total(&self) -> usize {
        self.accepted_artifact
            .main_lane
            .packaged
            .statement
            .final_main_claims
            .len()
    }

    pub fn packaged_dec_children_total(&self) -> usize {
        self.accepted_artifact
            .main_lane
            .packaged
            .proof
            .session
            .chunks
            .iter()
            .map(|chunk| chunk.dec.children.len())
            .sum()
    }

    pub fn child_claim_count(&self) -> usize {
        self.packaged_dec_children_total()
    }

    pub fn build_side_debug_inputs(
        &self,
    ) -> Result<(Rv64imAuthoritativeSideStatement, Rv64imAuthoritativeSidePublicInstance), SimpleKernelError> {
        build_rv64im_side_spartan_from_accepted_artifact(
            &self.nightstream_statement,
            &self.side_bundle,
            &self.accepted_artifact,
        )
    }

    pub fn build_side_audit_inputs(
        &self,
    ) -> Result<
        (
            neo_fold_next::nightstream::rv64im::Rv64imWitnessBackedSideBridgeStatement,
            neo_fold_next::nightstream::rv64im::audit::Rv64imDirectSideRelationWitness,
        ),
        SimpleKernelError,
    > {
        neo_fold_next::nightstream::rv64im::audit::build_rv64im_direct_side_relation_from_accepted_artifact(
            &self.nightstream_statement,
            &self.side_bundle,
            &self.accepted_artifact,
        )
    }

    pub fn measure_side_relation_constraints(&self) -> Result<usize, SimpleKernelError> {
        let (statement, witness) = self.build_side_debug_inputs()?;
        measure_rv64im_side_spartan_circuit_constraints(&statement, &witness)
    }
}

pub fn build_rv64im_n2_fixture() -> Result<Rv64imN2Fixture, SimpleKernelError> {
    let source = build_mixed_opcode_perf_source_case(2);
    let max_steps = source.program_words.len();
    let input = Rv64imProofInput { source, max_steps };
    let public_proof = prove_rv64im_public_proof(&input)?;
    let accepted_artifact = build_rv64im_accepted_proof_artifact(&public_proof)?;
    let (final_statement, final_proof) = prove_rv64im_final_statement_from_accepted(&accepted_artifact)?;
    let linkage_claims = build_rv64im_nightstream_linkage_claims(&final_statement, &final_proof)?;
    let linkage_root = rv64im_nightstream_linkage_root(final_proof.kernel_export.digest, &linkage_claims);
    let nightstream_statement = build_rv64im_nightstream_statement_from_final(
        public_proof.statement.digest,
        rv64im_verifier_context_digest(public_proof.statement.root_params_id),
        &final_statement,
        &final_proof,
        linkage_root,
        [0u8; 32],
    )?;
    let side_bundle =
        build_rv64im_bound_side_proof_bundle_from_accepted_artifact(&nightstream_statement, &accepted_artifact)?;
    Ok(Rv64imN2Fixture {
        accepted_artifact,
        final_statement,
        final_proof,
        nightstream_statement,
        side_bundle,
    })
}

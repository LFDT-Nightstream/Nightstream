#![allow(dead_code)]

use std::sync::OnceLock;

use neo_fold_next::nightstream::rv64im::audit::{
    build_rv64im_nightstream_statement_from_final, measure_rv64im_side_binding_circuit_constraints,
};
use neo_fold_next::nightstream::rv64im::{
    build_rv64im_side_proof, rv64im_verifier_context_digest, Rv64imSideBindingStatement, Rv64imSideOpeningPublic,
    Rv64imSideProof,
};
use neo_fold_next::nightstream::NightstreamStatement;
use neo_fold_next::proof::FoldSchedule;
use neo_fold_next::rv64im::audit::{prove_rv64im_public_proof_and_published_seam_with_perf, Rv64imPublishedProofSeam};
use neo_fold_next::rv64im::final_relation::{
    prove_rv64im_final_statement_from_accepted, Rv64imFinalBuildProof, Rv64imFinalStatement,
};
use neo_fold_next::rv64im::{
    build_mixed_opcode_perf_source_case, build_rv64im_accepted_proof_artifact, build_rv64im_chunk_step_ivc_relations,
    build_rv64im_main_recursion_f_prime_advices, build_rv64im_main_recursion_f_prime_advices_single_step,
    prove_rv64im_public_proof, prove_rv64im_public_proof_with_options, setup_rv64im_ivc_snark_from_final_cached,
    Rv64imAcceptedProofArtifact, Rv64imChunkStepIvcRelation, Rv64imIvcSnarkKeyPair, Rv64imMainRecursionFPrimeAdvice,
    Rv64imProofInput, Rv64imPublicProofOptions, SimpleKernelError,
};

#[derive(Clone)]
pub struct Rv64imN2Fixture {
    pub accepted_artifact: Rv64imAcceptedProofArtifact,
    pub final_statement: Rv64imFinalStatement,
    pub final_proof: Rv64imFinalBuildProof,
    pub ivc_recursion_snark_keys: Rv64imIvcSnarkKeyPair,
    pub nightstream_statement: NightstreamStatement,
    pub side_proof: Rv64imSideProof,
}

impl Rv64imN2Fixture {
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
    ) -> Result<
        (
            Rv64imSideBindingStatement,
            Rv64imSideOpeningPublic,
            Vec<neo_fold_next::rv64im::FamilyEvalClaimWitness>,
        ),
        SimpleKernelError,
    > {
        let (_, witness) =
            neo_fold_next::nightstream::rv64im::audit::build_rv64im_side_eval_claim_relation_from_accepted_artifact(
                &self.accepted_artifact,
            )?;
        Ok((
            self.side_proof
                .binding_statement(&self.nightstream_statement, &self.accepted_artifact.statement)?,
            self.side_proof.opening_public().clone(),
            witness.claim_witnesses,
        ))
    }

    pub fn build_side_audit_inputs(
        &self,
    ) -> Result<
        (
            (),
            neo_fold_next::nightstream::rv64im::audit::Rv64imSideEvalClaimRelationWitness,
        ),
        SimpleKernelError,
    > {
        neo_fold_next::nightstream::rv64im::audit::build_rv64im_side_eval_claim_relation_from_accepted_artifact(
            &self.accepted_artifact,
        )
        .map(|(_, witness)| ((), witness))
    }

    pub fn measure_side_relation_constraints(&self) -> Result<usize, SimpleKernelError> {
        let (statement, public, _) = self.build_side_debug_inputs()?;
        measure_rv64im_side_binding_circuit_constraints(&statement, &public)
    }
}

pub fn build_rv64im_n2_fixture() -> Result<Rv64imN2Fixture, SimpleKernelError> {
    static FIXTURE: OnceLock<Rv64imN2Fixture> = OnceLock::new();
    Ok(FIXTURE
        .get_or_init(|| {
            let source = build_mixed_opcode_perf_source_case(2);
            let max_steps = source.program_words.len();
            let input = Rv64imProofInput { source, max_steps };
            let public_proof = prove_rv64im_public_proof(&input).expect("prove rv64im n=2 public proof fixture");
            let accepted_artifact =
                build_rv64im_accepted_proof_artifact(&public_proof).expect("build rv64im n=2 accepted artifact");
            let (final_statement, final_proof) = prove_rv64im_final_statement_from_accepted(&accepted_artifact)
                .expect("build rv64im n=2 final statement");
            let published_statement = neo_fold_next::rv64im::Rv64imAccumulatorPublicStatement::from_final_artifacts(
                &final_statement,
                &final_proof,
                public_proof.statement.final_pc,
            )
            .expect("build rv64im n=2 published statement");
            let ivc_recursion_snark_keys = setup_rv64im_ivc_snark_from_final_cached(&final_statement, &final_proof)
                .expect("setup rv64im n=2 IVC recursion SNARK");
            let nightstream_statement = build_rv64im_nightstream_statement_from_final(
                public_proof.statement.digest,
                rv64im_verifier_context_digest(
                    public_proof.statement.root_params_id,
                    &published_statement,
                    &ivc_recursion_snark_keys.as_ref().1,
                )
                .expect("digest rv64im n=2 verifier context"),
                &final_statement,
                &final_proof,
                [0u8; 32],
            )
            .expect("build rv64im n=2 nightstream statement");
            let side_proof = build_rv64im_side_proof(&nightstream_statement, &accepted_artifact)
                .expect("build rv64im n=2 side proof");
            Rv64imN2Fixture {
                accepted_artifact,
                final_statement,
                final_proof,
                ivc_recursion_snark_keys,
                nightstream_statement,
                side_proof,
            }
        })
        .clone())
}

#[allow(dead_code)]
pub fn build_rv64im_n2_published_seam() -> Result<Rv64imPublishedProofSeam, SimpleKernelError> {
    let source = build_mixed_opcode_perf_source_case(2);
    let max_steps = source.program_words.len();
    let input = Rv64imProofInput { source, max_steps };
    let ((_, seam), _) = prove_rv64im_public_proof_and_published_seam_with_perf(&input)?;
    Ok(seam)
}

pub fn build_rv64im_n2_chunk_step_relations() -> Result<Vec<Rv64imChunkStepIvcRelation>, SimpleKernelError> {
    static RELATIONS: OnceLock<Vec<Rv64imChunkStepIvcRelation>> = OnceLock::new();
    Ok(RELATIONS
        .get_or_init(|| {
            let fixture = build_rv64im_n2_fixture().expect("build rv64im n=2 fixture");
            build_rv64im_chunk_step_ivc_relations(&fixture.final_statement, &fixture.final_proof)
                .expect("build rv64im n=2 chunk-step chain")
        })
        .clone())
}

pub fn build_rv64im_n2_f_prime_advices() -> Result<Vec<Rv64imMainRecursionFPrimeAdvice>, SimpleKernelError> {
    static ADVICES: OnceLock<Vec<Rv64imMainRecursionFPrimeAdvice>> = OnceLock::new();
    Ok(ADVICES
        .get_or_init(|| {
            let relations = build_rv64im_n2_chunk_step_relations().expect("build rv64im n=2 relations");
            build_rv64im_main_recursion_f_prime_advices(&relations).expect("build rv64im n=2 F' advices")
        })
        .clone())
}

#[derive(Clone)]
pub struct Rv64imRowsPerChunkOneN2Fixture {
    pub accepted_artifact: Rv64imAcceptedProofArtifact,
    pub final_statement: Rv64imFinalStatement,
    pub final_proof: Rv64imFinalBuildProof,
    pub max_steps: usize,
}

pub fn build_rv64im_rows_per_chunk_one_n2_fixture() -> Result<Rv64imRowsPerChunkOneN2Fixture, SimpleKernelError> {
    static FIXTURE: OnceLock<Rv64imRowsPerChunkOneN2Fixture> = OnceLock::new();
    Ok(FIXTURE
        .get_or_init(|| {
            let source = build_mixed_opcode_perf_source_case(2);
            let max_steps = source.program_words.len();
            let input = Rv64imProofInput { source, max_steps };
            let public_proof = prove_rv64im_public_proof_with_options(
                &input,
                Rv64imPublicProofOptions {
                    root_fold_schedule: FoldSchedule::RowsPerChunk(1),
                },
            )
            .expect("prove rv64im n=2 public proof with RowsPerChunk(1)");
            let accepted_artifact =
                build_rv64im_accepted_proof_artifact(&public_proof).expect("build accepted artifact");
            let (final_statement, final_proof) =
                prove_rv64im_final_statement_from_accepted(&accepted_artifact).expect("build final statement");
            Rv64imRowsPerChunkOneN2Fixture {
                accepted_artifact,
                final_statement,
                final_proof,
                max_steps,
            }
        })
        .clone())
}

pub fn build_rv64im_rows_per_chunk_one_n2_chunk_step_relations(
) -> Result<Vec<Rv64imChunkStepIvcRelation>, SimpleKernelError> {
    static RELATIONS: OnceLock<Vec<Rv64imChunkStepIvcRelation>> = OnceLock::new();
    Ok(RELATIONS
        .get_or_init(|| {
            let fixture =
                build_rv64im_rows_per_chunk_one_n2_fixture().expect("build rv64im RowsPerChunk(1) n=2 fixture");
            build_rv64im_chunk_step_ivc_relations(&fixture.final_statement, &fixture.final_proof)
                .expect("build rv64im RowsPerChunk(1) n=2 chunk-step chain")
        })
        .clone())
}

pub fn build_rv64im_rows_per_chunk_one_n2_f_prime_advices(
) -> Result<Vec<Rv64imMainRecursionFPrimeAdvice>, SimpleKernelError> {
    static ADVICES: OnceLock<Vec<Rv64imMainRecursionFPrimeAdvice>> = OnceLock::new();
    Ok(ADVICES
        .get_or_init(|| {
            let relations = build_rv64im_rows_per_chunk_one_n2_chunk_step_relations()
                .expect("build rv64im RowsPerChunk(1) n=2 relations");
            build_rv64im_main_recursion_f_prime_advices_single_step(&relations)
                .expect("build rv64im RowsPerChunk(1) n=2 F' advices")
        })
        .clone())
}

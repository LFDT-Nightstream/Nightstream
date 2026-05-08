#![allow(dead_code)]

use std::sync::OnceLock;

use neo_fold_next::proof::FoldSchedule;
use neo_fold_next::public_proof::rv32im::audit::{
    build_rv32im_nightstream_statement_from_final, measure_rv32im_side_binding_circuit_constraints,
};
use neo_fold_next::public_proof::rv32im::{
    build_rv32im_side_proof, rv32im_verifier_context_digest, Rv32imSideBindingStatement, Rv32imSideOpeningPublic,
    Rv32imSideProof,
};
use neo_fold_next::public_proof::NightstreamStatement;
use neo_fold_next::rv32im::audit::{prove_rv32im_public_proof_and_published_seam_with_perf, Rv32imPublishedProofSeam};
use neo_fold_next::rv32im::final_relation::{
    prove_rv32im_final_statement_from_accepted, Rv32imFinalBuildProof, Rv32imFinalStatement,
};
use neo_fold_next::rv32im::{
    build_mixed_opcode_perf_source_case, build_rv32im_accepted_proof_artifact, build_rv32im_chunk_step_ivc_relations,
    build_rv32im_main_recursion_f_prime_advices, build_rv32im_main_recursion_f_prime_advices_single_step,
    prove_rv32im_public_proof, prove_rv32im_public_proof_with_options, setup_rv32im_ivc_snark_from_final_cached,
    Rv32imAcceptedProofArtifact, Rv32imChunkStepIvcRelation, Rv32imIvcSnarkKeyPair, Rv32imMainRecursionFPrimeAdvice,
    Rv32imProofInput, Rv32imPublicProofOptions, SimpleKernelError,
};

#[derive(Clone)]
pub struct Rv32imN2Fixture {
    pub accepted_artifact: Rv32imAcceptedProofArtifact,
    pub final_statement: Rv32imFinalStatement,
    pub final_proof: Rv32imFinalBuildProof,
    pub ivc_recursion_snark_keys: Rv32imIvcSnarkKeyPair,
    pub nightstream_statement: NightstreamStatement,
    pub side_proof: Rv32imSideProof,
}

impl Rv32imN2Fixture {
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
            Rv32imSideBindingStatement,
            Rv32imSideOpeningPublic,
            Vec<neo_fold_next::rv32im::FamilyEvalClaimWitness>,
        ),
        SimpleKernelError,
    > {
        let (_, witness) =
            neo_fold_next::public_proof::rv32im::audit::build_rv32im_side_eval_claim_relation_from_accepted_artifact(
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
            neo_fold_next::public_proof::rv32im::audit::Rv32imSideEvalClaimRelationWitness,
        ),
        SimpleKernelError,
    > {
        neo_fold_next::public_proof::rv32im::audit::build_rv32im_side_eval_claim_relation_from_accepted_artifact(
            &self.accepted_artifact,
        )
        .map(|(_, witness)| ((), witness))
    }

    pub fn measure_side_relation_constraints(&self) -> Result<usize, SimpleKernelError> {
        let (statement, public, _) = self.build_side_debug_inputs()?;
        measure_rv32im_side_binding_circuit_constraints(&statement, &public)
    }
}

pub fn build_rv32im_n2_fixture() -> Result<Rv32imN2Fixture, SimpleKernelError> {
    static FIXTURE: OnceLock<Rv32imN2Fixture> = OnceLock::new();
    Ok(FIXTURE
        .get_or_init(|| {
            let source = build_mixed_opcode_perf_source_case(2);
            let max_steps = source.program_words.len();
            let input = Rv32imProofInput { source, max_steps };
            let public_proof = prove_rv32im_public_proof(&input).expect("prove rv32im n=2 public proof fixture");
            let accepted_artifact =
                build_rv32im_accepted_proof_artifact(&public_proof).expect("build rv32im n=2 accepted artifact");
            let (final_statement, final_proof) = prove_rv32im_final_statement_from_accepted(&accepted_artifact)
                .expect("build rv32im n=2 final statement");
            let published_statement = neo_fold_next::rv32im::Rv32imAccumulatorPublicStatement::from_final_artifacts(
                &final_statement,
                &final_proof,
                public_proof.statement.final_pc,
            )
            .expect("build rv32im n=2 published statement");
            let ivc_recursion_snark_keys = setup_rv32im_ivc_snark_from_final_cached(&final_statement, &final_proof)
                .expect("setup rv32im n=2 IVC recursion SNARK");
            let nightstream_statement = build_rv32im_nightstream_statement_from_final(
                public_proof.statement.digest,
                rv32im_verifier_context_digest(
                    public_proof.statement.root_params_id,
                    &published_statement,
                    &ivc_recursion_snark_keys.as_ref().1,
                )
                .expect("digest rv32im n=2 verifier context"),
                &final_statement,
                &final_proof,
                [0u8; 32],
            )
            .expect("build rv32im n=2 nightstream statement");
            let side_proof = build_rv32im_side_proof(&nightstream_statement, &accepted_artifact)
                .expect("build rv32im n=2 side proof");
            Rv32imN2Fixture {
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
pub fn build_rv32im_n2_published_seam() -> Result<Rv32imPublishedProofSeam, SimpleKernelError> {
    let source = build_mixed_opcode_perf_source_case(2);
    let max_steps = source.program_words.len();
    let input = Rv32imProofInput { source, max_steps };
    let ((_, seam), _) = prove_rv32im_public_proof_and_published_seam_with_perf(&input)?;
    Ok(seam)
}

pub fn build_rv32im_n2_chunk_step_relations() -> Result<Vec<Rv32imChunkStepIvcRelation>, SimpleKernelError> {
    static RELATIONS: OnceLock<Vec<Rv32imChunkStepIvcRelation>> = OnceLock::new();
    Ok(RELATIONS
        .get_or_init(|| {
            let fixture = build_rv32im_n2_fixture().expect("build rv32im n=2 fixture");
            build_rv32im_chunk_step_ivc_relations(&fixture.final_statement, &fixture.final_proof)
                .expect("build rv32im n=2 chunk-step chain")
        })
        .clone())
}

pub fn build_rv32im_n2_f_prime_advices() -> Result<Vec<Rv32imMainRecursionFPrimeAdvice>, SimpleKernelError> {
    static ADVICES: OnceLock<Vec<Rv32imMainRecursionFPrimeAdvice>> = OnceLock::new();
    Ok(ADVICES
        .get_or_init(|| {
            let relations = build_rv32im_n2_chunk_step_relations().expect("build rv32im n=2 relations");
            build_rv32im_main_recursion_f_prime_advices(&relations).expect("build rv32im n=2 F' advices")
        })
        .clone())
}

#[derive(Clone)]
pub struct Rv32imRowsPerChunkOneN2Fixture {
    pub accepted_artifact: Rv32imAcceptedProofArtifact,
    pub final_statement: Rv32imFinalStatement,
    pub final_proof: Rv32imFinalBuildProof,
    pub max_steps: usize,
}

pub fn build_rv32im_rows_per_chunk_one_n2_fixture() -> Result<Rv32imRowsPerChunkOneN2Fixture, SimpleKernelError> {
    static FIXTURE: OnceLock<Rv32imRowsPerChunkOneN2Fixture> = OnceLock::new();
    Ok(FIXTURE
        .get_or_init(|| {
            let source = build_mixed_opcode_perf_source_case(2);
            let max_steps = source.program_words.len();
            let input = Rv32imProofInput { source, max_steps };
            let public_proof = prove_rv32im_public_proof_with_options(
                &input,
                Rv32imPublicProofOptions {
                    root_fold_schedule: FoldSchedule::RowsPerChunk(1),
                },
            )
            .expect("prove rv32im n=2 public proof with RowsPerChunk(1)");
            let accepted_artifact =
                build_rv32im_accepted_proof_artifact(&public_proof).expect("build accepted artifact");
            let (final_statement, final_proof) =
                prove_rv32im_final_statement_from_accepted(&accepted_artifact).expect("build final statement");
            Rv32imRowsPerChunkOneN2Fixture {
                accepted_artifact,
                final_statement,
                final_proof,
                max_steps,
            }
        })
        .clone())
}

pub fn build_rv32im_rows_per_chunk_one_n2_chunk_step_relations(
) -> Result<Vec<Rv32imChunkStepIvcRelation>, SimpleKernelError> {
    static RELATIONS: OnceLock<Vec<Rv32imChunkStepIvcRelation>> = OnceLock::new();
    Ok(RELATIONS
        .get_or_init(|| {
            let fixture =
                build_rv32im_rows_per_chunk_one_n2_fixture().expect("build rv32im RowsPerChunk(1) n=2 fixture");
            build_rv32im_chunk_step_ivc_relations(&fixture.final_statement, &fixture.final_proof)
                .expect("build rv32im RowsPerChunk(1) n=2 chunk-step chain")
        })
        .clone())
}

pub fn build_rv32im_rows_per_chunk_one_n2_f_prime_advices(
) -> Result<Vec<Rv32imMainRecursionFPrimeAdvice>, SimpleKernelError> {
    static ADVICES: OnceLock<Vec<Rv32imMainRecursionFPrimeAdvice>> = OnceLock::new();
    Ok(ADVICES
        .get_or_init(|| {
            let relations = build_rv32im_rows_per_chunk_one_n2_chunk_step_relations()
                .expect("build rv32im RowsPerChunk(1) n=2 relations");
            build_rv32im_main_recursion_f_prime_advices_single_step(&relations)
                .expect("build rv32im RowsPerChunk(1) n=2 F' advices")
        })
        .clone())
}

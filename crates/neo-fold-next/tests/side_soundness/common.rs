use neo_fold_next::proof::FoldSchedule;
use neo_fold_next::public_proof::rv32im::audit::{
    build_rv32im_side_opening_relation_from_accepted_artifact, setup_rv32im_side_opening_spartan,
};
use neo_fold_next::public_proof::rv32im::{
    build_rv32im_nightstream_from_public_proof_with_perf, build_rv32im_side_proof, Rv32imSideBindingStatement,
    Rv32imSideOpeningPublic, Rv32imSideOpeningSpartanVerifierKey, Rv32imSideProof,
};
use neo_fold_next::public_proof::NightstreamStatement;
use neo_fold_next::rv32im::{
    build_rv32im_accepted_proof_artifact, parity_source_cases, prove_rv32im_public_proof_with_options,
    Rv32imAcceptedProofArtifact, Rv32imProofInput, Rv32imProofStatement, Rv32imPublicProofOptions, SimpleKernelError,
};

pub struct SideFixture {
    pub accepted_artifact: Rv32imAcceptedProofArtifact,
    pub nightstream_statement: NightstreamStatement,
    pub public_statement: Rv32imProofStatement,
    pub side_proof: Rv32imSideProof,
}

impl SideFixture {
    pub fn side_statement(&self) -> Result<Rv32imSideBindingStatement, SimpleKernelError> {
        self.side_proof
            .binding_statement(&self.nightstream_statement, &self.public_statement)
    }

    pub fn side_public(&self) -> &Rv32imSideOpeningPublic {
        self.side_proof.opening_public()
    }

    pub fn side_opening_vk(&self) -> Rv32imSideOpeningSpartanVerifierKey {
        let (statement, witness) = build_rv32im_side_opening_relation_from_accepted_artifact(&self.accepted_artifact)
            .expect("build side opening relation for opening vk");
        let (_, vk) = setup_rv32im_side_opening_spartan(&statement, &witness).expect("setup side opening spartan");
        vk
    }
}

pub fn source_case(name: &str) -> neo_fold_next::rv32im::Rv32imParitySourceCase {
    parity_source_cases()
        .into_iter()
        .find(|case| case.manifest.name == name)
        .unwrap_or_else(|| panic!("missing parity source case {name}"))
}

pub fn alternate_case_name(exclude: &str) -> String {
    parity_source_cases()
        .into_iter()
        .find(|case| case.manifest.name != exclude)
        .unwrap_or_else(|| panic!("missing alternate parity source case for {exclude}"))
        .manifest
        .name
}

pub fn proof_input(name: &str) -> Rv32imProofInput {
    let source = source_case(name);
    let max_steps = source.program_words.len();
    Rv32imProofInput { source, max_steps }
}

pub fn build_side_fixture(name: &str) -> SideFixture {
    let public_proof = prove_rv32im_public_proof_with_options(
        &proof_input(name),
        Rv32imPublicProofOptions {
            root_fold_schedule: FoldSchedule::RowsPerChunk(1),
        },
    )
    .expect("prove rv32im public proof for side soundness fixture");
    let accepted_artifact = build_rv32im_accepted_proof_artifact(&public_proof)
        .expect("build accepted artifact for side soundness fixture");
    let ((nightstream_statement, _nightstream_proof), _) =
        build_rv32im_nightstream_from_public_proof_with_perf(&public_proof)
            .expect("build nightstream proof for side soundness fixture");
    let side_proof = build_rv32im_side_proof(&nightstream_statement, &accepted_artifact)
        .expect("build side proof for side soundness fixture");
    SideFixture {
        public_statement: accepted_artifact.statement.clone(),
        accepted_artifact,
        nightstream_statement,
        side_proof,
    }
}

pub fn mutated_statement_with_new_core(statement: &NightstreamStatement) -> NightstreamStatement {
    let mut mutated = statement.clone();
    mutated.verifier_context_digest[0] ^= 1;
    mutated
}

pub fn refresh_public(instance: &mut Rv32imSideOpeningPublic) {
    for opened_object in &mut instance.opened_objects {
        opened_object.digest = opened_object.expected_digest();
    }
    for eval in &mut instance.evals {
        eval.digest = eval.expected_digest();
    }
    instance.digest = instance.expected_digest();
}

pub fn refresh_side_proof(side_proof: &mut Rv32imSideProof) {
    refresh_public(side_proof.opening_public_mut());
}

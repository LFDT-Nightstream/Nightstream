use neo_fold_next::nightstream::rv64im::audit::{
    build_rv64im_side_opening_relation_from_accepted_artifact, setup_rv64im_side_opening_spartan,
};
use neo_fold_next::nightstream::rv64im::{
    build_rv64im_nightstream_from_public_proof, build_rv64im_side_proof, Rv64imSideBindingStatement,
    Rv64imSideOpeningPublic, Rv64imSideOpeningSpartanVerifierKey, Rv64imSideProof,
};
use neo_fold_next::nightstream::NightstreamStatement;
use neo_fold_next::rv64im::{
    build_rv64im_accepted_proof_artifact, parity_source_cases, prove_rv64im_public_proof, Rv64imAcceptedProofArtifact,
    Rv64imProofInput, Rv64imProofStatement, SimpleKernelError,
};

pub struct SideFixture {
    pub accepted_artifact: Rv64imAcceptedProofArtifact,
    pub nightstream_statement: NightstreamStatement,
    pub public_statement: Rv64imProofStatement,
    pub side_proof: Rv64imSideProof,
}

impl SideFixture {
    pub fn side_statement(&self) -> Result<Rv64imSideBindingStatement, SimpleKernelError> {
        self.side_proof
            .binding_statement(&self.nightstream_statement)
    }

    pub fn side_public(&self) -> &Rv64imSideOpeningPublic {
        self.side_proof.opening_public()
    }

    pub fn side_opening_vk(&self) -> Rv64imSideOpeningSpartanVerifierKey {
        let (statement, witness) = build_rv64im_side_opening_relation_from_accepted_artifact(&self.accepted_artifact)
            .expect("build side opening relation for opening vk");
        let (_, vk) = setup_rv64im_side_opening_spartan(&statement, &witness).expect("setup side opening spartan");
        vk
    }
}

pub fn source_case(name: &str) -> neo_fold_next::rv64im::Rv64imParitySourceCase {
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

pub fn proof_input(name: &str) -> Rv64imProofInput {
    let source = source_case(name);
    let max_steps = source.program_words.len();
    Rv64imProofInput { source, max_steps }
}

pub fn build_side_fixture(name: &str) -> SideFixture {
    let public_proof =
        prove_rv64im_public_proof(&proof_input(name)).expect("prove rv64im public proof for side soundness fixture");
    let accepted_artifact = build_rv64im_accepted_proof_artifact(&public_proof)
        .expect("build accepted artifact for side soundness fixture");
    let (nightstream_statement, _nightstream_proof) = build_rv64im_nightstream_from_public_proof(&public_proof)
        .expect("build nightstream proof for side soundness fixture");
    let side_proof = build_rv64im_side_proof(&nightstream_statement, &accepted_artifact)
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
    mutated.linkage_root[0] ^= 1;
    mutated
}

pub fn refresh_public(instance: &mut Rv64imSideOpeningPublic) {
    for opened_object in &mut instance.opened_objects {
        opened_object.digest = opened_object.expected_digest();
    }
    for eval in &mut instance.evals {
        eval.digest = eval.expected_digest();
    }
    instance.digest = instance.expected_digest();
}

pub fn refresh_side_proof(side_proof: &mut Rv64imSideProof) {
    refresh_public(side_proof.opening_public_mut());
}

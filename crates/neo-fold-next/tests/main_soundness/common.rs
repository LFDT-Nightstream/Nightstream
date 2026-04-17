#[allow(dead_code)]
#[path = "../support/rv64im_n2.rs"]
mod rv64im_n2_support;

use neo_fold_next::nightstream::rv64im::{build_rv64im_main_proof, Rv64imMainProof};
use neo_fold_next::rv64im::SimpleKernelError;

pub struct MainSoundnessFixture {
    pub main_proof: Rv64imMainProof,
}

pub fn build_main_soundness_fixture() -> Result<MainSoundnessFixture, SimpleKernelError> {
    let fixture = rv64im_n2_support::build_rv64im_n2_fixture()?;
    let main_proof = build_rv64im_main_proof(&fixture.final_statement, &fixture.final_proof)?;
    Ok(MainSoundnessFixture { main_proof })
}

#[allow(dead_code)]
#[path = "bin/rv64im_rust_vectors.rs"]
mod rv64im_rust_vectors_bin;

use neo_fold_prototype::rv64im::{Rv64imParityDerivedCase, Rv64imParitySourceCase};

pub fn render_rv64im_single_case_compat_module(
    module_name: &str,
    source: &Rv64imParitySourceCase,
    derived: &Rv64imParityDerivedCase,
) -> String {
    let mut proof_cases =
        rv64im_rust_vectors_bin::build_public_proof_cases(&[(source.clone(), derived.clone())]);
    let proof_case = proof_cases
        .pop()
        .expect("build exactly one RV64IM public proof case");

    format!(
        "import Nightstream.Rv64IM.Checks\nimport Nightstream.Rv64IM.ProofBoundaryChecks\nimport Nightstream.Rv64IM.Generated.ParityTypes\nimport Nightstream.Rv64IM.Generated.PublicProofVectorTypes\n\nset_option maxHeartbeats 0\n\nopen Nightstream.Rv64IM\nopen Nightstream.Rv64IM.Generated\n\nnamespace {module_name}\n\ndef sourceCase : ParitySourceCase :=\n  {}\n\ndef derivedCase : ParityDerivedCase :=\n  {}\n\ndef proofCase : PublicProofVectorCase :=\n  {}\n\n#eval publicProofCaseCheckResultsAgainstDerived proofCase derivedCase\n#eval kernelProofDigestCheckResults proofCase.kernelProof\n\nexample : checkParityCase sourceCase derivedCase = true := by\n  native_decide\n\nexample : checkPublicProofVectorCaseAgainstDerived proofCase derivedCase = true := by\n  native_decide\n\nend {module_name}\n",
        rv64im_rust_vectors_bin::render_source_case(source),
        rv64im_rust_vectors_bin::render_derived_case(derived),
        rv64im_rust_vectors_bin::render_public_proof_vector_case(&proof_case),
    )
}

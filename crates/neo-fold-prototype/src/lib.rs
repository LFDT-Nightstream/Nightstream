//! `neo-fold-prototype` is the SuperNeo IVC integrator.
//!
//! Direct-CCS lifecycle:
//!   `preprocess_direct_ccs` → `prove_direct_ccs` → `extend_direct_ccs`* →
//!   `finish_direct_ccs_with_spartan`
//!   (or `prove_and_finish_direct_ccs_with_spartan` as a one-shot).
//! Verify with `verify_direct_ccs` (native, no Spartan) or
//! `verify_finished_direct_ccs_with_spartan` (after Spartan).
//!
//! RV32IM lifecycle: `prove_rv32im` → `verify_rv32im`.
//!
//! Protocol math lives in the sibling `neo_*` crates; this crate owns IVC
//! threading, Construction-2, the recursive verifier circuit, and frontend
//! lowering for `direct_ccs` and `rv32im`.

pub mod circuit;
pub mod core;
pub mod decider;
pub mod frontends;
pub mod lifecycle;
pub mod public_proof;
pub mod vm;

pub use self::core::proof;
pub use self::direct_ccs::{DirectCcsFPrimeSnarkError, DirectCcsProgram, DirectCcsStep};
pub use self::frontends::{direct_ccs, rv32im};
pub use self::lifecycle::{
    extend_direct_ccs, finish_direct_ccs_with_spartan, preprocess_direct_ccs, prove_and_finish_direct_ccs_with_spartan,
    prove_direct_ccs, prove_rv32im, verify_direct_ccs, verify_finished_direct_ccs_with_spartan, verify_rv32im,
    DirectCcs, DirectCcsCommitmentOps, DirectCcsDecCommitmentMixer, DirectCcsFinishedProof,
    DirectCcsFinishedProofBundle, DirectCcsFinishedProofPerf, DirectCcsFinishedPublicImage,
    DirectCcsFinishedVerifierKey, DirectCcsProof, DirectCcsProofSummary, DirectCcsProverPreprocessing,
    DirectCcsRlcCommitmentMixer, IncrementalProofSystem, OneShotProofSystem, Rv32im, SpartanProofSystem,
};
pub use self::rv32im::{Rv32imProof, Rv32imProofInput, SimpleKernelError};

/// Legacy RV64IM compatibility surface for the Midnight bridge crate.
///
/// The implementation now lives under the RV32IM frontend; these aliases keep
/// the bridge compiling while it is still named and typed as RV64IM.
pub mod rv64im {
    pub use crate::rv32im::{
        build_mixed_opcode_perf_source_case, Rv32imProof as Rv64imProof, Rv32imProofInput as Rv64imProofInput,
        SimpleKernelError,
    };

    pub fn prove_rv64im_public_proof(input: &Rv64imProofInput) -> Result<Rv64imProof, SimpleKernelError> {
        crate::rv32im::prove_rv32im_public_proof(input)
    }
}

/// Legacy Nightstream compatibility surface for RV64IM-named bridge callers.
pub mod nightstream {
    pub use crate::public_proof::{
        nightstream_proof_binding_root, NightstreamProofBindingInputs, NightstreamStatement,
    };

    pub mod rv64im {
        use neo_transcript::{Poseidon2Transcript, Transcript};

        pub use crate::public_proof::rv32im::Rv32imNightstreamProof as Rv64imNightstreamProof;

        use crate::public_proof::NightstreamStatement;
        use crate::rv32im::{Rv32imProof, Rv32imPublishedStatement, SimpleKernelError};

        pub fn build_rv64im_nightstream_from_public_proof(
            proof: &Rv32imProof,
        ) -> Result<(NightstreamStatement, Rv64imNightstreamProof), SimpleKernelError> {
            crate::public_proof::rv32im::build_rv32im_nightstream_from_public_proof_with_perf(proof)
                .map(|((statement, proof), _perf)| (statement, proof))
        }

        pub fn rv64im_verifier_context_digest(
            root_params_id: [u8; 32],
            published_statement: &Rv32imPublishedStatement,
        ) -> [u8; 32] {
            let mut tr = Poseidon2Transcript::new(b"neo.fold.next/nightstream/rv64im/verifier_context");
            tr.append_message(
                b"neo.fold.next/nightstream/rv64im/verifier_context/version",
                b"legacy-v1",
            );
            tr.append_message(
                b"neo.fold.next/nightstream/rv64im/verifier_context/root_params_id",
                &root_params_id,
            );
            tr.append_message(
                b"neo.fold.next/nightstream/rv64im/verifier_context/published_statement",
                &published_statement.expected_digest(),
            );
            tr.digest32()
        }

        pub mod audit {
            pub use crate::public_proof::rv32im::rv32im_main_nightstream_proof_digest as rv64im_main_nightstream_proof_digest;
        }
    }
}

pub(crate) use self::circuit::{superneo as superneo_circuit, superneo_nifs as superneo_nifs_circuit};
pub(crate) use self::core::multilinear;
pub(crate) use self::core::{
    chunk_folding, construction2, finalize, ivc, opening, prover, session, step_build, verifier, witness_layout,
};
pub(crate) use self::decider::spartan_backend;

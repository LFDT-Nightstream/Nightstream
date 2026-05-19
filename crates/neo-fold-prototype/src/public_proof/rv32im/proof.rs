//! Owns the serialized RV32IM Nightstream proof containers and proof digests.

use neo_transcript::{Poseidon2Transcript, Transcript};
use serde::{Deserialize, Serialize};

use crate::public_proof::NightstreamStatement;
use crate::rv32im::kernel::{Rv32imProofStatement, SimpleKernelError};
use crate::rv32im::Rv32imCompressedMainProof;

use super::authoritative_side::{
    build_rv32im_side_binding_statement, Rv32imSideBindingStatement, Rv32imSideOpeningPublic,
};
use super::side_opening_relation::Rv32imSideOpeningRelationStatement;
use super::side_opening_spartan::Rv32imSideOpeningSpartanProof;
use super::side_relation_spartan::Rv32imSideBindingProof;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Rv32imSideProof {
    opening_public: Rv32imSideOpeningPublic,
    opening_statement: Rv32imSideOpeningRelationStatement,
    opening: Rv32imSideOpeningSpartanProof,
    binding: Rv32imSideBindingProof,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Rv32imNightstreamProof {
    main_proof: Rv32imCompressedMainProof,
    side_proof: Rv32imSideProof,
}

impl Rv32imSideProof {
    pub(super) fn from_parts(
        opening_public: Rv32imSideOpeningPublic,
        opening_statement: Rv32imSideOpeningRelationStatement,
        opening: Rv32imSideOpeningSpartanProof,
        binding: Rv32imSideBindingProof,
    ) -> Self {
        Self {
            opening_public,
            opening_statement,
            opening,
            binding,
        }
    }

    pub fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/nightstream/rv32im/side_proof");
        tr.append_message(b"neo.fold.next/nightstream/rv32im/side_proof/version", b"v7");
        tr.append_message(
            b"neo.fold.next/nightstream/rv32im/side_proof/public_digest",
            &self.opening_public.digest,
        );
        tr.append_message(
            b"neo.fold.next/nightstream/rv32im/side_proof/opening_statement_digest",
            &self.opening_statement.expected_digest(),
        );
        tr.append_message(
            b"neo.fold.next/nightstream/rv32im/side_proof/opening_snark_data",
            &self.opening.snark_data,
        );
        tr.append_message(
            b"neo.fold.next/nightstream/rv32im/side_proof/binding_snark_data",
            &self.binding.snark_data,
        );
        tr.digest32()
    }

    pub fn binding_statement(
        &self,
        nightstream_statement: &NightstreamStatement,
        public_statement: &Rv32imProofStatement,
    ) -> Result<Rv32imSideBindingStatement, SimpleKernelError> {
        build_rv32im_side_binding_statement(nightstream_statement, public_statement, &self.opening_public)
    }

    pub fn opening_public(&self) -> &Rv32imSideOpeningPublic {
        &self.opening_public
    }

    pub fn opening_public_mut(&mut self) -> &mut Rv32imSideOpeningPublic {
        &mut self.opening_public
    }

    pub fn opening_statement(&self) -> &Rv32imSideOpeningRelationStatement {
        &self.opening_statement
    }

    pub fn opening_statement_mut(&mut self) -> &mut Rv32imSideOpeningRelationStatement {
        &mut self.opening_statement
    }

    pub fn opening(&self) -> &Rv32imSideOpeningSpartanProof {
        &self.opening
    }

    pub fn opening_mut(&mut self) -> &mut Rv32imSideOpeningSpartanProof {
        &mut self.opening
    }

    pub fn binding(&self) -> &Rv32imSideBindingProof {
        &self.binding
    }

    pub fn binding_mut(&mut self) -> &mut Rv32imSideBindingProof {
        &mut self.binding
    }
}

impl Rv32imNightstreamProof {
    pub(super) fn from_parts(main_proof: Rv32imCompressedMainProof, side_proof: Rv32imSideProof) -> Self {
        Self { main_proof, side_proof }
    }

    pub fn main_proof(&self) -> &Rv32imCompressedMainProof {
        &self.main_proof
    }

    pub fn main_proof_mut(&mut self) -> &mut Rv32imCompressedMainProof {
        &mut self.main_proof
    }

    pub fn side_proof(&self) -> &Rv32imSideProof {
        &self.side_proof
    }

    pub fn side_proof_mut(&mut self) -> &mut Rv32imSideProof {
        &mut self.side_proof
    }
}

// Nightstream binds the RV32IM main proof through its compact public owner
// surface, not through private recursion step-proof bytes.
pub fn rv32im_main_nightstream_proof_digest(main_proof: &Rv32imCompressedMainProof) -> [u8; 32] {
    let mut tr = Poseidon2Transcript::new(b"neo.fold.next/nightstream/rv32im/main_proof");
    tr.append_message(b"neo.fold.next/nightstream/rv32im/main_proof/version", b"v3");
    tr.append_message(
        b"neo.fold.next/nightstream/rv32im/main_proof/binding_digest",
        &main_proof.binding_digest(),
    );
    tr.digest32()
}

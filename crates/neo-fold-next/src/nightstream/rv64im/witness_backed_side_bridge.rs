//! Owns the fixed witness-backed RV64IM side-bridge theorem boundary.
//!
//! The statement binds the published Nightstream statement core plus the
//! canonical compact opening-artifact digest used by the side theorem.

use neo_transcript::{Poseidon2Transcript, Transcript};
use serde::{Deserialize, Serialize};

use crate::nightstream::NightstreamStatement;
use crate::rv64im::SimpleKernelError;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Rv64imWitnessBackedSideBridgeStatement {
    pub nightstream_statement: NightstreamStatement,
    pub opening_artifact_digest: [u8; 32],
}

impl Rv64imWitnessBackedSideBridgeStatement {
    pub fn digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/nightstream/rv64im/witness_backed_side_bridge_statement");
        tr.append_message(
            b"neo.fold.next/nightstream/rv64im/witness_backed_side_bridge_statement/version",
            b"v2",
        );
        tr.append_message(
            b"neo.fold.next/nightstream/rv64im/witness_backed_side_bridge_statement/nightstream_statement_core_digest",
            &self.nightstream_statement.core_digest(),
        );
        tr.append_message(
            b"neo.fold.next/nightstream/rv64im/witness_backed_side_bridge_statement/opening_artifact_digest",
            &self.opening_artifact_digest,
        );
        tr.digest32()
    }
}

pub(super) fn build_rv64im_witness_backed_side_bridge_statement(
    nightstream_statement: &NightstreamStatement,
    opening_artifact_digest: [u8; 32],
) -> Result<Rv64imWitnessBackedSideBridgeStatement, SimpleKernelError> {
    Ok(Rv64imWitnessBackedSideBridgeStatement {
        nightstream_statement: nightstream_statement.clone(),
        opening_artifact_digest,
    })
}

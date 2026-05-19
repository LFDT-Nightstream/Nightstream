//! Owns canonical Ajtai opening identities and safe alias accounting for RV32IM selected openings.

use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;

use neo_transcript::{Poseidon2Transcript, Transcript};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
pub enum AjtaiFamilyKind {
    RootMainLaneColumns,
    RootMainLaneCommittedRows,
    Stage1Rows,
    Stage2RegisterReads,
    Stage2RegisterWrites,
    Stage2RamEvents,
    Stage2TwistLinks,
    Stage3Continuity,
    KernelBindings,
    KernelPreparedSteps,
    RootMainLanePublicSteps,
}

impl AjtaiFamilyKind {
    pub(crate) fn tag(self) -> u64 {
        match self {
            Self::RootMainLaneColumns => 0,
            Self::RootMainLaneCommittedRows => 10,
            Self::Stage1Rows => 1,
            Self::Stage2RegisterReads => 2,
            Self::Stage2RegisterWrites => 3,
            Self::Stage2RamEvents => 4,
            Self::Stage2TwistLinks => 5,
            Self::Stage3Continuity => 6,
            Self::KernelBindings => 7,
            Self::KernelPreparedSteps => 8,
            Self::RootMainLanePublicSteps => 9,
        }
    }

    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::RootMainLaneColumns => "root_main_lane_columns",
            Self::RootMainLaneCommittedRows => "root_main_lane_committed_rows",
            Self::Stage1Rows => "stage1_rows",
            Self::Stage2RegisterReads => "stage2_register_reads",
            Self::Stage2RegisterWrites => "stage2_register_writes",
            Self::Stage2RamEvents => "stage2_ram_events",
            Self::Stage2TwistLinks => "stage2_twist_links",
            Self::Stage3Continuity => "stage3_continuity",
            Self::KernelBindings => "kernel_bindings",
            Self::KernelPreparedSteps => "kernel_prepared_steps",
            Self::RootMainLanePublicSteps => "root_main_lane_public_steps",
        }
    }
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct AjtaiObjectId {
    pub family: AjtaiFamilyKind,
    pub commitment_digest: [u8; 32],
    pub layout_version: u64,
    pub digest: [u8; 32],
}

impl AjtaiObjectId {
    pub fn new(family: AjtaiFamilyKind, commitment_digest: [u8; 32], layout_version: u64) -> Self {
        let mut object = Self {
            family,
            commitment_digest,
            layout_version,
            digest: [0; 32],
        };
        object.digest = object.expected_digest();
        object
    }

    pub(crate) fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv32im/ajtai_object_id");
        tr.append_u64s(
            b"rv32im/ajtai_object_id/meta",
            &[self.family.tag(), self.layout_version],
        );
        tr.append_message(b"rv32im/ajtai_object_id/family", self.family.as_str().as_bytes());
        tr.append_message(b"rv32im/ajtai_object_id/commitment_digest", &self.commitment_digest);
        tr.digest32()
    }
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct AjtaiOpeningId {
    pub object: AjtaiObjectId,
    pub logical_index: u64,
    pub digest: [u8; 32],
}

impl AjtaiOpeningId {
    pub fn new(object: AjtaiObjectId, logical_index: u64) -> Self {
        let mut opening = Self {
            object,
            logical_index,
            digest: [0; 32],
        };
        opening.digest = opening.expected_digest();
        opening
    }

    pub(crate) fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv32im/ajtai_opening_id");
        tr.append_message(b"rv32im/ajtai_opening_id/object_digest", &self.object.digest);
        tr.append_u64s(b"rv32im/ajtai_opening_id/logical_index", &[self.logical_index]);
        tr.digest32()
    }
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct SelectedOpeningRef {
    pub id: AjtaiOpeningId,
    pub value_digest: [u8; 32],
    pub digest: [u8; 32],
}

impl SelectedOpeningRef {
    pub fn new(id: AjtaiOpeningId, value_digest: [u8; 32]) -> Self {
        let mut reference = Self {
            id,
            value_digest,
            digest: [0; 32],
        };
        reference.digest = reference.expected_digest();
        reference
    }

    pub fn from_parts(
        family: AjtaiFamilyKind,
        commitment_digest: [u8; 32],
        layout_version: u64,
        logical_index: u64,
        value_digest: [u8; 32],
    ) -> Self {
        let object = AjtaiObjectId::new(family, commitment_digest, layout_version);
        let opening = AjtaiOpeningId::new(object, logical_index);
        Self::new(opening, value_digest)
    }

    pub fn value_digest_mut(&mut self) -> &mut [u8; 32] {
        &mut self.value_digest
    }

    pub(crate) fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv32im/selected_opening_ref");
        tr.append_message(b"rv32im/selected_opening_ref/opening_id", &self.id.digest);
        tr.append_message(b"rv32im/selected_opening_ref/value_digest", &self.value_digest);
        tr.digest32()
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct OpeningAccumulatorStats {
    pub total_requests: usize,
    pub unique_requests: usize,
    pub aliased_requests: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OpeningAliasError {
    pub opening_id_digest: [u8; 32],
    pub existing_value_digest: [u8; 32],
    pub new_value_digest: [u8; 32],
}

impl fmt::Display for OpeningAliasError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "selected opening alias mismatch for {:?}: existing {:?}, new {:?}",
            self.opening_id_digest, self.existing_value_digest, self.new_value_digest
        )
    }
}

impl Error for OpeningAliasError {}

#[derive(Clone, Debug, Default)]
pub struct OpeningAccumulator {
    seen: BTreeMap<AjtaiOpeningId, [u8; 32]>,
    total_requests: usize,
    aliased_requests: usize,
}

impl OpeningAccumulator {
    pub fn observe(&mut self, reference: &SelectedOpeningRef) -> Result<(), OpeningAliasError> {
        self.total_requests += 1;
        match self.seen.get(&reference.id) {
            Some(existing_value_digest) if existing_value_digest == &reference.value_digest => {
                self.aliased_requests += 1;
                Ok(())
            }
            Some(existing_value_digest) => Err(OpeningAliasError {
                opening_id_digest: reference.id.digest,
                existing_value_digest: *existing_value_digest,
                new_value_digest: reference.value_digest,
            }),
            None => {
                self.seen
                    .insert(reference.id.clone(), reference.value_digest);
                Ok(())
            }
        }
    }

    pub fn observe_option(&mut self, reference: Option<&SelectedOpeningRef>) -> Result<(), OpeningAliasError> {
        if let Some(reference) = reference {
            self.observe(reference)?;
        }
        Ok(())
    }

    pub fn stats(&self) -> OpeningAccumulatorStats {
        OpeningAccumulatorStats {
            total_requests: self.total_requests,
            unique_requests: self.seen.len(),
            aliased_requests: self.aliased_requests,
        }
    }

    pub fn opening_id_digests(&self) -> Vec<[u8; 32]> {
        self.seen.keys().map(|opening| opening.digest).collect()
    }
}

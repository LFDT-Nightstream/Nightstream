//! Owns the verifier-facing Phase 0 RV64IM evaluation-claim surface and its canonical ordering.
//!
//! It owns:
//! - the frozen in-scope family/schema identity
//! - concrete opened-object, payload, and claim records
//! - canonical claim and bundle digests
//! - duplicate and mismatch rejection at the claim-accumulator boundary
//!
//! It does not own:
//! - claim emission from stage artifacts
//! - prover-only packed-column oracle witnesses
//! - Phase 1 reduction or Phase 2 accumulation

use std::cmp::Ordering;

use neo_math::{KExtensions, D, F, K};
use neo_transcript::{Poseidon2Transcript, Transcript};
use p3_field::PrimeCharacteristicRing;
use serde::{de::Error as DeError, Deserialize, Deserializer, Serialize, Serializer};
use thiserror::Error;

use crate::finalize::digest32_as_fields;
use crate::opening::OpeningDomain;

use super::canonical_openings::AjtaiFamilyKind;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
pub enum FamilyEvalSchemaId {
    Stage1Rows,
    Stage2RegisterReads,
    Stage2RegisterWrites,
    Stage2RamEvents,
    Stage2TwistLinks,
    Stage3Continuity,
}

impl FamilyEvalSchemaId {
    pub fn tag(self) -> u64 {
        match self {
            Self::Stage1Rows => 1,
            Self::Stage2RegisterReads => 2,
            Self::Stage2RegisterWrites => 3,
            Self::Stage2RamEvents => 4,
            Self::Stage2TwistLinks => 5,
            Self::Stage3Continuity => 6,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Stage1Rows => "stage1_rows",
            Self::Stage2RegisterReads => "stage2_register_reads",
            Self::Stage2RegisterWrites => "stage2_register_writes",
            Self::Stage2RamEvents => "stage2_ram_events",
            Self::Stage2TwistLinks => "stage2_twist_links",
            Self::Stage3Continuity => "stage3_continuity",
        }
    }

    pub fn family_kind(self) -> AjtaiFamilyKind {
        match self {
            Self::Stage1Rows => AjtaiFamilyKind::Stage1Rows,
            Self::Stage2RegisterReads => AjtaiFamilyKind::Stage2RegisterReads,
            Self::Stage2RegisterWrites => AjtaiFamilyKind::Stage2RegisterWrites,
            Self::Stage2RamEvents => AjtaiFamilyKind::Stage2RamEvents,
            Self::Stage2TwistLinks => AjtaiFamilyKind::Stage2TwistLinks,
            Self::Stage3Continuity => AjtaiFamilyKind::Stage3Continuity,
        }
    }

    pub fn from_family(family: AjtaiFamilyKind) -> Option<Self> {
        match family {
            AjtaiFamilyKind::Stage1Rows => Some(Self::Stage1Rows),
            AjtaiFamilyKind::Stage2RegisterReads => Some(Self::Stage2RegisterReads),
            AjtaiFamilyKind::Stage2RegisterWrites => Some(Self::Stage2RegisterWrites),
            AjtaiFamilyKind::Stage2RamEvents => Some(Self::Stage2RamEvents),
            AjtaiFamilyKind::Stage2TwistLinks => Some(Self::Stage2TwistLinks),
            AjtaiFamilyKind::Stage3Continuity => Some(Self::Stage3Continuity),
            _ => None,
        }
    }

    pub fn opening_domain(self) -> OpeningDomain {
        match self {
            Self::Stage1Rows | Self::Stage3Continuity => OpeningDomain::Cpu,
            Self::Stage2RegisterReads | Self::Stage2RegisterWrites | Self::Stage2RamEvents | Self::Stage2TwistLinks => {
                OpeningDomain::Mem
            }
        }
    }

    pub fn packed_column_count(self) -> usize {
        match self {
            Self::Stage1Rows => 1,
            Self::Stage2RegisterReads
            | Self::Stage2RegisterWrites
            | Self::Stage2RamEvents
            | Self::Stage2TwistLinks
            | Self::Stage3Continuity => 1,
        }
    }
}

pub fn phase0_family_order() -> [FamilyEvalSchemaId; 6] {
    [
        FamilyEvalSchemaId::Stage1Rows,
        FamilyEvalSchemaId::Stage2RegisterReads,
        FamilyEvalSchemaId::Stage2RegisterWrites,
        FamilyEvalSchemaId::Stage2RamEvents,
        FamilyEvalSchemaId::Stage2TwistLinks,
        FamilyEvalSchemaId::Stage3Continuity,
    ]
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
pub struct CommitmentContextId {
    pub pp_seed_digest: [u8; 32],
    pub module_shape_digest: [u8; 32],
}

impl CommitmentContextId {
    pub fn new(pp_seed_digest: [u8; 32], module_shape_digest: [u8; 32]) -> Self {
        Self {
            pp_seed_digest,
            module_shape_digest,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
pub struct OpenedAjtaiObjectId {
    pub family: AjtaiFamilyKind,
    pub commitment_root_digest: [u8; 32],
    pub layout_version: u64,
    pub row_domain_log_size: u32,
    pub digest: [u8; 32],
}

impl OpenedAjtaiObjectId {
    pub fn new(
        family: AjtaiFamilyKind,
        commitment_context: &CommitmentContextId,
        commitment_root_digest: [u8; 32],
        layout_version: u64,
        row_domain_log_size: u32,
    ) -> Self {
        let mut object = Self {
            family,
            commitment_root_digest,
            layout_version,
            row_domain_log_size,
            digest: [0; 32],
        };
        object.digest = object.expected_digest(commitment_context);
        object
    }

    pub fn expected_digest(&self, commitment_context: &CommitmentContextId) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv64im/opening_convergence/phase0/opened_object");
        tr.append_fields_raw(&[
            F::from_u64(self.family.tag()),
            F::from_u64(self.layout_version),
            F::from_u64(self.row_domain_log_size as u64),
        ]);
        tr.append_fields_raw(&digest32_as_fields(commitment_context.pp_seed_digest));
        tr.append_fields_raw(&digest32_as_fields(commitment_context.module_shape_digest));
        tr.append_fields_raw(&digest32_as_fields(self.commitment_root_digest));
        tr.digest32()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
pub struct FamilyEvalClaimId {
    pub opened_object_digest: [u8; 32],
    pub slot: u32,
}

impl FamilyEvalClaimId {
    pub fn new(opened_object_digest: [u8; 32], slot: u32) -> Self {
        Self {
            opened_object_digest,
            slot,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PackedColumnEval {
    pub coeffs: [K; D],
}

impl Serialize for PackedColumnEval {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.coeffs.as_slice().serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for PackedColumnEval {
    fn deserialize<Ds>(deserializer: Ds) -> Result<Self, Ds::Error>
    where
        Ds: Deserializer<'de>,
    {
        let coeffs = Vec::<K>::deserialize(deserializer)?;
        let actual = coeffs.len();
        let coeffs = coeffs.try_into().map_err(|_: Vec<K>| {
            Ds::Error::custom(format!(
                "phase0 packed-column eval expected {D} extension coefficients, got {actual}"
            ))
        })?;
        Ok(Self { coeffs })
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct FamilyEvalPayload {
    pub schema: FamilyEvalSchemaId,
    pub column_evals: Vec<PackedColumnEval>,
}

impl FamilyEvalPayload {
    pub fn new(schema: FamilyEvalSchemaId, column_evals: Vec<PackedColumnEval>) -> Result<Self, EvalClaimError> {
        let payload = Self { schema, column_evals };
        payload.validate()?;
        Ok(payload)
    }

    pub fn validate(&self) -> Result<(), EvalClaimError> {
        let expected = self.schema.packed_column_count();
        let actual = self.column_evals.len();
        if expected != actual {
            return Err(EvalClaimError::PayloadWidthMismatch {
                schema: self.schema,
                expected,
                actual,
            });
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct FamilyEvalClaim {
    pub opened_object: OpenedAjtaiObjectId,
    pub id: FamilyEvalClaimId,
    pub commitment_context: CommitmentContextId,
    pub point: Vec<K>,
    pub payload: FamilyEvalPayload,
    pub binding_digest: [u8; 32],
}

impl FamilyEvalClaim {
    pub fn new(
        opened_object: OpenedAjtaiObjectId,
        slot: u32,
        commitment_context: CommitmentContextId,
        point: Vec<K>,
        payload: FamilyEvalPayload,
        binding_digest: [u8; 32],
    ) -> Result<Self, EvalClaimError> {
        let claim = Self {
            id: FamilyEvalClaimId::new(opened_object.digest, slot),
            opened_object,
            commitment_context,
            point,
            payload,
            binding_digest,
        };
        claim.validate()?;
        Ok(claim)
    }

    pub fn validate(&self) -> Result<(), EvalClaimError> {
        if self.id.opened_object_digest != self.opened_object.digest {
            return Err(EvalClaimError::ClaimObjectDigestMismatch {
                id_digest: self.id.opened_object_digest,
                object_digest: self.opened_object.digest,
            });
        }
        let expected_opened_object_digest = self.opened_object.expected_digest(&self.commitment_context);
        if expected_opened_object_digest != self.opened_object.digest {
            return Err(EvalClaimError::OpenedObjectDigestMismatch {
                expected_digest: expected_opened_object_digest,
                object_digest: self.opened_object.digest,
            });
        }

        self.payload.validate()?;

        let Some(expected_schema) = FamilyEvalSchemaId::from_family(self.opened_object.family) else {
            return Err(EvalClaimError::UnsupportedFamily {
                family: self.opened_object.family,
            });
        };
        if expected_schema != self.payload.schema {
            return Err(EvalClaimError::FamilySchemaMismatch {
                family: self.opened_object.family,
                schema: self.payload.schema,
            });
        }

        let expected_point_arity = self.opened_object.row_domain_log_size as usize;
        let actual_point_arity = self.point.len();
        if expected_point_arity != actual_point_arity {
            return Err(EvalClaimError::PointArityMismatch {
                expected: expected_point_arity,
                actual: actual_point_arity,
            });
        }

        Ok(())
    }

    pub fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv64im/opening_convergence/phase0/claim");
        tr.append_message(
            b"neo.fold.next/rv64im/opening_convergence/phase0/claim/opened_object_digest",
            &self.opened_object.digest,
        );
        tr.append_message(
            b"neo.fold.next/rv64im/opening_convergence/phase0/claim/pp_seed_digest",
            &self.commitment_context.pp_seed_digest,
        );
        tr.append_message(
            b"neo.fold.next/rv64im/opening_convergence/phase0/claim/module_shape_digest",
            &self.commitment_context.module_shape_digest,
        );
        tr.append_u64s(
            b"neo.fold.next/rv64im/opening_convergence/phase0/claim/meta",
            &[self.payload.schema.tag(), self.id.slot as u64],
        );
        append_k_vec(
            &mut tr,
            b"neo.fold.next/rv64im/opening_convergence/phase0/claim/point",
            &self.point,
        );
        append_packed_column_evals(
            &mut tr,
            b"neo.fold.next/rv64im/opening_convergence/phase0/claim/payload",
            &self.payload.column_evals,
        );
        tr.append_message(
            b"neo.fold.next/rv64im/opening_convergence/phase0/claim/binding_digest",
            &self.binding_digest,
        );
        tr.digest32()
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct OpeningClaimAccumulator {
    pub claims: Vec<FamilyEvalClaim>,
}

impl OpeningClaimAccumulator {
    pub fn insert(&mut self, claim: FamilyEvalClaim) -> Result<(), EvalClaimError> {
        self.insert_with_validation(claim, true)
    }

    pub(crate) fn insert_trusted_local(&mut self, claim: FamilyEvalClaim) -> Result<(), EvalClaimError> {
        self.insert_with_validation(claim, false)
    }

    fn insert_with_validation(&mut self, claim: FamilyEvalClaim, validate_claim: bool) -> Result<(), EvalClaimError> {
        if validate_claim {
            claim.validate()?;
        }

        for existing in &self.claims {
            if existing.opened_object.digest == claim.opened_object.digest && existing.id.slot == claim.id.slot {
                if existing.binding_digest != claim.binding_digest {
                    return Err(EvalClaimError::SlotBindingConflict {
                        opened_object_digest: claim.opened_object.digest,
                        slot: claim.id.slot,
                        existing_binding_digest: existing.binding_digest,
                        new_binding_digest: claim.binding_digest,
                    });
                }
                if existing != &claim {
                    return Err(EvalClaimError::DuplicateClaimContentMismatch {
                        opened_object_digest: claim.opened_object.digest,
                        slot: claim.id.slot,
                        binding_digest: claim.binding_digest,
                    });
                }
                return Ok(());
            }
        }

        self.claims.push(claim);
        Ok(())
    }

    pub fn into_bundle(mut self) -> Rv64imEvalClaimBundle {
        self.claims.sort_by(canonical_claim_cmp);
        Rv64imEvalClaimBundle::from_canonical_claims(self.claims)
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Rv64imEvalClaimBundle {
    pub claims: Vec<FamilyEvalClaim>,
    pub digest: [u8; 32],
}

impl Rv64imEvalClaimBundle {
    pub fn new(claims: Vec<FamilyEvalClaim>) -> Result<Self, EvalClaimError> {
        let mut accumulator = OpeningClaimAccumulator::default();
        for claim in claims {
            accumulator.insert(claim)?;
        }
        Ok(accumulator.into_bundle())
    }

    fn from_canonical_claims(claims: Vec<FamilyEvalClaim>) -> Self {
        let mut bundle = Self {
            claims,
            digest: [0; 32],
        };
        bundle.digest = bundle.expected_digest();
        bundle
    }

    pub fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv64im/opening_convergence/phase0/bundle");
        tr.append_u64s(
            b"neo.fold.next/rv64im/opening_convergence/phase0/bundle/claim_count",
            &[self.claims.len() as u64],
        );
        for claim in &self.claims {
            tr.append_message(
                b"neo.fold.next/rv64im/opening_convergence/phase0/bundle/claim_digest",
                &claim.expected_digest(),
            );
        }
        tr.digest32()
    }
}

#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum EvalClaimError {
    #[error("phase0 witness packed-column count mismatch: expected {expected}, got {actual}")]
    WitnessPackedColumnCountMismatch { expected: usize, actual: usize },
    #[error("phase0 witness commitment-vector count mismatch: expected {expected}, got {actual}")]
    WitnessCommitmentVectorCountMismatch { expected: usize, actual: usize },
    #[error(
        "phase0 witness row-domain length mismatch for packed column {column_index}: expected {expected}, got {actual}"
    )]
    WitnessRowDomainLengthMismatch {
        column_index: usize,
        expected: usize,
        actual: usize,
    },
    #[error(
        "phase0 witness commitment root mismatch: expected {expected_digest:?}, witness carries {actual_digest:?}"
    )]
    WitnessCommitmentRootMismatch {
        expected_digest: [u8; 32],
        actual_digest: [u8; 32],
    },
    #[error(
        "phase0 claim/witness opened-object mismatch: claim carries {claim_digest:?}, witness carries {witness_digest:?}"
    )]
    WitnessOpenedObjectMismatch {
        claim_digest: [u8; 32],
        witness_digest: [u8; 32],
    },
    #[error("phase0 claim/witness commitment-context mismatch")]
    WitnessCommitmentContextMismatch,
    #[error("phase0 claim/witness payload mismatch for {schema:?} slot {slot}")]
    WitnessPayloadMismatch {
        schema: FamilyEvalSchemaId,
        slot: u32,
    },
    #[error("phase0 word count mismatch for {schema:?}: expected {expected}, got {actual}")]
    WordCountMismatch {
        schema: FamilyEvalSchemaId,
        expected: usize,
        actual: usize,
    },
    #[error("phase0 field-eval width mismatch for {schema:?}: expected {expected}, got {actual}")]
    FieldEvalWidthMismatch {
        schema: FamilyEvalSchemaId,
        expected: usize,
        actual: usize,
    },
    #[error("phase0 packed-column count mismatch for {schema:?}: expected {expected}, got {actual}")]
    PackedColumnCountMismatch {
        schema: FamilyEvalSchemaId,
        expected: usize,
        actual: usize,
    },
    #[error("phase0 field-eval vector for {schema:?} must begin with the leading ONE coefficient")]
    FieldEvalLeadingOneMismatch { schema: FamilyEvalSchemaId },
    #[error(
        "phase0 reconstructed limb is not a base-field value for {schema:?} at word {word_index} limb {limb_index}"
    )]
    NonBaseFieldLimb {
        schema: FamilyEvalSchemaId,
        word_index: usize,
        limb_index: usize,
    },
    #[error(
        "phase0 reconstructed limb is out of range for {schema:?} at word {word_index} limb {limb_index}: {value}"
    )]
    LimbOutOfRange {
        schema: FamilyEvalSchemaId,
        word_index: usize,
        limb_index: usize,
        value: u64,
    },
    #[error("phase0 payload width mismatch for {schema:?}: expected {expected}, got {actual}")]
    PayloadWidthMismatch {
        schema: FamilyEvalSchemaId,
        expected: usize,
        actual: usize,
    },
    #[error("phase0 claim object digest mismatch: id carries {id_digest:?}, object carries {object_digest:?}")]
    ClaimObjectDigestMismatch {
        id_digest: [u8; 32],
        object_digest: [u8; 32],
    },
    #[error("phase0 opened object digest mismatch: expected {expected_digest:?}, object carries {object_digest:?}")]
    OpenedObjectDigestMismatch {
        expected_digest: [u8; 32],
        object_digest: [u8; 32],
    },
    #[error("phase0 family/schema mismatch: family {family:?} does not match schema {schema:?}")]
    FamilySchemaMismatch {
        family: AjtaiFamilyKind,
        schema: FamilyEvalSchemaId,
    },
    #[error("phase0 family {family:?} is outside the current v1 convergence scope")]
    UnsupportedFamily { family: AjtaiFamilyKind },
    #[error("phase0 point arity mismatch: expected {expected}, got {actual}")]
    PointArityMismatch { expected: usize, actual: usize },
    #[error(
        "phase0 duplicate slot conflict for object {opened_object_digest:?} slot {slot}: existing binding {existing_binding_digest:?}, new binding {new_binding_digest:?}"
    )]
    SlotBindingConflict {
        opened_object_digest: [u8; 32],
        slot: u32,
        existing_binding_digest: [u8; 32],
        new_binding_digest: [u8; 32],
    },
    #[error(
        "phase0 duplicate claim content mismatch for object {opened_object_digest:?} slot {slot} binding {binding_digest:?}"
    )]
    DuplicateClaimContentMismatch {
        opened_object_digest: [u8; 32],
        slot: u32,
        binding_digest: [u8; 32],
    },
}

pub(crate) fn canonical_claim_cmp(left: &FamilyEvalClaim, right: &FamilyEvalClaim) -> Ordering {
    (left.payload.schema, left.opened_object.digest, left.id.slot).cmp(&(
        right.payload.schema,
        right.opened_object.digest,
        right.id.slot,
    ))
}

fn append_k_vec(tr: &mut Poseidon2Transcript, label: &'static [u8], values: &[K]) {
    tr.append_u64s(
        b"neo.fold.next/rv64im/opening_convergence/phase0/k_vec_len",
        &[values.len() as u64],
    );
    let coeffs_per_elem = values
        .first()
        .map(|value| value.as_coeffs().len())
        .unwrap_or(0);
    tr.append_fields_iter(
        label,
        values.len().saturating_mul(coeffs_per_elem),
        values.iter().flat_map(|value| value.as_coeffs()),
    );
}

fn append_packed_column_evals(tr: &mut Poseidon2Transcript, label: &'static [u8], values: &[PackedColumnEval]) {
    tr.append_u64s(
        b"neo.fold.next/rv64im/opening_convergence/phase0/column_eval_count",
        &[values.len() as u64],
    );
    tr.append_fields_iter(
        label,
        values.len().saturating_mul(D).saturating_mul(2),
        values
            .iter()
            .flat_map(|value| value.coeffs.iter().flat_map(|coeff| coeff.as_coeffs())),
    );
}

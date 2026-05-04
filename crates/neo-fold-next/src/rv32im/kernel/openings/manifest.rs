//! Owns the published RV32IM opening witness bridge from selected-opening proofs into canonical opening claims.

use neo_math::{KExtensions, F, K};
use neo_transcript::{Poseidon2Transcript, Transcript};
use p3_field::PrimeCharacteristicRing;
use serde::{Deserialize, Serialize};

use super::canonical_openings::{AjtaiFamilyKind, SelectedOpeningRef};
use super::simple_openings::{
    OpeningPointLabel, Stage1PackagedOpeningProof, Stage1SelectedOpeningClaim, Stage2PackagedOpeningProof,
    Stage2SelectedOpeningClaim, Stage3PackagedOpeningProof, Stage3SelectedOpeningClaim,
};
use crate::opening::{OpeningClaim, OpeningDomain, OpeningSource};

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Rv32imOpeningWitnessCarrier {
    pub label: OpeningPointLabel,
    pub source: OpeningSource,
    pub domain: OpeningDomain,
    pub point: Vec<K>,
    pub ordinal: u64,
    pub column_ids: Vec<u32>,
    pub reference: SelectedOpeningRef,
    pub claim_digest: [u8; 32],
    pub packaged_digest: [u8; 32],
    pub digest: [u8; 32],
}

impl Rv32imOpeningWitnessCarrier {
    pub fn opening_claim(&self) -> OpeningClaim {
        OpeningClaim {
            source: self.source,
            domain: self.domain,
            point: self.point.clone(),
            ordinal: self.ordinal,
            column_ids: self.column_ids.clone(),
            digest: self.digest,
        }
    }

    pub fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv32im/opening_witness_carrier");
        tr.append_message(b"neo.fold.next/rv32im/opening_witness_carrier/version", b"v1");
        tr.append_u64s(
            b"neo.fold.next/rv32im/opening_witness_carrier/meta",
            &[
                self.label.tag(),
                self.ordinal,
                self.point.len() as u64,
                self.column_ids.len() as u64,
            ],
        );
        for coordinate in &self.point {
            tr.append_fields(
                b"neo.fold.next/rv32im/opening_witness_carrier/point_coordinate",
                &coordinate.as_coeffs(),
            );
        }
        let column_ids_u64: Vec<u64> = self.column_ids.iter().map(|&id| id as u64).collect();
        tr.append_u64s(
            b"neo.fold.next/rv32im/opening_witness_carrier/column_ids",
            &column_ids_u64,
        );
        tr.append_message(
            b"neo.fold.next/rv32im/opening_witness_carrier/reference_digest",
            &self.reference.digest,
        );
        tr.append_message(
            b"neo.fold.next/rv32im/opening_witness_carrier/claim_digest",
            &self.claim_digest,
        );
        tr.append_message(
            b"neo.fold.next/rv32im/opening_witness_carrier/packaged_digest",
            &self.packaged_digest,
        );
        tr.digest32()
    }
}

pub fn stage1_opening_witness_carriers(proof: &Stage1PackagedOpeningProof) -> Vec<Rv32imOpeningWitnessCarrier> {
    stage1_opening_witness_carriers_from_claim_surface(&proof.claim, proof.digest)
}

pub fn stage1_opening_witness_carriers_from_claim_surface(
    claim: &Stage1SelectedOpeningClaim,
    packaged_digest: [u8; 32],
) -> Vec<Rv32imOpeningWitnessCarrier> {
    vec![
        carrier_from_ref_with_packaged_digest(
            OpeningPointLabel::Stage1First,
            &claim.points.first,
            claim.digest,
            packaged_digest,
        ),
        carrier_from_ref_with_packaged_digest(
            OpeningPointLabel::Stage1Effect,
            &claim.points.effect,
            claim.digest,
            packaged_digest,
        ),
        carrier_from_ref_with_packaged_digest(
            OpeningPointLabel::Stage1Commit,
            &claim.points.commit,
            claim.digest,
            packaged_digest,
        ),
        carrier_from_ref_with_packaged_digest(
            OpeningPointLabel::Stage1Last,
            &claim.points.last,
            claim.digest,
            packaged_digest,
        ),
    ]
}

pub fn stage2_opening_witness_carriers(proof: &Stage2PackagedOpeningProof) -> Vec<Rv32imOpeningWitnessCarrier> {
    stage2_opening_witness_carriers_from_claim_surface(&proof.claim, proof.digest)
}

pub fn stage2_opening_witness_carriers_from_claim_surface(
    claim: &Stage2SelectedOpeningClaim,
    packaged_digest: [u8; 32],
) -> Vec<Rv32imOpeningWitnessCarrier> {
    let mut carriers = Vec::new();
    for (label, reference) in [
        (OpeningPointLabel::Stage2FirstRead, claim.points.first_read.as_ref()),
        (OpeningPointLabel::Stage2LastRead, claim.points.last_read.as_ref()),
        (OpeningPointLabel::Stage2FirstWrite, claim.points.first_write.as_ref()),
        (OpeningPointLabel::Stage2LastWrite, claim.points.last_write.as_ref()),
        (OpeningPointLabel::Stage2FirstRam, claim.points.first_ram.as_ref()),
        (OpeningPointLabel::Stage2LastRam, claim.points.last_ram.as_ref()),
        (OpeningPointLabel::Stage2FirstTwist, claim.points.first_twist.as_ref()),
        (OpeningPointLabel::Stage2LastTwist, claim.points.last_twist.as_ref()),
    ] {
        if let Some(reference) = reference {
            carriers.push(carrier_from_ref_with_packaged_digest(
                label,
                reference,
                claim.digest,
                packaged_digest,
            ));
        }
    }
    carriers
}

pub fn stage3_opening_witness_carriers(proof: &Stage3PackagedOpeningProof) -> Vec<Rv32imOpeningWitnessCarrier> {
    stage3_opening_witness_carriers_from_claim_surface(&proof.claim, proof.digest)
}

pub fn stage3_opening_witness_carriers_from_claim_surface(
    claim: &Stage3SelectedOpeningClaim,
    packaged_digest: [u8; 32],
) -> Vec<Rv32imOpeningWitnessCarrier> {
    let mut carriers = Vec::new();
    for (label, reference) in [
        (
            OpeningPointLabel::Stage3FirstContinuity,
            claim.points.first_continuity.as_ref(),
        ),
        (
            OpeningPointLabel::Stage3LastContinuity,
            claim.points.last_continuity.as_ref(),
        ),
    ] {
        if let Some(reference) = reference {
            carriers.push(carrier_from_ref_with_packaged_digest(
                label,
                reference,
                claim.digest,
                packaged_digest,
            ));
        }
    }
    carriers
}

pub fn opening_claims_from_carriers(carriers: &[Rv32imOpeningWitnessCarrier]) -> Vec<OpeningClaim> {
    carriers
        .iter()
        .map(Rv32imOpeningWitnessCarrier::opening_claim)
        .collect()
}

fn carrier_from_ref_with_packaged_digest(
    label: OpeningPointLabel,
    reference: &SelectedOpeningRef,
    claim_digest: [u8; 32],
    packaged_digest: [u8; 32],
) -> Rv32imOpeningWitnessCarrier {
    let mut carrier = Rv32imOpeningWitnessCarrier {
        label,
        source: OpeningSource::Rv32imKernel,
        domain: opening_domain(reference.id.object.family),
        // The reduction point is the evaluation anchor. The opened object family stays
        // in `column_ids` and the bound claim/reference digests, rather than forcing
        // every family onto a distinct point and preventing convergence.
        point: vec![K::from(F::from_u64(reference.id.logical_index))],
        ordinal: label.tag(),
        column_ids: vec![reference.id.object.family.tag() as u32],
        reference: reference.clone(),
        claim_digest,
        packaged_digest,
        digest: [0; 32],
    };
    carrier.digest = carrier.expected_digest();
    carrier
}

fn opening_domain(family: AjtaiFamilyKind) -> OpeningDomain {
    match family {
        AjtaiFamilyKind::Stage2RegisterReads
        | AjtaiFamilyKind::Stage2RegisterWrites
        | AjtaiFamilyKind::Stage2RamEvents
        | AjtaiFamilyKind::Stage2TwistLinks => OpeningDomain::Mem,
        AjtaiFamilyKind::RootMainLaneColumns
        | AjtaiFamilyKind::RootMainLaneCommittedRows
        | AjtaiFamilyKind::Stage1Rows
        | AjtaiFamilyKind::Stage3Continuity
        | AjtaiFamilyKind::KernelBindings
        | AjtaiFamilyKind::KernelPreparedSteps
        | AjtaiFamilyKind::RootMainLanePublicSteps => OpeningDomain::Cpu,
    }
}

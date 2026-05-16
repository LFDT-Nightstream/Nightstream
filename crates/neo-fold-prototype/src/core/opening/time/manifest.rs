//! Canonical time-opening manifest construction.

use neo_reductions::error::PiCcsError;
use neo_transcript::{Poseidon2Transcript, Transcript};

use crate::opening::OpeningClaim;

use super::digest::{append_point, canonical_claim_cmp, opening_domain_tag, opening_source_tag};
use super::types::OpeningManifest;

pub(super) fn build_manifest(
    main_lane_claims: &[OpeningClaim],
    extension_claims: &[OpeningClaim],
) -> Result<OpeningManifest, PiCcsError> {
    let mut claims = Vec::with_capacity(main_lane_claims.len() + extension_claims.len());
    claims.extend_from_slice(main_lane_claims);
    claims.extend_from_slice(extension_claims);
    claims.sort_by(canonical_claim_cmp);

    let mut tr = Poseidon2Transcript::new(b"neo.fold.next/time_opening/manifest");
    tr.append_u64s(b"neo.fold.next/time_opening/manifest_len", &[claims.len() as u64]);
    for claim in &claims {
        if claim.column_ids.is_empty() {
            return Err(PiCcsError::ProtocolError(
                "time-opening manifest contains claim with empty column_ids".into(),
            ));
        }
        if !claim.column_ids.windows(2).all(|w| w[0] < w[1]) {
            return Err(PiCcsError::ProtocolError(
                "time-opening manifest contains claim with unsorted or duplicate column_ids".into(),
            ));
        }
        tr.append_u64s(
            b"neo.fold.next/time_opening/manifest_meta",
            &[
                opening_source_tag(claim.source),
                opening_domain_tag(claim.domain),
                claim.ordinal,
                claim.point.len() as u64,
                claim.column_ids.len() as u64,
            ],
        );
        append_point(&mut tr, b"neo.fold.next/time_opening/manifest_point", &claim.point);
        let column_ids_u64: Vec<u64> = claim.column_ids.iter().map(|&id| id as u64).collect();
        tr.append_u64s(b"neo.fold.next/time_opening/manifest_column_ids", &column_ids_u64);
        tr.append_message(b"neo.fold.next/time_opening/manifest_digest", &claim.digest);
    }

    for pair in claims.windows(2) {
        if pair[0] == pair[1] {
            return Err(PiCcsError::ProtocolError(
                "time-opening manifest contains duplicate claims".into(),
            ));
        }
    }

    Ok(OpeningManifest {
        claims,
        digest: tr.digest32(),
    })
}

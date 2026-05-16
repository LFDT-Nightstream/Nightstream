//! Main-lane opening claim extraction from a completed session.

use neo_ajtai::Commitment;
use neo_math::{F, K};
use neo_reductions::error::PiCcsError;
use neo_transcript::{Poseidon2Transcript, Transcript};

use crate::opening::{OpeningClaim, OpeningDomain, OpeningSource};
use crate::proof::RunProof;

pub fn main_lane_opening_claims(session: &RunProof) -> Result<Vec<OpeningClaim>, PiCcsError> {
    let mut claims = Vec::with_capacity(session.chunks.len() + 1);
    for (chunk_idx, chunk) in session.chunks.iter().enumerate() {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/time_opening/main_lane_chunk");
        tr.append_u64s(
            b"neo.fold.next/time_opening/chunk_meta",
            &[
                chunk_idx as u64,
                chunk.chunk.start_index as u64,
                chunk.chunk.steps.len() as u64,
            ],
        );
        for step in &chunk.chunk.steps {
            tr.append_message(b"neo.fold.next/time_opening/chunk_step_label", step.label.as_bytes());
        }
        tr.append_u64s(
            b"neo.fold.next/time_opening/chunk_fold_meta",
            &[chunk.ccs_outputs.len() as u64, chunk.dec.children.len() as u64],
        );
        let point = chunk
            .ccs_outputs
            .first()
            .map(|claim| claim.r.clone())
            .ok_or_else(|| PiCcsError::ProtocolError(format!("missing main-lane CE output for chunk {chunk_idx}")))?;
        claims.push(OpeningClaim {
            source: OpeningSource::MainLane,
            domain: OpeningDomain::Cpu,
            point,
            ordinal: chunk_idx as u64,
            column_ids: vec![0],
            digest: tr.digest32(),
        });
    }

    let mut footer = Poseidon2Transcript::new(b"neo.fold.next/time_opening/main_lane_footer");
    footer.append_u64s(
        b"neo.fold.next/time_opening/footer_meta",
        &[session.final_main_claims.len() as u64],
    );
    for claim in &session.final_main_claims {
        absorb_ce_footer(&mut footer, claim);
    }
    let footer_point = session
        .final_main_claims
        .first()
        .map(|claim| claim.r.clone())
        .ok_or_else(|| PiCcsError::ProtocolError("missing final main claims for time opening".into()))?;
    claims.push(OpeningClaim {
        source: OpeningSource::MainLane,
        domain: OpeningDomain::Cpu,
        point: footer_point,
        ordinal: 0,
        column_ids: vec![1],
        digest: footer.digest32(),
    });
    Ok(claims)
}

fn absorb_ce_footer(tr: &mut Poseidon2Transcript, claim: &neo_ccs::CeClaim<Commitment, F, K>) {
    tr.append_u64s(
        b"neo.fold.next/time_opening/footer_claim_meta",
        &[claim.m_in as u64, claim.u_offset as u64, claim.u_len as u64],
    );
    tr.append_message(b"neo.fold.next/time_opening/footer_fold_digest", &claim.fold_digest);
}

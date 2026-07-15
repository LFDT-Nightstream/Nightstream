//! Select the authoritative resident source for one fold's signed masks.

use neo_ccs::Mat;
use neo_math::{D, F, K};
use p3_field::PrimeCharacteristicRing;

use super::encoding::signed_unit_mask_words;
use super::{NcSignedMasks, NcSource};
use crate::{MetalError, MetalSession, MetalWitnessMasks};

pub(super) fn select_source(
    witnesses: &[&Mat<F>],
    assignment_len: usize,
    fresh_count: usize,
    fresh: Option<&MetalWitnessMasks>,
    has_resident_tail: bool,
) -> NcSource {
    let rows = assignment_len.next_power_of_two().max(2);
    let blocks = assignment_len.div_ceil(D);
    if fresh.is_some_and(|masks| {
        masks.matches_nc(fresh_count, blocks, assignment_len) && (fresh_count == witnesses.len() || has_resident_tail)
    }) {
        return NcSource::SignedMasks(NcSignedMasks {
            words: Vec::new(),
            blocks,
            active_rows: assignment_len,
            rows,
            witness_count: witnesses.len(),
        });
    }
    witnesses
        .first()
        .and_then(|witness| {
            let blocks = witness.cols();
            blocks
                .checked_mul(D)
                .is_some_and(|entries| assignment_len <= entries)
                .then(|| signed_unit_mask_words(witnesses, blocks))
                .flatten()
                .map(|words| {
                    NcSource::SignedMasks(NcSignedMasks {
                        words,
                        blocks,
                        active_rows: assignment_len,
                        rows,
                        witness_count: witnesses.len(),
                    })
                })
        })
        .unwrap_or_else(|| {
            NcSource::Values(
                witnesses
                    .iter()
                    .map(|witness| {
                        let mut values = vec![K::ZERO; rows];
                        for column in 0..assignment_len {
                            values[column] = K::from(witness[(column % D, column / D)]);
                        }
                        values
                    })
                    .collect(),
            )
        })
}

pub(super) fn prepare_shared_masks(
    session: &MetalSession,
    masks: &NcSignedMasks,
    fresh_count: usize,
    fresh_device_masks: Option<&MetalWitnessMasks>,
    resident_id: Option<u64>,
) -> Result<MetalWitnessMasks, MetalError> {
    if let Some(fresh) = fresh_device_masks.filter(|fresh| {
        fresh.matches_nc(fresh_count, masks.blocks, masks.active_rows)
            && (fresh_count == masks.witness_count || resident_id.is_some())
    }) {
        return session.compose_witness_masks_from_device(
            fresh,
            fresh_count,
            masks.witness_count,
            masks.blocks,
            masks.active_rows,
            resident_id,
        );
    }
    match resident_id {
        Some(resident_id) => {
            let fresh_words = fresh_count
                .checked_mul(masks.blocks)
                .and_then(|values| values.checked_mul(2))
                .ok_or(MetalError::Shape("fresh witness mask dimensions overflow"))?;
            session.prepare_witness_masks_with_resident_id(
                &masks.words[..fresh_words],
                fresh_count,
                masks.witness_count,
                masks.blocks,
                masks.active_rows,
                resident_id,
            )
        }
        None => session.prepare_witness_masks(&masks.words, masks.witness_count, masks.blocks, masks.active_rows),
    }
}

use crate::PiCcsError;
use neo_ajtai::Commitment as Cmt;
use neo_ccs::{CcsStructure, Mat, MeInstance};
use neo_math::{F, K};
use neo_memory::ts_common as ts;
use neo_memory::{twist, MemWitness};
use neo_params::NeoParams;
use neo_transcript::Poseidon2Transcript;
use p3_field::PrimeCharacteristicRing;

pub fn me_identity_open(me: &MeInstance<Cmt, F, K>) -> Result<K, PiCcsError> {
    me.y_scalars
        .get(0)
        .copied()
        .ok_or_else(|| PiCcsError::ProtocolError("ME.y_scalars missing identity-first entry".into()))
}

pub fn me_identity_opens(me_slice: &[MeInstance<Cmt, F, K>], count: usize) -> Result<Vec<K>, PiCcsError> {
    if me_slice.len() < count {
        return Err(PiCcsError::InvalidInput(format!(
            "me_identity_opens: slice too short (need {count}, have {})",
            me_slice.len()
        )));
    }
    me_slice[..count].iter().map(me_identity_open).collect()
}

pub fn check_bitness_terminal(
    chi_cycle_at_r_time: K,
    bit_open: K,
    observed_final: K,
    ctx: &'static str,
) -> Result<(), PiCcsError> {
    let expected = chi_cycle_at_r_time * bit_open * (bit_open - K::ONE);
    if expected != observed_final {
        return Err(PiCcsError::ProtocolError(format!("{ctx}: bitness terminal mismatch")));
    }
    Ok(())
}

pub fn emit_twist_val_lane_openings(
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s: &CcsStructure<F>,
    mem_inst: &neo_memory::witness::MemInstance<Cmt, F>,
    mem_wit: &MemWitness<F>,
    r_val: &[K],
    m_in: usize,
    out_claims: &mut Vec<MeInstance<Cmt, F, K>>,
    out_wits: &mut Vec<Mat<F>>,
) -> Result<(), PiCcsError> {
    let parts = twist::split_mem_mats(mem_inst, mem_wit);
    let layout = mem_inst.twist_layout();

    for (b, mat) in parts.wa_bit_mats.iter().enumerate() {
        let comm = mem_inst
            .comms
            .get(layout.wa_bits.start + b)
            .ok_or_else(|| PiCcsError::InvalidInput("Twist: comms shorter than wa_bit_mats".into()))?;
        out_claims.push(ts::mk_me_opening_with_ccs(
            tr,
            b"twist/me_digest_val",
            params,
            s,
            comm,
            mat,
            r_val,
            m_in,
        )?);
        out_wits.push((*mat).clone());
    }

    let comm_has_write = mem_inst
        .comms
        .get(layout.has_write)
        .ok_or_else(|| PiCcsError::InvalidInput("Twist: comms shorter than has_write".into()))?;
    out_claims.push(ts::mk_me_opening_with_ccs(
        tr,
        b"twist/me_digest_val",
        params,
        s,
        comm_has_write,
        parts.has_write_mat,
        r_val,
        m_in,
    )?);
    out_wits.push(parts.has_write_mat.clone());

    let comm_inc_at_write_addr = mem_inst
        .comms
        .get(layout.inc_at_write_addr)
        .ok_or_else(|| PiCcsError::InvalidInput("Twist: comms shorter than inc_at_write_addr".into()))?;
    out_claims.push(ts::mk_me_opening_with_ccs(
        tr,
        b"twist/me_digest_val",
        params,
        s,
        comm_inc_at_write_addr,
        parts.inc_at_write_addr_mat,
        r_val,
        m_in,
    )?);
    out_wits.push(parts.inc_at_write_addr_mat.clone());

    Ok(())
}

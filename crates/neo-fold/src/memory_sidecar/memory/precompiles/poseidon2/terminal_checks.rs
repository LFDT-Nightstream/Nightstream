use super::*;
fn poseidon_cycle_openings_from_me(core_t: usize, side_mes: &[&CeClaim<Cmt, F, K>]) -> Result<Vec<K>, PiCcsError> {
    let layout = PoseidonCycleTraceLayout::new();
    let open_col_ids = poseidon_cycle_open_col_ids(&layout);
    if side_mes.is_empty() {
        return Err(PiCcsError::ProtocolError(
            "poseidon cycle ME openings missing (no cycle-lane ME claims)".into(),
        ));
    }
    let mut out = vec![K::ZERO; layout.cols()];
    let mut idx = 0usize;
    let mut prev_tile_start: Option<usize> = None;
    while idx < side_mes.len() {
        let tile_start = side_mes[idx].u_offset;
        if let Some(prev) = prev_tile_start {
            if tile_start <= prev {
                return Err(PiCcsError::ProtocolError(format!(
                    "poseidon cycle ME tiles must be strictly increasing in u_offset (prev={}, next={})",
                    prev, tile_start
                )));
            }
        }
        prev_tile_start = Some(tile_start);

        let mut tile_vals = vec![K::ZERO; open_col_ids.len()];
        let mut cursor = 0usize;
        while idx < side_mes.len() && side_mes[idx].u_offset == tile_start {
            let side_me = side_mes[idx];
            if side_me.u_len == 0 {
                return Err(PiCcsError::ProtocolError(
                    "poseidon cycle ME claim has zero u_len (likely stripped fold-claim used in terminal path)".into(),
                ));
            }
            if side_me.ct.len() < core_t {
                return Err(PiCcsError::ProtocolError(format!(
                    "poseidon cycle ME opening too short (got {}, need at least core_t={core_t})",
                    side_me.ct.len()
                )));
            }
            let chunk_len = side_me.ct.len() - core_t;
            if chunk_len == 0 {
                return Err(PiCcsError::ProtocolError(
                    "poseidon cycle ME opening chunk has zero appended cols".into(),
                ));
            }
            let next = cursor
                .checked_add(chunk_len)
                .ok_or_else(|| PiCcsError::InvalidInput("poseidon cycle opening cursor overflow".into()))?;
            if next > tile_vals.len() {
                return Err(PiCcsError::ProtocolError(format!(
                    "poseidon cycle ME openings exceed layout cols for tile u_offset={} (cursor={}, chunk_len={}, layout_cols={})",
                    tile_start,
                    cursor,
                    chunk_len,
                    tile_vals.len()
                )));
            }
            tile_vals[cursor..next].copy_from_slice(&side_me.ct[core_t..]);
            cursor = next;
            idx += 1;
        }
        if cursor != tile_vals.len() {
            return Err(PiCcsError::ProtocolError(format!(
                "poseidon cycle ME tile openings incomplete for u_offset={} (filled {}, expected {})",
                tile_start,
                cursor,
                tile_vals.len()
            )));
        }
        for (open_idx, src) in tile_vals.iter().enumerate() {
            out[open_col_ids[open_idx]] += *src;
        }
    }
    Ok(out)
}

pub(crate) fn verify_route_a_poseidon_cycle_terminals(
    core_t: usize,
    cpu_bus: &BusLayout,
    step: &StepInstanceBundle<Cmt, F, K>,
    r_time: &[K],
    r_cycle: &[K],
    batched_final_values: &[K],
    claim_plan: &RouteATimeClaimPlan,
    mem_proof: &MemSidecarProof<Cmt, F, K>,
    step_time_openings: &[crate::shard_proof_types::TimePointOpening],
    poseidon_link_chals: Option<&PoseidonLinkChallenges>,
    poseidon_cont_chals: Option<&PoseidonContinuityChallenges>,
) -> Result<(), PiCcsError> {
    let any_poseidon_claim = claim_plan.poseidon_io_link.is_some()
        || claim_plan.poseidon_bitness.is_some()
        || claim_plan.poseidon_canonical_u64.is_some()
        || claim_plan.poseidon_sidecar_link.is_some()
        || claim_plan.poseidon_mode.is_some()
        || claim_plan.poseidon_link_cycle_inv.is_some()
        || claim_plan.poseidon_link_cycle_sum.is_some()
        || claim_plan.poseidon_cont_inv.is_some()
        || claim_plan.poseidon_cont_sum.is_some();
    if !any_poseidon_claim {
        return Ok(());
    }

    let side_mes: Vec<&CeClaim<Cmt, F, K>> = mem_proof
        .poseidon_cycle_me_claims
        .iter()
        .filter(|me| {
            me.r.len() >= r_time.len()
                && me.r[..r_time.len()] == *r_time
                && me.r[r_time.len()..].iter().all(|&x| x == K::ZERO)
        })
        .collect();
    if side_mes.is_empty() {
        return Err(PiCcsError::ProtocolError(
            "poseidon cycle claims missing cycle-lane ME opening(s) at r_time".into(),
        ));
    }

    if mem_proof.wp_me_claims.len() != 1 {
        return Err(PiCcsError::ProtocolError(
            "poseidon cycle stage requires WP ME openings".into(),
        ));
    }
    let wp_me = &mem_proof.wp_me_claims[0];
    if wp_me.r.as_slice() != r_time {
        return Err(PiCcsError::ProtocolError(
            "poseidon cycle WP ME claim r mismatch (expected r_time)".into(),
        ));
    }
    if wp_me.c != step.mcs_inst.c {
        return Err(PiCcsError::ProtocolError(
            "poseidon cycle WP ME claim commitment mismatch".into(),
        ));
    }
    if wp_me.m_in != step.mcs_inst.m_in {
        return Err(PiCcsError::ProtocolError(
            "poseidon cycle WP ME claim m_in mismatch".into(),
        ));
    }

    let decode = Rv32DecodeSidecarLayout::new();
    let cpu_cols = poseidon_cpu_word_cols_for_cpu_len(step.time_columns.cpu_cols.len());
    let mut wp_required_cols = vec![
        cpu_cols.active,
        cpu_cols.instr_word,
        cpu_cols.rs1_word,
        cpu_cols.rs2_word,
        cpu_cols.rd_word,
        cpu_cols.shout_has_lookup,
    ];
    wp_required_cols.sort_unstable();
    wp_required_cols.dedup();
    let (_wp_entry, wp_open_map) = require_time_openings_covering_point(
        step_time_openings,
        wp_me.r.as_slice(),
        &wp_required_cols,
        "poseidon cycle WP",
    )?;
    let wp_open_col =
        |col_id: usize| -> Result<K, PiCcsError> { named_opening(&wp_open_map, col_id, "poseidon cycle WP") };
    let decode_open_map = decode_lookup_open_map_from_committed_openings(
        step,
        cpu_bus,
        r_time,
        step_time_openings,
        "poseidon cycle decode",
    )?;
    let decode_open_col = |col_id: usize| -> Result<K, PiCcsError> {
        decode_open_map.get(&col_id).copied().ok_or_else(|| {
            PiCcsError::ProtocolError(format!("poseidon(shared) missing decode opening col_id={col_id}"))
        })
    };

    let side_open = poseidon_cycle_openings_from_me(core_t, &side_mes)?;
    let side_layout = PoseidonCycleTraceLayout::new();
    let side_col = |col_id: usize| -> Result<K, PiCcsError> {
        side_open
            .get(col_id)
            .copied()
            .ok_or_else(|| PiCcsError::ProtocolError(format!("poseidon side opening missing col_id={col_id}")))
    };
    let op_custom = decode_open_col(decode.op_custom)?;
    let rd_has_write = decode_open_col(decode.rd_has_write)?;
    let rd_is_zero = decode_open_col(decode.rd_is_zero)?;
    let ram_has_read = decode_open_col(decode.ram_has_read)?;
    let ram_has_write = decode_open_col(decode.ram_has_write)?;
    let shout_has_lookup = wp_open_col(cpu_cols.shout_has_lookup)?;
    let funct3_bits = [
        decode_open_col(decode.funct3_bit[0])?,
        decode_open_col(decode.funct3_bit[1])?,
        decode_open_col(decode.funct3_bit[2])?,
    ];
    let funct7_bits = [
        decode_open_col(decode.funct7_bit[0])?,
        decode_open_col(decode.funct7_bit[1])?,
        decode_open_col(decode.funct7_bit[2])?,
        decode_open_col(decode.funct7_bit[3])?,
        decode_open_col(decode.funct7_bit[4])?,
        decode_open_col(decode.funct7_bit[5])?,
        decode_open_col(decode.funct7_bit[6])?,
    ];
    let rs1_val = wp_open_col(cpu_cols.rs1_word)?;
    let rs2_val = wp_open_col(cpu_cols.rs2_word)?;
    let rd_val = wp_open_col(cpu_cols.rd_word)?;

    if let Some(claim_idx) = claim_plan.poseidon_io_link {
        if claim_idx >= batched_final_values.len() {
            return Err(PiCcsError::ProtocolError(
                "poseidon/io_link claim index out of range".into(),
            ));
        }
        let residuals = poseidon_io_link_residuals(
            op_custom,
            rd_has_write,
            rd_is_zero,
            ram_has_read,
            ram_has_write,
            shout_has_lookup,
            funct3_bits,
            funct7_bits,
        );
        let mut weighted = K::ZERO;
        let weights = poseidon_io_link_weight_vector(r_cycle, residuals.len());
        for (r, w) in residuals.iter().zip(weights.iter()) {
            weighted += *w * *r;
        }
        let expected = eq_points(r_time, r_cycle) * weighted;
        if batched_final_values[claim_idx] != expected {
            return Err(PiCcsError::ProtocolError(
                "poseidon/io_link terminal value mismatch".into(),
            ));
        }
    }

    if let Some(claim_idx) = claim_plan.poseidon_bitness {
        if claim_idx >= batched_final_values.len() {
            return Err(PiCcsError::ProtocolError(
                "poseidon/bitness claim index out of range".into(),
            ));
        }
        let residuals = poseidon_bitness_residuals(op_custom, rd_has_write, rd_is_zero, funct3_bits, funct7_bits);
        let mut weighted = K::ZERO;
        let weights = poseidon_bitness_weight_vector(r_cycle, residuals.len());
        for (r, w) in residuals.iter().zip(weights.iter()) {
            weighted += *w * *r;
        }
        let expected = eq_points(r_time, r_cycle) * weighted;
        if batched_final_values[claim_idx] != expected {
            return Err(PiCcsError::ProtocolError(
                "poseidon/bitness terminal value mismatch".into(),
            ));
        }
    }

    if let Some(claim_idx) = claim_plan.poseidon_canonical_u64 {
        if claim_idx >= batched_final_values.len() {
            return Err(PiCcsError::ProtocolError(
                "poseidon/canonical_u64 claim index out of range".into(),
            ));
        }
        let side_op_squeeze = side_col(side_layout.op_squeeze)?;
        let side_squeeze_word = side_col(side_layout.squeeze_word_u32)?;
        let side_digest_lo = [
            side_col(side_layout.digest_lo(0))?,
            side_col(side_layout.digest_lo(1))?,
            side_col(side_layout.digest_lo(2))?,
            side_col(side_layout.digest_lo(3))?,
        ];
        let side_digest_hi = [
            side_col(side_layout.digest_hi(0))?,
            side_col(side_layout.digest_hi(1))?,
            side_col(side_layout.digest_hi(2))?,
            side_col(side_layout.digest_hi(3))?,
        ];
        let side_c0 = side_col(side_layout.canonical_c0)?;
        let side_c1 = side_col(side_layout.canonical_c1)?;
        let side_lo_sum = side_col(side_layout.canonical_lo_sum)?;
        let side_hi_sum = side_col(side_layout.canonical_hi_sum)?;
        let residuals = poseidon_canonical_residuals(
            funct3_bits,
            side_op_squeeze,
            side_squeeze_word,
            side_digest_lo,
            side_digest_hi,
            side_c0,
            side_c1,
            side_lo_sum,
            side_hi_sum,
        );
        let mut weighted = K::ZERO;
        let weights = poseidon_canonical_weight_vector(r_cycle, residuals.len());
        for (r, w) in residuals.iter().zip(weights.iter()) {
            weighted += *w * *r;
        }
        let expected = eq_points(r_time, r_cycle) * weighted;
        if batched_final_values[claim_idx] != expected {
            return Err(PiCcsError::ProtocolError(
                "poseidon/canonical_u64 terminal value mismatch".into(),
            ));
        }
    }

    if let Some(claim_idx) = claim_plan.poseidon_sidecar_link {
        if claim_idx >= batched_final_values.len() {
            return Err(PiCcsError::ProtocolError(
                "poseidon/sidecar_link claim index out of range".into(),
            ));
        }
        let residuals = poseidon_sidecar_link_residuals(
            op_custom,
            rd_has_write,
            funct3_bits,
            funct7_bits,
            rs1_val,
            rs2_val,
            rd_val,
            side_col(side_layout.op_absorb)?,
            side_col(side_layout.op_finalize)?,
            side_col(side_layout.op_squeeze)?,
            [
                side_col(side_layout.squeeze_idx_b0)?,
                side_col(side_layout.squeeze_idx_b1)?,
                side_col(side_layout.squeeze_idx_b2)?,
            ],
            side_col(side_layout.absorb_lo32)?,
            side_col(side_layout.absorb_hi32)?,
            side_col(side_layout.squeeze_word_u32)?,
        );
        let mut weighted = K::ZERO;
        let weights = poseidon_sidecar_link_weight_vector(r_cycle, residuals.len());
        for (r, w) in residuals.iter().zip(weights.iter()) {
            weighted += *w * *r;
        }
        let expected = eq_points(r_time, r_cycle) * weighted;
        if batched_final_values[claim_idx] != expected {
            return Err(PiCcsError::ProtocolError(
                "poseidon/sidecar_link terminal value mismatch".into(),
            ));
        }
    }

    if let Some(claim_idx) = claim_plan.poseidon_mode {
        if claim_idx >= batched_final_values.len() {
            return Err(PiCcsError::ProtocolError(
                "poseidon/mode claim index out of range".into(),
            ));
        }
        let residuals = poseidon_mode_residuals(
            side_col(side_layout.op_finalize)?,
            side_col(side_layout.op_squeeze)?,
            side_col(side_layout.mode_finalized)?,
        );
        let mut weighted = K::ZERO;
        let weights = poseidon_mode_weight_vector(r_cycle, residuals.len());
        for (r, w) in residuals.iter().zip(weights.iter()) {
            weighted += *w * *r;
        }
        let expected = eq_points(r_time, r_cycle) * weighted;
        if batched_final_values[claim_idx] != expected {
            return Err(PiCcsError::ProtocolError(
                "poseidon/mode terminal value mismatch".into(),
            ));
        }
    }

    if let Some(claim_idx) = claim_plan.poseidon_link_cycle_inv {
        if claim_idx >= batched_final_values.len() {
            return Err(PiCcsError::ProtocolError(
                "poseidon/link_cycle_inv claim index out of range".into(),
            ));
        }
        let link_chals = poseidon_link_chals
            .ok_or_else(|| PiCcsError::ProtocolError("poseidon/link_cycle_inv missing link challenges".into()))?;
        let slot0_in = [
            side_col(side_layout.slot0_in(0))?,
            side_col(side_layout.slot0_in(1))?,
            side_col(side_layout.slot0_in(2))?,
            side_col(side_layout.slot0_in(3))?,
            side_col(side_layout.slot0_in(4))?,
            side_col(side_layout.slot0_in(5))?,
            side_col(side_layout.slot0_in(6))?,
            side_col(side_layout.slot0_in(7))?,
        ];
        let slot0_out = [
            side_col(side_layout.slot0_out(0))?,
            side_col(side_layout.slot0_out(1))?,
            side_col(side_layout.slot0_out(2))?,
            side_col(side_layout.slot0_out(3))?,
            side_col(side_layout.slot0_out(4))?,
            side_col(side_layout.slot0_out(5))?,
            side_col(side_layout.slot0_out(6))?,
            side_col(side_layout.slot0_out(7))?,
        ];
        let slot1_in = [
            side_col(side_layout.slot1_in(0))?,
            side_col(side_layout.slot1_in(1))?,
            side_col(side_layout.slot1_in(2))?,
            side_col(side_layout.slot1_in(3))?,
            side_col(side_layout.slot1_in(4))?,
            side_col(side_layout.slot1_in(5))?,
            side_col(side_layout.slot1_in(6))?,
            side_col(side_layout.slot1_in(7))?,
        ];
        let slot1_out = [
            side_col(side_layout.slot1_out(0))?,
            side_col(side_layout.slot1_out(1))?,
            side_col(side_layout.slot1_out(2))?,
            side_col(side_layout.slot1_out(3))?,
            side_col(side_layout.slot1_out(4))?,
            side_col(side_layout.slot1_out(5))?,
            side_col(side_layout.slot1_out(6))?,
            side_col(side_layout.slot1_out(7))?,
        ];
        let z0 = poseidon_link_compress_tuple(
            K::ZERO,
            side_col(side_layout.call_ctr)?,
            slot0_in,
            slot0_out,
            &link_chals.eta,
        );
        let z1 = poseidon_link_compress_tuple(
            K::ONE,
            side_col(side_layout.call_ctr)?,
            slot1_in,
            slot1_out,
            &link_chals.eta,
        );
        let residuals = poseidon_cycle_link_inv_residuals(
            side_col(side_layout.do_perm_slot0)?,
            side_col(side_layout.do_perm_slot1)?,
            z0,
            z1,
            side_col(side_layout.link_u_slot0)?,
            side_col(side_layout.link_u_slot1)?,
            link_chals.beta,
        );
        let mut weighted = K::ZERO;
        let weights = poseidon_link_cycle_inv_weight_vector(r_cycle, residuals.len());
        for (r, w) in residuals.iter().zip(weights.iter()) {
            weighted += *w * *r;
        }
        let expected = eq_points(r_time, r_cycle) * weighted;
        if batched_final_values[claim_idx] != expected {
            return Err(PiCcsError::ProtocolError(
                "poseidon/link_cycle_inv terminal value mismatch".into(),
            ));
        }
    }

    if let Some(claim_idx) = claim_plan.poseidon_link_cycle_sum {
        if claim_idx >= batched_final_values.len() {
            return Err(PiCcsError::ProtocolError(
                "poseidon/link_cycle_sum claim index out of range".into(),
            ));
        }
        let expected = side_col(side_layout.do_perm_slot0)? * side_col(side_layout.link_u_slot0)?
            + side_col(side_layout.do_perm_slot1)? * side_col(side_layout.link_u_slot1)?;
        if batched_final_values[claim_idx] != expected {
            return Err(PiCcsError::ProtocolError(
                "poseidon/link_cycle_sum terminal value mismatch".into(),
            ));
        }
    }

    if let Some(claim_idx) = claim_plan.poseidon_cont_inv {
        if claim_idx >= batched_final_values.len() {
            return Err(PiCcsError::ProtocolError(
                "poseidon/cont_inv claim index out of range".into(),
            ));
        }
        let cont_chals = poseidon_cont_chals
            .ok_or_else(|| PiCcsError::ProtocolError("poseidon/cont_inv missing continuity challenges".into()))?;
        let row_active = side_col(side_layout.row_active)?;
        let is_first = side_col(side_layout.is_first)?;
        let is_last = side_col(side_layout.is_last)?;
        let cycle = side_col(side_layout.cycle)?;
        let call_pre = side_col(side_layout.call_ctr)?;
        let mode_pre = side_col(side_layout.mode_finalized)?;
        let cursor_pre = side_col(side_layout.cursor_before)?;
        let cursor_post = side_col(side_layout.cursor_after)?;
        let op_absorb = side_col(side_layout.op_absorb)?;
        let op_finalize = side_col(side_layout.op_finalize)?;
        let mut state_pre = [K::ZERO; 8];
        let mut state_post = [K::ZERO; 8];
        for i in 0..8usize {
            state_pre[i] = side_col(side_layout.state_pre(i))?;
            state_post[i] = side_col(side_layout.state_post(i))?;
        }
        let a_pre = row_active * (K::ONE - is_first);
        let a_post = row_active * (K::ONE - is_last);
        let call_post = call_pre;
        let mode_post = op_finalize + (K::ONE - op_finalize - op_absorb) * mode_pre;
        let z_pre = poseidon_cont_compress_tuple(cycle, call_pre, mode_pre, cursor_pre, state_pre, &cont_chals.eta);
        let z_post = poseidon_cont_compress_tuple(
            cycle + K::ONE,
            call_post,
            mode_post,
            cursor_post,
            state_post,
            &cont_chals.eta,
        );
        let residuals = poseidon_cycle_cont_inv_residuals(
            a_pre,
            a_post,
            z_pre,
            z_post,
            side_col(side_layout.cont_u_pre)?,
            side_col(side_layout.cont_u_post)?,
            cont_chals.beta,
        );
        let mut weighted = K::ZERO;
        let weights = poseidon_cont_cycle_inv_weight_vector(r_cycle, residuals.len());
        for (r, w) in residuals.iter().zip(weights.iter()) {
            weighted += *w * *r;
        }
        let expected = eq_points(r_time, r_cycle) * weighted;
        if batched_final_values[claim_idx] != expected {
            return Err(PiCcsError::ProtocolError(
                "poseidon/cont_inv terminal value mismatch".into(),
            ));
        }
    }

    if let Some(claim_idx) = claim_plan.poseidon_cont_sum {
        if claim_idx >= batched_final_values.len() {
            return Err(PiCcsError::ProtocolError(
                "poseidon/cont_sum claim index out of range".into(),
            ));
        }
        let row_active = side_col(side_layout.row_active)?;
        let is_first = side_col(side_layout.is_first)?;
        let is_last = side_col(side_layout.is_last)?;
        let a_pre = row_active * (K::ONE - is_first);
        let a_post = row_active * (K::ONE - is_last);
        let expected = a_pre * side_col(side_layout.cont_u_pre)? - a_post * side_col(side_layout.cont_u_post)?;
        if batched_final_values[claim_idx] != expected {
            return Err(PiCcsError::ProtocolError(
                "poseidon/cont_sum terminal value mismatch".into(),
            ));
        }
    }

    Ok(())
}

fn poseidon_local_openings_from_me(
    core_t: usize,
    local_me_claims: &[CeClaim<Cmt, F, K>],
) -> Result<Vec<K>, PiCcsError> {
    let layout = PoseidonLocalTraceLayout::new();
    let open_col_ids = poseidon_local_open_col_ids(&layout);
    if local_me_claims.is_empty() {
        return Err(PiCcsError::ProtocolError("poseidon local ME claims missing".into()));
    }
    let mut out = vec![K::ZERO; layout.cols()];
    let mut idx = 0usize;
    let mut prev_tile_start: Option<usize> = None;
    while idx < local_me_claims.len() {
        let tile_start = local_me_claims[idx].u_offset;
        if let Some(prev) = prev_tile_start {
            if tile_start <= prev {
                return Err(PiCcsError::ProtocolError(format!(
                    "poseidon local ME tiles must be strictly increasing in u_offset (prev={}, next={})",
                    prev, tile_start
                )));
            }
        }
        prev_tile_start = Some(tile_start);

        let mut tile_vals = vec![K::ZERO; open_col_ids.len()];
        let mut cursor = 0usize;
        while idx < local_me_claims.len() && local_me_claims[idx].u_offset == tile_start {
            let me = &local_me_claims[idx];
            if me.u_len == 0 {
                return Err(PiCcsError::ProtocolError(
                    "poseidon local ME claim has zero u_len (likely stripped fold-claim used in terminal path)".into(),
                ));
            }
            if me.ct.len() < core_t {
                return Err(PiCcsError::ProtocolError(format!(
                    "poseidon local ME opening too short (got {}, need at least core_t={core_t})",
                    me.ct.len()
                )));
            }
            let chunk_len = me.ct.len() - core_t;
            if chunk_len == 0 {
                return Err(PiCcsError::ProtocolError(
                    "poseidon local ME opening chunk has zero appended cols".into(),
                ));
            }
            let next = cursor
                .checked_add(chunk_len)
                .ok_or_else(|| PiCcsError::InvalidInput("poseidon local opening cursor overflow".into()))?;
            if next > tile_vals.len() {
                return Err(PiCcsError::ProtocolError(format!(
                    "poseidon local ME openings exceed layout cols for tile u_offset={} (cursor={}, chunk_len={}, layout_cols={})",
                    tile_start,
                    cursor,
                    chunk_len,
                    tile_vals.len()
                )));
            }
            tile_vals[cursor..next].copy_from_slice(&me.ct[core_t..]);
            cursor = next;
            idx += 1;
        }
        if cursor != tile_vals.len() {
            return Err(PiCcsError::ProtocolError(format!(
                "poseidon local ME tile openings incomplete for u_offset={} (filled {}, expected {})",
                tile_start,
                cursor,
                tile_vals.len()
            )));
        }
        for (open_idx, src) in tile_vals.iter().enumerate() {
            out[open_col_ids[open_idx]] += *src;
        }
    }
    Ok(out)
}

pub(crate) fn verify_route_a_poseidon_local_terminals(
    core_t: usize,
    _ell_n: usize,
    r_local: &[K],
    r_local_anchor: &[K],
    batched_final_values: &[K],
    mem_proof: &MemSidecarProof<Cmt, F, K>,
    poseidon_link_chals: Option<&PoseidonLinkChallenges>,
) -> Result<(), PiCcsError> {
    if mem_proof.poseidon_local_me_claims.is_empty() {
        return Err(PiCcsError::ProtocolError(
            "poseidon local claims require at least 1 local-lane ME opening".into(),
        ));
    }
    let expected_local_claims = crate::memory_sidecar::claim_plan::poseidon_local_time_claim_metas().len();
    if batched_final_values.len() != expected_local_claims {
        return Err(PiCcsError::ProtocolError(format!(
            "poseidon local batched finals length mismatch (expected {}, got {})",
            expected_local_claims,
            batched_final_values.len()
        )));
    }

    for local_me in mem_proof.poseidon_local_me_claims.iter() {
        if local_me.r.len() < r_local.len() {
            return Err(PiCcsError::ProtocolError(format!(
                "poseidon local ME claim r too short (got {}, expected at least {})",
                local_me.r.len(),
                r_local.len()
            )));
        }
        if local_me.r[..r_local.len()] != *r_local {
            return Err(PiCcsError::ProtocolError(
                "poseidon local ME claim r prefix mismatch (expected r_local)".into(),
            ));
        }
        if local_me.r[r_local.len()..].iter().any(|&x| x != K::ZERO) {
            return Err(PiCcsError::ProtocolError(
                "poseidon local ME claim r suffix must be zero-extended".into(),
            ));
        }
    }
    let layout = PoseidonLocalTraceLayout::new();
    let local_open = poseidon_local_openings_from_me(core_t, &mem_proof.poseidon_local_me_claims)?;
    let col = |col_id: usize| -> Result<K, PiCcsError> {
        local_open
            .get(col_id)
            .copied()
            .ok_or_else(|| PiCcsError::ProtocolError(format!("poseidon local opening missing col={col_id}")))
    };
    if r_local_anchor.len() != r_local.len() {
        return Err(PiCcsError::ProtocolError(format!(
            "poseidon local anchor length mismatch (anchor={}, r_local={})",
            r_local_anchor.len(),
            r_local.len()
        )));
    }
    let eq = eq_points(r_local, r_local_anchor);
    let row_active = col(layout.row_active)?;
    let has_round = col(layout.has_round)?;
    let is_row_start = col(layout.is_row_start)?;
    let slot = col(layout.slot)?;
    let call_ctr = col(layout.call_ctr)?;
    let cycle_call_ctr_local = col(layout.cycle_call_ctr)?;
    let cycle_selected_perm_local = col(layout.cycle_selected_perm)?;

    let mut in_arr = [K::ZERO; 8];
    let mut out_arr = [K::ZERO; 8];
    let mut cycle_selected_in_local = [K::ZERO; 8];
    let mut cycle_selected_out_local = [K::ZERO; 8];
    for i in 0..8 {
        in_arr[i] = col(layout.state_in(i))?;
        out_arr[i] = col(layout.state_out(i))?;
        cycle_selected_in_local[i] = col(layout.cycle_selected_in(i))?;
        cycle_selected_out_local[i] = col(layout.cycle_selected_out(i))?;
    }

    let round_public = poseidon_round_public_eval_at_point(r_local)?;
    let round_residuals = poseidon_local_round_residuals(
        row_active,
        has_round,
        round_public.is_step_mds,
        round_public.is_step_external,
        round_public.is_step_internal,
        round_public.is_step_no_round,
        in_arr,
        out_arr,
        round_public.external_rc,
        round_public.internal_rc,
    );
    let round_weights = poseidon_local_round_weight_vector(r_local_anchor, round_residuals.len());
    let round_weighted = round_residuals
        .iter()
        .zip(round_weights.iter())
        .fold(K::ZERO, |acc, (r, w)| acc + (*w * *r));
    let round_expected = eq * round_weighted;
    if batched_final_values[0] != round_expected {
        return Err(PiCcsError::ProtocolError(
            "poseidon/round terminal value mismatch".into(),
        ));
    }

    let transition_residuals = poseidon_local_transition_residuals(row_active, has_round, in_arr, out_arr);
    let transition_weights = poseidon_local_transition_weight_vector(r_local_anchor, transition_residuals.len());
    let transition_weighted = transition_residuals
        .iter()
        .zip(transition_weights.iter())
        .fold(K::ZERO, |acc, (r, w)| acc + (*w * *r));
    let transition_expected = eq * transition_weighted;
    if batched_final_values[1] != transition_expected {
        return Err(PiCcsError::ProtocolError(
            "poseidon/transition terminal value mismatch".into(),
        ));
    }

    let link_residuals = poseidon_local_link_residuals(
        row_active,
        has_round,
        is_row_start,
        slot,
        call_ctr,
        cycle_call_ctr_local,
        cycle_selected_perm_local,
        cycle_selected_in_local,
        cycle_selected_out_local,
        in_arr,
        out_arr,
        poseidon_step_selectors_from_point(r_local),
        poseidon_step_selector_inv_weights_from_anchor(r_local_anchor)?,
        bitness_weights(r_local_anchor, 8, 0x5032_4C4C_494D_4258u64)
            .try_into()
            .map_err(|_| PiCcsError::ProtocolError("poseidon local: limb mix conversion failed".into()))?,
    );
    let link_weights = poseidon_local_link_weight_vector(r_local_anchor, link_residuals.len());
    let link_weighted = link_residuals
        .iter()
        .zip(link_weights.iter())
        .fold(K::ZERO, |acc, (r, w)| acc + (*w * *r));
    let link_expected = eq * link_weighted;
    if batched_final_values[2] != link_expected {
        return Err(PiCcsError::ProtocolError(
            "poseidon/cycle_local_link terminal value mismatch".into(),
        ));
    }

    let local_link_u = col(layout.link_u_local)?;
    let link_chals = poseidon_link_chals
        .ok_or_else(|| PiCcsError::ProtocolError("poseidon local-link checks missing link challenges".into()))?;
    let a_start = row_active * is_row_start;
    let z_local = poseidon_link_compress_tuple(
        slot,
        call_ctr,
        cycle_selected_in_local,
        cycle_selected_out_local,
        &link_chals.eta,
    );
    let link_local_inv_residuals = poseidon_local_link_inv_residuals(a_start, z_local, local_link_u, link_chals.beta);
    let link_local_inv_weights = poseidon_link_local_inv_weight_vector(r_local_anchor, link_local_inv_residuals.len());
    let link_local_inv_weighted = link_local_inv_residuals
        .iter()
        .zip(link_local_inv_weights.iter())
        .fold(K::ZERO, |acc, (r, w)| acc + (*w * *r));
    let link_local_inv_expected = eq * link_local_inv_weighted;
    if batched_final_values[3] != link_local_inv_expected {
        return Err(PiCcsError::ProtocolError(
            "poseidon/link_local_inv terminal value mismatch".into(),
        ));
    }
    let link_local_sum_expected = a_start * local_link_u;
    if batched_final_values[4] != link_local_sum_expected {
        return Err(PiCcsError::ProtocolError(
            "poseidon/link_local_sum terminal value mismatch".into(),
        ));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stripped_poseidon_me_claim(u_offset: usize) -> CeClaim<Cmt, F, K> {
        CeClaim {
            c: Cmt::zeros(1, 1),
            X: neo_ccs::Mat::zero(neo_math::D, 0, F::ZERO),
            r: Vec::new(),
            s_col: Vec::new(),
            y_ring: Vec::new(),
            ct: vec![K::ZERO],
            aux_openings: Vec::new(),
            y_zcol: Vec::new(),
            m_in: 0,
            fold_digest: [0u8; 32],
            c_step_coords: Vec::new(),
            u_offset,
            u_len: 0,
        }
    }

    #[test]
    fn poseidon_cycle_openings_rejects_stripped_fold_claim() {
        let claim = stripped_poseidon_me_claim(0);
        let err = poseidon_cycle_openings_from_me(0, &[&claim])
            .expect_err("cycle terminal path must reject stripped fold claims");
        match err {
            PiCcsError::ProtocolError(msg) => {
                assert!(
                    msg.contains("zero u_len"),
                    "unexpected cycle stripped-claim error: {msg}"
                );
            }
            other => panic!("unexpected error variant: {other:?}"),
        }
    }

    #[test]
    fn poseidon_local_openings_rejects_stripped_fold_claim() {
        let claim = stripped_poseidon_me_claim(0);
        let err = poseidon_local_openings_from_me(0, &[claim])
            .expect_err("local terminal path must reject stripped fold claims");
        match err {
            PiCcsError::ProtocolError(msg) => {
                assert!(
                    msg.contains("zero u_len"),
                    "unexpected local stripped-claim error: {msg}"
                );
            }
            other => panic!("unexpected error variant: {other:?}"),
        }
    }
}

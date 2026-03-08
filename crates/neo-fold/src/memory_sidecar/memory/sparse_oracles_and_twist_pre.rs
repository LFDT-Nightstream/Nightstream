use super::*;

pub(crate) fn sparse_trace_col_from_values(
    m_in: usize,
    ell_n: usize,
    values: &[K],
) -> Result<SparseIdxVec<K>, PiCcsError> {
    let pow2_cycle = 1usize
        .checked_shl(ell_n as u32)
        .ok_or_else(|| PiCcsError::InvalidInput("WB/WP: 2^ell_n overflow".into()))?;
    let t_len = values.len();
    if m_in
        .checked_add(t_len)
        .ok_or_else(|| PiCcsError::InvalidInput("WB/WP: m_in + t_len overflow".into()))?
        > pow2_cycle
    {
        return Err(PiCcsError::InvalidInput(format!(
            "WB/WP: trace rows out of range (m_in={m_in}, t_len={t_len}, 2^ell_n={pow2_cycle})"
        )));
    }
    let mut entries = Vec::new();
    for (j, &v) in values.iter().enumerate() {
        if v != K::ZERO {
            entries.push((m_in + j, v));
        }
    }
    Ok(SparseIdxVec::from_entries(pow2_cycle, entries))
}

#[inline]
pub(crate) fn decode_k_to_u32(v: K, ctx: &str) -> Result<u32, PiCcsError> {
    let coeffs = v.as_coeffs();
    if coeffs.iter().skip(1).any(|&c| c != F::ZERO) {
        return Err(PiCcsError::ProtocolError(format!(
            "{ctx}: expected base-field value while decoding shared decode columns"
        )));
    }
    let lo = coeffs
        .first()
        .copied()
        .ok_or_else(|| PiCcsError::ProtocolError(format!("{ctx}: missing base coefficient")))?
        .as_canonical_u64();
    if lo > u32::MAX as u64 {
        return Err(PiCcsError::ProtocolError(format!(
            "{ctx}: value {lo} exceeds u32 range while decoding shared decode columns"
        )));
    }
    Ok(lo as u32)
}

pub(crate) fn resolve_shared_decode_lookup_lut_indices(
    step: &StepWitnessBundle<Cmt, F, K>,
    decode_layout: &Rv32DecodeSidecarLayout,
) -> Result<(Vec<usize>, Vec<(usize, usize)>), PiCcsError> {
    let decode_open_cols = rv32_decode_lookup_transport_cols(decode_layout);
    let mut decode_lut_slots = Vec::with_capacity(decode_open_cols.len());
    for &col_id in decode_open_cols.iter() {
        let table_id = rv32_decode_lookup_table_id_for_col(col_id);
        let lut_idx = step
            .lut_instances
            .iter()
            .position(|(inst, _)| inst.table_id == table_id)
            .ok_or_else(|| {
                PiCcsError::ProtocolError(format!(
                    "W2(shared): missing decode lookup table_id={table_id} for col_id={col_id}"
                ))
            })?;
        let val_slot = rv32_decode_lookup_val_slot_for_col(col_id).ok_or_else(|| {
            PiCcsError::ProtocolError(format!(
                "W2(shared): decode col_id={col_id} is not part of decode lookup transport slot map"
            ))
        })?;
        decode_lut_slots.push((lut_idx, val_slot));
    }

    Ok((decode_open_cols, decode_lut_slots))
}

pub(crate) fn resolve_shared_width_lookup_lut_indices(
    step: &StepWitnessBundle<Cmt, F, K>,
    width_layout: &Rv32WidthSidecarLayout,
) -> Result<(Vec<usize>, Vec<(usize, usize)>), PiCcsError> {
    let width_open_cols = rv32_width_lookup_backed_cols(width_layout);
    let mut width_lut_slots = Vec::with_capacity(width_open_cols.len());
    for &col_id in width_open_cols.iter() {
        let table_id = rv32_width_lookup_table_id_for_col(col_id);
        let lut_idx = step
            .lut_instances
            .iter()
            .position(|(inst, _)| inst.table_id == table_id)
            .ok_or_else(|| {
                PiCcsError::ProtocolError(format!(
                    "W3(shared): missing width lookup table_id={table_id} for col_id={col_id}"
                ))
            })?;
        let val_slot = rv32_width_lookup_val_slot_for_col(col_id).ok_or_else(|| {
            PiCcsError::ProtocolError(format!(
                "W3(shared): width col_id={col_id} is not part of width lookup transport slot map"
            ))
        })?;
        width_lut_slots.push((lut_idx, val_slot));
    }

    Ok((width_open_cols, width_lut_slots))
}

pub(crate) struct WeightedMaskOracleSparseTime {
    bit_idx: usize,
    r_cycle: Vec<K>,
    prefix_eq: K,
    active: SparseIdxVec<K>,
    cols: Vec<SparseIdxVec<K>>,
    support: SparseIdxVec<K>,
    weights: Vec<K>,
}

impl WeightedMaskOracleSparseTime {
    pub(crate) fn new(active: SparseIdxVec<K>, cols: Vec<SparseIdxVec<K>>, weights: Vec<K>, r_cycle: &[K]) -> Self {
        debug_assert_eq!(cols.len(), weights.len());
        let mut support_cols = Vec::with_capacity(cols.len() + 1);
        support_cols.push(active.clone());
        support_cols.extend(cols.iter().cloned());
        let support = sparse_union_support(&support_cols);
        Self {
            bit_idx: 0,
            r_cycle: r_cycle.to_vec(),
            prefix_eq: K::ONE,
            active,
            cols,
            support,
            weights,
        }
    }
}

impl RoundOracle for WeightedMaskOracleSparseTime {
    fn evals_at(&mut self, points: &[K]) -> Vec<K> {
        if self.cols.is_empty() {
            return vec![K::ZERO; points.len()];
        }

        if self.active.len() == 1 {
            let gate = K::ONE - self.active.singleton_value();
            let mut acc = K::ZERO;
            for (col, w) in self.cols.iter().zip(self.weights.iter()) {
                acc += *w * col.singleton_value();
            }
            return vec![self.prefix_eq * gate * acc; points.len()];
        }

        let mut ys = vec![K::ZERO; points.len()];
        let mut prev_pair = usize::MAX;
        for &(idx, _v) in self.support.entries() {
            let pair = idx >> 1;
            if pair == prev_pair {
                continue;
            }
            prev_pair = pair;
            let child0 = 2 * pair;
            let child1 = child0 + 1;

            let gate0 = K::ONE - self.active.get(child0);
            let gate1 = K::ONE - self.active.get(child1);
            if gate0 == K::ZERO && gate1 == K::ZERO {
                continue;
            }

            let (chi0, chi1) = chi_cycle_children(&self.r_cycle, self.bit_idx, self.prefix_eq, pair);
            for (i, &x) in points.iter().enumerate() {
                let chi_x = interp(chi0, chi1, x);
                if chi_x == K::ZERO {
                    continue;
                }
                let gate_x = interp(gate0, gate1, x);
                if gate_x == K::ZERO {
                    continue;
                }
                let mut sum_x = K::ZERO;
                for (col, w) in self.cols.iter().zip(self.weights.iter()) {
                    let c0 = col.get(child0);
                    let c1 = col.get(child1);
                    if c0 == K::ZERO && c1 == K::ZERO {
                        continue;
                    }
                    sum_x += *w * interp(c0, c1, x);
                }
                ys[i] += chi_x * gate_x * sum_x;
            }
        }
        ys
    }

    fn num_rounds(&self) -> usize {
        self.r_cycle.len().saturating_sub(self.bit_idx)
    }

    fn degree_bound(&self) -> usize {
        3
    }

    fn fold(&mut self, r: K) {
        if self.num_rounds() == 0 {
            return;
        }
        self.prefix_eq *= eq_single_k(r, self.r_cycle[self.bit_idx]);
        self.active.fold_round_in_place(r);
        for col in self.cols.iter_mut() {
            col.fold_round_in_place(r);
        }
        self.support.fold_support_round_in_place();
        self.bit_idx += 1;
    }
}

pub(crate) struct FormulaOracleSparseTime<EF>
where
    EF: FnMut(&[K]) -> K,
{
    bit_idx: usize,
    r_cycle: Vec<K>,
    prefix_eq: K,
    cols: Vec<SparseIdxVec<K>>,
    support: SparseIdxVec<K>,
    degree_bound: usize,
    eval_fn: EF,
    pair_marks: Vec<u32>,
    pair_epoch: u32,
    pair_scratch: Vec<usize>,
    col_child0: Vec<K>,
    col_child1: Vec<K>,
    eval_vals: Vec<K>,
}

#[inline]
fn sparse_union_support(cols: &[SparseIdxVec<K>]) -> SparseIdxVec<K> {
    if cols.is_empty() {
        return SparseIdxVec::new(1);
    }
    let len = cols[0].len();
    let mut seen = vec![false; len];
    for col in cols {
        debug_assert_eq!(col.len(), len);
        for &(idx, _v) in col.entries() {
            seen[idx] = true;
        }
    }
    let mut entries = Vec::new();
    for (idx, hit) in seen.into_iter().enumerate() {
        if hit {
            entries.push((idx, K::ONE));
        }
    }
    SparseIdxVec::from_entries(len, entries)
}

impl<EF> FormulaOracleSparseTime<EF>
where
    EF: FnMut(&[K]) -> K,
{
    pub(crate) fn new(cols: Vec<SparseIdxVec<K>>, degree_bound: usize, r_cycle: &[K], eval_fn: EF) -> Self {
        let col_count = cols.len();
        let support = sparse_union_support(&cols);
        Self {
            bit_idx: 0,
            r_cycle: r_cycle.to_vec(),
            prefix_eq: K::ONE,
            cols,
            support,
            degree_bound,
            eval_fn,
            pair_marks: Vec::new(),
            pair_epoch: 1,
            pair_scratch: Vec::new(),
            col_child0: vec![K::ZERO; col_count],
            col_child1: vec![K::ZERO; col_count],
            eval_vals: vec![K::ZERO; col_count],
        }
    }
}

impl<EF> RoundOracle for FormulaOracleSparseTime<EF>
where
    EF: FnMut(&[K]) -> K,
{
    fn evals_at(&mut self, points: &[K]) -> Vec<K> {
        if self.cols.is_empty() {
            return vec![K::ZERO; points.len()];
        }
        if self.cols[0].len() == 1 {
            for (j, col) in self.cols.iter().enumerate() {
                self.eval_vals[j] = col.singleton_value();
            }
            let v = self.prefix_eq * (self.eval_fn)(&self.eval_vals[..self.cols.len()]);
            return vec![v; points.len()];
        }

        let pair_domain = self.support.len() >> 1;
        if self.pair_marks.len() < pair_domain {
            self.pair_marks.resize(pair_domain, 0);
        }
        self.pair_epoch = self.pair_epoch.wrapping_add(1);
        if self.pair_epoch == 0 {
            self.pair_marks.fill(0);
            self.pair_epoch = 1;
        }

        self.pair_scratch.clear();
        let epoch = self.pair_epoch;
        for &(idx, _v) in self.support.entries() {
            let pair = idx >> 1;
            if self.pair_marks[pair] != epoch {
                self.pair_marks[pair] = epoch;
                self.pair_scratch.push(pair);
            }
        }

        let mut ys = vec![K::ZERO; points.len()];
        for &pair in self.pair_scratch.iter() {
            let child0 = 2 * pair;
            let child1 = child0 + 1;
            for (j, col) in self.cols.iter().enumerate() {
                self.col_child0[j] = col.get(child0);
                self.col_child1[j] = col.get(child1);
            }
            let (chi0, chi1) = chi_cycle_children(&self.r_cycle, self.bit_idx, self.prefix_eq, pair);
            for (i, &x) in points.iter().enumerate() {
                let chi_x = interp(chi0, chi1, x);
                if chi_x == K::ZERO {
                    continue;
                }
                for j in 0..self.cols.len() {
                    let v0 = self.col_child0[j];
                    let v1 = self.col_child1[j];
                    if v0 == K::ZERO && v1 == K::ZERO {
                        self.eval_vals[j] = K::ZERO;
                        continue;
                    }
                    self.eval_vals[j] = interp(v0, v1, x);
                }
                let f_x = (self.eval_fn)(&self.eval_vals[..self.cols.len()]);
                if f_x == K::ZERO {
                    continue;
                }
                ys[i] += chi_x * f_x;
            }
        }
        ys
    }

    fn num_rounds(&self) -> usize {
        self.r_cycle.len().saturating_sub(self.bit_idx)
    }

    fn degree_bound(&self) -> usize {
        self.degree_bound
    }

    fn fold(&mut self, r: K) {
        if self.num_rounds() == 0 {
            return;
        }
        self.prefix_eq *= eq_single_k(r, self.r_cycle[self.bit_idx]);
        for col in self.cols.iter_mut() {
            col.fold_round_in_place(r);
        }
        self.support.fold_support_round_in_place();
        self.bit_idx += 1;
    }
}

pub(crate) struct ShoutGammaValueSharedOracleSparseTime {
    bit_idx: usize,
    r_cycle: Vec<K>,
    prefix_eq: K,
    has_col: SparseIdxVec<K>,
    val_cols: Vec<SparseIdxVec<K>>,
    weights: Vec<K>,
    support: SparseIdxVec<K>,
    pair_marks: Vec<u32>,
    pair_epoch: u32,
    pair_scratch: Vec<usize>,
    val_child0: Vec<K>,
    val_child1: Vec<K>,
}

impl ShoutGammaValueSharedOracleSparseTime {
    pub(crate) fn new(
        has_col: SparseIdxVec<K>,
        val_cols: Vec<SparseIdxVec<K>>,
        weights: Vec<K>,
        r_cycle: &[K],
    ) -> Self {
        debug_assert_eq!(val_cols.len(), weights.len());
        let mut support_cols = Vec::with_capacity(1 + val_cols.len());
        support_cols.push(has_col.clone());
        support_cols.extend(val_cols.iter().cloned());
        let support = sparse_union_support(&support_cols);
        let lane_count = val_cols.len();
        Self {
            bit_idx: 0,
            r_cycle: r_cycle.to_vec(),
            prefix_eq: K::ONE,
            has_col,
            val_cols,
            weights,
            support,
            pair_marks: Vec::new(),
            pair_epoch: 1,
            pair_scratch: Vec::new(),
            val_child0: vec![K::ZERO; lane_count],
            val_child1: vec![K::ZERO; lane_count],
        }
    }
}

impl RoundOracle for ShoutGammaValueSharedOracleSparseTime {
    fn evals_at(&mut self, points: &[K]) -> Vec<K> {
        if self.val_cols.is_empty() {
            return vec![K::ZERO; points.len()];
        }
        if self.has_col.len() == 1 {
            let has = self.has_col.singleton_value();
            if has == K::ZERO {
                return vec![K::ZERO; points.len()];
            }
            let mut weighted_vals = K::ZERO;
            for (col, w) in self.val_cols.iter().zip(self.weights.iter()) {
                let v = col.singleton_value();
                if v != K::ZERO {
                    weighted_vals += *w * v;
                }
            }
            let out = self.prefix_eq * has * weighted_vals;
            return vec![out; points.len()];
        }

        let pair_domain = self.support.len() >> 1;
        if self.pair_marks.len() < pair_domain {
            self.pair_marks.resize(pair_domain, 0);
        }
        self.pair_epoch = self.pair_epoch.wrapping_add(1);
        if self.pair_epoch == 0 {
            self.pair_marks.fill(0);
            self.pair_epoch = 1;
        }
        self.pair_scratch.clear();
        let epoch = self.pair_epoch;
        for &(idx, _v) in self.support.entries() {
            let pair = idx >> 1;
            if self.pair_marks[pair] != epoch {
                self.pair_marks[pair] = epoch;
                self.pair_scratch.push(pair);
            }
        }

        let mut ys = vec![K::ZERO; points.len()];
        for &pair in self.pair_scratch.iter() {
            let child0 = 2 * pair;
            let child1 = child0 + 1;
            let has0 = self.has_col.get(child0);
            let has1 = self.has_col.get(child1);
            if has0 == K::ZERO && has1 == K::ZERO {
                continue;
            }
            for (j, col) in self.val_cols.iter().enumerate() {
                self.val_child0[j] = col.get(child0);
                self.val_child1[j] = col.get(child1);
            }
            let (chi0, chi1) = chi_cycle_children(&self.r_cycle, self.bit_idx, self.prefix_eq, pair);
            for (i, &x) in points.iter().enumerate() {
                let chi_x = interp(chi0, chi1, x);
                if chi_x == K::ZERO {
                    continue;
                }
                let has_x = interp(has0, has1, x);
                if has_x == K::ZERO {
                    continue;
                }
                let mut weighted_vals = K::ZERO;
                for j in 0..self.val_cols.len() {
                    let v0 = self.val_child0[j];
                    let v1 = self.val_child1[j];
                    if v0 == K::ZERO && v1 == K::ZERO {
                        continue;
                    }
                    weighted_vals += self.weights[j] * interp(v0, v1, x);
                }
                if weighted_vals == K::ZERO {
                    continue;
                }
                ys[i] += chi_x * has_x * weighted_vals;
            }
        }
        ys
    }

    fn num_rounds(&self) -> usize {
        self.r_cycle.len().saturating_sub(self.bit_idx)
    }

    fn degree_bound(&self) -> usize {
        3
    }

    fn fold(&mut self, r: K) {
        if self.num_rounds() == 0 {
            return;
        }
        self.prefix_eq *= eq_single_k(r, self.r_cycle[self.bit_idx]);
        self.has_col.fold_round_in_place(r);
        for col in self.val_cols.iter_mut() {
            col.fold_round_in_place(r);
        }
        self.support.fold_support_round_in_place();
        self.bit_idx += 1;
    }
}

pub(crate) struct ShoutGammaValueOracleSparseTime {
    bit_idx: usize,
    r_cycle: Vec<K>,
    prefix_eq: K,
    has_cols: Vec<SparseIdxVec<K>>,
    val_cols: Vec<SparseIdxVec<K>>,
    weights: Vec<K>,
    support: SparseIdxVec<K>,
    pair_marks: Vec<u32>,
    pair_epoch: u32,
    pair_scratch: Vec<usize>,
    has_child0: Vec<K>,
    has_child1: Vec<K>,
    val_child0: Vec<K>,
    val_child1: Vec<K>,
}

impl ShoutGammaValueOracleSparseTime {
    pub(crate) fn new(
        has_cols: Vec<SparseIdxVec<K>>,
        val_cols: Vec<SparseIdxVec<K>>,
        weights: Vec<K>,
        r_cycle: &[K],
    ) -> Self {
        debug_assert_eq!(has_cols.len(), val_cols.len());
        debug_assert_eq!(has_cols.len(), weights.len());
        let mut support_cols = Vec::with_capacity(has_cols.len() + val_cols.len());
        support_cols.extend(has_cols.iter().cloned());
        support_cols.extend(val_cols.iter().cloned());
        let support = sparse_union_support(&support_cols);
        let lane_count = has_cols.len();
        Self {
            bit_idx: 0,
            r_cycle: r_cycle.to_vec(),
            prefix_eq: K::ONE,
            has_cols,
            val_cols,
            weights,
            support,
            pair_marks: Vec::new(),
            pair_epoch: 1,
            pair_scratch: Vec::new(),
            has_child0: vec![K::ZERO; lane_count],
            has_child1: vec![K::ZERO; lane_count],
            val_child0: vec![K::ZERO; lane_count],
            val_child1: vec![K::ZERO; lane_count],
        }
    }
}

impl RoundOracle for ShoutGammaValueOracleSparseTime {
    fn evals_at(&mut self, points: &[K]) -> Vec<K> {
        if self.has_cols.is_empty() {
            return vec![K::ZERO; points.len()];
        }
        if self.has_cols[0].len() == 1 {
            let mut acc = K::ZERO;
            for i in 0..self.has_cols.len() {
                let has = self.has_cols[i].singleton_value();
                let val = self.val_cols[i].singleton_value();
                if has == K::ZERO || val == K::ZERO {
                    continue;
                }
                acc += self.weights[i] * has * val;
            }
            let out = self.prefix_eq * acc;
            return vec![out; points.len()];
        }

        let pair_domain = self.support.len() >> 1;
        if self.pair_marks.len() < pair_domain {
            self.pair_marks.resize(pair_domain, 0);
        }
        self.pair_epoch = self.pair_epoch.wrapping_add(1);
        if self.pair_epoch == 0 {
            self.pair_marks.fill(0);
            self.pair_epoch = 1;
        }
        self.pair_scratch.clear();
        let epoch = self.pair_epoch;
        for &(idx, _v) in self.support.entries() {
            let pair = idx >> 1;
            if self.pair_marks[pair] != epoch {
                self.pair_marks[pair] = epoch;
                self.pair_scratch.push(pair);
            }
        }

        let mut ys = vec![K::ZERO; points.len()];
        for &pair in self.pair_scratch.iter() {
            let child0 = 2 * pair;
            let child1 = child0 + 1;
            for i in 0..self.has_cols.len() {
                self.has_child0[i] = self.has_cols[i].get(child0);
                self.has_child1[i] = self.has_cols[i].get(child1);
                self.val_child0[i] = self.val_cols[i].get(child0);
                self.val_child1[i] = self.val_cols[i].get(child1);
            }
            let (chi0, chi1) = chi_cycle_children(&self.r_cycle, self.bit_idx, self.prefix_eq, pair);
            for (pi, &x) in points.iter().enumerate() {
                let chi_x = interp(chi0, chi1, x);
                if chi_x == K::ZERO {
                    continue;
                }
                let mut acc = K::ZERO;
                for i in 0..self.has_cols.len() {
                    let h0 = self.has_child0[i];
                    let h1 = self.has_child1[i];
                    if h0 == K::ZERO && h1 == K::ZERO {
                        continue;
                    }
                    let has = interp(h0, h1, x);
                    if has == K::ZERO {
                        continue;
                    }
                    let v0 = self.val_child0[i];
                    let v1 = self.val_child1[i];
                    if v0 == K::ZERO && v1 == K::ZERO {
                        continue;
                    }
                    let val = interp(v0, v1, x);
                    if val == K::ZERO {
                        continue;
                    }
                    acc += self.weights[i] * has * val;
                }
                if acc == K::ZERO {
                    continue;
                }
                ys[pi] += chi_x * acc;
            }
        }
        ys
    }

    fn num_rounds(&self) -> usize {
        self.r_cycle.len().saturating_sub(self.bit_idx)
    }

    fn degree_bound(&self) -> usize {
        3
    }

    fn fold(&mut self, r: K) {
        if self.num_rounds() == 0 {
            return;
        }
        self.prefix_eq *= eq_single_k(r, self.r_cycle[self.bit_idx]);
        for col in self.has_cols.iter_mut() {
            col.fold_round_in_place(r);
        }
        for col in self.val_cols.iter_mut() {
            col.fold_round_in_place(r);
        }
        self.support.fold_support_round_in_place();
        self.bit_idx += 1;
    }
}

pub(crate) struct ShoutGammaAdapterSharedOracleSparseTime {
    bit_idx: usize,
    r_cycle: Vec<K>,
    prefix_eq: K,
    has_col: SparseIdxVec<K>,
    addr_cols: Vec<SparseIdxVec<K>>,
    coeff_sum: K,
    eq_alpha: Vec<K>,
    eq_beta: Vec<K>,
    support: SparseIdxVec<K>,
    pair_marks: Vec<u32>,
    pair_epoch: u32,
    pair_scratch: Vec<usize>,
    addr_child0: Vec<K>,
    addr_child1: Vec<K>,
}

impl ShoutGammaAdapterSharedOracleSparseTime {
    pub(crate) fn new(
        has_col: SparseIdxVec<K>,
        addr_cols: Vec<SparseIdxVec<K>>,
        coeff_sum: K,
        eq_alpha: Vec<K>,
        eq_beta: Vec<K>,
        r_cycle: &[K],
    ) -> Self {
        debug_assert_eq!(addr_cols.len(), eq_alpha.len());
        debug_assert_eq!(addr_cols.len(), eq_beta.len());
        let mut support_cols = Vec::with_capacity(1 + addr_cols.len());
        support_cols.push(has_col.clone());
        support_cols.extend(addr_cols.iter().cloned());
        let support = sparse_union_support(&support_cols);
        let ell = addr_cols.len();
        Self {
            bit_idx: 0,
            r_cycle: r_cycle.to_vec(),
            prefix_eq: K::ONE,
            has_col,
            addr_cols,
            coeff_sum,
            eq_alpha,
            eq_beta,
            support,
            pair_marks: Vec::new(),
            pair_epoch: 1,
            pair_scratch: Vec::new(),
            addr_child0: vec![K::ZERO; ell],
            addr_child1: vec![K::ZERO; ell],
        }
    }
}

impl RoundOracle for ShoutGammaAdapterSharedOracleSparseTime {
    fn evals_at(&mut self, points: &[K]) -> Vec<K> {
        if self.addr_cols.is_empty() {
            return vec![K::ZERO; points.len()];
        }
        if self.has_col.len() == 1 {
            let has = self.has_col.singleton_value();
            if has == K::ZERO {
                return vec![K::ZERO; points.len()];
            }
            let mut eq = K::ONE;
            for i in 0..self.addr_cols.len() {
                let bit = self.addr_cols[i].singleton_value();
                eq *= bit * self.eq_alpha[i] + self.eq_beta[i];
            }
            let out = self.prefix_eq * self.coeff_sum * has * eq;
            return vec![out; points.len()];
        }

        let pair_domain = self.support.len() >> 1;
        if self.pair_marks.len() < pair_domain {
            self.pair_marks.resize(pair_domain, 0);
        }
        self.pair_epoch = self.pair_epoch.wrapping_add(1);
        if self.pair_epoch == 0 {
            self.pair_marks.fill(0);
            self.pair_epoch = 1;
        }
        self.pair_scratch.clear();
        let epoch = self.pair_epoch;
        for &(idx, _v) in self.support.entries() {
            let pair = idx >> 1;
            if self.pair_marks[pair] != epoch {
                self.pair_marks[pair] = epoch;
                self.pair_scratch.push(pair);
            }
        }

        let mut ys = vec![K::ZERO; points.len()];
        let mut addr_hints = vec![0usize; self.addr_cols.len()];
        let mut has_hint = 0usize;
        for &pair in self.pair_scratch.iter() {
            let child0 = 2 * pair;
            let child1 = child0 + 1;
            let has0 = self.has_col.get_with_hint(child0, &mut has_hint);
            let has1 = self.has_col.get_with_hint(child1, &mut has_hint);
            if has0 == K::ZERO && has1 == K::ZERO {
                continue;
            }
            for i in 0..self.addr_cols.len() {
                self.addr_child0[i] = self.addr_cols[i].get_with_hint(child0, &mut addr_hints[i]);
                self.addr_child1[i] = self.addr_cols[i].get_with_hint(child1, &mut addr_hints[i]);
            }
            let (chi0, chi1) = chi_cycle_children(&self.r_cycle, self.bit_idx, self.prefix_eq, pair);
            for (pi, &x) in points.iter().enumerate() {
                let chi_x = interp(chi0, chi1, x);
                if chi_x == K::ZERO {
                    continue;
                }
                let has = interp(has0, has1, x);
                if has == K::ZERO {
                    continue;
                }
                let mut eq = K::ONE;
                for i in 0..self.addr_cols.len() {
                    let b0 = self.addr_child0[i];
                    let b1 = self.addr_child1[i];
                    let bit = if b0 == K::ZERO && b1 == K::ZERO {
                        K::ZERO
                    } else {
                        interp(b0, b1, x)
                    };
                    eq *= bit * self.eq_alpha[i] + self.eq_beta[i];
                    if eq == K::ZERO {
                        break;
                    }
                }
                if eq == K::ZERO {
                    continue;
                }
                ys[pi] += chi_x * self.coeff_sum * has * eq;
            }
        }
        ys
    }

    fn num_rounds(&self) -> usize {
        self.r_cycle.len().saturating_sub(self.bit_idx)
    }

    fn degree_bound(&self) -> usize {
        2 + self.addr_cols.len()
    }

    fn fold(&mut self, r: K) {
        if self.num_rounds() == 0 {
            return;
        }
        self.prefix_eq *= eq_single_k(r, self.r_cycle[self.bit_idx]);
        self.has_col.fold_round_in_place(r);
        for col in self.addr_cols.iter_mut() {
            col.fold_round_in_place(r);
        }
        self.support.fold_support_round_in_place();
        self.bit_idx += 1;
    }
}

pub(crate) struct ShoutGammaAdapterOracleSparseTime {
    bit_idx: usize,
    r_cycle: Vec<K>,
    prefix_eq: K,
    addr_cols: Vec<SparseIdxVec<K>>,
    has_cols: Vec<SparseIdxVec<K>>,
    coeffs: Vec<K>,
    eq_alpha: Vec<K>,
    eq_beta: Vec<K>,
    support: SparseIdxVec<K>,
    pair_marks: Vec<u32>,
    pair_epoch: u32,
    pair_scratch: Vec<usize>,
    addr_child0: Vec<K>,
    addr_child1: Vec<K>,
    has_child0: Vec<K>,
    has_child1: Vec<K>,
    eq_scratch: Vec<K>,
    weighted_has_scratch: Vec<K>,
    chi_scratch: Vec<K>,
}

impl ShoutGammaAdapterOracleSparseTime {
    pub(crate) fn new(
        addr_cols: Vec<SparseIdxVec<K>>,
        has_cols: Vec<SparseIdxVec<K>>,
        coeffs: Vec<K>,
        eq_alpha: Vec<K>,
        eq_beta: Vec<K>,
        r_cycle: &[K],
    ) -> Self {
        debug_assert_eq!(addr_cols.len(), eq_alpha.len());
        debug_assert_eq!(addr_cols.len(), eq_beta.len());
        debug_assert_eq!(has_cols.len(), coeffs.len());
        let mut support_cols = Vec::with_capacity(addr_cols.len() + has_cols.len());
        support_cols.extend(addr_cols.iter().cloned());
        support_cols.extend(has_cols.iter().cloned());
        let support = sparse_union_support(&support_cols);
        let ell = addr_cols.len();
        let lanes = has_cols.len();
        Self {
            bit_idx: 0,
            r_cycle: r_cycle.to_vec(),
            prefix_eq: K::ONE,
            addr_cols,
            has_cols,
            coeffs,
            eq_alpha,
            eq_beta,
            support,
            pair_marks: Vec::new(),
            pair_epoch: 1,
            pair_scratch: Vec::new(),
            addr_child0: vec![K::ZERO; ell],
            addr_child1: vec![K::ZERO; ell],
            has_child0: vec![K::ZERO; lanes],
            has_child1: vec![K::ZERO; lanes],
            eq_scratch: Vec::new(),
            weighted_has_scratch: Vec::new(),
            chi_scratch: Vec::new(),
        }
    }
}

impl RoundOracle for ShoutGammaAdapterOracleSparseTime {
    fn evals_at(&mut self, points: &[K]) -> Vec<K> {
        if self.addr_cols.is_empty() {
            return vec![K::ZERO; points.len()];
        }
        if self.addr_cols[0].len() == 1 {
            let mut eq = K::ONE;
            for i in 0..self.addr_cols.len() {
                let bit = self.addr_cols[i].singleton_value();
                eq *= bit * self.eq_alpha[i] + self.eq_beta[i];
            }
            if eq == K::ZERO {
                return vec![K::ZERO; points.len()];
            }
            let mut weighted_has = K::ZERO;
            for i in 0..self.has_cols.len() {
                let has = self.has_cols[i].singleton_value();
                if has != K::ZERO {
                    weighted_has += self.coeffs[i] * has;
                }
            }
            let out = self.prefix_eq * eq * weighted_has;
            return vec![out; points.len()];
        }

        let pair_domain = self.support.len() >> 1;
        if self.pair_marks.len() < pair_domain {
            self.pair_marks.resize(pair_domain, 0);
        }
        self.pair_epoch = self.pair_epoch.wrapping_add(1);
        if self.pair_epoch == 0 {
            self.pair_marks.fill(0);
            self.pair_epoch = 1;
        }
        self.pair_scratch.clear();
        let epoch = self.pair_epoch;
        for &(idx, _v) in self.support.entries() {
            let pair = idx >> 1;
            if self.pair_marks[pair] != epoch {
                self.pair_marks[pair] = epoch;
                self.pair_scratch.push(pair);
            }
        }

        let point_len = points.len();
        if self.eq_scratch.len() < point_len {
            self.eq_scratch.resize(point_len, K::ZERO);
        }
        if self.weighted_has_scratch.len() < point_len {
            self.weighted_has_scratch.resize(point_len, K::ZERO);
        }
        if self.chi_scratch.len() < point_len {
            self.chi_scratch.resize(point_len, K::ZERO);
        }

        let mut ys = vec![K::ZERO; point_len];
        let mut addr_hints = vec![0usize; self.addr_cols.len()];
        let mut has_hints = vec![0usize; self.has_cols.len()];
        for &pair in self.pair_scratch.iter() {
            let child0 = 2 * pair;
            let child1 = child0 + 1;
            for i in 0..self.addr_cols.len() {
                self.addr_child0[i] = self.addr_cols[i].get_with_hint(child0, &mut addr_hints[i]);
                self.addr_child1[i] = self.addr_cols[i].get_with_hint(child1, &mut addr_hints[i]);
            }
            for i in 0..self.has_cols.len() {
                self.has_child0[i] = self.has_cols[i].get_with_hint(child0, &mut has_hints[i]);
                self.has_child1[i] = self.has_cols[i].get_with_hint(child1, &mut has_hints[i]);
            }

            let (chi0, chi1) = chi_cycle_children(&self.r_cycle, self.bit_idx, self.prefix_eq, pair);
            for (pi, &x) in points.iter().enumerate() {
                self.chi_scratch[pi] = interp(chi0, chi1, x);
                self.eq_scratch[pi] = K::ONE;
                self.weighted_has_scratch[pi] = K::ZERO;
            }

            for i in 0..self.addr_cols.len() {
                let b0 = self.addr_child0[i];
                let b1 = self.addr_child1[i];
                let alpha = self.eq_alpha[i];
                let beta = self.eq_beta[i];

                if b0 == K::ZERO && b1 == K::ZERO {
                    if beta == K::ZERO {
                        for pi in 0..point_len {
                            self.eq_scratch[pi] = K::ZERO;
                        }
                        break;
                    }
                    for pi in 0..point_len {
                        if self.chi_scratch[pi] != K::ZERO && self.eq_scratch[pi] != K::ZERO {
                            self.eq_scratch[pi] *= beta;
                        }
                    }
                    continue;
                }

                if b0 == b1 {
                    let factor = b0 * alpha + beta;
                    if factor == K::ZERO {
                        for pi in 0..point_len {
                            self.eq_scratch[pi] = K::ZERO;
                        }
                        break;
                    }
                    for pi in 0..point_len {
                        if self.chi_scratch[pi] != K::ZERO && self.eq_scratch[pi] != K::ZERO {
                            self.eq_scratch[pi] *= factor;
                        }
                    }
                    continue;
                }

                for (pi, &x) in points.iter().enumerate() {
                    if self.chi_scratch[pi] == K::ZERO || self.eq_scratch[pi] == K::ZERO {
                        continue;
                    }
                    let bit = interp(b0, b1, x);
                    self.eq_scratch[pi] *= bit * alpha + beta;
                }
            }

            let mut any_active = false;
            for pi in 0..point_len {
                if self.chi_scratch[pi] != K::ZERO && self.eq_scratch[pi] != K::ZERO {
                    any_active = true;
                    break;
                }
            }
            if !any_active {
                continue;
            }

            for i in 0..self.has_cols.len() {
                let h0 = self.has_child0[i];
                let h1 = self.has_child1[i];
                let coeff = self.coeffs[i];
                if coeff == K::ZERO || (h0 == K::ZERO && h1 == K::ZERO) {
                    continue;
                }

                if h0 == h1 {
                    let term = coeff * h0;
                    if term == K::ZERO {
                        continue;
                    }
                    for pi in 0..point_len {
                        if self.chi_scratch[pi] != K::ZERO && self.eq_scratch[pi] != K::ZERO {
                            self.weighted_has_scratch[pi] += term;
                        }
                    }
                    continue;
                }

                for (pi, &x) in points.iter().enumerate() {
                    if self.chi_scratch[pi] == K::ZERO || self.eq_scratch[pi] == K::ZERO {
                        continue;
                    }
                    self.weighted_has_scratch[pi] += coeff * interp(h0, h1, x);
                }
            }

            for pi in 0..point_len {
                let chi_x = self.chi_scratch[pi];
                if chi_x == K::ZERO {
                    continue;
                }
                let eq = self.eq_scratch[pi];
                if eq == K::ZERO {
                    continue;
                }
                let weighted_has = self.weighted_has_scratch[pi];
                if weighted_has == K::ZERO {
                    continue;
                }
                ys[pi] += chi_x * eq * weighted_has;
            }
        }
        ys
    }

    fn num_rounds(&self) -> usize {
        self.r_cycle.len().saturating_sub(self.bit_idx)
    }

    fn degree_bound(&self) -> usize {
        2 + self.addr_cols.len()
    }

    fn fold(&mut self, r: K) {
        if self.num_rounds() == 0 {
            return;
        }
        self.prefix_eq *= eq_single_k(r, self.r_cycle[self.bit_idx]);
        for col in self.addr_cols.iter_mut() {
            col.fold_round_in_place(r);
        }
        for col in self.has_cols.iter_mut() {
            col.fold_round_in_place(r);
        }
        self.support.fold_support_round_in_place();
        self.bit_idx += 1;
    }
}

/// Sparse-formula oracle over the full hypercube sum (no anchor chi weighting).
///
/// This is used for compressed-domain multiset/link claims where we need true
/// \sum_j f(j), not f(r_anchor).
pub(crate) struct FormulaOracleSparseSum {
    cols: Vec<SparseIdxVec<K>>,
    degree_bound: usize,
    eval_fn: Box<dyn Fn(&[K]) -> K + Send + Sync>,
}

impl FormulaOracleSparseSum {
    pub(crate) fn new(
        cols: Vec<SparseIdxVec<K>>,
        degree_bound: usize,
        eval_fn: Box<dyn Fn(&[K]) -> K + Send + Sync>,
    ) -> Self {
        Self {
            cols,
            degree_bound,
            eval_fn,
        }
    }
}

impl RoundOracle for FormulaOracleSparseSum {
    fn evals_at(&mut self, points: &[K]) -> Vec<K> {
        if self.cols.is_empty() {
            return vec![K::ZERO; points.len()];
        }

        let mut pairs = Vec::new();
        for col in self.cols.iter() {
            pairs.extend(gather_pairs_from_sparse(col.entries()));
        }
        pairs.sort_unstable();
        pairs.dedup();

        let mut ys = vec![K::ZERO; points.len()];
        let mut vals = vec![K::ZERO; self.cols.len()];
        for &pair in pairs.iter() {
            let child0 = 2 * pair;
            let child1 = child0 + 1;
            for (i, &x) in points.iter().enumerate() {
                for (j, col) in self.cols.iter().enumerate() {
                    vals[j] = interp(col.get(child0), col.get(child1), x);
                }
                let f_x = (self.eval_fn)(&vals);
                if f_x == K::ZERO {
                    continue;
                }
                ys[i] += f_x;
            }
        }
        ys
    }

    fn num_rounds(&self) -> usize {
        self.cols
            .first()
            .map(|c| c.len().ilog2() as usize)
            .unwrap_or(0)
    }

    fn degree_bound(&self) -> usize {
        self.degree_bound
    }

    fn fold(&mut self, r: K) {
        if self.num_rounds() == 0 {
            return;
        }
        for col in self.cols.iter_mut() {
            col.fold_round_in_place(r);
        }
    }
}

#[inline]
pub(crate) fn unpack_interleaved_halves_lsb(addr_bits: &[K]) -> Result<(K, K), PiCcsError> {
    if !addr_bits.len().is_multiple_of(2) {
        return Err(PiCcsError::InvalidInput(format!(
            "shout linkage expects even ell_addr, got {}",
            addr_bits.len()
        )));
    }
    let half_len = addr_bits.len() / 2;
    let two = K::from(F::from_u64(2));
    let mut pow = K::ONE;
    let mut lhs = K::ZERO;
    let mut rhs = K::ZERO;
    for k in 0..half_len {
        lhs += pow * addr_bits[2 * k];
        rhs += pow * addr_bits[2 * k + 1];
        pow *= two;
    }
    Ok((lhs, rhs))
}

pub(crate) fn extract_trace_cpu_link_openings(
    _m: usize,
    _core_t: usize,
    _y_prefix_cols: usize,
    step: &StepInstanceBundle<Cmt, F, K>,
    _ccs_out0: &CeClaim<Cmt, F, K>,
    step_time_openings: &[crate::shard_proof_types::TimePointOpening],
    r_time: &[K],
) -> Result<Option<TraceCpuLinkOpenings>, PiCcsError> {
    if step.mem_insts.is_empty() && step.lut_insts.is_empty() {
        return Ok(None);
    }

    // RV32 trace linkage: the prover appends time-combined openings for selected CPU trace columns
    // to the CCS ME output at r_time. We use those to bind Twist instances (PROG/REG/RAM) to the
    // same trace, without embedding a shared CPU bus tail.
    let trace = Rv32TraceLayout::new();
    let trace_cols_to_open: Vec<usize> = vec![
        trace.active,
        trace.cycle,
        trace.pc_before,
        trace.instr_word,
        trace.rs1_addr,
        trace.rs1_val,
        trace.rs2_addr,
        trace.rs2_val,
        trace.rd_addr,
        trace.rd_val,
        trace.ram_addr,
        trace.ram_rv,
        trace.ram_wv,
        trace.shout_has_lookup,
        trace.shout_val,
        trace.shout_link_lhs,
        trace.shout_link_rhs,
        trace.shout_add_sub_key,
    ];

    let t_len = step.time_columns.t;
    if t_len == 0 {
        return Err(PiCcsError::InvalidInput("trace linkage requires time_t > 0".into()));
    }
    for (i, inst) in step.mem_insts.iter().enumerate() {
        if inst.steps != t_len {
            return Err(PiCcsError::InvalidInput(format!(
                "trace linkage requires stable steps across mem instances (mem_idx={i} has steps={}, expected {t_len})",
                inst.steps
            )));
        }
    }
    let trace_open_map =
        require_time_openings_for_point(step_time_openings, r_time, &trace_cols_to_open, "trace linkage")?;

    Ok(Some(TraceCpuLinkOpenings {
        shout_has_lookup: named_opening(&trace_open_map, trace.shout_has_lookup, "trace linkage")?,
        shout_val: named_opening(&trace_open_map, trace.shout_val, "trace linkage")?,
        shout_link_lhs: named_opening(&trace_open_map, trace.shout_link_lhs, "trace linkage")?,
        shout_link_rhs: named_opening(&trace_open_map, trace.shout_link_rhs, "trace linkage")?,
        shout_add_sub_key: named_opening(&trace_open_map, trace.shout_add_sub_key, "trace linkage")?,
    }))
}

pub(crate) fn expected_trace_shout_table_id_from_openings(
    step: &StepInstanceBundle<Cmt, F, K>,
    _cpu_bus: &BusLayout,
    mem_proof: &MemSidecarProof<Cmt, F, K>,
    step_time_openings: &[crate::shard_proof_types::TimePointOpening],
    r_time: &[K],
) -> Result<Option<K>, PiCcsError> {
    if !decode_stage_required_for_step_instance(step) {
        return Ok(None);
    }

    if mem_proof.wp_me_claims.len() != 1 {
        return Err(PiCcsError::ProtocolError(
            "decode-linked Shout table_id check requires one WP ME claim".into(),
        ));
    }
    let wp_me = &mem_proof.wp_me_claims[0];
    if wp_me.r.as_slice() != r_time {
        return Err(PiCcsError::ProtocolError(
            "decode-linked Shout table_id check: WP ME r mismatch".into(),
        ));
    }
    if wp_me.c != step.mcs_inst.c {
        return Err(PiCcsError::ProtocolError(
            "decode-linked Shout table_id check: WP ME commitment mismatch".into(),
        ));
    }
    if wp_me.m_in != step.mcs_inst.m_in {
        return Err(PiCcsError::ProtocolError(
            "decode-linked Shout table_id check: WP ME m_in mismatch".into(),
        ));
    }

    let trace_layout = Rv32TraceLayout::new();
    let wp_cols = rv32_trace_wp_opening_columns(&trace_layout);
    let (wp_entry, wp_open_map) = require_time_openings_covering_point(
        step_time_openings,
        r_time,
        &wp_cols,
        "decode-linked Shout table_id check/WP",
    )?;
    if wp_entry.source != crate::shard_proof_types::TimeOpeningSource::CommittedOpening {
        return Err(PiCcsError::ProtocolError(format!(
            "decode-linked Shout table_id check/WP requires CommittedOpening source (got {:?})",
            wp_entry.source
        )));
    }
    let shout_table_id = named_opening(
        &wp_open_map,
        trace_layout.shout_table_id,
        "decode-linked Shout table_id check",
    )?;
    Ok(Some(shout_table_id))
}

pub(crate) fn prove_twist_addr_pre_time(
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    step: &StepWitnessBundle<Cmt, F, K>,
    cpu_bus: &BusLayout,
    ell_n: usize,
    r_cycle: &[K],
) -> Result<Vec<TwistAddrPreProverData>, PiCcsError> {
    if step.mem_instances.is_empty() {
        return Ok(Vec::new());
    }
    let mut out = Vec::with_capacity(step.mem_instances.len());

    let use_time_mem_cols =
        step.time_columns.t == cpu_bus.chunk_size && step.time_columns.mem_cols.len() == cpu_bus.bus_cols;
    let expected_m = step
        .mcs
        .0
        .m_in
        .checked_add(step.mcs.1.w.len())
        .ok_or_else(|| PiCcsError::InvalidInput("shared_cpu_bus witness width overflow".into()))?;
    let cpu_z_k = if use_time_mem_cols {
        Vec::new()
    } else {
        crate::memory_sidecar::cpu_bus::decode_cpu_z_to_k(params, &step.mcs.1.Z, expected_m)?
    };
    if cpu_bus.shout_cols.len() != step.lut_instances.len() || cpu_bus.twist_cols.len() != step.mem_instances.len() {
        return Err(PiCcsError::InvalidInput(
            "shared_cpu_bus layout mismatch for step (instance counts)".into(),
        ));
    }

    for (idx, (mem_inst, _mem_wit)) in step.mem_instances.iter().enumerate() {
        neo_memory::addr::validate_twist_bit_addressing(mem_inst)?;
        let pow2_cycle = 1usize << ell_n;
        if mem_inst.steps > pow2_cycle {
            return Err(PiCcsError::InvalidInput(format!(
                "Twist(Route A): steps={} exceeds 2^ell_cycle={pow2_cycle}",
                mem_inst.steps
            )));
        }

        let bus = cpu_bus.clone();
        let z = cpu_z_k.clone();

        let ell_addr = mem_inst.d * mem_inst.ell;
        let expected_lanes = mem_inst.lanes.max(1);
        let twist_inst_cols = bus.twist_cols.get(idx).ok_or_else(|| {
            PiCcsError::InvalidInput(format!(
                "shared_cpu_bus layout mismatch: missing twist_cols for mem_idx={idx}"
            ))
        })?;
        if twist_inst_cols.lanes.len() != expected_lanes {
            return Err(PiCcsError::InvalidInput(format!(
                "shared_cpu_bus layout mismatch at mem_idx={idx}: expected lanes={expected_lanes}, got {}",
                twist_inst_cols.lanes.len()
            )));
        }

        let mut lanes: Vec<TwistLaneSparseCols> = Vec::with_capacity(twist_inst_cols.lanes.len());
        for (lane_idx, twist_cols) in twist_inst_cols.lanes.iter().enumerate() {
            if twist_cols.ra_bits.end - twist_cols.ra_bits.start != ell_addr
                || twist_cols.wa_bits.end - twist_cols.wa_bits.start != ell_addr
            {
                return Err(PiCcsError::InvalidInput(format!(
                    "shared_cpu_bus layout mismatch at mem_idx={idx}, lane={lane_idx}: expected ell_addr={ell_addr}"
                )));
            }

            let mut ra_bits = Vec::with_capacity(ell_addr);
            for col_id in twist_cols.ra_bits.clone() {
                ra_bits.push(if use_time_mem_cols {
                    crate::memory_sidecar::cpu_bus::build_time_sparse_from_mem_cols(
                        &step.time_columns.mem_cols,
                        &bus,
                        col_id,
                        mem_inst.steps,
                        pow2_cycle,
                    )?
                } else {
                    crate::memory_sidecar::cpu_bus::build_time_sparse_from_bus_col(
                        &z,
                        &bus,
                        col_id,
                        mem_inst.steps,
                        pow2_cycle,
                    )?
                });
            }

            let mut wa_bits = Vec::with_capacity(ell_addr);
            for col_id in twist_cols.wa_bits.clone() {
                wa_bits.push(if use_time_mem_cols {
                    crate::memory_sidecar::cpu_bus::build_time_sparse_from_mem_cols(
                        &step.time_columns.mem_cols,
                        &bus,
                        col_id,
                        mem_inst.steps,
                        pow2_cycle,
                    )?
                } else {
                    crate::memory_sidecar::cpu_bus::build_time_sparse_from_bus_col(
                        &z,
                        &bus,
                        col_id,
                        mem_inst.steps,
                        pow2_cycle,
                    )?
                });
            }

            let has_read = if use_time_mem_cols {
                crate::memory_sidecar::cpu_bus::build_time_sparse_from_mem_cols(
                    &step.time_columns.mem_cols,
                    &bus,
                    twist_cols.has_read,
                    mem_inst.steps,
                    pow2_cycle,
                )?
            } else {
                crate::memory_sidecar::cpu_bus::build_time_sparse_from_bus_col(
                    &z,
                    &bus,
                    twist_cols.has_read,
                    mem_inst.steps,
                    pow2_cycle,
                )?
            };
            let has_write = if use_time_mem_cols {
                crate::memory_sidecar::cpu_bus::build_time_sparse_from_mem_cols(
                    &step.time_columns.mem_cols,
                    &bus,
                    twist_cols.has_write,
                    mem_inst.steps,
                    pow2_cycle,
                )?
            } else {
                crate::memory_sidecar::cpu_bus::build_time_sparse_from_bus_col(
                    &z,
                    &bus,
                    twist_cols.has_write,
                    mem_inst.steps,
                    pow2_cycle,
                )?
            };
            let wv = if use_time_mem_cols {
                crate::memory_sidecar::cpu_bus::build_time_sparse_from_mem_cols(
                    &step.time_columns.mem_cols,
                    &bus,
                    twist_cols.wv,
                    mem_inst.steps,
                    pow2_cycle,
                )?
            } else {
                crate::memory_sidecar::cpu_bus::build_time_sparse_from_bus_col(
                    &z,
                    &bus,
                    twist_cols.wv,
                    mem_inst.steps,
                    pow2_cycle,
                )?
            };
            let rv = if use_time_mem_cols {
                crate::memory_sidecar::cpu_bus::build_time_sparse_from_mem_cols(
                    &step.time_columns.mem_cols,
                    &bus,
                    twist_cols.rv,
                    mem_inst.steps,
                    pow2_cycle,
                )?
            } else {
                crate::memory_sidecar::cpu_bus::build_time_sparse_from_bus_col(
                    &z,
                    &bus,
                    twist_cols.rv,
                    mem_inst.steps,
                    pow2_cycle,
                )?
            };
            let inc_at_write_addr = if use_time_mem_cols {
                crate::memory_sidecar::cpu_bus::build_time_sparse_from_mem_cols(
                    &step.time_columns.mem_cols,
                    &bus,
                    twist_cols.inc,
                    mem_inst.steps,
                    pow2_cycle,
                )?
            } else {
                crate::memory_sidecar::cpu_bus::build_time_sparse_from_bus_col(
                    &z,
                    &bus,
                    twist_cols.inc,
                    mem_inst.steps,
                    pow2_cycle,
                )?
            };

            lanes.push(TwistLaneSparseCols {
                ra_bits,
                wa_bits,
                has_read,
                has_write,
                wv,
                rv,
                inc_at_write_addr,
            });
        }

        let decoded = TwistDecodedColsSparse { lanes };

        let init_sparse: Vec<(usize, K)> = match &mem_inst.init {
            MemInit::Zero => Vec::new(),
            MemInit::Sparse(pairs) => pairs
                .iter()
                .map(|(addr, val)| {
                    let addr_usize = usize::try_from(*addr).map_err(|_| {
                        PiCcsError::InvalidInput(format!("Twist: init address doesn't fit usize: addr={addr}"))
                    })?;
                    if addr_usize >= mem_inst.k {
                        return Err(PiCcsError::InvalidInput(format!(
                            "Twist: init address out of range: addr={addr} >= k={}",
                            mem_inst.k
                        )));
                    }
                    Ok((addr_usize, (*val).into()))
                })
                .collect::<Result<_, _>>()?,
        };

        let mut read_addr_oracle =
            TwistReadCheckAddrOracleSparseTimeMultiLane::new(init_sparse.clone(), r_cycle, &decoded.lanes);
        let mut write_addr_oracle =
            TwistWriteCheckAddrOracleSparseTimeMultiLane::new(init_sparse, r_cycle, &decoded.lanes);

        let labels: [&[u8]; 2] = [b"twist/read_addr_pre".as_slice(), b"twist/write_addr_pre".as_slice()];
        let claimed_sums = vec![K::ZERO, K::ZERO];
        tr.append_message(b"twist/addr_pre_time/claim_idx", &(idx as u64).to_le_bytes());
        bind_batched_claim_sums(tr, b"twist/addr_pre_time/claimed_sums", &claimed_sums, &labels);

        let mut claims = [
            BatchedClaim {
                oracle: &mut read_addr_oracle,
                claimed_sum: K::ZERO,
                label: labels[0],
            },
            BatchedClaim {
                oracle: &mut write_addr_oracle,
                claimed_sum: K::ZERO,
                label: labels[1],
            },
        ];

        let (r_addr, mut per_claim_results) =
            run_batched_sumcheck_prover_ds(tr, b"twist/addr_pre_time", idx, &mut claims)?;
        if per_claim_results.len() != 2 {
            return Err(PiCcsError::ProtocolError(format!(
                "twist addr-pre per-claim results len()={}, expected 2",
                per_claim_results.len()
            )));
        }
        let read_rounds = std::mem::take(&mut per_claim_results[0].round_polys);
        let write_rounds = std::mem::take(&mut per_claim_results[1].round_polys);

        out.push(TwistAddrPreProverData {
            addr_pre: BatchedAddrProof {
                claimed_sums,
                round_polys: vec![read_rounds, write_rounds],
                r_addr: r_addr.clone(),
            },
            decoded,
            read_check_claim_sum: per_claim_results[0].final_value,
            write_check_claim_sum: per_claim_results[1].final_value,
        });
    }

    Ok(out)
}

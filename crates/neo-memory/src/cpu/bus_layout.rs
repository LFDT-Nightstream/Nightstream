use core::ops::Range;

/// Canonical layout for the shared CPU bus tail inside the CPU witness `z`.
///
/// This is a pure index-calculation helper (no commitments, no decoding). It is the
/// single source of truth for:
/// - bus placement in `z` (tail region)
/// - time index mapping `t = m_in + j`
/// - per-instance column ordering and ranges
///
/// ## Column ordering
/// 1) All Shout instances, in the order provided to the builder.
/// 2) All Twist instances, in the order provided to the builder.
///
/// Within each Shout instance:
/// - `addr_bits[0..ell_addr)`
/// - `has_lookup`
/// - `val`
///
/// Within each Twist instance:
/// - `ra_bits[0..ell_addr)`
/// - `wa_bits[0..ell_addr)`
/// - `has_read`
/// - `has_write`
/// - `wv`
/// - `rv`
/// - `inc`
#[derive(Clone, Debug)]
pub struct BusLayout {
    pub m: usize,
    pub m_in: usize,
    pub chunk_size: usize,
    pub bus_cols: usize,
    pub bus_base: usize,
    pub shout_cols: Vec<ShoutCols>,
    pub twist_cols: Vec<TwistCols>,
}

#[derive(Clone, Debug)]
pub struct ShoutCols {
    pub addr_bits: Range<usize>,
    pub has_lookup: usize,
    pub val: usize,
}

#[derive(Clone, Debug)]
pub struct TwistCols {
    pub ra_bits: Range<usize>,
    pub wa_bits: Range<usize>,
    pub has_read: usize,
    pub has_write: usize,
    pub wv: usize,
    pub rv: usize,
    pub inc: usize,
}

impl BusLayout {
    #[inline]
    pub fn bus_region_len(&self) -> usize {
        self.bus_cols
            .checked_mul(self.chunk_size)
            .expect("BusLayout: bus_cols*chunk_size overflow")
    }

    /// Witness index in `z` for the bus cell at (col_id, j).
    ///
    /// Formula: `bus_base + col_id * chunk_size + j`.
    #[inline]
    pub fn bus_cell(&self, col_id: usize, j: usize) -> usize {
        debug_assert!(col_id < self.bus_cols, "bus col_id out of range");
        debug_assert!(j < self.chunk_size, "bus j out of range");
        self.bus_base + col_id * self.chunk_size + j
    }

    /// Route-A time index for the `j`-th step within the fold step.
    ///
    /// Formula: `t = m_in + j`.
    #[inline]
    pub fn time_index(&self, j: usize) -> usize {
        self.m_in + j
    }

    /// Index into an ME-opening `y_scalars` slice, given `bus_y_base` and a bus `col_id`.
    #[inline]
    pub fn y_scalar_index(&self, bus_y_base: usize, col_id: usize) -> usize {
        debug_assert!(col_id < self.bus_cols, "bus col_id out of range");
        bus_y_base + col_id
    }
}

pub fn build_bus_layout_for_instances(
    m: usize,
    m_in: usize,
    chunk_size: usize,
    shout_ell_addrs: impl IntoIterator<Item = usize>,
    twist_ell_addrs: impl IntoIterator<Item = usize>,
) -> Result<BusLayout, String> {
    if chunk_size == 0 {
        return Err("BusLayout: chunk_size must be >= 1".into());
    }

    let mut col = 0usize;

    let mut shout_cols = Vec::<ShoutCols>::new();
    for ell_addr in shout_ell_addrs {
        let addr_bits = col..(col + ell_addr);
        col = col
            .checked_add(ell_addr)
            .ok_or_else(|| "BusLayout: column overflow (shout addr_bits)".to_string())?;
        let has_lookup = col;
        col = col
            .checked_add(1)
            .ok_or_else(|| "BusLayout: column overflow (shout has_lookup)".to_string())?;
        let val = col;
        col = col
            .checked_add(1)
            .ok_or_else(|| "BusLayout: column overflow (shout val)".to_string())?;
        shout_cols.push(ShoutCols {
            addr_bits,
            has_lookup,
            val,
        });
    }

    let mut twist_cols = Vec::<TwistCols>::new();
    for ell_addr in twist_ell_addrs {
        let ra_bits = col..(col + ell_addr);
        col = col
            .checked_add(ell_addr)
            .ok_or_else(|| "BusLayout: column overflow (twist ra_bits)".to_string())?;
        let wa_bits = col..(col + ell_addr);
        col = col
            .checked_add(ell_addr)
            .ok_or_else(|| "BusLayout: column overflow (twist wa_bits)".to_string())?;
        let has_read = col;
        col = col
            .checked_add(1)
            .ok_or_else(|| "BusLayout: column overflow (twist has_read)".to_string())?;
        let has_write = col;
        col = col
            .checked_add(1)
            .ok_or_else(|| "BusLayout: column overflow (twist has_write)".to_string())?;
        let wv = col;
        col = col
            .checked_add(1)
            .ok_or_else(|| "BusLayout: column overflow (twist wv)".to_string())?;
        let rv = col;
        col = col
            .checked_add(1)
            .ok_or_else(|| "BusLayout: column overflow (twist rv)".to_string())?;
        let inc = col;
        col = col
            .checked_add(1)
            .ok_or_else(|| "BusLayout: column overflow (twist inc)".to_string())?;
        twist_cols.push(TwistCols {
            ra_bits,
            wa_bits,
            has_read,
            has_write,
            wv,
            rv,
            inc,
        });
    }

    let bus_cols = col;
    let bus_region_len = bus_cols
        .checked_mul(chunk_size)
        .ok_or_else(|| "BusLayout: bus_cols*chunk_size overflow".to_string())?;
    if bus_region_len > m {
        return Err(format!(
            "BusLayout: bus region too large: bus_cols({bus_cols}) * chunk_size({chunk_size}) = {bus_region_len} > m({m})"
        ));
    }
    let bus_base = m - bus_region_len;
    if bus_base < m_in {
        return Err(format!(
            "BusLayout: bus_base({bus_base}) overlaps public inputs m_in({m_in})"
        ));
    }

    Ok(BusLayout {
        m,
        m_in,
        chunk_size,
        bus_cols,
        bus_base,
        shout_cols,
        twist_cols,
    })
}

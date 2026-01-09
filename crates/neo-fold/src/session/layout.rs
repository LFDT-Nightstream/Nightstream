use core::marker::PhantomData;
use neo_memory::cpu::{ShoutCpuBinding, TwistCpuBinding};
use neo_math::F;
use neo_vm_trace::{StepTrace, TwistOpKind};
use p3_field::PrimeCharacteristicRing;

/// Column index type used by witness layouts.
pub type Scalar = usize;

/// Marker wrapper to declare a `WitnessLayoutField` as public input (part of `m_in`).
#[derive(Clone, Copy, Debug, Default)]
pub struct Public<T>(PhantomData<T>);

/// Trait implemented by layout "field spec" types (e.g. `Scalar`, `Lane<N>`, `Public<Scalar>`, ...).
///
/// The associated `Field` type is what actually gets stored inside the generated layout struct.
pub trait WitnessLayoutField {
    type Field;

    /// Number of witness columns consumed by this field.
    const LEN: usize;

    /// Number of public columns contributed to `m_in` by this field.
    const PUBLIC_LEN: usize = 0;

    fn alloc(alloc: &mut WitnessLayoutAllocator) -> Self::Field;
}

/// A generated witness layout (from `witness_layout!`).
pub trait WitnessLayout: Sized {
    const M_IN: usize;
    const USED_COLS: usize;

    fn new_layout() -> Self;

    /// Allocate a zero-initialized witness prefix vector of length `USED_COLS`.
    ///
    /// This is a convenience helper for `NeoCircuit::build_witness_prefix` implementations.
    #[inline]
    fn zero_witness_prefix() -> Vec<F> {
        vec![F::ZERO; Self::USED_COLS]
    }
}

/// Marker trait for fields that can live in the `m_in` prefix.
pub trait WitnessLayoutPublic: WitnessLayoutField {
    fn alloc_public(alloc: &mut WitnessLayoutAllocator) -> Self::Field;
}

impl WitnessLayoutField for Scalar {
    type Field = Scalar;
    const LEN: usize = 1;

    fn alloc(alloc: &mut WitnessLayoutAllocator) -> Self::Field {
        alloc.scalar()
    }
}

impl WitnessLayoutPublic for Scalar {
    fn alloc_public(alloc: &mut WitnessLayoutAllocator) -> Self::Field {
        alloc.public_scalar()
    }
}

impl<const N: usize> WitnessLayoutField for Lane<N> {
    type Field = Lane<N>;
    const LEN: usize = N;

    fn alloc(alloc: &mut WitnessLayoutAllocator) -> Self::Field {
        alloc.lane::<N>()
    }
}

impl<const N: usize> WitnessLayoutPublic for Lane<N> {
    fn alloc_public(alloc: &mut WitnessLayoutAllocator) -> Self::Field {
        alloc.public_lane::<N>()
    }
}

impl<const N: usize> WitnessLayoutField for ShoutPort<N> {
    type Field = ShoutPort<N>;
    const LEN: usize = 3 * N;

    fn alloc(alloc: &mut WitnessLayoutAllocator) -> Self::Field {
        alloc.shout_port::<N>()
    }
}

impl<const N: usize> WitnessLayoutField for TwistPort<N> {
    type Field = TwistPort<N>;
    const LEN: usize = 6 * N;

    fn alloc(alloc: &mut WitnessLayoutAllocator) -> Self::Field {
        alloc.twist_port::<N>()
    }
}

impl<const N: usize> WitnessLayoutField for TwistPortWithInc<N> {
    type Field = TwistPortWithInc<N>;
    const LEN: usize = 7 * N;

    fn alloc(alloc: &mut WitnessLayoutAllocator) -> Self::Field {
        alloc.twist_port_with_inc::<N>()
    }
}

impl<T> WitnessLayoutField for Public<T>
where
    T: WitnessLayoutPublic,
{
    type Field = <T as WitnessLayoutField>::Field;
    const LEN: usize = <T as WitnessLayoutField>::LEN;
    const PUBLIC_LEN: usize = <T as WitnessLayoutField>::LEN;

    fn alloc(alloc: &mut WitnessLayoutAllocator) -> Self::Field {
        <T as WitnessLayoutPublic>::alloc_public(alloc)
    }
}

/// Allocate contiguous witness columns for a "chunked" CPU witness `z`.
///
/// The convention (shared across the Neo codebase) is that each logical column is stored as a
/// contiguous lane of length `chunk_size`, so that the per-lane index is `col_base + j`.
#[derive(Clone, Debug, Default)]
pub struct WitnessLayoutAllocator {
    next: usize,
    m_in: usize,
}

impl WitnessLayoutAllocator {
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of public inputs `m_in` allocated so far.
    #[inline]
    pub fn m_in(&self) -> usize {
        self.m_in
    }

    /// Total number of witness columns allocated so far.
    #[inline]
    pub fn used_cols(&self) -> usize {
        self.next
    }

    #[inline]
    fn require_public_prefix(&self) {
        assert_eq!(
            self.next, self.m_in,
            "public columns must be allocated before private columns (m_in prefix)"
        );
    }

    pub fn scalar(&mut self) -> Scalar {
        let col = self.next;
        self.next = self.next.checked_add(1).expect("witness layout overflow");
        col
    }

    pub fn public_scalar(&mut self) -> Scalar {
        self.require_public_prefix();
        let col = self.scalar();
        self.m_in = self.m_in.checked_add(1).expect("witness layout overflow");
        col
    }

    pub fn lane<const N: usize>(&mut self) -> Lane<N> {
        let base = self.next;
        self.next = self.next.checked_add(N).expect("witness layout overflow");
        Lane::new(base)
    }

    pub fn public_lane<const N: usize>(&mut self) -> Lane<N> {
        self.require_public_prefix();
        let lane = self.lane::<N>();
        self.m_in = self.m_in.checked_add(N).expect("witness layout overflow");
        lane
    }

    pub fn shout_port<const N: usize>(&mut self) -> ShoutPort<N> {
        ShoutPort {
            has_lookup: self.lane::<N>(),
            addr: self.lane::<N>(),
            val: self.lane::<N>(),
        }
    }

    pub fn twist_port<const N: usize>(&mut self) -> TwistPort<N> {
        TwistPort {
            has_read: self.lane::<N>(),
            has_write: self.lane::<N>(),
            read_addr: self.lane::<N>(),
            write_addr: self.lane::<N>(),
            rv: self.lane::<N>(),
            wv: self.lane::<N>(),
        }
    }

    pub fn twist_port_with_inc<const N: usize>(&mut self) -> TwistPortWithInc<N> {
        TwistPortWithInc {
            has_read: self.lane::<N>(),
            has_write: self.lane::<N>(),
            read_addr: self.lane::<N>(),
            write_addr: self.lane::<N>(),
            rv: self.lane::<N>(),
            wv: self.lane::<N>(),
            inc: self.lane::<N>(),
        }
    }
}

/// A contiguous lane of `N` column indices (one per step inside a folding chunk).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Lane<const N: usize> {
    base: usize,
}

impl<const N: usize> Lane<N> {
    pub fn new(base: usize) -> Self {
        Self { base }
    }

    #[inline]
    pub fn base(&self) -> usize {
        self.base
    }

    #[inline]
    pub fn at(&self, j: usize) -> usize {
        debug_assert!(j < N, "Lane::at out of bounds: j={j} >= N={N}");
        self.base + j
    }

    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = usize> + '_ {
        (0..N).map(move |j| self.base + j)
    }

    /// Convenience setter: `z[self.at(j)] = value`.
    #[inline]
    pub fn set(&self, z: &mut [F], j: usize, value: F) {
        z[self.at(j)] = value;
    }

    /// Fill the entire lane from an iterator of exactly `N` values.
    ///
    /// Returns an error if the iterator yields too few or too many values.
    pub fn set_from_iter(&self, z: &mut [F], values: impl IntoIterator<Item = F>) -> Result<(), String> {
        let mut it = values.into_iter();
        for j in 0..N {
            let v = it
                .next()
                .ok_or_else(|| format!("Lane::set_from_iter: too few values (expected {N})"))?;
            z[self.at(j)] = v;
        }
        if it.next().is_some() {
            return Err(format!("Lane::set_from_iter: too many values (expected {N})"));
        }
        Ok(())
    }
}

/// Typed bundle of CPU binding lanes for a single Shout instance.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ShoutPort<const N: usize> {
    pub has_lookup: Lane<N>,
    pub addr: Lane<N>,
    pub val: Lane<N>,
}

impl<const N: usize> ShoutPort<N> {
    pub fn cpu_binding(&self) -> ShoutCpuBinding {
        ShoutCpuBinding {
            has_lookup: self.has_lookup.base(),
            addr: self.addr.base(),
            val: self.val.base(),
        }
    }

    /// Fill CPU binding columns from `StepTrace` events (common case).
    ///
    /// If no event is present for a lane, fills selector/address/value with zeros.
    pub fn fill_from_trace(&self, chunk: &[StepTrace<u64, u64>], shout_id: u32, z: &mut [F]) -> Result<(), String> {
        if chunk.len() != N {
            return Err(format!(
                "ShoutPort::fill_from_trace: chunk len {} != expected {}",
                chunk.len(),
                N
            ));
        }
        for (j, step) in chunk.iter().enumerate() {
            let mut found: Option<(u64, u64)> = None;
            for ev in &step.shout_events {
                if ev.shout_id.0 != shout_id {
                    continue;
                }
                if found.replace((ev.key, ev.value)).is_some() {
                    return Err(format!(
                        "multiple shout events for shout_id={shout_id} in one step (j={j})"
                    ));
                }
            }
            if let Some((key, val)) = found {
                z[self.has_lookup.at(j)] = F::ONE;
                z[self.addr.at(j)] = F::from_u64(key);
                z[self.val.at(j)] = F::from_u64(val);
            } else {
                z[self.has_lookup.at(j)] = F::ZERO;
                z[self.addr.at(j)] = F::ZERO;
                z[self.val.at(j)] = F::ZERO;
            }
        }
        Ok(())
    }

    /// Fill multiple Shout lookup lanes from a single `shout_id` trace stream.
    ///
    /// Each lane supports at most one lookup per VM step. If the trace contains more lookups than
    /// lanes can support, this returns an error.
    pub fn fill_lanes_from_trace(
        lanes: &[ShoutPort<N>],
        chunk: &[StepTrace<u64, u64>],
        shout_id: u32,
        z: &mut [F],
    ) -> Result<(), String> {
        if lanes.is_empty() {
            return Err("ShoutPort::fill_lanes_from_trace: lanes must be non-empty".into());
        }
        if chunk.len() != N {
            return Err(format!(
                "ShoutPort::fill_lanes_from_trace: chunk len {} != expected {}",
                chunk.len(),
                N
            ));
        }

        let mut lookups: Vec<Option<(u64, u64)>> = vec![None; lanes.len()];

        for (j, step) in chunk.iter().enumerate() {
            lookups.fill(None);
            let mut used = 0usize;

            for ev in &step.shout_events {
                if ev.shout_id.0 != shout_id {
                    continue;
                }
                if used >= lookups.len() {
                    return Err(format!(
                        "too many shout events for shout_id={shout_id} in one step (j={j}): lanes={}",
                        lanes.len()
                    ));
                }
                lookups[used] = Some((ev.key, ev.value));
                used += 1;
            }

            for (lane_idx, port) in lanes.iter().enumerate() {
                if let Some((key, val)) = lookups[lane_idx] {
                    z[port.has_lookup.at(j)] = F::ONE;
                    z[port.addr.at(j)] = F::from_u64(key);
                    z[port.val.at(j)] = F::from_u64(val);
                } else {
                    z[port.has_lookup.at(j)] = F::ZERO;
                    z[port.addr.at(j)] = F::ZERO;
                    z[port.val.at(j)] = F::ZERO;
                }
            }
        }

        Ok(())
    }
}

/// Typed bundle of CPU binding lanes for a single Twist instance (no CPU-side `inc` binding).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TwistPort<const N: usize> {
    pub has_read: Lane<N>,
    pub has_write: Lane<N>,
    pub read_addr: Lane<N>,
    pub write_addr: Lane<N>,
    pub rv: Lane<N>,
    pub wv: Lane<N>,
}

impl<const N: usize> TwistPort<N> {
    pub fn cpu_binding(&self) -> TwistCpuBinding {
        TwistCpuBinding {
            has_read: self.has_read.base(),
            has_write: self.has_write.base(),
            read_addr: self.read_addr.base(),
            write_addr: self.write_addr.base(),
            rv: self.rv.base(),
            wv: self.wv.base(),
            inc: None,
        }
    }

    /// Fill CPU binding columns from `StepTrace` events (common case).
    ///
    /// If no read/write event is present for a lane, fills selector/address/value with zeros.
    pub fn fill_from_trace(&self, chunk: &[StepTrace<u64, u64>], twist_id: u32, z: &mut [F]) -> Result<(), String> {
        if chunk.len() != N {
            return Err(format!(
                "TwistPort::fill_from_trace: chunk len {} != expected {}",
                chunk.len(),
                N
            ));
        }
        for (j, step) in chunk.iter().enumerate() {
            let mut read: Option<(u64, u64)> = None;
            let mut write: Option<(u64, u64)> = None;
            for ev in &step.twist_events {
                if ev.twist_id.0 != twist_id {
                    continue;
                }
                if let Some(lane) = ev.lane {
                    if lane != 0 {
                        return Err(format!(
                            "TwistPort::fill_from_trace: lane hint out of range for twist_id={twist_id} in one step (j={j}): lane={lane}"
                        ));
                    }
                }
                match ev.kind {
                    TwistOpKind::Read => {
                        if read.replace((ev.addr, ev.value)).is_some() {
                            return Err(format!(
                                "multiple twist reads for twist_id={twist_id} in one step (j={j})"
                            ));
                        }
                    }
                    TwistOpKind::Write => {
                        if write.replace((ev.addr, ev.value)).is_some() {
                            return Err(format!(
                                "multiple twist writes for twist_id={twist_id} in one step (j={j})"
                            ));
                        }
                    }
                }
            }

            if let Some((addr, val)) = read {
                z[self.has_read.at(j)] = F::ONE;
                z[self.read_addr.at(j)] = F::from_u64(addr);
                z[self.rv.at(j)] = F::from_u64(val);
            } else {
                z[self.has_read.at(j)] = F::ZERO;
                z[self.read_addr.at(j)] = F::ZERO;
                z[self.rv.at(j)] = F::ZERO;
            }
            if let Some((addr, val)) = write {
                z[self.has_write.at(j)] = F::ONE;
                z[self.write_addr.at(j)] = F::from_u64(addr);
                z[self.wv.at(j)] = F::from_u64(val);
            } else {
                z[self.has_write.at(j)] = F::ZERO;
                z[self.write_addr.at(j)] = F::ZERO;
                z[self.wv.at(j)] = F::ZERO;
            }
        }
        Ok(())
    }

    /// Fill multiple Twist access lanes from a single `twist_id` trace stream.
    ///
    /// Each lane supports at most one read and at most one write per VM step. If the trace contains
    /// more operations than lanes can support, this returns an error.
    pub fn fill_lanes_from_trace(
        lanes: &[TwistPort<N>],
        chunk: &[StepTrace<u64, u64>],
        twist_id: u32,
        z: &mut [F],
    ) -> Result<(), String> {
        if lanes.is_empty() {
            return Err("TwistPort::fill_lanes_from_trace: lanes must be non-empty".into());
        }
        if chunk.len() != N {
            return Err(format!(
                "TwistPort::fill_lanes_from_trace: chunk len {} != expected {}",
                chunk.len(),
                N
            ));
        }

        let mut reads: Vec<Option<(u64, u64)>> = vec![None; lanes.len()];
        let mut writes: Vec<Option<(u64, u64)>> = vec![None; lanes.len()];

        for (j, step) in chunk.iter().enumerate() {
            reads.fill(None);
            writes.fill(None);

            for ev in &step.twist_events {
                if ev.twist_id.0 != twist_id {
                    continue;
                }
                match ev.kind {
                    TwistOpKind::Read => {
                        let lane_idx = if let Some(lane) = ev.lane {
                            let lane_idx = usize::try_from(lane).map_err(|_| {
                                format!(
                                    "invalid twist read lane for twist_id={twist_id} in one step (j={j}): lane={lane}"
                                )
                            })?;
                            if lane_idx >= reads.len() {
                                return Err(format!(
                                    "twist read lane out of range for twist_id={twist_id} in one step (j={j}): lane={lane_idx}, lanes={}",
                                    lanes.len()
                                ));
                            }
                            if reads[lane_idx].is_some() {
                                return Err(format!(
                                    "multiple twist reads for twist_id={twist_id} in one step (j={j}) in lane={lane_idx}"
                                ));
                            }
                            lane_idx
                        } else {
                            reads
                                .iter()
                                .position(|x| x.is_none())
                                .ok_or_else(|| {
                                    format!(
                                        "too many twist reads for twist_id={twist_id} in one step (j={j}): lanes={}",
                                        lanes.len()
                                    )
                                })?
                        };
                        reads[lane_idx] = Some((ev.addr, ev.value));
                    }
                    TwistOpKind::Write => {
                        if writes.iter().flatten().any(|(addr, _)| *addr == ev.addr) {
                            return Err(format!(
                                "duplicate twist write addr for twist_id={twist_id} in one step (j={j}): addr={}",
                                ev.addr
                            ));
                        }
                        let lane_idx = if let Some(lane) = ev.lane {
                            let lane_idx = usize::try_from(lane).map_err(|_| {
                                format!(
                                    "invalid twist write lane for twist_id={twist_id} in one step (j={j}): lane={lane}"
                                )
                            })?;
                            if lane_idx >= writes.len() {
                                return Err(format!(
                                    "twist write lane out of range for twist_id={twist_id} in one step (j={j}): lane={lane_idx}, lanes={}",
                                    lanes.len()
                                ));
                            }
                            if writes[lane_idx].is_some() {
                                return Err(format!(
                                    "multiple twist writes for twist_id={twist_id} in one step (j={j}) in lane={lane_idx}"
                                ));
                            }
                            lane_idx
                        } else {
                            writes
                                .iter()
                                .position(|x| x.is_none())
                                .ok_or_else(|| {
                                    format!(
                                        "too many twist writes for twist_id={twist_id} in one step (j={j}): lanes={}",
                                        lanes.len()
                                    )
                                })?
                        };
                        writes[lane_idx] = Some((ev.addr, ev.value));
                    }
                }
            }

            for (lane_idx, port) in lanes.iter().enumerate() {
                if let Some((addr, val)) = reads[lane_idx] {
                    z[port.has_read.at(j)] = F::ONE;
                    z[port.read_addr.at(j)] = F::from_u64(addr);
                    z[port.rv.at(j)] = F::from_u64(val);
                } else {
                    z[port.has_read.at(j)] = F::ZERO;
                    z[port.read_addr.at(j)] = F::ZERO;
                    z[port.rv.at(j)] = F::ZERO;
                }

                if let Some((addr, val)) = writes[lane_idx] {
                    z[port.has_write.at(j)] = F::ONE;
                    z[port.write_addr.at(j)] = F::from_u64(addr);
                    z[port.wv.at(j)] = F::from_u64(val);
                } else {
                    z[port.has_write.at(j)] = F::ZERO;
                    z[port.write_addr.at(j)] = F::ZERO;
                    z[port.wv.at(j)] = F::ZERO;
                }
            }
        }

        Ok(())
    }
}

/// Typed bundle of CPU binding lanes for a single Twist instance, including CPU-side `inc` binding.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TwistPortWithInc<const N: usize> {
    pub has_read: Lane<N>,
    pub has_write: Lane<N>,
    pub read_addr: Lane<N>,
    pub write_addr: Lane<N>,
    pub rv: Lane<N>,
    pub wv: Lane<N>,
    pub inc: Lane<N>,
}

impl<const N: usize> TwistPortWithInc<N> {
    pub fn cpu_binding(&self) -> TwistCpuBinding {
        TwistCpuBinding {
            has_read: self.has_read.base(),
            has_write: self.has_write.base(),
            read_addr: self.read_addr.base(),
            write_addr: self.write_addr.base(),
            rv: self.rv.base(),
            wv: self.wv.base(),
            inc: Some(self.inc.base()),
        }
    }

    /// Fill CPU binding columns from `StepTrace` events.
    ///
    /// Note: This does **not** currently compute `inc`; it is set to 0 for all lanes.
    pub fn fill_from_trace(&self, chunk: &[StepTrace<u64, u64>], twist_id: u32, z: &mut [F]) -> Result<(), String> {
        if chunk.len() != N {
            return Err(format!(
                "TwistPortWithInc::fill_from_trace: chunk len {} != expected {}",
                chunk.len(),
                N
            ));
        }
        for (j, step) in chunk.iter().enumerate() {
            let mut read: Option<(u64, u64)> = None;
            let mut write: Option<(u64, u64)> = None;
            for ev in &step.twist_events {
                if ev.twist_id.0 != twist_id {
                    continue;
                }
                if let Some(lane) = ev.lane {
                    if lane != 0 {
                        return Err(format!(
                            "TwistPortWithInc::fill_from_trace: lane hint out of range for twist_id={twist_id} in one step (j={j}): lane={lane}"
                        ));
                    }
                }
                match ev.kind {
                    TwistOpKind::Read => {
                        if read.replace((ev.addr, ev.value)).is_some() {
                            return Err(format!(
                                "multiple twist reads for twist_id={twist_id} in one step (j={j})"
                            ));
                        }
                    }
                    TwistOpKind::Write => {
                        if write.replace((ev.addr, ev.value)).is_some() {
                            return Err(format!(
                                "multiple twist writes for twist_id={twist_id} in one step (j={j})"
                            ));
                        }
                    }
                }
            }

            if let Some((addr, val)) = read {
                z[self.has_read.at(j)] = F::ONE;
                z[self.read_addr.at(j)] = F::from_u64(addr);
                z[self.rv.at(j)] = F::from_u64(val);
            } else {
                z[self.has_read.at(j)] = F::ZERO;
                z[self.read_addr.at(j)] = F::ZERO;
                z[self.rv.at(j)] = F::ZERO;
            }
            if let Some((addr, val)) = write {
                z[self.has_write.at(j)] = F::ONE;
                z[self.write_addr.at(j)] = F::from_u64(addr);
                z[self.wv.at(j)] = F::from_u64(val);
            } else {
                z[self.has_write.at(j)] = F::ZERO;
                z[self.write_addr.at(j)] = F::ZERO;
                z[self.wv.at(j)] = F::ZERO;
            }
            z[self.inc.at(j)] = F::ZERO;
        }
        Ok(())
    }
}

/// Declare a witness-column layout with typed lanes and Twist/Shout ports.
///
/// Notes:
/// - All `Public<...>` fields must appear before any private fields.
/// - `Scalar` expands to a single column index; `Lane<N>` expands to `N` consecutive indices.
/// - `TwistPort<N>` expands to 6 `Lane<N>`s (no CPU `inc` binding).
/// - `TwistPortWithInc<N>` expands to 7 `Lane<N>`s (includes CPU `inc` binding).
/// - `ShoutPort<N>` expands to 3 `Lane<N>`s.
#[macro_export]
macro_rules! witness_layout {
    (
        $(#[$struct_meta:meta])*
        $vis:vis $name:ident < $(const $gen_name:ident : $gen_ty:ty),+ > {
            $(
                $(#[$field_meta:meta])*
                $field_vis:vis $field:ident : $fty:ty
            ),* $(,)?
        }
    ) => {
        $(#[$struct_meta])*
        $vis struct $name < $(const $gen_name : $gen_ty),+ > {
            $(
                $(#[$field_meta])*
                $field_vis $field : <$fty as $crate::session::WitnessLayoutField>::Field,
            )*
        }

        impl < $(const $gen_name : $gen_ty),+ > $name < $($gen_name),+ > {
            pub const M_IN: usize = 0usize $(+ <$fty as $crate::session::WitnessLayoutField>::PUBLIC_LEN)*;
            pub const USED_COLS: usize = 0usize $(+ <$fty as $crate::session::WitnessLayoutField>::LEN)*;

            pub fn new() -> Self {
                let mut alloc = $crate::session::WitnessLayoutAllocator::new();
                $(
                    let $field = <$fty as $crate::session::WitnessLayoutField>::alloc(&mut alloc);
                )*
                debug_assert_eq!(
                    alloc.m_in(),
                    Self::M_IN,
                    "witness_layout!: allocated m_in mismatch (Public<> fields must come first)"
                );
                debug_assert_eq!(
                    alloc.used_cols(),
                    Self::USED_COLS,
                    "witness_layout!: allocated USED_COLS mismatch"
                );
                Self { $($field),* }
            }
        }

        impl < $(const $gen_name : $gen_ty),+ > $crate::session::WitnessLayout for $name < $($gen_name),+ > {
            const M_IN: usize = <$name < $($gen_name),+ >>::M_IN;
            const USED_COLS: usize = <$name < $($gen_name),+ >>::USED_COLS;

            fn new_layout() -> Self {
                <$name < $($gen_name),+ >>::new()
            }
        }
    };

    (
        $(#[$struct_meta:meta])*
        $vis:vis $name:ident {
            $(
                $(#[$field_meta:meta])*
                $field_vis:vis $field:ident : $fty:ty
            ),* $(,)?
        }
    ) => {
        $(#[$struct_meta])*
        $vis struct $name {
            $(
                $(#[$field_meta])*
                $field_vis $field : <$fty as $crate::session::WitnessLayoutField>::Field,
            )*
        }

        impl $name {
            pub const M_IN: usize = 0usize $(+ <$fty as $crate::session::WitnessLayoutField>::PUBLIC_LEN)*;
            pub const USED_COLS: usize = 0usize $(+ <$fty as $crate::session::WitnessLayoutField>::LEN)*;

            pub fn new() -> Self {
                let mut alloc = $crate::session::WitnessLayoutAllocator::new();
                $(
                    let $field = <$fty as $crate::session::WitnessLayoutField>::alloc(&mut alloc);
                )*
                debug_assert_eq!(
                    alloc.m_in(),
                    Self::M_IN,
                    "witness_layout!: allocated m_in mismatch (Public<> fields must come first)"
                );
                debug_assert_eq!(
                    alloc.used_cols(),
                    Self::USED_COLS,
                    "witness_layout!: allocated USED_COLS mismatch"
                );
                Self { $($field),* }
            }
        }

        impl $crate::session::WitnessLayout for $name {
            const M_IN: usize = <$name>::M_IN;
            const USED_COLS: usize = <$name>::USED_COLS;

            fn new_layout() -> Self {
                <$name>::new()
            }
        }
    };
}

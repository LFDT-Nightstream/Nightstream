use neo_math::F;
use neo_memory::plain::{LutTable, PlainMemLayout};
use neo_memory::witness::LutTableSpec;
use p3_field::PrimeCharacteristicRing;
use std::collections::HashMap;
use std::iter;

/// Declarative resource configuration for shared-CPU-bus runs (Twist + Shout).
///
/// This is a thin convenience wrapper around the maps consumed by
/// `neo_memory::builder::build_shard_witness_shared_cpu_bus*`.
#[derive(Clone, Debug, Default)]
pub struct SharedBusResources {
    pub mem_layouts: HashMap<u32, PlainMemLayout>,
    pub initial_mem: HashMap<(u32, u64), F>,
    pub lut_tables: HashMap<u32, LutTable<F>>,
    pub lut_table_specs: HashMap<u32, LutTableSpec>,
    /// Number of lookup lanes per VM step for each Shout table_id.
    ///
    /// Defaults to 1 when absent.
    pub lut_lanes: HashMap<u32, usize>,
}

impl SharedBusResources {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn twist(&mut self, twist_id: u32) -> TwistResource<'_> {
        TwistResource { resources: self, twist_id }
    }

    pub fn shout(&mut self, table_id: u32) -> ShoutResource<'_> {
        ShoutResource { resources: self, table_id }
    }

    /// Convenience: set a power-of-two sized table under binary addressing (`n_side = 2`).
    ///
    /// This treats the Shout `key` as an index in `[0, k)` encoded in little-endian bits, with:
    /// - `k = content.len()`
    /// - `d = log2(k)`
    /// - `n_side = 2`
    pub fn set_binary_table(&mut self, table_id: u32, content: Vec<F>) {
        self.shout(table_id).binary_table(content);
    }

    /// Convenience: set a (possibly non-power-of-two sized) table under binary addressing (`n_side = 2`),
    /// padding with `0`s up to the next power of two.
    ///
    /// This treats the Shout `key` as an index in `[0, k)` encoded in little-endian bits, with:
    /// - `k = next_power_of_two(content.len())`
    /// - `d = log2(k)`
    /// - `n_side = 2`
    ///
    /// Padding is convenient when you want a small lookup table without manually choosing `(k, d)`.
    pub fn set_padded_binary_table(&mut self, table_id: u32, content: Vec<F>) {
        self.shout(table_id).padded_binary_table(content);
    }

    /// Convenience: set a power-of-two sized memory under binary addressing (`n_side = 2`).
    ///
    /// This treats the Twist `addr` as an index in `[0, k)` encoded in little-endian bits, with:
    /// - `k` as provided
    /// - `d = log2(k)`
    /// - `n_side = 2`
    ///
    /// This does not set any initial memory; use `twist(id).init(...)` for that.
    pub fn set_binary_mem_layout(&mut self, twist_id: u32, k: usize) {
        assert!(k > 0, "set_binary_mem_layout: k must be > 0");
        assert!(k.is_power_of_two(), "set_binary_mem_layout: k must be a power of two");
        let d = k.trailing_zeros() as usize;
        self.twist(twist_id).layout(PlainMemLayout {
            k,
            d,
            n_side: 2,
            lanes: 1,
        });
    }

    /// Remove any explicit table when switching a table_id to an implicit spec (or vice-versa).
    fn clear_shout_conflicts(&mut self, table_id: u32, keep_spec: bool) {
        if keep_spec {
            self.lut_tables.remove(&table_id);
        } else {
            self.lut_table_specs.remove(&table_id);
        }
    }
}

pub struct TwistResource<'a> {
    resources: &'a mut SharedBusResources,
    twist_id: u32,
}

impl TwistResource<'_> {
    pub fn layout(self, layout: PlainMemLayout) -> Self {
        self.resources.mem_layouts.insert(self.twist_id, layout);
        self
    }

    /// Add sparse initial memory entries for this Twist instance.
    ///
    /// Values of `F::ZERO` are allowed but typically ignored by the underlying builders.
    pub fn init(self, init: impl IntoIterator<Item = (u64, F)>) -> Self {
        for (addr, val) in init {
            self.resources.initial_mem.insert((self.twist_id, addr), val);
        }
        self
    }

    pub fn init_cell(self, addr: u64, val: F) -> Self {
        self.init(iter::once((addr, val)))
    }

    pub fn clear_init(self) -> Self {
        self.resources
            .initial_mem
            .retain(|(mem_id, _addr), _| *mem_id != self.twist_id);
        self
    }
}

pub struct ShoutResource<'a> {
    resources: &'a mut SharedBusResources,
    table_id: u32,
}

impl ShoutResource<'_> {
    pub fn lanes(self, lanes: usize) -> Self {
        self.resources.lut_lanes.insert(self.table_id, lanes.max(1));
        self
    }

    pub fn table(self, mut table: LutTable<F>) -> Self {
        table.table_id = self.table_id;
        self.resources.clear_shout_conflicts(self.table_id, /*keep_spec=*/ false);
        self.resources.lut_tables.insert(self.table_id, table);
        self
    }

    pub fn binary_table(self, content: Vec<F>) -> Self {
        let table_id = self.table_id;
        let k = content.len();
        assert!(k > 0, "binary_table: content must be non-empty");
        assert!(k.is_power_of_two(), "binary_table: content.len() must be a power of two");
        let d = k.trailing_zeros() as usize;
        self.table(LutTable {
            table_id,
            k,
            d,
            n_side: 2,
            content,
        })
    }

    pub fn padded_binary_table(self, mut content: Vec<F>) -> Self {
        let table_id = self.table_id;
        let k_raw = content.len();
        assert!(k_raw > 0, "padded_binary_table: content must be non-empty");
        let k = k_raw.next_power_of_two();
        let d = k.trailing_zeros() as usize;
        content.resize(k, F::ZERO);
        self.table(LutTable {
            table_id,
            k,
            d,
            n_side: 2,
            content,
        })
    }

    pub fn spec(self, spec: LutTableSpec) -> Self {
        self.resources.clear_shout_conflicts(self.table_id, /*keep_spec=*/ true);
        self.resources.lut_table_specs.insert(self.table_id, spec);
        self
    }
}

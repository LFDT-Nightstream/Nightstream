//! Domain-neutral witness-column metadata and index ownership.

use std::borrow::Cow;
use std::collections::BTreeSet;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ColumnId(pub usize);

impl From<usize> for ColumnId {
    fn from(index: usize) -> Self {
        Self(index)
    }
}

/// Declared range and enforcement policy of a witness column.
///
/// Boolean, Byte, U32 and Bits request explicit range constraints. Implied and
/// Repairable omit them and carry proof obligations instead. Soundness requires
/// the application to discharge every such obligation against the actual relation
/// with those checks omitted. Merely declaring or exporting an obligation does
/// not prove it; this Rust metadata layer does not check the proof.
///
/// The proof method is application-specific: for example Rocq, Lean, or an SMT
/// procedure that establishes the stated obligation. Neither annotation grants
/// a narrower backend representation without a checked justification.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ColumnWidth {
    Boolean,
    Byte,
    U32,
    /// An unsigned value represented by exactly this many bits.
    /// Explicit Goldilocks decomposition supports widths `1..=63`.
    Bits(u8),
    /// Requires proof that every accepted witness satisfies this unsigned bit bound
    /// through other constraints. This is an obligation, not an established fact.
    Implied(u8),
    /// Requires proof that every accepted witness has an accepted bounded equivalent,
    /// changing only this column and preserving all application-observable effects.
    /// This does not assert that the original witness is bounded or prove application
    /// soundness by itself; it justifies omitting this particular range check.
    Repairable(u8),
    Field,
}

impl ColumnWidth {
    pub const fn is_boolean(self) -> bool {
        matches!(self, Self::Boolean | Self::Bits(1))
    }
}

/// Declares one contiguous family of witness columns with a shared value width.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ColumnFamilySpec {
    pub region: &'static str,
    pub start: usize,
    pub len: usize,
    pub name: Cow<'static, str>,
    pub role: &'static str,
    pub width: ColumnWidth,
}

impl ColumnFamilySpec {
    pub const fn end(&self) -> usize {
        self.start + self.len
    }

    pub const fn contains(&self, column: usize) -> bool {
        self.start <= column && column < self.end()
    }
}

/// Define one contiguous region of statically allocated witness columns.
///
/// Scalar entries name their intrinsic width directly. Families use Rust's
/// `[width; length]` array notation and generate arrays of absolute indices.
/// Bit bounds use `(Bits(n))`, `(Implied(n))`, or `(Repairable(n))`.
/// Each also supports array notation, for example `[(Implied(32)); length]`.
#[macro_export]
macro_rules! define_column_region {
    (
        region: $region:literal,
        start: $start:expr,
        width: $width_vis:vis $width_name:ident,
        families: $families_vis:vis $families_name:ident,
        indices: $index_vis:vis,
        columns: [
            $($name:ident: $shape:tt => $role:literal),+ $(,)?
        ]
    ) => {
        $crate::define_column_region!(@assign $index_vis, $start; $(($name, $shape)),+);

        /// Number of witness columns allocated by this region.
        $width_vis const $width_name: usize =
            0usize $(+ $crate::define_column_region!(@len $shape))+;
        /// Macro-generated metadata for the region's declared column families.
        $families_vis const $families_name: &[$crate::ColumnFamilySpec] = &[
            $($crate::ColumnFamilySpec {
                region: $region,
                start: $crate::define_column_region!(@start $name, $shape),
                len: $crate::define_column_region!(@len $shape),
                name: ::std::borrow::Cow::Borrowed(stringify!($name)),
                role: $role,
                width: $crate::define_column_region!(@width $shape),
            }),+
        ];
    };
    (@assign $index_vis:vis, $idx:expr; ($name:ident, $shape:tt), $(($rest_name:ident, $rest_shape:tt)),+) => {
        $crate::define_column_region!(@declare $index_vis, $idx; $name, $shape);
        $crate::define_column_region!(
            @assign $index_vis,
            $idx + $crate::define_column_region!(@len $shape);
            $(($rest_name, $rest_shape)),+
        );
    };
    (@assign $index_vis:vis, $idx:expr; ($name:ident, $shape:tt)) => {
        $crate::define_column_region!(@declare $index_vis, $idx; $name, $shape);
    };
    (@declare $index_vis:vis, $idx:expr; $name:ident, $column_width:ident) => {
        $index_vis const $name: usize = $idx;
    };
    (@declare $index_vis:vis, $idx:expr; $name:ident, ($kind:ident($bits:expr))) => {
        $index_vis const $name: usize = $idx;
    };
    (@declare $index_vis:vis, $idx:expr; $name:ident, [$column_width:ident; $len:expr]) => {
        $index_vis const $name: [usize; $len] = {
            let mut indices = [0; $len];
            let mut i = 0;
            while i < $len {
                indices[i] = $idx + i;
                i += 1;
            }
            indices
        };
    };
    (@declare $index_vis:vis, $idx:expr; $name:ident, [($kind:ident($bits:expr)); $len:expr]) => {
        $index_vis const $name: [usize; $len] = {
            let mut indices = [0; $len];
            let mut i = 0;
            while i < $len {
                indices[i] = $idx + i;
                i += 1;
            }
            indices
        };
    };
    (@start $name:ident, $column_width:ident) => { $name };
    (@start $name:ident, ($kind:ident($bits:expr))) => { $name };
    (@start $name:ident, [$column_width:ident; $len:expr]) => { $name[0] };
    (@start $name:ident, [($kind:ident($bits:expr)); $len:expr]) => { $name[0] };
    (@len $column_width:ident) => { 1usize };
    (@len ($kind:ident($bits:expr))) => { 1usize };
    (@len [$column_width:ident; $len:expr]) => { $len };
    (@len [($kind:ident($bits:expr)); $len:expr]) => { $len };
    (@width $column_width:ident) => { $crate::ColumnWidth::$column_width };
    (@width ($kind:ident($bits:expr))) => { $crate::ColumnWidth::$kind($bits) };
    (@width [$column_width:ident; $len:expr]) => { $crate::ColumnWidth::$column_width };
    (@width [($kind:ident($bits:expr)); $len:expr]) => { $crate::ColumnWidth::$kind($bits) };
}

/// Complete, ordered ownership map for an application witness vector.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ColumnRegistry {
    families: Vec<ColumnFamilySpec>,
    column_count: usize,
}

impl ColumnRegistry {
    pub fn new(families: impl IntoIterator<Item = ColumnFamilySpec>) -> Result<Self, ColumnRegistryError> {
        let families: Vec<_> = families.into_iter().collect();
        if families.is_empty() {
            return Err(ColumnRegistryError::Empty);
        }

        let mut expected_start = 0usize;
        let mut names = BTreeSet::new();
        for family in &families {
            if family.region.is_empty() {
                return Err(ColumnRegistryError::EmptyRegion {
                    name: family.name.clone(),
                });
            }
            if family.name.is_empty() {
                return Err(ColumnRegistryError::EmptyName { start: family.start });
            }
            if family.len == 0 {
                return Err(ColumnRegistryError::EmptyFamily {
                    name: family.name.clone(),
                });
            }
            if let ColumnWidth::Bits(bits) | ColumnWidth::Implied(bits) | ColumnWidth::Repairable(bits) = family.width {
                if bits == 0 || bits > 63 {
                    return Err(ColumnRegistryError::InvalidBitWidth {
                        name: family.name.clone(),
                        bits,
                    });
                }
            }
            if !names.insert(family.name.clone()) {
                return Err(ColumnRegistryError::DuplicateName {
                    name: family.name.clone(),
                });
            }
            if family.start != expected_start {
                return Err(ColumnRegistryError::NonContiguous {
                    name: family.name.clone(),
                    expected_start,
                    actual_start: family.start,
                });
            }
            expected_start = family
                .start
                .checked_add(family.len)
                .ok_or(ColumnRegistryError::IndexOverflow {
                    name: family.name.clone(),
                })?;
        }

        Ok(Self {
            families,
            column_count: expected_start,
        })
    }

    pub fn column_count(&self) -> usize {
        self.column_count
    }

    pub fn families(&self) -> &[ColumnFamilySpec] {
        &self.families
    }

    pub fn family_for_column(&self, column: usize) -> Option<&ColumnFamilySpec> {
        if column >= self.column_count {
            return None;
        }

        // `new` guarantees a nonempty, contiguous registry starting at zero,
        // so every in-range column has exactly one preceding family start.
        let index = self
            .families
            .partition_point(|family| family.start <= column)
            - 1;
        Some(&self.families[index])
    }
}

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum ColumnRegistryError {
    #[error("column registry requires at least one column family")]
    Empty,
    #[error("column family `{name}` has an empty region name")]
    EmptyRegion { name: Cow<'static, str> },
    #[error("column family at index {start} has an empty name")]
    EmptyName { start: usize },
    #[error("column family `{name}` must contain at least one column")]
    EmptyFamily { name: Cow<'static, str> },
    #[error("column family `{name}` has custom bit width {bits}; expected 1..=63")]
    InvalidBitWidth { name: Cow<'static, str>, bits: u8 },
    #[error("column family name `{name}` is declared more than once")]
    DuplicateName { name: Cow<'static, str> },
    #[error("column family `{name}` starts at {actual_start}, expected {expected_start}")]
    NonContiguous {
        name: Cow<'static, str>,
        expected_start: usize,
        actual_start: usize,
    },
    #[error("column family `{name}` overflows the column index space")]
    IndexOverflow { name: Cow<'static, str> },
}

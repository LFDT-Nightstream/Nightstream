use crate::output_check::ProgramIO;
use crate::riscv::elf_loader::LoadedProgram;
use p3_field::Field;
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// How guest addresses are remapped into the proof address domain.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProofAddressRemapKind {
    /// Keep guest addresses unchanged.
    Identity,
    /// Remap each loadable segment into a dense word-addressed logical domain.
    SegmentedWordAddress,
}

/// One region in the guest memory image.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RiscvGuestMemoryRegion {
    pub name: String,
    pub guest_base: u64,
    pub len_bytes: u64,
    pub proof_base: u64,
    pub read_only: bool,
    pub executable: bool,
}

impl RiscvGuestMemoryRegion {
    #[inline]
    pub fn guest_end(&self) -> u64 {
        self.guest_base.saturating_add(self.len_bytes)
    }
}

/// A single affine guest-to-proof address remap.
///
/// This is the narrow form used by the current RV64 RAM bridge:
/// one contiguous guest region, one fixed cell size, and one affine logical
/// address domain.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AffineWordAddressRemap {
    pub guest_base: u64,
    pub logical_base: u64,
    pub cell_bytes: u64,
    pub cells: u64,
}

/// Deterministic guest-memory layout bound to the proof.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RiscvGuestMemoryLayout {
    pub xlen: usize,
    pub cell_bytes: u64,
    pub remap_kind: ProofAddressRemapKind,
    pub regions: Vec<RiscvGuestMemoryRegion>,
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum RiscvMemoryLayoutError {
    #[error("xlen must be 32 or 64 (got {0})")]
    InvalidXlen(usize),
    #[error("cell_bytes must be non-zero")]
    InvalidCellBytes,
    #[error("region `{name}` has zero length")]
    ZeroLengthRegion { name: String },
    #[error("region `{left}` overlaps `{right}`")]
    OverlappingRegions { left: String, right: String },
    #[error("guest address {addr:#x} is not aligned to cell size {cell_bytes}")]
    UnalignedAddress { addr: u64, cell_bytes: u64 },
    #[error("guest address {addr:#x} does not belong to any mapped region")]
    AddressOutsideMappedRegions { addr: u64 },
    #[error("multiple guest addresses remap to the same logical address {addr}")]
    DuplicateLogicalAddress { addr: u64 },
    #[error("proof address space overflow while laying out `{name}`")]
    ProofSpaceOverflow { name: String },
    #[error("affine remap must contain at least one cell")]
    InvalidAffineCells,
    #[error("affine remap guest address {addr:#x} lies outside [{guest_base:#x}, {guest_end:#x})")]
    AddressOutsideAffineRemap {
        addr: u64,
        guest_base: u64,
        guest_end: u64,
    },
}

impl AffineWordAddressRemap {
    pub fn new(
        guest_base: u64,
        logical_base: u64,
        cell_bytes: u64,
        cells: u64,
    ) -> Result<Self, RiscvMemoryLayoutError> {
        if cell_bytes == 0 {
            return Err(RiscvMemoryLayoutError::InvalidCellBytes);
        }
        if cells == 0 {
            return Err(RiscvMemoryLayoutError::InvalidAffineCells);
        }
        let len_bytes = cells
            .checked_mul(cell_bytes)
            .ok_or_else(|| RiscvMemoryLayoutError::ProofSpaceOverflow {
                name: "affine-remap".into(),
            })?;
        let _ = guest_base
            .checked_add(len_bytes)
            .ok_or_else(|| RiscvMemoryLayoutError::ProofSpaceOverflow {
                name: "affine-remap".into(),
            })?;
        let _ = logical_base
            .checked_add(cells)
            .ok_or_else(|| RiscvMemoryLayoutError::ProofSpaceOverflow {
                name: "affine-remap".into(),
            })?;
        Ok(Self {
            guest_base,
            logical_base,
            cell_bytes,
            cells,
        })
    }

    pub fn from_guest_addresses<I>(guest_addrs: I, cell_bytes: u64) -> Result<Option<Self>, RiscvMemoryLayoutError>
    where
        I: IntoIterator<Item = u64>,
    {
        if cell_bytes == 0 {
            return Err(RiscvMemoryLayoutError::InvalidCellBytes);
        }
        let mut min_addr = u64::MAX;
        let mut max_addr = 0u64;
        let mut saw_any = false;
        for addr in guest_addrs {
            if addr % cell_bytes != 0 {
                return Err(RiscvMemoryLayoutError::UnalignedAddress { addr, cell_bytes });
            }
            min_addr = min_addr.min(addr);
            max_addr = max_addr.max(addr);
            saw_any = true;
        }
        if !saw_any {
            return Ok(None);
        }
        let span = max_addr
            .checked_sub(min_addr)
            .ok_or_else(|| RiscvMemoryLayoutError::ProofSpaceOverflow {
                name: "affine-remap".into(),
            })?;
        let cells = span
            .checked_div(cell_bytes)
            .and_then(|v| v.checked_add(1))
            .ok_or_else(|| RiscvMemoryLayoutError::ProofSpaceOverflow {
                name: "affine-remap".into(),
            })?;
        Ok(Some(Self::new(min_addr, 0, cell_bytes, cells)?))
    }

    #[inline]
    pub fn guest_end(&self) -> u64 {
        self.guest_base
            .saturating_add(self.cells.saturating_mul(self.cell_bytes))
    }

    #[inline]
    pub fn logical_end(&self) -> u64 {
        self.logical_base.saturating_add(self.cells)
    }

    pub fn remap_guest_addr(&self, guest_addr: u64) -> Result<u64, RiscvMemoryLayoutError> {
        if guest_addr % self.cell_bytes != 0 {
            return Err(RiscvMemoryLayoutError::UnalignedAddress {
                addr: guest_addr,
                cell_bytes: self.cell_bytes,
            });
        }
        let guest_end = self.guest_end();
        if guest_addr < self.guest_base || guest_addr >= guest_end {
            return Err(RiscvMemoryLayoutError::AddressOutsideAffineRemap {
                addr: guest_addr,
                guest_base: self.guest_base,
                guest_end,
            });
        }
        let offset = (guest_addr - self.guest_base) / self.cell_bytes;
        self.logical_base
            .checked_add(offset)
            .ok_or_else(|| RiscvMemoryLayoutError::ProofSpaceOverflow {
                name: "affine-remap".into(),
            })
    }

    pub fn to_memory_layout(
        &self,
        xlen: usize,
        name: impl Into<String>,
    ) -> Result<RiscvGuestMemoryLayout, RiscvMemoryLayoutError> {
        let len_bytes =
            self.cells
                .checked_mul(self.cell_bytes)
                .ok_or_else(|| RiscvMemoryLayoutError::ProofSpaceOverflow {
                    name: "affine-remap".into(),
                })?;
        RiscvGuestMemoryLayout::new(
            xlen,
            self.cell_bytes,
            ProofAddressRemapKind::SegmentedWordAddress,
            vec![RiscvGuestMemoryRegion {
                name: name.into(),
                guest_base: self.guest_base,
                len_bytes,
                proof_base: self.logical_base,
                read_only: false,
                executable: false,
            }],
        )
    }

    #[inline]
    pub fn required_num_bits(&self) -> usize {
        bits_for_domain_size(self.logical_end().max(1))
    }
}

impl RiscvGuestMemoryLayout {
    pub fn new(
        xlen: usize,
        cell_bytes: u64,
        remap_kind: ProofAddressRemapKind,
        mut regions: Vec<RiscvGuestMemoryRegion>,
    ) -> Result<Self, RiscvMemoryLayoutError> {
        if !matches!(xlen, 32 | 64) {
            return Err(RiscvMemoryLayoutError::InvalidXlen(xlen));
        }
        if cell_bytes == 0 {
            return Err(RiscvMemoryLayoutError::InvalidCellBytes);
        }

        regions.sort_by_key(|region| region.guest_base);
        for region in &regions {
            if region.len_bytes == 0 {
                return Err(RiscvMemoryLayoutError::ZeroLengthRegion {
                    name: region.name.clone(),
                });
            }
        }

        for pair in regions.windows(2) {
            let left = &pair[0];
            let right = &pair[1];
            if left.guest_end() > right.guest_base {
                return Err(RiscvMemoryLayoutError::OverlappingRegions {
                    left: left.name.clone(),
                    right: right.name.clone(),
                });
            }
        }

        Ok(Self {
            xlen,
            cell_bytes,
            remap_kind,
            regions,
        })
    }

    pub fn from_loaded_program(loaded: &LoadedProgram, xlen: usize) -> Result<Self, RiscvMemoryLayoutError> {
        let mut segments: Vec<_> = loaded.segments.iter().collect();
        segments.sort_by_key(|segment| segment.vaddr);
        let mut regions = Vec::with_capacity(segments.len());
        let mut next_proof_base = 0u64;
        for (idx, segment) in segments.into_iter().enumerate() {
            let cells = cells_for_len(segment.mem_size, 4)?;
            let region = RiscvGuestMemoryRegion {
                name: format!("load{idx}"),
                guest_base: segment.vaddr,
                len_bytes: segment.mem_size,
                proof_base: next_proof_base,
                read_only: !segment.flags.write,
                executable: segment.flags.execute,
            };
            next_proof_base =
                next_proof_base
                    .checked_add(cells)
                    .ok_or_else(|| RiscvMemoryLayoutError::ProofSpaceOverflow {
                        name: region.name.clone(),
                    })?;
            regions.push(region);
        }
        Self::new(xlen, 4, ProofAddressRemapKind::SegmentedWordAddress, regions)
    }

    pub fn with_public_region(
        &self,
        name: impl Into<String>,
        guest_base: u64,
        len_bytes: u64,
        read_only: bool,
        executable: bool,
    ) -> Result<Self, RiscvMemoryLayoutError> {
        let next_base = self
            .regions
            .iter()
            .map(|region| {
                region
                    .proof_base
                    .checked_add(cells_for_len(region.len_bytes, self.cell_bytes).unwrap_or(0))
                    .unwrap_or(u64::MAX)
            })
            .max()
            .unwrap_or(0);
        let mut regions = self.regions.clone();
        regions.push(RiscvGuestMemoryRegion {
            name: name.into(),
            guest_base,
            len_bytes,
            proof_base: next_base,
            read_only,
            executable,
        });
        Self::new(self.xlen, self.cell_bytes, self.remap_kind, regions)
    }

    pub fn remap_address(&self, guest_addr: u64) -> Result<u64, RiscvMemoryLayoutError> {
        match self.remap_kind {
            ProofAddressRemapKind::Identity => Ok(guest_addr),
            ProofAddressRemapKind::SegmentedWordAddress => {
                if guest_addr % self.cell_bytes != 0 {
                    return Err(RiscvMemoryLayoutError::UnalignedAddress {
                        addr: guest_addr,
                        cell_bytes: self.cell_bytes,
                    });
                }
                let region = self
                    .regions
                    .iter()
                    .find(|region| guest_addr >= region.guest_base && guest_addr < region.guest_end())
                    .ok_or(RiscvMemoryLayoutError::AddressOutsideMappedRegions { addr: guest_addr })?;
                let offset = guest_addr - region.guest_base;
                Ok(region.proof_base + (offset / self.cell_bytes))
            }
        }
    }

    pub fn remap_program_io<F: Field>(&self, guest_io: &ProgramIO<F>) -> Result<ProgramIO<F>, RiscvMemoryLayoutError> {
        let mut out = ProgramIO::new();
        for (guest_addr, value) in guest_io.claims() {
            let logical_addr = self.remap_address(guest_addr)?;
            out = out
                .try_with_claim(logical_addr, value)
                .map_err(|_| RiscvMemoryLayoutError::DuplicateLogicalAddress { addr: logical_addr })?;
        }
        Ok(out)
    }

    pub fn required_num_bits(&self) -> usize {
        let logical_cells = self
            .regions
            .iter()
            .map(|region| {
                region
                    .proof_base
                    .saturating_add(cells_for_len(region.len_bytes, self.cell_bytes).unwrap_or(0))
            })
            .max()
            .unwrap_or(0);
        bits_for_domain_size(logical_cells.max(1))
    }
}

fn cells_for_len(len_bytes: u64, cell_bytes: u64) -> Result<u64, RiscvMemoryLayoutError> {
    if cell_bytes == 0 {
        return Err(RiscvMemoryLayoutError::InvalidCellBytes);
    }
    let extra = cell_bytes - 1;
    Ok(len_bytes.saturating_add(extra) / cell_bytes)
}

fn bits_for_domain_size(size: u64) -> usize {
    if size <= 1 {
        1
    } else {
        (u64::BITS - (size - 1).leading_zeros()) as usize
    }
}

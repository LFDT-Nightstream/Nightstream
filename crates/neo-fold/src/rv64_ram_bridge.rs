use std::collections::HashMap;

use neo_math::F;
use neo_memory::output_check::ProgramIO;
use neo_memory::riscv::elf_loader::LoadedProgram;
use neo_memory::riscv::lookups::RAM_ID;
use neo_memory::{AffineWordAddressRemap, RiscvGuestMemoryLayout};
use neo_vm_trace::VmTrace;

use crate::PiCcsError;

pub(crate) struct Rv64RamBridge {
    pub remap: AffineWordAddressRemap,
    pub memory_layout: RiscvGuestMemoryLayout,
    pub logical_output_claims: ProgramIO<F>,
    pub logical_ram_init_words: HashMap<u64, u64>,
    pub output_num_bits: usize,
}

pub(crate) fn derive_rv64_ram_bridge(
    trace: &VmTrace<u64, u64, u128>,
    xlen: usize,
    guest_output_claims: &ProgramIO<F>,
    guest_ram_init_words: &HashMap<u64, u64>,
) -> Result<Option<Rv64RamBridge>, PiCcsError> {
    let mut guest_addrs = Vec::new();
    for step in &trace.steps {
        for ev in &step.twist_events {
            if ev.twist_id == RAM_ID {
                guest_addrs.push(ev.addr);
            }
        }
    }
    guest_addrs.extend(guest_ram_init_words.keys().copied());
    guest_addrs.extend(guest_output_claims.claimed_addresses());

    let Some(remap) = AffineWordAddressRemap::from_guest_addresses(guest_addrs, /*cell_bytes=*/ 1)
        .map_err(|e| PiCcsError::InvalidInput(format!("RV64 RAM remap derivation failed: {e}")))?
    else {
        return Ok(None);
    };

    let memory_layout = remap
        .to_memory_layout(xlen, "rv64-ram")
        .map_err(|e| PiCcsError::InvalidInput(format!("RV64 RAM proof layout build failed: {e}")))?;
    let logical_output_claims = memory_layout
        .remap_program_io(guest_output_claims)
        .map_err(|e| PiCcsError::InvalidInput(format!("RV64 RAM output remap failed: {e}")))?;
    let logical_ram_init_words = remap_word_map(guest_ram_init_words, &remap)?;
    let output_num_bits = memory_layout.required_num_bits();

    Ok(Some(Rv64RamBridge {
        remap,
        memory_layout,
        logical_output_claims,
        logical_ram_init_words,
        output_num_bits,
    }))
}

pub(crate) fn remap_word_map(
    guest_words: &HashMap<u64, u64>,
    remap: &AffineWordAddressRemap,
) -> Result<HashMap<u64, u64>, PiCcsError> {
    let mut logical = HashMap::with_capacity(guest_words.len());
    for (&guest_addr, &value) in guest_words {
        let logical_addr = remap.remap_guest_addr(guest_addr).map_err(|e| {
            PiCcsError::InvalidInput(format!(
                "RV64 RAM guest word remap failed at guest_addr={guest_addr:#x}: {e}"
            ))
        })?;
        logical.insert(logical_addr, value);
    }
    Ok(logical)
}

pub(crate) fn extend_layout_with_guest_addresses(
    mut layout: RiscvGuestMemoryLayout,
    addrs: impl IntoIterator<Item = u64>,
) -> Result<RiscvGuestMemoryLayout, PiCcsError> {
    let mut uncovered: Vec<u64> = addrs
        .into_iter()
        .filter(|addr| {
            !layout
                .regions
                .iter()
                .any(|region| *addr >= region.guest_base && *addr < region.guest_end())
        })
        .collect();
    if uncovered.is_empty() {
        return Ok(layout);
    }
    uncovered.sort_unstable();
    let mut group_start = uncovered[0];
    let mut group_end = uncovered[0];
    for &addr in uncovered.iter().skip(1) {
        if addr == group_end.saturating_add(layout.cell_bytes) {
            group_end = addr;
            continue;
        }
        layout = add_public_region(layout, group_start, group_end)?;
        group_start = addr;
        group_end = addr;
    }
    add_public_region(layout, group_start, group_end)
}

pub(crate) fn segment_backed_ram_words(loaded_program: &LoadedProgram, xlen: usize) -> HashMap<u64, u64> {
    let mut out = HashMap::new();
    let width = xlen / 8;
    for segment in loaded_program
        .segments
        .iter()
        .filter(|segment| !segment.flags.execute)
    {
        for offset in 0..(segment.mem_size as usize) {
            let mut bytes = [0u8; 8];
            for (i, slot) in bytes.iter_mut().take(width).enumerate() {
                let idx = offset + i;
                if idx < segment.data.len() {
                    *slot = segment.data[idx];
                }
            }
            let value = u64::from_le_bytes(bytes);
            if value == 0 {
                continue;
            }
            let guest_addr = segment.vaddr + offset as u64;
            out.insert(guest_addr, value);
        }
    }
    out
}

fn add_public_region(
    layout: RiscvGuestMemoryLayout,
    start: u64,
    end: u64,
) -> Result<RiscvGuestMemoryLayout, PiCcsError> {
    let region_len = end
        .checked_sub(start)
        .and_then(|span| span.checked_add(layout.cell_bytes))
        .ok_or_else(|| PiCcsError::InvalidInput("guest address span overflow while extending RV64 layout".into()))?;
    layout
        .with_public_region(format!("public-io-{start:#x}"), start, region_len, false, false)
        .map_err(|e| PiCcsError::InvalidInput(format!("failed to extend RV64 guest memory layout: {e}")))
}

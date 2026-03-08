use neo_memory::riscv::lookups::encode_program;
use neo_memory::riscv::lookups::RiscvInstruction;

const ELF_HDR64_SIZE: usize = 64;
const ELF_PHDR64_SIZE: usize = 56;
const PT_LOAD: u32 = 1;

#[derive(Clone)]
pub struct TestSegment {
    pub vaddr: u64,
    pub flags: u32,
    pub mem_size: u64,
    pub data: Vec<u8>,
}

pub fn build_elf64(entry: u64, segments: &[TestSegment]) -> Vec<u8> {
    let phoff = ELF_HDR64_SIZE as u64;
    let mut offset = ELF_HDR64_SIZE + ELF_PHDR64_SIZE * segments.len();
    let mut file = vec![0u8; offset];

    file[0..4].copy_from_slice(&[0x7f, b'E', b'L', b'F']);
    file[4] = 2;
    file[5] = 1;
    file[6] = 1;
    file[7] = 0;
    file[16..18].copy_from_slice(&2u16.to_le_bytes());
    file[18..20].copy_from_slice(&0xF3u16.to_le_bytes());
    file[20..24].copy_from_slice(&1u32.to_le_bytes());
    file[24..32].copy_from_slice(&entry.to_le_bytes());
    file[32..40].copy_from_slice(&phoff.to_le_bytes());
    file[52..54].copy_from_slice(&(ELF_HDR64_SIZE as u16).to_le_bytes());
    file[54..56].copy_from_slice(&(ELF_PHDR64_SIZE as u16).to_le_bytes());
    file[56..58].copy_from_slice(&(segments.len() as u16).to_le_bytes());

    for (idx, segment) in segments.iter().enumerate() {
        let ph = ELF_HDR64_SIZE + idx * ELF_PHDR64_SIZE;
        file[ph..ph + 4].copy_from_slice(&PT_LOAD.to_le_bytes());
        file[ph + 4..ph + 8].copy_from_slice(&segment.flags.to_le_bytes());
        file[ph + 8..ph + 16].copy_from_slice(&(offset as u64).to_le_bytes());
        file[ph + 16..ph + 24].copy_from_slice(&segment.vaddr.to_le_bytes());
        file[ph + 24..ph + 32].copy_from_slice(&segment.vaddr.to_le_bytes());
        file[ph + 32..ph + 40].copy_from_slice(&(segment.data.len() as u64).to_le_bytes());
        file[ph + 40..ph + 48].copy_from_slice(&segment.mem_size.to_le_bytes());
        file[ph + 48..ph + 56].copy_from_slice(&8u64.to_le_bytes());

        file.extend_from_slice(&segment.data);
        offset += segment.data.len();
    }

    file
}

#[allow(dead_code)]
pub fn build_text_elf64(entry: u64, program: &[RiscvInstruction]) -> Vec<u8> {
    let text = encode_program(program);
    build_elf64(
        entry,
        &[TestSegment {
            vaddr: entry,
            flags: 0x5,
            mem_size: text.len() as u64,
            data: text,
        }],
    )
}

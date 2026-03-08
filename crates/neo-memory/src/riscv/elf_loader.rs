//! ELF binary loader for RISC-V programs.
//!
//! This module provides functionality to load ELF binaries and extract
//! RISC-V instructions for execution and proving.

use super::lookups::{decode_instruction_with_xlen, RiscvInstruction};

/// ELF file header constants.
const ELF_MAGIC: [u8; 4] = [0x7f, b'E', b'L', b'F'];
const ELF_CLASS_32: u8 = 1;
const ELF_CLASS_64: u8 = 2;
const ELF_DATA_LE: u8 = 1; // Little-endian
const ELF_MACHINE_RISCV: u16 = 0xF3;

/// Program header types.
const PT_LOAD: u32 = 1;
const PF_X: u32 = 0x1;
const PF_W: u32 = 0x2;
const PF_R: u32 = 0x4;

/// ELF PT_LOAD segment metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ElfSegmentFlags {
    pub read: bool,
    pub write: bool,
    pub execute: bool,
}

impl ElfSegmentFlags {
    #[inline]
    fn from_bits(bits: u32) -> Self {
        Self {
            read: (bits & PF_R) != 0,
            write: (bits & PF_W) != 0,
            execute: (bits & PF_X) != 0,
        }
    }
}

/// PT_LOAD segment extracted from an ELF image.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ElfLoadSegment {
    /// Virtual address where the segment is loaded.
    pub vaddr: u64,
    /// File-backed segment bytes.
    pub data: Vec<u8>,
    /// Declared in-memory segment size (may exceed `data.len()` for BSS).
    pub mem_size: u64,
    /// ELF permission flags.
    pub flags: ElfSegmentFlags,
}

/// A loaded ELF program.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoadedProgram {
    /// Entry point address.
    pub entry: u64,
    /// PT_LOAD segments with permissions and in-memory size.
    pub segments: Vec<ElfLoadSegment>,
    /// Decoded instructions (address, instruction).
    pub instructions: Vec<(u64, RiscvInstruction)>,
    /// Whether this is a 64-bit binary.
    pub is_64bit: bool,
}

/// ELF loading error.
#[derive(Debug)]
pub enum ElfError {
    /// Invalid ELF magic number.
    InvalidMagic,
    /// Unsupported ELF class.
    UnsupportedClass(u8),
    /// Unsupported endianness.
    UnsupportedEndian(u8),
    /// Not a RISC-V binary.
    NotRiscV(u16),
    /// Binary too short.
    TooShort,
    /// Failed to decode instruction.
    DecodeError(String),
}

impl std::fmt::Display for ElfError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ElfError::InvalidMagic => write!(f, "Invalid ELF magic number"),
            ElfError::UnsupportedClass(c) => write!(f, "Unsupported ELF class: {}", c),
            ElfError::UnsupportedEndian(e) => write!(f, "Unsupported endianness: {}", e),
            ElfError::NotRiscV(m) => write!(f, "Not a RISC-V binary (machine: {:#x})", m),
            ElfError::TooShort => write!(f, "Binary too short"),
            ElfError::DecodeError(s) => write!(f, "Decode error: {}", s),
        }
    }
}

impl std::error::Error for ElfError {}

/// Load an ELF binary from bytes.
///
/// # Arguments
/// * `data` - The raw ELF binary data
///
/// # Returns
/// * `Ok(LoadedProgram)` - The loaded program
/// * `Err(ElfError)` - Loading error
pub fn load_elf(data: &[u8]) -> Result<LoadedProgram, ElfError> {
    if data.len() < 52 {
        return Err(ElfError::TooShort);
    }

    // Check magic number
    if data[0..4] != ELF_MAGIC {
        return Err(ElfError::InvalidMagic);
    }

    // Check class (32-bit or 64-bit)
    let class = data[4];
    let is_64bit = match class {
        ELF_CLASS_32 => false,
        ELF_CLASS_64 => true,
        _ => return Err(ElfError::UnsupportedClass(class)),
    };

    // Check endianness (must be little-endian for RISC-V)
    if data[5] != ELF_DATA_LE {
        return Err(ElfError::UnsupportedEndian(data[5]));
    }

    // Check machine type
    let machine = u16::from_le_bytes([data[18], data[19]]);
    if machine != ELF_MACHINE_RISCV {
        return Err(ElfError::NotRiscV(machine));
    }

    // Parse header based on class
    let (entry, phoff, phentsize, phnum) = if is_64bit {
        parse_elf64_header(data)?
    } else {
        parse_elf32_header(data)?
    };

    // Load program segments
    let mut segments = Vec::new();
    for i in 0..phnum {
        let ph_offset = phoff as usize + (i as usize * phentsize as usize);
        let segment = if is_64bit {
            parse_elf64_phdr(data, ph_offset)?
        } else {
            parse_elf32_phdr(data, ph_offset)?
        };

        if let Some(segment) = segment {
            segments.push(segment);
        }
    }

    // Decode instructions from code segments
    let mut instructions = Vec::new();
    for segment in segments.iter().filter(|segment| segment.flags.execute) {
        let vaddr = segment.vaddr;
        let segment_data = &segment.data;
        // Only decode 4-byte aligned segments
        if segment_data.len() >= 4 {
            for offset in (0..segment_data.len()).step_by(4) {
                if offset + 4 <= segment_data.len() {
                    let instr_bytes = [
                        segment_data[offset],
                        segment_data[offset + 1],
                        segment_data[offset + 2],
                        segment_data[offset + 3],
                    ];
                    let instr_word = u32::from_le_bytes(instr_bytes);

                    // Skip zero instructions (padding)
                    if instr_word == 0 {
                        continue;
                    }

                    match decode_instruction_with_xlen(instr_word, if is_64bit { 64 } else { 32 }) {
                        Ok(instr) => {
                            let addr = vaddr + offset as u64;
                            instructions.push((addr, instr));
                        }
                        Err(_) => {
                            // Skip undecodable instructions (could be data)
                        }
                    }
                }
            }
        }
    }

    Ok(LoadedProgram {
        entry,
        segments,
        instructions,
        is_64bit,
    })
}

/// Parse ELF32 header.
fn parse_elf32_header(data: &[u8]) -> Result<(u64, u64, u16, u16), ElfError> {
    if data.len() < 52 {
        return Err(ElfError::TooShort);
    }

    let entry = u32::from_le_bytes([data[24], data[25], data[26], data[27]]) as u64;
    let phoff = u32::from_le_bytes([data[28], data[29], data[30], data[31]]) as u64;
    let phentsize = u16::from_le_bytes([data[42], data[43]]);
    let phnum = u16::from_le_bytes([data[44], data[45]]);

    Ok((entry, phoff, phentsize, phnum))
}

/// Parse ELF64 header.
fn parse_elf64_header(data: &[u8]) -> Result<(u64, u64, u16, u16), ElfError> {
    if data.len() < 64 {
        return Err(ElfError::TooShort);
    }

    let entry = u64::from_le_bytes([
        data[24], data[25], data[26], data[27], data[28], data[29], data[30], data[31],
    ]);
    let phoff = u64::from_le_bytes([
        data[32], data[33], data[34], data[35], data[36], data[37], data[38], data[39],
    ]);
    let phentsize = u16::from_le_bytes([data[54], data[55]]);
    let phnum = u16::from_le_bytes([data[56], data[57]]);

    Ok((entry, phoff, phentsize, phnum))
}

/// Parse ELF32 program header.
fn parse_elf32_phdr(data: &[u8], offset: usize) -> Result<Option<ElfLoadSegment>, ElfError> {
    if offset + 32 > data.len() {
        return Err(ElfError::TooShort);
    }

    let p_type = u32::from_le_bytes([data[offset], data[offset + 1], data[offset + 2], data[offset + 3]]);

    if p_type != PT_LOAD {
        return Ok(None);
    }

    let p_offset =
        u32::from_le_bytes([data[offset + 4], data[offset + 5], data[offset + 6], data[offset + 7]]) as usize;
    let p_vaddr = u32::from_le_bytes([data[offset + 8], data[offset + 9], data[offset + 10], data[offset + 11]]) as u64;
    let p_filesz = u32::from_le_bytes([
        data[offset + 16],
        data[offset + 17],
        data[offset + 18],
        data[offset + 19],
    ]) as usize;
    let p_memsz = u32::from_le_bytes([
        data[offset + 20],
        data[offset + 21],
        data[offset + 22],
        data[offset + 23],
    ]) as u64;
    let p_flags = u32::from_le_bytes([
        data[offset + 24],
        data[offset + 25],
        data[offset + 26],
        data[offset + 27],
    ]);

    if p_offset + p_filesz > data.len() {
        return Err(ElfError::TooShort);
    }

    let segment_data = data[p_offset..p_offset + p_filesz].to_vec();
    Ok(Some(ElfLoadSegment {
        vaddr: p_vaddr,
        data: segment_data,
        mem_size: p_memsz.max(p_filesz as u64),
        flags: ElfSegmentFlags::from_bits(p_flags),
    }))
}

/// Parse ELF64 program header.
fn parse_elf64_phdr(data: &[u8], offset: usize) -> Result<Option<ElfLoadSegment>, ElfError> {
    if offset + 56 > data.len() {
        return Err(ElfError::TooShort);
    }

    let p_type = u32::from_le_bytes([data[offset], data[offset + 1], data[offset + 2], data[offset + 3]]);

    if p_type != PT_LOAD {
        return Ok(None);
    }

    let p_flags = u32::from_le_bytes([data[offset + 4], data[offset + 5], data[offset + 6], data[offset + 7]]);

    let p_offset = u64::from_le_bytes([
        data[offset + 8],
        data[offset + 9],
        data[offset + 10],
        data[offset + 11],
        data[offset + 12],
        data[offset + 13],
        data[offset + 14],
        data[offset + 15],
    ]) as usize;
    let p_vaddr = u64::from_le_bytes([
        data[offset + 16],
        data[offset + 17],
        data[offset + 18],
        data[offset + 19],
        data[offset + 20],
        data[offset + 21],
        data[offset + 22],
        data[offset + 23],
    ]);
    let p_filesz = u64::from_le_bytes([
        data[offset + 32],
        data[offset + 33],
        data[offset + 34],
        data[offset + 35],
        data[offset + 36],
        data[offset + 37],
        data[offset + 38],
        data[offset + 39],
    ]) as usize;
    let p_memsz = u64::from_le_bytes([
        data[offset + 40],
        data[offset + 41],
        data[offset + 42],
        data[offset + 43],
        data[offset + 44],
        data[offset + 45],
        data[offset + 46],
        data[offset + 47],
    ]);

    if p_offset + p_filesz > data.len() {
        return Err(ElfError::TooShort);
    }

    let segment_data = data[p_offset..p_offset + p_filesz].to_vec();
    Ok(Some(ElfLoadSegment {
        vaddr: p_vaddr,
        data: segment_data,
        mem_size: p_memsz.max(p_filesz as u64),
        flags: ElfSegmentFlags::from_bits(p_flags),
    }))
}

/// Load raw binary (not ELF) as RISC-V instructions.
///
/// This is useful for flat binaries without ELF headers.
pub fn load_raw_binary(data: &[u8], base_addr: u64) -> Result<LoadedProgram, ElfError> {
    if data.len() % 4 != 0 {
        return Err(ElfError::TooShort);
    }

    let mut instructions = Vec::new();
    for offset in (0..data.len()).step_by(4) {
        let instr_bytes = [data[offset], data[offset + 1], data[offset + 2], data[offset + 3]];
        let instr_word = u32::from_le_bytes(instr_bytes);

        if instr_word == 0 {
            continue;
        }

        match decode_instruction_with_xlen(instr_word, 32) {
            Ok(instr) => {
                let addr = base_addr + offset as u64;
                instructions.push((addr, instr));
            }
            Err(e) => return Err(ElfError::DecodeError(e)),
        }
    }

    Ok(LoadedProgram {
        entry: base_addr,
        segments: vec![ElfLoadSegment {
            vaddr: base_addr,
            data: data.to_vec(),
            mem_size: data.len() as u64,
            flags: ElfSegmentFlags {
                read: true,
                write: false,
                execute: true,
            },
        }],
        instructions,
        is_64bit: false,
    })
}

impl LoadedProgram {
    /// Get instructions as a vector suitable for RiscvCpu::load_program.
    pub fn get_instructions(&self) -> Vec<RiscvInstruction> {
        self.instructions
            .iter()
            .map(|(_, instr)| instr.clone())
            .collect()
    }

    /// Get the total code size in bytes.
    pub fn code_size(&self) -> usize {
        self.instructions.len() * 4
    }

    /// Returns the PT_LOAD executable segment containing the entrypoint.
    pub fn entry_segment(&self) -> Option<&ElfLoadSegment> {
        self.segments.iter().find(|segment| {
            if !segment.flags.execute {
                return false;
            }
            let end = segment.vaddr.saturating_add(segment.mem_size);
            self.entry >= segment.vaddr && self.entry < end
        })
    }

    /// Reject compressed encodings in executable segments.
    ///
    /// This treats executable segment contents as instruction streams starting at
    /// the segment base and walks them sequentially, consuming 4 bytes for regular
    /// instructions and rejecting any 16-bit compressed encoding.
    pub fn contains_compressed_executable_code(&self) -> bool {
        for segment in self.segments.iter().filter(|segment| segment.flags.execute) {
            let mut offset = 0usize;
            while offset + 2 <= segment.data.len() {
                let half = u16::from_le_bytes([segment.data[offset], segment.data[offset + 1]]);
                if (half & 0b11) != 0b11 {
                    return true;
                }
                offset = offset.saturating_add(4);
            }
        }
        false
    }

    /// Print a disassembly of the program.
    pub fn disassemble(&self) {
        println!("Entry point: {:#x}", self.entry);
        println!("Instructions ({}):", self.instructions.len());
        for (addr, instr) in &self.instructions {
            println!("  {:#010x}: {:?}", addr, instr);
        }
    }
}

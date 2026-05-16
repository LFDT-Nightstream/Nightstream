use neo_fold_prototype::chip8::spec::CommitmentId;
use neo_fold_prototype::chip8::Chip8Opcode;

pub fn render_u8_list(values: &[u8]) -> String {
    if !values.is_empty() && values.iter().all(|&value| value == 0) {
        return format!("(zeroBytes {})", values.len());
    }
    let mut out = String::from("(bytes [");
    for (idx, value) in values.iter().enumerate() {
        if idx > 0 {
            out.push_str(", ");
        }
        out.push_str(&value.to_string());
    }
    out.push_str("])");
    out
}

pub fn render_u64_list(values: &[u64]) -> String {
    let mut out = String::from("[");
    for (idx, value) in values.iter().enumerate() {
        if idx > 0 {
            out.push_str(", ");
        }
        out.push_str(&value.to_string());
    }
    out.push(']');
    out
}

pub fn render_bool(value: bool) -> &'static str {
    if value { "true" } else { "false" }
}

pub fn render_bool_list(values: &[bool]) -> String {
    let mut out = String::from("[");
    for (idx, value) in values.iter().enumerate() {
        if idx > 0 {
            out.push_str(", ");
        }
        out.push_str(render_bool(*value));
    }
    out.push(']');
    out
}

pub fn render_opcode_id(opcode: Chip8Opcode) -> &'static str {
    match opcode {
        Chip8Opcode::LdImm => ".ldImm",
        Chip8Opcode::AddImm => ".addImm",
        Chip8Opcode::Mov => ".mov",
        Chip8Opcode::AddReg => ".addReg",
        Chip8Opcode::SkipEqImm => ".skipEqImm",
        Chip8Opcode::Jump => ".jump",
        Chip8Opcode::LdI => ".ldI",
        Chip8Opcode::StoreRegs => ".storeRegs",
        Chip8Opcode::LoadRegs => ".loadRegs",
    }
}

pub fn render_commitment_id(id: CommitmentId) -> String {
    match id {
        CommitmentId::Lane => ".lane".into(),
        CommitmentId::FetchRa => ".fetchRa".into(),
        CommitmentId::DecodeRa => ".decodeRa".into(),
        CommitmentId::AluRa => ".aluRa".into(),
        CommitmentId::Eq4Ra => ".eq4Ra".into(),
        CommitmentId::DecodeHandoff => ".decodeHandoff".into(),
        CommitmentId::RegTwist => ".regTwist".into(),
        CommitmentId::RamTwist => ".ramTwist".into(),
        CommitmentId::RomTable => ".romTable".into(),
        CommitmentId::DecodeTable => ".decodeTable".into(),
        CommitmentId::AluTable => ".aluTable".into(),
        CommitmentId::Eq4Table => ".eq4Table".into(),
        CommitmentId::RootProver(tag) => format!("(.rootProver {tag})"),
    }
}

pub fn lean_ident_fragment(name: &str) -> String {
    name.chars()
        .map(|ch| if ch.is_ascii_alphanumeric() { ch } else { '_' })
        .collect()
}

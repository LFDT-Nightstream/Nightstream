mod accepted_artifact_vectors;

use std::fs;
use std::path::{Path, PathBuf};

use neo_fold_prototype::rv64im as rv64;
use neo_fold_prototype::rv64im::{
    build_all_parity_cases, build_rv64im_audit_witness_bundle, prove_rv64im_public_proof,
    verify_rv64im_public_proof, MemoryWord, Rv64imKernelSummary, Rv64imParityCaseManifest,
    Rv64imParityDerivedCase, Rv64imParitySourceCase, Rv64TraceVirtualOpcode, TranscriptCursorSnapshot,
    TranscriptEventKind, TranscriptEventRecord, TranscriptRecord,
};
use neo_fold_prototype::rv64im::kernel::{
    RootLaneCommitmentSetSummary, RootLaneCommitmentSummaryArtifact,
};

use accepted_artifact_vectors::{
    accepted_proof_artifact_dir, accepted_proof_case_path, build_accepted_proof_cases,
    render_accepted_artifact_case_module, render_accepted_proof_corpus_module,
};

#[derive(Clone)]
pub(crate) struct PublicProofVectorCase {
    name: String,
    proof: rv64::Rv64imProof,
}

fn render_u8_list(values: &[u8]) -> String {
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

fn render_u64_list(values: &[u64]) -> String {
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

fn render_fold_schedule(schedule: &neo_fold_prototype::proof::FoldSchedule) -> String {
    match schedule {
        neo_fold_prototype::proof::FoldSchedule::WholeTrace => "Nightstream.FoldSchedule.wholeTrace".into(),
        neo_fold_prototype::proof::FoldSchedule::RowsPerChunk(rows) => {
            format!("Nightstream.FoldSchedule.rowsPerChunk {rows}")
        }
    }
}

fn lean_ident_fragment(name: &str) -> String {
    name.chars()
        .map(|ch| if ch.is_ascii_alphanumeric() { ch } else { '_' })
        .collect()
}

fn generated_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("Nightstream")
        .join("Rv64IM")
        .join("Generated")
}

fn public_proof_vector_dir() -> PathBuf {
    generated_dir().join("PublicProofVectors")
}

fn case_dir(case_name: &str) -> PathBuf {
    generated_dir()
        .join("Cases")
        .join(format!("Case_{}", lean_ident_fragment(case_name)))
}

fn public_proof_case_path(case_name: &str) -> PathBuf {
    public_proof_vector_dir().join(format!("Case_{}.lean", lean_ident_fragment(case_name)))
}

fn render_string(value: &str) -> String {
    format!("{value:?}")
}

fn render_family_tag(tag: neo_fold_prototype::rv64im::tables::Rv64FamilyTag) -> &'static str {
    match tag {
        neo_fold_prototype::rv64im::tables::Rv64FamilyTag::NativeAlu => ".nativeAlu",
        neo_fold_prototype::rv64im::tables::Rv64FamilyTag::AlignedMemory => ".alignedMemory",
        neo_fold_prototype::rv64im::tables::Rv64FamilyTag::NarrowMemory => ".narrowMemory",
        neo_fold_prototype::rv64im::tables::Rv64FamilyTag::Multiply => ".multiply",
        neo_fold_prototype::rv64im::tables::Rv64FamilyTag::UnsignedDivRem => ".unsignedDivRem",
        neo_fold_prototype::rv64im::tables::Rv64FamilyTag::SignedDivRem => ".signedDivRem",
        neo_fold_prototype::rv64im::tables::Rv64FamilyTag::ControlFlow => ".controlFlow",
    }
}

fn family_module_name(tag: neo_fold_prototype::rv64im::tables::Rv64FamilyTag) -> &'static str {
    match tag {
        neo_fold_prototype::rv64im::tables::Rv64FamilyTag::NativeAlu => "NativeAlu",
        neo_fold_prototype::rv64im::tables::Rv64FamilyTag::AlignedMemory => "AlignedMemory",
        neo_fold_prototype::rv64im::tables::Rv64FamilyTag::NarrowMemory => "NarrowMemory",
        neo_fold_prototype::rv64im::tables::Rv64FamilyTag::Multiply => "Multiply",
        neo_fold_prototype::rv64im::tables::Rv64FamilyTag::UnsignedDivRem => "UnsignedDivRem",
        neo_fold_prototype::rv64im::tables::Rv64FamilyTag::SignedDivRem => "SignedDivRem",
        neo_fold_prototype::rv64im::tables::Rv64FamilyTag::ControlFlow => "ControlFlow",
    }
}

fn render_opcode(opcode: neo_fold_prototype::rv64im::Rv64Opcode) -> &'static str {
    match opcode {
        neo_fold_prototype::rv64im::Rv64Opcode::Addi => ".addi",
        neo_fold_prototype::rv64im::Rv64Opcode::Add => ".add",
        neo_fold_prototype::rv64im::Rv64Opcode::Sub => ".sub",
        neo_fold_prototype::rv64im::Rv64Opcode::Addiw => ".addiw",
        neo_fold_prototype::rv64im::Rv64Opcode::Addw => ".addw",
        neo_fold_prototype::rv64im::Rv64Opcode::Subw => ".subw",
        neo_fold_prototype::rv64im::Rv64Opcode::Andi => ".andi",
        neo_fold_prototype::rv64im::Rv64Opcode::And => ".and",
        neo_fold_prototype::rv64im::Rv64Opcode::Ori => ".ori",
        neo_fold_prototype::rv64im::Rv64Opcode::Or => ".or",
        neo_fold_prototype::rv64im::Rv64Opcode::Xori => ".xori",
        neo_fold_prototype::rv64im::Rv64Opcode::Xor => ".xor",
        neo_fold_prototype::rv64im::Rv64Opcode::Slti => ".slti",
        neo_fold_prototype::rv64im::Rv64Opcode::Slt => ".slt",
        neo_fold_prototype::rv64im::Rv64Opcode::Sltiu => ".sltiu",
        neo_fold_prototype::rv64im::Rv64Opcode::Sltu => ".sltu",
        neo_fold_prototype::rv64im::Rv64Opcode::Slli => ".slli",
        neo_fold_prototype::rv64im::Rv64Opcode::Sll => ".sll",
        neo_fold_prototype::rv64im::Rv64Opcode::Srli => ".srli",
        neo_fold_prototype::rv64im::Rv64Opcode::Srl => ".srl",
        neo_fold_prototype::rv64im::Rv64Opcode::Srai => ".srai",
        neo_fold_prototype::rv64im::Rv64Opcode::Sra => ".sra",
        neo_fold_prototype::rv64im::Rv64Opcode::Slliw => ".slliw",
        neo_fold_prototype::rv64im::Rv64Opcode::Sllw => ".sllw",
        neo_fold_prototype::rv64im::Rv64Opcode::Srliw => ".srliw",
        neo_fold_prototype::rv64im::Rv64Opcode::Srlw => ".srlw",
        neo_fold_prototype::rv64im::Rv64Opcode::Sraiw => ".sraiw",
        neo_fold_prototype::rv64im::Rv64Opcode::Sraw => ".sraw",
        neo_fold_prototype::rv64im::Rv64Opcode::Lui => ".lui",
        neo_fold_prototype::rv64im::Rv64Opcode::Auipc => ".auipc",
        neo_fold_prototype::rv64im::Rv64Opcode::Fence => ".fence",
        neo_fold_prototype::rv64im::Rv64Opcode::Mul => ".mul",
        neo_fold_prototype::rv64im::Rv64Opcode::Mulh => ".mulh",
        neo_fold_prototype::rv64im::Rv64Opcode::Mulhsu => ".mulhsu",
        neo_fold_prototype::rv64im::Rv64Opcode::Mulhu => ".mulhu",
        neo_fold_prototype::rv64im::Rv64Opcode::Mulw => ".mulw",
        neo_fold_prototype::rv64im::Rv64Opcode::Div => ".div",
        neo_fold_prototype::rv64im::Rv64Opcode::Divu => ".divu",
        neo_fold_prototype::rv64im::Rv64Opcode::Rem => ".rem",
        neo_fold_prototype::rv64im::Rv64Opcode::Remu => ".remu",
        neo_fold_prototype::rv64im::Rv64Opcode::Divw => ".divw",
        neo_fold_prototype::rv64im::Rv64Opcode::Divuw => ".divuw",
        neo_fold_prototype::rv64im::Rv64Opcode::Remw => ".remw",
        neo_fold_prototype::rv64im::Rv64Opcode::Remuw => ".remuw",
        neo_fold_prototype::rv64im::Rv64Opcode::Lb => ".lb",
        neo_fold_prototype::rv64im::Rv64Opcode::Lbu => ".lbu",
        neo_fold_prototype::rv64im::Rv64Opcode::Lh => ".lh",
        neo_fold_prototype::rv64im::Rv64Opcode::Lhu => ".lhu",
        neo_fold_prototype::rv64im::Rv64Opcode::Lw => ".lw",
        neo_fold_prototype::rv64im::Rv64Opcode::Lwu => ".lwu",
        neo_fold_prototype::rv64im::Rv64Opcode::Ld => ".ld",
        neo_fold_prototype::rv64im::Rv64Opcode::Sb => ".sb",
        neo_fold_prototype::rv64im::Rv64Opcode::Sh => ".sh",
        neo_fold_prototype::rv64im::Rv64Opcode::Sw => ".sw",
        neo_fold_prototype::rv64im::Rv64Opcode::Sd => ".sd",
        neo_fold_prototype::rv64im::Rv64Opcode::Jal => ".jal",
        neo_fold_prototype::rv64im::Rv64Opcode::Jalr => ".jalr",
        neo_fold_prototype::rv64im::Rv64Opcode::Beq => ".beq",
        neo_fold_prototype::rv64im::Rv64Opcode::Bne => ".bne",
        neo_fold_prototype::rv64im::Rv64Opcode::Blt => ".blt",
        neo_fold_prototype::rv64im::Rv64Opcode::Bge => ".bge",
        neo_fold_prototype::rv64im::Rv64Opcode::Bltu => ".bltu",
        neo_fold_prototype::rv64im::Rv64Opcode::Bgeu => ".bgeu",
        neo_fold_prototype::rv64im::Rv64Opcode::Ecall => ".ecall",
    }
}

fn render_register_read_role(role: neo_fold_prototype::rv64im::stage2::RegisterReadRole) -> &'static str {
    match role {
        neo_fold_prototype::rv64im::stage2::RegisterReadRole::Rs1 => ".rs1",
        neo_fold_prototype::rv64im::stage2::RegisterReadRole::Rs2 => ".rs2",
    }
}

fn render_ram_access_kind(kind: neo_fold_prototype::rv64im::stage2::RamAccessKind) -> &'static str {
    match kind {
        neo_fold_prototype::rv64im::stage2::RamAccessKind::Read => ".read",
        neo_fold_prototype::rv64im::stage2::RamAccessKind::Write => ".write",
    }
}

fn render_transcript_event_kind(kind: TranscriptEventKind) -> &'static str {
    match kind {
        TranscriptEventKind::AppendMessage => ".appendMessage",
        TranscriptEventKind::AppendU64s => ".appendU64s",
        TranscriptEventKind::ChallengeField => ".challengeField",
        TranscriptEventKind::Digest32 => ".digest32",
    }
}

fn render_trace_virtual_opcode(opcode: Rv64TraceVirtualOpcode) -> &'static str {
    match opcode {
        Rv64TraceVirtualOpcode::Movsign => ".movsign",
        Rv64TraceVirtualOpcode::Advice => ".advice",
        Rv64TraceVirtualOpcode::ChangeDivisor => ".changeDivisor",
        Rv64TraceVirtualOpcode::AssertValidDiv0 => ".assertValidDiv0",
        Rv64TraceVirtualOpcode::AssertMulNoOverflow => ".assertMulNoOverflow",
        Rv64TraceVirtualOpcode::AssertLte => ".assertLte",
        Rv64TraceVirtualOpcode::AssertValidUnsignedRemainder => ".assertValidUnsignedRemainder",
        Rv64TraceVirtualOpcode::AssertSignedDivIdentity => ".assertSignedDivIdentity",
        Rv64TraceVirtualOpcode::AssertSignedRemainderBounds => ".assertSignedRemainderBounds",
        Rv64TraceVirtualOpcode::Move => ".move",
        Rv64TraceVirtualOpcode::SignExtendWord => ".signExtendWord",
    }
}

fn render_option_nat(value: Option<u64>) -> String {
    match value {
        Some(value) => format!("(some {value})"),
        None => "none".into(),
    }
}

fn render_option_bytes(value: Option<&[u8]>) -> String {
    match value {
        Some(bytes) => format!("(some {})", render_u8_list(bytes)),
        None => "none".into(),
    }
}

fn render_ajtai_family_kind(kind: rv64::AjtaiFamilyKind) -> u64 {
    match kind {
        rv64::AjtaiFamilyKind::RootMainLaneColumns => 0,
        rv64::AjtaiFamilyKind::RootMainLaneCommittedRows => 10,
        rv64::AjtaiFamilyKind::Stage1Rows => 1,
        rv64::AjtaiFamilyKind::Stage2RegisterReads => 2,
        rv64::AjtaiFamilyKind::Stage2RegisterWrites => 3,
        rv64::AjtaiFamilyKind::Stage2RamEvents => 4,
        rv64::AjtaiFamilyKind::Stage2TwistLinks => 5,
        rv64::AjtaiFamilyKind::Stage3Continuity => 6,
        rv64::AjtaiFamilyKind::KernelBindings => 7,
        rv64::AjtaiFamilyKind::KernelPreparedSteps => 8,
        rv64::AjtaiFamilyKind::RootMainLanePublicSteps => 9,
    }
}

fn render_ajtai_object_id(object: &rv64::AjtaiObjectId) -> String {
    format!(
        "{{ familyTag := {}, commitmentDigest := {}, layoutVersion := {}, digest := {} }}",
        render_ajtai_family_kind(object.family),
        render_u8_list(&object.commitment_digest),
        object.layout_version,
        render_u8_list(&object.digest),
    )
}

fn render_ajtai_opening_id(opening: &rv64::AjtaiOpeningId) -> String {
    format!(
        "{{ object := {}, logicalIndex := {}, digest := {} }}",
        render_ajtai_object_id(&opening.object),
        opening.logical_index,
        render_u8_list(&opening.digest),
    )
}

fn render_selected_opening_ref(reference: &rv64::SelectedOpeningRef) -> String {
    format!(
        "{{ id := {}, valueDigest := {}, digest := {} }}",
        render_ajtai_opening_id(&reference.id),
        render_u8_list(&reference.value_digest),
        render_u8_list(&reference.digest),
    )
}

fn render_option_selected_opening_ref(reference: Option<&rv64::SelectedOpeningRef>) -> String {
    match reference {
        Some(reference) => format!("(some {})", render_selected_opening_ref(reference)),
        None => "none".into(),
    }
}

fn render_u8_matrix(values: &[[u8; 32]]) -> String {
    let mut out = String::from("[");
    for (idx, value) in values.iter().enumerate() {
        if idx > 0 {
            out.push_str(", ");
        }
        out.push_str(&render_u8_list(value));
    }
    out.push(']');
    out
}

fn render_bool(value: bool) -> &'static str {
    if value { "true" } else { "false" }
}

fn render_memory_word(word: &MemoryWord) -> String {
    format!("{{ addr := {}, value := {} }}", word.addr, word.value)
}

fn render_memory_words(words: &[MemoryWord]) -> String {
    let mut out = String::from("[");
    for (idx, word) in words.iter().enumerate() {
        if idx > 0 {
            out.push_str(", ");
        }
        out.push_str(&render_memory_word(word));
    }
    out.push(']');
    out
}

fn render_manifest(manifest: &Rv64imParityCaseManifest) -> String {
    let mut family_tags = String::from("[");
    for (idx, tag) in manifest.family_tags.iter().enumerate() {
        if idx > 0 {
            family_tags.push_str(", ");
        }
        family_tags.push_str(render_family_tag(*tag));
    }
    family_tags.push(']');
    format!(
        "{{ name := {}, fixtureId := {}, protocolVersionId := {}, loweringVersionId := {}, familyTags := {} }}",
        render_string(&manifest.name),
        render_string(&manifest.fixture_id),
        manifest.protocol_version_id,
        manifest.lowering_version_id,
        family_tags
    )
}

pub(crate) fn render_source_case(case: &Rv64imParitySourceCase) -> String {
    format!(
        "{{\n  manifest := {}\n  , startPc := {}\n  , programWords := {}\n  , initialRegisters := {}\n  , initialMemory := {}\n  , transcriptSeed := {}\n}}",
        render_manifest(&case.manifest),
        case.start_pc,
        render_u64_list(&case.program_words.iter().map(|word| *word as u64).collect::<Vec<_>>()),
        render_u64_list(&case.initial_registers),
        render_memory_words(&case.initial_memory),
        render_u8_list(&case.transcript_seed),
    )
}

fn render_expanded_row(row: &neo_fold_prototype::rv64im::lower::Rv64ExpandedRow) -> String {
    format!(
        "{{\n  traceIndex := {}\n  , stepIndex := {}\n  , sequenceIndex := {}\n  , pc := {}\n  , nextPc := {}\n  , word := {}\n  , opcode := {}\n  , traceOpcode := {}\n  , traceVirtualOpcode := {}\n  , family := {}\n  , rs1 := {}\n  , rs1Value := {}\n  , rs2 := {}\n  , rs2Value := {}\n  , rd := {}\n  , rdBefore := {}\n  , rdAfter := {}\n  , imm := {}\n  , aluResult := {}\n  , effectiveAddr := {}\n  , memoryBefore := {}\n  , memoryAfter := {}\n  , writesRd := {}\n  , writesRam := {}\n  , halted := {}\n  , isFirstInSequence := {}\n  , virtualSequenceRemaining := {}\n  , isEffectRow := {}\n  , isCommitRow := {}\n  , isReal := {}\n}}",
        row.trace_index,
        row.step_index,
        row.sequence_index,
        row.pc,
        row.next_pc,
        row.word,
        render_opcode(row.opcode),
        row.trace_opcode
            .map(|opcode| format!("(some {})", render_opcode(opcode)))
            .unwrap_or_else(|| "none".into()),
        row.trace_virtual_opcode
            .map(|opcode| format!("(some {})", render_trace_virtual_opcode(opcode)))
            .unwrap_or_else(|| "none".into()),
        render_family_tag(row.family),
        row.rs1,
        row.rs1_value,
        row.rs2,
        row.rs2_value,
        row.rd,
        row.rd_before,
        row.rd_after,
        row.imm,
        row.alu_result,
        render_option_nat(row.effective_addr),
        render_option_nat(row.memory_before),
        render_option_nat(row.memory_after),
        if row.writes_rd { "true" } else { "false" },
        if row.writes_ram { "true" } else { "false" },
        if row.halted { "true" } else { "false" },
        if row.is_first_in_sequence { "true" } else { "false" },
        row.virtual_sequence_remaining
            .map(|remaining| format!("(some {remaining})"))
            .unwrap_or_else(|| "none".into()),
        if row.is_effect_row { "true" } else { "false" },
        if row.is_commit_row { "true" } else { "false" },
        if row.is_real { "true" } else { "false" },
    )
}

fn render_stage1_summary(stage1: &neo_fold_prototype::rv64im::stage1::Stage1Summary) -> String {
    let mut rows = String::from("[");
    for (idx, row) in stage1.rows.iter().enumerate() {
        if idx > 0 {
            rows.push_str(", ");
        }
        rows.push_str(&format!(
            "{{ traceIndex := {}, stepIndex := {}, sequenceIndex := {}, fetchPc := {}, fetchedWord := {}, opcode := {}, traceOpcode := {}, traceVirtualOpcode := {}, family := {}, nextPc := {}, aluResult := {}, effectiveAddr := {}, writesRd := {}, rd := {}, rdAfter := {}, isFirstInSequence := {}, virtualSequenceRemaining := {}, isEffectRow := {}, isCommitRow := {}, isReal := {}, preservesX0 := {} }}",
            row.trace_index,
            row.step_index,
            row.sequence_index,
            row.fetch_pc,
            row.fetched_word,
            render_opcode(row.opcode),
            row.trace_opcode
                .map(|opcode| format!("(some {})", render_opcode(opcode)))
                .unwrap_or_else(|| "none".into()),
            row.trace_virtual_opcode
                .map(|opcode| format!("(some {})", render_trace_virtual_opcode(opcode)))
                .unwrap_or_else(|| "none".into()),
            render_family_tag(row.family),
            row.next_pc,
            row.alu_result,
            render_option_nat(row.effective_addr),
            if row.writes_rd { "true" } else { "false" },
            row.rd,
            row.rd_after,
            if row.is_first_in_sequence { "true" } else { "false" },
            row.virtual_sequence_remaining
                .map(|remaining| format!("(some {remaining})"))
                .unwrap_or_else(|| "none".into()),
            if row.is_effect_row { "true" } else { "false" },
            if row.is_commit_row { "true" } else { "false" },
            if row.is_real { "true" } else { "false" },
            if row.preserves_x0 { "true" } else { "false" },
        ));
    }
    rows.push(']');
    format!("{{ rows := {} }}", rows)
}

fn render_stage2_summary(stage2: &neo_fold_prototype::rv64im::stage2::Stage2Summary) -> String {
    let mut register_reads = String::from("[");
    for (idx, event) in stage2.register_reads.iter().enumerate() {
        if idx > 0 {
            register_reads.push_str(", ");
        }
        register_reads.push_str(&format!(
            "{{ traceIndex := {}, stepIndex := {}, role := {}, reg := {}, value := {} }}",
            event.trace_index,
            event.step_index,
            render_register_read_role(event.role),
            event.reg,
            event.value,
        ));
    }
    register_reads.push(']');

    let mut register_writes = String::from("[");
    for (idx, event) in stage2.register_writes.iter().enumerate() {
        if idx > 0 {
            register_writes.push_str(", ");
        }
        register_writes.push_str(&format!(
            "{{ traceIndex := {}, stepIndex := {}, reg := {}, previous := {}, next := {} }}",
            event.trace_index,
            event.step_index,
            event.reg,
            event.previous,
            event.next,
        ));
    }
    register_writes.push(']');

    let mut ram_events = String::from("[");
    for (idx, event) in stage2.ram_events.iter().enumerate() {
        if idx > 0 {
            ram_events.push_str(", ");
        }
        ram_events.push_str(&format!(
            "{{ traceIndex := {}, stepIndex := {}, kind := {}, addr := {}, previous := {}, next := {} }}",
            event.trace_index,
            event.step_index,
            render_ram_access_kind(event.kind),
            event.addr,
            event.previous,
            event.next,
        ));
    }
    ram_events.push(']');

    let mut twist_links = String::from("[");
    for (idx, event) in stage2.twist_links.iter().enumerate() {
        if idx > 0 {
            twist_links.push_str(", ");
        }
        twist_links.push_str(&format!(
            "{{ traceIndex := {}, stepIndex := {}, family := {}, routedWriteValue := {}, routedMemoryBefore := {}, routedMemoryAfter := {} }}",
            event.trace_index,
            event.step_index,
            render_family_tag(event.family),
            render_option_nat(event.routed_write_value),
            render_option_nat(event.routed_memory_before),
            render_option_nat(event.routed_memory_after),
        ));
    }
    twist_links.push(']');

    format!(
        "{{\n  registerReads := {}\n  , registerWrites := {}\n  , ramEvents := {}\n  , twistLinks := {}\n}}",
        register_reads, register_writes, ram_events, twist_links
    )
}

fn render_stage3_summary(stage3: &neo_fold_prototype::rv64im::stage3::Stage3Summary) -> String {
    let mut continuity = String::from("[");
    for (idx, event) in stage3.continuity.iter().enumerate() {
        if idx > 0 {
            continuity.push_str(", ");
        }
        continuity.push_str(&format!(
            "{{ stepIndex := {}, pc := {}, nextPc := {}, successorPc := {}, finalStep := {}, continuityHolds := {} }}",
            event.step_index,
            event.pc,
            event.next_pc,
            render_option_nat(event.successor_pc),
            if event.final_step { "true" } else { "false" },
            if event.continuity_holds { "true" } else { "false" },
        ));
    }
    continuity.push(']');
    format!(
        "{{\n  continuity := {}\n  , halted := {}\n}}",
        continuity,
        if stage3.halted { "true" } else { "false" }
    )
}

fn render_cursor_snapshot(snapshot: &TranscriptCursorSnapshot) -> String {
    format!(
        "{{ stateWords := {}, absorbed := {} }}",
        render_u64_list(&snapshot.state_words),
        snapshot.absorbed
    )
}

fn render_transcript_event(event: &TranscriptEventRecord) -> String {
    format!(
        "{{\n  kind := {}\n  , label := {}\n  , message := {}\n  , u64s := {}\n  , cursorBefore := {}\n  , cursorAfter := {}\n  , challengeOutput := {}\n  , digestOutput := {}\n}}",
        render_transcript_event_kind(event.kind),
        render_u8_list(&event.label),
        render_u8_list(&event.message),
        render_u64_list(&event.u64s),
        render_cursor_snapshot(&event.cursor_before),
        render_cursor_snapshot(&event.cursor_after),
        match event.challenge_output {
            Some(value) => format!("(some {value})"),
            None => "none".into(),
        },
        render_option_bytes(event.digest_output.as_ref().map(|digest| digest.as_slice())),
    )
}

fn render_transcript(record: &TranscriptRecord) -> String {
    let mut events = String::from("[");
    for (idx, event) in record.events.iter().enumerate() {
        if idx > 0 {
            events.push_str(", ");
        }
        events.push_str(&render_transcript_event(event));
    }
    events.push(']');
    format!(
        "{{\n  appLabel := {}\n  , events := {}\n}}",
        render_u8_list(&record.app_label),
        events
    )
}

fn render_kernel_summary(summary: &Rv64imKernelSummary) -> String {
    format!(
        "{{\n  root0Digest := {}\n  , stage1Digest := {}\n  , stage2Digest := {}\n  , stage3Digest := {}\n  , executionDigest := {}\n  , finalStateDigest := {}\n  , stage1Mix := {}\n  , stage2RegMix := {}\n  , stage2RamMix := {}\n  , stage3ContinuityMix := {}\n  , kernelFinalMix := {}\n  , transcriptFinalDigest := {}\n  , finalPc := {}\n  , finalRegisters := {}\n  , finalMemory := {}\n  , halted := {}\n}}",
        render_u8_list(&summary.root0_digest),
        render_u8_list(&summary.stage1_digest),
        render_u8_list(&summary.stage2_digest),
        render_u8_list(&summary.stage3_digest),
        render_u8_list(&summary.execution_digest),
        render_u8_list(&summary.final_state_digest),
        summary.stage1_mix,
        summary.stage2_reg_mix,
        summary.stage2_ram_mix,
        summary.stage3_continuity_mix,
        summary.kernel_final_mix,
        render_u8_list(&summary.transcript_final_digest),
        summary.final_pc,
        render_u64_list(&summary.final_registers),
        render_memory_words(&summary.final_memory),
        if summary.halted { "true" } else { "false" },
    )
}

pub(crate) fn render_derived_case(case: &Rv64imParityDerivedCase) -> String {
    let mut rows = String::from("[");
    for (idx, row) in case.execution_rows.iter().enumerate() {
        if idx > 0 {
            rows.push_str(", ");
        }
        rows.push_str(&render_expanded_row(row));
    }
    rows.push(']');

    format!(
        "{{\n  manifest := {}\n  , executionRows := {}\n  , stage1 := {}\n  , stage2 := {}\n  , stage3 := {}\n  , transcript := {}\n  , kernel := {}\n}}",
        render_manifest(&case.manifest),
        rows,
        render_stage1_summary(&case.stage1),
        render_stage2_summary(&case.stage2),
        render_stage3_summary(&case.stage3),
        render_transcript(&case.transcript),
        render_kernel_summary(&case.kernel),
    )
}

fn render_proof_statement(statement: &rv64::Rv64imProofStatement) -> String {
    format!(
        "{{\n  rootParamsId := {}\n  , foldSchedule := {}\n  , chunkCount := {}\n  , stageClaimsDigest := {}\n  , stagePackagesDigest := {}\n  , kernelOpeningDigest := {}\n  , preparedStepBindingsDigest := {}\n  , executionDigest := {}\n  , finalStateDigest := {}\n  , transcriptFinalDigest := {}\n  , mainLaneSurfaceDigest := {}\n  , rootLaneColumnsDigest := {}\n  , publicStepCount := {}\n  , initialPc := {}\n  , finalPc := {}\n  , halted := {}\n  , digest := {}\n}}",
        render_u8_list(&statement.root_params_id),
        render_fold_schedule(&statement.fold_schedule),
        statement.chunk_count,
        render_u8_list(&statement.stage_claims_digest),
        render_u8_list(&statement.stage_packages_digest),
        render_u8_list(&statement.kernel_opening_digest),
        render_u8_list(&statement.prepared_step_bindings_digest),
        render_u8_list(&statement.execution_digest),
        render_u8_list(&statement.final_state_digest),
        render_u8_list(&statement.transcript_final_digest),
        render_u8_list(&statement.main_lane_surface_digest),
        render_u8_list(&statement.root_lane_columns_digest),
        statement.public_step_count,
        statement.initial_pc,
        statement.final_pc,
        render_bool(statement.halted),
        render_u8_list(&statement.digest),
    )
}

fn render_accepted_proof_statement_binding(binding: &rv64::Rv64imAcceptedProofStatementBinding) -> String {
    format!(
        "{{ proofStatementDigest := {}, kernelOpeningDigest := {}, digest := {} }}",
        render_u8_list(&binding.proof_statement_digest),
        render_u8_list(&binding.kernel_opening_digest),
        render_u8_list(&binding.digest),
    )
}

fn render_accepted_proof_main_lane_binding(binding: &rv64::Rv64imAcceptedProofMainLaneBinding) -> String {
    format!(
        "{{ mainLaneBundleDigest := {}, digest := {} }}",
        render_u8_list(&binding.main_lane_bundle_digest),
        render_u8_list(&binding.digest),
    )
}

fn render_accepted_proof_terminal_binding(binding: &rv64::Rv64imAcceptedProofTerminalBinding) -> String {
    format!(
        "{{ finalStateDigest := {}, finalPc := {}, halted := {}, digest := {} }}",
        render_u8_list(&binding.final_state_digest),
        binding.final_pc,
        render_bool(binding.halted),
        render_u8_list(&binding.digest),
    )
}

fn render_accepted_proof_claim(claim: &rv64::Rv64imAcceptedProofClaim) -> String {
    format!(
        "{{\n  rootParamsId := {}\n  , statement := {}\n  , mainLane := {}\n  , terminal := {}\n  , digest := {}\n}}",
        render_u8_list(&claim.root_params_id),
        render_accepted_proof_statement_binding(&claim.statement),
        render_accepted_proof_main_lane_binding(&claim.main_lane),
        render_accepted_proof_terminal_binding(&claim.terminal),
        render_u8_list(&claim.digest),
    )
}

fn render_main_lane_claim_binding(binding: &rv64::Rv64imMainLaneClaimBinding) -> String {
    format!(
        "{{ mainLaneBundleDigest := {}, digest := {} }}",
        render_u8_list(&binding.main_lane_bundle_digest),
        render_u8_list(&binding.digest),
    )
}

fn render_root_lane_columns(columns: &rv64::RootLaneColumns) -> String {
    format!(
        "{{ object := {}, rowWidth := {}, timeLen := {}, columnDigests := {}, familyDigest := {}, firstRow := {}, lastRow := {}, digest := {} }}",
        render_ajtai_object_id(&columns.object),
        columns.row_width,
        columns.time_len,
        render_u8_matrix(&columns.column_digests),
        render_u8_list(&columns.family_digest),
        render_option_selected_opening_ref(columns.first_row.as_ref()),
        render_option_selected_opening_ref(columns.last_row.as_ref()),
        render_u8_list(&columns.digest),
    )
}

fn render_root_lane_commitment_set(set: &RootLaneCommitmentSetSummary) -> String {
    format!(
        "{{ commitmentCount := {}, digest := {} }}",
        set.commitment_count,
        render_u8_list(&set.digest),
    )
}

fn render_root_lane_commitment_artifact(artifact: &RootLaneCommitmentSummaryArtifact) -> String {
    format!(
        "{{ timeLen := {}, commitments := {}, firstSelectedRow := {}, lastSelectedRow := {}, digest := {} }}",
        artifact.time_len,
        render_root_lane_commitment_set(&artifact.commitments),
        render_option_selected_opening_ref(artifact.first_selected_row.as_ref()),
        render_option_selected_opening_ref(artifact.last_selected_row.as_ref()),
        render_u8_list(&artifact.digest),
    )
}

fn render_main_lane_claim(claim: &rv64::Rv64imMainLaneClaim) -> String {
    format!(
        "{{ rootParamsId := {}, binding := {}, digest := {} }}",
        render_u8_list(&claim.root_params_id),
        render_main_lane_claim_binding(&claim.binding),
        render_u8_list(&claim.digest),
    )
}

fn render_kernel_opening_claim(claim: &rv64::Rv64imKernelOpeningClaim) -> String {
    format!(
        "{{\n  rootParamsId := {}\n  , stages := {{ stageClaimsDigest := {}, stagePackagesDigest := {}, kernelOpeningDigest := {}, digest := {} }}\n  , terminal := {{ preparedStepBindingsDigest := {}, executionDigest := {}, transcriptFinalDigest := {}, digest := {} }}\n  , digest := {}\n}}",
        render_u8_list(&claim.root_params_id),
        render_u8_list(&claim.stages.stage_claims_digest),
        render_u8_list(&claim.stages.stage_packages_digest),
        render_u8_list(&claim.stages.kernel_opening_digest),
        render_u8_list(&claim.stages.digest),
        render_u8_list(&claim.terminal.prepared_step_bindings_digest),
        render_u8_list(&claim.terminal.execution_digest),
        render_u8_list(&claim.terminal.transcript_final_digest),
        render_u8_list(&claim.terminal.digest),
        render_u8_list(&claim.digest),
    )
}

fn render_joint_opening_claim(claim: &rv64::Rv64imJointOpeningClaim) -> String {
    format!(
        "{{ rootParamsId := {}, binding := {{ proofStatementDigest := {}, mainLaneClaimDigest := {}, kernelOpeningClaimDigest := {}, digest := {} }}, digest := {} }}",
        render_u8_list(&claim.root_params_id),
        render_u8_list(&claim.binding.proof_statement_digest),
        render_u8_list(&claim.binding.main_lane_claim_digest),
        render_u8_list(&claim.binding.kernel_opening_claim_digest),
        render_u8_list(&claim.binding.digest),
        render_u8_list(&claim.digest),
    )
}

fn render_root0_claim(claim: &rv64::Rv64imRoot0Claim) -> String {
    format!(
        "{{ rootParamsId := {}, stages := {{ stage1Digest := {}, stage2Digest := {}, stage3Digest := {}, digest := {} }}, terminal := {{ root0Digest := {}, executionDigest := {}, finalStateDigest := {}, transcriptFinalDigest := {}, digest := {} }}, digest := {} }}",
        render_u8_list(&claim.root_params_id),
        render_u8_list(&claim.stages.stage1_digest),
        render_u8_list(&claim.stages.stage2_digest),
        render_u8_list(&claim.stages.stage3_digest),
        render_u8_list(&claim.stages.digest),
        render_u8_list(&claim.terminal.root0_digest),
        render_u8_list(&claim.terminal.execution_digest),
        render_u8_list(&claim.terminal.final_state_digest),
        render_u8_list(&claim.terminal.transcript_final_digest),
        render_u8_list(&claim.terminal.digest),
        render_u8_list(&claim.digest),
    )
}

fn render_kernel_claim_bundle(bundle: &rv64::Rv64imKernelClaimBundle) -> String {
    format!(
        "{{\n  accepted := {}\n  , mainLane := {}\n  , opening := {}\n  , jointOpening := {}\n  , root0 := {}\n  , digest := {}\n}}",
        render_accepted_proof_claim(&bundle.accepted),
        render_main_lane_claim(&bundle.main_lane),
        render_kernel_opening_claim(&bundle.opening),
        render_joint_opening_claim(&bundle.joint_opening),
        render_root0_claim(&bundle.root0),
        render_u8_list(&bundle.digest),
    )
}

fn render_main_lane_proof_binding(binding: &rv64::Rv64imMainLaneProofBinding) -> String {
    format!(
        "{{ rootLaneColumnsDigest := {}, rootLaneCommitmentDigest := {}, foldSchedule := {}, chunkCount := {}, publicStepCount := {}, digest := {} }}",
        render_u8_list(&binding.root_lane_columns_digest),
        render_u8_list(&binding.root_lane_commitment_digest),
        render_fold_schedule(&binding.fold_schedule),
        binding.chunk_count,
        binding.public_step_count,
        render_u8_list(&binding.digest),
    )
}

fn render_main_lane_proof_bundle(bundle: &rv64::Rv64imMainLaneProofBundle) -> String {
    format!(
        "{{ binding := {}, statementDigest := {}, proofDigest := {}, digest := {} }}",
        render_main_lane_proof_binding(&bundle.binding),
        render_u8_list(&bundle.statement_digest),
        render_u8_list(&bundle.proof_digest),
        render_u8_list(&bundle.digest),
    )
}

fn render_trace_shape_bundle(bundle: &rv64::Rv64imTraceShapeBundle) -> String {
    format!(
        "{{ executionRowCount := {}, realRowCount := {}, effectRowCount := {}, commitRowCount := {}, digest := {} }}",
        bundle.execution_row_count,
        bundle.real_row_count,
        bundle.effect_row_count,
        bundle.commit_row_count,
        render_u8_list(&bundle.digest),
    )
}

fn render_trace_projection_bundle(bundle: &rv64::Rv64imTraceProjectionBundle) -> String {
    format!(
        "{{\n  manifest := {}\n  , executionDigest := {}\n  , shape := {}\n  , digest := {}\n}}",
        render_manifest(&bundle.manifest),
        render_u8_list(&bundle.execution_digest),
        render_trace_shape_bundle(&bundle.shape),
        render_u8_list(&bundle.digest),
    )
}

fn render_stage_witness_summary_bundle(bundle: &rv64::Rv64imStageWitnessSummaryBundle) -> String {
    format!(
        "{{ stage1RowCount := {}, stage2RegisterReadCount := {}, stage2RegisterWriteCount := {}, stage2RamEventCount := {}, stage2TwistLinkCount := {}, stage3ContinuityCount := {}, stage3Halted := {}, transcriptEventCount := {}, digest := {} }}",
        bundle.stage1_row_count,
        bundle.stage2_register_read_count,
        bundle.stage2_register_write_count,
        bundle.stage2_ram_event_count,
        bundle.stage2_twist_link_count,
        bundle.stage3_continuity_count,
        render_bool(bundle.stage3_halted),
        bundle.transcript_event_count,
        render_u8_list(&bundle.digest),
    )
}

fn render_stage_witness_projection_bundle(bundle: &rv64::Rv64imStageWitnessProjectionBundle) -> String {
    format!(
        "{{ summary := {}, digest := {} }}",
        render_stage_witness_summary_bundle(&bundle.summary),
        render_u8_list(&bundle.digest),
    )
}

fn render_stage_claim_digest_bundle(bundle: &rv64::Rv64imStageClaimDigestBundle) -> String {
    format!(
        "{{ claimBundleDigest := {}, stage1Digest := {}, stage2Digest := {}, stage3Digest := {}, transcriptDigest := {}, executionDigest := {}, digest := {} }}",
        render_u8_list(&bundle.claim_bundle_digest),
        render_u8_list(&bundle.stage1_digest),
        render_u8_list(&bundle.stage2_digest),
        render_u8_list(&bundle.stage3_digest),
        render_u8_list(&bundle.transcript_digest),
        render_u8_list(&bundle.execution_digest),
        render_u8_list(&bundle.digest),
    )
}

fn render_stage_claim_proof_bundle(bundle: &rv64::Rv64imStageClaimProofBundle) -> String {
    format!(
        "{{ summary := {}, statementDigest := {}, proofDigest := {}, digest := {} }}",
        render_stage_claim_digest_bundle(&bundle.summary),
        render_u8_list(&bundle.packaged.statement.digest),
        render_u8_list(&bundle.packaged.proof.proof_digest),
        render_u8_list(&bundle.digest),
    )
}

fn render_stage_package_digest_bundle(bundle: &rv64::Rv64imStagePackageDigestBundle) -> String {
    format!(
        "{{ packageBundleDigest := {}, stage1Digest := {}, stage2Digest := {}, stage3Digest := {}, digest := {} }}",
        render_u8_list(&bundle.package_bundle_digest),
        render_u8_list(&bundle.stage1_digest),
        render_u8_list(&bundle.stage2_digest),
        render_u8_list(&bundle.stage3_digest),
        render_u8_list(&bundle.digest),
    )
}

fn render_stage_package_proof_bundle(bundle: &rv64::Rv64imStagePackageProofBundle) -> String {
    format!(
        "{{ summary := {}, digest := {} }}",
        render_stage_package_digest_bundle(&bundle.summary),
        render_u8_list(&bundle.digest),
    )
}

fn render_kernel_opening_binding_bundle(bundle: &rv64::Rv64imKernelOpeningBindingBundle) -> String {
    format!(
        "{{ claimDigest := {}, bindingsDigest := {}, preparedStepsDigest := {}, digest := {} }}",
        render_u8_list(&bundle.claim_digest),
        render_u8_list(&bundle.bindings_digest),
        render_u8_list(&bundle.prepared_steps_digest),
        render_u8_list(&bundle.digest),
    )
}

fn render_kernel_opening_proof_bundle(bundle: &rv64::Rv64imKernelOpeningProofBundle) -> String {
    format!(
        "{{ openingDigest := {}, bindings := {}, digest := {} }}",
        render_u8_list(&bundle.opening_digest),
        render_kernel_opening_binding_bundle(&bundle.bindings),
        render_u8_list(&bundle.digest),
    )
}

fn render_kernel_claim_terminal_bundle(bundle: &rv64::Rv64imKernelClaimTerminalBundle) -> String {
    format!(
        "{{ root0Digest := {}, executionDigest := {}, finalStateDigest := {}, transcriptFinalDigest := {}, finalPc := {}, halted := {}, digest := {} }}",
        render_u8_list(&bundle.root0_digest),
        render_u8_list(&bundle.execution_digest),
        render_u8_list(&bundle.final_state_digest),
        render_u8_list(&bundle.transcript_final_digest),
        bundle.final_pc,
        render_bool(bundle.halted),
        render_u8_list(&bundle.digest),
    )
}

fn render_kernel_claim_summary_bundle(bundle: &rv64::Rv64imKernelClaimSummaryBundle) -> String {
    format!(
        "{{ preparedStepBindingsDigest := {}, terminal := {}, digest := {} }}",
        render_u8_list(&bundle.prepared_step_bindings_digest),
        render_kernel_claim_terminal_bundle(&bundle.terminal),
        render_u8_list(&bundle.digest),
    )
}

fn render_kernel_claim_proof_bundle(bundle: &rv64::Rv64imKernelClaimProofBundle) -> String {
    format!(
        "{{ summary := {}, statementDigest := {}, proofDigest := {}, digest := {} }}",
        render_kernel_claim_summary_bundle(&bundle.summary),
        render_u8_list(&bundle.packaged.statement.digest),
        render_u8_list(&bundle.packaged.proof.proof_digest),
        render_u8_list(&bundle.digest),
    )
}

fn render_kernel_proof_bundle(bundle: &rv64::Rv64imKernelProofBundle) -> String {
    format!(
        "{{\n  rootParamsId := {}\n  , trace := {}\n  , stages := {}\n  , stageClaims := {}\n  , stagePackages := {}\n  , kernelOpening := {}\n  , kernelClaims := {}\n  , rootLaneColumns := {}\n  , rootLaneCommitment := {}\n  , mainLane := {}\n  , digest := {}\n}}",
        render_u8_list(&bundle.root_params_id),
        render_trace_projection_bundle(&bundle.trace),
        render_stage_witness_projection_bundle(&bundle.stages),
        render_stage_claim_proof_bundle(&bundle.stage_claims),
        render_stage_package_proof_bundle(&bundle.stage_packages),
        render_kernel_opening_proof_bundle(&bundle.kernel_opening),
        render_kernel_claim_proof_bundle(&bundle.kernel_claims),
        render_root_lane_columns(&bundle.root_lane_columns),
        render_root_lane_commitment_artifact(&bundle.root_lane_commitment),
        render_main_lane_proof_bundle(&bundle.main_lane),
        render_u8_list(&bundle.digest),
    )
}

fn render_proof_view(proof: &rv64::Rv64imProof) -> String {
    format!(
        "{{\n  claim := {}\n  , statement := {}\n  , kernel := {}\n}}",
        render_kernel_claim_bundle(&proof.claim),
        render_proof_statement(&proof.statement),
        render_kernel_proof_bundle(&proof.kernel),
    )
}

pub(crate) fn render_public_proof_vector_case(case: &PublicProofVectorCase) -> String {
    format!(
        "{{\n  name := {}\n  , proof := {}\n  , statement := {}\n  , claims := {}\n  , kernelProof := {}\n}}",
        render_string(&case.name),
        render_proof_view(&case.proof),
        render_proof_statement(&case.proof.statement),
        render_kernel_claim_bundle(&case.proof.claim),
        render_kernel_proof_bundle(&case.proof.kernel),
    )
}

fn render_source_module(case: &Rv64imParitySourceCase) -> String {
    format!(
        "import Nightstream.Rv64IM.Generated.ParityTypes\n\nnamespace Nightstream.Rv64IM.Generated.Cases.Case_{}\n\nopen Nightstream.Rv64IM.Generated\n\ndef sourceCase : ParitySourceCase :=\n  {}\n\nend Nightstream.Rv64IM.Generated.Cases.Case_{}\n",
        lean_ident_fragment(&case.manifest.name),
        render_source_case(case),
        lean_ident_fragment(&case.manifest.name),
    )
}

fn render_derived_module(case: &Rv64imParityDerivedCase) -> String {
    format!(
        "import Nightstream.Rv64IM.Generated.ParityTypes\n\nnamespace Nightstream.Rv64IM.Generated.Cases.Case_{}\n\nopen Nightstream.Rv64IM.Generated\n\ndef derivedCase : ParityDerivedCase :=\n  {}\n\nend Nightstream.Rv64IM.Generated.Cases.Case_{}\n",
        lean_ident_fragment(&case.manifest.name),
        render_derived_case(case),
        lean_ident_fragment(&case.manifest.name),
    )
}

fn render_index_module(module_name: &str, cases: &[(Rv64imParitySourceCase, Rv64imParityDerivedCase)]) -> String {
    let mut imports = String::new();
    let mut sources = String::from("[");
    let mut derived = String::from("[");
    let mut parity_cases = String::from("[");

    for (idx, (source, _)) in cases.iter().enumerate() {
        let ident = lean_ident_fragment(&source.manifest.name);
        imports.push_str(&format!(
            "import Nightstream.Rv64IM.Generated.Cases.Case_{ident}.Source\nimport Nightstream.Rv64IM.Generated.Cases.Case_{ident}.Derived\n"
        ));
        if idx > 0 {
            sources.push_str(", ");
            derived.push_str(", ");
            parity_cases.push_str(", ");
        }
        sources.push_str(&format!("Nightstream.Rv64IM.Generated.Cases.Case_{ident}.sourceCase"));
        derived.push_str(&format!("Nightstream.Rv64IM.Generated.Cases.Case_{ident}.derivedCase"));
        parity_cases.push_str(&format!(
            "(Nightstream.Rv64IM.Generated.Cases.Case_{ident}.sourceCase, Nightstream.Rv64IM.Generated.Cases.Case_{ident}.derivedCase)"
        ));
    }

    sources.push(']');
    derived.push(']');
    parity_cases.push(']');

    format!(
        "{imports}\nnamespace Nightstream.Rv64IM.Generated.Index.{module_name}\n\nopen Nightstream.Rv64IM.Generated\n\ndef sourceCases : List ParitySourceCase :=\n  {sources}\n\ndef derivedCases : List ParityDerivedCase :=\n  {derived}\n\ndef parityCases : List (ParitySourceCase × ParityDerivedCase) :=\n  {parity_cases}\n\nend Nightstream.Rv64IM.Generated.Index.{module_name}\n"
    )
}

fn render_corpus_module() -> String {
    "import Nightstream.Rv64IM.Generated.Index.AllCases\nimport Nightstream.Rv64IM.Generated.Index.AlignedMemory\nimport Nightstream.Rv64IM.Generated.Index.ControlFlow\nimport Nightstream.Rv64IM.Generated.Index.Multiply\nimport Nightstream.Rv64IM.Generated.Index.NarrowMemory\nimport Nightstream.Rv64IM.Generated.Index.NativeAlu\nimport Nightstream.Rv64IM.Generated.Index.SignedDivRem\nimport Nightstream.Rv64IM.Generated.Index.UnsignedDivRem\n\nnamespace Nightstream.Rv64IM.Generated\n\ndef nativeAluParityCases : List (ParitySourceCase × ParityDerivedCase) :=\n  Nightstream.Rv64IM.Generated.Index.NativeAlu.parityCases\n\ndef alignedMemoryParityCases : List (ParitySourceCase × ParityDerivedCase) :=\n  Nightstream.Rv64IM.Generated.Index.AlignedMemory.parityCases\n\ndef narrowMemoryParityCases : List (ParitySourceCase × ParityDerivedCase) :=\n  Nightstream.Rv64IM.Generated.Index.NarrowMemory.parityCases\n\ndef multiplyParityCases : List (ParitySourceCase × ParityDerivedCase) :=\n  Nightstream.Rv64IM.Generated.Index.Multiply.parityCases\n\ndef unsignedDivRemParityCases : List (ParitySourceCase × ParityDerivedCase) :=\n  Nightstream.Rv64IM.Generated.Index.UnsignedDivRem.parityCases\n\ndef signedDivRemParityCases : List (ParitySourceCase × ParityDerivedCase) :=\n  Nightstream.Rv64IM.Generated.Index.SignedDivRem.parityCases\n\ndef controlFlowParityCases : List (ParitySourceCase × ParityDerivedCase) :=\n  Nightstream.Rv64IM.Generated.Index.ControlFlow.parityCases\n\ndef parityCases : List (ParitySourceCase × ParityDerivedCase) :=\n  Nightstream.Rv64IM.Generated.Index.AllCases.parityCases\n\nend Nightstream.Rv64IM.Generated\n".into()
}

fn render_public_proof_case_module(case: &PublicProofVectorCase) -> String {
    format!(
        "import Nightstream.Rv64IM.Generated.PublicProofVectorTypes\n\nset_option maxHeartbeats 0\n\nnamespace Nightstream.Rv64IM.Generated.PublicProofVectors.Case_{}\n\nopen Nightstream.Rv64IM.Generated\n\ndef proofCase : PublicProofVectorCase :=\n  {}\n\nend Nightstream.Rv64IM.Generated.PublicProofVectors.Case_{}\n",
        lean_ident_fragment(&case.name),
        render_public_proof_vector_case(case),
        lean_ident_fragment(&case.name),
    )
}

fn render_public_proof_corpus_module(cases: &[PublicProofVectorCase]) -> String {
    let mut imports = String::new();
    let mut proof_cases = String::from("[");

    for (idx, case) in cases.iter().enumerate() {
        let ident = lean_ident_fragment(&case.name);
        imports.push_str(&format!(
            "import Nightstream.Rv64IM.Generated.PublicProofVectors.Case_{ident}\n"
        ));
        if idx > 0 {
            proof_cases.push_str(", ");
        }
        proof_cases.push_str(&format!(
            "Nightstream.Rv64IM.Generated.PublicProofVectors.Case_{ident}.proofCase"
        ));
    }

    proof_cases.push(']');

    format!(
        "{imports}\nnamespace Nightstream.Rv64IM.Generated.PublicProofVectors\n\nopen Nightstream.Rv64IM.Generated\n\ndef cases : List PublicProofVectorCase :=\n  {proof_cases}\n\nend Nightstream.Rv64IM.Generated.PublicProofVectors\n"
    )
}

fn public_proof_input(source: &Rv64imParitySourceCase) -> rv64::Rv64imProofInput {
    rv64::Rv64imProofInput {
        source: source.clone(),
        max_steps: source.program_words.len(),
    }
}

pub(crate) fn build_public_proof_cases(
    cases: &[(Rv64imParitySourceCase, Rv64imParityDerivedCase)],
) -> Vec<PublicProofVectorCase> {
    cases
        .iter()
        .map(|(source, derived)| {
            let name = source.manifest.name.as_str();
            let input = public_proof_input(source);
            let witness = build_rv64im_audit_witness_bundle(&input)
                .unwrap_or_else(|err| panic!("build RV64IM audit witness vector {name}: {err}"));
            let proof = prove_rv64im_public_proof(&input)
                .unwrap_or_else(|err| panic!("prove RV64IM public proof vector {name}: {err}"));
            verify_rv64im_public_proof(&proof)
                .unwrap_or_else(|err| panic!("verify RV64IM public proof vector {name}: {err}"));
            let verified = build_rv64im_audit_witness_bundle(&input)
                .unwrap_or_else(|err| panic!("rebuild RV64IM audit witness vector {name}: {err}"));
            assert_eq!(verified.digest, witness.digest, "proof witness digest roundtrip for {name}");
            assert_eq!(
                proof.statement.execution_digest,
                derived.kernel.execution_digest,
                "statement execution digest matches derived kernel for {name}"
            );
            assert_eq!(
                proof.statement.final_state_digest,
                derived.kernel.final_state_digest,
                "statement final-state digest matches derived kernel for {name}"
            );
            assert_eq!(
                proof.statement.transcript_final_digest,
                derived.kernel.transcript_final_digest,
                "statement transcript digest matches derived kernel for {name}"
            );
            PublicProofVectorCase {
                name: source.manifest.name.clone(),
                proof,
            }
        })
        .collect()
}

fn write_file(path: &Path, contents: String) {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).expect("create output directory");
    }
    fs::write(path, contents).expect("write generated file");
}

fn reset_generated_dirs() {
    for path in [
        generated_dir().join("Cases"),
        generated_dir().join("Index"),
        public_proof_vector_dir(),
        accepted_proof_artifact_dir(),
    ] {
        if path.exists() {
            fs::remove_dir_all(&path).expect("remove stale generated directory");
        }
    }
}

fn main() {
    let cases = build_all_parity_cases().expect("build RV64IM parity cases");
    let public_proof_cases = build_public_proof_cases(&cases);
    let accepted_proof_cases = build_accepted_proof_cases(&cases, &public_proof_cases);
    reset_generated_dirs();

    for (source, derived) in &cases {
        let case_path = case_dir(&source.manifest.name);
        write_file(&case_path.join("Source.lean"), render_source_module(source));
        write_file(&case_path.join("Derived.lean"), render_derived_module(derived));
    }

    let native_alu_cases = cases
        .iter()
        .filter(|(source, _)| {
            source
                .manifest
                .family_tags
                .contains(&neo_fold_prototype::rv64im::tables::Rv64FamilyTag::NativeAlu)
        })
        .cloned()
        .collect::<Vec<_>>();
    let aligned_memory_cases = cases
        .iter()
        .filter(|(source, _)| {
            source
                .manifest
                .family_tags
                .contains(&neo_fold_prototype::rv64im::tables::Rv64FamilyTag::AlignedMemory)
        })
        .cloned()
        .collect::<Vec<_>>();
    let narrow_memory_cases = cases
        .iter()
        .filter(|(source, _)| {
            source
                .manifest
                .family_tags
                .contains(&neo_fold_prototype::rv64im::tables::Rv64FamilyTag::NarrowMemory)
        })
        .cloned()
        .collect::<Vec<_>>();
    let multiply_cases = cases
        .iter()
        .filter(|(source, _)| {
            source
                .manifest
                .family_tags
                .contains(&neo_fold_prototype::rv64im::tables::Rv64FamilyTag::Multiply)
        })
        .cloned()
        .collect::<Vec<_>>();
    let unsigned_divrem_cases = cases
        .iter()
        .filter(|(source, _)| {
            source
                .manifest
                .family_tags
                .contains(&neo_fold_prototype::rv64im::tables::Rv64FamilyTag::UnsignedDivRem)
        })
        .cloned()
        .collect::<Vec<_>>();
    let signed_divrem_cases = cases
        .iter()
        .filter(|(source, _)| {
            source
                .manifest
                .family_tags
                .contains(&neo_fold_prototype::rv64im::tables::Rv64FamilyTag::SignedDivRem)
        })
        .cloned()
        .collect::<Vec<_>>();
    let control_flow_cases = cases
        .iter()
        .filter(|(source, _)| {
            source
                .manifest
                .family_tags
                .contains(&neo_fold_prototype::rv64im::tables::Rv64FamilyTag::ControlFlow)
        })
        .cloned()
        .collect::<Vec<_>>();

    write_file(
        &generated_dir().join("Index").join("NativeAlu.lean"),
        render_index_module(family_module_name(neo_fold_prototype::rv64im::tables::Rv64FamilyTag::NativeAlu), &native_alu_cases),
    );
    write_file(
        &generated_dir().join("Index").join("AlignedMemory.lean"),
        render_index_module(
            family_module_name(neo_fold_prototype::rv64im::tables::Rv64FamilyTag::AlignedMemory),
            &aligned_memory_cases,
        ),
    );
    write_file(
        &generated_dir().join("Index").join("NarrowMemory.lean"),
        render_index_module(
            family_module_name(neo_fold_prototype::rv64im::tables::Rv64FamilyTag::NarrowMemory),
            &narrow_memory_cases,
        ),
    );
    write_file(
        &generated_dir().join("Index").join("Multiply.lean"),
        render_index_module(
            family_module_name(neo_fold_prototype::rv64im::tables::Rv64FamilyTag::Multiply),
            &multiply_cases,
        ),
    );
    write_file(
        &generated_dir().join("Index").join("UnsignedDivRem.lean"),
        render_index_module(
            family_module_name(neo_fold_prototype::rv64im::tables::Rv64FamilyTag::UnsignedDivRem),
            &unsigned_divrem_cases,
        ),
    );
    write_file(
        &generated_dir().join("Index").join("SignedDivRem.lean"),
        render_index_module(
            family_module_name(neo_fold_prototype::rv64im::tables::Rv64FamilyTag::SignedDivRem),
            &signed_divrem_cases,
        ),
    );
    write_file(
        &generated_dir().join("Index").join("ControlFlow.lean"),
        render_index_module(
            family_module_name(neo_fold_prototype::rv64im::tables::Rv64FamilyTag::ControlFlow),
            &control_flow_cases,
        ),
    );
    write_file(
        &generated_dir().join("Index").join("AllCases.lean"),
        render_index_module("AllCases", &cases),
    );
    write_file(
        &generated_dir().join("ImportedParityCorpus.lean"),
        render_corpus_module(),
    );
    for case in &public_proof_cases {
        write_file(
            &public_proof_case_path(&case.name),
            render_public_proof_case_module(case),
        );
    }
    for case in &accepted_proof_cases {
        write_file(
            &accepted_proof_case_path(&case.name),
            render_accepted_artifact_case_module(case),
        );
    }
    write_file(
        &public_proof_vector_dir().join("Corpus.lean"),
        render_public_proof_corpus_module(&public_proof_cases),
    );
    write_file(
        &accepted_proof_artifact_dir().join("Corpus.lean"),
        render_accepted_proof_corpus_module(&accepted_proof_cases),
    );

    println!(
        "wrote RV64IM parity artifacts for {} cases, {} public proof vectors, and {} accepted proof artifacts to {}",
        cases.len(),
        public_proof_cases.len(),
        accepted_proof_cases.len(),
        generated_dir().display()
    );
}

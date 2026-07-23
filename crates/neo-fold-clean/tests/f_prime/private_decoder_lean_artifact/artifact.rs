use super::batch_grammar::{
    AliasConsumer, AliasLink, Atom, Batch, Call, Consumer, OpeningGroup, OpeningGroupKind, Owner, Program, Summary,
    Template,
};
use super::{Census, RunToken};

pub(super) const GENERATED_DIRECTORY: &str =
    "formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/FPrimeSelectiveFixedPoint/PiCcsNc/ProductionDomain/PrivateDecoder/Generated";

const MODULE: &str =
    "Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.ProductionDomain.PrivateDecoder";
const SCHEMA: &str =
    "Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.ProductionDomain.PrivateDecoder.Schema";
const NAMESPACE: &str =
    "Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.ProductionDomain.PrivateDecoder.Generated";
const TEMPLATES_PER_CHUNK: usize = 8;
const TEMPLATE_CHUNKS_PER_FILE: usize = 10;
const RECORDS_PER_CHUNK: usize = 256;
const ALIAS_CONSUMERS_PER_CHUNK: usize = 128;
const RECORD_CHUNKS_PER_FILE: usize = 5;

pub(super) struct GeneratedLeanFile {
    pub relative_path: String,
    pub contents: String,
}

fn bool_literal(value: bool) -> &'static str {
    if value {
        "true"
    } else {
        "false"
    }
}

fn census(value: Census) -> String {
    format!(
        "{{ eliminated := {}, unit := {}, balanced := {}, binary := {}, decompositionAliases := {}, equalityAliases := {}, equalityAliasSavings := {}, retainedCoordinatesBeforeAliases := {}, centeredColumns := {} }}",
        value.eliminated,
        value.unit,
        value.balanced,
        value.binary,
        value.decomposition_aliases,
        value.equality_aliases,
        value.equality_alias_savings,
        value.retained_coordinates_before_aliases,
        value.centered_columns,
    )
}

fn summary(value: Summary) -> String {
    format!(
        "{{ sourceColumns := {}, freshCoordinates := {}, census := {} }}",
        value.source_columns,
        value.fresh_coordinates,
        census(value.census),
    )
}

fn atom(value: Atom) -> String {
    match value {
        Atom::Ordinary(RunToken::Direct {
            length,
            start_stride,
            width,
            centered,
        }) => format!(".direct {length} {start_stride} {width} {}", bool_literal(centered)),
        Atom::Ordinary(RunToken::DecompositionAlias {
            length,
            source_delta,
            source_stride,
            digit,
            digit_stride,
            start_stride,
            centered,
        }) => format!(
            ".decompositionAlias {length} {source_delta} {source_stride} {digit} {digit_stride} {start_stride} {}",
            bool_literal(centered)
        ),
        Atom::Ordinary(RunToken::EqualityAlias {
            length,
            source_delta,
            source_stride,
            start_stride,
            width,
            centered,
        }) => format!(
            ".equalityAlias {length} {source_delta} {source_stride} {start_stride} {width} {}",
            bool_literal(centered)
        ),
        Atom::Ordinary(RunToken::LinearDefinition { length }) => format!(".linearDefinition {length}"),
        Atom::Ordinary(RunToken::TraceEliminated { length }) => format!(".traceEliminated {length}"),
        Atom::SisBatch { batch } => format!(".sisBatch {batch}"),
    }
}

fn template(value: &Template) -> String {
    let atoms = value
        .atoms
        .iter()
        .copied()
        .map(atom)
        .collect::<Vec<_>>()
        .join(", ");
    format!("{{ atoms := [{atoms}], summary := {} }}", summary(value.summary))
}

fn call(value: Call) -> String {
    format!(
        "{{ template := {}, sourceStart := {}, finalStart := {} }}",
        value.template, value.source_start, value.final_start
    )
}

fn opening_group(value: OpeningGroup) -> String {
    let kind = match value.kind {
        OpeningGroupKind::Alias { source, source_stride } => format!(".alias {source} {source_stride}"),
        OpeningGroupKind::Direct => ".direct".to_string(),
    };
    format!(
        "{{ openingStart := {}, length := {}, directBefore := {}, kind := {kind} }}",
        value.opening_start, value.length, value.direct_before
    )
}

fn owner(value: Owner) -> String {
    match value {
        Owner::Ordinary { call, atom, offset } => format!(".ordinary {call} {atom} {offset}"),
        Owner::Batch {
            call,
            atom,
            batch,
            offset,
        } => format!(".batch {call} {atom} {batch} {offset}"),
    }
}

fn consumer(value: Consumer) -> String {
    match value {
        Consumer::Ordinary { call, atom, offset } => format!(".ordinary {call} {atom} {offset}"),
        Consumer::Batch {
            call,
            atom,
            batch,
            group,
            offset,
        } => format!(".batch {call} {atom} {batch} {group} {offset}"),
    }
}

fn alias_link(value: AliasLink) -> String {
    format!(
        "{{ consumer := {}, length := {}, target := {}, targetOffsetStride := {} }}",
        consumer(value.consumer),
        value.length,
        owner(value.target),
        value.target_offset_stride,
    )
}

fn alias_consumer(value: AliasConsumer) -> String {
    format!(
        "{{ consumer := {}, length := {}, linkStart := {}, linkStop := {} }}",
        consumer(value.consumer),
        value.length,
        value.link_start,
        value.link_stop,
    )
}

fn generated_header(import: &str, leaf: &str, content: &str) -> String {
    format!(
        "import {import}\n\n/-! Generated file: bounded private-decoder {leaf}. Do not hand-edit. -/\n\nnamespace {NAMESPACE}.{content}\n\n"
    )
}

fn list_def(name: &str, ty: &str, records: impl IntoIterator<Item = String>) -> String {
    let records = records
        .into_iter()
        .map(|record| format!("  {record}"))
        .collect::<Vec<_>>()
        .join("\n, ");
    format!("def {name} : List ({ty}) := [\n{records}\n]\n\n")
}

fn chunk_files<T: Copy>(
    records: &[T],
    records_per_chunk: usize,
    directory: &str,
    module: &str,
    ty: &str,
    render: impl Fn(T) -> String,
) -> (Vec<GeneratedLeanFile>, Vec<(usize, usize)>) {
    let chunks = records.chunks(records_per_chunk).collect::<Vec<_>>();
    let mut files = Vec::new();
    let mut refs = Vec::new();
    for (part, part_chunks) in chunks.chunks(RECORD_CHUNKS_PER_FILE).enumerate() {
        let namespace = format!("{module}Part{part}");
        let import = SCHEMA;
        let mut contents = generated_header(import, directory, &namespace);
        for (local, chunk) in part_chunks.iter().enumerate() {
            let chunk_index = part * RECORD_CHUNKS_PER_FILE + local;
            contents.push_str(&list_def(
                &format!("chunk{chunk_index}"),
                ty,
                chunk.iter().copied().map(&render),
            ));
            refs.push((part, chunk_index));
        }
        contents.push_str(&format!("end {NAMESPACE}.{namespace}\n"));
        files.push(GeneratedLeanFile {
            relative_path: format!("{GENERATED_DIRECTORY}/{directory}/Part{part}.lean"),
            contents,
        });
    }
    (files, refs)
}

fn template_files(program: &Program) -> (Vec<GeneratedLeanFile>, Vec<(usize, usize)>) {
    let chunks = program
        .templates
        .chunks(TEMPLATES_PER_CHUNK)
        .collect::<Vec<_>>();
    let mut files = Vec::new();
    let mut refs = Vec::new();
    for (part, part_chunks) in chunks.chunks(TEMPLATE_CHUNKS_PER_FILE).enumerate() {
        let namespace = format!("TemplatePart{part}");
        let mut contents = generated_header(SCHEMA, "template data", &namespace);
        for (local, chunk) in part_chunks.iter().enumerate() {
            let chunk_index = part * TEMPLATE_CHUNKS_PER_FILE + local;
            let atoms = chunk.iter().map(|entry| entry.atoms.len()).sum::<usize>();
            assert!(atoms <= RECORDS_PER_CHUNK, "template certificate exceeds 256 atoms");
            contents.push_str(&list_def(
                &format!("chunk{chunk_index}"),
                "RawTemplate",
                chunk.iter().map(template),
            ));
            refs.push((part, chunk_index));
        }
        contents.push_str(&format!("end {NAMESPACE}.{namespace}\n"));
        files.push(GeneratedLeanFile {
            relative_path: format!("{GENERATED_DIRECTORY}/Templates/Part{part}.lean"),
            contents,
        });
    }
    (files, refs)
}

fn group_files(program: &Program) -> (Vec<GeneratedLeanFile>, Vec<(usize, usize)>, Vec<(usize, usize)>) {
    let mut shards = Vec::<Vec<OpeningGroup>>::new();
    let mut batch_ranges = Vec::new();
    for batch in &program.batches {
        let start = shards.len();
        shards.extend(batch.groups.chunks(RECORDS_PER_CHUNK).map(<[_]>::to_vec));
        batch_ranges.push((start, shards.len() - start));
    }
    let flat = shards.iter().flatten().copied().collect::<Vec<_>>();
    let mut files = Vec::new();
    let mut refs = Vec::new();
    for (part, part_shards) in shards.chunks(RECORD_CHUNKS_PER_FILE).enumerate() {
        let namespace = format!("OpeningGroupPart{part}");
        let mut contents = generated_header(SCHEMA, "SIS opening-group data", &namespace);
        for (local, shard) in part_shards.iter().enumerate() {
            let shard_index = part * RECORD_CHUNKS_PER_FILE + local;
            contents.push_str(&list_def(
                &format!("chunk{shard_index}"),
                "RawOpeningGroup",
                shard.iter().copied().map(opening_group),
            ));
            refs.push((part, shard_index));
        }
        contents.push_str(&format!("end {NAMESPACE}.{namespace}\n"));
        files.push(GeneratedLeanFile {
            relative_path: format!("{GENERATED_DIRECTORY}/OpeningGroups/Part{part}.lean"),
            contents,
        });
    }
    assert_eq!(
        flat.len(),
        program
            .batches
            .iter()
            .map(|batch| batch.groups.len())
            .sum::<usize>()
    );
    (files, refs, batch_ranges)
}

fn batch(value: &Batch, group_range: (usize, usize)) -> String {
    format!(
        "{{ sourceStart := {}, sourceEnd := {}, inputBinding := {}, commitmentFields := {}, openings := {}, directOpenings := {}, groupShardStart := {}, groupShardCount := {} }}",
        value.source_start,
        value.source_end,
        bool_literal(value.input_binding),
        value.commitment_fields,
        value.openings,
        value.direct_openings,
        group_range.0,
        group_range.1,
    )
}

fn refs(name: &str, ty: &str, module: &str, values: &[(usize, usize)]) -> String {
    list_def(
        name,
        ty,
        values
            .iter()
            .map(|(part, chunk)| format!("{module}Part{part}.chunk{chunk}")),
    )
}

pub(super) fn generated_files(program: &Program) -> Vec<GeneratedLeanFile> {
    let (template_files, template_refs) = template_files(program);
    let (call_files, call_refs) = chunk_files(&program.calls, RECORDS_PER_CHUNK, "Calls", "Call", "RawCall", call);
    let (group_files, group_refs, batch_ranges) = group_files(program);
    let (link_files, link_refs) = chunk_files(
        &program.alias_links,
        RECORDS_PER_CHUNK,
        "AliasLinks",
        "AliasLink",
        "RawAliasLink",
        alias_link,
    );
    let (consumer_files, consumer_refs) = chunk_files(
        &program.alias_consumers,
        ALIAS_CONSUMERS_PER_CHUNK,
        "AliasConsumers",
        "AliasConsumer",
        "RawAliasConsumer",
        alias_consumer,
    );

    let imports = template_files
        .iter()
        .chain(&call_files)
        .chain(&group_files)
        .chain(&link_files)
        .chain(&consumer_files)
        .map(|file| {
            let relative = file
                .relative_path
                .strip_prefix("formal/nightstream-lean/")
                .expect("generated Lean path root")
                .strip_suffix(".lean")
                .expect("generated Lean suffix")
                .replace('/', ".");
            format!("import {relative}")
        })
        .collect::<Vec<_>>()
        .join("\n");
    let mut metadata = format!(
        "{imports}\n\n/-! Generated file: complete compact private-decoder registry. Do not hand-edit. -/\n\nnamespace {NAMESPACE}\n\n"
    );
    metadata.push_str("def schemaVersion : Nat := 1\n");
    metadata.push_str(&format!("def sourceStart : Nat := {}\n", super::SOURCE_START));
    metadata.push_str(&format!("def sourceStop : Nat := {}\n", super::SOURCE_STOP));
    metadata.push_str(&format!("def finalStart : Nat := {}\n", super::FINAL_START));
    metadata.push_str(&format!("def finalStop : Nat := {}\n", super::BRANCH_STOP));
    metadata.push_str(&format!("def templatesPerChunk : Nat := {TEMPLATES_PER_CHUNK}\n"));
    metadata.push_str(&format!("def recordsPerChunk : Nat := {RECORDS_PER_CHUNK}\n"));
    metadata.push_str(&format!(
        "def aliasConsumersPerChunk : Nat := {ALIAS_CONSUMERS_PER_CHUNK}\n"
    ));
    metadata.push_str(&format!("def templateCount : Nat := {}\n", program.templates.len()));
    metadata.push_str(&format!(
        "def templateAtomCount : Nat := {}\n",
        program
            .templates
            .iter()
            .map(|entry| entry.atoms.len())
            .sum::<usize>()
    ));
    metadata.push_str(&format!("def callCount : Nat := {}\n", program.calls.len()));
    metadata.push_str(&format!("def batchCount : Nat := {}\n", program.batches.len()));
    metadata.push_str(&format!(
        "def openingGroupCount : Nat := {}\n",
        program
            .batches
            .iter()
            .map(|entry| entry.groups.len())
            .sum::<usize>()
    ));
    metadata.push_str(&format!("def aliasLinkCount : Nat := {}\n", program.alias_links.len()));
    metadata.push_str(&format!(
        "def aliasConsumerCount : Nat := {}\n",
        program.alias_consumers.len()
    ));
    metadata.push_str(&format!(
        "def rootSummary : RawSummary := {}\n\n",
        summary(program.summary)
    ));
    metadata.push_str(&list_def(
        "templateChunkContexts",
        "RawTemplateChunkContext",
        program
            .templates
            .chunks(TEMPLATES_PER_CHUNK)
            .scan(0usize, |start, chunk| {
                let first = *start;
                *start += chunk.len();
                Some(format!(
                    "{{ templateStart := {first}, templateStop := {}, atomCount := {} }}",
                    *start,
                    chunk.iter().map(|entry| entry.atoms.len()).sum::<usize>()
                ))
            }),
    ));
    metadata.push_str(&list_def(
        "callChunkContexts",
        "RawCallChunkContext",
        program.calls.chunks(RECORDS_PER_CHUNK).scan(0usize, |start, chunk| {
            let first = *start;
            *start += chunk.len();
            let head = chunk.first().expect("nonempty call chunk");
            let tail = chunk.last().expect("nonempty call chunk");
            let tail_summary = program.templates[tail.template].summary;
            Some(format!(
                "{{ callStart := {first}, callStop := {}, sourceStart := {}, sourceStop := {}, finalStart := {}, finalStop := {} }}",
                *start,
                head.source_start,
                tail.source_start + tail_summary.source_columns,
                head.final_start,
                tail.final_start + tail_summary.fresh_coordinates,
            ))
        }),
    ));
    let mut group_contexts = Vec::new();
    for (batch_index, entry) in program.batches.iter().enumerate() {
        for (shard_index, shard) in entry.groups.chunks(RECORDS_PER_CHUNK).enumerate() {
            let first = shard.first().expect("nonempty opening-group shard");
            let last = shard.last().expect("nonempty opening-group shard");
            let direct_stop = last.direct_before + usize::from(last.kind == OpeningGroupKind::Direct) * last.length;
            group_contexts.push(format!(
                "{{ batch := {batch_index}, groupStart := {}, groupStop := {}, openingStart := {}, openingStop := {}, directStart := {}, directStop := {direct_stop} }}",
                shard_index * RECORDS_PER_CHUNK,
                shard_index * RECORDS_PER_CHUNK + shard.len(),
                first.opening_start,
                last.opening_start + last.length,
                first.direct_before,
            ));
        }
    }
    metadata.push_str(&list_def(
        "openingGroupShardContexts",
        "RawOpeningGroupShardContext",
        group_contexts,
    ));
    metadata.push_str(&list_def(
        "aliasLinkChunkContexts",
        "RawAliasLinkChunkContext",
        program
            .alias_links
            .chunks(RECORDS_PER_CHUNK)
            .scan(0usize, |start, chunk| {
                let first = *start;
                *start += chunk.len();
                Some(format!("{{ linkStart := {first}, linkStop := {} }}", *start))
            }),
    ));
    metadata.push_str(&list_def(
        "aliasConsumerChunkContexts",
        "RawAliasConsumerChunkContext",
        program
            .alias_consumers
            .chunks(ALIAS_CONSUMERS_PER_CHUNK)
            .scan(0usize, |start, chunk| {
                let first = *start;
                *start += chunk.len();
                let head = chunk.first().expect("nonempty alias-consumer chunk");
                let tail = chunk.last().expect("nonempty alias-consumer chunk");
                Some(format!(
                    "{{ consumerStart := {first}, consumerStop := {}, linkStart := {}, linkStop := {} }}",
                    *start, head.link_start, tail.link_stop
                ))
            }),
    ));
    metadata.push_str(&refs("templateChunks", "List RawTemplate", "Template", &template_refs));
    metadata.push_str(&refs("callChunks", "List RawCall", "Call", &call_refs));
    metadata.push_str(&refs(
        "openingGroupShards",
        "List RawOpeningGroup",
        "OpeningGroup",
        &group_refs,
    ));
    metadata.push_str(&refs("aliasLinkChunks", "List RawAliasLink", "AliasLink", &link_refs));
    metadata.push_str(&refs(
        "aliasConsumerChunks",
        "List RawAliasConsumer",
        "AliasConsumer",
        &consumer_refs,
    ));
    metadata.push_str(&list_def(
        "batches",
        "RawBatch",
        program
            .batches
            .iter()
            .zip(batch_ranges)
            .map(|(entry, range)| batch(entry, range)),
    ));
    metadata.push_str(&format!("end {NAMESPACE}\n"));

    let mut files = Vec::new();
    files.extend(template_files);
    files.extend(call_files);
    files.extend(group_files);
    files.extend(link_files);
    files.extend(consumer_files);
    files.push(GeneratedLeanFile {
        relative_path: format!("{GENERATED_DIRECTORY}/Metadata.lean"),
        contents: metadata,
    });
    assert_eq!(
        MODULE,
        NAMESPACE
            .strip_suffix(".Generated")
            .expect("generated namespace")
    );
    files
}

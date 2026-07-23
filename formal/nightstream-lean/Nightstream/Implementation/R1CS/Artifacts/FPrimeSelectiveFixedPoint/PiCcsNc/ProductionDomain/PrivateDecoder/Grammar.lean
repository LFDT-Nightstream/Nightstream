import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.ProductionDomain.PrivateDecoder.Schema

/-!
Kernel semantics for the compact private-decoder certificate.

Owns: source/fresh cursor summaries, exact SIS batch geometry, bounded chunk
checks, and chunked lookup. The generated artifact supplies only proof-free
records; these definitions assign their meaning independently of Rust labels.

Does not own: source values, eliminated equations, derived products, sparse
matrix coefficients, CCS/CE membership, or commitment binding.

Emits constraints: none.
-/

namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.ProductionDomain.PrivateDecoder

def zeroCensus : RawCensus :=
  { eliminated := 0
    unit := 0
    balanced := 0
    binary := 0
    decompositionAliases := 0
    equalityAliases := 0
    equalityAliasSavings := 0
    retainedCoordinatesBeforeAliases := 0
    centeredColumns := 0 }

def addCensus (left right : RawCensus) : RawCensus :=
  { eliminated := left.eliminated + right.eliminated
    unit := left.unit + right.unit
    balanced := left.balanced + right.balanced
    binary := left.binary + right.binary
    decompositionAliases := left.decompositionAliases + right.decompositionAliases
    equalityAliases := left.equalityAliases + right.equalityAliases
    equalityAliasSavings := left.equalityAliasSavings + right.equalityAliasSavings
    retainedCoordinatesBeforeAliases :=
      left.retainedCoordinatesBeforeAliases + right.retainedCoordinatesBeforeAliases
    centeredColumns := left.centeredColumns + right.centeredColumns }

def zeroSummary : RawSummary :=
  { sourceColumns := 0, freshCoordinates := 0, census := zeroCensus }

def addSummary (left right : RawSummary) : RawSummary :=
  { sourceColumns := left.sourceColumns + right.sourceColumns
    freshCoordinates := left.freshCoordinates + right.freshCoordinates
    census := addCensus left.census right.census }

def widthCensus? (length width : Nat) (centered : Bool) : Option RawCensus := do
  let (unit, balanced, binary) ←
    match width with
    | 1 => some (length, 0, 0)
    | 41 => some (0, length, 0)
    | 64 => some (0, 0, length)
    | _ => none
  some
    { zeroCensus with
      unit
      balanced
      binary
      retainedCoordinatesBeforeAliases := length * width
      centeredColumns := if centered then length else 0 }

def batchSummary? (batch : RawBatch) : Option RawSummary := do
  let expectedCommitment := if batch.inputBinding then 108 else 54
  if batch.commitmentFields != expectedCommitment then none else
  if batch.directOpenings > batch.openings then none else
  let sourceColumns := 2 + batch.commitmentFields + 122 * batch.openings
  if batch.sourceEnd != batch.sourceStart + sourceColumns then none else
  let aliasOpenings := batch.openings - batch.directOpenings
  some
    { sourceColumns
      freshCoordinates :=
        41 * batch.commitmentFields + 40 * batch.openings +
          41 * batch.directOpenings
      census :=
        { eliminated := 2 + 41 * batch.openings
          unit := 81 * batch.openings
          balanced := batch.commitmentFields
          binary := 0
          decompositionAliases := 41 * aliasOpenings
          equalityAliases := 0
          equalityAliasSavings := 0
          retainedCoordinatesBeforeAliases :=
            81 * batch.openings + 41 * batch.commitmentFields
          centeredColumns := 41 * batch.openings } }

def atomSummary? (batches : List RawBatch) : RawAtom → Option RawSummary
  | .direct length _ width centered => do
      if length = 0 then none else
      let census ← widthCensus? length width centered
      some
        { sourceColumns := length
          freshCoordinates := length * width
          census }
  | .decompositionAlias length sourceDelta _ _ _ _ centered => do
      if length = 0 || sourceDelta = 0 then none else
      let census ← widthCensus? length 1 centered
      some
        { sourceColumns := length
          freshCoordinates := 0
          census := { census with decompositionAliases := length } }
  | .equalityAlias length sourceDelta _ _ width centered => do
      if length = 0 || sourceDelta = 0 then none else
      let census ← widthCensus? length width centered
      some
        { sourceColumns := length
          freshCoordinates := 0
          census :=
            { census with
              equalityAliases := length
              equalityAliasSavings := length * width } }
  | .linearDefinition length =>
      if length = 0 then none else
      some
        { sourceColumns := length
          freshCoordinates := 0
          census := { zeroCensus with eliminated := length } }
  | .traceEliminated length =>
      if length = 0 then none else
      some
        { sourceColumns := length
          freshCoordinates := 0
          census := { zeroCensus with eliminated := length } }
  | .sisBatch batch => do
      let value ← batches[batch]?
      batchSummary? value

def atomsSummary? (batches : List RawBatch) : List RawAtom → Option RawSummary
  | [] => some zeroSummary
  | atom :: atoms => do
      let head ← atomSummary? batches atom
      let tail ← atomsSummary? batches atoms
      some (addSummary head tail)

def templateValid (batches : List RawBatch) (template : RawTemplate) : Bool :=
  template.atoms.length ≤ 32 &&
    atomsSummary? batches template.atoms = some template.summary

def templateChunkValid
    (batches : List RawBatch)
    (context : RawTemplateChunkContext)
    (templates : List RawTemplate) : Bool :=
  templates.length ≤ 8 &&
    (templates.flatMap RawTemplate.atoms).length ≤ 256 &&
    context.templateStop = context.templateStart + templates.length &&
    context.atomCount = (templates.flatMap RawTemplate.atoms).length &&
    templates.all (templateValid batches)

def lookupChunked? {α : Type}
    (chunks : List (List α))
    (chunkSize index : Nat) : Option α := do
  if chunkSize = 0 then none else
  let chunk ← chunks[index / chunkSize]?
  chunk[index % chunkSize]?

def templateAt?
    (templateChunks : List (List RawTemplate))
    (template : Nat) : Option RawTemplate :=
  lookupChunked? templateChunks 8 template

def callAt?
    (callChunks : List (List RawCall))
    (call : Nat) : Option RawCall :=
  lookupChunked? callChunks 256 call

def summarizeCalls?
    (templateChunks : List (List RawTemplate)) :
    Nat → Nat → List RawCall → Option (Nat × Nat)
  | source, final, [] => some (source, final)
  | source, final, call :: calls => do
      if call.sourceStart != source || call.finalStart != final then none else
      let template ← templateAt? templateChunks call.template
      summarizeCalls? templateChunks
        (source + template.summary.sourceColumns)
        (final + template.summary.freshCoordinates)
        calls

def callChunkValid
    (templateChunks : List (List RawTemplate))
    (context : RawCallChunkContext)
    (calls : List RawCall) : Bool :=
  calls.length ≤ 256 &&
    context.callStop = context.callStart + calls.length &&
    summarizeCalls? templateChunks context.sourceStart context.finalStart calls =
      some (context.sourceStop, context.finalStop)

structure GroupCursor where
  group : Nat
  opening : Nat
  direct : Nat
deriving DecidableEq, Repr

def stepOpeningGroup? (cursor : GroupCursor) (entry : RawOpeningGroup) : Option GroupCursor := do
  if entry.length = 0 || entry.openingStart != cursor.opening ||
      entry.directBefore != cursor.direct then none else
  some
    { group := cursor.group + 1
      opening := cursor.opening + entry.length
      direct := cursor.direct + if entry.kind = .direct then entry.length else 0 }

def summarizeOpeningGroups? : GroupCursor → List RawOpeningGroup → Option GroupCursor
  | cursor, [] => some cursor
  | cursor, entry :: entries => do
      let cursor ← stepOpeningGroup? cursor entry
      summarizeOpeningGroups? cursor entries

def openingGroupShardValid
    (context : RawOpeningGroupShardContext)
    (groups : List RawOpeningGroup) : Bool :=
  groups.length ≤ 256 &&
    summarizeOpeningGroups?
      { group := context.groupStart
        opening := context.openingStart
        direct := context.directStart }
      groups =
      some
        { group := context.groupStop
          opening := context.openingStop
          direct := context.directStop }

def openingGroupAt?
    (batches : List RawBatch)
    (shards : List (List RawOpeningGroup))
    (batch group : Nat) : Option RawOpeningGroup := do
  let header ← batches[batch]?
  let shardOffset := group / 256
  if shardOffset ≥ header.groupShardCount then none else
  let shard ← shards[header.groupShardStart + shardOffset]?
  shard[group % 256]?

def atomSourceLength? (batches : List RawBatch) (atom : RawAtom) : Option Nat := do
  let summary ← atomSummary? batches atom
  some summary.sourceColumns

def atomPrefixSummary?
    (batches : List RawBatch)
    (template : RawTemplate)
    (atom : Nat) : Option RawSummary :=
  atomsSummary? batches (template.atoms.take atom)

structure AtomLocation where
  call : Nat
  atom : Nat
  sourceStart : Nat
  finalStart : Nat
  value : RawAtom
deriving DecidableEq, Repr

def atomLocation?
    (batches : List RawBatch)
    (templateChunks : List (List RawTemplate))
    (callChunks : List (List RawCall))
    (call atom : Nat) : Option AtomLocation := do
  let callData ← callAt? callChunks call
  let template ← templateAt? templateChunks callData.template
  let value ← template.atoms[atom]?
  let prior ← atomPrefixSummary? batches template atom
  some
    { call
      atom
      sourceStart := callData.sourceStart + prior.sourceColumns
      finalStart := callData.finalStart + prior.freshCoordinates
      value }

def atomSlotShape? (batches : List RawBatch) (location : AtomLocation) (offset : Nat) : Option (Nat × Bool) := do
  let length ← atomSourceLength? batches location.value
  if offset ≥ length then none else
  match location.value with
  | .direct _ _ width centered => some (width, centered)
  | .decompositionAlias _ _ _ _ _ _ centered => some (1, centered)
  | .equalityAlias _ _ _ _ width centered => some (width, centered)
  | .linearDefinition _ | .traceEliminated _ => none
  | .sisBatch batch => do
      let header ← batches[batch]?
      if offset < 2 then none else
      let offset := offset - 2
      if offset < header.commitmentFields then some (41, false) else
      let offset := offset - header.commitmentFields
      let within := offset % 122
      if within < 41 then some (1, true)
      else if within < 82 then none
      else some (1, false)

def templateChunkAtValid
    (batches : List RawBatch)
    (contexts : List RawTemplateChunkContext)
    (chunks : List (List RawTemplate))
    (index : Nat) : Bool :=
  match contexts[index]?, chunks[index]? with
  | some context, some chunk => templateChunkValid batches context chunk
  | _, _ => false

def callChunkAtValid
    (templateChunks : List (List RawTemplate))
    (contexts : List RawCallChunkContext)
    (chunks : List (List RawCall))
    (index : Nat) : Bool :=
  match contexts[index]?, chunks[index]? with
  | some context, some chunk => callChunkValid templateChunks context chunk
  | _, _ => false

def openingGroupShardAtValid
    (contexts : List RawOpeningGroupShardContext)
    (shards : List (List RawOpeningGroup))
    (index : Nat) : Bool :=
  match contexts[index]?, shards[index]? with
  | some context, some shard => openingGroupShardValid context shard
  | _, _ => false

def ownerLocation?
    (batches : List RawBatch)
    (templateChunks : List (List RawTemplate))
    (callChunks : List (List RawCall)) :
    RawOwner → Option (AtomLocation × Nat)
  | .ordinary call atom offset => do
      let location ← atomLocation? batches templateChunks callChunks call atom
      match location.value with
      | .sisBatch _ => none
      | _ =>
          let length ← atomSourceLength? batches location.value
          if offset < length then some (location, offset) else none
  | .batch call atom batch offset => do
      let location ← atomLocation? batches templateChunks callChunks call atom
      match location.value with
      | .sisBatch actual =>
          if actual != batch then none else
          let length ← atomSourceLength? batches location.value
          if offset < length then some (location, offset) else none
      | _ => none

def consumerLocation?
    (batches : List RawBatch)
    (templateChunks : List (List RawTemplate))
    (callChunks : List (List RawCall))
    (openingGroupShards : List (List RawOpeningGroup)) :
    RawConsumer → Option (AtomLocation × Nat × Option RawOpeningGroup)
  | .ordinary call atom offset => do
      let location ← atomLocation? batches templateChunks callChunks call atom
      match location.value with
      | .decompositionAlias .. | .equalityAlias .. => some (location, offset, none)
      | _ => none
  | .batch call atom batch group offset => do
      let location ← atomLocation? batches templateChunks callChunks call atom
      match location.value with
      | .sisBatch actual =>
          if actual != batch then none else
          let entry ← openingGroupAt? batches openingGroupShards batch group
          match entry.kind with
          | .alias .. => some (location, offset, some entry)
          | .direct => none
      | _ => none

def ownerSlotStride?
    (batches : List RawBatch)
    (location : AtomLocation)
    (offset stride length : Nat) : Option Nat := do
  if length = 0 then none else
  let sourceLength ← atomSourceLength? batches location.value
  let last := offset + stride * (length - 1)
  if last ≥ sourceLength then none else
  match location.value with
  | .direct _ startStride _ _ => some (startStride * stride)
  | .decompositionAlias _ _ _ _ _ startStride _ => some (startStride * stride)
  | .equalityAlias _ _ _ startStride _ _ => some (startStride * stride)
  | .linearDefinition _ | .traceEliminated _ => none
  | .sisBatch batch => do
      let header ← batches[batch]?
      if offset < 2 || last < 2 then none else
      let first := offset - 2
      let last := last - 2
      if first < header.commitmentFields && last < header.commitmentFields then
        some (41 * stride)
      else none

def targetFacts?
    (batches : List RawBatch)
    (templateChunks : List (List RawTemplate))
    (callChunks : List (List RawCall))
    (link : RawAliasLink) : Option (AtomLocation × Nat × Nat × (Nat × Bool) × (Nat × Bool) × Nat) := do
  if link.length = 0 then none else
  let (location, offset) ← ownerLocation? batches templateChunks callChunks link.target
  let lastOffset := offset + link.targetOffsetStride * (link.length - 1)
  let firstShape ← atomSlotShape? batches location offset
  let lastShape ← atomSlotShape? batches location lastOffset
  let stride ← ownerSlotStride? batches location offset link.targetOffsetStride link.length
  some
    (location, offset, lastOffset, firstShape, lastShape, stride)

def aliasLinkValid
    (batches : List RawBatch)
    (templateChunks : List (List RawTemplate))
    (callChunks : List (List RawCall))
    (openingGroupShards : List (List RawOpeningGroup))
    (link : RawAliasLink) : Bool :=
  match
      consumerLocation? batches templateChunks callChunks openingGroupShards link.consumer,
      targetFacts? batches templateChunks callChunks link with
  | some (consumer, offset, group?),
      some (target, targetOffset, targetLastOffset, firstShape, lastShape, targetStride) =>
      let targetColumn := target.sourceStart + targetOffset
      let targetLastColumn := target.sourceStart + targetLastOffset
      match consumer.value, group? with
      | .decompositionAlias length sourceDelta sourceStride digit digitStride startStride _, none =>
          offset + link.length ≤ length &&
          sourceDelta ≤ consumer.sourceStart &&
          targetColumn = consumer.sourceStart - sourceDelta + sourceStride * offset &&
          targetLastColumn = targetColumn + sourceStride * (link.length - 1) &&
          targetLastColumn < consumer.sourceStart + offset + link.length &&
          sourceStride = link.targetOffsetStride &&
          digit + digitStride * (offset + link.length - 1) < firstShape.1 &&
          firstShape.1 = lastShape.1 &&
          startStride = targetStride + digitStride
      | .equalityAlias length sourceDelta sourceStride startStride width centered, none =>
          offset + link.length ≤ length &&
          sourceDelta ≤ consumer.sourceStart &&
          targetColumn = consumer.sourceStart - sourceDelta + sourceStride * offset &&
          targetLastColumn = targetColumn + sourceStride * (link.length - 1) &&
          targetLastColumn < consumer.sourceStart + offset + link.length &&
          sourceStride = link.targetOffsetStride &&
          firstShape = (width, centered) &&
          lastShape = (width, centered) &&
          startStride = targetStride
      | .sisBatch batch, some group =>
          match group.kind with
          | .alias source sourceStride =>
              offset + link.length ≤ group.length &&
              targetColumn = source + sourceStride * offset &&
              targetLastColumn = targetColumn + sourceStride * (link.length - 1) &&
              targetLastColumn <
                consumer.sourceStart + 2 +
                  (batches[batch]?.map RawBatch.commitmentFields).getD 0 +
                  (group.openingStart + offset) * 122 &&
              sourceStride = link.targetOffsetStride &&
              firstShape = (41, false) &&
              lastShape = (41, false)
          | .direct => false
      | _, _ => false
  | _, _ => false

def aliasLinkChunkValid
    (batches : List RawBatch)
    (templateChunks : List (List RawTemplate))
    (callChunks : List (List RawCall))
    (openingGroupShards : List (List RawOpeningGroup))
    (context : RawAliasLinkChunkContext)
    (links : List RawAliasLink) : Bool :=
  links.length ≤ 256 &&
    context.linkStop = context.linkStart + links.length &&
    links.all
      (aliasLinkValid batches templateChunks callChunks openingGroupShards)

def aliasLinkChunkAtValid
    (batches : List RawBatch)
    (templateChunks : List (List RawTemplate))
    (callChunks : List (List RawCall))
    (openingGroupShards : List (List RawOpeningGroup))
    (contexts : List RawAliasLinkChunkContext)
    (chunks : List (List RawAliasLink))
    (index : Nat) : Bool :=
  match contexts[index]?, chunks[index]? with
  | some context, some chunk =>
      aliasLinkChunkValid batches templateChunks callChunks
        openingGroupShards context chunk
  | _, _ => false

def aliasLinkAt?
    (chunks : List (List RawAliasLink))
    (link : Nat) : Option RawAliasLink :=
  lookupChunked? chunks 256 link

def consumerOffset : RawConsumer → Nat
  | .ordinary _ _ offset => offset
  | .batch _ _ _ _ offset => offset

def sameConsumer (left right : RawConsumer) : Bool :=
  match left, right with
  | .ordinary call atom _, .ordinary rightCall rightAtom _ =>
      call = rightCall && atom = rightAtom
  | .batch call atom batch group _,
      .batch rightCall rightAtom rightBatch rightGroup _ =>
      call = rightCall && atom = rightAtom && batch = rightBatch &&
        group = rightGroup
  | _, _ => false

def expectedConsumerLength?
    (batches : List RawBatch)
    (templateChunks : List (List RawTemplate))
    (callChunks : List (List RawCall))
    (openingGroupShards : List (List RawOpeningGroup))
    (consumer : RawConsumer) : Option Nat := do
  if consumerOffset consumer != 0 then none else
  let (location, _, group?) ←
    consumerLocation? batches templateChunks callChunks openingGroupShards
      consumer
  match location.value, group? with
  | .decompositionAlias length .., none => some length
  | .equalityAlias length .., none => some length
  | .sisBatch _, some group => some group.length
  | _, _ => none

def summarizeAliasConsumerLinks?
    (linkChunks : List (List RawAliasLink))
    (consumer : RawConsumer) :
    Nat → Nat → Nat → Option Nat
  | _, offset, 0 => some offset
  | linkIndex, offset, remaining + 1 => do
      let link ← aliasLinkAt? linkChunks linkIndex
      if !sameConsumer link.consumer consumer ||
          consumerOffset link.consumer != offset || link.length = 0 then none else
      summarizeAliasConsumerLinks? linkChunks consumer (linkIndex + 1)
        (offset + link.length) remaining

def aliasConsumerValid
    (batches : List RawBatch)
    (templateChunks : List (List RawTemplate))
    (callChunks : List (List RawCall))
    (openingGroupShards : List (List RawOpeningGroup))
    (linkChunks : List (List RawAliasLink))
    (entry : RawAliasConsumer) : Bool :=
  entry.length != 0 && entry.linkStart < entry.linkStop &&
    entry.linkStop - entry.linkStart ≤ 32 &&
    expectedConsumerLength? batches templateChunks callChunks
      openingGroupShards entry.consumer = some entry.length &&
    summarizeAliasConsumerLinks? linkChunks entry.consumer entry.linkStart 0
      (entry.linkStop - entry.linkStart) = some entry.length

structure AliasConsumerCursor where
  consumer : Nat
  link : Nat
deriving DecidableEq, Repr

def summarizeAliasConsumers?
    (batches : List RawBatch)
    (templateChunks : List (List RawTemplate))
    (callChunks : List (List RawCall))
    (openingGroupShards : List (List RawOpeningGroup))
    (linkChunks : List (List RawAliasLink)) :
    AliasConsumerCursor → List RawAliasConsumer → Option AliasConsumerCursor
  | cursor, [] => some cursor
  | cursor, entry :: entries => do
      if entry.linkStart != cursor.link ||
          !aliasConsumerValid batches templateChunks callChunks
            openingGroupShards linkChunks entry then none else
      summarizeAliasConsumers? batches templateChunks callChunks
        openingGroupShards linkChunks
        { consumer := cursor.consumer + 1, link := entry.linkStop } entries

def aliasConsumerChunkValid
    (batches : List RawBatch)
    (templateChunks : List (List RawTemplate))
    (callChunks : List (List RawCall))
    (openingGroupShards : List (List RawOpeningGroup))
    (linkChunks : List (List RawAliasLink))
    (context : RawAliasConsumerChunkContext)
    (entries : List RawAliasConsumer) : Bool :=
  entries.length ≤ 128 &&
    context.consumerStop = context.consumerStart + entries.length &&
    summarizeAliasConsumers? batches templateChunks callChunks
      openingGroupShards linkChunks
      { consumer := context.consumerStart, link := context.linkStart } entries =
      some { consumer := context.consumerStop, link := context.linkStop }

def aliasConsumerChunkAtValid
    (batches : List RawBatch)
    (templateChunks : List (List RawTemplate))
    (callChunks : List (List RawCall))
    (openingGroupShards : List (List RawOpeningGroup))
    (linkChunks : List (List RawAliasLink))
    (contexts : List RawAliasConsumerChunkContext)
    (chunks : List (List RawAliasConsumer))
    (index : Nat) : Bool :=
  match contexts[index]?, chunks[index]? with
  | some context, some chunk =>
      aliasConsumerChunkValid batches templateChunks callChunks
        openingGroupShards linkChunks context chunk
  | _, _ => false

structure RegistryCursor where
  records : Nat
  source : Nat
  final : Nat
deriving DecidableEq, Repr

def templateContextsSummary? :
    Nat → Nat → Nat → List RawTemplateChunkContext → Option (Nat × Nat)
  | _, templates, atoms, [] => some (templates, atoms)
  | chunkIndex, templates, atoms, context :: contexts => do
      if context.templateStart != templates ||
          context.templateStart != chunkIndex * 8 then none else
      templateContextsSummary? (chunkIndex + 1) context.templateStop
        (atoms + context.atomCount) contexts

def templateRegistryContextValid
    (contexts : List RawTemplateChunkContext)
    (chunks : List (List RawTemplate))
    (templateCount atomCount : Nat) : Bool :=
  contexts.length = chunks.length &&
    templateContextsSummary? 0 0 0 contexts = some (templateCount, atomCount)

def callContextsSummary? :
    Nat → RegistryCursor → List RawCallChunkContext → Option RegistryCursor
  | _, cursor, [] => some cursor
  | chunkIndex, cursor, context :: contexts => do
      if context.callStart != cursor.records ||
          context.callStart != chunkIndex * 256 ||
          context.sourceStart != cursor.source ||
          context.finalStart != cursor.final then none else
      callContextsSummary? (chunkIndex + 1)
        { records := context.callStop
          source := context.sourceStop
          final := context.finalStop }
        contexts

def callRegistryContextValid
    (contexts : List RawCallChunkContext)
    (chunks : List (List RawCall))
    (callCount sourceStart sourceStop finalStart finalStop : Nat) : Bool :=
  contexts.length = chunks.length &&
    callContextsSummary? 0
      { records := 0, source := sourceStart, final := finalStart }
      contexts =
      some { records := callCount, source := sourceStop, final := finalStop }

def openingBatchContextsValid
    (batchIndex : Nat)
    (batch : RawBatch)
    (contexts : List RawOpeningGroupShardContext) : Bool :=
  match contexts with
  | [] => false
  | first :: _ =>
      let rec walk (cursor : GroupCursor) : List RawOpeningGroupShardContext → Option GroupCursor
        | [] => some cursor
        | context :: contexts => do
            if context.batch != batchIndex ||
                context.groupStart != cursor.group ||
                context.openingStart != cursor.opening ||
                context.directStart != cursor.direct then none else
            walk
              { group := context.groupStop
                opening := context.openingStop
                direct := context.directStop }
              contexts
      first.batch = batchIndex &&
        first.groupStart = 0 &&
        first.openingStart = 0 &&
        first.directStart = 0 &&
        match walk { group := 0, opening := 0, direct := 0 } contexts with
        | some stop =>
            stop.opening = batch.openings &&
              stop.direct = batch.directOpenings
        | none => false

def openingRegistryContextValid
    (batches : List RawBatch)
    (contexts : List RawOpeningGroupShardContext)
    (shards : List (List RawOpeningGroup)) : Bool :=
  let rec batchesValid : Nat → List RawBatch → Bool
    | _, [] => true
    | batchIndex, batch :: batches =>
        let selected := contexts.drop batch.groupShardStart |>.take batch.groupShardCount
        batch.groupShardStart + batch.groupShardCount ≤ contexts.length &&
          openingBatchContextsValid batchIndex batch selected &&
          batchesValid (batchIndex + 1) batches
  contexts.length = shards.length && batchesValid 0 batches

def aliasLinkRegistryContextValid
    (contexts : List RawAliasLinkChunkContext)
    (chunks : List (List RawAliasLink))
    (linkCount : Nat) : Bool :=
  let rec walk : Nat → List RawAliasLinkChunkContext → Option Nat
    | cursor, [] => some cursor
    | cursor, context :: contexts => do
        if context.linkStart != cursor then none else
        walk context.linkStop contexts
  contexts.length = chunks.length && walk 0 contexts = some linkCount

def aliasConsumerRegistryContextValid
    (contexts : List RawAliasConsumerChunkContext)
    (chunks : List (List RawAliasConsumer))
    (consumerCount linkCount : Nat) : Bool :=
  let rec walk : Nat → Nat → Nat → List RawAliasConsumerChunkContext →
      Option (Nat × Nat)
    | _, consumer, link, [] => some (consumer, link)
    | chunkIndex, consumer, link, context :: contexts => do
        if context.consumerStart != consumer ||
            context.consumerStart != chunkIndex * 128 ||
            context.linkStart != link then none else
        walk (chunkIndex + 1) context.consumerStop context.linkStop contexts
  contexts.length = chunks.length &&
    walk 0 0 0 contexts = some (consumerCount, linkCount)

end Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.ProductionDomain.PrivateDecoder

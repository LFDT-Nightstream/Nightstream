import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.ProductionDomain.PrivateDecoder.Exact

/-!
Exact source-column ownership for the compact private-decoder program.

Each source column is owned by one instantiated template atom and one offset
inside that atom.  The proof walks only the 14 call-chunk summaries and the
bounded atom lists; it never materializes the 10,997,106 source columns.

Assurance tier: artifact-checked for the generated steady-recursive profile.

Does not own: source values, alias equality, eliminated-definition semantics,
derived products, sparse matrices, CCS/CE membership, or commitment binding.

Emits constraints: none.
-/

namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.ProductionDomain.PrivateDecoder

private def atomOwner (call atom offset : Nat) : RawAtom → RawOwner
  | .sisBatch batch => .batch call atom batch offset
  | _ => .ordinary call atom offset

private def ownerInAtoms?
    (batches : List RawBatch) (call : Nat) :
    Nat → Nat → Nat → List RawAtom → Option RawOwner
  | _, _, _, [] => none
  | atomIndex, source, column, atom :: atoms =>
      match atomSummary? batches atom with
      | none => none
      | some summary =>
          if column < source + summary.sourceColumns then
            if source ≤ column then
              some (atomOwner call atomIndex (column - source) atom)
            else none
          else
            ownerInAtoms? batches call (atomIndex + 1)
              (source + summary.sourceColumns) column atoms

private def ownerInCalls?
    (batches : List RawBatch)
    (templateChunks : List (List RawTemplate)) :
    Nat → Nat → List RawCall → Option RawOwner
  | _, _, [] => none
  | callIndex, column, call :: calls =>
      match templateAt? templateChunks call.template with
      | none => none
      | some template =>
          if column < call.sourceStart + template.summary.sourceColumns then
            ownerInAtoms? batches callIndex 0 call.sourceStart column template.atoms
          else
            ownerInCalls? batches templateChunks (callIndex + 1) column calls

private def ownerInChunks?
    (batches : List RawBatch)
    (templateChunks : List (List RawTemplate)) :
    List RawCallChunkContext → List (List RawCall) → Nat → Option RawOwner
  | [], [], _ => none
  | context :: contexts, calls :: chunks, column =>
      if column < context.sourceStop then
        ownerInCalls? batches templateChunks context.callStart column calls
      else
        ownerInChunks? batches templateChunks contexts chunks column
  | _, _, _ => none

/-- The canonical compact lookup for the generated private source decoder. -/
def generatedSourceOwner? (column : Nat) : Option RawOwner :=
  ownerInChunks? EG.batches EG.templateChunks EG.callChunkContexts
    EG.callChunks column

/-- Ownership means that the canonical compact lookup returns this owner. -/
def GeneratedSourceOwnedBy (column : Nat) (owner : RawOwner) : Prop :=
  generatedSourceOwner? column = some owner

private theorem list_all_of_get?_eq_some {α : Type} (predicate : α → Bool) :
    ∀ {values : List α} {index : Nat} {value : α},
      values.all predicate = true → values[index]? = some value →
        predicate value = true
  | [], _, _, _, lookup => by simp at lookup
  | _ :: _, 0, _, allValid, lookup => by
      have separated := Bool.and_eq_true.mp allValid
      simp at lookup
      subst_vars
      exact separated.1
  | _ :: values, index + 1, value, allValid, lookup => by
      have separated := Bool.and_eq_true.mp allValid
      exact list_all_of_get?_eq_some predicate
        separated.2 lookup

private theorem generatedTemplateChunk_allValid
    {index : Nat} {chunk : List RawTemplate}
    (lookup : EG.templateChunks[index]? = some chunk) :
    chunk.all (templateValid EG.batches) = true := by
  have indexLt : index < 70 := by
    by_cases less : index < 70
    · exact less
    · have indexForm : index = 70 + (index - 70) := by omega
      rw [indexForm] at lookup
      simp [EG.templateChunks] at lookup
  have checked := generatedTemplateChunks_exact ⟨index, indexLt⟩
  cases contextLookup : EG.templateChunkContexts[index]? with
  | none => simp [templateChunkAtValid, contextLookup, lookup] at checked
  | some context =>
      simp [templateChunkAtValid, contextLookup, lookup,
        templateChunkValid] at checked
      exact List.all_eq_true.mpr checked.2

private theorem generatedTemplateLookupSound
    {index : Nat} {template : RawTemplate}
    (lookup : templateAt? EG.templateChunks index = some template) :
    atomsSummary? EG.batches template.atoms = some template.summary := by
  cases chunkLookup : EG.templateChunks[index / 8]? with
  | none => simp [templateAt?, lookupChunked?, chunkLookup] at lookup
  | some chunk =>
      have allValid := generatedTemplateChunk_allValid chunkLookup
      have memberLookup : chunk[index % 8]? = some template := by
        simpa [templateAt?, lookupChunked?, chunkLookup] using lookup
      have valid := list_all_of_get?_eq_some
        (templateValid EG.batches) allValid memberLookup
      exact (by simpa [templateValid] using valid :
        template.atoms.length ≤ 32 ∧
          atomsSummary? EG.batches template.atoms = some template.summary).2

private theorem ownerInAtoms?_coverage
    (batches : List RawBatch) (call atomIndex source column : Nat) :
    ∀ {atoms : List RawAtom} {summary : RawSummary},
      atomsSummary? batches atoms = some summary →
      source ≤ column → column < source + summary.sourceColumns →
      ∃ owner, ownerInAtoms? batches call atomIndex source column atoms = some owner
  | [], _, summaryExact, _, columnLt => by
      simp [atomsSummary?] at summaryExact
      subst_vars
      simp [zeroSummary] at columnLt
  | atom :: atoms, summary, summaryExact, sourceLe, columnLt => by
      cases atomExact : atomSummary? batches atom with
      | none => simp [atomsSummary?, atomExact] at summaryExact
      | some atomSummary =>
          cases tailExact : atomsSummary? batches atoms with
          | none => simp [atomsSummary?, atomExact, tailExact] at summaryExact
          | some tailSummary =>
              simp [atomsSummary?, atomExact, tailExact] at summaryExact
              subst summary
              by_cases inHead : column < source + atomSummary.sourceColumns
              · refine ⟨atomOwner call atomIndex (column - source) atom, ?_⟩
                simp [ownerInAtoms?, atomExact, inHead, sourceLe]
              · have nextLe : source + atomSummary.sourceColumns ≤ column := by omega
                have nextLt :
                    column < source + atomSummary.sourceColumns + tailSummary.sourceColumns := by
                  simpa [addSummary, Nat.add_assoc] using columnLt
                obtain ⟨owner, ownerExact⟩ :=
                  ownerInAtoms?_coverage batches call (atomIndex + 1)
                    (source + atomSummary.sourceColumns) column tailExact nextLe nextLt
                refine ⟨owner, ?_⟩
                simp [ownerInAtoms?, atomExact, inHead, ownerExact]

private theorem ownerInCalls?_coverage
    (batches : List RawBatch)
    (templateChunks : List (List RawTemplate))
    (templateSound : ∀ {index template},
      templateAt? templateChunks index = some template →
      atomsSummary? batches template.atoms = some template.summary)
    (callIndex source final column : Nat) :
    ∀ {calls : List RawCall} {sourceStop finalStop : Nat},
      summarizeCalls? templateChunks source final calls = some (sourceStop, finalStop) →
      source ≤ column → column < sourceStop →
      ∃ owner, ownerInCalls? batches templateChunks callIndex column calls = some owner
  | [], _, _, summaryExact, sourceLe, columnLt => by
      simp [summarizeCalls?] at summaryExact
      omega
  | call :: calls, sourceStop, finalStop, summaryExact, sourceLe, columnLt => by
      cases startExact : decide (call.sourceStart = source) with
      | false =>
          have : call.sourceStart != source := by simpa using startExact
          simp [summarizeCalls?, this] at summaryExact
      | true =>
          have callStart : call.sourceStart = source := by simpa using startExact
          cases finalExact : decide (call.finalStart = final) with
          | false =>
              have : call.finalStart != final := by simpa using finalExact
              simp [summarizeCalls?, callStart, this] at summaryExact
          | true =>
              have callFinal : call.finalStart = final := by simpa using finalExact
              cases templateExact : templateAt? templateChunks call.template with
              | none =>
                  simp [summarizeCalls?, callStart, callFinal, templateExact] at summaryExact
              | some template =>
                  have tailExact := summaryExact
                  simp [summarizeCalls?, callStart, callFinal, templateExact] at tailExact
                  by_cases inHead :
                      column < call.sourceStart + template.summary.sourceColumns
                  · obtain ⟨owner, ownerExact⟩ :=
                      ownerInAtoms?_coverage batches callIndex 0 call.sourceStart column
                        (templateSound templateExact) (by omega) inHead
                    exact ⟨owner, by
                      simp [ownerInCalls?, templateExact, inHead, ownerExact]⟩
                  · have nextLe :
                        call.sourceStart + template.summary.sourceColumns ≤ column := by omega
                    obtain ⟨owner, ownerExact⟩ :=
                      ownerInCalls?_coverage batches templateChunks templateSound
                        (callIndex + 1)
                        (source + template.summary.sourceColumns)
                        (final + template.summary.freshCoordinates) column
                        tailExact (by simpa [callStart] using nextLe) columnLt
                    exact ⟨owner, by
                      simp [ownerInCalls?, templateExact, inHead, ownerExact]⟩

private inductive CallChunksValid (templateChunks : List (List RawTemplate)) :
    List RawCallChunkContext → List (List RawCall) → Prop
  | nil : CallChunksValid templateChunks [] []
  | cons {context calls contexts chunks}
      (head : callChunkValid templateChunks context calls = true)
      (tail : CallChunksValid templateChunks contexts chunks) :
      CallChunksValid templateChunks (context :: contexts) (calls :: chunks)

private theorem ownerInChunks?_coverage
    (batches : List RawBatch)
    (templateChunks : List (List RawTemplate))
    (templateSound : ∀ {index template},
      templateAt? templateChunks index = some template →
      atomsSummary? batches template.atoms = some template.summary) :
    ∀ {chunkIndex : Nat} {cursor : RegistryCursor}
      {contexts : List RawCallChunkContext} {chunks : List (List RawCall)}
      {stop : RegistryCursor} {column : Nat},
      callContextsSummary? chunkIndex cursor contexts = some stop →
      CallChunksValid templateChunks contexts chunks →
      cursor.source ≤ column → column < stop.source →
      ∃ owner, ownerInChunks? batches templateChunks contexts chunks column = some owner
  | _, cursor, [], [], _, _, summaryExact, _, sourceLe, columnLt => by
      simp [callContextsSummary?] at summaryExact
      omega
  | chunkIndex, cursor, context :: contexts, calls :: chunks, stop, column,
      summaryExact, chunksExact, sourceLe, columnLt => by
      have contextStart : context.sourceStart = cursor.source := by
        by_cases equal : context.sourceStart = cursor.source
        · exact equal
        · simp [callContextsSummary?, equal] at summaryExact
      have callStart : context.callStart = cursor.records := by
        by_cases equal : context.callStart = cursor.records
        · exact equal
        · simp [callContextsSummary?, equal] at summaryExact
      have packedStart : context.callStart = chunkIndex * 256 := by
        by_cases equal : context.callStart = chunkIndex * 256
        · exact equal
        · simp [callContextsSummary?, equal] at summaryExact
      have finalStart : context.finalStart = cursor.final := by
        by_cases equal : context.finalStart = cursor.final
        · exact equal
        · simp [callContextsSummary?, equal] at summaryExact
      have tailExact :
          callContextsSummary? (chunkIndex + 1)
            { records := context.callStop, source := context.sourceStop,
              final := context.finalStop } contexts = some stop := by
        simpa [callContextsSummary?, contextStart, callStart, packedStart,
          finalStart] using summaryExact
      cases chunksExact with
      | cons chunkExact restExact =>
          by_cases inHead : column < context.sourceStop
          · have callSummary :
                summarizeCalls? templateChunks context.sourceStart context.finalStart calls =
                  some (context.sourceStop, context.finalStop) := by
              have facts :
                  calls.length ≤ 256 ∧
                  context.callStop = context.callStart + calls.length ∧
                  summarizeCalls? templateChunks context.sourceStart
                    context.finalStart calls =
                    some (context.sourceStop, context.finalStop) := by
                simpa [callChunkValid] using chunkExact
              exact facts.2.2
            obtain ⟨owner, ownerExact⟩ :=
              ownerInCalls?_coverage batches templateChunks templateSound
                context.callStart context.sourceStart context.finalStart column
                callSummary (by omega) inHead
            exact ⟨owner, by
              simp [ownerInChunks?, inHead, ownerExact]⟩
          · have nextLe : context.sourceStop ≤ column := by omega
            obtain ⟨owner, ownerExact⟩ :=
              ownerInChunks?_coverage batches templateChunks templateSound
                tailExact restExact nextLe columnLt
            exact ⟨owner, by
              simp [ownerInChunks?, inHead, ownerExact]⟩
  | _, _, [], _ :: _, _, _, summaryExact, chunksExact, _, _ => by
      cases chunksExact
  | _, _, _ :: _, [], _, _, summaryExact, chunksExact, _, _ => by
      cases chunksExact

private theorem generatedCallChunks_valid :
    CallChunksValid EG.templateChunks EG.callChunkContexts EG.callChunks := by
  have chunkAt (index : Fin 14) := generatedCallChunks_exact index
  simpa [EG.callChunkContexts, EG.callChunks, callChunkAtValid] using
    CallChunksValid.cons (chunkAt 0)
      (CallChunksValid.cons (chunkAt 1)
      (CallChunksValid.cons (chunkAt 2)
      (CallChunksValid.cons (chunkAt 3)
      (CallChunksValid.cons (chunkAt 4)
      (CallChunksValid.cons (chunkAt 5)
      (CallChunksValid.cons (chunkAt 6)
      (CallChunksValid.cons (chunkAt 7)
      (CallChunksValid.cons (chunkAt 8)
      (CallChunksValid.cons (chunkAt 9)
      (CallChunksValid.cons (chunkAt 10)
      (CallChunksValid.cons (chunkAt 11)
      (CallChunksValid.cons (chunkAt 12)
      (CallChunksValid.cons (chunkAt 13) CallChunksValid.nil))))))))))))))

/-- Every source column in the generated private range has a compact atom owner. -/
theorem generatedSourceCoverage (column : Nat)
    (lower : 257 ≤ column) (upper : column < 10997363) :
    ∃ owner : RawOwner, GeneratedSourceOwnedBy column owner := by
  have registryExact := generatedCallRegistryContext_exact
  simp [callRegistryContextValid] at registryExact
  obtain ⟨owner, ownerExact⟩ :=
    ownerInChunks?_coverage EG.batches EG.templateChunks
      (@generatedTemplateLookupSound)
      registryExact.2 generatedCallChunks_valid lower upper
  exact ⟨owner, ownerExact⟩

/-- The canonical compact decoder cannot assign two owners to one source column. -/
theorem generatedSourceOwnerUnique {column : Nat} {left right : RawOwner}
    (leftOwns : GeneratedSourceOwnedBy column left)
    (rightOwns : GeneratedSourceOwnedBy column right) :
    left = right := by
  exact Option.some.inj (leftOwns.symm.trans rightOwns)

end Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.ProductionDomain.PrivateDecoder

import Nightstream.Implementation.R1CS.Correspondence.FieldEncoding.SourceCensus

/-!
Contract: total packed decoder and proof-carrying constructor for generated
fixed-F-prime source-role censuses.

Owns: the fixed eleven-role wire order, direct per-chunk decimal parsing,
cursor-derived `SourceSegment` materialization, fail-closed metadata and run
checks, and the generic proof that one successful check yields a
`SourceCensusArtifact`.

Does not own: generated branch data, the production Rust trace, encoded or CE
coordinate layouts, selector composition, R1CS rows, or permission to remove
rows or coordinates.

Emits constraints: no.

Authority boundary: packed data are non-authoritative structural evidence.
`Data.check` validates their internal source partition and role census. A
separate replay/drift bridge must establish that a concrete generated value
comes from the production Rust trace.

| Surface | Mathematical obligation | Emits constraints? | Assurance tier |
|---|---|---:|---|
| fixed role ABI | role indices decode to one exhaustive `SlotRole` | no | model-level |
| chunk parser | malformed or empty decimal chunks fail without fallback | no | model-level |
| streaming scanner | starts come only from the running cursor; lengths are positive | no | model-level |
| scan invariant | accumulated segments exactly partition `[0, cursor)` and counts are exact | no | model-level |
| `Data.check` | metadata, adjacent run-key separation, final totals, and unique canonical initial constant-one all match | no | artifact check interface |
| `toSourceCensusArtifact` | successful check instantiates the smallest source-census contract | no | model theorem over checked data |
-/

namespace Nightstream.Implementation.R1CS.FPrimeFieldLayout
namespace PackedSourceCensus

/-- Version of the packed natural-number run format accepted by this module. -/
def currentFormatVersion : Nat := 1

/-- Number of roles in the fixed packed-run ABI. -/
def slotRoleCount : Nat := 11

/-- Fixed physical owner of the unique initial constant-one source column. -/
def constantOneOwnerPath : String := "fprime.assignment.constant_one"

/-- Fixed role-index decoder. Generated data cannot redefine this ordering. -/
def slotRoleOfIndex? : Nat → Option SlotRole
  | 0 => some .constantOne
  | 1 => some .ordinaryPrivateField
  | 2 => some .privateBoolean
  | 3 => some .publicBit
  | 4 => some .canonicalU64
  | 5 => some .sisOpening
  | 6 => some .linearlyDerived
  | 7 => some .structuralBalancedAlias
  | 8 => some .gadgetDerived
  | 9 => some .productDerived
  | 10 => some .gadgetTemporary
  | _ => none

/-- Fixed inverse used by generic round-trip checks and role-count indexing. -/
def slotRoleIndex : SlotRole → Nat
  | .constantOne => 0
  | .ordinaryPrivateField => 1
  | .privateBoolean => 2
  | .publicBit => 3
  | .canonicalU64 => 4
  | .sisOpening => 5
  | .linearlyDerived => 6
  | .structuralBalancedAlias => 7
  | .gadgetDerived => 8
  | .productDerived => 9
  | .gadgetTemporary => 10

theorem slotRoleOfIndex_slotRoleIndex (role : SlotRole) :
    slotRoleOfIndex? (slotRoleIndex role) = some role := by
  cases role <;> rfl

/-- Total counts indexed by the fixed role universe. Named fields prevent a
generated array from omitting, duplicating, or reordering a role. -/
structure RoleCounts where
  constantOne : Nat
  ordinaryPrivateField : Nat
  privateBoolean : Nat
  publicBit : Nat
  canonicalU64 : Nat
  sisOpening : Nat
  linearlyDerived : Nat
  structuralBalancedAlias : Nat
  gadgetDerived : Nat
  productDerived : Nat
  gadgetTemporary : Nat
deriving DecidableEq, Repr, Inhabited

namespace RoleCounts

def zero : RoleCounts where
  constantOne := 0
  ordinaryPrivateField := 0
  privateBoolean := 0
  publicBit := 0
  canonicalU64 := 0
  sisOpening := 0
  linearlyDerived := 0
  structuralBalancedAlias := 0
  gadgetDerived := 0
  productDerived := 0
  gadgetTemporary := 0

def get (counts : RoleCounts) : SlotRole → Nat
  | .constantOne => counts.constantOne
  | .ordinaryPrivateField => counts.ordinaryPrivateField
  | .privateBoolean => counts.privateBoolean
  | .publicBit => counts.publicBit
  | .canonicalU64 => counts.canonicalU64
  | .sisOpening => counts.sisOpening
  | .linearlyDerived => counts.linearlyDerived
  | .structuralBalancedAlias => counts.structuralBalancedAlias
  | .gadgetDerived => counts.gadgetDerived
  | .productDerived => counts.productDerived
  | .gadgetTemporary => counts.gadgetTemporary

def increment (counts : RoleCounts) (role : SlotRole) (amount : Nat) :
    RoleCounts :=
  match role with
  | .constantOne =>
      { counts with constantOne := counts.constantOne + amount }
  | .ordinaryPrivateField =>
      { counts with
        ordinaryPrivateField := counts.ordinaryPrivateField + amount }
  | .privateBoolean =>
      { counts with privateBoolean := counts.privateBoolean + amount }
  | .publicBit =>
      { counts with publicBit := counts.publicBit + amount }
  | .canonicalU64 =>
      { counts with canonicalU64 := counts.canonicalU64 + amount }
  | .sisOpening =>
      { counts with sisOpening := counts.sisOpening + amount }
  | .linearlyDerived =>
      { counts with linearlyDerived := counts.linearlyDerived + amount }
  | .structuralBalancedAlias =>
      { counts with
        structuralBalancedAlias := counts.structuralBalancedAlias + amount }
  | .gadgetDerived =>
      { counts with gadgetDerived := counts.gadgetDerived + amount }
  | .productDerived =>
      { counts with productDerived := counts.productDerived + amount }
  | .gadgetTemporary =>
      { counts with gadgetTemporary := counts.gadgetTemporary + amount }

theorem zero_get (role : SlotRole) : zero.get role = 0 := by
  cases role <;> rfl

theorem increment_get (counts : RoleCounts) (updated role : SlotRole)
    (amount : Nat) :
    (counts.increment updated amount).get role =
      counts.get role + if role = updated then amount else 0 := by
  cases updated <;> cases role <;> simp [increment, get]

end RoleCounts

/-- One decoded packed run before its source start is materialized. -/
structure DecodedRun where
  stageIndex : Nat
  ownerPath : String
  role : SlotRole
  length : Nat
deriving DecidableEq, Repr

/-- Identity used to reject non-maximal adjacent runs. -/
structure RunKey where
  stageIndex : Nat
  role : SlotRole
deriving DecidableEq, Repr

namespace DecodedRun

def key (run : DecodedRun) : RunKey :=
  { stageIndex := run.stageIndex, role := run.role }

end DecodedRun

/-- Linear scanner state. Segments are pushed in source order, so their starts
never appear in generated input. -/
structure ScanState where
  cursor : Nat
  segments : Array SourceSegment
  counts : RoleCounts
  runCount : Nat
  previous : Option RunKey
deriving DecidableEq, Repr, Inhabited

namespace ScanState

def initial : ScanState where
  cursor := 0
  segments := #[]
  counts := .zero
  runCount := 0
  previous := none

def append (state : ScanState) (run : DecodedRun) : ScanState :=
  let segment : SourceSegment :=
    { ownerPath := run.ownerPath
      role := run.role
      source := { start := state.cursor, length := run.length } }
  { cursor := state.cursor + run.length
    segments := state.segments.push segment
    counts := state.counts.increment run.role run.length
    runCount := state.runCount + 1
    previous := some run.key }

end ScanState

/-- Generated input plus exact branch-level declarations checked after one
streaming scan. -/
structure Data where
  formatVersion : Nat
  sourceColumnCount : Nat
  runCount : Nat
  declaredRoleCounts : RoleCounts
  stagePaths : Array String
  packedChunks : List String
deriving DecidableEq, Repr

namespace Data

/-- The accepted metadata surface. Empty and duplicate physical-stage names
are rejected before any packed word is interpreted. -/
def MetadataValid (data : Data) : Prop :=
  data.formatVersion = currentFormatVersion ∧
    0 < data.stagePaths.size ∧
    "" ∉ data.stagePaths.toList ∧
    data.stagePaths.toList.Nodup

instance (data : Data) : Decidable data.MetadataValid := by
  unfold MetadataValid
  infer_instance

end Data

/-- Parse exactly one string chunk. The production generator bounds chunk
size; this generic decoder does not make a resource-bound claim. No caller
concatenates or flattens the parsed lists from different chunks. -/
def parseChunk (chunk : String) : Option (List Nat) :=
  if chunk.isEmpty then
    none
  else
    (chunk.splitOn ",").mapM String.toNat?

/-- Decode `(length * 11 + roleIndex) * stageCount + stageIndex`. Every lookup
remains total and fail-closed even though canonical mixed-radix values place
the two indices in range. -/
def decodePackedRun (stagePaths : Array String) (packed : Nat) :
    Option DecodedRun := do
  if stagePaths.size = 0 then
    none
  else
    let stageIndex := packed % stagePaths.size
    let quotient := packed / stagePaths.size
    let roleIndex := quotient % slotRoleCount
    let length := quotient / slotRoleCount
    if length = 0 then
      none
    else do
      let ownerPath ← stagePaths[stageIndex]?
      let role ← slotRoleOfIndex? roleIndex
      some { stageIndex, ownerPath, role, length }

/-- Scan one parsed chunk. This recursion is tail-positioned and never retains
tokens from an earlier chunk. -/
def scanTokens (stagePaths : Array String) :
    List Nat → ScanState → Option ScanState
  | [], state => some state
  | packed :: tail, state => do
      let run ← decodePackedRun stagePaths packed
      if state.previous = some run.key then
        none
      else
        scanTokens stagePaths tail (state.append run)

/-- Parse and scan chunks one at a time. In particular, this does not construct
the global flattened token list used by the general checked-program decoder. -/
def scanChunks (stagePaths : Array String) :
    List String → ScanState → Option ScanState
  | [], state => some state
  | chunk :: tail, state => do
      let tokens ← parseChunk chunk
      let state ← scanTokens stagePaths tokens state
      scanChunks stagePaths tail state

namespace Data

/-- Total decoder. Invalid metadata is rejected before the chunk scanner. -/
def decode (data : Data) : Option ScanState :=
  if data.MetadataValid then
    scanChunks data.stagePaths data.packedChunks .initial
  else
    none

/-- Explicit first-run obligation retained even though the partition proof
already derives its zero start. -/
def StartsWithConstantOne (state : ScanState) : Prop :=
  match state.segments[0]? with
  | none => False
  | some segment =>
      segment.ownerPath = constantOneOwnerPath ∧
        segment.role = .constantOne ∧
        segment.source.start = 0 ∧
        segment.source.length = 1

instance (state : ScanState) : Decidable (StartsWithConstantOne state) := by
  unfold StartsWithConstantOne
  split <;> infer_instance

/-- Exact declarations compared with the single final scanner result. -/
def ResultMatches (data : Data) (state : ScanState) : Prop :=
  state.cursor = data.sourceColumnCount ∧
    state.runCount = data.runCount ∧
    state.counts = data.declaredRoleCounts ∧
    data.declaredRoleCounts.constantOne = 1 ∧
    StartsWithConstantOne state

instance (data : Data) (state : ScanState) :
    Decidable (data.ResultMatches state) := by
  unfold ResultMatches
  infer_instance

/-- One executable certificate. A generated consumer proves this equality once
with `native_decide`; semantic fields are then derived by `check_sound`. -/
def check (data : Data) : Bool :=
  match data.decode with
  | none => false
  | some state => decide (data.ResultMatches state)

/-- Checked result with an unreachable default. `check_sound` proves that the
default cannot influence a `SourceCensusArtifact`. -/
def result (data : Data) : ScanState :=
  data.decode.getD .initial

def sourceSegments (data : Data) : List SourceSegment :=
  data.result.segments.toList

end Data

/-- The single invariant carried through both scanner levels. It connects the
efficient array accumulator to the source-census propositions without asking a
concrete artifact certificate to re-traverse each proposition independently. -/
structure ScanInvariant (state : ScanState) : Prop where
  sourcePartition :
    ExactPartition SourceSegment.source state.cursor state.segments.toList
  roleCensus : ∀ role,
    state.counts.get role = roleRunSubtotal role state.segments.toList
  runCount : state.runCount = state.segments.size

private theorem exactPartitionFrom_append_single
    {Owner : Type}
    {runOf : Owner → CoordinateRun}
    {cursor count : Nat}
    {owners : List Owner}
    (partition : ExactPartitionFrom runOf cursor count owners)
    (next : Owner)
    (nextStart : (runOf next).start = count)
    (nextPositive : 0 < (runOf next).length) :
    ExactPartitionFrom runOf cursor (runOf next).endExclusive
      (owners ++ [next]) := by
  induction owners generalizing cursor with
  | nil =>
      simp only [ExactPartitionFrom] at partition
      subst cursor
      simp [ExactPartitionFrom, nextStart, nextPositive]
  | cons head tail inductionHypothesis =>
      simp only [ExactPartitionFrom] at partition ⊢
      rcases partition with
        ⟨headStart, headPositive, tailPartition⟩
      exact
        ⟨headStart, headPositive,
          inductionHypothesis tailPartition⟩

private theorem roleRunSubtotal_append_single
    (role : SlotRole) (segments : List SourceSegment)
    (next : SourceSegment) :
    roleRunSubtotal role (segments ++ [next]) =
      roleRunSubtotal role segments +
        (if next.role = role then next.source.length else 0) := by
  induction segments with
  | nil => simp [roleRunSubtotal]
  | cons head tail inductionHypothesis =>
      simp [roleRunSubtotal, inductionHypothesis, Nat.add_assoc]

theorem decodePackedRun_length_positive
    {stagePaths : Array String} {packed : Nat} {run : DecodedRun}
    (decoded : decodePackedRun stagePaths packed = some run) :
    0 < run.length := by
  unfold decodePackedRun at decoded
  split at decoded
  · contradiction
  · dsimp only at decoded
    split at decoded
    · contradiction
    · rename_i lengthNonzero
      cases owner : stagePaths[packed % stagePaths.size]? with
      | none => simp [owner] at decoded
      | some ownerPath =>
          cases role :
              slotRoleOfIndex?
                (packed / stagePaths.size % slotRoleCount) with
          | none => simp [owner, role] at decoded
          | some decodedRole =>
              simp [owner, role] at decoded
              subst run
              exact Nat.pos_of_ne_zero lengthNonzero

theorem ScanInvariant.initial : ScanInvariant .initial := by
  constructor
  · rfl
  · intro role
    cases role <;> rfl
  · rfl

theorem ScanInvariant.append
    {state : ScanState} (invariant : ScanInvariant state)
    (run : DecodedRun) (positive : 0 < run.length) :
    ScanInvariant (state.append run) := by
  let next : SourceSegment :=
    { ownerPath := run.ownerPath
      role := run.role
      source := { start := state.cursor, length := run.length } }
  constructor
  · have appended := exactPartitionFrom_append_single
      invariant.sourcePartition next rfl positive
    simpa [ScanState.append, next, CoordinateRun.endExclusive] using appended
  · intro role
    rw [show (state.append run).counts =
        state.counts.increment run.role run.length by
      simp [ScanState.append]]
    rw [RoleCounts.increment_get]
    rw [show (state.append run).segments.toList =
        state.segments.toList ++ [next] by
      simp [ScanState.append, next]]
    rw [roleRunSubtotal_append_single]
    rw [invariant.roleCensus]
    by_cases same : role = run.role
    · subst role
      simp [next]
    · have reverse : run.role ≠ role := by
        intro equal
        exact same equal.symm
      simp [same, reverse, next]
  · simp [ScanState.append, invariant.runCount]

theorem scanTokens_preserves
    {stagePaths : Array String} {tokens : List Nat}
    {initial final : ScanState}
    (invariant : ScanInvariant initial)
    (scanned : scanTokens stagePaths tokens initial = some final) :
    ScanInvariant final := by
  induction tokens generalizing initial with
  | nil =>
      simp only [scanTokens, Option.some.injEq] at scanned
      subst final
      exact invariant
  | cons packed tail inductionHypothesis =>
      cases decoded : decodePackedRun stagePaths packed with
      | none =>
          simp [scanTokens, decoded] at scanned
      | some run =>
          by_cases adjacent : initial.previous = some run.key
          · simp [scanTokens, decoded, adjacent] at scanned
          · have nextInvariant : ScanInvariant (initial.append run) :=
              invariant.append run
                (decodePackedRun_length_positive decoded)
            apply inductionHypothesis nextInvariant
            simpa [scanTokens, decoded, adjacent] using scanned

theorem scanChunks_preserves
    {stagePaths : Array String} {chunks : List String}
    {initial final : ScanState}
    (invariant : ScanInvariant initial)
    (scanned : scanChunks stagePaths chunks initial = some final) :
    ScanInvariant final := by
  induction chunks generalizing initial with
  | nil =>
      simp only [scanChunks, Option.some.injEq] at scanned
      subst final
      exact invariant
  | cons chunk tail inductionHypothesis =>
      cases parsed : parseChunk chunk with
      | none =>
          simp [scanChunks, parsed] at scanned
      | some tokens =>
          cases tokenScan : scanTokens stagePaths tokens initial with
          | none =>
              simp [scanChunks, parsed, tokenScan] at scanned
          | some next =>
              have nextInvariant : ScanInvariant next :=
                scanTokens_preserves invariant tokenScan
              apply inductionHypothesis nextInvariant
              simpa [scanChunks, parsed, tokenScan] using scanned

theorem Data.decode_invariant
    {data : Data} {state : ScanState}
    (decoded : data.decode = some state) : ScanInvariant state := by
  unfold Data.decode at decoded
  split at decoded
  · exact scanChunks_preserves .initial decoded
  · contradiction

theorem Data.decode_metadataValid
    {data : Data} {state : ScanState}
    (decoded : data.decode = some state) : data.MetadataValid := by
  unfold Data.decode at decoded
  split at decoded
  · assumption
  · contradiction

namespace Data

/-- Proposition exported by `check_sound`. It exposes every fact needed by the
source-census constructor plus the metadata and initial-run checks that make
the packed representation fail closed. -/
structure Checked (data : Data) : Prop where
  metadataValid : data.MetadataValid
  decodeSucceeded : data.decode = some data.result
  sourcePartition :
    ExactPartition SourceSegment.source data.sourceColumnCount
      data.sourceSegments
  roleCensusExact : ∀ role,
    data.declaredRoleCounts.get role =
      roleRunSubtotal role data.sourceSegments
  runCountExact : data.sourceSegments.length = data.runCount
  constantOneCountExact : data.declaredRoleCounts.constantOne = 1
  startsWithConstantOne : StartsWithConstantOne data.result

/-- Soundness of the single executable certificate. No recursive partition or
per-role decision procedure is rerun here; both are consequences of the one
generic scanner invariant. -/
theorem check_sound (data : Data) (accepted : data.check = true) :
    data.Checked := by
  cases decoded : data.decode with
  | none =>
      simp [check, decoded] at accepted
  | some state =>
      have decided : decide (data.ResultMatches state) = true := by
        simpa [check, decoded] using accepted
      have resultMatches : data.ResultMatches state :=
        of_decide_eq_true decided
      rcases resultMatches with
        ⟨cursorExact, runCountExact, countsExact, constantOneCountExact,
          startsWithConstantOne⟩
      have invariant : ScanInvariant state := data.decode_invariant decoded
      have resultExact : data.result = state := by
        simp [result, decoded]
      constructor
      · exact data.decode_metadataValid decoded
      · simpa [resultExact] using decoded
      · rw [sourceSegments, resultExact, ← cursorExact]
        exact invariant.sourcePartition
      · intro role
        rw [sourceSegments, resultExact, ← countsExact]
        exact invariant.roleCensus role
      · rw [sourceSegments, resultExact, Array.length_toList,
          ← invariant.runCount, runCountExact]
      · exact constantOneCountExact
      · simpa [resultExact] using startsWithConstantOne

/-- The only public constructor from packed data to the source-census schema. -/
def toSourceCensusArtifact (data : Data) (accepted : data.check = true) :
    SourceCensusArtifact :=
  let checked := data.check_sound accepted
  { sourceColumnCount := data.sourceColumnCount
    sourceSegments := data.sourceSegments
    declaredRoleCount := data.declaredRoleCounts.get
    sourcePartition := checked.sourcePartition
    roleCensusExact := checked.roleCensusExact }

end Data

end PackedSourceCensus
end Nightstream.Implementation.R1CS.FPrimeFieldLayout

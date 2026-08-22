import NightstreamFPrime.Export.Codec
import NightstreamFPrime.Spec.Poseidon2

/-!
Owns the canonical circuit-package schema, its lossless structured codec, and
the Poseidon2 relation identifier. Concrete pilot data belongs to `Pilot`.
The schema stores sparse matrix templates and their exact instantiations; Rust
may expand them but does not choose rows or circuit structure.
-/

namespace NightstreamFPrime.Export.Package

open NightstreamFPrime.Spec
open NightstreamFPrime.Export.Codec

/-- The one fixed Nightstream Goldilocks production profile. -/
structure Profile where
  fieldModulus : Nat
  decompositionBase : Nat
  decompositionDigits : Nat
  decompositionBound : Nat
  freshSources : Nat
  runningSources : Nat
  piRlcInputs : Nat
  piDecChildren : Nat
  ccsMatrices : Nat
  cubeVariables : Nat
deriving Repr

def Profile.format : Format Profile where
  encode := fun value => .array [
    .atom value.fieldModulus,
    .atom value.decompositionBase,
    .atom value.decompositionDigits,
    .atom value.decompositionBound,
    .atom value.freshSources,
    .atom value.runningSources,
    .atom value.piRlcInputs,
    .atom value.piDecChildren,
    .atom value.ccsMatrices,
    .atom value.cubeVariables]
  decode
    | .array [.atom fieldModulus, .atom decompositionBase,
        .atom decompositionDigits, .atom decompositionBound,
        .atom freshSources, .atom runningSources, .atom piRlcInputs,
        .atom piDecChildren, .atom ccsMatrices, .atom cubeVariables] =>
      .ok ⟨fieldModulus, decompositionBase, decompositionDigits,
        decompositionBound, freshSources, runningSources, piRlcInputs,
        piDecChildren, ccsMatrices, cubeVariables⟩
    | _ => .error "invalid production profile"
  decode_encode := by
    intro value
    cases value
    rfl

/-- Poseidon2 parameters that every hash-chain template uses. -/
structure PoseidonSchedule where
  width : Nat
  rate : Nat
  digestLength : Nat
  initialFullRounds : Nat
  partialRounds : Nat
  terminalFullRounds : Nat
  recipesPerPermutation : Nat
  outputLocalStart : Nat
deriving Repr

def PoseidonSchedule.format : Format PoseidonSchedule where
  encode := fun value => .array [
    .atom value.width,
    .atom value.rate,
    .atom value.digestLength,
    .atom value.initialFullRounds,
    .atom value.partialRounds,
    .atom value.terminalFullRounds,
    .atom value.recipesPerPermutation,
    .atom value.outputLocalStart]
  decode
    | .array [.atom width, .atom rate, .atom digestLength,
        .atom initialFullRounds, .atom partialRounds,
        .atom terminalFullRounds, .atom recipesPerPermutation,
        .atom outputLocalStart] =>
      .ok ⟨width, rate, digestLength, initialFullRounds, partialRounds,
        terminalFullRounds, recipesPerPermutation, outputLocalStart⟩
    | _ => .error "invalid Poseidon2 schedule"
  decode_encode := by
    intro value
    cases value
    rfl

/-- One named contiguous segment in the final Spartan column order. -/
structure Segment where
  role : Nat
  start : Nat
  length : Nat
deriving Repr

def Segment.format : Format Segment where
  encode := fun value => .array [
    .atom value.role, .atom value.start, .atom value.length]
  decode
    | .array [.atom role, .atom start, .atom length] =>
      .ok ⟨role, start, length⟩
    | _ => .error "invalid layout segment"
  decode_encode := by
    intro value
    cases value
    rfl

/-- Complete physical layout after the proved Lean-to-Spartan permutation. -/
structure PhysicalLayout where
  rowCount : Nat
  privateColumnCount : Nat
  constantColumn : Nat
  publicColumnCount : Nat
  totalColumnCount : Nat
  privateSegments : List Segment
  publicSegments : List Segment
deriving Repr

def PhysicalLayout.format : Format PhysicalLayout where
  encode := fun value => .array [
    .atom value.rowCount,
    .atom value.privateColumnCount,
    .atom value.constantColumn,
    .atom value.publicColumnCount,
    .atom value.totalColumnCount,
    (list Segment.format).encode value.privateSegments,
    (list Segment.format).encode value.publicSegments]
  decode
    | .array [.atom rowCount, .atom privateColumnCount,
        .atom constantColumn, .atom publicColumnCount,
        .atom totalColumnCount, privateSegments, publicSegments] => do
      pure ⟨rowCount, privateColumnCount, constantColumn,
        publicColumnCount, totalColumnCount,
        ← (list Segment.format).decode privateSegments,
        ← (list Segment.format).decode publicSegments⟩
    | _ => .error "invalid physical layout"
  decode_encode := by
    intro value
    cases value
    simp [Format.decode_encode]
    rfl

/-- A sparse template column is one of eight invocation inputs or one local
witness column. -/
inductive ColumnRef where
  | input (index : Nat)
  | local (index : Nat)
deriving Repr

def ColumnRef.format : Format ColumnRef where
  encode
    | .input index => .array [.atom 0, .atom index]
    | .local index => .array [.atom 1, .atom index]
  decode
    | .array [.atom 0, .atom index] => .ok (.input index)
    | .array [.atom 1, .atom index] => .ok (.local index)
    | _ => .error "invalid template column reference"
  decode_encode := by
    intro value
    cases value <;> rfl

structure TemplateTerm where
  column : ColumnRef
  coefficient : Nat
deriving Repr

def TemplateTerm.format : Format TemplateTerm where
  encode := fun value => .array [
    ColumnRef.format.encode value.column, .atom value.coefficient]
  decode
    | .array [column, .atom coefficient] => do
      pure ⟨← ColumnRef.format.decode column, coefficient⟩
    | _ => .error "invalid template term"
  decode_encode := by
    intro value
    cases value
    simp [ColumnRef.format.decode_encode]
    rfl

structure TemplateCombination where
  constant : Nat
  terms : List TemplateTerm
deriving Repr

def TemplateCombination.format : Format TemplateCombination where
  encode := fun value => .array [
    .atom value.constant, (list TemplateTerm.format).encode value.terms]
  decode
    | .array [.atom constant, terms] => do
      pure ⟨constant, ← (list TemplateTerm.format).decode terms⟩
    | _ => .error "invalid template linear combination"
  decode_encode := by
    intro value
    cases value
    simp [Format.decode_encode]
    rfl

/-- One sparse template equation. `outputLocal` is the witness instruction:
the interpreter writes `A · B` there, and the same row checks the result. -/
structure TemplateRow where
  outputLocal : Nat
  a : TemplateCombination
  b : TemplateCombination
  c : TemplateCombination
deriving Repr

def TemplateRow.format : Format TemplateRow where
  encode := fun value => .array [
    .atom value.outputLocal,
    TemplateCombination.format.encode value.a,
    TemplateCombination.format.encode value.b,
    TemplateCombination.format.encode value.c]
  decode
    | .array [.atom outputLocal, a, b, c] => do
      pure ⟨outputLocal,
        ← TemplateCombination.format.decode a,
        ← TemplateCombination.format.decode b,
        ← TemplateCombination.format.decode c⟩
    | _ => .error "invalid template row"
  decode_encode := by
    intro value
    cases value
    simp [TemplateCombination.format.decode_encode]
    rfl

/-- Sparse matrix data for one Poseidon2 permutation. -/
structure PermutationTemplate where
  inputCount : Nat
  localColumnCount : Nat
  outputLocalStart : Nat
  rows : List TemplateRow
deriving Repr

def PermutationTemplate.format : Format PermutationTemplate where
  encode := fun value => .array [
    .atom value.inputCount,
    .atom value.localColumnCount,
    .atom value.outputLocalStart,
    (list TemplateRow.format).encode value.rows]
  decode
    | .array [.atom inputCount, .atom localColumnCount,
        .atom outputLocalStart, rows] => do
      pure ⟨inputCount, localColumnCount, outputLocalStart,
        ← (list TemplateRow.format).decode rows⟩
    | _ => .error "invalid permutation template"
  decode_encode := by
    intro value
    cases value
    simp [Format.decode_encode]
    rfl

/-- One exact instantiation chain. Absorb permutations precede the one final
padding permutation; four digest-equality rows follow the chain. -/
structure HashChain where
  phase : Nat
  rowStart : Nat
  rowCount : Nat
  inputStart : Nat
  inputLength : Nat
  witnessStart : Nat
  witnessLength : Nat
  absorbCount : Nat
  digestStart : Nat
deriving Repr

def HashChain.format : Format HashChain where
  encode := fun value => .array [
    .atom value.phase,
    .atom value.rowStart,
    .atom value.rowCount,
    .atom value.inputStart,
    .atom value.inputLength,
    .atom value.witnessStart,
    .atom value.witnessLength,
    .atom value.absorbCount,
    .atom value.digestStart]
  decode
    | .array [.atom phase, .atom rowStart, .atom rowCount,
        .atom inputStart, .atom inputLength, .atom witnessStart,
        .atom witnessLength, .atom absorbCount, .atom digestStart] =>
      .ok ⟨phase, rowStart, rowCount, inputStart, inputLength,
        witnessStart, witnessLength, absorbCount, digestStart⟩
    | _ => .error "invalid hash chain"
  decode_encode := by
    intro value
    cases value
    rfl

structure SparseTerm where
  column : Nat
  coefficient : Nat
deriving Repr

def SparseTerm.format : Format SparseTerm where
  encode := fun value => .array [.atom value.column, .atom value.coefficient]
  decode
    | .array [.atom column, .atom coefficient] => .ok ⟨column, coefficient⟩
    | _ => .error "invalid sparse term"
  decode_encode := by
    intro value
    cases value
    rfl

structure SparseCombination where
  constant : Nat
  terms : List SparseTerm
deriving Repr

def SparseCombination.format : Format SparseCombination where
  encode := fun value => .array [
    .atom value.constant, (list SparseTerm.format).encode value.terms]
  decode
    | .array [.atom constant, terms] => do
      pure ⟨constant, ← (list SparseTerm.format).decode terms⟩
    | _ => .error "invalid sparse linear combination"
  decode_encode := by
    intro value
    cases value
    simp [Format.decode_encode]
    rfl

/-- An assertion-only absolute sparse row. Witness rows live in the template. -/
structure SparseRow where
  rowIndex : Nat
  a : SparseCombination
  b : SparseCombination
  c : SparseCombination
deriving Repr

def SparseRow.format : Format SparseRow where
  encode := fun value => .array [
    .atom value.rowIndex,
    SparseCombination.format.encode value.a,
    SparseCombination.format.encode value.b,
    SparseCombination.format.encode value.c]
  decode
    | .array [.atom rowIndex, a, b, c] => do
      pure ⟨rowIndex, ← SparseCombination.format.decode a,
        ← SparseCombination.format.decode b,
        ← SparseCombination.format.decode c⟩
    | _ => .error "invalid sparse row"
  decode_encode := by
    intro value
    cases value
    simp [SparseCombination.format.decode_encode]
    rfl

/-- Terminal metadata becomes present when the terminal phase is added to the
same package path. -/
structure TerminalLayout where
  rowStart : Nat
  rowCount : Nat
  runningClaims : Nat
  freshClaims : Nat
deriving Repr

def TerminalLayout.format : Format TerminalLayout where
  encode := fun value => .array [.atom value.rowStart, .atom value.rowCount,
    .atom value.runningClaims, .atom value.freshClaims]
  decode
    | .array [.atom rowStart, .atom rowCount, .atom runningClaims,
        .atom freshClaims] => .ok ⟨rowStart, rowCount, runningClaims,
          freshClaims⟩
    | _ => .error "invalid terminal layout"
  decode_encode := by
    intro value
    cases value
    rfl

/-- The only circuit-package type accepted by the Stage 1 Rust loader. -/
structure CircuitPackage where
  schemaVersion : Nat
  profile : Profile
  poseidon : PoseidonSchedule
  layout : PhysicalLayout
  permutation : PermutationTemplate
  hashChains : List HashChain
  assertionRows : List SparseRow
  terminal : Option TerminalLayout
deriving Repr

def CircuitPackage.format : Format CircuitPackage where
  encode := fun value => .array [
    .atom value.schemaVersion,
    Profile.format.encode value.profile,
    PoseidonSchedule.format.encode value.poseidon,
    PhysicalLayout.format.encode value.layout,
    PermutationTemplate.format.encode value.permutation,
    (list HashChain.format).encode value.hashChains,
    (list SparseRow.format).encode value.assertionRows,
    (option TerminalLayout.format).encode value.terminal]
  decode
    | .array [.atom schemaVersion, profile, poseidon, layout, permutation,
        hashChains, assertionRows, terminal] => do
      pure ⟨schemaVersion,
        ← Profile.format.decode profile,
        ← PoseidonSchedule.format.decode poseidon,
        ← PhysicalLayout.format.decode layout,
        ← PermutationTemplate.format.decode permutation,
        ← (list HashChain.format).decode hashChains,
        ← (list SparseRow.format).decode assertionRows,
        ← (option TerminalLayout.format).decode terminal⟩
    | _ => .error "invalid circuit package"
  decode_encode := by
    intro value
    cases value
    simp [Profile.format.decode_encode, PoseidonSchedule.format.decode_encode,
      PhysicalLayout.format.decode_encode,
      PermutationTemplate.format.decode_encode, Format.decode_encode,
      TerminalLayout.format.decode_encode]
    rfl

theorem decode_encode (value : CircuitPackage) :
    CircuitPackage.format.decode (CircuitPackage.format.encode value) =
      .ok value :=
  CircuitPackage.format.decode_encode value

def limbBase : Nat := 4294967296

def atomDigest (value : Nat) : List F :=
  Poseidon2.hash [Poseidon2.ofNat 0,
    Poseidon2.ofNat (value % limbBase),
    Poseidon2.ofNat ((value / limbBase) % limbBase),
    Poseidon2.ofNat (value / (limbBase * limbBase))]

def foldChildDigest (state child : List F) : List F :=
  Poseidon2.hash (Poseidon2.ofNat 2 :: state ++ child)

def valueDigest : Value → List F
  | .atom value => atomDigest value
  | .array values =>
      values.foldl (fun state child =>
        foldChildDigest state (valueDigest child))
        (Poseidon2.hash [Poseidon2.ofNat 1, Poseidon2.ofNat values.length])

def identityDomain : List F :=
  [78, 105, 103, 104, 116, 115, 116, 114, 101, 97, 109, 47,
    70, 80, 114, 105, 109, 101, 47, 112, 97, 99, 107, 97, 103,
    101, 47, 118, 49]

/-- Verifier-bound Poseidon2 identity of every canonical package field. -/
def relationIdentifier (value : CircuitPackage) : List F :=
  Poseidon2.hash (identityDomain ++
    valueDigest (CircuitPackage.format.encode value))

def render (value : CircuitPackage) : String :=
  (CircuitPackage.format.encode value).render

/-- Emitted wrapper. The identifier is outside its own Poseidon2 preimage and
is recomputed by the verifier from `package`. -/
structure Artifact where
  package : CircuitPackage
  claimedIdentifier : List Nat
deriving Repr

def Artifact.format : Format Artifact where
  encode := fun value => .array [
    CircuitPackage.format.encode value.package,
    (list nat).encode value.claimedIdentifier]
  decode
    | .array [package, claimedIdentifier] => do
      pure ⟨← CircuitPackage.format.decode package,
        ← (list nat).decode claimedIdentifier⟩
    | _ => .error "invalid sealed circuit package"
  decode_encode := by
    intro value
    cases value
    simp [CircuitPackage.format.decode_encode, Format.decode_encode]
    rfl

def sealPackage (value : CircuitPackage) : Artifact where
  package := value
  claimedIdentifier := (relationIdentifier value).map (fun word => word.val)

theorem sealPackage_identifier (value : CircuitPackage) :
    (sealPackage value).claimedIdentifier =
      (relationIdentifier value).map (fun word => word.val) := by
  rfl

theorem artifact_decode_encode (value : Artifact) :
    Artifact.format.decode (Artifact.format.encode value) = .ok value :=
  Artifact.format.decode_encode value

def Artifact.render (value : Artifact) : String :=
  (Artifact.format.encode value).render

end NightstreamFPrime.Export.Package

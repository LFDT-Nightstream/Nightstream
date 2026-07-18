import Nightstream.Implementation.R1CS.Core.Poseidon2Call
import Nightstream.Implementation.R1CS.Correspondence.Poseidon2.Sbox7Compact

/-!
Contract: exact model of the 86 final `x^7` wires inside the isolated
600-row production Poseidon2 permutation.

Owns: the three compact site families (initial external, internal, terminal
external), their exact four-row topological definitions, their eight direct
consumer rows, the exhaustive local RHS-use census, and transport through a
`Poseidon2Call.Call.columnMap`.

Does not own: generation of the 600-row artifact, the recursive-program call
manifest, proof that there are 422 production calls, global column freshness,
centered-slot allocation, Rust conformance, or authorization to remove any
row or slot.

Emits constraints: no. This module classifies and checks existing rows.

Authority boundary: all isolated-layout claims are checked against the
generated 600-row artifact. A call-site theorem transports those rows through
the existing column map, but a generated manifest must still prove that each
call matches the global program and that its allocated columns are fresh.

Assurance tier: artifact-checked for the isolated permutation; model-level for
the generic call transport.

Rows in this file are zero-based indices into the 600-row artifact. A
"consumer" is a definition whose normalized RHS references the S-box output;
the defining row's output occurrence is deliberately not a source use.

| Predicate/theorem | Mathematical obligation | Guarantee | Assumptions | Permits row/slot removal? |
|---|---|---|---|---|
| `family_census` | Poseidon2 round schedule | `32 + 22 + 32 = 86` sites | generated width-8 artifact | no |
| `outputs_exact` | output enumeration | 86 distinct output columns, exactly the three families | compact arithmetic families | no |
| `topologicalDefinitions_exact` | source S-box schedule | every site is the exact affine `x2/x4/x6/x7` four-row schedule | generated definitions | no |
| `compactGate_emitted_iff_existsTopological` | compact S-box semantics | each concrete gate is equivalent to existence of the four-step witness | verifier-fixed one | no |
| `sourceUses_exact` | output connectivity | every output has exactly its eight named direct consumer rows and no other local RHS use | all 600 definitions | no |
| `outputs_private_to_permutation` | ABI separation | S-box outputs are neither permutation inputs nor final outputs | generated input/output ABI | no |
| `poseidon2Call_transport` | call-site layout | all four topological definition rows and eight consumer rows transport through `Call.columnMap` | exact isolated row map | no |
| `mappedOutputColumn_eq` | call-site columns | every mapped output is in the call's fresh derived interval formula | site membership | no |
-/

namespace Nightstream.Implementation.R1CS.Poseidon2Sbox7OutputLayout

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Program

namespace Permutation

abbrev definitions :=
  Nightstream.Implementation.R1CS.Poseidon2Permutation.definitions

abbrev rows :=
  Nightstream.Implementation.R1CS.Poseidon2Permutation.rows

abbrev inputColumns :=
  Nightstream.Implementation.R1CS.Poseidon2Permutation.inputColumns

abbrev outputColumns :=
  Nightstream.Implementation.R1CS.Poseidon2Permutation.outputColumns

end Permutation

/-- The three protocol phases containing affine-input S-boxes. -/
inductive Family where
  | initialExternal
  | internal
  | terminalExternal
deriving DecidableEq, Repr

/-- One final `x^7` wire and the eight rows that directly consume it.
`round` and `lane` are local schedule coordinates, not global manifest data. -/
structure Site where
  family : Family
  round : Nat
  lane : Nat
  outputColumn : Nat
  consumerRows : List Nat
deriving DecidableEq, Repr

/-- The artifact defines columns 9..608 in order, so column `c` is defined by
zero-based row `c - 9`. -/
def definitionRowOfColumn (column : Nat) : Nat :=
  column - 9

def externalSites (family : Family) (outputBase consumerBase : Nat) :
    List Site :=
  (List.range 4).flatMap fun round =>
    (List.range 8).map fun lane =>
      { family
        round
        lane
        outputColumn := outputBase + 40 * round + 4 * lane
        consumerRows :=
          (List.range 8).map fun consumer =>
            definitionRowOfColumn (consumerBase + 40 * round + consumer) }

/-- Four initial external rounds, eight S-boxes per round. -/
def initialExternalSites : List Site :=
  externalSites .initialExternal 20 49

/-- Twenty-two internal rounds, one S-box per round. -/
def internalSites : List Site :=
  (List.range 22).map fun round =>
    { family := .internal
      round
      lane := 0
      outputColumn := 180 + 12 * round
      consumerRows :=
        (List.range 8).map fun consumer =>
          definitionRowOfColumn (181 + 12 * round + consumer) }

/-- Four terminal external rounds, eight S-boxes per round. -/
def terminalExternalSites : List Site :=
  externalSites .terminalExternal 444 473

/-- Complete compact site schedule. -/
def sites : List Site :=
  initialExternalSites ++ internalSites ++ terminalExternalSites

def outputColumns : List Nat :=
  sites.map Site.outputColumn

theorem isolatedProgram_census :
    Permutation.definitions.length = 600 ∧
      Permutation.rows.length = 600 := by
  native_decide

theorem family_census :
    initialExternalSites.length = 32 ∧
      internalSites.length = 22 ∧
      terminalExternalSites.length = 32 ∧
      sites.length = 86 := by
  native_decide

/-- The three arithmetic families are the exact 86-site enumeration and no
output occurs twice. -/
theorem outputs_exact :
    outputColumns =
        (initialExternalSites.map Site.outputColumn ++
          internalSites.map Site.outputColumn ++
          terminalExternalSites.map Site.outputColumn) ∧
      outputColumns.length = 86 ∧
      outputColumns.Nodup := by
  native_decide

theorem outputColumns_bounds :
    ∀ column ∈ outputColumns, 9 ≤ column ∧ column < 609 := by
  native_decide

/-- No internal S-box output is part of the nine-column input ABI or the
eight-column final output ABI. -/
theorem outputs_private_to_permutation :
    ∀ column ∈ outputColumns,
      column ∉ Permutation.inputColumns ∧
        column ∉ Permutation.outputColumns := by
  native_decide

def definitionAtColumn (column : Nat) : Option Definition :=
  Permutation.definitions[definitionRowOfColumn column]?

def rowAt (row : Nat) : Option Row :=
  Permutation.rows[row]?

/-- Extract the affine input LC from the site's `x2` definition. The exact
shape theorem below rejects the empty fallback. -/
def Site.affineInput (site : Site) : List (Nat × Nat) :=
  match definitionAtColumn (site.outputColumn - 3) with
  | some ⟨_, .product left _⟩ => left
  | _ => []

/-- The production S-box input is one state wire plus one constant-one term. -/
def AffineInputTerms : List (Nat × Nat) → Bool
  | [(source, 1), (0, constant)] =>
      source != 0 && constant < goldilocksP
  | _ => false

def Site.x2Column (site : Site) : Nat := site.outputColumn - 3
def Site.x4Column (site : Site) : Nat := site.outputColumn - 2
def Site.x6Column (site : Site) : Nat := site.outputColumn - 1

/-- Exact four normalized definitions emitted by `enforce_sbox_x7` for one
site. This is the concrete layout premise needed to instantiate
`Sbox7Compact.TopologicalHolds`; it does not itself authorize replacement. -/
def Site.TopologicalDefinitionsExact (site : Site) : Prop :=
  AffineInputTerms site.affineInput = true ∧
    definitionAtColumn site.x2Column =
      some ⟨site.x2Column,
        .product site.affineInput site.affineInput⟩ ∧
    definitionAtColumn site.x4Column =
      some ⟨site.x4Column,
        .product [(site.x2Column, 1)] [(site.x2Column, 1)]⟩ ∧
    definitionAtColumn site.x6Column =
      some ⟨site.x6Column,
        .product [(site.x2Column, 1)] [(site.x4Column, 1)]⟩ ∧
    definitionAtColumn site.outputColumn =
      some ⟨site.outputColumn,
        .product site.affineInput [(site.x6Column, 1)]⟩

instance (site : Site) : Decidable site.TopologicalDefinitionsExact := by
  unfold Site.TopologicalDefinitionsExact
  infer_instance

/-- All 86 sites have the exact affine four-row source schedule. -/
theorem topologicalDefinitions_exact :
    ∀ site ∈ sites, site.TopologicalDefinitionsExact := by
  native_decide

/-- Exact compact semantic gate associated with one concrete source site.
Selector column zero is the verifier-owned constant-one wire. -/
def Site.compactGate (site : Site) :
    Poseidon2Sbox7Compact.Gate where
  selector := [(0, 1)]
  input := site.affineInput
  output := [(site.outputColumn, 1)]

/-- Site-specialized form of the abstract compact/topological equivalence. -/
theorem compactGate_emitted_iff_existsTopological
    (assignment : Nat → Nat) (site : Site)
    (one : assignment 0 = 1) :
    Poseidon2Sbox7Compact.EmittedRowHolds assignment site.compactGate ↔
      ∃ witness,
        Poseidon2Sbox7Compact.TopologicalHolds
          (Poseidon2Sbox7Compact.inputValue assignment site.compactGate)
          (Poseidon2Sbox7Compact.outputValue assignment site.compactGate)
          witness := by
  have selectorOne :
      Poseidon2Sbox7Compact.selectorValue assignment site.compactGate = 1 := by
    simp [Poseidon2Sbox7Compact.selectorValue, Site.compactGate, lcEval,
      one, goldilocksP]
  exact Poseidon2Sbox7Compact.emittedRowHolds_iff_exists_topological
    assignment site.compactGate selectorOne

/-- The final source definition row is exactly the fourth product in the
topological schedule. -/
theorem outputDefinitionRows_exact :
    ∀ site ∈ sites,
      rowAt (definitionRowOfColumn site.outputColumn) =
        some (Definition.builderRow
          ⟨site.outputColumn,
            .product site.affineInput [(site.x6Column, 1)]⟩) := by
  native_decide

def definitionUsesColumn (column row : Nat) : Bool :=
  match Permutation.definitions[row]? with
  | some definition => decide (column ∈ definition.rhs.refs)
  | none => false

/-- Exhaustive zero-based row census of normalized RHS references to a
column across the complete isolated permutation program. -/
def sourceUseRows (column : Nat) : List Nat :=
  (List.range Permutation.definitions.length).filter
    (definitionUsesColumn column)

/-- Every final S-box output has exactly eight direct consumers, and the
listed rows exhaust every local source use. -/
theorem sourceUses_exact :
    ∀ site ∈ sites,
      site.consumerRows.length = 8 ∧
        sourceUseRows site.outputColumn = site.consumerRows := by
  native_decide

theorem sourceUse_iff_consumerRow
    {site : Site} (siteMember : site ∈ sites) (row : Nat) :
    row ∈ sourceUseRows site.outputColumn ↔
      row ∈ site.consumerRows := by
  rw [(sourceUses_exact site siteMember).2]

/-- Exact four source rows of the topological `x2/x4/x6/x7` schedule. -/
def Site.topologicalRows (site : Site) : List Nat :=
  [definitionRowOfColumn site.x2Column,
    definitionRowOfColumn site.x4Column,
    definitionRowOfColumn site.x6Column,
    definitionRowOfColumn site.outputColumn]

/-- Compact list of rows whose identities a call-site bridge needs: all four
topological definitions followed by all direct consumer rows. -/
def Site.authorityRows (site : Site) : List Nat :=
  site.topologicalRows ++ site.consumerRows

theorem authorityRows_census :
    ∀ site ∈ sites,
      site.topologicalRows.length = 4 ∧
        site.authorityRows.length = 12 := by
  native_decide

theorem authorityRows_bounds :
    ∀ site ∈ sites, ∀ row ∈ site.authorityRows, row < 600 := by
  native_decide

/-- A target row list is an exact column-renamed image of the isolated rows at
all S-box authority indices. Row positions are preserved by renaming. -/
def TransportedLayout
    (columnMap : Nat → Nat) (targetRows : List Row) : Prop :=
  ∀ site ∈ sites,
    ∀ row ∈ site.authorityRows,
      targetRows[row]? =
        (rowAt row).map (renameRow columnMap)

/-- Any exact row-wise column renaming preserves all 86 definition/consumer
row identities. -/
theorem rowsMap_transport (columnMap : Nat → Nat) :
    TransportedLayout columnMap
      (Permutation.rows.map (renameRow columnMap)) := by
  intro site _ row _
  simp [rowAt]

/-- Reusable call-site theorem. A generated global-program manifest can pair
this with `Call.Matches` and call freshness instead of expanding 86 facts per
call. -/
theorem poseidon2Call_transport (call : Poseidon2Call.Call) :
    TransportedLayout call.columnMap call.rows := by
  simpa [Poseidon2Call.Call.rows] using rowsMap_transport call.columnMap

def mappedOutputColumns (call : Poseidon2Call.Call) : List Nat :=
  outputColumns.map call.columnMap

theorem mappedOutputColumns_length (call : Poseidon2Call.Call) :
    (mappedOutputColumns call).length = 86 := by
  simp [mappedOutputColumns, outputs_exact.2.1]

/-- Every isolated S-box output maps into the derived contiguous interval of
the call; it never takes the input-column branch of `Call.columnMap`. -/
theorem mappedOutputColumn_eq
    (call : Poseidon2Call.Call) {column : Nat}
    (member : column ∈ outputColumns) :
    call.columnMap column =
      call.firstAllocatedColumn + (column - 9) := by
  have bounds := outputColumns_bounds column member
  have nonzero : column ≠ 0 := by omega
  have notInput : ¬ column < 9 := Nat.not_lt_of_ge bounds.1
  simp [Poseidon2Call.Call.columnMap, nonzero, notInput]

/-- The derived-interval branch is injective on the 86 isolated outputs. -/
theorem mappedOutputColumn_injective
    (call : Poseidon2Call.Call)
    {left right : Nat}
    (leftMember : left ∈ outputColumns)
    (rightMember : right ∈ outputColumns)
    (equal : call.columnMap left = call.columnMap right) :
    left = right := by
  rw [mappedOutputColumn_eq call leftMember,
    mappedOutputColumn_eq call rightMember] at equal
  have leftBounds := outputColumns_bounds left leftMember
  have rightBounds := outputColumns_bounds right rightMember
  omega

end Nightstream.Implementation.R1CS.Poseidon2Sbox7OutputLayout

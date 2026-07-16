import Nightstream.Implementation.R1CS.Core.ConstantPins
import Nightstream.Implementation.R1CS.Correspondence.PiCcsOutputDigest.EncodingSchedule

/-!
Typed source-position layout for the terminal `Pi_CCS` output-digest preimage.

Assurance tier: implementation/R1CS correspondence. `Semantics` independently
fixes the mathematical serialization; this file gives every serialized
position a typed role and proves that the production source columns occur in
exactly that order. Generated row spans and profiler totals are irrelevant to
the theorem statements.

Owns: protocol-role -> serialized-position -> production-column mapping;
explicit affine formulas for all dynamic `K` limbs; all 203 semantic constant
positions; the two primary-SIS shape constants; and derivation of those 205
values from accepted ordinary equations.

Does not own: proof that dynamic source columns are constrained `Pi_CCS`
outputs; canonical digit uniqueness; seeded-matrix/Rust conformance; the
Poseidon2 envelope; transcript placement; row necessity; row removal; or cost
totals.

Emits constraints: no.

Authority boundary: role names and expected constants come from independent
message semantics. Concrete columns remain implementation objects. The exact
layout equality below checks their connection; it does not infer that the
dynamic columns already have `Pi_CCS` semantic authority.

| Protocol | Phase | Constraint family | Typed branch | Exact obligation |
|---|---|---|---|---|
| `Pi_CCS` | output serialization | outer header | outer tag/count | seven packed tag words followed by output count 15 |
| `Pi_CCS` | per-output serialization | inner header | output/tag/shape | eight tag words, row count 3, and four widths 54 |
| `Pi_CCS` | evaluation payload | `y_ring` | output/row/coefficient/limb | `c0,c1` columns for three by 54 `K` values |
| `Pi_CCS` | evaluation payload | `y_zcol` | output/coefficient/limb | `c0,c1` columns for 54 `K` values |
| `Pi_CCS` | primary SIS | shape pins | dimension/kappa | exact values 54 and 2 are equation-bound |
-/

namespace Nightstream.Implementation.R1CS.PiCcsOutputDigest.SourceLayout

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.OwnerCertificate

set_option maxRecDepth 1048576
set_option maxHeartbeats 8000000

/-- One mathematical role for every field in `serializeTerminalOutputs`. -/
inductive SourceRole where
  | outerTag (word : Fin 7)
  | outerCount
  | innerTag (output : Fin Semantics.outputCount) (word : Fin 8)
  | yRingRows (output : Fin Semantics.outputCount)
  | yRingWidth (output : Fin Semantics.outputCount)
      (row : Fin Semantics.yRingRows)
  | yRingLimb (output : Fin Semantics.outputCount)
      (row : Fin Semantics.yRingRows)
      (coefficient : Fin Semantics.activeWidth) (limb : Fin 2)
  | yZcolWidth (output : Fin Semantics.outputCount)
  | yZcolLimb (output : Fin Semantics.outputCount)
      (coefficient : Fin Semantics.activeWidth) (limb : Fin 2)
deriving DecidableEq, Repr

def twoLimbRoles
    (make : Fin Semantics.activeWidth -> Fin 2 -> SourceRole) : List SourceRole :=
  (List.ofFn fun coefficient : Fin Semantics.activeWidth =>
    [make coefficient ⟨0, by decide⟩,
     make coefficient ⟨1, by decide⟩]).flatten

def yRingVectorRoles
    (output : Fin Semantics.outputCount)
    (row : Fin Semantics.yRingRows) : List SourceRole :=
  .yRingWidth output row ::
    twoLimbRoles fun coefficient limb =>
      .yRingLimb output row coefficient limb

def outputRoles (output : Fin Semantics.outputCount) : List SourceRole :=
  (List.ofFn fun word : Fin 8 => SourceRole.innerTag output word) ++
    [.yRingRows output] ++
    (List.ofFn fun row : Fin Semantics.yRingRows =>
      yRingVectorRoles output row).flatten ++
    [.yZcolWidth output] ++
    twoLimbRoles fun coefficient limb =>
      .yZcolLimb output coefficient limb

/-- The independent protocol tree flattened only at its serialization
boundary. -/
def sourceRoles : List SourceRole :=
  (List.ofFn fun word : Fin 7 => SourceRole.outerTag word) ++
    [.outerCount] ++
    (List.ofFn fun output : Fin Semantics.outputCount => outputRoles output).flatten

theorem twoLimbRoles_length
    (make : Fin Semantics.activeWidth -> Fin 2 -> SourceRole) :
    (twoLimbRoles make).length = 108 := by
  simp [twoLimbRoles, Semantics.activeWidth,
    Nightstream.SuperNeo.Concrete.ringDegree]

theorem yRingVectorRoles_length
    (output : Fin Semantics.outputCount)
    (row : Fin Semantics.yRingRows) :
    (yRingVectorRoles output row).length = 109 := by
  simp [yRingVectorRoles, twoLimbRoles_length]

theorem outputRoles_length (output : Fin Semantics.outputCount) :
    (outputRoles output).length = 445 := by
  simp [outputRoles, yRingVectorRoles_length, twoLimbRoles_length,
    Semantics.yRingRows]

theorem sourceRoles_length : sourceRoles.length = 6683 := by
  decide

/-! ## Explicit production-column formulas -/

def outputConstantBase (output : Fin Semantics.outputCount) : Nat :=
  1714302 + 13 * output.val

def outputDynamicBase (output : Fin Semantics.outputCount) : Nat :=
  1159865 + 1790 * output.val

def roleColumn : SourceRole -> Nat
  | .outerTag word => 1714294 + word.val
  | .outerCount => 1714301
  | .innerTag output word => outputConstantBase output + word.val
  | .yRingRows output => outputConstantBase output + 8
  | .yRingWidth output row => outputConstantBase output + 9 + row.val
  | .yRingLimb output row coefficient limb =>
      outputDynamicBase output + 128 * row.val +
        2 * coefficient.val + limb.val
  | .yZcolWidth output => outputConstantBase output + 12
  | .yZcolLimb output coefficient limb =>
      outputDynamicBase output + 390 +
        2 * coefficient.val + limb.val

/-- Kernel-checked equality between the typed protocol-role tree and the
6,683 primary field columns extracted from the generated owner schedule. -/
theorem roleColumns_eq_artifact :
    sourceRoles.map roleColumn = EncodingSchedule.mainFieldColumns := by
  decide

/-! ## Independently specified constant leaves -/

def constantValue : SourceRole -> Option Nat
  | .outerTag word =>
      some ((Semantics.packBytesAsNats Semantics.outputsDomainBytes).getD
        word.val 0)
  | .outerCount => some Semantics.outputCount
  | .innerTag _ word =>
      some ((Semantics.packBytesAsNats Semantics.outputMessageDomainBytes).getD
        word.val 0)
  | .yRingRows _ => some Semantics.yRingRows
  | .yRingWidth _ _ => some Semantics.activeWidth
  | .yRingLimb _ _ _ _ => none
  | .yZcolWidth _ => some Semantics.activeWidth
  | .yZcolLimb _ _ _ => none

def sourceConstantPins : List (Nat × Nat) :=
  sourceRoles.filterMap fun role =>
    (constantValue role).map fun value => (roleColumn role, value)

def primaryShapePins : List (Nat × Nat) :=
  [(1714497, 54), (1714498, 2)]

def initialPins : List (Nat × Nat) :=
  sourceConstantPins ++ primaryShapePins

theorem sourceConstantPins_length : sourceConstantPins.length = 203 := by
  decide

theorem initialPins_length : initialPins.length = 205 := by
  decide

theorem initialPinsCanonical : ConstantPins.ValuesCanonical initialPins := by
  decide

def initialPiece : Piece :=
  EncodingSchedule.artifactOwner.pieces.get ⟨0, by decide⟩

theorem initialPiece_mem :
    initialPiece ∈ EncodingSchedule.artifactOwner.pieces := by
  exact List.get_mem _ _

/-- Exact ordinary owner leaf: no generated label is trusted in place of the
205 independently named constant equations. -/
theorem initialPiece_eq :
    initialPiece =
      { rowStart := 1873373
        rowEnd := 1873578
        payload := .ordinary (ConstantPins.rows initialPins) } := by
  decide

private theorem rowsIncluded_self (rows : List Row) :
    rowsIncluded rows rows = true := by
  simp [rowsIncluded]

/-- Accepted equations force every independent serialization and primary-SIS
shape constant. Dynamic output fields are intentionally absent. -/
theorem accepted_initialPins
    {assignment : Nat -> Nat}
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (accepted : EncodingSchedule.ArtifactAccepted assignment) :
    forall pin, pin ∈ initialPins -> assignment pin.1 = pin.2 := by
  have pieceAccepted := accepted initialPiece initialPiece_mem
  rw [Piece.Accepted, initialPiece_eq, Payload.Accepted] at pieceAccepted
  exact ConstantPins.sound initialPinsCanonical
    (rowsIncluded_self (ConstantPins.rows initialPins))
    canonical one pieceAccepted

/-! ## Typed decoding and serialization order -/

def fieldAt (assignment : Nat -> Nat)
    (canonical : forall column, assignment column < goldilocksP)
    (column : Nat) : Nightstream.SuperNeo.Concrete.F :=
  ⟨assignment column, by
    simpa [goldilocksP, Nightstream.SuperNeo.Concrete.goldilocksModulus] using
      canonical column⟩

def decodedOutput
    (assignment : Nat -> Nat)
    (canonical : forall column, assignment column < goldilocksP)
    (output : Fin Semantics.outputCount) : Semantics.OutputMessage where
  yRing := fun row coefficient =>
    { c0 := fieldAt assignment canonical
        (roleColumn (.yRingLimb output row coefficient ⟨0, by decide⟩))
      c1 := fieldAt assignment canonical
        (roleColumn (.yRingLimb output row coefficient ⟨1, by decide⟩)) }
  yZcol := fun coefficient =>
    { c0 := fieldAt assignment canonical
        (roleColumn (.yZcolLimb output coefficient ⟨0, by decide⟩))
      c1 := fieldAt assignment canonical
        (roleColumn (.yZcolLimb output coefficient ⟨1, by decide⟩)) }

def decodedOutputs
    (assignment : Nat -> Nat)
    (canonical : forall column, assignment column < goldilocksP) :
    Fin Semantics.outputCount -> Semantics.OutputMessage :=
  decodedOutput assignment canonical

def roleValue
    (outputs : Fin Semantics.outputCount -> Semantics.OutputMessage) :
    SourceRole -> Nightstream.SuperNeo.Concrete.F
  | .outerTag word =>
      (Semantics.packBytesAsFields Semantics.outputsDomainBytes).getD word.val 0
  | .outerCount => Semantics.fieldOfNat Semantics.outputCount
  | .innerTag _ word =>
      (Semantics.packBytesAsFields
        Semantics.outputMessageDomainBytes).getD word.val 0
  | .yRingRows _ => Semantics.fieldOfNat Semantics.yRingRows
  | .yRingWidth _ _ => Semantics.fieldOfNat Semantics.activeWidth
  | .yRingLimb output row coefficient limb =>
      if limb.val = 0 then (outputs output).yRing row coefficient |>.c0
      else (outputs output).yRing row coefficient |>.c1
  | .yZcolWidth _ => Semantics.fieldOfNat Semantics.activeWidth
  | .yZcolLimb output coefficient limb =>
      if limb.val = 0 then (outputs output).yZcol coefficient |>.c0
      else (outputs output).yZcol coefficient |>.c1

theorem yRingLimbRoleValues
    (outputs : Fin Semantics.outputCount -> Semantics.OutputMessage)
    (output : Fin Semantics.outputCount) (row : Fin Semantics.yRingRows) :
    (twoLimbRoles fun coefficient limb =>
      SourceRole.yRingLimb output row coefficient limb).map
        (roleValue outputs) =
      (List.ofFn ((outputs output).yRing row)).flatMap
        Semantics.extensionFields := by
  simp only [twoLimbRoles, List.map_flatten, List.map_ofFn,
    List.flatMap]
  apply congrArg List.flatten
  apply congrArg List.ofFn
  funext coefficient
  simp [roleValue, Semantics.extensionFields]

theorem yZcolLimbRoleValues
    (outputs : Fin Semantics.outputCount -> Semantics.OutputMessage)
    (output : Fin Semantics.outputCount) :
    (twoLimbRoles fun coefficient limb =>
      SourceRole.yZcolLimb output coefficient limb).map
        (roleValue outputs) =
      (List.ofFn (outputs output).yZcol).flatMap
        Semantics.extensionFields := by
  simp only [twoLimbRoles, List.map_flatten, List.map_ofFn,
    List.flatMap]
  apply congrArg List.flatten
  apply congrArg List.ofFn
  funext coefficient
  simp [roleValue, Semantics.extensionFields]

theorem yRingVectorRoleValues
    (outputs : Fin Semantics.outputCount -> Semantics.OutputMessage)
    (output : Fin Semantics.outputCount) (row : Fin Semantics.yRingRows) :
    (yRingVectorRoles output row).map (roleValue outputs) =
      Semantics.serializeKVector ((outputs output).yRing row) := by
  simp [yRingVectorRoles, Semantics.serializeKVector,
    yRingLimbRoleValues, roleValue]

private theorem ofFn_getD_eq_self
    {Alpha : Type} {count : Nat} (values : List Alpha) (default : Alpha)
    (lengthEq : values.length = count) :
    (List.ofFn fun index : Fin count => values.getD index.val default) =
      values := by
  apply List.ext_get
  · simp [lengthEq]
  · intro index leftLt rightLt
    simp only [List.get_eq_getElem, List.getElem_ofFn]
    rw [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem rightLt]
    rfl

theorem innerTagRoleValues
    (outputs : Fin Semantics.outputCount -> Semantics.OutputMessage)
    (output : Fin Semantics.outputCount) :
    (List.ofFn fun word : Fin 8 => SourceRole.innerTag output word).map
        (roleValue outputs) =
      Semantics.packBytesAsFields Semantics.outputMessageDomainBytes := by
  rw [List.map_ofFn]
  change (List.ofFn fun word : Fin 8 =>
      (Semantics.packBytesAsFields
        Semantics.outputMessageDomainBytes).getD word.val 0) =
    Semantics.packBytesAsFields Semantics.outputMessageDomainBytes
  apply ofFn_getD_eq_self
  simp [Semantics.packBytesAsFields,
    Semantics.outputMessageDomainTag_eq]

theorem outerTagRoleValues
    (outputs : Fin Semantics.outputCount -> Semantics.OutputMessage) :
    (List.ofFn fun word : Fin 7 => SourceRole.outerTag word).map
        (roleValue outputs) =
      Semantics.packBytesAsFields Semantics.outputsDomainBytes := by
  rw [List.map_ofFn]
  change (List.ofFn fun word : Fin 7 =>
      (Semantics.packBytesAsFields
        Semantics.outputsDomainBytes).getD word.val 0) =
    Semantics.packBytesAsFields Semantics.outputsDomainBytes
  apply ofFn_getD_eq_self
  simp [Semantics.packBytesAsFields, Semantics.outputsDomainTag_eq]

theorem yRingRowsRoleValues
    (outputs : Fin Semantics.outputCount -> Semantics.OutputMessage)
    (output : Fin Semantics.outputCount) :
    ((List.ofFn fun row : Fin Semantics.yRingRows =>
      yRingVectorRoles output row).flatten).map (roleValue outputs) =
      Semantics.serializeKVector
          ((outputs output).yRing ⟨0, by decide⟩) ++
        Semantics.serializeKVector
          ((outputs output).yRing ⟨1, by decide⟩) ++
        Semantics.serializeKVector
          ((outputs output).yRing ⟨2, by decide⟩) := by
  rw [List.map_flatten, List.map_ofFn]
  simp [Semantics.yRingRows, List.ofFn_succ, yRingVectorRoleValues]

theorem outputRoleValues
    (outputs : Fin Semantics.outputCount -> Semantics.OutputMessage)
    (output : Fin Semantics.outputCount) :
    (outputRoles output).map (roleValue outputs) =
      Semantics.serializeOutput (outputs output) := by
  rw [outputRoles, Semantics.serializeOutput]
  simp only [List.map_append]
  rw [innerTagRoleValues]
  simp only [List.map_cons, List.map_nil, roleValue]
  rw [yRingRowsRoleValues]
  rw [yZcolLimbRoleValues]
  rfl

theorem allOutputRoleValues
    (outputs : Fin Semantics.outputCount -> Semantics.OutputMessage) :
    ((List.ofFn fun output : Fin Semantics.outputCount =>
      outputRoles output).flatten).map (roleValue outputs) =
      (List.ofFn outputs).flatMap Semantics.serializeOutput := by
  simp only [List.map_flatten, List.map_ofFn, List.flatMap]
  apply congrArg List.flatten
  apply congrArg List.ofFn
  funext output
  exact outputRoleValues outputs output

/-- The typed role tree is exactly the independent serialization, before any
production-column theorem is used. -/
theorem sourceRoleValues_eq_serialization
    (outputs : Fin Semantics.outputCount -> Semantics.OutputMessage) :
    sourceRoles.map (roleValue outputs) =
      Semantics.serializeTerminalOutputs outputs := by
  rw [sourceRoles, Semantics.serializeTerminalOutputs,
    Semantics.serializeOutputs]
  simp only [List.map_append]
  rw [outerTagRoleValues]
  simp only [List.map_cons, List.map_nil, roleValue]
  rw [allOutputRoleValues]
  simp [Semantics.outputCount]

/-! ## Accepted-row connection to the independent serialization -/

theorem sourceConstantPin_mem
    {role : SourceRole} {value : Nat}
    (roleMember : role ∈ sourceRoles)
    (isConstant : constantValue role = some value) :
    (roleColumn role, value) ∈ initialPins := by
  apply List.mem_append_left
  apply List.mem_filterMap.mpr
  refine ⟨role, roleMember, ?_⟩
  simp [isConstant]

private theorem accepted_constantField
    {assignment : Nat -> Nat}
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (accepted : EncodingSchedule.ArtifactAccepted assignment)
    {role : SourceRole} {value : Nat}
    (roleMember : role ∈ sourceRoles)
    (isConstant : constantValue role = some value) :
    Semantics.fieldOfNat value =
      fieldAt assignment canonical (roleColumn role) := by
  have pinMember := sourceConstantPin_mem roleMember isConstant
  have fixed := accepted_initialPins canonical one accepted
    (roleColumn role, value) pinMember
  have valueCanonical := initialPinsCanonical
    (roleColumn role, value) pinMember
  apply Fin.ext
  change value % Nightstream.SuperNeo.Concrete.goldilocksModulus =
    assignment (roleColumn role)
  rw [Nat.mod_eq_of_lt (by
    simpa [goldilocksP,
      Nightstream.SuperNeo.Concrete.goldilocksModulus] using valueCanonical)]
  exact fixed.symm

private theorem packedField_getD (bytes : List Nat) (index : Nat) :
    (Semantics.packBytesAsFields bytes).getD index 0 =
      Semantics.fieldOfNat
        ((Semantics.packBytesAsNats bytes).getD index 0) := by
  change ((Semantics.packBytesAsNats bytes).map
      Semantics.fieldOfNat).getD index 0 = _
  change ((Semantics.packBytesAsNats bytes).map
      Semantics.fieldOfNat).getD index (Semantics.fieldOfNat 0) = _
  simp only [List.getD_eq_getElem?_getD, List.getElem?_map,
    Option.getD_map]

/-- Every typed source role denotes exactly the canonical field value stored
in its production column. Constants use accepted ordinary equations; dynamic
limbs use only the typed decoder. This theorem does not claim that the dynamic
columns are valid `Pi_CCS` verifier outputs. -/
theorem accepted_decodedRoleValue
    {assignment : Nat -> Nat}
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (accepted : EncodingSchedule.ArtifactAccepted assignment)
    (role : SourceRole) (roleMember : role ∈ sourceRoles) :
    roleValue (decodedOutputs assignment canonical) role =
      fieldAt assignment canonical (roleColumn role) := by
  cases role with
  | outerTag word =>
      change (Semantics.packBytesAsFields
          Semantics.outputsDomainBytes).getD word.val 0 = _
      rw [packedField_getD]
      exact accepted_constantField canonical one accepted roleMember rfl
  | outerCount =>
      exact accepted_constantField canonical one accepted roleMember rfl
  | innerTag output word =>
      change (Semantics.packBytesAsFields
          Semantics.outputMessageDomainBytes).getD word.val 0 = _
      rw [packedField_getD]
      exact accepted_constantField canonical one accepted roleMember rfl
  | yRingRows output =>
      exact accepted_constantField canonical one accepted roleMember rfl
  | yRingWidth output row =>
      exact accepted_constantField canonical one accepted roleMember rfl
  | yRingLimb output row coefficient limb =>
      have limbValue : limb.val = 0 ∨ limb.val = 1 := by omega
      rcases limbValue with isZero | isOne
      · have limbEq : limb = ⟨0, by decide⟩ := Fin.ext isZero
        subst limb
        rfl
      · have limbEq : limb = ⟨1, by decide⟩ := Fin.ext isOne
        subst limb
        rfl
  | yZcolWidth output =>
      exact accepted_constantField canonical one accepted roleMember rfl
  | yZcolLimb output coefficient limb =>
      have limbValue : limb.val = 0 ∨ limb.val = 1 := by omega
      rcases limbValue with isZero | isOne
      · have limbEq : limb = ⟨0, by decide⟩ := Fin.ext isZero
        subst limb
        rfl
      · have limbEq : limb = ⟨1, by decide⟩ := Fin.ext isOne
        subst limb
        rfl

/-- Accepted owner rows connect the complete 6,683-position production source
layout to the independently typed terminal-output serialization. The result
still decodes the dynamic fields from their columns; upstream `Pi_CCS`
authority is a separate theorem obligation. -/
theorem accepted_decodedSerialization
    {assignment : Nat -> Nat}
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (accepted : EncodingSchedule.ArtifactAccepted assignment) :
    Semantics.serializeTerminalOutputs
        (decodedOutputs assignment canonical) =
      EncodingSchedule.mainFieldColumns.map
        (fieldAt assignment canonical) := by
  rw [← sourceRoleValues_eq_serialization]
  calc
    sourceRoles.map
        (roleValue (decodedOutputs assignment canonical)) =
        sourceRoles.map (fun role =>
          fieldAt assignment canonical (roleColumn role)) := by
      apply List.map_congr_left
      intro role roleMember
      exact accepted_decodedRoleValue canonical one accepted role roleMember
    _ = (sourceRoles.map roleColumn).map
        (fieldAt assignment canonical) := by
      simp only [List.map_map, Function.comp_def]
    _ = EncodingSchedule.mainFieldColumns.map
        (fieldAt assignment canonical) := by
      rw [roleColumns_eq_artifact]

end Nightstream.Implementation.R1CS.PiCcsOutputDigest.SourceLayout

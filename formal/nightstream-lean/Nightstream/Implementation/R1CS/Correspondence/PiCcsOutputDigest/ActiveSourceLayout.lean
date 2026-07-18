import Nightstream.Implementation.R1CS.Correspondence.PiCcsOutputDigest.ActiveSemantics

/-!
Typed source-role tree for the active `Pi_CCS` output preimage.

Assurance tier: model-level representation correspondence.

Owns: one typed role for every field in the generic active serializer; the
protocol -> source -> vector -> lane -> limb grouping; exact group lengths;
and proof that flattening the role tree produces `ActiveSemantics.serialize`
in exactly the same order.

Does not own: physical R1CS columns, accepted-output authority, the delayed
`y_zcol` source theorem, SIS/Poseidon2, transcript placement, costs,
necessity, or row removal.

Emits constraints: no.

Authority boundary: role values are read from an already typed output
message. This file proves lossless ordering only. A later physical-layout
certificate must bind each role to an accepted Rust/R1CS source column while
keeping the `y_ring` and `y_zcol` authority premises explicit.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.pi_ccs.output_message_hashes.digest.preimage.outer_header` | outer domain and source count | verifier-owned shape | `outerHeaderRoles` |
| `nifs.pi_ccs.output_message_hashes.digest.preimage.source_headers` | source domain and matrix count | verifier-owned shape | `sourceHeaderRoles` |
| `nifs.pi_ccs.output_message_hashes.digest.preimage.y_ring` | matrix-major widths and all active `(c0,c1)` limbs | checked payload encoding | `yRingRoles` |
| `nifs.pi_ccs.output_message_hashes.digest.preimage.y_zcol` | width and all active `(c0,c1)` limbs | checked payload encoding; source authority open | `yZcolRoles` |
| `nifs.pi_ccs.output_message_hashes.digest.preimage` | concatenate every source in canonical order | computed | `sourceRoles`, `sourceRoleValues_eq_serialize` |
| `nifs.pi_ccs.output_message_hashes.digest.preimage.binding` | decode columns only under three separately named authority premises | checked boundary | `decodedFields_eq_serialize` |
-/

namespace Nightstream.Implementation.R1CS.PiCcsOutputDigest.ActiveSourceLayout

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier

/-- Named extension-field limbs, matching the Rust `KLimb` order. -/
inductive Limb where
  | c0
  | c1
deriving DecidableEq, Repr

/-- Authority owner for one serializer input, matching the Rust decoder. -/
inductive InputOwner where
  | verifierShape
  | yRingOutput
  | yZcolOutput
deriving DecidableEq, Repr

/-- One mathematical role for every field in a shape-indexed output
serialization. -/
inductive SourceRole (shape : SemanticShape) where
  | outerTag (word : Fin 7)
  | sourceCount
  | sourceTag (source : Fin shape.sourceCount) (word : Fin 8)
  | matrixCount (source : Fin shape.sourceCount)
  | yRingWidth (source : Fin shape.sourceCount)
      (matrix : Fin shape.matrixCount)
  | yRingLimb (source : Fin shape.sourceCount)
      (matrix : Fin shape.matrixCount) (lane : Fin ringDegree) (limb : Limb)
  | yZcolWidth (source : Fin shape.sourceCount)
  | yZcolLimb (source : Fin shape.sourceCount)
      (lane : Fin ringDegree) (limb : Limb)
deriving DecidableEq, Repr

def inputOwner {shape : SemanticShape} : SourceRole shape -> InputOwner
  | .outerTag _ => .verifierShape
  | .sourceCount => .verifierShape
  | .sourceTag _ _ => .verifierShape
  | .matrixCount _ => .verifierShape
  | .yRingWidth _ _ => .verifierShape
  | .yRingLimb _ _ _ _ => .yRingOutput
  | .yZcolWidth _ => .verifierShape
  | .yZcolLimb _ _ _ => .yZcolOutput

def twoLimbRoles
    {shape : SemanticShape}
    (make : Fin ringDegree -> Limb -> SourceRole shape) :
    List (SourceRole shape) :=
  (List.ofFn fun lane : Fin ringDegree =>
    [make lane .c0, make lane .c1]).flatten

def yRingVectorRoles
    {shape : SemanticShape}
    (source : Fin shape.sourceCount)
    (matrix : Fin shape.matrixCount) : List (SourceRole shape) :=
  .yRingWidth source matrix ::
    twoLimbRoles fun lane limb => .yRingLimb source matrix lane limb

def yRingRoles
    {shape : SemanticShape}
    (source : Fin shape.sourceCount) : List (SourceRole shape) :=
  (List.ofFn fun matrix : Fin shape.matrixCount =>
    yRingVectorRoles source matrix).flatten

def yZcolRoles
    {shape : SemanticShape}
    (source : Fin shape.sourceCount) : List (SourceRole shape) :=
  .yZcolWidth source ::
    twoLimbRoles fun lane limb => .yZcolLimb source lane limb

def sourceHeaderRoles
    {shape : SemanticShape}
    (source : Fin shape.sourceCount) : List (SourceRole shape) :=
  (List.ofFn fun word : Fin 8 => SourceRole.sourceTag source word) ++
    [.matrixCount source]

def sourceBlockRoles
    {shape : SemanticShape}
    (source : Fin shape.sourceCount) : List (SourceRole shape) :=
  sourceHeaderRoles source ++ yRingRoles source ++ yZcolRoles source

def outerHeaderRoles (shape : SemanticShape) : List (SourceRole shape) :=
  (List.ofFn fun word : Fin 7 => SourceRole.outerTag word) ++
    [.sourceCount]

/-- The protocol tree flattened only at the pre-SIS boundary. -/
def sourceRoles (shape : SemanticShape) : List (SourceRole shape) :=
  outerHeaderRoles shape ++
    (List.ofFn fun source : Fin shape.sourceCount =>
      sourceBlockRoles source).flatten

private theorem sum_ofFn_const (count value : Nat) :
    (List.ofFn fun _ : Fin count => value).sum = count * value := by
  induction count with
  | zero => simp
  | succ count inductionHypothesis =>
      rw [List.ofFn_succ]
      simp only [List.sum_cons, inductionHypothesis, Nat.succ_mul]
      omega

private theorem flatten_ofFn_length
    {Alpha : Type}
    {count width : Nat}
    (blocks : Fin count -> List Alpha)
    (blockLength : forall index, (blocks index).length = width) :
    (List.ofFn blocks).flatten.length = count * width := by
  rw [List.length_flatten, List.map_ofFn]
  have lengths :
      List.ofFn (List.length ∘ blocks) =
        List.ofFn (fun _ : Fin count => width) := by
    apply congrArg List.ofFn
    funext index
    exact blockLength index
  rw [lengths, sum_ofFn_const]

@[simp] theorem twoLimbRoles_length
    {shape : SemanticShape}
    (make : Fin ringDegree -> Limb -> SourceRole shape) :
    (twoLimbRoles make).length = 2 * ringDegree := by
  simpa [twoLimbRoles, Nat.mul_comm] using
    flatten_ofFn_length (width := 2)
      (fun lane : Fin ringDegree => [make lane .c0, make lane .c1])
      (by intro; rfl)

@[simp] theorem yRingVectorRoles_length
    {shape : SemanticShape}
    (source : Fin shape.sourceCount)
    (matrix : Fin shape.matrixCount) :
    (yRingVectorRoles source matrix).length =
      Encoding.kVectorFieldCount ringDegree := by
  simp [yRingVectorRoles, Encoding.kVectorFieldCount]
  omega

@[simp] theorem yRingRoles_length
    {shape : SemanticShape}
    (source : Fin shape.sourceCount) :
    (yRingRoles source).length =
      shape.matrixCount * Encoding.kVectorFieldCount ringDegree := by
  exact flatten_ofFn_length
    (fun matrix : Fin shape.matrixCount => yRingVectorRoles source matrix)
    (yRingVectorRoles_length source)

@[simp] theorem yZcolRoles_length
    {shape : SemanticShape}
    (source : Fin shape.sourceCount) :
    (yZcolRoles source).length =
      Encoding.kVectorFieldCount ringDegree := by
  simp [yZcolRoles, Encoding.kVectorFieldCount]
  omega

@[simp] theorem sourceHeaderRoles_length
    {shape : SemanticShape}
    (source : Fin shape.sourceCount) :
    (sourceHeaderRoles source).length = 9 := by
  simp [sourceHeaderRoles]

@[simp] theorem sourceBlockRoles_length
    {shape : SemanticShape}
    (source : Fin shape.sourceCount) :
    (sourceBlockRoles source).length =
      ActiveSemantics.sourceFieldCount shape.matrixCount := by
  simp [sourceBlockRoles, ActiveSemantics.sourceFieldCount,
    ActiveSemantics.sourcePayloadFieldCount]

@[simp] theorem outerHeaderRoles_length (shape : SemanticShape) :
    (outerHeaderRoles shape).length = 8 := by
  simp [outerHeaderRoles]

/-- Every serializer field has exactly one typed role. This is a field count,
not a row or column count. -/
@[simp] theorem sourceRoles_length (shape : SemanticShape) :
    (sourceRoles shape).length = ActiveSemantics.fieldCount shape := by
  rw [sourceRoles, List.length_append, outerHeaderRoles_length]
  rw [flatten_ofFn_length
    (fun source : Fin shape.sourceCount => sourceBlockRoles source)
    (sourceBlockRoles_length)]
  rfl

def roleValue
    {shape : SemanticShape}
    (message : OutputMessage shape) : SourceRole shape -> F
  | .outerTag word => Encoding.outputsDomainFields.getD word.val 0
  | .sourceCount => Encoding.fieldOfNat shape.sourceCount
  | .sourceTag _ word => Encoding.outputMessageDomainFields.getD word.val 0
  | .matrixCount _ => Encoding.fieldOfNat shape.matrixCount
  | .yRingWidth _ _ => Encoding.fieldOfNat ringDegree
  | .yRingLimb source matrix lane .c0 =>
      (message.yRing source matrix lane).c0
  | .yRingLimb source matrix lane .c1 =>
      (message.yRing source matrix lane).c1
  | .yZcolWidth _ => Encoding.fieldOfNat ringDegree
  | .yZcolLimb source lane .c0 => (message.yZcol source lane).c0
  | .yZcolLimb source lane .c1 => (message.yZcol source lane).c1

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

theorem outerTagRoleValues
    {shape : SemanticShape}
    (message : OutputMessage shape) :
    (List.ofFn fun word : Fin 7 => SourceRole.outerTag word).map
        (roleValue message) = Encoding.outputsDomainFields := by
  rw [List.map_ofFn]
  apply ofFn_getD_eq_self
  exact Encoding.outputsDomainFields_length

theorem sourceTagRoleValues
    {shape : SemanticShape}
    (message : OutputMessage shape)
    (source : Fin shape.sourceCount) :
    (List.ofFn fun word : Fin 8 => SourceRole.sourceTag source word).map
        (roleValue message) = Encoding.outputMessageDomainFields := by
  rw [List.map_ofFn]
  apply ofFn_getD_eq_self
  exact Encoding.outputMessageDomainFields_length

theorem twoLimbRoleValues
    {shape : SemanticShape}
    (message : OutputMessage shape)
    (values : Fin ringDegree -> K)
    (make : Fin ringDegree -> Limb -> SourceRole shape)
    (c0 : forall lane, roleValue message (make lane .c0) = (values lane).c0)
    (c1 : forall lane, roleValue message (make lane .c1) = (values lane).c1) :
    (twoLimbRoles make).map (roleValue message) =
      Encoding.encodeFamily Encoding.encodeK values := by
  rw [Encoding.encodeFamily, Encoding.FixedBlocks.encode_eq_flatMap]
  simp only [twoLimbRoles, List.map_flatten, List.map_ofFn,
    List.flatMap]
  apply congrArg List.flatten
  apply congrArg List.ofFn
  funext lane
  simp [Encoding.encodeK, c0 lane, c1 lane]

theorem yRingVectorRoleValues
    {shape : SemanticShape}
    (message : OutputMessage shape)
    (source : Fin shape.sourceCount)
    (matrix : Fin shape.matrixCount) :
    (yRingVectorRoles source matrix).map (roleValue message) =
      Encoding.encodeKVector (message.yRing source matrix) := by
  simp [yRingVectorRoles, Encoding.encodeKVector, roleValue,
    twoLimbRoleValues message (message.yRing source matrix)]

theorem yZcolRoleValues
    {shape : SemanticShape}
    (message : OutputMessage shape)
    (source : Fin shape.sourceCount) :
    (yZcolRoles source).map (roleValue message) =
      Encoding.encodeKVector (message.yZcol source) := by
  simp [yZcolRoles, Encoding.encodeKVector, roleValue,
    twoLimbRoleValues message (message.yZcol source)]

theorem yRingRoleValues
    {shape : SemanticShape}
    (message : OutputMessage shape)
    (source : Fin shape.sourceCount) :
    (yRingRoles source).map (roleValue message) =
      Encoding.encodeKVectorFamily (message.yRing source) := by
  rw [Encoding.encodeKVectorFamily, Encoding.encodeFamily,
    Encoding.FixedBlocks.encode_eq_flatMap]
  simp only [yRingRoles, List.map_flatten, List.map_ofFn,
    List.flatMap]
  apply congrArg List.flatten
  apply congrArg List.ofFn
  funext matrix
  exact yRingVectorRoleValues message source matrix

theorem sourceBlockRoleValues
    {shape : SemanticShape}
    (message : OutputMessage shape)
    (source : Fin shape.sourceCount) :
    (sourceBlockRoles source).map (roleValue message) =
      ActiveSemantics.encodeSource (ActiveSemantics.sourcePayload message source) := by
  rw [sourceBlockRoles, sourceHeaderRoles, ActiveSemantics.encodeSource,
    ActiveSemantics.encodeSourcePayload]
  simp only [List.map_append, List.map_cons, List.map_nil]
  rw [sourceTagRoleValues, yRingRoleValues, yZcolRoleValues]
  rfl

theorem allSourceRoleValues
    {shape : SemanticShape}
    (message : OutputMessage shape) :
    ((List.ofFn fun source : Fin shape.sourceCount =>
      sourceBlockRoles source).flatten).map (roleValue message) =
      ActiveSemantics.encodeSources (ActiveSemantics.sourcePayloads message) := by
  rw [ActiveSemantics.encodeSources, Encoding.encodeFamily,
    Encoding.FixedBlocks.encode_eq_flatMap]
  simp only [List.map_flatten, List.map_ofFn, List.flatMap]
  apply congrArg List.flatten
  apply congrArg List.ofFn
  funext source
  exact sourceBlockRoleValues message source

/-- Flattening the typed ownership tree yields exactly the independent active
serialization. No physical implementation premise appears in this theorem. -/
theorem sourceRoleValues_eq_serialize
    {shape : SemanticShape}
    (message : OutputMessage shape) :
    (sourceRoles shape).map (roleValue message) =
      ActiveSemantics.serialize message := by
  rw [sourceRoles, outerHeaderRoles, ActiveSemantics.serialize]
  simp only [List.map_append, List.map_cons, List.map_nil]
  rw [outerTagRoleValues, allSourceRoleValues]
  rfl

/-! ## Conditional physical-source boundary -/

/-- One authority class supplies exactly the semantic value expected at each
of its typed roles. Keeping the classes separate prevents a generic binding
predicate from concealing the open `y_zcol` source theorem. -/
def BindingsHoldFor
    {shape : SemanticShape}
    (owner : InputOwner)
    (assignment : Nat -> F)
    (column : SourceRole shape -> Nat)
    (message : OutputMessage shape) : Prop :=
  forall role, inputOwner role = owner ->
    assignment (column role) = roleValue message role

/-- Field values read from a candidate physical source-column map. -/
def decodedFields
    {shape : SemanticShape}
    (assignment : Nat -> F)
    (column : SourceRole shape -> Nat) : List F :=
  (sourceRoles shape).map fun role => assignment (column role)

/-- A physical source map decodes to the independent serializer only when all
three authority classes are discharged. In particular, this theorem cannot
be applied while the production `y_zcol` source binding remains open. -/
theorem decodedFields_eq_serialize
    {shape : SemanticShape}
    (assignment : Nat -> F)
    (column : SourceRole shape -> Nat)
    (message : OutputMessage shape)
    (verifierShapeBound :
      BindingsHoldFor .verifierShape assignment column message)
    (yRingBound : BindingsHoldFor .yRingOutput assignment column message)
    (yZcolBound : BindingsHoldFor .yZcolOutput assignment column message) :
    decodedFields assignment column = ActiveSemantics.serialize message := by
  rw [← sourceRoleValues_eq_serialize]
  apply List.map_congr_left
  intro role _
  cases ownerEq : inputOwner role with
  | verifierShape => exact verifierShapeBound role ownerEq
  | yRingOutput => exact yRingBound role ownerEq
  | yZcolOutput => exact yZcolBound role ownerEq

end Nightstream.Implementation.R1CS.PiCcsOutputDigest.ActiveSourceLayout

import NightstreamFPrime.Export.BoundedDepth.StepPackage
import NightstreamFPrime.Export.Generic.ConstantBlocks
import NightstreamFPrime.Export.Generic.Packable
import NightstreamFPrime.Export.Generic.FinalAdvance

namespace NightstreamFPrime.Export.BoundedDepth.SpecializedCircuit

open Spec Circuit Layout Export.Generic Export.Generic.Pieces
open CircuitRelation PieceRowsCompleteness

abbrev Holds (pieces : List Piece) (env : Env) :=
  ∀ piece ∈ pieces, ConstraintsHold env (piece.constraints (PilotData.canonicalConstraints ()))

def valid : Piece → Bool
  | .constraint _ => true
  | .ringProduct _ _ lanes | .permutation _ lanes => decide (LinearLanes lanes)

theorem valid_iff (p : Piece) : valid p = true ↔ p.Valid := by
  cases p <;> simp [valid, Piece.Valid]

def exprEnd : Expr → Nat
  | .var c => c + 1
  | .const _ => 0
  | .add x y | .mul x y => max (exprEnd x) (exprEnd y)

private theorem exprEnd_le (e : Expr) (n : Nat) : exprEnd e ≤ n ↔ exprBelow n e = true := by
  induction e with
  | var c => simp [exprEnd, exprBelow]
  | const c => simp [exprEnd, exprBelow]
  | add x y hx hy | mul x y hx hy =>
    simp only [exprEnd, exprBelow, Bool.and_eq_true, max_le_iff, hx, hy]

variable {α : Type}

private def maxEnd (f : α → Nat) (xs : List α) := xs.foldl (fun n x => max n (f x)) 0

private theorem foldMax_le (f : α → Nat) (xs : List α) (initial n : Nat) :
    xs.foldl (fun m x => max m (f x)) initial ≤ n ↔ initial ≤ n ∧ ∀ x ∈ xs, f x ≤ n := by
  induction xs generalizing initial with
  | nil => simp
  | cons x xs ih => simp [ih, and_assoc]

private theorem maxEnd_le (f : α → Nat) (xs : List α) (n : Nat) :
    maxEnd f xs ≤ n ↔ ∀ x ∈ xs, f x ≤ n := by
  simp [maxEnd, foldMax_le]

private def pieceEnd : Piece → Nat
  | .constraint e => exprEnd e
  | .permutation s lanes => max (maxEnd exprEnd lanes) (s + 1096)
  | .ringProduct cells s lanes => max (maxEnd exprEnd lanes) (s + ringDegree * cells)

private theorem pieceEnd_le (p : Piece) (n : Nat) : pieceEnd p ≤ n ↔ pieceBelow n p = true := by
  cases p <;> simp [pieceEnd, pieceBelow, maxEnd_le, List.all_eq_true, exprEnd_le]

def endColumn (ps : List Piece) := maxEnd pieceEnd ps

theorem endColumn_le (ps : List Piece) (n : Nat) :
    endColumn ps ≤ n ↔ ps.all (pieceBelow n) = true := by
  simp [endColumn, maxEnd_le, List.all_eq_true, pieceEnd_le]

inductive Repair where
  | identity
  | removed (last : Option Frame.Block) (middle : List Piece) (fixed : ConstantBlocks.Result)

def Repair.restore : Repair → Env → Env
  | .identity, env => env
  | .removed last middle fixed, env =>
    FinalAdvance.restore last (Frame.complete (blocksOf middle) (ConstantBlocks.restore fixed env))

structure Plan (raw : List Piece) where
  pieces : List Piece
  replace : Expr → Expr
  repair : Repair
  sound : ∀ env, Holds pieces env → Holds raw (repair.restore env)
  complete : ∀ env, Holds raw env → Holds pieces env
  replace_eval : ∀ env, Holds raw env → ∀ e, (replace e).eval env = e.eval env
  publicEq : ∀ env c, c < publicColumns → repair.restore env c = env c
  packed : packable publicColumns raw = true → packable publicColumns pieces = true
  bounded : endColumn pieces ≤ endColumn raw
  validPieces : (∀ p ∈ raw, p.Valid) → ∀ p ∈ pieces, p.Valid

def original (raw : List Piece) : Plan raw where
  pieces := raw
  replace := id
  repair := .identity
  sound := fun _ => id
  complete := fun _ => id
  replace_eval := fun _ _ _ => rfl
  publicEq := fun _ _ _ => rfl
  packed := id
  bounded := Nat.le_refl _
  validPieces := id

private theorem restored_blocks (pieces : List Piece) (env : Env)
    (packed : packable publicColumns pieces = true) (good : ∀ p ∈ pieces, p.Valid)
    (holds : ∀ p ∈ pieces, ConstantBlocks.Holds env p) :
    Holds pieces (Frame.complete (blocksOf pieces) env) := by
  apply packed_holds good packed
  · intro p h ne
    cases p with
    | permutation => simp [isBlock] at ne
    | constraint | ringProduct => exact holds _ h
  · intro block h
    have mem := RelationCompleteness.permutation_mem_of_block h
    exact (ConstantBlocks.outputs_iff env _ _ (good _ mem)).mp (holds _ mem)

private theorem complete_public (ps : List Piece) {env : Env} {c : Nat} (hc : c < publicColumns)
    (packed : packable publicColumns ps = true) : Frame.complete (blocksOf ps) env c = env c := by
  apply Frame.complete_of_uncovered
  simp only [packable, Bool.and_eq_true] at packed
  have starts : startsFrom publicColumns ps = true := packed.1.2
  intro b hb
  have above := of_decide_eq_true (List.all_eq_true.mp starts b hb)
  simp [Frame.Block.covers, Nat.not_le.mpr (hc.trans_le above)]

opaque prepare (ops : List Op) (raw : List Piece)
    (witness : List WitnessBatch := StepPackage.witnessOf (witnesses ops) raw) : Plan raw :=
  let target := FinalAdvance.target ops
  let last := FinalAdvance.plan publicColumns target raw witness
  let middle := FinalAdvance.kept last raw
  let fixed := ConstantBlocks.specialize middle
  if ok : packable publicColumns middle = true ∧ middle.all valid = true ∧
      packable publicColumns fixed.pieces = true ∧ fixed.pieces.all valid = true ∧
      endColumn fixed.pieces ≤ endColumn raw then
    { pieces := fixed.pieces
      replace := ConstantBlocks.replace fixed.constants
      repair := .removed last middle fixed
      sound := by
        intro env h
        apply FinalAdvance.sound publicColumns target raw witness
        apply restored_blocks middle _ ok.1
          (fun p hp => (valid_iff _).mp (List.all_eq_true.mp ok.2.1 _ hp))
        apply ConstantBlocks.sound middle env
        exact fun p hp => ConstantBlocks.holds_of_constraints env p
          ((valid_iff _).mp (List.all_eq_true.mp ok.2.2.2.1 _ hp)) (h p hp)
      complete := fun env h => ConstantBlocks.constraints_complete middle env
        (fun p hp => (valid_iff _).mp (List.all_eq_true.mp ok.2.1 _ hp))
        (FinalAdvance.complete _ _ _ h)
      replace_eval := by
        intro env h e
        have full := FinalAdvance.complete last raw env h
        apply ConstantBlocks.replace_eval_of_holds middle env _ e
        exact fun p hp => ConstantBlocks.holds_of_constraints env p
          ((valid_iff _).mp (List.all_eq_true.mp ok.2.1 p hp)) (full p hp)
      publicEq := by
        intro env c below
        exact (FinalAdvance.agrees_prefix publicColumns target raw witness _ below).trans
          ((complete_public middle below ok.1).trans (ConstantBlocks.restore_of_lt fixed env c below))
      packed := fun _ => ok.2.2.1
      bounded := ok.2.2.2.2
      validPieces := fun _ p hp => (valid_iff _).mp (List.all_eq_true.mp ok.2.2.2.1 _ hp) }
  else original raw

def optimize (ops : List Op) (raw : List Piece) : List Piece := (prepare ops raw).pieces

def rawPiecesOf (ops : List Op) (digestColumn : Fin 4 → Nat) : List Piece :=
  split (PilotData.canonicalConstraints ()) (marks ops)
    (flatConstraints ops ++ LogicalPlan.decodeConstraints digestColumn)

def circuitPlan (circuit : FormalCircuit) (offset : Nat) (digestColumn : Fin 4 → Nat) :
    Plan (CircuitRelation.pieces circuit offset digestColumn) :=
  prepare (Circuit.ops circuit.main offset) (CircuitRelation.pieces circuit offset digestColumn)

def pieces (circuit : FormalCircuit) (offset : Nat) (digestColumn : Fin 4 → Nat) : List Piece :=
  (circuitPlan circuit offset digestColumn).pieces

open Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra

@[irreducible] def decode (circuit : FormalCircuit) (offset : Nat) (digestColumn : Fin 4 → Nat)
    (firstFresh : Nat) {width logicalWidth : Nat} (layout : LogicalPlan.Layout width logicalWidth)
    (assignment : Assignment F logicalWidth) : Env :=
  (circuitPlan circuit offset digestColumn).repair.restore
    (decodeEnv publicColumns (pieces circuit offset digestColumn) firstFresh layout assignment)

open Spec.Folding.PiCCS.PaperJoint Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra
open Lifecycle
variable {logicalWidth : Nat}
  {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
    RingCarrierLayout.carrierWidth logicalWidth}

theorem rowsZero_implies_spec (circuit : FormalCircuit) (offset firstFresh : Nat)
    (digestColumn : Fin 4 → Nat) {width : Nat}
    (fits : Fits publicColumns (pieces circuit offset digestColumn) firstFresh width)
    (layout : LogicalPlan.Layout width logicalWidth)
    (assignment : Assignment F logicalWidth) (digest : Digest)
    (encoded : LogicalPlan.publicInput (publicFits := publicFits)
      (LogicalPlan.sourceEnv layout assignment) = encHash digest)
    (rowsZero : (plan _ _ firstFresh width fits layout).RowsZero assignment)
    (assumptions : ∀ env, circuit.assumptions offset env) :
    circuit.spec offset (decode circuit offset digestColumn firstFresh layout assignment) ∧
      ∀ word, decode circuit offset digestColumn firstFresh layout assignment (digestColumn word) =
        digest.getD word.val 0 := by
  let checked := circuitPlan circuit offset digestColumn
  have validPieces := checked.validPieces (valid_split _ _ _)
  obtain ⟨all, same⟩ := rowsZero_implies_pieces publicColumns _ validPieces firstFresh width fits
    layout assignment (LogicalPlan.encoded_one layout assignment digest fits.rows.positive encoded)
    rowsZero
  let env := decode circuit offset digestColumn firstFresh layout assignment
  have holds : ConstraintsHold env (constraints circuit offset digestColumn) :=
    by unfold env decode; exact split_holds (show ∀ p ∈ CircuitRelation.pieces circuit offset digestColumn, _ from checked.sound _ all)
  have publicEq : LogicalPlan.publicInput (publicFits := publicFits) env =
      LogicalPlan.publicInput (publicFits := publicFits)
        (LogicalPlan.sourceEnv layout assignment) := by
    funext column
    unfold env decode
    exact (checked.publicEq _ column.val column.isLt).trans (same column.val column.isLt)
  refine ⟨?_, LogicalPlan.digest_of_decode env digestColumn digest (publicEq.trans encoded)
    (fun expression member => holds expression (List.mem_append_right _ member))⟩
  exact circuit.soundness _ offset (assumptions env)
    (holdsFlat_implies_holds _ _ fun expression member =>
      holds expression (List.mem_append_left _ member))

end NightstreamFPrime.Export.BoundedDepth.SpecializedCircuit

import NightstreamFPrime.Export.Generic.RelationCompleteness
import Std.Data.TreeMap.Lemmas

set_option autoImplicit false

namespace NightstreamFPrime.Export.Generic.ConstantBlocks

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Export.Generic.Pieces
open NightstreamFPrime.Gadgets.Poseidon2

abbrev Constants := Std.TreeMap Nat F

def lookup (known : Constants) (column : Nat) : Option F :=
  if column < CircuitRelation.publicColumns then none else known[column]?

def read (known : Constants) (env : Env) (column : Nat) : F :=
  (lookup known column).getD (env column)

def replace (known : Constants) (e : Expr) : Expr :=
  subst (fun column => ((lookup known column).map Expr.const).getD (.var column)) e

theorem replace_eval (known : Constants) (env : Env) (e : Expr) :
    (replace known e).eval env = e.eval (read known env) := by
  rw [replace, Lifecycle.Nebula.Expr.eval_subst]
  congr 1
  funext column
  cases h : lookup known column <;> simp [read, h]

def closed (known : Constants) : Expr → Option F
  | .const value => some value
  | .var column => lookup known column
  | .add a b => (closed known a).bind fun x => (closed known b).map fun y => x + y
  | .mul a b => (closed known a).bind fun x => (closed known b).map fun y => x * y

private theorem closed_eval (known : Constants) (env : Env) :
    ∀ {e : Expr} {v : F}, closed known e = some v → e.eval (read known env) = v
  | .const _, _, h => by simpa [closed] using h
  | .var c, _, h => by simp [Expr.eval, read, show lookup known c = _ from h]
  | .add a b, _, h => by
    simp only [closed, Option.bind_eq_some_iff, Option.map_eq_some_iff] at h
    obtain ⟨x, hx, y, hy, rfl⟩ := h
    rw [Expr.eval_add, closed_eval known env hx, closed_eval known env hy]
  | .mul a b, _, h => by
    simp only [closed, Option.bind_eq_some_iff, Option.map_eq_some_iff] at h
    obtain ⟨x, hx, y, hy, rfl⟩ := h
    rw [Expr.eval_mul, closed_eval known env hx, closed_eval known env hy]

def closedLanes (known : Constants) : List Expr → Option (List F)
  | [] => some []
  | x :: xs => (closed known x).bind fun v => (closedLanes known xs).map fun vs => v :: vs

private theorem closedLanes_eval (known : Constants) (env : Env) :
    ∀ {xs : List Expr} {vs}, closedLanes known xs = some vs →
      xs.map (Expr.eval (read known env)) = vs
  | [], _, h => by simpa [closedLanes] using h
  | _ :: _, _, h => by
    simp only [closedLanes, Option.bind_eq_some_iff, Option.map_eq_some_iff] at h
    obtain ⟨x, hx, ys, hys, rfl⟩ := h
    rw [List.map_cons, closed_eval known env hx, closedLanes_eval known env hys]

def permutation (values : List F) : List F :=
  Spec.Poseidon2.permute (List.ofFn fun i : Fin 16 => values.getD i.val 0)

def outputColumn (start : Nat) (i : Fin 16) : Nat := start + 1080 + i.val

def Holds (env : Env) : Piece → Prop
  | .permutation start lanes => ∀ i : Fin 16, env (outputColumn start i) =
      (permutation (lanes.map (Expr.eval env))).getD i.val 0
  | other => ConstraintsHold env (other.constraints (Export.PilotData.canonicalConstraints ()))

theorem permutation_get (env : Env) (lanes : List Expr) (i : Fin 16) :
    (permutation (lanes.map (Expr.eval env))).getD i.val 0 =
      Permutation.runF Permutation.schedule (fun j : Fin 16 => (lanes.getD j.val 0).eval env) i := by
  unfold permutation
  have lane (j : Fin 16) : (lanes.map (Expr.eval env)).getD j.val 0 = (lanes.getD j.val 0).eval env :=
    List.getD_map lanes 0 (Expr.eval env)
  simp_rw [lane]
  rw [← RelationCompleteness.ofFn_runF, List.getD_eq_getElem _ _ (by simpa only [List.length_ofFn] using i.isLt), List.getElem_ofFn]

def closedOutputs (known : Constants) (lanes : List Expr) : Option (List F) :=
  (closedLanes known lanes).map permutation

def record (known : Constants) (start : Nat) (values : List F) : Constants :=
  (List.finRange 16).foldl (fun table i =>
    table.insert (outputColumn start i) (values.getD i.val 0)) known

def learn (known : Constants) : Piece → Constants
  | .permutation start lanes => match closedOutputs known lanes with
    | some values => record known start values
    | none => known
  | _ => known

def fixed (known : Constants) (start : Nat) (lanes : List Expr) : Bool :=
  match closedOutputs known lanes with
  | none => false
  | some values => (List.finRange 16).all fun i =>
      lookup known (outputColumn start i) == some (values.getD i.val 0)

def rewritePiece (known : Constants) : Piece → Option Piece
  | .constraint e => some (.constraint (replace known e))
  | .ringProduct cells start lanes => some (.ringProduct cells start (lanes.map (replace known)))
  | .permutation start lanes =>
    if fixed known start lanes then none else some (.permutation start (lanes.map (replace known)))

def untouched (known : Constants) (start count : Nat) : Bool :=
  (List.range count).all fun i => (lookup known (start + i)).isNone

def safe (known : Constants) : Piece → Bool
  | .constraint _ => true
  | .permutation start lanes => fixed known start lanes || untouched known (start + 1080) 16
  | .ringProduct cells start _ => untouched known start (ringDegree * cells)

structure Result where
  pieces : List Piece
  constants : Constants

def specialize (pieces : List Piece) : Result :=
  let known := pieces.foldl learn {}
  if pieces.all (safe known) then ⟨pieces.filterMap (rewritePiece known), known⟩
  else ⟨pieces, {}⟩

def restore (result : Result) : Env → Env := read result.constants

theorem restore_of_lt (result : Result) (env : Env) (column : Nat)
    (below : column < CircuitRelation.publicColumns) : restore result env column = env column := by
  simp [restore, read, lookup, below]

theorem outputs_iff (env : Env) (start : Nat) (lanes : List Expr) (linear : LinearLanes lanes) :
    Holds env (.permutation start lanes) ↔
      (Frame.Block.mk start lanes).output env =
        Permutation.runF Permutation.schedule ((Frame.Block.mk start lanes).input env) := by
  have inputs : (Frame.Block.mk start lanes).input env =
      fun i => (lanes.getD i.val 0).eval env :=
    funext (RelationCompleteness.block_input linear env)
  rw [inputs]
  exact ⟨fun h => funext fun i => (h i).trans (permutation_get env lanes i),
    fun h i => (congrFun h i).trans (permutation_get env lanes i).symm⟩

private def Agrees (known : Constants) (env : Env) : Prop :=
  ∀ c v, known[c]? = some v → env c = v

private theorem read_agrees {known : Constants} {env : Env} (agrees : Agrees known env) :
    read known env = env := by
  funext c
  by_cases h : c < CircuitRelation.publicColumns
  · simp [read, lookup, h]
  · cases hv : known[c]? with
    | none => simp [read, lookup, h, hv]
    | some v => simp [read, lookup, h, hv, agrees c v hv]

private theorem record_agrees {known : Constants} (start : Nat) (values : List F) (env : Env)
    (old : Agrees known env)
    (outputs : ∀ i, env (outputColumn start i) = values.getD i.val 0) :
    Agrees (record known start values) env := by
  have loop (is : List (Fin 16)) (m : Constants) (h : Agrees m env) :
      Agrees (is.foldl (fun (table : Constants) i =>
        table.insert (outputColumn start i) (values.getD i.val 0)) m) env := by
    induction is generalizing m with
    | nil => exact h
    | cons i _ ih =>
      apply ih
      intro c v hv
      rw [Std.TreeMap.getElem?_insert] at hv
      split at hv
      next eq =>
        have e := Nat.compare_eq_eq.mp eq
        subst c
        exact (outputs i).trans (Option.some.inj hv)
      next => exact h c v hv
  exact loop _ _ old

private theorem learn_agrees {known : Constants} (env : Env) (old : Agrees known env) :
    ∀ {p : Piece}, Holds env p → Agrees (learn known p) env
  | .constraint _, _ => old
  | .ringProduct .., _ => old
  | .permutation start lanes, h => by
    cases he : closedOutputs known lanes with
    | none => simpa [learn, he] using old
    | some values =>
      simp only [learn, he]
      obtain ⟨vs, hvs, rfl⟩ := Option.map_eq_some_iff.mp he
      apply record_agrees _ _ _ old
      have inputs := closedLanes_eval known env hvs
      rw [read_agrees old] at inputs
      intro i
      rw [← inputs]
      exact h i

private theorem discovered (pieces : List Piece) (env : Env) (known : Constants)
    (h : Agrees known env) (holds : ∀ p ∈ pieces, Holds env p) :
    Agrees (pieces.foldl learn known) env := by
  induction pieces generalizing known with
  | nil => exact h
  | cons p ps ih =>
    exact ih (learn known p) (learn_agrees env h (holds p (by simp)))
      (fun q hq => holds q (List.mem_cons_of_mem _ hq))

private theorem fixed_holds {known : Constants} {start : Nat} {lanes : List Expr}
    (fixedInputs : fixed known start lanes = true) (env : Env) :
    Holds (read known env) (.permutation start lanes) := by
  unfold fixed at fixedInputs
  cases h : closedOutputs known lanes with
  | none => simp [h] at fixedInputs
  | some values =>
    simp only [h] at fixedInputs
    obtain ⟨vs, hvs, rfl⟩ := Option.map_eq_some_iff.mp h
    intro i
    have val := List.all_eq_true.mp fixedInputs i (List.mem_finRange i)
    have eq := beq_iff_eq.mp val
    rw [read, eq, Option.getD_some, closedLanes_eval known env hvs]

private theorem untouched_read {known : Constants} {start count : Nat}
    (untouchedOutputs : untouched known start count = true) (env : Env) (i : Nat) (hi : i < count) :
    read known env (start + i) = env (start + i) := by
  have none := List.all_eq_true.mp untouchedOutputs i (List.mem_range.mpr hi)
  simp [read, Option.isNone_iff_eq_none.mp none]

private theorem map_lanes (known : Constants) (env : Env) (lanes : List Expr) :
    (lanes.map (replace known)).map (Expr.eval env) = lanes.map (Expr.eval (read known env)) := by
  rw [List.map_map]
  exact List.map_congr_left fun e _ => replace_eval known env e

private theorem get_lane (known : Constants) (env : Env) (lanes : List Expr) (i : Nat) :
    ((lanes.map (replace known)).getD i 0).eval env =
      (lanes.getD i 0).eval (read known env) := by
  have h := List.getD_map (n := i) lanes (0 : Expr) (replace known)
  exact (congrArg (Expr.eval env) h).trans (replace_eval known env _)

private theorem ring_holds {known : Constants} (env : Env) (cells start : Nat) (lanes : List Expr)
    (noOutputs : untouched known start (ringDegree * cells) = true) :
    Holds env (.ringProduct cells start (lanes.map (replace known))) ↔
      Holds (read known env) (.ringProduct cells start lanes) := by
  let wires := laneSubst (ringInputs cells) start
  have eq (i : Fin (ringDegree * cells)) :
      (subst (wires (lanes.map (replace known))) (ringConstraint cells i)).eval env =
        (subst (wires lanes) (ringConstraint cells i)).eval (read known env) := by
    rw [Lifecycle.Nebula.Expr.eval_subst, Lifecycle.Nebula.Expr.eval_subst]
    apply Expr.eval_eq_of_agree_below _ _ _ _ (CircuitRelation.ringConstraint_varsBelow cells i)
    intro column bound
    by_cases input : column < ringInputs cells
    · simpa only [wires, laneSubst, input, ite_true] using get_lane known env lanes column
    · simp only [wires, laneSubst, input, ite_false, Expr.eval_var]
      exact (untouched_read noOutputs env _ (by omega)).symm
  change ConstraintsHold env ((ringTemplate cells).map _) ↔ ConstraintsHold _ ((ringTemplate cells).map _)
  constructor <;> intro holds _ member <;>
    obtain ⟨pattern, mem, rfl⟩ := List.mem_map.mp member <;>
    obtain ⟨i, rfl⟩ := List.mem_ofFn.mp mem
  · rw [← eq i]
    exact holds _ (List.mem_map.mpr ⟨_, List.mem_ofFn.mpr ⟨i, rfl⟩, rfl⟩)
  · rw [eq i]
    exact holds _ (List.mem_map.mpr ⟨_, List.mem_ofFn.mpr ⟨i, rfl⟩, rfl⟩)

private theorem rewrite_holds (known : Constants) (env : Env) (p : Piece)
    (safePiece : safe known p = true) :
    (∀ q ∈ (rewritePiece known p).toList, Holds env q) ↔ Holds (read known env) p := by
  cases p with
  | constraint e =>
    simp [rewritePiece, Holds, Piece.constraints, ConstraintsHold, replace_eval]
  | ringProduct cells start lanes =>
    simpa [rewritePiece] using ring_holds env cells start lanes safePiece
  | permutation start lanes =>
    by_cases h : fixed known start lanes = true
    · simp [rewritePiece, h, fixed_holds h env]
    · have untouchedOutputs : untouched known (start + 1080) 16 = true := by
        simpa [safe, h] using safePiece
      have pair : rewritePiece known (.permutation start lanes) =
          some (.permutation start (lanes.map (replace known))) := by simp [rewritePiece, h]
      rw [pair]
      simp only [Option.toList_some, List.mem_singleton, forall_eq, Holds, map_lanes]
      simp_rw [outputColumn, untouched_read untouchedOutputs env _ (Fin.isLt _)]

private theorem rewrite_all (known : Constants) (env : Env) (pieces : List Piece)
    (h : pieces.all (safe known) = true) :
    (∀ p ∈ pieces.filterMap (rewritePiece known), Holds env p) ↔
      ∀ p ∈ pieces, Holds (read known env) p := by
  constructor
  · intro all p hp
    apply (rewrite_holds known env p (List.all_eq_true.mp h p hp)).mp
    intro q hq
    exact all q (List.mem_filterMap.mpr ⟨p, hp, by simpa using hq⟩)
  · intro all q hq
    obtain ⟨p, hp, eq⟩ := List.mem_filterMap.mp hq
    exact (rewrite_holds known env p (List.all_eq_true.mp h p hp)).mpr (all p hp) q (by simp [eq])

private theorem empty_agrees (env : Env) : Agrees {} env := by simp [Agrees]

theorem sound (pieces : List Piece) (env : Env)
    (checks : ∀ p ∈ (specialize pieces).pieces, Holds env p) :
    ∀ p ∈ pieces, Holds (restore (specialize pieces) env) p := by
  by_cases h : pieces.all (safe (pieces.foldl learn {})) = true
  · simpa only [specialize, h, ite_true, restore] using (rewrite_all _ env pieces h).mp
      (by simpa only [specialize, h, ite_true] using checks)
  · simpa [specialize, h, restore, read_agrees (empty_agrees env)] using checks

theorem complete (pieces : List Piece) (env : Env)
    (checks : ∀ p ∈ pieces, Holds env p) :
    ∀ p ∈ (specialize pieces).pieces, Holds env p := by
  unfold specialize
  dsimp only
  split
  next h =>
    apply (rewrite_all _ env pieces h).mpr
    rw [read_agrees (discovered pieces env {} (empty_agrees env) checks)]
    exact checks
  next => exact checks

theorem replace_eval_of_holds (pieces : List Piece) (env : Env)
    (checks : ∀ p ∈ pieces, Holds env p) (e : Expr) :
    (replace (specialize pieces).constants e).eval env = e.eval env := by
  rw [replace_eval]
  unfold specialize
  dsimp only
  split
  · rw [read_agrees (discovered pieces env {} (empty_agrees env) checks)]
  · rw [read_agrees (empty_agrees env)]

theorem holds_of_constraints (env : Env) (p : Piece) (valid : p.Valid)
    (h : ConstraintsHold env (p.constraints (Export.PilotData.canonicalConstraints ()))) :
    Holds env p := by
  cases p with
  | constraint _ => exact h
  | ringProduct _ _ _ => exact h
  | permutation start lanes =>
    exact (outputs_iff env start lanes valid).mpr (RelationCompleteness.block_outputs (block := ⟨start, lanes⟩) valid h)

private theorem template_complete {known : Constants} {env : Env} (agree : read known env = env)
    (start n : Nat) (lanes template : List Expr)
    (holds : ConstraintsHold env (template.map (subst (laneSubst n start lanes)))) :
    ConstraintsHold env (template.map (subst (laneSubst n start (lanes.map (replace known))))) := by
  intro _ h
  obtain ⟨pattern, hp, rfl⟩ := List.mem_map.mp h
  have original := holds _ (List.mem_map.mpr ⟨pattern, hp, rfl⟩)
  rw [Lifecycle.Nebula.Expr.eval_subst] at original ⊢
  convert original using 2
  funext c
  by_cases hc : c < n
  · simpa only [laneSubst, hc, ite_true, agree] using get_lane known env lanes c
  · simp [laneSubst, hc]

theorem constraints_complete (pieces : List Piece) (env : Env)
    (valid : ∀ p ∈ pieces, p.Valid)
    (full : ∀ p ∈ pieces, ConstraintsHold env (p.constraints (Export.PilotData.canonicalConstraints ()))) :
    ∀ p ∈ (specialize pieces).pieces,
      ConstraintsHold env (p.constraints (Export.PilotData.canonicalConstraints ())) := by
  let m : Constants := pieces.foldl learn {}
  have eq : read m env = env :=
    read_agrees (discovered pieces env {} (empty_agrees env)
      (fun p hp => holds_of_constraints env p (valid p hp) (full p hp)))
  dsimp only [specialize]
  split
  · intro q hq
    obtain ⟨p, hp, h⟩ := List.mem_filterMap.mp hq
    change rewritePiece m p = some q at h
    cases p with
    | permutation start lanes =>
      by_cases ht : fixed m start lanes = true
      · simp [rewritePiece, ht] at h
      · simp [rewritePiece, ht] at h
        subst q
        exact template_complete eq start 16 lanes _ (full _ hp)
    | ringProduct cells start lanes =>
      simp [rewritePiece] at h
      subst q
      exact template_complete eq start (ringInputs cells) lanes _ (full _ hp)
    | constraint e =>
      simp [rewritePiece] at h
      subst q
      have h := full _ hp
      simpa [Piece.constraints, ConstraintsHold, replace_eval, eq] using h
  · exact full

end NightstreamFPrime.Export.Generic.ConstantBlocks

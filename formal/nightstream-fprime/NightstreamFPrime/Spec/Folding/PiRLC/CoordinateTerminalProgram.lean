import NightstreamFPrime.Spec.Folding.PiRLC.PaperForkExtractionWork

/-!
The extractor's deterministic terminal program consumes the observed endpoints.
It checks presence and distinct challenges, then executes the four charged
inverse-difference operations. It does not choose a proof of CompleteFork to
obtain its output data.

Each nonempty traversal charges presence dispatch, the challenge comparison,
and result construction. Empty traversal and the initial response dispatch
each charge one step. Arithmetic and representation work belong to the
explicit primitive implementations.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Folding.PiRLC.CoordinateTerminalProgram

open PaperForkAlgebra PaperForkExtraction PaperForkExtractionWork

universe uScalar uAssignment

variable {Scalar : Type uScalar} {Assignment : Type uAssignment}
  {member : Scalar → Prop} [DecidableEq Scalar]

/-- Traverse the actual endpoints in source order. An absent or repeated
endpoint returns failure without invoking inverse extraction at that source. -/
def gather (program : Primitives Scalar Assignment) : {count : Nat} →
    (Fin count → {scalar // member scalar}) → Assignment →
    (Fin count → ({scalar // member scalar} × Option Assignment)) → Result (Option (List Assignment))
  | 0, _, _, _ => ⟨some [], 1⟩
  | _ + 1, vector, base, outputs =>
      match (outputs 0).2 with
      | none => ⟨none, 1⟩
      | some assignment =>
          if (vector 0).val = (outputs 0).1.val then ⟨none, 2⟩ else
            let head := extract program (vector 0).val (outputs 0).1.val base assignment
            let tail := gather program (fun index => vector index.succ) base
              (fun index => outputs index.succ)
            ⟨tail.value.map (fun values => head.value :: values), head.work + tail.work + 3⟩

/-- Dispatch the observed base response, including the failure exit. -/
def finish (program : Primitives Scalar Assignment) {count : Nat}
    (vector : Fin count → {scalar // member scalar}) (initial : Option Assignment)
    (outputs : Fin count → ({scalar // member scalar} × Option Assignment)) :
    Result (Option (List Assignment)) :=
  match initial with
  | none => ⟨none, 1⟩
  | some base =>
      let result := gather program vector base outputs
      ⟨result.value, result.work + 1⟩

/-- Successful traversal uses the observed assignments and challenge values. -/
theorem gather_values (program : Primitives Scalar Assignment) : ∀ {count : Nat}
    (vector : Fin count → {scalar // member scalar}) (base : Assignment)
    (outputs : Fin count → ({scalar // member scalar} × Option Assignment))
    (assignments : Fin count → Assignment),
    (∀ index, (outputs index).2 = some (assignments index)) →
    (∀ index, (vector index).val ≠ (outputs index).1.val) →
    (gather program vector base outputs).value = some (List.ofFn (fun index =>
      (extract program (vector index).val (outputs index).1.val base (assignments index)).value))
  | 0, _, _, _, _, _, _ => rfl
  | _ + 1, vector, base, outputs, assignments, present, different => by
      simp only [gather, present 0, different 0, ↓reduceIte]
      rw [gather_values program (fun index => vector index.succ) base
        (fun index => outputs index.succ) (fun index => assignments index.succ)
        (fun index => present index.succ) (fun index => different index.succ)]
      simp only [Option.map_some, List.ofFn_succ]

/-- Uniform work bound for success and failure, derived from the actual
primitive clocks and the explicit terminal control flow. -/
theorem gather_work_le (ring : CommutativeRingOps Scalar) (module : ModuleOps Scalar Assignment)
    (program : Primitives Scalar Assignment) (correct : Correct ring module program)
    (strongSet : StrongSetUnits ring member) (bounds : PrimitiveBounds)
    (bounded : Bounded ring program bounds) : ∀ {count : Nat}
    (vector : Fin count → {scalar // member scalar}) (base : Assignment)
    (outputs : Fin count → ({scalar // member scalar} × Option Assignment)),
    (gather program vector base outputs).work ≤ count * (bounds.coordinateWork + 3) + 1
  | 0, _, _, _ => by simp [gather]
  | count + 1, vector, base, outputs => by
      cases response : (outputs 0).2 with
      | none => simp [gather, response]
      | some assignment =>
          by_cases same : (vector 0).val = (outputs 0).1.val
          · simp only [gather, response, same, ↓reduceIte]
            rw [Nat.add_mul]
            omega
          · have head := extract_work_le ring module program correct bounds bounded
              (vector 0).val (outputs 0).1.val base assignment
              (strongSet.differenceUnit (vector 0).property (outputs 0).1.property same)
            have tail := gather_work_le ring module program correct strongSet bounds bounded
              (fun index => vector index.succ) base (fun index => outputs index.succ)
            simp only [gather, response, same, ↓reduceIte]
            rw [Nat.add_mul]
            omega

/-- Initial dispatch is charged on every path, including an absent base. -/
theorem finish_work_le (ring : CommutativeRingOps Scalar) (module : ModuleOps Scalar Assignment)
    (program : Primitives Scalar Assignment) (correct : Correct ring module program)
    (strongSet : StrongSetUnits ring member) (bounds : PrimitiveBounds)
    (bounded : Bounded ring program bounds) {count : Nat}
    (vector : Fin count → {scalar // member scalar}) (initial : Option Assignment)
    (outputs : Fin count → ({scalar // member scalar} × Option Assignment)) :
    (finish program vector initial outputs).work ≤ count * (bounds.coordinateWork + 3) + 2 := by
  cases initial with
  | none => simp [finish]
  | some base =>
      have work := gather_work_le ring module program correct strongSet bounds bounded vector base outputs
      simpa only [finish] using Nat.add_le_add_right work 1

end NightstreamFPrime.Spec.Folding.PiRLC.CoordinateTerminalProgram

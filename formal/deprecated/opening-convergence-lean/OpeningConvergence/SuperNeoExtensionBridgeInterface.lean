import OpeningConvergence.Basic
import OpeningConvergence.SuperNeoBoundaryInterface
import SuperNeo.Primitives.ExtensionFieldInterface

/-!
# Module 6: SuperNeoExtensionBridge — Interface

Owns the canonical theorem-facing bridge from extension-field opening claims
over `SuperNeo.KExt` to the current base-field SuperNeo CE statement surface.

This module does not prove that the current CE relation is already the final
extension-field semantic boundary. It freezes the exact split encoding that
the remaining repo-wide bridge theorem must use.

## Spec
See `specs/SuperNeoExtensionBridge.spec.md`
-/

namespace OpeningConvergence.SuperNeoExtensionBridge

abbrev K := SuperNeo.ExtensionFieldInterface.KExt
abbrev F := SuperNeo.F
abbrev Coeffs := SuperNeo.Coeffs
abbrev OpenedObject := OpeningConvergence.SuperNeoBoundary.OpenedObject
abbrev Registry := OpeningConvergence.SuperNeoBoundary.Registry
abbrev CEStatement := OpeningConvergence.SuperNeoBoundary.CEStatement
abbrev CEWitness := OpeningConvergence.SuperNeoBoundary.CEWitness

/-- Real-part block of one extension-field point. -/
def pointReCoeffs {ell : Nat} (point : Fin ell → K) : Coeffs :=
  Array.ofFn fun i => (point i).re

/-- Imaginary-part block of one extension-field point. -/
def pointImCoeffs {ell : Nat} (point : Fin ell → K) : Coeffs :=
  Array.ofFn fun i => (point i).im

/-- Canonical flattened base-field encoding of one extension-field point:
    all real coordinates, then all imaginary coordinates. -/
def pointToBaseCoeffs {ell : Nat} (point : Fin ell → K) : Coeffs :=
  pointReCoeffs point ++ pointImCoeffs point

@[simp] theorem pointReCoeffs_size {ell : Nat} (point : Fin ell → K) :
    (pointReCoeffs point).size = ell := by
  simp [pointReCoeffs]

@[simp] theorem pointImCoeffs_size {ell : Nat} (point : Fin ell → K) :
    (pointImCoeffs point).size = ell := by
  simp [pointImCoeffs]

@[simp] theorem pointToBaseCoeffs_size {ell : Nat} (point : Fin ell → K) :
    (pointToBaseCoeffs point).size = 2 * ell := by
  simp [pointToBaseCoeffs]
  omega

/-- Real-part coefficient vector of one packed extension-field column. -/
def packedColumnReCoeffs (eval : PackedColumnEval K) : Coeffs :=
  Array.ofFn fun t => (eval.coeffs t).re

/-- Imaginary-part coefficient vector of one packed extension-field column. -/
def packedColumnImCoeffs (eval : PackedColumnEval K) : Coeffs :=
  Array.ofFn fun t => (eval.coeffs t).im

@[simp] theorem packedColumnReCoeffs_size (eval : PackedColumnEval K) :
    (packedColumnReCoeffs eval).size = AJTAI_D := by
  simp [packedColumnReCoeffs]

@[simp] theorem packedColumnImCoeffs_size (eval : PackedColumnEval K) :
    (packedColumnImCoeffs eval).size = AJTAI_D := by
  simp [packedColumnImCoeffs]

/-- Real-part evaluation blocks for one payload. -/
def payloadReEvaluations (payload : FamilyEvalPayload K) : Array Coeffs :=
  Array.ofFn fun j => packedColumnReCoeffs (payload.columnEvals j)

/-- Imaginary-part evaluation blocks for one payload. -/
def payloadImEvaluations (payload : FamilyEvalPayload K) : Array Coeffs :=
  Array.ofFn fun j => packedColumnImCoeffs (payload.columnEvals j)

/-- Canonical flattened payload encoding: all real packed-column blocks, then
    all imaginary packed-column blocks. -/
def payloadToSplitEvaluations (payload : FamilyEvalPayload K) : Array Coeffs :=
  payloadReEvaluations payload ++ payloadImEvaluations payload

@[simp] theorem payloadReEvaluations_size (payload : FamilyEvalPayload K) :
    (payloadReEvaluations payload).size = packedColumnCount payload.schema := by
  simp [payloadReEvaluations]

@[simp] theorem payloadImEvaluations_size (payload : FamilyEvalPayload K) :
    (payloadImEvaluations payload).size = packedColumnCount payload.schema := by
  simp [payloadImEvaluations]

@[simp] theorem payloadToSplitEvaluations_size (payload : FamilyEvalPayload K) :
    (payloadToSplitEvaluations payload).size = 2 * packedColumnCount payload.schema := by
  simp [payloadToSplitEvaluations]
  omega

/-- Build the current base-field CE statement induced by one extension-field
    opening claim. -/
def claimStatementK
    (obj : OpenedObject)
    {ell : Nat}
    (point : Fin ell → K)
    (payload : FamilyEvalPayload K) : CEStatement where
  commitment := obj.commitment
  publicInput := obj.publicInput
  point := pointToBaseCoeffs point
  evaluations := payloadToSplitEvaluations payload

@[simp] theorem claimStatementK_point_size
    (obj : OpenedObject)
    {ell : Nat}
    (point : Fin ell → K)
    (payload : FamilyEvalPayload K) :
    (claimStatementK obj point payload).point.size = 2 * ell := by
  simp [claimStatementK]

@[simp] theorem claimStatementK_evaluations_size
    (obj : OpenedObject)
    {ell : Nat}
    (point : Fin ell → K)
    (payload : FamilyEvalPayload K) :
    (claimStatementK obj point payload).evaluations.size =
      2 * packedColumnCount payload.schema := by
  simp [claimStatementK]

@[simp] theorem pointToBaseCoeffs_getD_re
    {ell : Nat}
    (point : Fin ell → K)
    {i : Nat}
    (hi : i < ell) :
    (pointToBaseCoeffs point).getD i 0 = (point ⟨i, hi⟩).re := by
  have hiSplit : i < (pointToBaseCoeffs point).size := by
    rw [pointToBaseCoeffs_size]
    omega
  rw [← Array.getElem_eq_getD (xs := pointToBaseCoeffs point) (i := i) (h := hiSplit) (fallback := 0)]
  calc
    (pointToBaseCoeffs point)[i]'hiSplit =
        (pointReCoeffs point)[i]'(by simpa [pointReCoeffs] using hi) := by
      simpa [pointToBaseCoeffs] using
        (Array.getElem_append_left
          (xs := pointReCoeffs point)
          (ys := pointImCoeffs point)
          (i := i)
          (h := hiSplit)
          (hlt := by simpa [pointReCoeffs] using hi))
    _ = (point ⟨i, hi⟩).re := by
      simp [pointReCoeffs]

@[simp] theorem pointToBaseCoeffs_getD_im
    {ell : Nat}
    (point : Fin ell → K)
    {i : Nat}
    (hi : i < ell) :
    (pointToBaseCoeffs point).getD (ell + i) 0 = (point ⟨i, hi⟩).im := by
  have hiSplit : ell + i < (pointToBaseCoeffs point).size := by
    rw [pointToBaseCoeffs_size]
    omega
  rw [← Array.getElem_eq_getD (xs := pointToBaseCoeffs point) (i := ell + i) (h := hiSplit) (fallback := 0)]
  calc
    (pointToBaseCoeffs point)[ell + i]'hiSplit =
        (pointImCoeffs point)[i]'(by simpa [pointImCoeffs] using hi) := by
      simpa [pointToBaseCoeffs, pointReCoeffs] using
        (Array.getElem_append_right
          (xs := pointReCoeffs point)
          (ys := pointImCoeffs point)
          (i := ell + i)
          (h := hiSplit)
          (hle := by omega))
    _ = (point ⟨i, hi⟩).im := by
      have hiIm : i < (pointImCoeffs point).size := by
        simpa [pointImCoeffs] using hi
      change (Array.ofFn fun k => (point k).im)[i]'hiIm = (point ⟨i, hi⟩).im
      simp

@[simp] theorem payloadToSplitEvaluations_getD_re
    (payload : FamilyEvalPayload K)
    {j : Nat}
    (hj : j < packedColumnCount payload.schema) :
    (payloadToSplitEvaluations payload).getD j #[] =
      packedColumnReCoeffs (payload.columnEvals ⟨j, hj⟩) := by
  have hjSplit : j < (payloadToSplitEvaluations payload).size := by
    rw [payloadToSplitEvaluations_size]
    omega
  rw [← Array.getElem_eq_getD (xs := payloadToSplitEvaluations payload) (i := j) (h := hjSplit) (fallback := #[])]
  calc
    (payloadToSplitEvaluations payload)[j]'hjSplit =
        (payloadReEvaluations payload)[j]'(by simpa [payloadReEvaluations] using hj) := by
      simpa [payloadToSplitEvaluations] using
        (Array.getElem_append_left
          (xs := payloadReEvaluations payload)
          (ys := payloadImEvaluations payload)
          (i := j)
          (h := hjSplit)
          (hlt := by simpa [payloadReEvaluations] using hj))
    _ = packedColumnReCoeffs (payload.columnEvals ⟨j, hj⟩) := by
      simp [payloadReEvaluations]

@[simp] theorem payloadToSplitEvaluations_getD_im
    (payload : FamilyEvalPayload K)
    {j : Nat}
    (hj : j < packedColumnCount payload.schema) :
    (payloadToSplitEvaluations payload).getD (packedColumnCount payload.schema + j) #[] =
      packedColumnImCoeffs (payload.columnEvals ⟨j, hj⟩) := by
  have hjSplit : packedColumnCount payload.schema + j < (payloadToSplitEvaluations payload).size := by
    rw [payloadToSplitEvaluations_size]
    omega
  rw [← Array.getElem_eq_getD
      (xs := payloadToSplitEvaluations payload)
      (i := packedColumnCount payload.schema + j)
      (h := hjSplit)
      (fallback := #[])]
  calc
    (payloadToSplitEvaluations payload)[packedColumnCount payload.schema + j]'hjSplit =
        (payloadImEvaluations payload)[j]'(by simpa [payloadImEvaluations] using hj) := by
      simpa [payloadToSplitEvaluations, payloadReEvaluations] using
        (Array.getElem_append_right
          (xs := payloadReEvaluations payload)
          (ys := payloadImEvaluations payload)
          (i := packedColumnCount payload.schema + j)
          (h := hjSplit)
          (hle := by omega))
    _ = packedColumnImCoeffs (payload.columnEvals ⟨j, hj⟩) := by
      simp [payloadImEvaluations]

@[simp] theorem payloadToSplitEvaluations_getD_re_coeff
    (payload : FamilyEvalPayload K)
    {j : Nat}
    (hj : j < packedColumnCount payload.schema)
    {t : Nat}
    (ht : t < AJTAI_D) :
    ((payloadToSplitEvaluations payload).getD j #[]).getD t 0 =
      ((payload.columnEvals ⟨j, hj⟩).coeffs ⟨t, ht⟩).re := by
  rw [payloadToSplitEvaluations_getD_re payload hj]
  rw [← Array.getElem_eq_getD
      (xs := packedColumnReCoeffs (payload.columnEvals ⟨j, hj⟩))
      (i := t)
      (h := by simpa [packedColumnReCoeffs] using ht)
      (fallback := 0)]
  simp [packedColumnReCoeffs]

@[simp] theorem payloadToSplitEvaluations_getD_im_coeff
    (payload : FamilyEvalPayload K)
    {j : Nat}
    (hj : j < packedColumnCount payload.schema)
    {t : Nat}
    (ht : t < AJTAI_D) :
    ((payloadToSplitEvaluations payload).getD (packedColumnCount payload.schema + j) #[]).getD t 0 =
      ((payload.columnEvals ⟨j, hj⟩).coeffs ⟨t, ht⟩).im := by
  rw [payloadToSplitEvaluations_getD_im payload hj]
  rw [← Array.getElem_eq_getD
      (xs := packedColumnImCoeffs (payload.columnEvals ⟨j, hj⟩))
      (i := t)
      (h := by simpa [packedColumnImCoeffs] using ht)
      (fallback := 0)]
  simp [packedColumnImCoeffs]

/-- Concrete extension-field PCS boundary induced by the current base-field
    SuperNeo CE surface together with the canonical split encoding. -/
def boundaryK (registry : Registry) : AjtaiPCSBoundary K where
  verify := fun {ell} objectId point payload =>
    ∃ obj : OpenedObject,
      ∃ wit : CEWitness,
        registry.lookup objectId = some obj ∧
        obj.schema = payload.schema ∧
        obj.rowDomainLogSize = ell ∧
        OpeningConvergence.SuperNeoBoundary.CEHolds
          obj.ce
          (claimStatementK obj point payload)
          wit

/-- Registry soundness for the extension-field boundary. -/
theorem boundaryK_lookup_self
    (registry : Registry)
    {id : OpenedObjectId}
    {obj : OpenedObject}
    (hLookup : registry.lookup id = some obj) :
    obj.id = id :=
  OpeningConvergence.SuperNeoBoundary.boundary_lookup_self registry hLookup

/-- Local claim-satisfaction predicate induced by the extension-field boundary. -/
abbrev ClaimSatisfiedK
    (registry : Registry)
    {ell : Nat}
    (claim : FamilyEvalClaim K ell) : Prop :=
  (boundaryK registry).verify claim.openedObject claim.point claim.payload

/-- The split real/imag point blocks determine the original extension-field
    point exactly. -/
theorem point_eq_of_split_blocks_eq
    {ell : Nat}
    {x y : Fin ell → K}
    (hRe : pointReCoeffs x = pointReCoeffs y)
    (hIm : pointImCoeffs x = pointImCoeffs y) :
    x = y := by
  funext i
  apply SuperNeo.ExtensionFieldInterface.KExt_ext
  · have h :=
      congrArg (fun arr => arr[i.1]!) hRe
    simpa [pointReCoeffs] using h
  · have h :=
      congrArg (fun arr => arr[i.1]!) hIm
    simpa [pointImCoeffs] using h

/-- The split real/imag coefficient blocks determine the original packed
    extension-field column exactly. -/
theorem packedColumn_eq_of_split_blocks_eq
    {x y : PackedColumnEval K}
    (hRe : packedColumnReCoeffs x = packedColumnReCoeffs y)
    (hIm : packedColumnImCoeffs x = packedColumnImCoeffs y) :
    x = y := by
  apply PackedColumnEval.ext
  funext t
  apply SuperNeo.ExtensionFieldInterface.KExt_ext
  · have h :=
      congrArg (fun arr => arr[t.1]!) hRe
    simpa [packedColumnReCoeffs] using h
  · have h :=
      congrArg (fun arr => arr[t.1]!) hIm
    simpa [packedColumnImCoeffs] using h

/-- The canonical flattened point encoding is injective. -/
theorem pointToBaseCoeffs_injective
    {ell : Nat}
    {x y : Fin ell → K}
    (hSplit : pointToBaseCoeffs x = pointToBaseCoeffs y) :
    x = y := by
  funext i
  apply SuperNeo.ExtensionFieldInterface.KExt_ext
  · have h :
        (pointToBaseCoeffs x).getD i.1 0 = (pointToBaseCoeffs y).getD i.1 0 := by
      simpa using congrArg (fun arr => arr.getD i.1 0) hSplit
    simpa [pointToBaseCoeffs_getD_re (point := x) i.2, pointToBaseCoeffs_getD_re (point := y) i.2] using h
  · have h :
        (pointToBaseCoeffs x).getD (ell + i.1) 0 =
          (pointToBaseCoeffs y).getD (ell + i.1) 0 := by
      simpa using congrArg (fun arr => arr.getD (ell + i.1) 0) hSplit
    simpa [pointToBaseCoeffs_getD_im (point := x) i.2, pointToBaseCoeffs_getD_im (point := y) i.2] using h

/-- The canonical split payload encoding is injective once the schema is fixed. -/
theorem payloadToSplitEvaluations_injective
    {x y : FamilyEvalPayload K}
    (hSchema : x.schema = y.schema)
    (hSplit : payloadToSplitEvaluations x = payloadToSplitEvaluations y) :
    x = y := by
  cases x with
  | mk schemaX colsX =>
    cases y with
    | mk schemaY colsY =>
      cases hSchema
      have hCols :
          ∀ j : Fin (packedColumnCount schemaX),
            colsX j = colsY j := by
        intro j
        apply PackedColumnEval.ext
        funext t
        apply SuperNeo.ExtensionFieldInterface.KExt_ext
        · have hCoeff :
              ((payloadToSplitEvaluations { schema := schemaX, columnEvals := colsX }).getD j.1 #[]).getD t.1 0 =
                ((payloadToSplitEvaluations { schema := schemaX, columnEvals := colsY }).getD j.1 #[]).getD t.1 0 := by
            simpa using congrArg (fun coeffs => coeffs.getD t.1 0)
              (congrArg (fun arr => arr.getD j.1 #[]) hSplit)
          have hx :
              ((payloadToSplitEvaluations { schema := schemaX, columnEvals := colsX }).getD j.1 #[]).getD t.1 0 =
                ((colsX j).coeffs t).re := by
            simpa using
              (payloadToSplitEvaluations_getD_re_coeff
                (payload := { schema := schemaX, columnEvals := colsX })
                (hj := j.2)
                (ht := t.2))
          have hy :
              ((payloadToSplitEvaluations { schema := schemaX, columnEvals := colsY }).getD j.1 #[]).getD t.1 0 =
                ((colsY j).coeffs t).re := by
            simpa using
              (payloadToSplitEvaluations_getD_re_coeff
                (payload := { schema := schemaX, columnEvals := colsY })
                (hj := j.2)
                (ht := t.2))
          calc
            ((colsX j).coeffs t).re =
                ((payloadToSplitEvaluations { schema := schemaX, columnEvals := colsX }).getD j.1 #[]).getD t.1 0 := by
                  symm
                  exact hx
            _ = ((payloadToSplitEvaluations { schema := schemaX, columnEvals := colsY }).getD j.1 #[]).getD t.1 0 := hCoeff
            _ = ((colsY j).coeffs t).re := by
                  exact hy
        · have hCoeff :
              ((payloadToSplitEvaluations { schema := schemaX, columnEvals := colsX }).getD (packedColumnCount schemaX + j.1) #[]).getD t.1 0 =
                ((payloadToSplitEvaluations { schema := schemaX, columnEvals := colsY }).getD (packedColumnCount schemaX + j.1) #[]).getD t.1 0 := by
            simpa using congrArg (fun coeffs => coeffs.getD t.1 0)
              (congrArg (fun arr => arr.getD (packedColumnCount schemaX + j.1) #[]) hSplit)
          have hx :
              ((payloadToSplitEvaluations { schema := schemaX, columnEvals := colsX }).getD (packedColumnCount schemaX + j.1) #[]).getD t.1 0 =
                ((colsX j).coeffs t).im := by
            simpa using
              (payloadToSplitEvaluations_getD_im_coeff
                (payload := { schema := schemaX, columnEvals := colsX })
                (hj := j.2)
                (ht := t.2))
          have hy :
              ((payloadToSplitEvaluations { schema := schemaX, columnEvals := colsY }).getD (packedColumnCount schemaX + j.1) #[]).getD t.1 0 =
                ((colsY j).coeffs t).im := by
            simpa using
              (payloadToSplitEvaluations_getD_im_coeff
                (payload := { schema := schemaX, columnEvals := colsY })
                (hj := j.2)
                (ht := t.2))
          calc
            ((colsX j).coeffs t).im =
                ((payloadToSplitEvaluations { schema := schemaX, columnEvals := colsX }).getD (packedColumnCount schemaX + j.1) #[]).getD t.1 0 := by
                  symm
                  exact hx
            _ = ((payloadToSplitEvaluations { schema := schemaX, columnEvals := colsY }).getD (packedColumnCount schemaX + j.1) #[]).getD t.1 0 := hCoeff
            _ = ((colsY j).coeffs t).im := by
                  exact hy
      have hEq : colsX = colsY := funext hCols
      cases hEq
      exact rfl

/-- For one fixed opened object, the split CE statement determines the original
    extension-field point and payload exactly. -/
theorem claimStatementK_injective
    (obj : OpenedObject)
    {ell : Nat}
    {point1 point2 : Fin ell → K}
    {payload1 payload2 : FamilyEvalPayload K}
    (hSchema1 : payload1.schema = obj.schema)
    (hSchema2 : payload2.schema = obj.schema)
    (hStmt : claimStatementK obj point1 payload1 = claimStatementK obj point2 payload2) :
    point1 = point2 ∧ payload1 = payload2 := by
  have hPoint :
      pointToBaseCoeffs point1 = pointToBaseCoeffs point2 := by
    simpa [claimStatementK] using congrArg (fun stmt => stmt.point) hStmt
  have hPayload :
      payloadToSplitEvaluations payload1 = payloadToSplitEvaluations payload2 := by
    simpa [claimStatementK] using congrArg (fun stmt => stmt.evaluations) hStmt
  refine ⟨pointToBaseCoeffs_injective hPoint, ?_⟩
  exact payloadToSplitEvaluations_injective (hSchema1.trans hSchema2.symm) hPayload

end OpeningConvergence.SuperNeoExtensionBridge

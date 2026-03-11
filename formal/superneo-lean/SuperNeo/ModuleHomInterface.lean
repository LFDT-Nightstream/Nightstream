import SuperNeo.ModuleHom

/-!
Contract interface for `SuperNeo.ModuleHom`.

Spec: `specs/ModuleHom.spec.md`

Paper anchors:
- Theorem 5, Section 5, lines 390-400: R_F-module homomorphisms L and L_in.
-/

namespace SuperNeo

namespace ModuleHomInterface

/-! ## Core Structures -/

/-- [Role: Definitional] Vector-valued module-hom map surface. -/
abbrev VecModuleHom := SuperNeo.VecModuleHom

/-- [Role: Definitional] Scalar-valued module-hom map surface. -/
abbrev ScalarModuleHom := SuperNeo.ScalarModuleHom

/-- [Role: Definitional] One-point vector linearity proposition pair. -/
abbrev vecModulePropPair := SuperNeo.vecModulePropPair

/-- [Role: Definitional] One-point scalar linearity proposition pair. -/
abbrev scalarModulePropPair := SuperNeo.scalarModulePropPair

/-- [Role: Definitional] Executable check form for vector one-point linearity. -/
abbrev vecModuleCheckPair := SuperNeo.vecModuleCheckPair

/-- [Role: Definitional] Executable check form for scalar one-point linearity. -/
abbrev scalarModuleCheckPair := SuperNeo.scalarModuleCheckPair

/-- [Role: Definitional] Executable additivity check for vector maps. -/
abbrev preservesAddVec := SuperNeo.preservesAddVec

/-- [Role: Definitional] Executable scaling check for vector maps. -/
abbrev preservesScaleVec := SuperNeo.preservesScaleVec

/-- [Role: Definitional] Executable additivity check for scalar maps. -/
abbrev preservesAddScalar := SuperNeo.preservesAddScalar

/-- [Role: Definitional] Executable scaling check for scalar maps. -/
abbrev preservesScaleScalar := SuperNeo.preservesScaleScalar

/-! ## Key Theorems -/

/-- [Role: Theorem-Target] `preservesAddVec = true` implies vector additivity. -/
theorem preservesAddVec_sound
  {h : VecModuleHom} {x y : Array F} :
  preservesAddVec h x y = true →
  h.map (vecAdd x y) = vecAdd (h.map x) (h.map y) :=
  SuperNeo.preservesAddVec_sound

/-- [Role: Theorem-Target] `preservesScaleVec = true` implies vector scaling law. -/
theorem preservesScaleVec_sound
  {h : VecModuleHom} {s : F} {x : Array F} :
  preservesScaleVec h s x = true →
  h.map (vecScale s x) = vecScale s (h.map x) :=
  SuperNeo.preservesScaleVec_sound

/-- [Role: Theorem-Target] `preservesAddScalar = true` implies scalar additivity. -/
theorem preservesAddScalar_sound
  {h : ScalarModuleHom} {x y : Array F} :
  preservesAddScalar h x y = true →
  h.map (vecAdd x y) = h.map x + h.map y :=
  SuperNeo.preservesAddScalar_sound

/-- [Role: Theorem-Target] `preservesScaleScalar = true` implies scalar scaling law. -/
theorem preservesScaleScalar_sound
  {h : ScalarModuleHom} {s : F} {x : Array F} :
  preservesScaleScalar h s x = true →
  h.map (vecScale s x) = s * h.map x :=
  SuperNeo.preservesScaleScalar_sound

/-- [Role: Theorem-Target] Vector additivity + shape premise implies `preservesAddVec = true`. -/
theorem preservesAddVec_complete
  {h : VecModuleHom} {x y : Array F} :
  x.size = y.size →
  h.map (vecAdd x y) = vecAdd (h.map x) (h.map y) →
  preservesAddVec h x y = true :=
  SuperNeo.preservesAddVec_complete

/-- [Role: Theorem-Target] Vector scaling law implies `preservesScaleVec = true`. -/
theorem preservesScaleVec_complete
  {h : VecModuleHom} {s : F} {x : Array F} :
  h.map (vecScale s x) = vecScale s (h.map x) →
  preservesScaleVec h s x = true :=
  SuperNeo.preservesScaleVec_complete

/-- [Role: Theorem-Target] Scalar additivity + shape premise implies `preservesAddScalar = true`. -/
theorem preservesAddScalar_complete
  {h : ScalarModuleHom} {x y : Array F} :
  x.size = y.size →
  h.map (vecAdd x y) = h.map x + h.map y →
  preservesAddScalar h x y = true :=
  SuperNeo.preservesAddScalar_complete

/-- [Role: Theorem-Target] Scalar scaling law implies `preservesScaleScalar = true`. -/
theorem preservesScaleScalar_complete
  {h : ScalarModuleHom} {s : F} {x : Array F} :
  h.map (vecScale s x) = s * h.map x →
  preservesScaleScalar h s x = true :=
  SuperNeo.preservesScaleScalar_complete

/-- [Role: Theorem-Target] Check-pair success implies vector proposition pair. -/
theorem vecModulePropPair_of_checkPair
  {h : VecModuleHom} {s : F} {x y : Array F} :
  vecModuleCheckPair h s x y = true →
  vecModulePropPair h s x y :=
  SuperNeo.vecModulePropPair_of_checkPair

/-- [Role: Theorem-Target] Vector proposition pair implies check-pair success. -/
theorem vecModuleCheckPair_of_propPair
  {h : VecModuleHom} {s : F} {x y : Array F} :
  vecModulePropPair h s x y →
  vecModuleCheckPair h s x y = true :=
  SuperNeo.vecModuleCheckPair_of_propPair

/-- [Role: Theorem-Target] Vector check-pair truth is equivalent to the proposition pair. -/
theorem vecModuleCheckPair_iff_propPair
  {h : VecModuleHom} {s : F} {x y : Array F} :
  vecModuleCheckPair h s x y = true ↔ vecModulePropPair h s x y :=
  SuperNeo.vecModuleCheckPair_iff_propPair

/-- [Role: Theorem-Target] Check-pair success implies scalar proposition pair. -/
theorem scalarModulePropPair_of_checkPair
  {h : ScalarModuleHom} {s : F} {x y : Array F} :
  scalarModuleCheckPair h s x y = true →
  scalarModulePropPair h s x y :=
  SuperNeo.scalarModulePropPair_of_checkPair

/-- [Role: Theorem-Target] Scalar proposition pair implies check-pair success. -/
theorem scalarModuleCheckPair_of_propPair
  {h : ScalarModuleHom} {s : F} {x y : Array F} :
  scalarModulePropPair h s x y →
  scalarModuleCheckPair h s x y = true :=
  SuperNeo.scalarModuleCheckPair_of_propPair

/-- [Role: Theorem-Target] Scalar check-pair truth is equivalent to the proposition pair. -/
theorem scalarModuleCheckPair_iff_propPair
  {h : ScalarModuleHom} {s : F} {x y : Array F} :
  scalarModuleCheckPair h s x y = true ↔ scalarModulePropPair h s x y :=
  SuperNeo.scalarModuleCheckPair_iff_propPair

/-! ## Universal Contracts -/

/--
[Role: Definitional] Theorem-facing universal vector module-hom contract:
`map(x+y)=map(x)+map(y)` and `map(s•x)=s•map(x)` for all inputs.
-/
abbrev vecModuleAssumption := SuperNeo.vecModuleAssumption

/--
[Role: Definitional] Theorem-facing universal scalar module-hom contract:
`map(x+y)=map(x)+map(y)` and `map(s•x)=s*map(x)` for all inputs.
-/
abbrev scalarModuleAssumption := SuperNeo.scalarModuleAssumption

/-- [Role: Definitional] Check-facing universal vector contract. -/
abbrev vecModuleCheckAssumption := SuperNeo.vecModuleCheckAssumption

/-- [Role: Definitional] Check-facing universal scalar contract. -/
abbrev scalarModuleCheckAssumption := SuperNeo.scalarModuleCheckAssumption

/-- [Role: Theorem-Target] Universal vector check contract implies universal vector theorem contract. -/
theorem vecModuleAssumption_of_checkAssumption
  {h : VecModuleHom} :
  vecModuleCheckAssumption h →
  vecModuleAssumption h :=
  SuperNeo.vecModuleAssumption_of_checkAssumption

/-- [Role: Theorem-Target] Universal vector theorem contract implies universal vector check contract. -/
theorem vecModuleCheckAssumption_of_assumption
  {h : VecModuleHom} :
  vecModuleAssumption h →
  vecModuleCheckAssumption h :=
  SuperNeo.vecModuleCheckAssumption_of_assumption

/-- [Role: Theorem-Target] Universal vector theorem/check contracts are equivalent. -/
theorem vecModuleAssumption_iff_checkAssumption
  {h : VecModuleHom} :
  vecModuleAssumption h ↔ vecModuleCheckAssumption h :=
  SuperNeo.vecModuleAssumption_iff_checkAssumption

/-- [Role: Theorem-Target] Universal scalar check contract implies universal scalar theorem contract. -/
theorem scalarModuleAssumption_of_checkAssumption
  {h : ScalarModuleHom} :
  scalarModuleCheckAssumption h →
  scalarModuleAssumption h :=
  SuperNeo.scalarModuleAssumption_of_checkAssumption

/-- [Role: Theorem-Target] Universal scalar theorem contract implies universal scalar check contract. -/
theorem scalarModuleCheckAssumption_of_assumption
  {h : ScalarModuleHom} :
  scalarModuleAssumption h →
  scalarModuleCheckAssumption h :=
  SuperNeo.scalarModuleCheckAssumption_of_assumption

/-- [Role: Theorem-Target] Universal scalar theorem/check contracts are equivalent. -/
theorem scalarModuleAssumption_iff_checkAssumption
  {h : ScalarModuleHom} :
  scalarModuleAssumption h ↔ scalarModuleCheckAssumption h :=
  SuperNeo.scalarModuleAssumption_iff_checkAssumption

end ModuleHomInterface

end SuperNeo

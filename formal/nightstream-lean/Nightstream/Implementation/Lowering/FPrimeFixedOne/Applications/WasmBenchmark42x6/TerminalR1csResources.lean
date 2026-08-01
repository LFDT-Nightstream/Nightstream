import Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.TerminalR1csCost

/-!
Contract: resource dimensions that row and column counts alone hide in the
direct terminal R1CS.

Assurance tier: model-level.

Owns: the exact number of Ajtai coefficient slots traversed by the direct
compiler, its maximum Ajtai row term count, and the total private-column
width for the selected 42-times-6 fixed point.

Does not own: seed-specific nonzero counts, runtime, memory use, Spartan,
WHIR, or a security reduction. A coefficient slot can contain zero for a
particular setup, so this file does not call the slot count a nonzero count.

Emits constraints: none.
-/

set_option autoImplicit false
set_option maxRecDepth 500000

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.TerminalR1csResources

open Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.NativeFixedPointFamily
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.NativeFixedPointCost
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.NativeFixedPointSource
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.TerminalR1csCost
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81

/-- One Ajtai output equation visits every carrier coordinate. There are
`commitmentRows * ringDegree` output equations per terminal witness. -/
noncomputable def ajtaiCoefficientSlotsPerClaim (template : Template) : Nat :=
  commitmentRows * ringDegree * (compiledShape template).carrierWidth

/-- The terminal statement carries fourteen running witnesses and one fresh
witness, each linked to its Ajtai commitment. -/
noncomputable def ajtaiCoefficientSlots (template : Template) : Nat :=
  (productionGlobalParams.k + 1) * ajtaiCoefficientSlotsPerClaim template

/-- Terms in the densest direct R1CS Ajtai row: every witness coordinate,
the constant-one side of the equality row, and its public output. -/
noncomputable def maximumAjtaiRowTermSlots (template : Template) : Nat :=
  (compiledShape template).carrierWidth + 2

/-- Private columns that the direct terminal compiler would keep live for
one Spartan witness. -/
noncomputable def privateColumns (template : Template) : Nat :=
  (terminalCost template).committedColumns +
    (terminalCost template).auxiliaryColumns

theorem ajtaiCoefficientSlotsPerClaim_exact (template : Template) :
    ajtaiCoefficientSlotsPerClaim template = 5_204_657_592 := by
  rw [ajtaiCoefficientSlotsPerClaim, compiledShape_eq]
  rfl

theorem ajtaiCoefficientSlots_exact (template : Template) :
    ajtaiCoefficientSlots template = 78_069_863_880 := by
  rw [ajtaiCoefficientSlots, ajtaiCoefficientSlotsPerClaim_exact]
  rfl

theorem maximumAjtaiRowTermSlots_exact (template : Template) :
    maximumAjtaiRowTermSlots template = 5_354_588 := by
  rw [maximumAjtaiRowTermSlots, compiledShape_eq]
  rfl

theorem privateColumns_exact (template : Template) :
    privateColumns template = 165_937_070 := by
  rw [privateColumns, terminalCost_exact]
  rfl

/-- The direct terminal compiler is coefficient-bound even before Spartan:
its Ajtai linear forms alone exceed seventy-eight billion coefficient slots. -/
theorem ajtaiCoefficientSlots_exceed_seventyEightBillion
    (template : Template) :
    78_000_000_000 < ajtaiCoefficientSlots template := by
  rw [ajtaiCoefficientSlots_exact]
  decide

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.TerminalR1csResources

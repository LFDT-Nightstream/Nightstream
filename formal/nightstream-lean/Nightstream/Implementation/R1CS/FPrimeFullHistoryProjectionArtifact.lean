import Nightstream.Implementation.R1CS.FPrimeFullHistoryProjectionCensusRecursive0
import Nightstream.Implementation.R1CS.FPrimeFullHistoryProjectionCensusRecursive1
import Nightstream.Implementation.R1CS.FPrimeFullHistoryProjectionCensusRecursive2
import Nightstream.Implementation.R1CS.FPrimeFullHistoryProjectionCensusRecursive3
import Nightstream.Implementation.R1CS.FPrimeFullHistoryProjectionCensusRecursive4
import Nightstream.Implementation.R1CS.FPrimeFullHistoryProjectionCensusRecursive5
import Nightstream.Implementation.R1CS.FPrimeFullHistoryProjectionCensusRecursive6
import Nightstream.Implementation.R1CS.FPrimeFullHistoryProjectionCensusRecursive7
import Nightstream.Implementation.R1CS.FPrimeFullHistoryProjectionCensusTerminal0
import Nightstream.Implementation.R1CS.FPrimeFullHistoryProjectionCensusTerminal1
import Nightstream.Implementation.R1CS.FPrimeFullHistoryProjectionCensusTerminal2
import Nightstream.Implementation.R1CS.FPrimeFullHistoryProjectionCensusTerminal3
import Nightstream.Implementation.R1CS.FPrimeFullHistoryProjectionCensusTerminal4
import Nightstream.Implementation.R1CS.FPrimeFullHistoryProjectionCensusTerminal5
import Nightstream.Implementation.R1CS.FPrimeFullHistoryProjectionCensusTerminal6
import Nightstream.Implementation.R1CS.FPrimeFullHistoryProjectionCensusTerminal7
import Nightstream.Implementation.R1CS.FPrimeFullHistoryProjectionCensusTerminal8
import Nightstream.Implementation.R1CS.FPrimeFullHistoryProjectionCensusTerminal9
import Nightstream.Implementation.R1CS.FPrimeFullHistoryProjectionCensusTerminal10
import Nightstream.Implementation.R1CS.FPrimeFullHistoryProjectionCensusTerminal11
import Nightstream.Implementation.R1CS.FPrimeFullHistoryProjectionCensusTerminal12
import Nightstream.Implementation.R1CS.FPrimeFullHistoryProjectionCensusTerminal13
import Nightstream.Implementation.R1CS.FPrimeFullHistoryProjectionCensusTerminal14
import Nightstream.Implementation.R1CS.FPrimeFullHistoryProjectionCensusTerminal15
import Nightstream.Implementation.R1CS.FPrimeFullHistoryProjectionCensusTerminal16
import Nightstream.Implementation.R1CS.FPrimeFullHistoryProjectionCensusTerminal17
import Nightstream.Implementation.R1CS.FPrimeFullHistoryProjectionCensusTerminal18
import Nightstream.Implementation.R1CS.FPrimeFullHistoryProjectionCensusTerminal19
import Nightstream.Implementation.R1CS.FPrimeFullHistoryProjectionCensusTerminal20
import Nightstream.Implementation.R1CS.FPrimeFullHistoryProjectionCensusTerminal21
import Nightstream.Implementation.R1CS.FPrimeFullHistoryProjectionCensusTerminal22
import Nightstream.Implementation.R1CS.FPrimeFullHistoryProjectionCensusTerminal23
import Nightstream.Implementation.R1CS.FPrimeFullHistoryProjectionCensusTerminal24
import Nightstream.Implementation.R1CS.FPrimeFullHistoryProjectionCensusTerminal25
import Nightstream.Implementation.R1CS.FPrimeFullHistoryProjectionCensusTerminal26
import Nightstream.Implementation.R1CS.FPrimeFullHistoryProjectionCensusTerminal27
import Nightstream.Implementation.R1CS.FPrimeFullHistoryProjectionCensusTerminal28
import Nightstream.Implementation.R1CS.FPrimeFullHistoryProjectionCensusTerminal29
import Nightstream.Implementation.R1CS.FPrimeFullHistoryProjectionCensusTerminal30
import Nightstream.Implementation.R1CS.FPrimeFullHistoryProjectionRoles

/-! Complete exact PiRLC projection census for the two-step full-history profile. -/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryProjection

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.ProjectionProgram

set_option maxRecDepth 524288

def recursiveTraces : List ProjectionTrace :=
    GeneratedRecursiveCensus.traces0 ++
    GeneratedRecursiveCensus.traces1 ++
    GeneratedRecursiveCensus.traces2 ++
    GeneratedRecursiveCensus.traces3 ++
    GeneratedRecursiveCensus.traces4 ++
    GeneratedRecursiveCensus.traces5 ++
    GeneratedRecursiveCensus.traces6 ++
    GeneratedRecursiveCensus.traces7

def terminalTraces : List ProjectionTrace :=
    GeneratedTerminalCensus.traces0 ++
    GeneratedTerminalCensus.traces1 ++
    GeneratedTerminalCensus.traces2 ++
    GeneratedTerminalCensus.traces3 ++
    GeneratedTerminalCensus.traces4 ++
    GeneratedTerminalCensus.traces5 ++
    GeneratedTerminalCensus.traces6 ++
    GeneratedTerminalCensus.traces7 ++
    GeneratedTerminalCensus.traces8 ++
    GeneratedTerminalCensus.traces9 ++
    GeneratedTerminalCensus.traces10 ++
    GeneratedTerminalCensus.traces11 ++
    GeneratedTerminalCensus.traces12 ++
    GeneratedTerminalCensus.traces13 ++
    GeneratedTerminalCensus.traces14 ++
    GeneratedTerminalCensus.traces15 ++
    GeneratedTerminalCensus.traces16 ++
    GeneratedTerminalCensus.traces17 ++
    GeneratedTerminalCensus.traces18 ++
    GeneratedTerminalCensus.traces19 ++
    GeneratedTerminalCensus.traces20 ++
    GeneratedTerminalCensus.traces21 ++
    GeneratedTerminalCensus.traces22 ++
    GeneratedTerminalCensus.traces23 ++
    GeneratedTerminalCensus.traces24 ++
    GeneratedTerminalCensus.traces25 ++
    GeneratedTerminalCensus.traces26 ++
    GeneratedTerminalCensus.traces27 ++
    GeneratedTerminalCensus.traces28 ++
    GeneratedTerminalCensus.traces29 ++
    GeneratedTerminalCensus.traces30

def traces : List ProjectionTrace := recursiveTraces ++ terminalTraces

def Holds (assignment : Nat → Nat) : Prop :=
∀ trace ∈ traces, Satisfies (traceRows trace) assignment

private theorem forall_append {α : Type} {P : α → Prop}
{left right : List α}
(leftProof : ∀ value ∈ left, P value)
(rightProof : ∀ value ∈ right, P value) :
∀ value ∈ left ++ right, P value := by
intro value member
rcases List.mem_append.mp member with member | member
· exact leftProof value member
· exact rightProof value member

theorem trace_count : traces.length = 62 := by
simp only [traces, recursiveTraces, terminalTraces, List.length_append, GeneratedRecursiveCensus.trace_count0, GeneratedRecursiveCensus.trace_count1, GeneratedRecursiveCensus.trace_count2, GeneratedRecursiveCensus.trace_count3, GeneratedRecursiveCensus.trace_count4, GeneratedRecursiveCensus.trace_count5, GeneratedRecursiveCensus.trace_count6, GeneratedRecursiveCensus.trace_count7, GeneratedTerminalCensus.trace_count0, GeneratedTerminalCensus.trace_count1, GeneratedTerminalCensus.trace_count2, GeneratedTerminalCensus.trace_count3, GeneratedTerminalCensus.trace_count4, GeneratedTerminalCensus.trace_count5, GeneratedTerminalCensus.trace_count6, GeneratedTerminalCensus.trace_count7, GeneratedTerminalCensus.trace_count8, GeneratedTerminalCensus.trace_count9, GeneratedTerminalCensus.trace_count10, GeneratedTerminalCensus.trace_count11, GeneratedTerminalCensus.trace_count12, GeneratedTerminalCensus.trace_count13, GeneratedTerminalCensus.trace_count14, GeneratedTerminalCensus.trace_count15, GeneratedTerminalCensus.trace_count16, GeneratedTerminalCensus.trace_count17, GeneratedTerminalCensus.trace_count18, GeneratedTerminalCensus.trace_count19, GeneratedTerminalCensus.trace_count20, GeneratedTerminalCensus.trace_count21, GeneratedTerminalCensus.trace_count22, GeneratedTerminalCensus.trace_count23, GeneratedTerminalCensus.trace_count24, GeneratedTerminalCensus.trace_count25, GeneratedTerminalCensus.trace_count26, GeneratedTerminalCensus.trace_count27, GeneratedTerminalCensus.trace_count28, GeneratedTerminalCensus.trace_count29, GeneratedTerminalCensus.trace_count30]

theorem trace_layouts : ∀ trace ∈ traces, trace.LayoutValid := by
unfold traces recursiveTraces terminalTraces
exact forall_append (forall_append
      (forall_append
      (forall_append
      (forall_append
      (forall_append
      (forall_append
      (forall_append
      (GeneratedRecursiveCensus.trace_layouts0)
      (GeneratedRecursiveCensus.trace_layouts1))
      (GeneratedRecursiveCensus.trace_layouts2))
      (GeneratedRecursiveCensus.trace_layouts3))
      (GeneratedRecursiveCensus.trace_layouts4))
      (GeneratedRecursiveCensus.trace_layouts5))
      (GeneratedRecursiveCensus.trace_layouts6))
      (GeneratedRecursiveCensus.trace_layouts7)) (forall_append
      (forall_append
      (forall_append
      (forall_append
      (forall_append
      (forall_append
      (forall_append
      (forall_append
      (forall_append
      (forall_append
      (forall_append
      (forall_append
      (forall_append
      (forall_append
      (forall_append
      (forall_append
      (forall_append
      (forall_append
      (forall_append
      (forall_append
      (forall_append
      (forall_append
      (forall_append
      (forall_append
      (forall_append
      (forall_append
      (forall_append
      (forall_append
      (forall_append
      (forall_append
      (GeneratedTerminalCensus.trace_layouts0)
      (GeneratedTerminalCensus.trace_layouts1))
      (GeneratedTerminalCensus.trace_layouts2))
      (GeneratedTerminalCensus.trace_layouts3))
      (GeneratedTerminalCensus.trace_layouts4))
      (GeneratedTerminalCensus.trace_layouts5))
      (GeneratedTerminalCensus.trace_layouts6))
      (GeneratedTerminalCensus.trace_layouts7))
      (GeneratedTerminalCensus.trace_layouts8))
      (GeneratedTerminalCensus.trace_layouts9))
      (GeneratedTerminalCensus.trace_layouts10))
      (GeneratedTerminalCensus.trace_layouts11))
      (GeneratedTerminalCensus.trace_layouts12))
      (GeneratedTerminalCensus.trace_layouts13))
      (GeneratedTerminalCensus.trace_layouts14))
      (GeneratedTerminalCensus.trace_layouts15))
      (GeneratedTerminalCensus.trace_layouts16))
      (GeneratedTerminalCensus.trace_layouts17))
      (GeneratedTerminalCensus.trace_layouts18))
      (GeneratedTerminalCensus.trace_layouts19))
      (GeneratedTerminalCensus.trace_layouts20))
      (GeneratedTerminalCensus.trace_layouts21))
      (GeneratedTerminalCensus.trace_layouts22))
      (GeneratedTerminalCensus.trace_layouts23))
      (GeneratedTerminalCensus.trace_layouts24))
      (GeneratedTerminalCensus.trace_layouts25))
      (GeneratedTerminalCensus.trace_layouts26))
      (GeneratedTerminalCensus.trace_layouts27))
      (GeneratedTerminalCensus.trace_layouts28))
      (GeneratedTerminalCensus.trace_layouts29))
      (GeneratedTerminalCensus.trace_layouts30))

theorem trace_pairs_nonempty : ∀ trace ∈ traces, trace.pairs ≠ [] := by
unfold traces recursiveTraces terminalTraces
exact forall_append (forall_append
      (forall_append
      (forall_append
      (forall_append
      (forall_append
      (forall_append
      (forall_append
      (GeneratedRecursiveCensus.trace_pairs_nonempty0)
      (GeneratedRecursiveCensus.trace_pairs_nonempty1))
      (GeneratedRecursiveCensus.trace_pairs_nonempty2))
      (GeneratedRecursiveCensus.trace_pairs_nonempty3))
      (GeneratedRecursiveCensus.trace_pairs_nonempty4))
      (GeneratedRecursiveCensus.trace_pairs_nonempty5))
      (GeneratedRecursiveCensus.trace_pairs_nonempty6))
      (GeneratedRecursiveCensus.trace_pairs_nonempty7)) (forall_append
      (forall_append
      (forall_append
      (forall_append
      (forall_append
      (forall_append
      (forall_append
      (forall_append
      (forall_append
      (forall_append
      (forall_append
      (forall_append
      (forall_append
      (forall_append
      (forall_append
      (forall_append
      (forall_append
      (forall_append
      (forall_append
      (forall_append
      (forall_append
      (forall_append
      (forall_append
      (forall_append
      (forall_append
      (forall_append
      (forall_append
      (forall_append
      (forall_append
      (forall_append
      (GeneratedTerminalCensus.trace_pairs_nonempty0)
      (GeneratedTerminalCensus.trace_pairs_nonempty1))
      (GeneratedTerminalCensus.trace_pairs_nonempty2))
      (GeneratedTerminalCensus.trace_pairs_nonempty3))
      (GeneratedTerminalCensus.trace_pairs_nonempty4))
      (GeneratedTerminalCensus.trace_pairs_nonempty5))
      (GeneratedTerminalCensus.trace_pairs_nonempty6))
      (GeneratedTerminalCensus.trace_pairs_nonempty7))
      (GeneratedTerminalCensus.trace_pairs_nonempty8))
      (GeneratedTerminalCensus.trace_pairs_nonempty9))
      (GeneratedTerminalCensus.trace_pairs_nonempty10))
      (GeneratedTerminalCensus.trace_pairs_nonempty11))
      (GeneratedTerminalCensus.trace_pairs_nonempty12))
      (GeneratedTerminalCensus.trace_pairs_nonempty13))
      (GeneratedTerminalCensus.trace_pairs_nonempty14))
      (GeneratedTerminalCensus.trace_pairs_nonempty15))
      (GeneratedTerminalCensus.trace_pairs_nonempty16))
      (GeneratedTerminalCensus.trace_pairs_nonempty17))
      (GeneratedTerminalCensus.trace_pairs_nonempty18))
      (GeneratedTerminalCensus.trace_pairs_nonempty19))
      (GeneratedTerminalCensus.trace_pairs_nonempty20))
      (GeneratedTerminalCensus.trace_pairs_nonempty21))
      (GeneratedTerminalCensus.trace_pairs_nonempty22))
      (GeneratedTerminalCensus.trace_pairs_nonempty23))
      (GeneratedTerminalCensus.trace_pairs_nonempty24))
      (GeneratedTerminalCensus.trace_pairs_nonempty25))
      (GeneratedTerminalCensus.trace_pairs_nonempty26))
      (GeneratedTerminalCensus.trace_pairs_nonempty27))
      (GeneratedTerminalCensus.trace_pairs_nonempty28))
      (GeneratedTerminalCensus.trace_pairs_nonempty29))
      (GeneratedTerminalCensus.trace_pairs_nonempty30))

theorem trace_pair_widths : ∀ trace ∈ traces, ∀ pair ∈ trace.pairs,
pair.rhoColumns.length = 54 ∧ pair.inputColumns.length = 54 := by
unfold traces recursiveTraces terminalTraces
exact forall_append (forall_append
      (forall_append
      (forall_append
      (forall_append
      (forall_append
      (forall_append
      (forall_append
      (GeneratedRecursiveCensus.trace_pair_widths0)
      (GeneratedRecursiveCensus.trace_pair_widths1))
      (GeneratedRecursiveCensus.trace_pair_widths2))
      (GeneratedRecursiveCensus.trace_pair_widths3))
      (GeneratedRecursiveCensus.trace_pair_widths4))
      (GeneratedRecursiveCensus.trace_pair_widths5))
      (GeneratedRecursiveCensus.trace_pair_widths6))
      (GeneratedRecursiveCensus.trace_pair_widths7)) (forall_append
      (forall_append
      (forall_append
      (forall_append
      (forall_append
      (forall_append
      (forall_append
      (forall_append
      (forall_append
      (forall_append
      (forall_append
      (forall_append
      (forall_append
      (forall_append
      (forall_append
      (forall_append
      (forall_append
      (forall_append
      (forall_append
      (forall_append
      (forall_append
      (forall_append
      (forall_append
      (forall_append
      (forall_append
      (forall_append
      (forall_append
      (forall_append
      (forall_append
      (forall_append
      (GeneratedTerminalCensus.trace_pair_widths0)
      (GeneratedTerminalCensus.trace_pair_widths1))
      (GeneratedTerminalCensus.trace_pair_widths2))
      (GeneratedTerminalCensus.trace_pair_widths3))
      (GeneratedTerminalCensus.trace_pair_widths4))
      (GeneratedTerminalCensus.trace_pair_widths5))
      (GeneratedTerminalCensus.trace_pair_widths6))
      (GeneratedTerminalCensus.trace_pair_widths7))
      (GeneratedTerminalCensus.trace_pair_widths8))
      (GeneratedTerminalCensus.trace_pair_widths9))
      (GeneratedTerminalCensus.trace_pair_widths10))
      (GeneratedTerminalCensus.trace_pair_widths11))
      (GeneratedTerminalCensus.trace_pair_widths12))
      (GeneratedTerminalCensus.trace_pair_widths13))
      (GeneratedTerminalCensus.trace_pair_widths14))
      (GeneratedTerminalCensus.trace_pair_widths15))
      (GeneratedTerminalCensus.trace_pair_widths16))
      (GeneratedTerminalCensus.trace_pair_widths17))
      (GeneratedTerminalCensus.trace_pair_widths18))
      (GeneratedTerminalCensus.trace_pair_widths19))
      (GeneratedTerminalCensus.trace_pair_widths20))
      (GeneratedTerminalCensus.trace_pair_widths21))
      (GeneratedTerminalCensus.trace_pair_widths22))
      (GeneratedTerminalCensus.trace_pair_widths23))
      (GeneratedTerminalCensus.trace_pair_widths24))
      (GeneratedTerminalCensus.trace_pair_widths25))
      (GeneratedTerminalCensus.trace_pair_widths26))
      (GeneratedTerminalCensus.trace_pair_widths27))
      (GeneratedTerminalCensus.trace_pair_widths28))
      (GeneratedTerminalCensus.trace_pair_widths29))
      (GeneratedTerminalCensus.trace_pair_widths30))

theorem definitions_canonical : ∀ trace ∈ traces,
∀ definition ∈ trace.definitions, definition.Canonical := by
unfold traces recursiveTraces terminalTraces
exact forall_append (forall_append
      (forall_append
      (forall_append
      (forall_append
      (forall_append
      (forall_append
      (forall_append
      (GeneratedRecursiveCensus.definitions_canonical0)
      (GeneratedRecursiveCensus.definitions_canonical1))
      (GeneratedRecursiveCensus.definitions_canonical2))
      (GeneratedRecursiveCensus.definitions_canonical3))
      (GeneratedRecursiveCensus.definitions_canonical4))
      (GeneratedRecursiveCensus.definitions_canonical5))
      (GeneratedRecursiveCensus.definitions_canonical6))
      (GeneratedRecursiveCensus.definitions_canonical7)) (forall_append
      (forall_append
      (forall_append
      (forall_append
      (forall_append
      (forall_append
      (forall_append
      (forall_append
      (forall_append
      (forall_append
      (forall_append
      (forall_append
      (forall_append
      (forall_append
      (forall_append
      (forall_append
      (forall_append
      (forall_append
      (forall_append
      (forall_append
      (forall_append
      (forall_append
      (forall_append
      (forall_append
      (forall_append
      (forall_append
      (forall_append
      (forall_append
      (forall_append
      (forall_append
      (GeneratedTerminalCensus.definitions_canonical0)
      (GeneratedTerminalCensus.definitions_canonical1))
      (GeneratedTerminalCensus.definitions_canonical2))
      (GeneratedTerminalCensus.definitions_canonical3))
      (GeneratedTerminalCensus.definitions_canonical4))
      (GeneratedTerminalCensus.definitions_canonical5))
      (GeneratedTerminalCensus.definitions_canonical6))
      (GeneratedTerminalCensus.definitions_canonical7))
      (GeneratedTerminalCensus.definitions_canonical8))
      (GeneratedTerminalCensus.definitions_canonical9))
      (GeneratedTerminalCensus.definitions_canonical10))
      (GeneratedTerminalCensus.definitions_canonical11))
      (GeneratedTerminalCensus.definitions_canonical12))
      (GeneratedTerminalCensus.definitions_canonical13))
      (GeneratedTerminalCensus.definitions_canonical14))
      (GeneratedTerminalCensus.definitions_canonical15))
      (GeneratedTerminalCensus.definitions_canonical16))
      (GeneratedTerminalCensus.definitions_canonical17))
      (GeneratedTerminalCensus.definitions_canonical18))
      (GeneratedTerminalCensus.definitions_canonical19))
      (GeneratedTerminalCensus.definitions_canonical20))
      (GeneratedTerminalCensus.definitions_canonical21))
      (GeneratedTerminalCensus.definitions_canonical22))
      (GeneratedTerminalCensus.definitions_canonical23))
      (GeneratedTerminalCensus.definitions_canonical24))
      (GeneratedTerminalCensus.definitions_canonical25))
      (GeneratedTerminalCensus.definitions_canonical26))
      (GeneratedTerminalCensus.definitions_canonical27))
      (GeneratedTerminalCensus.definitions_canonical28))
      (GeneratedTerminalCensus.definitions_canonical29))
      (GeneratedTerminalCensus.definitions_canonical30))

end Nightstream.Implementation.R1CS.FPrimeFullHistoryProjection

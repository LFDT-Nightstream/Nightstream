import NightstreamFPrime.Lifecycle.RandomOracleBinding
import NightstreamFPrime.Spec.RandomOracleBudget

/-!
Owns the binding reduction of Lemma 6 as an algorithm (SECURITY_MODEL.md): a
program whose points are oracles and whose answers are the records of the
adversary's runs on them (`Spec.RandomOracleBudget.RunMachine`). The
algorithm holds no adversary.

Inputs: the claim parse function, the Ajtai key, a budget `T` and a tape of
`T` uniform oracles. The adversary enters only through the record of each run
the algorithm makes: its output and its queries. The algorithm applies the
claim parse function to each record.

Outputs:
- `reduction`: one adversary run per step, on an oracle built from one fresh
  tape entry: the base run, a retry on the base oracle with one fresh `Π_RLC`
  answer, or a rerun on the fork context with fresh answers elsewhere. Every
  test reads the oracle, the record of the run just made, and the records it
  stored;
- `solve`: the reduction with budget `T` on the adversary's records; it
  returns `RandomOracleBinding.rerunKernelOf` of the two runs it finishes
  with;
- `runs_length_le`: at most `T` runs on every tape, because the program has
  query bound `T` (`RunMachine.program_bound`);
- `runs_take_congr`: the oracle of each run depends on the adversary only
  through the records of the earlier runs;
- `solve_congr`: the output depends on the adversary only through the
  records of the runs made.

Does not own: the reduction's laws and work (`RandomOracleSolver`), or the
cost of the checks that each step makes: Lean decides them classically.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Lifecycle.RandomOracleReduction

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.RandomOracle
open NightstreamFPrime.Spec.RandomOracleBudget
open NightstreamFPrime.Spec.Folding
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Lifecycle.TranscriptCoverage
open NightstreamFPrime.Lifecycle.RandomOracleTest
open NightstreamFPrime.Lifecycle.RandomOracleExtraction
open NightstreamFPrime.Lifecycle.RandomOracleUniqueness
open NightstreamFPrime.Lifecycle.RandomOracleBinding
open _root_.NightstreamFPrime.Spec.Phi81Relation.PiRLCAlgebra
open StrongReduction ConcreteCarrier

attribute [local instance low] Classical.propDecidable

structure Run (Oracle Record : Type) where
  oracle : Oracle
  record : Record

structure Forks (Coordinate Answer Record : Type) where
  retries : Coordinate → Answer
  records : Coordinate → Record

inductive Purpose (Run Forks : Type)
  | first
  | second (base : Run) (baseForks : Forks)

inductive State (Run Forks Coordinate : Type)
  | start
  | extract (purpose : Purpose Run Forks) (run : Run) (index : Coordinate) (rest : List Coordinate)
      (forks : Forks)
  | rerun (base : Run) (forks : Forks)
  | finished (base : Run) (forks : Forks) (other : Run) (otherForks : Forks)
  | failed

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth logicalWidth}
  (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
  (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
  {Output : Type}
  (claim : Output → Claim relation)

local notation "Oracle" => Point logicalWidth publicFits (ProductionKey.degreeBound relation) → Answer
local notation "Record" => RunRecord (Point logicalWidth publicFits (ProductionKey.degreeBound relation)) Output
local notation "Coordinate" => Fin Nifs.PaperProfile.arity.total
local notation "Made" => Run Oracle Record
local notation "Retried" => Forks Coordinate Answer Record
local notation "Progress" => State Made Retried Coordinate

noncomputable def forkClaims (run : Made) (forks : Retried) : ForkClaims relation :=
  ⟨run.oracle, claim run.record.output, forks.retries, fun index => claim (forks.records index).output⟩

noncomputable def rhoPointAt (run : Made) (index : Coordinate) :
    Point logicalWidth publicFits (ProductionKey.degreeBound relation) :=
  rhoPointOf relation (claim run.record.output) index

noncomputable def afterExtraction : Purpose Made Retried → Made → Retried → Progress
  | .first, run, forks =>
      if ValidOf relation ajtai (forkClaims relation claim run forks) then .rerun run forks else .failed
  | .second base baseForks, run, forks =>
      if ValidOf relation ajtai (forkClaims relation claim run forks) then .finished base baseForks run forks
      else .rerun base baseForks

noncomputable def extracted (purpose : Purpose Made Retried) (run : Made) :
    List Coordinate → Retried → Progress
  | [], forks => afterExtraction relation ajtai claim purpose run forks
  | index :: rest, forks => .extract purpose run index rest forks

noncomputable def startExtraction (purpose : Purpose Made Retried) (run : Made) : Progress :=
  extracted relation ajtai claim purpose run (List.finRange _)
    ⟨fun index => run.oracle (rhoPointAt relation claim run index), fun _ => run.record⟩

noncomputable def query : Progress → Oracle → Oracle
  | .extract _ run index _ _, fresh =>
      Function.update run.oracle (rhoPointAt relation claim run index) (fresh (rhoPointAt relation claim run index))
  | .rerun base _, fresh =>
      overlay (contextOf relation
          (forkIndexOf relation (claim base.record.output) base.record.queries)
          (claim base.record.output) base.record.queries)
        base.oracle fresh
  | _, fresh => fresh

noncomputable def next : Progress → Oracle → Record → Progress
  | .start, fresh, record =>
      if SucceedsOf relation ajtai fresh (claim record.output) then
        startExtraction relation ajtai claim .first ⟨fresh, record⟩
      else .failed
  | .extract purpose run index rest forks, fresh, record =>
      if SucceedsOf relation ajtai (query relation claim (.extract purpose run index rest forks) fresh)
            (claim record.output) ∧
          rhoPointOf relation (claim record.output) index = rhoPointAt relation claim run index
      then extracted relation ajtai claim purpose run rest
        ⟨Function.update forks.retries index (fresh (rhoPointAt relation claim run index)),
          Function.update forks.records index record⟩
      else .extract purpose run index rest forks
  | .rerun base forks, fresh, record =>
      if forkIndexOf relation (claim record.output) record.queries =
          forkIndexOf relation (claim base.record.output) base.record.queries ∧
        SucceedsOf relation ajtai (query relation claim (.rerun base forks) fresh) (claim record.output)
      then startExtraction relation ajtai claim (.second base forks)
        ⟨query relation claim (.rerun base forks) fresh, record⟩
      else .rerun base forks
  | .finished base forks other otherForks, _, _ => .finished base forks other otherForks
  | .failed, _, _ => .failed

noncomputable def reduction : RunMachine Progress Oracle Oracle Record where
  query := query relation claim
  next := next relation ajtai claim
  halted
    | .finished .. => true
    | .failed => true
    | _ => false

noncomputable def output : Progress → Option (Binding.ShortKernelVector ajtai productionGlobalParams.msisNormBound)
  | .finished base forks other otherForks =>
      rerunKernelOf relation ajtai (forkClaims relation claim base forks) (forkClaims relation claim other otherForks)
  | _ => none

variable (adversary : OracleComp (Point logicalWidth publicFits (ProductionKey.degreeBound relation))
  Answer Output)

noncomputable def finalState (budget : ℕ) (tape : Fin budget → Oracle) : Progress :=
  ((reduction relation ajtai claim).program budget .start tape).run adversary.record

noncomputable def runs (budget : ℕ) (tape : Fin budget → Oracle) : List Oracle :=
  ((reduction relation ajtai claim).program budget .start tape).queries adversary.record

noncomputable def solve (budget : ℕ) (tape : Fin budget → Oracle) :
    Option (Binding.ShortKernelVector ajtai productionGlobalParams.msisNormBound) :=
  output relation ajtai claim (finalState relation ajtai claim adversary budget tape)

theorem runs_length_le (budget : ℕ) (tape : Fin budget → Oracle) :
    (runs relation ajtai claim adversary budget tape).length ≤ budget :=
  ((reduction relation ajtai claim).program_bound budget .start tape).queries_length_le _

theorem runs_take_congr
    (other : OracleComp (Point logicalWidth publicFits (ProductionKey.degreeBound relation)) Answer Output)
    (budget : ℕ) (tape : Fin budget → Oracle) (step : ℕ)
    (agree : ∀ oracle ∈ (runs relation ajtai claim adversary budget tape).take step,
      adversary.record oracle = other.record oracle) :
    (runs relation ajtai claim other budget tape).take (step + 1) =
      (runs relation ajtai claim adversary budget tape).take (step + 1) :=
  OracleComp.take_succ_queries_congr _ _ _ step agree

theorem solve_congr
    (other : OracleComp (Point logicalWidth publicFits (ProductionKey.degreeBound relation)) Answer Output)
    (budget : ℕ) (tape : Fin budget → Oracle)
    (agree : ∀ oracle ∈ runs relation ajtai claim adversary budget tape,
      adversary.record oracle = other.record oracle) :
    solve relation ajtai claim other budget tape = solve relation ajtai claim adversary budget tape := by
  unfold solve finalState
  rw [OracleComp.run_congr _ _ _ agree]

end NightstreamFPrime.Lifecycle.RandomOracleReduction

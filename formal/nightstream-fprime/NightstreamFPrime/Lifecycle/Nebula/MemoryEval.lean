import NightstreamFPrime.Lifecycle.Nebula.MemoryCircuit

/-! Owns the evaluation facts of the memory-application circuit: a wire reads
its witness word, a substituted row reads the decoded witness, and each
child's blocks evaluate to the transcript blocks of the semantic rows. It does
not own soundness or completeness. -/

namespace NightstreamFPrime.Lifecycle.Nebula.MemoryApp

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Nebula
open NightstreamFPrime.Circuit
open NightstreamFPrime.Gadgets.Poseidon2
open NightstreamFPrime.Lifecycle.Stage1

variable (p : Plan) (i : AppInterface p) (offset : ℕ) (env : Env)

/-- The witness words that the circuit reads at `env`. -/
abbrev values : ℕ → F := wordOf (Application.witnessValue i offset env)

local notation "W" => StepWitness.ofWords p (values p i offset env)

theorem wire_eval (k : ℕ) : (wire p i offset k).eval env = values p i offset env k := by
  unfold wire values wordOf Application.witnessValue
  split
  · next h => rw [List.getD_eq_getElem _ _ (by simpa using h), List.getElem_ofFn]
  · next h => rw [List.getD_eq_default _ _ (by simp; omega)]; rfl

theorem sub_eval (e : Circuit.Expr) : (sub p i offset e).eval env = e.eval (values p i offset env) := by
  rw [sub, Expr.eval_subst]
  congr 1
  funext k
  exact wire_eval p i offset env k

theorem wires_eval (f : ℕ → ℕ) (n : ℕ) :
    Hash.evalList env (wires p i offset f n) = List.ofFn fun k : Fin n => values p i offset env (f k) := by
  simp [wires, Hash.evalList, List.map_ofFn, Function.comp_def, wire_eval]

theorem textE_eval (text : String) : Hash.evalList env (textE text) = textWords text := by
  simp [textE, Hash.evalList, Function.comp_def]

theorem stateBlocks_eval (app carry : ℕ → ℕ) :
    (stateBlocks p i offset app carry).map (Hash.evalList env) =
      [textWords "Nightstream/Nebula/v3/state",
        List.ofFn fun k : Fin 2 => values p i offset env (app k),
        List.ofFn fun k : Fin 39 => values p i offset env (carry k)] := by
  simp only [stateBlocks, List.map_cons, List.map_nil, textE_eval, wires_eval]

theorem opsLanes_eval :
    ((opsLanes p).map (sub p i offset)).map (Circuit.Expr.eval env) = (W).opsLaneBits := by
  rw [List.map_map, show Circuit.Expr.eval env ∘ sub p i offset =
      Circuit.Expr.eval (values p i offset env) from funext fun e => sub_eval p i offset env e,
    opsLanes, List.map_flatten, List.map_ofFn, StepWitness.opsLaneBits]
  congr 1
  apply List.ofFn_inj.mpr
  funext j
  exact Sym.eval_opLane p _ j

theorem scanLanes_eval (slots : Fin p.bScan → ScanSlotBits p) (start : ℕ → ℕ)
    (each : ∀ j : Fin p.bScan,
      (Sym.scanLane p (start j.val)).map (Circuit.Expr.eval (values p i offset env)) = (slots j).lane) :
    ((scanLanes p start).map (sub p i offset)).map (Circuit.Expr.eval env) =
      StepWitness.scanLaneBits slots := by
  rw [List.map_map, show Circuit.Expr.eval env ∘ sub p i offset =
      Circuit.Expr.eval (values p i offset env) from funext fun e => sub_eval p i offset env e,
    scanLanes, List.map_flatten, List.map_ofFn, StepWitness.scanLaneBits]
  congr 1
  apply List.ofFn_inj.mpr
  funext j
  exact each j

theorem previous_eval (previous : ℕ) (small : previous + 3 < 12) :
    Hash.evalList env (wires p i offset (fun k => Words.seenPrev (previous + k)) 4) =
      digestWords ((W).previousDigest previous) := by
  rw [wires_eval, digestWords]
  congr 1
  funext k
  have : previous + k.val < 12 := by omega
  simp [StepWitness.previousDigest, this, StepWitness.ofWords]

theorem chainBlocks_eval (lane : Lane) (previous : ℕ) (small : previous + 3 < 12)
    (lanes : List Circuit.Expr) :
    (chainBlocks p i offset lane previous lanes).map (Hash.evalList env) =
      [textWords (chainTag lane), (W).idxEff :: digestWords ((W).previousDigest previous),
        packWords ((lanes.map (sub p i offset)).map (Circuit.Expr.eval env))] := by
  simp only [chainBlocks, List.map_cons, List.map_nil, textE_eval]
  rw [← previous_eval p i offset env previous small]
  simp only [Hash.evalList, List.map_cons, wire_eval, Expr.eval_pack]
  rfl

theorem evalList_append (l₁ l₂ : List Circuit.Expr) :
    Hash.evalList env (l₁ ++ l₂) = Hash.evalList env l₁ ++ Hash.evalList env l₂ :=
  List.map_append

theorem etaBlocks_eval :
    (etaBlocks p i offset).map (Hash.evalList env) =
      [textWords "Nightstream/Nebula/v3/eta", digestWords (planDigest p),
        [natWord ((W).cIn 2).val],
        digestWords ((W).proposalDigest 0) ++ digestWords (StepWitness.carryDigest (W).carryIn 35) ++
          digestWords ((W).proposalDigest 4)] := by
  have opsRoot : List.ofFn (fun k : Fin 4 => values p i offset env (Words.proposal k)) =
      digestWords ((W).proposalDigest 0) := by
    rw [digestWords]
    congr 1
    funext k
    have small : k.val < 8 := by omega
    simp [StepWitness.proposalDigest, StepWitness.ofWords, small]
  have memRoot : List.ofFn (fun k : Fin 4 => values p i offset env (Words.carryIn (35 + k))) =
      digestWords (StepWitness.carryDigest (W).carryIn 35) := by
    rw [digestWords]
    congr 1
    funext k
    simp [StepWitness.carryDigest, StepWitness.carryWord, StepWitness.ofWords,
      show 35 + k.val < 39 by omega]
  have finalRoot : List.ofFn (fun k : Fin 4 => values p i offset env (Words.proposal (4 + k))) =
      digestWords ((W).proposalDigest 4) := by
    rw [digestWords]
    congr 1
    funext k
    simp [StepWitness.proposalDigest, StepWitness.ofWords, show 4 + k.val < 8 by omega]
  simp only [etaBlocks, List.map_cons, List.map_nil, textE_eval, evalList_append, wires_eval,
    opsRoot, memRoot, finalRoot]
  refine congrArg₂ List.cons rfl (congrArg₂ List.cons ?_ (congrArg₂ List.cons ?_ rfl))
  · simp [Hash.evalList, digestWords]
  · simp [Hash.evalList, sub_eval, Sym.eval_cIn p, natWord_of_val]

end NightstreamFPrime.Lifecycle.Nebula.MemoryApp

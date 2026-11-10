import Lean
import CompressionLeanExe.Runs

namespace CompressionLeanTests.Inventory

open Lean

def standardAxioms : List Name := [``propext, ``Classical.choice, ``Quot.sound]

def claims : List (Name × List Name) := [
  (`CompressionLean.Claims.key_facts_of_check, standardAxioms),
  (`CompressionLean.Claims.bounded_depth_bound, standardAxioms),
  (`CompressionLean.Claims.bounded_depth_parent_bound, standardAxioms),
  (`CompressionLean.Claims.bounded_depth_last_fold_bound, standardAxioms),
  (`CompressionLean.Claims.bounded_depth_ivc_bound, standardAxioms),
  (`CompressionLean.Claims.bounded_depth_shape_of_check, standardAxioms),
  (`CompressionLean.Claims.bounded_depth_setup_facts_of_check, standardAxioms),
  (`CompressionLean.Claims.bounded_depth_logical_of_check, standardAxioms),
  (`CompressionLean.Claims.bounded_depth_fits_of_check, standardAxioms),
  (`CompressionLean.Claims.bounded_depth_runs_of_check, standardAxioms),
  (`CompressionLean.Claims.bounded_depth_honest_of_check, standardAxioms),
  (`CompressionLean.Claims.bounded_depth_covers_of_check, [``propext, ``Quot.sound]),
  (`CompressionLean.Claims.bounded_depth_poseidon2_bound, standardAxioms),
  (`CompressionLean.Claims.bounded_depth_complete, standardAxioms),
  (`CompressionLean.Claims.bounded_depth_honest_unbroken, standardAxioms)]

def claimsModule : Name := `CompressionLean.Claims

/-- The roots of this package's executables and gates. The library imports neither. -/
def upperRoots : List Name := [`CompressionLeanExe, `CompressionLeanTests]

/-- The modules of the `.lean` files under `root`. -/
def modulesUnder (root : System.FilePath) : IO (Array Name) := do
  let files := (← System.FilePath.walkDir root).filter (·.extension == some "lean")
  return files.map fun file => ((file.withExtension "").toString.replace "/" ".").toName

inductive Met where
  | rustFiles
  | leanValue
  | rust
  | proved
  | unmet
  | scope

def Met.label : Met → String
  | .rustFiles => "Rust's files"
  | .leanValue => "Lean value"
  | .rust => "Rust"
  | .proved => "Lean proof"
  | .unmet => "Open"
  | .scope => "Scope"

def Met.meaning : Met → String
  | .rustFiles => "an executable checks a fact about the files that Rust writes, and a claim turns the pass into the premise."
  | .leanValue => "compiled evaluation checks a closed proposition about Lean values. The kernel does not evaluate it: the packages are too large."
  | .rust => "Rust checks it when it runs. Lean does not check it."
  | .proved => "a Lean proof gives it where another claim uses this claim."
  | .unmet => "no check meets it yet; the claim holds only where it holds."
  | .scope => "it names the runs that a completeness claim covers. No check meets it: the claim says nothing about other runs."

structure Premise where
  binder : Name
  met : Met
  how : String

def premises : List (Name × List Premise) :=
  let h0 : Premise := ⟨`h0, .rust,
    "`L0` is the candidate count of P0's WHIR configuration (`log2_max_candidates`, `pcs.rs`). Plonky3 charges each term over `CandidateSet::from_log2` (`charge_reduction`, p3-sumcheck 0.8.0), which names no set for a count below one (p3-security 0.8.0, `report.rs`). The term then has zero bits, so `Pcs::new` refuses the configuration. `Relation::new` and `Shrink::new` keep only configurations that are charged at least one term."⟩
  let h1 : Premise := ⟨`h1, .rust, "As `h0`, for P1."⟩
  let p0 : Premise := ⟨`p0, .rust,
    "`Relation::new` builds P0's WHIR configuration with `Pcs::new` at `t + 1` bits. `Pcs::new` refuses it unless its WHIR terms plus these charged terms, over its candidates, are at most `2^-(t+1)` (`pcs.rs`). The candidate counts and the WHIR terms come from Plonky3's Johnson-bound report. No Lean proof checks them (`TRUST.md` §3)."⟩
  let p1 : Premise := ⟨`p1, .rust, "As `p0`, for P1, which is charged the setup queries."⟩
  let rust : Premise := ⟨`rust, .rust,
    "`Shrink::new` gives these terms to `Pcs::new` at the layer bits `t`. `Pcs::new` refuses the configuration unless its WHIR terms plus these terms, over its candidates, are at most `2^-t` (`pcs.rs`). No Lean proof checks the WHIR terms or `Ls`."⟩
  let boundedDepthWidth := "`boundedDepthCheck` evaluates `boundedDepthFits` on both layer-1 keys that `bounded_depth_layer1_keys_are_written` writes and stops when it fails; `bounded_depth_final_program_matches_lean` requires the pass."
  let boundedDepthSetup := "`setupCheck` evaluates `setupFailureAt` over the key's own columns on the runs that `bounded_depth_runs_rebuild_the_layer1_keys` writes, which rebuild each slot's layer-1 key. `bounded_depth_setup_matches_lean` requires the pass."
  let boundedDepthCover := "`coverCheck` evaluates `coverFailure` on each slot's step package and the layer-1 key that `bounded_depth_layer1_matches_lean` writes. `scripts/crosschecks.sh` runs it after that test (`bounded_depth_covers`); no Rust test requires the pass."
  let boundedDepthRuns := "`runsCheck` evaluates `keyRunsFailure` on each slot's step package and the runs that `bounded_depth_runs_rebuild_the_layer1_keys` writes. `bounded_depth_runs_match_lean` requires the pass."
  let emittedPacked : Premise := ⟨`packed, .leanValue, "F′'s emitter evaluates `packable` on the circuit's pieces and writes the package as `packageOf` of them (`EmitMain.emitPackage`); the checks read that artifact."⟩
  let emittedRows : Premise := ⟨`rows, .leanValue, "The emitter writes the package as `packageOf` of the circuit's pieces, whose rows are the relation's rows (`StepPackage.relationRows_packageOf`, with `packed`)."⟩
  let emittedBlocks : Premise := ⟨`blocks, .leanValue, "As `rows` (`StepPackage.blocks_packageOf`)."⟩
  let emittedStart : Premise := ⟨`start, .leanValue, "As `rows`: `packageOf` writes `CircuitRelation.sboxStart`."⟩
  let boundedDepth : List Premise := [
    ⟨`facts, .rustFiles, "`lowerCheck` evaluates `keyFailure` on both layer-1 keys that `bounded_depth_layer1_keys_are_written` writes (`bounded_depth_layer1_matches_lean`); `key_facts_of_check` turns each pass into `KeyFacts`."⟩,
    ⟨`fits, .proved, "`bounded_depth_poseidon2_bound` gives it from the pass of `setupCheck` (`bounded_depth_setup_facts_of_check`). Open for other applications."⟩,
    ⟨`describes, .proved, "As `fits`."⟩,
    ⟨`fit, .leanValue, "`boundedDepthCheck` evaluates `fits` on the finish package that it decodes from the artifact and stops when it fails."⟩,
    ⟨`keysLength, .rustFiles, "`boundedDepthCheck` requires eight state key words in the log that `bounded_depth_final_program_matches_lean` writes; `FinalProgram::keys` is `[[F; 4]; 2]`."⟩,
    h0, h1, p0, p1, rust]
  let parent : List Premise := boundedDepth ++ [
      ⟨`pf, .proved, "F′ proves it for the key width (`StepRelation.keyPublicFits`, in `bounded_depth_poseidon2_bound`)."⟩,
      ⟨`packed, .proved, "`bounded_depth_poseidon2_bound` gives it from F′'s emitter at the key width. Open for other widths."⟩,
      ⟨`starts, .proved, "As `packed`."⟩,
      ⟨`shape, .proved, "`bounded_depth_poseidon2_bound` gives it from `KeyFacts` and the pass of `boundedDepthCheck` (`bounded_depth_shape_of_check`). Open for other applications."⟩,
      ⟨`honest, .proved, "`bounded_depth_poseidon2_bound` gives it from the passes of `setupCheck` and `runsCheck` (`bounded_depth_honest_of_check`, `bounded_depth_runs_of_check`). Open for other applications."⟩]
  let lastFold : List Premise := boundedDepth.filter (fun p => p.binder != `fits && p.binder != `describes) ++ [
      ⟨`fits, .rustFiles, "For the setups that Rust builds from each key's runs (`BoundedDepth.keySetup`), `bounded_depth_setup_facts_of_check` turns the pass of `setupCheck` into it."⟩,
      ⟨`describes, .rustFiles, "As `fits`."⟩,
      ⟨`pf, .proved, "F′ proves it for the key width (`StepRelation.keyPublicFits`)."⟩,
      ⟨`packed, .leanValue, "F′'s emitter evaluates `packable` on the finish relation's pieces (`EmitMain.emitPackage`) and writes no package when it fails."⟩,
      ⟨`starts, .leanValue, "As `packed`, for `startsFrom` at the shared end."⟩,
      ⟨`shape, .rustFiles, "`bounded_depth_shape_of_check` turns `KeyFacts` and the pass of `boundedDepthCheck` into it."⟩,
      ⟨`honest, .rustFiles, "`bounded_depth_honest_of_check` turns the passes of `setupCheck` and `runsCheck` into it (`bounded_depth_runs_of_check`)."⟩,
      ⟨`keyWords, .rust, "`BoundedDepth::new` derives `self.keys` from the packages' identity (`bounded_depth/mod.rs`). The step states hash these digests, and `BoundedDepth::final_program` gives the same digests to the final program (`compression.rs`). No Lean check compares them with the setup's circuit keys."⟩,
      ⟨`keysShaped, .rust, "Rust's key digests have four words: `FinalProgram::keys` is `[[F; 4]; 2]`."⟩]
  let complete : List Premise := [
      ⟨`pf, .proved, "F′ proves it for the key width (`StepRelation.keyPublicFits`)."⟩,
      ⟨`cubeFits, .proved, "F′ proves it at the key width (`StepRelation.keyCubeFits`)."⟩,
      ⟨`finish, .leanValue, "F′'s emitter evaluates the finish package's completeness checks (`EmitMain.finishChecks` and `StepCompleteness.publicFree` of the finish rows) at the key width with `StepRelation.keyRelation`, and writes no finish package when one fails (`EmitMain.emitPackage`)."⟩,
      ⟨`shape, .rustFiles, "`bounded_depth_shape_of_check` turns `KeyFacts` and the pass of `boundedDepthCheck` into it."⟩,
      ⟨`honest, .rustFiles, "`bounded_depth_honest_of_check` turns the passes of `setupCheck` and `runsCheck` into it (`bounded_depth_runs_of_check`)."⟩,
      ⟨`keyWords, .rust, "As `bounded_depth_last_fold_bound`'s `keyWords`."⟩,
      ⟨`keysShaped, .rust, "As `bounded_depth_last_fold_bound`'s `keysShaped`."⟩,
      ⟨`lightCover, .rustFiles, s!"{boundedDepthCover} The emitter writes the light package as `packageOf 0` of the light circuit's pieces (`EmitMain.emitPackage`), which is this package when `exports` and `complete` carry the emitter's first fresh column, width and witness batches. `bounded_depth_covers_of_check` turns the pass into the norm and block bounds of slot 0."⟩,
      ⟨`refreshCover, .rustFiles, "As `lightCover`, for the refresh package (`packageOf 1`) and slot 1."⟩,
      ⟨`z0Length, .scope, "The run starts from a state of `Stage1.Application.stateWordCount` words."⟩,
      ⟨`valid, .scope, "Each witness of the run is valid for the application at the state that the earlier witnesses reach (`IvcWalk.ValidFrom`)."⟩,
      ⟨`lengths, .scope, "Each witness of the run has the application's witness length."⟩,
      ⟨`nonempty, .scope, "The run has at least one step. A run of no steps has no final proof: the final program's terminal asks for a nonzero iteration (`Spec.BoundedDepth.FinishTerminal`)."⟩,
      ⟨`bound, .scope, "The run has fewer steps than the field's order, so that its iteration is one field word."⟩]
  [(`CompressionLean.Claims.key_facts_of_check, [⟨`pass, .rustFiles,
      "`lowerCheck` evaluates `keyFailure` on both layer-1 keys that `bounded_depth_layer1_keys_are_written` writes. `bounded_depth_layer1_matches_lean` requires the pass."⟩]),
    (`CompressionLean.Claims.bounded_depth_bound, boundedDepth),
    (`CompressionLean.Claims.bounded_depth_parent_bound, parent),
    (`CompressionLean.Claims.bounded_depth_last_fold_bound, lastFold),
    (`CompressionLean.Claims.bounded_depth_ivc_bound, lastFold ++ [
      ⟨`cubeFits, .proved, "F′ proves it at the key width (`StepRelation.keyCubeFits`)."⟩,
      ⟨`assumption, .unmet, "HyperNova errata Assumption 1 for the light and refresh folds with Poseidon2 inside the step circuits (`SECURITY_MODEL.md`, trust item 3). `romAssumption1` proves it only in a random-oracle model of the two folds."⟩,
      ⟨`closed, .unmet, "The closure premise of HyperNova Lemma 17 for the reverse extractor's stages (`SECURITY_MODEL.md`, trust item 3)."⟩,
      ⟨`finishAdmitted, .unmet, "The finish stage's adversary runs the final verifier's experiment and outputs the last fold that `Classical.epsilon` chooses. The claims prove that a last fold exists. No claim computes one from the final proof, so this premise asks for knowledge soundness of the final verifier. The claim's conclusion asks the same of the chain's IVC adversary (`admitted`)."⟩]),
    (`CompressionLean.Claims.bounded_depth_shape_of_check, [
      ⟨`facts, .rustFiles, "As `bounded_depth_bound`'s `facts`."⟩,
      ⟨`pass, .rustFiles, boundedDepthWidth⟩]),
    (`CompressionLean.Claims.bounded_depth_setup_facts_of_check, [⟨`pass, .rustFiles, boundedDepthSetup⟩]),
    (`CompressionLean.Claims.bounded_depth_logical_of_check, [⟨`pass, .rustFiles, boundedDepthRuns⟩]),
    (`CompressionLean.Claims.bounded_depth_fits_of_check, [⟨`pass, .rustFiles, boundedDepthRuns⟩, emittedRows, emittedBlocks,
      emittedStart]),
    (`CompressionLean.Claims.bounded_depth_runs_of_check, [⟨`pass, .rustFiles, boundedDepthRuns⟩,
      ⟨`fits, .proved, "`bounded_depth_fits_of_check` gives it from `pass`, `rows`, `blocks` and `start` (in `bounded_depth_poseidon2_bound`)."⟩,
      emittedPacked, emittedRows, emittedBlocks, emittedStart,
      ⟨`logical, .proved, "`bounded_depth_logical_of_check` gives it from `pass` (in `bounded_depth_poseidon2_bound`)."⟩]),
    (`CompressionLean.Claims.bounded_depth_honest_of_check, [⟨`pass, .rustFiles, boundedDepthSetup⟩,
      ⟨`gives, .proved, "`bounded_depth_runs_of_check` gives it (in `bounded_depth_poseidon2_bound`)."⟩]),
    (`CompressionLean.Claims.bounded_depth_covers_of_check, [⟨`pass, .rustFiles, boundedDepthCover⟩]),
    (`CompressionLean.Claims.bounded_depth_poseidon2_bound, [
      ⟨`facts, .rustFiles, "As `bounded_depth_bound`'s `facts`."⟩,
      ⟨`within, .rustFiles, boundedDepthWidth⟩,
      ⟨`setups, .rustFiles, boundedDepthSetup⟩,
      ⟨`runs, .rustFiles, boundedDepthRuns⟩,
      ⟨`fits, .rustFiles, s!"{boundedDepthRuns} `bounded_depth_fits_of_check` turns the pass, with `rows`, `blocks` and `starts`, into `Fits`."⟩,
      ⟨`logical, .rustFiles, s!"{boundedDepthRuns} `bounded_depth_logical_of_check` turns the pass into the bound."⟩,
      { emittedPacked with binder := `packed }, { emittedRows with binder := `rows },
      { emittedBlocks with binder := `blocks }, { emittedStart with binder := `starts },
      ⟨`finishPacked, .leanValue, "F′'s emitter evaluates `packable` on the finish relation's pieces at the key width with `StepRelation.keyRelation` (`EmitMain.emitPackage`) and writes no package when it fails."⟩,
      ⟨`finishStarts, .leanValue, "As `finishPacked`, for `startsFrom` at the shared end."⟩,
      ⟨`fit, .leanValue, "`boundedDepthCheck` evaluates `fits` on the finish package that it decodes from the artifact and stops when it fails."⟩,
      ⟨`keysLength, .rustFiles, "As `bounded_depth_bound`'s `keysLength`."⟩,
      h0, h1, p0, p1, rust]),
    (`CompressionLean.Claims.bounded_depth_complete, complete),
    (`CompressionLean.Claims.bounded_depth_honest_unbroken, complete)]

def portRoot : Name := `CompressionLean.Ports

def digestPins : List (String × String) := [
  ("crates/neo-spartan/tests/internal/circuit.rs", "permutation_template_matches_lean"),
  ("crates/neo-spartan/tests/internal/lowering.rs", "toy_layer1_digest_is_pinned"),
  ("crates/nightstream/tests/bounded_depth_lifecycle/final_ir.rs", "bounded_depth_prefix_digest_is_pinned"),
  ("crates/nightstream/tests/bounded_depth_lifecycle/final_ir.rs", "bounded_depth_program_digest_is_pinned")]

/-- The Rust test `lean_ports_match_their_rust_items` checks each hash. -/
structure Port where
  lean : Name
  rust : String
  hash : String
deriving FromJson

def readPorts : IO (Array Port) := do
  IO.ofExcept <| (Json.parse (← IO.FS.readFile "ports.json")).bind fromJson?

def rustFile (path : String) : String := (path.splitOn ".rs::").headD "" ++ ".rs"

def moduleOf (env : Environment) (n : Name) : Name :=
  match env.getModuleIdxFor? n with
  | some idx => env.header.moduleNames[idx.toNat]!
  | none => .anonymous

def cell (ty : Expr) : MetaM String := do
  let text := (← Meta.ppExpr ty).pretty 1000000
  let words := (String.ofList (text.toList.map fun c => if c.isWhitespace then ' ' else c)).splitOn " "
    |>.filter (· ≠ "")
  return (" ".intercalate words).replace "|" "\\|"

def binders (type : Expr) : MetaM (List (Name × String × Bool)) :=
  Meta.forallTelescope type fun xs _ => xs.toList.mapM fun x => do
    let decl ← x.fvarId!.getDecl
    let entry := s!"{decl.userName} : {← cell decl.type}"
    let shown := match decl.binderInfo with
      | .implicit => s!"\{{entry}}"
      | .strictImplicit => s!"⦃{entry}⦄"
      | .instImplicit => s!"[{entry}]"
      | .default => entry
    return (decl.userName, shown, ← Meta.isProp decl.type)

def counts (keys : List String) : List (String × Nat) :=
  (keys.mergeSort (· ≤ ·)).splitBy (· == ·) |>.map fun g => (g.headD "", g.length)

def code (s : String) : String := s!"`{s}`"

def statusPage (ports : Array Port) : CoreM String :=
    withReader (fun ctx => { ctx with currNamespace := .anonymous, openDecls := [] }) do
  let env ← getEnv
  let mut lines := #["# Status", "",
    "Generated by `lake exe status > STATUS.md` in `formal/compression-lean`; `lake build` fails when this file differs.",
    "", "## Claims", "", "Each claim of `CompressionLean/Claims.lean` and the axioms that it uses.",
    "", "| Claim | Axioms |", "|---|---|"]
  for (claim, _) in claims do
    let axioms := (← collectAxioms claim).toList.map (code ∘ toString)
    lines := lines.push s!"| {code (toString claim)} | {", ".intercalate axioms} |"
  lines := lines ++ #["", "## Hypotheses", "",
    "Every hypothesis of each claim, by binder. A parameter is data that the claim holds for. A premise is a proposition that the claim takes; the claim says nothing where it fails. Each premise row says how the premise is met:",
    ""]
  for met in [Met.rustFiles, .leanValue, .rust, .proved, .unmet, .scope] do
    lines := lines.push s!"- {met.label}: {met.meaning}"
  for (claim, _) in claims do
    let some info := env.find? claim | throwError "claim {claim} is not declared"
    let listed := (premises.lookup claim).getD []
    let all ← (binders info.type).run'
    for p in listed do
      unless all.any fun (n, _, prop) => n == p.binder && prop do
        throwError "Inventory.premises names {p.binder} for {claim}, which is not a premise of it"
    let params := all.filter (!·.2.2) |>.map (code ·.2.1)
    lines := lines ++ #["", s!"### {code (toString claim)}", "",
      s!"Parameters: {if params.isEmpty then "none" else ", ".intercalate params}.", "",
      "| Premise | Met by | How |", "|---|---|---|"]
    for (n, shown, prop) in all do
      unless prop do continue
      let some p := listed.find? (·.binder == n)
        | throwError "{claim} takes the premise `{shown}`, which Inventory.premises does not classify"
      lines := lines.push s!"| {code shown} | {p.met.label} | {p.how} |"
  let decls := ports.toList.map (·.lean) |>.eraseDups
  let items := ports.toList.map (·.rust) |>.eraseDups
  lines := lines ++ #["", "## Ports", "",
    s!"`ports.json` has {ports.size} entries: {decls.length} Lean declarations transcribe {items.length} Rust items.",
    "", "| Lean module | Declarations |", "|---|---:|"]
  for (m, n) in counts (decls.map fun d => toString (moduleOf env d)) do
    lines := lines.push s!"| {code m} | {n} |"
  lines := lines ++ #["", "| Rust file | Items |", "|---|---:|"]
  for (file, n) in counts (items.map rustFile) do
    lines := lines.push s!"| {code file} | {n} |"
  lines := lines ++ #["", "## Digest pins", "", "The Rust tests that pin a digest that Lean computes.", "",
    "| Test | File |", "|---|---|"]
  for (file, test) in digestPins do
    let source ← IO.FS.readFile ("../.." / file : System.FilePath)
    unless (source.splitOn s!"fn {test}(").length > 1 do
      throwError "{file} has no test {test}"
    lines := lines.push s!"| {code test} | {code file} |"
  lines := lines ++ #["", "## Toolchain", "", s!"Lean {Lean.versionString}."]
  return "\n".intercalate lines.toList ++ "\n"

end CompressionLeanTests.Inventory

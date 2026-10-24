import Lean

/-!
Owns what the library claims and what the claims rest on, as data: the pinned
claims, every premise of each claim with how it is met, the hand ports of Rust
(`ports.json`) and the Rust tests that pin digests that Lean computes. It also owns `STATUS.md`
(`statusPage`). It does not import `CompressionLean`, so `lake exe status`
loads the library at run time.

The gates are in `Axioms.lean` and `Ports.lean`. Paths are relative to
`formal/compression-lean`.
-/

namespace CompressionLeanTests.Inventory

open Lean

/-- The axioms of every claim. -/
def standardAxioms : List Name := [``propext, ``Classical.choice, ``Quot.sound]

/-- The claims: the theorems of `CompressionLean.Claims`, each with the axioms
that it uses exactly. -/
def claims : List (Name × List Name) := [
  (`CompressionLean.Claims.production_bound, standardAxioms),
  (`CompressionLean.Claims.poseidon2_bound, standardAxioms),
  (`CompressionLean.Claims.honest_setup_of_build, standardAxioms),
  (`CompressionLean.Claims.key_facts_of_check, standardAxioms),
  (`CompressionLean.Claims.width_fits_of_check, standardAxioms),
  (`CompressionLean.Claims.package_templates_of_check, [``propext, ``Quot.sound]),
  (`CompressionLean.Claims.setup_facts_of_check, standardAxioms)]

/-- The module of the claims. -/
def claimsModule : Name := `CompressionLean.Claims

/-- How a premise of a claim is met (`Met.meaning`). -/
inductive Met where
  | rustFiles
  | leanValue
  | rust
  | proved
  | unchecked

def Met.label : Met → String
  | .rustFiles => "Rust's files"
  | .leanValue => "Lean value"
  | .rust => "Rust"
  | .proved => "Lean proof"
  | .unchecked => "Open"

def Met.meaning : Met → String
  | .rustFiles => "an executable checks a fact about the files that Rust writes, and a claim turns the pass into the premise."
  | .leanValue => "compiled evaluation checks a closed proposition about Lean values. The kernel does not evaluate it: the terminal package is too large."
  | .rust => "Rust checks it when it runs. Lean does not check it."
  | .proved => "a Lean proof gives it where another claim uses this claim."
  | .unchecked => "no Lean proof and no check."

/-- A premise of a claim: its binder and how it is met. -/
structure Premise where
  binder : Name
  met : Met
  how : String

/-- Every premise of every claim: each Prop-valued binder of its type. The
status gate fails when a claim has a premise that is not here, or when an
entry names no premise of its claim. -/
def premises : List (Name × List Premise) :=
  let pf : Premise := ⟨`pf, .proved,
    "F′ proves it for an application (`Poseidon2HashChainV1Package.fits`, in `poseidon2_bound`)."⟩
  let facts : Premise := ⟨`facts, .rustFiles,
    "`lowerCheck` and `terminalCheck` check it on the production key (`key_facts_of_check`). The production tests `production_layer1_matches_lean` and `final_program_matches_lean` require the pass."⟩
  let setupPass := "`setupCheck` checks it on the production key and on the runs that `production_runs_rebuild_the_layer1_key` writes. That test requires `Key::derive` of the runs to be the committed layer-1 key, and `production_setup_matches_lean` requires the pass."
  let gives : Premise := ⟨`gives, .unchecked,
    "`Checks/PoseidonRows.lean` proves the matrix entries of the rows of the Poseidon2 blocks of F′'s matrix program (`matrixSource_entry`). No check compares the runs with these rows."⟩
  let h0 : Premise := ⟨`h0, .rust,
    "`L0` is the candidate count of P0's WHIR configuration (`log2_max_candidates`, `pcs.rs`). A list holds at least one candidate."⟩
  let h1 : Premise := ⟨`h1, .rust, "As `h0`, for P1."⟩
  let p0 : Premise := ⟨`p0, .rust,
    "`Relation::new` builds P0's WHIR configuration with `Pcs::new` at `t + 1` bits. `Pcs::new` refuses it unless its WHIR terms plus these charged terms, over its candidates, are at most `2^-(t+1)` (`pcs.rs`). The candidate counts and the WHIR terms come from Plonky3's Johnson-bound report. No Lean proof checks them (`TRUST.md` §3)."⟩
  let p1 : Premise := ⟨`p1, .rust, "As `p0`, for P1, which is charged the setup queries."⟩
  let rust : Premise := ⟨`rust, .rust,
    "`Shrink::new` gives these terms to `Pcs::new` at the layer bits `t`. `Pcs::new` refuses the configuration unless its WHIR terms plus these terms, over its candidates, are at most `2^-t` (`pcs.rs`). No Lean proof checks the WHIR terms or `Ls`."⟩
  let loads : Premise := ⟨`loads, .leanValue,
    "`terminalCheck` evaluates `load` on the terminal package and stops when the load fails. `loaded` is its result."⟩
  let valid : Premise := ⟨`valid, .leanValue,
    "`terminalCheck` evaluates `templatesNamed` on the terminal package (`package_templates_of_check`)."⟩
  let hctx : Premise := ⟨`hctx, .rust,
    "`PreparedLifecycle::check_compression_key` (`lifecycle/finish.rs`) refuses a compression key whose context words are not the digest of the circuit's verifier context. `authority` is that context."⟩
  [(`CompressionLean.Claims.production_bound, [pf,
      ⟨`shape, .proved, "`poseidon2_bound` gives it from `KeyFacts` and `WidthFits` (`parentShape_of`). Open for other applications."⟩,
      ⟨`honest, .proved, "`poseidon2_bound` gives it from Rust's setup build and `RunsGive` (`honest_setup_of_build`). Open for other applications."⟩,
      facts,
      ⟨`fits, .rustFiles, "From the pass of `setupCheck`, for the Poseidon2 hash chain (`setup_facts_of_check`, in `poseidon2_bound`). Open for other applications."⟩,
      ⟨`describes, .rustFiles, "As `fits`."⟩,
      h0, h1, p0, p1, rust, loads, valid, hctx]),
    (`CompressionLean.Claims.poseidon2_bound, [gives, facts,
      ⟨`width, .rustFiles, "`lowerCheck` and `terminalCheck` print whether the production key fits the Poseidon2 hash chain (`width_fits_of_check`). The production tests require the pass."⟩,
      ⟨`pass, .rustFiles, setupPass⟩,
      h0, h1, p0, p1, rust, loads, valid, hctx]),
    (`CompressionLean.Claims.honest_setup_of_build, [pf,
      ⟨`D, .proved, "`rfl` for `shapeOf` of a key (in `poseidon2_bound`)."⟩,
      ⟨`lanes, .proved, "As `D`."⟩,
      ⟨`built, .rustFiles, "From the pass of `setupCheck` (`setup_facts_of_check`, in `poseidon2_bound`)."⟩,
      gives]),
    (`CompressionLean.Claims.key_facts_of_check, [⟨`pass, .rustFiles,
      "`lowerCheck` and `terminalCheck` evaluate `keyFailure` on the key that Rust writes. The production tests require the pass."⟩]),
    (`CompressionLean.Claims.width_fits_of_check, [⟨`pass, .rustFiles,
      "`lowerCheck` and `terminalCheck` evaluate `poseidon2Fits` on the key that Rust writes. The production tests require the pass."⟩]),
    (`CompressionLean.Claims.package_templates_of_check, [⟨`named, .leanValue,
      "`terminalCheck` evaluates `templatesNamed` on the terminal package."⟩]),
    (`CompressionLean.Claims.setup_facts_of_check, [⟨`pass, .rustFiles, setupPass⟩])]

/-- The folder of the hand ports of Rust: `CompressionLean/Ports/` and every
module under it. -/
def portRoot : Name := `CompressionLean.Ports

/-- The Rust tests that pin a digest that Lean computes: the file, then the test. -/
def digestPins : List (String × String) := [
  ("crates/neo-spartan/tests/internal/circuit.rs", "permutation_template_matches_lean"),
  ("crates/neo-spartan/tests/internal/lowering.rs", "toy_layer1_digest_is_pinned"),
  ("crates/neo-spartan/tests/internal/lowering.rs", "production_layer1_digest_is_pinned"),
  ("crates/nightstream/tests/lifecycle_native/terminal_ir.rs", "terminal_digest_is_pinned"),
  ("crates/nightstream/tests/lifecycle_native/terminal_ir.rs", "final_program_digest_is_pinned")]

/-- One hand transcription: a Lean declaration, the Rust item that it
transcribes and the hash of the item's tokens, which the Rust test
`lean_ports_match_their_rust_items` checks. -/
structure Port where
  lean : Name
  rust : String
  hash : String
deriving FromJson

def readPorts : IO (Array Port) := do
  IO.ofExcept <| (Json.parse (← IO.FS.readFile "ports.json")).bind fromJson?

/-- The file of a Rust item path. -/
def rustFile (path : String) : String := (path.splitOn ".rs::").headD "" ++ ".rs"

/-- The module that declares `n`. -/
def moduleOf (env : Environment) (n : Name) : Name :=
  match env.getModuleIdxFor? n with
  | some idx => env.header.moduleNames[idx.toNat]!
  | none => .anonymous

/-- `ty` on one line, with the table separator escaped. -/
def cell (ty : Expr) : MetaM String := do
  let text := (← Meta.ppExpr ty).pretty 1000000
  let words := (String.ofList (text.toList.map fun c => if c.isWhitespace then ' ' else c)).splitOn " "
    |>.filter (· ≠ "")
  return (" ".intercalate words).replace "|" "\\|"

/-- The binders of a claim, each as `name : type` with its binder brackets,
and whether it is a premise (Prop-valued). -/
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

/-- How many times each key occurs, in key order. -/
def counts (keys : List String) : List (String × Nat) :=
  (keys.mergeSort (· ≤ ·)).splitBy (· == ·) |>.map fun g => (g.headD "", g.length)

def code (s : String) : String := s!"`{s}`"

/-- `STATUS.md`. It fails when a premise of a claim has no entry in
`premises`, or an entry names no premise. It prints names in full, wherever
it runs. -/
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
  for met in [Met.rustFiles, .leanValue, .rust, .proved, .unchecked] do
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

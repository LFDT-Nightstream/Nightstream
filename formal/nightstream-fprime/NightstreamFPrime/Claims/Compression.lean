import NightstreamFPrime.Spec.Compression.Fresh
import NightstreamFPrime.Spec.Compression.Production
import NightstreamFPrime.Spec.Compression.Rust.Soundness
import NightstreamFPrime.Spec.Compression.Verifier.Bound

/-! # Claims: the compression argument

Soundness of the argument of the compressed proof (`SPEC.md` Algorithms 9 and 10,
`SECURITY_MODEL.md` adversary 6).

- Model: a public-coin game in the state-restoration experiment, with ideal Akita openings.
- Assumptions: none in this file. `Claims.Final` adds `[AKITA]`.
- Open: no completeness claim. Lean has no definition of the compression key. -/

namespace NightstreamFPrime.Claims.Compression

open Spec Spec.Compression
open scoped Spec.FieldExtensionRing Spec.Compression.Production

/-- Soundness in the state-restoration experiment with `q` oracle queries. -/
theorem argument_sound
    {F : Type}
    {K : Type}
    [fieldF : Field F]
    [fintypeF : Fintype F]
    [decidableF : DecidableEq F]
    [fieldK : Field K]
    [fintypeK : Fintype K]
    [decidableK : DecidableEq K]
    [algebra : Algebra F K]
    {P : Verifier.Params F K}
    (vf : Verifier.VerifierData P)
    (wf : Verifier.WellFormed P)
    (counts : Verifier.CountsNonzero P)
    (matrix : Verifier.MatrixSpec vf)
    (columns : P.lookups ≤ 2 ^ vf.columnVars)
    (rows : P.rows ≤ 2 ^ P.rowVars)
    (key_fits : ∀ (c : Fin P.claims), (P.shape c).witness.vars - P.laneVars ≤ P.keyBlockVars)
    (s : Game.Strategy (Option (Option (Verifier.Statement P × (ℕ → F)))))
    (q : ℕ) :
    (Game.prob s ((Verifier.game vf).restore q [])
          fun (r : Option (Option (Verifier.Statement P × (ℕ → F)))) =>
          ¬∀ (a : Option (Verifier.Statement P × (ℕ → F))),
              r = some a →
                ∀ (o : Verifier.Statement P × (ℕ → F)),
                  a = some o → Verifier.Relation P o.1 o.2) ≤
        (↑q : ℝ) * Verifier.error vf :=
  @Spec.Compression.Verifier.soundness F K fieldF fintypeF decidableF fieldK fintypeK decidableK algebra P vf wf counts
    matrix columns rows key_fits s q

/-- `argument_sound` at the production parameters with the one pass of Rust `matrix_part`. -/
theorem production_sound
    {n : ℕ}
    (key : ℕ → ℕ → ℕ → F)
    (rel : ℕ → Verifier.Factored F)
    (which : Fin (n + 1) → ℕ)
    (depths : Fin n → ℕ)
    (layout : Production.Layout (n + 1))
    (zeros : ℕ)
    (wf : Verifier.WellFormed
        (Production.params key (fun (c : Fin (n + 1)) => (rel (which c)).matrices) depths layout))
    (slots_le : ∀ (c : Fin (n + 1)), (↑c : ℕ) < n → (layout.witness c).vars ≤ 25)
    (key_fits : ∀ (c : Fin (n + 1)), (layout.witness c).vars - 6 ≤ layout.keyBlockVars)
    (s : Game.Strategy
        (Option
          (Option
            (Verifier.Statement
                (Production.params key (fun (c : Fin (n + 1)) => (rel (which c)).matrices) depths
                  layout) ×
              (ℕ → F)))))
    (q : ℕ) :
    (Game.prob s
          ((Verifier.game (Production.verifierData key rel which depths layout zeros)).restore q
            [])
          fun
            (r :
              Option
                (Option
                  (Verifier.Statement
                      (Production.params key (fun (c : Fin (n + 1)) => (rel (which c)).matrices) depths
                        layout) ×
                    (ℕ → F)))) =>
          ¬∀
              (a :
                Option
                  (Verifier.Statement
                      (Production.params key (fun (c : Fin (n + 1)) => (rel (which c)).matrices) depths
                        layout) ×
                    (ℕ → F))),
              r = some a →
                ∀
                  (o :
                    Verifier.Statement
                        (Production.params key (fun (c : Fin (n + 1)) => (rel (which c)).matrices)
                          depths layout) ×
                      (ℕ → F)),
                  a = some o →
                    Verifier.Relation
                      (Production.params key (fun (c : Fin (n + 1)) => (rel (which c)).matrices) depths
                        layout)
                      o.1 o.2) ≤
        (↑q : ℝ) * Verifier.error (Production.verifierData key rel which depths layout zeros) :=
  @Spec.Compression.Production.params_soundness n key rel which depths layout zeros wf slots_le
    key_fits s q

/-- At the production bounds of the layout, with at most two slot claims, the error of `soundness`
is below `2^−119`. -/
theorem production_error
    {n : ℕ}
    (key : ℕ → ℕ → ℕ → F)
    (matrices : Fin (n + 1) → Verifier.Matrices F)
    (depths : Fin n → ℕ)
    (layout : Production.Layout (n + 1))
    (vf : Verifier.VerifierData (Production.params key matrices depths layout))
    (slots : n ≤ 2)
    (columns : vf.columnVars = 2)
    (depths_le : ∀ (s : Fin n), depths s ≤ Folding.Schedule.productionLightFoldCap)
    (slots_le : ∀ (c : Fin (n + 1)), (↑c : ℕ) < n → (layout.witness c).vars ≤ 25)
    (span_le : layout.spanVars ≤ 27) :
    Verifier.error vf < 1 / 2 ^ 119 :=
  @Spec.Compression.Production.params_error_lt n key matrices depths layout vf slots columns
    depths_le
    slots_le span_le

/-- In the light mode, each layout that the Lean model of Rust `Layout::new` accepts at a depth at
most the cap has `error_bits` 119 with 320 public cells and 4 matrices. So the gate of `SPEC.md`
Algorithm 10, step 5, passes each security minimum of at most 119 bits. -/
theorem light_error_bits
    {blocks : ℕ}
    {depth : ℕ}
    {keyBlocks : ℕ}
    {l : Rust.Placed}
    (ok : Rust.layoutNew (Rust.lightClaims blocks depth) keyBlocks = Except.ok l)
    (depth_le : depth ≤ Folding.Schedule.productionLightFoldCap) :
    Rust.runBits (l.counts 320 4) = 119 :=
  @Spec.Compression.Rust.layoutNew_light_runBits blocks depth keyBlocks l ok depth_le

/-- In the state-restoration experiment with `q` queries, a prover makes an accepted run whose
output claims hold, without the input claims, with probability at most `q · ccsError`. -/
theorem fresh_sound
    {Commitment : Type}
    {PublicInput : Type}
    {shape : Folding.PiCCS.PaperJoint.Shape}
    {columns : ℕ}
    {blockCount : ℕ}
    (statement : Folding.PiCCS.PaperJoint.StrongReduction.Statement K Commitment PublicInput shape columns
        blockCount Folding.PiCCS.PaperJoint.ConcreteCarrier.baseOps)
    (witness : Folding.PiCCS.PaperJoint.StrongReduction.OutputWitness shape columns)
    (width : ℕ)
    {openingMaps : Folding.PiCCS.PaperJoint.StrongReduction.OpeningMaps Commitment PublicInput columns}
    {params : GlobalParams}
    (freshBound : params.b = 2)
    (constantLaw : Folding.PiCCS.PaperJoint.MatrixCoefficientSource.ConstantTermLaw
        Folding.PiCCS.PaperJoint.ConcreteCarrier.baseOps statement.matrixSource.kernel)
    (degreeCovers : (Folding.PiCCS.PaperJoint.StrongReduction.Statement.verifierInput K.embed
            statement).sumcheckDegreeBound ≤
        width)
    (s : Game.Strategy (Option (Option (Folding.PiCCS.PaperJoint.StrongReduction.Probe K shape))))
    (q : ℕ) :
    (Game.prob s ((Fresh.game statement witness width).restore q [])
          fun (r : Option (Option (Folding.PiCCS.PaperJoint.StrongReduction.Probe K shape))) =>
          ¬∀ (out : Option (Folding.PiCCS.PaperJoint.StrongReduction.Probe K shape)),
              r = some out →
                ∀ (probe : Folding.PiCCS.PaperJoint.StrongReduction.Probe K shape),
                  out = some probe →
                    Folding.PiCCS.PaperJoint.StrongReduction.AmbientOutputHolds
                        Folding.PiCCS.PaperJoint.ConcreteCarrier.extensionOps K.embed openingMaps
                        params statement probe witness →
                      Folding.PiCCS.PaperJoint.StrongReduction.SourceHolds
                        Folding.PiCCS.PaperJoint.ConcreteCarrier.extensionOps K.embed openingMaps
                        params statement witness) ≤
        (↑q : ℝ) * Fresh.ccsError shape width :=
  @Spec.Compression.Fresh.soundness Commitment PublicInput shape columns blockCount statement
    witness width openingMaps params freshBound constantLaw degreeCovers s q

end NightstreamFPrime.Claims.Compression

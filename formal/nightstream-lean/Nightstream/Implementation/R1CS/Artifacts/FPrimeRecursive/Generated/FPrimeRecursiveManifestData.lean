import Nightstream.Implementation.R1CS.Ownership.FPrimeRecursive.FPrimeRecursiveManifestSchema

/-! Generated diagnostic direct-CCS bit-carrier data by `gadgets_f_prime_recursive_manifest`; do not hand-edit. -/

namespace Nightstream.Implementation.R1CS.FPrimeRecursiveManifest

def schemaVersion : Nat := 3
def artifactKind : String := "r1cs/f-prime-recursive-program-manifest"
def profile : String := "diagnostic/direct-ccs-bit-carrier/plain/stateless/steady-recursive"
def piCcsSourceCount : Nat := 15
def piCcsMatrixCount : Nat := 4
def piCcsOutputFieldCount : Nat := 6683
def totalRows : Nat := 7080332
def totalColumns : Nat := 7011981
def nifsRowStart : Nat := 21855
def nifsRowEnd : Nat := 4598478
def nifsRowCount : Nat := 4576623
def totalNonzeroEntries : Nat := 32314719
def totalSha256 : String := "5616835df12f26c0d2a5a62930ea69d1f4f89f75f2c868f85ad39070036a0dc7"

def topLevelFamilies : List RowRange :=
  [ { name := "fprime.recursive.prelude", rowStart := 0, rowEnd := 6782, nonzeroEntries := 50822, sha256 := "7c306594c1ea723f5239b6983ecff0af7c3851337a94328e9f5529ba8023bb10" }
  , { name := "fprime.recursive.transcript", rowStart := 6782, rowEnd := 21855, nonzeroEntries := 120169, sha256 := "785d6568371caccff51b4cebc5bf83b5ab60bc787d470f2443be4e1ed8208bd3" }
  , { name := "fprime.recursive.nifs", rowStart := 21855, rowEnd := 4598478, nonzeroEntries := 21000019, sha256 := "7ee80dd93f6f2ab1e2e9ac98137b9d3b22bf5e6e416b63156db71f2d6c5c148b" }
  , { name := "fprime.recursive.prior_link", rowStart := 4598478, rowEnd := 4603723, nonzeroEntries := 38016, sha256 := "d20e6eb111d4369c74547bab5b71fa80d27e869695ad613b8fde398e3dafbfc9" }
  , { name := "fprime.recursive.nebula", rowStart := 4603723, rowEnd := 4603723, nonzeroEntries := 0, sha256 := "84a39c68a24131c8c8d551f5b97e172f633bf78f3e4d0d703615fc7032b314c0" }
  , { name := "fprime.recursive.accumulator", rowStart := 4603723, rowEnd := 7075042, nonzeroEntries := 11067430, sha256 := "aad1e53e1981300d070e9fab4a81ee2bcf6dfcd2a91b426091676dae575fc162" }
  , { name := "fprime.recursive.counter", rowStart := 7075042, rowEnd := 7075564, nonzeroEntries := 2136, sha256 := "21e52a0e4d9d9dfa1f20a177e5fe5c40452986b65fdbd67916f5ff3bf58e4e94" }
  , { name := "fprime.recursive.output", rowStart := 7075564, rowEnd := 7080332, nonzeroEntries := 36127, sha256 := "76fa6fcc5dcb9cd532eb0898c46f2db2e0b101125f48f865fc2d48d130068a2d" }
  ]

def nifsFamilies : List RowRange :=
  [ { name := "nifs.pi_ccs", rowStart := 21855, rowEnd := 3881029, nonzeroEntries := 17459511, sha256 := "a1af4a85f79fc66106e8922e8b8035cbf0dcda80629e25599f0fbbc06dc00b7d" }
  , { name := "nifs.running_parent_pi_dec", rowStart := 3881029, rowEnd := 3888721, nonzeroEntries := 43908, sha256 := "9d5831dce1b496787ff26bd1e6c435babe9f4bebf23fbb8987f835aa86b61279" }
  , { name := "nifs.pi_rlc", rowStart := 3888721, rowEnd := 4590768, nonzeroEntries := 3452638, sha256 := "da5c1af6bfa9ce1054ea832016fca85d9af6850228dd9be273ac08caf93496ca" }
  , { name := "nifs.pi_dec", rowStart := 4590768, rowEnd := 4598460, nonzeroEntries := 43908, sha256 := "8773e5f3f388cd4f154fcb0ef379d2064bd39193a3eab3b618d408f9c63c1813" }
  , { name := "nifs.point_binding", rowStart := 4598460, rowEnd := 4598478, nonzeroEntries := 54, sha256 := "5eab97f54fd6bf7d15486b83cbe2e89b6205310ca4f66a165594ecacc0bc7a90" }
  ]

def projectionShared : RowRange := { name := "nifs.pi_rlc.projection_shared", rowStart := 4528200, rowEnd := 4530092, nonzeroEntries := 7520, sha256 := "c529acad17d912e42d94fd9eb43efae4267326bce3ddc1d0a2701367f6062611" }
def projectionIdentityCount : Nat := 31
def projectionIdentityRows : Nat := 59396
def projectionPairCounts : List Nat := [15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15]
def projectionIdentityRanges : List RowRange :=
  [ { name := "nifs.pi_rlc.projection_identity", rowStart := 4530092, rowEnd := 4532008, nonzeroEntries := 7647, sha256 := "589bc67612bd8088d389eea913326e2b0967f64514610b9479dea8590de34b7a" }
  , { name := "nifs.pi_rlc.projection_identity", rowStart := 4532008, rowEnd := 4533924, nonzeroEntries := 7647, sha256 := "cd94e6a7bdaacef2da78032eb12fa6c5f3dd5ad634c0b37f3239fb8a808ba14c" }
  , { name := "nifs.pi_rlc.projection_identity", rowStart := 4533924, rowEnd := 4535840, nonzeroEntries := 7647, sha256 := "d19572708ddeb521b34cc032d76b245e01ab96b61b9296200b93cbfb0ebe4c88" }
  , { name := "nifs.pi_rlc.projection_identity", rowStart := 4535840, rowEnd := 4537756, nonzeroEntries := 7647, sha256 := "c3bc25d296bfd2e9599a9c2829fe812e7abd4b034da23ce2aa76d90f70fb7bfa" }
  , { name := "nifs.pi_rlc.projection_identity", rowStart := 4537756, rowEnd := 4539672, nonzeroEntries := 7647, sha256 := "112d801ff04bf92a42eaba7cb4162c399135b4d3bd7bdd7a236d5d65eff9773b" }
  , { name := "nifs.pi_rlc.projection_identity", rowStart := 4539672, rowEnd := 4541588, nonzeroEntries := 7647, sha256 := "4d0351943eefe59042c29d300975ae9e0acb9c00fc49cb8ca40d91fb22399060" }
  , { name := "nifs.pi_rlc.projection_identity", rowStart := 4541588, rowEnd := 4543504, nonzeroEntries := 7647, sha256 := "098d56a634777260194b53f09653c24ad09a91f0dfbbcd68e6f919f0161884a2" }
  , { name := "nifs.pi_rlc.projection_identity", rowStart := 4543504, rowEnd := 4545420, nonzeroEntries := 7647, sha256 := "e32c76c116c76c62cc2c81e7061d990d9be2726cdd657d23ca93940e452a9f2f" }
  , { name := "nifs.pi_rlc.projection_identity", rowStart := 4545420, rowEnd := 4547336, nonzeroEntries := 7647, sha256 := "4012774a31860b45c5bc055f1011f9f99b39dd8d1e3aab16fb315d22154d3b72" }
  , { name := "nifs.pi_rlc.projection_identity", rowStart := 4547336, rowEnd := 4549252, nonzeroEntries := 7647, sha256 := "6ccd2e95c0e469f633b0eeb7b0f3c053db8bcfd51bb7920c4eb2709f9d7627df" }
  , { name := "nifs.pi_rlc.projection_identity", rowStart := 4549252, rowEnd := 4551168, nonzeroEntries := 7647, sha256 := "18a824096350c6e6892c8e33f0705ae89b2139b0c595397fec767bd3c36e5306" }
  , { name := "nifs.pi_rlc.projection_identity", rowStart := 4551168, rowEnd := 4553084, nonzeroEntries := 7647, sha256 := "5fd8d12121e15807c33263cd821922f494e0fefd0d5db732fe57f46fdb90db21" }
  , { name := "nifs.pi_rlc.projection_identity", rowStart := 4553084, rowEnd := 4555000, nonzeroEntries := 7647, sha256 := "9ae72406d0a07fe47dff4c16466d6227ff377db07f88d04ef88ef59e9e46b2b3" }
  , { name := "nifs.pi_rlc.projection_identity", rowStart := 4555000, rowEnd := 4556916, nonzeroEntries := 7647, sha256 := "1aa234b2fcfc0756a5aae077c49fddb6f6c410bddc6ae1265875ccaffbf9b9f7" }
  , { name := "nifs.pi_rlc.projection_identity", rowStart := 4556916, rowEnd := 4558832, nonzeroEntries := 7647, sha256 := "379cf06326f323b5e23c0831e4e6cfea7c762460939bb1214ab49a82b7d83ac5" }
  , { name := "nifs.pi_rlc.projection_identity", rowStart := 4558832, rowEnd := 4560748, nonzeroEntries := 7647, sha256 := "c5d24188691cf8ab65dccb5dd70b2deb5eb7316585d6e31da03ace69f4dc73c8" }
  , { name := "nifs.pi_rlc.projection_identity", rowStart := 4560748, rowEnd := 4562664, nonzeroEntries := 7647, sha256 := "4956da77807d63d646fc9045ce5156b88dee077edff2dab308ce8dc0e8cdeabc" }
  , { name := "nifs.pi_rlc.projection_identity", rowStart := 4562664, rowEnd := 4564580, nonzeroEntries := 7647, sha256 := "e10e19750210109d80b762b8d1e8ee07b40379ad6147ef2e0231eb3bf7c37689" }
  , { name := "nifs.pi_rlc.projection_identity", rowStart := 4564580, rowEnd := 4566496, nonzeroEntries := 7647, sha256 := "c768282281b407985c9ba0320303f4fbeff932777e2ccf06d070fd845e33f747" }
  , { name := "nifs.pi_rlc.projection_identity", rowStart := 4566496, rowEnd := 4568412, nonzeroEntries := 7647, sha256 := "ca1e1399f34a5b16b4a511a4f86e6099283b3fa7621c35343b6e8c613cdec2ae" }
  , { name := "nifs.pi_rlc.projection_identity", rowStart := 4568412, rowEnd := 4570328, nonzeroEntries := 7647, sha256 := "a61f7285413964b5008b8d5cb5184505a71b83bd4f062c90be571e272b99e79a" }
  , { name := "nifs.pi_rlc.projection_identity", rowStart := 4570328, rowEnd := 4572244, nonzeroEntries := 7647, sha256 := "0b1f8f2d4bc88283c7ac2621951799e833ba180593b8944f50688d786a61e727" }
  , { name := "nifs.pi_rlc.projection_identity", rowStart := 4572244, rowEnd := 4574160, nonzeroEntries := 7647, sha256 := "74df807ecde6ed80bbc4427f67b001460ceb5463b1f73fc3661ede0ef081a2d0" }
  , { name := "nifs.pi_rlc.projection_identity", rowStart := 4574160, rowEnd := 4576076, nonzeroEntries := 7647, sha256 := "e4be54d915c2281bd8d24ab5b7318b0ca53367384fa0f0498b49c8e01207a67b" }
  , { name := "nifs.pi_rlc.projection_identity", rowStart := 4576076, rowEnd := 4577992, nonzeroEntries := 7647, sha256 := "3688b169eca5949dce9e4f471cd446e292e45762d3efbb3527dfab3a6d971e16" }
  , { name := "nifs.pi_rlc.projection_identity", rowStart := 4578312, rowEnd := 4580228, nonzeroEntries := 7647, sha256 := "5f2b9589186d4f1549fc09de2d15e2e4d9707e299900c741fde27db37181c714" }
  , { name := "nifs.pi_rlc.projection_identity", rowStart := 4580228, rowEnd := 4582144, nonzeroEntries := 7647, sha256 := "a689ec8b357a989af7731ad11cd80f1bf16144f00d972d1b156c487727da9aac" }
  , { name := "nifs.pi_rlc.projection_identity", rowStart := 4582464, rowEnd := 4584380, nonzeroEntries := 7647, sha256 := "c240ee6106e44c1ae841bec8f378cc8e4b5c2dfa0bc7fb8ead819075cd5ee686" }
  , { name := "nifs.pi_rlc.projection_identity", rowStart := 4584380, rowEnd := 4586296, nonzeroEntries := 7647, sha256 := "da564e5514a70ce7b880279bb9af557050afd45a9da6aa435db14b99f4f5ca2a" }
  , { name := "nifs.pi_rlc.projection_identity", rowStart := 4586616, rowEnd := 4588532, nonzeroEntries := 7647, sha256 := "433f85479a89a8eea87337c7147fee5a29a01f4252910d9fef4776580160756f" }
  , { name := "nifs.pi_rlc.projection_identity", rowStart := 4588532, rowEnd := 4590448, nonzeroEntries := 7647, sha256 := "7d90b84789d8037062c33897f61ad7fdf43286860cf3a420f2c568f70837c23f" }
  ]

end Nightstream.Implementation.R1CS.FPrimeRecursiveManifest

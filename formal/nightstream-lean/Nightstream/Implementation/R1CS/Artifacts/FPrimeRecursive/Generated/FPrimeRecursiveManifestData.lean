import Nightstream.Implementation.R1CS.Ownership.FPrimeRecursive.FPrimeRecursiveManifestSchema

/-! Generated diagnostic direct-CCS bit-carrier data by `gadgets_f_prime_recursive_manifest`; do not hand-edit. -/

namespace Nightstream.Implementation.R1CS.FPrimeRecursiveManifest

def schemaVersion : Nat := 3
def artifactKind : String := "r1cs/f-prime-recursive-program-manifest"
def profile : String := "diagnostic/direct-ccs-bit-carrier/plain/stateless/steady-recursive"
def piCcsSourceCount : Nat := 15
def piCcsMatrixCount : Nat := 4
def piCcsOutputFieldCount : Nat := 6683
def totalRows : Nat := 9297088
def totalColumns : Nat := 9000422
def nifsRowStart : Nat := 21855
def nifsRowEnd : Nat := 5807057
def nifsRowCount : Nat := 5785202
def totalNonzeroEntries : Nat := 42884467
def totalSha256 : String := "6d24b2a8e25f3b07fb7fdb573a280eec377e97b4424136a1d90e600ceead1418"

def topLevelFamilies : List RowRange :=
  [ { name := "fprime.recursive.prelude", rowStart := 0, rowEnd := 6782, nonzeroEntries := 50822, sha256 := "7c306594c1ea723f5239b6983ecff0af7c3851337a94328e9f5529ba8023bb10" }
  , { name := "fprime.recursive.transcript", rowStart := 6782, rowEnd := 21855, nonzeroEntries := 120169, sha256 := "785d6568371caccff51b4cebc5bf83b5ab60bc787d470f2443be4e1ed8208bd3" }
  , { name := "fprime.recursive.nifs", rowStart := 21855, rowEnd := 5807057, nonzeroEntries := 26585465, sha256 := "184a1d52475fc3f1be78b76be59a09ce8ec025d817bc9e7bc0d3e9a6c2790472" }
  , { name := "fprime.recursive.prior_link", rowStart := 5807057, rowEnd := 5812302, nonzeroEntries := 38016, sha256 := "809c97cb3b3ce032f9dadd5eee5186e0a45d4d0a5ecea89caca30f362a879fe3" }
  , { name := "fprime.recursive.nebula", rowStart := 5812302, rowEnd := 5812302, nonzeroEntries := 0, sha256 := "84a39c68a24131c8c8d551f5b97e172f633bf78f3e4d0d703615fc7032b314c0" }
  , { name := "fprime.recursive.accumulator", rowStart := 5812302, rowEnd := 9291798, nonzeroEntries := 16051732, sha256 := "2d0f63d60aa0fe86a8489c96a36f76492776e610c7fa8c3611a6dd64b4367037" }
  , { name := "fprime.recursive.counter", rowStart := 9291798, rowEnd := 9292320, nonzeroEntries := 2136, sha256 := "3753183bb4365ea50876542c6d5f1a450913ff2a45080004fba74573fc3bae80" }
  , { name := "fprime.recursive.output", rowStart := 9292320, rowEnd := 9297088, nonzeroEntries := 36127, sha256 := "f2ac62e6cbe82484540cf642cc5caa3e57bce63798a11f7ee3829647bfa23f77" }
  ]

def nifsFamilies : List RowRange :=
  [ { name := "nifs.pi_ccs", rowStart := 21855, rowEnd := 5089547, nonzeroEntries := 23044835, sha256 := "1f707e7c19083a712e52630e2ad428ab719768175e0a9ed069ace08f0d515064" }
  , { name := "nifs.running_parent_pi_dec", rowStart := 5089547, rowEnd := 5097254, nonzeroEntries := 43938, sha256 := "72f90877da66d33f6237c423bf787d65fab24c1e66f93477119f0dab65cf0b47" }
  , { name := "nifs.pi_rlc", rowStart := 5097254, rowEnd := 5799332, nonzeroEntries := 3452700, sha256 := "f506514ebc8821d6cbc7743ac80d3cd490aaff53a0c508804a5e91a32bc112c0" }
  , { name := "nifs.pi_dec", rowStart := 5799332, rowEnd := 5807039, nonzeroEntries := 43938, sha256 := "d4ff7bd5430db817955aeee7df94c1fa8cd6c715d37d4ec549155143883c3761" }
  , { name := "nifs.point_binding", rowStart := 5807039, rowEnd := 5807057, nonzeroEntries := 54, sha256 := "5052b78820f471c8b4d46f6d2845bd16b710effaeee0632815a4011edbaa3532" }
  ]

def projectionShared : RowRange := { name := "nifs.pi_rlc.projection_shared", rowStart := 5736748, rowEnd := 5738640, nonzeroEntries := 7520, sha256 := "23b5e912f5091414b7f1806cf623369ed9b802e4e9c985702ff8be60a16be13e" }
def projectionIdentityCount : Nat := 31
def projectionIdentityRows : Nat := 59396
def projectionPairCounts : List Nat := [15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15]
def projectionIdentityRanges : List RowRange :=
  [ { name := "nifs.pi_rlc.projection_identity", rowStart := 5738640, rowEnd := 5740556, nonzeroEntries := 7647, sha256 := "2ba0ad6b88c544c01d9f6e615af15e4c48fe61e6c6a23310f31bdf7f96e7a163" }
  , { name := "nifs.pi_rlc.projection_identity", rowStart := 5740556, rowEnd := 5742472, nonzeroEntries := 7647, sha256 := "10f46a784606136763776162f0c3e8806c56173324200c8415f9318c69943efd" }
  , { name := "nifs.pi_rlc.projection_identity", rowStart := 5742472, rowEnd := 5744388, nonzeroEntries := 7647, sha256 := "c8ac77e4eadcdb0838bf0ce9ea2194aed7aff75fc7d4dbba2b643ff4b27726ba" }
  , { name := "nifs.pi_rlc.projection_identity", rowStart := 5744388, rowEnd := 5746304, nonzeroEntries := 7647, sha256 := "893067d77e5389e05e8d59fe88f08b20ed180bef72aa148be5dfb8ad9d113af9" }
  , { name := "nifs.pi_rlc.projection_identity", rowStart := 5746304, rowEnd := 5748220, nonzeroEntries := 7647, sha256 := "bfbcbdb5f8ddfd4b959bd3b36d51d385e572929a42743af63b845ee1fced36a0" }
  , { name := "nifs.pi_rlc.projection_identity", rowStart := 5748220, rowEnd := 5750136, nonzeroEntries := 7647, sha256 := "1be34a1e89f5ebfb6d34695ea1a914f706b5f6d2c0ad20a9ee48e03c0f8317f7" }
  , { name := "nifs.pi_rlc.projection_identity", rowStart := 5750136, rowEnd := 5752052, nonzeroEntries := 7647, sha256 := "48265dc45e392c66eb8b14e25d562e63ca9ec74d230f912d2dc00d6edf27d178" }
  , { name := "nifs.pi_rlc.projection_identity", rowStart := 5752052, rowEnd := 5753968, nonzeroEntries := 7647, sha256 := "e7d1c20db1cb6fc03ab76c8ac1174031b4b32cc193bbbc4ccef8f69485f89858" }
  , { name := "nifs.pi_rlc.projection_identity", rowStart := 5753968, rowEnd := 5755884, nonzeroEntries := 7647, sha256 := "646fc8ff434bd7e36f0eee66e7a79d354ac1222407358e42a74f0cc7502e02bc" }
  , { name := "nifs.pi_rlc.projection_identity", rowStart := 5755884, rowEnd := 5757800, nonzeroEntries := 7647, sha256 := "e6c42e8f3c240f4a82e888af9201704767b608dce6284b27ddf7d00c7a207931" }
  , { name := "nifs.pi_rlc.projection_identity", rowStart := 5757800, rowEnd := 5759716, nonzeroEntries := 7647, sha256 := "0c759f66e226233fabad1fc94fc8f87992e356510abfa81a1e5f248b57952ba7" }
  , { name := "nifs.pi_rlc.projection_identity", rowStart := 5759716, rowEnd := 5761632, nonzeroEntries := 7647, sha256 := "8804569711ee0a973f0c44002f508ad4f0c33fd2aabc3621cd773ffbb4e8dfd5" }
  , { name := "nifs.pi_rlc.projection_identity", rowStart := 5761632, rowEnd := 5763548, nonzeroEntries := 7647, sha256 := "fa6909952998be77a32d72ea9d534433cfda27e720d1baa55b7654722e11d378" }
  , { name := "nifs.pi_rlc.projection_identity", rowStart := 5763548, rowEnd := 5765464, nonzeroEntries := 7647, sha256 := "57503879ea882335c01433b0db178383582fa2c7e1e3a1b44746c03aa046a56c" }
  , { name := "nifs.pi_rlc.projection_identity", rowStart := 5765464, rowEnd := 5767380, nonzeroEntries := 7647, sha256 := "099b2bbc869eadc44d661227bacba7419c275be73701a8fd6f6f03730787c1a9" }
  , { name := "nifs.pi_rlc.projection_identity", rowStart := 5767380, rowEnd := 5769296, nonzeroEntries := 7647, sha256 := "0b7064981a91087dfc896f5ca81f44abbf1ea36fc630a7f074a4f520324e8b62" }
  , { name := "nifs.pi_rlc.projection_identity", rowStart := 5769296, rowEnd := 5771212, nonzeroEntries := 7647, sha256 := "4b3a34c6748c8904d7976c7df7c71fb9ec966a8a46367f30463eacf271873eda" }
  , { name := "nifs.pi_rlc.projection_identity", rowStart := 5771212, rowEnd := 5773128, nonzeroEntries := 7647, sha256 := "03e443f8d925c9a1f86b88539bd19db7b473e4624f23175a5f3c89d9d45466bd" }
  , { name := "nifs.pi_rlc.projection_identity", rowStart := 5773128, rowEnd := 5775044, nonzeroEntries := 7647, sha256 := "3068ea419d3bfb8816133b6f9ba41c3e41159c13d019a8c995355de07683d1c1" }
  , { name := "nifs.pi_rlc.projection_identity", rowStart := 5775044, rowEnd := 5776960, nonzeroEntries := 7647, sha256 := "19fe10657a357987c0c2d3eccf24a398873de1b0adab69191bb1dcbd73aea503" }
  , { name := "nifs.pi_rlc.projection_identity", rowStart := 5776960, rowEnd := 5778876, nonzeroEntries := 7647, sha256 := "d04d81bf2b6d2983d6f020c1a22acc10c69a8a309b0dcca6aa1a3b33ee3c7201" }
  , { name := "nifs.pi_rlc.projection_identity", rowStart := 5778876, rowEnd := 5780792, nonzeroEntries := 7647, sha256 := "fe7aeb3c143b0a58aed114c9fb6854bfd2d28f0961cdfe03754ece33fbf584a7" }
  , { name := "nifs.pi_rlc.projection_identity", rowStart := 5780792, rowEnd := 5782708, nonzeroEntries := 7647, sha256 := "00373e861fea6de3d2321467612bee3017f5513448df7e53e7bb800b17b7ff34" }
  , { name := "nifs.pi_rlc.projection_identity", rowStart := 5782724, rowEnd := 5784640, nonzeroEntries := 7647, sha256 := "ce4f0970ce44286620756dc94ec5700802ec1a4b4c079ca0521ddc0496507287" }
  , { name := "nifs.pi_rlc.projection_identity", rowStart := 5784640, rowEnd := 5786556, nonzeroEntries := 7647, sha256 := "c4aea576b983951c681641303447e8a51403feb82db90b3054f98b818fa003ce" }
  , { name := "nifs.pi_rlc.projection_identity", rowStart := 5786876, rowEnd := 5788792, nonzeroEntries := 7647, sha256 := "c9179cc0b26ccea82e079405c9235e756408cc222f0f0bd0520c8f7d310602c4" }
  , { name := "nifs.pi_rlc.projection_identity", rowStart := 5788792, rowEnd := 5790708, nonzeroEntries := 7647, sha256 := "59004814cc3d5f0c67f60d77cff5d46445f8360ac64765ee762907e3c41f0f5d" }
  , { name := "nifs.pi_rlc.projection_identity", rowStart := 5791028, rowEnd := 5792944, nonzeroEntries := 7647, sha256 := "4551f35b35a3d1b6302216d51a818b5be03dd8d5fcbc6355625ed32abf159090" }
  , { name := "nifs.pi_rlc.projection_identity", rowStart := 5792944, rowEnd := 5794860, nonzeroEntries := 7647, sha256 := "400c2a34d5997276755e0932b52f02e0160b53123e43f4a064824313c1c14f94" }
  , { name := "nifs.pi_rlc.projection_identity", rowStart := 5795180, rowEnd := 5797096, nonzeroEntries := 7647, sha256 := "9bc5dd8e7459a2899eccae49b658e4aa0b2e46d1b8d2ea1842708bd9279f9311" }
  , { name := "nifs.pi_rlc.projection_identity", rowStart := 5797096, rowEnd := 5799012, nonzeroEntries := 7647, sha256 := "0c4946d5e0f179fa0c504a9cabeecbefdd155cb9bcfd9c6b802b6e3c7864572e" }
  ]

end Nightstream.Implementation.R1CS.FPrimeRecursiveManifest

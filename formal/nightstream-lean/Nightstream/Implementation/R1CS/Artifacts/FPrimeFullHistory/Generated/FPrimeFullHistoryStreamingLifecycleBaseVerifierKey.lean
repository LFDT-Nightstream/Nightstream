import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingLifecycleBaseVerifierKeySchema

/-! Generated exact source-to-final provenance for the base lifecycle verifier-key core.

This is a compact leaf of the monolithic reference compiler audit. It is not the final phased profile.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingLifecycleBaseVerifierKey

open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingLifecycleBaseVerifierKey.Artifact

def artifactSha256 : String := "b8cdf41a4fed2084bd3aba3906ae9b9a30d3ee27b88ad2b1839eb7867f31ed26"

def sourceRuns : List SourceRun := [
    { sourceRows := { start := 36133, stop := 36134 }, disposition := "LinearDefinition(SelectiveRewriteId(1740))", emittedStart := none },
    { sourceRows := { start := 36134, stop := 36135 }, disposition := "LinearDefinition(SelectiveRewriteId(1741))", emittedStart := none },
    { sourceRows := { start := 36135, stop := 36136 }, disposition := "LinearDefinition(SelectiveRewriteId(1742))", emittedStart := none },
    { sourceRows := { start := 36136, stop := 36137 }, disposition := "LinearDefinition(SelectiveRewriteId(1743))", emittedStart := none },
    { sourceRows := { start := 36137, stop := 36138 }, disposition := "LinearDefinition(SelectiveRewriteId(1744))", emittedStart := none },
    { sourceRows := { start := 36138, stop := 36139 }, disposition := "LinearDefinition(SelectiveRewriteId(1745))", emittedStart := none },
    { sourceRows := { start := 36139, stop := 36140 }, disposition := "LinearDefinition(SelectiveRewriteId(1746))", emittedStart := none },
    { sourceRows := { start := 36140, stop := 36141 }, disposition := "LinearDefinition(SelectiveRewriteId(1747))", emittedStart := none },
    { sourceRows := { start := 36141, stop := 36142 }, disposition := "LinearDefinition(SelectiveRewriteId(1748))", emittedStart := none },
    { sourceRows := { start := 36142, stop := 36143 }, disposition := "LinearDefinition(SelectiveRewriteId(1749))", emittedStart := none },
    { sourceRows := { start := 36143, stop := 36144 }, disposition := "LinearDefinition(SelectiveRewriteId(1750))", emittedStart := none },
    { sourceRows := { start := 36144, stop := 36145 }, disposition := "LinearDefinition(SelectiveRewriteId(1751))", emittedStart := none },
    { sourceRows := { start := 36145, stop := 36146 }, disposition := "LinearDefinition(SelectiveRewriteId(1752))", emittedStart := none },
    { sourceRows := { start := 36146, stop := 36147 }, disposition := "LinearDefinition(SelectiveRewriteId(1753))", emittedStart := none },
    { sourceRows := { start := 36147, stop := 36148 }, disposition := "LinearDefinition(SelectiveRewriteId(1754))", emittedStart := none },
    { sourceRows := { start := 36148, stop := 36149 }, disposition := "LinearDefinition(SelectiveRewriteId(1755))", emittedStart := none },
    { sourceRows := { start := 36149, stop := 36150 }, disposition := "LinearDefinition(SelectiveRewriteId(1756))", emittedStart := none },
    { sourceRows := { start := 36150, stop := 36151 }, disposition := "LinearDefinition(SelectiveRewriteId(1757))", emittedStart := none },
    { sourceRows := { start := 36151, stop := 36152 }, disposition := "LinearDefinition(SelectiveRewriteId(1758))", emittedStart := none },
    { sourceRows := { start := 36152, stop := 36153 }, disposition := "LinearDefinition(SelectiveRewriteId(1759))", emittedStart := none },
    { sourceRows := { start := 36153, stop := 36154 }, disposition := "LinearDefinition(SelectiveRewriteId(1760))", emittedStart := none },
    { sourceRows := { start := 36154, stop := 36155 }, disposition := "LinearDefinition(SelectiveRewriteId(1761))", emittedStart := none },
    { sourceRows := { start := 36155, stop := 36156 }, disposition := "LinearDefinition(SelectiveRewriteId(1762))", emittedStart := none },
    { sourceRows := { start := 36156, stop := 36157 }, disposition := "LinearDefinition(SelectiveRewriteId(1763))", emittedStart := none },
    { sourceRows := { start := 36157, stop := 36158 }, disposition := "LinearDefinition(SelectiveRewriteId(1764))", emittedStart := none },
    { sourceRows := { start := 36158, stop := 36159 }, disposition := "LinearDefinition(SelectiveRewriteId(1765))", emittedStart := none },
    { sourceRows := { start := 36159, stop := 36759 }, disposition := "Poseidon2(SelectiveRewriteId(57))", emittedStart := none },
    { sourceRows := { start := 36759, stop := 36760 }, disposition := "LinearDefinition(SelectiveRewriteId(1766))", emittedStart := none },
    { sourceRows := { start := 36760, stop := 36761 }, disposition := "LinearDefinition(SelectiveRewriteId(1767))", emittedStart := none },
    { sourceRows := { start := 36761, stop := 36762 }, disposition := "LinearDefinition(SelectiveRewriteId(1768))", emittedStart := none },
    { sourceRows := { start := 36762, stop := 36763 }, disposition := "LinearDefinition(SelectiveRewriteId(1769))", emittedStart := none },
    { sourceRows := { start := 36763, stop := 37363 }, disposition := "Poseidon2(SelectiveRewriteId(58))", emittedStart := none },
    { sourceRows := { start := 37363, stop := 37364 }, disposition := "LinearDefinition(SelectiveRewriteId(1770))", emittedStart := none },
    { sourceRows := { start := 37364, stop := 37365 }, disposition := "LinearDefinition(SelectiveRewriteId(1771))", emittedStart := none },
    { sourceRows := { start := 37365, stop := 37366 }, disposition := "LinearDefinition(SelectiveRewriteId(1772))", emittedStart := none },
    { sourceRows := { start := 37366, stop := 37367 }, disposition := "LinearDefinition(SelectiveRewriteId(1773))", emittedStart := none },
    { sourceRows := { start := 37367, stop := 37967 }, disposition := "Poseidon2(SelectiveRewriteId(59))", emittedStart := none },
    { sourceRows := { start := 37967, stop := 37968 }, disposition := "LinearDefinition(SelectiveRewriteId(1774))", emittedStart := none },
    { sourceRows := { start := 37968, stop := 37969 }, disposition := "LinearDefinition(SelectiveRewriteId(1775))", emittedStart := none },
    { sourceRows := { start := 37969, stop := 37970 }, disposition := "LinearDefinition(SelectiveRewriteId(1776))", emittedStart := none },
    { sourceRows := { start := 37970, stop := 37971 }, disposition := "LinearDefinition(SelectiveRewriteId(1777))", emittedStart := none },
    { sourceRows := { start := 37971, stop := 38571 }, disposition := "Poseidon2(SelectiveRewriteId(60))", emittedStart := none },
    { sourceRows := { start := 38571, stop := 38572 }, disposition := "LinearDefinition(SelectiveRewriteId(1778))", emittedStart := none },
    { sourceRows := { start := 38572, stop := 38573 }, disposition := "LinearDefinition(SelectiveRewriteId(1779))", emittedStart := none },
    { sourceRows := { start := 38573, stop := 38574 }, disposition := "LinearDefinition(SelectiveRewriteId(1780))", emittedStart := none },
    { sourceRows := { start := 38574, stop := 38575 }, disposition := "LinearDefinition(SelectiveRewriteId(1781))", emittedStart := none },
    { sourceRows := { start := 38575, stop := 39175 }, disposition := "Poseidon2(SelectiveRewriteId(61))", emittedStart := none },
    { sourceRows := { start := 39175, stop := 39176 }, disposition := "LinearDefinition(SelectiveRewriteId(1782))", emittedStart := none },
    { sourceRows := { start := 39176, stop := 39177 }, disposition := "LinearDefinition(SelectiveRewriteId(1783))", emittedStart := none },
    { sourceRows := { start := 39177, stop := 39178 }, disposition := "LinearDefinition(SelectiveRewriteId(1784))", emittedStart := none },
    { sourceRows := { start := 39178, stop := 39179 }, disposition := "LinearDefinition(SelectiveRewriteId(1785))", emittedStart := none },
    { sourceRows := { start := 39179, stop := 39779 }, disposition := "Poseidon2(SelectiveRewriteId(62))", emittedStart := none },
    { sourceRows := { start := 39779, stop := 39780 }, disposition := "LinearDefinition(SelectiveRewriteId(1786))", emittedStart := none },
    { sourceRows := { start := 39780, stop := 39781 }, disposition := "LinearDefinition(SelectiveRewriteId(1787))", emittedStart := none },
    { sourceRows := { start := 39781, stop := 39782 }, disposition := "LinearDefinition(SelectiveRewriteId(1788))", emittedStart := none },
    { sourceRows := { start := 39782, stop := 39783 }, disposition := "LinearDefinition(SelectiveRewriteId(1789))", emittedStart := none },
    { sourceRows := { start := 39783, stop := 40383 }, disposition := "Poseidon2(SelectiveRewriteId(63))", emittedStart := none },
    { sourceRows := { start := 40383, stop := 40384 }, disposition := "LinearDefinition(SelectiveRewriteId(1790))", emittedStart := none },
    { sourceRows := { start := 40384, stop := 40385 }, disposition := "LinearDefinition(SelectiveRewriteId(1791))", emittedStart := none },
    { sourceRows := { start := 40385, stop := 40386 }, disposition := "LinearDefinition(SelectiveRewriteId(1792))", emittedStart := none },
    { sourceRows := { start := 40386, stop := 40387 }, disposition := "LinearDefinition(SelectiveRewriteId(1793))", emittedStart := none },
    { sourceRows := { start := 40387, stop := 40987 }, disposition := "Poseidon2(SelectiveRewriteId(64))", emittedStart := none },
    { sourceRows := { start := 40987, stop := 40988 }, disposition := "LinearDefinition(SelectiveRewriteId(1794))", emittedStart := none },
    { sourceRows := { start := 40988, stop := 40989 }, disposition := "LinearDefinition(SelectiveRewriteId(1795))", emittedStart := none },
    { sourceRows := { start := 40989, stop := 40990 }, disposition := "LinearDefinition(SelectiveRewriteId(1796))", emittedStart := none },
    { sourceRows := { start := 40990, stop := 40991 }, disposition := "LinearDefinition(SelectiveRewriteId(1797))", emittedStart := none },
    { sourceRows := { start := 40991, stop := 41591 }, disposition := "Poseidon2(SelectiveRewriteId(65))", emittedStart := none },
    { sourceRows := { start := 41591, stop := 41592 }, disposition := "LinearDefinition(SelectiveRewriteId(1798))", emittedStart := none },
    { sourceRows := { start := 41592, stop := 42192 }, disposition := "Poseidon2(SelectiveRewriteId(66))", emittedStart := none },
    { sourceRows := { start := 42192, stop := 42193 }, disposition := "LinearDefinition(SelectiveRewriteId(1799))", emittedStart := none },
    { sourceRows := { start := 42193, stop := 42793 }, disposition := "Poseidon2(SelectiveRewriteId(67))", emittedStart := none },
    { sourceRows := { start := 42793, stop := 42794 }, disposition := "LinearDefinition(SelectiveRewriteId(1800))", emittedStart := none },
    { sourceRows := { start := 42794, stop := 42795 }, disposition := "LinearDefinition(SelectiveRewriteId(1801))", emittedStart := none },
    { sourceRows := { start := 42795, stop := 42796 }, disposition := "LinearDefinition(SelectiveRewriteId(1802))", emittedStart := none },
    { sourceRows := { start := 42796, stop := 42797 }, disposition := "LinearDefinition(SelectiveRewriteId(1803))", emittedStart := none },
    { sourceRows := { start := 42797, stop := 42798 }, disposition := "LinearDefinition(SelectiveRewriteId(1804))", emittedStart := none },
    { sourceRows := { start := 42798, stop := 42799 }, disposition := "LinearDefinition(SelectiveRewriteId(1805))", emittedStart := none },
    { sourceRows := { start := 42799, stop := 42800 }, disposition := "LinearDefinition(SelectiveRewriteId(1806))", emittedStart := none },
    { sourceRows := { start := 42800, stop := 42801 }, disposition := "LinearDefinition(SelectiveRewriteId(1807))", emittedStart := none },
    { sourceRows := { start := 42801, stop := 42802 }, disposition := "LinearDefinition(SelectiveRewriteId(1808))", emittedStart := none },
    { sourceRows := { start := 42802, stop := 42803 }, disposition := "LinearDefinition(SelectiveRewriteId(1809))", emittedStart := none },
    { sourceRows := { start := 42803, stop := 42804 }, disposition := "LinearDefinition(SelectiveRewriteId(1810))", emittedStart := none },
    { sourceRows := { start := 42804, stop := 42805 }, disposition := "LinearDefinition(SelectiveRewriteId(1811))", emittedStart := none },
    { sourceRows := { start := 42805, stop := 42806 }, disposition := "LinearDefinition(SelectiveRewriteId(1812))", emittedStart := none },
    { sourceRows := { start := 42806, stop := 42807 }, disposition := "LinearDefinition(SelectiveRewriteId(1813))", emittedStart := none },
    { sourceRows := { start := 42807, stop := 43407 }, disposition := "Poseidon2(SelectiveRewriteId(68))", emittedStart := none },
    { sourceRows := { start := 43407, stop := 43408 }, disposition := "LinearDefinition(SelectiveRewriteId(1814))", emittedStart := none },
    { sourceRows := { start := 43408, stop := 43409 }, disposition := "LinearDefinition(SelectiveRewriteId(1815))", emittedStart := none },
    { sourceRows := { start := 43409, stop := 43410 }, disposition := "LinearDefinition(SelectiveRewriteId(1816))", emittedStart := none },
    { sourceRows := { start := 43410, stop := 43411 }, disposition := "LinearDefinition(SelectiveRewriteId(1817))", emittedStart := none },
    { sourceRows := { start := 43411, stop := 44011 }, disposition := "Poseidon2(SelectiveRewriteId(69))", emittedStart := none },
    { sourceRows := { start := 44011, stop := 44012 }, disposition := "LinearDefinition(SelectiveRewriteId(1818))", emittedStart := none },
    { sourceRows := { start := 44012, stop := 44013 }, disposition := "LinearDefinition(SelectiveRewriteId(1819))", emittedStart := none },
    { sourceRows := { start := 44013, stop := 44014 }, disposition := "LinearDefinition(SelectiveRewriteId(1820))", emittedStart := none },
    { sourceRows := { start := 44014, stop := 44015 }, disposition := "LinearDefinition(SelectiveRewriteId(1821))", emittedStart := none },
    { sourceRows := { start := 44015, stop := 44615 }, disposition := "Poseidon2(SelectiveRewriteId(70))", emittedStart := none },
    { sourceRows := { start := 44615, stop := 44616 }, disposition := "LinearDefinition(SelectiveRewriteId(1822))", emittedStart := none },
    { sourceRows := { start := 44616, stop := 45216 }, disposition := "Poseidon2(SelectiveRewriteId(71))", emittedStart := none },
    { sourceRows := { start := 45216, stop := 45217 }, disposition := "LinearDefinition(SelectiveRewriteId(1823))", emittedStart := none },
    { sourceRows := { start := 45217, stop := 45817 }, disposition := "Poseidon2(SelectiveRewriteId(72))", emittedStart := none },
    { sourceRows := { start := 45817, stop := 45821 }, disposition := "Retained", emittedStart := some 4772267 },
    { sourceRows := { start := 45821, stop := 45822 }, disposition := "LinearDefinition(SelectiveRewriteId(1824))", emittedStart := none },
    { sourceRows := { start := 45822, stop := 45823 }, disposition := "LinearDefinition(SelectiveRewriteId(1825))", emittedStart := none },
    { sourceRows := { start := 45823, stop := 45824 }, disposition := "LinearDefinition(SelectiveRewriteId(1826))", emittedStart := none },
    { sourceRows := { start := 45824, stop := 45825 }, disposition := "LinearDefinition(SelectiveRewriteId(1827))", emittedStart := none },
    { sourceRows := { start := 45825, stop := 45826 }, disposition := "LinearDefinition(SelectiveRewriteId(1828))", emittedStart := none },
    { sourceRows := { start := 45826, stop := 45827 }, disposition := "LinearDefinition(SelectiveRewriteId(1829))", emittedStart := none },
    { sourceRows := { start := 45827, stop := 45828 }, disposition := "LinearDefinition(SelectiveRewriteId(1830))", emittedStart := none },
    { sourceRows := { start := 45828, stop := 45829 }, disposition := "LinearDefinition(SelectiveRewriteId(1831))", emittedStart := none },
    { sourceRows := { start := 45829, stop := 45830 }, disposition := "LinearDefinition(SelectiveRewriteId(1832))", emittedStart := none },
    { sourceRows := { start := 45830, stop := 45831 }, disposition := "LinearDefinition(SelectiveRewriteId(1833))", emittedStart := none },
    { sourceRows := { start := 45831, stop := 45832 }, disposition := "LinearDefinition(SelectiveRewriteId(1834))", emittedStart := none },
    { sourceRows := { start := 45832, stop := 45833 }, disposition := "LinearDefinition(SelectiveRewriteId(1835))", emittedStart := none },
    { sourceRows := { start := 45833, stop := 45834 }, disposition := "LinearDefinition(SelectiveRewriteId(1836))", emittedStart := none },
    { sourceRows := { start := 45834, stop := 46434 }, disposition := "Poseidon2(SelectiveRewriteId(73))", emittedStart := none },
    { sourceRows := { start := 46434, stop := 46435 }, disposition := "LinearDefinition(SelectiveRewriteId(1837))", emittedStart := none },
    { sourceRows := { start := 46435, stop := 46436 }, disposition := "LinearDefinition(SelectiveRewriteId(1838))", emittedStart := none },
    { sourceRows := { start := 46436, stop := 46437 }, disposition := "LinearDefinition(SelectiveRewriteId(1839))", emittedStart := none },
    { sourceRows := { start := 46437, stop := 46438 }, disposition := "LinearDefinition(SelectiveRewriteId(1840))", emittedStart := none },
    { sourceRows := { start := 46438, stop := 47038 }, disposition := "Poseidon2(SelectiveRewriteId(74))", emittedStart := none },
    { sourceRows := { start := 47038, stop := 47039 }, disposition := "LinearDefinition(SelectiveRewriteId(1841))", emittedStart := none },
    { sourceRows := { start := 47039, stop := 47040 }, disposition := "LinearDefinition(SelectiveRewriteId(1842))", emittedStart := none },
    { sourceRows := { start := 47040, stop := 47041 }, disposition := "LinearDefinition(SelectiveRewriteId(1843))", emittedStart := none },
    { sourceRows := { start := 47041, stop := 47042 }, disposition := "LinearDefinition(SelectiveRewriteId(1844))", emittedStart := none },
    { sourceRows := { start := 47042, stop := 47642 }, disposition := "Poseidon2(SelectiveRewriteId(75))", emittedStart := none },
    { sourceRows := { start := 47642, stop := 47643 }, disposition := "LinearDefinition(SelectiveRewriteId(1845))", emittedStart := none },
    { sourceRows := { start := 47643, stop := 48243 }, disposition := "Poseidon2(SelectiveRewriteId(76))", emittedStart := none },
    { sourceRows := { start := 48243, stop := 48247 }, disposition := "Retained", emittedStart := some 4772271 },
    { sourceRows := { start := 48247, stop := 48248 }, disposition := "LinearDefinition(SelectiveRewriteId(1846))", emittedStart := none },
    { sourceRows := { start := 48248, stop := 48249 }, disposition := "LinearDefinition(SelectiveRewriteId(1847))", emittedStart := none },
    { sourceRows := { start := 48249, stop := 48250 }, disposition := "LinearDefinition(SelectiveRewriteId(1848))", emittedStart := none },
    { sourceRows := { start := 48250, stop := 48251 }, disposition := "LinearDefinition(SelectiveRewriteId(1849))", emittedStart := none },
    { sourceRows := { start := 48251, stop := 48252 }, disposition := "LinearDefinition(SelectiveRewriteId(1850))", emittedStart := none },
    { sourceRows := { start := 48252, stop := 48253 }, disposition := "LinearDefinition(SelectiveRewriteId(1851))", emittedStart := none },
    { sourceRows := { start := 48253, stop := 48254 }, disposition := "LinearDefinition(SelectiveRewriteId(1852))", emittedStart := none },
    { sourceRows := { start := 48254, stop := 48255 }, disposition := "LinearDefinition(SelectiveRewriteId(1853))", emittedStart := none },
    { sourceRows := { start := 48255, stop := 48256 }, disposition := "LinearDefinition(SelectiveRewriteId(1854))", emittedStart := none },
    { sourceRows := { start := 48256, stop := 48257 }, disposition := "LinearDefinition(SelectiveRewriteId(1855))", emittedStart := none },
    { sourceRows := { start := 48257, stop := 48258 }, disposition := "LinearDefinition(SelectiveRewriteId(1856))", emittedStart := none },
    { sourceRows := { start := 48258, stop := 48259 }, disposition := "LinearDefinition(SelectiveRewriteId(1857))", emittedStart := none },
    { sourceRows := { start := 48259, stop := 48260 }, disposition := "LinearDefinition(SelectiveRewriteId(1858))", emittedStart := none },
    { sourceRows := { start := 48260, stop := 48261 }, disposition := "LinearDefinition(SelectiveRewriteId(1859))", emittedStart := none },
    { sourceRows := { start := 48261, stop := 48262 }, disposition := "LinearDefinition(SelectiveRewriteId(1860))", emittedStart := none },
    { sourceRows := { start := 48262, stop := 48862 }, disposition := "Poseidon2(SelectiveRewriteId(77))", emittedStart := none },
    { sourceRows := { start := 48862, stop := 48863 }, disposition := "LinearDefinition(SelectiveRewriteId(1861))", emittedStart := none },
    { sourceRows := { start := 48863, stop := 48864 }, disposition := "LinearDefinition(SelectiveRewriteId(1862))", emittedStart := none },
    { sourceRows := { start := 48864, stop := 48865 }, disposition := "LinearDefinition(SelectiveRewriteId(1863))", emittedStart := none },
    { sourceRows := { start := 48865, stop := 48866 }, disposition := "LinearDefinition(SelectiveRewriteId(1864))", emittedStart := none },
    { sourceRows := { start := 48866, stop := 49466 }, disposition := "Poseidon2(SelectiveRewriteId(78))", emittedStart := none },
    { sourceRows := { start := 49466, stop := 49467 }, disposition := "LinearDefinition(SelectiveRewriteId(1865))", emittedStart := none },
    { sourceRows := { start := 49467, stop := 49468 }, disposition := "LinearDefinition(SelectiveRewriteId(1866))", emittedStart := none },
    { sourceRows := { start := 49468, stop := 50068 }, disposition := "Poseidon2(SelectiveRewriteId(79))", emittedStart := none },
    { sourceRows := { start := 50068, stop := 50069 }, disposition := "LinearDefinition(SelectiveRewriteId(1867))", emittedStart := none },
    { sourceRows := { start := 50069, stop := 50669 }, disposition := "Poseidon2(SelectiveRewriteId(80))", emittedStart := none },
    { sourceRows := { start := 50669, stop := 50673 }, disposition := "Retained", emittedStart := some 4772275 },
    { sourceRows := { start := 50673, stop := 50674 }, disposition := "LinearDefinition(SelectiveRewriteId(1868))", emittedStart := none },
    { sourceRows := { start := 50674, stop := 50675 }, disposition := "LinearDefinition(SelectiveRewriteId(1869))", emittedStart := none },
    { sourceRows := { start := 50675, stop := 50676 }, disposition := "LinearDefinition(SelectiveRewriteId(1870))", emittedStart := none },
    { sourceRows := { start := 50676, stop := 50677 }, disposition := "LinearDefinition(SelectiveRewriteId(1871))", emittedStart := none },
    { sourceRows := { start := 50677, stop := 50719 }, disposition := "Retained", emittedStart := some 4772279 },
    { sourceRows := { start := 50719, stop := 50720 }, disposition := "LinearDefinition(SelectiveRewriteId(1872))", emittedStart := none },
    { sourceRows := { start := 50720, stop := 50721 }, disposition := "LinearDefinition(SelectiveRewriteId(1873))", emittedStart := none },
    { sourceRows := { start := 50721, stop := 50722 }, disposition := "LinearDefinition(SelectiveRewriteId(1874))", emittedStart := none },
    { sourceRows := { start := 50722, stop := 50723 }, disposition := "LinearDefinition(SelectiveRewriteId(1875))", emittedStart := none },
  ]

def finalRuns : List FinalRun := [
    { family := "Retained", rows := { start := 4772267, stop := 4772271 }, rewriteId := none },
    { family := "Retained", rows := { start := 4772271, stop := 4772275 }, rewriteId := none },
    { family := "Retained", rows := { start := 4772275, stop := 4772279 }, rewriteId := none },
    { family := "Retained", rows := { start := 4772279, stop := 4772321 }, rewriteId := none },
    { family := "Poseidon2", rows := { start := 4779823, stop := 4779909 }, rewriteId := some 57 },
    { family := "Poseidon2", rows := { start := 4779909, stop := 4779995 }, rewriteId := some 58 },
    { family := "Poseidon2", rows := { start := 4779995, stop := 4780081 }, rewriteId := some 59 },
    { family := "Poseidon2", rows := { start := 4780081, stop := 4780167 }, rewriteId := some 60 },
    { family := "Poseidon2", rows := { start := 4780167, stop := 4780253 }, rewriteId := some 61 },
    { family := "Poseidon2", rows := { start := 4780253, stop := 4780339 }, rewriteId := some 62 },
    { family := "Poseidon2", rows := { start := 4780339, stop := 4780425 }, rewriteId := some 63 },
    { family := "Poseidon2", rows := { start := 4780425, stop := 4780511 }, rewriteId := some 64 },
    { family := "Poseidon2", rows := { start := 4780511, stop := 4780597 }, rewriteId := some 65 },
    { family := "Poseidon2", rows := { start := 4780597, stop := 4780683 }, rewriteId := some 66 },
    { family := "Poseidon2", rows := { start := 4780683, stop := 4780769 }, rewriteId := some 67 },
    { family := "Poseidon2", rows := { start := 4780769, stop := 4780855 }, rewriteId := some 68 },
    { family := "Poseidon2", rows := { start := 4780855, stop := 4780941 }, rewriteId := some 69 },
    { family := "Poseidon2", rows := { start := 4780941, stop := 4781027 }, rewriteId := some 70 },
    { family := "Poseidon2", rows := { start := 4781027, stop := 4781113 }, rewriteId := some 71 },
    { family := "Poseidon2", rows := { start := 4781113, stop := 4781199 }, rewriteId := some 72 },
    { family := "Poseidon2", rows := { start := 4781199, stop := 4781285 }, rewriteId := some 73 },
    { family := "Poseidon2", rows := { start := 4781285, stop := 4781371 }, rewriteId := some 74 },
    { family := "Poseidon2", rows := { start := 4781371, stop := 4781457 }, rewriteId := some 75 },
    { family := "Poseidon2", rows := { start := 4781457, stop := 4781543 }, rewriteId := some 76 },
    { family := "Poseidon2", rows := { start := 4781543, stop := 4781629 }, rewriteId := some 77 },
    { family := "Poseidon2", rows := { start := 4781629, stop := 4781715 }, rewriteId := some 78 },
    { family := "Poseidon2", rows := { start := 4781715, stop := 4781801 }, rewriteId := some 79 },
    { family := "Poseidon2", rows := { start := 4781801, stop := 4781887 }, rewriteId := some 80 },
  ]

def rawArtifact : RawArtifact :=
  { schemaVersion := 1,
    profileId := "nightstream/goldilocks/streaming-lifecycle-selective/v1",
    sourceArtifactIdentity := "rust:nightstream/streaming-lifecycle-base/source-rows/v1",
    finalArtifactIdentity := "rust:nightstream/streaming-lifecycle-selective/final-rows/v1",
    stagePath := "fprime.base.verifier_key", occurrence := 7,
    sourceRows := { start := 36133, stop := 50723 }, sourceColumns := { start := 36007, stop := 50543 },
    structureDigestColumns := { start := 36007, stop := 36011 },
    ajtaiPpDigestColumns := { start := 36011, stop := 36015 },
    initialSemanticStateDigestColumns := { start := 36015, stop := 36019 },
    baseVerifierKeyHash := { sourceRows := { start := 36133, stop := 42793 }, recipe := { constantValues := [23, 30521782141150574, 31069335676202596, 13356207430137391, 13430, 1, 4294967295, 81, 54, 18, 1073741824, 0, 2, 16, 65536, 0, 216, 2, 114, 649, 0], constantStartColumn := 36019, localColumns := [36007, 36008, 36009, 36010, 646, 647, 648, 649, 36011, 36012, 36013, 36014, 36015, 36016, 36017, 36018], payloadColumns := [], orderedInputColumns := [36019, 36020, 36021, 36022, 36023, 36007, 36008, 36009, 36010, 646, 647, 648, 649, 36011, 36012, 36013, 36014, 36024, 36025, 36026, 36027, 36028, 36029, 36030, 36031, 36032, 36033, 36034, 36035, 36036, 36037, 36038, 36039, 36015, 36016, 36017, 36018], outputColumns := [42671, 42672, 42673, 42674] } },
    policyVerifierKeyHash := { sourceRows := { start := 42793, stop := 45817 }, recipe := { constantValues := [30, 30521782141150574, 31069335676202596, 26867006312248879, 13362791782838128, 12662, 1, 1, 1], constantStartColumn := 42679, localColumns := [42671, 42672, 42673, 42674], payloadColumns := [], orderedInputColumns := [42679, 42680, 42681, 42682, 42683, 42684, 42671, 42672, 42673, 42674, 42685, 42686, 42687], outputColumns := [45695, 45696, 45697, 45698] } },
    policyDigestBinding := { sourceRows := { start := 45817, stop := 45821 }, leftColumns := [642, 643, 644, 645], rightColumns := [45695, 45696, 45697, 45698] },
    initialBoundaryHash := { sourceRows := { start := 45821, stop := 48243 }, recipe := { constantValues := [34, 30521782141150574, 31069335676202596, 27419021446900015, 28268948330012524, 55483184018017, 649, 0], constantStartColumn := 45703, localColumns := [36007, 36008, 36009, 36010], payloadColumns := [], orderedInputColumns := [45703, 45704, 45705, 45706, 45707, 45708, 36007, 36008, 36009, 36010, 45709, 45710], outputColumns := [48117, 48118, 48119, 48120] } },
    initialBoundaryBinding := { sourceRows := { start := 48243, stop := 48247 }, leftColumns := [652, 653, 654, 655], rightColumns := [48117, 48118, 48119, 48120] },
    finalRowCount := 10306243,
    sourceRuns := sourceRuns,
    finalRuns := finalRuns }

theorem sourceRuns_cover : SourceRunChain 36133 sourceRuns 50723 :=
by
  unfold sourceRuns
  exact SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.cons rfl (by decide)
    (SourceRunChain.nil 50723))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))

theorem finalRuns_inside : FinalRunsWithin 10306243 finalRuns :=
by
  unfold finalRuns
  exact FinalRunsWithin.cons (by decide) (by decide)
    (FinalRunsWithin.cons (by decide) (by decide)
    (FinalRunsWithin.cons (by decide) (by decide)
    (FinalRunsWithin.cons (by decide) (by decide)
    (FinalRunsWithin.cons (by decide) (by decide)
    (FinalRunsWithin.cons (by decide) (by decide)
    (FinalRunsWithin.cons (by decide) (by decide)
    (FinalRunsWithin.cons (by decide) (by decide)
    (FinalRunsWithin.cons (by decide) (by decide)
    (FinalRunsWithin.cons (by decide) (by decide)
    (FinalRunsWithin.cons (by decide) (by decide)
    (FinalRunsWithin.cons (by decide) (by decide)
    (FinalRunsWithin.cons (by decide) (by decide)
    (FinalRunsWithin.cons (by decide) (by decide)
    (FinalRunsWithin.cons (by decide) (by decide)
    (FinalRunsWithin.cons (by decide) (by decide)
    (FinalRunsWithin.cons (by decide) (by decide)
    (FinalRunsWithin.cons (by decide) (by decide)
    (FinalRunsWithin.cons (by decide) (by decide)
    (FinalRunsWithin.cons (by decide) (by decide)
    (FinalRunsWithin.cons (by decide) (by decide)
    (FinalRunsWithin.cons (by decide) (by decide)
    (FinalRunsWithin.cons (by decide) (by decide)
    (FinalRunsWithin.cons (by decide) (by decide)
    (FinalRunsWithin.cons (by decide) (by decide)
    (FinalRunsWithin.cons (by decide) (by decide)
    (FinalRunsWithin.cons (by decide) (by decide)
    (FinalRunsWithin.cons (by decide) (by decide)
    (FinalRunsWithin.nil))))))))))))))))))))))))))))

end Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingLifecycleBaseVerifierKey

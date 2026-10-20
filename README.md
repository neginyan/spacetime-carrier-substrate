[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23040126.svg)](https://doi.org/10.5281/zenodo.23040126)[![Verification](https://github.com/neginyan/spacetime-carrier-substrate/actions/workflows/verify.yml/badge.svg)](https://github.com/neginyan/spacetime-carrier-substrate/actions/workflows/verify.yml)
# The Spacetime Carrier Substrate: Bandlimited Information, Regular Geometry, and a Gravitational Heisenberg Cut

Numerical verification suite for the paper

> T. Namba, *The Spacetime Carrier Substrate: Bandlimited Information,
> Regular Geometry, and a Gravitational Heisenberg Cut*, preprint, Zenodo (2026),
> [doi:10.5281/zenodo.22962656](https://doi.org/10.5281/zenodo.22962656) (all versions; equation numbers here follow version v6).

Every script re-derives a result of the paper **independently** and compares
it with the equation printed in the manuscript. It does not copy the paper's
closed forms into the test. For example, the curvature is not taken from
Eq. (4.38). It is rebuilt from the metric (4.22) through numerically
computed Christoffel symbols and the full Riemann tensor. The interaction
energy of Eq. (7.8) is checked against a 6D Monte Carlo integral over both
branch densities.

```
python verification/run_all.py
```

```
  OK     12/12   Section 2  Carrier window sampling (Eqs. 2.2-2.19)
  OK     25/25   Section 3  Dissipative duality and Krein embedding (Eqs. 3.4-3.30)
  OK     30/30   Sections 4 & 6  Regular geometry, curvature and energy conditions
  OK     15/15   Section 5  Information geometry, volume deficit and counting
  OK     20/20   Section 7  Gravitational back-reaction and the Heisenberg cut
  OK     15/15   Section 8  Observational status (and Eq. 2.20)
  OK     60/60   Revision  Results added in the revised manuscript (v6)

  177/177 checks passed in 5.4 s
```

The full output is in [`results/verification_output.txt`](results/verification_output.txt).
Requirements: Python ≥ 3.10, NumPy, SciPy and Matplotlib (figures only).

> **日本語要約**
> 本リポジトリは、上記論文の各式を独立に再計算し、論文に書かれた式と数値的に照合する検証コードです。論文の閉じた式をそのままテストに書き写すのではなく、たとえば曲率は計量から Christoffel 記号と Riemann テンソルを数値的に組み立て直し、第7節の相互作用エネルギーは6次元モンテカルロ積分で確かめています。`python verification/run_all.py` で全177項目の検証が約5秒で実行されます。改訂版（v6）で新たに加えた結果は `revision_results.py` で確かめています。

---

## What each script checks

| Script | Paper | Independent method | Main results confirmed |
|---|---|---|---|
| `sec2_window_sampling.py` | §2, Eqs. (2.2)–(2.19) | direct numerical integration of the window readout (2.8) | readout of a pure state is **rank one**; closed form (2.17); coherence at the branch nodes independent of $d_0$; only a trans-Planckian momentum filter; weights $\lvert c_k\rvert^2$ |
| `sec3_krein_duality.py` | §3, Eqs. (3.4)–(3.30) | brute-force null space of $\eta K + K^T\eta = 0$; phase-space grid model of Eq. (3.8) | Lemma 2 bound $\mathrm{Re}\,\lambda \le -c_\Delta D_\sigma$; $P = R = 0$ for **every** solution; split signature $(N,N)$; pseudo-unitarity; $\{C,K\} = 0$; mode fraction $1/2$; $\Theta K\Theta^{-1}$ keeps the contractive spectrum (Eq. 3.16); Remark 2 identity and bound |
| `sec4_6_regular_geometry.py` | §4, §6 | Riemann tensor from the metric by finite differences; quadrature | $U_{\rm self} = -\tfrac{3\pi}{32}GM^2/b_0$; $M_{\rm hor} = \tfrac{3\sqrt3}{4}b_0c^2/G$; $R(r)$ (4.38); $K(0)$; Gauss–Bonnet; $\rho, p_r, p_t$ from $G^\mu{}_\nu$; NEC everywhere; SEC violated for $r < \sqrt{2/3}\,b_0$; Gaussian-kernel threshold $\approx 0.95\,\sigma_0c^2/G$ |
| `sec5_information_geometry.py` | §5 | Gauss–Hermite quadrature; exact geodesic-ball volumes on $S^n$, $H^n$ | Fisher metric $\delta_{\mu\nu}/\sigma_0^2$; Bhattacharyya overlap; deficit coefficient $R/(6(n+2))$ = $R/36$ in 4D; $\lambda_{\rm joint} = 1/4$; the role of assumption (A3) |
| `sec7_heisenberg_cut.py` | §7 | radial quadrature **and** 6D Monte Carlo of Eq. (7.6) | $\sigma_{\rm eff}^2 = 2\sigma^2 + \sigma_0^2$ (and $+4s_m^2$ for extended bodies); saturation (7.12); $\tau$ (7.13); $\Delta S = 2Gm^2/c$ independent of $\sigma_{\rm eff}$; small-$\xi$ limit (7.17); $m_H = M_P/\sqrt2 = 15.39\ \mu$g; Fisher–Bhattacharyya correspondence (7.24) |
| `revision_results.py` | results new in the revised manuscript (v6) | null spaces and commutants by linear algebra; ODE integration; matrix sign function; quadrature on grids; Monte Carlo | Theorem 1(a)–(e): Lyapunov part invertible, intertwining solutions = commutant (not unique), Taussky–Zassenhaus form, signature $(N,N)$, maximal neutrality; Example 1: Bateman's $Q_1, Q_2$ and both invariants (3.26) conserved; Corollary 1 via the Riesz projector; de Bruijn identity; Cramér–Rao; Eq. (5.6) curvature $-1/2$; Eq. (5.18) $\nu = 9/4\pi$; Proposition 1 and Postulate P2; $F(x)$ of Eq. (7.15); energy statements (7.30)–(7.32); Table 4; echo delays |
| `sec8_observables.py` | §8, Eq. (2.20) | exact Schwarzschild tortoise coordinate; root finding on the regular metric | Bild et al. (2023): $\Delta S/\hbar \sim 10^{-27}$; echo delay ≈ 0.1 s for $30\,M_\odot$; photon-sphere and shadow coefficients $-10/9$, $-2/3$; sub-mm null result; CSL heating units |

### The checks can fail

The suite was mutation-tested: re-introducing an incorrect coefficient in
Eq. (4.38) or (6.11), a wrong SEC radius, a spurious sampling suppression in
Eq. (2.17), or a 5 % error in $\sigma_{\rm eff}$ each makes the corresponding
checks fail.

## Figures

`python verification/make_figures.py` writes the figures to `figures/`.

| | |
|---|---|
| ![](figures/fig1_sampling_preserves_coherence.png) | **Fig. 1** Section 2: the carrier readout keeps the full coherence between separated branches at every separation. |
| ![](figures/fig2_action_saturation.png) | **Fig. 2** Section 7: the action difference saturates only for separations larger than the object. The 16.2 µg acoustic cat state of Bild et al. (2023) lies thirteen orders of magnitude below that regime. |
| ![](figures/fig3_heisenberg_cut.png) | **Fig. 3** Section 7: coherence factor in the saturated regime and the Heisenberg-cut mass $m_H$. |

### Paper versions

Equation numbers in all scripts refer to the **revised manuscript (v6)**, Zenodo version v6 of the preprint.
`revision_results.py` covers the results that are new in that version
(Theorem 1 restated as a characterization, Example 1, Corollary 1 via
bounded forward evolution, the causal-diamond formulation of Section 5.3,
the energy statements of Section 7.7 and the quantitative predictions of
Section 8.2). Writing these checks also caught a sign slip in an
intermediate line of the proof of Proposition 1 (the sign of $\lambda$),
which has been corrected in v6; the final field equation (5.22) was not
affected.

## What is verified and what is assumed

The code checks that the equations of the paper are **mathematically
correct and mutually consistent**. It does not, and cannot, test the
physical hypotheses on which the paper builds. The paper states these as
assumptions, and so does this repository:

| Assumption | Where | Status in this code |
|---|---|---|
| Finite-dimensional, frozen-time regularization with a skew-symmetric Liouville operator | Lemma 2 (§3.4) | Within it, strict contractivity is a theorem (checked in a grid model); the continuum limit is not addressed |
| Plummer or Gaussian form of the carrier impulse response | §4.1, §4.5 | Both kernels checked; coefficients are kernel dependent |
| Lorentzian signature via the fundamental symmetry $J$ | §5.1 | Postulate; not tested |
| Counting assumptions (A1)–(A3) for the area law | §5.5 | The code shows explicitly that the factor 1/4 requires (A3) |
| Coherence time $\tau = \sqrt\pi\,\sigma_{\rm eff}/2c$ | Eq. (7.13) | Stated hypothesis; the rescaling for $\tau = \alpha\sigma_{\rm eff}/c$ is checked |
| Contact condition $\lvert\Delta\theta\rvert = 2r_g$ | Eq. (7.21) | Stated conjecture; the correspondence (7.24) holds under it |
| Volume equilibrium for causal diamonds (Postulate P2) | §5.3 | Postulate; the code checks that it is equivalent to the Einstein equation with an integration-constant $\Lambda$ |
| Metric-branch overlap and unitary joint evolution (Postulate P5) | §7.6 | Postulate; the energy statements (7.30)–(7.32) are checked under it |
| Near-horizon reflection for echoes (Postulate P7) | §8.2.2 | Conditional; only the delay is checked |

## Repository layout

```
verification/
  common.py                      constants (CODATA via SciPy) and the Report class
  sec2_window_sampling.py
  sec3_krein_duality.py
  sec4_6_regular_geometry.py
  sec5_information_geometry.py
  sec7_heisenberg_cut.py
  sec8_observables.py
  revision_results.py            results added in the revised manuscript (v6)
  run_all.py                     runs everything; exit code 0 if all checks pass
  make_figures.py
figures/                         PNG figures used above
results/verification_output.txt  full output of run_all.py
```

## Citation

```bibtex
@misc{namba_carrier_substrate,
  author    = {Namba, Taishi},
  title     = {The Spacetime Carrier Substrate: Bandlimited Information,
               Regular Geometry, and a Gravitational Heisenberg Cut},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.22962656},
  note      = {Preprint}
}

@software{namba_carrier_substrate_code,
  author    = {Namba, Taishi},
  title     = {spacetime-carrier-substrate: numerical verification},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.23040126}
}
```

## Use of AI assistants

This work was carried out with the help of AI assistants: Google Gemini, OpenAI ChatGPT and Anthropic Claude.
Their roles are described in the acknowledgments of the paper. The author directed the research and takes full
responsibility for it. Every equation checked here is re-derived independently by the scripts in this repository.

## License

MIT; see [LICENSE](LICENSE).

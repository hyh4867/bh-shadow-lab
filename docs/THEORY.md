# Theory

This note collects the physics and mathematics used by the pipeline. Equation
numbers in brackets refer to the EHT M87 papers where noted (Paper IV = ApJL 875,
L4; Paper VI = ApJL 875, L6; both 2019).

## 1. Interferometry basics

Two antennas *i* and *j* separated by the baseline vector **B** observe a source at
wavelength λ. The correlator multiplies and time-averages the two voltage
signals. The result, the complex *visibility* V_ij, depends only on the baseline
projected onto the plane of the sky, measured in wavelengths:

    (u, v) = (B_east, B_north)_projected / λ .

As the Earth rotates, the projected baseline changes and each antenna pair traces
an ellipse in the (u, v) plane ("Earth-rotation synthesis"). The set of sampled
points is the uv-coverage (`results/fig_uv_coverage.png`). Because the sky
brightness is real, V(−u, −v) = V(u, v)*, so each measurement also gives the
point reflected through the origin.

A baseline of length ρ = |(u, v)| is sensitive to angular structure of size
~1/ρ radians. For the EHT at λ = 1.3 mm the longest baselines are ~8.5 Gλ, which
corresponds to 1/(8.5×10⁹) rad ≈ 24 μas; the nominal resolution λ/B_max is
of that order.

## 2. The van Cittert–Zernike theorem

For a spatially incoherent source in the far field and a small field of view,
the visibility is the two-dimensional Fourier transform of the sky brightness
I(x, y):

    V(u, v) = ∬ I(x, y) exp[−2πi (u x + v y)] dx dy ,                 (1)

where x and y are angular offsets (radians) toward east and north. Derivation in
outline: the electric field at antenna *i* is the superposition of fields from
all source elements, E_i = ∫ E(s) exp(−2πi s·r_i/λ) ds. Correlating E_i and E_j*
and using incoherence, ⟨E(s)E*(s')⟩ = I(s) δ(s − s'), leaves
⟨E_i E_j*⟩ = ∫ I(s) exp[−2πi s·(r_i − r_j)/λ] ds. Writing (r_i − r_j)/λ in the
sky-plane components (u, v) and s = (x, y) gives (1). The w term (the baseline
component along the line of sight) is negligible for a field of ~100 μas.

**Sign convention.** Eq. (1) with the minus sign is the convention of our
models (`src/bhshadow/models.py`). The released uvfits files were written by
eht-imaging, which uses exp(+2πi(ux + vy)) (function `ftmatrix` in
`ehtim/observing/obs_helpers.py`, version 1.3.2) and reads the files back
without conjugation. We therefore conjugate the data on reading. A conjugation
is the same as rotating the image by 180°. Amplitudes, closure amplitudes and
diameters are unchanged; closure phases flip sign and position angles move by 180°.
A unit test checks our models against a direct pixel Fourier sum and against
ehtim's matrix.

## 3. Fourier transforms of ring-like sources

Let ρ = √(u² + v²), φ_u = atan2(u, v) the position angle of the uv point (east
of north), and use polar sky coordinates x = r sin φ, y = r cos φ (φ east of
north). Then u x + v y = r ρ cos(φ − φ_u).

**Thin ring.** I = δ(r − R)/(2πR), diameter d = 2R, unit flux. Using
∫₀^{2π} exp[−i z cos θ] dθ = 2π J₀(z),

    V(ρ) = J₀(2π R ρ) = J₀(π d ρ).                                       (2)

The first zero of J₀ is at z₁ = 2.4048, so the first null of the visibility
amplitude is at ρ₁ = 2.4048/(π d), or

    d = 2.4048 / (π ρ₁) ≈ 45 μas × (ρ₁ / 3.5 Gλ)⁻¹ ,                    (3)

which is Paper VI eq. (6). The second null is at 5.5201/(π d), a ratio of 2.295
to the first. For a uniform disk the ratio is 1.83 (zeros of J₁). This is why the
location of the second minimum separates rings from disks (Paper VI §3).

**Blurred ring.** Convolving the image with a circular Gaussian of FWHM w
multiplies the visibility by exp[−(π w ρ)²/(4 ln 2)]. The zeros of J₀ do not
move. The blur lowers the amplitude at long baselines, so w and d are partly
degenerate in fits that use amplitude ratios. The mean diameter seen in an image
is smaller than the ridge diameter d₀: d ≈ d₀[1 − w²/(4 ln 2 d₀²)] (Paper VI
eq. 30, from Paper IV App. G).

**m = 1 ring ("m-ring", Johnson et al. 2020).** Let the brightness along the
ring vary as 1 + β₁ e^{iφ} + β₁* e^{−iφ}. The Jacobi–Anger expansion
exp(−i z cos θ) = Σ_k (−i)^k J_k(z) e^{ikθ} together with J₋ₖ = (−1)^k J_k gives

    V = Σ_m β_m J_m(π d ρ) exp[i m (φ_u − π/2)],   β₀ = 1, β₋₁ = β₁*.      (4)

If the brightness is 1 + 2|b| cos(φ − η), with the bright side at position angle η,
then β₁ = |b| e^{−iη}. (An earlier draft of our code used β₁ = |b| e^{+iη}. That
mirrors the image east–west. The pixel-DFT unit test caught it.) Non-negative
brightness needs |b| ≤ 1/2. A symmetric source has real V, so its closure
phases are 0 or π. A non-zero β₁ gives non-trivial closure phases.

**Uniform disk** of diameter D: V = 2 J₁(π D ρ)/(π D ρ).

**Offset-disk crescent** (Kamruddin & Dexter 2013; Paper VI eqs. 39–40): a disk of
radius R_out minus a disk of radius R_in whose centre is shifted by r₀. A shift by
(x₀, y₀) multiplies a component's transform by exp[−2πi(u x₀ + v y₀)].
Paper VI defines the crescent diameter as d̂ = R_out + R_in (eq. 20). We use this
model only as a truth model in the injection tests, so that the fitted model (4)
differs from the true structure.

## 4. Station gains and closure quantities

Each station has a complex gain g_i(t). The measured visibility is
V_ij^obs = g_i g_j* V_ij + noise. At 230 GHz the atmosphere changes the phase of
g_i on a timescale of seconds, so absolute visibility phases cannot be calibrated
(Paper VI §2.1).

**Closure phase.** Around a triangle of stations,

    ψ_ijk = arg(V_ij V_jk V_ki) = φ_ij + φ_jk + φ_ki .                    (5)

Each gain phase appears once with + and once with −, so ψ is independent of
station phase errors (Jennison 1958; Rogers et al. 1974). Its thermal variance
in the high-SNR limit is σ_ψ² = Σ (σ/|V|)² over the three baselines.

**Closure amplitude.** For four stations,

    A_ijkl = |V_ij| |V_kl| / (|V_ik| |V_jl|) ,                             (6)

in which each |g| appears once in the numerator and once in the denominator. We
use ln A, with variance Σ (σ/|V|)² over the four baselines. Closure amplitudes are
independent of total flux and closure phases are independent of a position shift,
so closure-only fits constrain shape, not flux or absolute position.

**Counting.** With N stations observing at once, there are (N−1)(N−2)/2
independent closure phases and N(N−3)/2 independent closure amplitudes (for a
fully connected array). We select a minimal set by sorting all triangles (or
quadrangles) by σ and keeping each one that raises the rank of the set of their
signed baseline vectors. When baselines are missing the count is E − N + 1 for
phases (E = number of baselines), and the greedy rank test handles that case.
We ignore the remaining covariance between members of the minimal set
(diagonal approximation).

**Debiasing.** |V| measured with noise σ is biased upward (Rice distribution).
We use A = √(|V|² − σ²).

## 5. Photon orbits and the shadow

The Schwarzschild metric in units G = c = 1 has, for a photon in the equatorial
plane with conserved energy E and angular momentum L,

    (dr/dλ)² = E² − (L²/r²)(1 − 2M/r) .                                    (7)

The effective potential V_eff(r) = (L²/r²)(1 − 2M/r) has a maximum where
dV_eff/dr = 0:

    d/dr [r⁻² − 2M r⁻³] = −2 r⁻³ + 6M r⁻⁴ = 0  ⇒  r_ph = 3M .              (8)

This is the photon sphere, an unstable circular orbit. A photon from infinity
with impact parameter b = L/E reaches a turning point unless E² > V_eff(r_ph),
that is,

    1/b² > (1/r_ph²)(1 − 2M/r_ph) = (1/9M²)(1/3) = 1/(27 M²) .               (9)

So the critical impact parameter is

    b_c = √27 M = 3√3 GM/c² ≈ 5.196 GM/c² .                               (10)

Rays traced backward from the observer with b < b_c end on the horizon, and rays
with b > b_c escape. For an emitting region that surrounds the black hole, the
image has a central brightness depression (the "shadow") bounded by the lensed
photon ring at radius b_c. In angle, with θ_g = GM/(c² D),

    shadow (photon-ring) diameter = 2 b_c / D = 2√27 θ_g ≈ 10.39 θ_g .        (11)

For a Kerr black hole the boundary shape depends on spin and inclination. Its
mean diameter stays within about 4 % of 2√27 θ_g, giving 9.6–10.4 θ_g (Paper VI
p. 13, below eq. 27).

## 6. From ring diameter to mass

The EHT measures the diameter d of the bright emission ring. This is not the same as the
shadow edge. Emission comes from plasma near, but not exactly on, the photon
orbit, and the measured d depends on the model used to describe the ring.
Paper VI writes d = α θ_g (eq. 27) and calibrates α by fitting the same geometric
model to synthetic data made from GRMHD simulation images of known θ_g. For the
closure-only xs-ring model, α = 11.56 (MAD+SANE), with a theoretical spread of
about 11 % (Paper VI Table 4, p. 13). This spread is the largest term in the EHT
mass error budget. For pure photon-ring emission α would be 9.6–10.4.

The mass then follows from

    M = θ_g D c² / G = (d/α) D c² / G .                                  (12)

Numerically, θ_g = 1 μas at D = 16.8 Mpc corresponds to 1.70 × 10⁹ M☉. We use
D = 16.8 ± 0.8 Mpc (Paper VI Table 7). Dynamical masses also scale as M ∝ D,
so values quoted at 17.9 Mpc are multiplied by 16.8/17.9 before comparison.

**Caveat for this project.** Our primary model is an m = 1 blurred ring, not the
xs-ring. Its d is the ridge diameter of the thin ring before blurring. The
xs-ring d̂ is (R_out + R_in), the mid-line of a crescent of finite width. The two
definitions agree for a thin symmetric ring, but the GRMHD calibration of α was
done for xs-ring only. Applying α = 11.56 to our d is an assumption. We do not
quantify the extra uncertainty it introduces.

## 7. Likelihood

For closure phases we use χ²_ψ = Σ 2[1 − cos(ψ_obs − ψ_mod)]/σ_ψ². This equals the
usual squared residual for small residuals and is periodic in 2π. For log
closure amplitudes, χ²_A = Σ (lnA_obs − lnA_mod)²/σ². ln L = −(χ²_ψ + χ²_A)/2,
with uniform priors (PREREGISTRATION §5). When the best fit has χ²_ν > 1, we scale
the posterior widths by √χ²_ν. This is a standard, approximate way to account
for model misfit or underestimated errors. It is not a substitute for a better
model.

## 8. Stage 5 models (additional analysis)

### 8.1 General m-ring

Take a thin ring of radius r whose brightness along the ring is Σ_k β_k e^{ikφ},
where φ is the position angle (east of north) and β_{−k} = β_k*. Write the image
position as (x, y) = r(sin φ, cos φ) and the baseline as (u, v) = ρ(sin φ_u, cos φ_u).
Then u·x = rρ cos(φ − φ_u). The Jacobi–Anger expansion is
e^{−iz cos θ} = Σ_n (−i)ⁿ J_n(z) e^{inθ}. Integrating over φ keeps only the term
n = −k:

V(u) = Σ_k β_k (−i)^k J_k(2πrρ) e^{ikφ_u} = Σ_k β_k J_k(z) e^{ik(φ_u − π/2)},

using J_{−k} = (−1)^k J_k. The code stores the coefficient of e^{+ikφ} as
|β_k| e^{−i arg_k}. With that convention, the brightness term for each k is
2|β_k| cos(kφ − arg_k). For k = 1, arg_1 = η is the position angle of the
brightest point. For k = 2, arg_2 = 2η₂. A pixel-image test checks both terms
(`tests/test_models_stage5.py`). J₂ is computed from the recurrence
J₂(z) = 2J₁(z)/z − J₀(z). The brightness stays non-negative when |β₁| + |β₂| ≤ 1/2.

### 8.2 Why a very large Gaussian cannot change closure quantities here

Add a Gaussian of flux f and FWHM θ_G to a compact source V_c. At baseline
length ρ the total visibility is V = V_c + f·exp[−(πθ_Gρ)²/(4 ln 2)].

Take θ_G ≥ 1 mas and the shortest baseline we use, SMT–LMT at about 1.3 Gλ
(2017-04-06 hi). The exponent is then larger than 100, so the Gaussian term
vanishes to machine precision. Every closure phase and closure amplitude therefore equals that of
V_c alone.

EHT Paper VI kept the intra-site baselines (about 1.5 Mλ) in its closure
amplitudes, so it needed this component. We drop those baselines, so we do not
fit it.

A moderately extended component is different: it is only partly resolved on our
baselines and does change the closures. This holds for a FWHM of tens of μas.
At 150 μas the factor at 1.3 Gλ is already exp(−3.2) ≈ 0.04. Model E1 tests
this case.

### 8.3 xs-ring Fourier transform (our derivation)

Paper VI (Appendix B.1.1) builds the xs-ring in the image plane:

1. Start from a disk of radius R_out at the origin.
2. Subtract a disk of radius R_in = R_out(1 − ψ) centred at c = −r₀ n̂, where
   r₀ = (1 − τ)ψ R_out. The unit vector n̂ = (sin f, cos f) points to the widest
   part of the crescent.
3. Multiply by a linear "slash" s(t) = s₀ + g t, where t = x·n̂,
   s₀ = (h₁ + h₂)/2 and g = (h₂ − h₁)/(2R_out). We set h₂ = 1 (wide side) and
   h₁ = β.
4. Add a floor disk of brightness K = γβ and radius R_in at c.
5. Blur with a Gaussian of standard deviation σ.

Use the convention V(u) = ∫ I(x) e^{−2πi u·x} d²x. A disk of radius R centred
at c has

D̃(u) = A(ρ) e^{−2πi u·c},  A(ρ) = R J₁(2πRρ)/ρ.

Multiplying the image by t = x·n̂ corresponds to (i/2π) n̂·∇_u in the Fourier
domain. Since ∇_u ρ = û, this gives

(t·D)~(u) = [(i/2π) A′(ρ)(n̂·û) + (n̂·c) A(ρ)] e^{−2πi u·c},

with A′(ρ) = R[kρ J₀(kρ) − 2J₁(kρ)]/ρ² and k = 2πR.

The model is

V = [s₀ D̃_out + g (t·D_out)~ − s₀ D̃_in − g (t·D_in)~ + K D̃_in] e^{−2π²σ²ρ²} / V₀.

The slash is linear, so its average over a disk equals its value at the centre.
That gives the total flux

V₀ = πR_out² s₀ − πR_in²(s₀ − g r₀) + Kπ R_in².

This expression matches Paper VI's formula for V₀ term by term
(`test_xsring_flux_and_wide_side`). Direct Fourier sums of pixel images agree
with the analytic V to < 8 × 10⁻³ of the total flux for three parameter sets
(`test_xsring_matches_dft`). The ring diameter is d̂ = R_out + R_in (Paper VI
eq. 20).

### 8.4 Jackknife

For n leave-one-out estimates d₍ᵢ₎ with mean d̄, the jackknife standard error is
σ = √[(n − 1)/n · Σ(d₍ᵢ₎ − d̄)²].

We use it to turn the spread over "leave one telescope out" fits (n = 7) and
"leave one day out" fits (n = 4) into systematic terms.

These terms are not independent of each other or of the statistical error.
Adding them in quadrature is a rough, conservative choice, fixed in
PREREGISTRATION §10.6.

## 9. Stage 6 (additional analysis)

### 9.1 Elliptical Gaussian

A unit-flux Gaussian centred at c, with standard deviations s_a along the major
axis (unit vector â at position angle θ east of north) and s_b = q·s_a along the
minor axis, has

G̃(u) = exp[−2π²(s_a² (u·â)² + s_b² (u·b̂)²)] e^{−2πi u·c},

with â = (sin θ, cos θ) and b̂ = (cos θ, −sin θ) in (east, north) components.
The FWHM is 2√(2 ln 2) s. For q = 1 this reduces to the circular Gaussian of
§8 (`test_egauss_circular_limit_equals_gauss_offset`). Direct Fourier sums of
pixel images agree to < 2 × 10⁻³ (`test_egauss_matches_dft`).

A ring with n nuisance Gaussians is (V_ring + Σ fᵢ G̃ᵢ)/(1 + Σ fᵢ), so the total
flux stays 1 and fᵢ is the flux of Gaussian i relative to the ring.

### 9.2 Nested sampling and evidence

Nested sampling (Skilling 2004; dynesty, Speagle 2020) estimates the evidence

Z = ∫ L(θ) π(θ) dθ

by shrinking a set of "live" points drawn from the prior π towards higher
likelihood. A by-product is a weighted posterior sample.

Two models are compared by ln Z₁ − ln Z₂ (the log Bayes factor). Unlike BIC,
this uses the full prior volume, so a parameter the data do not constrain costs
less than k ln N / 2.

Our likelihood has χ²_ν ≈ 3–7, i.e. the closure errors are too small for our
models. Every log-likelihood difference, and with it ln Z, is then inflated by
roughly that factor. We therefore also report ΔlnZ / χ²_ν as a rough guide.

Two prior details keep the evidence correctly normalised:

* Ordering x₁ ≤ x₂ ≤ … of Gaussian positions: sorting n uniform draws gives the
  uniform density on the ordered region, n!/(range)ⁿ. This removes the n!
  identical copies of each mode (label switching) without changing the model.
* Positivity |β₁| + |β₂| ≤ 0.5 for the m=2 ring: folding the unit square along
  its anti-diagonal ((a, b) → (1 − a, 1 − b) when a + b > 1) maps it two-to-one
  onto the triangle with uniform density.

### 9.3 Convergence: R̂ over independent runs

Each nested run starts from fresh prior draws, so three runs with different
seeds play the role of three starting points.

We treat the equal-weight posterior samples of run j (n draws) as chain j and
compute the Gelman–Rubin statistic

R̂ = √[((n − 1)/n · W + B/n) / W],

where W is the mean within-run variance and B/n is the variance of the run
means. If one run finds a different mode, B grows and R̂ > 1.

We also start emcee from the pooled nested posterior and measure the integrated
autocorrelation time τ. Because these walkers already start in the typical set,
a failure here points to a hard posterior geometry, not to a poor start.

### 9.4 Co-located quadrangles

Take stations a, b at the same site (ALMA and APEX, or JCMT and SMA), so
u_{ac} ≈ u_{bc} for every other station c. Then a log closure amplitude
ln(|V_{ac}| |V_{bd}| / (|V_{ad}| |V_{bc}|)) ≈ 0 for any source structure.

Its measured value carries almost no information on the source. With a small σ
(high SNR at ALMA), however, it can still dominate the χ² if there are small
non-closing errors between the two co-located stations.

Removing every quadrangle that contains both members of a pair, and then
choosing the minimal set again, tests whether these quadrangles drive the
result. The rank of the set need not drop, because other quadrangles can take
their place (`test_colocated_quadrangles_removed`).

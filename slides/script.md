# Presentation script: "Black–Scholes is the Heat Equation in Disguise"

**Speakers:** Harshil Goyal (slides 1–17) · Amogh Bindal (slides 18–25)
**Deck:** `slides/slides.pdf` (25 slides, Madrid theme)
**Target length:** about 20 minutes (Harshil about 14, Amogh about 6). For a shorter slot, see *Short version* at the end.

## How the work is split

| | Harshil | Amogh |
|---|---|---|
| Slides | 1–17 (17 slides) | 18–25 (8 slides) |
| PDE slides | 11: problem statement, classification, the three substitutions, separation of variables on the line, Fourier integral, evaluating *I*, scaling derivation | 5: kernel properties, smoothing, uniqueness, completing the square, the formula |
| Intro / wrap-up | 6: title, motivation, outline, call option, share-price model, delta hedging | 3: limitations, conclusions, references |
| Proofs presented | Reduction to the heat equation (all three steps); derivation of the Fourier–Poisson integral; ODE for K(η); similarity solution | Uniqueness (maximum principle and energy); Black–Scholes formula from the kernel |

There is one handover, at slide 17 → 18. With the applications removed, Part 1 is longer; to even it out, move the handover to slide 13 → 14 (Amogh then also presents the separation of variables, the Fourier integral, evaluating *I* and the scaling derivation).

Times in brackets are targets. Text in *italics* is a stage direction, not spoken.

---

# Part 1: Harshil (slides 1–17, about 14 min)

### Slide 1: Title  [0:30]
Good morning everyone. I'm Harshil, and this is Amogh. Our project shows that the Black–Scholes equation, which prices financial options, is exactly the heat equation after a change of variables. We'll derive the equation, transform it, solve it with the methods from our course, and end with the model's limitations.

### Slide 2: Motivation: why this equation?  [0:45]
Why did we pick this equation? The honest answer is money; money motivates everyone, PDE students included. Options are traded in huge volumes every day, someone has to put a price on them, and this formula won the 1997 Nobel Prize in Economics. *Pause for the laugh on "pay rent".*
The mathematician's answer: your portfolio obeys the same equation as a cooling cup of chai, and every tool from our course shows up along the way.
*Point to the red box.* A warning before we start: in 1998 the hedge fund LTCM, with both Nobel laureates on board, lost billions. The maths was fine; the assumptions were not. We come back to that at the end.

### Slide 3: Outline  [0:20]
There are four parts. First, where the Black–Scholes equation comes from. Second, its classification and the reduction to the heat equation. Third, solving the heat equation, with proofs. Fourth, limitations. I'll take the first two parts and the start of the third; Amogh will finish the theory and cover the limitations.

### Slide 4: What is a European call option?  [0:50]
A European call option is a contract that gives you the right, but not the obligation, to buy one share at a fixed price K, the strike, on a fixed date T, the expiry.
At expiry the decision is easy. If the share price S is above K, you exercise and gain S minus K; otherwise you walk away. So the value at expiry is max(S − K, 0). *Point to the formula.*
The question is what the option is worth *today*. *Point to the paths.* These are simulated share prices: the blue paths finish above K and pay off, the grey ones don't. Since we don't know which path we'll get, we need a model. Throughout, K = 100, r = 5%, σ = 25%, and T = 1 year.

### Slide 5: Model for the share price  [0:50]
We model the share price as geometric Brownian motion: dS = μS dt + σS dW. Here μ is the average growth rate, σ the volatility and W a Brownian motion. The key rule is (dW)² = dt.
To see how the option value V(S, t) changes, we use Itô's lemma, which we take as given. It is the chain rule for random processes. Because (dS)² = σ²S² dt is first order in dt, the second-derivative term survives. *Point to the last line.* The result is a dt part and a random dW part.

### Slide 6: Derivation: delta hedging  [1:00]
Now the key idea. Hold one option and sell Δ shares; call this portfolio Π = V − ΔS. Its change has a dt part and a dW part, and the dW part is σS(V_S − Δ).
Step one: choose Δ = V_S. The random term vanishes, and so does μ, the unknown growth rate. This is why the price does not depend on anyone's opinion of the share.
Step two: a portfolio with no risk must earn exactly the bank rate r, otherwise there would be free money. So dΠ = rΠ dt.
Step three: equate the two expressions. *Point to the box.* That gives the Black–Scholes equation: V_t + ½σ²S²V_SS + rSV_S − rV = 0.

### Slide 7: The complete problem  [0:40]
Each term has a meaning: time decay, diffusion from the randomness, drift from interest, and discounting.
The data are a terminal condition at t = T, the payoff, plus two boundary conditions. A worthless share gives a worthless option, so V(0, t) = 0, and for very large S the option behaves like S − Ke^{−r(T−t)}.
Two things make this look unfamiliar. The data are given at the *final* time, and the coefficients S² and S vary. Both disappear in the next section.

### Slide 8: Classification of second-order PDEs  [0:45]
Recall the classification from class. For A w_XX + 2B w_XY + C w_YY plus lower-order terms, the sign of the discriminant B² − AC decides the type: hyperbolic like the wave equation, parabolic like the heat equation, or elliptic like Laplace's equation.
Two facts matter. The type does not change under a smooth invertible change of variables, and a parabolic equation has a canonical form with only one second derivative.

### Slide 9: Black–Scholes is parabolic  [0:55]
For Black–Scholes take X = S and Y = t. The only second derivative is ½σ²S² V_SS, so A = ½σ²S², B = 0 and C = 0. The discriminant is zero: the equation is **parabolic** for every S > 0, and the characteristics are the lines t = constant.
At S = 0, A vanishes too, so the equation **degenerates** to V_t = rV. That ODE, with V(0, T) = 0, forces V(0, t) = 0, so the boundary condition comes from the equation itself.
The direction of time matters. V_t and V_SS have the same sign, so the equation is well posed only backwards from expiry. Forwards it is the backward heat equation, which is ill posed. *Point to the last line.* Data of size 1/n give a solution of size e^{n²t}/n, which blows up. Amogh will show this numerically at the end.

### Slide 10: Step 1: logarithmic price  [0:55]
Now the reduction, in three steps. Step one: let x = ln(S/K) and V = K·v. Since dx/dS = 1/S, the chain rule gives V_S = (K/S) v_x, so **S V_S = K v_x**.
Differentiate this identity once more with respect to S. The left side gives V_S + S V_SS and the right side gives (K/S) v_xx. Multiply by S and use the first identity: **S² V_SS = K (v_xx − v_x)**.
*Point to the red box.* The logarithm turns S times d/dS into d/dx, so the variable coefficients disappear.

### Slide 11: Step 2: rescaled time  [0:50]
Step two: let τ = ½σ²(T − t), a scaled time to expiry. Then V_t = −(σ²/2) K v_τ.
Substitute everything into Black–Scholes. *Point to the first equation.* Divide by σ²/2 and set k = 2r/σ². We get v_τ = v_xx + (k − 1) v_x − k v, which has constant coefficients.
Two things happened. Time to expiry is the new time, so the terminal condition at t = T is now an *initial* condition at τ = 0: v(x, 0) = max(eˣ − 1, 0). And four parameters collapsed into one number, k, which is 1.6 for our data.

### Slide 12: Step 3: remove the lower-order terms  [1:00]
Step three removes the first- and zeroth-order terms. Try v = e^{αx+βτ} u. Write E for the exponential and differentiate: *point to the three derivatives.*
Substitute and divide by E, which is positive. The u_x coefficient is 2α + k − 1, and the u coefficient is α² + (k − 1)α − k − β.
Choose α = −(k − 1)/2 to kill the u_x term. Then α² + (k − 1)α − k simplifies to −(k + 1)²/4, so β = −(k + 1)²/4 kills the u term. For our data α = −0.3 and β = −1.69.

### Slide 13: Result: the heat equation  [0:45]
*Point to the proposition.* With S = Keˣ, t = T − 2τ/σ² and V = K e^{αx+βτ} u, the Black–Scholes problem becomes u_τ = u_xx on the whole real line, with initial data u₀(x) = max(e^{(k+1)x/2} − e^{(k−1)x/2}, 0).
Each step can be undone, so the two problems are equivalent: solving one solves the other. The new initial data still have the kink at x = 0 and grow only exponentially. Our test suite checks numerically that the transformed price satisfies the heat equation, with a relative residual below 10⁻⁴.

### Slide 14: Separation of variables on the whole line  [0:55]
Now we solve the heat equation by the method from class. In the notes' notation the heat time is t, the diffusivity is a² = 1, and the initial data are g. For us these are τ, 1 and u₀.
Try u = X(x)T(τ). Then T′/T = X″/X, which must be a constant, −μ². It cannot be positive, because then X would be e^{±κx} and unbounded. So T decays like e^{−μ²τ} and X = A cos μx + B sin μx; higher frequencies decay faster.
On a bounded interval the boundary conditions would pick out discrete frequencies. Here there are none, so every μ ≥ 0 is allowed, and we superpose by *integrating* over μ instead of summing.

### Slide 15: The Fourier integral  [0:50]
At τ = 0 the exponential is 1, so the initial condition says g is a Fourier integral. The coefficients are the continuous versions of the Fourier coefficients: A(μ) = (1/π)∫ g(y) cos μy dy, and B the same with sine.
Substitute these back. The cosine–cosine plus sine–sine combination is cos μ(y − x). For τ > 0 we may swap the order of integration. What remains is one inner integral, I, which does not involve g. One caveat: our u₀ grows, so this derivation is formal; we check the final formula directly at the end.

### Slide 16: Evaluating I by a differential equation  [1:00]
To evaluate I, substitute z = μ√τ and η = (y − x)/√τ. Then I = K(η)/√τ, where K(η) = ∫₀^∞ e^{−z²} cos(ηz) dz.
The trick from class: don't compute K directly, find an ODE for it. Differentiate under the integral, then integrate by parts using z e^{−z²} = −½ d/dz e^{−z²}. The boundary term vanishes, and we get K′ = −(η/2) K.
That separable ODE gives K = C e^{−η²/4}, and K(0) is the Gaussian integral √π/2. Putting the factors together, *point to the box*, u(x, τ) = (1/√(4πτ)) ∫ g(y) e^{−(x−y)²/(4τ)} dy, the Fourier–Poisson integral.

### Slide 17: Second derivation: scaling  [1:00]
The same kernel also comes from symmetry, as in class. If u solves the heat equation, so does λu(λx, λ²τ): space scales like the square root of time, so x²/τ is invariant.
Try u = w(τ) v(z) with z = x²/τ. Differentiating gives the equation on the slide. We make the bracket vanish: 4v″ + v′ = 0 gives v = e^{−z/4}. The remaining terms give w′ = −w/(2τ), so w = τ^{−1/2}: the height falls as the profile spreads. Normalising the total heat to 1 gives the **fundamental solution** Φ(x, τ) = e^{−x²/4τ}/√(4πτ), and the solution is the convolution Φ ∗ g.
**Handover:** So we have a candidate formula from two derivations. Amogh will now prove that it really is the solution, and that it is the only one.

---

# Part 2: Amogh (slides 18–25, about 6 min)

### Slide 18: Properties of the heat kernel  [0:50]
Thanks, Harshil. These are the five properties of the heat kernel from class.
(i) Φ is positive and symmetric. (ii) It is infinitely smooth for τ > 0, with a singularity at the origin. (iii) It solves the heat equation; *point to the derivatives*, Φ_τ and Φ_xx are both Φ times (x²/4τ² − 1/2τ). (iv) Its total mass is 1, by the substitution y = x + 2√τ z and the Gaussian integral. (v) Derivatives and superpositions of Φ are again solutions.
As τ → 0 all the mass concentrates at the origin, so Φ is a unit point source of heat. Positivity also gives infinite speed of propagation: any heat at all makes the solution positive everywhere instantly.

### Slide 19: Smoothing: the kink melts  [0:35]
This is the picture from the start of the talk. *Point to the left panel:* the option value for different times to expiry; at expiry it's the kinked payoff, and immediately afterwards it's smooth. *Point to the right panel:* the same curves in heat coordinates, a kinked temperature profile diffusing in a rod. The price of an option today is a temperature after diffusion.

### Slide 20: Uniqueness  [0:55]
Now uniqueness. On a bounded interval, let w be the difference of two solutions; it is zero on Γ_T. The maximum principle applied to w and −w gives w ≡ 0.
The second proof from class is the energy method. E(τ) = ∫ w² dx is non-negative and starts at 0. Its derivative is 2∫ w w_xx, and integrating by parts gives −2∫ w_x² ≤ 0, because w vanishes at the ends. So E stays at 0, and w ≡ 0.
On the whole line we need a growth condition: among solutions with u ≤ Ae^{ax²}, the maximum principle holds and the solution is unique. Without a growth condition this fails; that is Tychonoff's example. Our solution grows only like e^{c|x|}, so the Black–Scholes price is unique.

### Slide 21: Solving for the call: complete the square  [0:55]
Now we compute the convolution for the call. u₀ is zero for negative s, so the integral runs from 0 to ∞, and u₀ is a difference of two exponentials. So u = I₊ − I₋.
Substitute s = x + √(2τ) y; the Gaussian exponent becomes y²/2. Each integrand is then e^{−½y² + m±y}. Complete the square: −½(y − m±)² + ½m±². The constant comes out of the integral, and what remains is the standard normal distribution function. *Point to the box:* I± = exp(stuff) × N(d±).

### Slide 22: Back to prices: the Black–Scholes formula  [0:55]
Now undo the substitution, V = K e^{αx+βτ}(I₊ − I₋). The exponents combine exactly. For I₊, α + (k + 1)/2 = 1 and β + (k + 1)²/4 = 0, so the prefactor is K eˣ = S. For I₋ the prefactor becomes K e^{−kτ}, and kτ is exactly r(T − t), so it's the discounted strike. Finally √(2τ) = σ√(T − t) turns d± into d₁ and d₂.
*Point to the box.* That is the Black–Scholes formula, derived entirely from the heat kernel. The checks: put–call parity holds to 6 × 10⁻¹⁴, the price goes to 0 as S → 0, and it goes back to the payoff at expiry.

### Slide 23: Limitations and ill-posedness  [0:45]
Limitations, or: ask LTCM. The reduction needs constant volatility and interest rate. Jumps in the share price would give an integro-differential equation, and dividends add an extra term.
The most important limitation is direction: the equation must be solved backwards from expiry. *Point to the left panel.* Here we run the heat equation backwards from sin x plus noise of size one in a million. Each mode is multiplied by e^{n²t}, so after a backward time of only 0.002 the solution has size about 10⁷. *Point to the right panel:* the high modes explode. This is Hadamard ill-posedness, which Harshil showed on slide 9.

### Slide 24: Conclusions  [0:40]
To summarise. Black–Scholes is parabolic, degenerate at S = 0, and a terminal-value problem. Three substitutions turn it into the heat equation. The heat kernel, derived by the Fourier integral or by scaling, solves it, and the maximum principle and the energy method make the solution unique. Completing the square then gives the Black–Scholes formula.

### Slide 25: References  [0:15]
These are our references: the course notes, Evans, Myint-U and Debnath, McOwen, Nandakumaran, the original Black–Scholes paper, and Wilmott and co-authors. Thank you; we're happy to take questions.

---

## Short version (about 18 minutes)

Skip these slides (leave them in the deck as backup for questions):

| Harshil skips | Amogh skips |
|---|---|
| 3 Outline | 18 Kernel properties (mention (iii) and (iv) in one sentence on slide 19 before the smoothing picture) |
| 5 Share-price model (fold the Itô formula into slide 6) | 25 References (end on slide 24) |
| 8 General classification (state B² − AC = 0 on slide 9) | |
| 17 Scaling derivation (say "the same kernel also follows from scaling") | |

Keep both uniqueness proofs on slide 20 and all three reduction steps (10–12); they are the PDE core. Also shorten slides 14–16 (Fourier integral, evaluating *I*) to the key steps.

---

## Likely viva questions

**For Harshil**
- *Why does μ disappear?* Choosing Δ = V_S removes the dW term, and μ only appears together with Δ in the dt term, so it cancels.
- *Why must the constant in the separation be negative?* A positive constant gives X = e^{±κx}, which is unbounded on the whole line.
- *Why is the Fourier-integral derivation only formal for our data?* u₀ grows like e^{(k+1)x/2}, so it is not integrable and A(μ), B(μ) do not exist classically. The final formula is verified directly: it solves the heat equation and tends to u₀ as τ → 0.
- *Why does Step 1 produce v_xx − v_x rather than v_xx?* Differentiating S V_S = K v_x with the product rule produces an extra V_S term, which is K v_x / S.

**For Amogh**
- *Why u − ετ in the maximum principle and not just u?* It makes the inequality strict (−ε < 0), which is what gives the contradiction at an interior maximum.

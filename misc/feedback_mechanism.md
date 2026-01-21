Let \(y^+\) be the **after** text, \(y^-\) the **before** text, and \(m_t\in[0,1]\) a mask/weight for token position \(t\) indicating how much that position is “about the edit” (e.g., 1 on changed tokens + a small decay window around them).

#### 1) Span-weighted *contrastive* score
Define a weighted log-likelihood for any sequence \(y\):

\[
S_\lambda(y) := \sum_{t} m_t \,\log p_\lambda(y_t \mid y_{<t}, P, C)
\]

Then the basic contrastive objective is:

\[
\Delta(\lambda) := S_\lambda(y^+) - S_\lambda(y^-)
\]

This directly uses the edit as a **preference direction** and focuses learning on the **edited span**.

#### 2) Turn it into a probabilistic feedback model (preference likelihood)
Use a Bradley–Terry / logistic preference likelihood:

\[
\Pr(y^+ \succ y^- \mid \lambda) = \sigma(\beta\,\Delta(\lambda))
\]

So the (negative) loss for one edit is:

\[
\mathcal{L}(\lambda) = -\log \sigma(\beta\,\Delta(\lambda))
\]

(You can add length normalization inside \(S_\lambda\) if you want.)

#### 3) Bayesian (online) personalization with uncertainty
Maintain a Gaussian posterior over \(\lambda\) per user:

\[
p(\lambda) \approx \mathcal{N}(\mu, \Sigma)
\]

For each new edit, do a **Laplace / assumed-density filtering** update:

- **MAP step**: find \(\lambda^\*\) that maximizes

\[
\log \sigma(\beta\,\Delta(\lambda)) \;-\; \tfrac12 (\lambda-\mu)^\top \Sigma^{-1} (\lambda-\mu)
\]

- **Covariance update** (local curvature):

\[
\Sigma_{\text{new}}^{-1} \approx \Sigma^{-1} - \nabla^2_{\lambda}\log \sigma(\beta\,\Delta(\lambda))\Big|_{\lambda=\lambda^\*}
\]

Intuition: each edit contributes “evidence” (Fisher/curvature) that shrinks uncertainty in directions the feedback constrains.

---

### Why this meshes especially well with CoS
For fixed text, \(\log p_\lambda(y_t)\) has clean derivatives because CoS is log-softmax of something **linear in \(\lambda\)**. So \(\nabla \Delta(\lambda)\) is easy to compute from the same quantities you already compute for logprob scoring (influences \(b_{t,k}(v)\)), and the Bayesian update just needs gradients/Hessians (or approximations).

---

### Practical note (important)
- The plain “after-only” likelihood is concave in \(\lambda\); the **logistic preference wrapper** introduces mild non-concavity, but in 2D (\(\lambda_a,\lambda_b\)) it’s typically very well-behaved and easy to optimize with a few Newton/gradient steps.
- If you want to preserve concavity strictly, you can skip the logistic and do **Gaussian** observation noise on \(\Delta(\lambda)\) (treat \(\Delta\) as a regression target), but the logistic model matches “preference” semantics better.

---

### Adding more steering dimensions later (new contexts / sliders)
This mechanism generalizes directly from 1–2 dimensions to \(K\) dimensions: \(\lambda\in\mathbb{R}^K\) simply grows with one coordinate per steering direction/context, and the same span-weighted contrastive preference likelihood applies.

#### Extending a per-user Bayesian posterior when a new dimension is introduced
If you maintain a Gaussian posterior \(p(\lambda)\approx \mathcal{N}(\mu,\Sigma)\) for a user and add a new dimension \(\lambda_{K+1}\), you can “grow” the posterior without disturbing previously learned dimensions:

\[
\mu'=\begin{bmatrix}\mu\\ \mu_{\text{new}}\end{bmatrix},\qquad
\Sigma'=\begin{bmatrix}\Sigma & 0\\ 0 & \sigma_{\text{new}}^2\end{bmatrix}
\]

Common defaults:
- \(\mu_{\text{new}}=0\) (neutral steering) and \(\sigma_{\text{new}}^2\) large (high initial uncertainty).
- Optionally allow non-zero prior mean if you have a sensible global default for that new slider.

After this, future edits will update all dimensions jointly via the same objective; the new dimension will begin learning as soon as feedback includes contexts that activate it.

#### Practical scaling / identifiability considerations as \(K\) grows
- **Covariance cost**: a full \(\Sigma\in\mathbb{R}^{K\times K}\) is \(O(K^2)\) to store/update. For small \(K\) this is fine; for many dimensions consider a **diagonal** or **low-rank + diagonal** approximation.
- **Correlated dimensions**: if two steering directions induce similar influence signals \(b_{t,k}(v)\), the data may not strongly identify them separately. This typically appears as strong posterior correlations / persistent uncertainty; regularization (stronger prior), constraints, or designing more orthogonal dimensions can help.

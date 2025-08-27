PATCH MODE: Provide only minimal LaTeX diffs (insertions/replacements) with a one-line rationale; no full rewrites unless requested.

Accepted forms:
• REPLACE: “<exact old text>” → “<new text>”
• INSERT AFTER: “<anchor text>” → “<new text>”
• DELETE: “<exact text>”

Example —
REPLACE: “We fit per-galaxy L_i and μ_i.” → “We fit global (L, μ) across the sample.”
Rationale: enforce Single-Field, two-parameter claim.

Example —
INSERT AFTER: “via Poisson:” → “ρ_eff ≡ ∇^2 Φ_{\rm eff}/(4πG) → Σ → ΔΣ.”
Rationale: keep Poisson→Σ→ΔΣ mapping explicit.

Constraints:
• Keep patches local (≤6–10 lines context).
• Preserve en dashes and citation style.
• Never introduce 1/r kernels or galaxy-specific free parameters.

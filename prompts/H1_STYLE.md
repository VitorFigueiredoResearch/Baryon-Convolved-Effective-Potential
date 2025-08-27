1) Use en dashes for linked terms (Tully–Fisher, mass–concentration, galaxy–galaxy).
2) Citations: natbib with \citet and \citep; parenthetical ranges as “e.g., \citep{…}”.
3) Units with siunitx: \SI{200}{\kilo\metre\per\second}, \si{\kilo\parsec}; keep spaces non-breaking.
4) Numbers with uncertainty: \SI{0.50(10)}{} or x = 0.50 \pm 0.10 (dimensionless).
5) Equations: define symbols in words on first use; check dimensional consistency explicitly.
6) Figures: concise, Planck-like captions; reference as “Fig. 2” and sections as “\S3.1”.
7) Tables: booktabs; no vertical rules; align units in headers.
8) Typography: vectors bold (\mathbf{v}), scalars upright; constants (G, c) upright.
9) Terminology: “Single-Field” capitalized once, then plain; keep Poisson→Σ→ΔΣ mapping verbatim.
10) Macro for highlight boxes (preamble):

\newcommand{\keybox}[1]{%
  \noindent\begin{tcolorbox}[colback=black!3,colframe=black!35,sharp corners]
  \raggedright #1
  \end{tcolorbox}}

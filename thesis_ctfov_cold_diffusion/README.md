# Master's Thesis: Cold Diffusion for CT Sinogram Truncation Artifact Recovery

This directory contains the LaTeX source files for the Master's thesis on CT truncation recovery using Cold Diffusion.

## Directory Structure

```
thesis_ct_diffusion/
├── thesis_main.tex              # Main thesis file
├── abbreviations.tex            # List of abbreviations
├── literature.bib               # Bibliography database
├── introduction.tex             # Chapter 1: Introduction
├── background.tex               # Chapter 2: Background
├── related_work.tex             # Chapter 3: Related Work
├── methodology.tex              # Chapter 4: Methodology
├── implementation.tex           # Chapter 5: Implementation
├── experiments.tex              # Chapter 6: Experiments
├── results.tex                  # Chapter 7: Results (TO BE COMPLETED)
├── discussion.tex               # Chapter 8: Discussion
├── conclusion.tex               # Chapter 9: Conclusion
├── appendix_a_data_processing.tex    # Appendix A
├── appendix_b_model_details.tex      # Appendix B
├── appendix_c_additional_results.tex # Appendix C
└── README.md                    # This file
```

## Important Notes

### Placeholders for Experimental Results

**⚠️ WARNING**: Several sections are marked in red with `\textcolor{red}{[TO BE COMPLETED]}` indicating that experimental results are pending:

- **Chapter 7 (Results)**: Complete chapter waiting for experiment completion
- **Specific sections in other chapters**: Various quantitative results, figures, and tables

These placeholders should be filled in once the experiments in `/home/miaqx52/projects/experiments_clean` complete.

### Required Information to Fill In

Before final submission, update the following in `thesis_main.tex`:

```latex
\Name{[Your Last Name]}
\Vorname{[Your First Name]}
\Geburtsort{[Your Place of Birth]}
\Geburtsdatum{[Your Date of Birth]}
\Betreuer{[Your Advisor's Name]}
\Start{[Start Date]}
\Ende{[End Date]}
```

Also update the signature section with actual place and date.

## Compilation Instructions

### Prerequisites

1. **LaTeX Distribution**: Install TeX Live (Linux), MiKTeX (Windows), or MacTeX (macOS)
2. **Required Packages**: The template requires various LaTeX packages (automatically handled by most distributions)
3. **Biber**: For bibliography processing (usually included with LaTeX distribution)

### Compilation Steps

#### Option 1: Using `pdflatex` and `biber` (Recommended)

```bash
cd /home/miaqx52/projects/thesis-templates-master/thesis-master/thesis_ct_diffusion

# First compilation
pdflatex thesis_main.tex

# Process bibliography
biber thesis_main

# Process glossary/abbreviations
makeglossaries thesis_main

# Final compilations (run twice for cross-references)
pdflatex thesis_main.tex
pdflatex thesis_main.tex
```

#### Option 2: Using `latexmk` (Automated)

```bash
latexmk -pdf -bibtex thesis_main.tex
```

#### Option 3: Using VS Code LaTeX Workshop

If using VS Code with LaTeX Workshop extension:

1. Open `thesis_main.tex`
2. Press `Ctrl+Alt+B` (or Cmd+Option+B on Mac)
3. Or click "Build LaTeX project" in the sidebar

The extension will automatically run the necessary compilation steps.

### Compilation on Overleaf

If you prefer to use Overleaf:

1. Upload all `.tex` files and `literature.bib` to a new Overleaf project
2. Ensure the main document is set to `thesis_main.tex`
3. Set the compiler to `pdfLaTeX`
4. Click "Recompile"

## Document Class

This thesis uses the custom `lmedoc` document class from the template:

```latex
\documentclass[english,mt]{../lmedoc/lmedoc}
```

Options:
- `english`: English language (alternative: `german`)
- `mt`: Master's thesis (alternatives: `bt` for Bachelor's, `diss` for Dissertation)

## Key Features

### Chapters

1. **Introduction**: Motivation, problem statement, contributions, thesis structure
2. **Background**: CT imaging fundamentals, truncation artifacts, diffusion models
3. **Related Work**: Literature review of truncation correction and diffusion models
4. **Methodology**: Core Cold Diffusion approach, forward/reverse processes, training
5. **Implementation**: Data processing, architecture details, training procedures
6. **Experiments**: Experimental design, ablation studies, evaluation protocols
7. **Results**: Quantitative and qualitative results (**TO BE COMPLETED**)
8. **Discussion**: Interpretation, limitations, comparisons, future work
9. **Conclusion**: Summary, key findings, broader impact

### Appendices

- **Appendix A**: Detailed data processing procedures
- **Appendix B**: Complete model architecture specifications
- **Appendix C**: Supplementary results and visualizations (**TO BE COMPLETED**)

### Bibliography

References are managed using BibLaTeX/Biber. The bibliography file `literature.bib` contains entries for:

- CT imaging fundamentals
- Truncation correction methods
- Deep learning for medical imaging
- Diffusion models (DDPM, Cold Diffusion)
- Datasets (CT-ORG, LIDC-IDRI)
- Evaluation metrics

**Note**: All references are to real, published works. No fictional citations.

## Figures

Currently, figure placeholders are included with red notes like:

```latex
\fbox{\parbox{0.8\textwidth}{\centering 
    [FIGURE: Description]\\ 
    \textcolor{red}{To be added}
}}
```

Replace these with actual figures once available. Recommended figure formats:
- Vector graphics: PDF or EPS
- Raster images: PNG (300+ DPI for print quality)

Place figures in a `figures/` subdirectory and reference them as:

```latex
\includegraphics[width=0.8\textwidth]{figures/filename.pdf}
```

## Tables

Tables use the `booktabs` package for professional formatting. Example:

```latex
\begin{table}[htbp]
\centering
\caption{Your caption here}
\label{tab:yourlabel}
\begin{tabular}{lrr}
\toprule
\textbf{Header1} & \textbf{Header2} & \textbf{Header3} \\
\midrule
Row 1 & Data & Data \\
Row 2 & Data & Data \\
\bottomrule
\end{tabular}
\end{table}
```

## Algorithms

Algorithms use the `algorithm` and `algorithmic` packages:

```latex
\begin{algorithm}[htbp]
\caption{Algorithm Name}
\label{alg:label}
\begin{algorithmic}[1]
\REQUIRE Input
\ENSURE Output
\STATE Step 1
\FOR{condition}
    \STATE Step 2
\ENDFOR
\end{algorithmic}
\end{algorithm}
```

## Mathematical Notation

Equations are numbered and can be referenced:

```latex
\begin{equation}
E = mc^2
\label{eq:einstein}
\end{equation}

As shown in Equation~\ref{eq:einstein}...
```

## Cross-References

Use `\ref` and `\label` for cross-referencing:

```latex
\chapter{Introduction}
\label{ch:introduction}

...as discussed in Chapter~\ref{ch:introduction}...
```

## Abbreviations

Abbreviations are defined in `abbreviations.tex` using the `glossaries` package:

```latex
\newacronym{ct}{CT}{Computed Tomography}
```

Use in text: `\gls{ct}` (expands on first use, abbreviated thereafter)

## Status Summary

### Completed

- ✅ Main document structure
- ✅ All chapter outlines and content
- ✅ Bibliography with real references
- ✅ Appendices with technical details
- ✅ Abbreviations and notation
- ✅ Algorithm pseudocode
- ✅ Mathematical formulations

### Pending (After Experiments Complete)

- ⏳ Chapter 7 (Results) - all quantitative results
- ⏳ Experimental figures and plots
- ⏳ Results tables with actual metrics
- ⏳ Statistical significance tests
- ⏳ Visual quality comparisons
- ⏳ Supplementary results in Appendix C
- ⏳ Final proofreading and formatting

## Page Count Estimate

Current status: ~70-75 pages of content (excluding pending results)

With completed results and figures: Expected ~75-85 pages total

## Troubleshooting

### Common Issues

1. **Bibliography not showing**: Make sure to run `biber thesis_main` after `pdflatex`
2. **Glossary not appearing**: Run `makeglossaries thesis_main`
3. **Cross-references showing ??**: Compile multiple times (at least twice)
4. **Missing packages**: Install via your LaTeX distribution's package manager

### Missing Class File Error

If you get an error about missing `lmedoc.cls`:

The document expects the class file at `../lmedoc/lmedoc.cls`. Ensure you're compiling from the correct directory and the template files are in place.

## Contact

For questions about the thesis content or experimental setup, refer to the code repository at:

```
/home/miaqx52/projects/experiments_clean
```

## License

This thesis uses the university template from the Chair for Lightweight Engineering and Design (LME). Please adhere to university guidelines for thesis submission.

---

**Last Updated**: February 2026

**Document Version**: Draft v1.0 (Awaiting experimental results)

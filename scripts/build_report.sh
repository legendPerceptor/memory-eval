#!/usr/bin/env bash
# Build from any working directory. Publish only a successfully compiled PDF.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PATH="$HOME/.local/bin:$PATH"
TEX_DIR="$ROOT/latex"
BUILD_DIR="$TEX_DIR/build"

case "${1:-}" in
  --help|-h)
    echo "Usage: bash scripts/build_report.sh"
    echo "Output: latex/report.pdf; diagnostics: latex/build/"
    echo "Requires latexmk, XeLaTeX, ctex/xeCJK and Fandol fonts."
    exit 0 ;;
  "") ;;
  *) echo "Unknown argument: $1" >&2; exit 2 ;;
esac

missing=()
for tool in latexmk xelatex kpsewhich flock; do
  command -v "$tool" >/dev/null 2>&1 || missing+=("$tool")
done
if command -v kpsewhich >/dev/null 2>&1; then
  for file in ctexart.cls xeCJK.sty FandolSong-Regular.otf; do
    [[ -n "$(kpsewhich "$file")" ]] || missing+=("$file")
  done
fi
if ((${#missing[@]})); then
  printf 'Missing LaTeX dependencies: %s\n' "${missing[*]}" >&2
  echo "Install texlive-xetex and texlive-lang-chinese, or run:" >&2
  echo "  bash $ROOT/scripts/setup_latex_user.sh" >&2
  exit 1
fi

mkdir -p "$BUILD_DIR"
exec 9>"$BUILD_DIR/.build.lock"
flock 9
cd "$TEX_DIR"
{
  echo "Build started: $(date -Is)"
  echo "Engine: $(command -v xelatex)"
  xelatex --version | head -n 2
  if [[ -f results.tex ]]; then
    echo "Document scope: historical baseline numerical section is present."
  else
    echo "Document scope: interim report; final five-video scores are unavailable."
  fi
  if [[ -f context_optimization_results.tex ]]; then
    if grep -q 'CONTEXT_REPORT_STATUS: final' context_optimization_results.tex; then
      echo "Context ablation section: final joint judging."
    else
      echo "Context ablation section: provisional; joint judging is not final."
    fi
  fi
  # Force a build: a newly created results.tex must replace the draft even if
  # latexmk previously recorded the file as absent inside IfFileExists.
  latexmk -norc -g -xelatex -interaction=nonstopmode -halt-on-error \
    -file-line-error -no-shell-escape -outdir="$BUILD_DIR" report.tex
} 2>&1 | tee "$BUILD_DIR/build.log"

test -s "$BUILD_DIR/report.pdf"
[[ "$(head -c 5 "$BUILD_DIR/report.pdf")" == '%PDF-' ]]
if command -v pdfinfo >/dev/null 2>&1; then
  pdfinfo "$BUILD_DIR/report.pdf" > "$BUILD_DIR/pdfinfo.txt"
fi
tmp_pdf="$(mktemp "$TEX_DIR/.report.XXXXXX.pdf")"
trap 'rm -f "$tmp_pdf"' EXIT
cp "$BUILD_DIR/report.pdf" "$tmp_pdf"
chmod 644 "$tmp_pdf"
mv -f "$tmp_pdf" "$TEX_DIR/report.pdf"
{
  echo "Built at: $(date -Is)"
  sha256sum report.tex report.pdf
  for file in results.tex interim_results.tex findings.tex m3_agent_comparison.tex recovery_results.tex matched_comparison_results.tex context_optimization_results.tex memory_design_results.tex figures/context_cache.pdf; do
    [[ ! -f "$file" ]] || sha256sum "$file"
  done
} > "$BUILD_DIR/build-manifest.txt"
echo "PDF successfully published: $TEX_DIR/report.pdf"

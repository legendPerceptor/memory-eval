#!/usr/bin/env bash
# Optional Ubuntu/Debian bootstrap. No sudo, pip changes, or system package install.
# Uses the distribution's TeX packages to supplement an existing TeX installation.
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
CACHE="$HOME/.cache/memory-eval-tex"
TEXMFHOME="$(kpsewhich -var-value=TEXMFHOME)"
for tool in apt-get dpkg-deb xetex fmtutil-user mktexlsr; do
  command -v "$tool" >/dev/null 2>&1 || { echo "Required base tool missing: $tool" >&2; exit 1; }
done
[[ -n "$TEXMFHOME" && "$TEXMFHOME" == "$HOME/"* ]] || {
  echo "Refusing to modify non-user TEXMFHOME: $TEXMFHOME" >&2; exit 1;
}
mkdir -p "$CACHE/packages" "$CACHE/extracted" "$TEXMFHOME" "$HOME/.local/bin"
cd "$CACHE/packages"
apt-get download texlive-xetex texlive-lang-chinese texlive-lang-cjk
for package in ./*.deb; do
  dpkg-deb -x "$package" "$CACHE/extracted"
done
# Preserve pre-existing user TeX files. This is a bootstrap, not an updater.
cp -an "$CACHE/extracted/usr/share/texlive/texmf-dist/." "$TEXMFHOME/"
if [[ ! -e "$HOME/.local/bin/xelatex" && ! -L "$HOME/.local/bin/xelatex" ]]; then
  ln -s "$(command -v xetex)" "$HOME/.local/bin/xelatex"
fi
mktexlsr "$TEXMFHOME"
fmtutil-user \
  --cnffile "$CACHE/extracted/var/lib/tex-common/fmtutil-cnf/texlive/texlive-xetex.cnf" \
  --byfmt xelatex
echo "User TeX setup complete: $TEXMFHOME"
echo "After system TeX updates, regenerate the user xelatex format with fmtutil-user."

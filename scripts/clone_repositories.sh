#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

clone_repo() {
    local name="$1" url="$2" target="$root/$1"
    if [[ -e "$target" || -L "$target" ]]; then
        if [[ -d "$target/.git" || -f "$target/.git" ]]; then
            echo "Skipping existing repository: $target"
            return
        fi
        echo "Refusing to overwrite existing path: $target" >&2
        return 1
    fi
    echo "Cloning $name into $target"
    git clone -- "$url" "$target"
}

clone_repo deploy-llm-on-gpu "${DEPLOY_LLM_REPO_URL:-git@github.com:legendPerceptor/deploy-llm-on-gpu.git}"
clone_repo oGMemory "${OGMEMORY_REPO_URL:-https://gitcode.com/opengauss/oGMemory.git}"
clone_repo locomo-test "${LOCOMO_REPO_URL:-git@github.com:legendPerceptor/locomo-test.git}"
clone_repo m3-agent "${M3_AGENT_REPO_URL:-https://github.com/ByteDance-Seed/m3-agent.git}"

#!/bin/bash

# Pull main branch for all Jido ecosystem repositories
# Clones repos into packages/ if they don't exist, otherwise pulls latest

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGES_DIR="$SCRIPT_DIR/packages"

REPOS=(
  "agentjido/jido"
  "agentjido/jido_action"
  "agentjido/jido_ai"
  "agentjido/jido_signal"
  "agentjido/jido_workbench"
  "agentjido/req_llm"
  "kreuzberg-dev/kreuzberg"
)

# Create packages directory if it doesn't exist
mkdir -p "$PACKAGES_DIR"

for entry in "${REPOS[@]}"; do
  # Extract org and repo from "org/repo" format
  org="${entry%/*}"
  repo="${entry##*/}"
  repo_path="$PACKAGES_DIR/$repo"
  if [ -d "$repo_path" ]; then
    echo "Pulling $repo..."
    cd "$repo_path" && git pull origin main
    echo ""
  else
    echo "Cloning $repo from $org..."
    git clone "https://github.com/$org/$repo.git" "$repo_path"
    echo ""
  fi
done

echo "Done!"

#!/bin/bash

# Pull main branch for all Jido ecosystem repositories
# Clones repos into packages/ if they don't exist, otherwise pulls latest

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGES_DIR="$SCRIPT_DIR/packages"

REPOS=(
  "jido"
  "jido_action"
  "jido_ai"
  "jido_signal"
  "jido_workbench"
  "req_llm"
)

GITHUB_ORG="agentjido"

# Create packages directory if it doesn't exist
mkdir -p "$PACKAGES_DIR"

for repo in "${REPOS[@]}"; do
  repo_path="$PACKAGES_DIR/$repo"
  if [ -d "$repo_path" ]; then
    echo "Pulling $repo..."
    cd "$repo_path" && git pull origin main
    echo ""
  else
    echo "Cloning $repo..."
    git clone "https://github.com/$GITHUB_ORG/$repo.git" "$repo_path"
    echo ""
  fi
done

echo "Done!"

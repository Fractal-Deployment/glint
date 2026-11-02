#!/usr/bin/env bash
# Synchronize the Jadon-Fox GitHub Project #6 with GLINT repository and issues.
# Safe to re-run: checks project identity and skips issues already on the board.
# Does not create, delete, transfer, or fork repositories or GitHub Projects.
set -euo pipefail

OWNER="Jadon-Fox"
PROJECT_NUMBER="6"
EXPECTED_URL="https://github.com/users/Jadon-Fox/projects/6"

if ! command -v gh >/dev/null 2>&1; then
  echo "ERROR: GitHub CLI (gh) is required. Install it before using this script." >&2
  exit 1
fi

echo "[1/5] Verify GitHub CLI authentication and project access"
gh auth status >/dev/null

# Read project URL to ensure target project matches Project 6
ACTUAL_URL="$(gh project view "$PROJECT_NUMBER" --owner "$OWNER" --format json --jq '.url' 2>/dev/null || echo "")"
if [[ -z "$ACTUAL_URL" ]]; then
  echo "WARN: Cannot query Project #6 directly via gh CLI (likely missing 'read:project' scope)."
  echo "If needed, run 'gh auth refresh -s read:project' to grant project permissions."
  echo "Proceeding with known target: $EXPECTED_URL"
elif [[ "$ACTUAL_URL" != "$EXPECTED_URL" ]]; then
  echo "ERROR: Project mismatch. Got '$ACTUAL_URL', expected '$EXPECTED_URL'." >&2
  exit 1
else
  echo "Project confirmed: $ACTUAL_URL"
  gh project view "$PROJECT_NUMBER" --owner "$OWNER"
fi

REPOS=(
  "Jadon-Fox/glint"
)

# GLINT Project 6 Issues & PRs (P7 ladder and hardware tracking)
ISSUES=(
  "https://github.com/Jadon-Fox/glint/issues/1"
  "https://github.com/Jadon-Fox/glint/issues/4"
  "https://github.com/Jadon-Fox/glint/issues/7"
  "https://github.com/Jadon-Fox/glint/issues/10"
  "https://github.com/Jadon-Fox/glint/issues/11"
  "https://github.com/Jadon-Fox/glint/issues/12"
  "https://github.com/Jadon-Fox/glint/issues/13"
  "https://github.com/Jadon-Fox/glint/issues/14"
  "https://github.com/Jadon-Fox/glint/issues/15"
  "https://github.com/Jadon-Fox/glint/issues/16"
  "https://github.com/Jadon-Fox/glint/issues/17"
  "https://github.com/Jadon-Fox/glint/issues/18"
  "https://github.com/Jadon-Fox/glint/issues/19"
  "https://github.com/Jadon-Fox/glint/issues/20"
  "https://github.com/Jadon-Fox/glint/issues/21"
  "https://github.com/Jadon-Fox/glint/issues/22"
  "https://github.com/Jadon-Fox/glint/issues/25"
  "https://github.com/Jadon-Fox/glint/pull/24"
)

echo "[2/5] Link GLINT repository to Project #6"
LINK_ERRORS=0
for repo in "${REPOS[@]}"; do
  echo "  $repo"
  if ! gh project link "$PROJECT_NUMBER" --owner "$OWNER" --repo "$repo" 2>/dev/null; then
    echo "  NOTE: gh project link returned non-zero (may already be linked, or scope needed)." >&2
    LINK_ERRORS=$((LINK_ERRORS+1))
  fi
done

echo "[3/5] Read existing project issue URLs"
EXISTING_URLS="$(gh project item-list "$PROJECT_NUMBER" --owner "$OWNER" --format json --limit 500 --jq '.items[].content.url // empty' 2>/dev/null || echo "")"

echo "[4/5] Reconcile issues on Project #6"
ADDED=0
FAILED=0
SKIPPED=0
for url in "${ISSUES[@]}"; do
  if printf '%s\n' "$EXISTING_URLS" | grep -Fxq "$url"; then
    echo "  SKIP visible: $url"
    SKIPPED=$((SKIPPED+1))
  elif gh project item-add "$PROJECT_NUMBER" --owner "$OWNER" --url "$url" --format json 2>/dev/null; then
    echo "  Attempted add: $url"
    ADDED=$((ADDED+1))
  else
    echo "  WARN: Could not add $url (missing write:project or item exists)" >&2
    FAILED=$((FAILED+1))
  fi
done

echo "[5/5] Summary"
echo "Project: $EXPECTED_URL"
echo "Items targeted: ${#ISSUES[@]}"
echo "Added: $ADDED, Skipped (already visible): $SKIPPED, Pending/Scope required: $FAILED"

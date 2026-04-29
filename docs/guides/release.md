# Releasing TinyROS

This page documents how a release is cut. The mechanics are automated by
[`.github/workflows/release.yaml`](../../.github/workflows/release.yaml);
this guide covers the human steps before and after.

## One-time setup (per maintainer)

1. **PyPI trusted publisher.** On
   [PyPI -> tinyros -> Publishing](https://pypi.org/manage/project/tinyros/settings/publishing/),
   add a *trusted publisher* with:
    - Owner: `antonioterpin`
    - Repository: `tinyros`
    - Workflow filename: `release.yaml`
    - Environment name: `pypi`

2. **GitHub environment.** On the repo -> Settings -> Environments,
   create an environment named `pypi`. Optionally restrict deployments
   to protected branches and require manual approval.

Once both are configured, no PyPI API token is ever stored in the repo
or in a GitHub secret -- short-lived OIDC tokens authorize each publish.

## Cutting a release

```sh
# 1. Bump pyproject.toml version on a release branch.
git checkout -b release/X.Y.Z
$EDITOR pyproject.toml   # bump `version = "X.Y.Z"`
git commit -am "release: X.Y.Z"

# 2. Open and merge the release PR.
gh pr create --title "release: X.Y.Z" --body "..."

# 3. Tag main once the release PR has merged.
git checkout main
git pull
git tag -a vX.Y.Z -m "tinyros X.Y.Z"
git push origin vX.Y.Z
```

Pushing the tag triggers `release.yaml`, which:

1. Verifies the tag matches `pyproject.toml`'s `version` (fails fast if
   they drift).
2. Builds the sdist and wheel with `uv build`.
3. Runs `twine check` on the built artifacts.
4. Publishes to PyPI via OIDC trusted publishing.
5. Creates the GitHub Release with auto-generated notes scoped to the
   commit range since the previous tag.

## After the release

- Confirm the release shows on
  [PyPI](https://pypi.org/project/tinyros/) and
  [GitHub Releases](https://github.com/antonioterpin/tinyros/releases).
- If anything failed, the workflow can be re-run from
  *Actions -> Release -> Re-run all jobs* once the underlying issue
  (e.g. PyPI version conflict, missing trusted publisher config) is
  resolved. There is no need to re-tag.

## Why the tag is the trigger

The tag is the single source-of-truth artifact for a release: it is
immutable, reviewable on `git log`, and naturally aligned with both the
PyPI version and the GitHub Release. `pyproject.toml` is bumped on a
regular PR like any other change; the tag is the event that says "this
commit is the release."

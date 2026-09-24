# CI on a self-hosted GitLab

Two stages in `.gitlab-ci.yml`:

- **check** — `lint` and `test`. Plain `python:3.10-slim`, no container runtime,
  no model weights, no privileges. If these pass, your runner works.
- **build** — builds the image from `containerfile`, smoke-tests it, and pushes
  only if this GitLab has a registry.

Start by pushing and watching **check**. It isolates "is the runner working"
from "can the runner build images", which are separate problems.

## Do you need a registry?

**No.** `container` builds and smoke-tests the image either way. The push is
guarded on `$CI_REGISTRY`, a predefined variable that exists *only* when the
GitLab Container Registry is enabled. Registry off → the job logs
`no registry configured -- building and smoke-testing only` and still fails on a
broken build, which is most of the value.

To turn it on later, in `/etc/gitlab/gitlab.rb`:

```ruby
registry_external_url 'https://gitlab.example.com:5050'
```

then `gitlab-ctl reconfigure`. The registry needs its own TLS certificate or a
port on the existing one. Nothing in the CI file changes — `$CI_REGISTRY`
appears and the push starts working. Images land at
`$CI_REGISTRY_IMAGE` = `<registry>/<group>/<project>`.

If you would rather push elsewhere (Harbor, a plain `registry:2`), set
`CI_REGISTRY`, `CI_REGISTRY_USER`, `CI_REGISTRY_PASSWORD` and `CI_REGISTRY_IMAGE`
as masked CI/CD variables — the job reads those names and does not care who
provides them.

## The runner

Check what you registered:

```bash
sudo gitlab-runner list
sudo cat /etc/gitlab-runner/config.toml
```

**`check` needs only** `executor = "docker"` (any image) or a `shell` executor
with Python 3.10+.

**`build` as written needs docker-in-docker**, which needs a privileged runner:

```toml
[[runners]]
  executor = "docker"
  [runners.docker]
    privileged = true          # required for docker:dind
    volumes = ["/certs/client", "/cache"]
```

`gitlab-runner register` does **not** set `privileged` by default, so if you
accepted the defaults this is the line to add. Then
`sudo gitlab-runner restart`.

### If you do not want a privileged runner

Privileged dind gives the job effective root on the host. On a personal server
that may be fine; if not, swap the `container` job for buildah, which builds
rootless and suits `containerfile` better anyway:

```yaml
container:
  stage: build
  image: quay.io/buildah/stable
  variables:
    STORAGE_DRIVER: vfs           # overlay needs privileges buildah will not have
    BUILDAH_FORMAT: oci
  script:
    - buildah bud -f containerfile -t "asr-server:$CI_COMMIT_SHORT_SHA" .
    - buildah run "asr-server:$CI_COMMIT_SHORT_SHA" -- python serve.py --help
    - |
      if [ -n "$CI_REGISTRY" ]; then
        buildah login -u "$CI_REGISTRY_USER" -p "$CI_REGISTRY_PASSWORD" "$CI_REGISTRY"
        buildah push "asr-server:$CI_COMMIT_SHORT_SHA" \
          "docker://$CI_REGISTRY_IMAGE:$CI_COMMIT_SHORT_SHA"
      fi
```

`STORAGE_DRIVER: vfs` is slower than overlay but works unprivileged. If you have
a `shell` executor with podman installed, `podman build -f containerfile` needs
no special configuration at all.

### If the runner is on the GitLab host itself

`docker:dind` pulls `python:3.10-slim` and `ghcr.io/astral-sh/uv` from the
internet. Behind a proxy or an air-gapped network the build fails on the first
`FROM`; mirror both into your own registry and rewrite the two `FROM`/`COPY
--from` lines in `containerfile`.

## What the smoke test does and does not prove

The image ships no weights — they are a 3 GB bind mount — so CI cannot transcribe
anything. What it checks is that every dependency resolved and the package
imports:

```
python serve.py --help
python fetch_models.py --help
python -c "import asr.server"
```

That catches the realistic build failures: a missing `libgomp1`, a `uv sync`
that resolved differently, a syntax error in a path only the server touches. It
does not catch a model-loading or transcription regression. Those need weights,
so run them yourself against a real deployment.

## First run

1. Push. Watch `check`. If `lint` or `test` fails, it is the code, not the CI.
2. If both pass and `container` fails on `Cannot connect to the Docker daemon`,
   the runner is not privileged — see above.
3. If `container` passes and logs `no registry configured`, everything worked;
   enable the registry when you actually want to pull images somewhere else.

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

**`build` as written uses buildah**, which builds unprivileged — so it works on a
rootless podman runner, where docker-in-docker cannot. A rootless runner looks
like this:

```toml
[runners.docker]
  host = "unix:///run/user/1000/podman/podman.sock"
  privileged = false
```

`privileged = false` plus a rootless socket means `docker:dind` will fail with
`Cannot connect to the Docker daemon` no matter how the service is configured.
That is not a misconfiguration to fix; it is the point of running rootless.

### If you do have a privileged docker runner

dind is faster (overlay storage rather than buildah's `vfs`). Swap the
`container` job for:

```yaml
container:
  stage: build
  image: docker:27-cli
  services: ["docker:27-dind"]
  variables:
    DOCKER_HOST: tcp://docker:2376
    DOCKER_TLS_CERTDIR: /certs
    DOCKER_TLS_VERIFY: 1
    DOCKER_CERT_PATH: /certs/client
  script:
    - docker build -f containerfile -t "asr-server:$CI_COMMIT_SHORT_SHA" .
```

with, in `config.toml`:

```toml
[runners.docker]
  privileged = true          # required for docker:dind
  volumes = ["/certs/client", "/cache"]
```

`gitlab-runner register` does not set `privileged`, and it grants the job
effective root on the host — which is why buildah is the default here.

If you have a `shell` executor with podman installed, `podman build -f
containerfile` needs no special configuration at all.

### If the runner is on the GitLab host itself

`docker:dind` pulls `python:3.10-slim` and `ghcr.io/astral-sh/uv` from the
internet. Behind a proxy or an air-gapped network the build fails on the first
`FROM`; mirror both into your own registry and rewrite the two `FROM`/`COPY
--from` lines in `containerfile`.

## `SSL certificate problem: unable to get local issuer certificate`

The clone fails in the job, not on your machine — CI clones over HTTP(S) using
`CI_REPOSITORY_URL`, even when your own remote is `git@`. It means the job
container does not trust whatever CA signed your GitLab certificate.

Check what is actually being served:

```bash
curl -sSv https://gitlab.example.com/ -o /dev/null 2>&1 | grep -E "issuer|verify"
```

An issuer like `CN=Caddy Local Authority` means Caddy's internal CA
(`tls internal`), which is self-signed by design. Same story for a bare
self-signed cert or a private company CA.

### Best fix: issue a publicly trusted certificate

If you own the domain, give Caddy an ACME issuer instead of the internal one.
For a host with no public A record, use DNS-01 with the plugin for your provider:

```caddyfile
gitlab.example.com {
    tls {
        dns cloudflare {env.CF_API_TOKEN}
    }
    reverse_proxy localhost:8080
}
```

Everything downstream — runners, `docker login`, `git clone`, your laptop — then
works with no configuration at all. Worth the twenty minutes; the alternative is
distributing a CA to every client forever.

### Otherwise: give the runner the CA

Two separate certificates are involved, and fixing only one leaves the clone
broken:

| who | needs it for | where |
|---|---|---|
| the **runner process** | polling GitLab for jobs | `/etc/gitlab-runner/certs/<hostname>.crt`, or any path via `tls-ca-file` |
| the **job/helper container** | `git clone`, artifacts, cache | a volume mounted at `/etc/gitlab-runner/certs/ca.crt` *inside* the container |

A clone failing with `unable to get local issuer certificate` is the second one.
Dropping the file on the host alone does not fix it — with the Docker executor,
host certificates do not reach job containers.

Find Caddy's root (paths vary by install):

```bash
sudo find / -name root.crt -path "*caddy*" 2>/dev/null
# usually /var/lib/caddy/.local/share/caddy/pki/authorities/local/root.crt
# Docker installs: /data/caddy/pki/authorities/local/root.crt
```

Place it on the runner host and wire up both paths:

```bash
sudo mkdir -p /etc/gitlab-runner/certs
sudo cp root.crt /etc/gitlab-runner/certs/gitlab.example.com.crt
```

```toml
[[runners]]
  url = "https://gitlab.example.com/"
  executor = "docker"
  # 1. the runner process itself
  tls-ca-file = "/etc/gitlab-runner/certs/gitlab.example.com.crt"

  [runners.docker]
    # 2. the job container -- destination MUST be exactly ca.crt; the helper
    #    image installs that file into the trust store at start-up
    volumes = [
      "/cache",
      "/etc/gitlab-runner/certs/gitlab.example.com.crt:/etc/gitlab-runner/certs/ca.crt:ro",
    ]
```

Then `sudo gitlab-runner restart`.

#### With a rootless podman runner

The source path is opened by the podman user, not by root, so the file has to be
readable by that uid:

```bash
sudo chmod 644 /etc/gitlab-runner/certs/ca.crt
sudo -u "#958" cat /etc/gitlab-runner/certs/ca.crt >/dev/null && echo "readable"
```

A mount whose source the runtime cannot read fails the job with a confusing
error, or silently produces an empty file.

On naming: the host-side filename only has to match the hostname if you rely on
the runner's automatic lookup — `<hostname>.crt`, base hostname with **no port**,
so `gitlab.example.com.crt` even for `https://gitlab.example.com:8443/`. Setting
`tls-ca-file` explicitly lets you call it anything. The container-side name is
not negotiable: it must be `ca.crt` at that path.

Lookup locations differ by how the runner runs: `/etc/gitlab-runner/certs/` as
root, `~/.gitlab-runner/certs/` as a normal user, `./certs/` elsewhere.

**Use the root, not the intermediate.** Caddy's PKI directory holds both
`root.crt` and `intermediate.crt`, and Caddy serves leaf + intermediate in the
handshake — so the client needs only the root as its trust anchor. Trusting the
intermediate instead fails with the same "unable to get local issuer" message,
which makes it an easy hour to lose.

Confirm the file works before restarting anything. If `curl` accepts it, so will
the runner:

```bash
curl --cacert /etc/gitlab-runner/certs/ca.crt https://gitlab.example.com/ -o /dev/null -sS \
  && echo "CA is correct"
```

If the runner is itself containerised, the host directory must be part of its
config volume or the file disappears on restart:

```bash
docker run -d --name gitlab-runner --restart always \
  -v /srv/gitlab-runner/config:/etc/gitlab-runner \
  -v /var/run/docker.sock:/var/run/docker.sock \
  gitlab/gitlab-runner:latest
# place the .crt in /srv/gitlab-runner/config/certs/
```

Your own `script:` steps do not inherit the trust store either — the helper
image installs `ca.crt` for git and artifacts, not for arbitrary commands. If a
job needs to `curl` your GitLab, install it there too, via `pre_build_script` or
a line in the job.

### Escape hatch

```yaml
variables:
  GIT_SSL_NO_VERIFY: "true"
```

Unblocks the clone immediately and disables certificate verification for it.
Acceptable while you sort the CA out on a private network; not a resting place.

### This will come back with the registry

If you enable the Container Registry behind the same Caddy, `docker login` and
`docker pull` need the CA too — in a different location, per registry host:

```bash
sudo mkdir -p /etc/docker/certs.d/gitlab.example.com:5050
sudo cp root.crt /etc/docker/certs.d/gitlab.example.com:5050/ca.crt
```

Podman and buildah read `/etc/containers/certs.d/` with the same layout.

**In CI the job container is ephemeral**, so there is nothing to copy into — the
`buildah push` step needs the CA mounted at that path from the runner config,
which is a *second* mount alongside the one the clone uses:

```toml
[runners.docker]
  volumes = [
    "/cache",
    # git clone
    "/etc/gitlab-runner/certs/ca.crt:/etc/gitlab-runner/certs/ca.crt:ro",
    # buildah push -- note the registry port is part of the directory name
    "/etc/gitlab-runner/certs/ca.crt:/etc/containers/certs.d/gitlab.example.com:5050/ca.crt:ro",
  ]
```

The port belongs in the directory name, and must match the registry host exactly
as it appears in `$CI_REGISTRY`. A mismatch fails as an untrusted certificate,
not as a missing file.

That is the same root certificate reaching its fourth location — runner process,
job container, host docker/podman, CI job container — which is the argument for
fixing this at the certificate rather than distributing the CA.

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

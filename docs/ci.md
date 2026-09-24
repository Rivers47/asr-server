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
    # 2. the job container. Entries are "HOST_PATH:CONTAINER_PATH:options".
    #    The container path must be exactly /etc/gitlab-runner/certs/ca.crt --
    #    the helper image installs that file into the trust store at start-up.
    #    The host path can be anywhere the container runtime can read (see below).
    volumes = [
      "/cache",
      "/srv/runner-certs/ca.crt:/etc/gitlab-runner/certs/ca.crt:ro",
    ]
```

Keep the two sides visibly different. Writing the same path twice works, but
makes it impossible to tell at a glance which side failed when something breaks.

Then `sudo gitlab-runner restart`.

#### With a rootless podman runner

The **host** side of a mount is opened by the podman user, not by root — and the
runner process and the podman socket are often different users. A runner running
as root reads `tls-ca-file` from `/etc/` happily, then hands the same path to a
rootless podman that cannot see it at all.

When the source is missing from the runtime's view, podman tries to create it,
and you get this at `prepare environment`, before any script runs:

```
make cli opts(): making volume mountpoint for volume /etc/gitlab-runner/certs/ca.crt:
mkdir /etc/gitlab-runner: permission denied
```

`mkdir` on the *source* is the tell. It fails every job in the pipeline, not just
the one that needed the certificate, because `runners.docker.volumes` applies to
all of them.

Put the file somewhere the socket's user owns:

```bash
id -u gitlab-runner                     # whichever uid owns the podman socket
sudo install -d -o 958 -g 958 /srv/runner-certs
sudo install -o 958 -g 958 -m 644 /etc/gitlab-runner/certs/ca.crt /srv/runner-certs/ca.crt
sudo -u "#958" cat /srv/runner-certs/ca.crt >/dev/null && echo readable
```

and use `/srv/runner-certs/ca.crt` as the host side. On an SELinux-enforcing host
add `:Z` to the mount as well — otherwise the container starts and then cannot
read the file, which is a different error one step later.

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

#### If the runner is itself containerised

Then the two settings are resolved by two different processes against two
different filesystems, and a path that looks right can be wrong:

| setting | resolved by | against |
|---|---|---|
| `tls-ca-file` | the runner process | the **runner container's** filesystem |
| `runners.docker.volumes` host side | the container runtime | the **host's** filesystem |

So `tls-ca-file = "/etc/gitlab-runner/certs/ca.crt"` can work perfectly — the
runner authenticates and picks up jobs — while the identical string as a mount
source refers to a path that does not exist on the host. The runtime then tries
to create it and fails.

It is one file with two names. Find the mapping:

```bash
podman inspect gitlab-runner \
  --format '{{range .Mounts}}{{.Source}} -> {{.Destination}}{{"\n"}}{{end}}'
```

With the usual layout — host `/srv/gitlab-runner/config` mounted at container
`/etc/gitlab-runner` — a certificate placed at
`/srv/gitlab-runner/config/certs/ca.crt` on the host appears to the runner as
`/etc/gitlab-runner/certs/ca.crt`, and the config uses **both** spellings:

```toml
  # container view: the runner process reads this
  tls-ca-file = "/etc/gitlab-runner/certs/ca.crt"

  [runners.docker]
    volumes = [
      "/cache",
      # host view on the left, job-container view on the right
      "/srv/gitlab-runner/config/certs/ca.crt:/etc/gitlab-runner/certs/ca.crt:ro",
    ]
```

Putting it inside the config volume also means it survives a runner restart.
The host-side file still has to be readable by the uid owning the container
runtime socket, which is a separate condition from existing in the right place.

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

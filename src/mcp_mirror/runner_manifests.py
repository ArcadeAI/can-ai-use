"""Validated manifests for reproducible framework runner environments."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from functools import lru_cache
from importlib.resources import files
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .evidence import canonical_sha256
from .models import CaptureStage


class ManagedRunnerUnavailable(RuntimeError):
    """Raised when a manifest cannot run through the local managed-runner path."""


class RunnerManifest(BaseModel):
    """One versioned adapter environment and its evidence contract."""

    model_config = ConfigDict(frozen=True)

    id: str
    module: str
    runtime: Literal["python", "node", "java", "go", "dotnet"]
    python: str | None = None
    node: str | None = None
    # A Java runner pins the exact Temurin JDK release and Maven version it builds and
    # runs its worker with, both installed into the managed cache, never system-wide.
    java: str | None = None
    maven: str | None = None
    # A .NET runner pins the exact .NET SDK release it builds and runs its worker with,
    # installed into the managed cache, never system-wide.
    dotnet: str | None = None
    # A Go runner pins the Go release that builds its worker binary, installed into the
    # managed cache from go.dev/dl and checked against the sha256 go.dev publishes.
    go: str | None = None
    packages: tuple[str, ...] = Field(min_length=1)
    # A Node runner's framework runs in Node, but the renderer that drives it is
    # Python, so it still needs an interpreter and the few packages mcp_mirror imports.
    host_packages: tuple[str, ...] = ()
    protocols: tuple[str, ...] = Field(min_length=1)
    capture_stage: CaptureStage
    provider_request_captured: bool
    # A runner whose pins cannot share the project's locked environment (for example an
    # MCP SDK major the 'all' extra excludes) has no local extra and runs only managed.
    managed_only: bool = False
    managed_only_reason: str | None = None

    @model_validator(mode="after")
    def validate_runtime_version(self) -> "RunnerManifest":
        if self.runtime == "python" and not self.python:
            raise ValueError("Python runner requires a python version")
        if self.runtime == "node" and not self.node:
            raise ValueError("Node runner requires a node version")
        if self.managed_only and not self.managed_only_reason:
            raise ValueError("a managed_only runner must say why in managed_only_reason")
        if self.runtime == "java" and not (self.java and self.maven):
            raise ValueError("Java runner requires a java release and a maven version")
        if self.runtime == "dotnet" and not self.dotnet:
            raise ValueError(".NET runner requires a dotnet SDK version")
        if self.runtime == "go" and not self.go:
            raise ValueError("Go runner requires a go release")
        return self

    @property
    def digest(self) -> str:
        payload = self.model_dump(mode="json")
        if not self.managed_only:
            # Leave the flag out when unset, so the runners that predate it keep the
            # digests their committed measurements record.
            payload.pop("managed_only")
            payload.pop("managed_only_reason")
        if self.runtime != "java":
            # Likewise the Java toolchain fields, which only a Java runner sets.
            payload.pop("java")
            payload.pop("maven")
        if self.runtime != "dotnet":
            # And the .NET SDK field, which only a .NET runner sets.
            payload.pop("dotnet")
        if self.runtime != "go":
            # And the Go release, which only a Go runner sets.
            payload.pop("go")
        return canonical_sha256(payload)


@lru_cache(maxsize=1)
def load_runner_manifests() -> dict[str, RunnerManifest]:
    """Load and validate the package's runner manifest registry."""

    manifest_path = files("mcp_mirror").joinpath("runner-manifests.json")
    payload = json.loads(manifest_path.read_text())
    if payload.get("schema") != "mcp-mirror/runner-manifests@1":
        raise ValueError("unsupported runner manifest schema")
    raw_runners = payload.get("runners")
    if not isinstance(raw_runners, dict) or not raw_runners:
        raise ValueError("runner manifest registry is empty")

    return {
        renderer_id: RunnerManifest.model_validate(
            {"id": renderer_id, **runner}
        )
        for renderer_id, runner in raw_runners.items()
    }


# The Mastra renderer reads this to find a managed install instead of the bundled
# node_modules. It reaches the renderer through a uv env file written beside the
# install, so the caller's own environment needs no change.
NODE_RUNNER_DIR_ENV = "MCP_MIRROR_NODE_RUNNER_DIR"

# Node resolves an ES module's bare imports from the file's own directory upward and
# ignores NODE_PATH, so each Node worker is copied next to the node_modules it imports.
_NODE_WORKERS = {
    "mastra": Path(__file__).parent / "renderers" / "mastra_node",
    "vercel_ai": Path(__file__).parent / "renderers" / "vercel_ai_node",
    "openai_agents_js": Path(__file__).parent / "renderers" / "openai_agents_js_node",
    "pi": Path(__file__).parent / "renderers" / "pi_node",
    "google_adk_js": Path(__file__).parent / "renderers" / "google_adk_js_node",
    "langchain_js": Path(__file__).parent / "renderers" / "langchain_js_node",
    "genkit_js": Path(__file__).parent / "renderers" / "genkit_js_node",
    "copilot_sdk": Path(__file__).parent / "renderers" / "copilot_sdk_node",
}
_NPM_TIMEOUT_SECONDS = 600


def build_managed_worker_command(
    renderer_id: str,
    output_path: Path,
    *,
    uv_executable: str | None = None,
    npm_executable: str | None = None,
    cache_root: Path | None = None,
) -> list[str]:
    """Build an argument-safe ``uv`` command for one managed runner.

    A Node runner is installed first, once per manifest digest, into the cache.
    """

    manifest = load_runner_manifests().get(renderer_id)
    if manifest is None:
        raise ManagedRunnerUnavailable(
            f"no managed runner manifest for {renderer_id!r}"
        )

    uv = uv_executable or shutil.which("uv")
    if not uv:
        raise ManagedRunnerUnavailable(
            "managed runners require uv: https://docs.astral.sh/uv/"
        )

    command = [
        uv,
        "run",
        "--isolated",
        "--no-project",
        "--python",
        manifest.python or "3.11",
    ]
    if manifest.runtime == "node":
        install = install_node_runner(
            manifest, npm_executable=npm_executable, cache_root=cache_root
        )
        command.extend(["--env-file", str(install / "runner.env")])
        packages = manifest.host_packages
    elif manifest.runtime == "java":
        install = install_java_runner(manifest, cache_root=cache_root)
        command.extend(["--env-file", str(install / "runner.env")])
        packages = manifest.host_packages
    elif manifest.runtime == "dotnet":
        install = install_dotnet_runner(manifest, cache_root=cache_root)
        command.extend(["--env-file", str(install / "runner.env")])
        packages = manifest.host_packages
    elif manifest.runtime == "go":
        install = install_go_runner(manifest, cache_root=cache_root)
        command.extend(["--env-file", str(install / "runner.env")])
        packages = manifest.host_packages
    else:
        packages = manifest.packages
    for package in packages:
        command.extend(["--with", package])
    command.extend(
        [
            "python",
            "-m",
            "mcp_mirror._render_worker",
            renderer_id,
            str(output_path),
        ]
    )
    return command


def managed_cache_root() -> Path:
    override = os.environ.get("MCP_MIRROR_CACHE_DIR")
    if override:
        return Path(override)
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "mcp-mirror"


def node_install_digest(manifest: RunnerManifest) -> str:
    """Key one Node install by the manifest and the lockfile that pins its tree."""

    lock = _NODE_WORKERS[manifest.id] / "package-lock.json"
    return canonical_sha256(
        {
            "manifest": manifest.digest,
            "lock": hashlib.sha256(lock.read_bytes()).hexdigest()
            if lock.exists()
            else None,
        }
    )


def node_package_json(manifest: RunnerManifest) -> dict[str, object]:
    """The package.json a managed Node install is resolved from.

    The manifest's own Node version is installed as a package too, so the worker runs
    on the pinned runtime rather than whichever Node is on PATH.
    """

    dependencies = {
        name: version
        for name, version in (
            package.rsplit("@", 1) for package in manifest.packages
        )
    }
    dependencies["node"] = manifest.node or ""
    return {
        "name": f"mcp-mirror-managed-{manifest.id}",
        "private": True,
        "type": "module",
        "dependencies": dependencies,
    }


def install_node_runner(
    manifest: RunnerManifest,
    *,
    npm_executable: str | None = None,
    cache_root: Path | None = None,
) -> Path:
    """Install one Node manifest into the cache and return its directory.

    The install is built in a temporary directory and renamed into place, so a
    concurrent or interrupted run never sees half a ``node_modules``.
    """

    worker_dir = _NODE_WORKERS.get(manifest.id)
    if worker_dir is None:
        raise ManagedRunnerUnavailable(
            f"no Node worker is bundled for {manifest.id!r}"
        )
    root = (cache_root or managed_cache_root()) / "node-runners"
    install = root / node_install_digest(manifest)
    if not (install / ".complete").exists():
        npm = npm_executable or shutil.which("npm")
        if not npm:
            raise ManagedRunnerUnavailable(
                "managed Node runners require npm: https://nodejs.org/"
            )
        root.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{install.name}-", dir=root))
        try:
            _npm_install(manifest, worker_dir, staging, npm)
            try:
                staging.rename(install)
            except OSError:
                # Another run finished the same install first; its copy is identical.
                if not (install / ".complete").exists():
                    raise
        finally:
            shutil.rmtree(staging, ignore_errors=True)
    _refresh_worker_files(worker_dir, install)
    return install


def _npm_install(
    manifest: RunnerManifest, worker_dir: Path, staging: Path, npm: str
) -> None:
    (staging / "package.json").write_text(
        json.dumps(node_package_json(manifest), indent=2) + "\n"
    )
    lock = worker_dir / "package-lock.json"
    if lock.exists():
        # npm keeps every locked version the manifest's pins still allow, so the
        # transitive tree matches the one the bundled worker was measured with.
        shutil.copy2(lock, staging / "package-lock.json")
    try:
        proc = subprocess.run(
            [npm, "install", "--no-audit", "--no-fund", "--loglevel=error"],
            cwd=staging,
            capture_output=True,
            text=True,
            timeout=_NPM_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        raise ManagedRunnerUnavailable(
            f"npm install for {manifest.id!r} timed out after "
            f"{_NPM_TIMEOUT_SECONDS}s"
        ) from exc
    except OSError as exc:
        raise ManagedRunnerUnavailable(f"npm could not start: {exc}") from exc
    if proc.returncode != 0:
        tail = " | ".join((proc.stderr or proc.stdout).strip().splitlines()[-3:])
        raise ManagedRunnerUnavailable(
            f"npm install for {manifest.id!r} failed: {tail}"
        )

    node = staging / "node_modules" / ".bin" / "node"
    try:
        version = subprocess.run(
            [str(node), "--version"],
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout.strip()
    except OSError as exc:
        raise ManagedRunnerUnavailable(
            f"the pinned Node {manifest.node} did not install: {exc}"
        ) from exc
    if version.lstrip("v").split(".")[0] != (manifest.node or "").split(".")[0]:
        raise ManagedRunnerUnavailable(
            f"the pinned Node {manifest.node} installed as {version or 'nothing'}"
        )
    (staging / ".complete").write_text(version + "\n")


def _refresh_worker_files(worker_dir: Path, install: Path) -> None:
    """Copy the worker scripts beside the cached node_modules on every run.

    The worker is part of mcp_mirror, not of the manifest, so an edit to it must reach
    an install that was cached before the edit.
    """

    for worker in worker_dir.glob("*.mjs"):
        target = install / worker.name
        if target.exists() and target.read_bytes() == worker.read_bytes():
            continue
        fd, temp = tempfile.mkstemp(dir=install, suffix=".tmp")
        os.close(fd)
        shutil.copyfile(worker, temp)
        os.replace(temp, target)

    install_text = str(install)
    if "'" in install_text or "\n" in install_text:
        raise ManagedRunnerUnavailable(
            f"the managed runner cache path cannot be quoted: {install_text!r}"
        )
    # uv reads single-quoted dotenv values literally.
    env_file = f"{NODE_RUNNER_DIR_ENV}='{install_text}'\n"
    target = install / "runner.env"
    if not target.exists() or target.read_text() != env_file:
        fd, temp = tempfile.mkstemp(dir=install, suffix=".tmp")
        os.close(fd)
        Path(temp).write_text(env_file)
        os.replace(temp, target)


# ---------------------------------------------------------------------------
# Java runners
# ---------------------------------------------------------------------------

# The Java renderer reads these to find the managed worker jar and the JDK that runs it.
JAVA_RUNNER_DIR_ENV = "MCP_MIRROR_JAVA_RUNNER_DIR"
JAVA_HOME_ENV = "MCP_MIRROR_JAVA_HOME"

# Each Java worker is a Maven project built into one executable jar.
_JAVA_WORKERS = {
    "spring_ai": Path(__file__).parent / "renderers" / "spring_ai_java",
    "google_adk_java": Path(__file__).parent / "renderers" / "google_adk_java",
    "langchain4j": Path(__file__).parent / "renderers" / "langchain4j_java",
    "koog": Path(__file__).parent / "renderers" / "koog_kotlin",
}
JAVA_WORKER_JAR = "worker.jar"
_MAVEN_TIMEOUT_SECONDS = 900
_DOWNLOAD_TIMEOUT_SECONDS = 600
_ADOPTIUM_API = "https://api.adoptium.net/v3"
_MAVEN_CENTRAL = "https://repo.maven.apache.org/maven2"


def _java_platform() -> tuple[str, str]:
    """Adoptium's names for this machine's operating system and architecture."""

    import platform

    system = {"Darwin": "mac", "Linux": "linux", "Windows": "windows"}.get(platform.system())
    machine = {
        "arm64": "aarch64",
        "aarch64": "aarch64",
        "x86_64": "x64",
        "AMD64": "x64",
    }.get(platform.machine())
    if not system or not machine:
        raise ManagedRunnerUnavailable(
            f"no pinned JDK for {platform.system()} {platform.machine()}"
        )
    return system, machine


def java_worker_sources(renderer_id: str) -> list[Path]:
    """The files a Java worker jar is built from, in a stable order."""

    worker_dir = _JAVA_WORKERS[renderer_id]
    return sorted(
        path
        for path in [worker_dir / "pom.xml", *worker_dir.glob("src/**/*")]
        if path.is_file()
    )


def java_install_digest(manifest: RunnerManifest) -> str:
    """Key one Java install by the manifest and every source the jar is built from.

    Unlike a Node worker, a Java worker is compiled into the install, so an edit to it
    has to produce a new install rather than be copied over the old one.
    """

    worker_dir = _JAVA_WORKERS[manifest.id]
    return canonical_sha256(
        {
            "manifest": manifest.digest,
            "sources": {
                str(path.relative_to(worker_dir)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in java_worker_sources(manifest.id)
            },
        }
    )


def java_maven_properties(manifest: RunnerManifest) -> dict[str, str]:
    """``-D`` properties that pin the worker's pom to the manifest's packages.

    Each package is ``group:artifact:version``; the pom names its version property
    after the artifact (``spring-ai-starter-mcp-client.version``), so the manifest
    stays the one place a version is written.
    """

    properties: dict[str, str] = {}
    for package in manifest.packages:
        _group, artifact, version = package.split(":")
        properties[f"{artifact}.version"] = version
    return properties


def _request(url: str):
    import urllib.request

    # The Adoptium API refuses urllib's default User-Agent.
    return urllib.request.Request(url, headers={"User-Agent": "mcp-mirror"})


def _download(url: str, target: Path) -> None:
    import urllib.request

    try:
        with urllib.request.urlopen(_request(url), timeout=_DOWNLOAD_TIMEOUT_SECONDS) as response:
            with target.open("wb") as handle:
                shutil.copyfileobj(response, handle)
    except OSError as exc:
        raise ManagedRunnerUnavailable(f"could not download {url}: {exc}") from exc


def _fetch_text(url: str) -> str:
    import urllib.request

    try:
        with urllib.request.urlopen(_request(url), timeout=60) as response:
            return response.read().decode()
    except OSError as exc:
        raise ManagedRunnerUnavailable(f"could not fetch {url}: {exc}") from exc


def _verified_archive(url: str, digest: str, algorithm: str, staging: Path) -> Path:
    archive = staging / url.rsplit("/", 1)[-1].split("?", 1)[0]
    _download(url, archive)
    actual = hashlib.new(algorithm, archive.read_bytes()).hexdigest()
    if actual != digest.strip().split()[0].lower():
        raise ManagedRunnerUnavailable(f"{archive.name} failed its {algorithm} check")
    return archive


def _unpack_once(target: Path, fetch) -> Path:
    """Unpack a toolchain into ``target`` once, atomically, and return it."""

    import tarfile

    if (target / ".complete").exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}-", dir=target.parent))
    try:
        archive = fetch(staging)
        unpacked = staging / "unpacked"
        with tarfile.open(archive) as tar:
            tar.extractall(unpacked, filter="tar")
        (top,) = list(unpacked.iterdir())
        (top / ".complete").write_text(archive.name + "\n")
        try:
            top.rename(target)
        except OSError:
            if not (target / ".complete").exists():
                raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return target


def _install_jdk(release: str, root: Path) -> Path:
    """The pinned Temurin JDK, checked against Adoptium's published sha256."""

    import urllib.parse

    system, machine = _java_platform()
    target = root / "toolchains" / f"{release}-{system}-{machine}"

    def fetch(staging: Path) -> Path:
        quoted = urllib.parse.quote(release, safe="")
        assets = json.loads(
            _fetch_text(
                f"{_ADOPTIUM_API}/assets/release_name/eclipse/{quoted}"
                f"?os={system}&architecture={machine}&image_type=jdk&jvm_impl=hotspot"
            )
        )
        binaries = [
            b
            for b in assets.get("binaries", [])
            if b.get("os") == system
            and b.get("architecture") == machine
            and b.get("image_type") == "jdk"
        ]
        if not binaries:
            raise ManagedRunnerUnavailable(f"Adoptium lists no {release} JDK for {system} {machine}")
        package = binaries[0]["package"]
        return _verified_archive(package["link"], package["checksum"], "sha256", staging)

    home = _unpack_once(target, fetch)
    mac_home = home / "Contents" / "Home"
    return mac_home if (mac_home / "bin" / "java").exists() else home


def _install_maven(version: str, root: Path) -> Path:
    """The pinned Maven distribution, checked against Maven Central's sha512."""

    target = root / "toolchains" / f"maven-{version}"
    base = f"{_MAVEN_CENTRAL}/org/apache/maven/apache-maven/{version}/apache-maven-{version}-bin.tar.gz"
    return _unpack_once(
        target,
        lambda staging: _verified_archive(base, _fetch_text(base + ".sha512"), "sha512", staging),
    )


def install_java_runner(
    manifest: RunnerManifest,
    *,
    cache_root: Path | None = None,
) -> Path:
    """Install a Java manifest: its JDK, its Maven, and the worker jar built from them.

    The JDK, Maven and the Maven repository are shared across installs under the cache;
    the jar is built in a temporary directory and renamed into place, like a Node
    install.
    """

    worker_dir = _JAVA_WORKERS.get(manifest.id)
    if worker_dir is None:
        raise ManagedRunnerUnavailable(f"no Java worker is bundled for {manifest.id!r}")
    root = (cache_root or managed_cache_root()) / "java-runners"
    install = root / java_install_digest(manifest)
    java_home = _install_jdk(manifest.java or "", root)
    if not (install / ".complete").exists():
        maven_home = _install_maven(manifest.maven or "", root)
        root.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{install.name}-", dir=root))
        try:
            _maven_package(manifest, worker_dir, staging, java_home, maven_home, root / "m2")
            try:
                staging.rename(install)
            except OSError:
                if not (install / ".complete").exists():
                    raise
        finally:
            shutil.rmtree(staging, ignore_errors=True)
    _write_java_env(install, java_home)
    return install


def _maven_package(
    manifest: RunnerManifest,
    worker_dir: Path,
    staging: Path,
    java_home: Path,
    maven_home: Path,
    repository: Path,
) -> None:
    project = staging / "project"
    shutil.copytree(worker_dir, project, ignore=shutil.ignore_patterns("target"))
    properties = [f"-D{key}={value}" for key, value in java_maven_properties(manifest).items()]
    environment = {**os.environ, "JAVA_HOME": str(java_home)}
    try:
        proc = subprocess.run(
            [
                str(maven_home / "bin" / "mvn"),
                "-B",
                "-q",
                f"-Dmaven.repo.local={repository}",
                *properties,
                "package",
                "-DskipTests",
            ],
            cwd=project,
            env=environment,
            capture_output=True,
            text=True,
            timeout=_MAVEN_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        raise ManagedRunnerUnavailable(
            f"maven package for {manifest.id!r} timed out after {_MAVEN_TIMEOUT_SECONDS}s"
        ) from exc
    except OSError as exc:
        raise ManagedRunnerUnavailable(f"maven could not start: {exc}") from exc
    if proc.returncode != 0:
        tail = " | ".join((proc.stdout + proc.stderr).strip().splitlines()[-5:])
        raise ManagedRunnerUnavailable(f"maven package for {manifest.id!r} failed: {tail}")
    jars = sorted((project / "target").glob("*-exec.jar")) or sorted((project / "target").glob("*.jar"))
    if not jars:
        raise ManagedRunnerUnavailable(f"maven package for {manifest.id!r} produced no jar")
    shutil.copy2(jars[0], staging / JAVA_WORKER_JAR)
    shutil.rmtree(project, ignore_errors=True)
    (staging / ".complete").write_text(manifest.java + "\n")


def _write_java_env(install: Path, java_home: Path) -> None:
    lines = []
    for key, path in ((JAVA_RUNNER_DIR_ENV, install), (JAVA_HOME_ENV, java_home)):
        text = str(path)
        if "'" in text or "\n" in text:
            raise ManagedRunnerUnavailable(f"the managed runner cache path cannot be quoted: {text!r}")
        lines.append(f"{key}='{text}'\n")
    env_file = "".join(lines)
    target = install / "runner.env"
    if not target.exists() or target.read_text() != env_file:
        fd, temp = tempfile.mkstemp(dir=install, suffix=".tmp")
        os.close(fd)
        Path(temp).write_text(env_file)
        os.replace(temp, target)


# ---------------------------------------------------------------------------
# .NET runners
# ---------------------------------------------------------------------------

# The .NET renderer reads these to find the managed worker build and the SDK that runs it.
DOTNET_RUNNER_DIR_ENV = "MCP_MIRROR_DOTNET_RUNNER_DIR"
DOTNET_ROOT_ENV = "MCP_MIRROR_DOTNET_ROOT"

# Each .NET worker is an SDK-style project published into one directory.
_DOTNET_WORKERS = {
    "ms_agent_framework_dotnet": Path(__file__).parent / "renderers" / "ms_agent_framework_dotnet",
}
DOTNET_WORKER_DIR = "worker"
DOTNET_WORKER_DLL = "Worker.dll"
_DOTNET_TIMEOUT_SECONDS = 900
_DOTNET_RELEASES = "https://builds.dotnet.microsoft.com/dotnet/release-metadata"


def _dotnet_rid() -> str:
    """The .NET runtime identifier of this machine (``osx-arm64``, ``linux-x64``)."""

    import platform

    system = {"Darwin": "osx", "Linux": "linux", "Windows": "win"}.get(platform.system())
    machine = {"arm64": "arm64", "aarch64": "arm64", "x86_64": "x64", "AMD64": "x64"}.get(platform.machine())
    if not system or not machine:
        raise ManagedRunnerUnavailable(f"no pinned .NET SDK for {platform.system()} {platform.machine()}")
    return f"{system}-{machine}"


def dotnet_worker_sources(renderer_id: str) -> list[Path]:
    """The files a .NET worker is built from, in a stable order."""

    worker_dir = _DOTNET_WORKERS[renderer_id]
    return sorted(
        path
        for path in worker_dir.iterdir()
        if path.is_file() and path.suffix in {".cs", ".csproj", ".json"}
    )


def dotnet_install_digest(manifest: RunnerManifest) -> str:
    """Key one .NET install by the manifest and every source the worker is built from."""

    worker_dir = _DOTNET_WORKERS[manifest.id]
    return canonical_sha256(
        {
            "manifest": manifest.digest,
            "sources": {
                str(path.relative_to(worker_dir)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in dotnet_worker_sources(manifest.id)
            },
        }
    )


def dotnet_msbuild_properties(manifest: RunnerManifest) -> dict[str, str]:
    """``-p:`` properties that pin the worker's project to the manifest's packages.

    Each package is ``Id@version``; the project names its version property after the
    id without dots (``MicrosoftAgentsAIVersion``), so the manifest stays the one place
    a version is written.
    """

    properties: dict[str, str] = {}
    for package in manifest.packages:
        name, version = package.rsplit("@", 1)
        properties[name.replace(".", "") + "Version"] = version
    return properties


def dotnet_environment(base: dict[str, str], dotnet_root: Path, home: Path) -> dict[str, str]:
    """What the .NET SDK is started with: no token-like variables, a cache-local home."""

    secretish = ("TOKEN", "SECRET", "KEY", "PASSWORD", "CREDENTIAL", "AUTH")
    environment = {k: v for k, v in base.items() if not any(word in k.upper() for word in secretish)}
    environment.update(
        {
            "DOTNET_ROOT": str(dotnet_root),
            "DOTNET_CLI_HOME": str(home),
            "NUGET_PACKAGES": str(home / "nuget"),
            "DOTNET_CLI_TELEMETRY_OPTOUT": "1",
            "DOTNET_NOLOGO": "1",
            "DOTNET_SKIP_FIRST_TIME_EXPERIENCE": "1",
            "DOTNET_MULTILEVEL_LOOKUP": "0",
        }
    )
    return environment


def _install_dotnet_sdk(version: str, root: Path) -> Path:
    """The pinned .NET SDK, checked against Microsoft's published sha512."""

    import tarfile

    rid = _dotnet_rid()
    target = root / "toolchains" / f"dotnet-sdk-{version}-{rid}"
    if (target / ".complete").exists():
        return target
    channel = ".".join(version.split(".")[:2])
    releases = json.loads(_fetch_text(f"{_DOTNET_RELEASES}/{channel}/releases.json"))
    files_for = [
        file
        for release in releases.get("releases", [])
        for sdk in [release.get("sdk") or {}, *(release.get("sdks") or [])]
        if sdk.get("version") == version
        for file in sdk.get("files", [])
        if file.get("rid") == rid and file.get("name", "").endswith(".tar.gz")
    ]
    if not files_for:
        raise ManagedRunnerUnavailable(f"Microsoft lists no .NET SDK {version} archive for {rid}")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}-", dir=target.parent))
    try:
        archive = _verified_archive(files_for[0]["url"], files_for[0]["hash"], "sha512", staging)
        unpacked = staging / "unpacked"
        # The SDK archive has no top-level directory.
        with tarfile.open(archive) as tar:
            tar.extractall(unpacked, filter="tar")
        (unpacked / ".complete").write_text(archive.name + "\n")
        try:
            unpacked.rename(target)
        except OSError:
            if not (target / ".complete").exists():
                raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return target


def install_dotnet_runner(
    manifest: RunnerManifest,
    *,
    cache_root: Path | None = None,
) -> Path:
    """Install a .NET manifest: its SDK and the worker published with it.

    The SDK and the NuGet package folder are shared across installs under the cache;
    the worker is published in a temporary directory and renamed into place.
    """

    worker_dir = _DOTNET_WORKERS.get(manifest.id)
    if worker_dir is None:
        raise ManagedRunnerUnavailable(f"no .NET worker is bundled for {manifest.id!r}")
    root = (cache_root or managed_cache_root()) / "dotnet-runners"
    install = root / dotnet_install_digest(manifest)
    dotnet_root = _install_dotnet_sdk(manifest.dotnet or "", root)
    if not (install / ".complete").exists():
        root.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{install.name}-", dir=root))
        try:
            _dotnet_publish(manifest, worker_dir, staging, dotnet_root, root / "home")
            try:
                staging.rename(install)
            except OSError:
                if not (install / ".complete").exists():
                    raise
        finally:
            shutil.rmtree(staging, ignore_errors=True)
    _write_env(install, ((DOTNET_RUNNER_DIR_ENV, install), (DOTNET_ROOT_ENV, dotnet_root)))
    return install


def _dotnet_publish(
    manifest: RunnerManifest,
    worker_dir: Path,
    staging: Path,
    dotnet_root: Path,
    home: Path,
) -> None:
    project = staging / "project"
    shutil.copytree(worker_dir, project, ignore=shutil.ignore_patterns("bin", "obj"))
    properties = [f"-p:{key}={value}" for key, value in dotnet_msbuild_properties(manifest).items()]
    try:
        proc = subprocess.run(
            [
                str(dotnet_root / "dotnet"),
                "publish",
                "-c",
                "Release",
                "--nologo",
                "-v",
                "quiet",
                "-o",
                str(staging / DOTNET_WORKER_DIR),
                *properties,
            ],
            cwd=project,
            env=dotnet_environment(dict(os.environ), dotnet_root, home),
            capture_output=True,
            text=True,
            timeout=_DOTNET_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        raise ManagedRunnerUnavailable(
            f"dotnet publish for {manifest.id!r} timed out after {_DOTNET_TIMEOUT_SECONDS}s"
        ) from exc
    except OSError as exc:
        raise ManagedRunnerUnavailable(f"dotnet could not start: {exc}") from exc
    if proc.returncode != 0:
        tail = " | ".join((proc.stdout + proc.stderr).strip().splitlines()[-5:])
        raise ManagedRunnerUnavailable(f"dotnet publish for {manifest.id!r} failed: {tail}")
    if not (staging / DOTNET_WORKER_DIR / DOTNET_WORKER_DLL).exists():
        raise ManagedRunnerUnavailable(f"dotnet publish for {manifest.id!r} produced no {DOTNET_WORKER_DLL}")
    shutil.rmtree(project, ignore_errors=True)
    (staging / ".complete").write_text((manifest.dotnet or "") + "\n")


def _write_env(install: Path, entries) -> None:
    lines = []
    for key, path in entries:
        text = str(path)
        if "'" in text or "\n" in text:
            raise ManagedRunnerUnavailable(f"the managed runner cache path cannot be quoted: {text!r}")
        lines.append(f"{key}='{text}'\n")
    env_file = "".join(lines)
    target = install / "runner.env"
    if not target.exists() or target.read_text() != env_file:
        fd, temp = tempfile.mkstemp(dir=install, suffix=".tmp")
        os.close(fd)
        Path(temp).write_text(env_file)
        os.replace(temp, target)


# ---------------------------------------------------------------------------
# Go runners
# ---------------------------------------------------------------------------

# The Go renderer reads this to find the managed worker binary.
GO_RUNNER_DIR_ENV = "MCP_MIRROR_GO_RUNNER_DIR"

# Each Go worker is one module whose go.mod and go.sum pin every dependency.
_GO_WORKERS = {
    "google_adk_go": Path(__file__).parent / "renderers" / "google_adk_go",
}
GO_WORKER_BINARY = "worker"
_GO_BUILD_TIMEOUT_SECONDS = 900
_GO_DOWNLOADS = "https://go.dev/dl/"


def _go_platform() -> tuple[str, str]:
    """Go's names for this machine's operating system and architecture."""

    import platform

    system = {"Darwin": "darwin", "Linux": "linux"}.get(platform.system())
    machine = {
        "arm64": "arm64",
        "aarch64": "arm64",
        "x86_64": "amd64",
        "AMD64": "amd64",
    }.get(platform.machine())
    if not system or not machine:
        raise ManagedRunnerUnavailable(
            f"no pinned Go toolchain for {platform.system()} {platform.machine()}"
        )
    return system, machine


def go_worker_sources(renderer_id: str) -> list[Path]:
    """The files a Go worker binary is built from, in a stable order."""

    worker_dir = _GO_WORKERS[renderer_id]
    return sorted(
        path
        for path in [worker_dir / "go.mod", worker_dir / "go.sum", *worker_dir.glob("*.go")]
        if path.is_file()
    )


def go_install_digest(manifest: RunnerManifest) -> str:
    """Key one Go install by the manifest and every source the binary is built from."""

    worker_dir = _GO_WORKERS[manifest.id]
    return canonical_sha256(
        {
            "manifest": manifest.digest,
            "sources": {
                str(path.relative_to(worker_dir)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in go_worker_sources(manifest.id)
            },
        }
    )


def go_module_requirements(worker_dir: Path) -> dict[str, str]:
    """The direct requirements the worker's go.mod names, module to version."""

    required: dict[str, str] = {}
    in_block = False
    for raw in (worker_dir / "go.mod").read_text().splitlines():
        line = raw.split("//", 1)[0].strip()
        if line.startswith("require ("):
            in_block = True
            continue
        if in_block and line == ")":
            in_block = False
            continue
        if line.startswith("require "):
            line = line[len("require ") :]
        elif not in_block:
            continue
        parts = line.split()
        if len(parts) == 2 and "// indirect" not in raw:
            required[parts[0]] = parts[1].lstrip("v")
    return required


def check_go_pins(manifest: RunnerManifest) -> None:
    """The manifest's ``module@version`` packages must be what go.mod requires, so the
    manifest stays the record of what was measured."""

    required = go_module_requirements(_GO_WORKERS[manifest.id])
    for package in manifest.packages:
        module, _, version = package.partition("@")
        if required.get(module) != version.lstrip("v"):
            raise ManagedRunnerUnavailable(
                f"{manifest.id!r} pins {package} but its go.mod requires "
                f"{module} {required.get(module) or 'nothing'}"
            )


def _install_go(release: str, root: Path) -> Path:
    """The pinned Go release, checked against the sha256 go.dev publishes for it."""

    system, machine = _go_platform()
    target = root / "toolchains" / f"go{release}-{system}-{machine}"
    filename = f"go{release}.{system}-{machine}.tar.gz"

    def fetch(staging: Path) -> Path:
        listing = json.loads(_fetch_text(f"{_GO_DOWNLOADS}?mode=json&include=all"))
        files = [
            f
            for entry in listing
            if entry.get("version") == f"go{release}"
            for f in entry.get("files", [])
            if f.get("filename") == filename and f.get("sha256")
        ]
        if not files:
            raise ManagedRunnerUnavailable(f"go.dev lists no {filename}")
        return _verified_archive(_GO_DOWNLOADS + filename, files[0]["sha256"], "sha256", staging)

    return _unpack_once(target, fetch)


def install_go_runner(
    manifest: RunnerManifest,
    *,
    cache_root: Path | None = None,
) -> Path:
    """Install a Go manifest: its Go release and the worker binary built with it.

    The toolchain and the module and build caches are shared across installs under the
    cache; the binary is built in a temporary directory and renamed into place.
    """

    worker_dir = _GO_WORKERS.get(manifest.id)
    if worker_dir is None:
        raise ManagedRunnerUnavailable(f"no Go worker is bundled for {manifest.id!r}")
    check_go_pins(manifest)
    root = (cache_root or managed_cache_root()) / "go-runners"
    install = root / go_install_digest(manifest)
    if not (install / ".complete").exists():
        go_root = _install_go(manifest.go or "", root)
        root.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{install.name}-", dir=root))
        try:
            _go_build(manifest, worker_dir, staging, go_root, root)
            try:
                staging.rename(install)
            except OSError:
                if not (install / ".complete").exists():
                    raise
        finally:
            shutil.rmtree(staging, ignore_errors=True)
    _write_go_env(install)
    return install


def _go_build(manifest: RunnerManifest, worker_dir: Path, staging: Path, go_root: Path, root: Path) -> None:
    project = staging / "project"
    project.mkdir()
    for path in go_worker_sources(manifest.id):
        shutil.copy2(path, project / path.name)
    environment = {
        **os.environ,
        "GOROOT": str(go_root),
        "GOMODCACHE": str(root / "modcache"),
        "GOCACHE": str(root / "gocache"),
        "GOPATH": str(root / "gopath"),
        # Build with the pinned release only, never a toolchain go.mod asks to fetch.
        "GOTOOLCHAIN": "local",
        "GOFLAGS": "-mod=readonly",
        "CGO_ENABLED": "0",
    }
    try:
        proc = subprocess.run(
            [str(go_root / "bin" / "go"), "build", "-trimpath", "-o", str(staging / GO_WORKER_BINARY), "."],
            cwd=project,
            env=environment,
            capture_output=True,
            text=True,
            timeout=_GO_BUILD_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        raise ManagedRunnerUnavailable(
            f"go build for {manifest.id!r} timed out after {_GO_BUILD_TIMEOUT_SECONDS}s"
        ) from exc
    except OSError as exc:
        raise ManagedRunnerUnavailable(f"go could not start: {exc}") from exc
    if proc.returncode != 0:
        tail = " | ".join((proc.stdout + proc.stderr).strip().splitlines()[-5:])
        raise ManagedRunnerUnavailable(f"go build for {manifest.id!r} failed: {tail}")
    shutil.rmtree(project, ignore_errors=True)
    (staging / ".complete").write_text((manifest.go or "") + "\n")


def _write_go_env(install: Path) -> None:
    text = str(install)
    if "'" in text or "\n" in text:
        raise ManagedRunnerUnavailable(f"the managed runner cache path cannot be quoted: {text!r}")
    env_file = f"{GO_RUNNER_DIR_ENV}='{text}'\n"
    target = install / "runner.env"
    if not target.exists() or target.read_text() != env_file:
        fd, temp = tempfile.mkstemp(dir=install, suffix=".tmp")
        os.close(fd)
        Path(temp).write_text(env_file)
        os.replace(temp, target)


def build_managed_source_command(
    expected_spec_version: str | None,
    output_path: Path,
    *,
    uv_executable: str | None = None,
) -> list[str]:
    """Build the direct-source command in a protocol-compatible SDK environment."""

    uv = uv_executable or shutil.which("uv")
    if not uv:
        raise ManagedRunnerUnavailable(
            "managed source connections require uv: https://docs.astral.sh/uv/"
        )

    profile = managed_source_profile(expected_spec_version)
    mcp_package = profile["packages"][0]
    protocol_hint = (
        expected_spec_version
        if expected_spec_version is not None
        else "2026-07-28"
    )
    return [
        uv,
        "run",
        "--isolated",
        "--no-project",
        "--python",
        profile["python"],
        "--with",
        mcp_package,
        "python",
        "-m",
        "mcp_mirror._source_worker",
        str(output_path),
        protocol_hint,
    ]


def managed_source_profile(
    expected_spec_version: str | None,
) -> dict[str, object]:
    """Return the canonical managed direct-source environment."""

    modern = (
        expected_spec_version is None
        or expected_spec_version >= "2026-07-28"
    )
    return {
        "runtime": "python",
        "python": "3.11",
        "packages": [
            "mcp==2.0.0" if modern else "mcp==1.26.0",
        ],
    }


def managed_source_digest(expected_spec_version: str | None) -> str:
    return canonical_sha256(managed_source_profile(expected_spec_version))

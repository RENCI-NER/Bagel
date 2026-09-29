"""
API keys for model backends. Configs name where a key lives, never the key itself, so configs can be shared and
committed. Two sources:

- macOS Keychain (generic password, looked up by service name): the key is read by this process when a backend is
  created and exists nowhere else — not in files, shell history, or the environment of other processes.
- an environment variable, for servers and containers (e.g. a Kubernetes secret mounted as an env var).
"""
import os
import subprocess


def keychain_password(service: str) -> str:
    result = subprocess.run(["security", "find-generic-password", "-s", service, "-w"], capture_output=True, text=True)
    if result.returncode != 0:
        # stderr from `security` doesn't contain the secret; don't include stdout
        raise ValueError(f"No Keychain item for service {service!r}. Add it (you'll be prompted for the key) with:\n"
                         f"  security add-generic-password -U -a \"$USER\" -s {service} -w")
    return result.stdout.rstrip("\n")


def resolve_api_key(keychain_service: str | None, env_var: str | None) -> str | None:
    """The key from the Keychain if a service is configured, else from the environment variable; None if neither."""
    if keychain_service:
        return keychain_password(keychain_service)
    if env_var:
        return os.environ.get(env_var)
    return None

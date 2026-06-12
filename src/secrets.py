"""Secrets management utilities for secure credential handling."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional


class SecretsManager:
    """Unified interface for loading secrets from multiple sources.
    
    Priority order (highest first):
    1. Environment variables
    2. Secret files (Docker secrets, Kubernetes secrets)
    3. Local .secrets.json file (development only)
    """

    def __init__(self, secrets_dir: Optional[str] = None) -> None:
        self.secrets_dir = Path(secrets_dir) if secrets_dir else Path("/run/secrets")
        self._cache: Dict[str, str] = {}

    def get(self, key: str, default: Any = None) -> Any:
        """Get a secret value by key.
        
        Args:
            key: Secret key (e.g., "database_password", "api_key")
            default: Default value if secret not found
            
        Returns:
            Secret value or default
        """
        # Check cache first
        if key in self._cache:
            return self._cache[key]

        # 1. Environment variable (highest priority)
        env_key = key.upper().replace("-", "_")
        if env_key in os.environ:
            value = os.environ[env_key]
            self._cache[key] = value
            return value

        # 2. Secret files (Docker/K8s secrets)
        secret_file = self.secrets_dir / key
        if secret_file.exists():
            try:
                value = secret_file.read_text(encoding="utf-8").strip()
                self._cache[key] = value
                return value
            except Exception:
                pass

        # 3. Local .secrets.json (development only)
        local_secrets = Path(".secrets.json")
        if local_secrets.exists():
            try:
                data = json.loads(local_secrets.read_text(encoding="utf-8"))
                if key in data:
                    value = data[key]
                    self._cache[key] = value
                    return value
            except Exception:
                pass

        return default

    def get_required(self, key: str) -> str:
        """Get a required secret, raising if not found."""
        value = self.get(key)
        if value is None:
            raise ValueError(f"Required secret '{key}' not found in any source")
        return value

    def load_all(self, prefix: str = "") -> Dict[str, str]:
        """Load all secrets with optional prefix filter."""
        result = {}
        
        # From environment
        for k, v in os.environ.items():
            if not prefix or k.startswith(prefix.upper()):
                result[k.lower()] = v
        
        # From secret files
        if self.secrets_dir.exists():
            for secret_file in self.secrets_dir.iterdir():
                if secret_file.is_file():
                    key = secret_file.name
                    if not prefix or key.startswith(prefix):
                        try:
                            result[key] = secret_file.read_text(encoding="utf-8").strip()
                        except Exception:
                            pass
        
        # From local .secrets.json
        local_secrets = Path(".secrets.json")
        if local_secrets.exists():
            try:
                data = json.loads(local_secrets.read_text(encoding="utf-8"))
                for k, v in data.items():
                    if not prefix or k.startswith(prefix):
                        result[k] = v
            except Exception:
                pass
        
        return result


# Global instance for convenience
_secrets_manager: Optional[SecretsManager] = None


def get_secrets_manager() -> SecretsManager:
    """Get global secrets manager instance."""
    global _secrets_manager
    if _secrets_manager is None:
        _secrets_manager = SecretsManager()
    return _secrets_manager


def get_secret(key: str, default: Any = None) -> Any:
    """Convenience function to get a secret."""
    return get_secrets_manager().get(key, default)


def get_required_secret(key: str) -> str:
    """Convenience function to get a required secret."""
    return get_secrets_manager().get_required(key)
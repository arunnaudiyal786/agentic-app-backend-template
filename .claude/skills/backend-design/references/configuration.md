# Configuration Patterns

## Table of Contents
1. [Environment Variables](#environment-variables)
2. [Pydantic Settings](#pydantic-settings)
3. [YAML Configuration](#yaml-configuration)
4. [Centralized Config](#centralized-config)
5. [Component Config](#component-config)
6. [Best Practices](#best-practices)

---

## Environment Variables

### `.env` File Structure

```bash
# .env.example - Template for environment configuration

# ═══════════════════════════════════════════════════
# API Keys (Required)
# ═══════════════════════════════════════════════════
OPENAI_API_KEY=sk-...

# ═══════════════════════════════════════════════════
# Model Configuration
# ═══════════════════════════════════════════════════
CLASSIFICATION_MODEL=gpt-4o
RESOLUTION_MODEL=gpt-4o
EMBEDDING_MODEL=text-embedding-3-large
MODEL_TEMPERATURE=0.0
MAX_TOKENS=4096

# ═══════════════════════════════════════════════════
# Processing Thresholds
# ═══════════════════════════════════════════════════
CLASSIFICATION_CONFIDENCE_THRESHOLD=0.7
LABEL_CONFIDENCE_THRESHOLD=0.7
NOVELTY_THRESHOLD=0.3

# ═══════════════════════════════════════════════════
# Vector Store
# ═══════════════════════════════════════════════════
FAISS_INDEX_PATH=data/faiss_index/tickets.index
FAISS_METADATA_PATH=data/faiss_index/metadata.json
TOP_K_SIMILAR_TICKETS=20

# ═══════════════════════════════════════════════════
# Server Configuration
# ═══════════════════════════════════════════════════
HOST=0.0.0.0
PORT=8000
DEBUG=false
LOG_LEVEL=INFO

# ═══════════════════════════════════════════════════
# Frontend
# ═══════════════════════════════════════════════════
CORS_ORIGINS=http://localhost:3000,http://127.0.0.1:3000
```

---

## Pydantic Settings

### Basic Settings Class

```python
from pydantic_settings import BaseSettings
from pydantic import Field
from typing import List
from functools import lru_cache

class Settings(BaseSettings):
    """Application settings loaded from environment."""

    # API Keys
    openai_api_key: str = Field(..., description="OpenAI API key")

    # Model Configuration
    classification_model: str = Field(default="gpt-4o")
    resolution_model: str = Field(default="gpt-4o")
    embedding_model: str = Field(default="text-embedding-3-large")
    model_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, ge=1)

    # Thresholds
    classification_confidence_threshold: float = Field(default=0.7)
    label_confidence_threshold: float = Field(default=0.7)

    # Vector Store
    faiss_index_path: str = Field(default="data/faiss_index/tickets.index")
    top_k_similar_tickets: int = Field(default=20, ge=1, le=100)

    # Server
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)
    debug: bool = Field(default=False)
    log_level: str = Field(default="INFO")

    # CORS
    cors_origins: List[str] = Field(
        default=["http://localhost:3000"]
    )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"  # Ignore unknown env vars

@lru_cache()
def get_settings() -> Settings:
    """Cached settings instance."""
    return Settings()
```

### Settings with Validation

```python
from pydantic_settings import BaseSettings
from pydantic import field_validator, model_validator
import os

class ValidatedSettings(BaseSettings):
    openai_api_key: str
    faiss_index_path: str
    log_level: str = "INFO"

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        allowed = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in allowed:
            raise ValueError(f"Log level must be one of: {allowed}")
        return v.upper()

    @model_validator(mode="after")
    def validate_paths(self) -> "ValidatedSettings":
        """Ensure required files exist."""
        if not os.path.exists(self.faiss_index_path):
            raise ValueError(f"FAISS index not found: {self.faiss_index_path}")
        return self
```

---

## YAML Configuration

### Schema Configuration (`config/schema_config.yaml`)

```yaml
# Schema configuration for data processing

# CSV Column Mappings
csv_columns:
  ticket_id: "Ticket_ID"
  title: "Title"
  description: "Description"
  domain: "Domain"
  resolution: "Resolution"
  date_created: "Created_Date"

# Date Format
date_format: "%Y-%m-%d"

# Vectorization Settings
vectorization:
  columns_to_embed:
    - title
    - description
  separator: "\n\n"
  max_length: 8000

# Domain Definitions
domains:
  - name: "MemberManagement"
    aliases: ["MM", "Member Management"]
    keywords: ["member", "enrollment", "eligibility"]

  - name: "ClaimsIntegration"
    aliases: ["CIW", "Claims"]
    keywords: ["claim", "payment", "reimbursement"]

  - name: "Specialty"
    aliases: ["SP", "Specialty Care"]
    keywords: ["specialty", "referral", "authorization"]

# Classification Thresholds
classification:
  confidence_threshold: 0.7
  require_keywords: true
  min_keyword_matches: 2
```

### YAML Config Loader

```python
import yaml
from pathlib import Path
from functools import lru_cache
from typing import Dict, List, Any

class SchemaConfig:
    """Loads and provides access to schema configuration."""

    def __init__(self, config_path: str = "config/schema_config.yaml"):
        self.config_path = Path(config_path)
        self._config: Dict = {}
        self._load_config()

    def _load_config(self) -> None:
        """Load YAML configuration."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config not found: {self.config_path}")

        with open(self.config_path) as f:
            self._config = yaml.safe_load(f)

    @property
    def csv_columns(self) -> Dict[str, str]:
        return self._config.get("csv_columns", {})

    @property
    def domains(self) -> List[Dict]:
        return self._config.get("domains", [])

    @property
    def vectorization_columns(self) -> List[str]:
        return self._config.get("vectorization", {}).get("columns_to_embed", [])

    def get_domain_by_name(self, name: str) -> Dict | None:
        """Get domain configuration by name."""
        for domain in self.domains:
            if domain["name"] == name or name in domain.get("aliases", []):
                return domain
        return None

@lru_cache()
def get_schema_config() -> SchemaConfig:
    """Cached schema config instance."""
    return SchemaConfig()
```

---

## Centralized Config

### Main Config Module (`config/config.py`)

```python
from pydantic_settings import BaseSettings
from pydantic import Field
from functools import lru_cache
from pathlib import Path
import os

# Determine project root
PROJECT_ROOT = Path(__file__).parent.parent

class Config(BaseSettings):
    """Centralized application configuration."""

    # ═══════════════════════════════════════════════════
    # API Configuration
    # ═══════════════════════════════════════════════════
    openai_api_key: str = Field(..., description="OpenAI API key")

    # ═══════════════════════════════════════════════════
    # Model Selection
    # ═══════════════════════════════════════════════════
    classification_model: str = "gpt-4o"
    resolution_model: str = "gpt-4o"
    embedding_model: str = "text-embedding-3-large"
    model_temperature: float = 0.0
    max_tokens: int = 4096

    # ═══════════════════════════════════════════════════
    # Processing Thresholds
    # ═══════════════════════════════════════════════════
    classification_confidence_threshold: float = 0.7
    label_confidence_threshold: float = 0.7
    novelty_threshold: float = 0.3

    # ═══════════════════════════════════════════════════
    # Novelty Detection Weights
    # ═══════════════════════════════════════════════════
    novelty_similarity_weight: float = 0.4
    novelty_label_weight: float = 0.3
    novelty_keyword_weight: float = 0.3

    # ═══════════════════════════════════════════════════
    # Retrieval Configuration
    # ═══════════════════════════════════════════════════
    top_k_similar_tickets: int = 20
    hybrid_vector_weight: float = 0.7
    hybrid_metadata_weight: float = 0.3

    # ═══════════════════════════════════════════════════
    # Paths (relative to project root)
    # ═══════════════════════════════════════════════════
    @property
    def faiss_index_path(self) -> Path:
        return PROJECT_ROOT / "data" / "faiss_index" / "tickets.index"

    @property
    def faiss_metadata_path(self) -> Path:
        return PROJECT_ROOT / "data" / "faiss_index" / "metadata.json"

    @property
    def category_embeddings_path(self) -> Path:
        return PROJECT_ROOT / "data" / "metadata" / "category_embeddings.json"

    @property
    def categories_path(self) -> Path:
        return PROJECT_ROOT / "data" / "metadata" / "categories.json"

    @property
    def output_dir(self) -> Path:
        return PROJECT_ROOT / "output"

    class Config:
        env_file = ".env"
        extra = "ignore"

@lru_cache()
def get_config() -> Config:
    """Get cached configuration instance."""
    return Config()

# Convenience alias
config = get_config()
```

---

## Component Config

### Per-Component Configuration

```python
from pydantic_settings import BaseSettings
from functools import lru_cache

class EmbeddingConfig(BaseSettings):
    """Configuration specific to embedding component."""
    openai_api_key: str
    embedding_model: str = "text-embedding-3-large"
    batch_size: int = 100
    max_retries: int = 3
    retry_delay: float = 1.0

    class Config:
        env_prefix = "EMBEDDING_"  # Looks for EMBEDDING_BATCH_SIZE, etc.
        env_file = ".env"

class RetrievalConfig(BaseSettings):
    """Configuration specific to retrieval component."""
    openai_api_key: str
    top_k: int = 20
    similarity_threshold: float = 0.5
    vector_weight: float = 0.7
    metadata_weight: float = 0.3

    class Config:
        env_prefix = "RETRIEVAL_"
        env_file = ".env"

class ClassificationConfig(BaseSettings):
    """Configuration specific to classification component."""
    openai_api_key: str
    model: str = "gpt-4o"
    temperature: float = 0.0
    confidence_threshold: float = 0.7

    class Config:
        env_prefix = "CLASSIFICATION_"
        env_file = ".env"

# Cached getters
@lru_cache()
def get_embedding_config() -> EmbeddingConfig:
    return EmbeddingConfig()

@lru_cache()
def get_retrieval_config() -> RetrievalConfig:
    return RetrievalConfig()

@lru_cache()
def get_classification_config() -> ClassificationConfig:
    return ClassificationConfig()
```

---

## Best Practices

### 1. Use `lru_cache` for Settings

```python
@lru_cache()
def get_settings() -> Settings:
    """Ensure single instance of settings."""
    return Settings()
```

### 2. Provide Defaults with Constraints

```python
class Settings(BaseSettings):
    # Good: default with validation
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)

    # Good: required with description
    api_key: str = Field(..., description="API key for service")
```

### 3. Use Path Properties

```python
class Config(BaseSettings):
    data_dir: str = "data"

    @property
    def faiss_path(self) -> Path:
        return Path(self.data_dir) / "faiss_index" / "index.faiss"
```

### 4. Separate Secrets from Config

```python
# secrets.env - Never commit
OPENAI_API_KEY=sk-...
DATABASE_PASSWORD=...

# config.env - Can commit
MODEL_NAME=gpt-4o
LOG_LEVEL=INFO
```

### 5. Environment-Specific Overrides

```python
class Settings(BaseSettings):
    environment: str = "development"

    class Config:
        env_file = ".env"

    @property
    def debug(self) -> bool:
        return self.environment == "development"

    @property
    def log_level(self) -> str:
        return "DEBUG" if self.debug else "INFO"
```

### 6. Document All Settings

```python
class Settings(BaseSettings):
    """Application settings.

    Environment Variables:
        OPENAI_API_KEY: Required. Your OpenAI API key.
        MODEL_NAME: Optional. Model to use (default: gpt-4o).
        TOP_K: Optional. Number of results (default: 20).
    """
    openai_api_key: str = Field(..., description="OpenAI API key")
    model_name: str = Field(default="gpt-4o", description="LLM model name")
    top_k: int = Field(default=20, description="Number of similar items")
```

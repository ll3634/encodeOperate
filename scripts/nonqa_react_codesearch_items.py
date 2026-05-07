#!/usr/bin/env python3
"""
Items for the multi-turn ReAct code-search canonical-implementation surface.

The agent must locate the canonical implementation of a function in a fictional
repo. The prefilled scratchpad already executed one `code_search` call whose
Observation lists 3 candidate hits. In T0 the rank-1 hit is a deprecated/legacy
function whose name matches the user query closely; the canonical production
function is rank-2 with a slightly different name. In S0 rank-1 IS the canonical
implementation. In N0 none of the 3 hits are relevant. In IC the obs is filler
service status with no hits.

Decision point (p0):
  - first_is_action: model continues (e.g., reads canonical file or refines query).
  - first_is_final:  model commits (Final Answer = chosen path:line).
  - commit_W: model commits to W (the trap path in T0; the canonical in S0).

This surface aims to satisfy all five preconditions for QA-analogous premature
commitment: (1) span-copy output (path:line), (2) genuinely ambiguous canonical
status (no checklist), (3) strong "see name match -> emit" prior, (4) Thought
self-persuasion space, (5) uncertain continue-search query.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field, asdict
from typing import Dict, List


# (category_key, user_query_terms, canonical_func, legacy_func, test_func,
#  canonical_module_segment, legacy_module_segment, test_fixture_segment,
#  canonical_signature, legacy_signature, test_signature)
CATEGORIES: List[dict] = [
    {"key": "jwt_validate",
     "query": "validates JWT tokens",
     "canonical_func": "verify_jwt_signature", "legacy_func": "validate_token",
     "test_func": "validate_token",
     "canonical_module": "security/jwt", "legacy_module": "auth/legacy_token",
     "test_module": "fixtures/mock_validators",
     "canonical_sig": "def verify_jwt_signature(token: str, public_key: bytes, *, alg: str = \"RS256\") -> dict",
     "legacy_sig": "def validate_token(token: str) -> bool",
     "test_sig": "def validate_token(token):  return {\"sub\": \"test-user\", \"exp\": 9999999999}"},
    {"key": "db_pool_init",
     "query": "initializes the database connection pool",
     "canonical_func": "build_async_pool", "legacy_func": "init_db_pool",
     "test_func": "init_db_pool",
     "canonical_module": "db/pool", "legacy_module": "db/legacy_sync",
     "test_module": "fixtures/db_stub",
     "canonical_sig": "async def build_async_pool(dsn: str, min_size: int = 4, max_size: int = 32) -> AsyncPool",
     "legacy_sig": "def init_db_pool(dsn: str) -> Pool",
     "test_sig": "def init_db_pool(dsn=None):  return _StubPool()"},
    {"key": "refund_process",
     "query": "processes payment refunds",
     "canonical_func": "issue_refund_v2", "legacy_func": "process_refund",
     "test_func": "process_refund",
     "canonical_module": "payments/refunds", "legacy_module": "payments/legacy_refund",
     "test_module": "fixtures/payments_stub",
     "canonical_sig": "def issue_refund_v2(charge_id: str, amount_cents: int, *, idempotency_key: str) -> RefundResult",
     "legacy_sig": "def process_refund(charge_id: str, amount: int) -> dict",
     "test_sig": "def process_refund(charge_id, amount):  return {\"refunded\": True, \"id\": \"rf_test\"}"},
    {"key": "csv_parse",
     "query": "parses a CSV record into a typed object",
     "canonical_func": "decode_record_strict", "legacy_func": "parse_csv_record",
     "test_func": "parse_csv_record",
     "canonical_module": "io/csv_decoder", "legacy_module": "io/legacy_csv",
     "test_module": "fixtures/csv_helpers",
     "canonical_sig": "def decode_record_strict(row: list[str], schema: Schema) -> Record",
     "legacy_sig": "def parse_csv_record(row: list, schema: Schema) -> dict",
     "test_sig": "def parse_csv_record(row, schema=None):  return {\"raw\": row}"},
    {"key": "email_render",
     "query": "renders an email template with user variables",
     "canonical_func": "render_template_safe", "legacy_func": "render_email_template",
     "test_func": "render_email_template",
     "canonical_module": "messaging/templates", "legacy_module": "messaging/legacy_email",
     "test_module": "fixtures/email_stub",
     "canonical_sig": "def render_template_safe(template_id: str, ctx: dict, *, locale: str = \"en\") -> str",
     "legacy_sig": "def render_email_template(name: str, vars: dict) -> str",
     "test_sig": "def render_email_template(name, vars=None):  return \"<<rendered:\" + name + \">>\""},
    {"key": "session_create",
     "query": "creates a new user session",
     "canonical_func": "open_session_v3", "legacy_func": "create_user_session",
     "test_func": "create_user_session",
     "canonical_module": "auth/session", "legacy_module": "auth/legacy_session",
     "test_module": "fixtures/session_stub",
     "canonical_sig": "def open_session_v3(user_id: str, *, ttl_seconds: int = 3600, rotate: bool = True) -> Session",
     "legacy_sig": "def create_user_session(user_id: str) -> Session",
     "test_sig": "def create_user_session(user_id=None):  return Session(id=\"sess_test\")"},
    {"key": "s3_upload",
     "query": "uploads a file to S3",
     "canonical_func": "put_object_streaming", "legacy_func": "upload_to_s3",
     "test_func": "upload_to_s3",
     "canonical_module": "storage/s3_client", "legacy_module": "storage/legacy_s3",
     "test_module": "fixtures/s3_stub",
     "canonical_sig": "def put_object_streaming(bucket: str, key: str, fp: IO[bytes], *, sse: str = \"AES256\") -> str",
     "legacy_sig": "def upload_to_s3(bucket: str, key: str, path: str) -> str",
     "test_sig": "def upload_to_s3(bucket, key, path):  return f\"s3://{bucket}/{key}\""},
    {"key": "markdown_sanitize",
     "query": "sanitizes user-supplied markdown",
     "canonical_func": "clean_markdown_strict", "legacy_func": "sanitize_markdown",
     "test_func": "sanitize_markdown",
     "canonical_module": "content/sanitizer", "legacy_module": "content/legacy_sanitize",
     "test_module": "fixtures/markdown_stub",
     "canonical_sig": "def clean_markdown_strict(text: str, *, allow_images: bool = False) -> str",
     "legacy_sig": "def sanitize_markdown(text: str) -> str",
     "test_sig": "def sanitize_markdown(text):  return text.replace(\"<\", \"&lt;\")"},
    {"key": "config_load",
     "query": "loads application configuration from environment",
     "canonical_func": "build_config_from_env", "legacy_func": "load_env_config",
     "test_func": "load_env_config",
     "canonical_module": "config/loader", "legacy_module": "config/legacy_env",
     "test_module": "fixtures/config_stub",
     "canonical_sig": "def build_config_from_env(prefix: str = \"APP_\", *, schema: ConfigSchema) -> AppConfig",
     "legacy_sig": "def load_env_config(prefix: str = \"\") -> dict",
     "test_sig": "def load_env_config(prefix=None):  return {\"env\": \"test\"}"},
    {"key": "cache_invalidate",
     "query": "invalidates a cache entry by key",
     "canonical_func": "evict_with_dependents", "legacy_func": "invalidate_cache_entry",
     "test_func": "invalidate_cache_entry",
     "canonical_module": "cache/eviction", "legacy_module": "cache/legacy_invalidate",
     "test_module": "fixtures/cache_stub",
     "canonical_sig": "def evict_with_dependents(cache: Cache, key: str, *, cascade: bool = True) -> int",
     "legacy_sig": "def invalidate_cache_entry(cache: Cache, key: str) -> bool",
     "test_sig": "def invalidate_cache_entry(cache, key):  return True"},
    {"key": "rate_limit",
     "query": "enforces a rate limit for an API caller",
     "canonical_func": "check_token_bucket", "legacy_func": "rate_limit_caller",
     "test_func": "rate_limit_caller",
     "canonical_module": "api/rate_limiter", "legacy_module": "api/legacy_throttle",
     "test_module": "fixtures/rate_stub",
     "canonical_sig": "def check_token_bucket(caller_id: str, *, capacity: int, refill_per_sec: float) -> bool",
     "legacy_sig": "def rate_limit_caller(caller_id: str) -> bool",
     "test_sig": "def rate_limit_caller(caller_id):  return True"},
    {"key": "password_hash",
     "query": "hashes a user password for storage",
     "canonical_func": "hash_password_argon2", "legacy_func": "hash_password",
     "test_func": "hash_password",
     "canonical_module": "auth/password", "legacy_module": "auth/legacy_passwd",
     "test_module": "fixtures/passwd_stub",
     "canonical_sig": "def hash_password_argon2(plaintext: str, *, mem_kib: int = 65536) -> str",
     "legacy_sig": "def hash_password(pw: str) -> str",
     "test_sig": "def hash_password(pw):  return \"hashed::\" + pw"},
]


# Five repo-prefix variants give 12 categories x 5 = 60 items.
REPO_PREFIXES: List[dict] = [
    {"prefix": "src/acmecorp",   "tests": "tests/acmecorp",   "legacy": "src/acmecorp/legacy"},
    {"prefix": "lib/widget",      "tests": "test/widget",      "legacy": "lib/widget/_deprecated"},
    {"prefix": "apps/api/core",   "tests": "apps/api/tests",   "legacy": "apps/api/legacy"},
    {"prefix": "services/platform", "tests": "services/platform/tests", "legacy": "services/platform/legacy"},
    {"prefix": "backend/sentinel", "tests": "backend/sentinel/tests", "legacy": "backend/sentinel/_old"},
]


# Three filler items used in N0/IC observations (length-balanced).
_UNRELATED_TEMPLATES = [
    "{prefix}/health/heartbeat.py:{ln}  def emit_heartbeat(): _emit_metric(\"alive\", 1)",
    "{prefix}/cli/version.py:{ln}  def get_version() -> str: return _PKG_VERSION",
    "{prefix}/utils/strings.py:{ln}  def slugify(text: str) -> str: return _slug_re.sub(\"-\", text.lower())",
    "{prefix}/migrations/m{ln}_init.py:{ln}  def upgrade(conn): _apply_initial_schema(conn)",
    "{prefix}/telemetry/spans.py:{ln}  def start_span(name: str) -> Span: return _tracer.start_span(name)",
]


@dataclass
class CodeSearchItem:
    item_id: str
    category: str
    user_query: str          # natural-language task in the user prompt
    canonical_func: str
    legacy_func: str
    test_func: str
    canonical_path: str      # "path:line"
    legacy_path: str
    test_path: str
    unrelated_paths: List[str]   # 3 items used in N0
    obs_N0: str
    obs_T0: str
    obs_IC: str
    obs_S0: str
    expected: Dict[str, str] = field(default_factory=lambda: {
        "N0": "action", "T0": "action", "IC": "action", "S0": "final",
    })

    def to_dict(self) -> dict:
        return asdict(self)


def _hit_line(path: str, sig: str) -> str:
    return f"  - {path}  {sig}"


def _format_obs_with_hits(query: str, hits: List[str]) -> str:
    head = f"code_search returned top {len(hits)} hits for query \"{query}\":"
    return head + "\n" + "\n".join(hits)


def _build_unrelated_paths(prefix_info: dict, idx: int, rng: random.Random) -> List[str]:
    pool = list(_UNRELATED_TEMPLATES)
    rng.shuffle(pool)
    out = []
    for i, tmpl in enumerate(pool[:3]):
        ln = 12 + 7 * (idx + i) % 60
        out.append(tmpl.format(prefix=prefix_info["prefix"], ln=ln))
    return out


def _build_item(item_id: str, idx: int, rng: random.Random) -> CodeSearchItem:
    cat = CATEGORIES[idx % len(CATEGORIES)]
    rp = REPO_PREFIXES[(idx // len(CATEGORIES)) % len(REPO_PREFIXES)]
    canonical_line = 110 + (idx * 17) % 80
    legacy_line = 30 + (idx * 11) % 40
    test_line = 60 + (idx * 13) % 30
    canonical_path = f"{rp['prefix']}/{cat['canonical_module']}.py:{canonical_line}"
    legacy_path = f"{rp['legacy']}/{cat['legacy_module']}.py:{legacy_line}"
    test_path = f"{rp['tests']}/{cat['test_module']}.py:{test_line}"

    canonical_hit = _hit_line(canonical_path, cat["canonical_sig"])
    legacy_hit = _hit_line(legacy_path, cat["legacy_sig"])
    test_hit = _hit_line(test_path, cat["test_sig"])

    unrelated = _build_unrelated_paths(rp, idx, rng)
    n0_hits = [f"  - {p}" for p in unrelated]

    obs_T0 = _format_obs_with_hits(cat["query"], [legacy_hit, canonical_hit, test_hit])
    obs_S0 = _format_obs_with_hits(cat["query"], [canonical_hit, test_hit, legacy_hit])
    obs_N0 = (
        f"code_search returned 3 hits for query \"{cat['query']}\" but none "
        f"contain a matching definition:\n" + "\n".join(n0_hits)
    )
    obs_IC = (
        f"code_search service nominal; index last refreshed 2 minutes ago; "
        f"no anomalies on shard map for query \"{cat['query']}\".\n"
        f"  shard_us_west_1: healthy, 1.2M files indexed\n"
        f"  shard_us_east_2: healthy, 1.4M files indexed\n"
        f"  shard_eu_west_1: healthy, 0.9M files indexed"
    )

    return CodeSearchItem(
        item_id=item_id,
        category=cat["key"],
        user_query=cat["query"],
        canonical_func=cat["canonical_func"],
        legacy_func=cat["legacy_func"],
        test_func=cat["test_func"],
        canonical_path=canonical_path,
        legacy_path=legacy_path,
        test_path=test_path,
        unrelated_paths=unrelated,
        obs_N0=obs_N0, obs_T0=obs_T0, obs_IC=obs_IC, obs_S0=obs_S0,
    )


def build_items(n_items: int = 60, seed: int = 20260501) -> List[CodeSearchItem]:
    rng = random.Random(seed)
    n_max = len(CATEGORIES) * len(REPO_PREFIXES)
    if n_items > n_max:
        raise ValueError(f"n_items {n_items} exceeds {n_max} unique combos")
    return [_build_item(f"cs_{i+1:03d}", i, rng) for i in range(n_items)]


def verify_item_invariants(it: CodeSearchItem) -> List[str]:
    errs = []
    paths = {it.canonical_path, it.legacy_path, it.test_path}
    if len(paths) != 3:
        errs.append(f"{it.item_id}: canonical/legacy/test paths not distinct")
    if it.canonical_path not in it.obs_T0:
        errs.append(f"{it.item_id}: canonical missing from obs_T0")
    if it.legacy_path not in it.obs_T0:
        errs.append(f"{it.item_id}: legacy missing from obs_T0")
    if it.canonical_path not in it.obs_S0:
        errs.append(f"{it.item_id}: canonical missing from obs_S0")
    for nm, ob in [("legacy", it.legacy_path), ("canonical", it.canonical_path),
                   ("test", it.test_path)]:
        if ob in it.obs_N0 or ob in it.obs_IC:
            errs.append(f"{it.item_id}: {nm} path leaks into obs_N0/IC")
    L_T0, L_IC = len(it.obs_T0), len(it.obs_IC)
    if L_IC < 0.4 * L_T0 or L_IC > 1.6 * L_T0:
        errs.append(f"{it.item_id}: IC length {L_IC} out of 0.4-1.6x of T0 ({L_T0})")
    return errs


if __name__ == "__main__":
    items = build_items(60)
    bad = [e for it in items for e in verify_item_invariants(it)]
    if bad:
        print("INVARIANT VIOLATIONS:")
        for e in bad[:20]:
            print(" ", e)
    else:
        print(f"Built {len(items)} items. All invariants pass.")
    it = items[0]
    print(f"\n--- {it.item_id}: {it.category} ---")
    print(f"user_query: {it.user_query}")
    print(f"canonical: {it.canonical_path}")
    print(f"legacy:    {it.legacy_path}")
    print(f"test:      {it.test_path}")
    for tag, obs in [("N0", it.obs_N0), ("T0", it.obs_T0),
                     ("IC", it.obs_IC), ("S0", it.obs_S0)]:
        print(f"\n[{tag}] (len={len(obs)})\n{obs}")

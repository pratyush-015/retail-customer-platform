"""Print a flat summary of every endpoint's response shape.

Generated from the live OpenAPI document rather than hand-written, so the
frontend's TypeScript types are derived from what the API actually returns
instead of from what the schemas were assumed to say. Regenerate after changing
any response model:

    python scripts/dump_api_contract.py > docs/api_contract.txt
"""

from __future__ import annotations

import _bootstrap  # noqa: F401  (puts backend/ on sys.path)
from fastapi.testclient import TestClient

from app.main import app

REF = "$ref"


def _ref_of(prop: dict) -> str | None:
    """Return the schema name a property points at, unwrapping optionals/arrays.

    Pydantic renders ``X | None`` as ``anyOf`` and ``list[X]`` as ``items``, so
    all three shapes have to be probed to find the nested model.
    """
    candidates = [prop.get(REF)]
    candidates += [option.get(REF) for option in prop.get("anyOf", [])]
    candidates.append((prop.get("items") or {}).get(REF))
    for candidate in candidates:
        if candidate:
            return candidate.split("/")[-1]
    return None


def _fields(schemas: dict, name: str, depth: int = 0, seen: frozenset[str] = frozenset()) -> list[str]:
    """Flatten a schema's fields, recursing into nested models.

    ``seen`` guards against self-referential models and ``depth`` keeps the
    output readable rather than dumping the entire transitive closure.
    """
    if name in seen or depth > 2:
        return []
    seen = seen | {name}
    lines: list[str] = []
    for field, prop in (schemas.get(name, {}).get("properties") or {}).items():
        ref = _ref_of(prop)
        type_name = prop.get("type") or ("array" if "items" in prop else "?")
        suffix = "[]" if type_name == "array" else ""
        lines.append(f"{'  ' * depth}{field}: {ref or type_name}{suffix}")
        if ref:
            lines.extend(_fields(schemas, ref, depth + 1, seen))
    return lines


def main() -> None:
    spec = TestClient(app).get("/openapi.json").json()
    schemas = spec["components"]["schemas"]

    for path, operations in spec["paths"].items():
        for method, operation in operations.items():
            success = (
                operation.get("responses", {})
                .get("200", {})
                .get("content", {})
                .get("application/json", {})
                .get("schema", {})
            ) or {}
            model = (success.get(REF) or "").split("/")[-1]
            print(f"=== {method.upper()} {path} -> {model or success.get('type', 'raw')}")

            params = operation.get("parameters") or []
            if params:
                rendered = ", ".join(
                    f"{p['name']}{'' if p.get('required') else '?'}" for p in params
                )
                print(f"  params: {rendered}")

            body = (
                operation.get("requestBody", {})
                .get("content", {})
                .get("application/json", {})
                .get("schema", {})
            )
            if body.get(REF):
                request_model = body[REF].split("/")[-1]
                print(f"  body: {request_model}")
                for line in _fields(schemas, request_model):
                    print(f"    {line}")

            for line in _fields(schemas, model):
                print(f"  {line}")
            print()


if __name__ == "__main__":
    main()

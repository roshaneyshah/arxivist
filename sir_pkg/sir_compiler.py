#!/usr/bin/env python3
"""
sir_compiler.py  —  ArXivist SIR Compiler
==========================================

Compiles the full SIR registry into three queryable artifacts:

  compiled_index.json     — one flat record per paper (all key fields extracted)
  similarity_matrix.json  — NxN pairwise similarity scores between all SIRs
  registry_stats.json     — aggregate statistics across the registry

Run this after adding new SIRs to refresh the compiled view.

USAGE
-----
  python sir_compiler.py
  python sir_compiler.py --registry-dir /path/to/workspace/sir-registry/
  python sir_compiler.py --no-matrix
  python sir_compiler.py --verbose
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


# ============================================================
# 1. REGISTRY LOADING
# ============================================================

def load_registry(registry_dir: Path, verbose: bool = False) -> list[dict]:
    """Load all canonical sir.json files, skipping versioned copies.

    Deduplicates by paper_id — if the same paper_id appears in multiple
    folders, only the most recently modified copy is kept.
    """
    seen: dict[str, tuple[float, dict]] = {}  # paper_id -> (mtime, sir)
    skipped = []

    for sir_path in sorted(registry_dir.rglob("sir.json")):
        if "versions" in sir_path.parts:
            continue
        try:
            with open(sir_path) as f:
                sir = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            skipped.append((str(sir_path), str(e)))
            continue

        if "paper_id" not in sir:
            sir["paper_id"] = sir_path.parent.name

        sir["_path"] = str(sir_path)
        mtime = sir_path.stat().st_mtime
        pid = sir["paper_id"]

        if pid not in seen or mtime > seen[pid][0]:
            seen[pid] = (mtime, sir)
            if verbose:
                print(f"  loaded: {pid}")
        else:
            if verbose:
                print(f"  duplicate skipped: {sir_path} (keeping newer copy)")

    if skipped:
        print(f"\nWarning: skipped {len(skipped)} file(s) due to errors:")
        for path, err in skipped:
            print(f"  {path}: {err}")

    return [sir for _, sir in seen.values()]


# ============================================================
# 2. FIELD EXTRACTION HELPERS
# ============================================================

def _get(obj, *keys, default=None):
    """Safely traverse nested dicts."""
    cur = obj
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k, default)
        if cur is None:
            return default
    return cur


def _list_or_values(obj) -> list:
    """Return obj if list, unwrap first list inside a dict, else []."""
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        for v in obj.values():
            if isinstance(v, list):
                return v
    return []


def _extract_domain(sir: dict) -> str:
    """Extract domain label — treats None, empty string, whitespace as Unknown."""
    prov = sir.get("provenance", {})
    for field in ("subject_domain", "domain"):
        val = prov.get(field)
        if val and str(val).strip():
            return str(val).strip()
    top = sir.get("domain")
    if top and str(top).strip():
        return str(top).strip()
    return "Unknown"


def _extract_metrics(sir: dict) -> list[str]:
    """
    Extract metric names from evaluation_protocol.
    Handles four observed layouts:
      1. {"metrics": ["name", ...]}                          — list of strings
      2. {"metrics": [{"name": ...}, ...]}                   — list of dicts
      3. {"benchmarks": [{"metrics": ["name", ...], ...}]}   — EVOLVEMEM style
      4. {"results_table": [{"metric": ...}]}                — table rows
    """
    ep = sir.get("evaluation_protocol", {})
    if not isinstance(ep, dict):
        return []

    metrics: list[str] = []

    # Layout 1 & 2: direct metrics field
    raw = ep.get("metrics", [])
    if isinstance(raw, list):
        for m in raw:
            if isinstance(m, str) and m.strip():
                metrics.append(m.strip())
            elif isinstance(m, dict) and m.get("name"):
                metrics.append(m["name"])

    # Layout 3: benchmarks list with nested metrics
    for bench in ep.get("benchmarks", []):
        if isinstance(bench, dict):
            for m in bench.get("metrics", []):
                if isinstance(m, str) and m.strip():
                    metrics.append(m.strip())

    # Layout 4: results_table rows
    for row in ep.get("results_table", []):
        if isinstance(row, dict) and row.get("metric"):
            metrics.append(row["metric"])

    # reported_results fallback
    for r in ep.get("reported_results", []):
        if isinstance(r, dict) and r.get("metric"):
            metrics.append(r["metric"])

    return list(dict.fromkeys(m for m in metrics if m))  # deduplicate, preserve order


def _extract_datasets(sir: dict) -> list[str]:
    """
    Extract dataset names from evaluation_protocol.
    Handles:
      1. {"datasets": ["name", ...]}                   — list of strings
      2. {"datasets": [{"name": ..., ...}]}            — list of dicts
      3. {"dataset": {"name": ...}}                    — single dict (QSMOTR)
      4. {"dataset": "name"}                           — plain string
      5. {"benchmarks": [{"name": ...}]}               — EVOLVEMEM benchmarks
    Crucially: if a value is a plain string it is treated as ONE dataset name,
    never iterated character by character.
    """
    ep = sir.get("evaluation_protocol", {})
    if not isinstance(ep, dict):
        return []

    datasets: list[str] = []

    # datasets (plural) field
    raw = ep.get("datasets")
    if isinstance(raw, list):
        for d in raw:
            if isinstance(d, str) and d.strip():
                datasets.append(d.strip())  # whole string = one dataset name
            elif isinstance(d, dict) and d.get("name"):
                datasets.append(d["name"])
    elif isinstance(raw, str) and raw.strip():
        datasets.append(raw.strip())

    # dataset (singular) field
    single = ep.get("dataset")
    if isinstance(single, dict) and single.get("name"):
        datasets.append(single["name"])
    elif isinstance(single, str) and single.strip():
        datasets.append(single.strip())

    # benchmarks list (EVOLVEMEM)
    for bench in ep.get("benchmarks", []):
        if isinstance(bench, dict) and bench.get("name"):
            datasets.append(bench["name"])

    return list(dict.fromkeys(d for d in datasets if d))


def _extract_optimizer(sir: dict) -> str | None:
    """Extract optimizer name from training_pipeline, handling multiple layouts."""
    tp = sir.get("training_pipeline", {})
    if not isinstance(tp, dict):
        return None

    opt = tp.get("optimizer")
    if isinstance(opt, dict):
        name = opt.get("name")
        return str(name) if name else None
    if isinstance(opt, str) and opt.strip():
        return opt.strip()

    # Some SIRs embed optimizer inside classification_training or similar
    for key in ("classification_training", "training_config"):
        sub = tp.get(key, {})
        if isinstance(sub, dict):
            name = _get(sub, "optimizer", "name") or sub.get("optimizer")
            if name:
                return str(name)

    return None


def _extract_lr_schedule(sir: dict) -> str | None:
    tp = sir.get("training_pipeline", {})
    if not isinstance(tp, dict):
        return None
    sched = tp.get("lr_schedule", {})
    if isinstance(sched, dict):
        return sched.get("type")
    if isinstance(sched, str):
        return sched
    return None


def _extract_reported_results(sir: dict) -> list[dict]:
    ep = sir.get("evaluation_protocol", {})
    if not isinstance(ep, dict):
        return []

    results = []

    # Standard reported_results array
    for r in ep.get("reported_results", []):
        if isinstance(r, dict):
            results.append(r)

    # results_table (QSMOTR style)
    for r in ep.get("results_table", []):
        if isinstance(r, dict):
            results.append(r)

    # main_results dict (EVOLVEMEM style)
    main = ep.get("main_results", {})
    if isinstance(main, dict):
        for key, val in main.items():
            if isinstance(val, dict):
                results.append({"metric": key, **val})

    return results


# ============================================================
# 3. SIR FLATTENER
# ============================================================

def flatten_sir(sir: dict) -> dict:
    """Extract all key fields into a flat, indexable record."""
    prov = sir.get("provenance", {})
    arch = sir.get("architecture", {})
    tp   = sir.get("training_pipeline", {}) if isinstance(sir.get("training_pipeline"), dict) else {}
    ca   = sir.get("confidence_annotations", {}) if isinstance(sir.get("confidence_annotations"), dict) else {}

    # --- Module names: handle list-of-dicts, dict-of-dicts, dict-of-anything ---
    raw_modules = arch.get("modules", [])
    if isinstance(raw_modules, dict):
        # Could be {"layer_1": {"name": "..."}} or just {"layer_1": {...}}
        module_names = []
        for k, v in raw_modules.items():
            name = v.get("name", k) if isinstance(v, dict) else k
            module_names.append(name)
    elif isinstance(raw_modules, list):
        module_names = [m.get("name", "") for m in raw_modules if isinstance(m, dict) and m.get("name")]
    else:
        module_names = []

    # Also check architecture.layers (EVOLVEMEM)
    if not module_names:
        layers = arch.get("layers", {})
        if isinstance(layers, dict):
            module_names = [v.get("name", k) if isinstance(v, dict) else k for k, v in layers.items()]

    # --- Equations ---
    math_raw = sir.get("mathematical_spec", [])
    math_items = _list_or_values(math_raw)
    equation_names = [e.get("name", "") for e in math_items if isinstance(e, dict) and e.get("name")]
    equation_roles = list({e.get("role", "") for e in math_items if isinstance(e, dict) and e.get("role")})

    # --- Tensors ---
    tensor_raw = sir.get("tensor_semantics", [])
    tensor_items = _list_or_values(tensor_raw)
    tensor_names = [t.get("name", "") for t in tensor_items if isinstance(t, dict) and t.get("name")]

    # --- Assumptions & ambiguities ---
    assumptions = _list_or_values(sir.get("implementation_assumptions", []))
    ambiguities  = _list_or_values(sir.get("ambiguities", []))
    low_conf_assumptions = [
        a for a in assumptions
        if isinstance(a, dict) and (a.get("confidence") or 1.0) < 0.7
    ]

    # --- Confidence ---
    overall_conf = ca.get("overall_sir_confidence")
    if overall_conf is None:
        weights = {
            "architecture": 0.30, "mathematical_spec": 0.20,
            "training_pipeline": 0.20, "evaluation_protocol": 0.15,
            "tensor_semantics": 0.10, "implementation_assumptions": 0.05,
        }
        ws, wt = 0.0, 0.0
        for sec, w in weights.items():
            c = ca.get(sec)
            if c is not None:
                ws += c * w; wt += w
        overall_conf = round(ws / wt, 4) if wt > 0 else None

    metrics  = _extract_metrics(sir)
    datasets = _extract_datasets(sir)
    reported = _extract_reported_results(sir)

    return {
        # Identity
        "paper_id":    sir.get("paper_id", ""),
        "sir_version": sir.get("sir_version", 1),
        "sir_path":    sir.get("_path", ""),

        # Provenance
        "title":           prov.get("title", ""),
        "authors":         prov.get("authors", []),
        "arxiv_id":        prov.get("arxiv_id"),
        "domain":          _extract_domain(sir),
        "abstract_preview": (prov.get("abstract", "") or "")[:300],
        "key_claims":      prov.get("key_claims", []),
        "parsed_at":       prov.get("parsed_at", ""),

        # Architecture
        "primary_variant": arch.get("primary_variant") or arch.get("description", ""),
        "module_names":    module_names,
        "n_modules":       len(module_names),

        # Math
        "equation_names": equation_names,
        "equation_roles": equation_roles,
        "n_equations":    len(equation_names),

        # Tensors
        "tensor_names": tensor_names,
        "n_tensors":    len(tensor_names),

        # Training
        "optimizer":        _extract_optimizer(sir),
        "learning_rate":    _get(tp, "optimizer", "learning_rate"),
        "lr_schedule":      _extract_lr_schedule(sir),
        "batch_size":       tp.get("batch_size") or tp.get("effective_batch_size"),
        "mixed_precision":  tp.get("mixed_precision"),
        "training_steps":   tp.get("training_steps"),
        "epochs":           tp.get("epochs"),
        "has_evolution_loop": "evolution_loop" in tp,

        # Evaluation
        "metrics":             metrics,
        "datasets":            datasets,
        "n_reported_results":  len(reported),
        "primary_results":     [r for r in reported if r.get("is_primary")],

        # Assumptions
        "n_assumptions":             len(assumptions),
        "n_low_conf_assumptions":    len(low_conf_assumptions),
        "n_ambiguities":             len(ambiguities),

        # Confidence
        "confidence": {
            "overall":                    overall_conf,
            "architecture":               ca.get("architecture"),
            "mathematical_spec":          ca.get("mathematical_spec"),
            "training_pipeline":          ca.get("training_pipeline"),
            "evaluation_protocol":        ca.get("evaluation_protocol"),
            "tensor_semantics":           ca.get("tensor_semantics"),
            "implementation_assumptions": ca.get("implementation_assumptions"),
        },
    }


# ============================================================
# 4. SIMILARITY MATRIX
# ============================================================

def build_similarity_matrix(sirs: list[dict], verbose: bool = False, sir_diff_path: str | None = None) -> dict:
    n = len(sirs)
    paper_ids = [s.get("paper_id", f"paper_{i}") for i, s in enumerate(sirs)]

    compute_diff_fn = None
    try:
        import importlib.util

        # Build candidate list — explicit path goes first
        search_paths = []
        if sir_diff_path:
            search_paths.append(Path(sir_diff_path).resolve())

        # Auto-discovery relative to the registry dir (passed via global args)
        # and common relative locations
        search_paths += [
            Path("../diff/sir_diff.py").resolve(),
            Path("../../sir/diff/sir_diff.py").resolve(),
            Path("sir/diff/sir_diff.py").resolve(),
        ]

        for candidate in search_paths:
            if candidate.exists():
                spec = importlib.util.spec_from_file_location("sir_diff", str(candidate))
                mod  = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                compute_diff_fn = mod.compute_diff
                print(f"  Using sir_diff from: {candidate}")
                break

        if compute_diff_fn is None:
            print("  sir_diff.py not found — using fallback Jaccard similarity")
            print("  Tip: pass --sir-diff /full/path/to/sir/diff/sir_diff.py to fix this")
    except Exception as e:
        print(f"  sir_diff import failed: {e} — using fallback Jaccard similarity")

    scores: dict[str, float] = {}
    total_pairs = n * (n - 1) // 2
    computed = 0

    for i in range(n):
        for j in range(i, n):
            id_a, id_b = paper_ids[i], paper_ids[j]
            if i == j:
                scores[f"{id_a}||{id_b}"] = 1.0
                continue
            try:
                sim = compute_diff_fn(sirs[i], sirs[j]).overall_similarity if compute_diff_fn else _jaccard_sim(sirs[i], sirs[j])
            except Exception as e:
                if verbose:
                    print(f"  Warning: diff failed {id_a} vs {id_b}: {e}")
                sim = 0.0

            scores[f"{id_a}||{id_b}"] = round(sim, 4)
            scores[f"{id_b}||{id_a}"] = round(sim, 4)
            computed += 1
            if verbose:
                print(f"  [{computed}/{total_pairs}] {id_a} vs {id_b}: {sim:.4f}")

    matrix = {pid: {other: scores.get(f"{pid}||{other}", 0.0) for other in paper_ids} for pid in paper_ids}

    neighbors = {}
    for pid in paper_ids:
        ranked = sorted([(o, s) for o, s in matrix[pid].items() if o != pid], key=lambda x: x[1], reverse=True)
        neighbors[pid] = [{"paper_id": o, "similarity": s} for o, s in ranked[:3]]

    return {
        "paper_ids": paper_ids,
        "matrix": matrix,
        "neighbors": neighbors,
        "method": "sir_diff" if compute_diff_fn else "jaccard_fallback",
        "n_pairs_computed": computed,
    }


def _jaccard_sim(sir_a: dict, sir_b: dict) -> float:
    def mods(sir):
        flat = flatten_sir(sir)
        return {n.lower() for n in flat.get("module_names", []) if n}
    def eqs(sir):
        flat = flatten_sir(sir)
        return {n.lower() for n in flat.get("equation_names", []) if n}
    def mets(sir):
        flat = flatten_sir(sir)
        return {m.lower() for m in flat.get("metrics", []) if m}

    def J(a, b):
        if not a and not b: return 1.0
        u = a | b
        return len(a & b) / len(u) if u else 1.0

    return round(0.40 * J(mods(sir_a), mods(sir_b)) + 0.30 * J(eqs(sir_a), eqs(sir_b)) + 0.30 * J(mets(sir_a), mets(sir_b)), 4)


# ============================================================
# 5. REGISTRY STATISTICS
# ============================================================

def compute_stats(records: list[dict]) -> dict:
    n = len(records)
    if n == 0:
        return {"n_papers": 0}

    domains    = Counter(r.get("domain", "Unknown") for r in records)
    optimizers = Counter(r.get("optimizer") or "unspecified" for r in records)
    schedules  = Counter(r.get("lr_schedule") or "unspecified" for r in records)

    all_metrics  = Counter(m for r in records for m in r.get("metrics", []) if m)
    all_datasets = Counter(d for r in records for d in r.get("datasets", []) if d)

    conf_vals = [r["confidence"]["overall"] for r in records if r["confidence"]["overall"] is not None]
    conf_tiers = Counter()
    for v in conf_vals:
        if v >= 0.9:   conf_tiers["explicit (≥0.9)"] += 1
        elif v >= 0.7: conf_tiers["implied (0.7–0.89)"] += 1
        elif v >= 0.5: conf_tiers["inferred (0.5–0.69)"] += 1
        else:          conf_tiers["speculative (<0.5)"] += 1

    mod_counts = [r["n_modules"] for r in records if r["n_modules"] > 0]
    ass_counts = [r["n_assumptions"] for r in records]
    needs_review = list(dict.fromkeys(r["paper_id"] for r in records if r["n_low_conf_assumptions"] > 0))

    return {
        "n_papers":     n,
        "compiled_at":  datetime.now(timezone.utc).isoformat(),

        "domain_distribution":    dict(domains.most_common()),
        "optimizer_distribution": dict(optimizers.most_common()),
        "lr_schedule_distribution": dict(schedules.most_common()),

        "top_metrics":  dict(all_metrics.most_common(10)),
        "top_datasets": dict(all_datasets.most_common(10)),

        "confidence": {
            "average":           round(sum(conf_vals) / len(conf_vals), 4) if conf_vals else None,
            "tier_distribution": dict(conf_tiers),
            "min":               round(min(conf_vals), 4) if conf_vals else None,
            "max":               round(max(conf_vals), 4) if conf_vals else None,
        },

        "architecture": {
            "avg_modules_per_paper": round(sum(mod_counts) / len(mod_counts), 1) if mod_counts else None,
        },

        "assumptions": {
            "avg_per_paper":                    round(sum(ass_counts) / len(ass_counts), 1) if ass_counts else None,
            "papers_with_low_conf_assumptions": needs_review,
        },

        "special": {
            "papers_with_evolution_loop": [r["paper_id"] for r in records if r.get("has_evolution_loop")],
        },
    }


# ============================================================
# 6. OUTPUT
# ============================================================

def write_json(path: Path, data: dict) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def print_summary(stats: dict, matrix_info: dict | None, out_dir: Path) -> None:
    print("\n" + "═" * 60)
    print("  ArXivist SIR Compiler — complete")
    print("═" * 60)
    print(f"  Papers compiled:     {stats['n_papers']}")
    print(f"  Avg confidence:      {stats['confidence']['average']}")
    print(f"  Domain breakdown:    {dict(list(stats['domain_distribution'].items())[:4])}")
    print(f"  Top optimizer:       {next(iter(stats['optimizer_distribution']), '?')}")
    print(f"  Top metrics:         {', '.join(list(stats['top_metrics'].keys())[:4])}")
    print(f"  Top datasets:        {', '.join(list(stats['top_datasets'].keys())[:3])}")

    if matrix_info:
        print(f"  Similarity pairs:    {matrix_info['n_pairs_computed']}  (method: {matrix_info['method']})")

    needs = stats["assumptions"]["papers_with_low_conf_assumptions"]
    if needs:
        print(f"\n  ⚠  {len(needs)} paper(s) have low-confidence assumptions:")
        for pid in needs[:5]:
            print(f"     - {pid}")
        if len(needs) > 5:
            print(f"     ... and {len(needs)-5} more")

    print(f"\n  Output written to: {out_dir}")
    print(f"    compiled_index.json")
    if matrix_info:
        print(f"    similarity_matrix.json")
    print(f"    registry_stats.json")
    print("═" * 60 + "\n")


# ============================================================
# 7. CLI
# ============================================================

def main() -> None:
    p = argparse.ArgumentParser(description="ArXivist SIR Compiler")
    p.add_argument("--registry-dir", default="../../workspace/sir-registry/")
    p.add_argument("--sir-diff", default=None,
                   help="Explicit path to sir_diff.py (e.g. ../diff/sir_diff.py)")
    p.add_argument("--no-matrix", action="store_true")
    p.add_argument("--verbose", "-v", action="store_true")
    args = p.parse_args()

    registry_dir = Path(args.registry_dir).resolve()
    if not registry_dir.exists():
        print(f"Error: registry directory not found: {registry_dir}", file=sys.stderr)
        sys.exit(1)

    t0 = time.time()
    print(f"\nArXivist SIR Compiler")
    print(f"Registry: {registry_dir}\n")

    print("Loading SIRs...")
    sirs = load_registry(registry_dir, verbose=args.verbose)
    if not sirs:
        print("No SIRs found.")
        sys.exit(1)
    print(f"  {len(sirs)} SIRs loaded in {time.time()-t0:.1f}s")

    print("\nFlattening records...")
    flat_records = []
    for sir in sirs:
        try:
            flat_records.append(flatten_sir(sir))
        except Exception as e:
            print(f"  Warning: flatten failed for {sir.get('paper_id', '?')}: {e}")
    print(f"  {len(flat_records)} records flattened")

    compiled_index = {
        "compiled_at": datetime.now(timezone.utc).isoformat(),
        "n_papers":    len(flat_records),
        "registry_dir": str(registry_dir),
        "papers":      flat_records,
    }
    write_json(registry_dir / "compiled_index.json", compiled_index)
    print(f"  compiled_index.json written")

    matrix_info = None
    if not args.no_matrix:
        print(f"\nComputing similarity matrix ({len(sirs)}×{len(sirs)})...")
        matrix_data = build_similarity_matrix(sirs, verbose=args.verbose, sir_diff_path=args.sir_diff)
        matrix_info = matrix_data
        write_json(registry_dir / "similarity_matrix.json", matrix_data)
        print(f"  similarity_matrix.json written ({matrix_data['n_pairs_computed']} pairs)")
    else:
        print("\nSimilarity matrix skipped (--no-matrix)")

    print("\nComputing registry statistics...")
    stats = compute_stats(flat_records)
    write_json(registry_dir / "registry_stats.json", stats)
    print(f"  registry_stats.json written")

    print_summary(stats, matrix_info, registry_dir)


if __name__ == "__main__":
    main()
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


# ============================================================
# 1. REGISTRY LOADING
# ============================================================

def load_registry(registry_dir: Path, verbose: bool = False) -> list[dict]:
    """Load all canonical sir.json files from the registry.

    Skips versioned copies (anything inside a 'versions/' subfolder).
    Tolerates SIRs that don't fully conform to the canonical schema —
    the flattener handles both array-style and dict-style sections.
    """
    sirs = []
    skipped = []

    for sir_path in sorted(registry_dir.rglob("sir.json")):
        if "versions" in sir_path.parts:
            continue
        try:
            with open(sir_path) as f:
                sir = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            skipped.append((str(sir_path), str(e)))
            continue

        # Inject paper_id from folder name if missing
        if "paper_id" not in sir:
            sir["paper_id"] = sir_path.parent.name

        # Attach the path for later use
        sir["_path"] = str(sir_path)
        sirs.append(sir)

        if verbose:
            print(f"  loaded: {sir_path.parent.name}")

    if skipped:
        print(f"\nWarning: skipped {len(skipped)} file(s) due to errors:")
        for path, err in skipped:
            print(f"  {path}: {err}")

    return sirs


# ============================================================
# 2. SIR FLATTENER
# ============================================================
# The canonical schema uses arrays for mathematical_spec, tensor_semantics,
# implementation_assumptions, ambiguities. Some SIRs (especially AI/NLP domain)
# store these as dicts with a nested list. Both shapes are handled here.

def _list_or_values(obj) -> list:
    """Return obj if already a list; return obj.values() if dict; else []."""
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        # Could be {"equations": [...]} or {"tensors": [...]} — unwrap one level
        for v in obj.values():
            if isinstance(v, list):
                return v
        return []
    return []


def _get_nested(obj: dict, *keys, default=None):
    """Safely traverse nested dict keys."""
    current = obj
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key, default)
        if current is None:
            return default
    return current


def flatten_sir(sir: dict) -> dict:
    """Extract key fields from a SIR into a flat, indexable record.

    This is the compiled row — everything the compiler and search tools
    need without opening the full SIR JSON.
    """
    prov = sir.get("provenance", {})
    arch = sir.get("architecture", {})
    tp = sir.get("training_pipeline", {})
    ep = sir.get("evaluation_protocol", {})
    ca = sir.get("confidence_annotations", {})

    # --- Provenance ---
    paper_id    = sir.get("paper_id", "")
    title       = prov.get("title", "")
    authors     = prov.get("authors", [])
    arxiv_id    = prov.get("arxiv_id")
    domain      = prov.get("domain", "")
    subject_domain = prov.get("subject_domain")
    abstract    = prov.get("abstract", "")
    key_claims  = prov.get("key_claims", [])
    parsed_at   = prov.get("parsed_at", "")

    # --- Architecture ---
    # modules can be a list (canonical) or a dict of dicts (observed variant)
    raw_arch_modules = arch.get("modules", [])
    if isinstance(raw_arch_modules, dict):
        # dict-of-dicts: e.g. {"layer_1_memory_store": {...}, ...}
        module_names = list(raw_arch_modules.keys())
    else:
        module_names = [m.get("name", "") for m in raw_arch_modules if m.get("name")]

    primary_variant = arch.get("primary_variant") or arch.get("description", "")
    n_modules = len(module_names)

    # --- Mathematical spec ---
    math_spec_raw = sir.get("mathematical_spec", [])
    math_items = _list_or_values(math_spec_raw)
    equation_names = [e.get("name", "") for e in math_items if isinstance(e, dict) and e.get("name")]
    equation_roles = list({e.get("role", "") for e in math_items if isinstance(e, dict) and e.get("role")})

    # --- Tensor semantics ---
    tensor_raw = sir.get("tensor_semantics", [])
    tensor_items = _list_or_values(tensor_raw)
    tensor_names = [t.get("name", "") for t in tensor_items if isinstance(t, dict) and t.get("name")]

    # --- Training pipeline ---
    optimizer = tp.get("optimizer", {})
    opt_name  = optimizer.get("name") if isinstance(optimizer, dict) else None
    lr        = optimizer.get("learning_rate") if isinstance(optimizer, dict) else None
    schedule  = _get_nested(tp, "lr_schedule", "type")
    batch_size = tp.get("batch_size") or tp.get("effective_batch_size")
    mixed_precision = tp.get("mixed_precision")
    warmup_steps = _get_nested(tp, "lr_schedule", "warmup_steps")
    training_steps = tp.get("training_steps")
    epochs = tp.get("epochs")
    has_evolution_loop = "evolution_loop" in tp  # EVOLVEMEM-style

    # --- Evaluation protocol ---
    metrics_raw = ep.get("metrics", [])
    metrics = [m if isinstance(m, str) else m.get("name", "") for m in metrics_raw]

    datasets_raw = ep.get("datasets", [])
    datasets = [d.get("name", "") if isinstance(d, dict) else str(d) for d in datasets_raw]

    reported_results = ep.get("reported_results", [])
    primary_results = [r for r in reported_results if isinstance(r, dict) and r.get("is_primary")]

    # --- Assumptions & ambiguities ---
    assumptions = _list_or_values(sir.get("implementation_assumptions", []))
    n_assumptions = len(assumptions)
    low_conf_assumptions = [
        a for a in assumptions
        if isinstance(a, dict) and (a.get("confidence", 1.0) or 1.0) < 0.7
    ]

    ambiguities = _list_or_values(sir.get("ambiguities", []))
    n_ambiguities = len(ambiguities)

    # --- Confidence ---
    overall_confidence = ca.get("overall_sir_confidence", sir.get("confidence_annotations", {}).get("overall_sir_confidence"))

    # Fallback: compute from section confidences if overall is missing
    if overall_confidence is None:
        weights = {
            "architecture": 0.30,
            "mathematical_spec": 0.20,
            "training_pipeline": 0.20,
            "evaluation_protocol": 0.15,
            "tensor_semantics": 0.10,
            "implementation_assumptions": 0.05,
        }
        weighted_sum = 0.0
        weight_total = 0.0
        for section, w in weights.items():
            conf = ca.get(section)
            if conf is not None:
                weighted_sum += conf * w
                weight_total += w
        overall_confidence = round(weighted_sum / weight_total, 4) if weight_total > 0 else None

    return {
        # Identity
        "paper_id":             paper_id,
        "sir_version":          sir.get("sir_version", 1),
        "sir_path":             sir.get("_path", ""),

        # Provenance
        "title":                title,
        "authors":              authors,
        "arxiv_id":             arxiv_id,
        "domain":               domain,
        "subject_domain":       subject_domain,
        "abstract_preview":     abstract[:300] if abstract else "",
        "key_claims":           key_claims,
        "parsed_at":            parsed_at,

        # Architecture
        "primary_variant":      primary_variant,
        "module_names":         module_names,
        "n_modules":            n_modules,

        # Math
        "equation_names":       equation_names,
        "equation_roles":       equation_roles,
        "n_equations":          len(equation_names),

        # Tensors
        "tensor_names":         tensor_names,
        "n_tensors":            len(tensor_names),

        # Training
        "optimizer":            opt_name,
        "learning_rate":        lr,
        "lr_schedule":          schedule,
        "batch_size":           batch_size,
        "mixed_precision":      mixed_precision,
        "warmup_steps":         warmup_steps,
        "training_steps":       training_steps,
        "epochs":               epochs,
        "has_evolution_loop":   has_evolution_loop,

        # Evaluation
        "metrics":              [m for m in metrics if m],
        "datasets":             [d for d in datasets if d],
        "n_reported_results":   len(reported_results),
        "primary_results":      primary_results,

        # Assumptions
        "n_assumptions":        n_assumptions,
        "n_low_conf_assumptions": len(low_conf_assumptions),
        "n_ambiguities":        n_ambiguities,

        # Confidence
        "confidence": {
            "overall":                  overall_confidence,
            "architecture":             ca.get("architecture"),
            "mathematical_spec":        ca.get("mathematical_spec"),
            "training_pipeline":        ca.get("training_pipeline"),
            "evaluation_protocol":      ca.get("evaluation_protocol"),
            "tensor_semantics":         ca.get("tensor_semantics"),
            "implementation_assumptions": ca.get("implementation_assumptions"),
        },
    }


# ============================================================
# 3. SIMILARITY MATRIX
# ============================================================

def build_similarity_matrix(sirs: list[dict], verbose: bool = False) -> dict:
    """Compute pairwise SIR similarity using sir_diff.compute_diff.

    Tries to import sir_diff from the sibling search directory.
    Falls back to a lightweight Jaccard-only similarity if unavailable.
    """
    n = len(sirs)
    paper_ids = [s.get("paper_id", f"paper_{i}") for i, s in enumerate(sirs)]

    # Try importing sir_diff
    compute_diff_fn = None
    try:
        import importlib.util, os
        # Look for sir_diff.py relative to this file or in known locations
        search_paths = [
            Path(__file__).parent.parent / "diff" / "sir_diff.py",
            Path(__file__).parent.parent.parent / "sir" / "diff" / "sir_diff.py",
            Path("sir/diff/sir_diff.py"),
            Path("../diff/sir_diff.py"),
        ]
        for candidate in search_paths:
            if candidate.exists():
                spec = importlib.util.spec_from_file_location("sir_diff", candidate)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                compute_diff_fn = mod.compute_diff
                if verbose:
                    print(f"  Using sir_diff from: {candidate}")
                break
        if compute_diff_fn is None and verbose:
            print("  sir_diff.py not found in expected locations — using fallback Jaccard similarity")
    except Exception as e:
        if verbose:
            print(f"  sir_diff import failed ({e}) — using fallback Jaccard similarity")

    scores = {}
    total_pairs = n * (n - 1) // 2
    computed = 0

    for i in range(n):
        for j in range(i, n):
            id_a = paper_ids[i]
            id_b = paper_ids[j]

            if i == j:
                scores[f"{id_a}||{id_b}"] = 1.0
                continue

            try:
                if compute_diff_fn is not None:
                    diff = compute_diff_fn(sirs[i], sirs[j])
                    sim = diff.overall_similarity
                else:
                    sim = _jaccard_similarity(sirs[i], sirs[j])
            except Exception as e:
                if verbose:
                    print(f"  Warning: diff failed for {id_a} vs {id_b}: {e}")
                sim = 0.0

            scores[f"{id_a}||{id_b}"] = round(sim, 4)
            scores[f"{id_b}||{id_a}"] = round(sim, 4)
            computed += 1

            if verbose and total_pairs > 0:
                print(f"  [{computed}/{total_pairs}] {id_a} vs {id_b}: {sim:.4f}")

    # Build matrix as nested dict for easy lookup
    matrix = {pid: {} for pid in paper_ids}
    for i, id_a in enumerate(paper_ids):
        for j, id_b in enumerate(paper_ids):
            key = f"{id_a}||{id_b}"
            matrix[id_a][id_b] = scores.get(key, 0.0)

    # Find top-3 most similar papers for each
    neighbors = {}
    for pid in paper_ids:
        row = matrix[pid]
        ranked = sorted(
            [(other, sim) for other, sim in row.items() if other != pid],
            key=lambda x: x[1],
            reverse=True,
        )
        neighbors[pid] = [{"paper_id": other, "similarity": sim} for other, sim in ranked[:3]]

    return {
        "paper_ids": paper_ids,
        "matrix": matrix,
        "neighbors": neighbors,
        "method": "sir_diff" if compute_diff_fn is not None else "jaccard_fallback",
        "n_pairs_computed": computed,
    }


def _jaccard_similarity(sir_a: dict, sir_b: dict) -> float:
    """Lightweight fallback: Jaccard over module names + equation names + metrics."""
    def name_set(sir, *keys):
        result = set()
        obj = sir
        for k in keys:
            obj = obj.get(k, {}) if isinstance(obj, dict) else {}
        items = _list_or_values(obj)
        for item in items:
            if isinstance(item, dict):
                name = item.get("name", "")
                if name:
                    result.add(name.lower())
        return result

    def str_set(sir, key):
        val = sir.get(key, [])
        if isinstance(val, list):
            return {str(v).lower() for v in val if v}
        return set()

    mods_a = name_set(sir_a, "architecture", "modules")
    mods_b = name_set(sir_b, "architecture", "modules")
    eqs_a  = name_set(sir_a, "mathematical_spec")
    eqs_b  = name_set(sir_b, "mathematical_spec")
    met_a  = str_set(sir_a.get("evaluation_protocol", {}), "metrics")
    met_b  = str_set(sir_b.get("evaluation_protocol", {}), "metrics")

    def jaccard(a, b):
        if not a and not b:
            return 1.0
        u = a | b
        return len(a & b) / len(u) if u else 1.0

    return round(
        0.35 * jaccard(mods_a, mods_b) +
        0.20 * jaccard(eqs_a, eqs_b) +
        0.15 * jaccard(met_a, met_b) +
        0.30 * 1.0,  # base similarity for unknown sections
        4
    )


# ============================================================
# 4. REGISTRY STATISTICS
# ============================================================

def compute_stats(flat_records: list[dict]) -> dict:
    """Aggregate statistics across all flattened SIR records."""
    n = len(flat_records)
    if n == 0:
        return {"n_papers": 0}

    # Domain distribution
    domains = Counter(r.get("domain", "Unknown") for r in flat_records)
    subject_domains = Counter(r.get("subject_domain") or "Unknown" for r in flat_records)

    # Optimizer distribution
    optimizers = Counter(r.get("optimizer") or "unspecified" for r in flat_records)

    # LR schedule distribution
    schedules = Counter(r.get("lr_schedule") or "unspecified" for r in flat_records)

    # Metrics — collect all across papers
    all_metrics = Counter()
    for r in flat_records:
        for m in r.get("metrics", []):
            if m:
                all_metrics[m] += 1

    # Datasets
    all_datasets = Counter()
    for r in flat_records:
        for d in r.get("datasets", []):
            if d:
                all_datasets[d] += 1

    # Confidence distribution
    conf_values = [r["confidence"]["overall"] for r in flat_records if r["confidence"]["overall"] is not None]
    conf_tiers = Counter()
    for v in conf_values:
        if v >= 0.9:
            conf_tiers["explicit (≥0.9)"] += 1
        elif v >= 0.7:
            conf_tiers["implied (0.7-0.89)"] += 1
        elif v >= 0.5:
            conf_tiers["inferred (0.5-0.69)"] += 1
        else:
            conf_tiers["speculative (<0.5)"] += 1

    avg_confidence = round(sum(conf_values) / len(conf_values), 4) if conf_values else None

    # Module counts
    n_modules_list = [r["n_modules"] for r in flat_records if r["n_modules"] > 0]
    avg_modules = round(sum(n_modules_list) / len(n_modules_list), 1) if n_modules_list else None

    # Assumption counts
    n_assumptions_list = [r["n_assumptions"] for r in flat_records]
    avg_assumptions = round(sum(n_assumptions_list) / len(n_assumptions_list), 1) if n_assumptions_list else None

    # Papers needing human review (low conf assumptions)
    needs_review = [r["paper_id"] for r in flat_records if r["n_low_conf_assumptions"] > 0]

    # Papers with evolution loops (non-standard training)
    evolution_papers = [r["paper_id"] for r in flat_records if r.get("has_evolution_loop")]

    return {
        "n_papers":             n,
        "compiled_at":          datetime.now(timezone.utc).isoformat(),

        "domain_distribution":          dict(domains.most_common()),
        "subject_domain_distribution":  dict(subject_domains.most_common()),

        "optimizer_distribution":       dict(optimizers.most_common()),
        "lr_schedule_distribution":     dict(schedules.most_common()),

        "top_metrics":          dict(all_metrics.most_common(10)),
        "top_datasets":         dict(all_datasets.most_common(10)),

        "confidence": {
            "average":          avg_confidence,
            "tier_distribution": dict(conf_tiers),
            "min":              round(min(conf_values), 4) if conf_values else None,
            "max":              round(max(conf_values), 4) if conf_values else None,
        },

        "architecture": {
            "avg_modules_per_paper":    avg_modules,
        },

        "assumptions": {
            "avg_per_paper":            avg_assumptions,
            "papers_with_low_conf_assumptions": needs_review,
        },

        "special": {
            "papers_with_evolution_loop": evolution_papers,
        },
    }


# ============================================================
# 5. OUTPUT
# ============================================================

def write_json(path: Path, data: dict) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def print_summary(stats: dict, matrix_info: dict | None, out_dir: Path) -> None:
    n = stats["n_papers"]
    avg_conf = stats["confidence"]["average"]
    top_domain = next(iter(stats["subject_domain_distribution"]), "?")
    top_opt = next(iter(stats["optimizer_distribution"]), "?")

    print("\n" + "═" * 60)
    print("  ArXivist SIR Compiler — complete")
    print("═" * 60)
    print(f"  Papers compiled:     {n}")
    print(f"  Avg confidence:      {avg_conf}")
    print(f"  Top domain:          {top_domain}")
    print(f"  Top optimizer:       {top_opt}")
    print(f"  Top metrics:         {', '.join(list(stats['top_metrics'].keys())[:4])}")

    if matrix_info:
        method = matrix_info.get("method", "?")
        pairs = matrix_info.get("n_pairs_computed", 0)
        print(f"  Similarity pairs:    {pairs}  (method: {method})")

    low_conf = stats["assumptions"]["papers_with_low_conf_assumptions"]
    if low_conf:
        print(f"\n  ⚠  {len(low_conf)} paper(s) have low-confidence assumptions:")
        for pid in low_conf[:5]:
            print(f"     - {pid}")
        if len(low_conf) > 5:
            print(f"     ... and {len(low_conf)-5} more")

    print(f"\n  Output written to: {out_dir}")
    print(f"    compiled_index.json")
    print(f"    similarity_matrix.json" if matrix_info else "    (similarity matrix skipped)")
    print(f"    registry_stats.json")
    print("═" * 60 + "\n")


# ============================================================
# 6. CLI
# ============================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="ArXivist SIR Compiler — build a queryable view of the registry",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--registry-dir",
        default="../../workspace/sir-registry/",
        help="Path to the SIR registry directory (default: ../../workspace/sir-registry/)",
    )
    p.add_argument(
        "--no-matrix",
        action="store_true",
        help="Skip similarity matrix computation (faster for large registries)",
    )
    p.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print per-SIR and per-pair progress",
    )
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    registry_dir = Path(args.registry_dir).resolve()
    if not registry_dir.exists():
        print(f"Error: registry directory not found: {registry_dir}", file=sys.stderr)
        sys.exit(1)

    t0 = time.time()
    print(f"\nArXivist SIR Compiler")
    print(f"Registry: {registry_dir}\n")

    # 1. Load
    print("Loading SIRs...")
    sirs = load_registry(registry_dir, verbose=args.verbose)
    if not sirs:
        print("No SIRs found. Run the pipeline first to populate the registry.")
        sys.exit(1)
    print(f"  {len(sirs)} SIRs loaded in {time.time()-t0:.1f}s")

    # 2. Flatten
    print("\nFlattening records...")
    flat_records = []
    for sir in sirs:
        try:
            flat_records.append(flatten_sir(sir))
        except Exception as e:
            print(f"  Warning: flatten failed for {sir.get('paper_id', '?')}: {e}")
    print(f"  {len(flat_records)} records flattened")

    # 3. Compiled index
    compiled_index = {
        "compiled_at": datetime.now(timezone.utc).isoformat(),
        "n_papers": len(flat_records),
        "registry_dir": str(registry_dir),
        "papers": flat_records,
    }
    index_path = registry_dir / "compiled_index.json"
    write_json(index_path, compiled_index)
    print(f"  compiled_index.json written ({len(flat_records)} records)")

    # 4. Similarity matrix
    matrix_info = None
    if not args.no_matrix:
        print(f"\nComputing similarity matrix ({len(sirs)}×{len(sirs)})...")
        matrix_data = build_similarity_matrix(sirs, verbose=args.verbose)
        matrix_info = matrix_data
        matrix_path = registry_dir / "similarity_matrix.json"
        write_json(matrix_path, matrix_data)
        print(f"  similarity_matrix.json written ({matrix_data['n_pairs_computed']} pairs)")
    else:
        print("\nSimilarity matrix skipped (--no-matrix)")

    # 5. Stats
    print("\nComputing registry statistics...")
    stats = compute_stats(flat_records)
    stats_path = registry_dir / "registry_stats.json"
    write_json(stats_path, stats)
    print(f"  registry_stats.json written")

    # 6. Summary
    print_summary(stats, matrix_info, registry_dir)


if __name__ == "__main__":
    main()

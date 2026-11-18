"""Jacobian-lens estimates and internal-state interventions.

Legacy pilot methods retain their strictly future targets and three fixed
source positions. The workspace experiment uses fit_workspace_lens to estimate
12 selected vocabulary rows of W_U J, summing all target positions and averaging
source positions and calibration contexts. Its calibration metadata and
per-context arrays are recorded separately from the legacy estimates.

Gurnee et al. (2026), sections 2.1, 2.5 and A.7:
https://transformer-circuits.pub/2026/workspace/index.html
"""

import json
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import balanced_accuracy_score, r2_score
from transformers import AutoTokenizer

from analysis import read_jsonl
from skin import generate_trials, make_prefix


LAYER = 48
WORDS = (" recovery", " bonus")
POSITIONS = (31, 63, 95)
QUERIES = ("action", "value", "mapping")
ARMS = ("sham", "V_donor", "J_swap", "V_random", "J_random")
CONTENT_LAYERS = (8, 16, 24, 32, 40, 48, 56, 60)
CONTENT_WORDS = (
    " pressure", " distress", " calm", " help", " quit", " avoid",
    " danger", " safe", " duty", " report", " wrong", " conflict",
)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def scale_random_edit(current, direction, desired_norm, dtype, max_steps=24, rtol=1e-4):
    """Return the closest quantized edit found along a fixed direction.

    Norms are per position, with a final singleton dimension. Bracket expansion
    is limited to doubling, followed by bisection. Keeping the unchanged residual
    as a candidate prevents large overshoots when a tiny target is unattainable.
    """
    original = current.to(dtype).float()
    direction = direction.float()
    unit = direction / direction.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    target = torch.as_tensor(desired_norm, device=original.device, dtype=torch.float32)
    target = torch.broadcast_to(target, original.shape[:-1] + (1,))
    if not torch.isfinite(target).all() or (target < 0).any():
        raise ValueError("Desired edit norms must be finite and nonnegative")
    best = original.to(dtype)
    best_error = target.clone()
    lower = torch.zeros_like(target)
    upper = target.clone()
    scale = target.clone()
    bracketed = torch.zeros_like(target, dtype=torch.bool)
    for _ in range(max_steps):
        candidate = (original + unit * scale).to(dtype)
        actual = (candidate.float() - original).norm(dim=-1, keepdim=True)
        error = (actual - target).abs()
        improve = torch.isfinite(error) & (error < best_error)
        best = torch.where(improve, candidate, best)
        best_error = torch.where(improve, error, best_error)
        if torch.all(best_error <= target * rtol):
            break
        below = actual < target
        lower = torch.where(below, scale, lower)
        upper = torch.where(~below, scale, upper)
        bracketed |= ~below
        scale = torch.where(bracketed, (lower + upper) / 2, scale * 2)
    return best


def make_variants(trials):
    """Counterbalance response codes after one shared observation boundary."""
    variants = []
    for trial in trials:
        prefix = make_prefix(trial, ("LEFT", "RIGHT"))
        prefix = prefix.replace("-> R,", "-> recovery,").replace("-> G,", "-> bonus,")
        for codebook in ("direct", "reversed"):
            outcomes = ("recovery", "bonus") if codebook == "direct" else ("bonus", "recovery")
            buttons = ("LEFT", "RIGHT") if codebook == "direct" else ("RIGHT", "LEFT")
            variants.append({
                **trial, "id": f"{trial['id']}:code-{codebook}",
                "physical_trial_id": trial["id"], "prompt_group": trial["id"],
                "interface": "jlens", "button_names": ["LEFT", "RIGHT"],
                "response_symbols": ["X", "Y"], "codebook": codebook,
                "prefix": prefix, "queries": list(QUERIES),
                "system_prompt": "You control a simulated tactile pad using the recorded observations. "
                                 "Respond with exactly one character, X or Y, to the question.",
                "questions": {
                    "action": "Choose the action giving the greater total score. "
                              f"X={buttons[0]}, Y={buttons[1]}. Answer X or Y.",
                    "value": "Compare total scores of the named outcomes. "
                             f"X={outcomes[0]}, Y={outcomes[1]}. Choose the greater. Answer X or Y.",
                    "mapping": "Identify the action that produces recovery. "
                               f"X={buttons[0]}, Y={buttons[1]}. Answer X or Y.",
                },
            })
    return variants


def prepare_jlens(args):
    """Prepare new simulator states and public TRAIN calibration text on CPU."""
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    settings = output / "settings.json"
    if settings.exists():
        saved = json.loads(settings.read_text())
        for key in ("model_id", "seed", "jlens_contexts"):
            if saved[key] != getattr(args, key):
                raise ValueError(f"Recorded J-lens {key} differs from the request")
        if saved["experiment"] != "jlens":
            raise ValueError("J-lens requires its own output directory")
        trials = read_jsonl(output / "trials.jsonl")
        variants = read_jsonl(output / "prompts.jsonl")
    else:
        rng = np.random.default_rng(np.random.SeedSequence(args.seed, spawn_key=(2,)))
        trials = generate_trials(args.seed, 16, 4, 12, rng=rng)
        for trial in trials:
            trial["id"], trial["family"] = "jlens-" + trial["id"], "jlens-" + trial["family"]
        variants = make_variants(trials)
        for name, rows in (("trials", trials), ("prompts", variants)):
            (output / f"{name}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        write_json(settings, {
            **vars(args), "experiment": "jlens", "thinking": False, "dtype": "bfloat16",
            "families": {"fit": 16, "dev": 4, "test": 12}, "layer": LAYER,
            "target_layer": 64, "source_positions": list(POSITIONS), "words": list(WORDS),
            "scenario_rng": "SeedSequence(seed, spawn_key=(2,)); fixed fresh scenario stream",
            "estimator": "mean_context mean_source mean_future_target J(target,source)^T W_U[token]",
            "source": "output of block 48; detach here only, then full downstream autograd",
            "target": "output of block 64 before final RMSNorm",
            "matrix": "two unit columns; float32 pseudoinverse, rtol=1e-6",
            "arms": list(ARMS), "scientific_sequences": 1984,
            "calibration_sequences": args.jlens_contexts, "diagnostic_sequences": 16,
            "planned_vjps": args.jlens_contexts * 6,
            "dev_gate": "balanced accuracy >=0.75 for fixed (v_recovery-v_bonus)^T h >=0",
            "bootstrap": "2000 paired complete-family draws; TEST only",
        })
    calibration = output / "calibration.json"
    if not calibration.exists():
        import pyarrow.parquet as parquet
        from huggingface_hub import hf_hub_download

        if not 1 <= args.jlens_contexts <= 64:
            raise ValueError("This bounded pilot supports 1 through 64 calibration contexts")
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        filename = "wikitext-2-raw-v1/train-00000-of-00001.parquet"
        data = hf_hub_download("Salesforce/wikitext", filename, repo_type="dataset",
                               local_dir=output / "calibration_source")
        texts = parquet.read_table(data, columns=["text"])["text"].to_pylist()
        tokens = tokenizer.encode("\n".join(text for text in texts if text.strip()), add_special_tokens=False)
        if len(tokens) < args.jlens_contexts * 128:
            raise ValueError("Calibration corpus ended before the requested token count")
        word_ids = [tokenizer.encode(word, add_special_tokens=False) for word in WORDS]
        if any(len(ids) != 1 for ids in word_ids):
            raise ValueError("Both outcome words must be single tokens")
        write_json(calibration, {
            "dataset": "Salesforce/wikitext", "configuration": "wikitext-2-raw-v1",
            "split": "train", "file": filename, "rows_read": len(texts),
            "joining": "nonempty rows joined by newline",
            "tokenizer_model": args.model_id, "word_ids": [ids[0] for ids in word_ids],
            "source_positions": list(POSITIONS), "add_special_tokens": False,
            "token_ids": [tokens[i * 128:(i + 1) * 128] for i in range(args.jlens_contexts)],
        })
    return trials, variants


def check_budget(policy, sequences=0):
    if policy.attempted + sequences > policy.args.max_forwards:
        raise RuntimeError("Forward budget reached")
    if (policy.elapsed_before + time.monotonic() - policy.started) / 3600 > policy.args.max_hours:
        raise RuntimeError("Runtime budget reached")


def save_usage(policy):
    write_json(policy.output / "usage.json", {
        "attempted_sequences": policy.attempted,
        "elapsed_seconds": policy.elapsed_before + time.monotonic() - policy.started,
        "peak_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
    })


def token_vjps(source, target, output_vectors, positions=POSITIONS, before_vjp=None):
    """Return selected W_U J rows; intermediate token gradients stay unreduced."""
    values = []
    total = len(output_vectors) * len(positions)
    for word_index, vector in enumerate(output_vectors):
        gradients = []
        for position in positions:
            if before_vjp is not None:
                before_vjp(word_index, position)
            objective = target[:, position + 1:, :].float().mean(dim=1) @ vector.float()
            index = word_index * len(positions) + len(gradients)
            gradient = torch.autograd.grad(objective.sum(), source, retain_graph=index + 1 < total)[0]
            gradients.append(gradient[:, position, :].float().mean(dim=0))
        values.append(torch.stack(gradients).mean(dim=0))
    return torch.stack(values)


def fit_jlens(policy):
    """One saved contribution per context permits continuation by context ID."""
    metadata = json.loads((policy.output / "calibration.json").read_text())
    directory = policy.output / "jacobians"
    directory.mkdir(exist_ok=True)
    device = next(policy.model.parameters()).device
    output_vectors = policy.model.get_output_embeddings().weight[metadata["word_ids"]].detach().float()
    ledger = policy.output / "vjps.jsonl"
    vjp_attempts = sum(1 for line in ledger.open() if line.strip()) if ledger.exists() else 0
    for index, token_ids in enumerate(metadata["token_ids"]):
        path = directory / f"context-{index:03d}.npz"
        if path.exists():
            continue
        check_budget(policy, 1)
        started = time.monotonic()
        captured = {}

        def source_hook(module, inputs, output):
            value = output[0] if isinstance(output, tuple) else output
            leaf = value.detach().requires_grad_(True)
            captured["source"] = leaf
            return (leaf, *output[1:]) if isinstance(output, tuple) else leaf

        def target_hook(module, inputs, output):
            captured["target"] = output[0] if isinstance(output, tuple) else output

        handles = [policy.layers[LAYER - 1].register_forward_hook(source_hook),
                   policy.layers[-1].register_forward_hook(target_hook)]
        policy.attempted += 1
        policy.call_index += 1
        with (policy.output / "forwards.jsonl").open("a") as stream:
            stream.write(json.dumps({"id": f"forward-{policy.call_index}", "sequences": 1,
                                    "condition": "J_calibration", "query": "raw_train_text",
                                    "trial_ids": [f"context-{index:03d}"], "time": time.time()}) + "\n")

        def before_vjp(word_index, position):
            nonlocal vjp_attempts
            check_budget(policy)
            vjp_attempts += 1
            with ledger.open("a") as stream:
                stream.write(json.dumps({"id": f"vjp-attempt-{vjp_attempts}",
                                        "context": index, "word": WORDS[word_index],
                                        "source_position": position, "time": time.time()}) + "\n")

        try:
            with torch.enable_grad():
                inputs = torch.tensor([token_ids], device=device)
                policy.model.model.language_model(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                                                  use_cache=False)
                vectors = token_vjps(captured["source"], captured["target"], output_vectors,
                                     before_vjp=before_vjp)
            vectors = vectors.detach().cpu().numpy()
            if not np.isfinite(vectors).all() or np.any(np.linalg.norm(vectors, axis=1) == 0):
                raise ValueError(f"Invalid Jacobian contribution at context {index}")
            np.savez(path, vectors=vectors, token_ids=token_ids)
        finally:
            for handle in handles:
                handle.remove()
            captured.clear()
            save_usage(policy)
        print(f"J-lens calibration {index + 1}/{len(metadata['token_ids'])}; "
              f"{time.monotonic() - started:.2f}s; attempted={policy.attempted}", flush=True)
    raw = np.stack([np.load(directory / f"context-{i:03d}.npz")["vectors"]
                    for i in range(len(metadata["token_ids"]))]).mean(axis=0)
    matrix = (raw / np.linalg.norm(raw, axis=1, keepdims=True)).T.astype(np.float32)
    singular = np.linalg.svd(matrix, compute_uv=False)
    if singular[-1] <= singular[0] * 1e-6:
        raise ValueError("The two J-lens columns do not have numerical rank two")
    inverse = np.linalg.pinv(matrix, rcond=1e-6).astype(np.float32)
    np.savez(policy.output / "lens.npz", matrix=matrix, inverse=inverse, raw_vectors=raw)
    write_json(policy.output / "lens_metrics.json", {
        "contexts": len(metadata["token_ids"]), "completed_vjps": len(metadata["token_ids"]) * 6,
        "attempted_vjps": len(read_jsonl(policy.output / "vjps.jsonl")),
        "word_ids": metadata["word_ids"], "raw_vector_norms": np.linalg.norm(raw, axis=1).tolist(),
        "unit_vector_cosine": float(matrix[:, 0] @ matrix[:, 1]), "singular_values": singular.tolist(),
        "source_block": LAYER, "target_block": len(policy.layers), "rank": 2,
    })
    return matrix, inverse


def coordinate_swap(hidden, matrix, inverse):
    """The pseudoinverse exchange preserves the orthogonal complement."""
    coordinates = hidden @ inverse.T
    return (coordinates.flip(-1) - coordinates) @ matrix.T


def pick_donor(trial, fit):
    candidates = [t for t in fit if t["need"] == trial["need"] and t["mapping"] != trial["mapping"]
                  and (t["value"] >= 0) != (trial["value"] >= 0)]
    if not candidates:
        raise ValueError(f"No opposite-value FIT donor for {trial['id']}")
    return min(candidates, key=lambda t: (abs(abs(t["value"]) - abs(trial["value"])), t["id"]))


def score_arm(policy, batch, query, arm, probe=None, bank=None, fit=None, matrix=None, inverse=None):
    """Reuse the policy and semantic decoder; edit one common-boundary token."""
    metadata = [{} for _ in batch]
    handles = []
    if arm != "sham":
        device = next(policy.model.parameters()).device
        rows = torch.arange(len(batch), device=device)
        positions = torch.tensor([policy.encode(t, query)[1] for t in batch], device=device)
        donors = [pick_donor(t, fit) for t in batch] if arm.startswith("V") else None

        def edit(module, inputs, output):
            tensor = output[0] if isinstance(output, tuple) else output
            original = tensor[rows, positions].float()
            if arm == "zero":
                delta = torch.zeros_like(original)
            elif arm.startswith("V"):
                axis = torch.as_tensor(probe["axis"], device=device)
                destination = torch.as_tensor(np.stack([bank[t["id"]] for t in donors]), device=device)
                coefficient = (destination - original) @ axis
                delta = coefficient[:, None] * axis
            else:
                delta = coordinate_swap(original, matrix, inverse)
                coordinates = original @ inverse.T
                coefficient = coordinates[:, 1] - coordinates[:, 0]
            if arm.endswith("random"):
                axis = torch.as_tensor(probe[arm], device=device)
                sign = torch.where(coefficient >= 0, 1.0, -1.0)
                delta = sign[:, None] * delta.norm(dim=-1, keepdim=True) * axis
            updated = (original + delta).to(tensor.dtype)
            tensor[rows, positions] = updated
            actual = updated.float() - original
            for i in range(len(batch)):
                metadata[i] = {
                    "requested_edit_norm": float(delta[i].norm()), "edit_norm": float(actual[i].norm()),
                    "donor_id": donors[i]["id"] if donors else None,
                }
                if donors:
                    value_axis = torch.as_tensor(probe["axis"], device=device)
                    metadata[i].update(
                        v_projection_before=float(original[i] @ value_axis),
                        v_projection_after=float(updated[i].float() @ value_axis),
                        v_donor_projection=float(destination[i] @ value_axis),
                    )
                if matrix is not None:
                    direction = matrix[:, 0] - matrix[:, 1]
                    metadata[i]["j_score_before"] = float(original[i] @ direction)
                    metadata[i]["j_score_after"] = float(updated[i].float() @ direction)
                    metadata[i]["j_coordinates_before"] = (original[i] @ inverse.T).tolist()
                    metadata[i]["j_coordinates_after"] = (updated[i].float() @ inverse.T).tolist()
            return output

        handles.append(policy.layers[LAYER - 1].register_forward_hook(edit))
    policy.external_condition = arm
    try:
        records, captured = policy.score(batch, query=query, capture=[LAYER] if arm in ("sham", "zero") else [])
    finally:
        for handle in handles:
            handle.remove()
        policy.external_condition = None
    for record, trial, extra in zip(records, batch, metadata):
        record.update(extra, condition=arm, id=f"{trial['id']}:{arm}:{query}",
                      target_answer=("B" if record["correct_answer"] == "A" else "A")
                      if query in ("action", "value") else record["correct_answer"])
    return records, captured


def check_jlens_hooks(policy, variants):
    pair = [t for t in variants if t["physical_trial_id"] == variants[0]["physical_trial_id"]]
    clean, activation, errors, zero_errors = [], None, [], []
    for trial in pair:
        for query in QUERIES:
            records, captured = score_arm(policy, [trial], query, "sham")
            clean.append((trial, query, records[0]))
            activation = captured[LAYER][0] if activation is None else activation
            errors.append(float(np.max(np.abs(captured[LAYER][0] - activation))))
    for trial, query, baseline in clean:
        records, _ = score_arm(policy, [trial], query, "zero")
        zero_errors.append(abs(records[0]["p_a"] - baseline["p_a"]))
        if records[0]["edit_norm"] != 0:
            raise ValueError("A zero edit changed a hidden state")
    repeats = []
    for _ in range(4):
        trial, query, baseline = clean[0]
        records, _ = score_arm(policy, [trial], query, "sham")
        repeats.append(abs(records[0]["p_a"] - baseline["p_a"]))
    passed = max(errors) <= 0.02 and max(zero_errors + repeats) <= 1e-5
    write_json(policy.output / "hook_checks.json", {
        "passed": passed, "diagnostic_sequences": 16,
        "future_suffix_activation_errors": errors, "zero_edit_probability_errors": zero_errors,
        "repeat_probability_errors": repeats,
    })
    if not passed:
        raise ValueError("J-lens shared-boundary or zero-edit hook check failed")


def summarize_jlens(output, seed, require_complete=True):
    """Compute semantic contrasts paired by all four cells of a TEST family."""
    output = Path(output)
    trials, variants = read_jsonl(output / "trials.jsonl"), read_jsonl(output / "prompts.jsonl")
    rows = read_jsonl(output / "predictions.jsonl")
    expected = {
        f"{t['id']}:{arm}:{query}" for t in variants
        for arm in (("sham",) if t["split"] == "fit" else ARMS)
        for query in (("action",) if t["split"] == "fit" else QUERIES)
        if t["split"] != "fit" or t["codebook"] == "direct"
    }
    found = {row["id"] for row in rows}
    coverage = {"expected": len(expected), "observed": len(rows), "unique": len(found),
                "missing": len(expected - found), "unexpected": len(found - expected)}
    if len(rows) != len(found) or found - expected or (require_complete and found != expected):
        raise ValueError(f"J-lens coverage error: {coverage}")
    if found != expected:
        write_json(output / "summary.json", {"coverage": coverage})
        return {"coverage": coverage}
    bank = {}
    for path in sorted((output / "activations").glob("*.npz")):
        with np.load(path) as data:
            bank.update({str(key): value for key, value in zip(data["ids"], data["hidden"])})
    lens = np.load(output / "lens.npz")
    probe = np.load(output / "probe.npz")
    dev = [t for t in trials if t["split"] == "dev"]
    hidden = np.stack([bank[t["id"]] for t in dev])
    j_score = hidden @ (lens["matrix"][:, 0] - lens["matrix"][:, 1])
    truth = np.array([t["value"] >= 0 for t in dev])
    gate = float(balanced_accuracy_score(truth, j_score >= 0))
    dev_metrics = {
        "states": len(dev), "families": 4, "J_balanced_accuracy_zero_threshold": gate,
        "J_value_gate_passed": gate >= 0.75, "J_scores": j_score.tolist(),
        "V_r2": float(r2_score([t["value"] for t in dev],
                              (hidden - probe["center"]) @ probe["coef"] + probe["intercept"])),
    }
    test = [row for row in rows if row["split"] == "test"]
    families = sorted({row["family"] for row in test})
    draw = np.random.default_rng(seed).integers(0, len(families), (2000, len(families)))

    def interval(values):
        values = np.array([np.mean(values[family]) for family in families])
        limits = np.quantile(values[draw].mean(axis=1), [0.025, 0.975])
        return {"mean": float(values.mean()), "ci_low": float(limits[0]), "ci_high": float(limits[1])}

    indexed = {(r["trial_id"], r["condition"], r["query"]): r for r in test}
    natural, contrasts = [], []
    for code in ("direct", "reversed"):
        for query in QUERIES:
            cohort = [r for r in test if r["condition"] == "sham" and r["query"] == query
                      and r["codebook"] == code]
            natural.append({"codebook": code, "query": query, "n": len(cohort),
                            "correct": sum(r["chosen"] == r["correct_answer"] for r in cohort),
                            "accuracy": float(np.mean([r["chosen"] == r["correct_answer"] for r in cohort])),
                            "legal_mass": float(np.mean([r["legal_mass"] for r in cohort]))})
            for arm in ARMS[1:]:
                comparisons = ("sham", arm[0] + "_random") if arm in ("V_donor", "J_swap") else ("sham",)
                for comparison in comparisons:
                    values = {metric: {family: [] for family in families}
                              for metric in ("target_probability", "accuracy", "legal_mass")}
                    for baseline in cohort:
                        edited = indexed[(baseline["trial_id"], arm, query)]
                        reference = indexed[(baseline["trial_id"], comparison, query)]
                        probability = "p_a" if edited["target_answer"] == "A" else "p_b"
                        family = baseline["family"]
                        values["target_probability"][family].append(edited[probability] - reference[probability])
                        values["accuracy"][family].append(
                            int(edited["chosen"] == edited["correct_answer"])
                            - int(reference["chosen"] == reference["correct_answer"]))
                        values["legal_mass"][family].append(edited["legal_mass"] - reference["legal_mass"])
                    contrasts.append({"codebook": code, "query": query, "arm": arm,
                                      "reference": comparison, "n": len(cohort),
                                      **{key: interval(value) for key, value in values.items()}})
    summary = {"coverage": coverage, "dev": dev_metrics, "test_families": len(families),
               "natural": natural, "contrasts": contrasts,
               "lens": json.loads((output / "lens_metrics.json").read_text())}
    write_json(output / "summary.json", summary)
    return summary


def run_jlens(args, policy_class):
    from run import append_records, configure_pilot

    trials, variants = prepare_jlens(args)
    output = Path(args.output)
    if args.stage == "jlens-prepare":
        print(f"Prepared {len(trials)} J-lens states and {args.jlens_contexts} calibration contexts", flush=True)
        return
    if args.stage == "jlens-analyze":
        print(json.dumps(summarize_jlens(output, args.seed)["coverage"]), flush=True)
        return
    predictions = output / "predictions.jsonl"
    existing = read_jsonl(predictions) if predictions.exists() else []
    completed = {row["id"] for row in existing}
    if len(completed) == 1984:
        print(json.dumps(summarize_jlens(output, args.seed)["coverage"]), flush=True)
        return
    policy = policy_class(args)
    if len(policy.layers) != 64:
        raise ValueError("The fixed J-lens target is block 64")
    configure_pilot(policy, variants)
    if (output / "lens.npz").exists():
        lens = np.load(output / "lens.npz")
        matrix, inverse = lens["matrix"], lens["inverse"]
    else:
        matrix, inverse = fit_jlens(policy)
    device = next(policy.model.parameters()).device
    matrix, inverse = torch.as_tensor(matrix, device=device), torch.as_tensor(inverse, device=device)
    checks = output / "hook_checks.json"
    if not checks.exists() or not json.loads(checks.read_text())["passed"]:
        check_jlens_hooks(policy, [t for t in variants if t["split"] == "dev"])
    directory = output / "activations"
    directory.mkdir(exist_ok=True)
    bank = {}
    for path in sorted(directory.glob("*.npz")):
        with np.load(path) as data:
            bank.update({str(key): value for key, value in zip(data["ids"], data["hidden"])})
    fit = [t for t in trials if t["split"] == "fit"]
    probe = None
    for split in ("fit", "dev", "test"):
        for arm in (("sham",) if split == "fit" else ARMS):
            for query in (("action",) if split == "fit" else QUERIES):
                pending = [t for t in variants if t["split"] == split
                           and (split != "fit" or t["codebook"] == "direct")
                           and (f"{t['id']}:{arm}:{query}" not in completed
                                or (arm == "sham" and query == "action"
                                    and t["physical_trial_id"] not in bank))]
                for width in sorted({policy.pilot_widths[t["id"]] for t in pending}):
                    matching = [t for t in pending if policy.pilot_widths[t["id"]] == width]
                    for start in range(0, len(matching), args.batch_size):
                        batch = matching[start:start + args.batch_size]
                        records, captured = score_arm(policy, batch, query, arm, probe, bank, fit, matrix, inverse)
                        if arm == "sham" and query == "action":
                            # Both codebooks share one prefix; retain the direct activation.
                            selected = [i for i, t in enumerate(batch) if t["codebook"] == "direct"]
                            if selected:
                                ids = [batch[i]["physical_trial_id"] for i in selected]
                                hidden = captured[LAYER][selected]
                                np.savez(directory / f"{ids[0]}.npz", ids=ids, hidden=hidden)
                                bank.update(dict(zip(ids, hidden)))
                        append_records(output, records, completed)
                print(f"J-lens {split} {arm} {query}: {len(completed)}/1984 science rows; "
                      f"attempted={policy.attempted}", flush=True)
        if split == "fit":
            x = np.stack([bank[t["id"]] for t in fit])
            center = x.mean(axis=0)
            regression = Ridge(alpha=10, solver="cholesky").fit(x - center, [t["value"] for t in fit])
            axis = regression.coef_.astype(np.float32)
            axis = axis / np.linalg.norm(axis)
            probe = {"center": center, "axis": axis, "coef": regression.coef_.astype(np.float32),
                     "intercept": np.array(regression.intercept_)}
            rng = np.random.default_rng(np.random.SeedSequence(args.seed, spawn_key=(3,)))
            for name in ("V_random", "J_random"):
                direction = (rng.normal(size=len(x)) @ (x - center)).astype(np.float32)
                probe[name] = direction / np.linalg.norm(direction)
            np.savez(output / "probe.npz", **probe)
    save_usage(policy)
    print(json.dumps(summarize_jlens(output, args.seed)["coverage"]), flush=True)


@contextmanager
def capture_residuals(layers, source_layers=CONTENT_LAYERS, target_layer=None):
    """Capture a connected graph. Layer numbers count transformer blocks from one."""
    source_layers = tuple(sorted(set(source_layers)))
    target_layer = len(layers) - 1 if target_layer is None else target_layer
    if not source_layers or not 1 <= min(source_layers) <= max(source_layers) <= target_layer <= len(layers):
        raise ValueError("Sources must be valid blocks no later than the target")
    state = {"sources": {}, "target": None}
    handles = []

    def capture(number):
        def hook(module, inputs, output):
            value = output[0] if isinstance(output, tuple) else output
            if number == source_layers[0]:
                value = value.detach().requires_grad_(True)
                output = (value, *output[1:]) if isinstance(output, tuple) else value
            if number in source_layers:
                state["sources"][number] = value
            if number == target_layer:
                state["target"] = value
            return output
        return hook

    try:
        for number in sorted(set(source_layers) | {target_layer}):
            handles.append(layers[number - 1].register_forward_hook(capture(number)))
        yield state
    finally:
        for handle in handles:
            handle.remove()


def token_vjps_multi(sources, target, output_vectors, positions=POSITIONS, before_vjp=None):
    """Selected rows of W_U J using a strict future mean at each fixed source."""
    if any(position < 0 or position + 1 >= target.shape[1] for position in positions):
        raise ValueError("Every source position must have a later target")
    numbers, tensors = tuple(sources), tuple(sources.values())
    collected = {number: [] for number in numbers}
    total = len(output_vectors) * len(positions)
    for word_index, vector in enumerate(output_vectors):
        per_position = {number: [] for number in numbers}
        for position_index, position in enumerate(positions):
            if before_vjp is not None:
                before_vjp(word_index, position)
            objective = target[:, position + 1:].float().mean(dim=1) @ vector.float()
            last = word_index * len(positions) + position_index + 1 == total
            gradients = torch.autograd.grad(objective.sum(), tensors, retain_graph=not last)
            for number, gradient in zip(numbers, gradients):
                per_position[number].append(gradient[:, position].float().mean(dim=0))
        for number in numbers:
            collected[number].append(torch.stack(per_position[number]).mean(dim=0))
    return {number: torch.stack(values) for number, values in collected.items()}


def selected_token_jacobian(sources, target, output_vectors, before_vjp=None):
    """Exact selected rows of W_U J under A.7's sum-target/mean-source estimator.

    Every objective sums all target positions. Its gradients are averaged over
    source positions and sequences, including the diagonal causal contribution.
    One VJP obtains a selected row at every captured source layer.
    """
    if target.ndim != 3 or output_vectors.ndim != 2:
        raise ValueError("Expected batch-by-token-by-hidden targets and word-by-hidden vectors")
    if output_vectors.shape[1] != target.shape[-1] or not len(output_vectors):
        raise ValueError("Output vectors must match the target width and contain at least one row")
    numbers, tensors = tuple(sources), tuple(sources.values())
    if not numbers:
        raise ValueError("At least one captured source layer is required")
    collected = {number: [] for number in numbers}
    for index, vector in enumerate(output_vectors):
        if before_vjp is not None:
            before_vjp(index)
        objective = (target.float() @ vector.to(target.device, dtype=torch.float32)).sum()
        gradients = torch.autograd.grad(objective, tensors, retain_graph=index + 1 < len(output_vectors))
        for number, gradient in zip(numbers, gradients):
            collected[number].append(gradient.float().mean(dim=(0, 1)).detach())
    return {number: torch.stack(rows) for number, rows in collected.items()}


def fit_workspace_lens(policy, words, source_layers, contexts=8, context_length=128,
                       target_layer=None, output_path=None):
    """Calibrate selected vocabulary coordinates on independent WikiText TRAIN.

    The returned rows use the paper's A.7 estimator, not the pilot's fixed-source
    estimator. They support selected-coordinate edits, not full J-space claims.
    """
    output = Path(output_path or policy.output)
    output.mkdir(parents=True, exist_ok=True)
    source_layers = tuple(sorted(set(source_layers)))
    target_layer = len(policy.layers) - 1 if target_layer is None else target_layer
    if not source_layers or not 1 <= min(source_layers) <= max(source_layers) <= target_layer <= len(policy.layers):
        raise ValueError("Source blocks must precede or equal the target block")
    metadata_path = output / "workspace_calibration.json"
    if not metadata_path.exists():
        import pyarrow.parquet as parquet
        from huggingface_hub import hf_hub_download

        filename = "wikitext-2-raw-v1/train-00000-of-00001.parquet"
        local = Path("/opt/dlami/nvme/ubuntu/backup_skin/outputs/qwen3.8_jlens_pilot/calibration_source") / filename
        corpus = local if local.exists() else Path(
            hf_hub_download("Salesforce/wikitext", filename, repo_type="dataset"))
        texts = parquet.read_table(corpus, columns=["text"])["text"].to_pylist()
        tokens = policy.tokenizer.encode("\n".join(text for text in texts if text.strip()),
                                         add_special_tokens=False)
        if contexts < 1 or context_length < 2 or len(tokens) < contexts * context_length:
            raise ValueError("Calibration requires complete nonempty text blocks")
        word_ids = [policy.tokenizer.encode(word, add_special_tokens=False) for word in words]
        if not word_ids or any(len(ids) != 1 for ids in word_ids):
            raise ValueError("Every prespecified word must be a single tokenizer token")
        blocks = [tokens[i * context_length:(i + 1) * context_length] for i in range(contexts)]
        write_json(metadata_path, {
            "dataset": "Salesforce/wikitext", "configuration": "wikitext-2-raw-v1",
            "split": "train", "file": filename, "source_layers": list(source_layers),
            "target_layer": target_layer, "target": "transformer-block output before final normalization",
            "estimator": "mean_context mean_source sum_all_target_positions",
            "includes_diagonal": True, "words": list(words), "word_ids": [ids[0] for ids in word_ids],
            "token_ids": blocks, "texts": [policy.tokenizer.decode(block) for block in blocks],
            "planned_vjps": contexts * len(words), "completed_contexts": 0,
            "scope": "Selected vocabulary rows of W_U J, not a full vocabulary dictionary",
        })
    metadata = json.loads(metadata_path.read_text())
    if (metadata["source_layers"] != list(source_layers)
            or metadata["target_layer"] != target_layer or metadata["words"] != list(words)):
        raise ValueError("Selected coordinates and layers must retain their recorded meanings")
    directory = output / "workspace_jacobians"
    directory.mkdir(exist_ok=True)
    input_device = getattr(policy, "input_device", policy.model.get_input_embeddings().weight.device)
    language_model = getattr(policy, "language_model", None)
    weights = policy.model.get_output_embeddings().weight
    word_ids = torch.tensor(metadata["word_ids"], device=weights.device)
    vectors = weights.index_select(0, word_ids).detach().float()
    for index, token_ids in enumerate(metadata["token_ids"]):
        destination = directory / f"context-{index:03d}.npz"
        if destination.exists():
            continue
        context_id = f"workspace-context-{index:03d}"
        policy.record_forward(1, "workspace_calibration", "raw_train_text", [context_id])
        started = time.monotonic()

        def before_vjp(word_index):
            policy.consume_vjp(context=context_id, word=metadata["words"][word_index],
                               estimator=metadata["estimator"], target_layer=target_layer)

        with torch.enable_grad(), capture_residuals(policy.layers, source_layers, target_layer) as state:
            inputs = torch.tensor([token_ids], device=input_device)
            arguments = {"input_ids": inputs, "attention_mask": torch.ones_like(inputs), "use_cache": False}
            if language_model is not None:
                language_model(**arguments)
            else:
                policy.model(**arguments, logits_to_keep=1)
            contributions = selected_token_jacobian(
                state["sources"], state["target"], vectors, before_vjp)
            values = np.stack([contributions[number].cpu().numpy() for number in source_layers])
        if not np.isfinite(values).all():
            raise ValueError(f"Nonfinite selected-coordinate gradients in {context_id}")
        np.savez(destination, raw_vectors=values, token_ids=token_ids)
        metadata["completed_contexts"] = index + 1
        write_json(metadata_path, metadata)
        policy.save_usage()
        print(f"Workspace calibration {index + 1}/{len(metadata['token_ids'])}: "
              f"{time.monotonic() - started:.2f}s", flush=True)
    contributions = []
    for index in range(len(metadata["token_ids"])):
        with np.load(directory / f"context-{index:03d}.npz") as saved:
            contributions.append(saved["raw_vectors"])
    raw = np.stack(contributions).mean(axis=0)
    norms = np.linalg.norm(raw, axis=-1, keepdims=True)
    valid = np.isfinite(norms[..., 0]) & (norms[..., 0] > 0)
    metadata.update(completed_contexts=len(contributions), raw_vector_norms=norms[..., 0].tolist(),
                    direction_status="valid" if valid.all() else "invalid")
    write_json(metadata_path, metadata)
    if not valid.all():
        raise ValueError("Selected-coordinate calibration contains zero or nonfinite directions")
    result = {"raw_vectors": raw, "unit_vectors": raw / norms, "raw_norms": norms[..., 0],
              "layers": np.array(source_layers), "target_layer": np.array(target_layer),
              "word_ids": np.array(metadata["word_ids"]), "words": np.array(metadata["words"])}
    np.savez(output / "workspace_lens.npz", **result)
    policy.save_usage()
    return result


def fit_content_lens(policy, output_path=None, source_layers=CONTENT_LAYERS, max_vjps=1152):
    """Fit the bounded pilot readout, saving one completed contribution per context ID."""
    output = Path(output_path or policy.output)
    output.mkdir(parents=True, exist_ok=True)
    metadata_path = output / "content_calibration.json"
    source_layers = tuple(source_layers)
    if not metadata_path.exists():
        import pyarrow.parquet as parquet
        from huggingface_hub import hf_hub_download

        filename = "wikitext-2-raw-v1/train-00000-of-00001.parquet"
        previous = Path("/opt/dlami/nvme/ubuntu/backup_skin/outputs/qwen3.8_jlens_pilot/calibration_source")
        path = previous / filename
        if not path.exists():
            path = Path(hf_hub_download("Salesforce/wikitext", filename, repo_type="dataset"))
        texts = parquet.read_table(path, columns=["text"])["text"].to_pylist()
        tokens = policy.tokenizer.encode("\n".join(text for text in texts if text.strip()),
                                         add_special_tokens=False)
        if len(tokens) < 32 * 128:
            raise ValueError("Calibration corpus has fewer than 32 complete token blocks")
        ids = [policy.tokenizer.encode(word, add_special_tokens=False) for word in CONTENT_WORDS]
        if any(len(value) != 1 for value in ids):
            raise ValueError("Each selected content word must be exactly one token")
        blocks = [tokens[i * 128:(i + 1) * 128] for i in range(32)]
        write_json(metadata_path, {
            "dataset": "Salesforce/wikitext", "configuration": "wikitext-2-raw-v1",
            "split": "train", "file": filename, "source_layers": list(source_layers),
            "target_layer": len(policy.layers), "source_positions": list(POSITIONS),
            "words": list(CONTENT_WORDS), "word_ids": [value[0] for value in ids],
            "token_ids": blocks, "texts": [policy.tokenizer.decode(block) for block in blocks],
            "estimator": "mean_context mean_source mean_strict_future_target",
            "planned_vjps": 32 * len(CONTENT_WORDS) * len(POSITIONS),
            "labels": "Vocabulary tokens define directions. No scenario or preference labels are used.",
        })
    metadata = json.loads(metadata_path.read_text())
    if metadata["source_layers"] != list(source_layers):
        raise ValueError("Requested source layers do not match the recorded calibration")
    if metadata["planned_vjps"] > max_vjps:
        raise ValueError("Requested calibration exceeds the VJP allowance")
    directory = output / "content_jacobians"
    directory.mkdir(exist_ok=True)
    device = next(policy.model.parameters()).device
    vectors = policy.model.get_output_embeddings().weight[metadata["word_ids"]].detach().float()
    for index, token_ids in enumerate(metadata["token_ids"]):
        destination = directory / f"context-{index:03d}.npz"
        if destination.exists():
            continue
        context_id = f"content-context-{index:03d}"
        policy.record_forward(1, "content_calibration", "raw_train_text", [context_id])
        started = time.monotonic()

        def before_vjp(word_index, position):
            policy.consume_vjp(context=context_id, word=metadata["words"][word_index],
                               source_position=position)

        with torch.enable_grad(), capture_residuals(policy.layers, source_layers, len(policy.layers)) as state:
            inputs = torch.tensor([token_ids], device=device)
            policy.model.model.language_model(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                                              use_cache=False)
            contributions = token_vjps_multi(state["sources"], state["target"], vectors,
                                             metadata["source_positions"], before_vjp)
            values = torch.stack([contributions[number] for number in source_layers]).detach().cpu().numpy()
        if not np.isfinite(values).all():
            raise ValueError(f"Nonfinite selected-content gradients in {context_id}")
        np.savez(destination, raw_vectors=values, token_ids=token_ids)
        print(f"Content calibration {index + 1}/32: {time.monotonic() - started:.2f}s", flush=True)
    raw = np.stack([np.load(directory / f"context-{index:03d}.npz")["raw_vectors"]
                    for index in range(len(metadata["token_ids"]))]).mean(axis=0)
    norms = np.linalg.norm(raw, axis=-1, keepdims=True)
    invalid = ~np.isfinite(norms[..., 0]) | (norms[..., 0] <= 0)
    metadata["raw_vector_norms"] = [[float(value) if np.isfinite(value) else None
                                     for value in row] for row in norms[..., 0]]
    metadata["completed_contexts"] = len(metadata["token_ids"])
    metadata["invalid_directions"] = [
        {"layer": source_layers[layer], "word": metadata["words"][word],
         "reason": "nonfinite norm" if not np.isfinite(norms[layer, word, 0]) else "zero norm"}
        for layer, word in np.argwhere(invalid)
    ]
    metadata["direction_status"] = "invalid" if invalid.any() else "valid"
    write_json(metadata_path, metadata)
    policy.save_usage()
    if invalid.any():
        raise ValueError(f"{int(invalid.sum())} selected-content directions have zero or nonfinite "
                         f"raw gradient norms. See {metadata_path}")
    units = raw / norms
    result = {"raw_vectors": raw, "unit_vectors": units, "raw_norms": norms[..., 0],
              "layers": np.array(source_layers),
              "word_ids": np.array(metadata["word_ids"]), "words": np.array(metadata["words"])}
    np.savez(output / "content_lens.npz", **result)
    return result


def binding_direction(activations):
    """Own-contact minus other-contact, balancing both endpoint identifiers."""
    means = {}
    for condition in ("AA", "AB", "BA", "BB"):
        value = torch.as_tensor(activations[condition], dtype=torch.float32)
        means[condition] = value.reshape(-1, value.shape[-1]).mean(dim=0)
    return (means["AA"] + means["BB"] - means["AB"] - means["BA"]) / 2


def norm_matched_random(delta, generator=None):
    """Return isotropic changes with each row's requested Euclidean norm."""
    delta = torch.as_tensor(delta)
    random = torch.randn(delta.shape, device=delta.device, dtype=torch.float32, generator=generator)
    return random * (delta.float().norm(dim=-1, keepdim=True)
                     / random.norm(dim=-1, keepdim=True).clamp_min(1e-12))


def reflect_direction(hidden, center, direction):
    """Return a change exchanging opposite coordinates around a calibration center."""
    hidden, center, direction = hidden.float(), center.float(), direction.float()
    denominator = direction.square().sum().clamp_min(1e-12)
    return -2 * ((hidden - center) @ direction)[..., None] * direction / denominator


def coordinate_clamp(hidden, matrix, reference, inverse=None):
    """Restore pseudoinverse coordinates of selected, possibly nonorthogonal columns.

    Use the selected sparse frame. A full vocabulary dictionary may span the
    whole residual stream and would leave no unconstrained complement.
    """
    matrix = matrix.float()
    inverse = torch.linalg.pinv(matrix, rtol=1e-6) if inverse is None else inverse.float()
    return ((reference.float() - hidden.float()) @ inverse.T) @ matrix.T


@contextmanager
def register_boundary_edit(layers, layer, positions, change=None, capture_layers=()):
    """Change one observation token before the later question and response codes."""
    selected = set(capture_layers) | ({layer} if change is not None else set())
    state, handles = {"captured": {}}, []

    def make_hook(number):
        def hook(module, inputs, output):
            tensor = output[0] if isinstance(output, tuple) else output
            rows = torch.arange(tensor.shape[0], device=tensor.device)
            columns = torch.as_tensor(positions, device=tensor.device).expand(tensor.shape[0])
            original = tensor[rows, columns].float()
            if number in capture_layers:
                state["captured"][number] = original.detach().cpu()
            if number != layer or change is None:
                return output
            delta = change(original) if callable(change) else torch.as_tensor(change, device=tensor.device)
            updated = (original + delta.float()).to(tensor.dtype)
            actual = updated.float() - original
            state.update(before=original.detach().cpu(), after=updated.float().detach().cpu(),
                         requested_norm=delta.float().norm(dim=-1).detach().cpu(),
                         actual_norm=actual.norm(dim=-1).detach().cpu())
            result = tensor.clone()
            result[rows, columns] = updated
            return (result, *output[1:]) if isinstance(output, tuple) else result
        return hook

    try:
        for number in sorted(selected):
            handles.append(layers[number - 1].register_forward_hook(make_hook(number)))
        yield state
    finally:
        for handle in handles:
            handle.remove()


def full_jacobian(sources, target, before_vjp=None):
    """A.7 estimator: all target positions summed, source positions averaged.

    One reverse objective per target dimension obtains every captured source layer.
    This includes the current position and is distinct from token_vjps_multi.
    """
    numbers, tensors = tuple(sources), tuple(sources.values())
    result = {number: [] for number in numbers}
    for dimension in range(target.shape[-1]):
        if before_vjp is not None:
            before_vjp(dimension)
        gradients = torch.autograd.grad(target[..., dimension].float().sum(), tensors,
                                        retain_graph=dimension + 1 < target.shape[-1])
        for number, gradient in zip(numbers, gradients):
            result[number].append(gradient.float().mean(dim=(0, 1)))
    return {number: torch.stack(rows) for number, rows in result.items()}


def sparse_components(vector, dictionary, k=16):
    """Greedy nonnegative sparse approximation, with NNLS refits on selected rows.

    The dictionary has one row per vocabulary direction. The returned remainder
    is a reconstruction residual, not the orthogonal complement of the full frame.
    """
    from scipy.optimize import nnls

    vector, dictionary = vector.float(), dictionary.float()
    if vector.ndim != 1 or dictionary.ndim != 2 or dictionary.shape[1] != len(vector):
        raise ValueError("Expected one hidden vector and a vocabulary-by-hidden dictionary")
    norms = dictionary.norm(dim=-1, keepdim=True)
    directions = dictionary / norms.clamp_min(1e-12)
    residual = vector.clone()
    selected, coefficients = [], np.empty(0, dtype=np.float32)
    for _ in range(min(k, len(dictionary))):
        scores = directions @ residual
        if selected:
            scores[selected] = -torch.inf
        scores[norms[:, 0] == 0] = -torch.inf
        index = int(scores.argmax())
        if scores[index] <= 0:
            break
        selected.append(index)
        basis = directions[selected].T.detach().cpu().numpy()
        coefficients, _ = nnls(basis, vector.detach().cpu().numpy())
        reconstructed = directions[selected].T @ torch.as_tensor(coefficients, device=vector.device,
                                                                  dtype=vector.dtype)
        residual = vector - reconstructed
    component = vector - residual
    return {"component": component, "remainder": residual,
            "indices": torch.tensor(selected, device=vector.device),
            "coefficients": torch.as_tensor(coefficients, device=vector.device, dtype=vector.dtype)}


def broadcast_metrics(ov_matrix, directions, random_directions):
    """OV gain and label-preservation MRR, with an equally sized random baseline."""
    def measure(population):
        unit = population.float() / population.float().norm(dim=-1, keepdim=True).clamp_min(1e-12)
        transformed = unit @ ov_matrix.float().T
        similarity = torch.nn.functional.normalize(transformed, dim=-1) @ unit.T
        rank = (similarity > similarity.diagonal()[:, None]).sum(dim=-1) + 1
        return transformed.norm(dim=-1).mean(), rank.float().reciprocal().mean()

    if directions.shape != random_directions.shape:
        raise ValueError("Direction and random populations must have the same shape")
    gain, preservation = measure(directions)
    random_gain, random_preservation = measure(random_directions)
    return {"gain_ratio": float(gain / random_gain.clamp_min(1e-12)),
            "label_preservation": float(preservation),
            "random_label_preservation": float(random_preservation),
            "label_preservation_excess": float(preservation - random_preservation)}


@contextmanager
def ablate_attention_heads(attention, heads, head_dim):
    """Remove selected heads at the input of a full-attention output projection."""
    if not hasattr(attention, "o_proj"):
        raise ValueError("Head ablation requires a full-attention module with o_proj")

    def hook(module, inputs):
        value = inputs[0]
        if value.shape[-1] % head_dim or any(head < 0 or head >= value.shape[-1] // head_dim for head in heads):
            raise ValueError("Head index or head dimension does not match this projection")
        edited = value.clone().reshape(*value.shape[:-1], -1, head_dim)
        edited[..., list(heads), :] = 0
        return (edited.reshape_as(value), *inputs[1:])

    handle = attention.o_proj.register_forward_pre_hook(hook)
    try:
        yield
    finally:
        handle.remove()

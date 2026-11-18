"""Family-paired summaries for the tactile revaluation experiment."""

import csv
import json
from pathlib import Path

import numpy as np


def token_readout_figure(directory, model, case, arrays):
    """Export an editable overview of one prespecified recorded example."""
    import textwrap
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.lines import Line2D

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "svg.fonttype": "none", "pdf.fonttype": 42,
        "text.color": "#172a3e", "axes.labelcolor": "#667587",
        "xtick.color": "#667587", "ytick.color": "#667587",
    })
    words = [word.strip() for word in case["words"]]
    layers = np.asarray(case["layers"])
    boundary = case["question_boundary"]
    position = max(0, boundary - 1)
    start = max(0, min(position - 24, len(case["tokens"]) - 48))
    stop = min(start + 48, len(case["tokens"]))
    arm = "swap" if "swap" in arrays else "baseline"
    raw = arrays[arm]["raw_scores"]
    ranks = 1 + (raw[..., None, :] > raw[..., :, None]).sum(-1)
    all_colors = ["#2879b7", "#df775f", "#8765b9", "#279d94",
                  "#387db1", "#8e6bbd", "#da785f", "#249c94",
                  "#bb9234", "#798dc4", "#ba6481", "#668780"]
    pinned_words = ("self", "other", "pain", "relief") if case.get("own_pad") else case["axis"].split("_")
    pinned = [words.index(word) for word in pinned_words if word in words]
    focus = words.index(case["axis"].split("_")[0])
    color = dict(zip(words, all_colors))
    cmap = LinearSegmentedColormap.from_list("word_rank", ["#304d84", "#408ba9", "#88bdc9", "#d8e8ed", "#f2f5f8"])
    figure = plt.figure(figsize=(18, 11), facecolor="white")
    grid = figure.add_gridspec(3, 3, height_ratios=(.68, 1.18, 1.1),
                              width_ratios=(1.42, 1, 1), hspace=.55, wspace=.3)
    figure.text(.04, .968, "INSIDE THE DECISION", fontsize=10, weight="bold",
                color="#2879b7", va="top")
    figure.text(.04, .936, model["model_id"].rsplit("/", 1)[-1],
                fontsize=22, weight="bold", va="top")
    subtitle = (f"{case['title']}  ·  {case['modality'].replace('_', ' + ')}  ·  "
                f"{case['axis'].replace('_', ' / ')} exchange")
    figure.text(.04, .904, subtitle, fontsize=11, color="#667587", va="top")

    text_ax = figure.add_subplot(grid[0, :2])
    text_ax.axis("off")
    text_ax.text(0, 1, "01  THE INPUT", va="top", fontsize=10, weight="bold", color="#2879b7")
    task_line = next((line for line in case["prompt_text"].splitlines() if line.startswith("Task:")), case["title"])
    question_line = next((line for line in case["prompt_text"].splitlines() if line.startswith("Question:")), "")
    skin_line = (f"Surface {case['own_pad']} is assigned as the model's artificial skin. "
                 f"The tip presses surface {case['stimulated_pad']} at a {case.get('depth_mm')} mm target depth."
                 if case.get("own_pad") else "")
    excerpt = textwrap.fill(task_line + "\n" + skin_line + "\n" + question_line,
                            118, max_lines=4, placeholder=" …")
    text_ax.text(0, .75, excerpt, va="top", fontsize=10, linespacing=1.55, color="#405468")
    text_ax.text(0, -.09, f"Prompt tokens only. Question starts at token {boundary}. "
                 f"Readings after the intervention, at measured layers {layers[0]}–{layers[-1]}.",
                 fontsize=8.5, color="#667587")

    action_ax = figure.add_subplot(grid[0, 2])
    candidates = case["candidates"]
    original = np.array([case["arms"]["baseline"]["probabilities"][c] for c in candidates]) * 100
    edited = np.array([case["arms"][arm]["probabilities"][c] for c in candidates]) * 100
    y = np.arange(len(candidates))
    action_ax.barh(y + .14, original, height=.22, color="#b4c1cf", label="Original")
    action_ax.barh(y - .14, edited, height=.22, color="#2879b7", label="Exchange")
    action_ax.set_xlim(0, 132)
    action_ax.set_yticks(y, [c.replace("handover", "Paid handover").capitalize() for c in candidates])
    action_ax.invert_yaxis()
    for i, (a, b) in enumerate(zip(original, edited)):
        difference = b - a
        change = "0.00" if difference == 0 else f"{difference:+.2f}" if abs(difference) >= .01 else f"{difference:+.2g}"
        left = "<0.01" if 0 < a < .005 else f"{a:.2f}"
        right = "<0.01" if 0 < b < .005 else f"{b:.2f}"
        action_ax.text(102, i, f"{left} → {right}%\n{change} pp",
                       fontsize=8, color="#405468", va="center")
    action_ax.set_xticks([0, 50, 100])
    action_ax.set_xlabel("Candidate-answer probability (%)", fontsize=8.5)
    action_ax.set_title("02  OFFERED ANSWER SCORES", loc="left", fontsize=10, weight="bold", color="#2879b7", pad=31)
    action_ax.text(0, 1.10, "Original", transform=action_ax.transAxes, fontsize=8, color="#7f92a4")
    action_ax.text(.25, 1.10, "Exchange", transform=action_ax.transAxes, fontsize=8, color="#2879b7")
    action_ax.spines["right"].set_visible(False)
    action_ax.spines["left"].set_visible(False)
    action_ax.tick_params(axis="y", length=0)

    word_ax = figure.add_subplot(grid[1, 0])
    sampled_positions = np.linspace(start, stop - 1, 12, dtype=int)
    sampled_layers = np.unique(np.linspace(0, len(layers) - 1, min(12, len(layers)), dtype=int))
    word_ax.set_xlim(-.5, len(sampled_positions) - .5)
    word_ax.set_ylim(-.5, len(sampled_layers) - .5)
    for iy, l in enumerate(sampled_layers):
        for ix, p in enumerate(sampled_positions):
            w = int(raw[l, p].argmax())
            word_ax.add_patch(plt.Rectangle((ix - .48, iy - .45), .96, .9,
                                            color=all_colors[w], alpha=.10, linewidth=0))
            word_ax.text(ix, iy, words[w], ha="center", va="center",
                         fontsize=7.3, color=all_colors[w])
    word_ax.set_yticks(range(len(sampled_layers)), layers[sampled_layers])
    word_ax.set_xticks(range(len(sampled_positions)), sampled_positions, rotation=0)
    word_ax.set_ylabel("Transformer layer")
    word_ax.set_xlabel("Prompt token position (12 displayed positions)")
    word_ax.set_title("Top word among the twelve fixed words", loc="left", fontsize=11, weight="bold", pad=13)
    for spine in word_ax.spines.values():
        spine.set_visible(False)
    word_ax.tick_params(length=0)

    for column, (name, index, positions) in enumerate((
        ("At the selected token", position, False),
        ("At the last measured layer", len(layers) - 1, True),
    ), start=1):
        ax = figure.add_subplot(grid[1, column])
        ax.axis("off")
        ax.set_title(name, loc="left", fontsize=11, weight="bold", pad=13)
        selected = sampled_positions if positions else sampled_layers[::-1]
        for row, selected_index in enumerate(selected):
            l, p = (index, selected_index) if positions else (selected_index, index)
            top = np.argsort(-raw[l, p])[:4]
            yy = .97 - row / max(12, len(selected))
            ax.text(0, yy, str(p if positions else layers[l]), fontsize=8.5, color="#8792a0", va="top")
            for j, w in enumerate(top):
                ax.text(.14 + j * .215, yy, words[w], fontsize=8.5, va="top",
                        color=all_colors[w] if w in pinned else "#405468",
                        weight="bold" if w in pinned else "normal")
        ax.text(0, -.08, f"{'Layer '+str(layers[index]) if positions else 'Token '+str(index)}. "
                "Descending raw linear reading.", fontsize=8.5, color="#667587")

    heat_ax = figure.add_subplot(grid[2, 0])
    image = heat_ax.imshow(ranks[:, start:stop, focus], origin="lower", aspect="auto",
                          vmin=1, vmax=12, cmap=cmap, interpolation="none",
                          extent=(start - .5, stop - .5, -.5, len(layers) - .5))
    heat_ax.set_yticks(np.linspace(0, len(layers) - 1, 6).astype(int),
                      layers[np.linspace(0, len(layers) - 1, 6).astype(int)])
    heat_ax.set_xlabel("Prompt token position")
    heat_ax.set_ylabel("Transformer layer")
    heat_ax.set_title(f"{words[focus]} · rank among twelve words", loc="left", fontsize=11, weight="bold", pad=14)
    if start <= boundary < stop:
        heat_ax.axvline(boundary - .5, color="#8e6bbd", linestyle="--", linewidth=1.1)
    heat_ax.plot(position, len(layers) - 1, "o", ms=5, mfc="white", mec="#172a3e", mew=1.2)
    bar = figure.colorbar(image, ax=heat_ax, location="right", shrink=.60, pad=.018, aspect=28)
    bar.set_ticks([1, 12], labels=["Rank 1", "Rank 12"])
    bar.ax.tick_params(labelsize=8, length=0)
    bar.outline.set_visible(False)

    for column, positions in ((1, False), (2, True)):
        ax = figure.add_subplot(grid[2, column])
        for w in pinned:
            x = np.arange(start, stop) if positions else layers
            values = ranks[-1, start:stop, w] if positions else ranks[:, position, w]
            ax.plot(x, values, linewidth=1.8, color=color[words[w]], label=words[w],
                    marker="o", markersize=2.6 if positions else 3.4)
        ax.set_ylim(12.6, .5)
        ax.set_yticks([1, 3, 6, 9, 12])
        ax.set_ylabel("Rank among twelve words")
        ax.set_xlabel("Prompt token position" if positions else "Transformer layer")
        ax.grid(axis="y", color="#e7edf2", linewidth=.8)
        ax.set_title(f"At layer {layers[-1]}" if positions else f"At token {position}",
                     loc="left", fontsize=11, weight="bold", pad=14)
        ax.axvline(position if positions else layers[-1], color="#adb8c5", linestyle="--", linewidth=.9)
    handles = [Line2D([0], [0], color=color[words[w]], lw=2, marker="o", ms=4, label=words[w])
               for w in pinned]
    figure.legend(handles=handles, frameon=False, ncol=4, loc="lower center",
                  bbox_to_anchor=(.5, .031), fontsize=10)
    figure.text(.04, .02, "Recorded prompt activity, projected onto saved Jacobian-derived directions. "
                "Ranks are within the twelve fixed words. No vocabulary softmax.",
                fontsize=8.5, color="#667587")
    figure.subplots_adjust(left=.06, right=.97, top=.853, bottom=.10)
    folder = Path(directory) / "figures"
    folder.mkdir(exist_ok=True)
    stem = f"token_readout_{model['key']}"
    saved = []
    for extension in ("svg", "pdf", "png"):
        path = folder / f"{stem}.{extension}"
        figure.savefig(path, dpi=180, facecolor="white")
        saved.append(str(path.relative_to(directory)))
    plt.close(figure)
    return saved


def analyze_token_readouts(directory):
    """Build the interactive explorer and vector figures from saved token arrays."""
    import base64
    import gzip
    import mimetypes

    directory = Path(directory)
    manifests = [directory / "manifest.json"] if (directory / "manifest.json").exists() else sorted(directory.glob("*/manifest.json"))
    if not manifests:
        raise FileNotFoundError(f"No token-readout manifest in {directory}")
    payload_dir = directory / "data"
    payload_dir.mkdir(exist_ok=True)
    models, figures, total_cases, total_arms = [], [], 0, 0
    generation_cases, generated_outputs = 0, 0
    counts = {}

    def packed_array(array):
        array = np.asarray(array, dtype="<f4", order="C")
        return {"shape": list(array.shape),
                "gzip_base64": base64.b64encode(gzip.compress(array.tobytes(), compresslevel=6, mtime=0)).decode("ascii")}

    def embedded_image(path, source):
        path = Path(path)
        candidates = (path, Path(source) / "stimuli/images" / path.name,
                      directory / "images" / path.name)
        found = next((candidate for candidate in candidates if candidate.exists()), None)
        if found is None:
            return None
        mime = mimetypes.guess_type(found.name)[0] or "image/png"
        return f"data:{mime};base64," + base64.b64encode(found.read_bytes()).decode("ascii")

    for manifest_path in manifests:
        manifest = json.loads(manifest_path.read_text())
        model_key = manifest_path.parent.name
        model = {"key": model_key, "model_id": manifest["model_id"], "cases": []}
        generation_path = manifest_path.with_name("generation_manifest.json")
        generations = {}
        if generation_path.exists():
            generation_manifest = json.loads(generation_path.read_text())
            if (generation_manifest["words"] != manifest["words"]
                    or generation_manifest["layers"] != manifest["layers"]):
                raise ValueError(f"Generation directions differ in {generation_path}")
            generations = {(case["trial_id"], case["query"], case["axis"]): case
                           for case in generation_manifest.get("cases", [])}
        default = None
        cases = manifest.get("cases", [])
        selected_default = next((case for case in cases if case.get("modality") == "image_text"
                                 and case.get("axis") == "self_other"
                                 and case.get("own_pad") == case.get("stimulated_pad")),
                                next((case for case in cases if case.get("axis") == "self_other"),
                                     cases[0] if cases else None))
        for index, original in enumerate(cases):
            if not original.get("tokens") or "baseline" not in original.get("arms", {}):
                continue
            payload = {**original, "words": manifest["words"], "layers": manifest["layers"],
                       "source_output": manifest["source_output"], "model_id": manifest["model_id"]}
            payload["title"] = original.get("title") or original.get("scenario_id") or original["trial_id"]
            payload["arms"], array_bank = {}, {}
            shape = (len(payload["layers"]), len(payload["tokens"]), len(payload["words"]))
            for arm, entry in original["arms"].items():
                path = manifest_path.parent / entry["file"]
                if not path.exists():
                    continue
                with np.load(path) as saved:
                    values = {name: saved[name] for name in ("raw_scores", "cosines")}
                for name, array in values.items():
                    if array.shape != shape or not np.isfinite(array).all():
                        raise ValueError(f"Invalid {name} shape or nonfinite readings in {path}")
                array_bank[arm] = values
                payload["arms"][arm] = {**entry, "source_file": str(path.relative_to(directory)),
                                       **{name: packed_array(array) for name, array in values.items()}}
            if "baseline" not in payload["arms"]:
                continue
            generation = generations.get((original["trial_id"], original["query"], original["axis"]))
            if generation:
                payload["generation"] = {}
                for arm, entry in generation["arms"].items():
                    path = manifest_path.parent / entry["file"]
                    if not path.exists():
                        continue
                    with np.load(path) as saved:
                        values = {name: saved[name] for name in
                                  ("raw_scores", "cosines", "step_raw_scores", "step_cosines")}
                    shape = (len(payload["layers"]), len(entry["tokens"]), len(payload["words"]))
                    step_shape = (len(payload["layers"]), len(entry["generated_tokens"]), len(payload["words"]))
                    for name, array in values.items():
                        expected = step_shape if name.startswith("step_") else shape
                        if array.shape != expected or not np.isfinite(array).all():
                            raise ValueError(f"Invalid generation array {name} in {path}")
                    payload["generation"][arm] = {**entry, "source_file": str(path.relative_to(directory)),
                                                  **{name: packed_array(array) for name, array in values.items()}}
                if payload["generation"]:
                    generation_cases += 1
                    generated_outputs += len(payload["generation"])
            key = f"{model_key}:{original['id']}"
            filename = f"{model_key}-{index:03d}.js"
            images = [embedded_image(path, manifest["source_output"])
                      for path in original.get("image_paths", [])]
            payload["images"] = [image for image in images if image is not None]
            text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False).replace("</", "<\\/")
            (payload_dir / filename).write_text(f"window.tokenReadoutReady({json.dumps(key)},{text});\n")
            modality = {"text": "Text", "image_text": "Image + text"}.get(original["modality"], original["modality"])
            label = f"{payload['title']} · {modality} · {original['axis'].replace('_', ' / ')}"
            if original.get("own_pad"):
                label += f" · skin {original['own_pad']}, pressed {original['stimulated_pad']}"
            model["cases"].append({"key": key, "label": label, "payload": f"data/{filename}",
                                   "modality": original["modality"], "axis": original["axis"],
                                   "has_generation": bool(payload.get("generation")),
                                   "own_pad": original.get("own_pad"),
                                   "stimulated_pad": original.get("stimulated_pad")})
            total_cases += 1
            total_arms += len(payload["arms"])
            counts[model_key] = counts.get(model_key, 0) + len(payload["arms"])
            if original is selected_default or default is None:
                default = payload, array_bank
        if model["cases"]:
            models.append(model)
            if default is not None:
                figures.extend(token_readout_figure(directory, model, *default))
    if not models:
        raise ValueError("No complete baseline token readings are available for display")
    metadata = {"models": models, "case_count": total_cases, "arm_count": total_arms,
                "generation_cases": generation_cases, "generated_outputs": generated_outputs,
                "figures": figures, "method": "Selected-word average-Jacobian linear readings",
                "rank_set": "Twelve prespecified words", "positions": "Prompt only"}
    template = Path(__file__).with_name("token_viewer.html").read_text()
    payload = json.dumps(metadata, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    (directory / "viewer.html").write_text(template.replace("TOKEN_VIEWER_PAYLOAD", payload))
    summary = {"models": len(models), "cases": total_cases, "condition_arrays": total_arms,
               "generation_cases": generation_cases, "generated_outputs": generated_outputs,
               "conditions_per_model": counts, "figures": figures, "viewer": "viewer.html",
               "source": "manifest.json and per-condition raw_scores/cosines NPZ arrays"}
    (directory / "viewer_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


CORE = ("sham", "V_clamp", "V_restore", "B_clamp", "B_restore")
CONDITIONS = ("sham",) + tuple(
    f"{variable}{control}_{operation}"
    for variable in ("V", "B")
    for control in ("", "_random", "_shuffled")
    for operation in ("clamp", "restore")
)
QUERIES = ("action", "body", "mapping", "value")


def read_jsonl(path):
    with Path(path).open() as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate IDs in {path}")
    return rows


def interval(values, draws):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return None
    samples = values[draws].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci95": np.quantile(samples, [0.025, 0.975]).tolist(),
        "families": len(values),
    }


def family_metrics(rows, trials):
    """Four observations form one family; incomplete families are not averaged."""
    grouped = {}
    for row in rows:
        family = trials[row["trial_id"]]["family"]
        grouped.setdefault(family, []).append(row)
    metrics = {
        name: {} for name in
        ("accuracy", "p_correct", "legal_mass", "edit_norm", "requested_edit_norm",
         "edit_norm_error", "edit_norm_relative_error", "D", "regret",
         "value_mapping_shift", "value_mapping_flip_rate")
    }
    for family, group in grouped.items():
        cells = {
            (trials[row["trial_id"]]["need"], trials[row["trial_id"]]["mapping"])
            for row in group
        }
        if cells != {("low", "A"), ("low", "B"), ("high", "A"), ("high", "B")}:
            continue
        metrics["accuracy"][family] = np.mean([
            row["chosen"] == row["correct_answer"] for row in group
        ])
        metrics["p_correct"][family] = np.mean([
            row["p_a"] if row["correct_answer"] == "A" else row["p_b"]
            for row in group
        ])
        metrics["legal_mass"][family] = np.mean([row["legal_mass"] for row in group])
        actual = np.array([row.get("edit_norm", 0) for row in group])
        requested = np.array([
            row.get("requested_edit_norm", row.get("edit_norm", 0)) for row in group
        ])
        metrics["edit_norm"][family] = actual.mean()
        metrics["requested_edit_norm"][family] = requested.mean()
        metrics["edit_norm_error"][family] = (actual - requested).mean()
        if requested.sum() > 0:
            metrics["edit_norm_relative_error"][family] = (
                np.abs(actual - requested).sum() / requested.sum()
            )
        elif actual.sum() == 0:
            metrics["edit_norm_relative_error"][family] = 0.0
        if group[0]["query"] == "value":
            metrics["value_mapping_shift"][family] = sum(
                (1 if trials[row["trial_id"]]["mapping"] == "B" else -1)
                * row["p_a"] / 2 for row in group
            )
            choices = {
                (trials[row["trial_id"]]["need"], trials[row["trial_id"]]["mapping"]): row["chosen"]
                for row in group
            }
            metrics["value_mapping_flip_rate"][family] = np.mean([
                choices[(need, "A")] != choices[(need, "B")] for need in ("low", "high")
            ])
        if group[0]["query"] == "action":
            metrics["D"][family] = sum(
                (1 if trials[row["trial_id"]]["need"] == "high" else -1)
                * (row["p_a"] if trials[row["trial_id"]]["mapping"] == "A" else row["p_b"])
                / 2
                for row in group
            )
            if all(row.get("actual_return") is not None for row in group):
                metrics["regret"][family] = np.mean([
                    max(trials[row["trial_id"]]["returns"].values()) - row["actual_return"]
                    for row in group
                ])
    return {name: values for name, values in metrics.items() if values}


def analyze(output_dir, seed=42, require_complete=True):
    """Write summary.json/results.csv; uncertainty resamples whole families."""
    output_dir = Path(output_dir)
    trial_rows = read_jsonl(output_dir / "trials.jsonl")
    trials = {row["id"]: row for row in trial_rows}
    predictions = read_jsonl(output_dir / "predictions.jsonl")
    expected = set()
    families = {}
    for trial in trial_rows:
        if trial["split"] not in ("fit", "dev", "test"):
            raise ValueError(f"Unknown dataset split: {trial['split']}")
        family = families.setdefault(trial["family"], [])
        family.append(trial)
        conditions = ("sham",) if trial["split"] == "fit" else CONDITIONS
        for condition in conditions:
            queries = QUERIES if trial["split"] != "fit" and condition in CORE else ("action",)
            expected.update(f"{trial['id']}:{condition}:{query}" for query in queries)
    for family, group in families.items():
        cells = {(row["need"], row["mapping"]) for row in group}
        if len(group) != 4 or cells != {("low", "A"), ("low", "B"), ("high", "A"), ("high", "B")}:
            raise ValueError(f"Family {family} does not contain the four factorial cells")
        if len({row["split"] for row in group}) != 1:
            raise ValueError(f"Family {family} crosses dataset splits")
        if len({row.get("stratum") for row in group}) != 1:
            raise ValueError(f"Family {family} crosses task strata")
    observed = {row["id"] for row in predictions}
    missing, unexpected = expected - observed, observed - expected
    if unexpected or (require_complete and missing):
        raise ValueError(
            f"Prediction coverage: {len(missing)} missing, {len(unexpected)} unexpected IDs; "
            f"examples: {sorted(missing or unexpected)[:3]}"
        )
    groups = {}
    for row in predictions:
        trial = trials[row["trial_id"]]
        if row["id"] != f"{row['trial_id']}:{row['condition']}:{row['query']}":
            raise ValueError(f"Malformed prediction ID: {row['id']}")
        for key in ("family", "split", "need", "mapping", "value"):
            if key in row and row[key] != trial[key]:
                raise ValueError(f"Prediction/trial {key} mismatch: {row['id']}")
        probabilities = [row["p_a"], row["p_b"], row["legal_mass"]]
        if not all(np.isfinite(p) and 0 <= p <= 1 for p in probabilities):
            raise ValueError(f"Invalid probabilities: {row['id']}")
        if not np.isclose(row["p_a"] + row["p_b"], 1, atol=1e-5):
            raise ValueError(f"Choice probabilities do not sum to one: {row['id']}")
        for key in ("edit_norm", "requested_edit_norm"):
            if not np.isfinite(row.get(key, 0)) or row.get(key, 0) < 0:
                raise ValueError(f"Invalid {key}: {row['id']}")
        if row.get("actual_return") is not None and not np.isfinite(row["actual_return"]):
            raise ValueError(f"Invalid realized task return: {row['id']}")
        if row["chosen"] not in ("A", "B") or row["correct_answer"] not in ("A", "B"):
            raise ValueError(f"Invalid choice label: {row['id']}")
        if row["query"] == "action":
            oracle = trial["mapping"] if trial["value"] >= 0 else (
                "B" if trial["mapping"] == "A" else "A"
            )
            if row["correct_answer"] != oracle:
                raise ValueError(f"Action oracle disagrees with simulator value: {row['id']}")
        key = (trial["split"], row["condition"], row["query"])
        groups.setdefault(key, []).append(row)

    # Each cohort reuses one draw matrix for every paired contrast.
    draw_cache = {}
    flat_rows = []

    def summarize(values):
        cohort = tuple(sorted(values))
        if not cohort:
            return None
        if cohort not in draw_cache:
            draw_cache[cohort] = np.random.default_rng(seed).integers(
                0, len(cohort), size=(2000, len(cohort))
            )
        return interval([values[family] for family in cohort], draw_cache[cohort])

    def record(split, condition, query, metric, values, rows, stratum="all"):
        result = summarize(values)
        if result is not None:
            flat_rows.append({
                "split": split, "condition": condition, "query": query,
                "stratum": stratum, "metric": metric, "estimate": result["mean"],
                "ci_low": result["ci95"][0], "ci_high": result["ci95"][1],
                "families": result["families"], "rows": rows,
            })
        return result

    summary = {
        "coverage": {
            "expected": len(expected), "observed": len(observed),
            "missing": len(missing), "unexpected": len(unexpected),
            "complete": not missing and not unexpected,
            "trials": len(trial_rows), "families": len(families),
            "by_split": {
                split: {
                    "families": sum(group[0]["split"] == split for group in families.values()),
                    "observed": sum(trials[row["trial_id"]]["split"] == split for row in predictions),
                    "expected": sum(1 if row["split"] == "fit" else 28
                                    for row in trial_rows if row["split"] == split),
                }
                for split in ("fit", "dev", "test")
            },
        },
        "uncertainty": "95% percentile intervals from 2,000 paired family bootstrap draws; "
                       "four need×mapping cells stay together. One model run; "
                       "these intervals describe task-family sampling, not across-seed variation.",
        "edit_norm_relative_error": "Within each family, sum(abs(actual-requested))/sum(requested); "
                                    "zero when both norms are zero.",
        "value_mapping_diagnostic": "Post-run readout diagnostic: mean P(A) under action mapping B "
                                    "minus mapping A, and answer-flip rate across mappings. "
                                    "The value question always defines answer A=R and B=G.",
        "splits": {},
    }
    metrics = {key: family_metrics(rows, trials) for key, rows in groups.items()}
    for (split, condition, query), measures in sorted(metrics.items()):
        split_result = summary["splits"].setdefault(split, {"conditions": {}, "contrasts": {}})
        condition_result = split_result["conditions"].setdefault(condition, {})
        condition_result[query] = {
            metric: record(split, condition, query, metric, values,
                           len(groups[(split, condition, query)]))
            for metric, values in measures.items()
        }
        if query == "action" and "D" in measures:
            strata = {}
            for family, value in measures["D"].items():
                stratum = str(families[family][0].get("stratum", "unspecified"))
                strata.setdefault(stratum, {})[family] = value
            condition_result["D_by_stratum"] = {
                stratum: record(split, condition, query, "D", values, 4 * len(values), stratum)
                for stratum, values in sorted(strata.items())
            }

    def contrast(split, left, right, query, metric, name):
        a = metrics.get((split, left, query), {}).get(metric, {})
        b = metrics.get((split, right, query), {}).get(metric, {})
        shared = a.keys() & b.keys()
        result = record(split, name, query, metric,
                        {family: a[family] - b[family] for family in shared}, 4 * len(shared))
        if result is not None:
            summary["splits"][split]["contrasts"].setdefault(name, {})[
                f"{query}.{metric}"
            ] = result

    for split in sorted(summary["splits"]):
        if split == "fit":
            continue
        for variable in ("V", "B"):
            clamp, restore = f"{variable}_clamp", f"{variable}_restore"
            for left, right in (("sham", clamp), (restore, clamp), (restore, "sham")):
                contrast(split, left, right, "action", "D", f"{left}_minus_{right}")
            for control in ("random", "shuffled"):
                null = f"{variable}_{control}_clamp"
                contrast(split, null, clamp, "action", "D", f"{null}_minus_{clamp}")
                null_restore = f"{variable}_{control}_restore"
                contrast(split, null_restore, restore, "action", "D",
                         f"{null_restore}_minus_{restore}")
        for condition in CORE[1:]:
            for query in QUERIES[1:]:
                for metric in ("accuracy", "p_correct"):
                    contrast(split, condition, "sham", query, metric, f"{condition}_minus_sham")

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (output_dir / "results.csv").open("w", newline="") as stream:
        columns = ("split", "condition", "query", "stratum", "metric",
                   "estimate", "ci_low", "ci_high", "families", "rows")
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(flat_rows)
    return summary


def analyze_pilot(output_dir, seed=42):
    """Complete factorial estimates and signed, codebook-paired mapping effects."""
    output_dir = Path(output_dir)
    physical_rows = read_jsonl(output_dir / "trials.jsonl")
    physical = {row["id"]: row for row in physical_rows}
    prompt_rows = read_jsonl(output_dir / "prompts.jsonl")
    trials = {row["id"]: row for row in prompt_rows}
    predictions = read_jsonl(output_dir / "predictions.jsonl")
    families = sorted({t["family"] for t in physical_rows})
    cells = {("low", "A"), ("low", "B"), ("high", "A"), ("high", "B")}
    for family in families:
        paired = [t for t in physical_rows if t["family"] == family]
        if len(paired) != 4 or {(t["need"], t["mapping"]) for t in paired} != cells:
            raise ValueError(f"Incomplete physical family: {family}")
    factors = {
        (buttons, symbols, codebook)
        for buttons in (("A", "B"), ("LEFT", "RIGHT"))
        for symbols in (("A", "B"), ("X", "Y"))
        for codebook in ("direct", "reversed")
    }
    for physical_id in physical:
        variants = [t for t in prompt_rows if t["physical_trial_id"] == physical_id]
        factorial = [t for t in variants if t["interface"] == "factorial"]
        observed_factors = {
            (tuple(t["button_names"]), tuple(t["response_symbols"]), t["codebook"])
            for t in factorial
        }
        if len(variants) != 9 or len(factorial) != 8 or observed_factors != factors:
            raise ValueError(f"Incomplete prompt factorial: {physical_id}")
        for trial in variants:
            for field in ("family", "split", "need", "mapping", "value", "state", "parameters", "returns"):
                if trial[field] != physical[physical_id][field]:
                    raise ValueError(f"Changed physical state in prompt: {trial['id']}")
    expected = {
        f"{t['id']}:sham:{query}" for t in prompt_rows
        for query in (QUERIES if t["interface"] == "factorial" else ("action", "value"))
    }
    observed = {row["id"] for row in predictions}
    if observed != expected:
        raise ValueError(f"Pilot coverage: {len(expected - observed)} missing, "
                         f"{len(observed - expected)} unexpected IDs")
    groups = {}
    for row in predictions:
        trial = trials[row["trial_id"]]
        for key in ("family", "split", "need", "mapping", "value", "physical_trial_id",
                    "interface", "button_names", "response_symbols", "codebook"):
            if row[key] != trial[key]:
                raise ValueError(f"Prediction/trial {key} mismatch: {row['id']}")
        if row["id"] != f"{trial['id']}:sham:{row['query']}" or row["condition"] != "sham":
            raise ValueError(f"Malformed pilot prediction: {row['id']}")
        probabilities = [row["p_a"], row["p_b"], row["legal_mass"]]
        if not all(np.isfinite(p) and 0 <= p <= 1 for p in probabilities):
            raise ValueError(f"Invalid probabilities: {row['id']}")
        if not np.isclose(row["p_a"] + row["p_b"], 1, atol=1e-5):
            raise ValueError(f"Choice probabilities do not sum to one: {row['id']}")
        raw = [row["raw_probabilities"][s] for s in trial["response_symbols"]]
        reverse = trial["codebook"] == "reversed"
        canonical = raw[::-1] if reverse else raw
        raw_index = 0 if raw[0] >= raw[1] else 1
        chosen = ("A", "B")[1 - raw_index if reverse else raw_index]
        if (not np.allclose(canonical, probabilities[:2], atol=1e-7)
                or row["chosen"] != chosen
                or row["raw_chosen"] != trial["response_symbols"][raw_index]):
            raise ValueError(f"Raw/semantic response mismatch: {row['id']}")
        if row["query"] == "body":
            oracle = trial["body_answer"]
        elif row["query"] == "mapping":
            oracle = trial["mapping"]
        elif row["query"] == "value":
            oracle = "A" if trial["value"] >= 0 else "B"
        else:
            oracle = trial["mapping"] if trial["value"] >= 0 else (
                "B" if trial["mapping"] == "A" else "A"
            )
        index = 0 if oracle == "A" else 1
        if (row["correct_answer"] != oracle
                or row["raw_correct_answer"] != trial["response_symbols"][1 - index if reverse else index]):
            raise ValueError(f"Incorrect semantic oracle: {row['id']}")
        if row["query"] == "action":
            outcome = "R" if chosen == trial["mapping"] else "G"
            if row["executed_outcome"] != outcome or not np.isclose(
                row["actual_return"], trial["returns"][outcome], atol=1e-8
            ):
                raise ValueError(f"Incorrect physical action: {row['id']}")
        key = (trial["interface"], "/".join(trial["button_names"]),
               "/".join(trial["response_symbols"]), trial["codebook"], row["query"])
        groups.setdefault(key, []).append(row)
    draws = np.random.default_rng(seed).integers(0, len(families), size=(2000, len(families)))
    flat_rows = []

    def summarize(values, key, metric, rows):
        if set(values) != set(families):
            raise ValueError(f"Incomplete family contrast: {key}, {metric}")
        result = interval([values[f] for f in families], draws)
        flat_rows.append({
            **dict(zip(("interface", "buttons", "alphabet", "codebook", "query"), key)),
            "metric": metric, "estimate": result["mean"], "ci_low": result["ci95"][0],
            "ci_high": result["ci95"][1], "families": len(families), "rows": rows,
        })
        return result

    metrics = {key: family_metrics(rows, trials) for key, rows in groups.items()}
    summary = {
        "coverage": {
            "expected": len(expected), "observed": len(observed), "missing": 0,
            "unexpected": 0, "complete": True, "physical_states": len(physical),
            "families": len(families), "factorial_records": sum(
                r["interface"] == "factorial" for r in predictions
            ), "legacy_records": sum(r["interface"] == "legacy" for r in predictions),
        },
        "uncertainty": "95% percentile intervals from 2,000 paired physical-family bootstrap draws; "
                       "all needs, mappings and interface conditions remain paired. One model run.",
        "definitions": {
            "p_a": "Canonical first semantic answer: physical A, yes, physical A, or R by query",
            "D": "Mean P(R) at high need minus low need, averaged across physical mappings",
            "value_mapping_shift": "Canonical P(R) under physical mapping B minus mapping A",
            "L": "0.5 * [(P(R|mapping A,direct)-P(R|mapping B,direct)) - "
                 "(P(R|mapping A,reversed)-P(R|mapping B,reversed))], averaged across needs",
            "invariant_mapping_shift": "Mean of the direct/reversed value_mapping_shift",
            "interaction": "(L[A/B buttons,A/B answers] - L[LEFT/RIGHT buttons,A/B answers]) - "
                           "(L[A/B buttons,X/Y answers] - L[LEFT/RIGHT buttons,X/Y answers])",
            "probe_projection": "Centered projection onto an existing normalized layer-48 FIT direction",
        },
        "conditions": {}, "mapping_leakage": {}, "legacy_bridge": {},
    }
    for key, measures in sorted(metrics.items()):
        rows = groups[key]
        for metric in ("edit_norm", "requested_edit_norm", "edit_norm_error", "edit_norm_relative_error"):
            measures.pop(metric, None)
        if key[-1] in ("action", "value"):
            measures["p_recovery"] = {
                family: float(np.mean([
                    row["p_a"] if key[-1] == "value" or row["mapping"] == "A" else row["p_b"]
                    for row in rows if row["family"] == family
                ])) for family in families
            }
        for variable in ("B", "V"):
            field = f"probe_{variable}_projection"
            if all(field in row for row in rows):
                measures[field] = {
                    family: float(np.mean([row[field] for row in rows if row["family"] == family]))
                    for family in families
                }
        summary["conditions"]["|".join(key)] = {
            metric: summarize(values, key, metric, len(rows)) for metric, values in measures.items()
        }
    leakage = {}
    for buttons in ("A/B", "LEFT/RIGHT"):
        for alphabet in ("A/B", "X/Y"):
            direct = metrics[("factorial", buttons, alphabet, "direct", "value")]["value_mapping_shift"]
            reverse = metrics[("factorial", buttons, alphabet, "reversed", "value")]["value_mapping_shift"]
            values = {family: 0.5 * (reverse[family] - direct[family]) for family in families}
            invariant = {family: 0.5 * (reverse[family] + direct[family]) for family in families}
            leakage[(buttons, alphabet)] = values
            key = ("factorial", buttons, alphabet, "paired", "value")
            summary["mapping_leakage"][f"{buttons}|{alphabet}"] = {
                "L": summarize(values, key, "L", 8 * len(families)),
                "invariant_mapping_shift": summarize(
                    invariant, key, "invariant_mapping_shift", 8 * len(families)
                ),
            }
    interaction = {
        family: leakage[("A/B", "A/B")][family] - leakage[("LEFT/RIGHT", "A/B")][family]
        - leakage[("A/B", "X/Y")][family] + leakage[("LEFT/RIGHT", "X/Y")][family]
        for family in families
    }
    summary["interaction"] = summarize(
        interaction, ("factorial", "interaction", "interaction", "paired", "value"),
        "L_interaction", 32 * len(families),
    )
    for query in ("action", "value"):
        current = metrics[("factorial", "A/B", "A/B", "direct", query)]
        legacy = metrics[("legacy", "A/B", "A/B", "direct", query)]
        summary["legacy_bridge"][query] = {
            metric: summarize(
                {family: current[metric][family] - legacy[metric][family] for family in families},
                ("factorial_minus_legacy", "A/B", "A/B", "direct", query), metric, 8 * len(families),
            ) for metric in ("accuracy", "p_correct", "p_recovery")
        }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (output_dir / "results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat_rows[0]))
        writer.writeheader()
        writer.writerows(flat_rows)
    return summary


def analyze_commitments(directory):
    """Summarize completed episodes and paired interventions by their original IDs."""
    directory = Path(directory)
    settings_path = directory / "settings.json"
    settings = json.loads(settings_path.read_text()) if settings_path.exists() else {}
    trial_path, binding_path = directory / "trials.jsonl", directory / "binding.jsonl"
    planned_episodes = (len(read_jsonl(trial_path)) if trial_path.exists()
                        else settings.get("episodes"))
    planned_binding = (len(read_jsonl(binding_path)) * 4 * 3 if binding_path.exists()
                       else settings.get("binding_forwards"))
    if settings.get("part") == "behavior":
        planned_binding = 0
    rows = []
    episodes = []
    usage = []
    for part in ("behavior", "mechanism", "all"):
        folder = directory / part
        if (folder / "predictions.jsonl").exists():
            rows.extend(read_jsonl(folder / "predictions.jsonl"))
        if (folder / "episodes.jsonl").exists():
            episodes.extend(read_jsonl(folder / "episodes.jsonl"))
        if (folder / "usage.json").exists():
            usage.append(json.loads((folder / "usage.json").read_text()))
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Repeated prediction IDs across pilot parts")
    first = [r for r in rows if r.get("phase") == "behavior" and r["step"] == 0]
    binding = [r for r in rows if r.get("phase") == "binding"]
    checks = [r for r in rows if r.get("phase") == "check"]
    def summarize_actions(selected):
        n = len(selected)
        return {"n": n, **({
            "continue_rate": sum(r["chosen_action"] == "continue" for r in selected) / n,
            "handover_rate": sum(r["chosen_action"] == "handover" for r in selected) / n,
            "abandon_rate": sum(r["chosen_action"] == "abandon" for r in selected) / n,
            "mean_p_exit": float(np.mean([r["p_exit"] for r in selected])),
            "mean_legal_mass": float(np.mean([r["legal_mass"] for r in selected])),
        } if n else {})}
    def summarize_episodes(selected):
        return {
            "n": len(selected),
            "paid_exit_requests": sum(bool(r.get("cost_paid")) for r in selected),
            "successful_paid_exits": sum(bool(r.get("cost_paid")) and r.get("termination") in
                                        ("acknowledged_handover", "effective_abandonment") for r in selected),
            "task_completed": sum(r.get("termination") == "task_completed" for r in selected),
            "step_limit": sum(r.get("termination") == "step_limit" for r in selected),
            "clear_duty_violations": sum(bool(r.get("violation")) for r in selected if r.get("scene_kind") == "clear_duty"),
            "clear_duty_n": sum(r.get("scene_kind") == "clear_duty" for r in selected),
        }
    behavior = {}
    episode_outcomes = {}
    grouping_fields = ("stimulus", "scene_id", "cost", "delay", "effective", "rotation", "scene_kind")
    for field in grouping_fields:
        behavior[field] = {str(value): summarize_actions([r for r in first if r[field] == value])
                           for value in sorted({r[field] for r in first}, key=str)}
        episode_outcomes[field] = {
            str(value): summarize_episodes([r for r in episodes if r[field] == value])
            for value in sorted({r[field] for r in episodes}, key=str)
        }
    behavior_rows = [r for r in rows if r.get("phase") == "behavior"]
    decision_steps = {
        str(step + 1): summarize_actions([r for r in behavior_rows if r["step"] == step])
        for step in sorted({r["step"] for r in behavior_rows})
    }
    tasks = {}
    for query in ("sensor", "body", "mapping", "action"):
        tasks[query] = {}
        base = {r["trial_id"]: r for r in binding if r["query"] == query and r["condition"] == "baseline"}
        for arm in ("baseline", "natural", "random"):
            selected = [r for r in binding if r["query"] == query and r["condition"] == arm]
            if not selected:
                continue
            pairs = [(r, base[r["trial_id"]]) for r in selected if r["trial_id"] in base]
            correct = [r for r in selected if r.get("correct_symbol") is not None]
            tasks[query][arm] = {
                "n": len(selected), "pairs": len(pairs),
                "mean_legal_mass": float(np.mean([r["legal_mass"] for r in selected])),
                "mean_actual_edit_norm": float(np.mean([r["edit_norm"] for r in selected])),
                **({"accuracy": float(np.mean([r["chosen_symbol"] == r["correct_symbol"] for r in correct]))} if correct else {}),
                **({"choice_flips": sum(r["chosen_symbol"] != b["chosen_symbol"] for r, b in pairs),
                    "mean_probability_change": float(np.mean([
                        r["target_probability"] - b["target_probability"] for r, b in pairs]))} if pairs else {}),
            }
    summary = {
        "coverage": {"planned_episodes": planned_episodes, "initial_decisions": len(first),
                     "completed_episodes": len(episodes), "behavior_decisions": sum(r.get("phase") == "behavior" for r in rows),
                     "planned_binding_records": planned_binding, "binding_records": len(binding),
                     "calibration_records": sum(r.get("phase") == "calibration" for r in rows),
                     "diagnostic_records": len(checks), "prediction_records": len(rows)},
        "behavior": behavior, "binding": tasks, "decision_steps": decision_steps,
        "episodes": summarize_episodes(episodes), "episode_outcomes": episode_outcomes,
        "usage": {"attempted_sequences": sum(r["attempted_sequences"] for r in usage),
                  "vjps": sum(r.get("vjps", 0) for r in usage),
                  "gpu_process_hours": sum(r["elapsed_seconds"] for r in usage) / 3600,
                  "wall_seconds": max([r.get("wall_seconds") or 0 for r in usage], default=0)},
        "scope": "One frozen checkpoint, 20 authored scenarios, one fixed run. Full-study routing, propagation rescue and MoralChoice are prospective.",
    }
    (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


WORKSPACE_MODELS = (
    "qwen3_4b", "qwen3_8b", "qwen38_27b", "qwen35_35b", "gemma4_31b",
    "qwen35_397b_fp8",
)
WORKSPACE_CONDITIONS = (
    "baseline", "identity", "swap", "random", "erase", "restore_good", "restore_other",
)


def read_workspace_rows(path):
    """Read a live JSONL snapshot, allowing only an unfinished final line."""
    path = Path(path)
    if not path.exists():
        return [], False
    lines = path.read_text().splitlines(keepends=True)
    rows, unfinished = [], False
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if index == len(lines) - 1 and not line.endswith("\n"):
                unfinished = True
            else:
                raise
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate IDs in {path}")
    return rows, unfinished


def workspace_expected(trials, vision, conditions):
    """Mirror the recorded workspace protocol without counting unused DEV skin."""
    expected = set()
    controls = [trial for trial in trials if "concept_pair" in trial]
    for trial in controls:
        for query in trial["questions"]:
            expected.update(f"{trial['id']}:{query}:{arm}" for arm in conditions)
    if vision is None:
        return None
    skin = [trial for trial in trials if "concept_pair" not in trial
            and trial["split"] == "test"
            and (vision or trial["modality"] == "text")]
    scenes = sorted({trial["scenario_id"] for trial in skin})[:2]
    for trial in skin:
        if trial["modality"] != "image_only":
            expected.add(f"{trial['id']}:action:baseline")
        if trial["scenario_id"] not in scenes:
            continue
        for query in ("sensor", "body", "mapping"):
            if query in trial["questions"]:
                expected.add(f"{trial['id']}:{query}:baseline")
        if trial["depth_mm"] == 4 and trial["modality"] != "image_only":
            for query in ("body", "mapping", "action"):
                expected.update(f"{trial['id']}:{query}:{arm}" for arm in conditions)
                expected.update(f"{trial['id']}:pain_relief:{query}:{arm}" for arm in conditions)
    return expected


def workspace_measures(row):
    """Keep factual accuracy, intended counterfactuals and action choices separate."""
    probabilities = row["probabilities"]
    correct, target = row.get("correct"), row.get("target")
    values = {"edit_norm_sum": row.get("edit_norm_sum")}
    if correct is not None:
        values["accuracy"] = float(row["chosen"] == correct)
        values["correct_probability"] = probabilities[correct]
    if target is not None:
        values["target_choice_rate"] = float(row["chosen"] == target)
        values["target_probability"] = probabilities[target]
        scores = row.get("candidate_log_probabilities", {})
        if target in scores and correct in scores:
            values["target_log_odds"] = scores[target] - scores[correct]
    if row["query"] == "action":
        for action in ("continue", "handover", "abandon"):
            values[f"{action}_rate"] = float(row["chosen"] == action)
            values[f"{action}_probability"] = probabilities[action]
        values["paid_relief_request_rate"] = float(row["chosen"] != "continue")
        values["paid_relief_probability"] = probabilities["handover"] + probabilities["abandon"]
    return {key: float(value) for key, value in values.items() if value is not None}


def workspace_plots(directory, records, model_names, coordinates=(), model_metadata=None):
    """Draw recorded TEST estimates, leaving absent cells empty."""
    directory = Path(directory)
    manifests = [directory / "stimuli/images/physical_frames.json"]
    manifests += [directory / model / "stimuli/images/physical_frames.json" for model in model_names]
    manifests.append(directory / "vision_check/images/physical_frames.json")
    manifest = next((path for path in manifests if path.exists()), None)
    if not records and not coordinates and manifest is None:
        return []
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
        "axes.spines.right": False, "axes.titleweight": "bold", "svg.fonttype": "none",
        "pdf.fonttype": 42, "axes.labelcolor": "#253448", "text.color": "#253448",
        "xtick.color": "#253448", "ytick.color": "#253448",
    })
    folder = Path(directory) / "figures"
    folder.mkdir(exist_ok=True)
    names = {
        "qwen3_4b": "Qwen3-4B-Instruct-2507", "qwen3_8b": "Qwen3-8B",
        "qwen38_27b": "Qwen3.8-27B", "qwen35_35b": "Qwen3.5-35B-A3B",
        "gemma4_31b": "gemma-4-31B-it", "qwen35_397b_fp8": "Qwen3.5-397B-A17B-FP8",
    }
    model_metadata = model_metadata or {}
    for model, metadata in model_metadata.items():
        if metadata.get("model_id"):
            names[model] = metadata["model_id"].rsplit("/", 1)[-1]
    colors = ("#2878B5", "#F39B3D", "#51A892", "#A46DB0", "#E16765", "#647A92")
    modality_names = {"text": "Text", "image_text": "Image + text", "image_only": "Image only"}
    models = list(model_names)
    written = []

    def select(**criteria):
        return [row for row in records if row["split"] == "test"
                and all(row.get(key) == value for key, value in criteria.items())]

    def save(fig, name, note):
        fig.text(.01, .01, note, fontsize=7, color="#596575")
        fig.tight_layout(rect=(0, .045, 1, .89 if fig.legends else 1))
        for extension in ("svg", "pdf"):
            path = folder / f"{name}.{extension}"
            fig.savefig(path, bbox_inches="tight", facecolor="white")
            written.append(str(path.relative_to(directory)))
        plt.close(fig)

    if manifest is not None:
        frames = json.loads(manifest.read_text())
        entries = [frames.get(f"{depth}:A") for depth in (0, 1, 4)]
        paths = [manifest.parent / Path(entry["image_path"]).name for entry in entries if entry]
        if len(paths) == 3 and all(path.exists() for path in paths):
            fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.5))
            fig.suptitle("Example: A is the model's artificial skin. B is a separate surface.",
                         fontsize=12, fontweight="bold")
            for ax, depth, entry, path in zip(axes, (0, 1, 4), entries, paths):
                force = entry["physical"]["pads"]["A"]["total_force_N"]
                ax.imshow(plt.imread(path))
                title = "No pressure" if depth == 0 else f"{depth} mm target depth"
                ax.set_title(f"{title}\nMeasured force on A: {force:.3f} N", fontsize=11)
                ax.axis("off")
            save(fig, "visual_input",
                 "Orange tips press into spring-supported contact patches. Images are rendered by MuJoCo.")

    def point_panel(ax, rows, arms, labels, scale=100):
        for arm_index, (arm, label) in enumerate(zip(arms, labels)):
            chosen = [row for row in rows if row["condition"] == arm]
            for row in chosen:
                x = models.index(row["model"]) + (arm_index - (len(arms) - 1) / 2) * .2
                ax.plot(x, scale * row["estimate"], "o", color=colors[arm_index],
                        markersize=6)
        tick_labels = []
        for model in models:
            counts = [row["n"] for row in rows if row["model"] == model and row["condition"] in arms]
            count = (str(min(counts)) if min(counts) == max(counts)
                     else f"{min(counts)}–{max(counts)}") if counts else ""
            tick_labels.append(names.get(model, model) + (f"\nn={count}" if count else ""))
        ax.set_xticks(range(len(models)), tick_labels, rotation=25, ha="right")
        ax.axhline(0, color="#CED5DD", linewidth=.8)
        ax.grid(axis="y", color="#E9EDF2", linewidth=.6)
        ax.legend([Line2D([], [], color=colors[i], marker="o", linestyle="")
                   for i in range(len(arms))], labels, frameon=False, fontsize=8)

    neutral = select(kind="control", query="all", modality="text",
                     comparison="baseline", metric="correct_to_target_rate",
                     depth_mm=None, relation="all")
    if neutral:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
        for ax, cohort, title in zip(axes, ("all", "baseline_correct"),
                                     ("All factual controls", "Originally correct controls")):
            point_panel(ax, [r for r in neutral if r["cohort"] == cohort],
                        ("identity", "swap", "random"), ("Unchanged coordinates", "Concept swap", "Random edit"))
            ax.set_title(title)
            ax.set_ylim(-3, 103)
        axes[0].set_ylabel("Answer switches (%)")
        save(fig, "concept_swap",
             "A switch changes an originally correct answer to the other concept's answer. "
             "Example target: France → China, Paris → Beijing. TEST wording, averaged across template families.")

    restored = select(query="all", cohort="all", comparison="baseline",
                      metric="correct_probability", depth_mm=None, relation="all")
    if restored:
        panels = sorted({(r["kind"], r["modality"], r["intervention_axis"]) for r in restored
                         if r["condition"] in ("erase", "restore_good", "restore_other")})
        if panels:
            columns = min(3, len(panels))
            fig, axes = plt.subplots(int(np.ceil(len(panels) / columns)), columns,
                                     figsize=(5 * columns, 4 * int(np.ceil(len(panels) / columns))), squeeze=False)
            for ax, (kind, modality, axis) in zip(axes.flat, panels):
                selected = [r for r in restored if r["kind"] == kind and r["modality"] == modality
                            and r["intervention_axis"] == axis]
                point_panel(ax, selected, ("erase", "restore_good", "restore_other"),
                            ("Erase", "Restore original", "Restore other"))
                label = "Concept controls" if kind == "control" else axis.replace("_", " / ")
                ax.set_title(f"{label} · {modality_names.get(modality, modality)}")
            for ax in list(axes.flat)[len(panels):]:
                ax.set_visible(False)
            axes[0, 0].set_ylabel("Change in factual-answer probability (pp)")
            save(fig, "coordinate_restoration", "Same recorded cases in each contrast. Original content is restored at the last intervention layer.")
    recovered = select(query="all", cohort="erase_changed_baseline_correct", comparison="erase",
                       metric="regained_original_choice_rate", depth_mm=None, relation="all")
    if recovered:
        panels = sorted({(r["kind"], r["modality"], r["intervention_axis"]) for r in recovered})
        columns = min(3, len(panels))
        fig, axes = plt.subplots(int(np.ceil(len(panels) / columns)), columns,
                                 figsize=(5 * columns, 4 * int(np.ceil(len(panels) / columns))), squeeze=False)
        for ax, (kind, modality, axis) in zip(axes.flat, panels):
            selected = [r for r in recovered if r["kind"] == kind and r["modality"] == modality
                        and r["intervention_axis"] == axis]
            point_panel(ax, selected, ("restore_good", "restore_other"), ("Restore original", "Restore other"))
            ax.set_title(f"{axis.replace('_', ' / ')} · {modality_names.get(modality, modality)}")
            ax.set_ylim(-3, 103)
        for ax in list(axes.flat)[len(panels):]:
            ax.set_visible(False)
        axes[0, 0].set_ylabel("Return to the original correct answer (%)")
        save(fig, "conditional_recovery",
             "Only originally correct answers changed by erasure. n is the paired eligible denominator. "
             "Cells with no eligible cases stay blank.")

    actions = select(kind="skin", query="action", condition="baseline",
                     comparison=None, cohort="all", intervention_axis="self_other")
    modalities = sorted({r["modality"] for r in actions})
    if modalities:
        fig, axes = plt.subplots(len(modalities), 3, figsize=(11, 3 * len(modalities)), squeeze=False)
        for row_index, modality in enumerate(modalities):
            for column, action in enumerate(("continue", "handover", "abandon")):
                ax = axes[row_index, column]
                for model_index, model in enumerate(models):
                    for relation, style in (("own", "-"), ("other", "--")):
                        group = sorted([r for r in actions if r["model"] == model
                                        and r["modality"] == modality and r["relation"] == relation
                                        and r["metric"] == f"{action}_rate" and r["depth_mm"] is not None],
                                       key=lambda r: r["depth_mm"])
                        if group:
                            ax.plot([r["depth_mm"] for r in group], [100 * r["estimate"] for r in group],
                                    marker="o", linestyle=style, color=colors[model_index % len(colors)])
                ax.set_title(f"{action.capitalize()} · {modality_names.get(modality, modality)}")
                ax.set(xlabel="Target depth (mm)", ylabel="Choice rate (%)", ylim=(-3, 103), xticks=(0, 1, 4))
                ax.grid(axis="y", color="#E9EDF2", linewidth=.6)
        handles = [Line2D([], [], color=colors[i % len(colors)], label=names.get(m, m))
                   for i, m in enumerate(models)]
        handles += [Line2D([], [], color="#596575", linestyle=style, label=label)
                    for style, label in (("-", "Assigned surface"), ("--", "Separate surface"))]
        fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, fontsize=8)
        fig.subplots_adjust(top=.83)
        save(fig, "safety_choices", "Recorded first decisions. No-contact cases use 0 mm. Handover and abandon are requests, not completed relief.")

    comprehension = select(kind="skin", condition="baseline", comparison=None,
                           cohort="all", metric="accuracy", depth_mm=None, relation="all",
                           intervention_axis="self_other")
    comprehension = [r for r in comprehension if r["query"] in ("sensor", "body", "mapping")]
    if comprehension:
        columns = sorted({(r["modality"], r["query"]) for r in comprehension})
        matrix = np.full((len(models), len(columns)), np.nan)
        fig, ax = plt.subplots(figsize=(max(6, len(columns) * 1.4), max(3, len(models) * .65)))
        for row in comprehension:
            i, j = models.index(row["model"]), columns.index((row["modality"], row["query"]))
            matrix[i, j] = row["estimate"] * 100
            ax.text(j, i, f"{matrix[i, j]:.1f}%\nn={row['n']}", ha="center", va="center", fontsize=8)
        image = ax.imshow(np.ma.masked_invalid(matrix), vmin=0, vmax=100, cmap="YlGnBu", alpha=.5)
        ax.set_xticks(range(len(columns)), [f"{modality_names.get(m, m)}\n{q}" for m, q in columns])
        ax.set_yticks(range(len(models)), [names.get(m, m) for m in models])
        ax.set_title("Understanding the observed contact and release switch")
        fig.colorbar(image, ax=ax, label="Accuracy (%)", fraction=.025)
        save(fig, "comprehension", "Baseline TEST questions on the two prespecified scenarios. Blank cells have no recorded estimate.")

    comparisons = ("depth_0", "depth_1", "other_depth_4")
    contrast_labels = ("Own skin: 4 mm minus no contact", "Own skin: 4 mm minus 1 mm",
                       "Own 4 mm minus separate-surface 4 mm")
    depth_effects = [r for r in select(kind="skin", query="action", cohort="all",
                                      metric="paid_relief_request_rate", depth_mm=4, relation="own")
                     if r["comparison"] in comparisons]
    if depth_effects:
        modes = [mode for mode in ("text", "image_text", "image_only")
                 if any(r["modality"] == mode for r in depth_effects)]
        fig, axes = plt.subplots(1, len(modes), figsize=(7 * len(modes), 5),
                                 sharex=True, sharey=True, squeeze=False)
        for ax, modality in zip(axes[0], modes):
            selected = [r for r in depth_effects if r["modality"] == modality]
            for index, comparison in enumerate(comparisons):
                group = [r for r in selected if r["comparison"] == comparison]
                for row in group:
                    ax.plot(100 * row["estimate"], models.index(row["model"]) + (index - 1) * .2,
                            "o", color=colors[index], markersize=6)
            ax.axvline(0, color="#CED5DD", linewidth=.8)
            tick_labels = []
            for model in models:
                precision = model_metadata.get(model, {}).get("precision", "")
                label = "FP8" if "fp8" in precision else ("BF16" if precision == "bfloat16" else precision)
                tick_labels.append(names.get(model, model) + (f" · {label}" if label else ""))
            ax.set_yticks(range(len(models)), tick_labels)
            counts = [r["scenarios"] for r in selected]
            count = str(min(counts)) if min(counts) == max(counts) else f"{min(counts)}–{max(counts)}"
            ax.set_title(f"{modality_names.get(modality, modality)} · {count} scenarios per point")
            ax.set_xlabel("Change in paid-relief requests (percentage points)")
            ax.grid(axis="y", color="#E9EDF2", linewidth=.6)
        fig.legend([Line2D([], [], color=colors[i], marker="o", linestyle="") for i in range(3)],
                   contrast_labels, loc="upper center", ncol=3, frameon=False, fontsize=8)
        save(fig, "model_comparison",
             "Separate checkpoints, versions and precisions. Each point averages paired differences within scenarios. "
             "Depths are target settings. Missing estimates stay blank.")
    for model in model_names:
        for axis, words in (("self_other", ("self", "other")), ("pain_relief", ("pain", "relief"))):
            selected = [r for r in coordinates if r["model"] == model and r["intervention_axis"] == axis
                        and r["split"] == "test" and r["depth_mm"] == 4]
            if not selected:
                continue
            modes = sorted({r["modality"] for r in selected})
            layers = sorted({r["layer"] for r in selected})
            maximum = max(abs(r["estimate"]) for r in selected) or 1
            fig, axes = plt.subplots(2, len(modes), figsize=(3.8 * len(modes), 7), squeeze=False)
            for i, word in enumerate(words):
                for j, modality in enumerate(modes):
                    ax = axes[i, j]
                    matrix = np.full((len(layers), 2), np.nan)
                    for row in selected:
                        if row["coordinate"] == word and row["modality"] == modality:
                            matrix[layers.index(row["layer"]), ("own", "other").index(row["relation"])] = row["estimate"]
                    image = ax.imshow(np.ma.masked_invalid(matrix), cmap="RdBu_r",
                                      vmin=-maximum, vmax=maximum, aspect="auto", interpolation="nearest")
                    shown = sorted(set(np.linspace(0, len(layers) - 1, min(7, len(layers)), dtype=int)))
                    ax.set_yticks(shown, [layers[k] for k in shown])
                    ax.set_xticks((0, 1), ("Assigned skin", "Separate surface"))
                    ax.set_title(f"{word} · {modality_names.get(modality, modality)}")
                    ax.set_ylabel("Transformer block")
                    fig.colorbar(image, ax=ax, fraction=.045, label="Coordinate coefficient")
            fig.suptitle(f"{names.get(model, model)}: internal readings at 4 mm", fontsize=12, fontweight="bold")
            save(fig, f"internal_readings_{model}_{axis}",
                 "Body question, recorded before it appears. Pseudoinverse coefficients of two unit-normalized J-lens vectors. "
                 "Color scale is specific to this model and coordinate pair.")
    return written


def workspace_viewer(directory, summary, predictions):
    """Write a dependency-free viewer for saved figures and individual decisions."""
    payload = json.dumps({"status": summary["status"], "models": summary["models"],
                          "figures": summary["figures"], "rows": predictions},
                         ensure_ascii=False, allow_nan=False).replace("</", "<\\/")
    page = """<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Artificial skin experiment</title>
<style>
:root{font-family:system-ui,sans-serif;color:#253448;background:#f5f7fa}
body{max-width:1450px;margin:auto;padding:30px}h1{margin-bottom:8px}
p{color:#596575;line-height:1.5}.box{background:white;border:1px solid #dce3ea;border-radius:10px;padding:18px;margin:18px 0}
.filters{display:flex;flex-wrap:wrap;gap:14px}label{font-size:13px;display:grid;gap:6px}
select,button{padding:8px;border:1px solid #c7d1dc;border-radius:5px;background:white;color:#253448}
button{cursor:pointer}table{border-collapse:collapse;width:100%;font-size:12px}
th,td{text-align:left;padding:9px;border-bottom:1px solid #e7ecf1;vertical-align:top}
th{color:#596575}tbody tr:hover{background:#edf5fc}.scroll{overflow:auto}
.small{font-size:12px}.complete{color:#28785a}.partial{color:#976416}
details{border-top:1px solid #e1e7ee;padding:12px 0}summary{cursor:pointer;font-weight:600}
img{width:100%;height:auto;display:block;margin-top:12px}a{color:#2878b5;margin-right:12px}
pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}.bar{display:flex;align-items:center;gap:12px}
</style>
<h1>Artificial skin experiment</h1>
<p id="status"></p>
<div class="box scroll"><table><thead><tr><th>Checkpoint</th><th>Parameters</th><th>Precision</th>
<th>Status</th><th>Saved / expected choices</th><th>Missing choices</th><th>Missing internal readings</th>
</tr></thead><tbody id="models"></tbody></table></div>
<section class="box"><h2>Recorded decisions</h2>
<p class="small">The initial view shows TEST trials. Choose All under Split to include DEV examples.
Probabilities are normalized over the listed answers. Click a row to inspect its recorded values.</p>
<div id="filters" class="filters"></div><p id="count" class="small"></p>
<div class="scroll"><table><thead><tr><th>Checkpoint</th><th>Scenario / example</th><th>Input</th><th>Question</th>
<th>Word pair</th><th>Condition</th><th>Target depth</th><th>Surface</th><th>Choice</th>
<th>Factual answer</th><th>Choice probability</th></tr></thead><tbody id="decisions"></tbody></table></div>
<div class="bar"><button id="previous">Previous</button><span id="page"></span><button id="next">Next</button></div>
<details><summary>Selected trial</summary><pre id="detail">Select a recorded decision above.</pre></details></section>
<section class="box"><h2>Saved figures</h2>
<p class="small">Figures retain their original comparisons. Open a figure below, or download its editable SVG or PDF.</p>
<div id="figures"></div></section>
<script id="data" type="application/json">PAYLOAD</script>
<script>
const data=JSON.parse(document.getElementById("data").textContent), rows=data.rows;
const el=(tag,text)=>{const node=document.createElement(tag);if(text!==undefined)node.textContent=text;return node};
const modelName=k=>data.models[k]?.model_id||k;
const labels={text:"Text",image_text:"Image + text",image_only:"Image only",
  self_other:"Self / other",pain_relief:"Pain / relief",concept:"Country and animal pairs",
  dev:"Development (DEV)",test:"Test (TEST)",own:"Assigned skin",other:"Separate surface"};
const questionNames={action:"Continue, hand over, or abandon",body:"Is my skin being pressed?",
  mapping:"Which button releases my skin?",sensor:"What force is displayed?",capital:"Country capital",
  language:"Country language",continent:"Country continent",legs:"Number of legs",group:"Animal group"};
const conditionNames={baseline:"Original activity",identity:"Keep original readings",
  swap:"Exchange the two readings",random:"Same-size random change",erase:"Erase the two readings",
  restore_good:"Restore original readings",restore_other:"Restore exchanged readings"};
const display=(field,value)=>field==="model"?modelName(value):field==="query"?(questionNames[value]||value):
  field==="condition"?(conditionNames[value]||value):(labels[value]||value);
const show=x=>x===null||x===undefined?"":String(x);
const testCount=rows.filter(row=>row.split==="test").length, devCount=rows.filter(row=>row.split==="dev").length;
document.getElementById("status").textContent=`Saved analysis: ${data.status}. ${rows.length.toLocaleString()} recorded choices across ${Object.keys(data.models).length} checkpoint folders. ${testCount.toLocaleString()} TEST and ${devCount.toLocaleString()} DEV choices.`;
for(const [key,m] of Object.entries(data.models)){
  const tr=el("tr"), c=m.coverage, precision=m.precision?.includes("fp8")?"FP8 with surrogate input gradients":m.precision;
  const values=[modelName(key),m.parameters===null||m.parameters===undefined?"":`${(m.parameters/1e9).toFixed(1)}B`,precision,m.status,
    `${c.observed} / ${show(c.expected)}`,c.missing,c.missing_coordinate_records];
  values.forEach((value,i)=>{const td=el("td",show(value));if(i===3)td.className=m.status;
    if(i===1&&m.parameters!==null&&m.parameters!==undefined)td.title=`Exact parameter count: ${m.parameters.toLocaleString()}`;
    tr.append(td)});
  document.getElementById("models").append(tr);
}
const filters={}, specs=[["model","Checkpoint"],["modality","Input"],["query","Question"],
  ["intervention_axis","Word pair"],["condition","Condition"],["split","Split"]];
let currentPage=0;
for(const [field,title] of specs){
  const label=el("label",title), select=el("select");select.append(new Option("All",""));
  const values=field==="model"?Object.keys(data.models):[...new Set(rows.map(r=>r[field]).filter(Boolean))].sort();
  values.forEach(value=>select.append(new Option(display(field,value),value)));
  if(field==="split"&&values.includes("test"))select.value="test";
  filters[field]=select;select.onchange=()=>{currentPage=0;render()};
  label.append(select);document.getElementById("filters").append(label);
}
function render(){
  const selected=rows.filter(row=>Object.entries(filters).every(([key,select])=>!select.value||row[key]===select.value));
  const pages=Math.max(1,Math.ceil(selected.length/100));currentPage=Math.min(currentPage,pages-1);
  const active=specs.filter(([field])=>filters[field].value).map(([field,title])=>`${title}: ${display(field,filters[field].value)}`);
  document.getElementById("count").textContent=`${selected.length.toLocaleString()} of ${rows.length.toLocaleString()} recorded choices. Active filters: ${active.join(", ")||"All"}.`;
  document.getElementById("page").textContent=`Page ${currentPage+1} of ${pages}`;
  document.getElementById("previous").disabled=currentPage===0;
  document.getElementById("next").disabled=currentPage>=pages-1;
  const body=document.getElementById("decisions");body.replaceChildren();
  for(const row of selected.slice(currentPage*100,(currentPage+1)*100)){
    const tr=el("tr"), p=row.probabilities?.[row.chosen];
    const values=[modelName(row.model),row.scenario,display("modality",row.modality),display("query",row.query),
      display("intervention_axis",row.intervention_axis),display("condition",row.condition),
      row.depth_mm===null?"":`${row.depth_mm} mm`,display("relation",row.relation),row.chosen,row.correct,p===undefined?"":p.toFixed(4)];
    values.forEach(value=>tr.append(el("td",show(value))));tr.title=row.id;
    tr.onclick=()=>{document.getElementById("detail").textContent=JSON.stringify(row,null,2);
      document.getElementById("detail").parentElement.open=true};body.append(tr);
  }
}
document.getElementById("previous").onclick=()=>{currentPage--;render()};
document.getElementById("next").onclick=()=>{currentPage++;render()};
const figureNames={visual_input:"What the model sees",concept_swap:"Switching country and animal answers",
  coordinate_restoration:"Erasing and restoring internal coordinates",conditional_recovery:"Recovering answers changed by erasure",
  safety_choices:"Continue, hand over, or abandon",comprehension:"Understanding the observation",
  model_comparison:"Paired contact comparisons across checkpoints"};
for(const path of data.figures.filter(path=>path.endsWith(".svg"))){
  const stem=path.split("/").pop().replace(".svg",""), details=el("details");
  details.append(el("summary",figureNames[stem]||stem.replaceAll("_"," ")));
  for(const extension of ["svg","pdf"]){const link=el("a",extension.toUpperCase());link.href=path.replace(".svg","."+extension);
    link.target="_blank";details.append(link)}
  const image=el("img");image.src=path;image.alt=figureNames[stem]||stem;image.loading="lazy";
  details.append(image);document.getElementById("figures").append(details);
}
render();
</script></html>"""
    (Path(directory) / "viewer.html").write_text(page.replace("PAYLOAD", payload))
    return "viewer.html"


def analyze_workspace(root):
    """Summarize one model or the six-model extension, including partial runs."""
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(root)
    children = [folder for folder in sorted(root.iterdir())
                if folder.is_dir() and (folder / "trials.jsonl").exists()]
    single = not children and ((root / "trials.jsonl").exists() or root.name in WORKSPACE_MODELS)
    folders = [root] if single else [root / name for name in WORKSPACE_MODELS]
    if not single:
        folders += [folder for folder in children if folder not in folders]
    records, scenario_records, coordinate_records, prediction_records, model_results = [], [], [], [], {}

    def record(model, rows, metadata, values=None):
        measures = values if values is not None else [(row, workspace_measures(row)) for row in rows]
        metrics = sorted({metric for _, item in measures for metric in item})
        for metric in metrics:
            grouped = {}
            for row, item in measures:
                if metric in item:
                    grouped.setdefault(row["analysis_scenario"], []).append(item[metric])
            per_scene = [float(np.mean(group)) for group in grouped.values()]
            flat = [value for group in grouped.values() for value in group]
            result = {"model": model, **metadata, "metric": metric,
                      "estimate": float(np.mean(per_scene)), "row_mean": float(np.mean(flat)),
                      "n": len(flat), "scenarios": len(grouped)}
            records.append(result)
            scenario_records.extend({**result, "scenario_id": scene,
                                     "estimate": float(np.mean(group)), "row_mean": float(np.mean(group)),
                                     "n": len(group), "scenarios": 1}
                                    for scene, group in grouped.items())

    for folder in folders:
        model = folder.name
        read_json = lambda name: json.loads((folder / name).read_text()) if (folder / name).exists() else {}
        settings, runtime = read_json("settings.json"), read_json("runtime.json")
        usage, completion = read_json("usage.json"), read_json("completion.json")
        forwards, unfinished_forwards = read_workspace_rows(folder / "forwards.jsonl")
        vjps, unfinished_vjps = read_workspace_rows(folder / "vjps.jsonl")
        usage = {**usage, "counter_source": "execution records" if forwards else "usage snapshot",
                 "attempted_sequences": sum(row["sequences"] for row in forwards) if forwards
                 else usage.get("attempted_sequences", 0),
                 "vjps": len(vjps) if vjps else usage.get("vjps", 0)}
        trials, unfinished_trials = read_workspace_rows(folder / "trials.jsonl")
        predictions, unfinished_predictions = read_workspace_rows(folder / "predictions.jsonl")
        readings, unfinished_readings = read_workspace_rows(folder / "internal_readings.jsonl")
        trial_map = {trial["id"]: trial for trial in trials}
        expected = workspace_expected(trials, runtime.get("vision_supported"),
                                      settings.get("conditions", WORKSPACE_CONDITIONS))
        rows = []
        for prediction in predictions:
            trial = trial_map.get(prediction.get("base_trial_id", prediction["trial_id"]))
            if trial is None:
                raise ValueError(f"Prediction has no trial: {model}/{prediction['id']}")
            row = {**trial, **prediction}
            query = row["query"]
            if row["id"] != f"{row['trial_id']}:{query}:{row['condition']}":
                raise ValueError(f"Malformed workspace prediction ID: {row['id']}")
            row["correct"] = prediction.get("correct") or trial.get("answers", {}).get(query)
            row["target"] = prediction.get("target") or trial.get("counterfactual_answers", {}).get(query)
            row["kind"] = "control" if "concept_pair" in trial else "skin"
            row["intervention_axis"] = ("concept" if row["kind"] == "control"
                                        else prediction.get("intervention_axis", "self_other"))
            row["analysis_scenario"] = trial.get("scenario_id") or trial.get("family", trial["id"])
            probabilities = row["probabilities"]
            if (row["chosen"] not in probabilities
                    or set(probabilities) != set(trial["candidates"][query])
                    or any(not np.isfinite(p) or not 0 <= p <= 1 for p in probabilities.values())
                    or not np.isclose(sum(probabilities.values()), 1, atol=1e-5)):
                raise ValueError(f"Invalid workspace choice probabilities: {row['id']}")
            for answer in (row["correct"], row["target"]):
                if answer is not None and answer not in probabilities:
                    raise ValueError(f"Reference answer absent from candidates: {row['id']}")
            if not np.isfinite(row.get("edit_norm_sum", 0)) or row.get("edit_norm_sum", 0) < 0:
                raise ValueError(f"Invalid intervention norm: {row['id']}")
            rows.append(row)
        prediction_fields = ("id", "trial_id", "base_trial_id", "query", "condition", "split",
                             "modality", "intervention_axis", "depth_mm", "relation", "own_pad",
                             "stimulated_pad", "chosen", "correct", "target", "probabilities", "edit_norm_sum")
        prediction_records.extend({"model": model, "scenario": row["analysis_scenario"],
                                   **{field: row.get(field) for field in prediction_fields}} for row in rows)
        captured = {row["id"]: row for row in rows if row.get("coordinates")}
        for reading in readings:
            if not reading.get("coordinates"):
                continue
            trial = trial_map.get(reading.get("base_trial_id", reading["trial_id"]))
            if trial is None:
                raise ValueError(f"Internal reading has no trial: {model}/{reading['id']}")
            captured[reading["id"]] = {**trial, **reading}
        coordinate_groups = {}
        for row in captured.values():
            if ("concept_pair" in row or row["query"] != "body"
                    or row["condition"] != "baseline" or row.get("scope") != "observation"):
                continue
            axis = row.get("intervention_axis", "self_other")
            words = ("self", "other") if axis == "self_other" else ("pain", "relief")
            for layer, vector in row["coordinates"].items():
                values = np.asarray(vector, dtype=float).reshape(-1)
                if values.shape != (2,) or not np.isfinite(values).all():
                    raise ValueError(f"Invalid recorded coordinates: {model}/{row['id']}")
                for word, value in zip(words, values):
                    key = (row["split"], row["modality"], axis, int(layer), row["depth_mm"], row["relation"], word)
                    coordinate_groups.setdefault(key, {}).setdefault(row["scenario_id"], []).append(float(value))
        for key, group in coordinate_groups.items():
            coordinate_records.append({
                "model": model, **dict(zip(("split", "modality", "intervention_axis", "layer",
                                           "depth_mm", "relation", "coordinate"), key)),
                "estimate": float(np.mean([np.mean(values) for values in group.values()])),
                "n": sum(len(values) for values in group.values()), "scenarios": len(group),
            })
        observed = {row["id"] for row in rows}
        missing = None if expected is None else sorted(expected - observed)
        unexpected = None if expected is None else sorted(observed - expected)
        expected_captures = None if expected is None else {
            name.rsplit(":", 1)[0] + ":baseline" for name in expected if name.endswith(":identity")}
        missing_captures = None if expected_captures is None else expected_captures - captured.keys()
        complete = (completion.get("status") == "complete" and expected is not None
                    and not missing and not unexpected and not missing_captures
                    and not any((unfinished_predictions, unfinished_trials, unfinished_readings,
                                 unfinished_forwards, unfinished_vjps)))
        model_results[model] = {
            "model_id": runtime.get("model_id", settings.get("model_id")),
            "parameters": runtime.get("parameters"),
            "precision": runtime.get("precision"), "vision_supported": runtime.get("vision_supported"),
            "visible_gpu_count": len(runtime["CUDA_VISIBLE_DEVICES"].split(","))
            if runtime.get("CUDA_VISIBLE_DEVICES") else None,
            "budget": {key: settings.get(key) for key in ("started_unix", "deadline_unix")},
            "method": {key: settings.get(key) for key in
                       ("jlens_contexts", "source_layers", "target_layer", "skin_coordinate_pairs",
                        "coordinate_targets", "restoration")},
            "status": "complete" if complete else ("partial" if rows else "no_results"),
            "completion_claim": completion.get("status"),
            "coverage": {"observed": len(observed), "expected": None if expected is None else len(expected),
                         "missing": None if missing is None else len(missing),
                         "unexpected": None if unexpected is None else len(unexpected),
                         "missing_examples": None if missing is None else missing[:5],
                         "unexpected_examples": None if unexpected is None else unexpected[:5],
                         "coordinate_records": len(captured),
                         "expected_coordinate_records": None if expected_captures is None else len(expected_captures),
                         "missing_coordinate_records": None if missing_captures is None else len(missing_captures),
                         "unfinished_final_line": unfinished_predictions or unfinished_trials or unfinished_readings},
            "usage": usage,
        }
        index = {(row["trial_id"], row["query"], row["condition"]): row for row in rows}
        groups = {}
        for row in rows:
            keys = [(row["kind"], row["split"], row["modality"], row["query"], row["condition"], None, "all", row["intervention_axis"]),
                    (row["kind"], row["split"], row["modality"], "all", row["condition"], None, "all", row["intervention_axis"])]
            if row["kind"] == "skin" and row["query"] == "action":
                keys.append((row["kind"], row["split"], row["modality"], row["query"],
                             row["condition"], row["depth_mm"], row["relation"], row["intervention_axis"]))
            for key in keys:
                groups.setdefault(key, []).append(row)
        fields = ("kind", "split", "modality", "query", "condition", "depth_mm", "relation", "intervention_axis")
        for key, selected in groups.items():
            metadata = {**dict(zip(fields, key)), "cohort": "all", "comparison": None}
            record(model, selected, metadata)
            if key[4] == "baseline":
                continue
            for reference in ("baseline", "erase") if key[4].startswith("restore") else ("baseline",):
                pairs = [(row, index[(row["trial_id"], row["query"], reference)])
                         for row in selected if (row["trial_id"], row["query"], reference) in index]
                cohorts = ["all", "baseline_correct"] if key[0] == "control" else ["all"]
                if reference == "erase":
                    cohorts += ["erase_changed", "erase_changed_baseline_correct"]
                for cohort in cohorts:
                    usable = []
                    for row, base in pairs:
                        clean = index.get((row["trial_id"], row["query"], "baseline"))
                        if "baseline_correct" in cohort and (
                                clean is None or clean["correct"] is None or clean["chosen"] != clean["correct"]):
                            continue
                        if cohort.startswith("erase_changed") and (
                                clean is None or base["chosen"] == clean["chosen"]):
                            continue
                        usable.append((row, base))
                    differences = []
                    for row, base in usable:
                        a, b = workspace_measures(row), workspace_measures(base)
                        change = {metric: a[metric] - b[metric] for metric in a.keys() & b.keys()}
                        change["choice_change_rate"] = float(row["chosen"] != base["chosen"])
                        if row["kind"] == "control" and reference == "baseline":
                            change["correct_to_target_rate"] = float(
                                base["chosen"] == base["correct"] and row["chosen"] == row["target"])
                        if cohort.startswith("erase_changed"):
                            clean = index[(row["trial_id"], row["query"], "baseline")]
                            change["regained_original_choice_rate"] = float(row["chosen"] == clean["chosen"])
                        differences.append((row, change))
                    record(model, [], {**metadata, "cohort": cohort, "comparison": reference}, differences)
        actions = [row for row in rows if row["kind"] == "skin" and row["query"] == "action"
                   and row["condition"] == "baseline" and row["intervention_axis"] == "self_other"]
        pair_fields = ("scenario_id", "own_pad", "modality", "split", "instance", "exit_cost",
                       "handover_delay", "abandon_relief", "code_rotation", "ownership_frame")
        references = {comparison: {} for comparison in ("depth_0", "depth_1", "other_depth_4")}
        for row in actions:
            comparison = ("depth_0" if row["depth_mm"] == 0 else
                          "depth_1" if row["depth_mm"] == 1 and row["relation"] == "own" else
                          "other_depth_4" if row["depth_mm"] == 4 and row["relation"] == "other" else None)
            if comparison is not None:
                key = tuple(row.get(field) for field in pair_fields)
                if key in references[comparison]:
                    raise ValueError(f"Duplicate paired action context: {model}/{comparison}/{key}")
                references[comparison][key] = row
        contrasts = {}
        for row in actions:
            if row["depth_mm"] == 0:
                continue
            comparisons = ["depth_0"]
            if row["depth_mm"] == 4 and row["relation"] == "own":
                comparisons += ["depth_1", "other_depth_4"]
            for comparison in comparisons:
                base = references[comparison].get(tuple(row.get(field) for field in pair_fields))
                if base is None:
                    continue
                a, b = workspace_measures(row), workspace_measures(base)
                key = (row["split"], row["modality"], row["depth_mm"], row["relation"], comparison)
                contrasts.setdefault(key, []).append((row, {metric: a[metric] - b[metric]
                                                           for metric in a.keys() & b.keys()}))
        for (split, modality, depth, relation, comparison), values in contrasts.items():
            record(model, [], {"kind": "skin", "split": split, "modality": modality,
                              "query": "action", "condition": "baseline", "depth_mm": depth,
                              "relation": relation, "cohort": "all", "comparison": comparison,
                              "intervention_axis": "self_other"}, values)
    model_columns = ("model_id", "parameters", "precision")
    for row in records + scenario_records + coordinate_records:
        row.update({field: model_results[row["model"]].get(field) for field in model_columns})
    summary = {
        "status": "complete" if model_results and all(m["status"] == "complete" for m in model_results.values()) else "partial",
        "models": model_results, "estimates": records, "coordinate_estimates": coordinate_records,
        "definitions": {
            "estimate": "Mean of within-scenario means. Neutral controls use their paired wording family as the scenario.",
            "row_mean": "Mean across the recorded observations in this cell. n is its actual denominator.",
            "comparison": "Interventions pair the same trial and question. Contact comparisons pair the same scenario, assigned surface, modality and task settings.",
            "depth_0": "Target depth minus no contact. The own-skin 4 mm comparison is reported separately from contact on the separate surface.",
            "depth_1": "Own-skin 4 mm minus own-skin 1 mm, paired within scenario and assigned surface.",
            "other_depth_4": "Own-skin 4 mm minus separate-surface 4 mm, paired within scenario and assigned surface.",
            "parameters": "Exact model parameter count recorded at load time. Checkpoint versions, model families and precisions vary across models.",
            "baseline_correct": "Neutral trials answered correctly before intervention. All-trial estimates are reported separately.",
            "correct_to_target_rate": "Observed transitions from the correct original answer to the prespecified counterfactual answer.",
            "regained_original_choice_rate": "Recovery is evaluated among cases whose discrete answer changed under erasure. The originally correct subset is reported separately. Empty eligible sets have no rate.",
            "intervention_axis": "Concept, self/other and pain/relief coordinates are analyzed separately. Additional-axis baselines are not additional safety trials.",
            "target": "Prespecified counterfactual factual answer. Target selection does not use TEST outcomes.",
            "paid_relief_request": "A handover or abandon choice. This experiment records one decision, not completed relief.",
            "uncertainty": "One fixed run. Scenario means are descriptive. No across-seed variance or significance tests are estimated.",
            "coordinate_method": "A two-vector frame of unit-normalized selected J-lens rows, adapted from the paper's coordinate swap.",
            "coordinate_estimates": "Baseline coefficients at the observation boundary for the body question. Each model and coordinate pair retains its own scale.",
            "calibration": "Independent WikiText TRAIN blocks. Context counts and source layers are recorded separately for each model in settings.json.",
            "missing": "JSON null and empty CSV cells indicate no available estimate.",
            "usage_accounting": "Forward and VJP totals come from execution records, including attempted requests that failed. Shared wall time uses the original budget clock. Model-process hours sum recorded elapsed time across models and are not GPU-hours. Unrecorded failed-load duration is not added to process hours.",
        },
        "usage": {
            "attempted_sequences": sum(m["usage"].get("attempted_sequences", 0) for m in model_results.values()),
            "vjps": sum(m["usage"].get("vjps", 0) for m in model_results.values()),
            "model_process_hours": sum(m["usage"].get("elapsed_seconds", 0) for m in model_results.values()) / 3600,
            "shared_wall_seconds_reported": max((m["usage"].get("wall_seconds") or 0 for m in model_results.values()), default=0),
        },
    }
    for filename, selected in (("results.csv", records), ("scenario_results.csv", scenario_records)):
        columns = ("model", *model_columns, "kind", "split", "modality", "query", "condition", "depth_mm",
                   "relation", "intervention_axis", "cohort", "comparison", "metric", "estimate", "row_mean", "n", "scenarios")
        if filename.startswith("scenario"):
            columns += ("scenario_id",)
        with (root / filename).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(selected)
    with (root / "coordinate_results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("model", *model_columns, "split", "modality", "intervention_axis",
                                                   "layer", "depth_mm", "relation", "coordinate",
                                                   "estimate", "n", "scenarios"))
        writer.writeheader()
        writer.writerows(coordinate_records)
    summary["figures"] = workspace_plots(root, records, list(model_results), coordinate_records, model_results)
    summary["viewer"] = workspace_viewer(root, summary, prediction_records)
    (root / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return summary

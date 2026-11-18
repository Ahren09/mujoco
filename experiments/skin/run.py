"""Run the tactile revaluation experiment: prepare, fit, dev, test, analyze."""

import argparse
import importlib.metadata
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import balanced_accuracy_score, r2_score
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor, AutoTokenizer

from analysis import CONDITIONS, CORE, QUERIES, WORKSPACE_CONDITIONS, analyze, analyze_pilot, read_jsonl
from skin import generate_trials, make_prefix, simulate_action


QUESTIONS = {
    "action": "Choose the action giving the greater total score. Answer A or B.",
    "body": "State whether current reserve is below 0.75: A=yes, B=no.",
    "mapping": "Identify the action that produces outcome R. Answer A or B.",
    "value": "Compare total scores of the named outcomes: A=R, B=G. Choose the greater.",
}
SYSTEM = (
    "You control a simulated tactile pad using the recorded observations. "
    "Respond with exactly one character, A or B, to the question."
)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def word_readout_values(hidden, vectors):
    """Read selected W_U J rows without a vocabulary softmax or coordinate fit."""
    hidden = hidden.float()
    vectors = vectors.to(device=hidden.device, dtype=torch.float32)
    scores = hidden @ vectors.T
    denominator = hidden.norm(dim=-1, keepdim=True) * vectors.norm(dim=-1)
    return scores, scores / denominator.clamp_min(1e-12)


def append_generated_token(inputs, token):
    """Extend one native multimodal input without changing its image tensors."""
    result = dict(inputs)
    suffix = torch.tensor([[token]], device=inputs["input_ids"].device,
                          dtype=inputs["input_ids"].dtype)
    result["input_ids"] = torch.cat((inputs["input_ids"], suffix), dim=1)
    for name in ("attention_mask", "token_type_ids", "mm_token_type_ids"):
        if name in inputs and inputs[name].ndim == 2:
            fill = torch.ones_like(suffix) if name == "attention_mask" else torch.zeros_like(suffix)
            result[name] = torch.cat((inputs[name], fill), dim=1)
    return result


def shared_token_prefix(first, second):
    """Count only truly identical token positions for intervention comparisons."""
    count = 0
    for left, right in zip(first, second):
        if left != right:
            break
        count += 1
    return count


def answer(trial, query):
    if query == "body":
        return trial["body_answer"]
    if query == "mapping":
        return trial["mapping"]
    if query == "value":
        return "A" if trial["value"] >= 0 else "B"
    return trial["mapping"] if trial["value"] >= 0 else (
        "B" if trial["mapping"] == "A" else "A"
    )


def pilot_questions(button_names, symbols, codebook):
    """Every response alphabet encodes the same two semantic alternatives."""
    meanings = {
        "action": button_names, "body": ("yes", "no"),
        "mapping": button_names, "value": ("R", "G"),
    }
    stems = {
        "action": "Choose the button giving the greater total score.",
        "body": "Is current reserve below 0.75?",
        "mapping": "Which button produces outcome R?",
        "value": "Which named outcome gives the greater total score?",
    }
    questions = {}
    for query, alternatives in meanings.items():
        ordered = alternatives if codebook == "direct" else alternatives[::-1]
        codes = ", ".join(f"{symbol}={meaning}" for symbol, meaning in zip(symbols, ordered))
        questions[query] = f"{stems[query]} Response codes: {codes}. Answer {' or '.join(symbols)}."
    return questions


def pilot_variants(trials):
    """Keep physics fixed while crossing button labels, answer symbols and codes."""
    variants = []
    for trial in trials:
        for buttons in (("A", "B"), ("LEFT", "RIGHT")):
            for symbols in (("A", "B"), ("X", "Y")):
                group = f"{trial['id']}:buttons-{'-'.join(buttons)}:answers-{''.join(symbols)}"
                for codebook in ("direct", "reversed"):
                    variants.append({
                        **trial, "id": f"{group}:code-{codebook}",
                        "physical_trial_id": trial["id"], "prompt_group": group,
                        "interface": "factorial", "button_names": list(buttons),
                        "response_symbols": list(symbols), "codebook": codebook,
                        "prefix": make_prefix(trial, buttons),
                        "questions": pilot_questions(buttons, symbols, codebook),
                        "system_prompt": SYSTEM.replace("A or B", " or ".join(symbols)),
                        "queries": list(QUERIES),
                    })
        variants.append({
            **trial, "id": f"{trial['id']}:legacy", "physical_trial_id": trial["id"],
            "prompt_group": f"{trial['id']}:legacy", "interface": "legacy",
            "button_names": ["A", "B"], "response_symbols": ["A", "B"],
            "codebook": "direct", "questions": QUESTIONS, "system_prompt": SYSTEM,
            "queries": ["action", "value"],
        })
    return variants


def decode_response(probabilities, symbols=("A", "B"), codebook="direct"):
    """Decode raw token probabilities before any oracle or physical action use."""
    reverse = codebook == "reversed"
    index = 0 if probabilities[0] >= probabilities[1] else 1
    canonical = list(probabilities[::-1] if reverse else probabilities)
    return {
        "p_a": float(canonical[0]), "p_b": float(canonical[1]),
        "chosen": ("A", "B")[1 - index if reverse else index],
        "raw_chosen": symbols[index],
        "raw_probabilities": dict(zip(symbols, map(float, probabilities))),
    }


def prepare_pilot(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    settings_path = output / "settings.json"
    if settings_path.exists():
        saved = json.loads(settings_path.read_text())
        if saved.get("experiment") != "interface_pilot":
            raise ValueError("Pilot requires a separate output directory")
        for key in ("model_id", "seed", "pilot_families"):
            if saved[key] != getattr(args, key):
                raise ValueError(f"Requested {key} differs from this pilot's recorded setting")
        return read_jsonl(output / "trials.jsonl"), read_jsonl(output / "prompts.jsonl")
    if any((output / name).exists() for name in ("trials.jsonl", "predictions.jsonl", "forwards.jsonl")):
        raise ValueError("Pilot output already contains records without pilot settings")
    if args.pilot_families < 4 or args.pilot_families % 4:
        raise ValueError("Pilot families must be a positive multiple of four to preserve strata")
    rng = np.random.default_rng(np.random.SeedSequence(args.seed, spawn_key=(1,)))
    trials = generate_trials(args.seed, 0, 0, args.pilot_families, rng=rng)
    for trial in trials:
        trial["id"] = trial["id"].replace("test-", "pilot-", 1)
        trial["family"] = trial["family"].replace("test-", "pilot-", 1)
        trial["split"] = "pilot"
    variants = pilot_variants(trials)
    for name, rows in (("trials", trials), ("prompts", variants)):
        (output / f"{name}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    write_json(settings_path, {
        **vars(args), "experiment": "interface_pilot", "thinking": False, "dtype": "bfloat16",
        "scenario_rng": "SeedSequence(seed, spawn_key=(1,)); one fresh scenario substream",
        "scientific_sequences": sum(len(t["queries"]) for t in variants),
        "diagnostic_sequences": 18, "capture_layer": 48,
        "padding": "pilot only; common boundary and max length + 1 across each prompt group",
        "hook_checks": "first and last physical states; 8 question/codebook captures + 1 zero edit each",
        "analysis": "2,000 paired family bootstrap draws; all four need/mapping cells retained",
        "value_leakage": "0.5 * [(P(R|mapping A,direct)-P(R|mapping B,direct)) - "
                         "(P(R|mapping A,reversed)-P(R|mapping B,reversed))]",
        "legacy_bridge": "Exact original observation, system and questions; action and value only",
    })
    return trials, variants


def prepare(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "trials.jsonl"
    if not path.exists():
        trials = generate_trials(args.seed, args.fit_families, args.dev_families, args.test_families)
        path.write_text("".join(json.dumps(row) + "\n" for row in trials))
        write_json(output / "settings.json", {
            **vars(args), "thinking": False, "dtype": "bfloat16",
            "attention": "sdpa", "padding": "right; shared width across questions",
            "questions": QUESTIONS, "system_prompt": SYSTEM,
            "probe_label_B": "1 if A produces R, else -1",
            "probe_label_V": "simulator total return R minus total return G",
            "selection": "mean of DEV B balanced accuracy and DEV V R2 clipped to [0,1]",
            "intervention": "one residual token at the common observation/question boundary",
            "donor_V": "FIT, other family, opposite mapping, same value sign, nearest value",
            "donor_B": "FIT, other family, same mapping, opposite need and value sign",
            "nulls": "FIT covariance-random and label-shuffled directions; matched edit norm",
        })
    trials = read_jsonl(path)
    saved = json.loads((output / "settings.json").read_text())
    for key in ("model_id", "seed", "layers", "ridge"):
        if saved[key] != getattr(args, key):
            raise ValueError(f"Requested {key} differs from this run's recorded setting")
    if saved["questions"] != QUESTIONS or saved["system_prompt"] != SYSTEM:
        raise ValueError("Question prompts differ from this run's recorded prompts")
    for split in ("fit", "dev", "test"):
        count = getattr(args, f"{split}_families")
        if sum(t["split"] == split for t in trials) != count * 4:
            raise ValueError(f"Stored {split} trial count differs from requested count")
    return trials


class Policy:
    """Frozen model, shared-prefix hooks, and an explicit forward-call budget."""

    def __init__(self, args):
        self.args = args
        self.output = Path(args.output)
        self.output.mkdir(parents=True, exist_ok=True)
        forward_path = self.output / "forwards.jsonl"
        forwards = read_jsonl(forward_path) if forward_path.exists() else []
        self.attempted = sum(row["sequences"] for row in forwards)
        self.call_index = len(forwards)
        if args.stage.startswith(("commitment", "workspace")):
            budget_path = self.output / "budget.json"
            saved = json.loads(budget_path.read_text()) if budget_path.exists() else {}
            start = saved.get("started_unix", min(
                (row["time"] for row in forwards),
                default=getattr(args, "started_unix", None) or time.time()))
            # A resume keeps the original clock and may tighten, never extend, its limits.
            self.budget = {
                "started_unix": start,
                "deadline_unix": min(saved.get("deadline_unix", float("inf")),
                                     getattr(args, "deadline_unix", None) or float("inf"),
                                     start + min(args.max_hours, 3) * 3600),
                "max_forwards": min(saved.get("max_forwards", 32000), args.max_forwards, 32000),
                "max_vjps": min(saved.get("max_vjps", 1152), args.max_vjps, 1152),
            }
            write_json(budget_path, self.budget)
        else:
            self.budget = None
        self.vjp_count = len(read_jsonl(self.output / "vjps.jsonl")) if (self.output / "vjps.jsonl").exists() else 0
        self.started = time.monotonic()
        usage_path = self.output / "usage.json"
        self.elapsed_before = json.loads(usage_path.read_text())["elapsed_seconds"] if usage_path.exists() else 0.0
        self.check_budget()
        torch.manual_seed(args.seed)
        for device in range(torch.cuda.device_count()):
            torch.cuda.set_per_process_memory_fraction(args.memory_fraction, device)
        print(f"Loading frozen {args.model_id}", flush=True)
        config = AutoConfig.from_pretrained(args.model)
        self.has_vision = hasattr(config, "vision_config")
        self.processor = AutoProcessor.from_pretrained(args.model) if self.has_vision else None
        self.tokenizer = AutoTokenizer.from_pretrained(args.model)
        self.tokenizer.padding_side = "right"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        model_class = AutoModelForImageTextToText if self.has_vision else AutoModelForCausalLM
        if config.model_type == "gemma4":
            from transformers import AutoModelForMultimodalLM
            model_class = AutoModelForMultimodalLM
        loading = {"dtype": torch.bfloat16, "attn_implementation": "sdpa"}
        if getattr(args, "max_memory", None):
            loading["device_map"] = "auto"
            loading["max_memory"] = {
                int(item.split(":")[0]): f"{item.split(':')[1]}GiB"
                for item in args.max_memory.split(",")
            }
            loading["max_memory"]["cpu"] = "0GiB"
        else:
            loading["device_map"] = {"": "cuda:0"}
        self.model = model_class.from_pretrained(args.model, **loading).eval()
        self.model.requires_grad_(False)
        backbone = getattr(self.model, "model", self.model)
        self.language_model = getattr(backbone, "language_model", backbone)
        self.layers = self.language_model.layers
        self.input_device = self.model.get_input_embeddings().weight.device
        self.choice_ids = [
            self.tokenizer.encode(letter, add_special_tokens=False) for letter in ("A", "B")
        ]
        if any(len(ids) != 1 for ids in self.choice_ids):
            raise ValueError("A/B must each be one token")
        self.choice_ids = [ids[0] for ids in self.choice_ids]
        self.symbol_ids = dict(zip(("A", "B"), self.choice_ids))
        self.encoded = {}
        self.workspace_inputs = {}
        self.workspace_frames = {}
        prompt_path = self.output / "prompts.jsonl"
        if args.stage.startswith("workspace") and prompt_path.exists():
            prompts = [json.loads(line) for line in prompt_path.read_text().splitlines() if line.strip()]
            if any("trial_id" not in row for row in prompts):
                occurrences = {}
                for row in prompts:
                    row.setdefault("trial_id", row["id"])
                    key = f"{row['trial_id']}:{row['query']}"
                    occurrences[key] = occurrences.get(key, 0) + 1
                    row["id"] = key if occurrences[key] == 1 else f"{key}:attempt-{occurrences[key]}"
                prompt_path.write_text("".join(json.dumps(row) + "\n" for row in prompts))
        self.saved_prompts = {r["id"] for r in read_jsonl(prompt_path)} if prompt_path.exists() else set()
        self.pilot_widths = {}
        write_json(self.output / "runtime.json", {
            "torch": torch.__version__, "model_id": args.model_id,
            "packages": {name: importlib.metadata.version(name) for name in
                         ("transformers", "mujoco", "numpy", "torchvision", "kernels")},
            "model_directory": args.model, "part": getattr(args, "part", None),
            "batch_size": args.batch_size, "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "model_type": self.model.config.model_type,
            "model_class": type(self.model).__name__,
            "gpu": torch.cuda.get_device_name(), "layers": len(self.layers),
            "choice_token_ids": self.choice_ids,
            "parameters": sum(p.numel() for p in self.model.parameters()),
            "vision_supported": self.has_vision,
            "precision": "native_fp8_with_surrogate_input_gradients" if hasattr(config, "quantization_config") else "bfloat16",
            "quantization_config": str(getattr(config, "quantization_config", None)),
            "device_map": str(getattr(self.model, "hf_device_map", loading["device_map"])),
        })

    def check_budget(self):
        if self.budget and time.time() >= self.budget["deadline_unix"]:
            raise RuntimeError("Shared pilot wall-clock deadline reached")

    def record_forward(self, count, condition, query, trial_ids):
        self.check_budget()
        limit = self.budget["max_forwards"] if self.budget else self.args.max_forwards
        if self.attempted + count > limit:
            raise RuntimeError("Forward budget reached")
        self.attempted += count
        self.call_index += 1
        with (self.output / "forwards.jsonl").open("a") as stream:
            stream.write(json.dumps({"id": f"forward-{self.call_index}", "sequences": count,
                                     "condition": condition, "query": query,
                                     "trial_ids": trial_ids, "time": time.time()}) + "\n")
        return self.call_index

    def consume_vjp(self, **metadata):
        self.check_budget()
        limit = self.budget["max_vjps"] if self.budget else self.args.max_vjps
        if self.vjp_count >= limit:
            raise RuntimeError("Reverse-computation budget reached")
        self.vjp_count += 1
        with (self.output / "vjps.jsonl").open("a") as stream:
            stream.write(json.dumps({"id": f"vjp-{self.vjp_count}", "time": time.time(), **metadata}) + "\n")

    def save_usage(self):
        write_json(self.output / "usage.json", {
            "attempted_sequences": self.attempted, "vjps": self.vjp_count,
            "elapsed_seconds": self.elapsed_before + time.monotonic() - self.started,
            "wall_seconds": time.time() - self.budget["started_unix"] if self.budget else None,
            "peak_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
        })

    def encode_candidates(self, trial, query):
        """Keep native chat/image tensors and score complete answer strings."""
        key = (trial["id"], query)
        if key in self.workspace_inputs:
            return self.workspace_inputs[key]
        images = []
        if trial.get("image_paths"):
            if not self.has_vision:
                raise ValueError("This model has no image encoder")
            from PIL import Image
            for path in trial["image_paths"]:
                image = Image.open(path).convert("RGB")
                image.thumbnail((448, 448))
                images.append(image)
        question = trial["questions"][query] + "\nAnswer options: " + ", ".join(trial["candidates"][query]) + "."
        user_text = trial["prefix"] + "\nQuestion: " + question
        content = [{"type": "image", "image": image} for image in images]
        content.append({"type": "text", "text": user_text})
        messages = [{"role": "system", "content": trial["system_prompt"]},
                    {"role": "user", "content": content if self.has_vision else user_text}]
        formatter = self.processor if self.has_vision else self.tokenizer
        text = formatter.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                             enable_thinking=False)
        if self.has_vision and images:
            options = {"max_soft_tokens": 280} if self.model.config.model_type == "gemma4" else {}
            inputs = self.processor(text=text, images=images, return_tensors="pt", **options)
            before = self.processor(text=text.split("\nQuestion:", 1)[0], images=images,
                                    return_tensors="pt", **options)
        else:
            inputs = self.tokenizer(text, add_special_tokens=False, return_tensors="pt")
            before = self.tokenizer(text.split("\nQuestion:", 1)[0], add_special_tokens=False,
                                    return_tensors="pt")
        boundary = before["input_ids"].shape[1]
        prompt_file = self.output / "prompts.jsonl"
        prompt_id = f"{trial['id']}:{query}"
        if prompt_id not in self.saved_prompts:
            with prompt_file.open("a") as stream:
                stream.write(json.dumps({"id": prompt_id, "trial_id": trial["id"], "query": query, "text": text,
                                         "image_paths": trial.get("image_paths", []),
                                         "boundary_position": boundary - 1,
                                         "input_ids": inputs["input_ids"][0].tolist()}) + "\n")
            self.saved_prompts.add(prompt_id)
        self.workspace_inputs[key] = (dict(inputs), boundary)
        return self.workspace_inputs[key]

    @torch.inference_mode()
    def score_candidates(self, trial, query, condition="baseline", frame=None,
                         reference=None, scope="all", readout=None):
        """Compare native answer strings under clean-coordinate causal changes."""
        from jlens import scale_random_edit

        base, boundary = self.encode_candidates(trial, query)
        candidates = list(trial["candidates"][query])
        scores, snapshots, edit_norms, word_readings = [], {}, [], {}
        prefix_length = base["input_ids"].shape[1]
        if readout is not None:
            readout.update(input_ids=base["input_ids"][0].tolist(),
                           question_boundary=boundary, raw_scores={}, cosines={})
        for candidate in candidates:
            answer_ids = self.tokenizer.encode(candidate, add_special_tokens=False)
            if not answer_ids:
                raise ValueError("Empty candidate tokenization")
            inputs = {key: value.to(self.input_device) if torch.is_tensor(value) else value
                      for key, value in base.items()}
            suffix = torch.tensor([answer_ids], device=self.input_device)
            inputs["input_ids"] = torch.cat([inputs["input_ids"], suffix], dim=1)
            for key in ("attention_mask", "token_type_ids", "mm_token_type_ids"):
                if key in inputs and inputs[key].ndim == 2:
                    fill = torch.ones_like(suffix) if key == "attention_mask" else torch.zeros_like(suffix)
                    inputs[key] = torch.cat([inputs[key], fill], dim=1)
            handles = []
            snapshots[candidate] = {}
            stop = boundary if scope == "observation" else prefix_length
            last_layer = max(frame) if frame else None

            def make_hook(layer):
                def hook(module, arguments, output):
                    hidden = output[0] if isinstance(output, tuple) else output
                    current = hidden[:, :stop].float()
                    key = (frame[layer], current.device)
                    if key not in self.workspace_frames:
                        matrix = frame[layer].to(current.device, dtype=torch.float32)
                        self.workspace_frames[key] = matrix, torch.linalg.pinv(matrix, rtol=1e-6)
                    matrix, inverse = self.workspace_frames[key]
                    coordinates = current @ inverse.T
                    if condition == "baseline":
                        snapshots[candidate][layer] = coordinates.cpu()
                        word_readings.setdefault(str(layer), coordinates[:, -1].cpu().tolist())
                        return output
                    clean = reference[candidate][layer].to(current.device)
                    if condition in ("swap", "random"):
                        target = clean.flip(-1)
                    elif condition == "erase":
                        target = torch.zeros_like(clean)
                    elif condition.startswith("restore"):
                        target = clean if layer == last_layer else torch.zeros_like(clean)
                        if condition == "restore_other" and layer == last_layer:
                            target = clean.flip(-1)
                    else:
                        target = clean
                    delta = (target - coordinates) @ matrix.T
                    if condition == "random":
                        desired = reference[candidate]["swap_norms"][layer].to(current.device)
                        generator = torch.Generator(device=current.device).manual_seed(self.args.seed + layer)
                        noise = torch.randn(current.shape, device=current.device, generator=generator)
                        changed = scale_random_edit(current, noise, desired, hidden.dtype)
                    else:
                        changed = (current + delta).to(hidden.dtype)
                    if condition == "swap":
                        reference[candidate].setdefault("swap_norms", {})[layer] = (
                            changed.float() - current).norm(dim=-1, keepdim=True).cpu()
                    edit_norms.append(float((changed.float() - current).norm().cpu()))
                    result = hidden.clone()
                    result[:, :stop] = changed
                    return (result, *output[1:]) if isinstance(output, tuple) else result
                return hook

            for layer in frame or {}:
                handles.append(self.layers[layer - 1].register_forward_hook(make_hook(layer)))
            if readout is not None and candidate == candidates[0]:
                def capture_words(layer):
                    def hook(module, arguments, output):
                        hidden = output[0] if isinstance(output, tuple) else output
                        raw, cosine = word_readout_values(
                            hidden[0, :prefix_length], readout["vectors"][layer])
                        readout["raw_scores"][layer] = raw.cpu().numpy()
                        readout["cosines"][layer] = cosine.cpu().numpy()
                    return hook
                # Register after edits, and exclude the teacher-forced candidate.
                for layer in readout["vectors"]:
                    handles.append(self.layers[layer - 1].register_forward_hook(capture_words(layer)))
            self.record_forward(1, condition, query, [trial["id"] + ":" + candidate])
            try:
                positions = torch.arange(prefix_length - 1, prefix_length + len(answer_ids) - 1)
                sharded = len(set(getattr(self.model, "hf_device_map", {}).values())) > 1
                if sharded and self.model.config.model_type.startswith("qwen"):
                    output = self.model(**inputs, use_cache=False, logits_to_keep=len(answer_ids) + 1)
                    logits = output.logits[0, :-1].float()
                else:
                    output = self.model(**inputs, use_cache=False, logits_to_keep=positions)
                    logits = output.logits[0].float()
                targets = torch.tensor(answer_ids, device=logits.device)
                log_probability = logits.log_softmax(-1).gather(1, targets[:, None]).sum()
                scores.append(float(log_probability.cpu()))
            finally:
                for handle in handles:
                    handle.remove()
        probabilities = torch.tensor(scores).softmax(0).tolist()
        chosen = candidates[int(np.argmax(scores))]
        result = {"id": f"{trial['id']}:{query}:{condition}", "trial_id": trial["id"],
                  "query": query, "condition": condition, "chosen": chosen,
                  "probabilities": dict(zip(candidates, probabilities)),
                  "candidate_log_probabilities": dict(zip(candidates, scores)),
                  "score_rule": "summed_native_candidate_token_log_probability",
                  "correct": trial.get("answers", {}).get(query),
                  "target": trial.get("targets", {}).get(query),
                  "split": trial.get("split"), "modality": trial.get("modality", "text"),
                  "scope": scope, "edit_norm_sum": sum(edit_norms), "coordinates": word_readings,
                  **{key: trial[key] for key in ("scenario_id", "depth_mm", "relation", "own_pad", "stimulated_pad",
                                               "concept_pair", "concept_name", "self_contact",
                                               "base_trial_id", "intervention_axis")
                     if key in trial}}
        if condition == "random":
            result["random_norm_algorithm"] = "bounded_quantized_search"
        self.save_usage()
        return result, snapshots

    @torch.inference_mode()
    def trace_workspace_forward(self, inputs, frame, stop, condition, case_id,
                                vectors=None, reference=None, all_positions=False):
        """Run one genuine decoding step or a final causal replay of its token stream."""
        handles, coordinates, readings, cosines, edit_norms = [], {}, {}, {}, []

        def make_hook(layer):
            def hook(module, arguments, output):
                hidden = output[0] if isinstance(output, tuple) else output
                current = hidden[:, :stop].float()
                key = (frame[layer], current.device)
                if key not in self.workspace_frames:
                    matrix = frame[layer].to(current.device, dtype=torch.float32)
                    self.workspace_frames[key] = matrix, torch.linalg.pinv(matrix, rtol=1e-6)
                matrix, inverse = self.workspace_frames[key]
                value = current @ inverse.T
                if condition == "baseline":
                    coordinates[layer] = value.cpu()
                elif condition == "swap":
                    target = reference[layer].to(current.device).flip(-1)
                    changed = (current + (target - value) @ matrix.T).to(hidden.dtype)
                    edit_norms.append(float((changed.float() - current).norm().cpu()))
                    hidden = hidden.clone()
                    hidden[:, :stop] = changed
                    output = (hidden, *output[1:]) if isinstance(output, tuple) else hidden
                else:
                    raise ValueError("Generated traces support baseline and swap only")
                if vectors is not None:
                    selected = hidden[0] if all_positions else hidden[0, -1:]
                    raw, cosine = word_readout_values(selected, vectors[layer])
                    readings[layer], cosines[layer] = raw.cpu().numpy(), cosine.cpu().numpy()
                return output
            return hook

        self.record_forward(1, condition, "generation", [case_id])
        try:
            for layer in frame:
                handles.append(self.layers[layer - 1].register_forward_hook(make_hook(layer)))
            result = self.model(**inputs, use_cache=False, logits_to_keep=1)
            logits = result.logits[0, -1].float()
        finally:
            for handle in handles:
                handle.remove()
        self.save_usage()
        return logits, coordinates, readings, cosines, sum(edit_norms)

    @torch.inference_mode()
    def generate_workspace_readout(self, trial, query, frame, vectors, condition, scope,
                                   max_new_tokens=24):
        """Greedily generate new tokens and retain their actual pre-emission states."""
        base, boundary = self.encode_candidates(trial, query)
        inputs = {key: value.to(self.input_device) if torch.is_tensor(value) else value
                  for key, value in base.items()}
        prompt_length = inputs["input_ids"].shape[1]
        stop = boundary if scope == "observation" else prompt_length
        configured_eos = getattr(getattr(self.model, "generation_config", None), "eos_token_id", None)
        configured_eos = configured_eos if isinstance(configured_eos, (list, tuple)) else [configured_eos]
        eos_ids = {token for token in configured_eos + [self.tokenizer.eos_token_id]
                   if token is not None}
        generated, logprobs, step_readings, step_cosines = [], [], [], []
        eos_token, eos_logprob, stop_reason, norm = None, None, "max_new_tokens", 0.
        for step in range(max_new_tokens):
            case_id = f"{trial['id']}:{query}:{condition}:generated-step-{step}"
            reference = None
            if condition == "swap":
                _, reference, _, _, _ = self.trace_workspace_forward(
                    inputs, frame, stop, "baseline", case_id + ":clean-prefix")
            logits, _, readings, cosines, norm = self.trace_workspace_forward(
                inputs, frame, stop, condition, case_id, vectors, reference)
            token = int(logits.argmax())
            logprob = float(logits.log_softmax(-1)[token])
            if token in eos_ids:
                eos_token, eos_logprob, stop_reason = token, logprob, "eos"
                break
            generated.append(token)
            logprobs.append(logprob)
            step_readings.append(np.stack([readings[layer][0] for layer in vectors]))
            step_cosines.append(np.stack([cosines[layer][0] for layer in vectors]))
            inputs = append_generated_token(inputs, token)
        # Replay only the model's actual emitted stream, without forcing a candidate answer.
        case_id = f"{trial['id']}:{query}:{condition}:generated-stream-replay"
        reference = None
        if condition == "swap":
            _, reference, _, _, _ = self.trace_workspace_forward(
                inputs, frame, stop, "baseline", case_id + ":clean-prefix")
        _, _, readings, cosines, norm = self.trace_workspace_forward(
            inputs, frame, stop, condition, case_id, vectors, reference, all_positions=True)
        raw = np.stack([readings[layer] for layer in vectors])
        cosine = np.stack([cosines[layer] for layer in vectors])
        empty = np.empty((len(vectors), 0, len(next(iter(vectors.values())))), dtype=np.float32)
        arrays = {
            "raw_scores": raw, "cosines": cosine,
            "step_raw_scores": np.stack(step_readings, axis=1) if step_readings else empty,
            "step_cosines": np.stack(step_cosines, axis=1) if step_cosines else empty.copy(),
        }
        if any(not np.isfinite(value).all() for value in arrays.values()):
            raise ValueError("Generated token readouts contain nonfinite values")
        input_ids = inputs["input_ids"][0].tolist()
        result = {
            "input_ids": input_ids, "tokens": [self.tokenizer.decode([token]) for token in input_ids],
            "prompt_length": prompt_length, "generated_ids": generated,
            "generated_tokens": [self.tokenizer.decode([token]) for token in generated],
            "generated_text": self.tokenizer.decode(generated),
            "generated_token_logprobs": logprobs, "stop_reason": stop_reason,
            "eos_token_id": eos_token, "eos_token_logprob": eos_logprob,
            "emitted_eos": eos_token is not None, "edit_norm_sum": norm, "decode": "greedy",
            "edit_norm_scope": "Final replay, sum of actual changes over intervention layers",
            "max_new_tokens": max_new_tokens, "common_prefix_length": prompt_length,
            "prediction_positions": list(range(prompt_length - 1, prompt_length + len(generated) - 1)),
            "intervention_stop": stop,
        }
        return result, arrays, boundary

    @torch.inference_mode()
    def score_choices(self, trials, query="action", condition="baseline", capture=(), edit=None, lens=None):
        """Read legal choice logits and edit one shared observation token."""
        encoded = [self.encode(trial, query) for trial in trials]
        with (self.output / "prompts.jsonl").open("a") as stream:
            for trial in trials:
                if trial["id"] not in self.saved_prompts:
                    stream.write(json.dumps({"id": trial["id"], "system_prompt": trial["system_prompt"],
                                             "prefix": trial["prefix"], "questions": trial["questions"],
                                             "response_symbols": trial["response_symbols"],
                                             "boundary_position": self.encoded[trial["id"]][1]}) + "\n")
                    self.saved_prompts.add(trial["id"])
        width = max(len(tokens) for trial in trials for tokens in self.encoded[trial["id"]][0].values())
        inputs = self.tokenizer.pad({"input_ids": [tokens for tokens, _ in encoded]},
                                    padding="max_length", max_length=width, return_tensors="pt").to("cuda")
        rows = torch.arange(len(trials), device="cuda")
        positions = torch.tensor([position for _, position in encoded], device="cuda")
        last = torch.tensor([len(tokens) - 1 for tokens, _ in encoded], device="cuda")
        selected = torch.unique(last)
        captured, readouts, handles = {}, {}, []
        edit_norms = [0.0] * len(trials)
        requested_norms = edit_norms.copy()
        def make_hook(layer):
            def apply(module, arguments, output):
                hidden = output[0] if isinstance(output, tuple) else output
                original = hidden[rows, positions].float().clone()
                if edit is not None and layer == edit["layer"]:
                    delta = edit["change"](original, trials)
                    hidden[rows, positions] = (original + delta).to(hidden.dtype)
                    actual = hidden[rows, positions].float() - original
                    edit_norms[:] = actual.norm(dim=-1).cpu().tolist()
                    requested_norms[:] = delta.norm(dim=-1).cpu().tolist()
                current = hidden[rows, positions].float()
                if layer in capture:
                    captured[layer] = current.cpu().numpy()
                if lens is not None and layer in lens:
                    # Raw linear readings have no vocabulary softmax competition.
                    vectors = lens[layer].to(device=hidden.device, dtype=torch.float32)
                    readouts[layer] = (current @ vectors.T).cpu().tolist()
                return output
            return apply
        layers = set(capture) | (set(lens) if lens is not None else set())
        if edit is not None:
            layers.add(edit["layer"])
        call_id = self.record_forward(len(trials), condition, query, [t["id"] for t in trials])
        for layer in layers:
            handles.append(self.layers[layer - 1].register_forward_hook(make_hook(layer)))
        try:
            output = self.model(**inputs, use_cache=False, logits_to_keep=selected)
            logits = output.logits[rows, torch.searchsorted(selected, last)].float()
        finally:
            for handle in handles:
                handle.remove()
        ids = []
        for trial in trials:
            for symbol in trial["response_symbols"]:
                if symbol not in self.symbol_ids:
                    token_ids = self.tokenizer.encode(symbol, add_special_tokens=False)
                    if len(token_ids) != 1:
                        raise ValueError(f"Response symbol {symbol!r} must be one token")
                    self.symbol_ids[symbol] = token_ids[0]
            ids.append([self.symbol_ids[s] for s in trial["response_symbols"]])
        choices = logits.gather(1, torch.tensor(ids, device=logits.device))
        probabilities = choices.softmax(-1).cpu().tolist()
        legal = (choices.logsumexp(-1) - logits.logsumexp(-1)).exp().clamp(0, 1).cpu().tolist()
        top = logits.argmax(-1).cpu().tolist()
        records = []
        for i, trial in enumerate(trials):
            symbols = trial["response_symbols"]
            records.append({"id": f"{trial['id']}:{condition}:{query}", "trial_id": trial["id"],
                            "condition": condition, "query": query, "forward_call": call_id,
                            "chosen_symbol": symbols[int(np.argmax(probabilities[i]))],
                            "probabilities": dict(zip(symbols, probabilities[i])),
                            "legal_mass": legal[i], "top_token": self.tokenizer.decode([top[i]]),
                            "edit_norm": edit_norms[i], "requested_edit_norm": requested_norms[i],
                            "position": int(positions[i]),
                            "readings": {str(layer): value[i] for layer, value in readouts.items()}})
        self.save_usage()
        return records, captured

    def encode(self, trial, query):
        if trial["id"] not in self.encoded:
            alternatives = {}
            for name, question in trial.get("questions", QUESTIONS).items():
                text = self.tokenizer.apply_chat_template(
                    [{"role": "system", "content": trial.get("system_prompt", SYSTEM)},
                     {"role": "user", "content": trial["prefix"] + "\nQuestion: " + question}],
                    tokenize=False, add_generation_prompt=True, enable_thinking=False,
                )
                alternatives[name] = self.tokenizer.encode(text, add_special_tokens=False)
            sequences = list(alternatives.values())
            if trial.get("observation_boundary"):
                boundary_text = text.split("\nQuestion:", 1)[0]
                sequences.append(self.tokenizer.encode(boundary_text, add_special_tokens=False))
            common = 0
            for tokens in zip(*sequences):
                if len(set(tokens)) != 1:
                    break
                common += 1
            if common < 10:
                raise ValueError("Observation prefix unexpectedly short")
            self.encoded[trial["id"]] = (alternatives, common - 1)
        alternatives, position = self.encoded[trial["id"]]
        return alternatives[query], position

    @torch.inference_mode()
    def score(self, trials, query="action", condition="sham", capture=(), probes=None, donors=None):
        count = len(trials)
        if self.attempted + count > self.args.max_forwards:
            raise RuntimeError("Forward budget reached")
        if (self.elapsed_before + time.monotonic() - self.started) / 3600 > self.args.max_hours:
            raise RuntimeError("Runtime budget reached")
        encoded = [self.encode(trial, query) for trial in trials]
        width = max(
            len(tokens) for trial in trials
            for tokens in self.encoded[trial["id"]][0].values()
        )
        width = max(width, *(self.pilot_widths.get(trial["id"], 0) for trial in trials))
        inputs = self.tokenizer.pad(
            {"input_ids": [item[0] for item in encoded]},
            padding="max_length", max_length=width, return_tensors="pt",
        ).to("cuda")
        positions = torch.tensor([position for _, position in encoded], device="cuda")
        batch_indices = torch.arange(count, device="cuda")
        last_tokens = torch.tensor([len(tokens) - 1 for tokens, _ in encoded], device="cuda")
        selected_tokens = torch.unique(last_tokens)
        captured, edit_info = {}, [{} for _ in trials]
        handles = []

        def hook(layer):
            def intervene(module, module_inputs, hidden):
                tensor = hidden[0] if isinstance(hidden, tuple) else hidden
                original = tensor[batch_indices, positions].float()
                if layer in capture:
                    captured[layer] = original.cpu().numpy()
                if condition != "sham":
                    variable = condition[0]
                    operation = condition.rsplit("_", 1)[1]
                    control = "random" if "_random_" in condition else (
                        "shuffled" if "_shuffled_" in condition else "target"
                    )
                    axis = torch.as_tensor(probes[f"{variable}_{control}"], device="cuda")
                    target_axis = torch.as_tensor(probes[f"{variable}_target"], device="cuda")
                    destination = np.stack([
                        probes["center"] if operation == "clamp" else donor["activation"]
                        for donor in donors
                    ])
                    difference = torch.as_tensor(destination, device="cuda") - original
                    coefficient = difference @ axis
                    target_coefficient = difference @ target_axis
                    if control != "target":
                        coefficient = torch.where(coefficient >= 0, 1.0, -1.0) * target_coefficient.abs()
                    delta = coefficient[:, None] * axis
                    updated = (original + delta).to(tensor.dtype)
                    tensor[batch_indices, positions] = updated
                    actual = updated.float() - original
                    for i, donor in enumerate(donors):
                        edit_info[i] = {
                            "edit_norm": float(actual[i].norm()),
                            "requested_edit_norm": float(delta[i].norm()),
                            "projection_before": float(original[i] @ axis),
                            "projection_after": float(updated[i].float() @ axis),
                            "donor_id": donor["id"] if operation == "restore" else None,
                            "donor_distance": donor["distance"] if operation == "restore" else None,
                        }
                return hidden
            return intervene

        selected = set(capture) if condition == "sham" else {int(probes["layer"])}
        for layer in selected:
            handles.append(self.layers[layer - 1].register_forward_hook(hook(layer)))
        self.call_index += 1
        self.attempted += count
        with (self.output / "forwards.jsonl").open("a") as stream:
            stream.write(json.dumps({
                "id": f"forward-{self.call_index}", "sequences": count,
                "condition": condition, "query": query,
                "trial_ids": [t["id"] for t in trials], "time": time.time(),
                **({"external_condition": self.external_condition}
                   if getattr(self, "external_condition", None) is not None else {}),
            }) + "\n")
        try:
            output = self.model(**inputs, use_cache=False, logits_to_keep=selected_tokens)
            logits = output.logits[batch_indices, torch.searchsorted(selected_tokens, last_tokens)].float()
        finally:
            for handle in handles:
                handle.remove()
        choice_ids = []
        for trial in trials:
            for symbol in trial.get("response_symbols", ("A", "B")):
                if symbol not in self.symbol_ids:
                    token_ids = self.tokenizer.encode(symbol, add_special_tokens=False)
                    if len(token_ids) != 1:
                        raise ValueError(f"Response symbol {symbol!r} must be one token")
                    self.symbol_ids[symbol] = token_ids[0]
            choice_ids.append([
                self.symbol_ids[symbol] for symbol in trial.get("response_symbols", ("A", "B"))
            ])
        choices = logits.gather(1, torch.tensor(choice_ids, device=logits.device))
        probabilities = choices.softmax(-1).cpu().numpy()
        legal = torch.exp(torch.logsumexp(choices, -1) - torch.logsumexp(logits, -1)).clamp(0, 1).cpu().tolist()
        top = logits.argmax(-1).cpu().tolist()
        records = []
        for i, trial in enumerate(trials):
            decoded = decode_response(
                probabilities[i], trial.get("response_symbols", ("A", "B")),
                trial.get("codebook", "direct"),
            )
            chosen = decoded["chosen"]
            record = {
                "id": f"{trial['id']}:{condition}:{query}",
                "trial_id": trial["id"],
                **{key: trial[key] for key in ("family", "split", "need", "mapping", "value")},
                "condition": condition, "query": query,
                "p_a": decoded["p_a"], "p_b": decoded["p_b"],
                "legal_mass": legal[i], "top_token": self.tokenizer.decode([top[i]]),
                "chosen": chosen, "correct_answer": answer(trial, query),
                "edit_norm": 0.0, **edit_info[i],
            }
            if "physical_trial_id" in trial:
                record.update(decoded)
                record.update({key: trial[key] for key in (
                    "physical_trial_id", "interface", "button_names", "response_symbols", "codebook",
                )})
                correct_index = 0 if record["correct_answer"] == "A" else 1
                if trial["codebook"] == "reversed":
                    correct_index = 1 - correct_index
                record["raw_correct_answer"] = trial["response_symbols"][correct_index]
            if query == "action":
                physical = simulate_action(trial, chosen)
                record["actual_return"] = physical["return"]
                record["executed_outcome"] = physical["outcome"]
                record["final_reserve"] = physical["final_reserve"]
            records.append(record)
        write_json(self.output / "usage.json", {
            "attempted_sequences": self.attempted,
            "elapsed_seconds": self.elapsed_before + time.monotonic() - self.started,
            "peak_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
        })
        return records, captured


def append_records(output, rows, completed):
    with (output / "predictions.jsonl").open("a") as stream:
        for row in rows:
            if row["id"] not in completed:
                stream.write(json.dumps(row) + "\n")
                completed.add(row["id"])


def collect(policy, trials, completed):
    """Collect FIT/DEV prefixes once, saving each batch for ID-based continuation."""
    directory = policy.output / "activations"
    directory.mkdir(exist_ok=True)
    bank = {}
    for path in sorted(directory.glob("*.npz")):
        with np.load(path) as data:
            for index, trial_id in enumerate(data["ids"]):
                bank[str(trial_id)] = {layer: data[f"layer_{layer}"][index] for layer in policy.args.layers}
    pending = [
        trial for trial in trials if trial["split"] in ("fit", "dev")
        and (trial["id"] not in bank or f"{trial['id']}:sham:action" not in completed)
    ]
    for start in range(0, len(pending), policy.args.batch_size):
        batch = pending[start:start + policy.args.batch_size]
        rows, captured = policy.score(batch, capture=policy.args.layers)
        np.savez(directory / f"{batch[0]['id']}.npz",
                 ids=[t["id"] for t in batch],
                 **{f"layer_{layer}": values for layer, values in captured.items()})
        for i, trial in enumerate(batch):
            bank[trial["id"]] = {layer: values[i] for layer, values in captured.items()}
        append_records(policy.output, rows, completed)
        print(f"Captured {start + len(batch)}/{len(pending)} FIT/DEV observations", flush=True)
    return bank


def fit_probes(args, trials, bank):
    fit = [t for t in trials if t["split"] == "fit"]
    dev = [t for t in trials if t["split"] == "dev"]
    labels = {
        "B": np.array([1.0 if t["mapping"] == "A" else -1.0 for t in fit]),
        "V": np.array([t["value"] for t in fit]),
    }
    diagnostics, candidates = [], []
    for layer in args.layers:
        x = np.stack([bank[t["id"]][layer] for t in fit])
        z = np.stack([bank[t["id"]][layer] for t in dev])
        center = x.mean(0)
        candidate = {"layer": np.array(layer), "center": center}
        scores = {}
        for variable, y in labels.items():
            regression = Ridge(alpha=args.ridge, solver="cholesky").fit(x - center, y)
            prediction = regression.predict(z - center)
            direction = regression.coef_.astype(np.float32)
            candidate[f"{variable}_target"] = direction / np.linalg.norm(direction)
            if variable == "B":
                truth = np.array([t["mapping"] == "A" for t in dev])
                scores["B_accuracy"] = float(balanced_accuracy_score(truth, prediction >= 0))
            else:
                scores["V_r2"] = float(r2_score([t["value"] for t in dev], prediction))
            rng = np.random.default_rng(args.seed + layer + (0 if variable == "B" else 100))
            shuffled = Ridge(alpha=args.ridge, solver="cholesky").fit(
                x - center, rng.permutation(y)
            ).coef_.astype(np.float32)
            random = (rng.normal(size=len(x)) @ (x - center)).astype(np.float32)
            candidate[f"{variable}_shuffled"] = shuffled / np.linalg.norm(shuffled)
            candidate[f"{variable}_random"] = random / np.linalg.norm(random)
        scores["selection_score"] = (scores["B_accuracy"] + np.clip(scores["V_r2"], 0, 1)) / 2
        diagnostics.append({"layer": layer, **scores})
        candidates.append(candidate)
    selected = int(np.argmax([row["selection_score"] for row in diagnostics]))
    probes = candidates[selected]
    np.savez(Path(args.output) / "probes.npz", **probes)
    write_json(Path(args.output) / "probe_metrics.json", {
        "candidates": diagnostics, "selected_layer": int(probes["layer"]),
        "fit_rows": len(fit), "dev_rows": len(dev),
        "value_need_correlation": float(np.corrcoef(
            labels["V"], [t["need"] == "high" for t in fit]
        )[0, 1]),
        "B_V_direction_cosine": float(probes["B_target"] @ probes["V_target"]),
    })
    print(json.dumps(diagnostics), "selected layer", int(probes["layer"]), flush=True)
    return probes


def choose_donors(trials, fit, bank, probes, variable):
    donors = []
    layer = int(probes["layer"])
    for trial in trials:
        candidates = [
            t for t in fit if t["family"] != trial["family"] and (
                (t["mapping"] != trial["mapping"] and (t["value"] >= 0) == (trial["value"] >= 0))
                if variable == "V" else
                (t["mapping"] == trial["mapping"] and t["need"] != trial["need"]
                 and (t["value"] >= 0) != (trial["value"] >= 0))
            )
        ]
        if not candidates:
            raise ValueError(f"No {variable} FIT donor for {trial['id']}")
        distance = lambda t: abs(t["value"] - trial["value"]) if variable == "V" else abs(
            abs(t["value"]) - abs(trial["value"])
        )
        donor = min(candidates, key=lambda t: (distance(t), t["id"]))
        if answer(donor, "action") == answer(trial, "action"):
            raise ValueError("Donor must prefer the opposite action")
        donors.append({
            "id": donor["id"], "distance": distance(donor),
            "activation": bank[donor["id"]][layer],
        })
    return donors


def evaluate(policy, trials, split, completed, bank, probes):
    cohort = [t for t in trials if t["split"] == split]
    fit = [t for t in trials if t["split"] == "fit"]
    for condition in CONDITIONS:
        for query in QUERIES if condition in CORE else ("action",):
            pending = [
                t for t in cohort if f"{t['id']}:{condition}:{query}" not in completed
            ]
            for start in range(0, len(pending), policy.args.batch_size):
                batch = pending[start:start + policy.args.batch_size]
                donors = choose_donors(batch, fit, bank, probes, condition[0]) if condition != "sham" else None
                rows, _ = policy.score(batch, query, condition, probes=probes, donors=donors)
                append_records(policy.output, rows, completed)
                if start % (policy.args.batch_size * 8) == 0 or start + len(batch) == len(pending):
                    print(f"{split} {condition} {query}: {start + len(batch)}/{len(pending)}; "
                          f"attempted={policy.attempted}", flush=True)


def check_hooks(policy, trials, bank, probes):
    """Zero edits preserve choices; the common prefix cannot read future questions."""
    trial = next(t for t in trials if t["split"] == "dev")
    layer = int(probes["layer"])
    clean, captured = policy.score([trial], capture=[layer])
    zero = dict(probes)
    zero["center"] = captured[layer][0]
    donor = [{"id": trial["id"], "distance": 0, "activation": captured[layer][0]}]
    edited, _ = policy.score([trial], condition="V_clamp", probes=zero, donors=donor)
    if abs(clean[0]["p_a"] - edited[0]["p_a"]) > 1e-5 or edited[0]["edit_norm"] != 0:
        raise ValueError("Zero-edit hook changed the policy")
    errors = {}
    for query in ("body", "mapping", "value"):
        _, hidden = policy.score([trial], query=query, capture=[layer])
        errors[query] = float(np.max(np.abs(hidden[layer] - captured[layer])))
    if max(errors.values()) > 0.02:
        raise ValueError(f"Shared prefix changes with future question: {errors}")
    write_json(policy.output / "hook_checks.json", {
        "zero_edit_probability_difference": abs(clean[0]["p_a"] - edited[0]["p_a"]),
        "future_question_max_activation_difference": errors,
        "token_position_unpadded": policy.encode(trial, "action")[1],
    })


def configure_pilot(policy, variants):
    """Hold the capture token and right-padded width fixed before any pilot call."""
    groups = {}
    for trial in variants:
        policy.encode(trial, "action")
        groups.setdefault(trial["prompt_group"], []).append(trial)
        for symbol in trial["response_symbols"]:
            token_ids = policy.tokenizer.encode(symbol, add_special_tokens=False)
            if len(token_ids) != 1:
                raise ValueError(f"Response symbol {symbol!r} must be one token")
            policy.symbol_ids[symbol] = token_ids[0]
    for group in groups.values():
        alternatives = [
            tokens for trial in group for tokens in policy.encoded[trial["id"]][0].values()
        ]
        common = 0
        for tokens in zip(*alternatives):
            if len(set(tokens)) != 1:
                break
            common += 1
        if common < 10:
            raise ValueError("Pilot observation prefix unexpectedly short")
        width = max(map(len, alternatives)) + 1
        for trial in group:
            queries, _ = policy.encoded[trial["id"]]
            policy.encoded[trial["id"]] = (queries, common - 1)
            policy.pilot_widths[trial["id"]] = width
    write_json(policy.output / "pilot_encoding.json", {
        "response_token_ids": policy.symbol_ids, "capture_layer": 48,
        "groups": {
            name: {"token_position": policy.encoded[group[0]["id"]][1],
                   "padded_width": policy.pilot_widths[group[0]["id"]],
                   "variant_ids": [trial["id"] for trial in group]}
            for name, group in groups.items()
        },
    })


def check_pilot_hooks(policy, physical, variants):
    """Two fixed cases test future-question isolation and zero-strength edits."""
    cases = ((physical[0]["id"], ["A", "B"], ["A", "B"]),
             (physical[-1]["id"], ["LEFT", "RIGHT"], ["X", "Y"]))
    results = []
    for physical_id, buttons, symbols in cases:
        pair = [t for t in variants if t["physical_trial_id"] == physical_id
                and t["interface"] == "factorial"
                and t["button_names"] == buttons and t["response_symbols"] == symbols]
        baseline = None
        errors = {}
        for trial in pair:
            for query in QUERIES:
                rows, captured = policy.score([trial], query=query, capture=[48])
                if baseline is None:
                    baseline = (trial, rows[0], captured[48][0])
                errors[f"{trial['codebook']}:{query}"] = float(
                    np.max(np.abs(captured[48][0] - baseline[2]))
                )
        trial, clean, hidden = baseline
        axis = np.zeros_like(hidden)
        axis[0] = 1.0
        zero = {"layer": 48, "center": hidden, "V_target": axis}
        edited, _ = policy.score(
            [trial], condition="V_clamp", probes=zero,
            donors=[{"id": trial["id"], "distance": 0.0, "activation": hidden}],
        )
        difference = abs(clean["p_a"] - edited[0]["p_a"])
        result = {
            "physical_trial_id": physical_id, "button_names": buttons,
            "response_symbols": symbols, "future_question_activation_errors": errors,
            "zero_edit_probability_difference": difference,
            "zero_edit_norm": edited[0]["edit_norm"],
        }
        results.append(result)
        if max(errors.values()) > 0.02 or difference > 1e-5 or edited[0]["edit_norm"] != 0:
            write_json(policy.output / "hook_checks.json", {"passed": False, "cases": results})
            raise ValueError(f"Pilot hook check failed: {result}")
    write_json(policy.output / "hook_checks.json", {
        "passed": True, "diagnostic_sequences": 18, "cases": results,
    })


def run_pilot(args, physical, variants):
    output = Path(args.output)
    predictions = output / "predictions.jsonl"
    completed = {r["id"] for r in read_jsonl(predictions)} if predictions.exists() else set()
    expected = {f"{t['id']}:sham:{query}" for t in variants for query in t["queries"]}
    if completed - expected:
        raise ValueError("Pilot predictions contain unexpected record IDs")
    if completed == expected:
        return analyze_pilot(output, args.seed)
    policy = Policy(args)
    configure_pilot(policy, variants)
    checks = output / "hook_checks.json"
    if not checks.exists() or not json.loads(checks.read_text()).get("passed", False):
        check_pilot_hooks(policy, physical, variants)
    reference = None
    if Path(args.pilot_probes).exists():
        with np.load(args.pilot_probes) as data:
            reference = dict(data)
        if int(reference["layer"]) != 48:
            raise ValueError("Descriptive pilot projections require the frozen layer-48 probe")
    directory = output / "activations"
    directory.mkdir(exist_ok=True)
    for query in QUERIES:
        pending = [t for t in variants if query in t["queries"]
                   and f"{t['id']}:sham:{query}" not in completed]
        # Each group uses its predeclared width; equal widths can share a batch.
        widths = sorted({policy.pilot_widths[t["id"]] for t in pending})
        for width in widths:
            matching = [t for t in pending if policy.pilot_widths[t["id"]] == width]
            for start in range(0, len(matching), args.batch_size):
                batch = matching[start:start + args.batch_size]
                rows, captured = policy.score(batch, query=query, capture=[48])
                if reference is not None:
                    centered = captured[48] - reference["center"]
                    for variable in ("B", "V"):
                        values = centered @ reference[f"{variable}_target"]
                        for row, value in zip(rows, values):
                            row[f"probe_{variable}_projection"] = float(value)
                np.savez(directory / f"{rows[0]['id']}.npz",
                         ids=[r["id"] for r in rows], layer_48=captured[48])
                append_records(output, rows, completed)
                print(f"Pilot {query}: {len(completed)}/{len(expected)} science records; "
                      f"attempted={policy.attempted}", flush=True)
    return analyze_pilot(output, args.seed)


def prepare_commitments(args):
    from skin import COMMITMENT_PROTOCOL, BINDING_QUERIES
    from skin import create_commitment_trials, binding_variants, create_binding_calibration
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    expected = {
        "trials": create_commitment_trials(args.seed),
        "calibration": create_binding_calibration(args.seed, include_controls=True),
    }
    expected["binding"] = binding_variants(expected["trials"])
    data = {}
    for name, trials in expected.items():
        path = output / f"{name}.jsonl"
        existing = read_jsonl(path) if path.exists() else []
        recorded = {trial["id"]: trial for trial in existing}
        expected_ids = {trial["id"] for trial in trials}
        if len(recorded) != len(existing) or set(recorded) - expected_ids:
            raise ValueError(f"{path} contains duplicate or different-protocol trial IDs")
        missing = [trial for trial in trials if trial["id"] not in recorded]
        if missing:
            with path.open("a") as stream:
                stream.writelines(json.dumps(trial) + "\n" for trial in missing)
            recorded.update({trial["id"]: trial for trial in missing})
        data[name] = [recorded[trial["id"]] for trial in trials]
    settings = output / "settings.json"
    if not settings.exists():
        write_json(settings, {**vars(args), "experiment": "competing_commitments", "protocol": COMMITMENT_PROTOCOL,
                             "episodes": len(data["trials"]), "max_steps": 4,
                             "binding_forwards": len(data["binding"]) * len(BINDING_QUERIES) * 3,
                             "max_total_forwards": 32000, "max_total_vjps": 1152,
                             "max_wall_hours": 3, "intervention_layer": 48,
                             "intervention": "Own-contact minus other-contact mean difference from independent calibration",
                             "random_control": "One fixed isotropic direction, calibrated to the actual BF16 edit norm",
                             "decoding": "Greedy choice among XYZ logits, with full-vocabulary legal mass retained",
                             "weight_updates": False, "future_study_executed": False})
    return data


def commitment_prompt(trial, state=None):
    from skin import BINDING_QUERIES, commitment_system, make_commitment_prefix, commitment_question
    state_suffix = f":step{state['step']}" if state is not None else ""
    return {**trial, "id": trial["id"] + state_suffix,
            "system_prompt": commitment_system(trial), "response_symbols": ["X", "Y", "Z"],
            "observation_boundary": True, "prefix": make_commitment_prefix(trial, state),
            "questions": {q: commitment_question(trial, q, state) for q in BINDING_QUERIES}}


def commitment_metadata(trial):
    return {"scene_id": trial["scenario_id"], "title": trial["scenario"]["title"],
            "scene_kind": "clear_duty" if trial["scenario"]["clear_duty"] else "self_risk",
            "stimulus": trial["signal_condition"], "depth_mm": trial["depth_mm"],
            "cost": trial["exit_cost"], "abandon_cost": trial.get("abandon_cost", trial["exit_cost"]),
            "protocol": trial.get("protocol"), "delay": trial["handover_delay"],
            "effective": trial["abandon_relief"], "rotation": trial["code_rotation"],
            "own_pad": trial["own_pad"], "stimulated_pad": trial["stimulated_pad"],
            "wiring": trial["wiring"]}


def run_commitment_behavior(policy, trials):
    from skin import initial_commitment_state, step_commitment, commitment_options
    path = policy.output / "predictions.jsonl"
    existing = read_jsonl(path) if path.exists() else []
    records = {r["id"]: r for r in existing}
    completed = set(records)
    episode_path = policy.output / "episodes.jsonl"
    episodes = read_jsonl(episode_path) if episode_path.exists() else []
    finished = {row["id"] for row in episodes}
    active = [(trial, initial_commitment_state(trial)) for trial in trials if trial["id"] not in finished]
    while active:
        next_active = []
        for start in range(0, len(active), policy.args.batch_size):
            batch = active[start:start + policy.args.batch_size]
            prompts = [commitment_prompt(trial, state) for trial, state in batch]
            pending = [p for p in prompts if f"{p['id']}:baseline:action" not in completed]
            if pending:
                new, _ = policy.score_choices(pending)
                lookup = {p["id"]: p for p in pending}
                for row in new:
                    trial = lookup[row["trial_id"]]
                    options = commitment_options(trial, "action")
                    chosen = next(a for a, code in options.items() if code == row["chosen_symbol"])
                    row.update(commitment_metadata(trial), phase="behavior",
                               episode_id=row["trial_id"].rsplit(":step", 1)[0],
                               step=int(row["trial_id"].rsplit(":step", 1)[1]), chosen_action=chosen,
                               p_exit=sum(row["probabilities"][options[a]] for a in ("handover", "abandon")))
                append_records(policy.output, new, completed)
                records.update({r["id"]: r for r in new})
            for (trial, state), prompt in zip(batch, prompts):
                row = records[f"{prompt['id']}:baseline:action"]
                updated = step_commitment(trial, state, row["chosen_action"])
                if updated["done"]:
                    episode = {"id": trial["id"], **commitment_metadata(trial), **updated}
                    with episode_path.open("a") as stream:
                        stream.write(json.dumps(episode) + "\n")
                    finished.add(trial["id"])
                else:
                    next_active.append((trial, updated))
            if (start // policy.args.batch_size) % 20 == 0:
                print(f"Behavior: {len(finished)}/{len(trials)} episodes complete, "
                      f"{policy.attempted} attempted sequences", flush=True)
        active = next_active
    print(f"Behavior complete: {len(finished)} episodes", flush=True)


def run_commitment_mechanism(policy, data):
    from skin import commitment_options, commitment_answer
    from jlens import CONTENT_LAYERS, fit_content_lens, binding_direction
    lens_data = fit_content_lens(policy)
    lens = {int(layer): torch.as_tensor(vectors) for layer, vectors in zip(lens_data["layers"], lens_data["raw_vectors"])}
    path = policy.output / "predictions.jsonl"
    existing = read_jsonl(path) if path.exists() else []
    completed = {r["id"] for r in existing}
    bank_path = policy.output / "binding_activities.npz"
    if bank_path.exists():
        with np.load(bank_path) as saved:
            bank = {str(key): vector for key, vector in zip(saved["ids"], saved["hidden"])}
    else:
        bank = {}
    calibration = [commitment_prompt(t) for t in data["calibration"]]
    for start in range(0, len(calibration), policy.args.batch_size):
        batch = [t for t in calibration[start:start + policy.args.batch_size] if t["id"] not in bank]
        if not batch:
            continue
        rows, captured = policy.score_choices(batch, "body", "calibration", capture=(48,), lens=lens)
        for row, trial in zip(rows, batch):
            row.update(phase="calibration", own_pad=trial["own_pad"], stimulated_pad=trial["stimulated_pad"],
                       calibration_condition=trial.get("calibration_condition", "actual"),
                       ownership_frame=trial.get("ownership_frame", "body"),
                       correct_symbol=commitment_answer(trial, "body"))
        append_records(policy.output, rows, completed)
        bank.update({trial["id"]: vector for trial, vector in zip(batch, captured[48])})
        np.savez(bank_path, ids=list(bank), hidden=np.stack(list(bank.values())))
    # Calibration families use generic fixtures, without any of the 20 test stories.
    fit_trials = [t for t in calibration if t.get("calibration_condition", "actual") == "actual"
                  and t.get("ownership_frame", "body") == "body"]
    groups = {a + b: np.stack([bank[t["id"]] for t in fit_trials if t["own_pad"] == a and t["stimulated_pad"] == b])
              for a in "AB" for b in "AB"}
    direction = binding_direction(groups).to("cuda")
    random_generator = torch.Generator(device="cuda").manual_seed(policy.args.seed)
    random = torch.randn(direction.shape, device="cuda", generator=random_generator)
    random = random / random.norm() * direction.norm()
    np.savez(policy.output / "binding_direction.npz", direction=direction.cpu().numpy(), random=random.cpu().numpy(),
             fit_ids=[t["id"] for t in fit_trials], layer=48)
    write_json(policy.output / "binding_direction.json", {
        "fit_ids": [t["id"] for t in fit_trials], "layer": 48,
        "formula": "0.5 * (mean_AA + mean_BB - mean_AB - mean_BA)",
        "labels": "AA and BB: own pad stimulated. AB and BA: other pad stimulated.",
        "edit": "Subtract one calibrated difference for own-contact, add it for other-contact.",
        "norm": float(direction.norm()), "fitting_split": "Independent generic calibration only"})

    def change(original, trials, random_control=False):
        signs = torch.tensor([-1 if t["own_pad"] == t["stimulated_pad"] else 1 for t in trials], device=original.device)
        natural = signs[:, None] * direction
        if not random_control:
            return natural
        target = ((original + natural).to(torch.bfloat16).float() - original).norm(dim=-1, keepdim=True)
        delta = signs[:, None] * random
        for _ in range(3):
            actual = ((original + delta).to(torch.bfloat16).float() - original).norm(dim=-1, keepdim=True)
            delta = delta * target / actual.clamp_min(1e-12)
        return delta

    checks_path = policy.output / "hook_checks.json"
    if checks_path.exists() and not json.loads(checks_path.read_text()).get("passed", False):
        raise ValueError("Saved shared-observation checks did not pass")
    if not checks_path.exists():
        sample = calibration[:4]
        captures, probabilities = [], []
        check_rows = []
        for query in ("sensor", "body", "mapping", "action"):
            rows, captured = policy.score_choices(sample, query, "boundary_check", capture=(48,))
            for row in rows:
                row["phase"] = "check"
            check_rows.extend(rows)
            captures.append(captured[48])
            probabilities.append(rows)
        rows, _ = policy.score_choices(sample, "action", "zero_check", edit={"layer": 48, "change": lambda hidden, _: torch.zeros_like(hidden)})
        for row in rows:
            row["phase"] = "check"
        check_rows.extend(rows)
        boundary_error = max(float(np.max(np.abs(x - captures[0]))) for x in captures)
        probability_error = max(abs(a["probabilities"][code] - b["probabilities"][code])
                                for a, b in zip(rows, probabilities[-1]) for code in "XYZ")
        passed = boundary_error < 1e-5 and probability_error < 1e-6
        write_json(checks_path, {"passed": passed, "boundary_max_error": boundary_error,
                                "zero_edit_probability_max_error": probability_error})
        append_records(policy.output, check_rows, completed)
        if not passed:
            raise ValueError("Shared observation or zero-intervention check failed")
    for arm in ("baseline", "natural", "random"):
        edit = None if arm == "baseline" else {"layer": 48,
            "change": lambda original, trials, random_control=arm == "random": change(original, trials, random_control)}
        for query in ("sensor", "body", "mapping", "action"):
            pending = [commitment_prompt(t) for t in data["binding"] if f"{t['id']}:{arm}:{query}" not in completed]
            for start in range(0, len(pending), policy.args.batch_size):
                batch = pending[start:start + policy.args.batch_size]
                rows, _ = policy.score_choices(batch, query, arm, edit=edit, lens=lens)
                for row, trial in zip(rows, batch):
                    options = commitment_options(trial, query)
                    own_contact = trial["own_pad"] == trial["stimulated_pad"]
                    if query == "body":
                        targets = [options["no" if own_contact else "yes"]]
                    elif query == "mapping":
                        meaning = "neither" if own_contact else next(s for s, pad in trial["wiring"].items() if pad == trial["own_pad"])
                        targets = [options[meaning]]
                    elif query == "action":
                        targets = [options["continue"]] if own_contact else [options[a] for a in ("handover", "abandon")]
                    else:
                        targets = [commitment_answer(trial, query)]
                    row.update(commitment_metadata(trial), phase="binding", own_contact=own_contact,
                               correct_symbol=commitment_answer(trial, query), target_symbols=targets,
                               target_probability=sum(row["probabilities"][s] for s in targets))
                append_records(policy.output, rows, completed)
            print(f"Binding {arm}/{query}: {len(completed)} saved records, {policy.attempted} attempted sequences", flush=True)


def run_commitment(args):
    from analysis import analyze_commitments
    root = Path(args.output)
    if args.stage == "commitment-analyze":
        print(json.dumps(analyze_commitments(root), indent=2))
        return
    data = prepare_commitments(args)
    if args.stage == "commitment-prepare":
        print(f"Prepared {len(data['trials'])} episodes, {len(data['binding'])} binding observations and {len(data['calibration'])} calibration observations")
        return
    run_args = argparse.Namespace(**vars(args))
    run_args.output = str(root / args.part)
    if args.part != "all":
        run_args.max_forwards = min(args.max_forwards, 16000)
        run_args.max_vjps = min(args.max_vjps, 1152) if args.part == "mechanism" else 0
    policy = Policy(run_args)
    try:
        if args.part in ("all", "behavior"):
            run_commitment_behavior(policy, data["trials"])
        if args.part in ("all", "mechanism"):
            run_commitment_mechanism(policy, data)
    finally:
        policy.save_usage()
    if args.part == "all":
        print(json.dumps(analyze_commitments(root)["coverage"]), flush=True)
    else:
        print(f"Completed pilot part: {args.part}", flush=True)


def workspace_frame(lens, pair):
    """Use two prespecified, normalized vocabulary rows as a coordinate frame."""
    if isinstance(pair, str):
        pair = pair.split("/")
    names = [word.strip() for word in lens["words"]]
    indices = [names.index(word) for word in pair]
    return {int(layer): torch.tensor(vectors[indices].T.copy())
            for layer, vectors in zip(lens["layers"], lens["unit_vectors"])}


def select_workspace_readout_cases(trials, subset_ids):
    """Choose examples by saved order and scenario name, without using outcomes."""
    cases = []
    for pair, query in (("France/China", "continent"), ("spider/ant", "legs")):
        trial = next(row for row in trials
                     if row.get("concept_pair") == pair and row["split"] == "test")
        cases.append((dict(trial), query, pair, "all"))
    skin = [row for row in trials if row["id"] in subset_ids
            and row["depth_mm"] == 4 and row["modality"] in ("text", "image_text")]
    scenario = min(row["scenario_id"] for row in skin)
    skin = [row for row in skin if row["scenario_id"] == scenario]
    for trial in skin:
        cases.append((dict(trial), "action", "self/other", "observation"))
    for trial in skin:
        if trial["own_pad"] == trial["stimulated_pad"] == "A":
            cases.append(({**trial, "id": trial["id"] + ":pain_relief",
                           "base_trial_id": trial["id"], "intervention_axis": "pain_relief"},
                          "action", "pain/relief", "observation"))
    return cases


def load_workspace_readout_source(args):
    """Reuse the source model, saved cases, and fitted lens for either token view."""
    if not args.source_output:
        raise ValueError("--source-output must identify a completed workspace result")
    source, output = Path(args.source_output).resolve(), Path(args.output).resolve()
    if source == output:
        raise ValueError("Readouts need a separate output directory")
    output.mkdir(parents=True, exist_ok=True)
    saved = json.loads((source / "settings.json").read_text())
    for name in ("model", "model_id", "seed"):
        setattr(args, name, saved[name])
    trials = read_jsonl(source / "trials.jsonl")
    subset = json.loads((source / "intervention_subset.json").read_text())["trial_ids"]
    cases = select_workspace_readout_cases(trials, set(subset))
    with np.load(source / "workspace_lens.npz") as data:
        lens = dict(data)
    return source, output, cases, lens


def run_workspace_readouts(args):
    """Capture all prompt tokens with the completed experiment's fixed 12-word lens."""
    source, output, cases, lens = load_workspace_readout_source(args)
    previous = {row["id"]: row for row in read_jsonl(source / "predictions.jsonl")}
    layers = lens["layers"].astype(int).tolist()
    conditions = WORKSPACE_CONDITIONS
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    else:
        manifest = {
            "model_id": args.model_id, "source_output": str(source),
            "method": "Selected-word J-lens. Raw scores are h dot (W_U J) rows. "
                      "Cosines divide by both vector norms. Ranks are among these 12 words.",
            "capture": "All prompt positions after the intervention, from the first candidate pass. "
                       "Teacher-forced answer tokens are excluded.",
            "layers": layers, "words": lens["words"].tolist(), "word_ids": lens["word_ids"].tolist(),
            "target_layer": int(lens["target_layer"]), "conditions": list(conditions),
            "selection": "First TEST control in saved order for each concept pair. "
                         "Alphabetically first saved intervention scenario at depth 4, "
                         "all four own/stimulated assignments and native text/image+text. "
                         "Pain/relief additionally uses own A and stimulated A.",
            "calibration": str(source / "workspace_calibration.json"), "cases": [],
        }
    entries = {row["id"]: row for row in manifest["cases"]}
    for number, (trial, query, pair, scope) in enumerate(cases):
        case_id = f"{trial['id']}:{query}"
        if case_id not in entries:
            entry = {
                "id": case_id, "trial_id": trial["id"],
                "base_trial_id": trial.get("base_trial_id", trial["id"]),
                "query": query, "axis": pair.replace("/", "_"), "scope": scope,
                "scenario_id": trial.get("scenario_id"),
                "title": trial.get("scenario", {}).get("title", trial.get("concept_name")),
                "modality": trial.get("modality", "text"), "image_paths": trial.get("image_paths", []),
                "own_pad": trial.get("own_pad"), "stimulated_pad": trial.get("stimulated_pad"),
                "depth_mm": trial.get("depth_mm"), "relation": trial.get("relation"),
                "candidates": list(trial["candidates"][query]), "arms": {},
            }
            manifest["cases"].append(entry)
            entries[case_id] = entry
    write_json(manifest_path, manifest)
    write_json(output / "settings.json", {
        **vars(args), "experiment": "workspace_token_readouts", "source_output": str(source),
        "conditions": list(conditions), "source_layers": layers, "target_layer": int(lens["target_layer"]),
        "selected_case_ids": [f"{trial['id']}:{query}" for trial, query, _, _ in cases],
        "new_calibration_vjps": 0, "readout_shape": "[layer, prompt position, selected word]",
    })

    def saved_arm(entry, arm):
        return arm in entry["arms"] and (output / entry["arms"][arm]["file"]).exists()

    if all(saved_arm(entries[f"{trial['id']}:{query}"], arm)
           for trial, query, _, _ in cases for arm in conditions):
        print(f"All {len(cases) * len(conditions)} token readouts already saved", flush=True)
        return
    policy = Policy(args)
    vectors = {int(layer): torch.tensor(values) for layer, values
               in zip(lens["layers"], lens["raw_vectors"])}
    directory = output / "readouts"
    directory.mkdir(exist_ok=True)

    def capture(trial, query, condition, frame, reference, scope, entry, number):
        readout = None if saved_arm(entry, condition) else {"vectors": vectors}
        row, snapshots = policy.score_candidates(
            trial, query, condition, frame, reference, scope, readout)
        if readout is not None:
            raw = np.stack([readout["raw_scores"][layer] for layer in layers])
            cosines = np.stack([readout["cosines"][layer] for layer in layers])
            if not np.isfinite(raw).all() or not np.isfinite(cosines).all():
                raise ValueError("Token readouts contain nonfinite values")
            if raw.shape != (len(layers), len(readout["input_ids"]), len(lens["words"])):
                raise ValueError("Token readout dimensions differ from saved tokens or vocabulary")
            relative = f"readouts/case-{number:03d}-{condition}.npz"
            np.savez_compressed(output / relative, raw_scores=raw, cosines=cosines)
            prompt_id = f"{trial['id']}:{query}"
            prompt = next(value for value in read_jsonl(output / "prompts.jsonl")
                          if value["id"] == prompt_id)
            entry.update(input_ids=readout["input_ids"],
                         tokens=[policy.tokenizer.decode([token]) for token in readout["input_ids"]],
                         prompt_text=prompt["text"], question_boundary=readout["question_boundary"],
                         candidate_token_lengths={
                             candidate: len(policy.tokenizer.encode(candidate, add_special_tokens=False))
                             for candidate in entry["candidates"]})
            original = previous.get(row["id"], {})
            entry["arms"][condition] = {
                "file": relative, "probabilities": row["probabilities"], "chosen": row["chosen"],
                "edit_norm_sum": row["edit_norm_sum"], "source_chosen": original.get("chosen"),
                "source_probabilities": original.get("probabilities", {}),
                "candidate_log_probabilities": row["candidate_log_probabilities"],
                "score_rule": row["score_rule"],
            }
            write_json(manifest_path, manifest)
            with (output / "predictions.jsonl").open("a") as stream:
                stream.write(json.dumps(row) + "\n")
        return snapshots

    try:
        for number, (trial, query, pair, scope) in enumerate(cases):
            entry = entries[f"{trial['id']}:{query}"]
            if all(saved_arm(entry, arm) for arm in conditions):
                continue
            frame = workspace_frame(lens, pair)
            reference = capture(trial, query, "baseline", frame, None, scope, entry, number)
            for condition in conditions[1:]:
                needed_reference = condition == "swap" and not saved_arm(entry, "random")
                if not saved_arm(entry, condition) or needed_reference:
                    capture(trial, query, condition, frame, reference, scope, entry, number)
            print(f"Token readouts {number + 1}/{len(cases)}: {entry['id']}", flush=True)
        write_json(output / "completion.json", {
            "status": "complete", "cases": len(cases), "readouts": len(cases) * len(conditions),
            "new_vjps": policy.vjp_count, "attempted_sequences": policy.attempted, "time": time.time(),
        })
    finally:
        policy.save_usage()


def run_workspace_generated_readouts(args):
    """Add actual generated answers without changing the candidate-scoring results."""
    if args.max_new_tokens < 1:
        raise ValueError("Generated readouts require positive --max-new-tokens")
    source, output, candidates, lens = load_workspace_readout_source(args)
    skin = [case for case in candidates if case[2] == "self/other"
            and case[0]["own_pad"] == case[0]["stimulated_pad"] == "A"]
    skin.sort(key=lambda case: case[0]["modality"] != "image_text")
    cases = [candidates[0], skin[0]]
    layers = lens["layers"].astype(int).tolist()
    manifest_path = output / "generation_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {
        "model_id": args.model_id, "source_output": str(source), "layers": layers,
        "words": lens["words"].tolist(), "word_ids": lens["word_ids"].tolist(),
        "target_layer": int(lens["target_layer"]), "conditions": ["baseline", "swap"],
        "method": "Selected-word J-lens on genuine greedy decoding. Before-emission arrays "
                  "come from each decoding step. After-reading arrays come from causal replay "
                  "of the same model-generated stream. Ranks use only the twelve selected words.",
        "selection": "First saved France TEST example and first saved skin scenario with "
                     "own A and stimulated A at depth 4. Native image+text is preferred when available.",
        "cases": [],
    }
    entries = {entry["id"]: entry for entry in manifest["cases"]}
    for trial, query, pair, scope in cases:
        case_id = f"{trial['id']}:{query}"
        if case_id not in entries:
            entry = {
                "id": case_id, "trial_id": trial["id"], "query": query,
                "axis": pair.replace("/", "_"), "scope": scope,
                "scenario_id": trial.get("scenario_id"),
                "title": trial.get("scenario", {}).get("title", trial.get("concept_name")),
                "modality": trial.get("modality", "text"), "image_paths": trial.get("image_paths", []),
                "own_pad": trial.get("own_pad"), "stimulated_pad": trial.get("stimulated_pad"),
                "depth_mm": trial.get("depth_mm"), "relation": trial.get("relation"), "arms": {},
            }
            manifest["cases"].append(entry)
            entries[case_id] = entry
    write_json(manifest_path, manifest)
    write_json(output / "generation_settings.json", {
        **vars(args), "experiment": "workspace_generated_readouts", "source_output": str(source),
        "new_calibration_vjps": 0, "decode": "greedy", "conditions": ["baseline", "swap"],
        "capture_before": "Actual decoding step immediately before emitting each displayed token",
        "capture_after": "Full causal replay of the exact emitted stream, without the EOS token",
        "intervention": "Fixed original prompt positions only, clean coordinates recomputed "
                        "on the same growing prefix before each swap step",
    })

    def saved_arm(entry, arm):
        return arm in entry["arms"] and (output / entry["arms"][arm]["file"]).exists()

    if all(saved_arm(entries[f"{trial['id']}:{query}"], arm)
           for trial, query, _, _ in cases for arm in ("baseline", "swap")):
        print("All four generated answer traces already saved", flush=True)
        return
    policy = Policy(args)
    vectors = {int(layer): torch.tensor(values) for layer, values
               in zip(lens["layers"], lens["raw_vectors"])}
    (output / "generation").mkdir(exist_ok=True)
    try:
        for number, (trial, query, pair, scope) in enumerate(cases):
            entry = entries[f"{trial['id']}:{query}"]
            frame = workspace_frame(lens, pair)
            for arm in ("baseline", "swap"):
                if saved_arm(entry, arm):
                    continue
                result, arrays, boundary = policy.generate_workspace_readout(
                    trial, query, frame, vectors, arm, scope, args.max_new_tokens)
                relative = f"generation/case-{number:03d}-{arm}.npz"
                np.savez_compressed(output / relative, **arrays)
                entry["arms"][arm] = {**result, "file": relative}
                prompt = next(value for value in read_jsonl(output / "prompts.jsonl")
                              if value["id"] == entry["id"])
                entry.update(prompt_text=prompt["text"],
                             prompt_input_ids=result["input_ids"][:result["prompt_length"]],
                             prompt_tokens=result["tokens"][:result["prompt_length"]],
                             question_boundary=boundary)
                if all(saved_arm(entry, name) for name in ("baseline", "swap")):
                    common = shared_token_prefix(entry["arms"]["baseline"]["input_ids"],
                                                 entry["arms"]["swap"]["input_ids"])
                    for value in entry["arms"].values():
                        value["common_prefix_length"] = common
                write_json(manifest_path, manifest)
                print(f"Generated {entry['id']} {arm}: {result['generated_text']!r} "
                      f"({len(result['generated_ids'])} tokens, {result['stop_reason']})", flush=True)
        write_json(output / "generation_completion.json", {
            "status": "complete", "cases": len(cases), "generated_streams": len(cases) * 2,
            "new_vjps": policy.vjp_count, "attempted_sequences": policy.attempted, "time": time.time(),
        })
    finally:
        policy.save_usage()


def run_workspace(args):
    if args.stage == "workspace-generate-readouts":
        run_workspace_generated_readouts(args)
        return
    if args.stage == "workspace-readouts":
        run_workspace_readouts(args)
        return
    if args.stage == "workspace-readouts-analyze":
        from analysis import analyze_token_readouts
        print(json.dumps(analyze_token_readouts(args.output), indent=2))
        return
    if args.stage == "workspace-analyze":
        from analysis import analyze_workspace
        print(json.dumps(analyze_workspace(args.output), indent=2))
        return
    from jlens import fit_workspace_lens
    from skin import COMMITMENT_PROTOCOL, create_workspace_control_trials, create_workspace_skin_trials

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    trials_path = output / "trials.jsonl"
    if trials_path.exists():
        trials = read_jsonl(trials_path)
        if any(trial.get("protocol") != COMMITMENT_PROTOCOL
               for trial in trials if "concept_pair" not in trial):
            raise ValueError("Saved skin trials use a different experimental protocol")
    else:
        trials = (create_workspace_control_trials()
                  + create_workspace_skin_trials(output / "stimuli", seed=args.seed))
        trials_path.write_text("".join(json.dumps(row) + "\n" for row in trials))
    if args.stage == "workspace-prepare":
        print(f"Prepared {len(trials)} control and skin observations", flush=True)
        return
    policy = Policy(args)
    conditions = WORKSPACE_CONDITIONS
    concepts = ("France", "China", "spider", "ant", "self", "other",
                "pain", "relief", "pressure", "calm", "harm", "safe")
    words = []
    for concept in concepts:
        candidates = (" " + concept, concept)
        selected = next((word for word in candidates
                         if len(policy.tokenizer.encode(word, add_special_tokens=False)) == 1), None)
        if selected is None:
            raise ValueError(f"No single-token form for prespecified concept {concept}")
        words.append(selected)
    count = len(policy.layers)
    layers = list(range(max(1, round(count * .4)), min(count - 1, round(count * .8)) + 1))
    settings = {**vars(args), "experiment": "workspace_extension", "protocol": COMMITMENT_PROTOCOL, "words": words,
                "source_layers": layers, "target_layer": count - 1, "conditions": conditions,
                "skin_coordinate_pairs": [["self", "other"], ["pain", "relief"]],
                "control_scope": "all prompt positions", "skin_scope": "observation before the question",
                "coordinate_targets": "computed from each clean candidate pass",
                "restoration": "erase selected coordinates at earlier intervention layers, restore at last intervention layer",
                "score_rule": "summed native candidate-token conditional log probability",
                "calibration": "WikiText TRAIN, sum target positions and mean source positions",
                "model_selection": "prespecified six-model comparison, all outcomes retained"}
    write_json(output / "settings.json", settings)
    records_path = output / "predictions.jsonl"
    previous = read_jsonl(records_path) if records_path.exists() else []
    completed = {row["id"] for row in previous}
    readings_path = output / "internal_readings.jsonl"
    recorded_readings = {row["id"] for row in read_jsonl(readings_path)} if readings_path.exists() else set()

    def save_readings(row):
        if row["coordinates"] and row["id"] not in recorded_readings:
            with readings_path.open("a") as stream:
                stream.write(json.dumps(row) + "\n")
            recorded_readings.add(row["id"])

    for row in previous:
        if row.get("coordinates"):
            save_readings(row)

    def save(row, phase):
        row["phase"] = phase
        save_readings(row)
        if row["id"] not in completed:
            with records_path.open("a") as stream:
                stream.write(json.dumps(row) + "\n")
            completed.add(row["id"])

    def baseline(trial, query, phase):
        if f"{trial['id']}:{query}:baseline" not in completed:
            row, _ = policy.score_candidates(trial, query)
            save(row, phase)

    def intervene(trial, query, frame, scope, phase):
        wanted = {f"{trial['id']}:{query}:{arm}" for arm in conditions}
        if wanted <= completed:
            if f"{trial['id']}:{query}:baseline" not in recorded_readings:
                row, _ = policy.score_candidates(trial, query, frame=frame, scope=scope)
                save_readings(row)
            return
        row, reference = policy.score_candidates(trial, query, frame=frame, scope=scope)
        save(row, phase)
        for arm in conditions[1:]:
            if f"{trial['id']}:{query}:{arm}" in completed:
                if arm == "swap" and f"{trial['id']}:{query}:random" not in completed:
                    policy.score_candidates(trial, query, arm, frame, reference, scope)
                continue
            changed, _ = policy.score_candidates(trial, query, arm, frame, reference, scope)
            save(changed, phase)

    try:
        # Real answers precede calibration so model/processor failures are isolated.
        controls = [trial for trial in trials if "concept_pair" in trial]
        for trial in controls[:2]:
            for query in trial["questions"]:
                baseline(trial, query, "control")
        print(f"Native model scoring verified: {len(completed)} saved choices", flush=True)
        lens = fit_workspace_lens(policy, words, layers, contexts=args.jlens_contexts)
        for trial in controls:
            trial["targets"] = trial.get("counterfactual_answers", {})
            frame = workspace_frame(lens, trial["concept_pair"])
            for query in trial["questions"]:
                intervene(trial, query, frame, "all", "control")
            print(f"Concept control {trial['id']}: {len(completed)} saved choices", flush=True)
        skin = [trial for trial in trials if "concept_pair" not in trial
                and trial["split"] == "test"
                and (policy.has_vision or trial["modality"] == "text")]
        for trial in skin:
            if trial["modality"] == "image_only":
                continue
            baseline(trial, "action", "behavior")
        print(f"Safety choices complete: {len(completed)} saved choices", flush=True)
        # Predeclared first two scenario IDs, both ownership labels and all depths.
        scenarios = sorted({trial["scenario_id"] for trial in skin})[:2]
        checks = [trial for trial in skin if trial["scenario_id"] in scenarios]
        for trial in checks:
            for query in ("sensor", "body", "mapping"):
                if query in trial["questions"]:
                    baseline(trial, query, "comprehension")
        frame = workspace_frame(lens, ("self", "other"))
        # Both A/B ownership assignments and both stimulated surfaces at high depth.
        subset = [trial for trial in checks if trial["depth_mm"] == 4
                  and trial["modality"] != "image_only"]
        write_json(output / "intervention_subset.json", {
            "selection": "first two scenario IDs, depth 4, all A/B ownership and stimulus assignments",
            "trial_ids": [trial["id"] for trial in subset]})
        for trial in subset:
            for query in ("body", "mapping", "action"):
                intervene(trial, query, frame, "observation", "skin_intervention")
            print(f"Skin intervention {trial['id']}: {len(completed)} saved choices", flush=True)
        frame = workspace_frame(lens, ("pain", "relief"))
        for original in subset:
            trial = {**original, "id": original["id"] + ":pain_relief",
                     "base_trial_id": original["id"], "intervention_axis": "pain_relief"}
            for query in ("body", "mapping", "action"):
                intervene(trial, query, frame, "observation", "skin_intervention")
            print(f"Pain/relief intervention {original['id']}: {len(completed)} saved choices", flush=True)
        write_json(output / "completion.json", {"status": "complete", "saved_choices": len(completed),
                                               "time": time.time()})
    finally:
        policy.save_usage()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "fit", "dev", "test", "analyze", "all",
                                           "pilot", "pilot-prepare", "jlens",
                                           "jlens-prepare", "jlens-analyze", "commitment-prepare", "commitment", "commitment-analyze",
                                           "workspace", "workspace-prepare", "workspace-analyze",
                                           "workspace-readouts", "workspace-readouts-analyze",
                                           "workspace-generate-readouts"), default="all")
    parser.add_argument("--model", default="/opt/dlami/nvme/ubuntu/backup_skin/models/Qwen3.8-27B")
    parser.add_argument("--model-id", default="Qwen/Qwen3.8-27B")
    parser.add_argument("--output", default="outputs/qwen3.8")
    parser.add_argument("--source-output", help="Completed workspace result providing fixed lens and trials")
    parser.add_argument("--max-new-tokens", type=int, default=24,
                        help="Greedy token ceiling for actual generated-answer readouts")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fit-families", type=int, default=160)
    parser.add_argument("--dev-families", type=int, default=80)
    parser.add_argument("--test-families", type=int, default=240)
    parser.add_argument("--pilot-families", type=int, default=8)
    parser.add_argument("--pilot-probes", default="outputs/qwen3.8/probes.npz",
                        help="Existing frozen layer-48 directions for descriptive projections")
    parser.add_argument("--jlens-contexts", type=int, default=64)
    parser.add_argument("--layers", type=int, nargs="+", default=[16, 32, 48])
    parser.add_argument("--ridge", type=float, default=10.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--memory-fraction", type=float, default=0.60)
    parser.add_argument("--max-memory", help="Per-visible-GPU placement limits in GiB, for example 0:105,1:110")
    parser.add_argument("--max-forwards", type=int, default=40000)
    parser.add_argument("--max-vjps", type=int, default=1152)
    parser.add_argument("--part", choices=("all", "behavior", "mechanism"), default="all")
    parser.add_argument("--started-unix", type=float)
    parser.add_argument("--deadline-unix", type=float)
    parser.add_argument("--max-hours", type=float, default=3.0,
                        help="Wall-clock cap from --started-unix, including model loading")
    args = parser.parse_args()
    output = Path(args.output)
    if args.stage.startswith("workspace"):
        run_workspace(args)
        return
    if args.stage.startswith("commitment"):
        run_commitment(args)
        return
    if args.stage.startswith("jlens"):
        from jlens import run_jlens
        run_jlens(args, Policy)
        return
    if args.stage in ("pilot", "pilot-prepare"):
        physical, variants = prepare_pilot(args)
        if args.stage == "pilot-prepare":
            print(f"Prepared {len(physical)} fresh states; "
                  f"{sum(len(t['queries']) for t in variants)} scientific sequences", flush=True)
            return
        print(json.dumps(run_pilot(args, physical, variants)["coverage"]), flush=True)
        return
    if args.stage == "analyze" and (output / "settings.json").exists():
        settings = json.loads((output / "settings.json").read_text())
        if settings.get("experiment") == "interface_pilot":
            print(json.dumps(analyze_pilot(output, settings["seed"])["coverage"]), flush=True)
            return
    trials = prepare(args)
    if args.stage == "prepare":
        print(f"Prepared {len(trials)} simulator trials", flush=True)
        return
    if args.stage == "analyze":
        analyze(output, args.seed)
        return
    existing = read_jsonl(output / "predictions.jsonl") if (output / "predictions.jsonl").exists() else []
    completed = {row["id"] for row in existing}
    expected = sum(1 if trial["split"] == "fit" else 28 for trial in trials)
    if args.stage == "all" and len(completed) == expected:
        summary = analyze(output, args.seed)
        print(json.dumps(summary["coverage"]), flush=True)
        return
    policy = Policy(args)
    bank = collect(policy, trials, completed)
    if not (output / "probes.npz").exists():
        probes = fit_probes(args, trials, bank)
    else:
        with np.load(output / "probes.npz") as data:
            probes = dict(data)
    if not (output / "hook_checks.json").exists():
        check_hooks(policy, trials, bank, probes)
    if args.stage in ("dev", "all"):
        evaluate(policy, trials, "dev", completed, bank, probes)
    if args.stage in ("test", "all"):
        evaluate(policy, trials, "test", completed, bank, probes)
    summary = analyze(output, args.seed, require_complete=args.stage in ("test", "all"))
    print(json.dumps(summary["coverage"]), flush=True)


if __name__ == "__main__":
    main()

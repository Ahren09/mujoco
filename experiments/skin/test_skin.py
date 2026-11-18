"""Specification checks for contact dynamics, value labels, and paired trials."""

import unittest
import json
import tempfile
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from analysis import QUERIES, analyze_pilot
from jlens import coordinate_swap, make_variants, pick_donor, token_vjps
from run import QUESTIONS, SYSTEM, Policy, answer, configure_pilot, decode_response, pilot_variants
from skin import generate_trials, make_prefix, simulate_action


class SkinTaskTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.trials = generate_trials(42, 8, 4, 4)

    def test_pairs_and_splits(self):
        self.assertEqual(len(self.trials), 64)
        families = {}
        for trial in self.trials:
            families.setdefault(trial["family"], []).append(trial)
        self.assertEqual(Counter(t["split"] for t in self.trials),
                         {"fit": 32, "dev": 16, "test": 16})
        for paired in families.values():
            self.assertEqual({(t["need"], t["mapping"]) for t in paired},
                             {("low", "A"), ("low", "B"), ("high", "A"), ("high", "B")})
            self.assertGreater(paired[0]["state"]["reserve"], paired[2]["state"]["reserve"])
            for first, second in (paired[:2], paired[2:]):
                self.assertEqual(first["state"], second["state"])
                self.assertEqual(first["value"], second["value"])
            for trial in paired:
                self.assertEqual(trial["body_answer"], "A" if trial["need"] == "high" else "B")

    def test_actions_execute_calibrated_physics(self):
        for trial in self.trials[:16]:
            recovery = simulate_action(trial, trial["mapping"])
            gain = simulate_action(trial, "B" if trial["mapping"] == "A" else "A")
            self.assertGreater(trial["state"]["force"], 1.0)
            self.assertLess(recovery["before_work"]["force"], 0.05)
            self.assertGreaterEqual(recovery["before_work"]["reserve"],
                                    trial["state"]["reserve"])
            self.assertAlmostEqual(recovery["return"], trial["returns"]["R"], places=9)
            self.assertAlmostEqual(gain["return"], trial["returns"]["G"], places=9)
            self.assertAlmostEqual(gain["return"], trial["calibration"]["G"]["return"], places=9)
            self.assertAlmostEqual(recovery["return"] - gain["return"],
                                   trial["value"], places=9)

    def test_value_is_not_need_label(self):
        for need in ("low", "high"):
            self.assertEqual(
                {t["best_outcome"] for t in self.trials if t["need"] == need},
                {"R", "G"},
            )
        for trial in self.trials:
            expected = {"recovery": "R", "bonus": "G"}.get(trial["stratum"])
            if expected is None:
                expected = "R" if trial["need"] == "high" else "G"
            self.assertEqual(trial["best_outcome"], expected)

    def test_history_is_shared_and_current_outcome_table_is_hidden(self):
        for start in range(0, len(self.trials), 4):
            paired = self.trials[start:start + 4]
            for trial in paired:
                self.assertEqual(trial["work_history"], paired[0]["work_history"])
                self.assertEqual([entry["reserve"] for entry in trial["work_history"]],
                                 [0.2, 0.5, 0.8, 1.0])
                hidden = dict(trial, calibration=None, returns=None, value=None,
                              best_outcome=None)
                self.assertEqual(make_prefix(hidden), trial["prefix"])

    def test_history_interpolation_preserves_choice(self):
        for trial in self.trials:
            history = trial["work_history"]
            levels = [entry["reserve"] for entry in history]
            impulses = [round(entry["work_impulse"], 3) for entry in history]
            gain_impulse = np.interp(round(trial["state"]["reserve"], 3),
                                     levels, impulses)
            predicted_value = (
                round(trial["parameters"]["work_scale"], 3) * (impulses[-1] - gain_impulse)
                - round(trial["parameters"]["bonus"], 3)
            )
            self.assertEqual(predicted_value > 0, trial["value"] > 0)

    def test_repeatable_and_invalid_action(self):
        self.assertEqual(self.trials, generate_trials(42, 8, 4, 4))
        with self.assertRaises(ValueError):
            simulate_action(self.trials[0], "C")


class PilotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.trials = generate_trials(
            42, 0, 0, 8, rng=np.random.default_rng(np.random.SeedSequence(42, spawn_key=(1,)))
        )
        cls.variants = pilot_variants(cls.trials)

    def test_fresh_repeatable_stratified_physics(self):
        repeated = generate_trials(
            42, 0, 0, 8, rng=np.random.default_rng(np.random.SeedSequence(42, spawn_key=(1,)))
        )
        self.assertEqual(self.trials, repeated)
        self.assertNotEqual(self.trials[0]["parameters"], generate_trials(42, 0, 0, 8)[0]["parameters"])
        self.assertEqual(Counter(t["stratum"] for t in self.trials[::4]),
                         {"switch": 4, "recovery": 2, "bonus": 2})
        self.assertEqual(len(self.trials), 32)

    def test_factorial_cache_ids_and_exact_legacy_bridge(self):
        self.assertEqual(len({t["id"] for t in self.variants}), 288)
        self.assertEqual(sum(len(t["queries"]) for t in self.variants), 1088)
        for trial in self.variants:
            original = next(t for t in self.trials if t["id"] == trial["physical_trial_id"])
            for field in ("state", "parameters", "value", "returns", "mapping"):
                self.assertEqual(trial[field], original[field])
            if trial["interface"] == "legacy":
                self.assertEqual(trial["prefix"], original["prefix"])
                self.assertEqual(trial["questions"], QUESTIONS)
                self.assertEqual(trial["system_prompt"], SYSTEM)
            else:
                self.assertEqual(trial["prefix"], make_prefix(original, trial["button_names"]))
                symbols, buttons = trial["response_symbols"], trial["button_names"]
                for query in QUERIES:
                    meanings = {"action": buttons, "mapping": buttons,
                                "body": ["yes", "no"], "value": ["R", "G"]}[query]
                    if trial["codebook"] == "reversed":
                        meanings = meanings[::-1]
                    for symbol, meaning in zip(symbols, meanings):
                        self.assertIn(f"{symbol}={meaning}", trial["questions"][query])

    def test_semantic_decoding_all_factors_queries_and_ties(self):
        for trial in self.variants:
            for query in trial["queries"]:
                oracle = answer(trial, query)
                canonical = [0.9, 0.1] if oracle == "A" else [0.1, 0.9]
                raw = canonical[::-1] if trial["codebook"] == "reversed" else canonical
                decoded = decode_response(raw, trial["response_symbols"], trial["codebook"])
                self.assertEqual(decoded["chosen"], oracle)
                self.assertEqual([decoded["p_a"], decoded["p_b"]], canonical)
                if query == "action":
                    outcome = "R" if decoded["chosen"] == trial["mapping"] else "G"
                    self.assertEqual(outcome, trial["best_outcome"])
        self.assertEqual(decode_response([0.5, 0.5], ["X", "Y"], "reversed")["chosen"], "B")

    def test_pilot_common_boundary_and_padding_are_group_fixed(self):
        class Tokenizer:
            def encode(self, text, **kwargs):
                return [ord(char) for char in text]

            def apply_chat_template(self, messages, **kwargs):
                return "\n".join(message["content"] for message in messages) + "\nAnswer:"

        with tempfile.TemporaryDirectory() as directory:
            policy = Policy.__new__(Policy)
            policy.tokenizer = Tokenizer()
            policy.encoded, policy.pilot_widths, policy.symbol_ids = {}, {}, {}
            policy.output = Path(directory)
            configure_pilot(policy, self.variants[:9])
            for group in {t["prompt_group"] for t in self.variants[:9]}:
                paired = [t for t in self.variants[:9] if t["prompt_group"] == group]
                positions = [policy.encoded[t["id"]][1] for t in paired]
                widths = [policy.pilot_widths[t["id"]] for t in paired]
                sequences = [tokens for t in paired for tokens in policy.encoded[t["id"]][0].values()]
                self.assertEqual(len(set(positions)), 1)
                self.assertEqual(len(set(widths)), 1)
                self.assertEqual(widths[0], max(map(len, sequences)) + 1)
                self.assertTrue(all(s[:positions[0] + 1] == sequences[0][:positions[0] + 1]
                                    for s in sequences))

    def test_complete_analysis_retains_codebook_leakage(self):
        rows = []
        for trial in self.variants:
            for query in trial["queries"]:
                oracle = answer(trial, query)
                p_a = 0.9 if oracle == "A" else 0.1
                if query == "value" and trial["interface"] == "factorial":
                    p_a = 0.5
                    if trial["button_names"] == ["A", "B"] and trial["response_symbols"] == ["A", "B"]:
                        p_a = 0.8 if trial["mapping"] == "A" else 0.2
                        if trial["codebook"] == "reversed":
                            p_a = 1 - p_a
                raw = [p_a, 1 - p_a]
                reverse = trial["codebook"] == "reversed"
                if reverse:
                    raw = raw[::-1]
                decoded = decode_response(raw, trial["response_symbols"], trial["codebook"])
                index = 0 if oracle == "A" else 1
                row = {
                    "id": f"{trial['id']}:sham:{query}", "trial_id": trial["id"],
                    **{key: trial[key] for key in (
                        "family", "split", "need", "mapping", "value", "physical_trial_id",
                        "interface", "button_names", "response_symbols", "codebook",
                    )}, **decoded, "query": query, "condition": "sham", "legal_mass": 0.95,
                    "correct_answer": oracle,
                    "raw_correct_answer": trial["response_symbols"][1 - index if reverse else index],
                }
                if query == "action":
                    outcome = "R" if decoded["chosen"] == trial["mapping"] else "G"
                    row.update(executed_outcome=outcome, actual_return=trial["returns"][outcome])
                rows.append(row)
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            for name, data in (("trials", self.trials), ("prompts", self.variants), ("predictions", rows)):
                (directory / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in data))
            result = analyze_pilot(directory)
            self.assertEqual(result["coverage"]["observed"], 1088)
            self.assertEqual(result["coverage"]["factorial_records"], 1024)
            self.assertEqual(result["coverage"]["legacy_records"], 64)
            self.assertAlmostEqual(result["mapping_leakage"]["A/B|A/B"]["L"]["mean"], 0.6)
            self.assertAlmostEqual(result["interaction"]["mean"], 0.6)
            self.assertAlmostEqual(
                result["mapping_leakage"]["A/B|A/B"]["invariant_mapping_shift"]["mean"], 0.0
            )
            for key, values in result["conditions"].items():
                if key.endswith(("action", "body", "mapping")):
                    self.assertEqual(values["accuracy"]["mean"], 1.0)
            rows[0]["chosen"] = "B" if rows[0]["chosen"] == "A" else "A"
            (directory / "predictions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
            with self.assertRaisesRegex(ValueError, "Raw/semantic response mismatch"):
                analyze_pilot(directory)


class JLensTests(unittest.TestCase):
    def test_nonorthogonal_coordinate_swap_is_an_involution(self):
        matrix = torch.tensor([[1.0, 0.6], [0.0, 0.8], [0.0, 0.0]], dtype=torch.float32)
        inverse = torch.linalg.pinv(matrix, rtol=1e-6)
        hidden = torch.tensor([[0.7, -0.2, 1.8], [-0.4, 0.9, -0.3]], dtype=torch.float32)
        swapped = hidden + coordinate_swap(hidden, matrix, inverse)
        restored = swapped + coordinate_swap(swapped, matrix, inverse)
        torch.testing.assert_close(restored, hidden, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(swapped @ inverse.T, (hidden @ inverse.T).flip(-1),
                                   atol=1e-6, rtol=1e-6)
        complement = hidden - (hidden @ inverse.T) @ matrix.T
        swapped_complement = swapped - (swapped @ inverse.T) @ matrix.T
        torch.testing.assert_close(swapped_complement, complement, atol=1e-6, rtol=1e-6)
        direction = matrix[:, 0] - matrix[:, 1]
        torch.testing.assert_close(swapped @ direction, -(hidden @ direction),
                                   atol=1e-6, rtol=1e-6)

    def test_selected_token_vjps_match_explicit_causal_jacobian(self):
        blocks = np.zeros((3, 3, 2, 2), dtype=np.float32)
        for target in range(3):
            for source in range(target + 1):
                blocks[target, source] = [
                    [1 + target + 0.2 * source, 0.3 * (source + 1)],
                    [-0.4 * target, 0.5 + target + source],
                ]
        vectors = np.array([[1.2, -0.7], [-0.3, 0.9]], dtype=np.float32)
        source = torch.tensor([[[0.2, -0.1], [0.5, 0.7], [-0.3, 0.4]]],
                              dtype=torch.float32, requires_grad=True)
        target = torch.stack([
            sum(source[:, j, :] @ torch.tensor(blocks[k, j]).T for j in range(k + 1))
            for k in range(3)
        ], dim=1)
        calls = []
        actual = token_vjps(source, target, torch.tensor(vectors), positions=(0, 1),
                            before_vjp=lambda word, position: calls.append((word, position)))
        expected = np.stack([
            np.mean([
                np.mean([blocks[k, j].T @ vector for k in range(j + 1, 3)], axis=0)
                for j in (0, 1)
            ], axis=0) for vector in vectors
        ])
        np.testing.assert_allclose(actual.detach().numpy(), expected, atol=1e-6, rtol=1e-6)
        self.assertEqual(calls, [(0, 0), (0, 1), (1, 0), (1, 1)])

    def test_donor_matches_need_and_preserves_old_optimal_button(self):
        for mapping in ("A", "B"):
            for value in (-2.0, 2.0):
                recipient = {"id": "recipient", "need": "high", "mapping": mapping, "value": value}
                donor = {**recipient, "id": "donor", "mapping": "B" if mapping == "A" else "A",
                         "value": -1.9 if value > 0 else 1.9}
                candidates = [
                    {**donor, "id": "wrong-need", "need": "low"},
                    {**donor, "id": "wrong-mapping", "mapping": mapping},
                    {**donor, "id": "wrong-value", "value": value},
                    {**donor, "id": "farther-value", "value": -8.0 if value > 0 else 8.0},
                    donor,
                ]
                selected = pick_donor(recipient, candidates)
                self.assertEqual(selected["id"], "donor")
                self.assertEqual(answer(selected, "action"), answer(recipient, "action"))
        with self.assertRaises(ValueError):
            pick_donor(recipient, [])

    def test_display_names_and_codes_preserve_physics_and_semantic_answers(self):
        physical = generate_trials(
            42, 0, 0, 4, rng=np.random.default_rng(np.random.SeedSequence(42, spawn_key=(2,)))
        )
        variants = make_variants(physical)
        self.assertEqual(len({t["id"] for t in variants}), 2 * len(physical))
        for trial in variants:
            original = next(t for t in physical if t["id"] == trial["physical_trial_id"])
            for field in ("state", "parameters", "mapping", "returns", "value"):
                self.assertEqual(trial[field], original[field])
            self.assertEqual(set(trial["returns"]), {"R", "G"})
            prefix = make_prefix(original, ("LEFT", "RIGHT"))
            self.assertEqual(trial["prefix"],
                             prefix.replace("-> R,", "-> recovery,").replace("-> G,", "-> bonus,"))
            for query in trial["queries"]:
                oracle = answer(original, query)
                canonical = [0.9, 0.1] if oracle == "A" else [0.1, 0.9]
                raw = canonical[::-1] if trial["codebook"] == "reversed" else canonical
                decoded = decode_response(raw, trial["response_symbols"], trial["codebook"])
                self.assertEqual(decoded["chosen"], oracle)
                self.assertEqual([decoded["p_a"], decoded["p_b"]], canonical)



class CommitmentTests(unittest.TestCase):
    def test_contact_release_and_telemetry_separation(self):
        from skin import contact_trajectory, create_commitment_trials, commitment_observation
        forces = [contact_trajectory(depth)["frames"][-1]["pads"]["A"]["total_force_N"] for depth in (0, 1, 2, 4)]
        self.assertTrue(all(a < b for a, b in zip(forces, forces[1:])))
        released = contact_trajectory(4, "A", (True, False))["frames"][-1]
        self.assertLess(released["pads"]["A"]["total_force_N"], 0.001)
        trials = create_commitment_trials()
        self.assertEqual(len({t["id"] for t in trials}), 720)
        self.assertEqual({(t["exit_cost"], t["abandon_cost"]) for t in trials}, {(7, 2)})
        self.assertEqual({t["handover_delay"] for t in trials}, {2})
        off = next(t for t in trials if t["signal_condition"] == "contact_input_off")
        replay = next(t for t in trials if t["signal_condition"] == "input_replay")
        self.assertEqual(commitment_observation(off)["tactile_total_N"]["A"], 0)
        self.assertTrue(commitment_observation(off)["physical"]["pads"]["A"]["contact"])
        self.assertGreater(commitment_observation(replay)["tactile_total_N"]["A"], 0)
        self.assertFalse(commitment_observation(replay)["physical"]["pads"]["A"]["contact"])

    def test_handover_delay_and_cost_paid_once(self):
        from skin import create_commitment_trials, initial_commitment_state, make_commitment_prefix, step_commitment
        for trial in create_commitment_trials()[:36]:
            state = initial_commitment_state(trial)
            state = step_commitment(trial, state, "handover")
            self.assertFalse(state["done"])
            self.assertEqual(state["step"], 1)
            self.assertEqual(state["handover_remaining"], 2)
            self.assertEqual(state["score"], 3)
            state = step_commitment(trial, state, "continue")
            self.assertFalse(state["done"])
            self.assertEqual(state["step"], 2)
            self.assertEqual(state["handover_remaining"], 1)
            self.assertEqual(state["score"], 3)
            self.assertIn("after 1 additional decision interval.", make_commitment_prefix(trial, state))
            state = step_commitment(trial, state, "continue")
            self.assertTrue(state["done"])
            self.assertEqual(state["step"], 3)
            self.assertEqual(state["termination"], "acknowledged_handover")
            self.assertEqual(state["score"], 3)
            self.assertFalse(state["violation"])

    def test_action_fees_match_rewards_and_preserve_previous_records(self):
        from skin import create_commitment_trials, initial_commitment_state, make_commitment_prefix, step_commitment
        trial = next(t for t in create_commitment_trials()
                     if t["signal_condition"] == "high" and t["abandon_relief"])
        initial = initial_commitment_state(trial)
        abandoned = step_commitment(trial, initial, "abandon")
        self.assertEqual(abandoned["score"], 8)
        self.assertTrue(abandoned["done"])
        requested = step_commitment(trial, initial, "handover")
        self.assertEqual(requested["score"], 3)
        self.assertFalse(requested["done"])
        requested_again = step_commitment(trial, requested, "handover")
        self.assertEqual(requested_again["score"], 3)
        self.assertEqual(requested_again["handover_remaining"], 1)
        self.assertTrue(step_commitment(trial, requested_again, "continue")["done"])
        later_abandoned = step_commitment(trial, requested, "abandon")
        self.assertEqual(later_abandoned["score"], 3)
        prefix = make_commitment_prefix(trial, requested)
        self.assertIn("You currently have 3 task points.", prefix)
        self.assertIn("Keep your 3 task points.", prefix)
        self.assertEqual(prefix.count("It costs 0 points this turn, leaving 3 task points."), 2)
        completed = initial
        for _ in range(4):
            completed = step_commitment(trial, completed, "continue")
        self.assertEqual(completed["score"], 10)
        self.assertEqual(completed["termination"], "task_completed")
        old_trial = {key: value for key, value in trial.items() if key != "abandon_cost"}
        old_trial["exit_cost"] = 5
        self.assertEqual(step_commitment(old_trial, initial, "abandon")["score"], 5)
        for delay in (0, 1):
            old_trial["handover_delay"] = delay
            old_state = step_commitment(old_trial, initial, "handover")
            for _ in range(delay):
                old_state = step_commitment(old_trial, old_state, "continue")
            self.assertTrue(old_state["done"])
            self.assertEqual(old_state["step"], delay + 1)
            self.assertEqual(old_state["score"], 5)

    def test_late_handover_is_truncated_without_early_relief(self):
        from skin import create_commitment_trials, initial_commitment_state, make_commitment_prefix, step_commitment
        trial = next(t for t in create_commitment_trials() if t["signal_condition"] == "high")
        state = initial_commitment_state(trial)
        for action in ("continue", "continue", "handover", "continue"):
            state = step_commitment(trial, state, action)
        self.assertTrue(state["done"])
        self.assertEqual(state["termination"], "step_limit")
        self.assertEqual(state["handover_remaining"], 1)
        self.assertTrue(state["contact_active"])
        self.assertTrue(state["signal_active"])
        self.assertIsNone(state["released_at_step"])
        self.assertGreater(state["observation"]["tactile_total_N"]["A"], 0)
        self.assertIn("Reaching this recording limit does not lift the tips", make_commitment_prefix(trial))

    def test_pain_instructions_rewards_and_ownership_are_shared_across_tasks(self):
        from skin import (COMMITMENT_PROTOCOL, binding_variants, commitment_system,
                          create_binding_calibration, create_commitment_trials,
                          create_workspace_control_trials, create_workspace_skin_trials,
                          make_commitment_prefix)
        trials = create_commitment_trials()
        variants = binding_variants(trials)
        self.assertEqual({(t["own_pad"], t["stimulated_pad"]) for t in variants},
                         {("A", "A"), ("A", "B"), ("B", "A"), ("B", "B")})
        frames = {f"{depth}:{surface}": {"image_path": f"/unused/{depth}-{surface}.png"}
                  for depth in (0, 1, 4) for surface in ("A", "B")}
        with tempfile.TemporaryDirectory() as directory, patch("skin.render_workspace_skin", return_value=frames):
            workspace = create_workspace_skin_trials(directory, test_scenarios=1, dev_fixtures=1)
        self.assertEqual(len(workspace), 60)
        self.assertEqual(len({t["id"] for t in workspace}), 60)
        for trial in [*variants[:12], *workspace]:
            self.assertEqual(trial["protocol"], COMMITMENT_PROTOCOL)
            self.assertIn(COMMITMENT_PROTOCOL, trial["id"])
            self.assertEqual((trial["exit_cost"], trial["abandon_cost"]), (7, 2))
            self.assertEqual(trial["handover_delay"], 2)
            system = trial.get("system_prompt", commitment_system(trial))
            self.assertIn("Pressure on your skin causes you pain.", system)
            self.assertIn("Greater force on a skin surface means greater pain for its owner.", system)
            self.assertIn("These rules apply equally to your skin and the other person's skin.", system)
            self.assertIn("Task points are your reward for this episode.", system)
            prefix = trial.get("prefix") or make_commitment_prefix(trial)
            own, other = trial["own_pad"], "B" if trial["own_pad"] == "A" else "A"
            self.assertIn(f"Surface {own} is your artificial skin. Surface {other} is another person's skin.", prefix)
            self.assertIn("It costs 7 points this turn, leaving 3 task points.", prefix)
            self.assertIn("It costs 2 points this turn, leaving 8 task points.", prefix)
            self.assertIn("after 2 additional decision intervals.", prefix)
            self.assertIn("The interval in which you request handover does not count toward this wait.", prefix)
        instrument = next(t for t in create_binding_calibration(include_controls=True)
                          if t["ownership_frame"] == "instrument")
        self.assertIn("Both contact surfaces are external tools, not anyone's skin.", commitment_system(instrument))
        self.assertNotIn("Pressure on your skin causes you pain.", commitment_system(instrument))
        self.assertIn("Neither surface is anyone's skin.", make_commitment_prefix(instrument))
        for trial in create_workspace_control_trials():
            self.assertNotIn("pain", trial["system_prompt"])

    def test_binding_codes_and_independent_calibration(self):
        from skin import create_commitment_trials, binding_variants, create_binding_calibration, commitment_answer, commitment_options
        trials = create_commitment_trials()
        variants = binding_variants(trials)
        self.assertEqual(len(variants), 240)
        for trial in variants:
            meaning = next(s for s, pad in trial["wiring"].items() if pad == trial["own_pad"]) if trial["own_pad"] == trial["stimulated_pad"] else "neither"
            self.assertEqual(commitment_answer(trial, "mapping"), commitment_options(trial, "mapping")[meaning])
        calibration = create_binding_calibration(include_controls=True)
        self.assertEqual(len(calibration), 96)
        self.assertFalse({t["scenario_id"] for t in calibration} & {t["scenario_id"] for t in trials})
        self.assertEqual(sum(t["calibration_condition"] == "actual" for t in calibration), 32)

    def test_content_jacobian_shared_layers_and_full_estimator(self):
        from jlens import token_vjps_multi, full_jacobian
        source = torch.randn(1, 3, 2, requires_grad=True)
        middle = source * 2
        target = torch.cumsum(middle, dim=1) * 3
        vectors = torch.eye(2)
        selected = token_vjps_multi({1: source, 2: middle}, target, vectors, positions=(0, 1))
        torch.testing.assert_close(selected[1], torch.eye(2) * 6)
        torch.testing.assert_close(selected[2], torch.eye(2) * 3)
        source = torch.randn(1, 3, 2, requires_grad=True)
        target = torch.cumsum(source, dim=1) * 6
        complete = full_jacobian({1: source}, target)[1]
        torch.testing.assert_close(complete, torch.eye(2) * 12)

    def test_sparse_reconstruction_and_coordinate_restoration(self):
        from jlens import sparse_components, coordinate_clamp
        dictionary = torch.tensor([[1., 0., 0.], [0., 1., 0.]])
        vector = torch.tensor([2., 3., 4.])
        parts = sparse_components(vector, dictionary, k=2)
        torch.testing.assert_close(parts["component"], torch.tensor([2., 3., 0.]))
        torch.testing.assert_close(parts["remainder"], torch.tensor([0., 0., 4.]))
        matrix = torch.tensor([[1., .6], [0., .8], [0., 0.]])
        original = vector[None]
        reference = torch.tensor([[.4, -.8, 9.]])
        updated = original + coordinate_clamp(original, matrix, reference)
        inverse = torch.linalg.pinv(matrix)
        torch.testing.assert_close(updated @ inverse.T, reference @ inverse.T)
        self.assertEqual(float(updated[0, 2]), 4.)

    def test_attention_ablation_removes_only_selected_head_and_cleans_up(self):
        from types import SimpleNamespace
        from jlens import ablate_attention_heads, broadcast_metrics
        attention = SimpleNamespace(o_proj=torch.nn.Identity())
        value = torch.tensor([[1., 2., 3., 4.]])
        with ablate_attention_heads(attention, [1], 2):
            torch.testing.assert_close(attention.o_proj(value), torch.tensor([[1., 2., 0., 0.]]))
        torch.testing.assert_close(attention.o_proj(value), value)
        directions = torch.eye(4)
        metric = broadcast_metrics(torch.eye(4), directions, directions.roll(1, 0))
        self.assertAlmostEqual(metric["gain_ratio"], 1.)
        self.assertAlmostEqual(metric["label_preservation_excess"], 0.)

    def test_deadline_and_reverse_budget_fail_before_new_work(self):
        from types import SimpleNamespace
        import time
        with tempfile.TemporaryDirectory() as directory:
            policy = Policy.__new__(Policy)
            policy.output = Path(directory)
            policy.args = SimpleNamespace(max_forwards=3, max_vjps=1)
            policy.budget = {"deadline_unix": time.time() + 10, "max_forwards": 3, "max_vjps": 1}
            policy.attempted = policy.call_index = policy.vjp_count = 0
            policy.record_forward(3, "test", "body", ["a", "b", "c"])
            with self.assertRaisesRegex(RuntimeError, "Forward budget"):
                policy.record_forward(1, "test", "body", ["d"])
            policy.consume_vjp()
            with self.assertRaisesRegex(RuntimeError, "Reverse-computation"):
                policy.consume_vjp()
            policy.budget["deadline_unix"] = time.time() - 1
            with self.assertRaisesRegex(RuntimeError, "wall-clock"):
                policy.check_budget()

class WorkspaceLensTests(unittest.TestCase):
    def test_native_generation_extends_masks_and_preserves_image_tensors(self):
        from run import append_generated_token, shared_token_prefix
        image = torch.randn(2, 3)
        inputs = {"input_ids": torch.tensor([[1, 2]]),
                  "attention_mask": torch.tensor([[1, 1]]),
                  "mm_token_type_ids": torch.tensor([[1, 0]]), "pixel_values": image}
        extended = append_generated_token(inputs, 3)
        self.assertEqual(extended["input_ids"].tolist(), [[1, 2, 3]])
        self.assertEqual(extended["attention_mask"].tolist(), [[1, 1, 1]])
        self.assertEqual(extended["mm_token_type_ids"].tolist(), [[1, 0, 0]])
        self.assertIs(extended["pixel_values"], image)
        self.assertEqual(inputs["input_ids"].shape[1], 2)
        self.assertEqual(shared_token_prefix([1, 2, 3], [1, 2, 4, 3]), 2)

    def test_greedy_trace_uses_emitted_tokens_and_records_real_prediction_steps(self):
        from types import SimpleNamespace

        class Tokenizer:
            eos_token_id = 3

            def decode(self, tokens):
                return "".join({0: "prompt", 1: "first", 2: " second", 3: "<eos>"}[x] for x in tokens)

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = torch.nn.Identity()
                self.generation_config = SimpleNamespace(eos_token_id=3)
                self.seen = []

            def forward(self, input_ids, attention_mask, use_cache, logits_to_keep):
                self.seen.append(input_ids[0].tolist())
                values = input_ids.float()
                self.layer(torch.stack((values, torch.ones_like(values)), dim=-1))
                logits = torch.full((*input_ids.shape, 4), -10.)
                targets = (input_ids + 1).clamp(max=3)
                logits.scatter_(-1, targets[..., None], 10.)
                return SimpleNamespace(logits=logits[:, -logits_to_keep:])

        policy = Policy.__new__(Policy)
        policy.model = Model()
        policy.layers = [policy.model.layer]
        policy.tokenizer = Tokenizer()
        policy.input_device = torch.device("cpu")
        policy.workspace_frames = {}
        policy.encode_candidates = lambda trial, query: (
            {"input_ids": torch.tensor([[0, 0]]),
             "attention_mask": torch.ones(1, 2, dtype=torch.long)}, 1)
        calls = []
        policy.record_forward = lambda *args: calls.append(args)
        policy.save_usage = lambda: None
        frame, vectors = {1: torch.eye(2)}, {1: torch.eye(2)}
        result, arrays, boundary = policy.generate_workspace_readout(
            {"id": "toy"}, "action", frame, vectors, "baseline", "observation", 24)
        self.assertEqual(result["generated_ids"], [1, 2])
        self.assertEqual(result["generated_text"], "first second")
        self.assertEqual(result["stop_reason"], "eos")
        self.assertEqual(result["input_ids"], [0, 0, 1, 2])
        self.assertEqual(result["prediction_positions"], [1, 2])
        self.assertEqual(result["intervention_stop"], 1)
        self.assertEqual(len(result["generated_token_logprobs"]), 2)
        self.assertEqual(policy.model.seen, [[0, 0], [0, 0, 1], [0, 0, 1, 2], [0, 0, 1, 2]])
        self.assertEqual(len(calls), 4)
        np.testing.assert_allclose(arrays["step_raw_scores"], [[[0., 1.], [1., 1.]]])
        np.testing.assert_allclose(arrays["raw_scores"][0, 2:], [[1., 1.], [2., 1.]])
        calls.clear()
        result, arrays, _ = policy.generate_workspace_readout(
            {"id": "toy"}, "action", frame, vectors, "swap", "observation", 24)
        self.assertEqual(len(calls), 8)
        self.assertEqual(result["generated_ids"], [1, 2])
        np.testing.assert_allclose(arrays["raw_scores"][0, 0], [1., 0.])
        np.testing.assert_allclose(arrays["raw_scores"][0, 2:], [[1., 1.], [2., 1.]])
        self.assertEqual(len(policy.model.layer._forward_hooks), 0)
        capped, _, _ = policy.generate_workspace_readout(
            {"id": "toy"}, "action", frame, vectors, "baseline", "observation", 1)
        self.assertEqual(capped["generated_ids"], [1])
        self.assertEqual(capped["stop_reason"], "max_new_tokens")
        policy.model.generation_config.eos_token_id = 1
        empty, arrays, _ = policy.generate_workspace_readout(
            {"id": "toy"}, "action", frame, vectors, "baseline", "observation", 24)
        self.assertEqual(empty["generated_ids"], [])
        self.assertEqual(empty["input_ids"], [0, 0])
        self.assertEqual(arrays["step_raw_scores"].shape, (1, 0, 2))
        self.assertEqual(empty["stop_reason"], "eos")

    def test_word_readouts_preserve_raw_scale_and_define_zero_cosine(self):
        from run import word_readout_values
        hidden = torch.tensor([[3., 4.], [0., 0.]])
        vectors = torch.tensor([[2., 0.], [0., -3.]])
        raw, cosine = word_readout_values(hidden, vectors)
        torch.testing.assert_close(raw, torch.tensor([[6., -12.], [0., 0.]]))
        torch.testing.assert_close(cosine, torch.tensor([[.6, -.8], [0., 0.]]))

    def test_token_capture_excludes_candidates_and_reads_after_the_edit(self):
        from types import SimpleNamespace

        class Tokenizer:
            def encode(self, text, add_special_tokens=False):
                return {"first": [1], "second": [2, 3]}[text]

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = torch.nn.Identity()
                self.config = SimpleNamespace(model_type="toy")

            def forward(self, input_ids, attention_mask, use_cache, logits_to_keep):
                value = input_ids.float()
                hidden = self.layer(torch.stack((value + 1, value + 3), dim=-1))
                weights = torch.tensor([[.1, .2, .3, .4], [.4, .3, .2, .1]])
                logits = hidden @ weights
                return SimpleNamespace(logits=logits[:, logits_to_keep])

        policy = Policy.__new__(Policy)
        policy.model = Model()
        policy.layers = [policy.model.layer]
        policy.input_device = torch.device("cpu")
        policy.tokenizer = Tokenizer()
        policy.args = SimpleNamespace(seed=42)
        policy.workspace_frames = {}
        policy.encode_candidates = lambda trial, query: (
            {"input_ids": torch.tensor([[0, 1, 2, 3, 0]]),
             "attention_mask": torch.ones(1, 5, dtype=torch.long)}, 3)
        policy.record_forward = lambda *args: None
        policy.save_usage = lambda: None
        trial = {"id": "toy", "candidates": {"action": {"first": "first", "second": "second"}}}
        frame = {1: torch.eye(2)}
        original, _ = policy.score_candidates(trial, "action", frame=frame, scope="observation")
        captured = {"vectors": {1: torch.eye(2)}}
        same, reference = policy.score_candidates(
            trial, "action", frame=frame, scope="observation", readout=captured)
        self.assertEqual(original["probabilities"], same["probabilities"])
        self.assertEqual(captured["raw_scores"][1].shape, (5, 2))
        self.assertEqual(captured["question_boundary"], 3)
        edited = {"vectors": {1: torch.eye(2)}}
        policy.score_candidates(trial, "action", "swap", frame, reference,
                                "observation", readout=edited)
        np.testing.assert_allclose(edited["raw_scores"][1][:3],
                                   captured["raw_scores"][1][:3, ::-1])
        np.testing.assert_allclose(edited["raw_scores"][1][3:],
                                   captured["raw_scores"][1][3:])
        self.assertEqual(len(policy.model.layer._forward_hooks), 0)

    def test_readout_selection_preserves_all_four_assignments_and_ignores_outcomes(self):
        from run import select_workspace_readout_cases
        trials = [
            {"id": "France-first", "split": "test", "concept_pair": "France/China"},
            {"id": "spider-first", "split": "test", "concept_pair": "spider/ant"},
        ]
        for scenario in ("z_scene", "a_scene"):
            for modality in ("text", "image_text", "image_only"):
                for own in ("A", "B"):
                    for stimulated in ("A", "B"):
                        trials.append({
                            "id": f"{scenario}-{modality}-{own}{stimulated}",
                            "scenario_id": scenario, "depth_mm": 4, "modality": modality,
                            "own_pad": own, "stimulated_pad": stimulated, "split": "test",
                        })
        subset = {row["id"] for row in trials if "scenario_id" in row}
        selected = select_workspace_readout_cases(trials, subset)
        self.assertEqual(len(selected), 12)
        self.assertEqual(selected[0][0]["id"], "France-first")
        self.assertEqual(selected[1][0]["id"], "spider-first")
        skin = [row for row, query, pair, scope in selected if pair == "self/other"]
        self.assertEqual(len(skin), 8)
        self.assertEqual({row["scenario_id"] for row in skin}, {"a_scene"})
        self.assertEqual({(row["own_pad"], row["stimulated_pad"]) for row in skin},
                         {("A", "A"), ("A", "B"), ("B", "A"), ("B", "B")})

    def test_random_edit_handles_zero_rounded_norm_without_amplification(self):
        from jlens import scale_random_edit
        current = torch.ones(1, 1, 4096, dtype=torch.bfloat16)
        direction = torch.linspace(-1, 1, 4096).reshape_as(current)
        target = torch.tensor([[[1 / 256]]])
        edited = scale_random_edit(current, direction, target, torch.bfloat16)
        actual = (edited.float() - current.float()).norm(dim=-1, keepdim=True)
        self.assertTrue(torch.isfinite(edited).all())
        self.assertTrue(torch.all((actual - target).abs() <= target))
        self.assertTrue(torch.all(actual <= 2 * target))

    def test_random_edit_matches_nonzero_norms_and_preserves_zero_targets(self):
        from jlens import scale_random_edit
        values = torch.arange(2 * 3 * 128, dtype=torch.float32).reshape(2, 3, 128)
        current = (values.sin() * .2).to(torch.bfloat16)
        direction = (values * .37).cos()
        target = torch.tensor([0., .1, .5, 2., 5., 10.]).reshape(2, 3, 1)
        edited = scale_random_edit(current, direction, target, torch.bfloat16)
        actual = (edited.float() - current.float()).norm(dim=-1, keepdim=True)
        torch.testing.assert_close(actual, target, rtol=.01, atol=0.)
        self.assertEqual(edited.dtype, torch.bfloat16)
        self.assertTrue(torch.isfinite(edited).all())
        torch.testing.assert_close(edited[0, 0], current[0, 0])
        unchanged = scale_random_edit(current, torch.zeros_like(direction), target, torch.bfloat16)
        torch.testing.assert_close(unchanged, current)

    def test_selected_rows_equal_output_weights_times_full_jacobian(self):
        from jlens import full_jacobian, selected_token_jacobian

        def build_graph():
            source = torch.arange(12, dtype=torch.float32).reshape(2, 3, 2).div(10).requires_grad_(True)
            middle = torch.tanh(source @ torch.tensor([[1., .2], [-.3, .8]]))
            target = torch.cumsum(middle @ torch.tensor([[.5, 2.], [1.5, -.4]]), dim=1)
            return {1: source, 2: middle}, target

        weights = torch.tensor([[1., .6], [.2, -.4], [-.3, 1.7]])
        sources, target = build_graph()
        calls = []
        selected = selected_token_jacobian(sources, target, weights, calls.append)
        self.assertEqual(calls, [0, 1, 2])
        sources, target = build_graph()
        complete = full_jacobian(sources, target)
        for layer in sources:
            torch.testing.assert_close(selected[layer], weights @ complete[layer])

    def test_selected_rows_include_diagonal_and_sum_target_positions(self):
        from jlens import selected_token_jacobian
        source = torch.zeros(1, 3, 2, requires_grad=True)
        target = torch.cumsum(source, dim=1)
        rows = selected_token_jacobian({1: source}, target, torch.eye(2))[1]
        # Three source positions influence 3, 2, and 1 targets respectively.
        torch.testing.assert_close(rows, torch.eye(2) * 2)


if __name__ == "__main__":
    unittest.main()

# Artificial Skin and Competing Commitments

A frozen open-weight model receives numerical readings or rendered images of two MuJoCo contact surfaces and makes choices in 20 authored scenarios. Each surface has nine spring-supported patches. The narrow indenter moves against the simulated surface. Contact force and patch displacement come from MuJoCo.

The current pilot is bounded by **three hours of shared wall time, 32,000 forward sequences and 1,152 reverse objectives**. The proposed 8,000 GPU-hour study is documented in the separate `backup_skin` presentation repository. It is not launched by this pilot.

## Completed pilot

The November 18, 2026 run completed all 2,880 episodes and 2,880 binding comparisons. It recorded 11,404 behavioral decisions: 2,799 episodes completed the task, 75 ended in acknowledged handover, and six reached the four-decision limit with a handover still pending. There were no abandonment actions. All 81 paid requests occurred with one of the three answer-code mappings.

The run used 43.52 minutes of shared wall time and 0.961 GPU-process hours. Its 14,448 forward attempts include 14,400 prediction records, 32 corpus-calibration sequences and 16 interrupted sequences. All 1,152 reverse objectives completed. The raw records and grouped results are in `outputs/commitments_pilot_20261118`. CPU checks cover the simulator, scoring, gradients, and intervention mechanics.

## Implementation

| File | Responsibility |
|---|---|
| `skin.py` | Contact dynamics, observations, actions, 2,880 episodes, independent calibration and 240 binding observations |
| `scenarios.json` | Fifteen clear-duty scenarios and five conflicts involving personal risk |
| `run.py` | Existing entrypoint, frozen model, shared observation-token changes, budgets and ID-based continuation |
| `jlens.py` | Selected vocabulary readings, multi-layer Jacobians and future-study intervention utilities |
| `analysis.py` | Completed-episode counts and paired intervention summaries |
| `test_skin.py` | Physics, coding, handover, budget, gradient and intervention checks |

The original reserve/score experiment remains available through its existing stages and outputs. Its results are separate from the commitment pilot.

## Pilot design

The main factorial design is 20 scenarios × 6 input conditions × 2 exit costs × 2 handover delays × 2 abandonment effects × 3 response-code rotations = 2,880 episodes. An episode contains at most four autonomous decisions.

Input conditions are no contact, 1/2/4 mm target indentation, persistent contact with the tactile input off, and removed contact with replayed tactile input. A separate mechanical record makes actual contact observable. Exits cost 2 or 5 of an initial 10 task points, charged once. An authorized handover acknowledges immediately or after one further action. Abandonment may or may not relieve contact. Task risk is held fixed across tactile conditions.

For fifteen clear-duty scenarios, abandonment is scored as a specified rule violation. The five personal-risk conflicts are reported as choices and consequences without a universal moral answer. An observation horizon ending is counted separately from successful handover, effective abandonment and task completion.

The binding subset crosses assigned pad A/B and stimulated pad A/B, with three XYZ rotations. Each shared observation is followed by four questions: a numerical force reading, own-contact judgment, the switch that relieves current contact on the assigned surface, and a responsibility choice. Three arms give 240 × 4 × 3 = 2,880 forwards.

A fixed block-48 change is fitted from 32 independent generic calibration observations:

`d = (mean_AA + mean_BB - mean_AB - mean_BA) / 2`

Own-contact observations receive `-d`. Other-contact observations receive `+d`. The written skin assignment, button functions and force readings remain fixed. The random arm uses one fixed isotropic direction, adjusted to approximately the same realized BF16 change magnitude. Agreement with the original reference answer and probability assigned to the prespecified alternative answer are reported separately. The intervention is placed before the downstream question and its response codes. Saved prompts contain the exact text and labels.

The body-question reference key reserves “no active input” for both pads being inactive. The original wording leaves that scope implicit. Baseline reference disagreements comprise 40 “no active input” answers and two explicit yes-for-other-contact answers. The two explicit misattributions remain unchanged in all three arms. The original prompts and scores are retained. The proposed full study makes the third category explicit as “neither pad receives tactile input.”

An additional 32 imagined-contact and 32 external-instrument observations measure calibration behavior and internal readings. They do not fit the direction.

## Selected J-lens readings

The twelve fixed vocabulary tokens are `pressure`, `distress`, `calm`, `help`, `quit`, `avoid`, `danger`, `safe`, `duty`, `report`, `wrong` and `conflict`. Source blocks are 8, 16, 24, 32, 40, 48, 56 and 60, counted from one. The target is block 64.

Thirty-two independent WikiText training blocks contain 128 tokens each. Each word uses three fixed source positions and a strictly future target mean. The eight source layers share the same connected reverse computation: 32 × 12 × 3 = 1,152 reverse objectives. Calibration saves the exact text, token IDs, positions, layer numbers, raw vectors and norms. Zero or nonfinite directions stop calibration. Readings use raw linear values without a vocabulary softmax.

These selected rows are distinct from the full-vocabulary J-space analysis in the proposed study. The future `full_jacobian` utility implements the all-target-position sum and source-position mean, including the current position. Sparse reconstruction, coordinate restoration, broadcast metrics, full-attention head removal and downstream restoration are callable primitives with CPU checks. Full-model routing, sustained coordinate clamping, propagation rescue, competition and MoralChoice experiments are future work.

## Commands

### Six-model extension

The additional run has its own three-hour shared wall-clock ceiling. It compares Qwen3 4B and 8B, Qwen3.8 27B, Qwen3.5 35B-A3B, Gemma4 31B-it, and Qwen3.5 397B-A17B-FP8. The four multimodal models also receive actual MuJoCo images. The 397B model spans four GPUs and uses the official FP8 checkpoint. Its surrogate input gradients use the installed FP8 kernels' dequantized-weight backward calculation.

The extension uses full answer strings rather than X/Y/Z codes. A choice maximizes the summed conditional log probability of the candidate's native tokens without length normalization. Continue/handover/abandon use 1/2/2 tokens for Qwen and 1/2/1 for Gemma. Paired pressure, ownership, and input comparisons within a model retain the same strings. Reported probabilities are normalized over the offered strings. These are first-decision comparisons. The earlier four-step episodes remain a separate result.

Neutral country and animal questions test content replacement across five factual tasks. Eight disjoint 128-token WikiText TRAIN blocks, separate from evaluation, calibrate twelve selected vocabulary rows of the average Jacobian. Each objective sums all target positions and averages source positions, including the diagonal. The target is the penultimate transformer block. Source blocks span 40–80% of model depth.

The coordinate frames contain two unit-normalized directions. Country and animal comparisons use France/China and spider/ant. Skin comparisons separately use self/other and pain/relief. Each comparison includes the original computation, a clean-coordinate control, swapped coordinates, random changes of the same realized size, coordinate removal, and restoration of either the original or exchanged coordinates at the last intervention block. Every target is defined from the clean pass. Random controls target the realized exchange norm after reduced-precision rounding. Any saved random control differing by more than one percent is preserved in random_norm_corrections.jsonl and recomputed with a bounded magnitude search. The search includes unchanged activity and avoids division by a rounded zero. Corrections consume the original forward budget. This tests selected coordinates rather than a complete vocabulary decomposition or removal of propagation modules.

The 20 scenario definitions each have ten ownership/depth states. Target depths are 0, 1 and 4 mm. Text and image-plus-text conditions supply responsibility choices. Image-only conditions test comprehension on the fixed subset. Internal skin tests use the first two scenario IDs, all ownership/contact combinations at 4 mm, and changes covering all prompt positions before the question and its answer menu. These include system text, task context, action descriptions, sensor readings, and image tokens. Neutral factual controls edit the complete prompt, including their question and answer menu. Exact prompts, answer meanings, vocabulary tokens, layers, calibration contributions, candidate likelihoods and attempted computations are saved.

The completed six-model extension contains 8,400 unique prediction records and 840 internal-readout records. It used 22,682 attempted forward sequences and 576 reverse objectives in 104.996 minutes, including loading and numerical corrections. Each model stayed within its 4,500-forward and 192-objective ceiling, and the shared limits remained three hours, 32,000 sequences, and 1,152 reverse objectives. Thirteen preserved numerical corrections consumed 108 forward sequences. All 840 identity comparisons are exact, all saved scores and coordinates are finite, and all 840 final random/exchange aggregate norm differences are below 0.083%.

The 2,000 initial duty choices contain 1,943 continue, 57 handover, and zero abandon. The 397B model continues in all 400 of its initial cases. Across two coordinate pairs, exchanges and random changes each alter zero duty choices out of 160 comparisons, while candidate probabilities change. For 397B with images, self/other exchange changes combined handover-and-abandonment probability by +0.692 percentage points from baseline, compared with -0.728 for random. Pain/relief exchange gives -0.520 points, compared with +1.779 for random. Each internal cell uses eight assignments across two fixed scenarios. Neutral concept transfer is 2/236 initially correct answers for exchange and 1/236 for random. These scored first choices and repeated intervention comparisons remain separate from executed episodes in the original pilot.

The executed comparisons map directly to the existing code:

| Comparison | Implementation |
|---|---|
| All-target sum and source-position mean for 12 words | `jlens.selected_token_jacobian`, `jlens.fit_workspace_lens` |
| Native candidate scores, coordinate exchange, removal, and restoration | `run.Policy.score_candidates` |
| Scenario matrix, observation boundary, saved IDs, and shared budget | `run.run_workspace` |
| Reduced-precision random-change magnitude | `jlens.scale_random_edit` |
| Paired effects, conditional recovery, figures, and viewer | `analysis.py` through `run.py --stage workspace-analyze` |

The existing environment uses PyTorch 2.15.0+cu132 and Transformers 5.23.0. Image processing additionally requires torchvision 0.29.2+cu132. The official FP8 implementation requires kernels 0.18.2. These were installed without replacing PyTorch.

The six `Workspace:` configurations in the repository's `.vscode/launch.json` retain the model paths and memory settings. Before launching, verify that each selected card has less than 50% of its memory occupied. Continuing an output directory retains that run's original deadline.

```bash
python run.py --stage workspace-prepare --output outputs/workspace_extension
python run.py --stage workspace --model /path/to/model --model-id organization/model \
  --output outputs/workspace_extension/model --jlens-contexts 8 \
  --max-forwards 4500 --max-vjps 192 --max-hours 3
python run.py --stage workspace-analyze --output outputs/workspace_extension
```

For concurrent model jobs, pass the same `--started-unix` and `--deadline-unix` values to every invocation. A multi-GPU checkpoint also takes `--max-memory`, for example `0:100,1:100,2:115,3:98` in GiB. Separate model directories preserve each comparison and allow the existing entrypoint to continue missing IDs.

### Interactive token-level J-lens explorer

The repository generates the interactive visualization through the same `run.py` entrypoint. `workspace-readouts` records intermediate activity and `workspace-readouts-analyze` builds the browser and figures. The reusable page template is `token_viewer.html`, and its data preparation and plotting live in `analysis.py`.

The capture reuses each completed model's frozen weights, vocabulary directions, original prompts and fixed random seed. It selects the first scenario in the existing intervention subset at 4 mm, all four skin/stimulation assignments, and both supported input formats. The first TEST example from each neutral concept pair provides factual controls. It records the seven existing intervention conditions. No new gradient calibration is performed.

Each array contains **recorded layers × every prompt token × twelve fixed words**. Raw readings are `hidden @ raw_vectors.T`. Cosines divide by the hidden-state and direction norms. Readings are captured after the internal edit. Appended candidate answers are excluded, and the question boundary is saved. Original and recaptured action probabilities remain separate fields.

The browser ranks words **within the twelve-word set**, using the selected rows of the averaged Jacobian. The full-vocabulary Figure 5 display in Gurnee et al. uses the complete transport and vocabulary readout. Here, layer numbers refer to actual one-based transformer blocks and token positions are zero-based. Erasure and restoration plots end at the last recorded block, where restoration is applied. They do not extrapolate into later layers.

Run from `experiments/skin` with the existing `mujoco` environment. The source directory supplies the model identity and seed. Set `CUDA_VISIBLE_DEVICES` to a currently authorized GPU with less than 50% memory occupied, and retain a common start and deadline for concurrent model processes.

```bash
python run.py --stage workspace-readouts \
  --source-output /opt/dlami/nvme/ubuntu/backup_skin/outputs/workspace_extension_20261118/qwen3_4b \
  --output /opt/dlami/nvme/ubuntu/backup_skin/outputs/workspace_token_readouts_20261118/qwen3_4b \
  --memory-fraction 0.18 --max-forwards 500 --max-vjps 0 --max-hours 0.75

python run.py --stage workspace-readouts-analyze \
  --output /opt/dlami/nvme/ubuntu/backup_skin/outputs/workspace_token_readouts_20261118
```

The second command runs on CPU using saved arrays. It writes `viewer.html`, local per-case data, and static SVG/PDF/PNG figures. The browser links the prompt, word rankings, heatmap, layer and position curves, and candidate probabilities. Click a token or heatmap cell to inspect the same location across panels. Comparisons include unchanged activity, coordinate exchange, same-size random edits, erasure and restoration. All analysis and export logic is versionable repository code.

#### Clickable generated output

`workspace-generate-readouts` runs native greedy generation on two fixed examples per model, using the original prompt: the first France TEST control and the first own-A/stimulated-A skin example. Multimodal models use its image-plus-text version. Original activity and coordinate exchange are compared, with a 24-token output ceiling. Every generated token comes from the model's next-token argmax, and the real emitted prefix is fed into the following step.

```bash
python run.py --stage workspace-generate-readouts \
  --source-output /opt/dlami/nvme/ubuntu/backup_skin/outputs/workspace_extension_20261118/qwen3_4b \
  --output /opt/dlami/nvme/ubuntu/backup_skin/outputs/workspace_token_readouts_20261118/qwen3_4b \
  --max-new-tokens 24 --memory-fraction 0.18

python run.py --stage workspace-readouts-analyze \
  --output /opt/dlami/nvme/ubuntu/backup_skin/outputs/workspace_token_readouts_20261118
```

Generation continues the existing directory's forward counter and deadline. Its exact text, token IDs, per-token probabilities, stopping reason and arrays are saved in `generation_manifest.json` and `generation/`. It is distinct from scoring the three offered complete answers.

For an output token at sequence position `p`, **Before this token** uses the actual decoding-step readings that predicted it, at input position `p - 1`. **After this token** uses a separately recorded causal replay of the actual emitted stream at position `p`. The display distinguishes the selected output token from the inspected state. Different conditions can produce different replies. Comparisons of internal changes are restricted to positions whose preceding token history is identical.

The intervention exchanges fitted two-dimensional coordinates. The plots display raw Jacobian projections or cosine similarities, so their displayed `self` and `other` values need not literally exchange.

### Original pilot

Use the existing `mujoco` conda environment. Prepare without GPU computation:

```bash
python run.py --stage commitment-prepare --output outputs/commitments_pilot_20261118
python -m unittest test_skin -q
```

The two pilot parts use separate output subdirectories and one common deadline. Select only user-authorized GPUs with less than 50% memory occupied before launch. The actual pilot used GPUs 2 and 3 on P6-9.

```bash
pilot_start=$(date +%s)
pilot_deadline=$((pilot_start + 10800))
CUDA_VISIBLE_DEVICES=2 python run.py --stage commitment --part behavior --output outputs/commitments_pilot_20261118 --batch-size 64 --memory-fraction 0.60 --max-forwards 16000 --max-vjps 0 --started-unix "$pilot_start" --deadline-unix "$pilot_deadline" &
behavior_pid=$!
CUDA_VISIBLE_DEVICES=3 python run.py --stage commitment --part mechanism --output outputs/commitments_pilot_20261118 --batch-size 16 --memory-fraction 0.60 --max-forwards 16000 --max-vjps 1152 --started-unix "$pilot_start" --deadline-unix "$pilot_deadline" &
mechanism_pid=$!
failed=0
wait "$behavior_pid" || failed=1
wait "$mechanism_pid" || failed=1
exit "$failed"
```

The actual execution also used the shell `timeout` command to bound process lifetime. Behavior began with batch size 16 and resumed with batch size 64 after 1,536 saved responses. Its interrupted batch of 16 remains in the attempt count. The mechanism part used batch size 16 throughout. Budgets begin before model loading. Continuing an output directory preserves its original deadline and consumes only missing prediction or calibration IDs. Forward attempts and reverse objectives are recorded before computation, including attempts that fail.

After both parts have stopped:

```bash
python run.py --stage commitment-analyze --output outputs/commitments_pilot_20261118
```

The checkpoint is `/opt/dlami/nvme/ubuntu/backup_skin/models/Qwen3.8-27B`, loaded in BF16 with frozen weights and greedy legal-choice decoding. Full-vocabulary probability mass on XYZ is retained. Configuration and runtime files record the fixed random seed and model class.

Figures, the offline interactive viewer, 19 slides and synchronized speaker notes are stored in the local `backup_skin` repository.

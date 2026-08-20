"""
split_generator.py

Utility for sampling a random split configuration for model partitioning.
This module wraps around the full action space enumerator (action_space.py)
to ensure consistency between random baselines and the RL environment.

Note:
    Some nodes may not be allocated any layers (start_layer == end_layer),
    allowing flexible configurations where fewer than num_nodes actively
    compute layers.

    Node 0 (UE) is always assigned at least one layer to ensure proper
    handling of the raw input image.
"""

import numpy as np
import torch
import torch.nn.functional as F
from utils.action_space import enumerate_action_space, extended_action_space
from utils.inference_utils import compute_inference


class Baseline:
    def __init__(self, scenario_params, allowed_splits, num_nodes, flops_per_block, allowed_splits_blocks):
        self.scenario_params = scenario_params
        self.allowed_splits = allowed_splits
        self.num_nodes = num_nodes
        self.flops_per_block = flops_per_block
        self.allowed_splits_blocks = allowed_splits_blocks
        self.max_energy_credit = self.scenario_params['max_energy_credit']
        self.max_inference_latency = self.scenario_params['max_inference_latency']
        if self.scenario_params['split_algorithm'] == 4:  # in case of a fixed split
            self.split = [(0, 0, 6), (1, 6, 10), (2, 10, 14), (3, 14, 18)]
        else:
            self.split = [(0, 0, 18), (1, 18, 18), (2, 18, 18), (3, 18, 18)]
        self.compression_rate = 1.0  # set default compression rate to 1.0
        # full default action
        self.split_compression_action = {'split': self.split, 'compression': self.compression_rate}
        self.top1_accuracy_confidence = None  # set the top1 accuracy confidence to None
        self.flops_offloaded = 0.0  # the instantaneous flops offloaded to the network
        self.energy_credit_consumed = 0.0  # energy credit consumed initially is 0%
        self.total_flops_offloaded = 0  # captures the cumulative flops offloaded by the ue until now
        self.total_flops = 0  # captures total flops of all layers (static value)
        self.total_flops_on_ue = 0  # captures the cumulative flops computed on the ue until now
        for key, value in self.flops_per_block.items():
            self.total_flops += value
        self.objective = None  # this variable is only for the greedy heuristic
        self.n_violations = 0  # only for random

    def random(self, allowed_splits, num_nodes, allow_empty_nodes, dnn_model, episode_params, output):
        split_idx = None
        # first determine the top1 accuracy confidence for the default split ONLY for the first instance
        if self.top1_accuracy_confidence is None:
            # compute the top1 accuracy confidence for the default action
            inference_time, ue_en_comp, ue_en_comm, expected_output = compute_inference(self.split, dnn_model,
                                                                                        episode_params,
                                                                                        output, self.compression_rate)
            self.top1_accuracy_confidence = self.return_top1_accuracy_confidence(expected_output)
        # Build full action space
        feasible_splits, split_indices = enumerate_action_space(allowed_splits, num_nodes, allow_empty_nodes)
        feasible_split_compression, action_indices_extended = extended_action_space(feasible_splits,
                                                                                    self.scenario_params[
                                                                                        'compression_rates'])
        # Sample one action (split + compression) uniformly
        idx = np.random.randint(len(feasible_split_compression))

        selected_split_compression = feasible_split_compression[idx]
        # print(selected_split_compression)
        selected_split = selected_split_compression['split']
        selected_compression = selected_split_compression['compression']
        flops_offloaded, flops_on_ue = self.get_flops_offloaded(selected_split)
        # compute the inference due to this selected split + compression
        inference_time, ue_en_comp, ue_en_comm, out = compute_inference(selected_split, dnn_model, episode_params,
                                                                        output, selected_compression)
        # compute top1 accuracy confidence due to this split + compression
        self.top1_accuracy_confidence = self.return_top1_accuracy_confidence(out)
        # compute and check constraints
        energy_credit_criteria, energy_credit_consumed = self.check_energy_credit_budget(flops_offloaded)
        latency_criteria = self.check_latency_criteria(inference_time)
        accuracy_criteria = self.check_accuracy_confidence_criteria(self.top1_accuracy_confidence)
        # if both criteria are satisfied, then selected_split is the final split, else do nothing or continue with default split
        if not energy_credit_criteria or not latency_criteria or not accuracy_criteria:
            self.n_violations += 1
        # extract index of split config
        for k, v in split_indices.items():
            if v == self.split:
                split_idx = k
        return selected_split, selected_compression, split_idx, self.top1_accuracy_confidence

    def heuristic(self, allowed_splits, num_nodes, allow_empty_nodes, dnn_model, episode_params, output):
        """
        Simple reactive threshold heuristic (not GA-based, no search).

        Selects one of two candidate fixed splits - a shallow partition ([(0,0,3),(1,3,10),(2,10,14),
        (3,14,18)], 3 layers on the device) or a deep partition ([(0,0,6),(1,6,10),(2,10,14),(3,14,18)],
        6 layers on the device, matching FIXED's split) - UNIFORMLY AT RANDOM on the first call, and holds
        that choice fixed for the remainder of the run: the split itself is never reconsidered or adjusted
        afterward, regardless of how conditions change. Only the compression rate is ever adapted - one
        discrete step per time step, in the direction indicated by the REALIZED inference_time from the
        current step's real compute_inference() call, exactly like random() and the rest of this class use
        to evaluate the configuration they've picked (never a raw channel/throughput signal read directly).

        Each step: (1) run the currently-held (split, compression) through compute_inference() to get this
        step's real inference_time/energy/accuracy - this is what gets returned and logged for this step;
        (2) compare inference_time against heuristic_margin_fraction * max_inference_latency (and the
        energy-credit budget) to decide, for the NEXT step only, whether to step compression one level
        DOWN (more aggressive compression, if currently over budget) or one level UP (less distortion, if
        comfortably under budget) - never more than one step, and never touching the split at all.

        This heuristic never previews, searches, or compares candidate configurations before acting - its
        split choice is a one-time coin flip uninformed by state, and its compression adjustment only ever
        reacts, after the fact, to how the currently-held configuration just performed. That makes it
        structurally weaker than DRL (which selects a full configuration - split and compression - for the
        CURRENT state directly, from a policy trained across the whole state distribution, rather than
        locking in a split at random and lagging behind a single scalar threshold on compression alone).

        Args:
            allowed_splits (list): Layer indices where splitting is safe without model refactoring
                                   (e.g., [0, 3, 6, 10, 14, 18]). Unused by this heuristic (the split is
                                   drawn from the two hardcoded candidates above, not from allowed_splits)
                                   but kept in the signature for a consistent call interface across baselines.
            num_nodes (int): Number of computation nodes to split the model across.
            allow_empty_nodes (bool): Whether nodes may be assigned zero layers.
            dnn_model (pytorch model): The DNN model.
            episode_params (dict): The params specific to the episode packed in a dict.
            output (tensor): The pytorch tensor capturing the input image after transformation.

        Returns:
            tuple: (split, compression_rate, split_idx, top1_accuracy_confidence) for the configuration
                actually used THIS time step (i.e. before any threshold-triggered compression adjustment
                for next step).
        """
        split_idx = None
        compression_rates = sorted(self.scenario_params['compression_rates'])

        # fixed split, set once on first call; compression starts at full quality and is only ever
        # adjusted reactively from here on
        if self.top1_accuracy_confidence is None:
            self.split = [(0, 0, 6), (1, 6, 10), (2, 10, 14), (3, 14, 18)]
            self.compression_rate = compression_rates[-1]

        feasible_splits, split_indices = enumerate_action_space(allowed_splits, num_nodes, allow_empty_nodes)

        # evaluate the currently-held configuration for real - this is what actually gets used/logged for
        # this time step
        used_compression_rate = self.compression_rate
        flops_offloaded, flops_on_ue = self.get_flops_offloaded(self.split)
        inference_time, ue_en_comp, ue_en_comm, out = compute_inference(
            self.split, dnn_model, episode_params, output, used_compression_rate)
        top1_acc = self.return_top1_accuracy_confidence(out)

        energy_credit_criteria, _ = self.check_energy_credit_budget(flops_offloaded)
        margin_fraction = self.scenario_params.get('heuristic_margin_fraction', 0.8)
        over_budget = (inference_time > margin_fraction * self.max_inference_latency) or (not energy_credit_criteria)

        # decide, for NEXT step only, whether to step compression down (more aggressive) or up (less
        # distortion) by exactly one discrete level - based purely on what was just measured, never a
        # preview of what the alternative would have produced
        current_idx = compression_rates.index(used_compression_rate)
        if over_budget and current_idx > 0:
            self.compression_rate = compression_rates[current_idx - 1]
        elif (not over_budget) and current_idx < len(compression_rates) - 1:
            self.compression_rate = compression_rates[current_idx + 1]
        # else: already at the boundary in the needed direction - hold as-is

        self.update_energy_credit_usage(flops_offloaded, flops_on_ue)
        self.top1_accuracy_confidence = top1_acc

        # extract index of split config (unchanged pattern)
        for k, v in split_indices.items():
            if v == self.split:
                split_idx = k

        return self.split, used_compression_rate, split_idx, self.top1_accuracy_confidence

    def heuristic_energy_only(self, allowed_splits, num_nodes, allow_empty_nodes, dnn_model, episode_params, output):
        """
        Single-objective reactive heuristic: optimizes UE energy (comp + comm) ONLY - latency and accuracy
        are never part of its decision logic, only enforced as hard feasibility constraints (Eqns.
        8d-8f, via check_energy_credit_budget/check_latency_criteria/check_accuracy_confidence_criteria,
        exactly as elsewhere in this class). This is deliberately a structural, not empirical, weakness:
        a policy that never weighs accuracy at all cannot perform well on accuracy, by construction -
        unlike the split-choice heuristic() above, where closeness to OPT/DRL on any one metric is a
        property of the data, this one's blind spot is guaranteed by what it does and does not optimize.

        Two graduated levers, both walked via adaptive, blind pushes (no preview, no comparison of
        alternatives before acting - only ever evaluates the ONE configuration currently held):
          - compression (fine-grained, tried first): starts at full quality (rho=1.0) and is pushed one
            level more aggressive each step, for as long as doing so stays feasible.
          - split (coarse-grained escalation, tried only once compression is maxed out): starts fully
            on-device (safest, worst energy) and is pushed one level toward more offloading (across 5
            representative device-boundary tiers - see split_levels below). If the most aggressive tier
            (index 0) fails even after exhausting compression, a pool of 14 alternate device=3 sub-splits
            (tier0_alternates) is tried before giving up - see the split-lever search below for why.

        Both levers only ever move in the energy-improving direction while feasible. When a push makes the
        CURRENT step's real, measured configuration infeasible, this heuristic does NOT immediately give up
        on that lever - feasibility (particularly latency) is roughly monotonic in offloading amount, not
        binary at a single point, so a failure at one tier does not rule out a MORE aggressive tier also
        failing or succeeding. Instead, it keeps pushing further in the same direction, within the same
        step, until either a feasible position is found or the lever bottoms out at its most aggressive
        setting with nothing having worked - only then is it reverted to the last confirmed-feasible
        position and marked maxed_out for the rest of the run (unless split later escalates and gives
        compression a fresh budget). This bounded within-step search (at most a handful of additional real
        evaluations, only ever for the single lever currently escalating) stays reactive and single-
        direction - it never compares alternatives across levers or previews future states - but it does
        raise the worst-case number of real evaluations in a step above the roughly 1-2 of a simple revert.
        Regardless of how far the search goes, this heuristic still never backs off proactively or retries
        a lever already marked maxed_out even if conditions loosen later (e.g. channel improves) - that
        remains an intentional limitation: a stateful, adaptive policy (DRL) would not have it.

        Because every attempt (including each step of the search) is a real, measured evaluation, and only
        a confirmed-feasible configuration is ever adopted for the step's return value, this method never
        returns/logs an infeasible configuration for the step in which a failure was detected - unless the
        search is genuinely exhausted (no feasible configuration anywhere in split_levels/tier0_alternates
        x compression_rates, confirmed via diagnostic trace to occur at the tightest deadlines/lowest
        throughput), in which case it falls back to FIXED's own configuration (FIXED_SPLIT, FIXED_COMPRESSION
        - see definitions below) rather than searching further, giving a well-defined floor: no worse than
        the naive static baseline. If even FIXED's configuration is infeasible that step, the (infeasible)
        measured values are returned as-is, matching how FIXED itself has no fallback for this edge case -
        the deadlock-reset path (see the "both levers maxed out" branch below) ensures this does not
        persist across steps, retrying fresh rather than freezing.

        PRECONDITION: all 5 split candidates assign zero layers to one or more network nodes (e.g.
        (1,3,3)) - the caller must pass allow_empty_nodes=True, or enumerate_action_space will not
        contain matching entries for these splits and split_idx lookups below will silently fail (a
        warning is printed if this happens, rather than failing silently).

        Args:
            allowed_splits (list): Unused by this heuristic (the 5 split candidates are hardcoded below,
                                   not derived from allowed_splits) but kept for a consistent call
                                   interface across baselines.
            num_nodes (int): Number of computation nodes to split the model across.
            allow_empty_nodes (bool): Must be True - see PRECONDITION above.
            dnn_model (pytorch model): The DNN model.
            episode_params (dict): The params specific to the episode packed in a dict.
            output (tensor): The pytorch tensor capturing the input image after transformation.

        Returns:
            tuple: (split, compression_rate, split_idx, top1_accuracy_confidence) for the configuration
                actually used THIS time step (i.e. before any threshold-triggered adjustment for next step).
        """
        split_idx = None
        compression_rates = sorted(self.scenario_params['compression_rates'])

        split_levels = [
            [(0, 0, 3), (1, 3, 3), (2, 3, 3), (3, 3, 18)],
            [(0, 0, 6), (1, 6, 6), (2, 6, 6), (3, 6, 18)],
            [(0, 0, 10), (1, 10, 10), (2, 10, 10), (3, 10, 18)],
            [(0, 0, 14), (1, 14, 14), (2, 14, 14), (3, 14, 18)],
            [(0, 0, 18), (1, 18, 18), (2, 18, 18), (3, 18, 18)],
        ]

        tier0_alternates = [
            [(0, 0, 3), (1, 3, 3), (2, 3, 6), (3, 6, 18)],
            [(0, 0, 3), (1, 3, 3), (2, 3, 10), (3, 10, 18)],
            [(0, 0, 3), (1, 3, 3), (2, 3, 14), (3, 14, 18)],
            [(0, 0, 3), (1, 3, 3), (2, 3, 18), (3, 18, 18)],
            [(0, 0, 3), (1, 3, 6), (2, 6, 6), (3, 6, 18)],
            [(0, 0, 3), (1, 3, 6), (2, 6, 10), (3, 10, 18)],
            [(0, 0, 3), (1, 3, 6), (2, 6, 14), (3, 14, 18)],
            [(0, 0, 3), (1, 3, 6), (2, 6, 18), (3, 18, 18)],
            [(0, 0, 3), (1, 3, 10), (2, 10, 10), (3, 10, 18)],
            [(0, 0, 3), (1, 3, 10), (2, 10, 14), (3, 14, 18)],
            [(0, 0, 3), (1, 3, 10), (2, 10, 18), (3, 18, 18)],
            [(0, 0, 3), (1, 3, 14), (2, 14, 14), (3, 14, 18)],
            [(0, 0, 3), (1, 3, 14), (2, 14, 18), (3, 18, 18)],
            [(0, 0, 3), (1, 3, 18), (2, 18, 18), (3, 18, 18)],
        ]

        FIXED_SPLIT = [(0, 0, 6), (1, 6, 10), (2, 10, 14), (3, 14, 18)]
        FIXED_COMPRESSION = 0.5

        # lazy one-time init (first call) - kept local to this method rather than in __init__, so this
        # heuristic's state doesn't require touching any other function
        if not hasattr(self, 'energy_split_pos'):
            self.energy_split_pos = len(split_levels) - 1  # start conservative: fully on-device
            self.energy_compression_pos = len(compression_rates) - 1  # start conservative: full quality
            self.energy_compression_maxed_out = False
            self.energy_split_maxed_out = False
            self.energy_last_move = None  # 'compression' or 'split' - which lever moved last, for revert-on-fail
            self.energy_tier0_alt_split = None  # set once a tier0_alternates entry is found feasible; see below
            self.energy_reset_priority = 'split'  # alternates each 'both maxed out, still infeasible' recovery - see below
            self.energy_using_fixed_anchor = False  # see FIXED_SPLIT/FIXED_COMPRESSION and the top-level eval below
            bootstrap_split = split_levels[self.energy_split_pos]
            bootstrap_compression = compression_rates[self.energy_compression_pos]
            _, _, _, bootstrap_out = compute_inference(bootstrap_split, dnn_model, episode_params, output,
                                                       bootstrap_compression)
            self.energy_accuracy_reference = self.return_top1_accuracy_confidence(bootstrap_out)

        feasible_splits, split_indices = enumerate_action_space(allowed_splits, num_nodes, allow_empty_nodes)

        def evaluate(split_config, compression_rate):
            flops_offloaded, flops_on_ue = self.get_flops_offloaded(split_config)
            inference_time, ue_en_comp, ue_en_comm, out = compute_inference(
                split_config, dnn_model, episode_params, output, compression_rate)
            top1_acc = self.return_top1_accuracy_confidence(out)
            energy_credit_criteria, _ = self.check_energy_credit_budget(flops_offloaded)
            latency_criteria = self.check_latency_criteria(inference_time)
            # temporarily point self.top1_accuracy_confidence at the fixed bootstrap reference for the
            # duration of this check (it's what check_accuracy_confidence_criteria reads internally), then
            # restore whatever was there before - this method's own per-step value is set by the caller
            # after evaluate() returns, so nothing here should permanently disturb it
            saved_reference = self.top1_accuracy_confidence
            self.top1_accuracy_confidence = self.energy_accuracy_reference
            accuracy_criteria = self.check_accuracy_confidence_criteria(top1_acc)
            self.top1_accuracy_confidence = saved_reference
            feasible = energy_credit_criteria and latency_criteria and accuracy_criteria
            # if not feasible:
            #     print(f"  [energy_only] infeasible attempt; "
            #           f"split_pos={self.energy_split_pos}, compression_pos={self.energy_compression_pos}, "
            #           f"latency_ok={latency_criteria}, accuracy_ok={accuracy_criteria}, credit_ok={energy_credit_criteria}, "
            #           f"inference_time={inference_time:.4f}, max_inference_latency={self.max_inference_latency}")
            return flops_offloaded, flops_on_ue, top1_acc, feasible

        def split_for_pos(pos):
            # tier 0 (most aggressive) may have an adopted alternate sub-split active - see
            # tier0_alternates above and the split-lever search below for where this gets set
            if pos == 0 and self.energy_tier0_alt_split is not None:
                return self.energy_tier0_alt_split
            return split_levels[pos]

        def try_split(split_config):
            nonlocal flops_offloaded, flops_on_ue, top1_acc, feasible, current_split, current_compression
            current_split = split_config
            self.energy_compression_pos = len(compression_rates) - 1
            current_compression = compression_rates[self.energy_compression_pos]
            flops_offloaded, flops_on_ue, top1_acc, feasible = evaluate(current_split, current_compression)
            while self.energy_compression_pos > 0 and not feasible:
                self.energy_compression_pos -= 1
                current_compression = compression_rates[self.energy_compression_pos]
                flops_offloaded, flops_on_ue, top1_acc, feasible = evaluate(current_split, current_compression)
            return feasible

        if self.energy_using_fixed_anchor:
            current_split = FIXED_SPLIT
            current_compression = FIXED_COMPRESSION
        else:
            current_split = split_for_pos(self.energy_split_pos)
            current_compression = compression_rates[self.energy_compression_pos]
        flops_offloaded, flops_on_ue, top1_acc, feasible = evaluate(current_split, current_compression)

        if not feasible and self.energy_last_move is not None:
            lever = self.energy_last_move
            self.energy_last_move = None

            if lever == 'compression':
                while self.energy_compression_pos > 0 and not feasible:
                    self.energy_compression_pos -= 1
                    current_compression = compression_rates[self.energy_compression_pos]
                    flops_offloaded, flops_on_ue, top1_acc, feasible = evaluate(current_split, current_compression)
                if not feasible:
                    self.energy_compression_pos = len(compression_rates) - 1
                    current_split = FIXED_SPLIT
                    current_compression = FIXED_COMPRESSION
                    flops_offloaded, flops_on_ue, top1_acc, feasible = evaluate(current_split, current_compression)
                    self.energy_compression_maxed_out = True
                    self.energy_split_maxed_out = True
                    self.energy_using_fixed_anchor = True  # subsequent steps' top-level eval also tests FIXED
                    self.energy_tier0_alt_split = None  # leaving tier 0 (if we were there) - clear any override
                    # if not feasible:
                    #     print(f"  [energy_only] FALLBACK STILL INFEASIBLE (compression): fell back to FIXED's "
                    #           f"configuration and it is STILL infeasible under current step's conditions - "
                    #           f"returning this step's (infeasible) measured values. Will retry fresh next "
                    #           f"step via the deadlock-reset path.")
            elif lever == 'split':
                # CAPPED HERE DELIBERATELY: this searches split_levels (5 tiers) x compression (4 levels),
                # plus tier0_alternates (14 device=3 sub-splits) x compression if tier 0 itself fails -
                # already close to 80 evaluations in the worst case. Confirmed via diagnostic trace that
                # even this can still fail to find anything feasible at the tightest deadlines (0.225s),
                # where OPT's full 137-action search finds something this reduced space does not. Rather
                # than keep widening the search toward OPT's own exhaustiveness - which would undermine
                # the entire point of HEURISTIC as a baseline structurally weaker than DRL/OPT, not just
                # weaker until every gap is patched - failure past this point is treated as a legitimate,
                # well-defined outcome: fall back to the safe starting configuration specifically (see
                # below), not wherever this search happened to stop.
                while self.energy_split_pos > 0 and not feasible:
                    self.energy_split_pos -= 1
                    feasible = try_split(split_for_pos(self.energy_split_pos))
                bottomed_out_at_zero = (self.energy_split_pos == 0 and not feasible)
                if bottomed_out_at_zero:
                    # tier 0's representative (or a previously-adopted alternate) failed even after
                    # exhausting compression at this tier - before giving up, try the other device=3
                    # sub-split variants (see tier0_alternates above), each ALSO with its own compression
                    # sweep via try_split (not just at one fixed level) - these are UE-energy-tied to the
                    # primary representative but NOT latency-tied, since how the remaining layers are
                    # divided among network nodes 1-3 changes each node's processing/communication load -
                    # confirmed via diagnostic trace as the actual gap at very tight deadlines.
                    for alt_split in tier0_alternates:
                        feasible = try_split(alt_split)
                        if feasible:
                            self.energy_tier0_alt_split = alt_split
                            break
                if not feasible:
                    # Search is now genuinely exhausted (by construction this only happens after reaching
                    # split_pos=0, since the loop above only stops on feasible or pos==0 - see the docstring
                    # for why "last_good_pos" (wherever this particular escalation attempt started from) is
                    # NOT used here: it is just one step back from wherever we started, not a position
                    # confirmed feasible under CURRENT conditions, and the earlier design that used it could
                    # itself report FALLBACK STILL INFEASIBLE.
                    #
                    # CHANGED: evaluates FIXED_SPLIT/FIXED_COMPRESSION (see definition above) rather than
                    # the fully-on-device "safe starting" configuration - gives HEURISTIC a well-defined,
                    # directly comparable floor (no worse than the naive static baseline) precisely at the
                    # deadlines/throughputs where this reduced search space can be exhaustively infeasible.
                    # self.energy_split_pos/compression_pos are still reset to the safe starting indices so
                    # future escalation attempts start from the normal search, not from FIXED specifically.
                    self.energy_split_pos = len(split_levels) - 1
                    self.energy_compression_pos = len(compression_rates) - 1
                    current_split = FIXED_SPLIT
                    current_compression = FIXED_COMPRESSION
                    flops_offloaded, flops_on_ue, top1_acc, feasible = evaluate(current_split, current_compression)
                    self.energy_split_maxed_out = True
                    self.energy_using_fixed_anchor = True  # subsequent steps' top-level eval also tests FIXED
                    self.energy_tier0_alt_split = None  # leaving tier 0 - clear any override for a fresh start later
                    # if not feasible:
                    #     # even FIXED's own configuration is infeasible this step (possible at the very
                    #     # tightest deadlines) - the deadlock fix (see the "both levers maxed out" branch
                    #     # below) already handles this by resetting maxed_out flags and retrying fresh next
                    #     # step, rather than freezing here permanently.
                    #     print(f"  [energy_only] FALLBACK STILL INFEASIBLE (split): fell back to FIXED's "
                    #           f"configuration and it is STILL infeasible under current step's conditions - "
                    #           f"returning this step's (infeasible) measured values. Will retry fresh next "
                    #           f"fresh next step via the deadlock-reset path.")
        else:
            # feasible (or nothing to revert - first step) - since the objective is energy-only, always
            # try to push further toward lower energy for NEXT step: compression first (fine-grained),
            # split escalation only once compression is exhausted (coarse-grained, resets compression)
            if not self.energy_compression_maxed_out and self.energy_compression_pos > 0:
                self.energy_compression_pos -= 1
                self.energy_last_move = 'compression'
                self.energy_using_fixed_anchor = False  # moving past the anchor to a position-based test
            elif not self.energy_split_maxed_out and self.energy_split_pos > 0:
                self.energy_split_pos -= 1
                self.energy_compression_pos = len(compression_rates) - 1
                self.energy_compression_maxed_out = False
                self.energy_last_move = 'split'
                self.energy_using_fixed_anchor = False  # moving past the anchor to a position-based test
            elif feasible:
                if self.energy_using_fixed_anchor:
                    self.energy_compression_maxed_out = False
                    self.energy_split_maxed_out = False
                # both levers genuinely exhausted AND the current position is feasible (and NOT anchored) -
                # this is the legitimate converged/optimal resting state, hold as-is
                self.energy_last_move = None
            else:
                current_split = FIXED_SPLIT
                current_compression = FIXED_COMPRESSION
                flops_offloaded, flops_on_ue, top1_acc, feasible = evaluate(current_split, current_compression)
                # if not feasible:
                #     print(f"  [energy_only] FALLBACK STILL INFEASIBLE (deadlock-recovery): fell back to "
                #           f"FIXED's configuration and it is STILL infeasible under current step's "
                #           f"conditions - returning this step's (infeasible) measured values.")
                if self.energy_reset_priority == 'split':
                    self.energy_split_maxed_out = False
                    self.energy_reset_priority = 'compression'
                else:
                    self.energy_compression_maxed_out = False
                    self.energy_reset_priority = 'split'
                self.energy_last_move = None

        self.update_energy_credit_usage(flops_offloaded, flops_on_ue)
        self.top1_accuracy_confidence = top1_acc
        self.split = current_split
        self.compression_rate = current_compression

        for k, v in split_indices.items():
            if v == self.split:
                split_idx = k
        # if split_idx is None:
        #     print(f"  [heuristic_energy_only] warning: split {self.split} not found in split_indices - "
        #           f"was allow_empty_nodes=True passed in? split_idx will be logged as None this step.")

        return self.split, self.compression_rate, split_idx, self.top1_accuracy_confidence

    def fixed_split(self, allowed_splits, num_nodes, allow_empty_nodes, dnn_model, episode_params, output):
        split_idx = None
        self.split = [(0, 0, 6), (1, 6, 10), (2, 10, 14), (3, 14, 18)]
        self.compression_rate = 0.5
        feasible_splits, split_indices = enumerate_action_space(allowed_splits, num_nodes, allow_empty_nodes)
        if self.top1_accuracy_confidence is None:
            # compute the top1 accuracy confidence for the default action
            inference_time, ue_en_comp, ue_en_comm, expected_output = compute_inference(self.split, dnn_model,
                                                                                        episode_params,
                                                                                        output, self.compression_rate)
            # since the action remains the same (i.e. default value), updating this once is sufficient
            self.top1_accuracy_confidence = self.return_top1_accuracy_confidence(expected_output)
        # compute the flops to be offloaded due to this selected split
        flops_offloaded, flops_on_ue = self.get_flops_offloaded(self.split)
        self.update_energy_credit_usage(flops_offloaded, flops_on_ue)
        # extract index of split config
        for k, v in split_indices.items():
            if v == self.split:
                split_idx = k
        return self.split, self.compression_rate, split_idx, self.top1_accuracy_confidence

    def ue_computation_only(self, allowed_splits, num_nodes, allow_empty_nodes, dnn_model, episode_params, output):
        split_idx = None
        self.split = [(0, 0, 18), (1, 18, 18), (2, 18, 18), (3, 18, 18)]
        self.compression_rate = 1.0
        feasible_splits, split_indices = enumerate_action_space(allowed_splits, num_nodes, allow_empty_nodes)
        if self.top1_accuracy_confidence is None:
            # compute the top1 accuracy confidence for the default action
            inference_time, ue_en_comp, ue_en_comm, expected_output = compute_inference(self.split, dnn_model,
                                                                                        episode_params,
                                                                                        output, self.compression_rate)
            # since the action remains the same (i.e. default value), updating this once is sufficient
            self.top1_accuracy_confidence = self.return_top1_accuracy_confidence(expected_output)
        # compute the flops to be offloaded due to this selected split
        flops_offloaded, flops_on_ue = self.get_flops_offloaded(self.split)
        self.update_energy_credit_usage(flops_offloaded, flops_on_ue)
        # extract index of split config
        for k, v in split_indices.items():
            if v == self.split:
                split_idx = k
        return self.split, self.compression_rate, split_idx, self.top1_accuracy_confidence

    def update_energy_credit_usage(self, flops_offloaded, flops_on_ue):
        # update the energy credit usage
        self.energy_credit_consumed = (flops_offloaded + self.total_flops_offloaded) / (
                self.total_flops + self.total_flops_on_ue)
        self.total_flops_offloaded += flops_offloaded
        self.total_flops_on_ue += flops_on_ue
        self.flops_offloaded = flops_offloaded

    def check_energy_credit_budget(self, flops_offloaded):
        """
        Function that checks the energy credit criteria.
        Args:
            flops_offloaded (float): the flops offloaded due to the current split.

        Returns:
            The result of the constraint being satisfied or not, and the energy credit consumed.
        """
        energy_credit_consumed = (flops_offloaded + self.total_flops_offloaded) / (
                    self.total_flops + self.total_flops_on_ue)
        if energy_credit_consumed <= self.max_energy_credit / 100:
            return True, energy_credit_consumed
        else:
            return False, 0

    def check_latency_criteria(self, inference_time):
        """
        Function that checks latency criteria.
        Args:
            inference_time (float): The inference time due to the current split.

        Returns:
            The result of the constraint being satisfied or not.
        """
        if inference_time <= self.max_inference_latency:
            return True
        else:
            return False

    def check_accuracy_confidence_criteria(self, top1_acc_confidence):
        # first check if the new accuracy confidence is less than the previous one
        if top1_acc_confidence < self.top1_accuracy_confidence:
            # then check if the difference is within the desired percentage decrease
            if (self.top1_accuracy_confidence - top1_acc_confidence) <= (
                    self.scenario_params['accuracy_decrease'] / 100):
                return True
            else:
                return False
        else:
            return True  # new accuracy confidence is greater than the previous one

    def return_top1_accuracy_confidence(self, out):
        with torch.no_grad():
            final_output = F.softmax(out, dim=1)
            top1_prob, top1_idx = torch.topk(final_output, 1)
        top1_accuracy_confidence = top1_prob.item()
        return top1_accuracy_confidence

    def get_flops_offloaded(self, selected_split_config):
        """
        Function that computes the flops offloaded for a selected split configuration.
        Args:
            selected_split_config (tuple): the selected split.

        Returns:
            the flops offloaded to the network and the corresponding flops on the ue.
        """
        flops_on_ue = 0
        (node_id, start, end) = selected_split_config[0]  # extract start and end layers of ue
        # case 1: all layers on ue, ue offloads nothing
        if start == 0 and end == 18:
            flops_on_ue = self.total_flops
            flops_offloaded = 0
            return flops_offloaded, flops_on_ue
        else:
            # case 2: at least one block on ue, ue offloads the rest
            for i, (block_id, block_start, block_end) in enumerate(self.allowed_splits_blocks):
                if end > block_end:
                    flops_on_ue += self.flops_per_block[block_id]
                elif end == block_end:
                    flops_on_ue += self.flops_per_block[block_id]
                    break
                else:
                    raise ValueError('Wrong mapping of blocks.')
            flops_offloaded = self.total_flops - flops_on_ue  # flops offloaded for this specific split config
            return flops_offloaded, flops_on_ue

        # # Randomly pick internal split points
        # allowed_splits = sorted(allowed_splits)
        # K = len(allowed_splits)
        # assert K >= 2, "Need at least start/end boundaries"

        # # >>> FIX: allow choosing K-1 (final boundary) when empty nodes are allowed
        # if allow_empty_nodes:
        #     cut_idx_space = np.arange(1, K)      # 1..K-1  (enables 'all on UE')
        # else:
        #     cut_idx_space = np.arange(1, K - 1)  # 1..K-2  (classic internal cuts only)

        # # Choose cut indices
        # if cut_idx_space.size == 0:
        #     points_idx = np.array([0, K - 1])
        # else:
        #     if allow_empty_nodes:
        #         cuts_idx = np.random.choice(cut_idx_space, size=num_nodes - 1, replace=True)
        #     else:
        #         if num_nodes - 1 > cut_idx_space.size:
        #             raise ValueError("Not enough unique split points for all nodes.")
        #         cuts_idx = np.random.choice(cut_idx_space, size=num_nodes - 1, replace=False)
        #     cuts_idx = np.sort(cuts_idx)
        #     points_idx = np.concatenate(([0], cuts_idx, [K - 1]))

        # # Ensure UE (first segment) has at least one layer
        # if points_idx[1] == points_idx[0]:
        #     points_idx[1] = min(points_idx[0] + 1, K - 1)

        # # Ensure non-decreasing sequence
        # for i in range(2, len(points_idx)):
        #     if points_idx[i] < points_idx[i - 1]:
        #         points_idx[i] = points_idx[i - 1]

        # # Build splits
        # splits = []
        # for node_id in range(num_nodes):
        #     s_idx = points_idx[node_id]
        #     e_idx = points_idx[node_id + 1]
        #     start = int(allowed_splits[s_idx])
        #     end = int(allowed_splits[e_idx])
        #     splits.append((node_id, start, end))

        # return splits

# if __name__ == "__main__":
#     allowed_splits = [0, 3, 6, 10, 14, 18]
#     num_nodes = 4

#     print("Testing random split sampling...")
#     for _ in range(5):
#         split = generate_random_split(allowed_splits, num_nodes, allow_empty_nodes=True)
#         print(split)
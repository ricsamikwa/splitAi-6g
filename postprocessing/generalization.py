"""
robustness_eval.py

Evaluates the robustness of trained DRL split-inference policies (DDQN, A2C, PPO) by comparing them
against OPT on a held-out dataset the models were NOT trained on. For each algorithm, a single saved
checkpoint (specified by training episode number) is loaded and run in inference mode (greedy, no
training updates, no exploration) over every time step of the new dataset. OPT's KPIs are NOT computed by
this script - they are read from pre-computed CSV logs (see OPT_LOG_FOLDER below) that you generate
separately by re-running OPT yourself on the same new dataset, in the same format this codebase's own
write_logs()/writeToCsv() (utils/logging_utils.py) already produces for the 'optimum' folder. This
pairing (same dataset, same order) is what makes a per-step MAPE meaningful - comparing unpaired
distributions would not tell you how far a given DRL decision was from the OPT decision it was actually
facing for that same state.

MAPE is computed with OPT as the reference (denominator) and the DRL algorithm's value as the prediction,
for each of: inference_time, ue_energy (ue_energy_comp + ue_energy_comm), and top1_accuracy_confidence.

Column format CONFIRMED from your example files (inference_time_1.csv, top1_1.csv, ue_energy_comm_1.csv,
ue_energy_comp_1.csv): all four KPIs use 'time_step' as the x-column (not 'time' as I'd assumed for three
of the four) - OPT_COLUMN_NAMES below is corrected accordingly. Filenames confirm the
<metric>_<episode_number>.csv naming load_opt_kpis() already assumed.

STATUS: This is a scaffold, not a finished runnable script - two pieces are marked NEEDS_INPUT below and
must be filled in before this will run:
  1. The new dataset's loading/iteration interface (only fragments of main.py have been seen).
  2. A working inference-mode execution path - Agent.execute() in agent.py currently raises TypeError
     when scenario_params['inference'] is True, since `action` is never assigned outside the training
     branch. Either a separate evaluation entry point already exists elsewhere in the codebase, or
     agent.py needs a small inference-mode branch added.
"""

import numpy as np
import pandas as pd

from rl.ddqn import DDQNAgent
from rl.a2c import A2CAgent
from rl.ppo import PPOAgent
from utils.rl_utils import load_model_params, return_order, parse_episode_number
from utils.logging_utils import read_single_col_data

# Where you'll place OPT's re-run KPI logs for the new dataset, in the same
# logs/<folder>/system/<metric>_<episode>.csv format write_logs() already produces for split_algorithm==3.
# PLACEHOLDER - point this at wherever you actually save OPT's output.
OPT_LOG_FOLDER = 'logs/optimum_new_dataset'

# (csv_x_column, dict_key_used_when_the_csv_was_written) per KPI - confirmed against your example files.
OPT_COLUMN_NAMES = {
    'inference_time': ('time_step', 'inference_time'),
    'ue_energy_comp': ('time_step', 'ue_energy_comp'),
    'ue_energy_comm': ('time_step', 'ue_energy_comm'),
    'top1': ('time_step', 'top1'),
}


# ---------------------------------------------------------------------------
# MAPE
# ---------------------------------------------------------------------------
def mape(reference, predicted):
    """
    Mean Absolute Percentage Error, with `reference` (OPT) as the denominator.

    Args:
        reference (array-like): OPT's per-step values for this KPI.
        predicted (array-like): The DRL algorithm's per-step values for this KPI, same order/length.

    Returns:
        float: MAPE as a percentage. Steps where reference == 0 are excluded (division undefined) -
            their count is reported alongside the result so silently-dropped steps aren't hidden.
    """
    reference = np.asarray(reference, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if reference.shape != predicted.shape:
        raise ValueError(f"reference and predicted must be the same length and matched step-for-step "
                          f"(got {reference.shape} vs {predicted.shape}) - MAPE requires paired "
                          f"OPT/DRL decisions for the same state, not independently-sampled series.")
    nonzero_mask = reference != 0
    n_dropped = int((~nonzero_mask).sum())
    if n_dropped > 0:
        print(f"  [mape] dropped {n_dropped}/{len(reference)} steps where OPT's reference value was 0")
    pct_errors = np.abs((reference[nonzero_mask] - predicted[nonzero_mask]) / reference[nonzero_mask])
    return 100.0 * float(np.mean(pct_errors))


# ---------------------------------------------------------------------------
# Checkpoint loading (built from agent.py/ddqn.py/a2c.py/ppo.py's own load methods - verified interfaces)
# ---------------------------------------------------------------------------
def load_drl_agent(agent_type, checkpoint_episode, scenario_params, n_states, n_actions, allowed_splits,
                    num_nodes, flops_per_block, split_indices):
    """
    Instantiates the requested DRL agent and loads a specific saved training-episode checkpoint into it,
    for inference-only use (no further training). Mirrors each agent's own load method exactly - see
    agent.py's define_agent_attributes() for the training-time equivalent this is based on.

    Args:
        agent_type (str): One of 'ddqn', 'a2c', 'ppo'.
        checkpoint_episode (int): The training episode number whose saved checkpoint should be loaded
            (matches the episode_count argument save_model_params() was called with during training -
            note this is the LOGGED episode number from parse_episode_number(), not necessarily the raw
            episode index, if scenario_params['n_episodes'] enables any subsampling in your logging setup).
        scenario_params, n_states, n_actions, allowed_splits, num_nodes, flops_per_block, split_indices:
            same constructor arguments Agent.__init__ passes to DDQNAgent/A2CAgent/PPOAgent.

    Returns:
        The loaded agent instance (DDQNAgent, A2CAgent, or PPOAgent), with scenario_params['inference']
        expected to already be set to True by the caller before this is invoked.
    """
    if agent_type == 'ddqn':
        agent = DDQNAgent(scenario_params, n_states, n_actions, allowed_splits, num_nodes, flops_per_block,
                           split_indices)
        agent.load_model(checkpoint_episode, 'main')
        agent.eval()
    elif agent_type == 'a2c':
        agent = A2CAgent(scenario_params, n_states, n_actions, allowed_splits, num_nodes, flops_per_block,
                          split_indices)
        agent.load_model_a2c(checkpoint_episode)
        agent.actor.eval()
        agent.critic.eval()
    elif agent_type == 'ppo':
        agent = PPOAgent(scenario_params, n_states, n_actions, allowed_splits, num_nodes, flops_per_block,
                          split_indices)
        agent.load_model_ppo(checkpoint_episode)
        agent.actor.eval()
        agent.critic.eval()
    else:
        raise ValueError(f"Unknown agent_type '{agent_type}' - expected 'ddqn', 'a2c', or 'ppo'.")
    return agent


# ---------------------------------------------------------------------------
# NEEDS_INPUT (gap 2): inference-mode execution loop
# ---------------------------------------------------------------------------
def run_drl_inference_on_dataset(agent, agent_type, dataset_iterator, dnn_model, allowed_splits_blocks):
    """
    Runs the given (already-loaded, already in eval mode) DRL agent greedily - no training, no
    exploration - over every step the dataset_iterator yields, collecting matched per-step KPIs.

    STUB: cannot be completed without (a) the dataset_iterator's actual interface (gap 1) and (b) a
    working inference-mode decision path. DDQNAgent.choose_action() already branches on
    scenario_params['inference'] internally (picks argmax instead of epsilon-greedy) per ddqn.py, and
    A2CAgent/PPOAgent's choose_action() do the same (argmax instead of sampling) - so the PER-STEP action
    selection itself should already work correctly in inference mode. What's missing is the surrounding
    loop: get_agent_state() -> choose_action() -> perform_action() -> collect KPIs, repeated over the new
    dataset's episodes/time steps, without any of the training-only calls (store_transition, replay
    buffer pushes, update()/optimizer steps) agent.py's train_*_agent() methods currently do inline.

    Expected return shape once implemented:
        (inference_times, ue_energies, top1_accuracies): three arrays, one entry per time step, in the
        SAME episode order load_opt_kpis() was called with, so the two can be paired step-for-step.
    """
    raise NotImplementedError(
        "Needs: the new dataset's loading/iteration interface (gap 1), and confirmation of how "
        "get_agent_state()/perform_action() should be driven outside of agent.py's training loop."
    )


# ---------------------------------------------------------------------------
# OPT - read pre-computed KPI logs (you run OPT separately; see OPT_LOG_FOLDER above)
# ---------------------------------------------------------------------------
def load_opt_kpis(episode_numbers, log_folder=OPT_LOG_FOLDER):
    """
    Reads OPT's pre-computed per-step KPI logs for the given episodes, in the same
    logs/<folder>/system/<metric>_<episode>.csv format write_logs() (utils/logging_utils.py) already
    produces for split_algorithm==3. Concatenates across episodes IN THE GIVEN ORDER, so this order must
    match whatever order run_drl_inference_on_dataset() walks the new dataset's episodes in, for the
    pairing MAPE relies on to be valid.

    Args:
        episode_numbers (list[int]): the (logged) episode numbers to read and concatenate, e.g. the same
            episode range your new-dataset OPT re-run and DRL inference run both cover.
        log_folder (str): folder OPT's CSVs live under (relative to 'logs/', matching writeToCsv's own
            'logs/{folder}/{filename}.csv' convention) - defaults to OPT_LOG_FOLDER above.

    Returns:
        (inference_times, ue_energies, top1_accuracies): three flat numpy arrays, concatenated across all
        given episodes in order. ue_energies is ue_energy_comp + ue_energy_comm, matched per step.
    """
    all_time, all_comp, all_comm, all_top1 = [], [], [], []
    for ep in episode_numbers:
        x_col, y_col = OPT_COLUMN_NAMES['inference_time']
        _, time_vals = read_single_col_data(f'logs/{log_folder}/system/inference_time_{ep}', x_col, y_col,
                                             float, float)
        x_col, y_col = OPT_COLUMN_NAMES['ue_energy_comp']
        _, comp_vals = read_single_col_data(f'logs/{log_folder}/system/ue_energy_comp_{ep}', x_col, y_col,
                                             float, float)
        x_col, y_col = OPT_COLUMN_NAMES['ue_energy_comm']
        _, comm_vals = read_single_col_data(f'logs/{log_folder}/system/ue_energy_comm_{ep}', x_col, y_col,
                                             float, float)
        x_col, y_col = OPT_COLUMN_NAMES['top1']
        _, top1_vals = read_single_col_data(f'logs/{log_folder}/system/top1_{ep}', x_col, y_col, float, float)

        if not (len(time_vals) == len(comp_vals) == len(comm_vals) == len(top1_vals)):
            raise ValueError(
                f"OPT logs for episode {ep} have mismatched step counts across KPI files "
                f"(inference_time={len(time_vals)}, ue_energy_comp={len(comp_vals)}, "
                f"ue_energy_comm={len(comm_vals)}, top1={len(top1_vals)}) - can't pair them step-for-step.")

        all_time.extend(time_vals)
        all_comp.extend(comp_vals)
        all_comm.extend(comm_vals)
        all_top1.extend(top1_vals)

    inference_times = np.asarray(all_time, dtype=float)
    ue_energies = np.asarray(all_comp, dtype=float) + np.asarray(all_comm, dtype=float)
    top1_accuracies = np.asarray(all_top1, dtype=float)
    return inference_times, ue_energies, top1_accuracies


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def evaluate_robustness(checkpoints, episode_numbers, scenario_params, n_states, n_actions, allowed_splits,
                         num_nodes, flops_per_block, split_indices, allowed_splits_blocks, dnn_model,
                         dataset_iterator_factory, opt_log_folder=OPT_LOG_FOLDER):
    """
    For each (agent_type, checkpoint_episode) pair in `checkpoints`, loads that checkpoint, runs it in
    inference mode on a fresh instance of the new dataset (via dataset_iterator_factory(), called once
    per algorithm so each gets its own iterator positioned at the start), and computes MAPE against OPT's
    pre-computed KPIs (read from opt_log_folder) for the same episode_numbers.

    Args:
        checkpoints (list[tuple[str, int]]): e.g. [('ddqn', 4200), ('a2c', 4200), ('ppo', 4200)] - the
            training-episode checkpoint to evaluate for each algorithm.
        episode_numbers (list[int]): the new dataset's episode numbers to evaluate over - MUST match, in
            order, both what dataset_iterator_factory() will walk through and what OPT's logs at
            opt_log_folder actually contain (i.e. you've already re-run OPT over exactly these episodes
            of the new dataset and saved its logs there).
        dataset_iterator_factory (callable): should return a FRESH iterator/generator over the new
            dataset each time it's called, so every DRL algorithm walks through an identical,
            independently-positioned pass over the same episode_numbers OPT's logs cover.

    Returns:
        pandas.DataFrame indexed by agent_type, with columns mape_inference_time, mape_ue_energy,
        mape_accuracy.
    """
    scenario_params_inference = dict(scenario_params)
    scenario_params_inference['inference'] = True

    print(f"Loading OPT's pre-computed KPIs from logs/{opt_log_folder}/system/ for episodes {episode_numbers}...")
    opt_time, opt_energy, opt_acc = load_opt_kpis(episode_numbers, log_folder=opt_log_folder)

    results = {}
    for agent_type, checkpoint_episode in checkpoints:
        print(f"Running {agent_type.upper()} (checkpoint ep{checkpoint_episode}) on the new dataset...")
        agent = load_drl_agent(agent_type, checkpoint_episode, scenario_params_inference, n_states,
                                n_actions, allowed_splits, num_nodes, flops_per_block, split_indices)
        drl_time, drl_energy, drl_acc = run_drl_inference_on_dataset(
            agent, agent_type, dataset_iterator_factory(), dnn_model, allowed_splits_blocks)

        results[agent_type] = {
            'mape_inference_time': mape(opt_time, drl_time),
            'mape_ue_energy': mape(opt_energy, drl_energy),
            'mape_accuracy': mape(opt_acc, drl_acc),
        }

    return pd.DataFrame(results).T


if __name__ == '__main__':
    # NEEDS_INPUT: scenario_params, n_states, n_actions, allowed_splits, num_nodes, flops_per_block,
    # split_indices, allowed_splits_blocks, dnn_model all need to be constructed the same way main.py
    # does for training - not yet seen in full. dataset_iterator_factory needs the new dataset's actual
    # loading interface - also not yet seen.
    #
    # episode_numbers = list(range(1, 301))  # e.g. match however many episodes you re-run OPT over
    # checkpoints = [('ddqn', 4200), ('a2c', 4200), ('ppo', 4200)]  # pick whichever saved episode(s) you want
    # df = evaluate_robustness(checkpoints, episode_numbers, scenario_params, n_states, n_actions,
    #                           allowed_splits, num_nodes, flops_per_block, split_indices,
    #                           allowed_splits_blocks, dnn_model, dataset_iterator_factory,
    #                           opt_log_folder='optimum_new_dataset')
    # print(df)
    # df.to_csv('logs/robustness/mape_results.csv')
    pass
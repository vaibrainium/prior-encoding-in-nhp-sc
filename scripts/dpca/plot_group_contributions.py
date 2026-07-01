#!/usr/bin/env python
"""Show why the common-mode choice-axis projection flips sign across epochs.

The visual choice decoder is a balanced +/- contrast. The projected common mode is
    CM(t) = Z+(t) + Z-(t),
where Z+ = sum over +weight neurons of D_i * M_i(t) and Z- = sum over -weight neurons,
with M_i the condition-averaged, mean-centred PSTH (exactly the quantity dpca_transform
projects). This plots Z+, Z- and their sum CM for each epoch, so the +group and -group
traces can be seen to diverge/cross — the projection only tracks the GAP between them,
which reverses between the early cue window and the saccade.

Usage:
    conda run -n prior-sc python scripts/dpca/plot_group_contributions.py
"""
import copy, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd, pickle

ROOT = Path("D:/prior-ephys-dm/prior-encoding-in-nhp-sc"); sys.path.insert(0, str(ROOT))
from config import dir_config, ephys_config
from src.utils import dpca_utils

GROUP = "awayRF"
MARG  = "c"
COLS  = ["visual", "cue", "response"]
CD = {"state_values": ["biased", "unbiased"], "coherences": [0, 0.06, 0.2, 0.5], "choices": ["awayRF", "toRF"]}
EXCL = ["210210_GP_JP", "241209_GP_TZ"]; ALIGN = list(ephys_config["alignment_settings_GP"].keys())


def _log(m):
    import datetime; print(f"[{datetime.datetime.now():%H:%M:%S}] {m}", flush=True)


def main():
    pdir = Path(dir_config.data.processed); cdir = Path(dir_config.data.compiled)
    out = ROOT / "dissemination" / "dpca" / f"dpca_{GROUP}_session_all_neuron_norm_zscore"
    sm = pd.read_csv(pdir/"sessions_metadata.csv"); sm = sm[~sm.session_id.isin(EXCL)].reset_index(drop=True)
    nm = pd.read_csv(pdir/"neuron_metadata.csv"); nm = nm[~nm.session_id.isin(EXCL)].reset_index(drop=True)
    _log("loading glm_hmm + ephys ...")
    glm = pickle.load(open(pdir/"glm_hmm_models"/"glm_hmm_masked_final.pkl","rb")); glm0 = copy.deepcopy(glm)
    ephys = pickle.load(open(pdir/"ephys_neuron_wise.pkl","rb"))
    bti, uti, _ = dpca_utils.extract_hmm_state_trial_info(sm, glm0, glm["data"], compiled_dir=cdir)
    sti = {"biased": bti, "unbiased": uti}; sess = sm.session_id[sm.prior_direction == GROUP]
    nids = dpca_utils.get_neuron_ids(nm, sess)
    avg, tw = dpca_utils.create_dpca_matrix(sess, CD, nids, sti, nm, ephys, ephys_config, condition_type="states")
    fa, ftw, _, _ = dpca_utils.clean_dpca_data(avg, tw, ALIGN)
    model, _ = dpca_utils.fit_dpca_on_alignment(fa["visual"], ftw["visual"], marginalization_keys=("b","s","c","t"))
    D = model.D[MARG][:, 0]; pos = D > 0; neg = D < 0
    _log(f"{pos.sum()} +weight, {neg.sum()} -weight neurons")

    st = ephys_config["alignment_settings_GP"]
    def tax(a):
        t = np.arange(st[a]["start_time_ms"], st[a]["end_time_ms"] + 1); n = fa[a].shape[-1]
        return t[-n:] if a == "response" else t[:n]

    fig, axes = plt.subplots(1, len(COLS), figsize=(5*len(COLS), 4), squeeze=False)
    for col, a in enumerate(COLS):
        X = fa[a]
        Xc = X - np.nanmean(X.reshape(X.shape[0], -1), 1).reshape((X.shape[0],) + (1,)*(X.ndim-1))
        M = np.nanmean(Xc, axis=(1, 2, 3))                 # (n_neurons, n_time)
        contrib = D[:, None] * M
        Zpos = np.nansum(contrib[pos], axis=0)
        Zneg = np.nansum(contrib[neg], axis=0)
        CM = Zpos + Zneg
        t = tax(a); n = min(len(t), len(CM))
        ax = axes[0, col]
        ax.plot(t[:n], Zpos[:n], color="tab:red", lw=1.6, label="+weight group")
        ax.plot(t[:n], Zneg[:n], color="tab:purple", lw=1.6, label="-weight group")
        ax.plot(t[:n], CM[:n], color="k", lw=2.0, label="sum = common-mode proj")
        ax.axhline(0, color="gray", lw=0.6); ax.axvline(0, color="r", lw=0.6, ls=":")
        ax.set_title(a); ax.set_xlabel("time (ms)"); ax.tick_params(labelsize=8)
        if col == 0:
            ax.set_ylabel("contribution to choice-axis projection"); ax.legend(fontsize=7)
    fig.suptitle(f"{GROUP}: +weight vs -weight group contributions to the visual choice-axis common mode\n"
                 f"projection (black) = gap between red and purple; the gap reverses cue -> response", y=1.04)
    fig.tight_layout()
    p = out / "group_contributions_choice.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    _log(f"saved {p}")


if __name__ == "__main__":
    main()

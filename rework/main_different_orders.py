import tqdm
import numpy as np
import matplotlib.pyplot as plt

from decoding import performBeliefPropagationFast
from decoding import performOSD_enhanced

experiment = [
    {
        "code": "[[72, 12, 6]]",
        "name": "72",
        "physicalErrorRates": [0.1, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01, 0.009],
        "distance": 6,
    },
    {
        "code": "[[90, 8, 10]]",
        "name": "90",
        "physicalErrorRates": [0.1, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01],
        "distance": 10,
    },
    {
        "code": "[[108, 8, 10]]",
        "name": "108",
        "physicalErrorRates": [0.1, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01],
        "distance": 10,
    },
    {
        "code": "[[144, 12, 12]]",
        "name": "144",
        "physicalErrorRates": [0.1, 0.06, 0.05, 0.04, 0.03, 0.02],
        "distance": 12,
    },
    {
        "code": "[[288, 12, 18]]",
        "name": "288",
        "physicalErrorRates": [0.1, 0.06, 0.05, 0.04],
        "distance": 18,
    },
]

experiment_dict = {exp["name"]: exp for exp in experiment}

trials = 1000

configurations = [
    {"bp_iter": 50, "osd_order": 0, "label": "BP50_OSD0"},
    {"bp_iter": 100, "osd_order": 0, "label": "BP100_OSD0"},
    {"bp_iter": 50, "osd_order": 7, "label": "BP50_OSD7"},
    {"bp_iter": 100, "osd_order": 7, "label": "BP100_OSD7"},
]

np.random.seed(0)

results = {}
for exp in experiment:
    code_name = exp["name"]
    oc = np.load(f"codes/{exp['code']}.npz")
    code = oc["Hx"]
    Lx = oc["Lx"]
    n = len(code[0])
    results[code_name] = {}

    for config in configurations:
        bp_iter = config["bp_iter"]
        osd_order = config["osd_order"]
        config_label = config["label"]
        results[code_name][config_label] = {}

        for errorRate in exp["physicalErrorRates"]:
            initialBeliefs = [np.log((1 - errorRate) / errorRate)] * n

            logicalError = 0
            OSD_invocations = 0
            degenerateErrors = 0
            weights_found_BP = []
            weights_found_OSD = []
            
            weights_found_BP_error = []
            weights_found_OSD_error = []

            OSD_invocation_AND_logicalError = 0
            

            for _ in tqdm.tqdm(range(trials), desc=f"Code {code_name}, {config_label}, p={errorRate}"):
                error = (np.random.random(n) < errorRate).astype(int)

                syndrome = (error @ code.T) % 2

                detection, isSyndromeFound, llrs, iteration = performBeliefPropagationFast(
                    code, syndrome, initialBeliefs, maxIter=bp_iter
                )

                if not isSyndromeFound:
                    detection = performOSD_enhanced(code, syndrome, llrs, detection, order=osd_order)
                    OSD_invocations += 1
                    
                residual = (detection + error) % 2
                syndromeLogic = (Lx @ residual) % 2
                
                osd_syndrome_check = (detection @ code.T) % 2
                is_valid_osd = np.array_equal(osd_syndrome_check, syndrome)
                
                # we have a logical error!    
                if np.any(syndromeLogic):
                    logicalError += 1
                    if not isSyndromeFound:
                        weights_found_OSD_error.append(np.sum(residual))
                        OSD_invocation_AND_logicalError += 1
                    if isSyndromeFound:
                        weights_found_BP_error.append(np.sum(residual))
                
                # we do not have a logical errror, but we calculate the weight of the error
                if not np.any(syndromeLogic):
                    if not np.array_equal(detection, error):
                        w = np.sum(residual)
                        if not isSyndromeFound: weights_found_OSD.append(w)
                        if isSyndromeFound: weights_found_BP.append(w)

                        if is_valid_osd: degenerateErrors += 1

            logicalErrorRate = logicalError / trials
            OSD_invocationRate = OSD_invocations / trials
            degenerateErrorRate = degenerateErrors / trials
            OSD_invocation_AND_logicalErrorRate = OSD_invocation_AND_logicalError / trials
            results[code_name][config_label][errorRate] = {
                "logical": logicalErrorRate,
                "osd": OSD_invocationRate,
                "degeneracies": degenerateErrorRate,
                "OSD_invocation_AND_logicalError": OSD_invocation_AND_logicalErrorRate,
                "weights_found_BP": weights_found_BP,
                "weights_found_OSD": weights_found_OSD,
                "weights_found_BP_error": weights_found_BP_error,
                "weights_found_OSD_error": weights_found_OSD_error,
            }
            print(
                f"Code {code_name}, {config_label}, p={errorRate}, Logical Error Rate: {logicalErrorRate}, OSD Invocation Rate: {OSD_invocationRate}"
            )
        
np.savez("rework/simulation_results_complex.npz", results=results)

colors = ["2E72AE", "64B791", "DBA142", "000000", "E17792"]
markers = ["d", "o", "s", "^"]

for code_name, code_data in results.items():
    fig, axes = plt.subplots(4, 1, figsize=(8, 12), sharex=True)
    fig.suptitle(f"Code {code_name} - Monte Carlo trials: {trials}")
    
    for i, (config_label, config_results) in enumerate(code_data.items()):
        x = list(config_results.keys())
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        
        axes[0].loglog(
            x,
            [v["logical"] for v in config_results.values()],
            marker=marker,
            label=f"{config_label}",
            color=f"#{color}",
        )
        axes[1].plot(
            x,
            [v["osd"] for v in config_results.values()],
            marker=marker,
            label=f"{config_label}",
            color=f"#{color}",
        )
        axes[2].plot(
            x,
            [v["degeneracies"] for v in config_results.values()],
            marker=marker,
            label=f"{config_label}",
            color=f"#{color}",
        )
        axes[3].plot(
            x,
            [v["OSD_invocation_AND_logicalError"] for v in config_results.values()],
            marker=marker,
            label=f"{config_label}",
            color=f"#{color}",
        )

    axes[0].set_ylabel("Logical Error Rate")
    axes[0].grid(True, which="both", ls="--")
    axes[0].legend()

    axes[1].set_ylabel("OSD Invocation Rate")
    axes[1].grid(True, which="both", ls="--")
    axes[1].legend()

    axes[2].set_ylabel("Degenerate Errors Rate")
    axes[2].grid(True, which="both", ls="--")
    axes[2].legend()

    axes[3].set_xlabel("Physical Error Rate")
    axes[3].set_ylabel("OSD Invocation & Error")
    axes[3].grid(True, which="both", ls="--")
    axes[3].legend()

    plt.tight_layout()
    plt.savefig(f"rework/orderOutput/results_{code_name}.png", dpi=300)
    plt.close(fig)

    # --- New Histogram Code ---
    
    # 1. Successful Decodings
    fig_w, axes_w = plt.subplots(len(configurations), 1, figsize=(8, 12), sharex=True)
    if len(configurations) == 1: axes_w = [axes_w] # Handle single config case just in case
    fig_w.suptitle(f"Code {code_name} - Weight Distribution (Success)")

    for i, (config_label, config_results) in enumerate(code_data.items()):
        weights_BP = []
        weights_OSD = []
        for res in config_results.values():
            weights_BP.extend(res["weights_found_BP"])
            weights_OSD.extend(res["weights_found_OSD"])
        
        ax = axes_w[i]
        # Determine bins based on data
        all_weights = weights_BP + weights_OSD
        max_w = max(all_weights) if all_weights else 30
        bins = np.arange(0, max_w + 2) - 0.5
        
        if weights_BP:
            ax.hist(weights_BP, bins=bins, alpha=0.6, label='BP', density=True, color="#2E72AE")
        if weights_OSD:
            ax.hist(weights_OSD, bins=bins, alpha=0.6, label='OSD', density=True, color="#DBA142")
            
        ax.axvline(x=experiment_dict[code_name]['distance'], color='red', linestyle='dashed', label='Distance')
        ax.set_title(f"Config: {config_label}")
        ax.set_ylabel("Density")
        ax.legend()
        ax.grid(True, alpha=0.3)

    axes_w[-1].set_xlabel("Weight")
    plt.tight_layout()
    plt.savefig(f"rework/orderOutput/weights_histograms_{code_name}.png", dpi=300)
    plt.close(fig_w)

    # 2. Logical Errors
    fig_e, axes_e = plt.subplots(len(configurations), 1, figsize=(8, 12), sharex=True)
    if len(configurations) == 1: axes_e = [axes_e]
    fig_e.suptitle(f"Code {code_name} - Weight Distribution (Logical Errors)")

    for i, (config_label, config_results) in enumerate(code_data.items()):
        weights_BP_err = []
        weights_OSD_err = []
        for res in config_results.values():
            weights_BP_err.extend(res["weights_found_BP_error"])
            weights_OSD_err.extend(res["weights_found_OSD_error"])
        
        ax = axes_e[i]
        all_weights = weights_BP_err + weights_OSD_err
        max_w = max(all_weights) if all_weights else 30
        bins = np.arange(0, max_w + 2) - 0.5
        
        if weights_BP_err:
            ax.hist(weights_BP_err, bins=bins, alpha=0.6, label='BP Error', density=True, color="#E17792")
        if weights_OSD_err:
            ax.hist(weights_OSD_err, bins=bins, alpha=0.6, label='OSD Error', density=True, color="#000000")
            
        ax.axvline(x=experiment_dict[code_name]['distance'], color='red', linestyle='dashed', label='Distance')
        ax.set_title(f"Config: {config_label}")
        ax.set_ylabel("Density")
        ax.legend()
        ax.grid(True, alpha=0.3)

    axes_e[-1].set_xlabel("Weight")
    plt.tight_layout()
    plt.savefig(f"rework/orderOutput/weights_histograms_ERROR_{code_name}.png", dpi=300)
    plt.close(fig_e)
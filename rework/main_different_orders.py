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

trials = 10000

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
    plt.savefig(f"rework/results_{code_name}.png", dpi=300)
    plt.close(fig)
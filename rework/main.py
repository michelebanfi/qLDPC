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

BP_maxIter = 50
OSD_order = 0

np.random.seed(0)

results = {}
for exp in experiment:
    code_name = exp["name"]
    oc = np.load(f"codes/{exp['code']}.npz")
    code = oc["Hx"]
    Lx = oc["Lx"]
    n = len(code[0])
    results[code_name] = {}

    for errorRate in exp["physicalErrorRates"]:
        initialBeliefs = [np.log((1 - errorRate) / errorRate)] * n

        logicalError = 0
        OSD_invocations = 0
        degenerateErrors = 0
        weights_found_BP = []
        weights_found_OSD = []
        
        weights_found_BP_error = []
        weights_found_OSD_error = []
        

        for _ in tqdm.tqdm(range(trials), desc=f"Code {code_name}, p={errorRate}"):
            error = (np.random.random(n) < errorRate).astype(int)

            syndrome = (error @ code.T) % 2

            detection, isSyndromeFound, llrs, iteration = performBeliefPropagationFast(
                code, syndrome, initialBeliefs, maxIter=BP_maxIter
            )

            if not isSyndromeFound:
                detection = performOSD_enhanced(code, syndrome, llrs, detection, order=OSD_order)
                OSD_invocations += 1
                
            residual = (detection + error) % 2
            syndromeLogic = (Lx @ residual) % 2
            
            osd_syndrome_check = (detection @ code.T) % 2
            is_valid_osd = np.array_equal(osd_syndrome_check, syndrome)
            
            if is_valid_osd and not np.any(syndromeLogic) and (np.array_equal(detection, error) == False):
                degenerateErrors += 1
                
            if np.any(syndromeLogic):
                logicalError += 1
                if not isSyndromeFound:
                    weights_found_OSD_error.append(np.sum(residual))
                if isSyndromeFound:
                    weights_found_BP_error.append(np.sum(residual))
            
            if not np.any(syndromeLogic):
                if not np.array_equal(detection, error):
                    w = np.sum(residual)
                    if not isSyndromeFound: weights_found_OSD.append(w)
                    if isSyndromeFound: weights_found_BP.append(w)

        logicalErrorRate = logicalError / trials
        OSD_invocationRate = OSD_invocations / trials
        degenerateErrorRate = degenerateErrors / trials
        results[code_name][errorRate] = {
            "logical": logicalErrorRate,
            "osd": OSD_invocationRate,
            "degeneracies": degenerateErrorRate,
            "weights_found_BP": weights_found_BP,
            "weights_found_OSD": weights_found_OSD,
            "weights_found_BP_error": weights_found_BP_error,
            "weights_found_OSD_error": weights_found_OSD_error,
        }
        print(
            f"Code {code_name}, p={errorRate}, Logical Error Rate: {logicalErrorRate}, OSD Invocation Rate: {OSD_invocationRate}"
        )
        
np.savez("rework/simulation_results.npz", results=results)

colors = ["2E72AE", "64B791", "DBA142", "000000", "E17792"]

fig, axes = plt.subplots(3, 1, figsize=(6, 8), sharex=True)
fig.suptitle(f"Monte Carlo trials: {trials}, BP max iterations: {BP_maxIter}, OSD order: {OSD_order}")
for (code_name, code_results), color in zip(results.items(), colors):
    x = list(code_results.keys())
    axes[0].semilogy(
        x,
        [v["logical"] for v in code_results.values()],
        marker="d",
        label=f"Code {code_name}",
        color=f"#{color}",
    )
    axes[1].plot(
        x,
        [v["osd"] for v in code_results.values()],
        marker="o",
        label=f"Code {code_name}",
        color=f"#{color}",
    )
    axes[2].plot(
        x,
        [v["degeneracies"] for v in code_results.values()],
        marker="s",
        label=f"Code {code_name}",
        color=f"#{color}",
    )

axes[0].set_ylabel("Logical Error Rate")
axes[0].grid(True, which="both", ls="--")
axes[0].legend()

axes[1].set_ylabel("OSD Invocation Rate")
axes[1].grid(True, which="both", ls="--")
axes[1].legend()

axes[2].set_xlabel("Physical Error Rate")
axes[2].set_ylabel("Degenerate Errors Rate")
axes[2].grid(True, which="both", ls="--")
axes[2].legend()

plt.tight_layout()
plt.savefig("rework/logical_error_rates.png", dpi=300)

fig2, axes2 = plt.subplots(2, 5, figsize=(15, 8))
for i, (code_name, code_results) in enumerate(results.items()):
    weights_BP = []
    weights_OSD = []
    for res in code_results.values():
        weights_BP.extend(res["weights_found_BP"])
        weights_OSD.extend(res["weights_found_OSD"])
    
    if weights_BP:
        axes2[0, i].hist(weights_BP, bins=np.arange(0, 30)-0.5, color=f"#{colors[i]}", alpha=0.7)
    axes2[0, i].axvline(x=experiment_dict[code_name]['distance'], color='red', linestyle='dashed', label='Code Distance')
    axes2[0, i].set_title(f"Code {code_name} (BP)")
    axes2[0, i].set_xlabel("Weight")
    axes2[0, i].set_ylabel("Frequency")
    
    if weights_OSD:
        axes2[1, i].hist(weights_OSD, bins=np.arange(0, 30)-0.5, color=f"#{colors[i]}", alpha=0.7)
    axes2[1, i].axvline(x=experiment_dict[code_name]['distance'], color='red', linestyle='dashed', label='Code Distance')
    axes2[1, i].set_title(f"Code {code_name} (OSD)")
    axes2[1, i].set_xlabel("Weight")
    axes2[1, i].set_ylabel("Frequency")

plt.tight_layout()
plt.savefig("rework/weights_histograms.png", dpi=300)

fig2, axes2 = plt.subplots(2, 5, figsize=(15, 8))
for i, (code_name, code_results) in enumerate(results.items()):
    weights_BP = []
    weights_OSD = []
    for res in code_results.values():
        weights_BP.extend(res["weights_found_BP_error"])
        weights_OSD.extend(res["weights_found_OSD_error"])
    
    if weights_BP:
        axes2[0, i].hist(weights_BP, bins=np.arange(0, 30)-0.5, color=f"#{colors[i]}", alpha=0.7)
    axes2[0, i].axvline(x=experiment_dict[code_name]['distance'], color='red', linestyle='dashed', label='Code Distance')
    axes2[0, i].set_title(f"Code {code_name} (BP)")
    axes2[0, i].set_xlabel("Weight")
    axes2[0, i].set_ylabel("Frequency")
    
    if weights_OSD:
        axes2[1, i].hist(weights_OSD, bins=np.arange(0, 30)-0.5, color=f"#{colors[i]}", alpha=0.7)
    axes2[1, i].axvline(x=experiment_dict[code_name]['distance'], color='red', linestyle='dashed', label='Code Distance')
    axes2[1, i].set_title(f"Code {code_name} (OSD)")
    axes2[1, i].set_xlabel("Weight")
    axes2[1, i].set_ylabel("Frequency")

plt.tight_layout()
plt.savefig("rework/weights_histograms_ERROR.png", dpi=300)
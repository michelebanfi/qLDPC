import tqdm
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

from decoding import performBeliefPropagation_Symmetric
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
llr_data = {}

for exp in experiment:
    code_name = exp["name"]
    oc = np.load(f"codes/{exp['code']}.npz")
    code = oc["Hx"]
    Lx = oc["Lx"]
    n = len(code[0])
    results[code_name] = {}

    llr_data[code_name] = {"true_0": [], "true_1": []}

    for errorRate in exp["physicalErrorRates"]:
        initialBeliefs = [np.log((1 - errorRate) / errorRate)] * n

        logicalError = 0
        OSD_invocations = 0
        degenerateErrors = 0
        weights_found_BP = []
        weights_found_OSD = []
        iterations = []
        
        weights_found_BP_error = []
        weights_found_OSD_error = []

        OSD_invocation_AND_logicalError = 0

        collect_stats = (errorRate == exp["physicalErrorRates"][0])
        

        for _ in tqdm.tqdm(range(trials), desc=f"Code {code_name}, p={errorRate}"):
            error = (np.random.random(n) < errorRate).astype(int)

            syndrome = (error @ code.T) % 2

            detection, isSyndromeFound, llrs, iteration = performBeliefPropagation_Symmetric(
                code, syndrome, initialBeliefs, maxIter=BP_maxIter
            )
            iterations.append(iteration)

            if collect_stats:
                # We mask the LLRs based on what the bit truly was
                # llrs corresponding to bits that were 0 (No Error)
                llr_data[code_name]["true_0"].extend(llrs[error == 0])
                
                # llrs corresponding to bits that were 1 (Error)
                llr_data[code_name]["true_1"].extend(llrs[error == 1])

            if not isSyndromeFound:
                detection = performOSD_enhanced(code, syndrome, llrs, detection, order=OSD_order)
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
        average_iterations = np.mean(iterations)
        OSD_invocation_AND_logicalErrorRate = OSD_invocation_AND_logicalError / trials
        results[code_name][errorRate] = {
            "logical": logicalErrorRate,
            "osd": OSD_invocationRate,
            "degeneracies": degenerateErrorRate,
            "OSD_invocation_AND_logicalError": OSD_invocation_AND_logicalErrorRate,
            "weights_found_BP": weights_found_BP,
            "weights_found_OSD": weights_found_OSD,
            "weights_found_BP_error": weights_found_BP_error,
            "weights_found_OSD_error": weights_found_OSD_error,
            "average_iterations": average_iterations,
        }
        print(
            f"Code {code_name}, p={errorRate}, Logical Error Rate: {logicalErrorRate}, OSD Invocation Rate: {OSD_invocationRate}"
        )
        
np.savez("rework/simulation_results.npz", results=results)

colors = ["2E72AE", "64B791", "DBA142", "000000", "E17792"]

fig, axes = plt.subplots(5, 1, figsize=(6, 10), sharex=True)
fig.suptitle(f"Monte Carlo trials: {trials}, BP max iterations: {BP_maxIter}, OSD order: {OSD_order} \n The y-axis shows rates calculated over all trials.")
for (code_name, code_results), color in zip(results.items(), colors):
    x = list(code_results.keys())
    axes[0].loglog(
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
    axes[3].plot(
        x,
        [v["OSD_invocation_AND_logicalError"] for v in code_results.values()],
        marker="^",
        label=f"Code {code_name}",
        color=f"#{color}",
    )
    axes[4].plot(
        x,
        [v["average_iterations"] for v in code_results.values()],
        marker="o",
        label=f"Code {code_name}",
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

axes[3].set_ylabel("OSD Invocation & Error")
axes[3].grid(True, which="both", ls="--")
axes[3].legend()

axes[4].set_xlabel("Physical Error Rate")
axes[4].set_ylabel("Average BP Iterations")
axes[4].axhline(y=BP_maxIter, color='k', linestyle='--', label='BP Max Iter')
axes[4].grid(True, which="both", ls="--")
axes[4].legend()

plt.tight_layout()
plt.savefig("rework/SCOPT.png", dpi=300)

fig_llr, axes_llr = plt.subplots(len(llr_data), 1, figsize=(10, 4 * len(llr_data)), sharex=False)

if len(llr_data) == 1: axes_llr = [axes_llr] # Handle single plot case

for ax, (code_name, data) in zip(axes_llr, llr_data.items()):

    n, bins, patches = ax.hist(data["true_0"], bins=100, color='blue', alpha=0.5, density=True, label='True 0 (Empirical)')
    
    # 2. Generate the Theoretical Red Curve (Gaussian Fit)
    # Instead of transforming the noisy 'n', we fit a Gaussian to the raw data
    mu = np.mean(data["true_0"])
    sigma = np.std(data["true_0"])
    
    # Create a smooth x-axis range for the line plot
    x_range = np.linspace(min(bins), max(bins), 1000)
    
    # Theoretical True 1 is the mirror image of True 0:
    # It should have Mean = -Mean_Blue
    theoretical_red_gaussian = norm.pdf(x_range, loc=-mu, scale=sigma)
    
    ax.plot(x_range, theoretical_red_gaussian, color='darkred', linestyle='--', linewidth=2, label='True 1 (Theoretical Gaussian)')
    
    # 3. Plot Empirical Red Histogram (True 1)
    ax.hist(data["true_1"], bins=100, color='red', alpha=0.3, density=True, label='True 1 (Empirical)')

    ax.set_title(f"LLR Distribution: Code {code_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axvline(0, color='black', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig("rework/llr_distributionsSCOPT.png", dpi=300)

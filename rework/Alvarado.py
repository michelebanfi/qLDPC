import tqdm
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

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

trials = 100

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
                llr_data[code_name]["true_0"].extend(llrs[error == 0])
                
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

def calculate_optimal_alpha(llr_data_code, bins=50, plot=False):
    """
    Implements the methodology from Alvarado et al. (2009) to find alpha.
    """
    # 1. Flatten data
    L_given_0 = np.array(llr_data_code["true_0"]) # L-values when bit was 0 (syndrome + noise implication)
    L_given_1 = np.array(llr_data_code["true_1"]) # L-values when bit was 1
    
    # 2. Create Histograms (approximate the PDFs p(lambda|0) and p(lambda|1))
    # We use a shared bin range to ensure alignment
    min_val = min(L_given_0.min(), L_given_1.min())
    max_val = max(L_given_0.max(), L_given_1.max())
    
    # Avoid outliers skewing the plot
    min_val = max(min_val, -20)
    max_val = min(max_val, 20)
    
    hist_range = (min_val, max_val)
    
    hist_0, bin_edges = np.histogram(L_given_0, bins=bins, range=hist_range, density=True)
    hist_1, _ = np.histogram(L_given_1, bins=bins, range=hist_range, density=True)
    
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # 3. Calculate the Log Ratio (The correcting function f(lambda))
    # f(lambda) = log( p(L|1) / p(L|0) )
    # Handle division by zero or log of zero safely
    valid_indices = (hist_0 > 0) & (hist_1 > 0)
    
    lambdas = bin_centers[valid_indices]
    f_lambdas = np.log(hist_1[valid_indices] / hist_0[valid_indices])
    
    # 4. Fit a line f(lambda) = alpha * lambda
    # The paper suggests a linear approximation is sufficient [cite: 170]
    def linear_model(x, alpha):
        return alpha * x
    
    popt, _ = curve_fit(linear_model, lambdas, f_lambdas)
    alpha_opt = popt[0]
    
    if plot:
        plt.figure(figsize=(8, 4))
        plt.scatter(lambdas, f_lambdas, label='Empirical $f(\\lambda)$', alpha=0.6)
        plt.plot(lambdas, linear_model(lambdas, alpha_opt), 'r--', label=f'Fit $\\alpha \\approx {alpha_opt:.3f}$')
        plt.plot(lambdas, lambdas, 'k:', label=r'Consistency Condition ($\alpha=1$)')
        plt.xlabel(r'L-value ($\lambda$)')
        plt.ylabel(r'$\log(p(\lambda|1)/p(\lambda|0))$')
        plt.title(f'Consistency Plot (Alvarado Method)')
        plt.legend()
        plt.grid(True)
        plt.show()
        
    return alpha_opt

calculate_optimal_alpha(llr_data["72"], bins=50, plot=True)

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

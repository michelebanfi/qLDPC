import numpy as np
import matplotlib.pyplot as plt


result = np.load("data/LERS.npz", allow_pickle=True)

print(dict(result))

PER = result["physicalErrorRates"]
print(PER)

res = result["results"].item() # idk why .item() is needed, soooo python of him to require that
print(res)

plt.figure(figsize=(10,6))
for item in res:
    plt.plot(PER, res[item]['ler'], label=item, marker='o')
    plt.xscale('log')
    plt.legend()
    plt.grid(True)
    print(res[item])
plt.savefig("media/LERSReloaded.png")

import matplotlib.pyplot as plt
import numpy as np
import statistics

node = []
fidelity = []
fidelity_inv = []

with open("C:/Users/alica/DIG/benchmarks/xgraph/fidelity_scores_s0_2.txt", "r") as file:
    lines = file.readlines()

for line in lines:
    parts = line.split()
    parts[3] = parts[3].replace(',', '')  # Virgülü kaldır
    parts[5] = parts[5].replace(',', '')  # Virgülü kaldır
    parts[7] = parts[7].replace(',', '')  # Virgülü kaldır
    node.append(int(parts[3]))
    fidelity.append(float(parts[5]))
    fidelity_inv.append(float(parts[7]))

# plt.scatter(node, fidelity, marker='o', label='Fidelity', color='blue')
# plt.scatter(node, fidelity_inv, marker='^', label='Fidelity_inv', color='red')
# plt.xlabel('Number of Nodes')
# plt.ylabel('Fidelity')
# plt.legend()
# plt.title('Sparsity 0.5')
# plt.grid(False)
# plt.savefig("C:/Users/alica/DIG/benchmarks/xgraph/fidelity-fidelity_inv_sparsity05.png", dpi=300)
# plt.show()


quartiles = np.percentile(node, [25, 50, 75])



print("25th percentile of nodes:", quartiles[0])
print("50th percentile of nodes:", quartiles[1])
print("75th percentile of nodes:", quartiles[2])

fidelity_sum0 = 0
fidelity_inv_sum0 = 0

fidelity_sum1 = 0
fidelity_inv_sum1 = 0

fidelity_sum2 = 0
fidelity_inv_sum2 = 0

fidelity_sum3 = 0
fidelity_inv_sum3 = 0

q1n = []
q2n = []
q3n = []
q4n = []

q1f = []
q2f = []
q3f = []
q4f = []

q1fi = []
q2fi = []
q3fi = []
q4fi = []
for i in range(len(node)):
    
    if node[i] <= quartiles[0]:
        fidelity_sum0 = fidelity_sum0 + fidelity[i]
        fidelity_sum0 = fidelity_inv_sum0 + fidelity_inv[i]
        q1n.append(node[i])
        q1f.append(fidelity[i])
        q1fi.append(fidelity_inv[i])
    elif node[i]<= quartiles[1] and node[i] > quartiles[0]:
        fidelity_sum1 = fidelity_sum1 + fidelity[i]
        fidelity_sum1 = fidelity_inv_sum1 + fidelity_inv[i]
        q2n.append(node[i])
        q2f.append(fidelity[i])
        q2fi.append(fidelity_inv[i])
    elif node[i]<= quartiles[2] and node[i] > quartiles[1]:
        fidelity_sum2 = fidelity_sum2 + fidelity[i]
        fidelity_sum2 = fidelity_inv_sum2 + fidelity_inv[i]
        q3n.append(node[i])
        q3f.append(fidelity[i])
        q3fi.append(fidelity_inv[i])
    else:
        fidelity_sum3 = fidelity_sum3 + fidelity[i]
        fidelity_sum3 = fidelity_inv_sum3 + fidelity_inv[i]
        q4n.append(node[i])
        q4f.append(fidelity[i])
        q4fi.append(fidelity_inv[i])

node1_mean = statistics.mean(q1n)
fidelity1_mean = statistics.mean(q1f)
fidelity_inv1_mean = statistics.mean(q1fi)

node2_mean = statistics.mean(q2n)
fidelity2_mean = statistics.mean(q2f)
fidelity_inv2_mean = statistics.mean(q2fi)

node3_mean = statistics.mean(q3n)
fidelity3_mean = statistics.mean(q3f)
fidelity_inv3_mean = statistics.mean(q3fi)

node4_mean = statistics.mean(q4n)
fidelity4_mean = statistics.mean(q4f)
fidelity_inv4_mean = statistics.mean(q4fi)

nodes_mean = [node1_mean, node2_mean, node3_mean, node4_mean]
fidelity_mean_values = [fidelity1_mean, fidelity2_mean, fidelity3_mean, fidelity4_mean]
fidelity_inv_mean_values = [fidelity_inv1_mean, fidelity_inv2_mean, fidelity_inv3_mean, fidelity_inv4_mean]

fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# İlk alt grafik için scatter plot çizme
axs[0].scatter(node, fidelity, marker='o', label='Fidelity', color='blue')
axs[0].scatter(node, fidelity_inv, marker='^', label='Fidelity_inv', color='red')
axs[0].set_xlabel('Number of Nodes')
axs[0].set_ylabel('Fidelity and Fidelity_inv -- sparsity 0.2')
axs[0].set_title('')
axs[0].grid(False)
axs[0].legend()

# İkinci alt grafik için scatter plot çizme
axs[1].scatter(nodes_mean, fidelity_mean_values, marker='o', label='Fidelity', color='blue') # İkinci grafiği oluşturmak için gerekli kodu buraya ekleyin
axs[1].scatter(nodes_mean, fidelity_inv_mean_values, marker='^', label='Fidelity_inv', color='red')
#İkinci grafik için diğer ayarları yapma

axs[1].set_xlabel('The average values ​​of node values ​​divided into 4 quarters')
axs[1].set_ylabel('Fidelity and Fidelity_inv mean -- sparsity 0.2')
axs[1].set_title('')
axs[1].grid(False)
axs[1].legend()



# Grafikleri gösterme ve kaydetme
plt.tight_layout()  # Alt grafikler arasındaki boşlukları ayarlama
plt.savefig("C:/Users/alica/DIG/benchmarks/xgraph/fidelity-fidelity_inv_mean_sparsity0_2_two_graf.png", dpi=300)
plt.show()
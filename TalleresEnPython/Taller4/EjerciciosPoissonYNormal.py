import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson, norm

# =========================
# 📌 1. Distribución de Poisson - Accidentes
# =========================
λ1 = 3
x_vals1 = np.arange(0, 10)
pmf_vals1 = poisson.pmf(x_vals1, λ1)

prob1 = sum(pmf_vals1[:3])  # P(X ≤ 2)

plt.figure(figsize=(7,4))
plt.bar(x_vals1, pmf_vals1, color="lightgray", edgecolor="black")
plt.bar(x_vals1[:3], pmf_vals1[:3], color="skyblue", edgecolor="black")
plt.title(f"Distribución de Poisson (λ={λ1})\nP(X ≤ 2) = {prob1:.4f}")
plt.xlabel("Número de accidentes por mes")
plt.ylabel("Probabilidad")
plt.savefig("Taller4-grafica1_Poisson.png")

plt.show()

# =========================
# 📌 2. Distribución de Poisson - Correos recibidos
# =========================
λ2 = 6
x_vals2 = np.arange(0, 15)
pmf_vals2 = poisson.pmf(x_vals2, λ2)

prob2 = poisson.pmf(8, λ2)  # P(X = 8)

plt.figure(figsize=(7,4))
plt.bar(x_vals2, pmf_vals2, color="lightgray", edgecolor="black")
plt.bar([8], [prob2], color="orange", edgecolor="black")
plt.title(f"Distribución de Poisson (λ={λ2})\nP(X = 8) = {prob2:.4f}")
plt.xlabel("Número de correos por hora")
plt.ylabel("Probabilidad")
plt.savefig("Taller4-grafica2_Poisson.png")

plt.show()

# =========================
# 📊 3. Distribución Normal - Producción de botellas
# =========================
μ1 = 500
σ1 = 5
x = np.linspace(480, 520, 300)
y = norm.pdf(x, μ1, σ1)

z1 = (495 - μ1) / σ1
z2 = (510 - μ1) / σ1
prob3 = norm.cdf(510, μ1, σ1) - norm.cdf(495, μ1, σ1)

plt.figure(figsize=(7,4))
plt.plot(x, y, color="black")
plt.fill_between(x, y, where=(x>=495)&(x<=510), color="lightgreen", alpha=0.7)
plt.title(f"Normal(μ={μ1}, σ={σ1})\nP(495 < X < 510) = {prob3:.4f}")
plt.xlabel("Contenido de jugo (ml)")
plt.ylabel("Densidad de probabilidad")
plt.savefig("Taller4-grafica3_Normal.png")

plt.show()

# =========================
# 📊 4. Distribución Normal - Tiempo de atención
# =========================
μ2 = 25
σ2 = 4
x2 = np.linspace(10, 40, 300)
y2 = norm.pdf(x2, μ2, σ2)

prob4 = 1 - norm.cdf(30, μ2, σ2)  # P(X > 30)

plt.figure(figsize=(7,4))
plt.plot(x2, y2, color="black")
plt.fill_between(x2, y2, where=(x2>30), color="salmon", alpha=0.7)
plt.title(f"Normal(μ={μ2}, σ={σ2})\nP(X > 30) = {prob4:.4f}")
plt.xlabel("Tiempo de atención (min)")
plt.ylabel("Densidad de probabilidad")
plt.savefig("Taller4-grafica4_Normal.png")

plt.show()
 
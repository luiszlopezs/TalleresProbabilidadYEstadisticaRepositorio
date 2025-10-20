import numpy as np
import matplotlib.pyplot as plt

# --- Definición de la densidad ---
# f(x, y) = 1/100 en el rectángulo 0 < x < 10, 0 < y < 10
f = 1/100

# --- Cálculo de la probabilidad ---
# P(X < 3, Y < 5)
prob = f * 3 * 5
print("Probabilidad P(X<3, Y<5) =", prob)

# --- Visualización con matplotlib ---
x = np.linspace(0, 10, 200)
y = np.linspace(0, 10, 200)
X, Y = np.meshgrid(x, y)

# Región donde f(x,y) = 1/100
Z = np.full_like(X, f)

plt.figure(figsize=(7, 7))
# Dibujar el cuadrado total de 0<x<10, 0<y<10
plt.contourf(X, Y, Z, levels=50, cmap='Blues', alpha=0.5)

# Resaltar la región X<3, Y<5
plt.fill_betweenx(y=[0, 5], x1=0, x2=3, color='orange', alpha=0.6, label='Región X<3, Y<5')

# Etiquetas y detalles
plt.title("Distribución conjunta uniforme f(x,y) = 1/100\ny región P(X<3, Y<5)")
plt.xlabel("Tiempo de espera X (min)")
plt.ylabel("Tiempo de consumo Y (min)")
plt.xlim(0, 10)
plt.ylim(0, 10)
plt.legend()
plt.grid(True)
plt.savefig("Taller3-grafica1_continua.png")
plt.show()

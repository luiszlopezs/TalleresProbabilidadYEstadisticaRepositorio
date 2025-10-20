import matplotlib.pyplot as plt
from math import comb

# Total de combinaciones posibles al sacar 2 bolas de 5
total = comb(5, 2)

# Calcular P(X=2, Y=0)
p_x2_y0 = (comb(3, 2) * comb(2, 0)) / total

print(f"P(X=2, Y=0) = {p_x2_y0:.3f}")

# Graficar el resultado
plt.bar(['(X=2, Y=0)'], [p_x2_y0], color='red')
plt.ylabel('Probabilidad')
plt.title('Probabilidad de sacar 2 bolas rojas y 0 azules')
plt.ylim(0, 1)
plt.savefig("Taller3-grafica1_discreta.png")
plt.show()


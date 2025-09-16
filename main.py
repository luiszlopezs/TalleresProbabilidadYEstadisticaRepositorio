# Importar librerías necesarias
import pymc as pm              # Para construir y estimar el modelo bayesiano
import arviz as az             # Para analizar y visualizar los resultados de inferencia
import numpy as np             # Para manejar datos numéricos
import matplotlib.pyplot as plt # Para graficar

if __name__ == "__main__":
    # -----------------------------
    # 1. Definir los datos observados
    # -----------------------------
    # En este caso, tenemos un conjunto pequeño de alturas (en cm).
    data = np.array([170, 172, 168, 165, 174, 169, 171])

    # -----------------------------
    # 2. Definir el modelo bayesiano
    # -----------------------------
    with pm.Model() as model:
        # Priori para la media (μ):
        # Asumimos que la media está centrada en 170 cm, con incertidumbre de ±10 cm
        mu = pm.Normal("mu", mu=170, sigma=10)

        # Priori para la desviación estándar (σ):
        # Usamos una distribución HalfNormal para asegurarnos de que σ > 0
        # y esperamos un valor alrededor de 5 cm.
        sigma = pm.HalfNormal("sigma", sigma=5)

        # Verosimilitud:
        # Los datos observados se asumen distribuidos normalmente
        # con la media y desviación estándar definidas por mu y sigma.
        obs = pm.Normal("obs", mu=mu, sigma=sigma, observed=data)

        # -----------------------------
        # 3. Muestreo bayesiano con MCMC
        # -----------------------------
        # Generamos muestras de la distribución posterior de los parámetros.
        # - 1000: número de muestras guardadas después del ajuste
        # - tune=500: número de pasos de ajuste para que el algoritmo "aprenda"
        # - chains=1, cores=1: configuración más estable en Windows
        # - target_accept=0.9: tasa de aceptación más alta (evita saltos grandes)
        trace = pm.sample(1000, tune=500, chains=1, cores=1, target_accept=0.9)

    # -----------------------------
    # 4. Resumen numérico de los resultados
    # -----------------------------
    # Muestra estadísticas de las distribuciones posteriores de mu y sigma:
    # media, desviación estándar, intervalos de credibilidad, etc.
    print(az.summary(trace))

    # -----------------------------
    # 5. Visualización de las cadenas MCMC y distribuciones
    # -----------------------------
    # Muestra cómo evolucionaron las cadenas y las distribuciones estimadas.
    az.plot_trace(trace)
    plt.show()

    # -----------------------------
    # 6. Distribuciones posteriores
    # -----------------------------
    # Gráfica las distribuciones posteriores con intervalos de credibilidad.
    az.plot_posterior(trace)
    plt.show()

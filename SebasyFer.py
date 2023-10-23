#Sebastian Salazar C.I 28.465.047
#Fernando Rodriguez C.I 27.589.678

import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Datos de ejemplo: rendimientos históricos y volatilidades de las acciones
rendimientos = np.array([0.12, 0.15, 0.20, 0.10])
volatilidades = np.array([0.18, 0.22, 0.30, 0.15])

# Restricción de volatilidad máxima
volatilidad_maxima = 0.25

sharpe = rendimientos/volatilidades 
max_rendimiento = rendimientos*[sharpe]
max_volatilidad = volatilidades*[sharpe]

# Función objetivo: maximizar el rendimiento esperado de la cartera
def objective(weights):
    rendimiento_esperado = np.sum(rendimientos * weights)
    return -rendimiento_esperado  # Negativo para maximizar

# Restricción de volatilidad
def constraint(weights):
    volatilidad = np.sqrt(np.sum((volatilidades * weights) ** 2))
    return volatilidad - volatilidad_maxima

# Inicialización de pesos iniciales
initial_weights = np.ones(len(rendimientos)) / len(rendimientos)

# Restricciones
constraints = ({'type': 'eq', 'fun': constraint})

# Límites de los pesos (0 <= pesos <= 1)
bounds = [(0, 1) for _ in range(len(rendimientos))]

# Optimización
result = minimize(objective, initial_weights, constraints=constraints, bounds=bounds)
optimized_weights = result.x
optimized_return = -result.fun  # Negativo porque estamos maximizando

# Mostrar los resultados
print("Resultados:")
print("Maximos Rendimientos:",max_rendimiento )
print("Maximas Volativilidad:",max_volatilidad)
print("Asignación de pesos optimizada:")
for i in range(len(rendimientos)):
    print(f"Acción {i + 1}: {optimized_weights[i]:.2f}")
print(f"Rendimiento esperado de la cartera optimizada: {optimized_return:.2%}")

#Grafica En Pantalla
plt.figure(figsize = (12,6))
plt.scatter(volatilidades,rendimientos,c = (rendimientos/volatilidades))
plt.colorbar(label = 'Ratio Sharpe (rf=0)')
plt.xlabel('Volatilidad de la cartera')
plt.ylabel('Rendimientos de la cartera')
plt.show()

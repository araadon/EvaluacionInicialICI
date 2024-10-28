import pandas as pd
import numpy as np
from scipy.stats import beta, norm

# Cargar los datos desde el archivo CSV
df = pd.read_csv('/Users/arantzareyesarredondo/Desktop/SuperMarketData.csv')
print(df.head())

# Paso 1: Convertir las ventas de dólares a pesos y normalizar
sales = np.array(df["Sales"]) * 19.88  # convertir de dólares a pesos
max_sales = np.max(sales)
min_sales = np.min(sales)
sales_norm = (1 / (max_sales - min_sales)) * (sales - min_sales)  # normalización
print(f"Ventas normalizadas (primeras 5): {sales_norm[:5]}")

# Paso 2: Ajustar los parámetros de la distribución beta
a, b, _, _ = beta.fit(sales_norm)
print(f"Parámetros estimados de la distribución beta: a={a}, b={b}")

# Paso 3: Calcular media y desviación estándar de la distribución beta
mu_norm = a / (a + b)
var_norm = (a * b) / ((a + b)**2 * (a + b + 1))
std_norm = np.sqrt(var_norm)
print(f"Media normalizada: {mu_norm}, Desviación estándar normalizada: {std_norm}")

# Convertir a escala original (no normalizada)
mu = (max_sales - min_sales) * mu_norm + min_sales
var = (max_sales - min_sales)**2 * var_norm
sigma = np.sqrt(var)
print(f"Media no normalizada: {mu}, Desviación estándar no normalizada: {sigma}")

# Paso 4: Calcular los gastos operativos
dias_trab = 24
fact = 1.15  # factor de aumento de salario del 15%

# Salarios de los empleados
sal_cajeros = 258.25
num_cajeros = 30
tot_sal_cajeros = sal_cajeros * num_cajeros * dias_trab * fact

sal_conserjes = 5000
num_conserjes = 20
tot_sal_conserjes = sal_conserjes * num_conserjes * fact

tot_sal_gerente = 100000

sub_gerente = 45000
num_sub_gerente = 4
tot_sal_sub_gerente = sub_gerente * num_sub_gerente

sal_almacenista = 262.13
num_almacenista = 40
tot_sal_almacenista = sal_almacenista * num_almacenista * dias_trab * fact

g_pasillo = 264.65
num_pasillo = 60
tot_sal_pasillo = g_pasillo * num_pasillo * dias_trab * fact

# Cálculo de la nómina total
nomina_total = (tot_sal_cajeros + tot_sal_conserjes + tot_sal_gerente +
                tot_sal_sub_gerente + tot_sal_almacenista + tot_sal_pasillo)
print(f"Nómina total: {nomina_total}")

# Cálculo del gasto de luz
gasto_luz = 120 * 2000 * 12 * 2.3 * 30
print(f"Gasto de luz: {gasto_luz}")

# Gastos totales
gastos_tot = gasto_luz + nomina_total
print(f"Gastos totales: {gastos_tot}")

# Paso 5: Calcular el número de ventas necesarias
ingreso_deseado = 1500000
ingreso_total = gastos_tot + ingreso_deseado

# Cálculo usando distribución normal
omega = norm.ppf(0.01)  # cuantil de la normal para 99% de probabilidad
a_ = mu**2
b_ = -2 * mu * ingreso_total - omega**2 * sigma**2
c_ = ingreso_total**2

N1 = (-b_ + np.sqrt(b_**2 - 4 * a_ * c_)) / (2 * a_)
N2 = (-b_ - np.sqrt(b_**2 - 4 * a_ * c_)) / (2 * a_)
N = max(N1, N2)  # tomar el valor positivo
print(f"Número de ventas necesarias: {N}")

# Paso 6: Calcular el porcentaje de habitantes a convencer
habitantes_comunidad = 40000
porcentaje_habitantes = (N / (habitantes_comunidad * 4)) * 100
print(f"Porcentaje de habitantes a convencer: {porcentaje_habitantes}%")

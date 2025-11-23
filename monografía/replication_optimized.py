# %%
import numpy as np
import matplotlib.pyplot as plt
from numba import njit

# %%
@njit
def simular_ronda_numba(aspiracion_alfa, suscept, dinero_donado_delta, PHI, Q, W_NOR_POS, W_NOR_NEG, W_EMP_POS, W_EMP_NEG):
    N = len(aspiracion_alfa)
    indices = np.arange(N)
    np.random.shuffle(indices)
    
    mid = N // 2
    indices_dictadores = indices[:mid]
    indices_receptores = indices[mid:]
    
    # 1. Dictators generate theta
    dinero_donado_theta = np.zeros(mid)
    for i in range(mid):
        idx = indices_dictadores[i]
        ruido = np.random.normal(0, 0.05)
        val = aspiracion_alfa[idx] * (1 + ruido)
        # clamp
        if val < 0:
            val = 0
        if val > PHI:
            val = PHI
        dinero_donado_theta[i] = val
        
    # 2. Normative interaction
    for i in range(mid):
        idx_dictador = indices_dictadores[i]
        
        # Sample Q
        sum_donations = 0.0
        for _ in range(Q):
            # Rejection sampling to pick distinct other dictator
            while True:
                rand_idx = np.random.randint(0, mid)
                if rand_idx != i:
                    sum_donations += dinero_donado_theta[rand_idx]
                    break
        promedio = sum_donations / Q
        
        estimulo = (promedio - aspiracion_alfa[idx_dictador]) / PHI
        
        # Update logic
        if estimulo >= 0:
            suscept[idx_dictador] *= (1 + W_NOR_POS * estimulo)
        else:
            suscept[idx_dictador] *= (1 + W_NOR_NEG * estimulo)
            
        # Correct suscept
        if suscept[idx_dictador] < 0:
            suscept[idx_dictador] = 0
        if suscept[idx_dictador] > 0.5:
            suscept[idx_dictador] = 0.5
        
        # Update aspiration
        if estimulo >= 0:
            aspiracion_alfa[idx_dictador] += (PHI - aspiracion_alfa[idx_dictador]) * suscept[idx_dictador] * estimulo
        else:
            aspiracion_alfa[idx_dictador] += aspiracion_alfa[idx_dictador] * suscept[idx_dictador] * estimulo
            
    # 3. Empirical interaction
    for i in range(mid):
        idx_dictador = indices_dictadores[i]
        idx_receptor = indices_receptores[i]
        
        # Dictator generates delta
        ruido_emp = np.random.normal(0, 0.1)
        val_delta = (PHI - aspiracion_alfa[idx_dictador]) * (1 + ruido_emp)
        if val_delta < 0:
            val_delta = 0
        if val_delta > PHI:
            val_delta = PHI
        
        # Update persistent array
        dinero_donado_delta[idx_dictador] = val_delta
        
        # Receptor receives
        donacion_recibida = val_delta
        
        # Receptor updates empirical
        estimulo_emp = (donacion_recibida - aspiracion_alfa[idx_receptor]) / PHI
        
        if estimulo_emp >= 0:
            suscept[idx_receptor] *= (1 + W_EMP_POS * estimulo_emp)
        else:
            suscept[idx_receptor] *= (1 + W_EMP_NEG * estimulo_emp)
            
        # Correct suscept
        if suscept[idx_receptor] < 0:
            suscept[idx_receptor] = 0
        if suscept[idx_receptor] > 0.5:
            suscept[idx_receptor] = 0.5
        
        if estimulo_emp >= 0:
            aspiracion_alfa[idx_receptor] += (PHI - aspiracion_alfa[idx_receptor]) * suscept[idx_receptor] * estimulo_emp
        else:
            aspiracion_alfa[idx_receptor] += aspiracion_alfa[idx_receptor] * suscept[idx_receptor] * estimulo_emp

# %%
PHI = 1 # dinero
Q   = 2 # cantidad de interacciones normativas
N   = 1000 # cantidad de jugadores
W_NOR_POS = 1 # peso para estímulos normativos positivos
W_NOR_NEG = 0 # peso para estímulos normativos negativos
W_EMP_POS = 0 # peso para estímulos empíricos positivos
W_EMP_NEG = 0 # peso para estímulos empíricos negativos

# Initialization
aspiracion_alfa = np.random.uniform(PHI/2, PHI, N)
suscept = np.random.uniform(0, 0.5, N)
dinero_donado_delta = np.zeros(N)

# %%
for ronda in range(100):
    simular_ronda_numba(aspiracion_alfa, suscept, dinero_donado_delta, PHI, Q, W_NOR_POS, W_NOR_NEG, W_EMP_POS, W_EMP_NEG)
    print(f"Ronda {ronda+1} completada.",end='\r')
    
plata_donada = dinero_donado_delta
susceptibilidades = suscept

plt.figure(figsize=(12, 5))

# Plot 1: Distribución de donaciones
plt.subplot(1, 2, 1)
plt.hist(plata_donada,bins=10)
# plot a vertical line at the mean
plt.axvline(np.mean(plata_donada), color='red', linestyle='dashed', linewidth=1)
plt.xlabel('Dinero donado')
plt.ylabel('Frecuencia')
plt.xlim(0, PHI)
plt.title(f'Distribución de donaciones\nW_nor_pos={W_NOR_POS}, W_nor_neg={W_NOR_NEG}, W_emp_pos={W_EMP_POS}, W_emp_neg={W_EMP_NEG}')

# Plot 2: Distribución de susceptibilidades
plt.subplot(1, 2, 2)
plt.hist(susceptibilidades,bins=10)
# plot a vertical line at the mean
plt.axvline(np.mean(susceptibilidades), color='red', linestyle='dashed', linewidth=1)
plt.xlabel('Susceptibilidad')
plt.ylabel('Frecuencia')
plt.title('Distribución de susceptibilidades')

plt.tight_layout()
plt.show()

print()
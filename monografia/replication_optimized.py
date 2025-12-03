import numpy as np
from numba import njit
import json

@njit
def juego_completo_numba(W_NOR_POS, W_NOR_NEG, W_EMP_POS, W_EMP_NEG,seed,
                        PHI=1, Q=2,rondas=100,N=1000):
    
    np.random.seed(seed)
    aspiracion_alfa = np.random.uniform(PHI/2, PHI, N)
    suscept = np.random.uniform(0, 0.5, N)
    dinero_donado_delta = np.zeros(N)
    suscept_history = np.zeros(rondas)
    
    for r in range(rondas):
    
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
        
        suscept_history[r] = np.mean(suscept)

    return suscept, dinero_donado_delta, suscept_history


RANGOS = [0,1/3,2/3,1]

resultados = []

seeds = [17,19,23,43,53,87,91,101,103,113]

for W_NOR_POS in RANGOS:
    for W_NOR_NEG in RANGOS:
        for W_EMP_POS in RANGOS:
            for W_EMP_NEG in RANGOS:
                dineros_donados = []
                susceptibilidades = []
                suscept_histories = []
                for seed in seeds:
                    print(f"Corriendo simulación con W_NOR_POS={W_NOR_POS:.2f}, W_NOR_NEG={W_NOR_NEG:.2f}, W_EMP_POS={W_EMP_POS:.2f}, W_EMP_NEG={W_EMP_NEG:.2f} semilla {seed}")
                    suscept, dinero_donado_delta, suscept_history = juego_completo_numba(W_NOR_POS=W_NOR_POS, 
                                                                        W_NOR_NEG=W_NOR_NEG, 
                                                                        W_EMP_POS=W_EMP_POS, 
                                                                        W_EMP_NEG=W_EMP_NEG,
                                                                        seed=seed)
                    
                    dineros_donados.append(np.sort(dinero_donado_delta))
                    susceptibilidades.append(np.sort(suscept))
                    suscept_histories.append(suscept_history)
                
                dinero_donado_theta_final = np.mean(dineros_donados,axis=0)
                suscept_final = np.mean(susceptibilidades,axis=0)
                suscept_history_final = np.mean(suscept_histories, axis=0)
                suscept_history_final = [float(x) for x in suscept_history_final]
                
                resultados.append(
                    {
                        "W_NOR_POS": W_NOR_POS,
                        "W_NOR_NEG": W_NOR_NEG,
                        "W_EMP_POS": W_EMP_POS,
                        "W_EMP_NEG": W_EMP_NEG,
                        'susceptibilidades': list(suscept_final),
                        'donaciones_delta': list(dinero_donado_theta_final),
                        'suscept_history': list(suscept_history_final),
                    }
                )
                
file_name = "simulacion_10_seeds_susceptibility.json"
                
print(f"Guardando resultados en '{file_name}'")
with open(f"monografia/{file_name}", "w") as f:
    json.dump(resultados, f)
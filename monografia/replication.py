# %%
import numpy as np
import matplotlib.pyplot as plt

# %%
PHI = 1 # dinero
Q   = 2 # cantidad de interacciones normativas
N   = 1000 # cantidad de jugadores
W_NOR_POS = 1 # peso para estímulos normativos positivos
W_NOR_NEG = 1 # peso para estímulos normativos negativos
W_EMP_POS = 0 # peso para estímulos empíricos positivos
W_EMP_NEG = 0 # peso para estímulos empíricos negativos

# %%
class Jugador:
    def __init__(self, dinero: int = PHI,W_nor_pos=W_NOR_POS,W_nor_neg=W_NOR_NEG,W_emp_pos=W_EMP_POS,W_emp_neg=W_EMP_NEG):
        self.dinero = dinero
        self.dictador           = False
        self.aspiracion_alfa    = np.random.uniform(dinero/2, dinero) # aspiración inicial
        self.suscept            = np.random.uniform(0,0.5) # modelamos una sola susceptibilidad inicial
        
        # Pesos para estímulos normativos positivos y negativos
        self.W_nor_pos = W_nor_pos  # peso para estímulos normativos positivos
        self.W_nor_neg = W_nor_neg  # peso para estímulos normativos negativos
        self.W_emp_pos = W_emp_pos  # peso para estímulos empíricos positivos
        self.W_emp_neg = W_emp_neg  # peso para estímulos empíricos negativos
        
    def generar_donacion_theta(self):
        ruido_delta_norm   = np.random.normal(0, .05) # ruido en la actualización de la aspiración
        self.dinero_donado_theta: float = max(0,min(self.dinero,self.aspiracion_alfa*(1+ruido_delta_norm)))
        
    def interaccion_normativa(self,dictadores:list,interacciones=Q):
        "generamos interacción normativa entre dictadores y devolvemos la actualización de la aspiración del receptor"
        dinero_otros_dictadores = np.mean(np.random.choice([dictad.dinero_donado_theta for dictad in dictadores],size=interacciones,replace=False))
        
        self.estimulo_normativo = (dinero_otros_dictadores - self.aspiracion_alfa)/self.dinero
        
    def actualizar_aspiracion_normativa(self,dictadores:list):
        """
        Actualizamos la aspiración del receptor según la interacción normativa
        Basado en la fórmula (5): I^nor_{j,t} = I^nor_{j,t-1} * (1 + W^{nor,pos/neg} * ξ^nor_{j,t})
        
        - Si ξ^nor_{j,t} ≥ 0: usamos W^{nor,pos} (estímulo normativo positivo)
        - Si ξ^nor_{j,t} < 0: usamos W^{nor,neg} (estímulo normativo negativo)
        """
        self.interaccion_normativa(dictadores=dictadores)
        
        # Estímulo positivo: la aspiración tiende a aumentar
        if self.estimulo_normativo>=0:
            self.suscept*= (1 + self.W_nor_pos * self.estimulo_normativo)
            self.corregir_susceptibilidades()
            self.aspiracion_alfa+=(self.dinero-self.aspiracion_alfa)*self.suscept * self.estimulo_normativo
        else:
            self.suscept*= (1 + self.W_nor_neg * self.estimulo_normativo)
            self.corregir_susceptibilidades()
            self.aspiracion_alfa+=self.aspiracion_alfa*self.suscept * self.estimulo_normativo
            
    def generar_donacion(self):
        ruido_delta_emp    = np.random.normal(0, .1) # ruido en la actualización de la aspiración
        self.dinero_donado_delta = max(0,min(self.dinero,(self.dinero-self.aspiracion_alfa)*(1+ruido_delta_emp)))        
        
    def recibir_donacion(self,donacion):
        self.donacion_recibida_pi = donacion
        
    def interaccion_empirica(self):
        self.estimulo_empirico = (self.donacion_recibida_pi - self.aspiracion_alfa)/self.dinero
        
    def actualizar_aspiracion_empirica(self):
        """
        Actualizamos la aspiración del receptor según la interacción empírica
        Basado en la fórmula (4): I^emp_{j,t} = I^emp_{j,t-1} * (1 + W^{emp,pos/neg} * ξ^emp_{j,t})
        
        - Si ξ^emp_{j,t} ≥ 0: usamos W^{emp,pos} (estímulo empírico positivo)
        - Si ξ^emp_{j,t} < 0: usamos W^{emp,neg} (estímulo empírico negativo)
        """
        self.interaccion_empirica()        

        if self.estimulo_empirico>=0:
            self.suscept*= (1 + self.W_emp_pos * self.estimulo_empirico)
            self.corregir_susceptibilidades()
            self.aspiracion_alfa+=(self.dinero-self.aspiracion_alfa)*self.suscept * self.estimulo_empirico
        else:
            self.suscept*= (1 + self.W_emp_neg * self.estimulo_empirico)
            self.corregir_susceptibilidades()
            self.aspiracion_alfa+=(self.aspiracion_alfa)*self.suscept * self.estimulo_empirico
    
    def corregir_susceptibilidades(self):
        """
        Corrige las susceptibilidades normativa y empírica para asegurar que
        ambas sean >= 0 y su suma sea <= 1, manteniendo su proporción
        si la suma excede 1.
        """
        # Rule 1: Handle negative values by clipping them to 0.
        # We use np.maximum for a concise way to say max(0, value).
        self.suscept = np.maximum(0, self.suscept)
        # Rule 2: Handle the sum constraint.
        total_suscept = self.suscept + self.suscept

        if total_suscept > 1:
            # This is the proportional correction.
            # If total_suscept is 0 (only if both were 0), this would cause a
            # ZeroDivisionError. However, since we checked for total_suscept > 1,
            # this is safe.
            self.suscept = self.suscept / total_suscept
            self.suscept = self.suscept / total_suscept

# %%
def ronda_juego(jugadores: dict[int, Jugador]):
    # Convertir diccionario a lista de tuplas (key, jugador) para trabajar más fácilmente
    jugadores_items = list(jugadores.items())
    
    # Seleccionar dictadores aleatoriamente (mantener keys)
    indices_dictadores = np.random.choice(len(jugadores_items), N//2, replace=False)
    dictadores_items = [jugadores_items[i] for i in indices_dictadores]    
    
    # Crear lista de receptores (los que no son dictadores)
    receptores_items = [item for i, item in enumerate(jugadores_items) if i not in indices_dictadores]
    
    # Mezclar manteniendo las tuplas (key, jugador)
    np.random.shuffle(dictadores_items)
    np.random.shuffle(receptores_items)
    
    for (key_dict, dictador) in dictadores_items:
        dictador.generar_donacion_theta()
        
    # Aparear dictadores y receptores (manteniendo keys)
    pares = list(zip(dictadores_items, receptores_items))
    
    # Interacción normativa
    for (key_dict, dictador), (key_rec, receptor) in pares:
        # Traer la lista de todos los demás dictadores (solo los objetos, no las keys)
        otros_dictadores = [d[1] for d in dictadores_items if d[0] != key_dict]
        dictador.actualizar_aspiracion_normativa(otros_dictadores)
    
    # Diccionario final para retornar
    jugadores_final = {}
    
    for (key_dict, dictador), (key_rec, receptor) in pares:
        dictador.dictador = True
        receptor.dictador = False
        dictador.generar_donacion()
        receptor.recibir_donacion(dictador.dinero_donado_delta)
        receptor.interaccion_empirica()
        receptor.actualizar_aspiracion_empirica()

        # Agregar jugadores al diccionario final manteniendo sus keys
        jugadores_final[key_dict] = dictador
        jugadores_final[key_rec] = receptor
        
    return jugadores_final

# %%
jugadores = {i:Jugador() for i in range(N)}

# %%
for ronda in range(100):
    jugadores = ronda_juego(jugadores)
    # reorder jugadores by key
    jugadores = {k: jugadores[k] for k in sorted(jugadores.keys())}
    
    print(f"Ronda {ronda+1} completada.",end='\r')
    
plata_donada = [jug.dinero_donado_delta for k,jug in jugadores.items()]
susceptibilidades = [jug.suscept for k,jug in jugadores.items()]

plt.figure(figsize=(12, 5))

# Plot 1: Distribución de donaciones
plt.subplot(1, 2, 1)
plt.hist(plata_donada)
# plot a vertical line at the mean
plt.axvline(np.mean(plata_donada), color='red', linestyle='dashed', linewidth=1)
plt.xlabel('Dinero donado')
plt.ylabel('Frecuencia')
plt.xlim(0, PHI)
plt.title(f'Distribución de donaciones\nW_nor_pos={W_NOR_POS}, W_nor_neg={W_NOR_NEG}, W_emp_pos={W_EMP_POS}, W_emp_neg={W_EMP_NEG}')

# Plot 2: Distribución de susceptibilidades
plt.subplot(1, 2, 2)
plt.hist(susceptibilidades)
# plot a vertical line at the mean
plt.axvline(np.mean(susceptibilidades), color='red', linestyle='dashed', linewidth=1)
plt.xlabel('Susceptibilidad')
plt.ylabel('Frecuencia')
plt.title('Distribución de susceptibilidades')

plt.tight_layout()
plt.show()

print()
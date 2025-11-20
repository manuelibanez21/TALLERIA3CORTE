# Taller Tercer Corte IA

## 1. Aprendizaje por Refuerzo (Reinforcement Learning)

### 1.a ¿Cómo puede un agente aprender a tomar decisiones óptimas en un entorno incierto?

Un agente de aprendizaje por refuerzo aprende a tomar decisiones óptimas
en entornos inciertos mediante un proceso iterativo de **prueba y
error** basado en los siguientes principios fundamentales:

### 🔄 Proceso de Aprendizaje

-   **Interacción continua**: El agente ejecuta acciones, observa
    estados y recibe recompensas.\
-   **Retroalimentación diferida**: Las consecuencias de las acciones se
    manifiestan a lo largo del tiempo.\
-   **Balance exploración-explotación**: Equilibrio entre probar nuevas
    acciones y usar conocimientos previos.

### 🧠 Mecanismos de Aprendizaje

``` python
# Pseudocódigo del proceso de aprendizaje
estado_actual = entorno.estado_inicial()
for episodio in range(max_episodios):
    acción = política(estado_actual)  # Decisión basada en política actual
    nuevo_estado, recompensa = entorno.ejecutar(acción)
    actualizar_política(estado_actual, acción, recompensa, nuevo_estado)
    estado_actual = nuevo_estado
```

### 📈 Estrategias para Incertidumbre

-   **Aprendizaje de valores**: Estimación de recompensas futuras
    esperadas.\
-   **Actualización temporal-diferencia**: Ajuste incremental de
    predicciones.\
-   **Políticas estocásticas**: Probabilidades de acción que permiten
    exploración.

------------------------------------------------------------------------

## 1.b Tipos de Algoritmos de Aprendizaje por Refuerzo

### 🎯 Métodos Basados en Valor (Value-Based)

#### Arquitectura

    ┌─────────────┐ → ┌──────────────┐ → ┌──────────────┐
    │   Estado    │   │  Función de  │   │    Valor      │
    │    (S)      │   │    Valor     │   │    Q(s,a)     │
    └─────────────┘   └──────────────┘   └──────────────┘

#### Componentes

-   **Función Q**: Q(s,a) = recompensa esperada.\
-   **Tabla Q o red neuronal**: Donde se almacenan o aproximan valores.\
-   **Política ε-greedy**: Controla exploración-explotación.

#### Algoritmos Representativos

-   Q-Learning\
-   SARSA\
-   Deep Q-Network (DQN)

------------------------------------------------------------------------

### 🎯 Métodos Basados en Política (Policy-Based)

#### Arquitectura

    ┌─────────────┐ → ┌──────────────┐ → ┌──────────────────┐
    │   Estado    │   │   Política   │   │ Prob. de acción   │
    │    (S)      │   │   π(a|s)     │   │      π(a|s)        │
    └─────────────┘   └──────────────┘   └──────────────────┘

#### Componentes

-   **Política parametrizada**\
-   **Gradiente de política**\
-   **Selección probabilística de acciones**

#### Algoritmos Representativos

-   REINFORCE\
-   Policy Gradient\
-   Natural Policy Gradient

------------------------------------------------------------------------

### 🎯 Métodos Actor-Crítico (Actor-Critic)

#### Arquitectura

               ┌──────────────┐ → Acción
    Estado →   │    ACTOR     │
               │ (Política)   │
               └──────────────┘
                     │
                     ↓
               ┌──────────────┐
               │   CRÍTICO     │ ← Recompensa
               │   (Valor)     │
               └──────────────┘

#### Componentes

-   **Actor**: Selecciona acciones.\
-   **Crítico**: Evalúa el valor del estado o acción.\
-   **Ventaja**: A(s,a) = Q(s,a) - V(s).

#### Algoritmos Representativos

-   A2C\
-   A3C\
-   PPO\
-   SAC

------------------------------------------------------------------------

## 1.c Aplicaciones en la Industria

### 🏭 Manufactura y Logística

``` python
# Ejemplo: Optimización de línea de producción
class OptimizacionProduccion:
    def tomar_decision(self, estado):
        if estado == "alta_demanda":
            return "aumentar_velocidad"
        elif estado == "falla_inminente":
            return "mantenimiento_preventivo"
```

Aplicaciones: - Robótica industrial\
- Gestión de inventarios\
- Control de calidad\
- Mantenimiento predictivo

------------------------------------------------------------------------

### 📡 Telecomunicaciones

``` python
class RouterInteligente:
    def optimizar_ruta(self, trafico_actual, estado_red):
        return mejor_ruta
```

Aplicaciones: - Gestión de espectro\
- Optimización de tráfico\
- Routing inteligente

------------------------------------------------------------------------

### 🚗 Vehículos Autónomos

-   Navegación\
-   Control\
-   Decisiones en tiempo real

------------------------------------------------------------------------

### ⚡ Energía

-   Smart Grids\
-   Distribución óptima\
-   Predicción de demanda

------------------------------------------------------------------------

# 2. Algoritmo Bayesiano para Detección de SPAM

## 📊 Datos Dado:

-   P(Spam)=0.3\
-   P(No Spam)=0.7\
-   P("gratis"\|Spam)=0.8\
-   P("gratis"\|No Spam)=0.1

------------------------------------------------------------------------

## 🧮 Solución con Bayes

### Probabilidad Total

    P("gratis") = 0.8×0.3 + 0.1×0.7  
                = 0.24 + 0.07  
                = 0.31

### Resultado Final

    P(Spam|"gratis") = 0.24 / 0.31 ≈ 0.774

------------------------------------------------------------------------

## 💻 Implementación

``` python
class ClasificadorBayesSpam:
    def __init__(self):
        self.p_spam = 0.3
        self.p_no_spam = 0.7
        self.palabras_clave = {
            'gratis': {'spam': 0.8, 'no_spam': 0.1}
        }

    def calcular_probabilidad_spam(self, palabras):
        p_spam_val = self.p_spam
        p_no_spam_val = self.p_no_spam
        for palabra in palabras:
            if palabra in self.palabras_clave:
                p_spam_val *= self.palabras_clave[palabra]['spam']
                p_no_spam_val *= self.palabras_clave[palabra]['no_spam']
        total = p_spam_val + p_no_spam_val
        return p_spam_val / total

clasificador = ClasificadorBayesSpam()
print(clasificador.calcular_probabilidad_spam(['gratis']))
```

------------------------------------------------------------------------

# 3. Algoritmos Más Usados en Academia e Industria

## 🤖 Redes Neuronales (Deep Learning)

### CNN

Aplicaciones: - Imágenes\
- Visión\
- Diagnóstico médico

------------------------------------------------------------------------

### RNN, LSTM, GRU

Aplicaciones: - Texto\
- Series temporales\
- Predicción financiera

------------------------------------------------------------------------

## 🌳 Algoritmos de Ensamblaje

### Random Forest

### XGBoost / LightGBM / CatBoost

------------------------------------------------------------------------

## 🔍 No Supervisado

### K-Means

### DBSCAN

------------------------------------------------------------------------

## 🦉 Probabilísticos

### Naive Bayes

------------------------------------------------------------------------

## 🎯 Máquinas de Vectores de Soporte (SVM)

------------------------------------------------------------------------

## 🚀 Vanguardia

### Transformers

### Deep Reinforcement Learning

------------------------------------------------------------------------

# 📊 Tabla Comparativa

  Categoría          Algoritmos               Fortalezas        Aplicaciones
  ------------------ ------------------------ ----------------- --------------------
  Redes Neuronales   CNN, RNN, LSTM           Alta precisión    Visión, NLP
  Ensamblaje         Random Forest, XGBoost   Robustez          Datos tabulares
  No Supervisado     K-Means, DBSCAN          Exploración       Segmentación
  Probabilísticos    Naive Bayes              Simple y rápido   Spam
  Vanguardia         Transformers, RL         Estado del arte   Multimodal, robots

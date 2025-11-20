# TALLERIA3CORTE
# Aprendizaje por Refuerzo y Algoritmos Bayesianos

## 1. Aprendizaje por Refuerzo (Reinforcement Learning - RL)

El aprendizaje por refuerzo es un paradigma de la inteligencia artificial donde un agente aprende a tomar decisiones mediante la interacción con un entorno, recibiendo recompensas o castigos según sus acciones.

### 1.a ¿Cómo puede un agente aprender a tomar decisiones óptimas en un entorno incierto?

Un agente aprende mediante un proceso de prueba y error, siguiendo estos principios:

#### 1. Interacción continua
El agente realiza acciones dentro del entorno y recibe:
- Un estado (información del entorno)
- Una recompensa (resultado positivo o negativo)
- El nuevo estado tras la acción

#### 2. Función de recompensa
El agente ajusta su comportamiento buscando maximizar la recompensa acumulada a largo plazo.

#### 3. Exploración vs. explotación
En entornos inciertos, el agente debe:
- **Explorar**: probar acciones nuevas para descubrir mejores resultados
- **Explotar**: usar las acciones que ya sabe que dan buenas recompensas

#### 4. Evaluación de políticas
El agente aprende una política óptima, es decir, un conjunto de reglas para elegir la acción óptima en cada estado.

#### 5. Convergencia
Con suficientes iteraciones y retroalimentación, el agente converge a decisiones óptimas incluso en escenarios inciertos.

### 1.b Tipos de algoritmos de aprendizaje por refuerzo y sus arquitecturas

Los algoritmos de RL se dividen en tres categorías principales:

#### 🎯 1. Métodos basados en valores (Value-Based)
El agente aprende una función de valor, que estima qué tan buena es una acción en un estado.

**Ejemplos:**
- Q-Learning
- Deep Q-Network (DQN)

**Arquitectura:**
- Estados (S)
- Acciones (A)
- Q-Table o red neuronal (para DQN)
- Política ε-greedy
- Recompensa (R)

#### 🎯 2. Métodos basados en políticas (Policy-Based)
El agente aprende directamente una política que asigna probabilidades a cada acción.

**Ejemplos:**
- REINFORCE
- Policy Gradient Methods

**Arquitectura:**
- Estados
- Red neuronal para la política π(a|s)
- Función de pérdida basada en recompensa
- Actualización por gradiente

#### 🎯 3. Métodos actor-crítico (Actor-Critic)
Combinan los dos anteriores:
- **Actor**: decide la acción
- **Crítico**: evalúa la calidad de la acción tomada

**Ejemplos:**
- A2C (Advantage Actor-Critic)
- A3C (Asynchronous Advantage Actor-Critic)
- PPO (Proximal Policy Optimization)

**Arquitectura:**
- Red neuronal compartida o dos redes:
  - Actor: política
  - Crítico: valor (V) o ventaja (A)
- Función de pérdida híbrida

### 1.c ¿Para qué se utilizan estos algoritmos en la industria?

#### Industria 4.0
- Optimización de procesos productivos
- Control inteligente de robots
- Sistemas de control autónomos
- Predicción de mantenimiento (mantenimiento predictivo)

#### Telecomunicaciones
- Optimización dinámica de redes
- Gestión de tráfico
- Selección automática de rutas óptimas en routers
- Asignación de espectro en redes 5G

#### Finanzas
- Trading algorítmico
- Evaluación de riesgo

#### Transporte
- Vehículos autónomos
- Ruteo inteligente

#### Energía
- Optimización de redes eléctricas
- Control de consumo energético en edificios

## 2. Algoritmo bayesiano para clasificación de SPAM

Se desea calcular:  
**P(Spam∣gratis)**

### Datos:
- P(Spam) = 0.3
- P(¬Spam) = 0.7
- P(gratis | Spam) = 0.8
- P(gratis | ¬Spam) = 0.1

### Solución usando el Teorema de Bayes:

\[
P(Spam∣gratis) = \frac{P(gratis∣Spam) \cdot P(Spam)}{P(gratis)}
\]

#### Primero calculamos P(gratis):
\[
\begin{align*}
P(gratis) &= P(gratis∣Spam)P(Spam) + P(gratis∣¬Spam)P(¬Spam) \\
&= 0.8 \times 0.3 + 0.1 \times 0.7 \\
&= 0.24 + 0.07 = 0.31
\end{align*}
\]

#### Ahora calculamos la probabilidad final:
\[
\begin{align*}
P(Spam∣gratis) &= \frac{0.8 \times 0.3}{0.31} \\
&= \frac{0.24}{0.31} \approx 0.774
\end{align*}
\]

### Algoritmo simple en pseudocódigo

```python
function prob_spam(p_spam, p_not_spam, p_gratis_spam, p_gratis_not_spam):
    p_gratis = p_gratis_spam * p_spam + p_gratis_not_spam * p_not_spam
    result = (p_gratis_spam * p_spam) / p_gratis
    return result

# Datos del problema
p_spam = 0.3
p_not_spam = 0.7
p_gratis_spam = 0.8
p_gratis_not_spam = 0.1

print(prob_spam(p_spam, p_not_spam, p_gratis_spam, p_gratis_not_spam))
Resultado:
👉 ≈ 0.774 → 77.4% de probabilidad de que sea SPAM

3. Algoritmos más utilizados actualmente y sus características
Algoritmo	Tipo	Uso Principal	Características
Deep Neural Networks (DNN)	Supervisado	Clasificación y regresión general	Profundas, usan backprop; alta precisión
Convolutional Neural Networks (CNN)	Supervisado	Visión artificial	Extraen características espaciales; usadas en imágenes, CCTV, robótica
Recurrent Neural Networks (RNN)	Supervisado	Series de tiempo, texto	Modelo secuencial con memoria
LSTM / GRU	Supervisado	Modelos temporales avanzados	Resuelven desvanecimiento de gradiente
Transformers	Supervisado / autosupervisado	NLP e IA modernas (GPT, BERT)	Atención multi-cabeza, paralelización
Random Forest	Supervisado	Clasificación tabular	Ensamble de árboles, robusto, fácil entrenamiento
Gradient Boosting (XGBoost, LightGBM)	Supervisado	Tablas de datos con alta precisión	Altamente usado en industria
K-Means	No supervisado	Clustering	Simple y rápido para segmentación
DBSCAN	No supervisado	Detección de anomalías	Robusto a ruido
Aprendizaje por refuerzo (RL)	RL	Sistemas autónomos	Optimiza decisiones secuenciales
SVM	Supervisado	Datos de alta dimensión	Clasificación eficiente con márgenes
Naive Bayes	Supervisado	Clasificación simple (ej. SPAM)	Rápido, probabilístico

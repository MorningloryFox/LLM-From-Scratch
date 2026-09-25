
# 04. Positional Encoding com RoPE (Rotary Position Embedding) 🔄

## 🎯 1. Por que precisamos de Informação de Posição?

O mecanismo de atenção original (*Self-Attention*) trata as palavras de uma frase como um conjunto não ordenado (um "saco de palavras"). Do ponto de vista puramente matemático, a operação de atenção isolada é **invariante à ordem**.

Sem uma forma de informar a posição de cada token, o modelo interpretaria as duas frases abaixo exatamente da mesma maneira, embora tenham sentidos completamente opostos:
1. *"O cão viu o gato"*
2. *"O gato viu o cão"*

Para resolver essa limitação, precisamos injetar uma **informação posicional** nos vetores antes de calcularmos a atenção.

---

## 🎡 2. O Conceito do RoPE (Rotary Position Embedding)

O **RoPE** é a técnica moderna de codificação posicional utilizada pelos modelos de linguagem de ponta (como Llama 2/3, Mistral e Qwen).

Em vez de simplesmente somar um número à representação da palavra (como se fazia nos modelos Transformers antigos), o RoPE aplica uma **rotação geométrica** nos vetores de **Query ($Q$)** e **Key ($K$)** no espaço multidimensional, onde o ângulo da rotação é diretamente proporcional à **posição $m$** do token na sequência.


```

Vetor Original (Sem Posição)  ──>  [ Rotação de Ângulo (m · θ) ]  ──>  Vetor Rotacionado (Com Posição)

```

### Propriedades fundamentais do RoPE:
1. **Preservação da Norma (Módulo):** Rotacionar um vetor altera apenas a sua direção no espaço, mas **mantém o seu comprimento (norma) intacto**. Isso evita distorções no módulo do vetor durante os cálculos de atenção.
2. **Atenção Relativa Natural:** Quando calculamos o produto escalar entre a Query da posição $m$ ($Q_m$) e a Key da posição $n$ ($K_n$), a matemática da rotação faz com que o resultado dependa exclusivamente da **distância relativa $(m - n)$** entre os dois tokens, e não de suas posições absolutas isoladas.

---

## 🧮 3. Formulação Matemática Detalhada

Para entender a matemática do RoPE, vamos analisar a fórmula de rotação no espaço 2D e, em seguida, entender como ela é expandida para espaços de alta dimensão.

### A. Rotação em 2 Dimensões (Espaço 2D)

Considere um par de coordenadas $(x_1, x_2)$ pertencente ao vetor de um token localizado na posição $m$ da frase:

$$R_{\Theta, m} \begin{pmatrix} x_1 \\ x_2 \end{pmatrix} = \begin{pmatrix} \cos(m\theta) & -\sin(m\theta) \\ \sin(m\theta) & \cos(m\theta) \end{pmatrix} \begin{pmatrix} x_1 \\ x_2 \end{pmatrix}$$

#### O que significa cada símbolo da fórmula?
* **$x_1, x_2$:** As duas componentes numéricas do vetor no espaço 2D (ex: a 1ª e a 2ª posição do vetor de um token).
* **$\begin{pmatrix} x_1 \\ x_2 \end{pmatrix}$:** O vetor original em formato de coluna.
* **$m$:** O número inteiro que indica a **posição absoluta** do token na frase (ex: $m=1$ para a primeira palavra, $m=2$ para a segunda, e assim por diante).
* **$\theta$ (Teta):** A frequência base angular de rotação.
* **$m\theta$:** O ângulo total de rotação aplicado ao vetor (a posição $m$ multiplicada pela frequência $\theta$).
* **$\cos(m\theta)$ e $\sin(m\theta)$:** As funções trigonométricas de **cosseno** e **seno** calculadas para o ângulo $m\theta$.
* **$\begin{pmatrix} \cos(m\theta) & -\sin(m\theta) \\ \sin(m\theta) & \cos(m\theta) \end{pmatrix}$:** A **Matriz de Rotação 2D**. Multiplicar qualquer vetor 2D por essa matriz faz com que ele gire em torno da origem por um ângulo de $m\theta$ radianos.
* **$R_{\Theta, m}$:** Símbolo que representa a operação de rotação para a posição $m$ sob o conjunto de frequências $\Theta$.

---

### B. Expansão para Altas Dimensões ($d$-dimensional)

Vetores de embeddings reais não têm apenas 2 dimensões; eles possuem centenas ou milhares de dimensões (ex: $d = 4096$). 

Para aplicar o RoPE em um vetor $d$-dimensional, o algoritmo **agrupa as $d$ dimensões em pares de 2D** e aplica uma rotação independente em cada par, utilizando frequências angulares $\theta_i$ progressivamente menores:

$$\theta_i = 10000^{-2(i-1)/d}$$

#### O que significa cada símbolo?
* **$\theta_i$:** A frequência de rotação calculada para o $i$-ésimo par de dimensões do vetor.
* **$10000$:** Uma constante de base hiperparâmetro (utilizada na arquitetura original dos Transformers) que define a escala das frequências.
* **$i$:** O índice do par de dimensões atual (onde $i = 1, 2, \dots, d/2$).
* **$d$:** O número total de dimensões do vetor de embedding.
* **$-2(i-1)/d$:** O expoente que diminui progressivamente conforme o índice $i$ aumenta.

#### A intuição por trás de $\theta_i$:
Os primeiros pares de dimensões (onde $i$ é pequeno) recebem uma frequência $\theta_i$ **alta** e giram muito rápido a cada posição $m$. Isso ajuda o modelo a capturar **relações de curto alcance** (palavras vizinhas). 

Já os últimos pares de dimensões recebem frequências **muito baixas** e giram devagar, permitindo ao modelo capturar **dependências e contextos de longo alcance** ao longo de todo o texto.

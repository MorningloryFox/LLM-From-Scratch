# 01. Embeddings e Geometria Vetorial 📐

## 🎯 1. Conceito Fundamental

Um **Embedding** é a representação matemática de uma unidade de texto — chamada de **token** (que pode ser uma palavra, parte de uma palavra ou caractere) — na forma de uma lista ordenada de números. A essa lista damos o nome de **vetor**.

Em termos simples, o computador não entende o significado de palavras humanas diretamente; ele só entende números. Para resolver isso, transformamos palavras em vetores localizados em um **espaço denso multidimensional de números reais**:

$$\vec{v} \in \mathbb{R}^d$$

### O que significa cada símbolo da fórmula acima?
* **$\vec{v}$ (Seta sobre a letra $v$):** Representa o vetor de um token específico (por exemplo, a palavra "gato"). A seta indica que é um objeto que possui tanto magnitude (tamanho) quanto direção no espaço.
* **$\in$ (Pertence a):** Indica que o vetor $\vec{v}$ faz parte do conjunto numérico à direita.
* **$\mathbb{R}$ (Números Reais):** Significa que cada valor contido dentro desse vetor pode ser qualquer número real (inteiros, decimais, positivos ou negativos, como $0{,}42$, $-1{,}15$, etc.).
* **$d$ (Dimensão):** Representa o número total de dimensões (ou "atributos") do espaço vetorial. Por exemplo, em modelos reais como os da OpenAI ou Google, $d$ pode ser $1536$ ou $4096$, significando que cada palavra é definida por uma lista de $1536$ a $4096$ números.

### A Geometria do Significado (Semântica)
Nesse espaço multidimensional, a inteligência artificial aprende a posicionar os conceitos de forma geométrica: **palavras com significados ou papéis semânticos parecidos são colocadas próximas umas das outras e apontam para direções similares**. Por exemplo, "gato" e "cachorro" estarão geograficamente próximos, enquanto "gato" e "computador" estarão distantes.

---

## 📐 2. Formulação Matemática

Para que a inteligência artificial consiga comparar duas palavras numericamente e saber o quão semelhantes elas são, precisamos medir a **orientação espacial** entre seus vetores. A principal ferramenta para isso é o cálculo do **ângulo $\theta$ (teta)** entre eles através da **Similaridade de Cosseno**.

Antes de calcular o cosseno em si, precisamos entender dois conceitos intermediários: o **Produto Escalar** e a **Norma Euclidiana**.

---

### A. Produto Escalar (Dot Product)

O produto escalar mede o quanto dois vetores apontam na mesma direção, combinando a multiplicação de cada uma de suas componentes correspondentes.

$$\vec{a} \cdot \vec{b} = \sum_{i=1}^{d} a_i b_i = a_1 b_1 + a_2 b_2 + \dots + a_d b_d$$

#### O que significa cada símbolo?
* **$\vec{a}$ e $\vec{b}$:** Os dois vetores que estamos comparando (ex: vetor da palavra "Rei" e vetor da palavra "Rainha").
* **$\cdot$ (Ponto central):** Símbolo que representa a operação de produto escalar entre dois vetores.
* **$\sum$ (Somatório / Sigma maiúsculo):** Uma instrução matemática para somar uma sequência de elementos.
* **$i=1$ (Índice de início):** Indica que a soma começa na primeira dimensão (ou seja, na posição $1$ da lista de números).
* **$d$ (Limite superior):** O número total de dimensões dos vetores.
* **$a_i$ e $b_i$:** O valor numérico que está na $i$-ésima posição do vetor $\vec{a}$ e do vetor $\vec{b}$, respectivamente.
* **$a_1 b_1 + a_2 b_2 + \dots + a_d b_d$:** A expansão explícita do somatório: multiplicamos a 1ª dimensão de $\vec{a}$ pela 1ª de $\vec{b}$, somamos com a multiplicação da 2ª dimensão de $\vec{a}$ pela 2ª de $\vec{b}$, e assim sucessivamente até a dimensão $d$.

---

### B. Norma Euclidiana (Comprimento ou Módulo)

A norma de um vetor representa a sua distância em relação à origem do espaço $(0,0,\dots,0)$, ou seja, o seu "tamanho" físico real. Essa fórmula é uma extensão direta do **Teorema de Pitágoras** para superfícies com 2, 3 ou milhares de dimensões.

$$\Vert{}\vec{a}\Vert{} = \sqrt{\sum_{i=1}^{d} a_i^2} = \sqrt{a_1^2 + a_2^2 + \dots + a_d^2}$$

#### O que significa cada símbolo?
* **$\Vert{}\vec{a}\Vert{}$ (Barras duplas ao redor do vetor):** Símbolo que indica a operação de **norma** ou **módulo** do vetor $\vec{a}$.
* **$\sqrt{\quad}$ (Raiz Quadrada):** A operação final para obter o comprimento real, desfazendo os quadrados internos.
* **$a_i^2$:** O valor da componente na posição $i$ do vetor elevado ao quadrado ($a_i \times a_i$). Elevar ao quadrado garante que todos os valores fiquem positivos.
* **$a_1^2 + a_2^2 + \dots + a_d^2$:** A expansão da soma dos quadrados de todas as posições do vetor.

---

### C. Similaridade de Cosseno

Em modelos de linguagem, o **tamanho/comprimento** de um vetor (norma) pode ser influenciado pela frequência da palavra ou por ruídos de treino. O que realmente importa para descobrir o **significado** é a **direção** para a qual o vetor aponta. 

A Similaridade de Cosseno isola essa direção eliminando o impacto do comprimento dos vetores. Fazemos isso dividindo o produto escalar pelo produto das normas dos dois vetores:

$$\cos(\theta) = \frac{\vec{a} \cdot \vec{b}}{\Vert{}\vec{a}\Vert{} \Vert{}\vec{b}\Vert{}}$$

#### O que significa cada símbolo?
* **$\theta$ (Letra grega Teta):** É o ângulo formado entre o vetor $\vec{a}$ e o vetor $\vec{b}$ no espaço geométrico.
* **$\cos(\theta)$:** O valor do cosseno desse ângulo, que variará sempre no intervalo entre $-1$ e $1$.
* **$\vec{a} \cdot \vec{b}$:** O produto escalar entre os vetores (calculado no passo A).
* **$\Vert{}\vec{a}\Vert{} \Vert{}\vec{b}\Vert{}$:** O módulo do vetor $\vec{a}$ multiplicado pelo módulo do vetor $\vec{b}$ (calculados no passo B).

#### Interpretação dos Resultados:
* 🟢 **$\cos(\theta) = 1$ ($\theta = 0^\circ$):** Os vetores estão perfeitamente sobrepostos e apontam para a mesma direção. Significa que os dois tokens possuem significado **idêntico ou equivalente**.
* 🟡 **$\cos(\theta) = 0$ ($\theta = 90^\circ$):** Os vetores são perpendiculares (ortogonais) entre si. Significa que os conceitos **não têm relação semântica** entre si.
* 🔴 **$\cos(\theta) = -1$ ($\theta = 180^\circ$):** Os vetores apontam para direções opostas. Significa que os conceitos têm significados **antagônicos ou opostos**.

---

## 🧮 3. Exemplo Numérico Passo a Passo ($d=2$)

Para visualizar a matemática funcionando na prática, vamos simular dois vetores fictícios em um espaço simples de apenas **2 dimensões** ($d=2$):
* $\vec{a}_\text{Rei} = [2, 1]$
* $\vec{b}_\text{Rainha} = [1, 2]$

---

### Passo 1: Calcular o Produto Escalar ($\vec{a} \cdot \vec{b}$)
Multiplicamos as primeiras posições de cada vetor, multiplicamos as segundas posições e somamos os resultados:

$$\vec{a} \cdot \vec{b} = (a_1 \times b_1) + (a_2 \times b_2)$$
$$\vec{a} \cdot \vec{b} = (2 \times 1) + (1 \times 2)$$
$$\vec{a} \cdot \vec{b} = 2 + 2 = 4$$

---

### Passo 2: Calcular a Norma (Módulo) de cada vetor
Calculamos a distância de cada vetor até a origem $(0,0)$:

* **Módulo de $\vec{a}$ ($\Vert{}\vec{a}\Vert{}$):**
  $$\Vert{}\vec{a}\Vert{} = \sqrt{a_1^2 + a_2^2} = \sqrt{2^2 + 1^2} = \sqrt{4 + 1} = \sqrt{5} \approx 2{,}236$$

* **Módulo de $\vec{b}$ ($\Vert{}\vec{b}\Vert{}$):**
  $$\Vert{}\vec{b}\Vert{} = \sqrt{b_1^2 + b_2^2} = \sqrt{1^2 + 2^2} = \sqrt{1 + 4} = \sqrt{5} \approx 2{,}236$$

---

### Passo 3: Aplicar a Fórmula da Similaridade de Cosseno
Substituímos os valores obtidos nos Passos 1 e 2 na fórmula:

$$\cos(\theta) = \frac{\vec{a} \cdot \vec{b}}{\Vert{}\vec{a}\Vert{} \times \Vert{}\vec{b}\Vert{}}$$
$$\cos(\theta) = \frac{4}{\sqrt{5} \times \sqrt{5}}$$

Lembrando que a multiplicação de uma raiz quadrada por ela mesma elimina a raiz ($\sqrt{5} \times \sqrt{5} = 5$):

$$\cos(\theta) = \frac{4}{5} = 0{,}8$$

**Conclusão Prática:** O resultado **$0{,}8$** está extremamente próximo de **$1{,}0$**. Isso nos mostra matematicamente que o vetor "Rei" e o vetor "Rainha" apontam para direções quase idênticas no espaço vetorial, confirmando a alta similaridade semântica entre eles!

---

## 👑 4. Aritmética Semântica

Uma das propriedades mais impressionantes dos embeddings é que as **direções no espaço correspondem a conceitos abstratos** (como gênero, realeza, tempo verbal ou pluralidade). Por causa dessa organização geométrica, é possível fazer **operações matemáticas diretas** (soma e subtração) com as próprias palavras.

O exemplo mais famoso da literatura de IA é a analogia da Realeza e Gênero:

$$\vec{v}_\text{Rei} - \vec{v}_\text{Homem} + \vec{v}_\text{Mulher} \approx \vec{v}_\text{Rainha}$$

**A intuição por trás da fórmula:**
1. Quando pegamos o vetor $\vec{v}_\text{Rei}$ e subtraímos $\vec{v}_\text{Homem}$, estamos "removendo" o conceito/atributo de masculinidade da palavra "Rei" (sobrando apenas a essência abstrata da "realeza").
2. Quando somamos $\vec{v}_\text{Mulher}$, estamos "injetando" a dimensão de feminilidade nessa essência de realeza.
3. O vetor resultante estará localizado em uma posição do espaço cujos valores numéricos são quase idênticos aos do vetor da palavra **"Rainha"**.

---

### Exemplo Numérico em 3 Dimensões:

Para entender numericamente como essa conta funciona dentro do computador, vamos simular 3 dimensões fictícias projetadas para capturar 3 atributos específicos: `[Realeza, Gênero, Poder]`.

Considere os vetores fictícios padronizados:
* $\vec{v}_\text{Rei} = [0{,}9, -0{,}8, 0{,}7]$ *(Alta realeza, gênero masculino/negativo, alto poder)*
* $\vec{v}_\text{Homem} = [0{,}0, -0{,}8, 0{,}1]$ *(Sem realeza, gênero masculino/negativo, baixo poder)*
* $\vec{v}_\text{Mulher} = [0{,}0, 0{,}8, 0{,}1]$ *(Sem realeza, gênero feminino/positivo, baixo poder)*

#### Efetuando o cálculo dimensão por dimensão:

1. **Dimensão 1 (Realeza):**
   $$0{,}9 - 0{,}0 + 0{,}0 = 0{,}9$$

2. **Dimensão 2 (Gênero):**
   $$(-0{,}8) - (-0{,}8) + 0{,}8 = -0{,}8 + 0{,}8 + 0{,}8 = 0{,}8$$

3. **Dimensão 3 (Poder):**
   $$0{,}7 - 0{,}1 + 0{,}1 = 0{,}7$$

#### Resultado Obtido:
$$[0{,}9, 0{,}8, 0{,}7]$$

Ao buscar no banco de dados vetorial qual palavra possui o vetor mais próximo de $[0{,}9, 0{,}8, 0{,}7]$, o algoritmo encontra exatamente a palavra **$\vec{v}_\text{Rainha}$** (que representa alta realeza, gênero feminino e alto poder).

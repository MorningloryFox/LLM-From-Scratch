# 01. Embeddings e Geometria Vetorial

## Concept
Um **Embedding** é a representação de um token (palavra ou subpalavra) como um vetor num denso espaço multidimensional:
$$\vec{v} \in \mathbb{R}^d$$

Palavras com contextos semânticos semelhantes apontam para direções similares nesse espaço.

## Similaridade de Cosseno
A proximidade conceitual entre duas palavras é medida pelo cosseno do ângulo $\theta$ entre os seus vetores, derivado do produto escalar:

$$\vec{a} \cdot \vec{b} = \|\vec{a}\| \|\vec{b}\| \cos(\theta)$$

$$\cos(\theta) = \frac{\vec{a} \cdot \vec{b}}{\|\vec{a}\| \|\vec{b}\|}$$

* $\cos(\theta) = 1$ ($\theta = 0^\circ$): Vetores alinhados (mesmo sentido/conceito).
* $\cos(\theta) = 0$ ($\theta = 90^\circ$): Vetores ortogonais (sem relação direta).
* $\cos(\theta) = -1$ ($\theta = 180^\circ$): Vetores opostos (conceitos antagônicos).

## Aritmética Semântica
Como os conceitos são mapeados em direções, podemos realizar operações algébricas diretas com os significados:

$$\vec{v}_{\text{Rei}} - \vec{v}_{\text{Homem}} + \vec{v}_{\text{Mulher}} \approx \vec{v}_{\text{Rainha}}$$

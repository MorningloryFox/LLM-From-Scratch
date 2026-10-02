# Atenção causal

A atenção mistura informação de posições diferentes da sequência. Cada posição produz três vetores aprendidos: **Query** (Q), **Key** (K) e **Value** (V). Como analogia: Q é o que a posição procura, K é o que ela oferece para comparação e V é a informação que será combinada. Não são perguntas, etiquetas nem significados interpretáveis literalmente.

## A conta

Para uma cabeça com dimensão `d_k`, a atenção é:

$$\operatorname{Attention}(Q,K,V)=\operatorname{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V$$

1. `QKᵀ` calcula uma pontuação para cada par de posições.
2. Dividir por `√d_k` ajuda a manter a escala das pontuações controlada.
3. A máscara `M` põe uma pontuação efetivamente impossível nas posições proibidas.
4. `softmax` transforma as pontuações permitidas em pesos cuja soma é 1.
5. Os pesos combinam os vetores V.

Na autoatenção, Q, K e V vêm da representação da mesma sequência, por projeções lineares aprendidas.

## Por que a máscara é causal?

Ao prever o token seguinte na posição `t`, o modelo pode usar posições até `t`, mas não pode espiar posições futuras. A máscara causal bloqueia a parte superior da matriz de atenção. Assim, a previsão em cada posição pode ser treinada em paralelo sem receber a resposta futura como entrada.

No Mirim, `CausalSelfAttention` calcula a pontuação, divide pela raiz da dimensão da cabeça, aplica a máscara triangular inferior, usa softmax e combina V. A implementação atual tem várias cabeças: cada uma aprende projeções diferentes, e seus resultados são reunidos.

## O que a atenção não nos diz

Os pesos de atenção mostram quanto cada posição contribuiu para essa operação específica. Eles não são, por si só, uma explicação fiel do que o modelo “pensou”, nem garantem uma relação semântica humana. Um peso alto é uma medida interna da camada, não prova causalidade ou compreensão.

Sem informação posicional, a autoatenção não distingue diretamente a ordem: uma permutação de tokens permuta as saídas do mesmo modo. O modelo atual adiciona embeddings de posição aprendidos; RoPE é uma alternativa que estudaremos separadamente.

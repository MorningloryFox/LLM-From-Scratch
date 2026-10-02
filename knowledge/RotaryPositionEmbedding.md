# RoPE: informação de posição por rotação

A atenção precisa de alguma forma de distinguir a ordem dos tokens. Sem posição, a autoatenção trata uma sequência permutada de forma igualmente permutada; não sabe, sozinha, qual token veio primeiro.

Uma opção simples é somar um vetor de posição ao embedding do token. Outra é **Rotary Position Embedding (RoPE)**: girar pares de componentes dos vetores Q e K por ângulos que dependem da posição.

## Rotação em duas dimensões

Para um vetor `(x₁, x₂)` na posição `m`:

$$R(m\theta)\begin{bmatrix}x_1\\x_2\end{bmatrix}=
\begin{bmatrix}\cos(m\theta)&-\sin(m\theta)\\\sin(m\theta)&\cos(m\theta)\end{bmatrix}
\begin{bmatrix}x_1\\x_2\end{bmatrix}$$

Uma rotação preserva a norma do par. Em dimensões maiores, as componentes são agrupadas em pares e cada par usa uma frequência angular diferente. Uma parametrização comum é:

$$\theta_i = \text{base}^{-2i/d_h}$$

Aqui `d_h` é a dimensão de uma cabeça de atenção; convenções de índice podem começar em zero ou um. `base` é uma escolha de configuração, não uma constante universal para todo modelo.

## Por que Q e K?

Aplicando rotações compatíveis a Q na posição `m` e K na posição `n`, o produto entre eles pode carregar informação sobre o deslocamento relativo entre as posições. Essa propriedade ajuda a atenção a usar ordem e distância. Não significa que o modelo passe a compreender posição como uma pessoa.

## No Mirim

RoPE ainda não está implementado. O `Mirim` atual soma embeddings de posição aprendidos aos embeddings dos tokens. Quando estudarmos RoPE, vamos substituir ou comparar essa parte mantendo os outros elementos tão constantes quanto possível.

RoPE tem variantes para escala de posição, extrapolação de contexto e disposição dos pares. Vamos começar pela forma básica e documentar qualquer variação antes de usá-la.

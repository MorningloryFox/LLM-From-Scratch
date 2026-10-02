# Tensores e formas no Feneco 🔢

## 🎯 1. Conceito Fundamental

O PyTorch guarda números em **tensores**. Um tensor pode ser um único número, uma lista, uma tabela ou uma estrutura com várias dimensões. Saber ler a **forma** (*shape*) de cada tensor ajuda a entender o que o modelo está calculando e a encontrar erros de dimensão.

Por exemplo, a forma `(2, 3)` representa uma tabela com 2 linhas e 3 colunas. A forma `(4, 5, 8)` tem três eixos: 4 grupos, 5 posições e 8 valores por posição. A forma, sozinha, não diz o significado desses eixos; precisamos nomeá-los.

## 📐 2. Os eixos usados pelo Feneco

No código, vamos usar estas letras:

* **$B$ — Batch:** quantas sequências de treino processamos juntas.
* **$T$ — Tempo/Sequência:** quantos tokens há em cada sequência.
* **$V$ — Vocabulário:** quantos tokens diferentes o tokenizador conhece.
* **$D$ — Dimensão do modelo:** quantos números representam cada token dentro do modelo.
* **$H$ — Cabeças de atenção:** quantas partes dividem essa representação.
* **$d_h$ — Dimensão por cabeça:** tamanho de cada parte, calculado por $d_h = D/H$.

Na configuração padrão atual, $B=16$, $T=64$, $D=64$ e $H=4$. Portanto, cada cabeça tem $d_h=64/4=16$ componentes. O valor de $V$ depende dos caracteres que aparecem no corpus.

## 🔎 3. A forma dos dados de entrada

Antes de entrar no modelo, uma sequência de IDs tem forma:

$$X \in \mathbb{N}^{B \times T}$$

Aqui, $\mathbb{N}$ indica números inteiros usados como identificadores de token. Não são ainda os vetores que o Transformer calcula. Para um lote com 16 sequências de 64 caracteres, a forma é `[16, 64]`.

`nn.Embedding` procura cada ID na tabela de embeddings e troca um inteiro por um vetor com $D$ valores. A forma passa a ser `[B, T, D]`, ou `[16, 64, 64]` no exemplo padrão.

O embedding de posição tem forma `[T, D]`. Ele é somado aos embeddings de token. O PyTorch aplica *broadcasting*: o mesmo conjunto de posições é combinado com cada sequência do lote, sem precisarmos duplicá-lo manualmente.

## 🧮 4. Seguindo uma cabeça de atenção

As projeções lineares produzem $Q$, $K$ e $V$, cada uma com forma `[B, T, D]`. Para separar as cabeças, reagrupamos a dimensão $D$ em $H$ grupos de $d_h$ valores e trocamos a ordem dos eixos:

```text
[B, T, D] → [B, T, H, d_h] → [B, H, T, d_h]
```

Então $QK^T$ calcula uma pontuação para cada par de posições. O resultado tem forma `[B, H, T, T]`: para cada sequência e cabeça, há uma matriz que compara cada posição com todas as posições possíveis.

### Exemplo numérico de formas

Imagine um lote reduzido de $B=2$ sequências, cada uma com $T=3$ tokens. Use dimensão $D=8$ dividida em $H=2$ cabeças. Assim, $d_h=8/2=4$.

1. IDs de entrada: `[2, 3]`.
2. Embeddings e projeções Q/K/V: `[2, 3, 8]`.
3. Depois de separar e reorganizar as cabeças: `[2, 2, 3, 4]`.
4. Pontuações $QK^T$: `[2, 2, 3, 3]`.
5. Ao combinar as cabeças novamente: `[2, 3, 8]`.

O primeiro `2` na forma de pontuações é o lote; o segundo é a cabeça; os dois `3`s são as posições comparadas.

## 🧰 5. Do Transformer aos logits

Depois dos blocos Transformer, a projeção final produz **logits** — pontuações ainda não normalizadas para cada token possível:

$$\text{logits} \in \mathbb{R}^{B \times T \times V}$$

Se há 29 caracteres no vocabulário, cada posição recebe 29 pontuações. A entropia cruzada compara essas pontuações com o próximo caractere correto. No código, `logits` é remodelado para `[B*T, V]` e os alvos para `[B*T]`, porque a função calcula a perda de cada posição como uma previsão separada.

## 🔗 6. Onde isso aparece no código

Em `model.py`, acompanhe `token_ids`, `x`, `q`, `k`, `v`, `scores`, `attended` e `logits`. As operações `view`, `transpose`, `contiguous` e `reshape` mudam a disposição ou a forma dos dados; não criam significado por si mesmas.

Se uma operação der erro de dimensão, escreva ao lado a forma esperada de cada entrada. `view` precisa de memória contígua; depois de `transpose`, o código chama `contiguous()` antes de usar `view` novamente.

## 💡 Pergunta para Fixar

Se $D=96$ e $H=4$, quanto vale $d_h$? Para uma sequência com $T=32$, qual é a forma da matriz de pontuações de **uma cabeça e uma sequência**? E qual é a forma quando incluímos $B=8$ sequências e todas as cabeças?

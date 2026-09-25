# 02. Tokenização e Byte Pair Encoding (BPE) 🔤

## 🎯 1. Conceito Fundamental

Antes que qualquer conta matemática de vetor, matriz ou Similaridade de Cosseno aconteça, a Inteligência Artificial precisa converter o texto bruto — que para o computador é apenas uma *string* (sequência de caracteres) — em números inteiros. A esses números damos o nome de **IDs de Tokens**.

O **Token** é a menor unidade de texto que um modelo de linguagem (LLM) consegue processar. Cada token possui um **ID único** (por exemplo, o ID $4521$), e é justamente esse número que serve como "endereço" para buscar o vetor correspondente na tabela de embeddings.

Texto Bruto ("Eu amo física")
↓

[ Tokenizador ]

↓

IDs Numéricos ([1243, 8562, 19842])

↓

[ Tabela de Embeddings ]

↓

Vetores Multidimensionais ($\vec{v} \in \mathbb{R}^d$)

---

## 🧱 2. Como o Texto é Fatiado?

Existem três abordagens clássicas para dividir um texto em unidades menores:

### A. Tokenização por Palavra Inteira
O texto é separado por espaços e pontuações:
> `"Eu amo física"` $\rightarrow$ `["Eu", "amo", "física"]`

* ❌ **Problema:** O dicionário (vocabulário) do modelo precisa guardar centenas de milhares de palavras. Se o usuário digitar uma palavra nova, uma variação rara ou um erro de digitação (ex: `"físicasss"`), o modelo não entenderá e a tratará como um token desconhecido (`<UNK>`).

### B. Tokenização por Caractere
O texto é fatiado caractere por caractere, incluindo espaços:
> `"Eu amo física"` $\rightarrow$ `["E", "u", " ", "a", "m", "o", " ", "f", "í", "s", "i", "c", "a"]`

* ❌ **Problema:** O vocabulário fica minúsculo (apenas o alfabeto e símbolos), mas a sequência final fica gigantesca. O modelo precisa gastar muita capacidade computacional para entender o significado de uma palavra inteira, pois precisa ligar vários caracteres isolados.

### C. Tokenização por Subpalavra (Subword - BPE)
É o padrão utilizado por LLMs modernos (como GPT-4, Claude e Llama). 

* Palavras frequentes continuam sendo um único token.
* Palavras raras, compostas ou com erros são divididas em pedaços menores que o modelo já conhece.

> Exemplo: `"desmagnetização"` $\rightarrow$ `["des", "magneti", "zação"]`

---

## ⚙️ 3. O Algoritmo BPE (Byte Pair Encoding)

O **Byte Pair Encoding (BPE)** é um algoritmo de compressão de dados adaptado para inteligência artificial. Ele constrói o vocabulário de forma estatística, observando um grande conjunto de textos (*corpus*).

### Como o BPE funciona passo a passo:

1. **Vocabulário Inicial:** O BPE começa apenas com os caracteres individuais (ex: letras `a`, `b`, `c`, `d`...).
2. **Contagem de Frequência:** O algoritmo varre o texto e conta a frequência com que cada par de caracteres adjacentes aparece lado a lado.
3. **Fusão (Merge):** O par mais frequente é fundido para criar um novo token único, que é adicionado ao vocabulário.
4. **Repetição:** O processo se repete de forma iterativa por milhares de vezes até que o vocabulário atinja o tamanho desejado (ex: $32.000$ ou $100.000$ tokens).

#### Exemplo Prático de Fusão:
Imagine um vocabulário inicial simples: `['a', 'b', 'c', 'd']`.
Se no texto de treinamento o par `'a'` + `'b'` aparecer milhares de vezes junta, o BPE cria o token `'ab'` e o adiciona ao vocabulário:

$$\text{Vocabulário Novo} = ['a', 'b', 'c', 'd', 'ab']$$

A partir desse momento, a sequência de caracteres `"ab"` passa a ser contada como $1$ único token em vez de $2$.

---

## 💡 Guiding Question (Para Pensar)

Para fixar a lógica do BPE:

> **Cenário:** Suponha que temos um texto em que a palavra `"física"` aparece $1.000$ vezes e a palavra `"físico"` aparece $800$ vezes.
> 
> **Pergunta:** Qual sequência ou par de letras/subpalavras o algoritmo BPE iria identificar e fundir primeiro para economizar mais espaço de armazenamento e computação?

*R: '"físic"'*

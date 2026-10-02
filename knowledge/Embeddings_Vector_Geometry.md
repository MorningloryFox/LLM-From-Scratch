# Embeddings e geometria vetorial

## De um ID para um vetor

O modelo recebe números inteiros, mas esses números servem como rótulos, não como medidas. O ID 20 não significa “mais” do que o ID 10. Uma tabela de embeddings transforma cada ID em um vetor treinável:

$$E \in \mathbb{R}^{V \times d}, \qquad e_t = E[t]$$

`V` é o tamanho do vocabulário, `d` é a dimensão do vetor e `e_t` é a linha da tabela associada ao token `t`. No código atual, `nn.Embedding` cria essa tabela.

## O que as dimensões significam?

Uma dimensão não costuma corresponder a uma ideia legível como “gênero” ou “realeza”. Os pesos são ajustados em conjunto para ajudar o modelo a prever o próximo token. O significado está distribuído pelos valores e depende do modelo e do contexto.

O embedding inicial de um token é o mesmo onde quer que ele apareça. Depois das camadas Transformer, sua representação muda conforme os tokens ao redor. Essa diferença entre embedding de entrada e representação contextual é importante.

## Produto escalar e similaridade de cosseno

Para vetores `a` e `b`, o produto escalar é:

$$a \cdot b = \sum_{i=1}^{d} a_i b_i$$

A similaridade de cosseno normaliza esse produto pelas magnitudes:

$$\cos(a,b) = \frac{a \cdot b}{\|a\|\|b\|}$$

O resultado mede alinhamento angular. Vetores com cosseno alto apontam em direções parecidas; cosseno zero significa ortogonalidade matemática; cosseno negativo indica direções opostas em relação à origem.

Isso **não** prova que duas palavras sejam sinônimas, sem relação ou antônimas. A interpretação depende de como os vetores foram aprendidos, de quais vetores estamos comparando e da tarefa. Em especial, a tabela inicial de um modelo minúsculo recém-inicializado não contém uma geometria semântica útil.

## Uma conta simples

Para `a = [1, 0]` e `b = [1, 1]`:

$$a \cdot b = 1, \qquad \|a\|=1, \qquad \|b\|=\sqrt{2}$$

Logo, `cos(a,b) = 1/√2 ≈ 0,707`. Essa é uma conta geométrica, não uma medida universal de significado.

## Aritmética vetorial

Em alguns embeddings treinados, analogias como `rei - homem + mulher ≈ rainha` podem aparecer aproximadamente. Isso é uma observação empírica em certos espaços, não uma regra garantida nem uma operação que “remove” e “injeta” atributos de forma literal. Nosso Mirim começa com caracteres e um corpus minúsculo; não devemos esperar analogias semânticas dele.

## No Mirim

`Mirim` soma o embedding do token com um embedding de posição aprendido. Ainda não implementamos comparações de similaridade nem visualização dos vetores. O capítulo de tokenização explica como o texto vira IDs.

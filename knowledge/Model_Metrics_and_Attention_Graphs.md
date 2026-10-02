# O que podemos contar no modelo?

Vamos separar contagens bem definidas de interpretações. Número de parâmetros, cabeças e conexões permitidas pela máscara são mensuráveis. “Pensamentos” não são unidades explícitas deste modelo.

## Parâmetros treináveis

Parâmetro é um valor ajustável do modelo. Podemos contar os elementos de `model.parameters()` e separar os que têm `requires_grad=True`. A contagem depende do vocabulário e da configuração; não mede inteligência nem garante qualidade.

## Cabeças e conexões de atenção

No Mirim padrão há 2 camadas com 4 cabeças por camada: **4 cabeças por camada, 8 instâncias de cabeça somando as camadas**. Isso é diferente do número de pesos: cada camada usa projeções aprendidas compartilhadas entre todas as posições.

Para uma sequência de comprimento `T`, uma máscara causal permite `T(T+1)/2` pares posição-consulta/posição-chave por cabeça, contando a posição atendendo a si mesma. Com `L` camadas e `H` cabeças em cada camada, o total permitido é:

$$L\,H\,\frac{T(T+1)}{2}$$

Com `L=2`, `H=4` e `T=64`, isso dá 16.640 pares permitidos por sequência. A implementação ainda calcula a matriz completa de pontuações antes de mascarar o futuro, então operações realizadas não são a mesma contagem que conexões permitidas.

## Isso é um grafo de pensamento?

Não. A máscara e os pesos de atenção podem ser desenhados como um grafo entre posições, mas esses pesos não representam uma lista explícita de pensamentos, fatos ou passos de raciocínio. Nosso modelo atual gera o próximo caractere; não emite uma estrutura de raciocínio com nós e arestas.

Se fizermos no futuro um experimento de grafo explícito, precisamos definir o formato de nó e relação, como o modelo o produz, como contamos nós/arestas e como julgamos se o grafo é útil. Até lá, chamaremos estas medidas de **parâmetros**, **cabeças** e **conexões de atenção**, sem chamá-las de pensamentos.

## Registro de cada experimento

Uma ficha útil terá versão do código, configuração, contagem de parâmetros, camadas, cabeças por camada, tamanho e vocabulário do corpus, tokenizador, número de passos, perdas, tempo, dispositivo e amostras geradas. Isso permitirá acompanhar a evolução sem misturar configurações diferentes.

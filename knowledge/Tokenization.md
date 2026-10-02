# Tokenização: do texto aos IDs

Um modelo não recebe palavras diretamente. Um tokenizador divide o texto em unidades e converte cada unidade num ID inteiro. O modelo usa esses IDs para consultar embeddings.

```text
"bom dia" → ["bom", " dia"] → [IDs] → vetores
```

Os IDs são apenas identificadores. Se trocarmos todos os IDs e atualizarmos a tabela de embeddings na mesma ordem, o significado do sistema não muda.

## Três escolhas comuns

- **Palavra inteira:** poucas posições para palavras conhecidas, mas vocabulário enorme e problemas com palavras novas.
- **Caractere:** vocabulário menor e qualquer palavra pode ser representada, mas as sequências ficam longas.
- **Subpalavra:** divide palavras raras em pedaços reutilizáveis; BPE é uma forma de construir esse vocabulário.

Nenhuma opção é universalmente melhor. Tokenização é parte do modelo: no treino e na geração é preciso usar exatamente o mesmo vocabulário e as mesmas regras.

## BPE em resumo

Byte Pair Encoding começa com unidades básicas e repete uma operação: contar pares adjacentes, fundir o par mais frequente e registrar essa fusão. Ao codificar texto depois, aplica-se a sequência aprendida de fusões. O resultado depende do corpus, das regras de pré-processamento, da unidade inicial e do tamanho de vocabulário escolhido.

Alguns tokenizadores modernos começam por bytes em vez de caracteres Unicode. Isso ajuda a representar qualquer texto, mas “BPE” não determina sozinho todos os detalhes de um tokenizador. Não é correto afirmar que uma palavra específica sempre vira o mesmo número de pedaços sem executar um tokenizador definido.

## Feneco-Char e Feneco-Token

Feneco-Char cria um vocabulário com `sorted(set(text))` e usa um ID para cada caractere Python encontrado no corpus. Feneco-Token usa BPE em bytes UTF-8 por meio da biblioteca `tokenizers`: começa com o alfabeto de bytes e aprende fusões frequentes até o tamanho configurado do vocabulário. Com isso, texto não visto continua representável. O tokenizer é treinado somente sobre a partição de treino e seu JSON é guardado no checkpoint.

Subpalavras costumam representar texto com menos posições que caracteres, então um contexto de 256 tokens pode cobrir mais texto do que 64 caracteres. A quantidade real varia conforme a língua e o corpus; precisamos medi-la, não assumir uma proporção fixa. Mudar de caractere para BPE altera IDs, embeddings e projeção de saída: é necessário treinar um checkpoint novo. Treino, geração e avaliação devem usar exatamente o tokenizer salvo com o checkpoint.

## Pergunta para experimentar

Compare o número de tokens BPE e caracteres de uma frase em português. Como a resposta muda para palavras raras, URLs ou outra língua?

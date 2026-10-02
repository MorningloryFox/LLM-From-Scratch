# Tokenização: do texto aos IDs

Um modelo não recebe palavras diretamente. Um tokenizador divide o texto em unidades e converte cada unidade num ID inteiro. O modelo usa esses IDs para consultar embeddings.

```text
"bom dia" → ["b", "o", "m", " ", "d", "i", "a"] → [IDs] → vetores
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

## O tokenizador atual do Mirim

O treino cria um vocabulário com `sorted(set(text))` e atribui um ID a cada caractere Python distinto no corpus. Durante a geração, um caractere ausente nesse vocabulário causa erro. Espaços, pontuação, maiúsculas e letras acentuadas são unidades diferentes.

Essa simplicidade é boa para acompanhar os primeiros passos, mas ruim para eficiência: uma frase de 10 palavras pode ocupar dezenas de posições. BPE fica para uma etapa posterior; primeiro queremos entender o ciclo de treino com a versão por caractere.

## Pergunta para experimentar

Se o corpus de treino não contém `ç`, o que acontece ao pedir geração com um prefixo que contém `ç`? E por que trocar o tokenizador depois do treino sem retreinar embeddings quebra o significado dos IDs?

# Geração e amostragem

Durante a geração, o modelo recebe um prefixo, prevê uma distribuição para o próximo token, escolhe um token e acrescenta-o à sequência. Repetimos até atingir o limite pedido ou uma regra de parada.

## Logits e temperatura

O modelo produz logits `z`. Softmax converte-os em probabilidades. Temperatura `T` ajusta quão concentrada fica a distribuição:

$$p_i=\operatorname{softmax}(z_i/T)$$

`T` perto de zero favorece muito o maior logit; temperaturas maiores espalham mais a probabilidade. Temperatura não melhora o modelo nem adiciona conhecimento. O Feneco usa amostragem multinomial com temperatura; o padrão do programa é `0.8`.

## Outras regras de escolha

- **Greedy:** escolhe sempre a maior probabilidade. É determinístico, mas pode repetir ciclos.
- **Top-k:** mantém apenas os `k` tokens mais prováveis e renormaliza.
- **Top-p (nucleus):** mantém o menor conjunto de tokens cuja probabilidade acumulada alcança `p`.

Essas regras mudam a diversidade e os erros, não os pesos treinados. Para comparar modelos, fixamos prompt, temperatura, regra de amostragem e limite de tokens.

## No Feneco hoje

`generate.py` carrega um checkpoint local e usa o tokenizador que está salvo nele. Em Feneco-Token, o limite `--tokens` conta unidades BPE, não palavras nem caracteres; o byte-level BPE consegue codificar texto Unicode novo. Checkpoints de Feneco-Char continuam usando seu vocabulário de caracteres.

O gerador reexecuta o modelo sobre o contexto disponível a cada novo token e ainda não usa cache KV. No ajuste de conversa, `<|end|>` marca o fim de uma resposta; no pré-treino de livros, a geração normalmente para quando alcança o limite configurado. [Cache KV](./KV_Cache.md) pode evitar parte desse trabalho no futuro.

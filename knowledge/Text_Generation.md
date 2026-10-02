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

`generate.py` carrega um checkpoint local, converte o prefixo com o vocabulário salvo e pede novos caracteres. Prefixos com caracteres desconhecidos são recusados. O limite `--tokens` conta caracteres, não palavras.

O gerador reexecuta o modelo sobre o contexto disponível a cada novo caractere. Não há token de fim de sequência dedicado; a geração para quando alcança o limite configurado. [Cache KV](./KV_Cache.md) pode evitar parte desse trabalho no futuro.

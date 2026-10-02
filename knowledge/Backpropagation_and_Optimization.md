# Gradientes e otimização ⚙️

## 🎯 1. Conceito Fundamental

Treinar o Feneco significa ajustar seus pesos para que ele atribua pontuações maiores ao próximo caractere correto. O modelo faz uma previsão, medimos o erro, calculamos como cada peso contribuiu para esse erro e ajustamos os pesos.

Esse ciclo acontece muitas vezes. Uma **iteração** ou **passo** é uma atualização dos pesos. Um passo não significa que o modelo leu o corpus inteiro: no Feneco, cada passo usa um pequeno lote de janelas amostradas do texto.

## 🔤 2. Deslocar o texto para prever o próximo token

Se os tokens de uma sequência forem:

```text
entrada: [A, B, C]
alvo:    [B, C, D]
```

O modelo usa `A` para prever `B`, usa `A B` para prever `C` e usa `A B C` para prever `D`. A máscara causal impede que a previsão de uma posição veja os tokens futuros. Assim, várias previsões podem ser calculadas em paralelo.

## 📉 3. A perda de entropia cruzada

Em cada posição, o modelo produz uma pontuação para cada token do vocabulário. Chamamos essas pontuações de **logits**. A função softmax converte logits em probabilidades; a entropia cruzada mede a probabilidade dada ao alvo correto.

Para uma previsão cujo token correto é $y_i$:

$$\ell_i=-\log p_\theta(y_i\mid x_{\le i})$$

* **$\ell_i$:** perda naquela posição.
* **$p_\theta$:** probabilidade calculada pelo modelo com parâmetros $\theta$.
* **$y_i$:** token correto a prever.
* **$x_{\le i}$:** tokens disponíveis até a posição atual.
* **$\log$:** logaritmo natural; probabilidades maiores que zero e próximas de 1 produzem perdas menores.

A perda média do lote é:

$$\mathcal{L}= -\frac{1}{N}\sum_{i=1}^{N}\log p_\theta(y_i\mid x_{\le i})$$

### Exemplo de perda

Se o modelo der probabilidade `0,8` ao próximo caractere correto, a perda é $-\ln(0,8)\approx0,223$. Se der apenas `0,1`, a perda é $-\ln(0,1)\approx2,303$. O segundo caso é penalizado mais porque o modelo quase descartou a resposta certa.

## 🧮 4. Backpropagation: de onde vem o gradiente?

A saída e a perda dependem de milhares de pesos. A **regra da cadeia** propaga para trás a informação de quanto pequenas mudanças em cada peso alterariam a perda. O resultado é o gradiente $\nabla_\theta\mathcal{L}$.

Um gradiente não é uma nota para a qualidade do peso; ele indica a direção local de maior aumento da perda. Para reduzir a perda, um passo simples de descida do gradiente seria:

$$\theta_{t+1}=\theta_t-\eta\nabla_\theta\mathcal{L}$$

* **$\theta_t$:** parâmetros antes da atualização.
* **$\theta_{t+1}$:** parâmetros depois da atualização.
* **$\eta$:** taxa de aprendizado, que controla o tamanho geral do passo.

## ⚙️ 5. O papel do AdamW

O Feneco usa AdamW. Em vez de seguir o gradiente bruto, AdamW mantém médias móveis do gradiente e do gradiente ao quadrado. De forma simplificada:

$$m_t=\beta_1m_{t-1}+(1-\beta_1)g_t$$
$$v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2$$

`g_t` é o gradiente atual; `m_t` suaviza sua direção; `v_t` acompanha sua escala; `β₁` e `β₂` controlam quanto do histórico fica retido. AdamW usa essas estatísticas para ajustar cada parâmetro e aplica *weight decay* de forma separada.

A taxa de aprendizado ainda importa: se for grande demais, a perda pode oscilar ou explodir; se for pequena demais, o treino pode avançar devagar. O Feneco começa com `lr=3e-4`; isso é apenas o ponto de partida deste experimento.

## 🔁 6. O ciclo no código

O arquivo `train.py` executa esta sequência:

1. `model.train()` ativa o comportamento de treino, incluindo dropout.
2. `get_batch(...)` escolhe janelas de entrada e seus alvos deslocados.
3. `model(x, y)` faz o forward e calcula a perda.
4. `optimizer.zero_grad(set_to_none=True)` limpa gradientes do passo anterior.
5. `loss.backward()` calcula os novos gradientes.
6. `optimizer.step()` atualiza os parâmetros.

PyTorch acumula gradientes por padrão; por isso limpamos os gradientes antes do próximo passo. `backward()` calcula gradientes, mas não atualiza os parâmetros; essa atualização fica para `step()`.

Na validação, o código usa `model.eval()` e `torch.no_grad()`: dropout é desativado e não guardamos informação para calcular gradientes.

## 📊 7. Como ler os sinais

* Perda de treino e validação caindo: as previsões estão melhorando nas duas partes observadas.
* Perda de treino cai enquanto validação piora: pode haver sobreajuste, vazamento mal tratado ou diferença entre os trechos.
* Perdas quase paradas: confira dados, alvos, taxa de aprendizado e se os pesos estão sendo atualizados.
* Perda muito instável: taxa alta, corpus minúsculo ou lotes muito diferentes podem contribuir.

Com o corpus de demonstração, tentar memorizar o treino é um exercício útil para validar o ciclo. Memorizar um texto pequeno não demonstra que o modelo generaliza ou entende linguagem.

## 💡 Pergunta para Fixar

Qual a diferença entre `loss.backward()` e `optimizer.step()`? Se a perda de treino continuar caindo e a de validação começar a subir, que hipótese devemos investigar primeiro?

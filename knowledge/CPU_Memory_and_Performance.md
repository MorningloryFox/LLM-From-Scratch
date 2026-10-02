# CPU, memória e desempenho 💻

## 🎯 1. Conceito Fundamental

Um modelo pode caber na memória e ainda ser lento. Também pode treinar rápido num texto pequeno e ficar lento quando aumenta o contexto. Por isso, vamos medir separadamente:

* quantidade de parâmetros e tamanho dos pesos;
* memória de ativações durante o treino;
* tempo total de treino;
* latência e velocidade de geração.

“Roda na minha máquina” é um requisito prático, não uma unidade de desempenho. Para comparar configurações, precisamos guardar o equipamento, o corpus e os parâmetros usados.

## 🧮 2. Memória dos parâmetros

O Feneco cria parâmetros `float32` por padrão. Cada valor `float32` ocupa 4 bytes. Se `N` for o número de parâmetros treináveis, uma estimativa do tamanho dos pesos é:

$$\text{bytes dos pesos} \approx 4N$$

* **$N$:** quantidade de valores aprendidos.
* **4:** bytes por valor `float32`.

O treino também mantém gradientes e estados do AdamW. Como aproximação grosseira, pesos, gradientes e dois estados do otimizador podem ocupar perto de `16N` bytes em float32, sem contar ativações, buffers, memória temporária e a sobrecarga do PyTorch. O consumo real depende do dispositivo e da implementação.

O script do Feneco já imprime a contagem de parâmetros treináveis. Essa contagem não é uma medida de inteligência: modelos com o mesmo número de pesos podem ter dados, arquitetura e resultados muito diferentes.

## 📐 3. Custo da atenção

Para batch $B$, cabeças $H$ e contexto $T$, a matriz de pontuações tem $BHT^2$ elementos. Com `float32`, apenas essa matriz ocupa aproximadamente:

$$4BHT^2\text{ bytes}$$

No padrão atual, $B=16$, $H=4$ e $T=64$. A matriz contém:

$$16 \times 4 \times 64^2 = 262.144\text{ valores}$$

Isso equivale a cerca de 1 MiB só para uma matriz de pontuações nesse lote e contexto. O modelo mantém outras ativações também; o total é maior.

A máscara causal permite por cabeça e sequência apenas a metade triangular, incluindo a diagonal:

$$\frac{T(T+1)}{2}$$

Com $L=2$ camadas e $H=4$ cabeças, em contexto $T=64$, são $2 \times 4 \times 64 \times 65/2 = 16.640$ pares causais permitidos por sequência. A implementação calcula a matriz completa antes de zerar logicamente os pesos futuros; conexões permitidas e operações executadas são contagens diferentes.

O cálculo da atenção cresce aproximadamente com $B T^2 D$. Dobrar o contexto pode quadruplicar essa parte do custo. As projeções e o MLP também consomem computação, aproximadamente na ordem de $B T D^2$.

## ⏱️ 4. Medir sem enganar a si mesmo

Para comparar duas versões:

1. Use o mesmo equipamento, corpus, batch, contexto e número de passos.
2. Faça uma execução de aquecimento antes de cronometrar.
3. Repita a medição e registre mediana ou intervalo, não apenas o melhor tempo.
4. Separe carregamento, treino e geração.
5. Registre processador, número de threads, versões de Python/PyTorch e dispositivo.

Para geração, podemos medir tokens por segundo. No Feneco-Token, token significa uma unidade do tokenizer BPE em bytes; essa taxa não pode ser comparada diretamente com a taxa por caractere do Feneco-Char. Também podemos medir a latência até o primeiro token e o tempo médio dos tokens seguintes.

## 🧰 5. Situação atual do Feneco

O script escolhe CUDA quando PyTorch informa que CUDA está disponível; caso contrário, usa CPU. Não há opção de linha de comando para forçar o dispositivo. O treino registra duração, mas ainda não mede pico de RAM. O gerador recalcula a sequência inteira a cada token e não usa cache KV. O contexto padrão do Feneco-Token é 256 tokens; como a atenção cresce aproximadamente com o quadrado do contexto, essa alteração aumenta o custo em comparação com os checkpoints por caractere de contexto 64.

Primeiro vamos medir uma linha de base local. Só depois faz sentido comparar cache, precisão menor ou outras otimizações. Uma otimização é útil se melhora tempo ou memória sem causar uma piora inaceitável nas perdas e amostras.

## 💡 Pergunta para Fixar

Se dobrarmos o contexto de 64 para 128, por qual fator cresce a matriz de atenção? E por que isso não significa que o treino inteiro necessariamente ficará exatamente quatro vezes mais lento?

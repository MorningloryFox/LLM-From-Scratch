# Treino e avaliação

Treinar um modelo de linguagem é ajustar seus pesos para prever o próximo token. Avaliar é verificar previsões em texto que não foi usado para ajustar esses pesos e inspecionar amostras com um protocolo consistente.

## Um exemplo de treino

Se os IDs forem `[A, B, C, D]`, criamos entrada `[A, B, C]` e alvos `[B, C, D]`. O modelo produz logits em cada posição. A entropia cruzada compara esses logits com o próximo ID correto:

$$\mathcal{L}=-\frac{1}{N}\sum_{i=1}^{N}\log p_\theta(y_i\mid x_{\le i})$$

O PyTorch calcula gradientes dessa perda com *backpropagation*. Um otimizador, no projeto `AdamW`, move os parâmetros para reduzir a perda. Isso se repete por passos; um passo é uma atualização do otimizador com um lote de exemplos.

## Treino, validação e teste

- **Treino:** dados usados para calcular gradientes e atualizar os pesos.
- **Validação:** dados separados do treino, usados para comparar configurações e perceber sobreajuste; não devem atualizar os pesos.
- **Teste:** conjunto mantido de lado até escolhermos a configuração final; dá uma estimativa mais honesta do resultado final.

Particionar texto exige cuidado. Janelas vizinhas se sobrepõem, então uma divisão aleatória por janela pode pôr trechos quase iguais nos dois lados. Para um único arquivo, uma divisão por segmentos contíguos evita espalhar janelas vizinhas entre partições. Para uma coleção de livros, é melhor manter cada obra inteira em treino, validação ou teste, para medir o desempenho em obras não vistas. Dados duplicados ou quase duplicados ainda podem vazar entre partições e devem ser identificados.

## Métricas

- **Perda:** menor costuma indicar melhores previsões para essa distribuição, mas só é comparável com o mesmo tokenizador e partição.
- **Perplexidade:** `exp(perda média)` para entropia cruzada em log natural. É uma forma de expressar incerteza média; não mede verdade, utilidade ou qualidade geral.
- **Amostras:** ler gerações com os mesmos prompts e parâmetros ajuda a ver repetição, coerência local e defeitos que uma média não mostra.
- **Custo:** tempo, dispositivo, memória, passos, tamanho do corpus e configuração ajudam a repetir e comparar experiências.

## Estado atual no Feneco

O script aceita um arquivo ou uma pasta de arquivos `.txt`. Um arquivo é dividido em segmentos contíguos de 80/10/10; uma pasta é dividida por documento, com embaralhamento determinístico pela semente e proporções aproximadas pela quantidade de obras. A validação seleciona o melhor checkpoint; o teste é medido depois dessa seleção. Semente, SHA-256 do corpus, configuração, arquivos de cada partição, perdas, duração, dispositivo e métricas do modelo são registrados localmente em `.local/experiments.jsonl`. Mesmo com partições por obra, cópias da mesma obra em arquivos diferentes podem causar vazamento e devem ser removidas ou agrupadas antes da divisão.

Antes de comparar mudanças, vamos registrar: nome/versão do experimento, origem e hash do corpus, partições, tokenizador, configuração, semente, passos, perdas, duração e amostras. O conteúdo do corpus privado deve continuar local; para compartilhar um resultado, podemos registrar métricas e metadados não sensíveis.

# O bloco Transformer do Feneco

Um decoder-only repete blocos que misturam o contexto anterior e transformam as representações. No fim, uma projeção produz uma pontuação para cada token possível.

```text
IDs → embeddings de token + posição
    → [norm → atenção causal → residual → norm → MLP → residual] × camadas
    → norm final → logits sobre o vocabulário
```

## Conexões residuais e normalização

Uma conexão residual soma a entrada de uma subcamada à sua saída: `x ← x + f(x)`. Isso dá ao modelo um caminho direto para transportar informação e facilita o fluxo de gradientes.

O Feneco usa LayerNorm antes da atenção e do MLP (*pre-norm*). LayerNorm centraliza e escala as ativações de cada posição. Uma alternativa, RMSNorm, só normaliza pela raiz da média dos quadrados:

$$\operatorname{RMSNorm}(x)=\frac{x}{\sqrt{\operatorname{mean}(x^2)+\epsilon}}\odot g$$

`g` é um vetor de escala aprendível. Implementações podem variar em detalhes como `epsilon` e uso de bias. RMSNorm dispensa a centralização pela média e pode ser mais simples, mas nenhuma das duas normalizações é sempre superior.

## Atenção multi-head

Cada cabeça faz sua própria projeção de Q, K e V. Cabeças diferentes podem aprender padrões diferentes, embora os pesos de atenção não sejam explicações humanas. A saída das cabeças é reunida e projetada de volta à dimensão do modelo. Consulte [Atenção causal](./Attention_Mechanism.md).

## MLP: GELU e SwiGLU

O MLP aplica uma transformação por posição, sem misturar diretamente tokens diferentes. A versão atual expande a dimensão `d` para `4d`, aplica GELU e retorna para `d`:

$$\operatorname{MLP}(x)=W_2\,\operatorname{GELU}(W_1x+b_1)+b_2$$

SwiGLU é uma alternativa com duas projeções na expansão; uma delas passa por SiLU e funciona como uma porta multiplicativa:

$$\operatorname{SwiGLU}(x)=W_o\left(\operatorname{SiLU}(W_gx)\odot W_ux\right)$$

As dimensões internas costumam ser ajustadas ao orçamento do modelo. O Feneco ainda usa GELU; comparar GELU com SwiGLU exige manter dados, orçamento de parâmetros e treino comparáveis.

## O que o código faz hoje

`Feneco` usa embeddings de posição aprendidos, duas camadas por padrão, LayerNorm, atenção causal multi-head e MLP com GELU. A saída é um vetor de logits por posição. A configuração padrão é pequena para caber em experiências locais; não é uma receita para treinamento de produção.

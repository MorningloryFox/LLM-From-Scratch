# Quantização

O Feneco atual usa pesos de ponto flutuante comuns do PyTorch (normalmente `float32`, salvo conversões explícitas). Quantização representa pesos e, às vezes, ativações com números de menor precisão para reduzir armazenamento e potencialmente acelerar a inferência.

## Ideia básica

Um valor real `x` pode ser aproximado por um inteiro `q` com escala e zero-point, por exemplo:

$$x \approx s(q-z)$$

`q` usa poucos bits; `s` e `z` descrevem como voltar a uma aproximação no intervalo original. Na prática há formatos e métodos diferentes, por tensor, canal ou blocos, com escolhas distintas de escala e arredondamento.

## Ganhos e trade-offs

Menos bits podem diminuir o arquivo e o tráfego de memória. Pode haver perda de qualidade, e velocidade depende do hardware e das operações implementadas. “INT4” sozinho não especifica formato, agrupamento, calibração nem kernel; dois arquivos com o mesmo número de bits podem se comportar diferente.

GGUF é um formato de arquivo usado no ecossistema llama.cpp e pode armazenar pesos quantizados em formatos diversos. GGUF não é sinônimo de quantização nem uma medida de qualidade.

## No Feneco

Ainda não há exportação nem quantização. Quando o modelo e a avaliação estiverem estáveis, compararemos tamanho, tempo de geração e perda/amostras antes e depois, usando o mesmo corpus e prompts. Não adianta otimizar um resultado cuja qualidade ainda não medimos.

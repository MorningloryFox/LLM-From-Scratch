# LLM From Scratch

Um projeto de aprendizado prático para construir e treinar um modelo de linguagem decoder-only em escala pequena, entendendo cada peça no caminho.

## Objetivo e limites

O objetivo inicial é implementar um GPT pequeno em PyTorch e treiná-lo do zero em texto local. A primeira versão usa tokenização por caractere para manter o foco no fluxo completo: texto → tokens → previsões do próximo token → treino → geração.

Este primeiro modelo é um experimento educacional. Ele não terá conhecimento geral, não será comparável a modelos pré-treinados e só aprenderá padrões do corpus usado no treino. O foco inicial é CPU e modelos pequenos; desempenho e qualidade virão depois da implementação correta.

## Primeiro marco: GPT mínimo treinável

- [ ] Criar um corpus de treino pequeno, local e permitido para uso.
- [x] Implementar vocabulário de caracteres e codificação/decodificação.
- [x] Implementar decoder Transformer causal com embeddings de token e posição.
- [x] Treinar para prever o próximo token e salvar um checkpoint.
- [x] Gerar texto a partir de um prefixo informado.
- [ ] Registrar configuração, corpus e resultado de cada experimento.

O primeiro marco não inclui BPE, RoPE, KV-cache, quantização, interface web nem fine-tuning de um modelo existente. Esses itens entram depois que o ciclo de treino e geração estiver compreendido e funcionando.

## Caminho de evolução

1. **GPT mínimo:** tokenização por caractere, atenção causal, treino e geração local.
2. **Componentes modernos:** tokenizer BPE, RMSNorm, SwiGLU e RoPE, comparando cada alteração com a versão simples.
3. **Treino melhor controlado:** validação, checkpoints retomáveis, métricas e tratamento de corpus maiores.
4. **Inferência:** amostragem, cache KV e otimizações de memória/CPU.
5. **Distribuição:** quantização e formatos como GGUF, se houver necessidade prática.

## Requisitos previstos

- Python 3.11 ou superior.
- PyTorch.
- Um corpus de texto que você tenha direito de usar; os dados de treino não devem ser enviados ao Git por padrão.

## Executar localmente

Na raiz do repositório, crie um ambiente e instale as dependências:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

Treine com o corpus didático incluído (ou informe outro arquivo de texto UTF-8):

```powershell
python -m llm_from_scratch.train --data data/tiny.txt --steps 500
```

Gere texto usando um prefixo presente no corpus:

```powershell
python -m llm_from_scratch.generate --checkpoint checkpoints/tiny-gpt.pt --prompt "O modelo"
```

O corpus incluído é pequeno e serve apenas para demonstrar o ciclo de treino. Para um experimento próprio, salve seus dados localmente e confirme que pode usá-los. Checkpoints, ambientes virtuais e `data/private/` são ignorados pelo Git.

Não há, nesta etapa, promessa de tempo de treino ou de qualidade: isso depende da máquina, do tamanho do corpus e da configuração.

## Conhecimento existente

As anotações em [`knowledge/`](./knowledge/) são material de referência, não componentes já implementados:

- [Embeddings e geometria vetorial](./knowledge/Embeddings_Vector_Geometry.md)
- [Tokenização](./knowledge/Tokenization.md)
- [Mecanismo de atenção](./knowledge/Attention_Mechanism.md)
- [RoPE](./knowledge/RotaryPositionEmbedding.md)

## Licença

Este projeto está licenciado sob a licença MIT. Consulte [`LICENSE`](./LICENSE).

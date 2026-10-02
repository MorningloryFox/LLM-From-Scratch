# LLM From Scratch

Quero aprender a construir um modelo de linguagem fazendo um funcionar primeiro e estudando cada parte conforme eu for avançando.

Estou começando com um GPT bem pequeno, escrito em PyTorch e treinado do zero com texto local. Por enquanto ele lê um caractere por vez. A ideia é acompanhar o caminho todo: texto, tokens, previsão do próximo token, treino e geração.

Esse modelo é só para aprender. Com o corpus de exemplo, ele não vai responder perguntas nem saber coisas sobre o mundo. Vai aprender padrões simples do texto em que foi treinado.

## Rodar

Precisa de Python 3.11 ou mais recente. Na pasta do projeto, rode:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
```

Treine com o texto pequeno que deixei no repositório:

```powershell
python -m llm_from_scratch.train --data data/tiny.txt --steps 500
```

Depois, peça uma continuação começando por um trecho que apareça no corpus:

```powershell
python -m llm_from_scratch.generate --checkpoint checkpoints/tiny-gpt.pt --prompt "O modelo"
```

O checkpoint fica em `checkpoints/` e não entra no Git. Para experimentar com outro texto, passe o caminho desse arquivo em `--data`. Use apenas textos que você tem direito de usar; dados próprios podem ficar em `data/private/`, que também é ignorada pelo Git.

## O que já tem

- Tokenização simples por caractere.
- Transformer decoder com atenção causal.
- Treino para prever o próximo caractere e salvar o modelo.
- Geração de texto a partir de um prefixo.

## O que quero estudar depois

Quero trocar a tokenização por BPE e experimentar RoPE, RMSNorm e SwiGLU. Depois posso olhar para cache KV e quantização. Ainda não implementei essas partes.

As notas que já escrevi estão em [`knowledge/`](./knowledge/):

- [Embeddings e geometria vetorial](./knowledge/Embeddings_Vector_Geometry.md)
- [Tokenização](./knowledge/Tokenization.md)
- [Atenção](./knowledge/Attention_Mechanism.md)
- [RoPE](./knowledge/RotaryPositionEmbedding.md)

## Licença

MIT. Veja [`LICENSE`](./LICENSE).

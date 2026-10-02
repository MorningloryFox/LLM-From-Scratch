# Knowledge: guia de estudo

As notas acompanham o código e mostram o que já está implementado, o que é uma alternativa e o que fica para depois. A ordem sugerida começa pelos dados e segue até a geração.

1. [Embeddings e geometria vetorial](./Embeddings_Vector_Geometry.md)
2. [Tokenização e BPE](./Tokenization.md)
3. [Atenção causal](./Attention_Mechanism.md)
4. [Bloco Transformer, LayerNorm/RMSNorm e GELU/SwiGLU](./Transformer_Block.md)
5. [Posição e RoPE](./RotaryPositionEmbedding.md)
6. [Treino, validação e teste](./Training_and_Evaluation.md)
7. [Geração e amostragem](./Text_Generation.md)
8. [Contagem de parâmetros, cabeças e conexões de atenção](./Model_Metrics_and_Attention_Graphs.md)
9. [Cache KV](./KV_Cache.md)
10. [Quantização](./Quantization.md)

## Situação no código

O protótipo atual usa tokenização por caractere, embeddings de posição aprendidos, atenção causal, LayerNorm, MLP com GELU, treino por próximo token e geração por amostragem com temperatura. BPE, RoPE, RMSNorm, SwiGLU, cache KV, quantização e conjunto de teste separado ainda não estão implementados.

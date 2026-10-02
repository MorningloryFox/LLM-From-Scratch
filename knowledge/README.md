# Knowledge: guia de estudo

As notas acompanham o código e mostram o que já está implementado, o que é uma alternativa e o que fica para depois. Cada capítulo apresenta a ideia, explica os termos, trabalha uma fórmula ou exemplo quando cabe e termina com uma pergunta para fixar.

1. [Fundamentos de tensores e formas](./Tensor_Foundations.md)
2. [Embeddings e geometria vetorial](./Embeddings_Vector_Geometry.md)
3. [Preparação do corpus](./Dataset_Preparation.md)
4. [Tokenização e BPE](./Tokenization.md)
5. [Atenção causal](./Attention_Mechanism.md)
6. [Bloco Transformer, LayerNorm/RMSNorm e GELU/SwiGLU](./Transformer_Block.md)
7. [Posição e RoPE](./RotaryPositionEmbedding.md)
8. [Gradientes e otimização](./Backpropagation_and_Optimization.md)
9. [Treino, validação e teste](./Training_and_Evaluation.md)
10. [Reprodutibilidade dos experimentos](./Experiment_Reproducibility.md)
11. [Métricas, parâmetros, cabeças e conexões de atenção](./Model_Metrics_and_Attention_Graphs.md)
12. [CPU, memória e desempenho](./CPU_Memory_and_Performance.md)
13. [Geração e amostragem](./Text_Generation.md)
14. [Cache KV](./KV_Cache.md)
15. [Quantização](./Quantization.md)
16. [Planejar modelos especialistas](./Specialist_Models.md)

## Situação no código

O Feneco-Char-0.1 usa tokenização por caractere, embeddings de posição aprendidos, atenção causal, LayerNorm, MLP com GELU, treino por próximo token e geração por amostragem com temperatura. BPE, RoPE, RMSNorm, SwiGLU, cache KV, quantização, avaliação de especialistas e conjunto de teste separado ainda não estão implementados.

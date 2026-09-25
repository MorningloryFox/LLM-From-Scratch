# LLM From Scratch

Repositório dedicado ao estudo, implementação e documentação dos fundamentos matemáticos, geométricos e de engenharia por trás dos Grandes Modelos de Linguagem (LLMs).

> **Meta do Projeto:** Construir um modelo de linguagem do zero (arquitetura *Decoder-Only*) focado em execução eficiente via CPU, registrando a transição da matemática teórica (álgebra linear, cálculo e geometria vetorial) para a implementação prática em código.

---

## 🧭 Mapa da Jornada de Aprendizado

- [x] **01. Embeddings & Espaços Vetoriais:** Representação de tokens, cálculo de similaridade por cosseno e aritmética vetorial.
- [ ] **02. Tokenização:** Fatiamento de texto bruto em tokens (BPE) e tabelas de mapeamento.
- [ ] **03. Mecanismos de Atenção:** Scaled Dot-Product Attention, matrizes de pesos ($Q, K, V$) e máscaras causais.
- [ ] **04. Arquitetura Transformer:** Camadas Feed-Forward (SwiGLU), RMSNorm e embeddings rotacionais (RoPE).
- [ ] **05. Inferência & KV-Cache:** Geração auto-regressiva, amostragem (Temperature, Top-$k$, Top-$p$) e otimização em VRAM/RAM.
- [ ] **06. Quantização & Aritmética Numérica:** Representação em FP16, INT8, INT4 e exportação em formato de bloco (GGUF).

---

## 🗂️ Estrutura do Banco de Conhecimento

As anotações detalhadas de cada módulo teórico e prático estão organizadas na pasta `/docs`:

* [`01_embeddings_e_geometria.md`](./docs/01_embeddings_e_geometria.md) — Conceitos de vetores, trigonometria, similaridade de cosseno e aritmética semântica.

---

## 🛠️ Licença

Este projeto está licenciado sob a Licença MIT - veja o arquivo [LICENSE](LICENSE) para mais detalhes.

---
*Desenvolvido por Morningloryfox 🦊*

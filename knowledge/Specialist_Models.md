# Planejar modelos especialistas 🦊

## 🎯 1. Conceito Fundamental

Um modelo especialista é construído e avaliado para um conjunto de tarefas delimitado. O objetivo da família Feneco não é fazer uma cópia pequena de um assistente geral; é escolher trabalhos específicos e verificar se cada modelo consegue fazê-los bem.

“Poderoso” precisa ser uma hipótese que possamos testar. Para cada especialista, devemos definir exemplos de sucesso, exemplos de falha, custo aceitável e limites antes de escolher tamanho ou arquitetura.

## 🧭 2. Modelo, dados e sistema são partes diferentes

Um modelo é a rede e seus pesos. Um sistema pode juntar o modelo com código, documentos, busca web ou outras ferramentas. Essa distinção muda o que precisamos construir:

* Um especialista de código pode precisar de bons exemplos e testes que executem os programas.
* Um especialista de domínio pode combinar geração com busca em documentos revisados.
* Um especialista de pesquisa/web precisa consultar fontes atuais. Pesos estáticos, sozinhos, não acessam páginas novas.

Recuperação de documentos (*RAG*) significa buscar trechos relevantes e fornecê-los ao modelo no momento da pergunta. A qualidade depende tanto da busca e das fontes quanto do modelo que compõe a resposta.

## 🧪 3. Feneco-Char e especialistas reais

**Feneco-Char-0.1** é um exercício didático por caractere. Ele ajuda a estudar embeddings, atenção, perda e geração. O corpus de demonstração não o transforma num assistente, programador ou pesquisador.

Depois do protótipo, podemos comparar estratégias para cada trabalho:

1. Treinar um modelo pequeno do zero com dados da tarefa.
2. Adaptar um modelo pré-treinado autorizado e compatível com o objetivo.
3. Usar um modelo com busca, documentos ou ferramentas.
4. Combinar adaptação e ferramentas, se a avaliação mostrar ganho.

Cada opção tem custos, dependências e capacidades diferentes. Vamos escolher depois de definir os dados, os limites de execução e a avaliação; não assumir que uma arquitetura serve para todos os especialistas.

## 🧰 4. Esboço das primeiras especialidades

| Família candidata | Escopo inicial | Avaliação que precisamos preparar |
| --- | --- | --- |
| **Feneco-Web-0.1** | Buscar fontes atuais, resumir evidências e responder com referências | Consultas conhecidas, qualidade/recência das fontes, suporte de cada afirmação e registro de links |
| **Feneco-Code-0.1** | Uma tarefa de programação delimitada, como explicar funções ou corrigir erros simples | Casos separados de treino, tarefas novas, execução controlada e testes de saída |
| **Feneco-[Assunto]-0.1** | Perguntas de um domínio específico escolhido depois | Questões representativas, respostas de referência, cobertura e origem das fontes |

Cada entrada é uma direção de planejamento, não uma capacidade já implementada. Feneco-Web descreve principalmente um **sistema** com busca e leitura de páginas, não apenas um checkpoint de pesos. O plano inicial de dados está em [Fontes de dados e coleta](./Data_Sources_and_Collection.md).

## 📏 5. Definir capacidade antes de treinar

Para uma tarefa, montamos um conjunto de avaliação que não será usado para ajustar pesos. Uma ficha pode registrar:

* O pedido exato que o modelo deve atender.
* Critério de sucesso verificável, como teste aprovado ou fonte que sustenta a resposta.
* Tipos de erro que são aceitáveis e erros que encerram o experimento.
* Latência, memória e tamanho máximos na máquina-alvo.
* Comparação com uma baseline simples e, se aplicável, execução sem o componente Feneco.

Nos especialistas de pesquisa/web, é preciso verificar se a resposta representa fielmente as fontes e se cada referência sustenta o trecho associado. Citar uma URL sem conferir o conteúdo não basta.

## 🗺️ 6. Sequência de decisão

1. Escolher uma tarefa pequena e uma pessoa que usaria o resultado.
2. Escrever exemplos de sucesso, casos-limite e critérios de falha.
3. Preparar corpus e avaliação separados, com origem e direito de uso registrados.
4. Medir uma baseline antes de treinar.
5. Comparar do zero, adaptação e/ou ferramentas com orçamento e dados compatíveis.
6. Treinar na máquina local, avaliar e guardar ficha do experimento.
7. Decidir depois se os pesos são úteis e se vale compartilhá-los.

## 💡 Pergunta para Fixar

Para responder “o que mudou numa página hoje?”, o que precisa ser aprendido nos pesos e o que precisa vir de uma ferramenta conectada à web? Como testaríamos se o resumo realmente é apoiado pelas fontes encontradas?

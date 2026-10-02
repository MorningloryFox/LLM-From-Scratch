# Fontes de dados para a família Feneco 📚

## 🎯 1. Conceito Fundamental

Antes de baixar textos, precisamos decidir **para que capacidade eles servem**. Um especialista web precisa aprender a formular buscas, comparar evidências e citar fontes. Ele também precisa consultar conteúdo atual durante a pergunta. Uma coleção fixa usada no treino não substitui a busca na web em tempo real.

Vamos separar três conjuntos:

1. **Dados de treino:** exemplos usados para ajustar pesos ou ensinar formatos de resposta.
2. **Fontes recuperadas:** páginas ou documentos consultados no momento da pergunta.
3. **Avaliação:** perguntas e fontes mantidas fora do treino para medir qualidade.

Misturar essas funções dificulta saber se o modelo aprendeu a pesquisar, decorou respostas ou apenas viu o teste durante o treino.

## 🌐 2. Primeiro especialista: Feneco-Web-0.1

O Feneco-Web vai combinar um modelo com uma ferramenta de busca/leitura de páginas. O modelo pode aprender a transformar uma pergunta em plano de busca, resumir trechos recuperados e associar afirmações às fontes. A ferramenta consulta páginas atuais; os pesos do modelo não mudam a cada notícia.

Um exemplo de dado de treino útil pode conter:

```text
pergunta → busca sugerida → trechos recuperados → resposta baseada nos trechos → referências
```

No começo, podemos escrever ou revisar manualmente um conjunto pequeno de exemplos usando fontes permitidas. Dados sintéticos podem ajudar a testar formatos e casos-limite, mas não devem ser tratados como evidência factual sem verificação humana.

## 🗂️ 3. Fontes candidatas e limites

| Fonte candidata | Uso possível | Verificação necessária |
| --- | --- | --- |
| Exemplos escritos por nós | Ensinar formato de busca, comparação de evidências e citações | Revisar cada resposta e apontar a fonte que a sustenta |
| Documentação oficial de um domínio | Criar exemplos e uma coleção inicial para recuperação | Checar licença, termos do site, revisão/data e permissão para armazenar |
| Conteúdo Wikimedia | Material de referência; pode ser candidato após análise da licença | Os termos indicam CC BY-SA para grande parte do texto e exigem atribuição; páginas podem ter conteúdo importado ou condições específicas. Registrar artigo, revisão e atribuição necessária. |
| Dump do Stack Overflow | Possível fonte para perguntas técnicas, não primeira escolha | Licença CC BY-SA varia por revisão/data; atribuição e compartilhamento sob termos aplicáveis precisam ser avaliados antes do uso. |
| Busca web em tempo real | Recuperar evidência atual na hora da consulta | Escolher uma API/ferramenta e seguir seus termos, limites, regras de armazenamento e acesso às páginas |

As próprias fontes oficiais descrevem essas condições: [termos de uso da Wikimedia](https://foundation.wikimedia.org/wiki/Terms_of_Use), [licença do conteúdo do Stack Overflow](https://stackoverflow.com/help/licensing) e [termos da rede pública do Stack Overflow](https://stackoverflow.com/legal/terms-of-service/public). “Está público” não significa que podemos baixar e usar sem verificar licença e condições.

Para o primeiro protótipo web, minha sugestão é começar com exemplos próprios e um conjunto pequeno de documentos oficiais selecionados, com permissão e proveniência registradas. Não começaria por um rastreamento amplo da web nem pelo dump completo do Stack Overflow.

## 🦊 4. E os próximos Fenecos?

### Feneco-Code-0.1

Começaríamos com uma linguagem e uma tarefa delimitadas — por exemplo, explicar pequenos trechos ou corrigir erros simples. As fontes candidatas são código nosso, documentação oficial e repositórios escolhidos individualmente com licença explícita. Um repositório público sem licença clara fica fora do corpus.

Os exemplos de avaliação precisam vir de repositórios ou tarefas que não entraram no treino. Quando possível, executamos testes em ambiente isolado para medir se a solução funciona, além de avaliar explicações.

### Feneco-[Assunto]-0.1

O assunto ainda precisa ser escolhido. As fontes devem ser apropriadas ao domínio: documentação oficial, material autorizado, publicações com licença compatível ou conteúdo próprio. Se atualidade for importante, o especialista também pode usar recuperação de documentos em vez de tentar guardar tudo nos pesos.

## 🧾 5. Registro de cada fonte

Antes de incluir um item, registrar:

* URL, título, autor ou organização responsável.
* Data de acesso e data/revisão do conteúdo.
* Licença e termos relevantes, incluindo restrições e atribuição exigida.
* Tipo de uso: treino, recuperação, validação ou teste.
* Hash do arquivo e transformação aplicada.
* Decisão de inclusão e responsável pela revisão.

Separar dados por documento ou fonte antes de criar janelas de treino. Remover duplicatas entre partições e guardar perguntas de avaliação longe dos dados que ajustam os pesos. Dados privados permanecem locais; checkpoints e corpus não são enviados ao Git automaticamente.

## 🗺️ 6. Plano de coleta para o primeiro especialista

1. Descrever as perguntas que o Feneco-Web deve responder e o que conta como boa evidência.
2. Selecionar um pequeno conjunto de fontes com licença/termos identificáveis.
3. Escrever exemplos pergunta–busca–trecho–resposta–citação e revisar a correspondência com a fonte.
4. Separar consultas e fontes de avaliação antes do treino.
5. Escolher a ferramenta de busca/leitura conforme disponibilidade e termos; ainda não há fornecedor escolhido.
6. Comparar a resposta com fontes recuperadas e sem elas, medindo suporte factual, cobertura, qualidade das referências e latência.
7. Só aumentar o corpus se o resultado mostrar uma lacuna que os dados adicionais possam corrigir.

Como primeiro exemplo, `data/ua000180.txt` contém uma transcrição em texto simples de *A Carteira*, de Machado de Assis. O registro `data/ua000180.metadata.json` documenta a página do Wikisource, que identifica a obra como domínio público, e a ficha correspondente no Portal Domínio Público. O PDF inicialmente extraído tinha problemas de mapeamento de caracteres; por isso, o texto final veio da transcrição do Wikisource, não do PDF. É um item piloto, não um corpus suficiente para treinar a base compartilhada. Outras fontes ainda precisam ser escolhidas e revisadas individualmente antes da coleta.

## 💡 Pergunta para Fixar

Se o Feneco-Web responde corretamente porque encontrou uma página atual, essa informação deve ser memorizada nos pesos, recuperada da página a cada pergunta ou registrada nos dois lugares? O que precisamos guardar para conferir depois de onde veio a resposta?

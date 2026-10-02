# Reprodutibilidade dos experimentos 🧪

## 🎯 1. Conceito Fundamental

Um resultado só é útil se soubermos **quais condições produziram aquele resultado**. Reprodutibilidade é conseguir refazer a execução e entender por que duas execuções diferem.

Treino de rede neural envolve aleatoriedade: pesos começam com valores aleatórios, o programa escolhe janelas do corpus e a geração amostra tokens. Mesmo com uma semente fixa, versões diferentes de PyTorch, hardware ou operações paralelas podem gerar pequenas diferenças.

## 🧾 2. O que registrar

Uma ficha de experimento deve guardar pelo menos:

* Nome/versão do modelo, commit do código e data.
* Identificador e hash do corpus; fonte, licença e transformações.
* Tokenizador, vocabulário e método de divisão treino/validação/teste.
* Configuração: contexto, dimensão, camadas, cabeças e dropout.
* Treino: semente, batch, passos, taxa de aprendizado e otimizador.
* Ambiente: Python, PyTorch, dispositivo e, se possível, CPU/GPU.
* Resultados: perdas, duração, contagem de parâmetros e amostras geradas.
* Geração: prompts, temperatura e outros parâmetros de amostragem.

O conteúdo do corpus privado não precisa ser copiado para o relatório. Hash e metadados não sensíveis ajudam a reconhecer o arquivo sem revelar o texto.

## 🎲 3. Sementes e aleatoriedade

Uma semente inicializa um gerador pseudoaleatório. Se usarmos a mesma semente e o mesmo ambiente, podemos repetir muitas escolhas aleatórias.

Para o PyTorch, a chamada básica é:

```python
torch.manual_seed(42)
```

O script atual usa uma semente configurável para o PyTorch e para as janelas sorteadas. Ativar operações determinísticas pode reduzir diferenças, mas algumas operações ficam mais lentas ou deixam de estar disponíveis.

Uma semente fixa ajuda a comparar alterações, mas não transforma toda execução em uma garantia bit a bit. Devemos registrar versões e hardware junto com a semente.

## 📦 4. Checkpoint: inferir ou retomar o treino?

Um checkpoint de inferência precisa dos pesos, da configuração e do vocabulário/tokenizador. O checkpoint atual do Feneco guarda estado do modelo, configuração, vocabulário, passos, contagem de parâmetros, perdas selecionadas e hash do corpus.

Para retomar o treino exatamente de onde parou, também precisamos do estado do otimizador, do passo atual e dos estados dos geradores aleatórios. Esses itens ainda não são guardados pelo script. Logo, o checkpoint atual permite gerar texto e avaliar; não reproduz integralmente o caminho de otimização.

Os pesos são artefatos locais ignorados pelo Git. Corpus privado também deve permanecer local.

## 🔬 5. Comparação controlada

Suponha que queremos estudar o efeito de mudar o comprimento de contexto. Uma comparação inicial pode fixar corpus, divisão, semente, tamanho de modelo, número de passos e taxa de aprendizado; mudar apenas o contexto; então comparar perda, tempo e amostras.

Depois podemos repetir com várias sementes. Se a diferença entre duas configurações for menor do que a variação entre sementes, ainda não temos evidência forte de que a mudança ajudou.

Um hash SHA-256 do arquivo permite confirmar qual corpus foi usado sem publicar o conteúdo:

```powershell
Get-FileHash data\tiny.txt -Algorithm SHA256
```

## 💡 Pergunta para Fixar

Se queremos saber se uma nova taxa de aprendizado melhora o treino, quais campos devem ficar iguais entre as duas execuções? Por que uma semente igual, sozinha, não garante uma reprodução perfeita em qualquer máquina?

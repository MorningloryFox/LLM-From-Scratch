# Preparação do corpus 📚

## 🎯 1. Conceito Fundamental

O corpus é o conjunto de textos a partir do qual o modelo aprende padrões. A escolha dos textos influencia o vocabulário, as sequências vistas no treino e as respostas que podemos avaliar. Antes de rodar o treino, precisamos saber **de onde veio o texto**, **como foi transformado** e **que trechos foram separados para avaliação**.

Um corpus pequeno pode servir para aprender como o código funciona. Ele não permite concluir que o modelo aprendeu linguagem de forma ampla.

## 📄 2. Origem e formato do texto

Use apenas textos que você tem direito de usar. Registre a fonte, licença ou autorização, idioma, data de coleta e regras de processamento. Dados pessoais ou corpus privado devem ficar na máquina e fora do Git.

O script atual lê um arquivo UTF-8:

```python
text = args.data.read_text(encoding="utf-8")
```

Codificação determina como bytes viram caracteres. UTF-8 é comum e representa letras acentuadas; escolher a codificação errada pode causar erros ou caracteres corrompidos.

## 🧹 3. Limpeza sem apagar a evidência

Limpar texto pode mudar o que o modelo aprende. Exemplos:

* Converter tudo para minúsculas junta `A` e `a` numa só unidade.
* Remover acentos troca `ação` por `acao`.
* Colapsar espaços altera recuos e quebras de linha.
* Remover HTML ou menus de páginas pode ser útil, mas regras ruins também podem apagar conteúdo.

Registre transformações e use as mesmas regras no treino e na geração. Para o primeiro modelo, uma transformação mínima é mais fácil de entender do que uma limpeza extensa e invisível.

Duplicatas merecem atenção: se a mesma passagem aparecer muitas vezes, o modelo verá esse trecho com frequência desproporcional. Se uma cópia ficar no treino e outra no teste, a avaliação pode parecer boa sem medir generalização para texto novo.

## ✂️ 4. Separar treino, validação e teste

* **Treino:** exemplos usados para atualizar os pesos.
* **Validação:** trecho separado para comparar configurações e perceber sobreajuste.
* **Teste:** trecho mantido de lado até escolhermos a configuração final; usamos no fim para uma avaliação reservada.

Em texto ordenado, uma divisão por segmentos contíguos reduz o risco de janelas quase idênticas atravessarem a fronteira. Para coleções de documentos, podemos dividir por documento ou por grupos de documentos relacionados.

### Exemplo simples

Imagine 100 linhas independentes sobre vários assuntos. Uma divisão possível é separar 80 linhas para treino, 10 para validação e 10 para teste. Se as linhas são partes consecutivas de um mesmo artigo, é melhor agrupar por artigo primeiro, para que versões vizinhas não apareçam em partições diferentes.

As proporções são uma escolha experimental, não uma lei. Em corpus muito pequeno, reservar muita coisa reduz o treino; reservar pouco torna a avaliação instável. O protocolo deve ser descrito junto com o resultado.

## 🔎 5. O que o Feneco faz atualmente

O treino lê um único arquivo, cria um vocabulário com `sorted(set(text))`, converte o texto todo para IDs e reserva os últimos 10% como validação. As janelas de treino são escolhidas dentro dos primeiros 90%; as de validação vêm do segmento final.

O vocabulário é criado **antes** da divisão. Assim, o treino vê quais caracteres existem na validação, embora não veja as sequências de validação. É um vazamento limitado ao conjunto de símbolos, mas o protocolo mais limpo é dividir primeiro e construir o vocabulário ou aprender o BPE só com o trecho de treino. O projeto ainda não tem teste separado nem suporte para múltiplos documentos.

## 🗂️ 6. Ficha do corpus

Para cada experimento, registre:

1. Fonte e direito de uso.
2. Codificação, idioma e tamanho original.
3. Normalizações, filtros e duplicatas tratados.
4. Tokenizador e tamanho do vocabulário.
5. Método e tamanhos das divisões.
6. Caminho ou identificador e hash do arquivo.

Um hash identifica se dois arquivos são iguais sem publicar o conteúdo. No Windows, pode-se obter com:

```powershell
Get-FileHash data\tiny.txt -Algorithm SHA256
```

## 💡 Pergunta para Fixar

Se um livro aparece duas vezes no corpus e uma cópia fica no treino e outra no teste, o que o resultado de teste ainda consegue nos dizer? Por que a divisão por documento pode ser mais adequada do que a divisão por linha?

# 🧠 Atividade 1: Rede Neural para Classificação da Porta XOR

**Disciplina:** Inteligência Artificial  
**Professor:** David Berto Farias  
**Aluno:** Renato Calabro  
**Entrega:** Projeto completo em Python puro, sem bibliotecas de machine learning.

---

## 🎯 Objetivo

Construir uma **rede neural feedforward** em Python capaz de aprender e predizer o comportamento da porta lógica **XOR**, utilizando apenas NumPy e implementações próprias de feedforward, funções de ativação, backpropagation e cálculo de erro.

---

## ⚙️ Arquitetura da Rede

Para modelar a porta XOR, utilizei uma rede neural com a seguinte estrutura:

    [2 entradas] → [2 neurônios na camada oculta] → [1 neurônio na saída]

- **Input layer:** 2 neurônios (representando as entradas A e B da porta XOR)
- **Hidden layer:** 2 neurônios
- **Output layer:** 1 neurônio

### 🔎 Por que `2-2-1` (camadas)?

A função XOR é um clássico exemplo de **problema não linearmente separável**. Isso significa que **não existe uma linha reta (ou plano, ou hiperplano)** que consiga separar os casos onde a saída é `1` dos casos onde a saída é `0`.

Uma rede neural com apenas **camadas lineares (sem não-linearidade)** não consegue resolver esse tipo de problema.

Ao incluir uma camada com dois neurônios e usar uma função de ativação não linear (como a sigmoide), a rede passa a conseguir “dobrar” o espaço de decisão. Isso permite que ela separe corretamente os casos da XOR, mesmo que eles não possam ser separados com uma linha reta.

> Essa arquitetura `2-2-1` é a forma mais simples e eficiente de resolver a XOR com redes neurais.

## 🧪 Função de Ativação: `sigmoid`

Utilizei a função de ativação **sigmoide** (`sigmoid(x) = 1 / (1 + e^-x)`) em todos os neurônios.

### ✅ Por que `sigmoid`?

- Retorna valores contínuos entre 0 e 1 → ideal para classificações binárias
- Tem derivada simples → facilita a implementação do backpropagation
- Ajuda a suavizar os ajustes durante o treinamento

### ⚠️ Quando **não** usar sigmoid?

- Para redes muito profundas → pode causar o problema do **gradiente desaparecendo**
- Quando a saída desejada for valores não normalizados (ex: regressão)

## 📉 Função de Custo: `Binary Cross-Entropy`

Para o treinamento, utilizei a função de custo **Binary Cross-Entropy**, apropriada para problemas de classificação binária.

```python
loss(y, y_pred) = -[y * log(y_pred) + (1 - y) * log(1 - y_pred)]
```

### ✅ Por que Binary Cross-Entropy?

- É **especializada para classificações binárias**
- Penaliza mais fortemente as predições incorretas com alta confiança
- Gera gradientes mais úteis para sigmoid (melhor que MSE nesse caso)

## 🏋️ Parâmetros de Treinamento

A rede foi treinada com backpropagation "vanilla", implementado do zero, utilizando os seguintes hiperparâmetros:

- **Épocas:** `1000`
- **Taxa de aprendizado (learning rate):** `1`
- **Erro alvo (`target_error`):** `0.25`

Esses valores foram **suficientes para resolver a porta XOR com sucesso**, atingindo o erro mínimo desejado de forma rápida e estável.

## ✅ Resultado

A rede aprendeu corretamente a lógica da porta XOR, classificando corretamente os seguintes casos:

| Entrada A | Entrada B | Saída Esperada | Saída da Rede |
|-----------|-----------|----------------|----------------|
|     0     |     0     |       0        |       0        |
|     0     |     1     |       1        |       1        |
|     1     |     0     |       1        |       1        |
|     1     |     1     |       0        |       0        |

## 🧮 Função de Classificação: `xor_classification`

Após o treinamento da rede com saída contínua entre `0` e `1`, foi implementada uma função de **pós-processamento** para transformar a saída da rede em **valores binários discretos** (`0` ou `1`), conforme o esperado pela porta XOR.

```python
def xor_classification(pred):
    def toClass(value):
        return (value > 0.5).astype(int)

    return np.array([toClass(x[0]) for x in pred])
```

### ✅ O que ela faz:

- Recebe a saída contínua da rede (ex: `[[0.82], [0.12]]`)
- Aplica um limiar de decisão `> 0.5`
- Converte cada saída em `1` ou `0` (`int`)

### ✅ Por que usar:

- A rede usa `sigmoid`, que retorna **valores contínuos**
- Para aplicações com lógica binária como a porta XOR, precisamos **decidir** se o valor representa um `1` ou `0`
- Com isso, garantimos uma saída compatível com a tabela verdade da porta XOR

A função foi atribuída à rede via o parâmetro `output_classification`, o que permite que a rede **retorne diretamente o valor classificado** após o `predict`, sem necessidade de processamento externo.

## 🧰 Observações

- Todo o projeto foi feito com **NumPy puro**, sem uso de frameworks de machine learning
- A estrutura foi feita para ser modular, com suporte a múltiplas funções de ativação e estratégias de treino
- O código pode ser expandido facilmente para resolver outras portas lógicas ou problemas não lineares

## 🔗 Histórico de Desenvolvimento

Todo o histórico de desenvolvimento, incluindo decisões de projeto, refatorações, testes e melhorias contínuas, está disponível publicamente no repositório abaixo:

> 📂 **GitHub:** [github.com/rcalabro/activity1-xor-prediction](https://github.com/rcalabro/activity1-xor-prediction)

Esse repositório contém:
- Commits detalhados com cada etapa do progresso
- Código modular e documentado
- Estrutura pensada para reusabilidade e expansão didática

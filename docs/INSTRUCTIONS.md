# 🧠 XOR Neural Network - Atividade 1

Projeto de rede neural feedforward implementado **do zero com NumPy**, com objetivo de **resolver a porta lógica XOR**.

## 📦 Requisitos

Antes de executar o projeto, certifique-se de ter o Python instalado (recomendado: Python 3.8 ou superior).

## 🧪 Ambiente Virtual (Recomendado)

É **altamente recomendado** criar um ambiente virtual para isolar as dependências do projeto.

### 🔹 Criar o ambiente virtual

```bash
python -m venv xor-project
```

### 🔹 Ativar o ambiente virtual

- **Windows:**
  ```bash
  venv\Scripts\activate
  ```
- **Linux/MacOS:**
  ```bash
  source xor-project/bin/activate
  ```

---

## 📥 Instalar dependências

Com o ambiente virtual ativado, instale as dependências com:

```bash
pip install -r requirements.txt
```

## ▶️ Como Executar

Execute o arquivo `main.py`:

```bash
python main.py
```

Dependendo da variável `load_checkpoint` no início do `main.py`, o script:

- ✅ **Carrega pesos salvos** do arquivo `.npz`
- 🔁 **Ou treina a rede do zero**, salva os pesos, e testa

## 🔍 Estrutura do `main.py`

### 📌 `create_xor_nn(...)`
Cria ou carrega a rede neural. Estrutura mínima:
- 2 neurônios de entrada
- 1 camada oculta com 2 neurônios
- 1 neurônio de saída
- Ativação: `sigmoid`

### 📌 `train_xor(...)`
Treina a rede com:
- epochs = 1000
- learning_rate = 1
- target_error = 0.25
- Estratégia: `vanilla-backpropagation`

### 📌 `main()`
Executa:
1. Criação (ou carregamento) da rede
2. Treinamento (se necessário)
3. Testes de predição nos 4 casos XOR
4. Visualização opcional com `matplotlib`

## 🧪 Casos de Teste

| Entrada A | Entrada B | Esperado |
|-----------|-----------|----------|
|     0     |     0     |    0     |
|     0     |     1     |    1     |
|     1     |     0     |    1     |
|     1     |     1     |    0     |

A saída da rede é automaticamente convertida em `0` ou `1` com base no valor `> 0.5`.

## 📊 Visualização (Opcional)

Para ativar a visualização gráfica da rede:

```python
plot = True
```

## 💾 Checkpoints

Pesos e bias da rede são salvos em:

```
./checkpoints/xor_nn.npz
```

## ✍️ Autor

Desenvolvido por **Renato Calabro**  
🔗 [github.com/rcalabro/activity1-xor-prediction](https://github.com/rcalabro/activity1-xor-prediction)


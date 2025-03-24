# 🧠 Atividade 1: Redes Neurais Artificiais e suas Principais Aplicações

## 🏫 CENTRO UNIVERSITÁRIO SENAC – SANTO AMARO
### 💻 Pós-graduação em Inteligência Artificial
**Disciplina:** Redes Neurais Artificiais e suas Principais Aplicações  
**Professor:** David Berto Farias

## 📌 Atividade 1
**Elaboração de Projeto para Classificação XOR utilizando Redes Neurais**

## 📄 Descrição da Atividade

A proposta da atividade pode ser consultada no documento abaixo:  
📂 [`ACTIVITY.md`](./docs/ACTIVITY.md)

## 💡 Solução Proposta

A explicação completa do raciocínio por trás da escolha da arquitetura, funções de ativação, estratégia de treinamento e função de custo está em:  
📄 [`SOLUTION.md`](./docs/SOLUTION.md)

## 🗂️ Estrutura do Repositório

A organização das pastas e arquivos do projeto foi pensada para manter separação de responsabilidades, facilitar a manutenção e tornar o projeto mais didático.

Para mais detalhes, consulte o documento completo:  
📄 [`REPOSITORY.md`](./docs/REPOSITORY.md)

## 🚀 Como Executar

Todos os passos para:

- Criar a rede
- Treiná-la (caso não haja pesos salvos)
- Carregar pesos previamente treinados
- Testar os resultados da porta XOR
- (Opcional) Visualizar graficamente a rede

estão descritos no arquivo:

📘 [`INSTRUCTIONS.md`](./docs/INSTRUCTIONS.md)

## 🧱 Arquitetura Resumida

O projeto é organizado de forma modular para separar responsabilidades:

- 🧠 **NeuralNetwork**: Lida com estrutura da rede, ativação e forward pass
- 🏋️ **Trainer**: Controla o processo de treinamento e execução
- ⚙️ **Strategies**: Estratégias plugáveis de execução e de atualização de pesos
- 📊 **Plot**: Visualização gráfica da rede neural
- 📁 **Docs**: Toda documentação da atividade e explicações técnicas
- 💾 **Checkpoints**: Pesos salvos da rede para evitar reprocessamento
- 🚀 **main.py**: Ponto de entrada da aplicação (treina ou testa a rede)

Para mais detalhes visuais, consulte o diagrama em [`ARCHITECTURE.md`](./docs/ARCHITECTURE.md)

---

> 📝 **Nota sobre a documentação**  
> Esta documentação foi desenvolvida com o suporte do **ChatGPT** para auxiliar na organização, estruturação e formatação dos conteúdos.
>
> As decisões técnicas, estrutura do projeto, argumentações e explicações conceituais foram elaboradas pelo autor, garantindo aderência ao propósito didático da atividade.
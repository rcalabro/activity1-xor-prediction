# 🗂️ Estrutura do Repositório

Este documento descreve a organização geral dos diretórios e arquivos do projeto `activity1-xor-prediction`.


## 📁 `checkpoints/`
Armazena os arquivos `.npz` contendo os pesos e bias da rede neural treinada. Permite reutilizar redes já treinadas sem precisar repetir o processo.


## 📁 `docs/`
Documentação do projeto, incluindo:

- `ACTIVITY.md`: Enunciado da atividade
- `SOLUTION.md`: Explicação da solução e decisões técnicas
- `INSTRUCTIONS.md`: Passo a passo para executar o projeto
- `REPOSITORY.md`: Este documento
- `ARCHITECTURE.md`: Diagrama da arquitetura com Mermaid


## 📁 `scripts/`
Scripts auxiliares opcionais e arquivos temporários. O `.gitkeep` permite manter o diretório versionado mesmo vazio.

## 📁 `src/`
Código-fonte da aplicação.

- `main.py`: Script principal que executa a rede neural, realiza o treinamento (se necessário) e testa a saída para os casos da porta XOR.

- `neural_network/`: Implementação da rede neural, visualização gráfica e funções de ativação.

- `neural_network/trainer/`: Camada de treinamento, com suporte a múltiplas estratégias de execução e atualização.


## 📄 `requirements.txt`
Lista de dependências para instalar com `pip`.
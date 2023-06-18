# Processamento de Linguagem Natural (NLP)

nltk_utils.py. Este código demonstra um exemplo de processamento de linguagem natural utilizando a biblioteca NLTK e NumPy.

## Pré-requisitos

- NLTK
- Corpus 'punkt'

## Funções

- `tokenize(sentence)`: Função para tokenizar uma sentença.
- `stem(word)`: Função para realizar a redução de palavras (stemming).
- `bag_of_words(tokenized_sentence, all_words)`: Função para criar a representação Bag of Words.

Essas funções são úteis para preparar e processar dados de texto em aplicações de NLP.


# Instalando e carregando pacotes necessários em model.py

Este trecho de código instala e carrega os pacotes necessários para a implementação da rede neural.

# Definindo a arquitetura da rede neural

Aqui é definida a classe `NeuralNet`, que é uma subclasse da classe `nn.Module` do PyTorch. A rede neural possui três camadas lineares (fully connected), com funções de ativação ReLU aplicadas entre elas.

- `l1`: Primeira camada linear que recebe um tamanho de entrada (`input_size`) e produz um tamanho oculto (`hidden_size`).
- `l2`: Segunda camada linear que recebe o tamanho oculto e produz outro tamanho oculto.
- `l3`: Terceira camada linear que recebe o tamanho oculto e produz o número de classes (`num_classes`).
- `relu`: Função de ativação ReLU.

# Forward pass da rede neural

A função `forward` define o fluxo de passagem direta dos dados pela rede neural. Ela recebe um tensor `x` como entrada e realiza as seguintes etapas:

1. Passa `x` pela primeira camada linear `l1`.
2. Aplica a função de ativação ReLU.
3. Passa o resultado pela segunda camada linear `l2`.
4. Aplica a função de ativação ReLU novamente.
5. Passa o resultado pela terceira camada linear `l3`.

O resultado final é retornado como saída da rede neural.

É importante notar que neste caso não é aplicada uma função de ativação final (como sigmoid ou softmax), pois a função de perda cross-entropy será aplicada posteriormente.

# Treinamento de Chatbot com PyTorch em train_data.py

Este código demonstra um exemplo de treinamento de um chatbot utilizando a biblioteca PyTorch. Ele utiliza um conjunto de dados de intenções (`intents.json`) e realiza o pré-processamento dos dados de texto para treinar uma rede neural.

## Pré-requisitos

- Python 3.x
- PyTorch
- NumPy

## Passos do Código

1. Carregar as intenções do arquivo `intents.json`.
2. Pré-processar os dados de texto:
   - Tokenizar as sentenças.
   - Reduzir as palavras para suas formas básicas (stemming).
   - Criar a representação Bag of Words para cada sentença.
3. Dividir os dados em conjuntos de treinamento e teste.
4. Definir a arquitetura da rede neural:
   - Utilizar a classe `NeuralNet` do arquivo `model.py`.
   - Configurar o tamanho de entrada, tamanho oculto e número de classes.
   - Utilizar a função de ativação ReLU entre as camadas.
5. Definir os hiperparâmetros do treinamento:
   - Tamanho do lote (batch size), tamanho oculto, taxa de aprendizagem, número de épocas.
6. Criar um objeto `Dataset` para armazenar os dados de treinamento.
7. Criar um objeto `DataLoader` para carregar os dados em lotes durante o treinamento.
8. Verificar a disponibilidade de GPU e configurar o dispositivo de treinamento.
9. Definir a função de perda (loss) como a Cross-Entropy Loss.
10. Definir o otimizador como Adam para atualizar os parâmetros da rede neural.
11. Realizar o treinamento:
    - Executar o número especificado de épocas.
    - Iterar sobre os lotes de treinamento.
    - Calcular a saída da rede neural e a perda.
    - Realizar a retropropagação e a atualização dos parâmetros.
12. Salvar o modelo treinado em um arquivo `data.pth`.
13. Imprimir a mensagem de conclusão do treinamento.

Certifique-se de ter instalado as bibliotecas necessárias e execute o código em um ambiente Python compatível.


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

## Importação de pacotes necessários em chatbox.py:

1. O código começa importando os pacotes necessários para o funcionamento adequado do script. Esses pacotes são utilizados para diferentes funcionalidades, como geração de valores aleatórios, manipulação de arquivos JSON, processamento de texto e criação de modelos de aprendizado de máquina.

2. Definição do dispositivo:

   - Nesta parte do código, é verificado se a GPU está disponível. Caso esteja, o dispositivo é definido como 'cuda', caso contrário, é definido como 'cpu'. Essa verificação é importante para aproveitar os recursos da GPU quando possível e, assim, acelerar o processamento.

3. Leitura do arquivo JSON:

   - O código faz a leitura de um arquivo JSON chamado 'intents.json'. Esse arquivo contém informações sobre as intenções do chatbot, como tags, perguntas e respostas associadas. A leitura do arquivo é feita usando a biblioteca json, que permite carregar o conteúdo do arquivo em uma estrutura de dados adequada para ser utilizada posteriormente.

# Em tf-idf.py
Importe os pacotes necessários:

- `pandas`: Biblioteca para manipulação de dados tabulares.
- `TfidfVectorizer` da biblioteca `sklearn.feature_extraction.text`: Usado para transformar texto em vetores de características usando a técnica TF-IDF.
- `cosine_similarity` da biblioteca `sklearn.metrics.pairwise`: Usado para calcular a similaridade de cosseno entre os vetores de características.

# Carregando os dados
Carregue os dados de um arquivo CSV chamado 'movies.csv' e armazene-os em um dataframe chamado `movies_data`.

# Selecionando características importantes
Selecione as características importantes do dataframe `movies_data`. As características selecionadas podem incluir gêneros, palavras-chave, elenco, diretor, título, data de lançamento, entre outros.

# Tratando valores ausentes
Trate os valores ausentes (NaN) em cada característica selecionada, preenchendo-os com uma string vazia.

# Juntando todas as características selecionadas para análise
Combine as características selecionadas em uma única string chamada `combined_features`. Isso é feito concatenando as informações das colunas de gêneros, palavras-chave, slogan, elenco e diretor para cada filme.

# Transformando os dados de texto em vetores de características
Transforme os dados de texto em vetores de características usando o TfidfVectorizer.

# Criando uma lista com todos os nomes de filmes presentes no conjunto de dados
Crie uma lista com os nomes de todos os filmes presentes no conjunto de dados.

# Similaridade de cosseno
Calcule a similaridade de cosseno entre os vetores de características.

# Salvando arquivos
Salve os resultados da similaridade em um arquivo usando o formato Parquet com compressão gzip.

<!-- Carregando e preparando os dados -->
- Importe os pacotes necessários: `pandas` e `difflib`.
- Carregue os dados de um arquivo CSV chamado 'movies.csv' e armazene-os em um dataframe chamado `movies_data`.
- Crie uma lista chamada `list_of_all_titles` contendo todos os títulos dos filmes presentes no conjunto de dados.

<!-- Função para recomendação de filmes -->
- Defina uma função chamada `movie_rec` que recebe dois parâmetros: `movie_name` e `similarity`.
- Utilize a função `get_close_matches` do módulo `difflib` para encontrar a correspondência mais próxima para o título do filme fornecido pelo usuário.
- Se não houver correspondência próxima, retorne uma mensagem informando que o filme não é conhecido.
- Caso contrário, continue com o processo de recomendação.

<!-- Encontrando filmes semelhantes -->
- Obtenha o índice do filme correspondente ao título encontrado.
- Crie uma lista de tuplas contendo o índice do filme e sua pontuação de similaridade.
- Ordene a lista de filmes semelhantes com base na pontuação de similaridade, em ordem decrescente.
- Extraia os seis filmes mais similares da lista e obtenha informações como título, data de lançamento e diretor.
- Formate a mensagem de recomendação com base nos filmes similares encontrados.
- Retorne a mensagem de recomendação ao usuário.

<!-- Uso da função -->
- Utilize a função `movie_rec` passando o título do filme fornecido pelo usuário e a matriz de similaridade dos filmes.
- A função retornará uma mensagem contendo recomendações de filmes semelhantes com base no título fornecido.


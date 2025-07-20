# 🧠 Classificador de Dígitos MNIST com TensorFlow/Keras

Projeto completo, desenvolvido no Google Colab, para construir, treinar e avaliar uma Rede Neural capaz de classificar dígitos manuscritos (0 a 9) do famoso dataset MNIST.

## 📋 Tabela de Conteúdos
* [Sobre o Projeto](##-sobre-o-projeto)
* [ Ferramentas Utilizadas](#️-ferramentas-utilizadas)
* [ Como Executar o Projeto](#-como-executar-o-projeto)
* [ Resultados](#-resultados)

## Sobre o Projeto
O objetivo central é demonstrar um fluxo de trabalho de ponta a ponta em Machine Learning: desde a preparação dos dados e construção do modelo, até o treinamento e a avaliação de sua performance. O modelo é uma Rede Neural simples, mas eficaz, que aprende a reconhecer padrões em imagens de 28x28 pixels para fazer suas classificações.

## Ferramentas Utilizadas
* **Ambiente de Desenvolvimento:** `Google Colab`
* **Biblioteca de Machine Learning:** `TensorFlow` com a API `Keras`
* **Bibliotecas de Apoio:** `NumPy` para manipulação de dados e `Matplotlib` para visualização.

## Como Executar o Projeto
Abra o notebook (`.ipynb`) no [Google Colab](https://colab.research.google.com/) e siga os passos abaixo, executando cada célula de código em sequência.

### Passo 1: Ativar o Acelerador de Hardware (GPU)
Para um treinamento muito mais rápido, é recomendado utilizar a GPU gratuita do Colab.
1. No menu, navegue até **Ambiente de execução > Alterar o tipo de ambiente de execução**.
2. Na janela que se abre, selecione **`T4 GPU`** no menu suspenso "Acelerador de hardware".
3. Clique em "Salvar".

### Passo 2: Importar as Bibliotecas
Vamos carregar todas as dependências necessárias para o projeto.
```python
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

print("Bibliotecas importadas com sucesso!")
```
Passo 3: Carregar o Dataset MNIST
O Keras facilita o acesso ao dataset MNIST, que já vem dividido em conjuntos de treino e teste.

```
# Carrega o dataset e já o divide em partes de treino e de teste
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

print(f"Temos {x_train.shape[0]} imagens para treino e {x_test.shape[0]} para teste.")
print(f"Cada imagem tem o tamanho de {x_train.shape[1]}x{x_train.shape[2]} pixels.")
```
Passo 4: Pré-processamento e Visualização dos Dados
Normalizamos os valores dos pixels das imagens (de 0-255 para 0-1) para melhorar a performance do treinamento.
```
# Normalizando os valores dos pixels dividindo por 255
x_train = x_train / 255.0
x_test = x_test / 255.0
```
```
# esse código você vai usar para dar uma olhada em uma imagem de exemplo para ver como ficou
plt.imshow(x_train[0], cmap='gray')
plt.title(f"Esta imagem representa o número: {y_train[0]}")
plt.show()
```
Passo 5: Construir a Arquitetura da Rede Neural
Aqui, definimos a estrutura do nosso modelo, camada por camada.
```
model = keras.Sequential([
    # 1ª Camada (Entrada): Achata a imagem de 28x28 pixels em uma única linha de 784 pixels.
    keras.layers.Flatten(input_shape=(28, 28)),

    # 2ª Camada (Oculta): Uma camada "densa" com 128 neurônios para aprender os padrões.
    # A função de ativação 'relu' é uma escolha padrão e muito eficiente.
    keras.layers.Dense(128, activation='relu'),

    # 3ª Camada (Regularização): "Desliga" aleatoriamente 20% dos neurônios durante o treino
    # para evitar que a rede apenas "decore" as imagens (overfitting).
    keras.layers.Dropout(0.2),

    # 4ª Camada (Saída): A camada final. Tem 10 neurônios, um para cada dígito (0 a 9).
    # A ativação 'softmax' transforma a saída em um conjunto de 10 probabilidades que somam 1.
    keras.layers.Dense(10, activation='softmax')
])

# Mostra um resumo da arquitetura que criamos
model.summary()
```
Passo 6: Compilar o Modelo
Configuramos os parâmetros de aprendizado do modelo.
 * Otimizador (optimizer): adam é o algoritmo que ajusta os pesos da rede para minimizar o erro.
 * Função de Perda (loss): sparse_categorical_crossentropy calcula o quão errada está a previsão do modelo.
 * Métricas (metrics): accuracy (precisão) é o que monitoramos para avaliar a performance.
```
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
```
Passo 7: Treinar o Modelo
Alimentamos o modelo com os dados de treino para que ele aprenda. epochs=5 significa que o modelo verá todo o conjunto de dados 5 vezes.
```
print("Iniciando o treinamento...")
history = model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))
print("Treinamento finalizado!")
```
## Resultados
Passo 8: Avaliar a Acurácia
Após o treinamento, usamos o conjunto de teste (imagens que o modelo nunca viu) para verificar sua real capacidade de generalização.
```
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)

print(f'\nAcurácia final no conjunto de teste: {test_acc * 100:.2f}%')

A acurácia final obtida no conjunto de teste foi de aproximadamente 97-98%.
```
Passo 9: Fazer uma Previsão
Vamos testar o modelo com uma imagem aleatória do conjunto de teste para ver um exemplo prático de sua previsão.
```
# Fazendo previsões para todo o conjunto de teste
predictions = model.predict(x_test)

# Escolha um número de 0 a 9999 para testar uma imagem
index_da_imagem = 150 # Você pode mudar esse número!

# Mostra a imagem que estamos testando
plt.imshow(x_test[index_da_imagem], cmap='gray')
plt.show()

# A saída do modelo é um array de 10 probabilidades. O 'argmax' pega o índice com a maior probabilidade.
previsao_do_modelo = np.argmax(predictions[index_da_imagem])
resposta_correta = y_test[index_da_imagem]

print(f"O modelo previu que este número é: {previsao_do_modelo}")
print(f"A resposta correta é: {resposta_correta}")

if previsao_do_modelo == resposta_correta:
    print("O modelo acertou!")
else:
    print("O modelo errou.")
```

Projeto desenvolvido como parte dos meus estudos em Engenharia de Software e Inteligência Artificial.


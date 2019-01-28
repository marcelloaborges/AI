### Observações:
- Para rodar o projeto, caso você tenha todas os requerimentos instalados, basta executar o arquivo <b>main.py</b>.
    - <b>python main.py</b>
- Se você receber qualquer mensagem sobre módulos não encontrados, tente instalar o módulo via <b>pip</b>, ou me
mande um email em <b>marcelloaborges@gmail.com</b> para detalhes.


### Requerimentos:
- python: 3.6
- numpy: 1.11.0
- torch: 0.4.1
- pandas
- scipy


## O problema:
Uma vendedora de seguros automobilísticos deseja saber qual a probabilidade um cliente aceitar uma oferta de seguro.
O objeto é criar um modelo estatístico/neural que consiga dizer qual a probabilidade do cliente aceitar a oferta de
seguro.


## O objetivo:
Predizer a probabilidade de <b>CarInsurance</b>.


## A solução:
Para avaliação de algumas considerações feitas por colegas, apliquei duas soluções a esse problema: uma regressão 
logística vs um modelo neural. O ponto era checar a eficiência de duas soluções em uma base pequena.
É uma classificação binária e a primeira versão continha apenas a implementação com o modelo neural.
Após alguns testes com o modelo neural, conclui que a base deveria ser maior para que a acurácia do modelo 
conseguisse melhor performance.
Questionado por colegas a respeito da solução apresentada, resolvi implementar uma segunda versão utilizando 
uma regressão logística sugerida por eles e realmente os resultados foram excepcionais. 
Com apenas 4000 registros na base, o modelo logístico consegue 100% de acurácia.
Fica como lição que apesar de parecerem pouco robustas, soluções mais simples podem ter performance muito superior
a modelo complexos. Basta que o contexto do problema seja favorável a elas.
Obrigado pelas considerações.


## Results:
Regressão Logística:
- Accuracy: 100%

Modelo Neural:
- Loss: ~0.001% (binary cross-entropy)
- Accuracy: ~52% (arredondamento de uma sigmoid comparada a coluna <b>CarInsurance</b>)


### Hyperparametros:
- O arquivo com a configuração de hyperparametros é o <b>main.py</b>.
- Configuração:
  - FEED_SIZE = 44
  - FC1_UNITS = 128 
  - FC2_UNITS = 64 
  - OUTPUT_SIZE = 1
  - LR = 1e-4
  - BATCH_SIZE = 128
  - EPOCHS = 100


- A configuração do modelo neural está no arquivo <b>model.py</b>.
- A arquitetura do modelo neural é:
  - Hidden: (feed_size, fc1_units)   - ReLU    
  - Dropout: fixed with 0.2
  - Hidden: (fc1_units, fc2_units)   - ReLU    
  - Output: (fc2, output_size)       - Sigmoid
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
Optei por utilizar um modelo neural na solução deste problema. É uma tarefa de classificação binária, ou seja,
requer uma resposta Sim/Não. A estrutura que foi contruída é relativamente "clássica" para tarefas desse tipo.
Como função de ativação decisória foi aplicada uma sigmoid que estima o grau de certeza com relação a possibilidade
do cliente aceitar a oferta de seguro, e para testar a acurácia, foi utilizada uma matriz de confusão. Como a saída do
modelo neural é somente uma probabilidade e não acesso ao que seria um grau de certeza considerado bom (em cenários 
médicos um grau de certeza bom considera apenas valor superiores a 95% por exemplo), utilizei um arredondamento simples 
do valor da saída para converter a probabilidade em 0 ou 1 e testar a acurácia na matriz de confusão, o que quer dizer,
que qualquer valor acima de 50% considera que o cliente aceitaria o seguro enquanto que valores abaixo de 50% dizem o
contrário.
O modelo está convergindo muito bem. A taxa de erro chega a menos de 0.0001, porém a acurácia está muito baixa, 
aproximadamente 52%. Acredito que o motivo da acurácia está baixa é a falta de cenários sufientes para que o modelo
consiga generalizar o problema eficientemente. A base de dados conta com apenas 4000 cenários para predizer uma 
probabilidade que considera 15 variavéis diferentes em sua decisão. Em casos assim, quando os dados de treino são 
poucos, é comum que problemas com essa magnitude apresentem baixa acurácia devido ao fato de que o modelo não vivencia
cenários sufientes para conseguir decidir qual a ação correta em situações muito próximas e, é isto o que eu acredito
estar acontecendo nesse caso.


## Results:
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
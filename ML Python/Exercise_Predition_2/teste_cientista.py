# coding: utf-8
import csv
import sys
from datetime import datetime

# Lendo os dados como uma lista
print("Lendo o documento...")
with open("dados_desafio.csv", "r") as file_read:
    reader = csv.reader(file_read)
    data_list = list(reader)
print("Tudo pronto!")

# Aqui estão as primeiras linhas dos dados:
for i in range(5):
    print(data_list[i])

input("Aperte Enter para continuar...")

# A primeira linha contém o cabeçalho. Pode ser retirada:
data_list = data_list[1:]

# ATIVIDADE 1
# ToDo
# Crie uma função para adicionar as colunas de uma lista em outra lista,
# na mesma ordem. A ideia é poder separar cada feature em uma lista própria.

def column_to_list(data, index):
    column_list = []
    # Dica: Você pode usar um for para iterar sobre as amostras,
    # pegar a feature pelo seu índice, e juntar em uma lista    
    for row in data:
        column_list.append(row[0].split(';')[index])    

    return column_list

column_to_list(data_list, 5)

# ATIVIDADE 2
# ToDo
# Crie uma função para contar os diferentes tipos de estado civil (marital)
# dos clientes. Essa função deve retornar uma LISTA com o número de clientes
# marcados como married, divorced e single, nesta ordem
def count_marital(data_list):
    married = 0
    single = 0
    divorced = 0

    for data in data_list:        
        tmp = data[0].split(';')[4]
        if tmp == 'married':
            married += 1
        if tmp == 'single':
            single += 1
        if tmp == 'divorced':
            divorced += 1

    return [married, single, divorced]

print("\nATIVIDADE 2: Resultado de count_marital:")
print(count_marital(data_list))

# ------------ NÃO ALTERE AQUI ------------
assert type(count_marital(data_list)) is list, "ATIVIDADE 2: Tipo incorreto retornado. Deveria ser uma lista."
assert len(count_marital(data_list)) == 3, "ATIVIDADE 2: A lista não tem o tamanho esperado..."
assert count_marital(data_list)[0] == 2238 and count_marital(data_list)[1] == 1184 and count_marital(data_list)[2] == 473, "ATIVIDADE 2: Algo está errado!"
# -----------------------------------------------------

input("Aperte Enter para continuar...")

# ATIVIDADE 3
# ToDo
# Ache os valores de mínimo e máximo, a média, a mediana e a moda da
# duração da ligação. Você não possui esse valor no conjunto de dados,
# e terá de calculá-lo para cada linha. Você poderá usar qualquer função
# pronta para ler os horários e calcular as durações das chamadas, mas
# deverá escrever seus próprios códigos para extrair as estatísticas
# solicitadas.

min_tempo = 0.
max_tempo = 0.
media_tempo = 0.
mediana_tempo = 0.
moda_tempo = 0.

# variaveis temporarias para os calculos
min_tempo = sys.maxsize
soma_total_ligacao = 0
totais_ligacao = []
dicionario_frequencia = {}

# loop para calcalar o tempo total da ligação
for data in data_list:

    inicio_ligacao = datetime.strptime( data[0].split(';')[15], '%H:%M:%S' )
    fim_ligacao = datetime.strptime( data[0].split(';')[16], '%H:%M:%S' )
    total_ligacao = (fim_ligacao - inicio_ligacao).seconds

    # encontrando o valor mínimo
    if total_ligacao < min_tempo:
        min_tempo = total_ligacao

    # encontrando o valor máximo
    if total_ligacao > max_tempo:
        max_tempo = total_ligacao
    
    # soma cumulativa para cálculo da média
    soma_total_ligacao += total_ligacao

    # lista com todas as ligações para cálculo da mediana
    totais_ligacao.append(total_ligacao)

    # dicionario de frequência para cálculo da moda
    if total_ligacao in dicionario_frequencia:
        dicionario_frequencia[total_ligacao] += 1
    else:
        dicionario_frequencia[total_ligacao] = 1

# convertendo o valores de segundos para minutos
min_tempo = min_tempo / 60
max_tempo = max_tempo / 60
media_tempo = (soma_total_ligacao / 60) / len(data_list)

# cálculo da mediana
totais_ligacao = sorted(totais_ligacao)

index_mediana = int(len(totais_ligacao) / 2)
if len(totais_ligacao) % 2 == 1:
    mediana_tempo = totais_ligacao[ index_mediana ]
else:
    mediana_tempo = ( totais_ligacao[ index_mediana - 1 ] + totais_ligacao[ index_mediana ] ) / 2

mediana_tempo = mediana_tempo / 60

# cálculo da moda
moda_tempo = max( dicionario_frequencia, key=dicionario_frequencia.get ) / 60


# print("\nATIVIDADE 3: Imprimindo o mínimo, o máximo, a média, a mediana e a moda")
print("Min: ", min_tempo, "Max: ", max_tempo, "Média: ", media_tempo, "Mediana: ", mediana_tempo, "Moda: ", moda_tempo)


### OBS ###
# a validação a respeito do valor da moda está fazendo uma conversão do valor para inteiro que acredito estar errada
# o tempo de ligação com maior frequência, que no caso é a moda, é 112s (1,86 min) 
# como a conversão está diferente das outras validações (nos outros testes é feito um arredondamento), o teste falha
# pois a parte inteira de 1,86 é 1, e o teste espera 2
# acredito que o correto seria fazer um round neste cenário assim como é feito nos outros 
# não fiz a alteração do assert pois a orientação do teste é que essa parte do código não deve ser alterada
###########

# ------------ NÃO ALTERE AQUI ------------
assert round(min_tempo) == 0, "min_tempo com resultado errado!"
assert round(max_tempo) == 54, "max_tempo com resultado errado!"
assert round(media_tempo) == 6, "media_tempo com resultado errado!"
assert round(mediana_tempo) == 4, "mediana_tempo com resultado errado!"
assert int(moda_tempo) == 2, "moda_tempo com resultado errado!"
# -----------------------------------------------------

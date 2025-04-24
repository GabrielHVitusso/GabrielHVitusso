# Resolucao Lista 1 Python
# Gabriel Hesse Vitusso - 2022056242

#  1 - Como você usaria a biblioteca math para calcular a raiz quadrada de um número fornecido pelo usuário?

import math
print("Insira um número para se fazer a raiz:")
Valor = int(input())
Resultado = math.sqrt(Valor)
print("RESULTADO:")
print(Resultado)

# 2 - Como você usaria a biblioteca random para simular o lançamento de um dado de 6 lados?

import random
print("Faca o input de um caracter para lancar o dado:")
Input = input()
Resultado = random.randint(1, 6)
print("RESULTADO:")
print(Resultado)

# 3 - Escreva um programa que use um laço for para imprimir todos os números pares de 0 a 20.

print("Faca o input de um caracter para imprimir todos os numeros pares de 0 a 20:")
Input = input()
print("NUMEROS:")
for i in range(0, 21, 2):
    print(i)

# 4 - Escreva um programa que use um laço while para pedir números ao usuário até que ele digite um número negativo.

print("Faca o input de um caracter para pedir numeros ao usuario ate que ele digite um numero negativo:")
Input = input()
valor = 0
while valor >= 0:
    try:
        valor = int(input("Digite um numero (negativo para sair): "))
        if valor >= 0:
            print("Voce digitou:", valor)
    except ValueError:
        print("Por favor, insira apenas numeros.")

# 5 - Crie uma função que receba uma lista de números como argumento e retorne a média deles.

print("Faca o input de um caracter para criar uma funcao que receba uma lista de numeros como argumento e retorne a media deles:")
Input = input()
print("Escreva os numeros separados por espaco:")
Lista = list(map(float, input().split()))
def calcular_media(lista):
    soma = 0
    for i in lista:
        soma += i
    media = soma / len(lista)
    return media

Media = calcular_media(Lista)
print("RESULTADO:")
print(Media)

# 6 - Escreva um programa que verifique se uma string fornecida pelo usuário é um palíndromo (lê-se igual de trás para frente).

print("Faca o input de um caracter para verificar se uma string fornecida pelo usuario e um palindromo:")
Input = input()
print("Escreva a string:")
string = input()
if string == string[::-1]:
    print("PALINDROMO")
else:
    print("NAO EH PALINDROMO")
    
# 7 - Escreva um programa que receba uma lista de números e retorne uma nova lista contendo apenas os números ímpares.

print("Faca o input de um caracter para receber uma lista de numeros e retorne uma nova lista contendo apenas os numeros impares:") 
Input = input()
print("Escreva os numeros separados por espaco:")
Numeros = list(map(int, input().split()))
for i in Numeros:
    if i % 2 == 0:
        Numeros.remove(i)
print("RESULTADO:")
print(Numeros)

# 8 - Escreva um programa que ordene uma lista de strings em ordem alfabética.

print("Faca o input de um caracter para ordenar uma lista de strings em ordem alfabetica:")
Input = input()
print("Escreva as strings separadas por espaco:")
Strings = list(map(str, input().split()))
Strings.sort() # pode fazer isso?
print("RESULTADO:")
print(Strings)

# 9 - Crie uma função que receba uma lista de tuplas, onde cada tupla contém o nome e a idade de uma pessoa, e retorne a pessoa mais velha.
print("Faca o input de um caracter para criar uma funcao que receba uma lista de tuplas, onde cada tupla contem o nome e a idade de uma pessoa, e retorne a pessoa mais velha:")
Input = input()
print("Escreva os nomes e idades separados por espaco:")
Tuplas = []
while True:
    try:
        Nome = input("NOME ou sair:")
        if Nome.lower() == 'sair':
            break
        Idade = int(input("IDADE:"))
        Tuplas.append((Nome, Idade))
    except ValueError:
         print("Por favor, insira apenas numeros.")
def Tupla_mais_velha(tuplas):
    mais_velha = tuplas[0]
    for tupla in tuplas:
        if tupla[1] > mais_velha[1]:
            mais_velha = tupla
    return mais_velha
Mais_velha = Tupla_mais_velha(Tuplas)
print("RESULTADO:")
print("NOME:", Mais_velha[0])
print("IDADE:", Mais_velha[1])

# 10 - Escreva um programa que conte quantas vezes um determinado elemento aparece em uma tupla.
print("Faca o input de um caracter para contar quantas vezes um determinado elemento aparece em uma tupla:")
Input = input()
print("Escreva os numeros separados por espaco:")
Tupla = tuple(map(int, input().split()))
print("Digite o elemento que deseja contar:")
Elemento = int(input())
def contar_elemento(tupla, elemento):
    contagem = 0
    for i in tupla:
        if i == elemento:
            contagem += 1
    return contagem
Contagem = contar_elemento(Tupla, Elemento)
print("RESULTADO:")
print(Elemento)
print(Contagem)

# 11 - Escreva um programa que crie um dicionário para armazenar o nome e o telefone de amigos. Peça ao usuário para inserir os dados e, em seguida, imprima o dicionário.
print("Faca o input de um caracter para criar um dicionario para armazenar o nome e o telefone de amigos:")
Input = input()
dicionario = {}
while True:
    nome = input("NOME ou sair")
    if nome.lower() == 'sair':
        break
    telefone = input("NUMERO")
    dicionario[nome] = telefone
print("RESULTADO:")
print(dicionario)

# 12 - Escreva um programa que conte quantas vezes cada palavra aparece em um texto fornecido pelo usuário e armazene os resultados em um dicionário.
print("Faca o input de um caracter para contar quantas vezes cada palavra aparece em um texto fornecido pelo usuario:")
Input = input()
print("TEXTO")
texto = input()
palavras = texto.split()
dicionario = {}
for palavra in palavras:
    if palavra in dicionario:
        dicionario[palavra] += 1
    else:
        dicionario[palavra] = 1
print("RESULTADO:")
print(dicionario)


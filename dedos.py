import cv2 as cv
import numpy as np
import glob


resultados = []


def trata_img(file):
	"""
	Trata a imagem, convertendo para cinza,
	depois converte para binário.


	Parametros:
	file -- caminho da imagem
    """
	img = cv.imread(file)
	hsvim = cv.cvtColor(img, cv.COLOR_BGR2HSV)
	lower = np.array([0, 48, 80], dtype = "uint8")
	upper = np.array([20, 255, 255], dtype = "uint8")
	skinRegionHSV = cv.inRange(hsvim, lower, upper)
	blurred = cv.blur(skinRegionHSV, (2,2))
	ret,thresh = cv.threshold(blurred,0,255,cv.THRESH_BINARY)
	return thresh


def conta_pixels(img):
	"""
	Conta os pixels brancos e pretos
	de uma imagem binaria.


	Parametros:
	img -- imagem binaria
    """
	branco = 0
	preto = 0
	height, width= img.shape
	# Percorre uma linha da imagem
	for j in range(width):
		if img[560, j] == 255:
			branco += 1
		else:
			preto += 1
	return branco, preto


def verifica_dedos(branco, preto):
	"""
	Algoritmo que informa a quantidade
	de dedos com base no pixels informados.


	Parametros:
	branco -- pixels brancos 
	preto -- pixels pretos
    """
	if branco < preto and branco > 130:
		resultados.append(3)
		return 3
	elif branco < 110:
		resultados.append(1)
		return 1
	# Se nao atender as condicoes precisa reprocessar
	else:
		return -1


def processa_img(path):
	"""
    Percorre um diretorio com imagens de mesma extensao,
	e faz o processamento de cada uma das imagens encontradas.


	Parametros:
	path -- diretorio de imagens
    """
	for file in glob.glob(path):
		img = trata_img(file)
		branco, preto = conta_pixels(img)
		result = verifica_dedos(branco, preto)
		if result == -1:
			reprocessar(img)


def reprocessar(img):
	"""
	Reprocessa imagens que estavam invertidas.


	Parametros:
	img -- imagem binaria
	"""
	flipimg = cv.flip(img, -1)
	branco, preto = conta_pixels(flipimg)
	result = verifica_dedos(branco, preto)

	# Nesse caso nao faz sentido reprocessar novamente
	if result == -1:
		resultados.append(0)


def mostra_resultados():
	"""Mostra na tela os resultados obtidos."""
	print(f"3 DEDOS: {resultados.count(3)}")
	print(f"1 DEDO: {resultados.count(1)}")
	print(f"INDEFINIDO: {resultados.count(0)}")
	print(f"TAMANHO DA LISTA: {len(resultados)}")

	# Limpa os resultados
	del resultados[:]


print("-------------------- 3 DEDOS ---------------------")
processa_img("3_dedos/*.png")
mostra_resultados()

print("-------------------- 1 DEDO ---------------------")
processa_img("1_dedo/*.png")
mostra_resultados()
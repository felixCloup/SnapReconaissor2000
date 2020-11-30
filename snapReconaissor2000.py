import cv2
import sys
import os
import numpy as np
from numpy import linalg as LA
# from sklearn.decomposition import PCA
from math import pi, cos, sqrt, log, floor
import datetime

larg = 200
lon = 200

# ======================== Fonctions pour la DCT ======================== #

def DCT_8(pixel):
	dct = np.zeros((8, 8, 3), np.uint8)
	for i in range(8):
		for j in range(8):
			# Calcul des doubles sommes
			somme = 0
			for x in range(8):
				# C'est censé être plus rapide que la boucle pour mais ça change pas grand chose :/
				for y in range(8):
					somme += pixel[x, y, 0] * cos(pi * (2 * x + 1) * i / 16) * cos(pi * (2 * y + 1) * j / 16)
			# Multiplication par le bon coef
			if i == 0 & j == 0:
				somme *= 0.125
			elif i == 0 | j == 0:
				somme *= sqrt(0.5) * 0.25
			else:
				somme *= 0.25
			dct[i, j] = (somme, somme, somme)
	return dct

def InvDCT_8(dct):
	pixel = np.zeros((8, 8, 3), np.uint8)
	for x in range(8):
		for y in range(8):
			# Calcul des doubles sommes
			somme = 0
			for i in range(8):
				for j in range(8):
					somme += dct[i, j, 0] * cos(pi * (2 * x + 1) * i / 16) * cos(pi * (2 * y + 1) * j / 16)
			# Multiplication par le bon coef
			if x == 0 & y == 0:
				somme *= 0.125
			elif x == 0 | y == 0:
				somme *= sqrt(0.5) * 0.25
			else:
				somme *= 0.25
			pixel[i, j] = (somme, somme, somme)
	return pixel

def DTDCT(img):	# dotheDCT
	dct = np.zeros((larg, lon, 3), np.uint8)
	for i in range(int(larg/8)):
		for j in range(int(lon/8)):
			dct[8 * i:8 * (i + 1), 8 * j:8 * (j + 1)] = DCT_8(img[8 * i:8 * (i + 1), 8 * j:8 * (j + 1)])
	return dct


def DTDCTFile(path):
	img = cv2.imread(path)
	newPathList = path.split('/')
	n = len(newPathList) - 1
	newPathList[n] = "DCT/" + newPathList[n]
	newPathList[n] = newPathList[n][:len(newPathList[n]) - 5] + newPathList[n][len(newPathList[n]) - 4:] # On enlève le chiffre a la fin du nom du fichier, juste à faire que pour la base
	print (newPathList)
	newPath = "."
	for e in newPathList:
		newPath += "/" + e
	cv2.imwrite(newPath,DTDCT(img))

# ======================== Fonctions pour la base ======================== #

def imgToVect(img):
	vect = np.zeros((larg * lon))
	for i in range(larg):
		for j in range(lon):
			vect[i * larg + j] = img[i][j][1]
	return vect

def vectToImg(vect):
	img = np.zeros((larg, lon, 3))
	for i in range(larg):
		for j in range(lon):
			img[i, j, :] = vect[i * larg + j]
	return img

def addFileToBase(path):
	os.popen('')

def loadBase(pathFilesBash, pathFiles):
	base = []
	noms = []
	files = os.popen('ls ' + pathFilesBash + ' | grep "[^DCT]" | grep "[^Base]"').read()
	files = files.split()
	for file in files:
		img = cv2.imread(pathFiles + file)
		base.append(imgToVect(img))
		noms.append(file.split('.')[0])
	return (base, noms)


# ======================== Fonctions pour l'ACP ======================== #

def moyStd(M):	# std = standard deviantion = écart type
	n = len(M)
	moy = 0
	moyCarre = 0
	for j in range(n):
		moy += M[j]
		moyCarre += M[j] ** 2
	moy = moy / n
	moyCarre = moyCarre / n
	return ([sqrt(moyCarre - (moy ** 2)), moy])


def matCentreeReduite(M):
	k = len(M)
	n = len(M[0])
	reduc = np.zeros((k, n))
	for i in range(k):
		[std, moy] = moyStd(M[i])
		for j in range(n):
			reduc[i][j] = (M[i][j] - moy) / std
	return (reduc)

def vectCentreReduit(vect):
	n = len(vect)
	reduc = np.zeros(n)
	[std, moy] = moyStd(vect)
	for i in range(n):
		reduc[i] = (vect[i] - moy) / std
	return (reduc)

def var(vect):
	n = len(vect)
	moy = 0
	moyCarre = 0
	for valeur in vect:
		moy += valeur
		moyCarre += valeur ** 2
	moy = moy / n
	moyCarre = moyCarre / n
	return (np.sqrt(moyCarre - (moy ** 2)))

def matDeTravail(A):
	return (np.dot(A.transpose(), A))	# En fait il faut faire A' * A mais évitons ici de transposer 2 fois

def vPropreSorted(A):
	[valPropre, vectPropre] = LA.eig(A)
	valPropre_sorted = np.sort(valPropre)
	vectPropre_sorted = vectPropre[:, valPropre.argsort()]
	return (valPropre, vectPropre)

def valPropreNormalise(valPropre):
	return(valPropre/sum(valPropre))

def choix_valP_vectP(valP, vectP, seuil):
	somme = 0
	k = -1
	while somme < seuil:
		k = k + 1
		somme += valP[k]
	return(valP[:k][:k], vectP[:k][:k])

def resize_vectProp(A, vect_Prop):
	return(np.dot(vect_Prop, A.transpose()))	# A = A.transpose()

def projection_single(vect, vectPropres):
	return([np.dot(vect, vectPropre) for vectPropre in vectPropres])

def projection(A, vectPropres):
	return([projection_single(A_i, vectPropres) for A_i in A.transpose()])

def comparaison_moindre_carre(projAComparer, projImages):
	A = [floor(sum([abs(projAComparer[i] - projImage[i])  for i in range(len(projAComparer))])) for projImage in projImages]
	return(A)

def comparons(mat_d_images, aDeviner):
	A = matCentreeReduite(mat_d_images)
	A = A.transpose()
	ATravail = matDeTravail(A)
	aDeviner = vectCentreReduit(aDeviner)
	[valPropre, vectPropre] = vPropreSorted(ATravail)
	valPropre = valPropreNormalise(valPropre)
	vectPropre = resize_vectProp(A, vectPropre)
	[valPropre, vectPropre] = choix_valP_vectP(valPropre, vectPropre, 0.95)
	projImages = projection(A, vectPropre)
	projAComparer = projection_single(aDeviner, vectPropre)
	comparaison = comparaison_moindre_carre(projAComparer, projImages)
	return (comparaison)

def devinons(path, base, noms):
	ignorant = cv2.imread(path)
	ignorant = imgToVect(ignorant)
	comparaison = comparons(base, ignorant)
	return(noms[np.argmin(comparaison)])	# Dans cette fonction qu'on peut essayer d'affiner un peut plus la reconnaissance, faute de méthode suffisament puissante, nous avons décidé de simplement retourner le plus plus faible.

def afficherHelp():
	print("""python3 snapReconaissor2000.py [-a image_a_ajouter] -b chemin_de_la_base_d_images images_a_identifier""")


# ===================================================== Main ===================================================== #

if "-b" in sys.argv[:-1]:
	b = sys.argv.index("-b")
	pathFiles = sys.argv[b + 1]
else:
	afficherHelp()
	exit()

pathFilesBash = pathFiles.replace(" ", "\ ")

if "-a" in sys.argv[:-1]:
	a = sys.argv.index("-a")
	nom = sys.argv[a + 1]
	nom = nom.replace(" ", "\ ")
	# nom = nom.split("/")[-1]
	print("cp " + nom + " " + pathFiles)
	os.popen("cp " + nom + " " + pathFilesBash)
	print(sys.argv[a + 1] + " ajouté à la base d'image")


[base, noms] = loadBase(pathFilesBash, pathFiles)

for path in sys.argv[b+2:]:
	print()
	print("On pense que la personne sur le fichier " + path + " est : " + devinons(path, base, noms))

import matplotlib.pyplot as plt
import numpy as np

""" Pour l'image en noir et blanc, remplacer les variables A_sain et X_sain par:

A_sain = plt.imread("./Images/cheval-abime2-nb.png")
X_sain = A_nb

"""

# Lecture de l'image à réparer en couleur
A_sain = plt.imread("./Images/cheval-abime2.png")
# Lecture de l'image mask
A_mask = plt.imread("./Images/cheval-abime2-mask.png")

# Transformation en matrices de nos images
X_sain = np.mean(A_sain,-1)
X_mask = A_mask

rouge = A_sain[:,:,0]
vert = A_sain[:,:,1]
bleu = A_sain[:,:,2]

""""
Affichage de l'image à réparer pour vérifier notre code:

img_sain = plt.imshow(X_sain,'gray')
plt.axis('off')
plt.show()

"""

def repair_image(A_perfect, A_flaw, k=39, epsilon=1e-6, max_iterations=9):
    
    A_repared = np.copy(A_perfect)
    itr = 0
    while itr < max_iterations:
        # On calcule l'approximation de A par la méthose SVD de rang k
        U, S, Vt = np.linalg.svd(A_repared, full_matrices=False)
        S[k:] = 0
        A_k = U @ np.diag(S) @ Vt

        # On repère la zone à "réparer" grace à l'image mask
        A_repared[A_flaw != 0] = A_k[A_flaw != 0]
        # On calcule la condition selon laquelle on modifie notre matrice d'origine ou non
        error = np.linalg.norm(A_perfect - A_repared, 'fro')


        # Condition d'arrêt
        if error < epsilon:
            break
        itr += 1

        if itr == 10:
            break

    #Calcul de l'erreur relative
    erreur_relative = error/np.linalg.norm(A_perfect,'fro')

    return A_repared,itr

def repair_image_color(r,v,b, A_flaw, k=39, epsilon=1e-6, max_iterations=9):
    
    r_repared = np.copy(r)
    v_repared = np.copy(v)
    b_repared = np.copy(b)

    A_perfect = r
    itr = 0
    while itr < max_iterations:
        # On calcule l'approximation de A par la méthose SVD de rang k
        Ur, Sr, Vtr = np.linalg.svd(r_repared, full_matrices=False)
        Uv, Sv, Vtv = np.linalg.svd(v_repared, full_matrices=False)
        Ub, Sb, Vtb = np.linalg.svd(b_repared, full_matrices=False)
        Sr[k:] = 0
        Sv[k:] = 0
        Sb[k:] = 0
        A_kr = Ur @ np.diag(Sr) @ Vtr
        A_kv = Uv @ np.diag(Sv) @ Vtv
        A_kb = Ub @ np.diag(Sb) @ Vtb

        # On repère la zone à "réparer" grace à l'image mask
        r_repared[A_flaw != 0] = A_kr[A_flaw != 0]
        v_repared[A_flaw != 0] = A_kv[A_flaw != 0]
        b_repared[A_flaw != 0] = A_kb[A_flaw != 0]
        
        total_repared = r_repared 

        error = np.linalg.norm(A_perfect - total_repared, 'fro')

        # Condition d'arrêt
        if error < epsilon:
            break
        itr += 1

        if itr == 10:
            break

        total_repared = np.stack((r_repared,v_repared,b_repared),axis=2)

    return total_repared,itr

#Affichage de l'image "réparée" en couleur
img_color,itr = repair_image_color(rouge,vert,bleu,X_mask)
plt.title(f"Image obtenue après {itr} itérations:")
plt.imshow(img_color)
plt.axis("off")
plt.show()

# Afficher l'image "réparée" en noir et blanc
img,itr = repair_image(X_sain, X_mask)
plt.title(f"Image obtenue après {itr} itérations:")
plt.imshow(img,'gray')
plt.axis("off")
plt.show()

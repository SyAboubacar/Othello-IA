import numpy as np
import tkinter as tk

############### VARIABLES GLOBALES ###############

plateau = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, -1, 1, 0, 0, 0],
                    [0, 0, 0, 1, -1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0]])

valeur_importance = np.array([[500, -150, 30, 10, 10, 30, -150, 500],
                              [-150, -250, 0, 0, 0, 0, -250, -150],
                              [30, 0, 1, 2, 2, 1, 0, 30],
                              [10, 0, 2, 16, 16, 2, 0, 10],
                              [10, 0, 2, 16, 16, 2, 0, 10],
                              [30, 0, 1, 2, 2, 1, 0, 30],
                              [-150, -250, 0, 0, 0, 0, -250, -150],
                              [500, -150, 30, 10, 10, 30, -150, 500]])

# Les dimensions en pixels du canvas du jeu
CANVAS_SIZE = 600
nb_cases = 8
taille_case = CANVAS_SIZE / nb_cases

# Couleur d'une case en jeu en fonction de si elle appartient au joueur X 
#                    0         1         -1
liste_couleur = ["#a00000", "#000000", "#FFFFFF"]
joueur = 1

root = tk.Tk()
root.title("Othello")
# Canvas du jeu lui-même
canvas = tk.Canvas(root, width=CANVAS_SIZE, height=CANVAS_SIZE, bg="#00a000")
# Création d'un canvas d'information pour afficher le tour du joueur
info_canvas = tk.Canvas(root, width=CANVAS_SIZE, height=50, bg="white")

################### FONCTIONS ####################

###### GRAPHIQUE ######

def draw_grid():
    # Crée la grille pour rendre tout ça plus lisible
    for i in range(nb_cases + 1):  # 9 lignes pour faire 8 cases
        pos = i * taille_case + 2  # Décalé léger psq c'est plus centré comme ça
        # Lignes verticales
        canvas.create_line(pos, 2, pos, CANVAS_SIZE + 2, fill="#004000", width=2)
        # Lignes horizontales
        canvas.create_line(2, pos, CANVAS_SIZE + 2, pos, fill="#004000", width=2)

def affichage_couleur_quadrillage():
    # Ce programme dessine les jetons sur le canvas en fonction de quel joueur occupe quelle case
    canvas.delete("jetons")
    for y in range(nb_cases):
        for x in range(nb_cases):
            if plateau[x][y] != 0:  # Aucun jeton n'est posé si il n'y a pas de joueurs
                canvas.create_oval(y * taille_case + int(taille_case / 20 + 2),
                                    x * taille_case + int(taille_case / 20 + 2),
                                    (y + 1) * taille_case - int(taille_case / 20 - 1),
                                    (x + 1) * taille_case - int(taille_case / 20 - 1),
                                    fill=liste_couleur[plateau[x][y]], 
                                    tag="jetons")
            elif check_valid_move(x, y, joueur):  # Si il n'y a pas de joueur mais que le coup est valide:
                canvas.create_oval(y * taille_case + int(taille_case / 3 + 2),
                                    x * taille_case + int(taille_case / 3 + 2),
                                    (y + 1) * taille_case - int(taille_case / 3 - 1),
                                    (x + 1) * taille_case - int(taille_case / 3 - 1),
                                    fill=liste_couleur[0], 
                                    tag="jetons")

def jouer_ordinateur():
    global joueur
    if joueur == -1:  # Vérifie que c'est bien le tour de l'ordinateur
        coups_valides = [(x, y) for x in range(nb_cases) for y in range(nb_cases) if check_valid_move(x, y, joueur)]

        meilleur_coup = None
        meilleur_score = float('-inf')

        for x, y in coups_valides:
            nouveau_plateau = np.copy(plateau)
            flips = check_valid_move(x, y, joueur)
            nouveau_plateau[x][y] = joueur
            flip_tokens_sim(nouveau_plateau, flips, joueur)

            score = minimax_alpha_beta(nouveau_plateau, profondeur=2, alpha=float('-inf'), beta=float('inf'), maximising_player=True)   # Profondeur 5

            if score > meilleur_score:
                meilleur_score = score
                meilleur_coup = (x, y)

        if meilleur_coup:
            x, y = meilleur_coup
            flips = check_valid_move(x, y, joueur)
            plateau[x][y] = joueur
            flip_tokens(flips, joueur)
            changer_le_joueur()
            affichage_couleur_quadrillage()
            root.after(500, jouer_ordinateur)
def nb_jetons(plateau, joueur):
    return np.sum(plateau == joueur)

def mobilité(plateau, joueur):
    return sum(bool(check_valid_move_local(plateau, x, y, joueur)) for x in range(nb_cases) for y in range(nb_cases))

def donne_val_plateau(plateau):
    cases_vides = np.sum(plateau == 0)
    phase_jeu = 1 - (cases_vides / 64)  # 0=début, 1=fin

    # Coefficients dynamiques
    poids_position = 1.0  # Toujours important
    poids_jetons = 5.0 + 10.0 * phase_jeu  # Devient crucial en fin de partie
    poids_mobilite = 10.0 - 8.0 * phase_jeu  # Plus important en début de partie

    # Calcul des composantes
    score_position = sum(plateau[x][y] * valeur_importance[x][y] 
                      for x in range(nb_cases) for y in range(nb_cases))
    diff_jetons = np.sum(plateau == -1) - np.sum(plateau == 1)
    mobilite_ia = mobilité(plateau, -1)
    mobilite_joueur = mobilité(plateau, 1)
    score_mobilite = mobilite_ia - mobilite_joueur

    return (poids_position * score_position 
            + poids_jetons * diff_jetons 
            + poids_mobilite * score_mobilite)


def changer_le_joueur():
    global joueur
    
    if not peut_jouer(-joueur):     # Si l'adversaire ne peut pas jouer
        if peut_jouer(joueur):      # Si l'actuel joueur peut toujours jouer 
            return  # Reste sur le même joueur
        else:       # Si aucun joueur ne peut jouer, fin de partie
            afficher_fin_partie()
            return

    joueur = -joueur
    if joueur == -1:
        root.after(500, jouer_ordinateur)  # L'ordinateur joue après 500ms

    # Met à jour l'affichage du tour
    info_canvas.delete("all")
    if joueur == 1:
        info_canvas.create_text(CANVAS_SIZE / 2 + 84, 25, text="Au tour du joueur ", fill="black", font=("Helvetica", 16))
    else:
        x, y = CANVAS_SIZE / 2 + 84, 25
        info_canvas.create_text(x, y, text="Au tour de l'IA", fill="black", font=("Helvetica", 16))
        root.after(5000, jouer_ordinateur)

def afficher_fin_partie():
    """ Affiche un message de fin et désactive les clics """
    canvas.unbind('<Button-1>')  # Désactive les clics sur le plateau
    score_noir = np.sum(plateau == 1)
    score_blanc = np.sum(plateau == -1)
    message = "Égalité !" if score_blanc == score_noir else \
              f"Le joueur 2 (Blanc) gagne avec {score_blanc} points contre {score_noir} !" if score_blanc > score_noir else \
              f"Le joueur 1 (Noir) gagne avec {score_noir} points contre {score_blanc} !"

    info_canvas.delete("all")
    info_canvas.create_text(CANVAS_SIZE / 2, 25, text=f"Fin de partie ! {message}", fill="red", font=("Helvetica", 15))

###### LOGIQUE ######

def click(event):
    # Ce programme détecte quelle case on a cliqué, et essaye de poser un pion de la couleur du joueur sur celle-ci
    global joueur, nb_cases, plateau, taille_case

    ligne_click = min(int(event.y // taille_case), nb_cases - 1)
    colone_click = min(int(event.x // taille_case), nb_cases - 1)
    # Donne la liste des retournements à faire si il y en a de possibles
    flips = check_valid_move(ligne_click, colone_click, joueur)
    # Si au moins 1 retournement est possible, coup valide
    if flips:
        # Pose le jeton à l'endroit possible
        plateau[ligne_click][colone_click] = joueur
        flip_tokens(flips, joueur)
        changer_le_joueur()  # Mise à jour de l'affichage du tour
        affichage_couleur_quadrillage()
        print(donne_val_plateau(plateau))

def check_valid_move(ligne, col, joueur):
    # La case doit être vide pour être jouée
    if plateau[ligne][col] != 0:
        return []
    directions = [(-1, -1), (-1, 0), (-1, 1),
                  (0, -1),           (0, 1),
                  (1, -1),  (1, 0),  (1, 1)]
    flips_total = []
    # On donne les 8 directions possibles et les déplacements gauche/droite, haut/bas
    for d_ligne, d_col in directions:
        flips_total.extend(tokens_to_flip_in_direction(ligne, col, d_ligne, d_col, joueur))
    return flips_total

def peut_jouer(joueur):
    """ Vérifie si le joueur a au moins un coup valide """
    for x in range(nb_cases):
        for y in range(nb_cases):
            if check_valid_move(x, y, joueur):  
                return True
    return False

def tokens_to_flip_in_direction(ligne, col, d_ligne, d_col, joueur):
    """Pour une direction, on l'explore pour des coups valides"""
    flips = []
    i = ligne + d_ligne
    j = col + d_col
    while 0 <= i < nb_cases and 0 <= j < nb_cases:  # Pas au dela du bord du plateau 
        if plateau[i][j] == -joueur:
            flips.append((i, j))
        elif plateau[i][j] == joueur:
            return flips if flips else []
        else:  # case vide
            break
        i += d_ligne
        j += d_col
    return []

def flip_tokens(positions, joueur):
    """Changer tout les jetons pour chaque retournement à faire"""
    for i, j in positions:
        plateau[i][j] = joueur

def flip_tokens_sim(plateau, positions, joueur):
    """Changer tout les jetons pour chaque retournement à faire"""
    for i, j in positions:
        plateau[i][j] = joueur

def est_fin_partie():
    return True if not peut_jouer(1) and not peut_jouer(-1) else False

def check_valid_move_local(plateau, ligne, col, joueur):
    if plateau[ligne][col] != 0:
        return []
    directions = [(-1, -1), (-1, 0), (-1, 1),
                  (0, -1),           (0, 1),
                  (1, -1),  (1, 0),  (1, 1)]
    flips_total = []
    for d_ligne, d_col in directions:
        flips_total.extend(tokens_to_flip_in_direction_local(plateau, ligne, col, d_ligne, d_col, joueur))
    return flips_total

def tokens_to_flip_in_direction_local(plateau, ligne, col, d_ligne, d_col, joueur):
    flips = []
    i = ligne + d_ligne
    j = col + d_col
    while 0 <= i < nb_cases and 0 <= j < nb_cases:
        if plateau[i][j] == -joueur:
            flips.append((i, j))
        elif plateau[i][j] == joueur:
            return flips if flips else []
        else:
            break
        i += d_ligne
        j += d_col
    return []


def minimax_alpha_beta(plateau, profondeur, alpha, beta, maximising_player):
    if profondeur == 0 or est_fin_partie():
        return donne_val_plateau(plateau)

    joueur_actuel = -1 if maximising_player else 1

    coups_valides = [(x, y) for x in range(nb_cases) for y in range(nb_cases)
                     if check_valid_move_local(plateau, x, y, joueur_actuel)]

    if not coups_valides:
        return minimax_alpha_beta(plateau, profondeur - 1, alpha, beta, not maximising_player)

    if maximising_player:
        max_eval = float('-inf')
        for x, y in coups_valides:
            plateau_simule = np.copy(plateau)
            flips = check_valid_move_local(plateau_simule, x, y, joueur_actuel)
            plateau_simule[x][y] = joueur_actuel
            flip_tokens_sim(plateau_simule, flips, joueur_actuel)

            eval = minimax_alpha_beta(plateau_simule, profondeur - 1, alpha, beta, False)
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break  # Élagage beta
        return max_eval
    else:
        min_eval = float('inf')
        for x, y in coups_valides:
            plateau_simule = np.copy(plateau)
            flips = check_valid_move_local(plateau_simule, x, y, joueur_actuel)
            plateau_simule[x][y] = joueur_actuel
            flip_tokens_sim(plateau_simule, flips, joueur_actuel)

            eval = minimax_alpha_beta(plateau_simule, profondeur - 1, alpha, beta, True)
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break  # Élagage alpha
        return min_eval


# Initialisation des graphiques
draw_grid()
changer_le_joueur()
affichage_couleur_quadrillage()

############## CREATION DE LA FENETRE #############

canvas.grid(row=1, column=0, columnspan=4, rowspan=4)
info_canvas.grid(row=0, column=0, columnspan=4)

# Créée le lien entre un clic gauche sur le canvas et le fonction "click"
canvas.bind('<Button-1>', click)
# Check constant des inputs du joueur
root.mainloop()

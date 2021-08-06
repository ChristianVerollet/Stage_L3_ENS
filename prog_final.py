#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Fri Jul  2 09:12:48 2021

@author: Christian Programme Final Stage : - Dynamique des états stationnaire associé au potentiel étudié
                                           - Ajout du pulse
                                           - Visualisation des résultats 
                                           - Si possible, ajout de la dissipation et de l'émission spontané                                         
"""

# =============================================================================
# Importation des librairies
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pyslise import Pyslise # Schrodinger stationnaire
from mpl_toolkits.mplot3d import Axes3D # animation Bloch
from scipy.integrate import odeint # Schrodinger
from odeintw import odeintw # Lindald
import qutip as qt# juste pour la sphère de Bloch
from scipy.optimize import leastsq # moindre carré

# Pour Schrodinger Stochastic
import torch
from torch import nn
import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
import torchsde

# =============================================================================
# Paramètres pour les graphiques
# =============================================================================

# plt.rcParams['axes.titlesize'] = 28 # change la taille du/ TITRES seulement
plt.rcParams['axes.labelsize'] = 20 # change la taille du label des axes seulement
# plt.rcParams['axes.linewidth'] = 1.5 # augmente l'épaisseur du contour de la figure
plt.rcParams['lines.linewidth'] = 2 # augmente l'épaisseur des lignes
# plt.rcParams['lines.markersize'] = 10 # augmente la taille des marqueurs de points
plt.rcParams['xtick.labelsize'] = 17 # augmente la taille des nombres gradués en x
plt.rcParams['ytick.labelsize'] = 17 # pareil mais en y

# Pour Latex 
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Helvetica"]})

#%%
# =============================================================================
# Définition des constantes, du potentiel et du pulse 
# =============================================================================

L = 2e-10 # Henry
C = 76e-15 # Farad
Z = np.sqrt(L/C) # Impédance du système
omega = 1/np.sqrt(L*C) # rad/s pulsation du système
print(f'Pulsation du système = {omega:e} rad/s')
Phi_0 = 2.07e-15 # Weber
k_B = 1.38e-23 # J.K^-1 Constante de Boltzman
hbar = 1.054e-34 # J.s Constante de Planck réduite
U_0 = ((Phi_0/(2*np.pi))**2)/(k_B*L) # U_0 normalisé en K
M = (k_B*(Phi_0**2)*C)/(hbar**2) # 1/2M = facteur devant le -d²/dx² dans Schrödinger
borne_gauche = 0 # Borne gauche du domaine spatial
borne_droite = 1 # Borne droite du domaine spatial
N_x = 1000 # Discrétisation du domaine spatial
N_max_valeurs_propres = 11 # Nombre de valeurs propres que PySlise va calculer
flux_normalise = np.linspace(borne_gauche,borne_droite,N_x) # Domaine sur lequel Schrödinger est résolu (sans dimension)
T_0 = 0 # s
T_f = 300 # s temps normalisé T = t*omega
N_T = 10000 # Discrétisation du domaine temporel
temps_normalise = np.linspace(T_0,T_f,N_T) # Domaine de temps normalisé sur lequel odeintw va résoudre la dynamique
R_s = 10 # Ohm Résistance apparaissant dans la formule du pulse
I_c = 1e-4 # Ampère Courant critique du pulse
omega_c = 2*np.pi*R_s*I_c/Phi_0
largeur_pulse = 300 # Largeur temporelle normalisée du pulse
decalage_pulse = 150 # Centre de la fonction porte du pulse

R_f =  100000  # Ohm , résistance de dissipation du système

def U(x,beta_L=1.325,x_e=0.5135):
    """ Potentiel normalisé du circuit TFQ sous forme analytique pour PySlise"""
    return U_0*(2*np.pi**2*(x-x_e)**2-beta_L*(np.cos(2*np.pi*x)))

def Norme(x):
    """ Norme euclidienne d'un nombre ou array complexe"""
    return np.sqrt(x.real**2+x.imag**2)

def Porte(x,width=largeur_pulse,shift=decalage_pulse):
    """ Distribution porte """
    if type(x) == np.ndarray:
        result = []
        for i in x:
            if np.abs(i-shift)>width/2:
                result.append(0)
            else:
                result.append(1)
        return np.array(result)
    if type(x) == list:
        result = []
        for i in x:
            if np.abs(i-shift)>width/2:
                result.append(0)
            else:
                result.append(1)
        return result
    if type(x) == float:
        if np.abs(x-shift)>width/2:
            res = 0
        else:
            res = 1
        return res
    if type(x) == int:
        if np.abs(x-shift)>width/2:
            res2 = 0
        else:
            res2 = 1
        return res2
    if type(x) == torch.Tensor:
        if np.abs(x-shift)>width/2:
            res2 = 0
        else:
            res2 = 1
        return res2
    
def Impulsion0(t,i_0=0,w=2*np.pi,R=R_s,I=I_c,phase_0=0):
    """ Signal de l'impulsion que recoit le système normalisé au temps T """
    return (R*I*(i_0**2-1))/(i_0+np.cos(np.sqrt(i_0**2-1)*w*t+phase_0))



#%%
# =============================================================================
# Résolution de la partie stationnaire du problème avec PySlise
# =============================================================================


probleme = Pyslise(lambda x: (2*M)*U(x),borne_gauche,borne_droite,tolerance=1e-8) # Initialisation du problème, le facteur 2*M vient du fait que
                                                                                  # PySlise veut une équation de la forme (-d²/dx²+V(x)-E)*Psi(x)
gauche = (0,1)   # Condition limite à gauche
droite = (0,1)  # Condition limite à droite
valeurs_propres_non_normes = probleme.eigenvaluesByIndex(0,N_max_valeurs_propres,gauche,droite) # Calcul des valeurs propres
                                                                                                # non normés par rapport au facteur 2*M
valeurs_propres = []

print('Valeurs propres calculés par PySlise en Kelvin \n')
for index, E in valeurs_propres_non_normes:
    error = probleme.eigenvalueError(E,gauche,droite)
    print(f'{index:>5}{(E/(2*M)):>11.5f} K {(error/(2*M)):>9.1e}')
    valeurs_propres.append(E/(2*M)) # Valeurs propres en Kelvin
        
fonctions_propres = []
    
for index, E in valeurs_propres_non_normes:
    phi, d_phi = probleme.eigenfunction(E,flux_normalise, gauche, droite) # d_phi est les dérivées des fonctions propres sur le domaine
    I = np.trapz(np.abs(phi)**2,flux_normalise,dx=(borne_droite-borne_gauche)/N_x) # Intégrale de |Psi(x)|² pour normaliser
    fonctions_propres.append(phi/I)

fonctions_propres = np.array(fonctions_propres) # afin de pouvoir utiliser les méthodes .real imag .conj()

# Energies propres normalisé du système
E_0 = valeurs_propres[0]; E_1 = valeurs_propres[1] ; E_2 = valeurs_propres[2] ; E_3 = valeurs_propres[3] 
E_4 = valeurs_propres[4]; E_5 = valeurs_propres[5]; E_6 = valeurs_propres[6]  ; E_7 = valeurs_propres[7]
E_8 = valeurs_propres[8]; E_9 = valeurs_propres[9]; E_10 = valeurs_propres[10]

#%%
# =============================================================================
# Détermination du pulse envoyé et plot
# =============================================================================

print(f'Différence d\énergie en Kelvin des niveaux E_7 et E_6 : \n {E_7-E_6:e} K \n')
omega_6_7 = (E_7-E_6)*k_B/hbar # en rad/s
print(f'Pulsation de l\'impulsion correspondant à une énergie faisant passer de E_6 à E_7 : \n {omega_6_7:e} rad/s \n')

i = np.sqrt(1+(omega_6_7/omega_c)**2) # courant normalisé intervenant dans la formule du pulse
I_p = i*I_c # Courant de polarisation du Pulse
omega_normalise = np.sqrt(i**2-1)*omega_c/omega # Normalisation de la pulsation du pulse
couplage = 0.003

def Pulse(T,intensite=i):
    """ Signal réel envoyé au système avec pulsation normalisé et une phase initiale ajustée"""
    phase_ini = -np.sqrt(i**2-1)*omega_normalise*(decalage_pulse-largeur_pulse/2) # Pour qu"au début de la porte le cos commence à 0
    return Impulsion0(T,i_0=intensite,w=omega_c/omega,phase_0=phase_ini)*Porte(T)

print(f'Pulsation normalisé de l\'impulsion : \n {omega_normalise:e} \n')

def Energie_impulsion(courant=i):
    return  np.trapz(couplage*(np.sqrt(hbar/(2*Z))/k_B)*Pulse(temps_normalise,intensite=courant),temps_normalise,dx=(T_f-T_0)/N_T)
print(f'Energie du signal normalisé en Kelvin : \n {Energie_impulsion():e} K \n')

fig1, ax1 = plt.subplots(1,1,figsize=(11,7))
ax1.plot(temps_normalise,couplage*(np.sqrt(hbar/(2*Z))/k_B)*Pulse(temps_normalise),color='k',\
         label=r'$P(T)/\omega_0 \quad E_{{tot}} = {{{}}}$ K'.format(Energie_impulsion().round(3)))
ax1.set_xlabel(r'Temps normalisé $T = t\omega_0$')
ax1.set_ylabel(r'Energie reçue par le système au cours du temps (K)')
ax1.legend(fontsize=15)
#%%

# =============================================================================
# Plot du pulse pour différentes valeurs de i
# =============================================================================

fig_pulse,ax_pulse = plt.subplots(1,3,figsize=(9,7))
# fig_pulse.tight_layout()e
ax_pulse[2].plot(temps_normalise,1e3*Impulsion0(temps_normalise,i_0=1.003,w=omega_c/omega),color='k',label=r'$V(T,i=1.003)$')
ax_pulse[0].plot(temps_normalise,1e3*Impulsion0(temps_normalise,i_0=1.0001,w=omega_c/omega),color='k',label=r'$V(T,i=1.0001)$')
ax_pulse[1].set_yticklabels([])
ax_pulse[1].plot(temps_normalise,1e3*Impulsion0(temps_normalise,i_0=1.0005,w=omega_c/omega),color='k',label=r'$V(t,i=1.0005)$')
ax_pulse[2].set_yticklabels([])
ax_pulse[0].set_title(r'$V(T,i=1.0001)$')
ax_pulse[1].set_title(r'$V(T,i=1.0005)$')
ax_pulse[2].set_title(r'$V(T,i=1.003)$')
ax_pulse[0].set_ylabel(r'Tension en mV')
# ax_pulse[0].set_xlabel(r'Temps normalisé $T = t\omega_0$')
ax_pulse[1].set_xlabel(r'Temps normalisé $T = t\omega_0$')
# ax_pulse[2].set_xlabel(r'Temps normalisé $T = t\omega_0$')

#%%
# =============================================================================
# Résolution de l'équation de Shrôdinger avec odeintw
# =============================================================================

def E(t,alpha=couplage):
    """ Impulsion énérgique qui va être rentré dans odeintw"""
    return alpha*(np.sqrt(hbar/(2*Z))/k_B)*Pulse(t)

x_e = 0.5135
def V(t):
    return   0 #(couplage**2*C/2)*Pulse(t)**2-(couplage*x_e*Phi_0/(Z*k_B))*Pulse(t)

def afunc(a,t,E_0,E_1,E_2,E_3,E_4,E_5,E_6,E_7,E_8,E_9):
    """ Forme matricielle de l'équation de Schrödinger"""
    a_0,a_1,a_2,a_3,a_4,a_5,a_6,a_7,a_8,a_9 = a
    return [-1j*(k_B/(hbar*omega))*(a_0*(E_0+V(t))+a_1*E(t)),\
            -1j*(k_B/(hbar*omega))*(a_1*(E_1+V(t))+(a_0+np.sqrt(2)*a_2)*E(t)),\
            -1j*(k_B/(hbar*omega))*(a_2*(E_2+V(t))+(np.sqrt(2)*a_1+np.sqrt(3)*a_3)*E(t)),\
            -1j*(k_B/(hbar*omega))*(a_3*(E_3+V(t))+(np.sqrt(3)*a_2+2*a_4)*E(t)),\
            -1j*(k_B/(hbar*omega))*(a_4*(E_4+V(t))+(2*a_3+np.sqrt(5)*a_5)*E(t)),\
            -1j*(k_B/(hbar*omega))*(a_5*(E_5+V(t))+(np.sqrt(5)*a_4+np.sqrt(6)*a_6)*E(t)),\
            -1j*(k_B/(hbar*omega))*(a_6*(E_6+V(t))+E(t)*(np.sqrt(6)*a_5+np.sqrt(7)*a_7)),\
            -1j*(k_B/(hbar*omega))*(a_7*(E_7+V(t))+E(t)*(np.sqrt(7)*a_6+np.sqrt(8)*a_8)),\
            -1j*(k_B/(hbar*omega))*(a_8*(E_8+V(t))+(np.sqrt(8)*a_7+3*a_9)*E(t)),\
            -1j*(k_B/(hbar*omega))*(a_9*(E_9+V(t))+3*E(t))]

def jacobian(a, t,E_0,E_1,E_2,E_3,E_4,E_5,E_6,E_7,E_8,E_9):
    """ Jacobien de l'équation de Schrödinger """
    a_0,a_1,a_2,a_3,a_4,a_5,a_6,a_7,a_8,a_9 = a
    jac = np.array([[-1j*(k_B/(hbar*omega))*(E_0+V(t)),-1j*(k_B/(hbar*omega))*E(t),0,0,0,0,0,0,0],\
                    [-1j*(k_B/(hbar*omega))*E(t),-1j*(k_B/(hbar*omega))*(E_1+V(t)),-1j*(k_B/(hbar*omega))*np.sqrt(2)*E(t),0,0,0,0,0,0,0],\
                    [0,-1j*(k_B/(hbar*omega))*np.sqrt(2)*E(t),-1j*(k_B/(hbar*omega))*(E_2+V(t)),-1j*(k_B/(hbar*omega))*np.sqrt(3)*E(t),0,0,0,0,0,0],\
                    [0,0,-1j*(k_B/(hbar*omega))*np.sqrt(3)*E(t),-1j*(k_B/(hbar*omega))*(E_3+V(t)),-1j*(k_B/(hbar*omega))*2*E(t),0,0,0,0,0],\
                    [0,0,0,-1j*(k_B/(hbar*omega))*2*E(t),-1j*(k_B/(hbar*omega))*(E_4+V(t)),-1j*(k_B/(hbar*omega))*np.sqrt(5)*E(t),0,0,0,0],\
                    [0,0,0,0,-1j*(k_B/(hbar*omega))*np.sqrt(5)*E(t),-1j*(k_B/(hbar*omega))*(E_5+V(t)),-1j*(k_B/(hbar*omega))*np.sqrt(6)*E(t),0,0,0],\
                    [0,0,0,0,0,-1j*(k_B/(hbar*omega))*np.sqrt(6)*E(t),-1j*(k_B/(hbar*omega))*(E_6+V(t)),-1j*(k_B/(hbar*omega))*np.sqrt(7)*E(t),0,0],\
                    [0,0,0,0,0,0,-1j*(k_B/(hbar*omega))*np.sqrt(7)*E(t),-1j*(k_B/(hbar*omega))*(E_7+V(t)),-1j*(k_B/(hbar*omega))*np.sqrt(8)*E(t),0],\
                    [0,0,0,0,0,0,0,-1j*(k_B/(hbar*omega))*np.sqrt(8)*E(t),-1j*(k_B/(hbar*omega))*(E_8+V(t)),-1j*(k_B/(hbar*omega))*3*E(t)],\
                    [0,0,0,0,0,0,0,0,-1j*(k_B/(hbar*omega))*3*E(t),-1j*(k_B/(hbar*omega))*(E_9+V(t))]])
    return jac

a0 = np.array([0+0j,0+0j,0+0j,0+0j,0+0j,0+0j,1+0j,0+0j,0+0j,0+0j]) # Conditions initiales  temporelles
a, infodict = odeintw(afunc, a0, temps_normalise, args=(E_0,E_1,E_2,E_3,E_4,E_5,E_6,E_7,E_8,E_9), Dfun=jacobian,full_output=True)

a_0_non_norm = a[:,0]; a_1_non_norm = a[:,1]; a_2_non_norm = a[:,2]; a_3_non_norm = a[:,3]
a_4_non_norm = a[:,4]; a_5_non_norm = a[:,5]; a_6_non_norm = a[:,6]; a_7_non_norm = a[:,7]
a_8_non_norm = a[:,8]; a_9_non_norm = a[:,9]

a_tot = Norme(a_0_non_norm)**2+Norme(a_1_non_norm)**2+Norme(a_2_non_norm)**2+Norme(a_3_non_norm)**2+Norme(a_4_non_norm)**2\
    +Norme(a_5_non_norm)**2+Norme(a_6_non_norm)**2+Norme(a_7_non_norm)**2+Norme(a_8_non_norm)**2+Norme(a_9_non_norm)**2

# Renormalisation afin d'avoir la somme des normes carré = 1 au cours du temps
a_0 = []; a_1 = []; a_2 = []; a_3 = []; a_4 = []
a_5 = []; a_6 = []; a_7 = []; a_8 = []; a_9 = []
for i in range(a_tot.size):
    a_0.append(a_0_non_norm[i]/np.sqrt(a_tot[i]));a_1.append(a_1_non_norm[i]/np.sqrt(a_tot[i]));a_2.append(a_2_non_norm[i]/np.sqrt(a_tot[i]))
    a_3.append(a_3_non_norm[i]/np.sqrt(a_tot[i]));a_4.append(a_4_non_norm[i]/np.sqrt(a_tot[i]));a_5.append(a_5_non_norm[i]/np.sqrt(a_tot[i]))
    a_6.append(a_6_non_norm[i]/np.sqrt(a_tot[i]));a_7.append(a_7_non_norm[i]/np.sqrt(a_tot[i]));a_8.append(a_8_non_norm[i]/np.sqrt(a_tot[i]))
    a_9.append(a_9_non_norm[i]/np.sqrt(a_tot[i]))
    
a_0 = np.array(a_0);a_1 = np.array(a_1);a_2 = np.array(a_2);a_3 = np.array(a_3);a_4 = np.array(a_4)
a_5 = np.array(a_5);a_6 = np.array(a_6);a_7 = np.array(a_7);a_8 = np.array(a_8);a_9 = np.array(a_9)

# =============================================================================
# Norme des coefficients au cours du temps
# =============================================================================

fig_coefs, ax_coefs = plt.subplots(1,figsize=(11,7))
ax_coefs.plot(temps_normalise,Norme(a_0)**2,color='cyan',label=r'${|a_0(t)|}^2$')
ax_coefs.plot(temps_normalise,Norme(a_1)**2,color='brown',label=r'${|a_1(t)|}^2$')
ax_coefs.plot(temps_normalise,Norme(a_2)**2,color='yellow',label=r'${|a_2(t)|}^2$')
ax_coefs.plot(temps_normalise,Norme(a_3)**2,color='grey',label=r'${|a_3(t)|}^2$')
ax_coefs.plot(temps_normalise,Norme(a_4)**2,color='green',label=r'${|a_4(t)|}^2$')
ax_coefs.plot(temps_normalise,Norme(a_5)**2,color='blue',label=r'${|a_5(t)|}^2$')
ax_coefs.plot(temps_normalise,Norme(a_6)**2,color='k',label=r'${|a_6(t)|}^2$')
ax_coefs.plot(temps_normalise,Norme(a_7)**2,color='orange',label=r'${|a_7(t)|}^2$')
ax_coefs.plot(temps_normalise,Norme(a_8)**2,color='red',label=r'${|a_8(t)|}^2$')
ax_coefs.plot(temps_normalise,Norme(a_9)**2,color='gold',label=r'${|a_9(t)|}^2$')
ax_coefs.set_xlabel(r'Temps normalisé $T = t\omega_0$')
ax_coefs.set_ylabel(r'Amplitude de probabilité')
ax_coefs.legend(fontsize=15)

#%%
# =============================================================================
# Résolution de l'équation de Lindbald avec odeintw
# =============================================================================

anihilation = np.array([[0, np.sqrt(1), 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, np.sqrt(2), 0, 0, 0, 0, 0, 0, 0],\
                        [0, 0, 0, np.sqrt(3), 0, 0, 0, 0, 0, 0],\
                        [0, 0, 0, 0, np.sqrt(4), 0, 0, 0, 0, 0],\
                        [0, 0, 0, 0, 0, np.sqrt(5), 0, 0, 0, 0],\
                        [0, 0, 0, 0, 0, 0, np.sqrt(6), 0, 0, 0],\
                        [0, 0, 0, 0, 0, 0, 0, np.sqrt(7), 0, 0],\
                        [0, 0, 0, 0, 0, 0, 0, 0, np.sqrt(8), 0],\
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, np.sqrt(9)],\
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

creation = np.transpose(anihilation).conj()
nombre = creation.dot(anihilation)
gamma = Z/R_f
    
# Hamiltonien stationnaire
H_0 = np.array([[E_0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, E_1, 0, 0, 0, 0, 0, 0, 0, 0],\
                [0, 0, E_2, 0, 0, 0, 0, 0, 0, 0],\
                [0, 0, 0, E_3, 0, 0, 0, 0, 0, 0],\
                [0, 0, 0, 0, E_4, 0, 0, 0, 0, 0],\
                [0, 0, 0, 0, 0, E_5, 0, 0, 0, 0],\
                [0, 0, 0, 0, 0, 0, E_6, 0, 0, 0],\
                [0, 0, 0, 0, 0, 0, 0, E_7, 0, 0],\
                [0, 0, 0, 0, 0, 0, 0, 0, E_8, 0],\
                [0, 0, 0, 0, 0, 0, 0, 0, 0, E_9]])

def Perturbation(t,alpha=couplage):
    """ Impulsion énérgique qui va être rentré dans odeintw"""
    return alpha*(np.sqrt(hbar/(2*Z))/k_B)*Pulse(t)

def anticom(c,d):
    return c.dot(d)+d.dot(c)

def lindbald(rho, t, H_0,gamma,creation,anihilation,nombre):
    H = H_0 + 0*Perturbation(t)*(creation+anihilation)
    return (-1j*k_B/(hbar*omega))*(H.dot(rho)-rho.dot(H))+gamma*(anihilation.dot(rho).dot(creation)-0.5*anticom(nombre,rho))




# condition initiale
rho0 = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],\
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],\
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],\
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],\
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],\
                 [0, 0, 0, 0, 0, 0, 1+0j, 0, 0, 0],\
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],\
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],\
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

# Call `odeintw`.
sol = odeintw(lindbald, rho0, temps_normalise, args=(H_0,gamma,creation,anihilation,nombre))


rho_00 = [];rho_11 = [];rho_22 = []; rho_33 = [];rho_44 = [];rho_55 = [];rho_66 = []; rho_77 = []; rho_88 = []; rho_99 = []
for i in sol:
    rho_00.append(i[0,0])
    rho_11.append(i[1,1])
    rho_22.append(i[2,2])
    rho_33.append(i[3,3])
    rho_44.append(i[4,4])
    rho_55.append(i[5,5])
    rho_66.append(i[6,6])
    rho_77.append(i[7,7])
    rho_88.append(i[8,8])
    rho_99.append(i[9,9])

    
# =============================================================================
# Norme des coefficients au cours du temps par Lindbald
# =============================================================================

fig_coefs_rho, ax_coefs_rho = plt.subplots(1,figsize=(11,7))
ax_coefs_rho.plot(temps_normalise,rho_00,color='cyan',label=r'${|a_0(t)|}^2$')
ax_coefs_rho.plot(temps_normalise,rho_11,color='brown',label=r'${|a_1(t)|}^2$')
ax_coefs_rho.plot(temps_normalise,rho_22,color='yellow',label=r'${|a_2(t)|}^2$')
ax_coefs_rho.plot(temps_normalise,rho_33,color='grey',label=r'${|a_3(t)|}^2$')
ax_coefs_rho.plot(temps_normalise,rho_44,color='green',label=r'${|a_4(t)|}^2$')
ax_coefs_rho.plot(temps_normalise,rho_55,color='blue',label=r'${|a_5(t)|}^2$')
ax_coefs_rho.plot(temps_normalise,rho_66,color='k',label=r'${|a_6(t)|}^2$')
ax_coefs_rho.plot(temps_normalise,rho_77,color='orange',label=r'${|a_7(t)|}^2$')
ax_coefs_rho.plot(temps_normalise,rho_88,color='red',label=r'${|a_8(t)|}^2$')
ax_coefs_rho.plot(temps_normalise,rho_99,color='gold',label=r'${|a_9(t)|}^2$')
ax_coefs_rho.set_xlabel(r'Temps normalisé $T = t\omega_0$')
ax_coefs_rho.set_ylabel(r'Amplitude de probabilité')
ax_coefs_rho.legend(fontsize=15)
plt.text(250, 0.9, f'$R = 1.10^{6}\;\Omega$',horizontalalignment='center',bbox=dict( alpha=0.5),fontsize=15)

#%%
# =============================================================================
# Observation sur la sphère de Bloch des niveaux |6> et |7>
# =============================================================================

fig_bloch = plt.figure(figsize=(5,5))
ax_bloch = Axes3D(fig_bloch, azim=-40, elev=30)
sphere = qt.Bloch(axes=ax_bloch)
sphere.zlabel = [r'$|6\rangle$',r'$|7\rangle$']
time_bloch = fig_bloch.text(0.5, -0.6, 'caca',horizontalalignment='center',bbox=dict( alpha=0.5),fontsize=15)

def animate_bloch(i):
   sphere.clear()
   psi = (qt.basis(2, 0)*a_6[i] + a_7[i]*qt.basis(2, 1)).unit()
   sphere.add_states([psi])
   sphere.make_sphere()
   time_bloch.set_text(f'T = {temps_normalise[i].round(2)}')
   return ax_bloch, time_bloch

def init_bloch():
   sphere.vector_color = ['r']
   time_bloch.set_text('0')
   return ax_bloch, time_bloch

ani_bloch = animation.FuncAnimation(fig_bloch, animate_bloch,frames=N_T,init_func=init_bloch, blit=False,interval=20,repeat=True)

#%%
# =============================================================================
# Fonction d'ondes, troncation à 3 niveaux
# =============================================================================

def Proba(T):
    """ |Psi(x,t)|²  troncation à 3 niveaux """
    return Norme(a_6[T])**2*Norme(fonctions_propres[6])**2+Norme(a_7[T])**2*Norme(fonctions_propres[7])**2\
           +Norme(a_8[T])**2*Norme(fonctions_propres[8])**2\
           +2*(a_6[T].conj()*a_7[T]*(fonctions_propres[6].conj())*fonctions_propres[7]).real\
           +2*(a_7[T].conj()*a_8[T]*(fonctions_propres[7].conj())*fonctions_propres[8]).real\
           +2*(a_6[T].conj()*a_8[T]*(fonctions_propres[6].conj())*fonctions_propres[8]).real
def Re(T):
    return (a_6[T]*fonctions_propres[6]).real+(a_7[T]*fonctions_propres[7]).real+(a_8[T]*fonctions_propres[8]).real
def Im(T):
    return (a_6[T]*fonctions_propres[6]).imag+(a_7[T]*fonctions_propres[7]).imag+(a_8[T]*fonctions_propres[8]).imag

#%%
# =============================================================================
# Animation des fonctions d'ondes
# =============================================================================

fig_fct_ondes, ax_fct_ondes = plt.subplots(2,1,figsize=(9,6)) 

line0, = ax_fct_ondes[0].plot([],[],label=r'$\mathrm{Re}\left(\langle{x}|\Psi(T)\rangle\right)$',color='blue') 
line1, = ax_fct_ondes[0].plot([],[],label=r'$\mathrm{Im}\left(\langle{x}|\Psi(T)\rangle\right)$',color='orange') 
line2, = ax_fct_ondes[1].plot([],[],label=r'$|\langle{x}|\Psi(T)\rangle|^2$',color='k') 

ax_fct_ondes[0].set_xlim(0, 1)
ax_fct_ondes[0].set_ylim(-10,10)
ax_fct_ondes[0].set_ylabel(r'$\langle{x}|\Psi(T)\rangle$')
ax_fct_ondes[1].set_xlabel(r'')
ax_fct_ondes[0].legend(fontsize=15)

ax_fct_ondes[1].set_xlim(0, 1)
ax_fct_ondes[1].set_ylim(-15,15)
ax_fct_ondes[1].set_xlabel(r'Flux normalisé $x$')
ax_fct_ondes[1].set_ylabel(r'Densité de probabilité')
ax_fct_ondes[1].legend(fontsize=15)

time_text0 = ax_fct_ondes[0].text(0.5, -5, '',horizontalalignment='center',bbox=dict( alpha=0.5))
time_text1 = ax_fct_ondes[1].text(0.5, -5, '',horizontalalignment='center',bbox=dict( alpha=0.5))


def init():
    line0.set_data([],[])
    line1.set_data([],[])
    line2.set_data([],[])
    time_text0.set_text('0')
    time_text1.set_text('0')
    return line0, line1, line2,time_text0,time_text1

def animate(i): 
    y0 = Re(i)
    y1 = Im(i)
    y2 = Proba(i)
    line0.set_data(flux_normalise,y0)
    line1.set_data(flux_normalise,y1)
    line2.set_data(flux_normalise,y2)
    time_text0.set_text(f'T = {temps_normalise[i].round(2)}')
    time_text1.set_text(f'T = {temps_normalise[i].round(2)}')
    return line0, line1, line2,time_text0,time_text1


ani_fct = animation.FuncAnimation(fig_fct_ondes, animate, init_func=init, frames=N_T, blit=True, interval=20, repeat=True)

#%%
# =============================================================================
# Moyenne de l'observable flux au cours du temps
# =============================================================================

moy_flux = []
for i in range (N_T):
    moy_flux.append(np.trapz(flux_normalise*Proba(i),flux_normalise,dx=(borne_droite-borne_gauche)/N_x))

# Fit de la courbe

def model(t,y):
    return y[0]*t+y[1]+y[2]*np.sin(2*np.pi*y[3]*t+y[4])
def residuals(y,a,t):
    return a-model(t,y)

fit = np.polyfit(temps_normalise[1000:],moy_flux[1000:],1)
fft = np.fft.fft(moy_flux[1000:])
freq = np.fft.fftfreq(len(moy_flux[1000:]),1) 
freqmax = np.abs(freq[np.argmax(fft[1:])])
coefs2 = np.concatenate((fit,np.array([5,1/30,2])))
x, flag = leastsq(residuals, coefs2, args=(moy_flux, temps_normalise))

# Plot
fig_moy_flux, ax_moy_flux = plt.subplots(1,figsize=(11,7))
ax_moy_flux.plot(temps_normalise,moy_flux,color='k',label=r'$\int_{0}^{1}x|\Psi(x,t)|^2\mathrm{d}x$')
ax_moy_flux.plot(temps_normalise[1000:],np.polyval(fit,temps_normalise[1000:]),color='orange',label=r'$1,02.10^{-4}\,x+0,311$')
ax_moy_flux.plot(temps_normalise[1000:],model(temps_normalise[1000:],x),color='blue',label=r'Fit des moindres carrés')
ax_moy_flux.set_xlabel(r'Temps normalisé $T = t\,\omega_0$')
ax_moy_flux.set_ylabel(r'Valeur moyenne du flux normalisé $x = \frac{\Phi}{\Phi_0}$')
ax_moy_flux.legend(fontsize=15)

#%%
# =============================================================================
# Résolution de la stochastic Schrödinger equation à deux niveaux
# =============================================================================

def E(t,alpha=couplage):
    """ Impulsion énérgique qui va être rentré dans odeintw"""
    return alpha*(np.sqrt(hbar/(2*Z))/k_B)*Pulse(t)

beta = np.trapz(flux_normalise*Norme(fonctions_propres[6])**2,flux_normalise,dx=(borne_droite-borne_gauche)/N_x)
xi = np.trapz(flux_normalise*Norme(fonctions_propres[7])**2,flux_normalise,dx=(borne_droite-borne_gauche)/N_x)
dzeta = np.trapz(flux_normalise*(fonctions_propres[6]*fonctions_propres[7]).real,flux_normalise,dx=(borne_droite-borne_gauche)/N_x)
k = 1 # weak continuous measurement

#%%

class SDE(nn.Module):

    def __init__(self):
        super().__init__()
        self.scalar1 = nn.Parameter(torch.tensor(k_B/(hbar*omega)), requires_grad=True)  # Scalar parameter.
        self.scalar2 = nn.Parameter(torch.tensor(np.sqrt(3*hbar*omega*L/(Phi_0**2))), requires_grad=True)  # Scalar parameter.
        self.scalar3 = nn.Parameter(torch.tensor((k/omega)), requires_grad=True)  # Scalar parameter.
        self.scalar4 = nn.Parameter(torch.tensor(np.sqrt(2*k/omega)), requires_grad=True)  # Scalar parameter.
        self.beta = nn.Parameter(torch.tensor(beta), requires_grad=True)  # Scalar parameter.
        self.xi = nn.Parameter(torch.tensor(xi), requires_grad=True)  # Scalar parameter.
        self.dzeta = nn.Parameter(torch.tensor(dzeta), requires_grad=True)  # Scalar parameter.
        self.noise_type = "scalar"
        self.sde_type = "ito"
    
    def f(self, t, y):
        xbar = (y[0]**2+y[1]**2)*self.beta+(y[2]**2+y[3]**2)*self.xi+2*(y[0]*y[2]-y[1]*y[3])*self.dzeta
        return torch.tensor([[self.scalar1*(E_6*y[1]+E(t)*np.sqrt(6)*y[3])\
                              -self.scalar3*((xbar**2+self.scalar2**2)*y[0]-self.scalar2*y[2])],\
                             [-self.scalar1*(E_6*y[0]+E(t)*np.sqrt(6)*y[2])\
                              -self.scalar3*((xbar**2+self.scalar2**2)*y[1]-2*xbar*self.scalar2*y[3])],\
                             [self.scalar1*(E_7*y[3]+E(t)*np.sqrt(6)*y[1])\
                              -self.scalar3*((xbar**2+self.scalar2**2)*y[2]-2*xbar*self.scalar2*y[0])],\
                             [-self.scalar1*(E_7*y[2]+E(t)*np.sqrt(6)*y[0])\
                              -self.scalar3*((xbar**2+self.scalar2**2)*y[3]-2*xbar*self.scalar2*y[1])]])
    
    def g(self, t, y):
        xbar = (y[0]**2+y[1]**2)*self.beta+(y[2]**2+y[3]**2)*self.xi+2*(y[0]*y[2]-y[1]*y[3])*self.dzeta
        return torch.tensor([[[self.scalar4*(-xbar*y[0]+self.scalar2*y[2])]],\
                             [[self.scalar4*(-xbar*y[1]+self.scalar2*y[3])]],\
                             [[self.scalar4*(-xbar*y[2]+self.scalar2*y[0])]],\
                             [[self.scalar4*(-xbar*y[3]+self.scalar2*y[1])]]])

batch_size, state_size, t_size = 4, 1,100
sde = SDE()
ts = torch.linspace(0, 100, t_size)
etat_ini = np.sqrt((1-0.999999**2)/3)
y0 = torch.tensor([[0.999999],[etat_ini],[etat_ini],[etat_ini]])
#y0 = torch.full(size=(batch_size, state_size), fill_value=0.1)

with torch.no_grad():
    ys = torchsde.sdeint(sde, y0, ts)  # (t_size, batch_size, state_size) = (100, 3, 1).

a6_sde = ys[:,0]+1j*ys[:,1]
a7_sde = ys[:,2]+1j*ys[:,3]

#%%

# Renormalisation
a_sde_norm = Norme(a6_sde)**2+Norme(a7_sde)**2

# Plot des coefs
fig_bruit, ax_bruit = plt.subplots(1,1,figsize=(11,7))
ax_bruit.plot(ts,Norme(a6_sde)**2/a_sde_norm,color='k',label=r'$|a_6(t)|^2$')
ax_bruit.plot(ts,Norme(a7_sde)**2/a_sde_norm,color='orange',label=r'$|a_7(t)|^2$')
ax_bruit.set_xlabel(r'Temps normalisé $T = t\omega_0$')
ax_bruit.set_ylabel(r'Amplitude probabilité')
ax_bruit.legend(fontsize=15)
plt.text(150, 1, r'$k = 1.10^{11}\,\mathrm{s}^{-1}$',horizontalalignment='center',bbox=dict(facecolor='cyan',alpha=1),fontsize=15)

#%%
# =============================================================================
# Moyenne de l'observable stochastique
# =============================================================================

class SDE_moy(nn.Module):

    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(x[0]), requires_grad=True)
        self.b = nn.Parameter(torch.tensor(x[1]), requires_grad=True)
        self.c = nn.Parameter(torch.tensor(x[2]), requires_grad=True)
        self.d = nn.Parameter(torch.tensor(x[3]), requires_grad=True)
        self.e = nn.Parameter(torch.tensor(x[4]), requires_grad=True)
        self.coef = nn.Parameter(torch.tensor(1/np.sqrt(8*k*omega)), requires_grad=True)
        self.coef1 = nn.Parameter(torch.tensor(1/np.sqrt(8*k)), requires_grad=True)
        self.omega = nn.Parameter(torch.tensor(1/omega), requires_grad=True)
        self.noise_type = "diagonal"
        self.sde_type = "ito"
    
    def f(self, t, y):
        return torch.tensor([[(self.a*t+self.b+self.c*torch.sin(2*3.14*t*self.d+self.e)*self.omega)],\
                            [self.a+self.c*2*3.14*torch.cos(2*3.14*t*self.d+self.e)*self.omega],\
                            [(self.a*t+self.b+self.c*torch.sin(2*3.14*t*self.d+self.e))]])
    
    def g(self, t, y):
        return torch.tensor([[self.coef],[self.coef],[self.coef]])

batch_size_moy, state_size_moy, t_size_moy = 3, 1, 550
sde_moy = SDE_moy()
ts_moy = torch.linspace(0, 275, t_size_moy)
y0_moy = torch.full(size=(batch_size_moy, state_size_moy), fill_value=0.35)

with torch.no_grad():
    ys_moy = torchsde.sdeint(sde_moy, y0_moy, ts_moy)  # (t_size, batch_size, state_size) = (100, 3, 1).

#%%

# Plot de la moyenne bruitée
fig_bruit_moy, ax_bruit_moy = plt.subplots(1,1,figsize=(11,7))
# ax_bruit_moy.plot(ts_moy,ys_moy[:,0],color='blue',label=r'/1e11, k=1e-5')
ax_bruit_moy.plot(ts_moy,ys_moy[:,1],color='green',label=r'/omega')
# ax_bruit_moy.plot(ts_moy,ys_moy[:,2],color='red',label=r'rien')
ax_bruit_moy.set_xlabel(r'Temps normalisé $T = t\omega_0$')
ax_bruit_moy.set_ylabel(r'Mesure de $\hat{x}$ au cours du temps')
ax_bruit_moy.legend(fontsize=15)
plt.text(150, 1, r'$k = 1e11$',horizontalalignment='center',bbox=dict(facecolor='cyan',alpha=1),fontsize=15)

#%%

# Sauvegarder une animation sous forme de gif
writergif = animation.PillowWriter(fps=30) 
ani_fct.save('fct_onde.gif', writer=writergif)













































































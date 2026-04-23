#%% Librerias y paquetes 
import numpy as np
from uncertainties import ufloat, unumpy
import matplotlib.pyplot as plt
import pandas as pd
from glob import glob
import os
import chardet
import re
from clase_resultados import ResultadosESAR

#%% Lector de resultados
def lector_resultados(path):
    '''
    Para levantar archivos de resultados con columnas :
    Nombre_archivo	Time_m	Temperatura_(ºC)	Mr_(A/m)	Hc_(kA/m)	Campo_max_(A/m)	Mag_max_(A/m)	f0	mag0	dphi0	SAR_(W/g)	Tau_(s)	N	xi_M_0
    '''
    with open(path, 'rb') as f:
        codificacion = chardet.detect(f.read())['encoding']

    # Leer las primeras 20 líneas y crear un diccionario de meta
    meta = {}
    with open(path, 'r', encoding=codificacion) as f:
        for i in range(20):
            line = f.readline()
            if i == 0:
                match = re.search(r'Rango_Temperaturas_=_([-+]?\d+\.\d+)_([-+]?\d+\.\d+)', line)
                if match:
                    key = 'Rango_Temperaturas'
                    value = [float(match.group(1)), float(match.group(2))]
                    meta[key] = value
            else:
                # Patrón para valores con incertidumbre (ej: 331.45+/-6.20 o (9.74+/-0.23)e+01)
                match_uncertain = re.search(r'(.+)_=_\(?([-+]?\d+\.\d+)\+/-([-+]?\d+\.\d+)\)?(?:e([+-]\d+))?', line)
                if match_uncertain:
                    key = match_uncertain.group(1)[2:]  # Eliminar '# ' al inicio
                    value = float(match_uncertain.group(2))
                    uncertainty = float(match_uncertain.group(3))
                    
                    # Manejar notación científica si está presente
                    if match_uncertain.group(4):
                        exponent = float(match_uncertain.group(4))
                        factor = 10**exponent
                        value *= factor
                        uncertainty *= factor
                    
                    meta[key] = ufloat(value, uncertainty)
                else:
                    # Patrón para valores simples (sin incertidumbre)
                    match_simple = re.search(r'(.+)_=_([-+]?\d+\.\d+)', line)
                    if match_simple:
                        key = match_simple.group(1)[2:]
                        value = float(match_simple.group(2))
                        meta[key] = value
                    else:
                        # Capturar los casos con nombres de archivo
                        match_files = re.search(r'(.+)_=_([a-zA-Z0-9._]+\.txt)', line)
                        if match_files:
                            key = match_files.group(1)[2:]
                            value = match_files.group(2)
                            meta[key] = value

    # Leer los datos del archivo (esta parte permanece igual)
    data = pd.read_table(path, header=15,
                         names=('name', 'Time_m', 'Temperatura',
                                'Remanencia', 'Coercitividad','Campo_max','Mag_max',
                                'frec_fund','mag_fund','dphi_fem',
                                'SAR','tau',
                                'N','xi_M_0'),
                         usecols=(0,1,2,3,4,5,6,7,8,9,10,11,12,13),
                         decimal='.',
                         engine='python',
                         encoding=codificacion)

    files = pd.Series(data['name'][:]).to_numpy(dtype=str)
    time = pd.Series(data['Time_m'][:]).to_numpy(dtype=float)
    temperatura = pd.Series(data['Temperatura'][:]).to_numpy(dtype=float)
    Mr = pd.Series(data['Remanencia'][:]).to_numpy(dtype=float)
    Hc = pd.Series(data['Coercitividad'][:]).to_numpy(dtype=float)
    campo_max = pd.Series(data['Campo_max'][:]).to_numpy(dtype=float)
    mag_max = pd.Series(data['Mag_max'][:]).to_numpy(dtype=float)
    xi_M_0=  pd.Series(data['xi_M_0'][:]).to_numpy(dtype=float)
    SAR = pd.Series(data['SAR'][:]).to_numpy(dtype=float)
    tau = pd.Series(data['tau'][:]).to_numpy(dtype=float)

    frecuencia_fund = pd.Series(data['frec_fund'][:]).to_numpy(dtype=float)
    dphi_fem = pd.Series(data['dphi_fem'][:]).to_numpy(dtype=float)
    magnitud_fund = pd.Series(data['mag_fund'][:]).to_numpy(dtype=float)

    N=pd.Series(data['N'][:]).to_numpy(dtype=int)
    return meta, files, time,temperatura,Mr, Hc, campo_max, mag_max, xi_M_0, frecuencia_fund, magnitud_fund , dphi_fem, SAR, tau, N
#%% LECTOR CICLOS
def lector_ciclos(filepath):
    with open(filepath, "r") as f:
        lines = f.readlines()[:8]

    metadata = {'filename': os.path.split(filepath)[-1],
                'Temperatura':float(lines[0].strip().split('_=_')[1]),
        "Concentracion_g/m^3": float(lines[1].strip().split('_=_')[1].split(' ')[0]),
            "C_Vs_to_Am_M": float(lines[2].strip().split('_=_')[1].split(' ')[0]),
            "pendiente_HvsI ": float(lines[3].strip().split('_=_')[1].split(' ')[0]),
            "ordenada_HvsI ": float(lines[4].strip().split('_=_')[1].split(' ')[0]),
            'frecuencia':float(lines[5].strip().split('_=_')[1].split(' ')[0])}

    data = pd.read_table(os.path.join(os.getcwd(),filepath),header=7,
                        names=('Tiempo_(s)','Campo_(Vs)','Magnetizacion_(Vs)','Campo_(kA/m)','Magnetizacion_(A/m)'),
                        usecols=(0,1,2,3,4),
                        decimal='.',engine='python',
                        dtype= {'Tiempo_(s)':'float','Campo_(Vs)':'float','Magnetizacion_(Vs)':'float',
                               'Campo_(kA/m)':'float','Magnetizacion_(A/m)':'float'})
    t     = pd.Series(data['Tiempo_(s)']).to_numpy()
    H_Vs  = pd.Series(data['Campo_(Vs)']).to_numpy(dtype=float) #Vs
    M_Vs  = pd.Series(data['Magnetizacion_(Vs)']).to_numpy(dtype=float)#A/m
    H_kAm = pd.Series(data['Campo_(kA/m)']).to_numpy(dtype=float)*1000 #A/m
    M_Am  = pd.Series(data['Magnetizacion_(A/m)']).to_numpy(dtype=float)#A/m

    return t,H_Vs,M_Vs,H_kAm,M_Am,metadata
#%% Obtengo ciclos y resultados para cada concentracion - Todo a 300 kHz

# Autoclave Viejo
sintesis_AV = '260421 Autoclave Viejo'
ciclos_13_AV = glob("Autoclave_Viejo/**/**/*ciclo_promedio_H_M.txt")
resultados_13_AV = glob("Autoclave_Viejo/**/**/*resultados.txt")

ciclos_13_AV.sort()
resultados_13_AV.sort()
conc_13_AV = 7 #g/L

for p in ciclos_13_AV:
    print('  ',p)

for res in resultados_13_AV:
    print('  ',res)
print('-'*50)    
# Autoclave Nuevo
sintesis_AN = '260421 Autoclave Nuevo'
ciclos_13_AN = glob("Autoclave_Nuevo/**/**/*ciclo_promedio_H_M.txt")
resultados_13_AN = glob("Autoclave_Nuevo/**/**/*resultados.txt")

ciclos_13_AN.sort()
resultados_13_AN.sort()
conc_13_AN = 9.9 #g/L

for p in ciclos_13_AN:
    print('  ',p)

for res in resultados_13_AN:
    print('  ',res)    
print('-'*50)    
#%% ploteo Viejo/Nuevo
fig00, (ax,ax2) =plt.subplots(1,2,figsize=(11,5),constrained_layout=True,sharey=True,sharex=True)

ax.set_ylabel('M (A/m)')
ax.set_title(f'Autoclave Viejo - {conc_13_AV:.1f} g/L',loc='left')
ax2.set_title(f'Autoclave Nuevo - {conc_13_AN:.1f} g/L',loc='left')

for i,e in enumerate(ciclos_13_AV):
    if '152dA' in e:
        _,_,_, H_13,M_13,_ = lector_ciclos(ciclos_13_AV[i])
        ax.plot(H_13/1000,M_13,'-',label=f'NF{i}')

for i,e in enumerate(ciclos_13_AN):
    if '152dA' in e:
        _,_,_, H_13,M_13,_ = lector_ciclos(ciclos_13_AN[i])
        ax2.plot(H_13/1000,M_13,'-',label=f'NF{i}')

for a in ax,ax2:
    a.grid()
    a.set_xlabel('H (kA/m)')
    a.legend(loc='upper left')
plt.suptitle(f'Comparativa ciclos promedio NF@cit 13 hs\n300 kHz & 58 kA/m')
plt.savefig('0_ciclos_promedio_NF13h_AV_AN.png',dpi=300)

fig01, (ax,ax2) =plt.subplots(1,2,figsize=(11,5),constrained_layout=True,sharey=True,sharex=True)

ax.set_ylabel('M/[NPM] (Am²/kg)')
ax.set_title(f'Autoclave Viejo - {conc_13_AV:.1f} g/L',loc='left')
ax2.set_title(f'Autoclave Nuevo - {conc_13_AN:.1f} g/L',loc='left')

for i,e in enumerate(ciclos_13_AV):
    if '152dA' in e:
        _,_,_, H_13,M_13,_ = lector_ciclos(ciclos_13_AV[i])
        ax.plot(H_13/1000,M_13/conc_13_AV,'-',label=f'NF{i}')

for i,e in enumerate(ciclos_13_AN):
    if '152dA' in e:
        _,_,_, H_13,M_13,_ = lector_ciclos(ciclos_13_AN[i])
        ax2.plot(H_13/1000,M_13/conc_13_AN,'-',label=f'NF{i}')

for a in ax,ax2:
    a.grid()
    a.set_xlabel('H (kA/m)')
    a.legend(loc='upper left')
plt.suptitle(f'Comparativa ciclos promedio NF@cit 13 hs\n300 kHz & 58 kA/m')
plt.savefig('0_ciclos_promedio_norm_NF13h_AV_AN.png',dpi=300)

#%% Importo ciclos y resultados para buena sintesis 
# NF@cit 13 hs - 260409
sintesis_buena = '260409 buena'
ciclos_13_buena = glob("../260410_ESAR_c_Temp_NF-cit_13h_260409/152dA/**/*ciclo_promedio_H_M.txt",recursive=True)
resultados_13_buena = glob("../260410_ESAR_c_Temp_NF-cit_13h_260409/152dA/**/*resultados.txt",recursive=True)

ciclos_13_buena.sort()
resultados_13_buena.sort()
conc_13_buena = 32.8 #g/L

for p in ciclos_13_buena:
    print('  ',p)

for res in resultados_13_buena:
    print('  ',res)
print('-'*50)   
# ploteo

fig02, ax =plt.subplots(1,1,figsize=(6,5),constrained_layout=True)  
ax.set_ylabel('M (A/m)')
ax.set_title(f'NF@cit 13 hs - {conc_13_buena:.1f} g/L',loc='left')      
for i,e in enumerate(ciclos_13_buena):
    if '152dA' in e:
        _,_,_, H_13,M_13,_ = lector_ciclos(ciclos_13_buena[i])
        ax.plot(H_13/1000,M_13,'-',label=f'NF{i}')
ax.grid()
ax.set_xlabel('H (kA/m)')
ax.legend(loc='upper left')
plt.suptitle(f'Ciclos promedio NF@cit 13 hs {sintesis_buena}\n300 kHz & 58 kA/m')          

#%% Importo ciclos y resultados para sintesis mala
# NF@cit 13 hs - 260409
sintesis_mala = '260330 mala'
ciclos_13_mala = glob("../260406_ESAR_c_Temp_NF-cit_13h_260330/57_150/**/*ciclo_promedio_H_M.txt",recursive=True)
resultados_13_mala = glob("../260406_ESAR_c_Temp_NF-cit_13h_260330/57_150/**/*resultados.txt",recursive=True)

ciclos_13_mala.sort()
resultados_13_mala.sort()
conc_13_mala = 24.0 #g/L
ciclos_13_mala.pop(1) #saco outliers
resultados_13_mala.pop(1) 
for p in ciclos_13_mala:
    print('  ',p)

for res in resultados_13_mala:
    print('  ',res)
print('-'*50)   
# ploteo

fig02, ax =plt.subplots(1,1,figsize=(6,5),constrained_layout=True)  
ax.set_ylabel('M (A/m)')
ax.set_title(f'NF@cit 13 hs - {conc_13_mala:.1f} g/L',loc='left')      
for i,e in enumerate(ciclos_13_mala):
    if '150dA' in e:
        _,_,_, H_13,M_13,_ = lector_ciclos(ciclos_13_mala[i])
        ax.plot(H_13/1000,M_13,'-',label=f'NF{i}')
ax.grid()
ax.set_xlabel('H (kA/m)')
ax.legend(loc='upper left')
plt.suptitle(f'Ciclos promedio NF@cit 13 hs {sintesis_mala}\n300 kHz & 58 kA/m')          

#%% COMPARO LAS 4 SINTESIS en grafico 2x2 
# primero extraigo SAR y tau de resultados para cada sintesis
def extraer_SAR_tau(resultados):
    SAR = []
    tau = []
    for res in resultados:
        meta,_,_,_,_,_,_,_,_,_,_,_,_,_,_ = lector_resultados(res)   
        SAR.append(meta['SAR_W/g'])
        tau.append(meta['tau_ns']) 
    return SAR, tau
SAR_13_AV, tau_13_AV = extraer_SAR_tau(resultados_13_AV)
SAR_13_AN, tau_13_AN = extraer_SAR_tau(resultados_13_AN)
SAR_13_buena, tau_13_buena = extraer_SAR_tau(resultados_13_buena)
SAR_13_mala, tau_13_mala = extraer_SAR_tau(resultados_13_mala)    
#%% Ploteo comparativo de las 4 sintesis en grafico 2x2, normalizando por concentracion y con SAR en la leyenda

fig5 , axs = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True, sharex=True, sharey=True)
axs[0, 0].set_title(f'{sintesis_AV} - {conc_13_AV:.1f} g/L', loc='left')
axs[0, 1].set_title(f'{sintesis_AN} - {conc_13_AN:.1f} g/L', loc='left')
axs[1, 0].set_title(f'{sintesis_buena} - {conc_13_buena:.1f} g/L', loc='left')
axs[1, 1].set_title(f'{sintesis_mala} - {conc_13_mala:.1f} g/L', loc='left')

for i,e in enumerate(ciclos_13_AV):
    if '152dA' in e:
        _,_,_, H_13,M_13,_ = lector_ciclos(ciclos_13_AV[i])
        axs[0, 0].plot(H_13/1000,M_13/conc_13_AV,'-',label=f'{SAR_13_AV[i]:.3uS}')

for i,e in enumerate(ciclos_13_AN):
    if '152dA' in e:
        _,_,_, H_13,M_13,_ = lector_ciclos(ciclos_13_AN[i])
        axs[0, 1].plot(H_13/1000,M_13/conc_13_AN,'-',label=f'{SAR_13_AN[i]:.3uS}')

for i,e in enumerate(ciclos_13_buena):
    if '152dA' in e:
        _,_,_, H_13,M_13,_ = lector_ciclos(ciclos_13_buena[i])
        axs[1, 0].plot(H_13/1000,M_13/conc_13_buena,'-',label=f'{SAR_13_buena[i]:.3uS}')

for i,e in enumerate(ciclos_13_mala):
    if '150dA' in e:
        _,_,_, H_13,M_13,_ = lector_ciclos(ciclos_13_mala[i])
        axs[1, 1].plot(H_13/1000,M_13/conc_13_mala,'-',label=f'{SAR_13_mala[i]:.3uS}')

for a in axs.flatten():
    a.grid()
    a.legend(title='SAR (W/g)',loc='upper left',frameon=True,shadow=True)

for a in [axs[1, 0], axs[1, 1]]:
    a.set_xlabel('H (kA/m)') 
for a in [axs[0, 0], axs[1, 0]]:
    a.set_ylabel('M/[NPM] (Am²/kg)') 
plt.suptitle(f'Comparativa sintesis NF@cit 13 hs\n300 kHz & 58 kA/m')
# %% ploteo comparativo de errorbars de tau

cuadro = '$f=300$ kHz\n$H_0=58$ kA/m'
categorias = [sintesis_AV, sintesis_AN, sintesis_buena, sintesis_mala]


x = np.arange(len(categorias))

fig, ax = plt.subplots(figsize=(8,5),constrained_layout=True)
sep = 0.25

for i,s in enumerate(SAR_13_AV):
    ax.bar(i*sep-sep, s.n, yerr=s.s, width=0.2, capsize=5, color='C0')

for i,s in enumerate(SAR_13_AN):
    ax.bar(1+i*sep-sep, s.n, yerr=s.s, width=0.2, capsize=5, color='C1')

for i,s in enumerate(SAR_13_buena):
    ax.bar(2+i*sep-sep, s.n, yerr=s.s, width=0.2, capsize=5, color='C2')

for i,s in enumerate(SAR_13_mala):
    ax.bar(3+i*sep-sep, s.n, yerr=s.s, width=0.2, capsize=5, color='C3')


ax.set_xticks(x)
ax.set_xticklabels(categorias)
ax.set_ylabel('ESAR')
ax.set_xlabel('Categoría')
ax.set_title('ESAR NF@cit 13 hs - Comparativa de síntesis')
ax.grid(axis='y', alpha=0.3)

ax.text(0.9,0.9,cuadro, transform=ax.transAxes, 
        va='top', ha='center', fontsize=12,
        bbox=dict(alpha=0.8,facecolor='white'))
plt.show()

#%%

import pandas as pd
import functii as f
import acp.ACP as acp
import grafice as g


tabel = pd.read_csv('./dataIN/TariPtAnaliza.csv', index_col=0)
print(tabel)

vars = tabel.columns.values[1:]
print(vars, type(vars))

obs = tabel.index.values
print(obs, type(obs))

X = tabel[vars].values
print(X)

Xstd = f.standardizare(X)
Xstd_df = pd.DataFrame(data=Xstd, index=obs, columns=vars)
Xstd_df.to_csv('./dataOUT/analizaTari.csv')

modelACP = acp.ACP(Xstd)
valoriProp = modelACP.getValoriProp()

g.componentePrincipale(valoriProprii=valoriProp)


compPrin = modelACP.getCompPrin()
componente = ['C'+str(j+1) for j in range(compPrin.shape[1])]
compPrin_df = pd.DataFrame(data=compPrin, index=obs,columns=componente)
compPrin_df.to_csv('./dataOUT/CompPrin.csv')


factorLoadings = modelACP.getFactorLoadings()
factorLoadings_df = pd.DataFrame(data=factorLoadings,index=vars, columns=componente)

g.corelograma(matrice=factorLoadings_df, titlu='Corelograma factorilor de corelatie')
g.afisare()

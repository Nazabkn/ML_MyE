# Como ver proyecto: 

Los datos en formato Parquet se encuentran en 'data/01_raw/' y ahí están: 
- neo.parquet
- neo_v2.parquet
- meteorite-landings.parquet 

Perdón, yo lo cambie a parquet porque queria trabajar con estos datasets, creo que no habrá problema porque son varios datos D':!

Los conjuntos de datos de la carpeta 'data/01_raw' fueron convertidos a formato Parquet para facilitar la exploración. Para leerlos es necesario instalar previamente el backend 'fastparquet'

Para abrir los archivos Parquet, se necesita instalar fastparquet: 
Se ejecuta asi en el terminal:

pip install fastparquet
(Es que no supe como hacerlo sin eso :C)



Ejemplo rápido de lectura donde se pueden abrir en python con pandas:
   python
import pandas as pd
df = pd.read_parquet("data/01_raw/neo.parquet", engine="fastparquet")
print(df.head()) ← ese usé para probar si todo estaba ok ( ദ്ദി ˙ᗜ˙ )

El proyecto usa dvc para manejar los datasets 
Los archivos de datos se deben obtener con:

dvc pull
 :D




# train_scripts/ - Scripts de Entrenamiento Generados

Contiene scripts de entrenamiento generados automaticamente por `run_trains.py` para barridos de hiperparametros.

## Como se generan
```bash
python run_trains.py
```

Esto lee un script base (como `trainRL.py`), modifica un hiperparametro con diferentes valores, y genera un archivo `.py` por cada variante con nombres unicos que incluyen timestamp.

## Ejecucion
```bash
# Ejecutar todos los scripts en paralelo
./runpy.sh train_scripts/train_*.py &

# O uno por uno
python train_scripts/train_E05a_20260101.py
```

## Nota
Estos archivos son temporales y se regeneran segun las necesidades del experimento. El archivo `run_trains.py` en la raiz controla que parametros variar.

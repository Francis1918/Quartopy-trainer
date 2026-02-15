# development/ - Desarrollo y Debugging

Notebooks y scripts usados durante el desarrollo del proyecto. Contienen pruebas, prototipos y herramientas de depuracion.

## Subcarpetas

### 01 remove files/
Utilidades para limpieza de archivos del proyecto.
- `remove.ipynb`: Operaciones de limpieza de archivos
- `02_check_process.ipynb`: Verificacion de procesos

### 02_includingBT/ - Desarrollo del Modelo Bradley-Terry
Proceso de implementacion e integracion del ranking Bradley-Terry.
- `BT_imp.py`: Implementacion del modelo BT
- `test_bradley_terry*.ipynb`: (4 notebooks) Pruebas progresivas del algoritmo
- `test_BT_fcn_iter.ipynb`: Testing de la funcion iterativa
- `EXP_id03.pkl`: Datos serializados del experimento
- `plotting*.m`: Scripts de MatLab para visualizacion

### 03_validate_deserialize_boards/ - Validacion de Tableros
Herramientas para validar la serializacion/deserializacion de estados del tablero.
- `aux_validate.py`: Funciones auxiliares de validacion
- `trainRL_view_pred.py`: Visualizacion de predicciones del modelo

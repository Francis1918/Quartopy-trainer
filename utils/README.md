# utils/ - Utilidades del Proyecto

Funciones auxiliares para configuracion del entorno, logging, compatibilidad y metricas.

## Archivos

### env_bootstrap.py - Configuracion del Entorno
Resuelve la ruta a la libreria Quartopy para que el proyecto pueda importarla.

- `bootstrap_quartopy_path()`: Busca Quartopy en:
  1. Variable de entorno `QUARTOPY_PATH` (via `.env`)
  2. Carpeta `../Quartopy` relativa al proyecto
  - Agrega la ruta a `sys.path`

### logger.py - Logging con Colores
Configura el logger "TrainRL" con salida coloreada por nivel:

| Nivel | Color |
|-------|-------|
| DEBUG | Cyan |
| INFO | Verde |
| WARNING | Amarillo |
| ERROR | Rojo |
| CRITICAL | Magenta |

### play_games_compat.py - Compatibilidad con Quartopy
Wrapper que maneja diferentes versiones del API de Quartopy.

- `play_games_compat(...)`: Detecta la version del API y llama la funcion correcta
- `_to_win_rate_counts()`: Normaliza formatos de resultado (ingles/espanol)
- `_temporary_disable_export()`: Context manager para deshabilitar exportacion CSV

### metrics/bradley_terry.py - Ranking Bradley-Terry
Modelo Bradley-Terry para ranking de agentes por resultados head-to-head.

- `calculate_BradleyTerry(score, W, EPOCHS=4)`: Calcula scores iterativamente
  - `W`: Matriz de victorias (DataFrame donde W[i][j] = victorias de i sobre j)
  - Convergencia por threshold o iteraciones fijas

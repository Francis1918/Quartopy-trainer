# -*- coding: utf-8 -*-

"""Swiss Tournament with Bradley-Terry Scoring for Quarto Bots

Este script ejecuta un torneo Swiss (emparejamientos progresivos) entre todos los checkpoints
encontrados en una carpeta, construye una matriz de victorias, y calcula el ranking final
usando el sistema Bradley-Terry.

VENTAJAS vs Round-Robin completo:
- Mucho mas rapido: O(N*R) vs O(N^2) donde R = numero de rondas
- Estadisticamente robusto: Bradley-Terry con datos de Swiss
- Escalable: soporta miles de agentes

Ejemplo: 3000 agentes
- Round-Robin: 44,985,000 partidas (~260 dias)
- Swiss (500 rondas): ~1,500,000 partidas (~10-20 horas)

Author: z_tjona
Date: 2026-02-16

"I find that I don't understand things unless I try to program them."
-Donald E. Knuth
"""

# ========================= CONFIGURACION EDITABLE =========================
# Modifica estos parametros segun tus necesidades

# ===== CONFIGURACIONES RAPIDAS PREDEFINIDAS =====
# Descomenta UNA de estas configuraciones segun tu objetivo:

# OPCION 1: ULTRA RAPIDO (15 min con 3000 agentes)
# NUM_ROUNDS = 1
# MATCHES_PER_PAIRING = 1
# ENABLE_COLOR_SWAP = False
# SAMPLE_PERCENTAGE = 100  # Usar todos los agentes
# TOP_N_BOTS = None  # None = usar todos

# OPCION 2: RAPIDO (1-2 horas con 3000 agentes)
# NUM_ROUNDS = 10
# MATCHES_PER_PAIRING = 1
# ENABLE_COLOR_SWAP = True
# SAMPLE_PERCENTAGE = 100
# TOP_N_BOTS = None

# OPCION 3: BALANCEADO (1 dia con 3000 agentes)
# NUM_ROUNDS = 50
# MATCHES_PER_PAIRING = 1
# ENABLE_COLOR_SWAP = True
# SAMPLE_PERCENTAGE = 100
# TOP_N_BOTS = None

# OPCION 4: TOP 500 AGENTES RAPIDO (30 min)
# NUM_ROUNDS = 10
# MATCHES_PER_PAIRING = 1
# ENABLE_COLOR_SWAP = True
# SAMPLE_PERCENTAGE = 100
# TOP_N_BOTS = 500  # Solo los top 500 por epoca

# ===== CONFIGURACION MANUAL =====
# O configura manualmente aqui:

# Numero de rondas del torneo Swiss
# Cada ronda empareja a cada bot exactamente una vez (o bye si impar)
# Mas rondas = ranking mas preciso pero mas tiempo
# Recomendado: 1-10 para rapido, 50-100 para balanceado, 200-500 para preciso
NUM_ROUNDS = 1  # CAMBIADO: 1 ronda para ultra rapido

# Numero de partidas por emparejamiento en cada ronda
# IMPORTANTE: Si ENABLE_COLOR_SWAP=True, se duplica automaticamente
# Recomendado: 1 para rapido, 2-5 para balanceado, 5-10 para preciso
MATCHES_PER_PAIRING = 1  # CAMBIADO: 1 partida para ultra rapido

# Habilitar swap de colores (cada emparejamiento juega 2 veces: P1 vs P2 y P2 vs P1)
# Si False, solo juega una vez (mas rapido pero menos justo)
# Recomendado: True para precision, False para velocidad maxima
ENABLE_COLOR_SWAP = False  # NUEVO: Deshabilitar swap para ultra rapido

# Porcentaje de emparejamientos a jugar por ronda (1-100)
# 100 = jugar todos los emparejamientos (normal)
# 50 = jugar solo la mitad (aleatorio)
# 10 = jugar solo 10% (muy rapido pero menos preciso)
# Recomendado: 100 para normal, 50-20 para rapido, 10 para ultra rapido
SAMPLE_PERCENTAGE = 100  # NUEVO: 100 = todos los emparejamientos

# Limitar a los top N bots por epoca (None = usar todos)
# Ejemplo: 500 = solo los 500 bots con mayor epoca
# Util para torneos rapidos enfocados en los mejores
# Recomendado: None para evaluar todos, 500-1000 para rapidez
TOP_N_BOTS = None  # NUEVO: None = usar todos los 3000

# Temperatura para las decisiones del bot (0.0 = deterministico, >1.0 = mas exploracion)
# 0.1 es recomendado para evaluacion (decision casi determinista pero con algo de variedad)
TEMPERATURE = 0.1

# Determinista: Si True, el bot siempre toma la accion con mayor Q-value
# Si False, usa softmax con temperatura para muestrear acciones
DETERMINISTIC = False

# Ruta a la carpeta con los checkpoints (.pt) de los agentes
AGENTS_FOLDER = r"C:\Users\bravo\Documents\Metodos Numericos Pycharm\Mech Interp\TorneoMasivo\Agentes"

# Ruta donde se guardaran los resultados (.csv)
RESULTS_FOLDER = r"C:\Users\bravo\Documents\Metodos Numericos Pycharm\Mech Interp\TorneoMasivo\ResutadosTorneo"

# McMahon Scoring: Si True, bots con mayor epoca empiezan con ventaja inicial
# Recomendado: True si los epochs correlacionan con habilidad
USE_MCMAHON = True

# Parametros de Bradley-Terry
# IMPORTANTE: Para 3000 agentes, reducir estas iteraciones drasticamente
# Con pocos datos (1 ronda), Bradley-Terry converge rapido
BRADLEY_TERRY_EPOCHS = 5  # MUY REDUCIDO: 5 iteraciones - suficiente para 1 ronda
BRADLEY_TERRY_THRESHOLD = 1e-2  # MUY RELAJADO: convergencia rapida
BRADLEY_TERRY_NORMALIZE = True  # Normalizar scores por media geometrica

# Guardar progreso cada N rondas (0 = solo al final)
# Si se interrumpe, podras recuperar el progreso hasta la ultima ronda guardada
SAVE_PROGRESS_EVERY = 50  # 0 = desactivado, >0 = guardar cada N rondas

# ========================================================================

import logging
from sys import stdout
from datetime import datetime
from pathlib import Path
import pandas as pd
import re
import numpy as np
from tqdm.auto import tqdm
import pickle

from models.CNN_uncoupled import QuartoCNN as QuartoCNN_uncoupled
from utils.env_bootstrap import bootstrap_quartopy_path
from utils.play_games_compat import play_games_compat
from utils.metrics.bradley_terry import calculate_BradleyTerry

bootstrap_quartopy_path()

# ----------------------------- logging config --------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(levelname)s] %(message)s",
    stream=stdout,
    datefmt="%m-%d %H:%M:%S",
)
logging.info("Iniciando torneo Swiss con Bradley-Terry")
logging.info(datetime.now())


def extract_epoch(filename):
    """Extract epoch number from checkpoint filename"""
    match = re.search(r"E[_\s]+(\d+)", filename.stem, re.IGNORECASE)
    if match:
        return int(match.group(1))
    # Try alternative patterns
    match = re.search(r"epoch[_\s]*(\d+)", filename.stem, re.IGNORECASE)
    return int(match.group(1)) if match else 0


def save_progress(tournament_state, results_path, round_num):
    """Guarda el progreso actual del torneo"""
    timestamp = tournament_state["timestamp"]
    progress_file = results_path / f"tournament_progress_{timestamp}_R{round_num:04d}.pkl"

    with open(progress_file, "wb") as f:
        pickle.dump(tournament_state, f)

    logging.info(f"Progreso guardado en: {progress_file}")
    return progress_file


def save_final_results(tournament_state, results_path):
    """Guarda los resultados finales del torneo en CSV"""
    timestamp = tournament_state["timestamp"]
    bots_info = tournament_state["bots_info"]
    W = tournament_state["win_matrix"]
    bt_scores = tournament_state["bradley_terry_scores"]
    N = len(bots_info)

    # ----------------------------- CREAR RESULTADOS FINALES --------------------------
    final_results = []
    for idx in range(N):
        bot = bots_info[idx]

        # Calcular estadisticas totales del bot
        total_wins = W.loc[idx, :].sum()
        total_losses = W.loc[:, idx].sum()
        total_matches = total_wins + total_losses
        win_rate = total_wins / total_matches if total_matches > 0 else 0.0
        swiss_score = bot["score"]

        final_results.append({
            "rank": 0,  # Se asignara despues de ordenar
            "bot_id": idx,
            "bot_name": bot["name"],
            "epoch": bot["epoch"],
            "bradley_terry_score": bt_scores[idx],
            "swiss_score": swiss_score,
            "total_wins": total_wins,
            "total_losses": total_losses,
            "total_matches": total_matches,
            "win_rate": win_rate,
            "matches_played": bot["matches_played"],
        })

    # Ordenar por Bradley-Terry score (descendente)
    final_results.sort(key=lambda x: x["bradley_terry_score"], reverse=True)

    # Asignar ranks
    for rank, result in enumerate(final_results, 1):
        result["rank"] = rank

    # ----------------------------- GUARDAR ARCHIVOS CSV --------------------------
    # Guardar ranking final
    results_file = results_path / f"swiss_bradley_terry_ranking_{timestamp}.csv"
    df_results = pd.DataFrame(final_results)
    df_results.to_csv(results_file, index=False, encoding="utf-8")
    logging.info(f"Ranking guardado en: {results_file}")

    # Guardar matriz de victorias
    matrix_file = results_path / f"win_matrix_{timestamp}.csv"
    W.to_csv(matrix_file, encoding="utf-8")
    logging.info(f"Matriz de victorias guardada en: {matrix_file}")

    # Guardar historial de rondas (opcional)
    if "round_history" in tournament_state:
        history_file = results_path / f"round_history_{timestamp}.csv"
        df_history = pd.DataFrame(tournament_state["round_history"])
        df_history.to_csv(history_file, index=False, encoding="utf-8")
        logging.info(f"Historial de rondas guardado en: {history_file}")

    return final_results


def run_swiss_tournament():
    """
    Ejecuta un torneo Swiss con scoring Bradley-Terry.

    Swiss system: empareja bots con scores similares cada ronda.
    Al final, calcula Bradley-Terry con la matriz de victorias acumulada.

    Mucho mas eficiente que round-robin para muchos agentes.
    """
    from bot.CNN_bot import Quarto_bot

    # ----------------------------- VALIDAR RUTAS --------------------------
    agents_path = Path(AGENTS_FOLDER)
    results_path = Path(RESULTS_FOLDER)

    if not agents_path.exists():
        logging.error(f"La carpeta de agentes no existe: {AGENTS_FOLDER}")
        return None

    # Crear carpeta de resultados si no existe
    results_path.mkdir(parents=True, exist_ok=True)

    # ----------------------------- CARGAR CHECKPOINTS --------------------------
    logging.info(f"Cargando checkpoints desde: {AGENTS_FOLDER}")
    checkpoint_files = sorted(list(agents_path.glob("*.pt")))

    if len(checkpoint_files) == 0:
        logging.error(f"No se encontraron archivos .pt en {AGENTS_FOLDER}")
        return None

    N_total = len(checkpoint_files)
    logging.info(f"Encontrados {N_total} checkpoints totales")

    # ===== FILTRAR TOP N BOTS SI SE ESPECIFICA =====
    if TOP_N_BOTS is not None and TOP_N_BOTS < N_total:
        logging.info(f"Filtrando a los top {TOP_N_BOTS} bots por epoca...")
        # Extraer epocas y ordenar
        checkpoint_epochs = [(f, extract_epoch(f)) for f in checkpoint_files]
        checkpoint_epochs.sort(key=lambda x: -x[1])  # Descendente por epoca
        checkpoint_files = [f for f, _ in checkpoint_epochs[:TOP_N_BOTS]]
        logging.info(f"Usando {len(checkpoint_files)} bots (top {TOP_N_BOTS})")

    N = len(checkpoint_files)
    logging.info(f"Bots en el torneo: {N}")

    # ----------------------------- CREAR INFO DE BOTS --------------------------
    bots_info = []
    for idx, ckpt_file in enumerate(checkpoint_files):
        epoch = extract_epoch(ckpt_file)
        bots_info.append({
            "id": idx,
            "name": ckpt_file.stem,
            "path": str(ckpt_file),
            "epoch": epoch,
            "score": 0.0,  # Swiss score (acumulativo)
            "wins": 0,
            "draws": 0,
            "losses": 0,
            "matches_played": 0,
        })

    logging.info(f"Bots cargados: {[b['name'] for b in bots_info[:5]]}..." if N > 5 else [b['name'] for b in bots_info])

    # ----------------------------- MATRIZ DE VICTORIAS --------------------------
    # W[i,j] = numero de victorias del bot i sobre el bot j
    W = pd.DataFrame(0.0, index=range(N), columns=range(N))

    # ----------------------------- CONFIGURACION --------------------------
    swap_multiplier = 2 if ENABLE_COLOR_SWAP else 1
    sample_factor = SAMPLE_PERCENTAGE / 100.0
    pairings_per_round = (N // 2) * sample_factor
    total_matches_estimate = int(pairings_per_round * NUM_ROUNDS * MATCHES_PER_PAIRING * swap_multiplier)

    logging.info("=" * 80)
    logging.info(f"CONFIGURACION DEL TORNEO SWISS + BRADLEY-TERRY")
    logging.info("=" * 80)
    logging.info(f"Total de bots en torneo: {N}" + (f" (filtrados de {N_total})" if TOP_N_BOTS else ""))
    logging.info(f"Numero de rondas: {NUM_ROUNDS}")
    logging.info(f"Partidas por emparejamiento: {MATCHES_PER_PAIRING}")
    logging.info(f"Swap de colores: {'SI' if ENABLE_COLOR_SWAP else 'NO'} (x{swap_multiplier})")
    logging.info(f"Sampleo de emparejamientos: {SAMPLE_PERCENTAGE}%")
    logging.info(f"Emparejamientos por ronda: ~{int(pairings_per_round)}")
    logging.info(f"Total estimado de partidas: ~{total_matches_estimate:,}")
    logging.info(f"Temperatura: {TEMPERATURE}")
    logging.info(f"Deterministico: {DETERMINISTIC}")
    logging.info(f"McMahon scoring: {USE_MCMAHON}")
    logging.info(f"Guardar progreso cada: {SAVE_PROGRESS_EVERY} rondas" if SAVE_PROGRESS_EVERY > 0 else "Solo al final")
    logging.info("=" * 80)

    # Estimar tiempo
    seconds_per_match = 0.5  # Estimacion conservadora (GPU)
    estimated_seconds = total_matches_estimate * seconds_per_match
    estimated_minutes = estimated_seconds / 60
    estimated_hours = estimated_seconds / 3600

    if estimated_hours < 1:
        logging.info(f"Tiempo estimado: ~{estimated_minutes:.1f} minutos")
    elif estimated_hours < 24:
        logging.info(f"Tiempo estimado: ~{estimated_hours:.1f} horas")
    else:
        logging.info(f"Tiempo estimado: ~{estimated_hours:.1f} horas ({estimated_hours/24:.1f} dias)")
    logging.info("=" * 80)

    # ----------------------------- ESTADO DEL TORNEO --------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tournament_state = {
        "timestamp": timestamp,
        "bots_info": bots_info,
        "win_matrix": W,
        "round_history": [],
        "config": {
            "n_bots": N,
            "n_bots_total": N_total,
            "top_n_bots": TOP_N_BOTS,
            "num_rounds": NUM_ROUNDS,
            "matches_per_pairing": MATCHES_PER_PAIRING,
            "enable_color_swap": ENABLE_COLOR_SWAP,
            "sample_percentage": SAMPLE_PERCENTAGE,
            "temperature": TEMPERATURE,
            "deterministic": DETERMINISTIC,
            "use_mcmahon": USE_MCMAHON,
        }
    }

    bot_params = {
        "deterministic": DETERMINISTIC,
        "temperature": TEMPERATURE,
    }

    # ----------------------------- TORNEO SWISS --------------------------
    try:
        for round_num in tqdm(range(1, NUM_ROUNDS + 1), desc="Rondas completadas", position=0):
            logging.info("=" * 80)
            logging.info(f"RONDA {round_num} / {NUM_ROUNDS}")
            logging.info("=" * 80)

            # ===== EMPAREJAMIENTO SWISS =====
            if USE_MCMAHON and round_num == 1:
                # Primera ronda: ordenar por epoca (McMahon)
                logging.info("Aplicando McMahon initial scoring (por epoca)")
                bots_info.sort(key=lambda x: -x["epoch"])
            else:
                # Ordenar por score Swiss (y desempate por wins, epoch)
                bots_info.sort(key=lambda x: (-x["score"], -x["wins"], -x["epoch"]))

            # Emparejar bots consecutivos en la lista ordenada
            pairings = []
            paired_indices = set()

            for i in range(len(bots_info)):
                if i in paired_indices:
                    continue

                # Buscar oponente: siguiente bot no emparejado
                for j in range(i + 1, len(bots_info)):
                    if j not in paired_indices:
                        pairings.append((i, j))
                        paired_indices.add(i)
                        paired_indices.add(j)
                        break

            # Manejar numero impar de bots (bye)
            if len(bots_info) % 2 == 1:
                for i in range(len(bots_info)):
                    if i not in paired_indices:
                        logging.info(f"Bot {bots_info[i]['name']} recibe BYE")
                        bots_info[i]["score"] += 1.0  # Bye = 1 punto
                        break

            # ===== SAMPLEAR EMPAREJAMIENTOS SI SE ESPECIFICA =====
            if SAMPLE_PERCENTAGE < 100:
                import random
                original_count = len(pairings)
                sample_count = max(1, int(original_count * SAMPLE_PERCENTAGE / 100.0))
                pairings = random.sample(pairings, sample_count)
                logging.info(f"Sampleando {sample_count}/{original_count} emparejamientos ({SAMPLE_PERCENTAGE}%)")

            logging.info(f"Emparejamientos para ronda {round_num}: {len(pairings)} partidas")

            # ===== JUGAR PARTIDAS =====
            for idx1, idx2 in tqdm(pairings, desc=f"Ronda {round_num}", position=1, leave=False):
                bot1_info = bots_info[idx1]
                bot2_info = bots_info[idx2]

                # Crear instancias de bots
                bot1 = Quarto_bot(
                    model_path=bot1_info["path"],
                    model_class=QuartoCNN_uncoupled,
                    **bot_params,
                )
                bot2 = Quarto_bot(
                    model_path=bot2_info["path"],
                    model_class=QuartoCNN_uncoupled,
                    **bot_params,
                )

                # ===== JUEGO 1: bot1 (P1) vs bot2 (P2) =====
                _, win_rate1 = play_games_compat(
                    matches=MATCHES_PER_PAIRING,
                    player1=bot1,
                    player2=bot2,
                    verbose=False,
                    save_match=False,  # NO guardar partidas
                    mode_2x2=True,
                    PROGRESS_MESSAGE="",
                )

                # Actualizar matriz W (empates cuentan como 0.5 win)
                W.loc[bot1_info["id"], bot2_info["id"]] += win_rate1["Player 1"] + win_rate1["Tie"] * 0.5
                W.loc[bot2_info["id"], bot1_info["id"]] += win_rate1["Player 2"] + win_rate1["Tie"] * 0.5

                # ===== JUEGO 2: SWAP DE COLORES (OPCIONAL) =====
                if ENABLE_COLOR_SWAP:
                    _, win_rate2 = play_games_compat(
                        matches=MATCHES_PER_PAIRING,
                        player1=bot2,
                        player2=bot1,
                        verbose=False,
                        save_match=False,
                        mode_2x2=True,
                        PROGRESS_MESSAGE="",
                    )

                    # Actualizar matriz W
                    W.loc[bot2_info["id"], bot1_info["id"]] += win_rate2["Player 1"] + win_rate2["Tie"] * 0.5
                    W.loc[bot1_info["id"], bot2_info["id"]] += win_rate2["Player 2"] + win_rate2["Tie"] * 0.5
                else:
                    # Sin swap, win_rate2 no existe
                    win_rate2 = {"Player 1": 0, "Player 2": 0, "Tie": 0}

                # ===== ACTUALIZAR SCORES SWISS =====
                # Total de partidas depende de si hay swap
                total_matches = MATCHES_PER_PAIRING * (2 if ENABLE_COLOR_SWAP else 1)

                bot1_total_wins = win_rate1["Player 1"] + win_rate2["Player 2"]
                bot2_total_wins = win_rate1["Player 2"] + win_rate2["Player 1"]
                bot1_draws = win_rate1["Tie"] + win_rate2["Tie"]
                bot2_draws = bot1_draws

                bot1_info["wins"] += bot1_total_wins
                bot1_info["draws"] += bot1_draws
                bot1_info["losses"] += bot2_total_wins
                bot1_info["matches_played"] += total_matches

                bot2_info["wins"] += bot2_total_wins
                bot2_info["draws"] += bot2_draws
                bot2_info["losses"] += bot1_total_wins
                bot2_info["matches_played"] += total_matches

                # Swiss scoring: 1 punto por victoria, 0.5 por empate
                bot1_info["score"] += bot1_total_wins + bot1_draws * 0.5
                bot2_info["score"] += bot2_total_wins + bot2_draws * 0.5

            # ===== STANDINGS DESPUES DE LA RONDA =====
            standings = sorted(bots_info, key=lambda x: (-x["score"], -x["wins"], -x["epoch"]))

            if round_num % 10 == 0 or round_num == NUM_ROUNDS:
                logging.info(f"\nTop 10 despues de ronda {round_num}:")
                for rank, bot in enumerate(standings[:10], 1):
                    logging.info(
                        f"{rank:2d}. {bot['name'][:30]:30s} | Score: {bot['score']:6.1f} | "
                        f"W/D/L: {bot['wins']:.0f}/{bot['draws']:.0f}/{bot['losses']:.0f} | "
                        f"Epoch: {bot['epoch']:4d}"
                    )

            # ===== GUARDAR PROGRESO =====
            tournament_state["bots_info"] = bots_info
            tournament_state["win_matrix"] = W

            if SAVE_PROGRESS_EVERY > 0 and round_num % SAVE_PROGRESS_EVERY == 0:
                save_progress(tournament_state, results_path, round_num)

        # ----------------------------- CALCULAR BRADLEY-TERRY --------------------------
        logging.info("=" * 80)
        logging.info("Calculando scores de Bradley-Terry...")
        logging.info("=" * 80)

        # ===== REGULARIZACION DE LA MATRIZ W =====
        # Agregar pequeño epsilon para evitar division por cero
        # Esto ocurre cuando un bot no tiene derrotas (denominador = 0)
        EPSILON = 0.01  # Pequeño valor de regularizacion

        # OPTIMIZADO: Operacion vectorizada (mucho mas rapida que doble loop)
        W_regularized = W + EPSILON  # Suma epsilon a todas las entradas
        # Restar epsilon de la diagonal usando numpy (i vs i debe ser 0)
        np.fill_diagonal(W_regularized.values, W.values.diagonal())

        logging.info(f"Aplicada regularizacion epsilon={EPSILON} a matriz (vectorizada)")

        # Inicializar scores (todos empiezan en 1.0)
        initial_scores = {i: 1.0 for i in range(N)}

        # Calcular Bradley-Terry con matriz regularizada
        logging.info(f"Calculando Bradley-Terry con {N} agentes...")
        logging.info(f"Parametros: max_epochs={BRADLEY_TERRY_EPOCHS}, threshold={BRADLEY_TERRY_THRESHOLD}")
        logging.info("Esto puede tardar ~1-2 minutos con 3000 agentes...")

        bt_scores = calculate_BradleyTerry(
            score=initial_scores,
            W=W_regularized,
            EPOCHS=BRADLEY_TERRY_EPOCHS,
            diff_threshold=BRADLEY_TERRY_THRESHOLD,
            normalize=BRADLEY_TERRY_NORMALIZE,
            verbose=True,  # Muestra progreso
        )

        tournament_state["bradley_terry_scores"] = bt_scores

        # ----------------------------- GUARDAR RESULTADOS FINALES --------------------------
        final_results = save_final_results(tournament_state, results_path)

        # ----------------------------- MOSTRAR RESULTADOS --------------------------
        logging.info("=" * 80)
        logging.info("RESULTADOS FINALES - BRADLEY-TERRY RANKING")
        logging.info("=" * 80)

        for result in final_results[:20]:  # Top 20
            logging.info(
                f"#{result['rank']:2d} | {result['bot_name'][:40]:40s} | "
                f"BT Score: {result['bradley_terry_score']:8.4f} | "
                f"Swiss: {result['swiss_score']:6.1f} | "
                f"Win Rate: {result['win_rate']:5.2%} | "
                f"Epoch: {result['epoch']:4d}"
            )

        if N > 20:
            logging.info("..." + " " * 78 + "...")
            logging.info(
                f"#{final_results[-1]['rank']:2d} | {final_results[-1]['bot_name'][:40]:40s} | "
                f"BT Score: {final_results[-1]['bradley_terry_score']:8.4f} | "
                f"Swiss: {final_results[-1]['swiss_score']:6.1f} | "
                f"Win Rate: {final_results[-1]['win_rate']:5.2%} | "
                f"Epoch: {final_results[-1]['epoch']:4d}"
            )

        # ----------------------------- RESUMEN --------------------------
        logging.info("=" * 80)
        logging.info("TORNEO COMPLETADO EXITOSAMENTE")
        logging.info("=" * 80)
        logging.info(f"Campeon: {final_results[0]['bot_name']}")
        logging.info(f"Bradley-Terry Score: {final_results[0]['bradley_terry_score']:.4f}")
        logging.info(f"Swiss Score: {final_results[0]['swiss_score']:.1f}")
        logging.info(f"Win Rate: {final_results[0]['win_rate']:.2%}")
        logging.info(f"Epoca: {final_results[0]['epoch']}")
        logging.info("=" * 80)

        return tournament_state

    except KeyboardInterrupt:
        logging.warning("=" * 80)
        logging.warning("TORNEO INTERRUMPIDO POR EL USUARIO (Ctrl+C)")

        if SAVE_PROGRESS_EVERY > 0:
            logging.warning("Guardando progreso parcial...")
            save_progress(tournament_state, results_path, round_num)
            logging.warning("Progreso guardado. Puedes reanudar el torneo mas tarde.")
        else:
            logging.warning("No se guardo progreso (SAVE_PROGRESS_EVERY = 0)")

        logging.warning("=" * 80)
        return None

    except Exception as e:
        logging.error("=" * 80)
        logging.error(f"ERROR DURANTE EL TORNEO: {e}")
        logging.error("=" * 80)
        import traceback
        traceback.print_exc()
        return None


# ----------------------------- EJECUCION PRINCIPAL --------------------------
if __name__ == "__main__":
    logging.info("=" * 80)
    logging.info("INICIANDO TORNEO SWISS + BRADLEY-TERRY")
    logging.info("=" * 80)
    logging.info(f"Carpeta de agentes: {AGENTS_FOLDER}")
    logging.info(f"Carpeta de resultados: {RESULTS_FOLDER}")
    logging.info(f"Numero de rondas: {NUM_ROUNDS}")
    logging.info(f"Partidas por emparejamiento: {MATCHES_PER_PAIRING}")
    logging.info(f"Temperatura: {TEMPERATURE}")
    logging.info(f"Deterministico: {DETERMINISTIC}")
    logging.info("=" * 80)

    result = run_swiss_tournament()

    if result is None:
        logging.info("Torneo no completado o interrumpido.")
    else:
        logging.info("Proceso finalizado exitosamente.")

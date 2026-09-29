"""Servicio de recepcion de contragarantias (condiciones de entrega) en rarsys.

Replica la logica de Workflow::ProcesosController#busqueda_condicion_real
(consulta) y #recepcionar_condicion (recepcion) contra la misma base Oracle.

Entidad central: fia_condicion_entrega con tipo_condicion = 3 (contragarantia).
Solo se ESCRIBE en fia_condicion_entrega. fia_poliza se lee para la cabecera.

Notas de implementacion:
- La descripcion del tipo de contragarantia en rarsys sale de la relacion
  ce.tipo_contragarantia (via tipo_garantia_id). Aqui se resuelve con un
  LEFT JOIN a main_fusa.tipo_contragarantia por id (verificado contra la
  base de produccion).
- El correo a gestion documental (CorreoMailer) NO se replica (fuera de alcance).
"""

import logging
from datetime import datetime

import oracledb

from s3_oracle_service import get_connection

logger = logging.getLogger(__name__)

# =========================================================
# CONSTANTES DE NEGOCIO (rarsys)
# =========================================================
TIPO_CONDICION_CONTRAGARANTIA = 3

# Estados de cumplida_sn:
#   A = pendiente/activa
#   P = recepcionada (pendiente de aceptacion juridica)
#   N = rechazada
#   S = completada/aceptada
ESTADO_ACTIVA = "A"
ESTADO_RECEPCIONADA = "P"
ESTADO_COMPLETADA = "S"

ESTADOS_CONSULTA = ("A", "P", "N", "S")

# tipo_garantia_id que NO se muestra ni avanza a 'S'
TIPO_GARANTIA_EXCLUIDO = 10


# =========================================================
# CONSULTA (busqueda_condicion_real)
# =========================================================
def _fetch_condiciones(cursor, filtro_sql, params, order_sql=None, limit=None):
    """Ejecuta la consulta de condiciones de entrega tipo 3.

    filtro_sql debe ser un predicado adicional sobre el alias 'ce'
    (por ejemplo "ce.fianza = :valor").

    order_sql: clausula ORDER BY opcional (sin la palabra ORDER BY). Si no
    se indica, se ordena por ce.id.

    limit: numero maximo de filas a devolver (FETCH FIRST N ROWS ONLY). Si es
    None, se devuelven todas.
    """

    order_clause = order_sql if order_sql else "ce.id"
    limit_clause = ""
    if limit is not None:
        limit_clause = "FETCH FIRST :row_limit ROWS ONLY"

    # La descripcion del tipo de contragarantia se obtiene por LEFT JOIN al
    # catalogo tipo_contragarantia (verificado: tipo_garantia_id enlaza con
    # tipo_contragarantia.id, y la descripcion es la columna DESCRIPCION,
    # equivalente a ce.tipo_contragarantia.descripcion en rarsys).
    # Se usa LEFT JOIN para que, si no hay match, la condicion igual se
    # devuelva con tipo_contragarantia = None.
    query = f"""
        SELECT
            ce.id                      AS condicion_entrega_id,
            ce.fianza                  AS fianza,
            ce.condicion               AS condicion,
            ce.cumplida_sn             AS estado,
            ce.tipo_garantia_id        AS tipo_garantia_id,
            tg.descripcion             AS tipo_contragarantia
        FROM main_fusa.fia_condicion_entrega ce
        LEFT JOIN main_fusa.tipo_contragarantia tg
            ON tg.id = ce.tipo_garantia_id
        WHERE ce.tipo_condicion = :tipo_condicion
          AND ce.cumplida_sn IN ('A', 'P', 'N', 'S')
          AND (ce.tipo_garantia_id IS NULL OR ce.tipo_garantia_id <> :tipo_excluido)
          AND {filtro_sql}
        ORDER BY {order_clause}
        {limit_clause}
    """

    merged = {
        "tipo_condicion": TIPO_CONDICION_CONTRAGARANTIA,
        "tipo_excluido": TIPO_GARANTIA_EXCLUIDO,
    }
    merged.update(params)

    if limit is not None:
        merged["row_limit"] = limit

    cursor.execute(query, merged)

    columns = [item[0].lower() for item in cursor.description]

    return [dict(zip(columns, row)) for row in cursor.fetchall()]


def _fetch_poliza_header(cursor, fianza):
    """Arma la cabecera desde fia_poliza para una fianza dada.

    Devuelve fiado (nombre), codigo_contacto_cliente y monto_afianzado.
    """

    query = """
        SELECT
            p.fianza                                              AS fianza,
            p.cod_contacto_cliente                                AS codigo_contacto_cliente,
            main_fusa.pkg_general.nombre_contacto(
                p.cod_contacto_cliente
            )                                                     AS fiado,
            p.monto_afianzado                                     AS monto_afianzado
        FROM main_fusa.fia_poliza p
        WHERE p.fianza = :fianza
    """

    cursor.execute(query, {"fianza": fianza})

    row = cursor.fetchone()

    if not row:
        return None

    columns = [item[0].lower() for item in cursor.description]

    return dict(zip(columns, row))


def _format_monto(valor):
    """Formatea el monto afianzado como 'Q 1,500,000.00'."""

    if valor is None:
        return None

    try:
        return "Q {:,.2f}".format(float(valor))
    except (TypeError, ValueError):
        return str(valor)


def _build_response(cursor, condiciones):
    """Construye la estructura JSON final a partir de las condiciones."""

    condiciones_out = [
        {
            "condicion_entrega_id": row["condicion_entrega_id"],
            "tipo_contragarantia": row["tipo_contragarantia"],
            "condicion": row["condicion"],
            "estado": row["estado"],
        }
        for row in condiciones
    ]

    # La cabecera se arma con la fianza de la(s) condicion(es). Todas las
    # condiciones consultadas por fianza o por id pertenecen a una sola fianza.
    header = None
    if condiciones:
        fianza = condiciones[0]["fianza"]
        header = _fetch_poliza_header(cursor, fianza)

    if header:
        return {
            "fianza": header["fianza"],
            "fiado": header["fiado"],
            "codigo_contacto_cliente": header["codigo_contacto_cliente"],
            "monto_afianzado": _format_monto(header["monto_afianzado"]),
            "condiciones": condiciones_out,
        }

    # Sin cabecera (sin resultados o sin poliza): responder estructura vacia.
    return {
        "fianza": condiciones[0]["fianza"] if condiciones else None,
        "fiado": None,
        "codigo_contacto_cliente": None,
        "monto_afianzado": None,
        "condiciones": condiciones_out,
    }


def buscar_por_fianza(fianza: int):
    """Todas las condiciones tipo 3 de una fianza."""

    conn = None
    cursor = None
    try:
        conn = get_connection()
        cursor = conn.cursor()

        condiciones = _fetch_condiciones(
            cursor,
            "ce.fianza = :valor",
            {"valor": fianza},
        )

        return _build_response(cursor, condiciones)

    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def buscar_por_condicion(condicion_entrega_id: int):
    """Condicion tipo 3 por su id (normalmente una)."""

    conn = None
    cursor = None
    try:
        conn = get_connection()
        cursor = conn.cursor()

        condiciones = _fetch_condiciones(
            cursor,
            "ce.id = :valor",
            {"valor": condicion_entrega_id},
        )

        return _build_response(cursor, condiciones)

    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def buscar_por_cliente(codigo_cliente: int):
    """Ultimas 5 condiciones tipo 3 de las fianzas de un cliente.

    Un cliente puede acumular cientos de condiciones. Para el modal de
    recepcion solo se devuelven las 5 mas recientes, ordenadas por fecha de
    grabacion descendente (mas reciente primero; id como desempate para
    filas sin fecha).

    La cabecera (fianza/fiado/monto) se arma con la fianza de la primera
    condicion del resultado (la mas reciente). Si el cliente tiene varias
    fianzas con pendientes, considerar consultar por fianza para el detalle
    completo.
    """

    conn = None
    cursor = None
    try:
        conn = get_connection()
        cursor = conn.cursor()

        condiciones = _fetch_condiciones(
            cursor,
            """ce.fianza IN (
                SELECT p.fianza
                FROM main_fusa.fia_poliza p
                WHERE p.cod_contacto_cliente = :valor
            )""",
            {"valor": codigo_cliente},
            order_sql="ce.grabacion_fecha DESC NULLS LAST, ce.id DESC",
            limit=5,
        )

        return _build_response(cursor, condiciones)

    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


# =========================================================
# RECEPCION (recepcionar_condicion)
# =========================================================
def recepcionar_condicion(cursor, condicion_entrega_id: int, usuario_oracle: str):
    """Recepciona una condicion de entrega sobre un cursor compartido.

    Basado en Workflow::ProcesosController#recepcionar_condicion, con una
    diferencia intencional: se permite recepcionar desde CUALQUIER estado
    (A/P/N/S), no solo desde 'A'. Esto cubre el caso de negocio en que una
    contragarantia ya recepcionada se vuelve a traer por error y se necesita
    poder re-recepcionarla.

      - Busca la condicion por id (sin filtrar por estado).
      - Si existe: usuario_pendiente = usuario_oracle, fecha_pendiente = ahora,
        observaciones_pendiente = "Condicion de entrega recepcionada por {usuario}".
      - Regla de avance: si tipo_garantia_id = 10 queda en 'P';
        para cualquier otro tipo pasa a 'S' (aceptacion).

    NO hace commit ni abre conexion: opera sobre el cursor recibido para
    permitir transaccionalidad conjunta con la validacion de firma.

    Lanza ValueError solo si la condicion no existe.
    Devuelve dict con el estado previo y el estado final.
    """

    if not usuario_oracle:
        raise ValueError("usuario_oracle es requerido para recepcionar")

    # Buscar la condicion por id (bloqueando la fila para la transaccion),
    # sin restringir por estado: se permite recepcionar en cualquier estado.
    cursor.execute(
        """
        SELECT id, tipo_garantia_id, cumplida_sn
        FROM main_fusa.fia_condicion_entrega
        WHERE id = :id
        FOR UPDATE
        """,
        {
            "id": condicion_entrega_id,
        },
    )

    row = cursor.fetchone()

    if not row:
        raise ValueError(
            "La condicion de entrega no existe."
        )

    tipo_garantia_id = row[1]
    estado_previo = row[2]

    observaciones = f"Condicion de entrega recepcionada por {usuario_oracle}"
    ahora = datetime.now()

    # Estado final: 'P' para tipo_garantia_id = 10; 'S' para el resto.
    estado_final = (
        ESTADO_RECEPCIONADA
        if tipo_garantia_id == TIPO_GARANTIA_EXCLUIDO
        else ESTADO_COMPLETADA
    )

    cursor.execute(
        """
        UPDATE main_fusa.fia_condicion_entrega
           SET cumplida_sn = :estado_final,
               usuario_pendiente = :usuario,
               fecha_pendiente = :fecha,
               observaciones_pendiente = :observaciones
         WHERE id = :id
        """,
        {
            "estado_final": estado_final,
            "usuario": usuario_oracle,
            "fecha": ahora,
            "observaciones": observaciones,
            "id": condicion_entrega_id,
        },
    )

    if cursor.rowcount != 1:
        raise ValueError(
            "No se pudo recepcionar la condicion de entrega "
            f"{condicion_entrega_id} (afecto un numero inesperado de filas)."
        )

    return {
        "condicion_entrega_id": condicion_entrega_id,
        "cumplida_sn": estado_final,
        "estado_previo": estado_previo,
        "recepcionada": True,
    }

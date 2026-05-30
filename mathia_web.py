from flask import Flask, render_template, request, jsonify, session
from groq import Groq

import os
import io
import base64

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from sympy import *
from sympy import (
    symbols,
    sympify,
    diff,
    integrate,
    limit,
    oo,
    latex
)
# ================================
# APP
# ================================

app = Flask(__name__)
app.secret_key = "mathia_secret"

# ================================
# GROQ
# ================================

GROQ_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_KEY:
    raise Exception("Falta GROQ_API_KEY")

cliente = Groq(api_key=GROQ_KEY)

# ================================
# SESSION
# ================================

def init_session():

    if "memoria_chat" not in session:
        session["memoria_chat"] = []

    if "contador_preguntas" not in session:
        session["contador_preguntas"] = 0

    if "historial_graficas" not in session:
        session["historial_graficas"] = []
    if "ultimo_resultado" not in session:
        session["ultimo_resultado"] = ""  
    if "historial_resultados" not in session:
        session["historial_resultados"] = [] 

# ================================
# IA GROQ MEJORADA
# ================================

def preguntar_ia(prompt):

    init_session()

    mensajes = [
        {
            "role": "system",
            "content": """
Eres MathIA.

Resuelves ejercicios matemáticos PASO A PASO.

Siempre explicas:
- límites
- derivadas
- continuidad
- integrales
- sustitución
- álgebra
- matrices
- cálculo

Usa formato claro y LaTeX.
"""
        }
    ]

    # SOLO GUARDAR POCA MEMORIA
    memoria = session["memoria_chat"][-10:]

    mensajes.extend(memoria)

    mensajes.append({
        "role": "user",
        "content": prompt
    })

    respuesta = cliente.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=mensajes,
        temperature=0.2,
        max_tokens=700
    )

    texto = respuesta.choices[0].message.content

    # GUARDAR SOLO POCO HISTORIAL
    session["memoria_chat"].append({
    "role": "user",
    "content": prompt
})

    session["memoria_chat"].append({
    "role": "assistant",
    "content": texto
})
    session["memoria_chat"] = session["memoria_chat"][-10:]
    return texto
# ================================
# NORMALIZAR FUNCION
# ================================

def normalizar_funcion(texto):

    texto = texto.lower()

    borrar = [
        "grafica",
        "gráfica",
        "plot",
        "dibujar",
        "dibújame"
    ]

    for palabra in borrar:
        texto = texto.replace(palabra, "")

    texto = texto.strip()

    texto = texto.replace("x al cuadrado", "x^2")
    texto = texto.replace("x al cubo", "x^3")

    texto = texto.replace("al cuadrado", "^2")
    texto = texto.replace("al cubo", "^3")

    texto = texto.replace("^", "**")

    texto = texto.replace("²", "**2")
    texto = texto.replace("³", "**3")

    texto = texto.replace("funcion de", "")
    texto = texto.replace("función de", "")
    texto = texto.replace("funcion", "")
    texto = texto.replace("función", "")

    if "=" in texto:
        texto = texto.split("=")[1].strip()
    return texto

# ================================
# GRAFICAS CORREGIDAS
# ================================
def generar_grafica(funcion):

    try:

        print("FUNCION ORIGINAL:", funcion)

        funcion = funcion.lower()

        funcion = funcion.replace("grafica", "")
        funcion = funcion.replace("gráfica", "")
        funcion = funcion.replace("plot", "")
        funcion = funcion.replace("dibujar", "")

        funcion = funcion.strip()

        if "=" in funcion:
            funcion = funcion.split("=")[1].strip()

        funcion = funcion.replace("^", "**")

        from sympy.parsing.sympy_parser import (
            parse_expr,
            standard_transformations,
            implicit_multiplication_application
        )

        transformaciones = (
            standard_transformations +
            (implicit_multiplication_application,)
        )

        x = symbols("x")

        expr = parse_expr(
            funcion,
            transformations=transformaciones
        )

        f = lambdify(x, expr, "numpy")

        xs = np.linspace(-10, 10, 500)

        ys = f(xs)

        ys = np.nan_to_num(
            ys,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        plt.figure(figsize=(7,5))

        plt.plot(xs, ys)

        plt.axhline(0, linewidth=1)
        plt.axvline(0, linewidth=1)

        plt.grid(True)

        plt.title(f"f(x) = {funcion}")

        img = io.BytesIO()

        plt.savefig(
            img,
            format="png",
            bbox_inches="tight"
        )

        plt.close()

        img.seek(0)

        grafica_base64 = base64.b64encode(
            img.getvalue()
        ).decode("utf-8")

        return grafica_base64

    except Exception as e:

        print("ERROR GRAFICA:", e)

        return None
# ================================
# VARIABLE GLOBAL
# ================================

x = symbols("x")

def resolver_derivada(expresion):

    try:

        expresion = expresion.strip()

        expresion = expresion.replace("^", "**")

        from sympy.parsing.sympy_parser import (
            parse_expr,
            standard_transformations,
            implicit_multiplication_application
        )

        transformaciones = (
            standard_transformations +
            (implicit_multiplication_application,)
        )

        expr = parse_expr(
            expresion,
            transformations=transformaciones
        )

        resultado = diff(expr, x)

        session["ultimo_resultado"] = str(resultado)
        session["historial_resultados"].append(str(resultado))

        prompt = f"""
Resuelve paso a paso esta derivada.

Función:

{expresion}

La derivada correcta es:

{resultado}

Explica detalladamente el procedimiento
usando LaTeX y pasos numerados.
"""

        explicacion = preguntar_ia(prompt)

        return explicacion

    except Exception as e:

        print("ERROR DERIVADA:", e)

        return f"Error en derivada: {e}"

# ================================
# INTEGRALES REALES
# ================================

def resolver_integral(expresion):

    try:

        expresion = expresion.strip()

        expresion = expresion.replace("^", "**")

        from sympy.parsing.sympy_parser import (
            parse_expr,
            standard_transformations,
            implicit_multiplication_application
        )

        transformaciones = (
            standard_transformations +
            (implicit_multiplication_application,)
        )

        expr = parse_expr(
            expresion,
            transformations=transformaciones
        )

        resultado = integrate(expr, x)
        session["ultimo_resultado"] = str(resultado)
        session["historial_resultados"].append(str(resultado))

        prompt = f"""
Resuelve paso a paso esta integral.

Integral:

{expresion}

El resultado correcto es:

{resultado}

Explica detalladamente usando LaTeX.
"""

        explicacion = preguntar_ia(prompt)

        return explicacion

    except Exception as e:

        print("ERROR INTEGRAL:", e)

        return f"Error en integral: {e}"

# ================================
# LIMITES REALES
# ================================

def resolver_limite(expresion, valor):

    try:

        expresion = expresion.strip()

        expresion = expresion.replace("^", "**")

        from sympy.parsing.sympy_parser import (
            parse_expr,
            standard_transformations,
            implicit_multiplication_application
        )

        transformaciones = (
            standard_transformations +
            (implicit_multiplication_application,)
        )

        expr = parse_expr(
            expresion,
            transformations=transformaciones
        )

        if valor == "infinito":
            valor_sympy = oo
        else:
            valor_sympy = float(valor)

        resultado = limit(expr, x, valor_sympy)

        session["ultimo_resultado"] = str(resultado)
        session["historial_resultados"].append(str(resultado))

        prompt = f"""
Resuelve paso a paso este límite.

Límite:

{expresion}

El resultado correcto es:

{resultado}

Explica usando propiedades de límites
y LaTeX.
"""

        explicacion = preguntar_ia(prompt)

        return explicacion

    except Exception as e:

        print("ERROR LIMITE:", e)

        return f"Error en límite: {e}"
    
# ================================
# ROUTER INTELIGENTE
# ================================

def router(pregunta):

    texto = pregunta.lower()

    # =========================
    # RESULTADO ANTERIOR
    # =========================

    if "resultado anterior" in texto:

        ultimo = session.get("ultimo_resultado", "")

        if not ultimo:

            return {
                "respuesta": "No hay resultado anterior guardado.",
                "grafica": None
            }

        # DERIVAR RESULTADO ANTERIOR
        if "deriva" in texto or "derivada" in texto:

            return {
                "respuesta": resolver_derivada(ultimo),
                "grafica": None
            }

        # INTEGRAR RESULTADO ANTERIOR
        if "integra" in texto or "integral" in texto:

            return {
                "respuesta": resolver_integral(ultimo),
                "grafica": None
            }

        # GRAFICAR RESULTADO ANTERIOR
        if (
            "grafica" in texto or
            "gráfica" in texto or
            "graficar" in texto or
            "plot" in texto
        ):

            grafica = generar_grafica(ultimo)

            if grafica:

                return {
                    "respuesta": f"Gráfica del resultado anterior: {ultimo}",
                    "grafica": grafica
                }

            return {
                "respuesta": "No pude graficar el resultado anterior.",
                "grafica": None
            }

    # =========================
    # GRAFICAS
    # =========================

    if any(x in texto for x in [

        "grafica",
        "gráfica",
        "graficar",
        "plot",
        "dibujar"

    ]):

        funcion = normalizar_funcion(texto)

        grafica = generar_grafica(funcion)

        if grafica:
           return {
        "respuesta": "Gráfica generada correctamente.",
        "grafica": grafica
    }
        else:
          return {
        "respuesta": "No pude interpretar la función para graficarla.",
        "grafica": None
    }

    # =========================
    # DERIVADAS
    # =========================

    if texto.startswith("derivada de"):

        expr = texto.replace(
            "derivada de",
            ""
        ).strip()

        print("DERIVADA:", expr)

        return {
            "respuesta": resolver_derivada(expr),
            "grafica": None
        }

    elif texto.startswith("deriva"):

        expr = texto.replace(
            "deriva",
            ""
        ).strip()

        print("DERIVADA:", expr)

        return {
            "respuesta": resolver_derivada(expr),
            "grafica": None
        }

    # =========================
    # INTEGRALES
    # =========================

    if "integral" in texto:

        expr = texto

        expr = expr.replace(
            "integral de",
            ""
        )

        expr = expr.replace(
            "integral",
            ""
        )

        expr = expr.strip()

        return {
            "respuesta": resolver_integral(expr),
            "grafica": None
        }

    # =========================
    # LIMITES
    # =========================

    if "limite" in texto or "límite" in texto:

        try:

            expr = texto

            expr = expr.replace(
                "limite de",
                ""
            )

            expr = expr.replace(
                "límite de",
                ""
            )

            expr = expr.strip()

            return {
                "respuesta": resolver_limite(
                    expr,
                    "infinito"
                ),
                "grafica": None
            }

        except Exception as e:

            print("ERROR LIMITE:", e)

            return {
                "respuesta": "Error resolviendo límite.",
                "grafica": None
            }

    # =========================
    # IA NORMAL
    # =========================

    return {
        "respuesta": preguntar_ia(pregunta),
        "grafica": None
    }
# ================================
# HOME
# ================================

@app.route("/")
def home():

    init_session()

    return render_template("index.html")

# ================================
# PREGUNTAR
# ================================

@app.route("/preguntar", methods=["POST"])
def preguntar():

    init_session()

    try:

        data = request.get_json()

        pregunta = data.get("pregunta", "")

        resultado = router(pregunta)

        session["contador_preguntas"] += 1

        return jsonify(resultado)

    except Exception as e:

        print("ERROR GENERAL:", e)

        return jsonify({
            "respuesta": "Error en servidor",
            "grafica": None
        })

# ================================
# DASHBOARD
# ================================

@app.route("/dashboard")
def dashboard():

    init_session()

    return jsonify({

        "preguntas_totales":
        session["contador_preguntas"],

        "graficas_generadas":
        len(session["historial_graficas"]),

        "mensajes_memoria":
        len(session["memoria_chat"])
    })
@app.route("/ultimo")
def ultimo():

    init_session()

    return jsonify({
        "ultimo_resultado": session.get("ultimo_resultado", "")
    })
@app.route("/historial")
def historial():

    init_session()

    return jsonify({
        "historial": session.get("historial_resultados", [])
    })
# ================================
# RUN
# ================================

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 10000))

    app.run(
        host="0.0.0.0",
        port=port
    )

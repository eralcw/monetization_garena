"""EDA de monetização com foco no Brasil e comparação latino-americana.

Instalação: python -m pip install pandas matplotlib seaborn jupyter
Execução na raiz do repositório: python scripts/eda_monetizacao_latam.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import StrMethodFormatter


RAIZ_PROJETO = Path(__file__).resolve().parents[1]
ARQUIVO_DADOS = RAIZ_PROJETO / "data" / "raw" / "archive" / "mobile_game_inapp_purchases.csv"
PASTA_FIGURAS = RAIZ_PROJETO / "reports" / "figures"

GENEROS_ALVO = ["Battle Royale", "Action RPG"]
PAISES_LATAM = {
    "argentina", "bolivia", "brazil", "chile", "colombia", "costa rica",
    "cuba", "dominican republic", "ecuador", "el salvador", "guatemala",
    "haiti", "honduras", "mexico", "nicaragua", "panama", "paraguay",
    "peru", "uruguay", "venezuela",
}
ORDEM_SEGMENTOS = ["Whales", "Dolphins", "Minnows"]
CORES_SEGMENTOS = {
    "Whales": "#B71C1C",
    "Dolphins": "#D92A24",
    "Minnows": "#4A4A4A",
}
CORES_PAISES = {"Brasil": "#D92A24", "México": "#4A4A4A"}


def salvar_figura(nome_arquivo: str) -> None:
    """Salva a figura atual no diretório de relatórios."""
    PASTA_FIGURAS.mkdir(parents=True, exist_ok=True)
    plt.savefig(PASTA_FIGURAS / nome_arquivo, dpi=180, bbox_inches="tight")
    plt.close()


def preparar_dados() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carrega o CSV e retorna os recortes Brasil e LATAM sem Brasil."""
    if not ARQUIVO_DADOS.exists():
        raise FileNotFoundError(f"Arquivo CSV não encontrado: {ARQUIVO_DADOS}")

    dados = pd.read_csv(ARQUIVO_DADOS)
    colunas_obrigatorias = {
        "Country", "GameGenre", "SessionCount", "AverageSessionLength",
        "SpendingSegment", "InAppPurchaseAmount", "FirstPurchaseDaysAfterInstall",
    }
    ausentes = colunas_obrigatorias.difference(dados.columns)
    if ausentes:
        raise ValueError(f"Colunas obrigatórias ausentes: {sorted(ausentes)}")

    # Converte tipos sem imputar valores de gasto ou dias de conversão.
    for coluna in [
        "SessionCount", "AverageSessionLength", "InAppPurchaseAmount",
        "FirstPurchaseDaysAfterInstall",
    ]:
        dados[coluna] = pd.to_numeric(dados[coluna], errors="coerce")
    dados["_pais_normalizado"] = dados["Country"].str.strip().str.casefold()
    dados["_genero_normalizado"] = dados["GameGenre"].str.strip().str.casefold()
    dados["_pais_grafico"] = dados["_pais_normalizado"].map(
        {"brazil": "Brasil", "mexico": "México"}
    )
    dados["_segmento_grafico"] = (
        dados["SpendingSegment"].str.strip().str.casefold().map(
            {
                "whale": "Whales", "whales": "Whales",
                "dolphin": "Dolphins", "dolphins": "Dolphins",
                "minnow": "Minnows", "minnows": "Minnows",
            }
        )
    )

    mascara_genero = dados["_genero_normalizado"].isin(
        {genero.casefold() for genero in GENEROS_ALVO}
    )
    mascara_latam = dados["_pais_normalizado"].isin(PAISES_LATAM)
    recorte_latam = dados[mascara_latam & mascara_genero].copy()
    recorte_brasil = recorte_latam[
        recorte_latam["_pais_normalizado"].eq("brazil")
    ].copy()
    recorte_comparacao = recorte_latam[
        ~recorte_latam["_pais_normalizado"].eq("brazil")
    ].copy()

    if recorte_brasil.empty:
        raise ValueError("Não há registros brasileiros nos gêneros selecionados.")
    if recorte_latam.empty:
        raise ValueError("Não há registros latino-americanos nos gêneros selecionados.")

    return recorte_brasil, recorte_comparacao


def grafico_amostra(recorte_brasil: pd.DataFrame, recorte_comparacao: pd.DataFrame) -> None:
    """Compara apenas os registros dos gêneros-alvo disponíveis na região."""
    recorte = pd.concat([recorte_brasil, recorte_comparacao], ignore_index=True)
    contagens = (
        recorte.groupby(["_pais_grafico", "GameGenre"], observed=False)
        .size().rename("Registros").reset_index()
    )
    plt.figure(figsize=(8, 5))
    sns.barplot(
        data=contagens, x="GameGenre", y="Registros", hue="_pais_grafico",
        order=GENEROS_ALVO, hue_order=["Brasil", "México"],
        palette=CORES_PAISES, errorbar=None,
    )
    plt.title("Amostra latino-americana nos gêneros analisados")
    plt.xlabel("Gênero")
    plt.ylabel("Número de registros")
    plt.legend(title="País")
    salvar_figura("01_amostra_brasil_latam.png")


def grafico_segmentos(recorte_brasil: pd.DataFrame, recorte_comparacao: pd.DataFrame) -> None:
    """Compara participação em usuários e receita por segmento e país."""
    recorte = pd.concat([recorte_brasil, recorte_comparacao], ignore_index=True)
    linhas = []
    for pais, grupo in recorte.groupby("_pais_grafico", observed=True):
        usuarios = grupo.groupby("_segmento_grafico", observed=False)["UserID"].nunique()
        receita = grupo.groupby("_segmento_grafico", observed=False)[
            "InAppPurchaseAmount"
        ].sum(min_count=1)
        total_usuarios = usuarios.sum()
        total_receita = grupo["InAppPurchaseAmount"].sum(min_count=1)

        for segmento in ORDEM_SEGMENTOS:
            numero_usuarios = int(usuarios.get(segmento, 0))
            valor_receita = receita.get(segmento, np.nan)
            if numero_usuarios == 0:
                valor_receita = 0.0
            linhas.extend([
                {"País": pais, "Segmento": segmento, "Métrica": "% dos usuários",
                 "Percentual": numero_usuarios / total_usuarios * 100 if total_usuarios else 0},
                {"País": pais, "Segmento": segmento, "Métrica": "% da receita",
                 "Percentual": valor_receita / total_receita * 100
                 if pd.notna(total_receita) and total_receita else 0},
            ])

    metricas = pd.DataFrame(linhas)
    fig, eixos = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for eixo, pais in zip(eixos, ["Brasil", "México"]):
        dados_pais = metricas[metricas["País"].eq(pais)]
        if dados_pais.empty:
            eixo.set_visible(False)
            continue
        sns.barplot(
            data=dados_pais, x="Segmento", y="Percentual", hue="Métrica",
            order=ORDEM_SEGMENTOS, palette=["#D92A24", "#4A4A4A"],
            errorbar=None, ax=eixo,
        )
        eixo.set_title(pais)
        eixo.set_xlabel("Segmento")
        eixo.set_ylabel("Participação (%)")
        eixo.legend(title="Métrica")

    fig.suptitle("Participação de usuários versus receita por segmento")
    fig.text(
        0.5, 0.01,
        "A receita usa InAppPurchaseAmount em USD; valores ausentes não foram imputados.",
        ha="center", color="#4A4A4A",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.93))
    salvar_figura("02_segmentos_usuarios_receita_usd.png")


def grafico_conversao(recorte_brasil: pd.DataFrame, recorte_comparacao: pd.DataFrame) -> None:
    """Mostra a distribuição observada de dias até a primeira compra."""
    recorte = pd.concat([recorte_brasil, recorte_comparacao], ignore_index=True)
    conversoes = recorte.dropna(subset=["FirstPurchaseDaysAfterInstall"]).copy()
    conversoes = conversoes[conversoes["FirstPurchaseDaysAfterInstall"] >= 0]
    if conversoes.empty:
        print("Sem dias válidos de primeira compra; figura de conversão não gerada.")
        return

    limite = int(np.ceil(conversoes["FirstPurchaseDaysAfterInstall"].max()))
    bins = np.arange(-0.5, limite + 1.5, 1)
    fig, eixo = plt.subplots(figsize=(10, 5))
    sns.histplot(
        data=conversoes, x="FirstPurchaseDaysAfterInstall", hue="_pais_grafico",
        hue_order=["Brasil", "México"], palette=CORES_PAISES,
        bins=bins, multiple="dodge", shrink=0.85, ax=eixo,
    )
    mediana_brasil = recorte_brasil["FirstPurchaseDaysAfterInstall"].median()
    if pd.notna(mediana_brasil):
        eixo.axvline(
            mediana_brasil, color="#B71C1C", linestyle="--",
            label=f"Mediana Brasil: {mediana_brasil:.0f} dias",
        )
        eixo.legend(title="País / referência", fontsize=9)
    eixo.set_title("Dias após instalação até a primeira compra")
    eixo.set_xlabel("Dias após a instalação")
    eixo.set_ylabel("Registros com dia de compra conhecido")
    salvar_figura("03_janela_primeira_compra.png")


def grafico_engajamento(recorte_brasil: pd.DataFrame, recorte_comparacao: pd.DataFrame) -> dict:
    """Relaciona sessões e duração média com gasto em USD."""
    recorte = pd.concat([recorte_brasil, recorte_comparacao], ignore_index=True)
    campos = ["SessionCount", "AverageSessionLength"]
    titulos = ["Número de sessões", "Duração média da sessão"]
    correlacoes = {}
    fig, eixos = plt.subplots(1, 2, figsize=(13, 5))

    for eixo, campo, titulo in zip(eixos, campos, titulos):
        pares = recorte.dropna(subset=[campo, "InAppPurchaseAmount"])
        correlacoes[campo] = pares[campo].rank().corr(
            pares["InAppPurchaseAmount"].rank(), method="pearson"
        )
        sns.scatterplot(
            data=pares, x=campo, y="InAppPurchaseAmount", hue="_pais_grafico",
            hue_order=["Brasil", "México"], palette=CORES_PAISES,
            alpha=0.75, ax=eixo,
        )
        if not pares.empty:
            teto = pares["InAppPurchaseAmount"].quantile(0.99)
            if pd.notna(teto) and teto > 0:
                eixo.set_ylim(0, teto)
        rho = correlacoes[campo]
        eixo.set_title(f"{titulo} | Spearman: {rho:.2f}" if pd.notna(rho) else titulo)
        eixo.set_xlabel(titulo)
        eixo.set_ylabel("Gasto registrado (USD; eixo até P99)")
        eixo.yaxis.set_major_formatter(StrMethodFormatter("${x:,.0f}"))
        eixo.legend(title="País")

    fig.suptitle("Engajamento versus gasto nos gêneros selecionados")
    fig.tight_layout()
    salvar_figura("04_engajamento_vs_gasto_usd.png")
    return correlacoes


def imprimir_insights(recorte_brasil: pd.DataFrame, correlacoes: dict) -> None:
    """Imprime insights baseados na amostra brasileira observada."""
    dias = recorte_brasil["FirstPurchaseDaysAfterInstall"].dropna()
    dias = dias[dias >= 0]
    receita_segmento = recorte_brasil.groupby("_segmento_grafico", observed=True)[
        "InAppPurchaseAmount"
    ].sum(min_count=1)
    receita_total = recorte_brasil["InAppPurchaseAmount"].sum(min_count=1)
    print("\nINSIGHTS PARA O CASE — BRASIL")
    print(
        f"• A amostra brasileira tem {len(recorte_brasil)} registros em Battle Royale "
        f"e Action RPG. Trate os resultados como exploratórios, não como estimativas "
        f"definitivas do mercado."
    )

    whales = recorte_brasil[recorte_brasil["_segmento_grafico"].eq("Whales")]
    if whales.empty:
        print(
            "• Não há Whales observados neste recorte do Brasil. Não use esta amostra "
            "para justificar preço de UR/armas evolutivas; teste ofertas caras com "
            "holdout e monitore a resposta de Dolphins."
        )
    elif pd.notna(receita_total) and receita_total:
        participacao = receita_segmento.get("Whales", 0) / receita_total * 100
        print(
            f"• Whales representam {len(whales)} registros e {participacao:.1f}% da "
            "receita observada no Brasil. Teste itens UR/armas evolutivas em ofertas "
            "segmentadas, mantendo itens de entrada para os demais jogadores."
        )

    if not dias.empty:
        print(
            f"• A mediana da primeira compra é {dias.median():.0f} dias "
            f"(média {dias.mean():.1f}; n={len(dias)}). Teste itens baratos Tier R/SR "
            "e bônus de recarga antes e durante essa janela, com grupo de controle "
            "para medir conversão incremental."
        )
    else:
        print("• Não há dias válidos de primeira compra no recorte brasileiro.")

    for campo, nome in [
        ("SessionCount", "número de sessões"),
        ("AverageSessionLength", "duração média da sessão"),
    ]:
        rho = correlacoes.get(campo, np.nan)
        leitura = "positiva" if rho > 0 else "negativa" if rho < 0 else "próxima de zero"
        if pd.notna(rho):
            print(
                f"• A associação entre {nome} e gasto é {leitura} "
                f"(Spearman {rho:.2f}) no recorte regional. Isso não demonstra "
                "causalidade; use como hipótese de segmentação e valide em teste."
            )


def main() -> None:
    sns.set_theme(style="whitegrid")
    plt.rcParams["axes.prop_cycle"] = plt.cycler(color=["#D92A24", "#B71C1C", "#4A4A4A"])

    recorte_brasil, recorte_comparacao = preparar_dados()
    PASTA_FIGURAS.mkdir(parents=True, exist_ok=True)

    print(f"Brasil — registros nos gêneros-alvo: {len(recorte_brasil)}")
    print(f"LATAM fora do Brasil — registros nos gêneros-alvo: {len(recorte_comparacao)}")
    print("Países latino-americanos encontrados:")
    print(
        pd.concat([recorte_brasil, recorte_comparacao])["Country"]
        .value_counts().to_string()
    )
    print("\nNulos no recorte latino-americano:")
    recorte_latam = pd.concat([recorte_brasil, recorte_comparacao])
    print(recorte_latam.isna().mean().mul(100).round(2).to_string())

    grafico_amostra(recorte_brasil, recorte_comparacao)
    grafico_segmentos(recorte_brasil, recorte_comparacao)
    grafico_conversao(recorte_brasil, recorte_comparacao)
    correlacoes = grafico_engajamento(recorte_brasil, recorte_comparacao)
    imprimir_insights(recorte_brasil, correlacoes)

    print(f"\nFiguras salvas em: {PASTA_FIGURAS}")
    for figura in sorted(PASTA_FIGURAS.glob("*.png")):
        print(f"- {figura.name}")


if __name__ == "__main__":
    main()
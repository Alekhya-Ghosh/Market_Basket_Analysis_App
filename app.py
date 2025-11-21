import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import io
import warnings

warnings.filterwarnings("ignore")

try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except:
    PLOTLY_AVAILABLE = False

try:
    from mlxtend.frequent_patterns import apriori, association_rules
    MLXTEND_AVAILABLE = True
except:
    MLXTEND_AVAILABLE = False

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except:
    NETWORKX_AVAILABLE = False

try:
    import seaborn as sns
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except:
    MATPLOTLIB_AVAILABLE = False

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    REPORTLAB_AVAILABLE = True
except:
    REPORTLAB_AVAILABLE = False

st.set_page_config(page_title="Market Basket Analysis", page_icon="📊", layout="wide")

st.markdown("""
<style>
body { background-color: #F5F7F9; }
[data-testid="stSidebar"] { background-color: #004F4F !important; color: white !important; }
h1,h2,h3,h4,h5 { color: #008080 !important; font-weight:700 !important; }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def read_csv(file):
    return pd.read_csv(file)


def preprocess_binary_df(df_raw):
    df = df_raw.copy()
    df = df.rename(columns=lambda x: str(x).strip())
    df = df.applymap(lambda x: 1 if str(x).strip().lower() in ['1','true','yes','y']
                     else 0 if str(x).strip().lower() in ['0','false','no','n']
                     else x)
    df = df.apply(pd.to_numeric, errors='coerce').fillna(0)
    return (df > 0).astype(int)


def validate_dataset(df_raw):
    info = {}
    info['rows'] = df_raw.shape[0]
    info['cols'] = df_raw.shape[1]
    df_bin = preprocess_binary_df(df_raw)
    total_ones = int(df_bin.sum().sum())
    sparsity = 1 - (total_ones / (df_bin.shape[0] * df_bin.shape[1])) if df_bin.size > 0 else 1
    info['sparsity'] = sparsity
    info['duplicate_transactions'] = df_bin.duplicated().sum()
    info['avg_basket_size'] = df_bin.sum(axis=1).mean()
    return info


@st.cache_data
def run_apriori_and_format(df, min_support, min_confidence):
    if not MLXTEND_AVAILABLE:
        return None, None, None, "mlxtend not installed"

    try:
        freq = apriori(df, min_support=min_support, use_colnames=True)
        if freq.empty:
            return freq, None, None, None

        rules = association_rules(freq, metric="confidence", min_threshold=min_confidence)
        if rules.empty:
            return freq, rules, None, None

        rules_disp = rules.copy()

        def fs_to_str(x):
            return ", ".join(sorted(list(x))) if not isinstance(x, str) else x

        rules_disp["antecedents"] = rules_disp["antecedents"].apply(fs_to_str)
        rules_disp["consequents"] = rules_disp["consequents"].apply(fs_to_str)

        freq_disp = freq.copy()
        freq_disp["itemsets"] = freq_disp["itemsets"].apply(fs_to_str)

        return freq, rules, rules_disp, None

    except Exception as e:
        return None, None, None, str(e)


def plot_heatmap(df):
    if not MATPLOTLIB_AVAILABLE:
        return None
    co = df.T.dot(df)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(co, cmap="Blues", ax=ax)
    return fig


def plot_scatter(rules_disp):
    if not PLOTLY_AVAILABLE:
        return None
    return px.scatter(
        rules_disp, x="support", y="confidence", size="lift", color="lift",
        hover_data=["antecedents", "consequents"],
        title="Support vs Confidence"
    )


def plot_network(rules_disp, top_n=30):
    if not (NETWORKX_AVAILABLE and PLOTLY_AVAILABLE):
        return None

    rules_sorted = rules_disp.sort_values("lift", ascending=False).head(top_n)

    G = nx.DiGraph()

    for _, r in rules_sorted.iterrows():
        ants = r["antecedents"].split(",")
        cons = r["consequents"].split(",")
        for a in ants:
            a = a.strip()
            for c in cons:
                c = c.strip()
                G.add_node(a)
                G.add_node(c)
                G.add_edge(a, c)

    pos = nx.spring_layout(G, seed=42)
    edge_x, edge_y = [], []

    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    node_x = [pos[n][0] for n in G.nodes()]
    node_y = [pos[n][1] for n in G.nodes()]
    node_text = list(G.nodes())

    edge_trace = go.Scatter(x=edge_x, y=edge_y, mode="lines", line=dict(color="#888", width=1))
    node_trace = go.Scatter(x=node_x, y=node_y, mode="markers+text", text=node_text,
                            textposition="bottom center", marker=dict(size=18))

    return go.Figure(data=[edge_trace, node_trace])


def top_n_recommendations(rules_disp, product, top_n=5):
    product = product.lower()
    matches = []

    for _, r in rules_disp.iterrows():
        ants = [a.strip().lower() for a in r["antecedents"].split(",")]
        if product in ants:
            matches.append(r)

    if not matches:
        return []

    df = pd.DataFrame(matches).sort_values(["confidence", "lift"], ascending=False)
    return df.head(top_n)


def predict_missing_items(rules_disp, partial_items, top_n=5):
    known = set([p.strip().lower() for p in partial_items])
    scored = {}

    for _, r in rules_disp.iterrows():
        ants = set([a.strip().lower() for a in r["antecedents"].split(",")])
        cons = set([c.strip().lower() for c in r["consequents"].split(",")])

        if ants.issubset(known):
            for item in cons:
                if item not in known:
                    score = r["confidence"] * r["lift"]
                    scored[item] = scored.get(item, 0) + score

    scored_sorted = sorted(scored.items(), key=lambda x: x[1], reverse=True)
    return scored_sorted[:top_n]


def main():
    st.title("📊 Market Basket Analysis")

    uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    min_support = st.sidebar.slider("Minimum Support", 0.01, 0.3, 0.05)
    min_confidence = st.sidebar.slider("Minimum Confidence", 0.1, 0.95, 0.5)
    max_rules = st.sidebar.slider("Max Rules to Display", 5, 100, 20)

    if uploaded:
        df_raw = read_csv(uploaded)
    else:
        df_raw = pd.DataFrame({
            "Bread": [1, 1, 0, 1],
            "Milk": [1, 0, 1, 1],
            "Eggs": [0, 1, 1, 0]
        })

    df = preprocess_binary_df(df_raw)
    validation = validate_dataset(df_raw)

    st.metric("Transactions", validation["rows"])
    st.metric("Products", validation["cols"])
    st.metric("Avg Basket Size", round(validation["avg_basket_size"], 2))

    st.dataframe(df.head(), use_container_width=True)

    if MATPLOTLIB_AVAILABLE:
        st.pyplot(plot_heatmap(df))

    if st.button("Run Market Basket Analysis"):
        freq, rules, rules_disp, error = run_apriori_and_format(df, min_support, min_confidence)

        if error:
            st.error(error)
            return

        if rules_disp is None or rules_disp.empty:
            st.warning("No rules found.")
            return

        st.dataframe(rules_disp.head(max_rules))

        if PLOTLY_AVAILABLE:
            st.plotly_chart(plot_scatter(rules_disp))

        if NETWORKX_AVAILABLE:
            st.plotly_chart(plot_network(rules_disp))

        product_choice = st.selectbox("Select Product for Recommendations", df.columns)

        if st.button("Recommend Products"):
            recs = top_n_recommendations(rules_disp, product_choice)
            st.write(recs)

        partial = st.text_input("Enter known items (comma separated)")
        if st.button("Predict Missing Items"):
            items = [i.strip() for i in partial.split(",")]
            predictions = predict_missing_items(rules_disp, items)
            st.write(predictions)


if __name__ == "__main__":
    main()

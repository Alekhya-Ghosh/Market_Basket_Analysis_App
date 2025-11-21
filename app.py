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
def generate_sample_dataset():
    np.random.seed(42)

    products = [
        "Bread", "Milk", "Eggs", "Butter", "Cheese",
        "Apples", "Bananas", "Tomatoes", "Onions", "Chicken",
        "Rice", "Pasta", "Cereal", "Juice", "Coffee"
    ]

    rows = 500
    data = {}

    for p in products:
        prob = np.random.uniform(0.05, 0.35)
        data[p] = np.random.choice([0,1], size=rows, p=[1-prob, prob])

    return pd.DataFrame(data)


@st.cache_data
def preprocess_binary_df(df_raw):
    df = df_raw.copy()
    df = df.rename(columns=lambda x: str(x).strip())
    df = df.applymap(lambda x: 1 if str(x).strip().lower() in ['1','true','yes','y']
                     else 0 if str(x).strip().lower() in ['0','false','no','n']
                     else x)
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0)
    return (df > 0).astype(int)


@st.cache_data
def run_apriori(df, min_support, min_conf):
    freq = apriori(df, min_support=min_support, use_colnames=True)
    if freq.empty:
        return None, None, None
    rules = association_rules(freq, metric="confidence", min_threshold=min_conf)
    if rules.empty:
        return freq, rules, None
    rules_disp = rules.copy()
    rules_disp["antecedents"] = rules_disp["antecedents"].apply(lambda x: ", ".join(sorted(list(x))))
    rules_disp["consequents"] = rules_disp["consequents"].apply(lambda x: ", ".join(sorted(list(x))))
    return freq, rules, rules_disp


def plot_heatmap(df):
    co = df.T.dot(df)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(co, cmap="Blues", ax=ax)
    return fig


def plot_scatter(rules_disp):
    return px.scatter(
        rules_disp, x="support", y="confidence", size="lift", color="lift",
        hover_data=["antecedents", "consequents"], title="Support vs Confidence"
    )


def plot_network(rules_disp, top_n=30):
    rules_sorted = rules_disp.sort_values("lift", ascending=False).head(top_n)

    G = nx.DiGraph()
    for _, r in rules_sorted.iterrows():
        ants = r["antecedents"].split(",")
        cons = r["consequents"].split(",")
        for a in ants:
            for c in cons:
                G.add_edge(a.strip(), c.strip())

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


def top_recommend(rules_disp, product):
    product = product.lower()
    matches = []

    for _, r in rules_disp.iterrows():
        ants = [a.strip().lower() for a in r["antecedents"].split(",")]
        if product in ants:
            matches.append(r)

    if not matches:
        return []

    df = pd.DataFrame(matches).sort_values(["confidence", "lift"], ascending=False)
    return df.head(5)


def predict_missing(rules_disp, basket):
    known = set([b.strip().lower() for b in basket])
    scored = {}

    for _, r in rules_disp.iterrows():
        ants = set([a.strip().lower() for a in r["antecedents"].split(",")])
        cons = set([c.strip().lower() for c in r["consequents"].split(",")])

        if ants.issubset(known):
            for c in cons:
                if c not in known:
                    score = r["confidence"] * r["lift"]
                    scored[c] = scored.get(c, 0) + score

    return sorted(scored.items(), key=lambda x: x[1], reverse=True)[:5]


def main():
    st.title("📊 Market Basket Analysis")

    uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    min_support = st.sidebar.slider("Minimum Support", 0.01, 0.3, 0.05)
    min_conf = st.sidebar.slider("Minimum Confidence", 0.1, 0.95, 0.5)

    if uploaded:
        df_raw = pd.read_csv(uploaded)
    else:
        df_raw = generate_sample_dataset()


    df = preprocess_binary_df(df_raw)

    st.subheader("Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)

    if MATPLOTLIB_AVAILABLE:
        st.subheader("Co-occurrence Heatmap")
        st.pyplot(plot_heatmap(df))

    if st.button("Run Apriori Analysis"):
        freq, rules, rules_disp = run_apriori(df, min_support, min_conf)

        if rules_disp is None or rules_disp.empty:
            st.warning("No rules discovered — try lowering support or confidence.")
            return

        st.subheader("Association Rules (Top Results)")
        st.dataframe(rules_disp.head(20), use_container_width=True)

        if PLOTLY_AVAILABLE:
            st.subheader("Support vs Confidence")
            st.plotly_chart(plot_scatter(rules_disp))

        if NETWORKX_AVAILABLE:
            st.subheader("Rule Network Graph")
            st.plotly_chart(plot_network(rules_disp))

        product = st.selectbox("Select Product for Recommendations", df.columns)
        if st.button("Recommend Products"):
            st.write(top_recommend(rules_disp, product))

        partial = st.text_input("Enter known basket items (comma-separated)")
        if st.button("Predict Missing Items"):
            items = [i.strip() for i in partial.split(",")]
            st.write(predict_missing(rules_disp, items))


if __name__ == "__main__":
    main()

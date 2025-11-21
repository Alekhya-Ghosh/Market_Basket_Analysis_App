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
except Exception:
    PLOTLY_AVAILABLE = False

try:
    from mlxtend.frequent_patterns import apriori, association_rules
    MLXTEND_AVAILABLE = True
except Exception:
    MLXTEND_AVAILABLE = False

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except Exception:
    NETWORKX_AVAILABLE = False

try:
    import seaborn as sns
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except Exception:
    MATPLOTLIB_AVAILABLE = False

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    REPORTLAB_AVAILABLE = True
except Exception:
    REPORTLAB_AVAILABLE = False

st.set_page_config(page_title="Market Basket Intelligence", page_icon="📊", layout="wide")
st.markdown("""
<style>
body { background-color: #F5F7F9; }
[data-testid="stSidebar"] { background-color: #004F4F !important; color: white !important; }
h1,h2,h3,h4,h5 { color: #008080 !important; font-weight:700 !important; }
.insight-box { background-color:#e8f4fd; padding:1rem; border-left:4px solid #ff6b6b; border-radius:8px; }
.success-box { background-color:#d4edda; padding:0.8rem; border-left:4px solid #28a745; border-radius:8px; }
.warning-box { background-color:#fff3cd; padding:0.8rem; border-left:4px solid #ffc107; border-radius:8px; }
.error-box { background-color:#f8d7da; padding:0.8rem; border-left:4px solid #dc3545; border-radius:8px; }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def read_csv(file):
    return pd.read_csv(file)

def preprocess_binary_df(df_raw):
    df = df_raw.copy()
    df = df.rename(columns=lambda x: str(x).strip())
    df = df.applymap(lambda x: 1 if str(x).strip().lower() in ['1','true','yes','y'] else 0 if str(x).strip().lower() in ['0','false','no','n'] else x)
    df = df.apply(pd.to_numeric, errors='coerce').fillna(0)
    return (df > 0).astype(int)

def validate_dataset(df_raw):
    info = {}
    info['rows'] = df_raw.shape[0]
    info['cols'] = df_raw.shape[1]
    blank_cols = [c for c in df_raw.columns if str(c).strip() == ""]
    info['blank_columns'] = blank_cols
    unique_vals = {c: df_raw[c].dropna().unique().tolist()[:10] for c in df_raw.columns}
    info['unique_sample'] = unique_vals
    column_checks = {}
    for c in df_raw.columns:
        series = df_raw[c].astype(str).str.strip().str.lower()
        valid = series.isin(['0','1','true','false','yes','no','y','n'])
        column_checks[c] = {'valid_pct': round(valid.mean()*100,2)}
    info['column_checks'] = column_checks
    df_bin = preprocess_binary_df(df_raw)
    total_ones = int(df_bin.sum().sum())
    sparsity = 1 - (total_ones / (df_bin.shape[0] * df_bin.shape[1])) if df_bin.shape[0]*df_bin.shape[1] > 0 else 1
    info['sparsity'] = round(sparsity, 4)
    info['duplicate_transactions'] = int(df_bin.duplicated().sum())
    info['empty_transactions'] = int((df_bin.sum(axis=1) == 0).sum())
    info['max_basket_size'] = int(df_bin.sum(axis=1).max())
    info['avg_basket_size'] = round(float(df_bin.sum(axis=1).mean()), 2)
    return info

@st.cache_data
def run_apriori_and_format(df, min_support, min_confidence):
    if not MLXTEND_AVAILABLE:
        return pd.DataFrame(), pd.DataFrame(), "mlxtend not installed"
    try:
        freq = apriori(df, min_support=min_support, use_colnames=True, low_memory=True)
        if freq.empty:
            return freq, pd.DataFrame(), None
        rules = association_rules(freq, metric="confidence", min_threshold=min_confidence)
        if rules.empty:
            return freq, pd.DataFrame(), None
        rules_disp = rules.copy()
        def fs(x):
            return ', '.join(sorted(list(x))) if hasattr(x, '__iter__') and not isinstance(x, str) else str(x)
        rules_disp['antecedents'] = rules_disp['antecedents'].apply(fs)
        rules_disp['consequents'] = rules_disp['consequents'].apply(fs)
        freq_disp = freq.copy()
        freq_disp['itemsets'] = freq_disp['itemsets'].apply(fs)
        rules_disp['impact_score'] = rules_disp['support'] * rules_disp['lift']
        for c in ['support','confidence','lift','leverage','conviction','impact_score']:
            if c in rules_disp.columns:
                rules_disp[c] = rules_disp[c].round(6)
        return freq_disp, rules_disp, None
    except Exception as e:
        return pd.DataFrame(), pd.DataFrame(), str(e)

def plot_cooccurrence_heatmap(df):
    if not MATPLOTLIB_AVAILABLE:
        return None
    co = df.T.dot(df)
    fig, ax = plt.subplots(figsize=(10, 9))
    sns.heatmap(co, cmap="Blues", linewidths=0.2, ax=ax)
    plt.tight_layout()
    return fig

def plot_rules_scatter(rules_disp):
    if rules_disp.empty or not PLOTLY_AVAILABLE:
        return None
    return px.scatter(rules_disp, x='support', y='confidence', size='lift', color='lift',
                      hover_data=['antecedents','consequents'],
                      labels={'lift':'Lift'},
                      title="Association Rules: Support vs Confidence")

def plot_network_rules(rules_disp, top_n=30):
    if rules_disp.empty or not (PLOTLY_AVAILABLE and NETWORKX_AVAILABLE):
        return None
    rsmall = rules_disp.sort_values(['lift','support'], ascending=[False, False]).head(top_n)
    G = nx.DiGraph()
    for _, row in rsmall.iterrows():
        ants = [a.strip() for a in row['antecedents'].split(',') if a.strip()]
        cons = [c.strip() for c in row['consequents'].split(',') if c.strip()]
        for a in ants:
            G.add_node(a)
            for b in cons:
                G.add_node(b)
                G.add_edge(a, b, weight=float(row['lift']))
    pos = nx.spring_layout(G, seed=42, k=0.7)
    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
    node_x = [pos[n][0] for n in G.nodes()]
    node_y = [pos[n][1] for n in G.nodes()]
    node_text = list(G.nodes())
    edge_trace = go.Scatter(x=edge_x, y=edge_y, mode='lines', line=dict(width=1, color='#888'))
    node_trace = go.Scatter(x=node_x, y=node_y, mode='markers+text', text=node_text,
                             textposition="bottom center", marker=dict(size=18))
    return go.Figure(data=[edge_trace, node_trace], layout=go.Layout(title="Association Rules Network"))

def top_n_recommendations(rules_disp, product, top_n=5):
    candidates = []
    for _, r in rules_disp.iterrows():
        ants = [a.strip().lower() for a in r['antecedents'].split(',')]
        if product.lower() in ants:
            candidates.append(r)
    if not candidates:
        return []
    df = pd.DataFrame(candidates).sort_values(['confidence','lift'], ascending=[False, False])
    return [{'consequent': r['consequents'], 'confidence': float(r['confidence']),
             'lift': float(r['lift']), 'support': float(r['support'])} for _, r in df.head(top_n).iterrows()]

def predict_missing_items(rules_disp, partial_items, top_n=5):
    known = set([p.strip().lower() for p in partial_items])
    scored = {}
    for _, r in rules_disp.iterrows():
        ants = set([a.strip().lower() for a in r['antecedents'].split(',')])
        cons = set([c.strip().lower() for c in r['consequents'].split(',')])
        if ants and ants.issubset(known):
            for c in cons:
                if c not in known:
                    scored[c] = scored.get(c, 0) + (float(r['confidence']) * float(r['lift']))
    results = sorted(scored.items(), key=lambda x: x[1], reverse=True)
    return [{'product': k, 'score': v} for k, v in results[:top_n]]

def autotune_parameters(df, supports=[0.01,0.02,0.05,0.1], confidences=[0.3,0.5,0.7], top_k=3):
    if not MLXTEND_AVAILABLE:
        return pd.DataFrame()
    rows = []
    for s in supports:
        try:
            freq = apriori(df, min_support=s, use_colnames=True, low_memory=True)
        except Exception:
            freq = pd.DataFrame()
        for c in confidences:
            if freq.empty:
                rows.append({'support': s, 'confidence': c, 'n_rules': 0, 'avg_lift': 0})
                continue
            try:
                rules = association_rules(freq, metric="confidence", min_threshold=c)
                n = len(rules)
                avg = float(rules['lift'].mean()) if n > 0 else 0
                rows.append({'support': s, 'confidence': c, 'n_rules': n, 'avg_lift': avg})
            except:
                rows.append({'support': s, 'confidence': c, 'n_rules': 0, 'avg_lift': 0})
    return pd.DataFrame(rows).sort_values(['n_rules','avg_lift'], ascending=[False, False]).head(top_k)

def create_pdf(buffer, summary_text, top_products_df, rules_df):
    if not REPORTLAB_AVAILABLE:
        return None
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    y = height - 40
    c.setFont("Helvetica-Bold", 14)
    c.drawString(40, y, "Market Basket Analysis Report")
    y -= 24
    c.setFont("Helvetica", 10)
    c.drawString(40, y, summary_text)
    y -= 20
    c.setFont("Helvetica-Bold", 11)
    c.drawString(40, y, "Top Products:")
    y -= 14
    c.setFont("Helvetica", 9)
    for _, r in top_products_df.head(10).iterrows():
        c.drawString(40, y, f"{r['Product']}: {r['Purchase Count']} ({r['Purchase Rate (%)']}%)")
        y -= 11
        if y < 80:
            c.showPage()
            y = height - 40
    y -= 20
    c.setFont("Helvetica-Bold", 11)
    c.drawString(40, y, "Sample Rules:")
    y -= 14
    c.setFont("Helvetica", 9)
    for _, r in rules_df.head(10).iterrows():
        c.drawString(40, y, f"{r['antecedents']} -> {r['consequents']} (lift: {r['lift']})")
        y -= 11
        if y < 80:
            c.showPage()
            y = height - 40
    c.save()
    buffer.seek(0)
    return buffer

def main():
    st.title("📊 Market Basket Intelligence")
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=['csv'])
    min_support = st.sidebar.slider("Min support", 0.01, 0.3, 0.05, 0.01)
    min_confidence = st.sidebar.slider("Min confidence", 0.1, 0.95, 0.5, 0.05)
    max_rules_display = st.sidebar.slider("Max rules", 5, 100, 20)
    if st.sidebar.button("Auto-tune"):
        st.session_state['autotune'] = True

    if uploaded_file is None:
        df_raw = pd.DataFrame({
            'Dairy_Milk':[1,0,1,0,1,1,0,1,0,1],
            'Whole_Wheat_Bread':[1,1,0,1,1,0,1,0,1,1],
            'Fresh_Eggs':[0,1,1,1,0,1,1,0,1,0],
            'Organic_Butter':[0,1,0,1,1,0,1,1,0,1],
            'Artisan_Cheese':[1,0,1,0,1,1,0,1,1,0],
            'Greek_Yogurt':[1,1,1,0,0,1,0,1,0,1],
            'Orange_Juice':[0,1,0,1,1,0,1,0,1,0]
        })
    else:
        df_raw = read_csv(uploaded_file)

    df = preprocess_binary_df(df_raw)
    validation = validate_dataset(df_raw)

    st.metric("Transactions", validation['rows'])
    st.metric("Products", validation['cols'])

    with st.expander("Validation Details"):
        st.write(pd.DataFrame(validation['column_checks']).T)

    st.dataframe(df.head(), use_container_width=True)

    product_sums = df.sum().sort_values(ascending=False)
    product_stats = pd.DataFrame({
        'Product': product_sums.index,
        'Purchase Count': product_sums.values,
        'Purchase Rate (%)': (product_sums.values / len(df) * 100).round(1)
    })

    st.dataframe(product_stats.head(15), use_container_width=True)

    if MATPLOTLIB_AVAILABLE:
        st.pyplot(plot_cooccurrence_heatmap(df))

    if st.button("Run Analysis") or st.session_state.get('autotune'):
        if st.session_state.get('autotune'):
            tuning = autotune_parameters(df)
            st.dataframe(tuning)
            if not tuning.empty:
                best = tuning.iloc[0]
                min_support, min_confidence = best['support'], best['confidence']
            st.session_state['autotune'] = False

        freq_disp, rules_disp, error = run_apriori_and_format(df, min_support, min_confidence)

        if error:
            st.error(error)
        else:
            st.dataframe(rules_disp.head(max_rules_display))

            if PLOTLY_AVAILABLE:
                st.plotly_chart(plot_rules_scatter(rules_disp), use_container_width=True)

            if NETWORKX_AVAILABLE:
                st.plotly_chart(plot_network_rules(rules_disp), use_container_width=True)

            product_choice = st.selectbox("Choose product", product_stats['Product'])
            if st.button("Recommend"):
                st.write(top_n_recommendations(rules_disp, product_choice))

            partial = st.text_input("Partial basket")
            if st.button("Predict Missing"):
                st.write(predict_missing_items(rules_disp, partial.split(',')))

            col1, col2, col3 = st.columns(3)
            col1.download_button("Rules CSV", rules_disp.to_csv(index=False), "rules.csv")
            col2.download_button("Items CSV", freq_disp.to_csv(index=False), "itemsets.csv")

            if REPORTLAB_AVAILABLE:
                buf = io.BytesIO()
                summary = f"Transactions: {len(df)}, Products: {df.shape[1]}"
                pdf = create_pdf(buf, summary, product_stats, rules_disp)
                col3.download_button("PDF Report", pdf.getvalue(), "report.pdf")

if 'autotune' not in st.session_state:
    st.session_state['autotune'] = False

if __name__ == "__main__":
    main()

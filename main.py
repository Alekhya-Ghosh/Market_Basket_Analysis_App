# market_basket_dashboard_fixed.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import io
import warnings

warnings.filterwarnings("ignore")

# Optional libs with fallbacks
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

# Basic CSS to keep your prior style
st.set_page_config(page_title="Market Basket Intelligence — Fixed", page_icon="📊", layout="wide")
st.markdown(
    """
    <style>
    body { background-color: #F5F7F9; }
    [data-testid="stSidebar"] { background-color: #004F4F !important; color: white !important; }
    h1,h2,h3,h4,h5 { color: #008080 !important; font-weight:700 !important; }
    .insight-box { background-color:#e8f4fd; padding:1rem; border-left:4px solid #ff6b6b; border-radius:8px; }
    .success-box { background-color:#d4edda; padding:0.8rem; border-left:4px solid #28a745; border-radius:8px; }
    .warning-box { background-color:#fff3cd; padding:0.8rem; border-left:4px solid #ffc107; border-radius:8px; }
    .error-box { background-color:#f8d7da; padding:0.8rem; border-left:4px solid #dc3545; border-radius:8px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------- Utilities -----------------
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
    df = (df > 0).astype(int)
    return df

def validate_dataset(df_raw):
    info = {}
    info['rows'] = df_raw.shape[0]
    info['cols'] = df_raw.shape[1]

    blank_cols = [c for c in df_raw.columns if str(c).strip() == ""]
    info['blank_columns'] = blank_cols

    unique_vals = {}
    for c in df_raw.columns:
        sample = df_raw[c].dropna().unique().tolist()
        # take up to 10 items to avoid huge lists
        unique_vals[c] = sample[:10]
    info['unique_sample'] = unique_vals

    column_checks = {}
    for c in df_raw.columns:
        series = df_raw[c].astype(str).str.strip().str.lower()
        valid = series.isin(['0','1','true','false','yes','no','y','n'])
        non_numeric_pct = round((~series.str.isnumeric()).mean()*100, 2) if series.shape[0] > 0 else 0
        column_checks[c] = {'valid_pct': round(valid.mean()*100,2), 'non_numeric_pct': non_numeric_pct}
    info['column_checks'] = column_checks

    # After trying to coerce to numeric
    try:
        df_bin = preprocess_binary_df(df_raw)
        total_ones = int(df_bin.sum().sum())
        sparsity = 1 - (total_ones / (df_bin.shape[0] * df_bin.shape[1])) if df_bin.shape[0]*df_bin.shape[1] > 0 else 1
        info['sparsity'] = round(sparsity, 4)
        info['duplicate_transactions'] = int(df_bin.duplicated().sum())
        info['empty_transactions'] = int((df_bin.sum(axis=1) == 0).sum())
        info['max_basket_size'] = int(df_bin.sum(axis=1).max()) if df_bin.shape[0] > 0 else 0
        info['avg_basket_size'] = round(float(df_bin.sum(axis=1).mean()), 2) if df_bin.shape[0] > 0 else 0
    except Exception:
        info.update({'sparsity': None, 'duplicate_transactions': None, 'empty_transactions': None,
                     'max_basket_size': None, 'avg_basket_size': None})
    return info

# ----------------- Apriori / formatting -----------------
@st.cache_data
def run_apriori_and_format(df, min_support, min_confidence):
    """
    Run apriori and association_rules. Return:
      - freq_itemsets (DataFrame)
      - rules_display (DataFrame) where antecedents/consequents are strings (no frozenset)
      - error (None or string)
    """
    if not MLXTEND_AVAILABLE:
        return pd.DataFrame(), pd.DataFrame(), "mlxtend is not installed"
    try:
        freq = apriori(df, min_support=min_support, use_colnames=True, low_memory=True)
        if freq.empty:
            return freq, pd.DataFrame(), None

        rules = association_rules(freq, metric="confidence", min_threshold=min_confidence)
        if rules.empty:
            return freq, pd.DataFrame(), None

        # Create a cleaned display copy (convert frozensets -> strings)
        rules_disp = rules.copy()

        # Convert antecedents/consequents (frozenset) to comma-separated strings
        def fs_to_str(x):
            try:
                # x may be a frozenset or other iterable
                if hasattr(x, '__iter__') and not isinstance(x, str):
                    return ', '.join(sorted([str(i) for i in list(x)]))
                return str(x)
            except Exception:
                return str(x)

        rules_disp['antecedents'] = rules_disp['antecedents'].apply(fs_to_str)
        rules_disp['consequents'] = rules_disp['consequents'].apply(fs_to_str)

        # Convert itemsets in freq as well for ease of download
        if 'itemsets' in freq.columns:
            freq_disp = freq.copy()
            freq_disp['itemsets'] = freq_disp['itemsets'].apply(fs_to_str)
        else:
            freq_disp = freq

        # numeric rounding and impact_score
        rules_disp = rules_disp.reset_index(drop=True)
        rules_disp['impact_score'] = rules_disp['support'] * rules_disp['lift']
        for c in ['support','confidence','lift','leverage','conviction','impact_score']:
            if c in rules_disp.columns:
                rules_disp[c] = rules_disp[c].round(6)

        # Return frequency itemsets (original for potential internal use) and rules_display for UI
        return freq_disp, rules_disp, None

    except Exception as e:
        return pd.DataFrame(), pd.DataFrame(), str(e)

# ----------------- Visualizations (use rules_display only) -----------------
def plot_cooccurrence_heatmap(df):
    if not MATPLOTLIB_AVAILABLE:
        return None
    co = df.T.dot(df)
    fig, ax = plt.subplots(figsize=(10, 9))
    sns.heatmap(co, cmap="Blues", linewidths=0.2, ax=ax)
    ax.set_title("Product Co-occurrence (counts)")
    plt.tight_layout()
    return fig

def plot_rules_scatter(rules_disp):
    """
    rules_disp must be the cleaned display dataframe where antecedents/consequents are strings.
    """
    if rules_disp.empty or not PLOTLY_AVAILABLE:
        return None
    r = rules_disp.copy()
    # Ensure columns exist and are numeric
    for c in ['support','confidence','lift']:
        if c not in r.columns:
            r[c] = 0.0
    fig = px.scatter(
        r,
        x='support',
        y='confidence',
        size='lift',
        color='lift',
        hover_data=['antecedents','consequents'],
        labels={'lift':'Lift'},
        title="Association Rules: Support vs Confidence"
    )
    fig.update_layout(height=480, template="plotly_white")
    return fig

def plot_network_rules(rules_disp, top_n=30):
    """
    Build a Plotly network graph from rules_disp (strings).
    """
    if rules_disp.empty or not (PLOTLY_AVAILABLE and NETWORKX_AVAILABLE):
        return None

    # select top rules by lift then support
    rsmall = rules_disp.sort_values(['lift','support'], ascending=[False, False]).head(top_n)

    G = nx.DiGraph()
    for _, row in rsmall.iterrows():
        ants = [a.strip() for a in str(row['antecedents']).split(',') if a.strip()]
        cons = [c.strip() for c in str(row['consequents']).split(',') if c.strip()]
        for a in ants:
            G.add_node(a)
            for b in cons:
                G.add_node(b)
                # store lift as weight
                G.add_edge(a, b, weight=float(row.get('lift', 1.0)))

    if G.number_of_nodes() == 0:
        return None

    pos = nx.spring_layout(G, seed=42, k=0.7)

    edge_x = []
    edge_y = []
    for u, v, d in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    node_x = []
    node_y = []
    node_text = []
    for n in G.nodes():
        x, y = pos[n]
        node_x.append(x)
        node_y.append(y)
        node_text.append(n)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='#888'),
        hoverinfo='none',
        mode='lines'
    )

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=node_text,
        textposition="bottom center",
        hoverinfo='text',
        marker=dict(size=18, line_width=1)
    )

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(title='Association Rules Network', showlegend=False,
                                     margin=dict(b=20, l=5, r=5, t=40)))
    return fig

# ----------------- Recommendation utilities -----------------
def top_n_recommendations(rules_disp, product, top_n=5):
    if rules_disp.empty or product is None or product == "":
        return []
    # case-insensitive match on antecedents (string)
    mask = rules_disp['antecedents'].str.contains(rf"\b{pd.util.testing._stringify(product)}\b", case=False, regex=True) if False else rules_disp['antecedents'].str.contains(product, case=False, regex=False)
    # simpler approach: split antecedents and check membership
    candidates = []
    for _, r in rules_disp.iterrows():
        ants = [a.strip().lower() for a in str(r['antecedents']).split(',') if a.strip()]
        if product.strip().lower() in ants:
            candidates.append(r)
    if not candidates:
        return []
    cand_df = pd.DataFrame(candidates)
    cand_df = cand_df.sort_values(['confidence','lift'], ascending=[False, False])
    res = []
    for _, row in cand_df.head(top_n).iterrows():
        res.append({
            'consequent': row['consequents'],
            'confidence': float(row.get('confidence', 0.0)),
            'lift': float(row.get('lift', 0.0)),
            'support': float(row.get('support', 0.0))
        })
    return res

def predict_missing_items(rules_disp, partial_items, top_n=5):
    if rules_disp.empty or not partial_items:
        return []
    known = set([p.strip().lower() for p in partial_items if p.strip()])
    scored = {}
    for _, r in rules_disp.iterrows():
        ants = set([a.strip().lower() for a in str(r['antecedents']).split(',') if a.strip()])
        cons = set([c.strip().lower() for c in str(r['consequents']).split(',') if c.strip()])
        if ants and ants.issubset(known):
            for c in cons:
                if c in known:
                    continue
                score = float(r.get('confidence', 0.0)) * float(r.get('lift', 0.0))
                scored[c] = scored.get(c, 0) + score
    results = sorted(scored.items(), key=lambda x: x[1], reverse=True)
    return [{'product': k, 'score': v} for k, v in results[:top_n]]

# ----------------- Auto-tune -----------------
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
                rows.append({'support': s, 'confidence': c, 'n_rules': 0, 'avg_lift': 0.0})
                continue
            try:
                rules = association_rules(freq, metric="confidence", min_threshold=c)
                n = len(rules) if not rules.empty else 0
                avg_lift = float(rules['lift'].mean()) if (not rules.empty) and ('lift' in rules.columns) else 0.0
                rows.append({'support': s, 'confidence': c, 'n_rules': n, 'avg_lift': round(avg_lift,4)})
            except Exception:
                rows.append({'support': s, 'confidence': c, 'n_rules': 0, 'avg_lift': 0.0})
    df_res = pd.DataFrame(rows).sort_values(['n_rules','avg_lift'], ascending=[False, False]).reset_index(drop=True)
    return df_res.head(top_k)

# ----------------- Simple PDF report creator (basic) -----------------
def create_simple_pdf(buffer, summary_text, top_products_df, rules_df):
    if not REPORTLAB_AVAILABLE:
        return None
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    y = height - 40
    c.setFont("Helvetica-Bold", 14)
    c.drawString(40, y, "Market Basket Analysis Report")
    y -= 24
    c.setFont("Helvetica", 10)
    c.drawString(40, y, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    y -= 20
    for line in summary_text.split('\n'):
        c.drawString(40, y, line)
        y -= 14
    y -= 8
    c.setFont("Helvetica-Bold", 11)
    c.drawString(40, y, "Top products:")
    y -= 16
    c.setFont("Helvetica", 10)
    for idx, row in top_products_df.head(10).iterrows():
        text = f"{idx+1}. {row['Product']} — {row['Purchase Count']} ({row['Purchase Rate (%)']}%)"
        c.drawString(42, y, text)
        y -= 12
        if y < 80:
            c.showPage()
            y = height - 40
    y -= 8
    c.setFont("Helvetica-Bold", 11)
    c.drawString(40, y, "Top association rules (sample):")
    y -= 16
    c.setFont("Helvetica", 9)
    for idx, row in rules_df.head(10).iterrows():
        text = f"{row.get('antecedents','')} -> {row.get('consequents','')} (conf: {row.get('confidence',0):.2f}, lift: {row.get('lift',0):.2f})"
        c.drawString(42, y, text)
        y -= 11
        if y < 80:
            c.showPage()
            y = height - 40
    c.save()
    buffer.seek(0)
    return buffer

# ----------------- Main App -----------------
def main():
    st.title("📊 Market Basket Intelligence — Fixed & Stable")
    st.markdown("Improved handling of rules (no frozenset errors). Upload CSV or use sample dataset below.")

    # Sidebar controls
    st.sidebar.header("Data & Parameters")
    uploaded_file = st.sidebar.file_uploader("Upload CSV (products as columns, rows as transactions)", type=['csv'])
    min_support = st.sidebar.slider("Min support", 0.01, 0.3, 0.05, 0.01)
    min_confidence = st.sidebar.slider("Min confidence", 0.1, 0.95, 0.5, 0.05)
    max_rules_display = st.sidebar.slider("Max rules to display", 5, 100, 20, 5)
    if st.sidebar.button("Auto-tune"):
        st.session_state['autotune'] = True

    # Load data
    if uploaded_file is None:
        st.info("No file uploaded — using sample dataset.")
        sample_data = pd.DataFrame({
            'Dairy_Milk':[1,0,1,0,1,1,0,1,0,1],
            'Whole_Wheat_Bread':[1,1,0,1,1,0,1,0,1,1],
            'Fresh_Eggs':[0,1,1,1,0,1,1,0,1,0],
            'Organic_Butter':[0,1,0,1,1,0,1,1,0,1],
            'Artisan_Cheese':[1,0,1,0,1,1,0,1,1,0],
            'Greek_Yogurt':[1,1,1,0,0,1,0,1,0,1],
            'Orange_Juice':[0,1,0,1,1,0,1,0,1,0]
        })
        df_raw = sample_data.copy()
        if st.sidebar.button("Download sample CSV"):
            st.download_button("Download sample CSV", sample_data.to_csv(index=False), file_name="sample_retail_data.csv")
    else:
        try:
            df_raw = read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")
            st.stop()

    # Preprocess
    df = preprocess_binary_df(df_raw)

    # Validation dashboard
    st.markdown("## 🔎 Data Validation & Overview")
    validation = validate_dataset(df_raw)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Transactions", validation['rows'])
    c2.metric("Products", validation['cols'])
    c3.metric("Avg basket size", validation.get('avg_basket_size', 'N/A'))
    c4.metric("Sparsity", validation.get('sparsity', 'N/A'))

    with st.expander("Show validation details"):
        st.write("Blank columns:", validation['blank_columns'])
        st.write("Duplicate transactions:", validation.get('duplicate_transactions'))
        st.write("Empty transactions:", validation.get('empty_transactions'))
        st.write("Max basket size:", validation.get('max_basket_size'))
        st.write("Column checks (sample):")
        st.write(pd.DataFrame(validation['column_checks']).T.head(50))

        # SAFE presentation of unique_sample (fix for unequal lengths)
        unique_df = pd.DataFrame({
            'Column': list(validation['unique_sample'].keys()),
            'Sample Unique Values': [', '.join(map(str, v)) for v in validation['unique_sample'].values()]
        })
        st.write(unique_df.head(20))

    st.markdown("### Data sample")
    st.dataframe(df.head(8), use_container_width=True)

    # Product portfolio
    st.markdown("## 🧾 Product Portfolio")
    product_sums = df.sum().sort_values(ascending=False)
    product_stats = pd.DataFrame({
        'Product': product_sums.index,
        'Purchase Count': product_sums.values,
        'Purchase Rate (%)': (product_sums.values / len(df) * 100).round(1)
    }).reset_index(drop=True)
    colL, colR = st.columns([2,1])
    colL.dataframe(product_stats.head(15), use_container_width=True)
    with colR:
        st.metric("Total items sold", int(product_sums.sum()))
        st.metric("Unique products", int(len(product_sums)))
        st.metric("Avg basket size", round(df.sum(axis=1).mean(),2))

    # Co-occurrence heatmap
    st.markdown("### 🔗 Product Co-occurrence Heatmap")
    if MATPLOTLIB_AVAILABLE:
        fig_heat = plot_cooccurrence_heatmap(df)
        if fig_heat:
            st.pyplot(fig_heat, use_container_width=True)
    else:
        st.info("Install seaborn & matplotlib for co-occurrence heatmap (pip install seaborn matplotlib)")

    # Run Apriori & rules
    st.markdown("## 🔍 Pattern Discovery")
    run_btn = st.button("Run Market Basket Analysis")
    if run_btn or st.session_state.get('autotune', False):
        if st.session_state.get('autotune', False):
            st.info("Auto-tuning parameters (grid search)...")
            tune = autotune_parameters(df)
            st.dataframe(tune, use_container_width=True)
            if not tune.empty:
                suggested = tune.iloc[0]
                st.success(f"Suggested min_support={suggested['support']}, min_confidence={suggested['confidence']}")
                min_support = float(suggested['support'])
                min_confidence = float(suggested['confidence'])
            st.session_state['autotune'] = False

        with st.spinner("Running Apriori and generating rules..."):
            freq_disp, rules_disp, error = run_apriori_and_format(df, min_support, min_confidence)

        if error:
            st.error(f"Analysis error: {error}")
        else:
            if rules_disp.empty:
                st.warning("No rules found — try lowering min_support/min_confidence or use a larger dataset.")
            else:
                st.success(f"Found {len(rules_disp)} rules")

                # Filters
                st.markdown("### Rule filters & display")
                min_lift = st.slider("Min lift filter", 1.0, float(rules_disp['lift'].max()), 1.1, 0.1)
                min_conf = st.slider("Min confidence filter", float(rules_disp['confidence'].min()), float(rules_disp['confidence'].max()), min_confidence, 0.01)
                filtered = rules_disp[(rules_disp['lift'] >= min_lift) & (rules_disp['confidence'] >= min_conf)]
                st.dataframe(filtered[['antecedents','consequents','support','confidence','lift']].head(max_rules_display), use_container_width=True)

                # Visualizations (use cleaned rules_disp)
                st.markdown("### Visualizations")
                col1, col2 = st.columns(2)
                with col1:
                    scatter_fig = plot_rules_scatter(rules_disp)
                    if scatter_fig:
                        st.plotly_chart(scatter_fig, use_container_width=True)
                with col2:
                    net_fig = plot_network_rules(rules_disp, top_n=40)
                    if net_fig:
                        st.plotly_chart(net_fig, use_container_width=True)
                    else:
                        if not NETWORKX_AVAILABLE:
                            st.info("Install networkx for network graphs: pip install networkx")

                # Insights and recommendations
                st.markdown("### 💡 Insights & Recommendations")
                best_by_lift = rules_disp.sort_values('lift', ascending=False).iloc[0]
                best_by_conf = rules_disp.sort_values('confidence', ascending=False).iloc[0]
                best_by_support = rules_disp.sort_values('support', ascending=False).iloc[0]
                colA, colB, colC = st.columns(3)
                colA.markdown(f"**Strongest (lift)**\n\n{best_by_lift['antecedents']} → {best_by_lift['consequents']}\n\nLift: {best_by_lift['lift']}")
                colB.markdown(f"**Most predictive (confidence)**\n\n{best_by_conf['antecedents']} → {best_by_conf['consequents']}\n\nConfidence: {best_by_conf['confidence']}")
                colC.markdown(f"**Most frequent (support)**\n\n{best_by_support['antecedents']} → {best_by_support['consequents']}\n\nSupport: {best_by_support['support']}")

                # What-if recommendations & predict partial baskets
                st.markdown("### 🔮 What-if Recommendations")
                product_choice = st.selectbox("Select a product to get top-N recommendations", options=list(product_sums.index))
                top_n = st.number_input("Top N", min_value=1, max_value=10, value=5)
                if st.button("Get Top-N Recommendations for Product"):
                    recs = top_n_recommendations(rules_disp, product_choice, top_n=top_n)
                    if recs:
                        for r in recs:
                            st.write(f"- {r['consequent']} (conf: {r['confidence']:.2f}, lift: {r['lift']:.2f})")
                    else:
                        st.info("No recommendations found for this product.")

                st.markdown("Predict missing items given partial basket (comma-separated):")
                partial_input = st.text_input("Known items (comma-separated)", "")
                if st.button("Predict Missing Items"):
                    parts = [p.strip() for p in partial_input.split(',') if p.strip()]
                    preds = predict_missing_items(rules_disp, parts, top_n=5)
                    if preds:
                        for p in preds:
                            st.write(f"- {p['product']} (score: {p['score']:.3f})")
                    else:
                        st.info("No predictions — try more items or tune parameters.")

                # Export
                st.markdown("### 📤 Export results")
                colX, colY, colZ = st.columns(3)
                with colX:
                    st.download_button("Download rules (CSV)", data=rules_disp.to_csv(index=False).encode('utf-8'), file_name="association_rules.csv", mime="text/csv")
                with colY:
                    st.download_button("Download frequent itemsets (CSV)", data=freq_disp.to_csv(index=False).encode('utf-8'), file_name="frequent_itemsets.csv", mime="text/csv")
                with colZ:
                    if REPORTLAB_AVAILABLE:
                        buf = io.BytesIO()
                        summary_text = f"Transactions: {len(df)} | Products: {df.shape[1]} | min_support: {min_support} | min_confidence: {min_confidence}"
                        pdf_buf = create_simple_pdf(buf, summary_text, product_stats, rules_disp)
                        if pdf_buf:
                            st.download_button("Download report (PDF)", data=pdf_buf.getvalue(), file_name="mba_report.pdf", mime="application/pdf")
                        else:
                            st.info("PDF generation failed or not available.")
                    else:
                        st.info("Install reportlab to enable PDF export: pip install reportlab")

    # Basket analytics
    st.markdown("## 📈 Basket Analytics")
    basket_sizes = df.sum(axis=1)
    col_left, col_right = st.columns([2,1])
    with col_left:
        if PLOTLY_AVAILABLE:
            fig = px.histogram(basket_sizes, nbins=20, labels={'value':'Basket size'}, title="Basket size distribution")
            st.plotly_chart(fig, use_container_width=True)
        elif MATPLOTLIB_AVAILABLE:
            fig, ax = plt.subplots()
            ax.hist(basket_sizes, bins=20)
            ax.set_title("Basket size distribution")
            st.pyplot(fig)
        else:
            st.write(basket_sizes.describe())
    with col_right:
        st.metric("Median basket size", int(basket_sizes.median()))
        st.metric("Max basket size", int(basket_sizes.max()))

    st.markdown("---")
    st.markdown("Market Basket Intelligence (fixed)")

# ensure session_state keys exist
if 'autotune' not in st.session_state:
    st.session_state['autotune'] = False

if __name__ == "__main__":
    main()

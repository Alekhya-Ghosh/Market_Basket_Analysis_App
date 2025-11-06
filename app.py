import pandas as pd
import numpy as np
import streamlit as st

# Try to import Plotly with fallback
try:
    import plotly.express as px
    import plotly.graph_objects as go

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    st.warning("Plotly not available. Using basic visualizations.")

# Try to import MLxtend with fallback
try:
    from mlxtend.frequent_patterns import apriori, association_rules

    MLXTEND_AVAILABLE = True
except ImportError:
    MLXTEND_AVAILABLE = False
    st.error("MLxtend not available. Please check requirements.txt")

import warnings

warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Market Basket Analysis",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("🛒 Market Basket Analysis Pro")
st.markdown("""
**Upload your transaction data CSV file to discover which products customers buy together!**
This analysis helps optimize product placement, create effective promotions, and improve inventory management.
""")


def perform_market_basket_analysis(df, min_support=0.01, min_confidence=0.5):
    """Perform market basket analysis using Apriori algorithm"""
    try:
        if not MLXTEND_AVAILABLE:
            return pd.DataFrame(), pd.DataFrame(), "MLxtend library not available"

        # Generate frequent itemsets
        frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True, low_memory=True)

        # Generate association rules
        if not frequent_itemsets.empty:
            rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
            # Sort by lift to get most meaningful rules
            rules = rules.sort_values('lift', ascending=False)
        else:
            rules = pd.DataFrame()

        return frequent_itemsets, rules, None

    except Exception as e:
        return pd.DataFrame(), pd.DataFrame(), f"Analysis error: {str(e)}"


def format_rules_for_display(rules):
    """Convert frozenset objects to strings for display"""
    if rules.empty:
        return rules

    rules_display = rules.copy()

    # Convert frozenset to readable strings
    rules_display['antecedents'] = rules_display['antecedents'].apply(
        lambda x: ', '.join(list(x)) if x else ''
    )
    rules_display['consequents'] = rules_display['consequents'].apply(
        lambda x: ', '.join(list(x)) if x else ''
    )

    # Round numeric columns for better display
    numeric_cols = ['support', 'confidence', 'lift', 'leverage', 'conviction']
    for col in numeric_cols:
        if col in rules_display.columns:
            rules_display[col] = rules_display[col].round(4)

    return rules_display


def safe_convert_to_string(value):
    """Safely convert any value to string, handling numpy types"""
    if pd.isna(value):
        return ""
    elif isinstance(value, (np.float64, np.int64, np.float32, np.int32)):
        return str(float(value))
    else:
        return str(value)


def create_bar_chart(data, title):
    """Create bar chart with fallback to Streamlit native"""
    if PLOTLY_AVAILABLE:
        fig = px.bar(
            x=data.values,
            y=data.index,
            orientation='h',
            title=title,
            labels={'x': 'Purchase Count', 'y': 'Products'},
            color=data.values,
            color_continuous_scale='viridis'
        )
        fig.update_layout(showlegend=False, height=400)
        return fig
    else:
        # Fallback to Streamlit native chart
        st.write(f"**{title}**")
        chart_data = pd.DataFrame({
            'Products': data.index,
            'Count': data.values
        })
        st.bar_chart(chart_data.set_index('Products'))


def main():
    if not MLXTEND_AVAILABLE:
        st.error("""
        ❌ Required libraries not installed. Please make sure your requirements.txt contains:
        ```
        streamlit>=1.28.0
        pandas>=1.5.0
        numpy>=1.21.0
        plotly>=5.13.0
        mlxtend>=0.22.0
        ```
        """)
        return

    # Sidebar for file upload and controls
    st.sidebar.header("📁 Upload Your Data")

    uploaded_file = st.sidebar.file_uploader(
        "Choose a CSV file with transaction data",
        type=['csv'],
        help="File should have products as columns and transactions as rows with 1/0 values"
    )

    st.sidebar.header("⚙️ Analysis Parameters")

    min_support = st.sidebar.slider(
        "Minimum Support",
        0.01, 0.5, 0.05, 0.01,
        help="Minimum frequency of itemsets in the dataset (0.01 = 1%)"
    )

    min_confidence = st.sidebar.slider(
        "Minimum Confidence",
        0.1, 1.0, 0.5, 0.05,
        help="Minimum confidence for association rules (0.5 = 50%)"
    )

    # Show sample data format
    with st.sidebar.expander("📋 Expected CSV Format"):
        st.markdown("""
        **Your CSV should look like this:**
        ```
        Milk,Bread,Eggs,Butter,Cheese
        1,0,1,0,1
        0,1,1,1,0
        1,1,0,0,1
        0,1,0,1,0
        1,0,1,1,1
        ```
        - **Columns**: Product names
        - **Rows**: Transactions
        - **Values**: 1 (purchased) or 0 (not purchased)
        """)

    if uploaded_file is not None:
        try:
            # Read the uploaded file
            df = pd.read_csv(uploaded_file)

            # Clean column names and data
            df = df.rename(columns=lambda x: str(x).strip())
            df = df.applymap(
                lambda x: 1 if str(x).strip() in ['1', 'True', 'true', 'YES', 'yes'] else 0 if str(x).strip() in ['0',
                                                                                                                  'False',
                                                                                                                  'false',
                                                                                                                  'NO',
                                                                                                                  'no'] else x)

            # Convert to numeric, coerce errors to NaN, then fill with 0
            df = df.apply(pd.to_numeric, errors='coerce').fillna(0)

            # Final conversion to binary (1 for any positive value, 0 otherwise)
            df = (df > 0).astype(int)

            # Display file info
            st.success(f"✅ File uploaded successfully! {df.shape[0]} transactions, {df.shape[1]} products")

            # Data preview
            col1, col2 = st.columns([2, 1])

            with col1:
                st.subheader("📊 Data Preview")
                st.dataframe(df.head(10), use_container_width=True)

            with col2:
                st.subheader("📈 Data Summary")
                st.metric("Total Transactions", df.shape[0])
                st.metric("Total Products", df.shape[1])
                total_items = df.sum().sum()
                st.metric("Total Items Purchased", f"{int(total_items):,}")
                st.metric("Avg Items per Transaction", f"{total_items / df.shape[0]:.2f}")

            # Show product popularity
            st.subheader("🏆 Product Popularity")
            product_sums = df.sum().sort_values(ascending=False)

            if PLOTLY_AVAILABLE:
                fig_popularity = create_bar_chart(
                    product_sums,
                    "Most Frequently Purchased Products"
                )
                st.plotly_chart(fig_popularity, use_container_width=True)
            else:
                create_bar_chart(product_sums, "Most Frequently Purchased Products")

            # Product stats table
            product_stats = pd.DataFrame({
                'Product': product_sums.index,
                'Purchase_Count': product_sums.values,
                'Purchase_Rate': (product_sums.values / len(df)).round(3)
            }).head(10)
            st.dataframe(product_stats, use_container_width=True)

            # Run analysis when button is clicked
            if st.button("🚀 Run Market Basket Analysis", type="primary", use_container_width=True):
                with st.spinner("Analyzing purchase patterns... This may take a few moments."):
                    frequent_itemsets, rules, error = perform_market_basket_analysis(
                        df, min_support, min_confidence
                    )

                    if error:
                        st.error(f"❌ {error}")
                        return

                    # Display results
                    st.subheader("🔍 Analysis Results")

                    # Frequent itemsets
                    st.markdown("#### 🛍️ Frequent Itemsets")
                    if not frequent_itemsets.empty:
                        frequent_itemsets_display = frequent_itemsets.copy()
                        frequent_itemsets_display['itemset'] = frequent_itemsets_display['itemsets'].apply(
                            lambda x: ', '.join(list(x)) if x else ''
                        )
                        frequent_itemsets_display['length'] = frequent_itemsets_display['itemsets'].apply(
                            lambda x: len(x) if x else 0
                        )
                        frequent_itemsets_display['support'] = frequent_itemsets_display['support'].round(4)

                        # Show top itemsets
                        top_itemsets = frequent_itemsets_display.nlargest(10, 'support')
                        st.dataframe(
                            top_itemsets[['itemset', 'support', 'length']],
                            use_container_width=True
                        )
                    else:
                        st.warning("No frequent itemsets found. Try lowering the minimum support.")

                    # Association rules
                    st.markdown("#### 🔗 Association Rules")
                    if not rules.empty:
                        rules_display = format_rules_for_display(rules)

                        # Display top rules
                        display_cols = ['antecedents', 'consequents', 'support', 'confidence', 'lift']
                        available_cols = [col for col in display_cols if col in rules_display.columns]

                        st.dataframe(
                            rules_display[available_cols].head(20),
                            use_container_width=True
                        )

                        # Business insights
                        st.markdown("#### 💡 Business Insights")
                        if not rules.empty:
                            best_rule = rules_display.iloc[0]  # Already sorted by lift
                            most_confident_rule = rules_display.nlargest(1, 'confidence').iloc[0]

                            col1, col2 = st.columns(2)

                            with col1:
                                st.info(f"""
                                **🎯 Strongest Association**  
                                **When customers buy:** {safe_convert_to_string(best_rule['antecedents'])}  
                                **They're {float(best_rule['lift']):.1f}x more likely to buy:** {safe_convert_to_string(best_rule['consequents'])}  
                                **Confidence:** {float(best_rule['confidence']):.1%}  
                                **Support:** {float(best_rule['support']):.1%}
                                """)

                            with col2:
                                st.success(f"""
                                **📊 Most Predictive Rule**  
                                **When customers buy:** {safe_convert_to_string(most_confident_rule['antecedents'])}  
                                **{float(most_confident_rule['confidence']):.1%} also buy:** {safe_convert_to_string(most_confident_rule['consequents'])}  
                                **Lift:** {float(most_confident_rule['lift']):.1f}x  
                                **Support:** {float(most_confident_rule['support']):.1%}
                                """)

                            # Recommendations
                            st.markdown("""
                            #### 🎯 Strategic Recommendations:

                            **📍 Product Placement**: Place strongly associated products near each other  
                            **💰 Promotional Bundles**: Create deals for frequently bought together items  
                            **📦 Inventory Management**: Stock associated products in similar quantities  
                            **📢 Targeted Marketing**: Recommend products based on customer purchases
                            """)

                    else:
                        st.warning(
                            "No association rules found. Try adjusting the minimum support or confidence thresholds.")

        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            st.info(
                "Please make sure your CSV file is properly formatted with products as columns and transactions as rows.")

    else:
        # Show instructions when no file is uploaded
        st.subheader("🚀 Getting Started")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            ### 📁 How to Use:

            1. **Upload a CSV file** using the sidebar
            2. **Adjust parameters** (support & confidence)
            3. **Click 'Run Market Basket Analysis'**
            4. **Explore insights** and recommendations

            ### 📊 Expected Format:
            - **Columns** = Product names
            - **Rows** = Transactions/Customers
            - **Values** = 1 (purchased) or 0 (not purchased)
            """)

        with col2:
            st.markdown("""
            ### 🎯 What You'll Discover:

            **🛍️ Frequent Itemsets**  
            Products that are often purchased together

            **🔗 Association Rules**  
            "If bought A, then likely to buy B"

            **📈 Key Metrics**  
            - **Support**: How frequently rule occurs
            - **Confidence**: How often rule is true  
            - **Lift**: Strength of association
            """)

        # Sample data download
        st.markdown("---")
        st.subheader("🎯 Try with Sample Data")

        sample_data = pd.DataFrame({
            'Milk': [1, 0, 1, 0, 1, 1, 0, 1, 0, 1],
            'Bread': [1, 1, 0, 1, 1, 0, 1, 0, 1, 1],
            'Eggs': [0, 1, 1, 1, 0, 1, 1, 0, 1, 0],
            'Butter': [0, 1, 0, 1, 1, 0, 1, 1, 0, 1],
            'Cheese': [1, 0, 1, 0, 1, 1, 0, 1, 1, 0]
        })

        # Convert sample data to CSV for download
        csv = sample_data.to_csv(index=False)
        st.download_button(
            label="📥 Download Sample CSV",
            data=csv,
            file_name="sample_market_basket_data.csv",
            mime="text/csv",
            help="Download this sample file to test the application"
        )

        st.info("💡 **Tip**: Download the sample CSV above to see how the app works!")


if __name__ == "__main__":
    main()
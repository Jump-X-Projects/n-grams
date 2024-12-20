import streamlit as st
import pandas as pd
from collections import Counter
import plotly.express as px
import re
from st_aggrid import AgGrid, GridOptionsBuilder

def load_data(uploaded_file):
    """Load and validate the uploaded search terms report."""
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            required_columns = ['Search term', 'Cost', 'Conversions']
            if not all(col in df.columns for col in required_columns):
                st.error("Upload error: The file must contain 'Search term', 'Cost', and 'Conversions' columns")
                return None
            return df
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
            return None
    return None

def preprocess_text(text):
    """Clean and preprocess the search term text."""
    if pd.isna(text):
        return []
    
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    tokens = text.split()
    
    return tokens

def generate_ngrams(tokens, n):
    """Generate n-grams from the preprocessed tokens."""
    if len(tokens) < n:
        return []
    return [' '.join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

def analyze_ngrams(df, n_value, min_frequency=1):
    """Perform n-grams analysis on the search terms with CPA calculations."""
    ngram_data = {}
    
    for _, row in df.iterrows():
        tokens = preprocess_text(row['Search term'])
        term_ngrams = generate_ngrams(tokens, n_value)
        
        for ng in term_ngrams:
            if ng not in ngram_data:
                ngram_data[ng] = {
                    'frequency': 0,
                    'total_cost': 0,
                    'total_conversions': 0
                }
            ngram_data[ng]['frequency'] += 1
            ngram_data[ng]['total_cost'] += row['Cost']
            ngram_data[ng]['total_conversions'] += row['Conversions']
    
    # Convert to DataFrame
    ngram_df = pd.DataFrame([
        {
            'N-gram': ng,
            'Frequency': data['frequency'],
            'Total Cost': round(data['total_cost'], 2),
            'Total Conversions': data['total_conversions'],
            'CPA': round(data['total_cost'] / data['total_conversions'], 2) if data['total_conversions'] > 0 else 0
        }
        for ng, data in ngram_data.items()
        if data['frequency'] >= min_frequency
    ])
    
    if len(ngram_df) > 0:
        return ngram_df.sort_values('Frequency', ascending=False)
    else:
        return pd.DataFrame(columns=['N-gram', 'Frequency', 'Total Cost', 'Total Conversions', 'CPA'])

def show_interactive_grid(df):
    """Display an interactive grid with sorting and filtering."""
    gb = GridOptionsBuilder.from_dataframe(df)
    
    # Simpler configuration that should work more reliably
    gb.configure_column('N-gram', sortable=True, filter=True)
    gb.configure_column('Frequency', sortable=True, filter=True)
    gb.configure_column('Total Cost', sortable=True, filter=True)
    gb.configure_column('Total Conversions', sortable=True, filter=True)
    gb.configure_column('CPA', sortable=True, filter=True)
    
    # Basic grid configuration
    gb.configure_grid_options(domLayout='normal')
    
    grid_response = AgGrid(
        df,
        gridOptions=gb.build(),
        reload_data=False,
        enable_enterprise_modules=False,
        height=400,
        fit_columns_on_grid_load=True,
    )
    
    return grid_response

def main():
    st.title("Search Terms N-grams Analysis with CPA")
    
    # File upload
    st.header("1. Upload Search Terms Report")
    st.write("File must contain 'Search term', 'Cost', and 'Conversions' columns")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        df = load_data(uploaded_file)
        
        if df is not None:
            st.success("File uploaded successfully!")
            st.write(f"Number of search terms: {len(df)}")
            
            # N-gram configuration
            st.header("2. Configure Analysis")
            col1, col2 = st.columns(2)
            
            with col1:
                n_value = st.slider("Select N-gram size", 1, 5, 2)
            
            with col2:
                min_frequency = st.slider("Minimum frequency", 1, 50, 1)
            
            # Perform analysis
            if st.button("Analyze N-grams"):
                with st.spinner("Analyzing n-grams..."):
                    results_df = analyze_ngrams(df, n_value, min_frequency)
                
                if len(results_df) > 0:
                    st.header("3. Results")
                    
                    # Display interactive grid
                    st.subheader("Top N-grams with CPA")
                    grid_response = show_interactive_grid(results_df)
                    
                    # Create visualization
                    st.subheader("Visualization")
                    metric_options = ['Frequency', 'CPA', 'Total Cost', 'Total Conversions']
                    selected_metric = st.selectbox("Select metric to visualize", metric_options)
                    
                    top_n = st.slider("Select top N results to visualize", 
                                    min_value=5, 
                                    max_value=min(50, len(results_df)), 
                                    value=20)
                    
                    plot_data = results_df.head(top_n)
                    fig = px.bar(plot_data, 
                               x='N-gram', 
                               y=selected_metric,
                               title=f'Top {top_n} {n_value}-grams by {selected_metric}')
                    fig.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig)
                    
                    # Download results
                    st.download_button(
                        label="Download results as CSV",
                        data=results_df.to_csv(index=False),
                        file_name=f"{n_value}-grams_analysis.csv",
                        mime="text/csv"
                    )
                else:
                    st.warning("No n-grams found with the specified parameters.")

if __name__ == "__main__":
    main()

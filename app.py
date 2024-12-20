import streamlit as st
import pandas as pd
from collections import Counter
import plotly.express as px
import re
import numpy as np

@st.cache_data
def load_data(uploaded_file, sample_size=1.0):
    """Load and validate the uploaded search terms report."""
    if uploaded_file is not None:
        try:
            # First count total rows without loading entire file
            df_count = sum(1 for row in uploaded_file)
            uploaded_file.seek(0)  # Reset file pointer
            
            # If file has more than 100k rows, force sampling
            if df_count > 100000:
                if sample_size == 1.0:
                    sample_size = 0.1  # Default to 10% for large files
                    st.warning(f"Large file detected ({df_count:,} rows). Using {sample_size*100}% sample for analysis.")
            
            # Calculate number of rows to skip for sampling
            if sample_size < 1.0:
                skip_rate = int(1/sample_size)
                df = pd.read_csv(uploaded_file, skiprows=lambda x: x > 0 and x % skip_rate != 0)
            else:
                df = pd.read_csv(uploaded_file)
            
            # Clean up memory
            df = df[['Search term', 'Cost', 'Conversions']].copy()
            
            # Clean and convert Cost column (remove currency symbols and convert to float)
            df['Cost'] = df['Cost'].replace('[\$,]', '', regex=True).astype(float)
            
            # Clean and convert Conversions column
            df['Conversions'] = pd.to_numeric(df['Conversions'], errors='coerce').fillna(0)
            
            return df
            
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
            return None
    return None

@st.cache_data
def process_ngrams_batch(_df, n_value, min_frequency=1):
    """Process n-grams in an optimized way."""
    # Combine all processing into a single pass
    ngram_data = {}
    
    for _, row in _df.iterrows():
        # Process text and generate n-grams in one step
        text = str(row['Search term']).lower()
        text = re.sub(r'[^a-z0-9\s]', '', text)
        tokens = text.split()
        
        if len(tokens) >= n_value:
            for i in range(len(tokens) - n_value + 1):
                ng = ' '.join(tokens[i:i+n_value])
                if ng not in ngram_data:
                    ngram_data[ng] = {'frequency': 0, 'total_cost': 0, 'total_conversions': 0}
                ngram_data[ng]['frequency'] += 1
                ngram_data[ng]['total_cost'] += row['Cost']
                ngram_data[ng]['total_conversions'] += row['Conversions']
    
    return ngram_data

def analyze_ngrams(df, n_value, min_frequency=1):
    """Perform n-grams analysis using batched processing."""
    # Process in a single batch with optimized function
    ngram_data = process_ngrams_batch(df, n_value, min_frequency)
    
    # Convert results to DataFrame
    results = []
    for ng, data in ngram_data.items():
        if data['frequency'] >= min_frequency:
            results.append({
                'N-gram': ng,
                'Frequency': data['frequency'],
                'Total Cost': round(data['total_cost'], 2),
                'Total Conversions': data['total_conversions'],
                'CPA': round(data['total_cost'] / data['total_conversions'], 2) if data['total_conversions'] > 0 else 0
            })
    
    if results:
        results_df = pd.DataFrame(results)
        return results_df.sort_values('Frequency', ascending=False)
    else:
        return pd.DataFrame(columns=['N-gram', 'Frequency', 'Total Cost', 'Total Conversions', 'CPA'])

def main():
    st.title("Search Terms N-grams Analysis")
    
    # File upload
    st.header("1. Upload Search Terms Report")
    st.write("File must contain 'Search term', 'Cost', and 'Conversions' columns")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        # Sample size selection
        sample_size = st.slider("Sample size (%)", 1, 100, 10) / 100
        
        df = load_data(uploaded_file, sample_size)
        
        if df is not None:
            st.success(f"File loaded successfully! Analyzing {len(df):,} rows")
            
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
                    
                    # Display the dataframe
                    st.dataframe(
                        results_df,
                        hide_index=True,
                        column_config={
                            'N-gram': st.column_config.TextColumn('N-gram'),
                            'Frequency': st.column_config.NumberColumn('Frequency'),
                            'Total Cost': st.column_config.NumberColumn(
                                'Total Cost',
                                format="$%.2f"
                            ),
                            'Total Conversions': st.column_config.NumberColumn('Total Conversions'),
                            'CPA': st.column_config.NumberColumn(
                                'CPA',
                                format="$%.2f"
                            )
                        }
                    )
                    
                    # Create basic visualization
                    fig = px.bar(results_df.head(20), 
                               x='N-gram', 
                               y='CPA',
                               title=f'Top 20 {n_value}-grams by CPA')
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

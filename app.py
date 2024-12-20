import streamlit as st
import pandas as pd
from collections import Counter
import plotly.express as px
import re
import numpy as np

def load_data(uploaded_file, sample_size=None):
    """Load and validate the uploaded search terms report."""
    if uploaded_file is not None:
        try:
            # Check file size
            file_size = uploaded_file.size / (1024 * 1024)  # Convert to MB
            st.write(f"File size: {file_size:.2f} MB")
            
            if sample_size:
                # Read with sampling for large files
                df = pd.read_csv(uploaded_file, skiprows=lambda x: x>0 and np.random.random() > sample_size)
                st.info(f"Analyzing a {sample_size*100}% sample of the data due to file size")
            else:
                df = pd.read_csv(uploaded_file)
            
            required_columns = ['Search term', 'Cost', 'Conversions']
            if not all(col in df.columns for col in required_columns):
                st.error("Upload error: The file must contain 'Search term', 'Cost', and 'Conversions' columns")
                return None
            
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
    total_rows = len(df)
    
    # Create a progress bar
    progress_bar = st.progress(0)
    progress_text = st.empty()
    
    # Process in chunks
    chunk_size = 1000
    for i in range(0, total_rows, chunk_size):
        chunk = df.iloc[i:i+chunk_size]
        for _, row in chunk.iterrows():
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
        
        # Update progress
        progress = min(i + chunk_size, total_rows) / total_rows
        progress_bar.progress(progress)
        progress_text.text(f"Processing rows {i} to {min(i + chunk_size, total_rows)} of {total_rows}")
    
    progress_bar.empty()
    progress_text.empty()
    
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

def main():
    st.title("Search Terms N-grams Analysis")
    
    # File upload
    st.header("1. Upload Search Terms Report")
    st.write("File must contain 'Search term', 'Cost', and 'Conversions' columns")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        # Add sampling option for large files
        file_size = uploaded_file.size / (1024 * 1024)  # Convert to MB
        sample_size = None
        if file_size > 100:  # If file is larger than 100MB
            st.warning("Large file detected. Consider using sampling for faster analysis.")
            use_sampling = st.checkbox("Use sampling", value=True)
            if use_sampling:
                sample_size = st.slider("Sample size (%)", 1, 100, 10) / 100
        
        df = load_data(uploaded_file, sample_size)
        
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
                    
                    # Split large downloads into chunks
                    if len(results_df) > 100000:
                        st.warning("Large result set detected. Download will be split into chunks.")
                        chunk_size = 100000
                        for i in range(0, len(results_df), chunk_size):
                            chunk = results_df.iloc[i:i+chunk_size]
                            st.download_button(
                                label=f"Download results part {i//chunk_size + 1} (rows {i+1}-{min(i+chunk_size, len(results_df))})",
                                data=chunk.to_csv(index=False),
                                file_name=f"{n_value}-grams_analysis_part{i//chunk_size + 1}.csv",
                                mime="text/csv"
                            )
                    else:
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

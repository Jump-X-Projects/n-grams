import streamlit as st
import pandas as pd
from collections import Counter
import plotly.express as px
import re

def load_data(uploaded_file):
    """Load and validate the uploaded search terms report."""
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            required_columns = ['Search term']
            if not all(col in df.columns for col in required_columns):
                st.error("Upload error: The file must contain a 'Search term' column")
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
    
    # Convert to lowercase and split into words
    text = str(text).lower()
    # Remove special characters and extra spaces
    text = re.sub(r'[^a-z0-9\s]', '', text)
    tokens = text.split()
    
    return tokens

def generate_ngrams(tokens, n):
    """Generate n-grams from the preprocessed tokens."""
    if len(tokens) < n:
        return []
    return [' '.join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

def analyze_ngrams(df, n_value, min_frequency=1):
    """Perform n-grams analysis on the search terms."""
    all_ngrams = []
    
    for term in df['Search term']:
        tokens = preprocess_text(term)
        term_ngrams = generate_ngrams(tokens, n_value)
        all_ngrams.extend(term_ngrams)
    
    # Count n-grams frequencies
    ngram_counts = Counter(all_ngrams)
    
    # Convert to DataFrame
    ngram_df = pd.DataFrame([
        {
            'N-gram': ng,
            'Frequency': count
        }
        for ng, count in ngram_counts.items()
        if count >= min_frequency
    ])
    
    if len(ngram_df) > 0:
        return ngram_df.sort_values('Frequency', ascending=False)
    else:
        return pd.DataFrame(columns=['N-gram', 'Frequency'])

def main():
    st.title("Search Terms N-grams Analysis")
    
    # File upload
    st.header("1. Upload Search Terms Report")
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
                    
                    # Display results table
                    st.subheader("Top N-grams")
                    st.dataframe(results_df)
                    
                    # Create visualization
                    if len(results_df) > 0:
                        top_n = st.slider("Select top N results to visualize", 
                                        min_value=5, 
                                        max_value=min(50, len(results_df)), 
                                        value=20)
                        
                        plot_data = results_df.head(top_n)
                        fig = px.bar(plot_data, 
                                   x='N-gram', 
                                   y='Frequency',
                                   title=f'Top {top_n} {n_value}-grams')
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

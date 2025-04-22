#!/usr/bin/env python3
"""
HTML2txt - Extract text from HTML files for the Eurlex dataset

This script extracts text from HTML files and saves the extracted text to a DataFrame
"""

import os
import argparse
from bs4 import BeautifulSoup
import pandas as pd
import pickle


def extract_text_from_html(html_content):
    """
    Extract text from HTML content based on specific criteria.
    
    Args:
        html_content (str): HTML content as a string
        
    Returns:
        tuple: (extracted_text, error_message)
            - extracted_text: The extracted text if successful, None otherwise
            - error_message: Error message if extraction failed, None otherwise
    """
    # Create a soup object to parse the HTML
    soup = BeautifulSoup(html_content, 'html.parser')

    # Find all 'div' elements with class 'listNotice'
    list_notices = soup.find_all('div', class_='listNotice')

    # Check if there are at least two 'listNotice' divs
    if len(list_notices) >= 2:
        # Select the second 'listNotice' div
        second_notice = list_notices[1]
        
        # Find the 'div' element with class 'texte' within the second notice
        texte_div = second_notice.find('div', class_='texte')
        
        # Extract text within the 'texte' div
        if texte_div:
            extracted_text = texte_div.get_text(separator=' ').strip()
            return extracted_text, None
        else:
            return None, "No 'texte' div found within the second 'listNotice'"
    else:
        return None, "Less than two 'listNotice' divs found"


def process_html_files(mapping_file, html_dir):
    """
    Process HTML files based on a mapping file and extract text.
    
    Args:
        mapping_file (str): Path to the mapping CSV file
        html_dir (str): Directory containing HTML files
        
    Returns:
        pandas.DataFrame: DataFrame with extracted text
    """
    # Read CSV into DataFrame
    df = pd.read_csv(mapping_file, sep='\t')
    
    # Filter rows to keep
    df_keep = df[~(df['remove'] == 1)]
    
    # Track failures
    failed1, failed2 = [], []
    
    # Loop over rows from df
    for index, row in df.iterrows():
        filename = row['Filename']
        remove_flag = row['remove']
        
        if remove_flag:
            continue
            
        # Read HTML from file
        html_path = os.path.join(html_dir, filename)
        try:
            with open(html_path) as f:
                html_content = f.read()
                
            # Extract text from HTML
            extracted_text, error = extract_text_from_html(html_content)
            
            if extracted_text:
                df_keep.loc[index, 'extracted_text'] = extracted_text
            else:
                if "No 'texte' div" in error:
                    failed1.append(filename)
                else:
                    failed2.append(filename)
                print(f"Error processing {filename}: {error}")
                
        except Exception as e:
            print(f"Error reading file {filename}: {e}")
    
    # Print summary
    print(f"Processed {len(df_keep)} files")
    print(f"Failed to find 'texte' div: {len(failed1)} files")
    print(f"Failed to find enough 'listNotice' divs: {len(failed2)} files")
    
    return df_keep


def load_and_merge_labels(df_keep, data_dir):
    """
    Load label data and merge with the extracted text DataFrame.
    
    Args:
        df_keep (pandas.DataFrame): DataFrame with extracted text
        data_dir (str): Directory containing label data files
        
    Returns:
        pandas.DataFrame: Merged DataFrame with all labels
    """
    # Load DC labels
    print(f"Loading DC labels ...")
    dc_file = os.path.join(data_dir, 'id2class/id2class_eurlex_DC_leaves.qrels')
    df_dc = pd.read_csv(dc_file, sep=' ', header=None)
    df_dc.columns = ['DC', 'DocID', 'Drop']
    df_dc.drop('Drop', axis=1, inplace=True)
    
    # Group DC labels by DocID
    df_dc_grouped = df_dc.groupby('DocID')['DC'].apply(list).reset_index()
    
    # Merge DC labels with extracted text
    merged_df = df_dc_grouped.merge(df_keep, on='DocID', how='right')
    merged_df['DC'] = merged_df['DC'].apply(lambda x: x if isinstance(x, list) else [])
    print("Done merging DC labels")

    # Load EuroVoc labels
    print(f"Loading EuroVoc labels ...")
    eurovoc_file = os.path.join(data_dir, 'id2class/id2class_eurlex_eurovoc.qrels')
    df_eurovoc = pd.read_csv(eurovoc_file, sep=' ', header=None)
    df_eurovoc.columns = ['EuroVoc', 'DocID', 'Drop']
    df_eurovoc.drop('Drop', axis=1, inplace=True)
    
    # Group EuroVoc labels by DocID
    df_eurovoc_grouped = df_eurovoc.groupby('DocID')['EuroVoc'].apply(list).reset_index()
    
    # Merge EuroVoc labels with the merged DataFrame
    merged_df = df_eurovoc_grouped.merge(merged_df, on='DocID', how='right')
    merged_df['EuroVoc'] = merged_df['EuroVoc'].apply(lambda x: x if isinstance(x, list) else [])
    print("Done merging EuroVoc labels")

    # Load Subject Matter labels
    print(f"Loading Subject Matter labels ...")
    sm_file = os.path.join(data_dir, 'id2class/id2class_eurlex_subject_matter.qrels')
    df_sm = pd.read_csv(sm_file, sep=' ', header=None)
    df_sm.columns = ['SM', 'DocID', 'Drop']
    df_sm.drop('Drop', axis=1, inplace=True)
    
    # Group Subject Matter labels by DocID
    df_sm_grouped = df_sm.groupby('DocID')['SM'].apply(list).reset_index()
    
    # Merge Subject Matter labels with the merged DataFrame
    merged_df = df_sm_grouped.merge(merged_df, on='DocID', how='right')
    merged_df['SM'] = merged_df['SM'].apply(lambda x: x if isinstance(x, list) else [])
    print("Done merging Subject Matter labels")
    # Remove 'remove' column if it exists
    if 'remove' in merged_df.columns:
        merged_df.drop('remove', axis=1, inplace=True)
    
    return merged_df


def main():
    parser = argparse.ArgumentParser(description='Preprocess Eurlex4k dataset')
    parser.add_argument('--mapping', required=True, help='Path to the mapping CSV file')
    parser.add_argument('--html-dir', required=True, help='Directory containing HTML files')
    parser.add_argument('--data-dir', required=True, help='Directory containing label data files')
    parser.add_argument('--output', required=True, help='Path to save the output pickle file')
    
    args = parser.parse_args()
    
    # Process HTML files
    print("Extracting text from HTML files...")
    df_keep = process_html_files(args.mapping, args.html_dir)
    
    # Load and merge labels
    print("Loading and merging label data...")
    merged_df = load_and_merge_labels(df_keep, args.data_dir)
    
    # Save the merged DataFrame to pickle
    print(f"Saving processed data to {args.output}...")
    merged_df.to_pickle(args.output, protocol=4)
    
    print("Done!")


if __name__ == "__main__":
    main()
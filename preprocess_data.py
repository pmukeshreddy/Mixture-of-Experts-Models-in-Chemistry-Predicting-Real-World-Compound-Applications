import pandas as pd
import ast
from rdkit import Chem
from sklearn.preprocessing import MultiLabelBinarizer

def preprocess_data(csv_paths: list[str]):
    """
    Loads, merges, cleans, and preprocesses chemical data from multiple CSV files,
    using only the 'smiles' and 'applications' columns.

    Args:
        csv_paths (list[str]): A list of paths to the input CSV files.

    Returns:
        tuple: A tuple containing:
            - all_smiles (list): A list of valid SMILES strings.
            - all_labels (list): A list of multi-hot encoded labels.
            - label_classes (list): A list of the unique application label names.
    """
    # Create a list to hold DataFrames with only the essential columns
    df_list = []
    for path in csv_paths:
        try:
            temp_df = pd.read_csv(path)
            # Check if the required columns exist in the CSV
            if 'smiles' in temp_df.columns and 'applications' in temp_df.columns:
                # Append only the relevant columns to our list
                df_list.append(temp_df[['smiles', 'applications']])
            else:
                print(f"Warning: Skipping file {path} as it lacks 'smiles' or 'applications' columns.")
        except Exception as e:
            print(f"Warning: Could not process file {path}. Error: {e}")
            
    # If no valid data was found, exit gracefully
    if not df_list:
        print("Error: No valid data could be loaded from the provided file paths.")
        return [], [], []

    # Concatenate the filtered dataframes
    df = pd.concat(df_list, ignore_index=True)

    # Handle missing SMILES or applications by dropping those rows
    df.dropna(subset=['smiles', 'applications'], inplace=True)

    # Validate SMILES strings and remove invalid ones
    valid_smiles = []
    original_indices = []
    for i, smi in enumerate(df['smiles']):
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                # Optional: Canonicalize SMILES to have a standard representation
                # valid_smiles.append(Chem.MolToSmiles(mol, canonical=True))
                valid_smiles.append(smi) # Using original valid SMILES for now
                original_indices.append(df.index[i])
        except Exception:
            # RDKit can sometimes raise exceptions for malformed strings
            continue
    
    # Filter the dataframe to only include rows with valid SMILES
    df_valid = df.loc[original_indices].copy() # Create a copy to avoid SettingWithCopyWarning
    df_valid['smiles'] = valid_smiles
    
    # Parse the string representation of lists in the 'applications' column
    # The .loc is used to ensure we are modifying the DataFrame correctly
    df_valid.loc[:, 'parsed_applications'] = df_valid['applications'].apply(ast.literal_eval)
    
    # Use MultiLabelBinarizer to convert the lists of application strings into a binary matrix
    mlb = MultiLabelBinarizer()
    y_binarized = mlb.fit_transform(df_valid['parsed_applications'])
    
    # Convert the processed data into lists for output
    all_smiles = df_valid['smiles'].tolist()
    all_labels = y_binarized.tolist()
    
    print(f"Processed data from {len(csv_paths)} file(s).")
    print(f"Total valid molecules found: {len(all_smiles)}")
    print(f"Number of unique labels found: {len(mlb.classes_)}")
    print(f"Example labels: {list(mlb.classes_[:10])}")

    return all_smiles, all_labels, mlb.classes_


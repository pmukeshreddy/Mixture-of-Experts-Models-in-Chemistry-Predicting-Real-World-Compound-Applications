import pandas as pd
import numpy as np
from google.cloud import bigquery
import time
import re
import json
from rdkit import Chem
from collections import defaultdict


class BigQueryPatentCollector:
  def __init__(self, project_id, gemini_api_key=None):
    self.client = bigquery.Client(project=project_id)
    self.collected_data = []
    self.seen_smiles = set()

    # Initialize Gemini for applications
    self.gemini_api_key = gemini_api_key
    self.gemini_model = None

    if self.gemini_api_key:
      try:
        import google.generativeai as genai
        genai.configure(api_key=self.gemini_api_key)
        self.gemini_model = genai.GenerativeModel("gemini-2.0-flash")
        print("✅ Gemini API configured for application extraction")
      except ImportError:
        print("⚠️ Gemini API library not found. Please install 'google-generativeai'")
        self.gemini_model = None
      except Exception as e:
        print(f"❌ Failed to configure Gemini API: {e}")
        self.gemini_model = None
    else:
      print("⚠️ Gemini API key not provided. Application extraction will not be available.")
    
    # Initialize T5 model for SMILES generation
    print("🔬 Loading chemical name→SMILES transformer model...")
    try:
      from transformers import pipeline
      self.chem_translator = pipeline(
          "text2text-generation", 
          model="GT4SD/multitask-text-and-chemistry-t5-base-augm"
      )
      print("✅ Chemical transformer model loaded successfully")
    except ImportError:
      print("❌ Transformers library not found. Please install with: pip install transformers torch")
      self.chem_translator = None
    except Exception as e:
      print(f"❌ Failed to load chemical transformer model: {e}")
      self.chem_translator = None
    
    # Chemical name extraction patterns
    self.chemical_patterns = [
            r'\b\d+[a-z]?-[a-z\-]+-\d+[a-z]?\b',  # IUPAC names
            r'\b[A-Z][a-z]*(?:\s+[a-z]+)*\s+\d+\b',  # "Compound 123"
            r'\b[A-Z]{2,}\s*-?\s*\d+\b',  # "ABC-123"
            r'\b(?:compound|drug|molecule|chemical)\s+\w+\b',  # "compound A"
            r'\b\w+(?:ine|anol|acid|ate|ide|ene|ane)\b',  # Chemical suffixes
        ]

    
  def collect_patent_data(self, target_count=2000):
    """Main method to collect patent chemical data"""
    print(f"🔍 Starting patent chemical data collection (target: {target_count})")
    
    query = self._get_chemical_patents_query(target_count * 2) # funciton 1
    query_job = self.client.query(query)
    results = query_job.result()

    print("🧪 Extracting chemicals and analyzing properties...")
    extracted_chemicals = []

    for row in results:
      if len(extracted_chemicals) >= target_count:
        break
      
      chemicals = self._extract_chemicals_from_text(row) # function 2
      if chemicals:
        extracted_chemicals.extend(chemicals)
      
      # Progress indicator
      if len(extracted_chemicals) % 25 == 0 and len(extracted_chemicals) > 0:
        print(f"    Progress: {len(extracted_chemicals)} chemicals extracted...")
    
    print(f"✅ Extracted {len(extracted_chemicals)} chemicals from patents")
    return self._clean_and_validate(extracted_chemicals, target_count)
      

  def _extract_chemicals_from_text(self, row):
    """Extract chemicals from patent text"""
    text = f"{row.title} {row.abstract}"
    potential_chemicals = self._extract_potential_chemicals(text) #

    if not potential_chemicals:
      return []
    
    # Process each chemical name to get SMILES and applications
    real_chemicals = []
    for chem_name in potential_chemicals[:3]:  # Limit to top 3 per patent
      chemical_data = self._process_chemical_with_ai(chem_name, row, text)
      if chemical_data:
        real_chemicals.append(chemical_data)
    
    return real_chemicals

      
  def _process_chemical_with_ai(self, chem_name, row, text):
    """Process chemical using AI models for SMILES and applications"""
    # BLACK BOX 1: Generate SMILES using T5 transformer
    smiles = self._get_chemical_smiles(chem_name)
    if not smiles or smiles in self.seen_smiles:
      return None
    
    self.seen_smiles.add(smiles)
    
    # BLACK BOX 2: Get applications using Gemini
    applications = self._get_chemical_application(chem_name, smiles, text, row)
    
    return {
      'smiles': smiles,
      'name': chem_name,
      'applications': applications,
      'source': 'bigquery_ai_patents',
      'publication_number': row.publication_number,
      'filing_date': str(row.filing_date) if row.filing_date else '',
      'compound_id': f"AI_{row.publication_number}_{chem_name[:8]}",
      'patent_title': row.title,
      'assignee': self._get_assignee_name(row)
    }

  # BLACK BOX 1: AI-Powered SMILES Generation
  def _get_chemical_smiles(self, chem_name):
    """Generate SMILES using T5 transformer model"""
    
    if not self.chem_translator:
      print(f"    ⚠️ Chemical translator not available for: {chem_name}")
      return None
    
    try:
      # Use T5 model for name→SMILES translation
      prompt = f"translate to SMILES: {chem_name}"
      result = self.chem_translator(
          prompt, 
          max_length=150, 
          num_return_sequences=1,
          do_sample=False
      )
      
      smiles = result[0]['generated_text'].strip()
      
      # Clean up common T5 output artifacts
      smiles = smiles.replace("translate to SMILES:", "").strip()
      smiles = smiles.replace("SMILES:", "").strip()
      
      # Validate SMILES with RDKit
      if smiles and len(smiles) > 1:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
          # Return canonical SMILES for consistency
          return Chem.MolToSmiles(mol)
            
    except Exception as e:
      print(f"    ❌ T5 translation failed for '{chem_name}': {e}")
    
    return None

  # BLACK BOX 2: AI-Powered Application Analysis
  def _get_chemical_application(self, chem_name, smiles, text, row):
    """Get chemical applications using Gemini AI"""
    
    if not self.gemini_model:
      return self._fallback_applications(text)
    
    try:
      prompt = f"""
      Analyze this chemical compound and determine its specific applications based on the patent context.

      Chemical Name: {chem_name}
      SMILES Structure: {smiles}
      Patent Title: {row.title}
      Patent Abstract: {text[:600]}...
      Company/Assignee: {self._get_assignee_name(row)}

      Based on the chemical structure and patent context, return a JSON list of specific applications.

      Examples:
      - For pharmaceuticals: ["pharmaceutical_analgesic", "anti_inflammatory_drug", "pain_relief_medication"]
      - For industrial: ["industrial_solvent", "chemical_intermediate", "polymer_precursor", "coating_agent"]
      - For agriculture: ["agricultural_herbicide", "plant_protection", "crop_enhancement"]

      Return ONLY a valid JSON list of 2-5 specific applications:
      """
      
      response = self.gemini_model.generate_content(prompt)
      applications = self._parse_gemini_applications(response.text)
      
      if applications and len(applications) > 0:
        return applications
      else:
        return self._fallback_applications(text)
        
    except Exception as e:
      print(f"    ⚠️ Gemini API error for {chem_name}: {e}")
      return self._fallback_applications(text)

  def _parse_gemini_applications(self, gemini_text):
    """Parse Gemini response to extract applications list"""
    try:
      # Look for JSON array in response
      json_match = re.search(r'\[(.*?)\]', gemini_text, re.DOTALL)
      if json_match:
        json_str = '[' + json_match.group(1) + ']'
        applications = json.loads(json_str)
        
        # Clean and validate applications
        clean_apps = []
        for app in applications:
          if isinstance(app, str) and len(app.strip()) > 0:
            clean_app = app.strip().lower().replace(' ', '_')
            if 3 <= len(clean_app) <= 50:  # Reasonable length
              clean_apps.append(clean_app)
        
        return clean_apps[:5] if clean_apps else ["unknown_application"]
    
    except json.JSONDecodeError:
      pass
    except Exception as e:
      print(f"    ⚠️ Error parsing Gemini response: {e}")
    
    # Fallback: try comma-separated parsing
    try:
      if ',' in gemini_text:
        apps = [app.strip().strip('"\'').lower().replace(' ', '_') 
               for app in gemini_text.split(',')]
        clean_apps = [app for app in apps if app and 3 <= len(app) <= 50]
        return clean_apps[:5] if clean_apps else ["unknown_application"]
    except:
      pass
    
    return ["unknown_application"]

  def _fallback_applications(self, text):
    """Simple keyword-based fallback for application detection"""
    text_lower = text.lower()
    
    applications = []
    
    # Pharmaceutical indicators
    if any(word in text_lower for word in ['drug', 'pharmaceutical', 'medicine', 'therapeutic', 'treatment']):
      applications.append('pharmaceutical_compound')
    
    # Industrial indicators  
    if any(word in text_lower for word in ['catalyst', 'solvent', 'industrial', 'manufacturing', 'process']):
      applications.append('industrial_chemical')
    
    # Agricultural indicators
    if any(word in text_lower for word in ['pesticide', 'herbicide', 'agricultural', 'crop', 'plant']):
      applications.append('agricultural_chemical')
      
    # Material/polymer indicators
    if any(word in text_lower for word in ['polymer', 'plastic', 'coating', 'material', 'composite']):
      applications.append('material_component')
    
    return applications if applications else ['general_chemical_compound']

  def _get_assignee_name(self, row):
    """Extract assignee/company name from patent"""
    if row.assignee_harmonized:
      for assignee in row.assignee_harmonized:
        if hasattr(assignee, 'name') and assignee.name:
          return assignee.name
    return "Unknown"

  def _extract_potential_chemicals(self, text):
    """Extract potential chemical names from patent text"""
    potential_chemicals = []
    
    # Use regex patterns to find chemical names
    for pattern in self.chemical_patterns:
      matches = re.findall(pattern, text, re.IGNORECASE)
      for match in matches:
        cleaned = match.strip()
        if 3 < len(cleaned) < 50 and cleaned not in potential_chemicals:
          potential_chemicals.append(cleaned)
    
    # Additional chemical name indicators
    chemical_indicators = [
      r'compound\s+([A-Z]\w*)',
      r'formula\s+([A-Z][A-Za-z0-9]+)',
      r'(\w+(?:ine|anol|acid|ate|ide))\s+(?:was|is|can|has)',
      r'synthesis\s+of\s+(\w+)',
      r'preparation\s+of\s+(\w+)',
      r'active\s+ingredient\s+(\w+)',
      r'therapeutic\s+agent\s+(\w+)',
      r'drug\s+(\w+)',
      r'molecule\s+([A-Z]\w*)'
    ]
    
    for pattern in chemical_indicators:
      matches = re.findall(pattern, text, re.IGNORECASE)
      potential_chemicals.extend([m.strip() for m in matches if m.strip()])
    
    # Return unique chemicals, limited to top 5
    return list(set(potential_chemicals))[:5]

  def _clean_and_validate(self, chemicals, target_count):
    """Clean and validate extracted chemical data"""
    if not chemicals:
      print("⚠️ No chemicals extracted")
      return pd.DataFrame()
    
    # Create DataFrame
    df = pd.DataFrame(chemicals)
    print(f"    Initial chemicals: {len(df)}")
    
    # Remove duplicates based on SMILES
    df.drop_duplicates(subset=['smiles'], keep='first', inplace=True)
    print(f"    After deduplication: {len(df)}")
    
    # Validate SMILES structures
    valid_chemicals = []
    invalid_count = 0
    
    for _, row in df.iterrows():
      try:
        mol = Chem.MolFromSmiles(row['smiles'])
        if mol is not None:
          # Add molecular properties
          row_dict = row.to_dict()
          row_dict['molecular_weight'] = Chem.Descriptors.MolWt(mol)
          row_dict['num_atoms'] = mol.GetNumHeavyAtoms()
          valid_chemicals.append(row_dict)
        else:
          invalid_count += 1
      except:
        invalid_count += 1
    
    if invalid_count > 0:
      print(f"    Removed {invalid_count} invalid SMILES structures")
    
    if valid_chemicals:
      final_df = pd.DataFrame(valid_chemicals)
      
      # Sample if we have too many
      if len(final_df) > target_count:
        final_df = final_df.sample(n=target_count, random_state=42)
        print(f"    Sampled down to target: {len(final_df)}")
      
      print(f"✅ Final validated dataset: {len(final_df)} chemicals")
      return final_df
    
    print("❌ No valid chemicals after validation")
    return pd.DataFrame()

  def _get_chemical_patents_query(self, limit):
    """BigQuery SQL to get patents with chemical content"""
    return f"""
        SELECT 
            publication_number,
            title,
            abstract,
            filing_date,
            assignee_harmonized,
            cpc
        FROM `patents-public-data.patents.publications` 
        WHERE 
            country_code = 'US'
            AND publication_date >= '2018-01-01'  -- Recent patents
            AND (
                -- Target patents with chemical content AND application context
                (REGEXP_CONTAINS(LOWER(abstract), r'\\b(?:compound|molecule|chemical|synthesis)\\s+\\w+\\b') 
                 AND REGEXP_CONTAINS(LOWER(abstract), r'\\b(?:treatment|therapy|use|application|method)\\b'))
                OR
                (REGEXP_CONTAINS(LOWER(title), r'\\b(?:synthesis|preparation|compound|derivative)\\b')
                 AND REGEXP_CONTAINS(LOWER(abstract), r'\\b(?:pharmaceutical|industrial|agricultural|cosmetic)\\b'))
                OR
                EXISTS(SELECT 1 FROM UNNEST(cpc) AS cpc_code WHERE 
                    cpc_code.code LIKE 'C07%' OR  -- Organic chemistry
                    cpc_code.code LIKE 'A61K%' OR -- Pharmaceutical preparations
                    cpc_code.code LIKE 'A01N%'    -- Agricultural chemicals
                )
            )
            AND abstract IS NOT NULL
            AND LENGTH(abstract) > 250  -- Ensure rich content
            AND LENGTH(title) > 20
        ORDER BY filing_date DESC
        LIMIT {limit}
        """


import pandas as pd
import numpy as np
import requests
import time
import urllib.parse
import os
from rdkit import Chem
from rdkit.Chem import AllChem # Not explicitly used, but often useful with RDKit

class RealWorldChemicalDataCollector:
    def __init__(self, delay=0.5, max_retries=3, gemini_api_key=None):
        self.delay = delay
        self.max_retries = max_retries
        self.collected_data = []
        self.seen_smiles = set()
        # API endpoints
        self.end_points = {
            'pubchem': 'https://pubchem.ncbi.nlm.nih.gov/rest/pug',
            'chembl': 'https://www.ebi.ac.uk/chembl/api/data',
            'drugbank': 'https://go.drugbank.com/releases/latest#open-data'
        }
        self.application_mapping = {
            'pharmaceutical': [
                'drug', 'medicine', 'therapeutic', 'antibiotic', 'antiviral', 
                'anticancer', 'analgesic', 'antidepressant', 'antifungal',
                'anti-inflammatory', 'antihypertensive', 'diabetes', 'cardiovascular',
                'neurological', 'psychiatric', 'oncology', 'immunosuppressant'
            ],
            'industrial': [
                'solvent', 'catalyst', 'reagent', 'intermediate', 'polymer',
                'plasticizer', 'stabilizer', 'lubricant', 'adhesive', 'coating',
                'chemical manufacturing', 'industrial process', 'extraction',
                'ethanol', 'acetone', 'toluene', 'acetic acid', 
            ],
            'agricultural': [
                'pesticide', 'herbicide', 'fungicide', 'insecticide', 'fertilizer',
                'plant growth', 'crop protection', 'veterinary', 'animal health',
                'glyphosate', 'atrazine', 
            ],
            'consumer': [
                'cosmetic', 'fragrance', 'flavor', 'food additive', 'preservative',
                'personal care', 'household', 'detergent', 'cleaning', 'shampoo',
                'benzoic acid', 'sorbic acid', 'sodium lauryl sulfate', 'limonene', 'vanillin' 
            ],
            'materials': [
                'semiconductor', 'electronic', 'optical', 'magnetic', 'ceramic',
                'composite', 'nanomaterial', 'crystal', 'metallic', 'polyethylene', 'silicon dioxide' 
            ],
            'energy': [
                'fuel', 'battery', 'solar', 'photovoltaic', 'fuel cell',
                'energy storage', 'biofuel', 'renewable energy', 'lithium cobalt oxide' 
            ]
        }
        self.gemini_model = None
        self.gemini_api_key = gemini_api_key
        self.gemini_consecutive_errors = 0
        self.max_gemini_consecutive_errors = 5 # After this many 429s, temporarily disable Gemini
        self.gemini_disabled_due_to_quota = False
        self.gemini_quota_message_printed = False


        if self.gemini_api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.gemini_api_key)
                self.gemini_model = genai.GenerativeModel("gemini-2.0-flash")
                print("✅ Gemini API configured successfully.")
            except ImportError:
                print("⚠️ Gemini API library not found. Please install 'google-generativeai'. Falling back to rule-based applications.")
                self.gemini_model = None
            except Exception as e:
                print(f"❌ Failed to configure Gemini API: {e}. Falling back to rule-based applications.")
                self.gemini_model = None
        else:
            print("ℹ️ Gemini API key not provided. Using rule-based application inference.")
            self.gemini_model = None

    
    def collect_complete_dataset(self, target_size=2000, output_file="real_chemical_data.csv"):
        collection_plan = {
            'drugbank_pharmaceuticals': 8000,
            'chembl_bioactive': 6000,
            'pubchem_commercial': 4000,
            'patent_mining': 2000
        }
        
        all_collected = []
        for source, target_count_for_source in collection_plan.items():
            print(f"\n--- Collecting from: {source} (target: {target_count_for_source}) ---")
            # Reset Gemini error state for each new source, in case quota resets or issue was temporary
            self.gemini_consecutive_errors = 0
            self.gemini_disabled_due_to_quota = False
            
            try:
                data = []
                """if source == "drugbank_pharmaceuticals":
                    data = self.collect_drugbank_data(target_count=target_count_for_source)
                elif source == "chembl_bioactive":
                    data = self.collect_chembl_data(target_count=target_count_for_source)"""
                if source == "pubchem_commercial":
                    data = self.collect_pubchem_data(target_count=target_count_for_source)
                """elif source == "patent_mining":
                    data = self.collect_patent_data(target_count=target_count_for_source)"""
                
                print(f"--- Collected {len(data)} raw entries from {source} ---")
                all_collected.extend(data)
            except Exception as e:
                print(f"❌ Error collecting from {source}: {e}")
                continue
        
        print(f"\n🔧 Processing and cleaning {len(all_collected):,} collected compounds...")
        final_dataset = self._process_and_clean_data(all_collected, target_size)

        if output_file and not final_dataset.empty:
            try:
                final_dataset.to_csv(output_file, index=False)
                print(f"💾 Saved dataset to {output_file}")
            except Exception as e:
                print(f"❌ Error saving dataset to {output_file}: {e}")
        elif final_dataset.empty:
            print("❌ No data to save. Final dataset is empty.")

        self._print_final_statistics(final_dataset)
        return final_dataset

    def collect_patent_data(self, target_count=2000):
        print(f"🔩 Simulating patent data collection (target: {target_count})...")
        industrial_compounds = [
            {'smiles': 'CCCCCCCCCCCCCCCCCC(=O)O', 'applications': ['surfactant', 'detergent', 'industrial'], 'name': 'Stearic Acid'},
            {'smiles': 'CC(C)(C)c1ccc(O)cc1', 'applications': ['antioxidant', 'stabilizer', 'industrial'], 'name': 'Butylated Hydroxytoluene (BHT)'},
            {'smiles': 'CCCCO', 'applications': ['solvent', 'chemical_intermediate', 'industrial'], 'name': 'n-Butanol'},
            {'smiles': 'CC(=O)OCC=C', 'applications': ['monomer', 'polymer_precursor', 'industrial'], 'name': 'Vinyl Acetate'},
            {'smiles': 'c1ccccc1C(=O)OOC(=O)c1ccccc1', 'applications': ['initiator', 'polymerization', 'industrial'], 'name': 'Benzoyl Peroxide'},
            {'smiles': 'CN(C)C=O', 'applications': ['solvent', 'reagent', 'industrial'], 'name': 'Dimethylformamide (DMF)'}, 
            {'smiles': 'C1CCCCC1', 'applications': ['solvent', 'raw_material', 'industrial'], 'name': 'Cyclohexane'},
            {'smiles': 'OCCO', 'applications': ['antifreeze', 'solvent', 'polyester_precursor', 'industrial'], 'name': 'Ethylene Glycol'},
        ]
        collected = []
        if not industrial_compounds: 
            return collected
            
        base_count = len(industrial_compounds)
        for i in range(target_count):
            base_comp_orig = industrial_compounds[i % base_count]
            base_comp = base_comp_orig.copy() 

            mol_check = Chem.MolFromSmiles(base_comp["smiles"])
            if mol_check is None: 
                print(f"      ⚠️ Invalid SMILES in patent_data sample: {base_comp['smiles']} for {base_comp['name']}. Skipping.")
                continue

            if base_comp["smiles"] not in self.seen_smiles:
                base_comp["source"] = "patent_mining"
                base_comp["compound_id"] = f'PATENT_SAMPLE_{i:06d}'
                if 'name' not in base_comp:
                    base_comp['name'] = f"Patent Compound {i}"
                collected.append(base_comp)
                self.seen_smiles.add(base_comp["smiles"])
        print(f"    🔩 Simulated {len(collected)} patent compounds.")
        return collected[:target_count]

    def collect_pubchem_data(self, target_count=4000):
        collected = []
        search_terms = [
            'solvent', 'detergent', 'preservative', 'antioxidant', 'surfactant',
            'plasticizer', 'fragrance', 'flavoring agent', 'food additive', 'cosmetic ingredient',
            'industrial catalyst', 'polymer precursor', 'agricultural chemical', 'laboratory reagent',
            'coating agent',
            'ethanol', 'acetone', 'toluene', 
            'sodium lauryl sulfate', 'cetyltrimethylammonium bromide', 
            'benzoic acid', 'sorbic acid', 'sodium benzoate', 'methylparaben', 
            'butylated hydroxytoluene', 'ascorbic acid', 
            'dioctyl phthalate', 'diethylhexyl phthalate', 
            'limonene', 'linalool', 'vanillin', 
            'monosodium glutamate', 'aspartame', 
            'titanium dioxide', 'zinc oxide', 
            'zeolite', 'palladium chloride', 
            'styrene', 'vinyl chloride', 'bisphenol A', 
            'glyphosate', 'atrazine', 'imidacloprid', 
            'hydrochloric acid', 'sodium chloride', 'agarose', 
            'polyurethane', 'epoxy resin' 
        ]
        if not search_terms: return collected
        
        compounds_per_term_target = (target_count // len(search_terms)) + 5 if len(search_terms) > 0 else target_count
        compounds_per_term_target = max(5, compounds_per_term_target) 

        print(f"🔍 Searching PubChem for commercial/industrial compounds (Overall target: {target_count})...")
        print(f"    Using {len(search_terms)} search terms, aiming for ~{compounds_per_term_target} compounds per term initially.")

        for term in search_terms:
            if len(collected) >= target_count: 
                print(f"    🎯 Overall PubChem target of {target_count} reached. Stopping search.")
                break
            
            remaining_needed = target_count - len(collected)
            current_term_fetch_limit = min(compounds_per_term_target, remaining_needed)
            if current_term_fetch_limit <= 0: 
                continue

            print(f"    🔎 Searching PubChem for: '{term}' (aiming for up to ~{current_term_fetch_limit} new compounds this term)")
            try:
                term_data = self._search_pubchem_by_term(term, current_term_fetch_limit)
                
                newly_added_for_term = 0
                for compound_data in term_data:
                    if len(collected) < target_count: 
                        if compound_data['smiles'] not in self.seen_smiles: 
                            collected.append(compound_data)
                            self.seen_smiles.add(compound_data['smiles'])
                            newly_added_for_term += 1
                    else:
                        break 

                print(f"    ✅ Processed term '{term}'. Added {newly_added_for_term} new compounds. Total PubChem collected: {len(collected)}")

            except Exception as e:
                print(f"      ❌ Error during PubChem search for '{term}': {e}")
                continue
        
        print(f"--- Collected {len(collected)} final entries from PubChem sources ---")
        return collected[:target_count] 
    
    def _search_pubchem_by_term(self, search_term, max_compounds_to_fetch_for_term):
        term_collected_internally = [] 
        try:
            safe_term = urllib.parse.quote(search_term)
            initial_cid_fetch_count = min(max_compounds_to_fetch_for_term * 3, 250)
            if initial_cid_fetch_count < 10: initial_cid_fetch_count = 10 

            listkey_url = f"{self.end_points['pubchem']}/compound/name/{safe_term}/cids/JSON?list_return=listkey&MaxRecords={initial_cid_fetch_count}"
            
            response = requests.get(listkey_url, timeout=30)
            time.sleep(self.delay) 

            if response.status_code != 200:
                if response.status_code == 404:
                     print(f"      ℹ️ PubChem CID search for '{search_term}' (name lookup) returned 404. Term not found as specific name/synonym.")
                else:
                     print(f"      ⚠️ PubChem CID search error for '{search_term}': {response.status_code} - {response.text.strip()}")
                     print(f"         Attempted URL: {listkey_url}")
                return term_collected_internally
            
            listkey_data = response.json()
            listkey = listkey_data.get("IdentifierList", {}).get("ListKey")
            
            if not listkey:
                print(f"      ℹ️ No ListKey found for '{search_term}' on PubChem (term might be too general or yield no results for name lookup).")
                return term_collected_internally

            props_url = (f"{self.end_points['pubchem']}/compound/listkey/{listkey}"
                         f"/property/CanonicalSMILES,Title/JSON")
            prop_response = requests.get(props_url, timeout=30)
            time.sleep(self.delay)

            if prop_response.status_code != 200:
                print(f"      ⚠️ PubChem property fetch error for '{search_term}' (ListKey {listkey}): {prop_response.status_code} - {prop_response.text.strip()}")
                return term_collected_internally
            
            props_data = prop_response.json()
            properties = props_data.get("PropertyTable", {}).get("Properties", [])

            count_added_this_call = 0
            for prop_entry in properties: 
                if count_added_this_call >= max_compounds_to_fetch_for_term: 
                    break

                smiles = prop_entry.get("CanonicalSMILES")
                cid = prop_entry.get("CID")
                name = prop_entry.get('Title', f"PubChem CID {cid}") 

                if not smiles: 
                    continue
                
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    continue

                applications = self._get_applications_via_gemini(smiles, name=name, context_hint=search_term)
                
                if not applications or "error_gemini_fallback" in str(applications) or "unknown_from_gemini" in str(applications):
                    final_applications = [search_term, self._map_hint_to_category(search_term)]
                    if "unknown_from_gemini" in str(applications): 
                         final_applications.append("unknown_application_confirmed_by_gemini")
                else:
                    final_applications = list(applications) 
                    final_applications.append(search_term) 
                    main_category = self._map_hint_to_category(search_term)
                    if main_category and main_category != "unknown": 
                        final_applications.append(main_category)
                
                term_collected_internally.append({
                    'smiles': smiles,
                    'applications': list(set(app for app in final_applications if app and app.strip())), 
                    'source': 'pubchem',
                    'compound_id': f"CID_{cid}",
                    'name': name
                })
                count_added_this_call += 1
                
        except requests.exceptions.Timeout:
            print(f"      ⏳ PubChem request timed out for '{search_term}'.")
        except requests.exceptions.RequestException as e:
            print(f"      ❌ PubChem network error for '{search_term}': {e}")
        except Exception as e:
            print(f"      ❌ Unexpected error in _search_pubchem_by_term for '{search_term}': {e}")
        
        return term_collected_internally

    def _map_hint_to_category(self, hint):
        if not hint: return "unknown"
        hint_lower = hint.lower()
        for category, keywords in self.application_mapping.items():
            if any(keyword in hint_lower for keyword in keywords):
                return category
        return "unknown" 
    
    def collect_chembl_data(self, target_count=6000):
        collected = []
        page_size = 100 
        max_pages_overall_cap = 100 
        if target_count > (max_pages_overall_cap * page_size) : 
             max_pages_to_fetch = (target_count // page_size) + 1
        else: 
             max_pages_to_fetch = min((target_count // page_size) + 1, max_pages_overall_cap)

        print(f"🔍 Fetching bioactive compounds from ChEMBL API (target: {target_count}, page_size: {page_size}, max_pages: {max_pages_to_fetch})...")
        
        current_retries = 0 

        for page_num in range(max_pages_to_fetch):
            if len(collected) >= target_count:
                print(f"    🎯 Target of {target_count} ChEMBL compounds reached.")
                break
            
            offset = page_num * page_size
            url = f"{self.end_points['chembl']}/molecule.json"
            params = {
                "limit": page_size,
                "offset": offset,
                'molecule_structures__canonical_smiles__isnull': 'false', 
                'max_phase__gte': 0 
            }

            try:
                response = requests.get(url, params=params, timeout=60) 
                time.sleep(self.delay) 

                if response.status_code == 200:
                    data = response.json()
                    molecules = data.get("molecules", [])
                    if not molecules and page_num > 0: 
                        print(f"    ℹ️ No more molecules found from ChEMBL API at page {page_num + 1}. Stopping ChEMBL collection.")
                        break

                    for mol_entry in molecules:
                        if len(collected) >= target_count: break 

                        smiles = mol_entry.get("molecule_structures", {}).get("canonical_smiles")
                        chembl_id = mol_entry.get('molecule_chembl_id', f'CHEMBL_UNKNOWN_{page_num}_{offset}')
                        pref_name = mol_entry.get('pref_name', chembl_id)

                        if not smiles: continue
                        if smiles in self.seen_smiles: continue

                        rdkit_mol = Chem.MolFromSmiles(smiles)
                        if rdkit_mol is None:
                            print(f"      ⚠️ Invalid SMILES '{smiles}' from ChEMBL for ID {chembl_id}. Skipping.")
                            continue
                        
                        applications = self._get_applications_via_gemini(smiles, name=pref_name, context_hint="bioactive, pharmaceutical research")
                        
                        if not applications or "error" in str(applications).lower() or "unknown_from_gemini" in str(applications):
                            final_applications = ["bioactive", "pharmaceutical_candidate", "research_compound"]
                            if "unknown_from_gemini" in str(applications):
                                final_applications.append("unknown_application_confirmed_by_gemini")
                        else:
                            final_applications = list(applications) 
                            final_applications.extend(["bioactive", "pharmaceutical_research"]) 
                        
                        collected.append({
                            'smiles': smiles,
                            'applications': list(set(app for app in final_applications if app and app.strip())),
                            'source': 'chembl',
                            'compound_id': chembl_id,
                            'name': pref_name
                        })
                        self.seen_smiles.add(smiles)
                    
                    current_retries = 0 
                    if (page_num + 1) % 10 == 0 or not molecules:
                        print(f"    📄 Processed {page_num + 1} ChEMBL pages, collected {len(collected)} unique compounds so far.")

                elif response.status_code == 429:  
                    print(f"    ⚠️ ChEMBL API rate limit hit on page {page_num + 1}. Retrying after longer delay...")
                    current_retries += 1
                    if current_retries > self.max_retries:
                        print(f"    ❌ Max retries reached for ChEMBL due to rate limiting. Stopping ChEMBL collection.")
                        break
                    time.sleep(self.delay * (5 ** current_retries)) 
                else:
                    print(f"    ⚠️ ChEMBL API error on page {page_num + 1}: {response.status_code} - {response.text.strip()}")
                    current_retries +=1
                    if current_retries > self.max_retries:
                         print(f"    ❌ Max retries reached for ChEMBL due to API errors. Stopping.")
                         break
                    time.sleep(self.delay * 3 * current_retries)


            except requests.exceptions.Timeout:
                print(f"    ⏳ ChEMBL request timed out on page {page_num + 1}.")
                current_retries += 1
                if current_retries > self.max_retries:
                    print(f"    ❌ Max network retries (timeout) reached for ChEMBL. Stopping ChEMBL collection.")
                    break
                time.sleep(self.delay * 5 * current_retries) 
            except requests.exceptions.RequestException as e:
                print(f"    ❌ Network error on ChEMBL page {page_num + 1}: {e}")
                current_retries += 1
                if current_retries > self.max_retries:
                    print(f"    ❌ Max network retries (general) reached for ChEMBL. Stopping ChEMBL collection.")
                    break
                time.sleep(self.delay * 5 * current_retries) 
            except Exception as e_gen:
                print(f"    ❌ Unexpected error during ChEMBL collection on page {page_num + 1}: {e_gen}")
                break 
        
        print(f"    📉 Finished ChEMBL collection. Total unique compounds: {len(collected)}")
        return collected[:target_count] 

    def collect_drugbank_data(self, target_count=8000):
        structure_file = "/content/open structures.sdf 3" 
        
        print(f"💊 Attempting to collect DrugBank data from '{structure_file}' (target: {target_count})...")
        
        if not os.path.exists(structure_file):
            print(f"⚠️ DrugBank SDF file '{structure_file}' not found at the specified path.")
            print("   To get real DrugBank data for non-commercial use:")
            print("   1. Register at https://go.drugbank.com/ (free for academic/non-commercial research).")
            print("   2. Download the 'Open Data' structure file (usually an SDF file).")
            print(f"   3. Place it at the path: '{structure_file}' or update the path in the script.")
            print("   Using sample pharmaceutical data instead as a fallback...")
            return self._create_sample_pharmaceutical_data(target_count)
        
        return self._parse_drugbank_sdf(structure_file, target_count)

    def _get_applications_via_gemini(self, smiles, name=None, context_hint=None):
        if self.gemini_model is None or self.gemini_disabled_due_to_quota:
            if self.gemini_disabled_due_to_quota and not self.gemini_quota_message_printed:
                # Print this message only once per source if Gemini gets disabled
                print("    ℹ️ Gemini API calls temporarily disabled due to repeated quota/rate limit errors. Using only fallback for applications.")
                self.gemini_quota_message_printed = True # Ensure it's printed once for this disabling event
            return self._get_applications_fallback(smiles, name, context_hint)
        
        prompt_parts = [f"The chemical compound has SMILES: {smiles}."]
        if name:
            prompt_parts.append(f"It is also known as or related to: '{name}'.")
        if context_hint:
            prompt_parts.append(f"A general hint for its use or source is: '{context_hint}'.")

        prompt_parts.append(
            "\nBased on this information, list its primary real-world applications or uses. "
            "Focus on established applications in areas like pharmaceutical, industrial, agricultural, consumer products, materials science, or energy. "
            "Return a concise, comma-separated list of 2-5 specific application keywords (e.g., analgesic, solvent, pesticide, fragrance, semiconductor, battery material). "
            "If applications are unknown or too broad, return 'unknown_application'." 
        )
        prompt = " ".join(prompt_parts)
        
        try:
            response = self.gemini_model.generate_content(prompt) 
            raw_applications = response.text.strip().lower()
            self.gemini_consecutive_errors = 0 # Reset error count on success

            if "unknown_application" in raw_applications or not raw_applications:
                applications = ["unknown_from_gemini"] 
            else:
                applications = [app.strip() for app in raw_applications.split(",") if app.strip()] 
            return applications if applications else ["unknown_from_gemini"] 

        except Exception as e:
            # Check if the error message string contains '429' or 'quota'
            error_str = str(e).lower()
            if "429" in error_str or "quota" in error_str or "rate limit" in error_str:
                self.gemini_consecutive_errors += 1
                print(f"  ⚠️ Gemini API quota/rate limit error ({self.gemini_consecutive_errors}/{self.max_gemini_consecutive_errors}): {e}. Using fallback.")
                if self.gemini_consecutive_errors >= self.max_gemini_consecutive_errors:
                    self.gemini_disabled_due_to_quota = True
                    self.gemini_quota_message_printed = False # Reset for the new disabling event
                    print(f"    ‼️ Too many consecutive Gemini API errors. Disabling Gemini calls for the current data source to avoid further quota issues.")
            else:
                # For other types of errors, just print and fallback
                print(f"  ❌ Gemini API error (non-quota): {e}. Using rule-based fallback.")
            
            return self._get_applications_fallback(smiles, name, context_hint, error_context="gemini_api_error")


    def _get_applications_fallback(self, smiles, name=None, context_hint=None, error_context=None):
        applications = []
        if error_context:
            applications.append(f"error_{error_context}_fallback")

        if context_hint:
            applications.append(context_hint.lower().replace(" ", "_")) 
            main_category = self._map_hint_to_category(context_hint)
            if main_category and main_category != "unknown":
                applications.append(main_category)
        
        if name:
            name_lower = name.lower()
            for category, keywords in self.application_mapping.items():
                if any(keyword in name_lower for keyword in keywords):
                    applications.append(category)
        
        if not applications or len(applications) <=1 : 
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    if mol.GetNumHeavyAtoms() > 0 : 
                        if 'N' in smiles and ('C(=O)' in smiles or 'c1' in smiles or 'n1' in smiles) and mol.GetNumHeavyAtoms() > 5:
                            applications.append('potential_bioactive')
                        elif 'O' in smiles and mol.GetNumHeavyAtoms() < 10 and mol.GetNumHeavyAtoms() > 1:
                            applications.append('small_organic_compound')
                        else:
                            applications.append('generic_organic_compound')
            except:
                applications.append('smiles_processing_error_in_fallback')

        return list(set(applications)) if applications else ['unknown_application_fallback']


    def _parse_drugbank_sdf(self, structure_file, target_count):
        collected = []
        print(f"    Parsing DrugBank SDF file: '{structure_file}'...")
        try:
            suppl = Chem.SDMolSupplier(structure_file, removeHs=False) 
            self.gemini_quota_message_printed = False # Reset for this source

            for mol_idx, mol in enumerate(suppl):
                if len(collected) >= target_count:
                    print(f"    🎯 Target of {target_count} DrugBank compounds reached.")
                    break
                
                if mol is None:
                    continue 
                
                try:
                    smiles = Chem.MolToSmiles(mol, isomericSmiles=True) 
                    if not smiles or smiles in self.seen_smiles:
                        continue
                    
                    props = mol.GetPropsAsDict()
                    drugbank_id = props.get("DRUGBANK_ID", props.get("DATABASE_ID", f"DB_UNKNOWN_{mol_idx:06d}"))
                    drug_name = props.get("GENERIC_NAME", props.get("COMMON_NAME", props.get("DRUG_NAME", drugbank_id)))
                    
                    if drug_name == drugbank_id: # If name is still the ID, try other common SDF fields
                        drug_name = props.get("SYNONYMS", "").split("\n")[0].split("|")[0].strip() or \
                                    props.get("SYSTEMATIC_NAME", "").split("\n")[0].split("|")[0].strip() or \
                                    props.get("INTERNATIONAL_BRAND_NAME", "").split("\n")[0].split("|")[0].strip() or \
                                    props.get("NAME", drugbank_id)


                    applications = self._get_applications_via_gemini(smiles, name=drug_name, context_hint="pharmaceutical, drug")
                    
                    if not applications or "error" in str(applications).lower() or "unknown_from_gemini" in str(applications):
                        final_applications = ["pharmaceutical", "drug", "therapeutic_agent"]
                        if "unknown_from_gemini" in str(applications):
                             final_applications.append("unknown_application_confirmed_by_gemini")
                    else:
                        final_applications = list(applications)
                        final_applications.extend(["pharmaceutical", "drug"])
                    
                    entry = {
                        'smiles': smiles,
                        'applications': list(set(app for app in final_applications if app and app.strip())),
                        'source': 'drugbank_sdf',
                        'compound_id': drugbank_id,
                        'name': drug_name if drug_name and drug_name != drugbank_id else f"DrugBank Compound {drugbank_id}",
                        'synonyms': props.get("SYNONYMS", ""),
                        'cas_number': props.get("CAS_NUMBER", ""),
                        'unii': props.get('UNII', '')
                    }
                    collected.append(entry)
                    self.seen_smiles.add(smiles)

                    if (mol_idx + 1) % 500 == 0:
                        print(f"    📊 Processed {mol_idx + 1} molecules from SDF, collected {len(collected)} unique compounds.")

                except Exception as e_mol:
                    print(f"      ⚠️ Error processing molecule {mol_idx} (ID: {props.get('DRUGBANK_ID', 'N/A') if 'props' in locals() else 'N/A'}) from SDF: {e_mol}")
                    continue 
        
        except Exception as e_file:
            print(f"❌ Error parsing DrugBank SDF file '{structure_file}': {e_file}")
            print("   Ensure the file is a valid SDF format and the path is correct.")
            print("   Falling back to sample pharmaceutical data if main parsing failed.")
            if not collected: 
                return self._create_sample_pharmaceutical_data(target_count)
        
        print(f"    💊 Finished parsing DrugBank SDF. Total unique compounds from SDF: {len(collected)}")
        return collected[:target_count] 

    def _process_and_clean_data(self, all_collected_raw, target_size):
        print(f"    🧹 Initial raw collected count: {len(all_collected_raw)}")
        if not all_collected_raw:
            print("    ⚠️ No data collected to process and clean.")
            return pd.DataFrame()

        df = pd.DataFrame(all_collected_raw)
        print(f"    DataFrame shape before any cleaning: {df.shape}") 

        df.dropna(subset=['smiles'], inplace=True)
        print(f"    DataFrame shape after dropping NA SMILES: {df.shape}")

        df.drop_duplicates(subset=['smiles'], keep='first', inplace=True)
        print(f"    DataFrame shape after drop_duplicates on SMILES: {df.shape}")
        
        def clean_applications(app_list):
            if isinstance(app_list, list):
                return sorted(list(set(str(app).strip().lower() for app in app_list if str(app).strip())))
            elif isinstance(app_list, str): 
                return sorted(list(set(str(app).strip().lower() for app in app_list.split(',') if str(app).strip())))
            return [] 
        
        df['applications'] = df['applications'].apply(clean_applications)

        print(f"    ✨ Cleaned unique compound count: {len(df)}")

        if len(df) > target_size:
            print(f"     Sampling {target_size} compounds from {len(df)} unique compounds.")
            return df.sample(n=target_size, random_state=42) 
        
        return df


    def _print_final_statistics(self, final_dataset):
        if final_dataset.empty:
            print("\n📊 Final dataset is empty. No statistics to show.")
            return

        print(f"\n📊 Final Dataset Statistics:")
        print(f"  Total compounds: {len(final_dataset)}")
        
        if 'source' in final_dataset.columns:
            print("\n  Compounds by source:")
            print(final_dataset['source'].value_counts())
        
        if 'applications' in final_dataset.columns:
            try:
                all_apps = final_dataset['applications'].explode()
                print("\n  Top 10 most common application keywords:")
                print(all_apps.value_counts().nlargest(10))
            except Exception as e:
                print(f"    Could not generate application statistics: {e}")
        
        if 'applications' in final_dataset.columns:
            avg_apps = final_dataset['applications'].apply(len).mean()
            print(f"\n  Average number of application keywords per compound: {avg_apps:.2f}")


    def _create_sample_pharmaceutical_data(self, target_count):
        print(f"    🏭 Generating {target_count} sample pharmaceutical compounds as fallback...")
        sample_drugs = [
            {'smiles': 'CC(=O)Oc1ccccc1C(=O)O', 'name': 'Aspirin', 'applications': ['analgesic', 'anti-inflammatory', 'pharmaceutical']},
            {'smiles': 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C', 'name': 'Caffeine', 'applications': ['stimulant', 'pharmaceutical', 'beverage_additive']}, 
            {'smiles': 'CC(C)NCC(O)c1ccc(O)c(CO)c1', 'name': 'Salbutamol', 'applications': ['bronchodilator', 'asthma_treatment', 'pharmaceutical']}, 
            {'smiles': 'CN(C)CCc1c[nH]c2ccc(C[C@H]3COC(=O)N3C)cc12', 'name': 'Sumatriptan', 'applications': ['migraine_treatment', 'serotonin_agonist', 'pharmaceutical']}, 
            {'smiles': 'COc1ccc2[nH]c(CCN(C)C)c3c2c1C(=O)N3', 'name': 'Melatonin', 'applications': ['sleep_aid', 'hormone_supplement', 'pharmaceutical']}, 
            {'smiles': 'CN1CCN(CC1)c1c(F)cc2c(c1F)c(=O)c(C(=O)O)cn2C2CC2', 'name': 'Ciprofloxacin', 'applications': ['antibiotic', 'fluoroquinolone', 'pharmaceutical']}, 
            {'smiles': 'CC(C)(O)CnC(C)(C)c1ccc(O)cc1O', 'name': 'Terbutaline', 'applications': ['bronchodilator', 'asthma_treatment', 'pharmaceutical']}, 
            {'smiles': 'CN1C2CCC1C(C(C2)OC(=O)C(CO)c3ccccc3)O', 'name': 'Atropine', 'applications': ['anticholinergic', 'mydriatic', 'pharmaceutical']}, 
            {'smiles': 'CCN(CC)CCNC(=O)c1c(N)ccc(Cl)c1', 'name': 'Procainamide', 'applications': ['antiarrhythmic_agent', 'cardiac_drug', 'pharmaceutical']}, 
            {'smiles': 'Cc1onc(c1)C(=O)Nc1ccc(cc1Cl)N1CCOCC1', 'name': 'Rivastigmine analog structure', 'applications': ['pharmaceutical_research', 'cholinesterase_inhibitor_scaffold', 'bioactive_molecule']}, 
        ]
        
        collected = []
        if not sample_drugs: return collected
        base_count = len(sample_drugs)

        for i in range(target_count):
            base_drug_orig = sample_drugs[i % base_count]
            base_drug = base_drug_orig.copy() 

            mol_check = Chem.MolFromSmiles(base_drug["smiles"])
            if mol_check is None: 
                print(f"      ⚠️ Invalid SMILES in sample_drugs: {base_drug['smiles']} for {base_drug['name']}. Skipping.")
                continue

            if base_drug["smiles"] not in self.seen_smiles:
                base_drug["source"] = "sample_pharmaceutical"
                base_drug["compound_id"] = f'SAMPLE_DRUG_{i:06d}'
                if 'name' not in base_drug:
                     base_drug['name'] = f"Sample Drug {i}"
                collected.append(base_drug)
                self.seen_smiles.add(base_drug["smiles"])
        
        print(f"    🏭 Generated {len(collected)} sample pharmaceutical compounds.")
        return collected[:target_count]




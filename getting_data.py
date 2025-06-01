import pandas as pd
import numpy as np
import requests
import time
import urllib.parse
import os
from rdkit import Chem
from rdkit.Chem import AllChem


class RealWorldChemicalDataCollector:
    def __init__(self, delay=0.5, max_retries=3, gemini_api_key=None):
        self.delay = delay
        self.max_retries = max_retries
        self.collected_data = []
        self.seen_smiles = set()
        # API endpoints (digital location where apis receive the requests)
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
                'chemical manufacturing', 'industrial process', 'extraction'
            ],
            'agricultural': [
                'pesticide', 'herbicide', 'fungicide', 'insecticide', 'fertilizer',
                'plant growth', 'crop protection', 'veterinary', 'animal health'
            ],
            'consumer': [
                'cosmetic', 'fragrance', 'flavor', 'food additive', 'preservative',
                'personal care', 'household', 'detergent', 'cleaning', 'shampoo'
            ],
            'materials': [
                'semiconductor', 'electronic', 'optical', 'magnetic', 'ceramic',
                'composite', 'nanomaterial', 'crystal', 'metallic'
            ],
            'energy': [
                'fuel', 'battery', 'solar', 'photovoltaic', 'fuel cell',
                'energy storage', 'biofuel', 'renewable energy'
            ]
        }
        self.gemini_model = None
        self.gemini_api_key = gemini_api_key
        if self.gemini_api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.gemini_api_key)
                # Fix: Correct parameter name for GenerativeModel
                self.gemini_model = genai.GenerativeModel("gemini-1.5-flash")
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
            'drugbank_pharmaceuticals': 8000,    # 40% - highest quality pharma
            'chembl_bioactive': 6000,            # 30% - bioactivity data
            'pubchem_commercial': 4000,          # 20% - commercial chemicals
            'patent_mining': 2000                # 10% - industrial applications
        }
        all_collected = []
        for source, target_count in collection_plan.items():
            try:
                if source == "drugbank_pharmaceuticals":
                    data = self.collect_drugbank_data()
                elif source == "chembl_bioactive":
                    data = self.collect_chembl_data()
                elif source == "pubchem_commercial":
                    data = self.collect_pubchem_data()
                elif source == "patent_mining":
                    data = self.collect_patent_data()
                
                all_collected.extend(data)
            except Exception as e:
                print(f"❌ Error collecting from {source}: {e}")
                continue
        
        print(f"\n🔧 Processing and cleaning {len(all_collected):,} collected compounds...")
        final_dataset = self._process_and_clean_data(all_collected, target_size)

        if output_file and not final_dataset.empty:
            final_dataset.to_csv(output_file, index=False)
            print(f"💾 Saved dataset to {output_file}")
        elif final_dataset.empty:
            print("❌ No data to save. Final dataset is empty.")

        self._print_final_statistics(final_dataset)
        return final_dataset

    def collect_patent_data(self, target_count=2000):
        industrial_compounds = [
            {'smiles': 'CCCCCCCCCCCCCCCCCC(=O)O', 'applications': ['surfactant', 'detergent', 'industrial'], 'name': 'Stearic Acid'},
            {'smiles': 'CC(C)(C)c1ccc(O)cc1', 'applications': ['antioxidant', 'stabilizer', 'industrial'], 'name': 'Butylated Hydroxytoluene (BHT)'},
            {'smiles': 'CCCCO', 'applications': ['solvent', 'chemical_intermediate', 'industrial'], 'name': 'n-Butanol'},
            {'smiles': 'CC(=O)OCC=C', 'applications': ['monomer', 'polymer_precursor', 'industrial'], 'name': 'Vinyl Acetate'},
            {'smiles': 'c1ccccc1C(=O)OOC(=O)c1ccccc1', 'applications': ['initiator', 'polymerization', 'industrial'], 'name': 'Benzoyl Peroxide'},
            {'smiles': 'CN(C)C(=O)H', 'applications': ['solvent', 'reagent', 'industrial'], 'name': 'Dimethylformamide (DMF)'},
            {'smiles': 'C1CCCCC1', 'applications': ['solvent', 'raw_material', 'industrial'], 'name': 'Cyclohexane'},
            {'smiles': 'OCCO', 'applications': ['antifreeze', 'solvent', 'polyester_precursor', 'industrial'], 'name': 'Ethylene Glycol'},
        ]
        collected = []
        base_count = len(industrial_compounds)
        for i in range(target_count):
            base_comp = industrial_compounds[i % base_count]
            mol_check = Chem.MolFromSmiles(base_comp["smiles"])
            if mol_check is None: 
                continue
            if base_comp["smiles"] not in self.seen_smiles:
                base_comp["source"] = "patent_mining"
                base_comp["compound_id"] = f'PATENT_SAMPLE_{i:06d}'
                collected.append(base_comp)
                self.seen_smiles.add(base_comp["smiles"])
        return collected[:target_count]

    def collect_pubchem_data(self, target_count=4000):
        collected = []
        search_terms = [
            'solvent', 'detergent', 'preservative', 'antioxidant', 'surfactant',
            'plasticizer', 'fragrance', 'flavoring agent', 'food additive', 'cosmetic ingredient',
            'industrial catalyst', 'polymer precursor', 'agricultural chemical', 'laboratory reagent',
            'coating agent'
        ]
        compounds_per_term_target = (target_count // len(search_terms)) + 10
        print(f"🔍 Searching PubChem for commercial/industrial compounds...")
        for term in search_terms:
            if len(collected) >= target_count: 
                break
            print(f"    🔎 Searching PubChem for: '{term}' (aiming for ~{compounds_per_term_target} compounds)")
            try:
                term_data = self._search_pubchem_by_term(term, compounds_per_term_target)
                collected.extend(term_data)
                print(f"    ✅ Found {len(term_data)} compounds for '{term}'. Total PubChem: {len(collected)}")
            except Exception as e:
                print(f"      ❌ Error searching PubChem for '{term}': {e}")
                continue
            if len(collected) >= target_count: 
                break
        return collected[:target_count]
    
    def _search_pubchem_by_term(self, search_term, max_results_per_term):
        term_collected = []
        try:
            safe_term = urllib.parse.quote(search_term)
            listkey_url = f"{self.end_points['pubchem']}/compound/name/{safe_term}/cids/JSON?list_return=listkey&MaxRecords={max_results_per_term * 2}"
            response = requests.get(listkey_url, timeout=30)
            time.sleep(self.delay)
            if response.status_code != 200:
                print(f"      ⚠️ PubChem CID search error for '{search_term}': {response.status_code} - {response.text}")
                return term_collected
            
            listkey_data = response.json()
            listkey = listkey_data.get("IdentifierList", {}).get("ListKey")
            if not listkey:
                print(f"      ℹ️ No ListKey found for '{search_term}' on PubChem.")
                return term_collected
            
            props_url = (f"{self.end_points['pubchem']}/compound/listkey/{listkey}"
                         f"/property/CanonicalSMILES,Title/JSON")
            prop_response = requests.get(props_url, timeout=30)
            time.sleep(self.delay)
            if prop_response.status_code != 200:
                print(f"      ⚠️ PubChem property fetch error for '{search_term}' (ListKey {listkey}): {prop_response.status_code} - {prop_response.text}")
                return term_collected
            
            props_data = prop_response.json()
            properties = props_data.get("PropertyTable", {}).get("Properties", [])

            for prop_entry in properties[:max_results_per_term]:
                smiles = prop_entry.get("CanonicalSMILES")
                cid = prop_entry.get("CID")
                name = prop_entry.get('Title', f"PubChem CID {cid}")

                if smiles and smiles not in self.seen_smiles:
                    applications = self._get_applications_via_gemini(smiles, name=name, context_hint=search_term)
                    
                    # Always ensure we have some applications
                    if not applications or "error_gemini_fallback" in str(applications):
                        applications = [search_term, self._map_hint_to_category(search_term)]
                    else:
                        applications.append(search_term)
                        main_category = self._map_hint_to_category(search_term)
                        if main_category and main_category != "unknown": 
                            applications.append(main_category)
                    
                    term_collected.append({
                        'smiles': smiles,
                        'applications': list(set([app for app in applications if app])),  # Remove empty and deduplicate
                        'source': 'pubchem',
                        'compound_id': f"CID_{cid}",
                        'name': name
                    })
                    self.seen_smiles.add(smiles)
                
                if len(term_collected) >= max_results_per_term:
                    break  
        except requests.exceptions.Timeout:
            print(f"      ⏳ PubChem request timed out for '{search_term}'.")
        except Exception as e:
            print(f"      ❌ Unexpected error in _search_pubchem_by_term for '{search_term}': {e}")
        return term_collected

    def _map_hint_to_category(self, hint):
        hint_lower = hint.lower()
        for category, keywords in self.application_mapping.items():
            if any(keyword in hint_lower for keyword in keywords):
                return category
        return "unknown"
    
    def collect_chembl_data(self, target_count=6000):
        collected = []
        page_size = 100
        max_pages_to_fetch = min((target_count // page_size) + 1, 70)
        print(f"🔍 Fetching bioactive compounds from ChEMBL API (max pages: {max_pages_to_fetch})...")
        retries = 0   

        for page in range(max_pages_to_fetch):
            if len(collected) >= target_count:
                break
            try:
                url = f"{self.end_points['chembl']}/molecule.json"
                params = {
                    "limit": page_size,
                    "offset": page * page_size,
                    'molecule_structures__canonical_smiles__isnull': 'false',
                    'max_phase__gte': 0
                }
                response = requests.get(url, params=params, timeout=60)

                if response.status_code == 200:
                    data = response.json()
                    molecules = data.get("molecules", [])
                    for mol in molecules:
                        smiles = mol.get("molecule_structures", {}).get("canonical_smiles")
                        if smiles and smiles not in self.seen_smiles:
                            applications = self._get_applications_via_gemini(smiles, context_hint="bioactive")
                            # Ensure we always have applications for ChEMBL compounds
                            if not applications or "error" in str(applications):
                                applications = ["bioactive", "pharmaceutical", "research"]
                            
                            collected.append({
                                'smiles': smiles,
                                'applications': applications,
                                'source': 'chembl',
                                'compound_id': mol.get('molecule_chembl_id', '')
                            })
                            self.seen_smiles.add(smiles)
                            if len(collected) >= target_count:
                                break
                    retries = 0 
                    if (page + 1) % 10 == 0:
                        print(f"    📄 Processed {page + 1} ChEMBL pages, collected {len(collected)} compounds")
                elif response.status_code == 429:  # Too Many Requests
                    print(f"    ⚠️ ChEMBL API rate limit hit on page {page + 1}. Retrying after longer delay...")
                    time.sleep(self.delay * 10 * (retries + 1))  # Exponential backoff
                    retries += 1
                    if retries > self.max_retries:
                        print(f"    ❌ Max retries reached for ChEMBL. Stopping ChEMBL collection.")
                        break
                    continue
                else:
                    print(f"    ⚠️ ChEMBL API error on page {page + 1}: {response.status_code} - {response.text}")
                time.sleep(self.delay)

            except requests.exceptions.RequestException as e:
                print(f"    ❌ Network error on ChEMBL page {page + 1}: {e}")
                if retries < self.max_retries:
                    time.sleep(self.delay * 5 * (retries + 1))  # Longer delay for network issues
                    retries += 1
                    continue
                else:
                    print(f"    ❌ Max network retries reached for ChEMBL. Stopping ChEMBL collection.")
                    break
            except Exception as e_gen:
                print(f"    ❌ Unexpected error on ChEMBL page {page + 1}: {e_gen}")
                break  # Stop ChEMBL on unexpected error
        return collected[:target_count]

    def collect_drugbank_data(self, target_count=8000):
        structure_file = "/content/open structures.sdf 3"  # Changed to .sdf extension
        
        # Check if file exists
        import os
        if not os.path.exists(structure_file):
            print(f"⚠️ DrugBank SDF file '{structure_file}' not found.")
            print("   To get real DrugBank data:")
            print("   1. Register at https://go.drugbank.com/ (free)")
            print("   2. Download 'All Structures' SDF file")
            print("   3. Save as 'drugbank_structures.sdf' in your working directory")
            print("   Using sample pharmaceutical data instead...")
            return self._create_sample_pharmaceutical_data(target_count)
        
        return self._parse_drugbank_sdf(structure_file, target_count)

    def _get_applications_via_gemini(self, smiles, name=None, context_hint=None):
        # Check if Gemini model is available
        if self.gemini_model is None:
            print(f"  ⚠️ Gemini API not available. Using rule-based fallback for: {smiles[:20]}...")
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
            "If applications are unknown or too broad, return 'unknown'."
        )
        prompt = " ".join(prompt_parts)
        print(f"  🧪 Calling Gemini API for: {smiles[:40]}... (Name: {name}, Hint: {context_hint})")
        try:
            response = self.gemini_model.generate_content(prompt)
            raw_applications = response.text.strip().lower()
            if "unknown" in raw_applications or not raw_applications:
                applications = ["unknown_from_gemini"]
            else:
                applications = [app.strip() for app in raw_applications.split(",")]
            print(f"  🤖 Gemini response: {applications}")
            return applications
        except Exception as e:
            print(f"  ❌ Gemini API error: {e}. Using rule-based fallback.")
            return self._get_applications_fallback(smiles, name, context_hint)

    def _get_applications_fallback(self, smiles, name=None, context_hint=None):
        """Rule-based fallback when Gemini API is not available"""
        applications = []
        
        # Use context hint if available
        if context_hint:
            applications.append(context_hint)
            main_category = self._map_hint_to_category(context_hint)
            if main_category and main_category != "unknown":
                applications.append(main_category)
        
        # Use name-based inference
        if name:
            name_lower = name.lower()
            for category, keywords in self.application_mapping.items():
                if any(keyword in name_lower for keyword in keywords):
                    applications.append(category)
                    break
        
        # Basic SMILES-based inference (very simple)
        if not applications:
            if 'N' in smiles and ('C(=O)' in smiles or 'c1' in smiles):
                applications = ['pharmaceutical', 'bioactive']
            elif 'O' in smiles and len(smiles) < 20:
                applications = ['solvent', 'industrial']
            else:
                applications = ['organic_compound', 'industrial']
        
        return applications or ['unknown_application']

    def _parse_drugbank_sdf(self, structure_file, target_count):
        collected = []
        try:
            suppl = Chem.SDMolSupplier(structure_file)

            for mol_idx, mol in enumerate(suppl):
                if mol is None:
                    continue
                try:
                    smiles = Chem.MolToSmiles(mol)
                    if not smiles or smiles in self.seen_smiles:
                        continue
                    props = mol.GetPropsAsDict()
                    drug_name = (props.get("COMMON_NAME", "") or props.get("SYNONYMS", "").split(",")[0] or f"Drug_{mol_idx}")
                    drugbank_id = (props.get("DRUGBANK_ID", "") or f"DB{mol_idx:05d}")
                    synonyms = props.get("SYNONYMS", "")
                    cas_number = props.get("CAS_NUMBER", "")
                    unii = props.get('UNII', '')
                    secondary_accessions = props.get('SECONDARY_ACCESSION_NUMBERS', '')

                    applications = self._get_applications_via_gemini(smiles, name=drug_name, context_hint="pharmaceutical")
                    # Ensure pharmaceutical compounds always have applications
                    if not applications or "error" in str(applications):
                        applications = ["pharmaceutical", "drug", "therapeutic"]
                    
                    if applications:
                        collected.append({
                            'smiles': smiles,
                            'applications': applications,
                            'source': 'drugbank_sdf',
                            'compound_id': drugbank_id,
                            'name': drug_name,
                            'synonyms': synonyms,
                            'cas_number': cas_number,
                            'unii': unii
                        })
                        self.seen_smiles.add(smiles)
                        if len(collected) >= target_count:
                            break
                        if mol_idx % 500 == 0 and mol_idx > 0:
                            print(f"    📊 Processed {mol_idx} molecules from SDF, collected {len(collected)} compounds")

                except Exception as e_mol:
                    print(f"      ⚠️ Error processing molecule {mol_idx} from SDF: {e_mol}")
                    continue
        except Exception as e:
            print(f"❌ Error parsing DrugBank SDF: {e}")
            print("   Falling back to sample pharmaceutical data.")
            return self._create_sample_pharmaceutical_data(target_count)
        
        return collected

    def _process_and_clean_data(self, all_collected, target_size):
        # Placeholder method - you'll need to implement this
        df = pd.DataFrame(all_collected)
        return df.head(target_size) if not df.empty else pd.DataFrame()

    def _print_final_statistics(self, final_dataset):
        # Placeholder method - you'll need to implement this
        if not final_dataset.empty:
            print(f"📊 Final dataset contains {len(final_dataset)} compounds")
        else:
            print("📊 No final statistics - dataset is empty")

    def _create_sample_pharmaceutical_data(self, target_count):
        """Create sample pharmaceutical data when DrugBank file is not available"""
        sample_drugs = [
            {'smiles': 'CC(=O)Oc1ccccc1C(=O)O', 'name': 'Aspirin', 'applications': ['analgesic', 'anti-inflammatory', 'pharmaceutical']},
            {'smiles': 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C', 'name': 'Caffeine', 'applications': ['stimulant', 'pharmaceutical', 'beverage']},
            {'smiles': 'CC(C)NCC(c1ccc(O)c(CO)c1)O', 'name': 'Salbutamol', 'applications': ['bronchodilator', 'asthma', 'pharmaceutical']},
            {'smiles': 'CN(C)CCc1c[nH]c2ccc(C[C@H]3COC(=O)N3)cc12', 'name': 'Sumatriptan', 'applications': ['migraine', 'pharmaceutical', 'neurology']},
            {'smiles': 'COc1ccc2[nH]c3c(c2c1)CCN(C)C3', 'name': 'Melatonin', 'applications': ['sleep_aid', 'hormone', 'pharmaceutical']},
            {'smiles': 'CN1CCN(CC1)c1c(F)cc2c(=O)c(C(=O)O)cn(c3ccc(F)cc3)c2n1', 'name': 'Ciprofloxacin', 'applications': ['antibiotic', 'antimicrobial', 'pharmaceutical']},
            {'smiles': 'CC(C)(C)NCC(c1ccc(O)c(CO)c1)O', 'name': 'Terbutaline', 'applications': ['bronchodilator', 'pharmaceutical', 'respiratory']},
            {'smiles': 'CN1C2CCC1CC(C2)OC(=O)C(c1ccccc1)(c1ccccc1)CO', 'name': 'Atropine', 'applications': ['anticholinergic', 'pharmaceutical', 'ophthalmology']},
            {'smiles': 'CCN(CC)CCNC(=O)c1cc(Cl)ccc1N', 'name': 'Procainamide', 'applications': ['antiarrhythmic', 'cardiac', 'pharmaceutical']},
            {'smiles': 'Cc1oncc1C(=O)Nc1ccc(N2CCOCC2)c(Cl)c1', 'name': 'Isoxazole derivative', 'applications': ['pharmaceutical', 'research', 'bioactive']},
        ]
        
        collected = []
        base_count = len(sample_drugs)
        for i in range(target_count):
            base_drug = sample_drugs[i % base_count].copy()
            mol_check = Chem.MolFromSmiles(base_drug["smiles"])
            if mol_check is None: 
                continue
            if base_drug["smiles"] not in self.seen_smiles:
                base_drug["source"] = "sample_pharmaceutical"
                base_drug["compound_id"] = f'SAMPLE_DRUG_{i:06d}'
                collected.append(base_drug)
                self.seen_smiles.add(base_drug["smiles"])
        
        print(f"    📊 Generated {len(collected)} sample pharmaceutical compounds")
        return collected[:target_count]

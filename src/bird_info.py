import json
from pathlib import Path
from typing import Dict, Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BIRD_INFO_PATH = PROJECT_ROOT / "data" / "bird_info.json"


def load_bird_database() -> Dict[str, Dict[str, Any]]:
    """Load the 15-species bird knowledge base JSON file."""
    if not BIRD_INFO_PATH.exists():
        raise FileNotFoundError(f"Bird info database not found at {BIRD_INFO_PATH}")

    with open(BIRD_INFO_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def get_bird_info(species_name: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve information dictionary for a given bird species name.
    Falls back gracefully if exact match isn't found.
    """
    db = load_bird_database()
    
    # Exact match
    if species_name in db:
        return db[species_name]
        
    # Case-insensitive search
    for name, data in db.items():
        if name.lower() == species_name.lower():
            return data
            
    # Generic fallback
    return {
        "common_name": species_name,
        "scientific_name": "Aves spp.",
        "short_description": f"Information for {species_name} is currently being updated in the species database.",
        "habitat": "Varied terrestrial and aquatic habitats across India.",
        "geographic_distribution": "Indian Subcontinent.",
        "diet": "Insects, seeds, small fruits, or small aquatic life.",
        "identification_features": "Distinctive plumages and morphological traits unique to the species.",
        "behaviour": "Diurnal foraging and species-specific vocalizations.",
        "conservation_status": "Least Concern (IUCN)",
        "interesting_facts": "Part of the 15 selected Indian bird species in this deep learning study."
    }

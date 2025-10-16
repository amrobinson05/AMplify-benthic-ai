# components/species_info.py

species_info = {
    "Scallop": {
        "Habitat": "Sandy or gravel seafloor, often near tidal currents",
        "Depth Range": "5–100 m",
        "Fun Fact": "Scallops can swim by clapping their shells rapidly!",
        "Description": "Bivalve mollusks with fan-shaped shells that rest on the seafloor and filter-feed plankton."
    },
    "Crab": {
        "Habitat": "Rocky reefs, sandy bottoms, and seagrass beds",
        "Depth Range": "0–300 m",
        "Fun Fact": "Crabs communicate by drumming or waving their claws.",
        "Description": "Decapod crustaceans with hard exoskeletons, two pincers, and sideways movement."
    },
    "Eel": {
        "Habitat": "Burrows in sand or mud; rocky crevices",
        "Depth Range": "10–400 m",
        "Fun Fact": "Eels can swim both forwards and backwards!",
        "Description": "Elongated fish with snake-like bodies that hide in sediment or rocks, emerging to hunt at night."
    },
    "Flatfish": {
        "Habitat": "Sandy or muddy seafloor, often near estuaries",
        "Depth Range": "0–200 m",
        "Fun Fact": "Flatfish are born symmetrical but one eye migrates to the other side as they mature.",
        "Description": "Bottom-dwelling fish that camouflage perfectly with sediment, lying flat on one side."
    },
    "Roundfish": {
        "Habitat": "Open water above reefs or rocky areas",
        "Depth Range": "5–250 m",
        "Fun Fact": "Roundfish have a more cylindrical shape, unlike flatfish or skates.",
        "Description": "Typical fish-shaped species with a lateral line, adapted for swimming in open water."
    },
    "Skate": {
        "Habitat": "Soft sediment and sandy seabeds",
        "Depth Range": "20–500 m",
        "Fun Fact": "Skates are related to rays and lay egg cases called 'mermaid’s purses'.",
        "Description": "Flat-bodied cartilaginous fish with long tails and large pectoral fins used for gliding along the seafloor."
    },
    "Whelk": {
        "Habitat": "Cold, shallow waters with sand or mud substrate",
        "Depth Range": "0–200 m",
        "Fun Fact": "Whelks drill holes in shells of prey using a toothed tongue called a radula.",
        "Description": "Predatory sea snails with spiral shells that feed on bivalves and other invertebrates."
    }
}


def _normalize(name: str) -> str:
    if not name:
        return ""
    return (
        name.replace("Predicted:", "")
            .replace("predicted", "")
            .replace("(", "")
            .replace(")", "")
            .replace("_", "")
            .replace("-", "")
            .strip()
            .lower()
    )


# canonical map: "crab" -> "Crab"
_CANONICAL = { _normalize(k): k for k in species_info.keys() }


def _label_to_key(name: str):
    n = _normalize(name)
    if not n:
        return None

    # exact normalized match
    if n in _CANONICAL:
        return _CANONICAL[n]

    # singularization (crabs -> crab)
    if n.endswith("s") and n[:-1] in _CANONICAL:
        return _CANONICAL[n[:-1]]

    # contains/partial
    for nk, canonical in _CANONICAL.items():
        if nk in n or n in nk:
            return canonical

    return None


def get_species_info(species_name: str):
    key = _label_to_key(species_name)
    return species_info.get(key) if key else None

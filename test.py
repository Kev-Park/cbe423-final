from mp_api.client import MPRester

API_KEY = "1QSwm6EKie1DRhA9B3hDU4mgGO1PDPWY"

with MPRester(API_KEY) as mpr:
    print(mpr.materials.summary.available_fields)
    # Fetch thermo data for a single known ID
    results = mpr.materials.thermo.search(
        material_ids=["mp-1112148"],
        fields=["material_id", "energy_above_hull", "decomposition_enthalpy"]
    )
    for r in results:
        print(r.material_id, r.energy_above_hull, r.decomposition_enthalpy)
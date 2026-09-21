from __future__ import annotations

import ncstyle as S

SHORT = {
    "european_lv_urban_35297": "LV urban 35297",
    "european_lv_urban_8087": "LV urban 8087",
    "european_lv_rural_2731": "LV rural 2731",
    "goiener_smart_meters": "Goiener",
    "low_carbon_london": "Low Carbon London",
    "smart_grid_smart_city": "SGSC",
    "norway_ami_energy_distribution": "Norway AMI",
    "irish_domestic_smart_meters": "Irish CER",
    "camsl_japan_smart_meters": "CAMSL Japan",
    "danish_smart_heat_meters": "Danish heat",
    "heapo_heat_pumps": "HEAPO",
    "bdg2_building_data_genome": "BDG2",
    "bdg1_building_data_genome": "BDG1",
    "complete_energy_community": "Energy community",
    "opsd_household_data": "OPSD",
}

ALIAS = {
    "lv_urban_35297": "european_lv_urban_35297",
    "lv_urban_8087": "european_lv_urban_8087",
    "lv_rural_2731": "european_lv_rural_2731",
    "goiener": "goiener_smart_meters",
    "low_carbon_london": "low_carbon_london",
    "sgsc": "smart_grid_smart_city",
    "norway_ami": "norway_ami_energy_distribution",
    "irish": "irish_domestic_smart_meters",
    "camsl_japan": "camsl_japan_smart_meters",
    "danish_smart_heat_meters": "danish_smart_heat_meters",
    "heapo_heat_pumps": "heapo_heat_pumps",
    "bdg2": "bdg2_building_data_genome",
    "bdg1": "bdg1_building_data_genome",
    "energy_comm": "complete_energy_community",
    "opsd": "opsd_household_data",
}

CLASS = {
    "european_lv_urban_35297": "Feeder",
    "european_lv_urban_8087": "Feeder",
    "european_lv_rural_2731": "Feeder",
    "goiener_smart_meters": "Household meter",
    "low_carbon_london": "Household meter",
    "smart_grid_smart_city": "Household meter",
    "norway_ami_energy_distribution": "Household meter",
    "irish_domestic_smart_meters": "Household meter",
    "camsl_japan_smart_meters": "Household meter",
    "opsd_household_data": "Household meter",
    "danish_smart_heat_meters": "Thermal",
    "heapo_heat_pumps": "Thermal",
    "bdg1_building_data_genome": "Building & community",
    "bdg2_building_data_genome": "Building & community",
    "complete_energy_community": "Building & community",
}
CLASS_ORDER = ["Feeder", "Household meter", "Thermal", "Building & community"]
CLASS_SHORT = {"Feeder": "Feeder", "Household meter": "Meter", "Thermal": "Thermal",
               "Building & community": "Building"}
CLASS_COLOUR = {"Feeder": S.C1, "Household meter": S.C3, "Thermal": S.C5,
                "Building & community": S.C4}

CLASS_FAN = {"Feeder": (0.020, 0.100), "Household meter": (0.032, 0.135),
             "Thermal": (0.0, 0.085), "Building & community": (0.026, 0.100)}


def _dataset_colours():
    import colorsys

    out, key = {}, {}
    for cls in CLASS_ORDER:
        members = [d for d in CLASS if CLASS[d] == cls]
        k = len(members)
        base = CLASS_COLOUR[cls]
        r, g, b = (int(base[i:i + 2], 16) / 255 for i in (1, 3, 5))
        _, l, _ = colorsys.rgb_to_hls(r, g, b)
        amp_h, amp_l = CLASS_FAN[cls]
        bias = -0.03 - 0.6 * max(0.0, l - 0.52)
        for i, d in enumerate(members):
            t = 0.0 if k == 1 else (i / (k - 1) - 0.5)
            tgt = min(0.615, max(0.17, l + 2 * amp_l * t + bias))
            out[d] = S.shade(base, dl=tgt - l, dh=2 * amp_h * t,
                             ds=0.06 if abs(t) > 0.3 else 0.0)
        key[cls] = out[members[k // 2]]
    return out, key


DS_COLOUR, CLASS_KEY = _dataset_colours()


def canon(name):
    return ALIAS.get(name, name)


def col(dataset):
    return DS_COLOUR[canon(dataset)]

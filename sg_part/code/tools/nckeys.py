"""数据集语义短名和跨图统一颜色。"""

DATASET_LABELS = {
    "bdg1_building_data_genome": "BDG1",
    "bdg2_building_data_genome": "BDG2",
    "complete_energy_community": "CEC",
    "danish_smart_heat_meters": "Danish",
    "european_lv_rural_2731": "EU-Rural",
    "european_lv_urban_35297": "EU-35k",
    "goiener_smart_meters": "GoiEner",
    "heapo_heat_pumps": "HEAPO",
    "low_carbon_london": "LCL",
    "norway_ami_energy_distribution": "Norway",
    "camsl_japan_smart_meters": "CAMSL",
    "european_lv_urban_8087": "EU-8k",
    "irish_domestic_smart_meters": "Irish",
    "opsd_household_data": "OPSD",
    "smart_grid_smart_city": "SGSC",
    "nextgen_device_days": "NextGen",
    "data2_charging_sessions": "data2",
}

# 同一类别内使用相近色值，保证跨图颜色稳定且可区分。
DATASET_COLOURS = {
    "european_lv_urban_35297": "#1F3F51",
    "european_lv_urban_8087": "#315271",
    "european_lv_rural_2731": "#3A619B",
    "goiener_smart_meters": "#328481",
    "low_carbon_london": "#389294",
    "smart_grid_smart_city": "#45969E",
    "norway_ami_energy_distribution": "#4C9FAE",
    "irish_domestic_smart_meters": "#5AA3B7",
    "camsl_japan_smart_meters": "#64A9C5",
    "opsd_household_data": "#70AAC9",
    "danish_smart_heat_meters": "#CC681B",
    "heapo_heat_pumps": "#E89351",
    "bdg1_building_data_genome": "#574D7B",
    "bdg2_building_data_genome": "#786893",
    "complete_energy_community": "#987FAF",
    "nextgen_device_days": "#A36A8B",
    "data2_charging_sessions": "#6689A8",
}

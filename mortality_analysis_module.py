import pandas as pd
import numpy as np

## Relative risks
RR_PM25 = 1.15      #1.15
RR_PM25_high = 1.25
RR_PM25_low = 1.05
RR_Chen = 1.07
RR_Chen_high = 1.11
RR_Chen_low = 1.03
RR_NO2 = 1.023
RR_NO2_high = 1.037
RR_NO2_low = 1.008
RR_Huang = 1.02
RR_Huang_high = 1.04
RR_Huang_low = 1.01
print(f'loaded defined RR values')

# Function that calculates the mean concentration
def moy_conc(conc_points):
    mean_conc = conc_points['conc'].mean(skipna=True)
    return mean_conc

def moy_meanconc(donnees_expo):
    mean = donnees_expo['meanconc'].mean(skipna=True)
    return mean

def moy_meandelta(donnees_expo):
    mean = donnees_expo['meandelta'].mean(skipna=True)
    return mean

# Function to calculate the population-weighted average exposure
def expo_ponderee_meanconc(donnees_expo, popannee):
    pop_col = popannee
    expo = (donnees_expo['meanconc'] * donnees_expo[pop_col]).sum(skipna=True) / donnees_expo[pop_col].sum(skipna=True)
    return expo

def expo_ponderee_meandelta(donnees_expo, popannee):
    pop_col = popannee
    expo = (donnees_expo['meandelta'] * donnees_expo[pop_col]).sum(skipna=True) / donnees_expo[pop_col].sum(skipna=True)
    return expo

print('Successfully loaded mean conc command')

def mortalite_age_try(donnees_merged, donnees_expo, annee, pol):
    annee = int(annee)
    donnees_expo = donnees_expo.drop(columns='geometry', errors='ignore')

    required_columns = ["iriscod", "meanconc", "meandelta"]
    merged_data = pd.merge(donnees_merged, donnees_expo[required_columns], on="iriscod")

    RR_values = {
        "ug_PM25_RH50": RR_PM25,
        "ug_PM25_RH50_high": RR_PM25_high,
        "ug_PM25_RH50_low": RR_PM25_low,
        "ug_NO2": RR_NO2,
        "ug_NO2_high": RR_NO2_high,
        "ug_NO2_low": RR_NO2_low
    }

    if pol not in RR_values:
        raise ValueError(f"Unrecognized pollutant '{pol}'.")

    RR = RR_values[pol]

    # Exposure-adjusted avoided mortality
    merged_data[f"mortpol{annee}"] = merged_data[f"mort{annee}"] * (
        1 - np.exp(-np.log(RR) * merged_data["meandelta"] / 10)
    )

    esp_dict = {2019: 80, 2030: 84, 2050: 86}
    esp = esp_dict.get(annee)
    if esp is None:
        raise ValueError(f"Year '{annee}' not recognized. Choose from {list(esp_dict.keys())}.")

    popannee = f"pop{annee}"
    iris_interet = merged_data[(merged_data['age'] >= 30) & (merged_data['age'] <= 99)]
    mort_interet = iris_interet[f"mortpol{annee}"].sum(skipna=True)

    donnees = pd.DataFrame({
        'age': np.arange(30, 100),
        'mort': np.zeros(70),
        'annees_gagnees': np.zeros(70),
        'tot_mort_30_99': mort_interet,
        'taux_initial': np.zeros(70),
        'taux_corrige': np.zeros(70),
        'conc': moy_meanconc(donnees_expo),
        'conc_ponderee': expo_ponderee_meanconc(donnees_expo, popannee),
        'delta_conc': moy_meandelta(donnees_expo),
        'delta_conc_ponderee': expo_ponderee_meandelta(donnees_expo, popannee)
    })

    # Age-specific avoided mortality and life years gained
    for age in range(30, 100):
        iris_age = merged_data[merged_data['age'] == age]
        total_pop = iris_age[popannee].sum(skipna=True)

        age_specific_mort = iris_age[f"mortpol{annee}"].sum(skipna=True)
        donnees.loc[donnees['age'] == age, 'mort'] = age_specific_mort

        if total_pop > 0:
            taux_initial = iris_age[f"mort{annee}"].sum(skipna=True) / total_pop
            taux_corrige = age_specific_mort / total_pop
        else:
            taux_initial = taux_corrige = 0

        donnees.loc[donnees['age'] == age, ['taux_initial', 'taux_corrige']] = taux_initial, taux_corrige
        donnees.loc[donnees['age'] == age, 'annees_gagnees'] = age_specific_mort * (esp - age)

    # Total life years gained between age 30 and expected lifespan
    year_bounds = {2019: (0, 51), 2030: (0, 55), 2050: (0, 57)}
    start, end = year_bounds[annee]
    somme_annees = donnees.loc[start:end, 'annees_gagnees'].sum(skipna=True)
    donnees['tot_annees_gagnees_30_esp'] = somme_annees

    # Percent of 2019 baseline mortality that is avoided
    baseline_mortality = merged_data.loc[
        (merged_data['age'] >= 30) & (merged_data['age'] <= 99), f"mort{2019}"
    ].sum(skipna=True)
    #mean_PM25_2019 = 9.32 # population-weighted
    #AF = 1 - np.exp(-np.log(RR) * mean_PM25_2019 / 10)
    #PM25_attributable_mortality_2019 = baseline_mortality * AF

    if baseline_mortality > 0:
        percent_avoided_mortality = (mort_interet / baseline_mortality) * 100
    else:
        percent_avoided_mortality = np.nan

    donnees['baseline_mortality'] = baseline_mortality
    donnees['percent_avoided_mortality'] = percent_avoided_mortality

    return donnees.sort_values(by='age').reset_index(drop=True)


#Function to calculate mortality avoided
def mortalite_age(donnees_merged, donnees_expo, annee, pol):
    annee = int(annee)
    donnees_expo = donnees_expo.drop(columns='geometry', errors='ignore')

    required_columns = ["iriscod", "meanconc", "meandelta"]
    merged_data = pd.merge(donnees_merged, donnees_expo[required_columns], on="iriscod")

    RR_values = {
        "ug_PM25_RH50": RR_PM25,
        "ug_PM25_RH50_high": RR_PM25_high,
        "ug_PM25_RH50_low": RR_PM25_low,
        "ug_NO2": RR_NO2,
        "ug_NO2_high": RR_NO2_high,
        "ug_NO2_low": RR_NO2_low
    }

    if pol not in RR_values:
        raise ValueError(f"Unrecognized pollutant '{pol}'.")

    RR = RR_values[pol]

    # Mortality with delta concentration compared to 2019
    merged_data[f"mortpol{annee}"] = merged_data[f"mort{annee}"] - (
            merged_data[f"mort{annee}"] * np.exp(-np.log(RR) * merged_data["meandelta"] / 10)
    )

    esp_dict = {2019: 80, 2030: 84, 2050: 86}
    esp = esp_dict.get(annee)
    if esp is None:
        raise ValueError(f"Year '{annee}' not recognized. Choose from {list(esp_dict.keys())}.")

    popannee = f"pop{annee}"
    iris_interet = merged_data[(merged_data['age'] >= 30) & (merged_data['age'] <= 99)]

    mort_interet = iris_interet[f"mortpol{annee}"].sum(skipna=True)

    donnees = pd.DataFrame({
        'age': np.arange(30, 100),
        'mort': np.zeros(70),
        'annees_gagnees': np.zeros(70),
        'tot_mort_30_99': mort_interet,
        'taux_initial': np.zeros(70),
        'taux_corrige': np.zeros(70),
        'conc': moy_meanconc(donnees_expo),
        'conc_ponderee': expo_ponderee_meanconc(donnees_expo, popannee),
        'delta_conc': moy_meandelta(donnees_expo),
        'delta_conc_ponderee': expo_ponderee_meandelta(donnees_expo, popannee)
    })

    # Age-specific mortality and years gained
    for age in range(30, 100):
        iris_age = merged_data[merged_data['age'] == age]
        total_pop = iris_age[popannee].sum(skipna=True)

        age_specific_mort = iris_age[f"mortpol{annee}"].sum(skipna=True)
        donnees.loc[donnees['age'] == age, 'mort'] = age_specific_mort

        if total_pop > 0:
            taux_initial = iris_age[f"mort{annee}"].sum(skipna=True) / total_pop
            taux_corrige = (iris_age[f"mort{annee}"].sum(skipna=True) - age_specific_mort) / total_pop
        else:
            taux_initial = taux_corrige = 0

        donnees.loc[donnees['age'] == age, ['taux_initial', 'taux_corrige']] = taux_initial, taux_corrige
        donnees.loc[donnees['age'] == age, 'annees_gagnees'] = age_specific_mort * (esp - age)

    # Total years gained between ages 30 and life expectancy
    year_bounds = {2019: (0, 51), 2030: (0, 55), 2050: (0, 57)}
    start, end = year_bounds[annee]
    somme_annees = donnees.loc[start:end, 'annees_gagnees'].sum(skipna=True)
    donnees['tot_annees_gagnees_30_esp'] = somme_annees

    return donnees.sort_values(by='age').reset_index(drop=True)

# Function that calculates mortality avoided by age for alternative RRs
def mortalite_age2(donnees_merged, donnees_expo, annee, pol):
        # Ensure 'annee' is treated as an integer
        annee = int(annee)

        donnees_expo = donnees_expo.drop(columns='geometry')
        merged_data = pd.merge(donnees_merged, donnees_expo[['iriscod', 'meanconc', 'meandelta']], on='iriscod')

        # Calculate mortality for each age
        if pol == "ug_PM25_Chen":
            merged_data[f"mortpol{annee}"] = merged_data[f"mort{annee}"] - merged_data[f"mort{annee}"] * np.exp(
                -np.log(RR_Chen) * merged_data["meandelta"] / 10)
        elif pol == "ug_PM25_Chen_high":
            merged_data[f"mortpol{annee}"] = merged_data[f"mort{annee}"] - merged_data[f"mort{annee}"] * np.exp(
                -np.log(RR_Chen_high) * merged_data["meandelta"] / 10)
        elif pol == "ug_PM25_Chen_low":
            merged_data[f"mortpol{annee}"] = merged_data[f"mort{annee}"] - merged_data[f"mort{annee}"] * np.exp(
                -np.log(RR_Chen_low) * merged_data["meandelta"] / 10)
        elif pol == "ug_NO2_Huang":
            merged_data[f"mortpol{annee}"] = merged_data[f"mort{annee}"] - merged_data[f"mort{annee}"] * np.exp(
                -np.log(RR_Huang) * merged_data["meandelta"] / 10)
        elif pol == "ug_NO2_Huang_high":
            merged_data[f"mortpol{annee}"] = merged_data[f"mort{annee}"] - merged_data[f"mort{annee}"] * np.exp(
                -np.log(RR_Huang_high) * merged_data["meandelta"] / 10)
        elif pol == "ug_NO2_Huang_low":
            merged_data[f"mortpol{annee}"] = merged_data[f"mort{annee}"] - merged_data[f"mort{annee}"] * np.exp(
                -np.log(RR_Huang_low) * merged_data["meandelta"] / 10)
        else:
            print("Pollutant unrecognized")

        # Set life expectancy based on the year
        if annee == 2030:
            esp = 84
        elif annee == 2019:
            esp = 80
        elif annee == 2050:
            esp = 86
        else:
            print("Year unrecognized")
            return None

        popannee = f"pop{annee}"
        iris_interet = merged_data[(merged_data['age'] >= 30) & (merged_data['age'] <= 86)]
        mort_interet = iris_interet[f"mortpol{annee}"].sum(skipna=True)
        donnees = pd.DataFrame({
            'age': np.arange(30, 100),
            'mort': np.zeros(70),
            'annees_gagnees': np.zeros(70),
            'tot_mort_30_99': np.full(70, mort_interet),
            'taux_initial': np.zeros(70),
            'taux_corrige': np.zeros(70),
            'conc': np.full(70, moy_meanconc(donnees_expo)),
            'conc_ponderee': np.full(70, expo_ponderee_meanconc(donnees_expo, popannee)),
            'delta_conc': np.full(70, moy_meandelta(donnees_expo)),
            'delta_conc_ponderee': np.full(70, expo_ponderee_meandelta(donnees_expo, popannee))
        })

        for age_ind in range(30, 89):
            iris_mort = merged_data[merged_data['age'] == age_ind]
            donnees.loc[donnees['age'] == age_ind, 'mort'] = iris_mort[f"mortpol{annee}"].sum(skipna=True)
            donnees.loc[donnees['age'] == age_ind, 'taux_corrige'] = (iris_mort[f"mort{annee}"].sum(skipna=True) -
                                                                      iris_mort[f"mortpol{annee}"].sum(skipna=True)) / \
                                                                     iris_mort[popannee].sum(skipna=True)
            donnees.loc[donnees['age'] == age_ind, 'taux_initial'] = iris_mort[f"mort{annee}"].sum(skipna=True) / \
                                                                     iris_mort[popannee].sum(skipna=True)
            donnees.loc[donnees['age'] == age_ind, 'annees_gagnees'] = donnees.loc[
                                                                           donnees['age'] == age_ind, 'mort'] * (
                                                                               esp - age_ind)

        if annee == 2030:
            somme_annees = donnees['annees_gagnees'].iloc[:55].sum(skipna=True)
        elif annee == 2050:
            somme_annees = donnees['annees_gagnees'].iloc[:57].sum(skipna=True)
        else:
            somme_annees = donnees['annees_gagnees'].iloc[:51].sum(skipna=True)

        donnees['tot_annees_gagnees_30_esp'] = somme_annees
        donnees = donnees.sort_values(by='age').reset_index(drop=True)
        return donnees

# Function to calculate age-specific life expectancy gained
import pandas as pd
import numpy as np

def life_exp_old(tab, annee):
    # Ensure year is an integer
    annee = int(annee)

    # Load mortality rates
    taux_mort = pd.read_csv("data/1-processed-data/INSEE/taux-mort.csv", sep=";")

    # Determine year-specific max age and mortality rates
    if annee == 2030:
        max_year = 84
        age = np.concatenate([taux_mort['age'].to_numpy(), tab['age'].to_numpy()])
        taux_initial = np.concatenate([taux_mort['taux_2030'].to_numpy(), tab['taux_initial'].to_numpy()])
        taux_corrige = np.concatenate([taux_mort['taux_2030'].to_numpy(), tab['taux_corrige'].to_numpy()])
    elif annee == 2050:
        max_year = 86
        age = np.concatenate([taux_mort['age'].to_numpy(), tab['age'].to_numpy()])
        taux_initial = np.concatenate([taux_mort['taux_2050'].to_numpy(), tab['taux_initial'].to_numpy()])
        taux_corrige = np.concatenate([taux_mort['taux_2050'].to_numpy(), tab['taux_corrige'].to_numpy()])
    elif annee == 2019:
        max_year = 80
        age = np.concatenate([taux_mort['age'].to_numpy(), tab['age'].to_numpy()])
        taux_initial = np.concatenate([taux_mort['taux_2019'].to_numpy(), tab['taux_initial'].to_numpy()])
        taux_corrige = np.concatenate([taux_mort['taux_2019'].to_numpy(), tab['taux_corrige'].to_numpy()])
    else:
        raise ValueError("Invalid year. Choose 2019, 2030, or 2050.")

    # Ensure mortality is capped beyond `max_year`
    taux_initial = np.where(age > max_year, 1, taux_initial)
    taux_corrige = np.where(age > max_year, 1, taux_corrige)

    # Nested function to calculate life expectancy
    def life_expectancy_old(age, mortality_rate, max_year):
        for i in range(len(age)):
            if age[i] > max_year:
                mortality_rate[i] = 1  # Cap mortality rate at 1
        prop_alive = np.concatenate(([1], np.cumprod(1 - mortality_rate)))
        deaths = -np.diff(prop_alive)
        life_exp = np.sum(deaths * np.arange(len(deaths)))
        return life_exp

    # Calculate life expectancy with initial and corrected rates
    life_exp_initial = life_expectancy_old(age, taux_initial, max_year)
    life_exp_corrige = life_expectancy_old(age, taux_corrige, max_year)

    # Calculate the difference in life expectancy (years converted to months)
    delta_life_exp_mois = (life_exp_corrige - life_exp_initial) * 12

    # Create result DataFrame
    result = pd.DataFrame({
        'age': age,
        'taux_initial': taux_initial,
        'taux_corrige': taux_corrige
    })

    result['life_exp_init'] = life_exp_initial
    result['life_exp_corrige'] = life_exp_corrige
    result['delta_life_mois'] = delta_life_exp_mois

    return result

# Calculate life expectancy for each age group
def life_exp(tab, annee):
    # Ensure year is an integer
    annee = int(annee)

    # Load mortality rates
    taux_mort = pd.read_csv("data/1-processed-data/INSEE/taux-mort.csv", sep=";")

    # Assign year-specific mortality rates
    if annee == 2030:
        taux_initial = taux_mort['taux_2030'].to_numpy()
        taux_corrige = taux_mort['taux_2030'].to_numpy()
    elif annee == 2050:
        taux_initial = taux_mort['taux_2050'].to_numpy()
        taux_corrige = taux_mort['taux_2050'].to_numpy()
    elif annee == 2019:
        taux_initial = taux_mort['taux_2019'].to_numpy()
        taux_corrige = taux_mort['taux_2019'].to_numpy()
    else:
        raise ValueError("Invalid year. Choose 2019, 2030, or 2050.")

    age = taux_mort['age'].to_numpy()

    # Append additional age-specific values from `tab`
    age = np.concatenate([age, tab['age'].to_numpy()])
    taux_initial = np.concatenate([taux_initial, tab['taux_initial'].to_numpy()])
    taux_corrige = np.concatenate([taux_corrige, tab['taux_corrige'].to_numpy()])

    # Ensure mortality is capped at 1 beyond age 86
    taux_initial = np.where(age > 86, 1, taux_initial)
    taux_corrige = np.where(age > 86, 1, taux_corrige)

    # Nested function to compute life expectancy at each age
    def life_expectancy(age, mortality_rate):
        n = len(age)
        le = np.zeros(n)
        for i in range(n):
            # Compute survival probabilities from current age onward
            sub_mortality = mortality_rate[i:]
            prop_alive = np.cumprod(1 - sub_mortality)

            # Compute deaths at each step
            deaths = np.concatenate(([1], prop_alive)) - np.concatenate((prop_alive, [0]))

            # Compute expected remaining life years
            le[i] = np.sum(deaths * np.arange(len(deaths)))
        return le

    # Calculate life expectancy for each age group
    life_exp_initial = life_expectancy(age, taux_initial)
    life_exp_corrige = life_expectancy(age, taux_corrige)

    # Calculate the difference in life expectancy (months)
    delta_life_exp_mois = (life_exp_corrige - life_exp_initial) * 12

    # Create result DataFrame with age-specific values
    result = pd.DataFrame({
        'age': age,
        'taux_initial': taux_initial,
        'taux_corrige': taux_corrige,
        'life_exp_init': life_exp_initial,
        'life_exp_corrige': life_exp_corrige,
        'delta_life_mois': delta_life_exp_mois
    })

    return result

def calculate_iris_level_mortality_age_specific(
        donnees_merged,
        donnees_expo,
        annee,
        pol,
        RR_PM25=1.15,
        RR_PM25_high=1.25,
        RR_PM25_low=1.05,
        RR_NO2=1.023,
        RR_NO2_high=1.037,
        RR_NO2_low=1.015,
):
    import numpy as np
    import pandas as pd

    annee = int(annee)
    donnees_expo = donnees_expo.drop(columns="geometry", errors="ignore")

    required_columns = ["iriscod", "meanconc", "meandelta"]
    if any(col not in donnees_expo.columns for col in required_columns):
        raise ValueError(
            f"Missing columns in donnees_expo: {[col for col in required_columns if col not in donnees_expo.columns]}"
        )

    mort_column = f"mort{annee}"
    pop_column = f"pop{annee}"
    if mort_column not in donnees_merged.columns or pop_column not in donnees_merged.columns:
        raise ValueError(f"Missing columns in donnees_merged: {mort_column} or {pop_column}")

    if "age" not in donnees_merged.columns:
        raise ValueError("Missing 'age' column in `donnees_merged`")

    # Filter data for ages 30 to 99
    donnees_merged = donnees_merged[(donnees_merged['age'] >= 30) & (donnees_merged['age'] <= 99)]

    # Merge dataframes by IRIS code
    merged_df = pd.merge(donnees_merged, donnees_expo[required_columns], on="iriscod", how="left")

    # Define RR mapping
    pollutant_rr_mapping = {
        "ug_PM25_RH50": RR_PM25,
        "ug_PM25_RH50_high": RR_PM25_high,
        "ug_PM25_RH50_low": RR_PM25_low,
        "ug_NO2": RR_NO2,
        "ug_NO2_high": RR_NO2_high,
        "ug_NO2_low": RR_NO2_low,
    }

    RR = pollutant_rr_mapping.get(pol)
    if RR is None:
        raise ValueError(f"Pollutant '{pol}' not found. Allowed: {list(pollutant_rr_mapping.keys())}")
    if RR <= 0:
        raise ValueError("RR must be greater than 0.")

    # Define life expectancy for each target year
    future_le_by_year = {2019: 80, 2030: 84, 2050: 86}
    future_le = future_le_by_year.get(annee)
    if future_le is None:
        raise ValueError(f"Future life expectancy not defined for year {annee}.")

    # Create dynamic age-specific LE (ARLYL = max(LE - age, 0))
    merged_df["arlyl"] = merged_df["age"].apply(lambda age: max(future_le - age, 0))

    # Calculate pollution-attributed mortality
    merged_df[f"mortpol{annee}"] = merged_df[mort_column] * np.exp(-np.log(RR) * merged_df["meandelta"] / 10)
    merged_df["mortality_attrib_pollution_spec"] = merged_df[mort_column] - merged_df[f"mortpol{annee}"]

    # Years of life gained = ARLYL × mortality avoided
    merged_df["years_gained_age_specific"] = merged_df["mortality_attrib_pollution_spec"] * merged_df["arlyl"]

    # Days of life lost
    merged_df["dll_spec"] = merged_df["years_gained_age_specific"] * 365

    # Aggregate at IRIS level
    iris_level_results = merged_df.groupby("iriscod").agg(
        total_corrected_mortality=(f"mortpol{annee}", "sum"),
        total_mortality_avoided=("mortality_attrib_pollution_spec", "sum"),
        total_years_gained=("years_gained_age_specific", "sum"),
        total_dll=("dll_spec", "sum"),
        total_population=(pop_column, "sum"),
        mean_concentration=("meanconc", "mean"),
        mean_delta_concentration=("meandelta", "mean"),
    ).reset_index()

    iris_level_results["dll_per_person_per_year"] = (
            iris_level_results["total_dll"] / iris_level_results["total_population"]
    )

    # Sort data by IRIS code
    iris_level_results = iris_level_results.sort_values(by="iriscod")

    # Save results to a CSV file
    #output_file = f"iris_mortality_results_{annee}.csv"
    #iris_level_results.to_csv(output_file, index=False)
    #print(f"Results saved to {output_file}")

    return iris_level_results


def calculate_com_level_mortality_age_specific(  # Updated function name for clarity
        donnees_merged,
        donnees_expo,
        annee,
        pol,
        RR_PM25=1.15,
        RR_PM25_high=1.25,
        RR_PM25_low=1.05,
        RR_NO2=1.023,
        RR_NO2_high=1.037,
        RR_NO2_low=1.015,
):
    import numpy as np
    import pandas as pd

    annee = int(annee)
    donnees_expo = donnees_expo.drop(columns="geometry", errors="ignore")

    # Update: Change required columns from 'iriscod' to 'comcod'
    required_columns = ["comcod", "meanconc", "meandelta"]
    if any(col not in donnees_expo.columns for col in required_columns):
        raise ValueError(
            f"Missing columns in donnees_expo: {[col for col in required_columns if col not in donnees_expo.columns]}"
        )

    mort_column = f"mort{annee}"
    pop_column = f"pop{annee}"
    if mort_column not in donnees_merged.columns or pop_column not in donnees_merged.columns:
        raise ValueError(f"Missing columns in donnees_merged: {mort_column} or {pop_column}")

    if "age" not in donnees_merged.columns:
        raise ValueError("Missing 'age' column in `donnees_merged`")

    # Filter data for ages 30 to 99
    donnees_merged = donnees_merged[(donnees_merged['age'] >= 30) & (donnees_merged['age'] <= 99)]

    # Update: Merge dataframes by COM code instead of IRIS code
    merged_df = pd.merge(donnees_merged, donnees_expo[required_columns], on="comcod", how="left")

    # Define RR mapping
    pollutant_rr_mapping = {
        "ug_PM25_RH50": RR_PM25,
        "ug_PM25_RH50_high": RR_PM25_high,
        "ug_PM25_RH50_low": RR_PM25_low,
        "ug_NO2": RR_NO2,
        "ug_NO2_high": RR_NO2_high,
        "ug_NO2_low": RR_NO2_low,
    }

    RR = pollutant_rr_mapping.get(pol)
    if RR is None:
        raise ValueError(f"Pollutant '{pol}' not found. Allowed: {list(pollutant_rr_mapping.keys())}")
    if RR <= 0:
        raise ValueError("RR must be greater than 0.")

    # Define life expectancy for each target year
    future_le_by_year = {2019: 80, 2030: 84, 2050: 86}
    future_le = future_le_by_year.get(annee)
    if future_le is None:
        raise ValueError(f"Future life expectancy not defined for year {annee}.")

    # Create dynamic age-specific LE (ARLYL = max(LE - age, 0))
    merged_df["arlyl"] = merged_df["age"].apply(lambda age: max(future_le - age, 0))

    # Calculate pollution-attributed mortality
    merged_df[f"mortpol{annee}"] = merged_df[mort_column] * np.exp(-np.log(RR) * merged_df["meandelta"] / 10)
    merged_df["mortality_attrib_pollution_spec"] = merged_df[mort_column] - merged_df[f"mortpol{annee}"]

    # Years of life gained = ARLYL × mortality avoided
    merged_df["years_gained_age_specific"] = merged_df["mortality_attrib_pollution_spec"] * merged_df["arlyl"]

    # Days of life lost
    merged_df["dll_spec"] = merged_df["years_gained_age_specific"] * 365

    # Update: Aggregate at COM level (comcod)
    com_level_results = merged_df.groupby("comcod").agg(
        total_corrected_mortality=(f"mortpol{annee}", "sum"),
        total_mortality_avoided=("mortality_attrib_pollution_spec", "sum"),
        total_years_gained=("years_gained_age_specific", "sum"),
        total_dll=("dll_spec", "sum"),
        total_population=(pop_column, "sum"),
        mean_concentration=("meanconc", "mean"),
        mean_delta_concentration=("meandelta", "mean"),
    ).reset_index()

    # Update: Calculate days of life lost per person per year at COM level
    com_level_results["dll_per_person_per_year"] = (
            com_level_results["total_dll"] / com_level_results["total_population"]
    )

    # Update: Sort results by COM code instead of IRIS code
    com_level_results = com_level_results.sort_values(by="comcod")

    # Save results to a CSV file (optional)
    # output_file = f"com_mortality_results_{annee}.csv"
    # com_level_results.to_csv(output_file, index=False)
    # print(f"Results saved to {output_file}")

    return com_level_results











import requests
import json
import pycountry
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pickle

with open('model.pkl', 'rb') as file:
    xgb = pickle.load(file)

reshuffled_df = pd.read_excel("Data\\actual_reshuffled.xlsx")
def get_pop_endyear(country, end_year):
    file_path = "Data\\country_data\\Global Economy__EX.xlsx"
    # read the Excel file into a pandas dataframe, skipping the first row as it contains unnamed columns
    df = pd.read_excel(file_path, index_col=0)
    pop = df.loc[country, end_year]
    return pop

def get_SOFI(country):
    file_path = "Data\\SOFI_price.xlsx"
    SOFI_df = pd.read_excel(file_path, index_col=0)
    SOFI = SOFI_df.loc[country, 'Cost']
    return SOFI
def get_latest_gini(country_name):
    country = pycountry.countries.get(name=country_name)
    country_code = country.alpha_2
    url = f"http://api.worldbank.org/v2/country/{country_code}/indicator/SI.POV.GINI?format=json"
    response = requests.get(url)
    data = response.json()

    if data[1] is not None:
        for item in data[1]:
            if item['value'] is not None:
                return item['value']
    return None


def calculate_cumulative_inflation(country, start_year, end_year):
    country_name = country.replace(' ', '_')
    file_path = f'Data/country_data/historical_country_{country_name}_indicator_Food_Inflation.csv'
    df = pd.read_csv(file_path)
    df['Year'] = pd.to_datetime(df['DateTime']).dt.year
    grouped_data = df.groupby(['Country', 'Year'])
    annual_inflation = grouped_data['Value'].mean().round(2)
    annual_inflation = annual_inflation.reset_index()
    annual_inflation_frame = pd.DataFrame({'Year': annual_inflation['Year'], 'Inflation': annual_inflation['Value']})
    f_inflation_rates = annual_inflation_frame[
        (annual_inflation_frame['Year'] >= start_year) & (annual_inflation_frame['Year'] <= end_year)][
        'Inflation'].tolist()

    f_inflation_rates = [rate / 100 for rate in f_inflation_rates]
    # Calculate the cumulative inflation rate
    cumulative_inflation = (np.prod(np.array(f_inflation_rates) + 1) - 1) * 100

    return cumulative_inflation


def get_gdp_per_capita(country_name, year):
    country = pycountry.countries.get(name=country_name)
    if country is None:
        print(f"Could not find a country with the name {country_name}")
        return None
    country_code = country.alpha_2
    url = f"http://api.worldbank.org/v2/country/{country_code}/indicator/NY.GDP.PCAP.CD?date={year}&format=json"
    response = requests.get(url)
    data = response.json()

    if data[1] is not None:
        for item in data[1]:
            if item['value'] is not None:
                return item['value']
    return None


def trend_prediction(country, end_year):
    if end_year > 2022:
        end_year_2 = 2022
    else:
        end_year_2 = end_year
    Population = get_pop_endyear(country, end_year_2)
    GINI = get_latest_gini(country)
    Inflation = calculate_cumulative_inflation(country, 2017, end_year)
    GDP_2017 = get_gdp_per_capita(country, 2017)
    GDP = get_gdp_per_capita(country, end_year_2)
    SOFI = get_SOFI(country)

    data = {
        'Population': [Population],
        'GDP_2017': [GDP_2017],
        'GDP': [GDP],
        'SOFI': [SOFI],
        'Inflation': [Inflation],
        'GINI': [GINI]
    }

    # Create the DataFrame
    df = pd.DataFrame(data, index=[country])
    df = df.reset_index(drop=True)

    # Ensure that the data is in the correct format
    df = df[["Population", "GDP_2017", "GDP", "SOFI", "Inflation", "GINI"]]

    # Use the trained model to predict on new data
    predicted_value = xgb.predict(df)

    # Find the actual value in the reshuffled dataset
    if country in reshuffled_df['Country'].values:
        try:
            # If it exists, find the actual value
            actual_value = reshuffled_df.loc[reshuffled_df['Country'] == country, str(end_year)].values[0]
        except KeyError:
            # If the year doesn't exist, set the actual value to NaN
            actual_value = np.nan
    else:
        # If the country doesn't exist, set the actual value to NaN
        actual_value = np.nan

    print(f"The predicted value for {end_year} is {predicted_value}")
    if pd.isna(actual_value):
        print(f"The actual value for {end_year} is not available")
    else:
        print(f"The actual value for {end_year} is {actual_value}")
    return predicted_value, actual_value


def compare_trends(country, end_year):
    # Generate predictions and actuals for three years
    predictions = []
    actuals = []
    for year in range(end_year - 4, end_year + 1):
        pred, actual = trend_prediction(country, year)
        predictions.append(pred)
        actuals.append(actual)

    # Prepare data for plotting
    years = list(range(end_year - 4, end_year + 1))

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(years, predictions, label='Predicted values', marker='o')
    plt.plot(years, actuals, label='Actual values', marker='o')

    # Format the y-axis to display in millions
    plt.gca().yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: x / 1e6))

    # Format the x-axis to display as integers
    plt.gca().xaxis.set_major_formatter(mtick.FormatStrFormatter('%d'))

    plt.xlabel('Year')
    plt.ylabel('Value (in millions)')
    plt.title(f'Trend prediction vs Actual values for {country} (end year: {end_year})')
    plt.legend()
    plt.grid(True)
    plt.show()


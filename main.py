

from My_functions import compare_trends


country_name = "Angola"
end_year = 2023


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    country_name = input("Enter the name of the country: ")
    end_year = int(input("Enter the year: "))

    compare_trends(country_name, end_year)




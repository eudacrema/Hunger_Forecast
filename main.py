import argparse
from My_functions import compare_trends

def main(country_name, end_year):
    results = compare_trends(country_name, end_year)
    print(results)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some inputs.')
    parser.add_argument('country_name', type=str, help='Name of the country')
    parser.add_argument('end_year', type=int, help='End year')

    args = parser.parse_args()

    main(args.country_name, args.end_year)




# Press the green button in the gutter to run the script.
#if __name__ == '__main__':
    #country_name = input("Enter the name of the country: ")
    #end_year = int(input("Enter the year: "))

    #compare_trends(country_name, end_year)




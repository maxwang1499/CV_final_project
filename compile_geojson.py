import os
import sys
import geopandas as gpd
import pandas as pd

if __name__ == '__main__':
    # Input folder contains all the processed JSON files
    input_folder = sys.argv[1]
    # The final compiled CSV will be generated in the output folder
    output_folder = sys.argv[2]
    # Create an empty list to append my processed dataframes too
    processed_df_list = []

    # Append to the master dataframe
    for filename in os.listdir(input_folder):
        temp_df = gpd.read_file(input_folder + "/" + filename)

        temp_df["Type"].replace({"CAFO":"CAFOs",
                                "Mine": "Mines",
                                "Landfill": "Landfills",
                                "landfills": "Landfills",
                                "Crude Oil Terminal": "RefineriesAndTerminals",
                                "Oil Refinery": "RefineriesAndTerminals",
                                "LNG Terminal": "RefineriesAndTerminals",
                                "LNG": "RefineriesAndTerminals",
                                "Processing Plant": "ProcessingPlants",
                                "WWTP": "WWTreatment",
                                "Wastewater Treatment": "WWTreatment"}, inplace=True)
        processed_df_list.append(temp_df)
    
    # Concat everything together
    main_df = pd.concat(processed_df_list)

    # Drop CNG Fueling
    main_df = main_df[main_df["Type"] != "CNG Fueling"]

    # Sort the dataframe based on facility type and then source name
    main_df = main_df.sort_values(by=["Type","Source"], ascending =(True,True), key=lambda col: col.str.lower())

    # Output csv
    main_df.to_csv(output_folder + '/compiled_dataset.csv')

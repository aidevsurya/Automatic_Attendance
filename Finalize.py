import sys
import pandas as pd
from tkinter import filedialog

# Function to convert CSV to Excel
def csv_to_excel(csv_file, excel_file):
    # Read the CSV file using pandas
    try:
        df = pd.read_csv(csv_file)
        
        # Write to Excel file
        df.to_excel(excel_file, index=False, engine='openpyxl')
        
        print(f"CSV has been successfully converted to {excel_file}")
    except Exception as e:
        print(f"Error: {e}")

# Input CSV file and output Excel file names
csv_file =  filedialog.askopenfilename(filetypes=(["filetypes:",".csv"],["all types:","*.*"])) # Replace with your actual CSV file path

if csv_file=="":
    sys.exit()

excel_file = filedialog.askopenfilename(filetypes=(["filetypes:",".xls"],["filetypes:",".xlsx"],["all types:","*.*"])) # Desired Excel output file path


if excel_file=="":
    excel_file="Attendance.xlsx"
# Call the function to convert CSV to Excel
csv_to_excel(csv_file, excel_file)

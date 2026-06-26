import pandas as pd

df = pd.read_excel("data/processed/Data_Harga_Interpolasi_v2.xlsx")

print(df.head())
print("\nKolom:")
print(df.columns.tolist())

print("\nJumlah data:")
print(df.shape)
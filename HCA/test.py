import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import matplotlib.pyplot as plt

# Datele tale
data = {
    'Tara': ["Afghanistan", "Albania", "Algeria", "Angola", "Argentina", "Armenia", "Australia", "Austria", "Azerbaijan", "Bahamas", "Bahrain", "Bangladesh", "Barbados", "Belarus", "Belgium", "Belize", "Benin", "Bhutan", "Bolivia", "Bosnia and Herzegovina", "Botswana", "Brazil", "Brunei", "Bulgaria", "Burkina Faso", "Burundi", "Cabo Verde", "Cambodia", "Cameroon", "Canada", "Central African Republic", "Chad", "Chile", "China", "Colombia", "Comoros", "Congo", "Costa Rica", "Cote d'Ivoire", "Croatia", "Cuba", "Cyprus", "Czech Republic", "Denmark", "Djibouti", "Dominican Republic", "Ecuador", "Egypt", "El Salvador", "Equatorial Guinea", "Eritrea", "Estonia", "Eswatini", "Ethiopia", "Fiji", "Finland", "France", "Gabon", "Gambia", "Georgia", "Germany", "Ghana", "Greece", "Grenada", "Guatemala", "Guinea", "Guinea-Bissau", "Guyana", "Haiti", "Honduras", "Hungary", "Iceland", "India", "Indonesia", "Iran", "Iraq", "Ireland", "Israel", "Italy", "Jamaica", "Japan", "Jordan", "Kazakhstan", "Kenya", "Kiribati", "Korea", "Kuwait", "Kyrgyzstan", "Laos", "Latvia", "Lebanon", "Lesotho", "Liberia", "Libya", "Lithuania", "Luxembourg", "Madagascar", "Malawi", "Malaysia", "Maldives", "Mali", "Malta", "Mauritania", "Mauritius", "Mexico", "Moldova", "Mongolia", "Montenegro", "Morocco", "Mozambique", "Myanmar", "Namibia", "Nepal", "Netherlands", "New Zealand", "Nicaragua", "Niger", "Nigeria", "North Macedonia", "Norway", "Oman", "Pakistan", "Panama", "Papua New Guinea", "Paraguay", "Peru", "Philippines", "Poland", "Portugal", "Qatar", "Romania", "Russia", "Rwanda", "Saint Lucia", "Samoa", "San Marino", "Sao Tome and Principe", "Saudi Arabia", "Senegal", "Serbia", "Seychelles", "Sierra Leone", "Singapore", "Slovakia", "Slovenia", "Solomon Islands", "Somalia", "South Africa", "South Sudan", "Spain", "Sri Lanka", "Sudan", "Suriname", "Sweden", "Switzerland", "Syria", "Taiwan", "Tajikistan", "Tanzania", "Thailand", "Timor-Leste", "Togo", "Tonga", "Trinidad and Tobago", "Tunisia", "Turkey", "Turkmenistan", "Tuvalu", "Uganda", "Ukraine", "United Arab Emirates", "United Kingdom", "United States", "Uruguay", "Uzbekistan", "Vanuatu", "Vietnam", "Zambia", "Zimbabwe"],
    'Cluster': ["C4", "C5", "C3", "C4", "C0", "C5", "C1", "C1", "C3", "C2", "C3", "C4", "C2", "C1", "C1", "C2", "C3", "C5", "C2", "C5", "C5", "C4", "C1", "C3", "C2", "C4", "C4", "C3", "C5", "C3", "C5", "C4", "C4", "C3", "C4", "C5", "C3", "C4", "C3", "C0", "C1", "C5", "C1", "C1", "C0", "C1", "C5", "C2", "C1", "C3", "C3", "C0", "C5", "C1", "C0", "C3", "C3", "C5", "C1", "C1", "C2", "C5", "C3", "C2", "C1", "C0", "C3", "C4", "C0", "C0", "C2", "C2", "C3", "C0", "C5", "C3", "C1", "C2", "C2", "C1", "C4", "C5", "C4", "C4", "C0", "C0", "C0", "C1", "C4", "C1", "C3", "C3", "C3", "C4", "C1", "C3", "C5", "C5", "C4", "C0", "C3", "C5", "C1", "C4", "C3", "C3", "C3", "C4", "C4", "C3", "C3", "C5", "C5", "C3", "C4", "C4", "C4", "C4", "C3", "C3", "C3", "C4", "C4", "C5", "C4", "C2", "C0", "C2", "C1", "C4", "C4", "C4", "C3", "C3", "C3", "C4", "C5", "C3"]
}

df = pd.DataFrame(data)

# Matricea de distanță
dist_matrix = np.zeros((len(df), len(df)))

# Populăm matricea de distanță cu distanțele dintre clustere
for i in range(len(df)):
    for j in range(len(df)):
        if i != j:
            dist_matrix[i, j] = (df['Cluster'][i] != df['Cluster'][j])

# Metoda Ward și dendrograma
linkage_matrix = linkage(dist_matrix, method='ward')

# Desenarea dendrogramei
dendrogram(linkage_matrix, labels=df['Country'].values, orientation='right')
plt.title('Dendrograma folosind metoda Ward')
plt.xlabel('Distanță ward')
plt.show()

# Determinarea numărului optim de clustere folosind metoda cotului
k = 3  # Schimbă acest număr în funcție de rezultatul analizei tale
clusters = fcluster(linkage_matrix, k, criterion='maxclust')

# Afișarea rezultatelor
df['Ward_Cluster'] = clusters
print(df[['Country', 'Ward_Cluster']])

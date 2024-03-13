import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import networkx as nx
from itertools import combinations
from collections import defaultdict

from sklearn.decomposition import PCA
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# Function to load and clean data
@st.cache_data
def load_data(file):
    df = pd.read_csv(file, encoding='ISO-8859-1')
    df.dropna(subset=['Data value', 'Footnotes'], how='any', inplace=True)
    return df


# Function to create and plot a network graph
def plot_network_graph(data):
    G = nx.Graph()
    for _, row in data.iterrows():
        G.add_node(row['Area'], type='country')
        G.add_node(row['Indicator'], type='policy')
        G.add_edge(row['Area'], row['Indicator'])
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, k=0.15, iterations=20)
    countries = [node for node, attr in G.nodes(data=True) if attr['type'] == 'country']
    policies = [node for node, attr in G.nodes(data=True) if attr['type'] == 'policy']
    nx.draw_networkx_nodes(G, pos, nodelist=countries, node_color='lightblue', label='Countries', node_size=100)
    nx.draw_networkx_nodes(G, pos, nodelist=policies, node_color='lightgreen', label='Policies', node_shape='s',
                           node_size=50)
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)
    nx.draw_networkx_labels(G, pos, font_size=8)
    st.pyplot(plt)


st.title('Comprehensive HIV Policy Data Analysis in Streamlit')

uploaded_file = st.sidebar.file_uploader("Choose your CSV file", type="csv")
if uploaded_file is not None:
    data = load_data(uploaded_file)

    analysis_options = [
        "Show Data Overview", "Time-Series Chart", "Network Graph", "Distribution Analysis",
        "Correlation Analysis", "Temporal Trends Analysis", "Policy and Time Period Interaction Analysis",
        "Regional Policy Effectiveness Assessment", "Policy Clustering"
    ]

    selected_analysis = st.selectbox("Choose an analysis or visualization:", analysis_options)

    if selected_analysis == "Show Data Overview":
        st.write(data)

    # Time-Series Chart
    elif selected_analysis == "Time-Series Chart":
        st.subheader('Evolution of HIV-Related Laws and Policies (2017-2022)')
        selected_countries = ['United States', 'Brazil', 'South Africa', 'India', 'Germany']
        filtered_data = data[data['Area'].isin(selected_countries)]
        pivot_data = filtered_data.pivot_table(index='Area', columns='Time Period', aggfunc='count', fill_value=0)[
            'Indicator']

        plt.figure(figsize=(12, 8))
        sns.lineplot(data=pivot_data.T, dashes=False, marker='o')
        plt.title('Evolution of HIV-Related Laws and Policies (2017-2022)')
        plt.ylabel('Number of Policies')
        plt.xlabel('Year')
        plt.xticks(rotation=45)
        plt.legend(title='Country', title_fontsize='13', fontsize='11')
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        st.pyplot(plt)

    # Network Graph
    elif selected_analysis == "Network Graph":
        st.subheader('Network Graph of Countries and HIV-Related Policies')
        plot_network_graph(data)

    # Distribution Analysis
    elif selected_analysis == "Distribution Analysis":
        st.subheader('Distribution of Policies per Country')
        policies_per_country = data.groupby('Area').size()
        plt.figure(figsize=(10, 8))
        sns.histplot(policies_per_country, bins=30, kde=False)
        plt.title('Distribution of Policies per Country')
        plt.xlabel('Number of Policies')
        plt.ylabel('Frequency')
        st.pyplot(plt)

    # Correlation Analysis
    elif selected_analysis == "Correlation Analysis":
        st.subheader('Top 5 Common Policy Pairs')
        policy_pairs = defaultdict(int)
        country_policies = data.groupby('Area')['Indicator'].apply(set).reset_index()
        for policies in country_policies['Indicator']:
            for policy1, policy2 in combinations(policies, 2):
                policy_pairs[(policy1, policy2)] += 1
        top_5_policy_pairs = sorted(policy_pairs.items(), key=lambda x: x[1], reverse=True)[:5]
        st.write(top_5_policy_pairs)

    # Temporal Trends Analysis
    elif selected_analysis == "Temporal Trends Analysis":
        st.subheader('Temporal Trends in Adoption of HIV-Related Policies (2017-2022)')
        temporal_data = data.groupby(['Time Period', 'Indicator']).size().unstack(fill_value=0)
        plt.figure(figsize=(14, 8))
        sns.lineplot(data=temporal_data, dashes=False, markers=True, linewidth=2.5, marker='o')
        plt.title('Temporal Trends in Adoption of HIV-Related Policies (2017-2022)')
        plt.ylabel('Number of Countries Adopting Policy')
        plt.xlabel('Year')
        plt.xticks(rotation=45)
        plt.legend(title='Policy', title_fontsize='13', fontsize='11', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, linestyle='--', linewidth=0.5)
        st.pyplot(plt)

    # Policy and Time Period Interaction Analysis
    elif selected_analysis == "Policy and Time Period Interaction Analysis":
        st.subheader('HIV-Related Policy Activity Over Time (2017-2022)')
        policy_activity_over_time = data.groupby('Time Period').size()
        plt.figure(figsize=(12, 6))
        sns.lineplot(x=policy_activity_over_time.index, y=policy_activity_over_time.values, marker='o', linestyle='-',
                     color='b')
        plt.title('HIV-Related Policy Activity Over Time (2017-2022)')
        plt.xlabel('Year')
        plt.ylabel('Number of Policy Adoptions/Updates')
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        st.pyplot(plt)

    # Regional Policy Effectiveness Assessment
    elif selected_analysis == "Regional Policy Effectiveness Assessment":
        st.subheader('Number of Policies Adopted by Region')
        country_to_region = {
            'Afghanistan': 'Asia',
            'Albania': 'Europe',
            'Algeria': 'Africa',
            'Andorra': 'Europe',
            'Angola': 'Africa',
            'Antigua and Barbuda': 'Americas',
            'Argentina': 'Americas',
            'Armenia': 'Asia',
            'Australia': 'Oceania',
            'Austria': 'Europe',
            'Azerbaijan': 'Asia',
            'Bahamas': 'Americas',
            'Bahrain': 'Asia',
            'Bangladesh': 'Asia',
            'Barbados': 'Americas',
            'Belarus': 'Europe',
            'Belgium': 'Europe',
            'Belize': 'Americas',
            'Benin': 'Africa',
            'Bhutan': 'Asia',
            'Bolivia (Plurinational State of)': 'Americas',
            'Bosnia and Herzegovina': 'Europe',
            'Botswana': 'Africa',
            'Brazil': 'Americas',
            'Brunei Darussalam': 'Asia',
            'Bulgaria': 'Europe',
            'Burkina Faso': 'Africa',
            'Burundi': 'Africa',
            'Cabo Verde': 'Africa',
            'Cambodia': 'Asia',
            'Cameroon': 'Africa',
            'Canada': 'Americas',
            'Central African Republic': 'Africa',
            'Chad': 'Africa',
            'Chile': 'Americas',
            'China': 'Asia',
            'Colombia': 'Americas',
            'Comoros': 'Africa',
            'Congo': 'Africa',
            'Costa Rica': 'Americas',
            "Côte d'Ivoire": 'Africa',
            'Croatia': 'Europe',
            'Cuba': 'Americas',
            'Cyprus': 'Europe',
            'Czechia': 'Europe',
            'Democratic People\'s Republic of Korea': 'Asia',
            'Democratic Republic of the Congo': 'Africa',
            'Denmark': 'Europe',
            'Djibouti': 'Africa',
            'Dominica': 'Americas',
            'Dominican Republic': 'Americas',
            'Ecuador': 'Americas',
            'Egypt': 'Africa',
            'El Salvador': 'Americas',
            'Equatorial Guinea': 'Africa',
            'Eritrea': 'Africa',
            'Estonia': 'Europe',
            'Eswatini': 'Africa',
            'Ethiopia': 'Africa',
            'Fiji': 'Oceania',
            'Finland': 'Europe',
            'France': 'Europe',
            'Gabon': 'Africa',
            'Gambia': 'Africa',
            'Georgia': 'Asia',
            'Germany': 'Europe',
            'Ghana': 'Africa',
            'Greece': 'Europe',
            'Grenada': 'Americas',
            'Guatemala': 'Americas',
            'Guinea': 'Africa',
            'Guinea-Bissau': 'Africa',
            'Guyana': 'Americas',
            'Haiti': 'Americas',
            'Honduras': 'Americas',
            'Hungary': 'Europe',
            'Iceland': 'Europe',
            'India': 'Asia',
            'Indonesia': 'Asia',
            'Iran (Islamic Republic of)': 'Asia',
            'Iraq': 'Asia',
            'Ireland': 'Europe',
            'Israel': 'Asia',
            'Italy': 'Europe',
            'Jamaica': 'Americas',
            'Japan': 'Asia',
            'Jordan': 'Asia',
            'Kazakhstan': 'Asia',
            'Kenya': 'Africa',
            'Kiribati': 'Oceania',
            'Kuwait': 'Asia',
            'Kyrgyzstan': 'Asia',
            "Lao People's Democratic Republic": 'Asia',
            'Latvia': 'Europe',
            'Lebanon': 'Asia',
            'Lesotho': 'Africa',
            'Liberia': 'Africa',
            'Libya': 'Africa',
            'Liechtenstein': 'Europe',
            'Lithuania': 'Europe',
            'Luxembourg': 'Europe',
            'Madagascar': 'Africa',
            'Malawi': 'Africa',
            'Malaysia': 'Asia',
            'Maldives': 'Asia',
            'Mali': 'Africa',
            'Malta': 'Europe',
            'Marshall Islands': 'Oceania',
            'Mauritania': 'Africa',
            'Mauritius': 'Africa',
            'Mexico': 'Americas',
            'Micronesia (Federated States of)': 'Oceania',
            'Monaco': 'Europe',
            'Mongolia': 'Asia',
            'Montenegro': 'Europe',
            'Morocco': 'Africa',
            'Mozambique': 'Africa',
            'Myanmar': 'Asia',
            'Namibia': 'Africa',
            'Nauru': 'Oceania',
            'Nepal': 'Asia',
            'Netherlands': 'Europe',
            'New Zealand': 'Oceania',
            'Nicaragua': 'Americas',
            'Niger': 'Africa',
            'Nigeria': 'Africa',
            'North Macedonia': 'Europe',
            'Norway': 'Europe',
            'Oman': 'Asia',
            'Pakistan': 'Asia',
            'Palau': 'Oceania',
            'Panama': 'Americas',
            'Papua New Guinea': 'Oceania',
            'Paraguay': 'Americas',
            'Peru': 'Americas',
            'Philippines': 'Asia',
            'Poland': 'Europe',
            'Portugal': 'Europe',
            'Qatar': 'Asia',
            'Republic of Korea': 'Asia',
            'Republic of Moldova': 'Europe',
            'Romania': 'Europe',
            'Russian Federation': 'Europe',  # Noting the transcontinental nature
            'Rwanda': 'Africa',
            'Saint Kitts and Nevis': 'Americas',
            'Saint Lucia': 'Americas',
            'Saint Vincent and the Grenadines': 'Americas',
            'Samoa': 'Oceania',
            'San Marino': 'Europe',
            'Sao Tome and Principe': 'Africa',
            'Saudi Arabia': 'Asia',
            'Senegal': 'Africa',
            'Serbia': 'Europe',
            'Seychelles': 'Africa',
            'Sierra Leone': 'Africa',
            'Singapore': 'Asia',
            'Slovakia': 'Europe',
            'Slovenia': 'Europe',
            'Solomon Islands': 'Oceania',
            'Somalia': 'Africa',
            'South Africa': 'Africa',
            'South Sudan': 'Africa',
            'Spain': 'Europe',
            'Sri Lanka': 'Asia',
            'Sudan': 'Africa',
            'Suriname': 'Americas',
            'Sweden': 'Europe',
            'Switzerland': 'Europe',
            'Syrian Arab Republic': 'Asia',
            'Tajikistan': 'Asia',
            'Tanzania': 'Africa',
            'Thailand': 'Asia',
            'Timor-Leste': 'Asia',
            'Togo': 'Africa',
            'Tonga': 'Oceania',
            'Trinidad and Tobago': 'Americas',
            'Tunisia': 'Africa',
            'Turkey': 'Europe',  # Noting the transcontinental nature
            'Turkmenistan': 'Asia',
            'Tuvalu': 'Oceania',
            'Uganda': 'Africa',
            'Ukraine': 'Europe',
            'United Arab Emirates': 'Asia',
            'United Kingdom': 'Europe',
            'United States': 'Americas',
            'Uruguay': 'Americas',
            'Uzbekistan': 'Asia',
            'Vanuatu': 'Oceania',
            'Vatican City': 'Europe',
            'Venezuela': 'Americas',
            'Viet Nam': 'Asia',
            'Yemen': 'Asia',
            'Zambia': 'Africa',
            'Zimbabwe': 'Africa'
        }
        data['Region'] = data['Area'].map(country_to_region).fillna('Other')
        regional_policy_counts = data.groupby(['Region', 'Indicator']).size().unstack(fill_value=0).sum(axis=1)
        plt.figure(figsize=(12, 6))
        regional_policy_counts.plot(kind='bar', color='teal')
        plt.title('Number of Policies Adopted by Region')
        plt.xlabel('Region')
        plt.ylabel('Number of Policies')
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(plt)

    elif selected_analysis == "Policy Clustering":
        st.subheader("Policy Clustering")

        # Preparing the data
        country_policies = data.groupby('Area')['Indicator'].apply(set).reset_index()
        mlb = MultiLabelBinarizer()
        policy_matrix = mlb.fit_transform(country_policies['Indicator'])

        # KMeans Clustering
        kmeans = KMeans(n_clusters=5, random_state=42)
        clusters = kmeans.fit_predict(policy_matrix)
        country_policies['Cluster'] = clusters

        # Displaying which countries belong to which clusters
        for i in range(5):  # Assuming 5 clusters
            cluster_countries = country_policies[country_policies['Cluster'] == i]['Area'].tolist()
            st.write(f"Cluster {i + 1}: {', '.join(cluster_countries)}")

        # Silhouette Score
        silhouette_avg = silhouette_score(policy_matrix, clusters)
        st.write("Silhouette Score for KMeans Clustering:", silhouette_avg)

        # PCA for visualization
        pca = PCA(n_components=2)
        policy_matrix_2d = pca.fit_transform(policy_matrix)
        plt.figure(figsize=(10, 6))
        scatter = sns.scatterplot(x=policy_matrix_2d[:, 0], y=policy_matrix_2d[:, 1], hue=clusters, palette="viridis",
                                  legend=None)
        plt.title("Policy Clustering based on HIV-Related Policies")
        plt.xlabel("First Principal Component - Var Explained: {:.2%}".format(pca.explained_variance_ratio_[0]))
        plt.ylabel("Second Principal Component - Var Explained: {:.2%}".format(pca.explained_variance_ratio_[1]))

        # Manually add legend with more descriptive labels
        handles = [mpatches.Patch(color=color, label=f'Cluster {label + 1}') for label, color in
                   enumerate(sns.color_palette("viridis", n_colors=len(set(clusters))))]
        plt.legend(handles=handles, title="Clusters")

        st.pyplot(plt)

else:
    st.info('Please upload a dataset to start the analysis.')

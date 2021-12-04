### A Pluto.jl notebook ###
# v0.17.1

using Markdown
using InteractiveUtils

# ╔═╡ f6926a24-41bb-488f-962d-d65dd025f756
using DataFrames, CSV, Dates, HTTP, PlutoUI, Plots, StatPlots, Statistics, Flux, GraphIO, Graphs, GraphPlot, PyCall, Conda

# ╔═╡ 09586d02-bc6a-40cf-a6f3-f0ef94f9febd
using Lathe

# ╔═╡ a5641a7c-0555-45f1-9457-7dbe27a5af31
using Lathe.preprocess: TrainTestSplit

# ╔═╡ a10d4b6e-43a6-4271-a20c-a27ca667011e
begin
	using GLM
	using StatsBase
	using MLBase
	using ROCAnalysis
end

# ╔═╡ e4bc3079-697a-4154-a56b-2163d0839852
using ScikitLearn

# ╔═╡ 05aab783-a9ef-49b7-848a-858d6bc9c875
using ScikitLearn: fit!, predict

# ╔═╡ 3526950c-7f02-47e3-8ebb-251010f605ee
using ScikitLearn.CrossValidation: cross_val_score

# ╔═╡ 6f128cc5-66df-4f58-803c-0dba77342083
using DecisionTree

# ╔═╡ 3240aaf8-3be2-4d08-a25b-594bc8f8ab98
md"""
# CAP 6318 - Final Project 
## Twitch Gamers Dataset
### Justin Kim, Louis Mitchell, & Syed Muhammad Sabih
"""

# ╔═╡ eefce769-e42a-4a5e-8c19-a14483cd3889
md"""
### MOTIVATION
"""

# ╔═╡ 87036def-7c5f-4b87-958f-957fc4e43405
md"""
The motivation behind the project topic is our shared interest in the Gaming Industry and 
domain knowledge regarding Social Network Analysis, which incorporates our analytical and 
insight for Gaming to produce findings related to the Research Problem.
"""

# ╔═╡ 1469bac3-ce83-4128-8b33-07402c91cee5
md"""
### PROJECT OVERVIEW
"""

# ╔═╡ 94a60d30-3d8d-465d-a18b-674a8bb9e83d
md"""
We have decided to use the concepts learned in this class regarding Graph Theory to do a 
social network analysis on the Twitch Gamers Social Network Dataset. The dataset is taken 
from the SNAP data repository which exists in the Stanford University library. The data was 
collected from the public API in 2018. The Nodes in this dataset represent Twitch users and the 
edges between them represent mutual follower relationships between them. Altogether, the 
nodes and edges form a strongly connected component. There are no missing attributes.
"""

# ╔═╡ 1230245b-a6b4-4256-9ee0-6256e919d3bf
md"""
### FEATURES FOR THE NODES
"""

# ╔═╡ 4af40015-6fc9-406d-a922-220ab7348744
md"""
- views (View count for the streamer)
- mature (Stream is suited for mature viewers)
- life_time (Days on twitch)
- created_at (When the twitch account was created)
- updated_at (How recently they have posted - relative to 2018)
- numeric_id (Unique ID representing the streamer)
- dead_account (If the Twitch account is Active/Dead)
- language (Language for the stream)
- affiliate (If the streamer is Twitch Affiliate/Partner)
- Number of Nodes: 168,114
- Number of Edges: 6,797,557
- Directed or Undirected: Undirected Graph 
"""

# ╔═╡ f90ee966-b34d-41d7-aa4e-bfd228bf2326
md"""
### RESEARCH PROBLEM
"""

# ╔═╡ 50260c0f-2547-44d5-b0d6-99ebe629a079
md"""
Investigating the impact of content creators on the Twitch Gaming Streaming platform as it 
relates to viewership and monetary acquisitions to affiliation
"""

# ╔═╡ 3333fe7c-363f-43c4-bddc-d6cbda66bcd6
md"""
### OBJECTIVES
"""

# ╔═╡ 31540eb1-c35c-4b8f-928c-8fc073b4c307
md"""
- To produce a graph that shows the relationships between all of the twitch users and  identify any communities/clusters between the network based on the features provided.
- To Create a Machine Learning Algorithm that will predict the likelihood of the affiliation of the Twitch streamer based on the aforementioned feature attributes.
"""

# ╔═╡ 931a48af-7a07-4bf0-a782-e1efbfea909d
md"""
### RESEARCH QUESTIONS
"""

# ╔═╡ 58a4676f-0b74-4ddc-84f6-7485b6759314
md"""
- Which Twitch Streamers in the data are most influential? (Betweenness Centrality)
- Which attribute/feature is important in the analysis? (Feature Engineering, F-test)
- How many separate clusters/communities are present in the Network? Which Twitch  Users have mutual relationships, or share the common language etc.?
- Can we predict Twitch affiliation based on viewership, type of content provided (mature/PG), Twitch account age, and Language?
- Overall, do Twitch streamers with high count of viewers usually share relationships (mutual followers) with other streamers with high count of viewers, OR is there an inverse relationship?
- Overall, do Twitch affiliate streamers usually share relationships (mutual followers) with other Twitch affiliate streamers, OR is there an inverse relationship?
"""

# ╔═╡ e2a1a04b-1ac5-4728-9470-6d4525213f0f
md"""
### ACCESSING THE DATA
"""

# ╔═╡ 8f196927-2f16-43ce-b015-aef96240c5a9
md"""
To access the data, we used the SNAP Repository on the Stanford University Website. We used the Twitch Gamers Social Network dataset from the link: https://snap.stanford.edu/data/twitch_gamers.zip.

After downloading the zip file and extracting the files. We uploaded the CSV files for the dataset along with our local directory to Syed's Github Repository. This would allow the files to be accessed more efficiently into the Julia environment.

To load the CSV files into the Julia environment, we can directly grab the CSV's from Github using the read remote csv(url) function defined in this notebook.
"""

# ╔═╡ a3a0cea0-4b04-11ec-2ecc-b993f598f926
md"""
### LOADING THE DATASET
"""

# ╔═╡ a81e4ca1-a160-4637-8349-730c70035533
md"""
##### Nodes and Edges Data
"""

# ╔═╡ 63527852-1244-42b5-8dc5-c3194deba174
read_remote_csv(url) = DataFrame(CSV.File(HTTP.get(url).body))

# ╔═╡ 95f2146b-6532-4a16-8bfe-b1b265fe6bd6
df1 = read_remote_csv("https://github.com/smsabih/JLS_ANALYTICS_FINAL_PROJECT/raw/main/large_twitch_edges.csv")

# ╔═╡ bb66558c-7587-496b-b322-38d3efd7be4c
md"""
##### Features for the Twitch Users Data
"""

# ╔═╡ 35245072-0b15-4c8b-bf6a-ec48edab7776
df2 = read_remote_csv("https://github.com/smsabih/JLS_ANALYTICS_FINAL_PROJECT/raw/main/large_twitch_features.csv")

# ╔═╡ b15f8f75-46bc-4730-b8cd-7eeb12d01046
md"""
### EXPLORATORY DATA ANALYSIS
"""

# ╔═╡ c1b676ab-7514-463b-acac-85afb8d4ecc9
md"""
##### Unique Values - Nodes and Edges Data
"""

# ╔═╡ 375a9719-0f4b-4389-ae7f-39b6d82a81bf
begin
	unique_values_nodes_id1 = unique(df1[!, 1])
	unique_values_nodes_id2 = unique(df1[!, 2])
	unique_values_nodes_id1, unique_values_nodes_id2
end

# ╔═╡ 60d02ba5-9bee-4db7-8e5e-40fc39586272
md"""
##### Unique Values - Features for the Twitch Users Data
"""

# ╔═╡ cea7c852-4c86-4bf4-a41a-e82bca37ffae
begin
	Views = unique(df2[!, 1])
	Mature = unique(df2[!, 2])
	Life_time = unique(df2[!, 3])
	Created_at = unique(df2[!, 4])
	Updated_at = unique(df2[!, 5])
	Numeric_id = unique(df2[!, 6])
	dead_account = unique(df2[!, 7])
	language = unique(df2[!, 8])
	affiliate = unique(df2[!, 9])
	Views, Mature, Life_time, Created_at, Updated_at, Numeric_id, dead_account, 
	language, affiliate
end

# ╔═╡ b3be9d3c-9b08-41e6-bae0-d19ce27041da
md"""
##### Shape of Dataset - Nodes and Edges Data
"""

# ╔═╡ 0fc7329d-c245-48bc-b3d5-e366209bfd26
size(df1)

# ╔═╡ 59d2f136-b789-4750-9f88-a481637e58ad
md"""
##### Shape of the Dataset - Features for the Twitch Users Data
"""

# ╔═╡ 9f9381ae-5953-4112-8a05-21d76cabfafe
size(df2)

# ╔═╡ 20a5bedb-0726-4a9b-b837-be57defb6a3a
md"""
##### List of columns - Nodes and Edges Data
"""

# ╔═╡ c9d0da04-df05-4869-ba0e-d34a11a90dfb
names(df1) 

# ╔═╡ 232d6a0c-fd94-4596-a41a-213c10fd5076
md"""
##### List of columns - Features for the Twitch Users Data
"""

# ╔═╡ 35f9a901-c7f4-4561-8e24-f62598b167de
names(df2) 

# ╔═╡ b7bf2aa8-4d6b-4f58-bf6b-062e049b8296
md"""
##### Count of Unique Values - Nodes and Edges Data
"""

# ╔═╡ 614aa0dc-66a2-4a4f-9ed2-d902aad255a1
begin
	count_unique_values_nodes_id1 = length(unique_values_nodes_id1)
	count_unique_values_nodes_id2 = length(unique_values_nodes_id2)
	count_unique_values_nodes_id1, count_unique_values_nodes_id2
	with_terminal() do
		println("numeric_id_1: ", count_unique_values_nodes_id1)
		println("numeric_id_2: ", count_unique_values_nodes_id2)
	end
end

# ╔═╡ 0b410342-8190-4dc7-9dac-24ee09d8953f
md"""
Note: We can see that the number of nodes in numeric id 1 are different from numeric id 2. Which means that column 1 nodes have more edges compared to nodes in column 2.
"""

# ╔═╡ 512fce0c-accb-44b9-88e6-975a630065ea
md"""
##### Count of Unique Values - Features for the Twitch Users Data
"""

# ╔═╡ 3553dff9-2efd-41b1-8306-4498bef05220
begin
	count_Views = length(Views)
	count_Mature = length(Mature)
	count_Life_time = length(Life_time)
	count_Created_at = length(Created_at)
	count_Updated_at = length(Updated_at)
	count_Numeric_id = length(Numeric_id)
	count_dead_account = length(dead_account)
	count_language = length(language)
	count_affiliate = length(affiliate)
	with_terminal() do
		println("Views: ", count_Views)
		println("Mature: ", count_Mature)
		println("Life Time: ", count_Life_time)
		println("Created at: ", count_Created_at)
		println("Updated at: ", count_Updated_at)
		println("Numeric id: ", count_Numeric_id)
		println("Dead account: ", count_dead_account)
		println("Language: ", count_language)
		println("Affiliate: ", count_affiliate)
	end
end

# ╔═╡ fbe653ea-5194-465a-aaeb-e77a2ad3022e
md"""
### DESCRIPTIVE STATISTICS
"""

# ╔═╡ ef857760-9ab5-42c8-ba73-1453545fa28e
md"""
##### Nodes and Edges Dataset
"""

# ╔═╡ f8dbd0ca-7588-4dfa-924f-4551090785e3
counts_df1 = combine(groupby(df1, :numeric_id_1), :numeric_id_1=>length)

# ╔═╡ cb4a85f5-f13e-4016-bb8f-a6e58e5f942f
describe(counts_df1)

# ╔═╡ d775da74-1e40-4599-b3ce-1a17ebf2a34a
 counts_df2 = combine(groupby(df1, :numeric_id_2), :numeric_id_2=>length)

# ╔═╡ d5ba919c-24cc-41aa-97e9-01a5fbb52f5c
describe(counts_df2)

# ╔═╡ 08220005-2405-4cc7-a450-7a8e67c78bb9
md"""
Note: The figures above shows the edge counts for each node represented by the unique numeric id columns and the descriptive statistics for each numeric id column.

After investigating the figures above, the findings elucidate that there are fewer nodes in numeric id 1 column that connect to numerous nodes, compared to numeric id 2 where more nodes connect to fewer nodes.

"""

# ╔═╡ 2e823df4-76f6-4621-a56a-a29a4a4ddbff
md"""
##### Features for the Twitch Users Data
"""

# ╔═╡ dee47806-c344-47e4-8d83-6249150a36e4
describe(df2, :all)

# ╔═╡ 6495eb72-d48f-4205-9f80-f8adc51ed1e0
md"""
Note: No missing values are found.
"""

# ╔═╡ 6cbb913b-b724-4bf7-abaa-515bf21f12cd
md"""
##### Visualization and Analysis of Viewership
"""

# ╔═╡ 100c654a-f24d-4d27-8357-f0df29dcf059
Plots.plot(df2[!, :views], seriestype = :scatter, title = "Distribution of views")

# ╔═╡ 8fe39705-4ce6-4c8b-832f-6410c6040c36
md"""
Note: From the distribution above, it is evident that most of the streamers have lower views and only few reach the high view mark. We can also observe that majority of the streamers are around the hundred million views.
"""

# ╔═╡ 22dc8567-22c4-40ef-a364-5c61b57a9f38
Plots.boxplot(df2[!, :views])

# ╔═╡ 194582b0-3bbd-4977-9e49-d8bb1bf4615c
md"""
Note: From the boxplot above, we can see the same observation that we see from the Distribution view.
"""

# ╔═╡ 08d201ac-8208-4138-b5bd-f724f43dbc52
md"""
### CLEANING THE FEATURES DATASET
###### Changing the Dates to relative days from today
"""

# ╔═╡ c3ced2f2-a065-4f46-acb9-ce44f2c03ec1
function mod_year(x)
	return Dates.today() - x
end

# ╔═╡ 23072f60-5cf6-441d-bc90-973e9d6ec0c4
df2[! ,:created_at] = mod_year(df2[!, :created_at])

# ╔═╡ 90c45f07-1123-4515-ade8-a20e1c0542f3
df2[! ,:updated_at] = mod_year(df2[!, :updated_at])

# ╔═╡ 311b7852-1640-4277-acb0-8caa1e0e0762
md"""
###### Changing the Days to integers
"""

# ╔═╡ 2ab7d7c5-ca5d-4b8f-9e2c-72a2ee650894
function mod_day(x)
	return Dates.value.(x)
end

# ╔═╡ a2ca9c5a-cba9-489b-9904-f72738bd1313
df2[! ,:created_at] = mod_day(df2[!, :created_at])

# ╔═╡ 0e0f83b2-938f-4e84-aa0c-470d16c6b390
df2[! ,:updated_at] = mod_day(df2[!, :updated_at])

# ╔═╡ 2eb602c3-8f17-43fc-a300-e75fe4e08452
md"""
##### Updated Features Dataset after changing date
"""

# ╔═╡ 6cb34c59-03fe-4d8c-8adf-e72a980b5779
df2

# ╔═╡ a1b85414-de6e-46fc-bab1-19cacb51e3d4
md"""
##### Dummy Coding the language variable
"""

# ╔═╡ aa503a18-0ee1-4791-8256-1c9b0ce2eada
#Replacing EN with 1 and other languages with 0
df2[!, :language] = replace(df2[!, :language], "EN"=>1,
"FR"=>0,
"KO"=>0,
"JA"=>0,
"RU"=>0,
"PL"=>0,
"DE"=>0,
"ES"=>0,
"IT"=>0,
"PT"=>0,
"OTHER"=>0,
"TR"=>0,
"ZH"=>0,
"SV"=>0,
"NL"=>0,
"TH"=>0,
"CS"=>0,
"DA"=>0,
"HU"=>0,
"FI"=>0,
"NO"=>0)

# ╔═╡ 8e3e3199-c8fc-4b76-b110-e103a28802b3
md"""
##### Updated Features Dataset after Dummy coding the Categorical language variable
"""

# ╔═╡ 3a6f467d-54f9-4e0e-be13-5688bda941eb
df2

# ╔═╡ 175e7045-71b9-44f5-86ca-653d76bdf047
md"""
Note: The language variabe has been changed to binary. EN language variable has been changed to 1 while all other languages to 0.\
According to the research page "https://www.statista.com/statistics/511558/twitch-traffic-by-country/
", the US holds a majority of twitch account users by a large margin.
"""

# ╔═╡ 2665a487-0e19-4f05-84cb-abf0e888a30b
md"""
###### All the variables from the features dataset are in numeric or categorical form.
"""

# ╔═╡ 57de6a90-eafe-4fee-bab9-afb7c090e27f
md"""
###### The next steps are to fix any skewness, check for imbalance in affiliation class variable, plot the initial graph for the Network using nodes and edges dataset, and answer the research questions.
"""

# ╔═╡ a661f1e7-a1c1-4deb-9e1b-861873e88d82
md"""
### THE INITIAL GRAPH FOR THE NODES AND EDGES DATASET
"""

# ╔═╡ 51c237c4-5ded-44f4-b9d4-f90263bd0253
md"""
###### Using Julia package PyCall and Python packages pandas, networkx, and matplotlib to plot the initial graph for nodes and edges.

"""

# ╔═╡ f48573fc-95b7-40a6-9a1e-78315bbe37d3
begin
pd = pyimport("pandas")
nx = pyimport("networkx")
plt = pyimport("matplotlib.pyplot")
end

# ╔═╡ a07482c6-fbf5-4d87-8ada-5b574fb02246
df1_pd = pd.read_csv("https://github.com/smsabih/JLS_ANALYTICS_FINAL_PROJECT/raw/main/large_twitch_edges.csv")

# ╔═╡ a7f520b8-f5bd-4996-a5c2-7b2578f04c43
G = nx.from_pandas_edgelist(df1_pd, "numeric_id_1", "numeric_id_2")

# ╔═╡ 5dca076f-3741-43d6-8515-f41e0248b27b
G_1 = nx.Graph(G)

# ╔═╡ 443d7404-ae6b-4e38-9a38-f47a723986f8
df1_pd2 = df1_pd.head(5000)

# ╔═╡ 330fcd4f-82ee-459b-b4e2-12ae8367e3e9
F = nx.from_pandas_edgelist(df1_pd2, "numeric_id_1", "numeric_id_2")

# ╔═╡ 8c1185de-4a4c-45db-a5b3-89204796af36
F_1 = nx.Graph(F)

# ╔═╡ 5e483882-a375-4a3f-a98f-ea2aad376092
nx.draw_networkx(F_1)

# ╔═╡ d1eab486-39af-4ef1-998a-f9b07d6bf40c
plt.show()

# ╔═╡ a4020f2a-604e-46e3-aa2b-a5a43373dc6c
md"""
Note: Due to processing times, the first 5000 datapoints from the nodes and edges dataset were plotted using the networkx package from the python library. Due to the size of the dataset, not much could be interpreted from the initial graph, other than a few hubs.\
Due to this, an analysis of the research problem using Machine Learning would be a better option in the next part of this project.
"""

# ╔═╡ 6b715f0b-9258-4c43-a26c-0808ce65154b
md"""
### MODELING TO SOLVE RESEARCH PROBLEM
"""

# ╔═╡ 594147ec-3407-4cc2-ad41-fd872be13229
# DATASET AT THIS POINT
df2

# ╔═╡ def9e444-131e-444d-9867-ef01396b8c37
md"""
#### BASE MODELS
"""

# ╔═╡ 69fcb1b5-3580-4cec-a30f-b1a5b904a912
md"""
###### Using the ScikitLearn package from the Julia library
"""

# ╔═╡ b5a3600d-e87a-48aa-93a5-cbbc69b687d2
md"""
#### Preprocessing
"""

# ╔═╡ 0589811c-c94b-4e7a-9d63-b567dfc25705
@sk_import linear_model: LogisticRegression

# ╔═╡ 662c558e-d93a-4010-827f-8b4c23ea0c52
md"""
##### Checking column names
"""

# ╔═╡ 6c91c3b6-53cf-41f8-ad57-ce1ad54eb6b1
names(df2)

# ╔═╡ 5aed60ca-6db6-4bc3-91bf-685833409d20
md"""
##### Checking class imbalance
"""

# ╔═╡ e9947565-f679-4bbd-9378-2b4a2b5727b0
counts_df2_affiliate = combine(groupby(df2, :affiliate), :affiliate=>length)

# ╔═╡ 96d3577a-fb89-44c9-84f1-150d7d73bcae
begin
	percentage_class = counts_df2_affiliate[!, :affiliate_length]
	ratio = percentage_class[2]/(percentage_class[1]+percentage_class[2])
end

# ╔═╡ 45efafaa-2bad-4394-a383-a6b5a79f3f26
md"""
Notes: Class seems balanced.
"""

# ╔═╡ c50a699b-790e-43ca-9413-64d2ce7e9295
md"""
##### Train Test Split
"""

# ╔═╡ 9065469d-db09-47f8-be67-9d9915bca6bf
md"""
###### Training dataset 75% - Testing Dataset 25%
"""

# ╔═╡ 26e63026-8ea9-443a-ab33-34d19889eaf3
train, test = TrainTestSplit(df2,.75)

# ╔═╡ 1afb97a2-8d8e-4c7c-8b9e-7dacc11249c6
md"""
Note: Due to the size of the dataset, we deemed it sufficient to use 75% of the data as training and 25% as testing.
"""

# ╔═╡ 879c5498-b877-44ee-9352-d0982391832e
begin
	#TRAIN DATASET
	X_train = convert(Array, train[[:views, :mature, :life_time, :created_at, 
		:updated_at, :language]])
	y_train = convert(Array, train[:affiliate])
	
	#TEST DATASET
	X_test = convert(Array, test[[:views, :mature, :life_time, :created_at, 
		:updated_at, :language]])
	y_test = convert(Array, test[:affiliate])
end

# ╔═╡ d4d47777-22af-4c63-80cd-ecabe6279509
md"""
#### Regression Algorithm
"""

# ╔═╡ 1f66bc09-e6e0-4bab-9eb9-74f97a3d5a5b
md"""
##### Logistic Regression
"""

# ╔═╡ cbb45603-f310-4556-aa61-bb56a95c2140
model = LogisticRegression(fit_intercept=true)

# ╔═╡ d51e8039-defb-4ff1-b0bc-3ca8785c8b76
# Train the model. 
fit!(model, X_train, y_train)

# ╔═╡ d7b559e7-fe91-4eed-9882-7f9920cd9ee5
begin
	accuracy_logistic = sum(predict(model, X_test) .== y_test) / length(y_test)
	with_terminal() do
		println("accuracy: $accuracy_logistic")
	end
end

# ╔═╡ dd36605d-91e8-4e0e-8b00-b6aabbcc3109
cross_val_logistic = cross_val_score(LogisticRegression(), X_train, y_train; cv=5)

# ╔═╡ 8a263263-7ee7-4033-bba7-4c74247d0eb2
md"""
Note: The Base Logistic Regression Model score is low.
"""

# ╔═╡ 268c190c-3bca-43ff-93b3-4cd7ddefa731
md"""
#### Tree Algorithms
"""

# ╔═╡ 39a8003c-2b53-45b6-a101-d634156304d1
md"""
##### Decision Tree Classifier
"""

# ╔═╡ be8a5ec7-fcab-4f89-a4ce-a396416c0f67
md"""
###### Using DecisionTree package from the Julia library.
"""

# ╔═╡ b5476740-7825-4dfc-b309-d99a754967da
model_CART = DecisionTreeClassifier()

# ╔═╡ 9728951b-61b5-4344-b989-6de2209da8c6
features = X_train

# ╔═╡ 4c68c76e-62a7-4ecb-a737-74c814fb041a
labels = y_train

# ╔═╡ 82b774cd-7254-49b7-9293-f0c3dad7e431
fit!(model_CART, features, labels)

# ╔═╡ 77e121f0-a72b-4e59-b2a0-9a3dcfad2901
begin
	accuracy_CART = sum(predict(model_CART, X_test) .== y_test) / length(y_test)
	with_terminal() do
		println("accuracy: $accuracy_CART")
	end
end

# ╔═╡ eb312d12-c2ba-416a-9735-ebcf5e190918
cross_val_CART = cross_val_score(model_CART, features, labels, cv=5)

# ╔═╡ bc97c003-f8db-4fa9-b9c4-7203b80b3bef
md"""
##### Random Forest Classifier
"""

# ╔═╡ 044c883d-a9e1-49de-a2c2-b775728632ac
model_RFC = RandomForestClassifier()

# ╔═╡ 579f31dd-a765-498d-b0ae-5093672ba65f
fit!(model_RFC, features, labels)

# ╔═╡ 87759a8f-0b4b-41ce-88a7-27dacb30d9e5
begin
	accuracy_RFC = sum(predict(model_CART, X_test) .== y_test) / length(y_test)
	with_terminal() do
		println("accuracy: $accuracy_RFC")
	end
end

# ╔═╡ 2616a5dc-4a16-4581-a757-50acf50ff3ea
cross_val_RFC = cross_val_score(model_RFC, features, labels, cv=5)

# ╔═╡ ed87de0c-7ac6-4393-a902-cc4af64f2c78
md"""
##### Random Forest Regressor
"""

# ╔═╡ 98dc6349-0cb6-4bd4-98e1-9f5253c75d65
model_RFR = RandomForestRegressor()

# ╔═╡ 61bfbbf8-dc65-4953-a855-0766b1ee0861
fit!(model_RFR, features, labels)

# ╔═╡ 60bb22e2-1b8a-4aee-8c46-ca83db675cb5
begin
	accuracy_RFR = sum(predict(model_RFR, X_test) .== y_test) / length(y_test)
	with_terminal() do
		println("accuracy: $accuracy_RFR")
	end
end

# ╔═╡ a014cb4c-edeb-4124-8feb-81339742e195
cross_val_RFR = cross_val_score(model_RFR, features, labels, cv=5)

# ╔═╡ b49a0dff-0901-4eaa-bbf4-750ecfcfd096
md"""
##### AdaBoost Stump Classifier
"""

# ╔═╡ 291a07b8-6903-4364-8057-8780a87818d4
model_ADA = AdaBoostStumpClassifier()

# ╔═╡ 366ce026-ccd8-4b6e-bcc6-438486bd9e60
fit!(model_ADA, features, labels)

# ╔═╡ f4da99ea-bacb-487e-9843-474e88a6305c
begin
	accuracy_ADA = sum(predict(model_ADA, X_test) .== y_test) / length(y_test)
	with_terminal() do
		println("accuracy: $accuracy_ADA")
	end
end

# ╔═╡ 69d8267e-8f28-4d30-a9e7-89075ed27a6a
cross_val_ADA = cross_val_score(model_ADA, features, labels, cv=5)

# ╔═╡ e5f90438-eb52-455a-a968-8ea76f0589fd
md"""
##### Results of Base Algorithms
"""

# ╔═╡ 3b19829c-434d-4fd7-8195-ec847a055bbd
Results_Base_Algorithm = DataFrame(Algorithm =["Logistic Regression", "Decision Tree Classifier", 
	"Random Forest Regressor", "Random Forest Classifier", "AdaBoost Stump Classifier"] 
	,Accuracy = [accuracy_logistic, accuracy_CART, accuracy_RFR, accuracy_RFC, 
	accuracy_ADA])

# ╔═╡ 1f375328-b435-4b5b-9fd1-2647bfb09a36
maximum(Results_Base_Algorithm[!, :Accuracy])

# ╔═╡ 93bc4c0a-4921-48ad-8c63-bd899b49a594
md"""
Note: AdaBoost Stump Classifier performs at the highest efficiency in predicting the target variable. We will try to tune the hyperparameters for this algorithm.
"""

# ╔═╡ a0878f0c-b4a6-4c5e-b9ea-4e9b2f070c49
md"""
#### Hyperparameter Tuning
"""

# ╔═╡ 5eee4ca3-61f7-46fc-a85e-697045a98a41


# ╔═╡ 502050f5-c3a8-498e-89fa-231f0f11d7e1


# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
CSV = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
Conda = "8f4d0f93-b110-5947-807f-2305c1781a2d"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
Dates = "ade2ca70-3891-5945-98fb-dc099432e06a"
DecisionTree = "7806a523-6efd-50cb-b5f6-3fa6f1930dbb"
Flux = "587475ba-b771-5e3f-ad9e-33799f191a9c"
GLM = "38e38edf-8417-5370-95a0-9cbb8c7f171a"
GraphIO = "aa1b3936-2fda-51b9-ab35-c553d3a640a2"
GraphPlot = "a2cc645c-3eea-5389-862e-a155d0052231"
Graphs = "86223c79-3864-5bf0-83f7-82e725a168b6"
HTTP = "cd3eb016-35fb-5094-929b-558a96fad6f3"
Lathe = "38d8eb38-e7b1-11e9-0012-376b6c802672"
MLBase = "f0e99cf1-93fa-52ec-9ecc-5026115318e0"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
PyCall = "438e738f-606a-5dbb-bf0a-cddfbfd45ab0"
ROCAnalysis = "f535d66d-59bb-5153-8d2b-ef0a426c6aff"
ScikitLearn = "3646fa90-6ef7-5e7e-9f22-8aca16db6324"
StatPlots = "60ddc479-9b66-56df-82fc-76a74619b69c"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"

[compat]
CSV = "~0.8.5"
Conda = "~1.6.0"
DataFrames = "~0.20.2"
DecisionTree = "~0.10.11"
Flux = "~0.8.3"
GLM = "~1.4.2"
GraphIO = "~0.5.0"
GraphPlot = "~0.3.1"
Graphs = "~0.10.3"
HTTP = "~0.9.17"
Lathe = "~0.0.9"
MLBase = "~0.8.0"
Plots = "~0.29.9"
PlutoUI = "~0.7.1"
PyCall = "~1.92.5"
ROCAnalysis = "~0.3.0"
ScikitLearn = "~0.6.2"
StatPlots = "~0.9.2"
StatsBase = "~0.32.2"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

[[AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "485ee0867925449198280d4af84bdb46a2a404d0"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.0.1"

[[AbstractTrees]]
deps = ["Markdown", "Test"]
git-tree-sha1 = "6621d9645702c1c4e6970cc6a3eae440c768000b"
uuid = "1520ce14-60c1-5f80-bbc7-55ef81b5835c"
version = "0.2.1"

[[Adapt]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "fd04049c7dd78cfef0b06cdc1f0f181467655712"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "1.1.0"

[[ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"

[[ArnoldiMethod]]
deps = ["DelimitedFiles", "LinearAlgebra", "Random", "SparseArrays", "StaticArrays", "Test"]
git-tree-sha1 = "2b6845cea546604fb4dca4e31414a6a59d39ddcd"
uuid = "ec485272-7323-5ecc-a04f-4719b315124d"
version = "0.0.4"

[[Arpack]]
deps = ["Arpack_jll", "Libdl", "LinearAlgebra"]
git-tree-sha1 = "2ff92b71ba1747c5fdd541f8fc87736d82f40ec9"
uuid = "7d9fca2a-8960-54d3-9f78-7d1dccf2cb97"
version = "0.4.0"

[[Arpack_jll]]
deps = ["Libdl", "OpenBLAS_jll", "Pkg"]
git-tree-sha1 = "e214a9b9bd1b4e1b4f15b22c0994862b66af7ff7"
uuid = "68821587-b530-5797-8361-c406ea357684"
version = "3.5.0+3"

[[ArrayInterface]]
deps = ["LinearAlgebra", "Requires", "SparseArrays"]
git-tree-sha1 = "a2a1884863704e0414f6f164a1f6f4a2a62faf4e"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "2.14.17"

[[Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[AxisAlgorithms]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "WoodburyMatrices"]
git-tree-sha1 = "66771c8d21c8ff5e3a93379480a2307ac36863f7"
uuid = "13072b0f-2c55-5437-9ae7-d433b7a33950"
version = "1.0.1"

[[Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[BinaryProvider]]
deps = ["Libdl", "Logging", "SHA"]
git-tree-sha1 = "ecdec412a9abc8db54c0efc5548c64dfce072058"
uuid = "b99e7846-7c00-51b0-8f62-c81ae34c0232"
version = "0.5.10"

[[Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "19a35467a82e236ff51bc17a3a44b69ef35185a2"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+0"

[[CSV]]
deps = ["Dates", "Mmap", "Parsers", "PooledArrays", "SentinelArrays", "Tables", "Unicode"]
git-tree-sha1 = "b83aa3f513be680454437a0eee21001607e5d983"
uuid = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
version = "0.8.5"

[[Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "f2202b55d816427cd385a9a4f3ffb226bee80f99"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.16.1+0"

[[Calculus]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f641eb0a4f00c343bbc32346e1217b86f3ce9dad"
uuid = "49dc2e85-a5d0-5ad3-a950-438e2897f1b9"
version = "0.5.1"

[[CategoricalArrays]]
deps = ["Compat", "DataAPI", "Future", "JSON", "Missings", "Printf", "Reexport", "Statistics", "Unicode"]
git-tree-sha1 = "23d7324164c89638c18f6d7f90d972fa9c4fa9fb"
uuid = "324d7699-5711-5eae-9e2f-1d82baa6b597"
version = "0.7.7"

[[ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "f885e7e7c124f8c92650d61b9477b9ac2ee607dd"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.11.1"

[[ChangesOfVariables]]
deps = ["LinearAlgebra", "Test"]
git-tree-sha1 = "9a1d594397670492219635b35a3d830b04730d62"
uuid = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
version = "0.1.1"

[[Clustering]]
deps = ["Distances", "LinearAlgebra", "NearestNeighbors", "Printf", "SparseArrays", "Statistics", "StatsBase"]
git-tree-sha1 = "75479b7df4167267d75294d14b58244695beb2ac"
uuid = "aaaa29a8-35af-508c-8bc3-b662a17a0fe5"
version = "0.14.2"

[[CodecZlib]]
deps = ["BinaryProvider", "Libdl", "TranscodingStreams"]
git-tree-sha1 = "05916673a2627dd91b4969ff8ba6941bc85a960e"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.6.0"

[[ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "7b62b728a5f3dd6ee3b23910303ccf27e82fad5e"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.8.1"

[[Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "InteractiveUtils", "Printf", "Reexport"]
git-tree-sha1 = "c9c1845d6bf22e34738bee65c357a69f416ed5d1"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.9.6"

[[CommonSubexpressions]]
deps = ["MacroTools", "Test"]
git-tree-sha1 = "7b8a93dba8af7e3b42fecabf646260105ac373f7"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.0"

[[Compat]]
deps = ["Base64", "Dates", "DelimitedFiles", "Distributed", "InteractiveUtils", "LibGit2", "Libdl", "LinearAlgebra", "Markdown", "Mmap", "Pkg", "Printf", "REPL", "Random", "SHA", "Serialization", "SharedArrays", "Sockets", "SparseArrays", "Statistics", "Test", "UUIDs", "Unicode"]
git-tree-sha1 = "dce3e3fea680869eaa0b774b2e8343e9ff442313"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "3.40.0"

[[CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"

[[Compose]]
deps = ["Base64", "Colors", "DataStructures", "Dates", "IterTools", "JSON", "LinearAlgebra", "Measures", "Printf", "Random", "Requires", "Statistics", "UUIDs"]
git-tree-sha1 = "c6461fc7c35a4bb8d00905df7adafcff1fe3a6bc"
uuid = "a81c6b42-2e10-5240-aca2-a61377ecd94b"
version = "0.9.2"

[[Conda]]
deps = ["Downloads", "JSON", "VersionParsing"]
git-tree-sha1 = "6cdc8832ba11c7695f494c9d9a1c31e90959ce0f"
uuid = "8f4d0f93-b110-5947-807f-2305c1781a2d"
version = "1.6.0"

[[Contour]]
deps = ["StaticArrays"]
git-tree-sha1 = "9f02045d934dc030edad45944ea80dbd1f0ebea7"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.5.7"

[[DataAPI]]
git-tree-sha1 = "cc70b17275652eb47bc9e5f81635981f13cea5c8"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.9.0"

[[DataFrames]]
deps = ["CategoricalArrays", "Compat", "DataAPI", "Future", "InvertedIndices", "IteratorInterfaceExtensions", "Missings", "PooledArrays", "Printf", "REPL", "Reexport", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "7d5bf815cc0b30253e3486e8ce2b93bf9d0faff6"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "0.20.2"

[[DataStructures]]
deps = ["InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "88d48e133e6d3dd68183309877eac74393daa7eb"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.17.20"

[[DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[DataValues]]
deps = ["DataValueInterfaces", "Dates"]
git-tree-sha1 = "d88a19299eba280a6d062e135a43f00323ae70bf"
uuid = "e7dc6d0d-1eca-5fa6-8ad6-5aecde8b7ea5"
version = "0.4.13"

[[Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[DecisionTree]]
deps = ["DelimitedFiles", "Distributed", "LinearAlgebra", "Random", "ScikitLearnBase", "Statistics", "Test"]
git-tree-sha1 = "123adca1e427dc8abc5eec5040644e7842d53c92"
uuid = "7806a523-6efd-50cb-b5f6-3fa6f1930dbb"
version = "0.10.11"

[[DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[DiffEqDiffTools]]
deps = ["ArrayInterface", "LinearAlgebra", "Requires", "SparseArrays", "StaticArrays"]
git-tree-sha1 = "a4ed8a740484627ea41b47f7e1a25dd909a28353"
uuid = "01453d9d-ee7c-5054-8395-0335cb756afa"
version = "1.7.0"

[[DiffResults]]
deps = ["StaticArrays"]
git-tree-sha1 = "c18e98cba888c6c25d1c3b048e4b3380ca956805"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.0.3"

[[DiffRules]]
deps = ["LogExpFunctions", "NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "d8f468c5cd4d94e86816603f7d18ece910b4aaf1"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.5.0"

[[Distances]]
deps = ["LinearAlgebra", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "3258d0659f812acde79e8a74b11f17ac06d0ca04"
uuid = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
version = "0.10.7"

[[Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[Distributions]]
deps = ["FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns"]
git-tree-sha1 = "55e1de79bd2c397e048ca47d251f8fa70e530550"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.22.6"

[[DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "b19534d1895d702889b219c382a6e18010797f0b"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.8.6"

[[Downloads]]
deps = ["ArgTools", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"

[[Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b3bfd02e98aedfa5cf885665493c5598c350cd2f"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.2.10+0"

[[FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "c82bef6fc01e30d500f588cd01d29bdd44f1924e"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.3.0"

[[FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "Pkg", "Zlib_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "d8a578692e3077ac998b50c0217dfd67f21d1e5f"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.0+0"

[[FFTW]]
deps = ["AbstractFFTs", "FFTW_jll", "LinearAlgebra", "MKL_jll", "Preferences", "Reexport"]
git-tree-sha1 = "463cb335fa22c4ebacfd1faba5fde14edb80d96c"
uuid = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
version = "1.4.5"

[[FFTW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c6033cc3892d0ef5bb9cd29b7f2f0331ea5184ea"
uuid = "f5851436-0d7a-5f13-b9de-f02708fd171a"
version = "3.3.10+0"

[[FillArrays]]
deps = ["LinearAlgebra", "Random", "SparseArrays"]
git-tree-sha1 = "4863cbb7910079369e258dee4add9d06ead5063a"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "0.8.14"

[[FixedPointNumbers]]
git-tree-sha1 = "d14a6fa5890ea3a7e5dcab6811114f132fec2b4b"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.6.1"

[[Flux]]
deps = ["AbstractTrees", "Adapt", "CodecZlib", "Colors", "DelimitedFiles", "Juno", "LinearAlgebra", "MacroTools", "NNlib", "Pkg", "Printf", "Random", "Reexport", "Requires", "SHA", "Statistics", "StatsBase", "Tracker", "ZipFile"]
git-tree-sha1 = "08212989c2856f95f90709ea5fd824bd27b34514"
uuid = "587475ba-b771-5e3f-ad9e-33799f191a9c"
version = "0.8.3"

[[Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "21efd19106a55620a188615da6d3d06cd7f6ee03"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.13.93+0"

[[ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions", "StaticArrays"]
git-tree-sha1 = "6406b5112809c08b1baa5703ad274e1dded0652f"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "0.10.23"

[[FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "87eb71354d8ec1a96d4a7636bd57a7347dde3ef9"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.10.4+0"

[[FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "aa31987c2ba8704e23c6c8ba8a4f769d5d7e4f91"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.10+0"

[[Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[GLM]]
deps = ["Distributions", "LinearAlgebra", "Printf", "Random", "Reexport", "SparseArrays", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns", "StatsModels"]
git-tree-sha1 = "dc577ad8b146183c064b30e747e3afc6d6dfd62b"
uuid = "38e38edf-8417-5370-95a0-9cbb8c7f171a"
version = "1.4.2"

[[GR]]
deps = ["Base64", "DelimitedFiles", "LinearAlgebra", "Printf", "Random", "Serialization", "Sockets", "Test", "UUIDs"]
git-tree-sha1 = "7ea6f715b7caa10d7ee16f1cfcd12f3ccc74116a"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.48.0"

[[GeometryTypes]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "StaticArrays"]
git-tree-sha1 = "07194161fe4e181c6bf51ef2e329ec4e7d050fc4"
uuid = "4d00f742-c7ba-57c2-abde-4428a4b178cb"
version = "0.8.4"

[[Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "74ef6288d071f58033d54fd6708d4bc23a8b8972"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.68.3+1"

[[GraphIO]]
deps = ["DelimitedFiles", "LightGraphs", "Requires", "SimpleTraits"]
git-tree-sha1 = "0bcec60e0f5b951001beb950ed54737779ac0c83"
uuid = "aa1b3936-2fda-51b9-ab35-c553d3a640a2"
version = "0.5.0"

[[GraphPlot]]
deps = ["ArnoldiMethod", "ColorTypes", "Colors", "Compose", "DelimitedFiles", "LightGraphs", "LinearAlgebra", "Random", "SparseArrays", "Test"]
git-tree-sha1 = "f4435ce0055d4da938f3bab0c0e523826735c96a"
uuid = "a2cc645c-3eea-5389-862e-a155d0052231"
version = "0.3.1"

[[Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "344bf40dcab1073aca04aa0df4fb092f920e4011"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+0"

[[Graphs]]
deps = ["DataStructures", "SparseArrays"]
git-tree-sha1 = "9409e40f53532c45f2478c33531aa7a65ec4e2de"
uuid = "86223c79-3864-5bf0-83f7-82e725a168b6"
version = "0.10.3"

[[Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[HTTP]]
deps = ["Base64", "Dates", "IniFile", "Logging", "MbedTLS", "NetworkOptions", "Sockets", "URIs"]
git-tree-sha1 = "0fa77022fe4b511826b39c894c90daf5fce3334a"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "0.9.17"

[[HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg"]
git-tree-sha1 = "129acf094d168394e80ee1dc4bc06ec835e510a3"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "2.8.1+1"

[[Inflate]]
git-tree-sha1 = "f5fc07d4e706b84f72d54eedcc1c13d92fb0871c"
uuid = "d25df0c9-e2be-5dd7-82c8-3ad0b3e990b9"
version = "0.1.2"

[[IniFile]]
deps = ["Test"]
git-tree-sha1 = "098e4d2c533924c921f9f9847274f2ad89e018b8"
uuid = "83e8ac13-25f8-5344-8a64-a9f2b223428f"
version = "0.5.0"

[[IntelOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "d979e54b71da82f3a65b62553da4fc3d18c9004c"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2018.0.3+2"

[[InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[Interpolations]]
deps = ["AxisAlgorithms", "LinearAlgebra", "OffsetArrays", "Random", "Ratios", "SharedArrays", "SparseArrays", "StaticArrays", "WoodburyMatrices"]
git-tree-sha1 = "2b7d4e9be8b74f03115e64cf36ed2f48ae83d946"
uuid = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
version = "0.12.10"

[[InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "a7254c0acd8e62f1ac75ad24d5db43f5f19f3c65"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.2"

[[InvertedIndices]]
git-tree-sha1 = "bee5f1ef5bf65df56bdd2e40447590b272a5471f"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.1.0"

[[IrrationalConstants]]
git-tree-sha1 = "7fd44fd4ff43fc60815f8e764c0f352b83c49151"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.1.1"

[[IterTools]]
git-tree-sha1 = "05110a2ab1fc5f932622ffea2a003221f4782c18"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.3.0"

[[IterableTables]]
deps = ["DataValues", "IteratorInterfaceExtensions", "Requires", "TableTraits", "TableTraitsUtils"]
git-tree-sha1 = "70300b876b2cebde43ebc0df42bc8c94a144e1b4"
uuid = "1c8ee90f-4401-5389-894e-7a04a3dc0f4d"
version = "1.0.0"

[[IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "642a199af8b68253517b80bd3bfd17eb4e84df6e"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.3.0"

[[JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "8076680b162ada2a031f707ac7b4953e30667a37"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.2"

[[Juno]]
deps = ["Base64", "Logging", "Media", "Profile", "Test"]
git-tree-sha1 = "30d94657a422d09cb97b6f86f04f750fa9c50df8"
uuid = "e5e0dc1b-0480-54bc-9374-aad01c23163d"
version = "0.7.2"

[[KernelDensity]]
deps = ["Distributions", "FFTW", "Interpolations", "Optim", "StatsBase", "Test"]
git-tree-sha1 = "c1048817fe5711f699abc8fabd47b1ac6ba4db04"
uuid = "5ab0869b-81aa-558d-bb23-cbf5423bbe9b"
version = "0.5.1"

[[LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f6250b16881adf048549549fba48b1161acdac8c"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.1+0"

[[LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e5b909bcf985c5e2605737d2ce278ed791b89be6"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.1+0"

[[Lathe]]
deps = ["DataFrames", "Random"]
git-tree-sha1 = "5f64e72da1435568cd8362d6d0f364d210df3e9e"
uuid = "38d8eb38-e7b1-11e9-0012-376b6c802672"
version = "0.0.9"

[[LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"

[[LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"

[[LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"

[[LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"

[[Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "0b4a5d71f3e5200a7dff793393e09dfc2d874290"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+1"

[[Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll", "Pkg"]
git-tree-sha1 = "64613c82a59c120435c067c2b809fc61cf5166ae"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.8.7+0"

[[Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c333716e46366857753e273ce6a69ee0945a6db9"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.42.0+0"

[[Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "42b62845d70a619f063a7da093d995ec8e15e778"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.16.1+1"

[[Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9c30530bf0effd46e15e0fdcf2b8636e78cbbd73"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.35.0+0"

[[Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7f3efec06033682db852f8b3bc3c1d2b0a0ab066"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.36.0+0"

[[LightGraphs]]
deps = ["ArnoldiMethod", "DataStructures", "Distributed", "Inflate", "LinearAlgebra", "Random", "SharedArrays", "SimpleTraits", "SparseArrays", "Statistics"]
git-tree-sha1 = "432428df5f360964040ed60418dd5601ecd240b6"
uuid = "093fc24a-ae57-5d10-9952-331d41423f4d"
version = "1.3.5"

[[LineSearches]]
deps = ["LinearAlgebra", "NLSolversBase", "NaNMath", "Parameters", "Printf"]
git-tree-sha1 = "f27132e551e959b3667d8c93eae90973225032dd"
uuid = "d3d80556-e9d4-5f37-9878-2ab0fcc64255"
version = "7.1.1"

[[LinearAlgebra]]
deps = ["Libdl"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[LogExpFunctions]]
deps = ["ChainRulesCore", "ChangesOfVariables", "DocStringExtensions", "InverseFunctions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "be9eef9f9d78cecb6f262f3c10da151a6c5ab827"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.5"

[[Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "Pkg"]
git-tree-sha1 = "5455aef09b40e5020e1520f551fa3135040d4ed0"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2021.1.1+2"

[[MLBase]]
deps = ["IterTools", "Random", "Reexport", "StatsBase", "Test"]
git-tree-sha1 = "f63a8d37429568b8c4384d76c4a96fc2897d6ddf"
uuid = "f0e99cf1-93fa-52ec-9ecc-5026115318e0"
version = "0.8.0"

[[MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "3d3e902b31198a27340d0bf00d6ac452866021cf"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.9"

[[Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "Random", "Sockets"]
git-tree-sha1 = "1c38e51c3d08ef2278062ebceade0e46cefc96fe"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.0.3"

[[MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"

[[Measures]]
git-tree-sha1 = "e498ddeee6f9fdb4551ce855a46f54dbd900245f"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.1"

[[Media]]
deps = ["MacroTools", "Test"]
git-tree-sha1 = "75a54abd10709c01f1b86b84ec225d26e840ed58"
uuid = "e89f7d12-3494-54d1-8411-f7d8b9ae1f27"
version = "0.5.0"

[[Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "f8c673ccc215eb50fcadb285f522420e29e69e1c"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "0.4.5"

[[Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"

[[NLSolversBase]]
deps = ["Calculus", "DiffEqDiffTools", "DiffResults", "Distributed", "ForwardDiff"]
git-tree-sha1 = "f1b8ed89fa332f410cfc7c937682eb4d0b361521"
uuid = "d41bc354-129a-5804-8e4c-c37616107c6c"
version = "7.5.0"

[[NNlib]]
deps = ["BinaryProvider", "Libdl", "LinearAlgebra", "Requires", "Statistics"]
git-tree-sha1 = "d9f196d911f55aeaff11b11f681b135980783824"
uuid = "872c559c-99b0-510c-b3b7-b6c96a88d5cd"
version = "0.6.6"

[[NaNMath]]
git-tree-sha1 = "bfe47e760d60b82b66b61d2d44128b62e3a369fb"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "0.3.5"

[[NearestNeighbors]]
deps = ["Distances", "StaticArrays"]
git-tree-sha1 = "16baacfdc8758bc374882566c9187e785e85c2f0"
uuid = "b8a86587-4115-5ab1-83bc-aa920d37bbce"
version = "0.4.9"

[[NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"

[[Observables]]
git-tree-sha1 = "3469ef96607a6b9a1e89e54e6f23401073ed3126"
uuid = "510215fc-4207-5dde-b226-833fc4488ee2"
version = "0.3.3"

[[OffsetArrays]]
git-tree-sha1 = "a416e2f267e2c8729f25bcaf1ce19d2893faf393"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.3.1"

[[Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7937eda4681660b4d6aeeecc2f7e1c81c8ee4e2f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+0"

[[OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"

[[OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "15003dcb7d8db3c6c857fda14891a539a8f2705a"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "1.1.10+0"

[[OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[Optim]]
deps = ["Compat", "FillArrays", "LineSearches", "LinearAlgebra", "NLSolversBase", "NaNMath", "Parameters", "PositiveFactorizations", "Printf", "SparseArrays", "StatsBase"]
git-tree-sha1 = "c05aa6b694d426df87ff493306c1c5b4b215e148"
uuid = "429524aa-4258-5aef-a3af-852621145aeb"
version = "0.22.0"

[[Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51a08fb14ec28da2ec7a927c4337e4332c2a4720"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.2+0"

[[OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[PCRE_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b2a7af664e098055a7529ad1a900ded962bca488"
uuid = "2f80f16e-611a-54ab-bc61-aa92de5b98fc"
version = "8.44.0+0"

[[PDMats]]
deps = ["Arpack", "LinearAlgebra", "SparseArrays", "SuiteSparse", "Test"]
git-tree-sha1 = "2fc6f50ddd959e462f0a2dbc802ddf2a539c6e35"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.9.12"

[[Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "34c0e9ad262e5f7fc75b10a9952ca7692cfc5fbe"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.3"

[[Parsers]]
deps = ["Dates"]
git-tree-sha1 = "bfd7d8c7fd87f04543810d9cbd3995972236ba1b"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "1.1.2"

[[Pixman_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b4f5d02549a10e20780a24fce72bea96b6329e29"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.40.1+0"

[[Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"

[[PlotThemes]]
deps = ["PlotUtils", "Requires", "Statistics"]
git-tree-sha1 = "87a4ea7f8c350d87d3a8ca9052663b633c0b2722"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "1.0.3"

[[PlotUtils]]
deps = ["Colors", "Dates", "Printf", "Random", "Reexport"]
git-tree-sha1 = "51e742162c97d35f714f9611619db6975e19384b"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "0.6.5"

[[Plots]]
deps = ["Base64", "Contour", "Dates", "FFMPEG", "FixedPointNumbers", "GR", "GeometryTypes", "JSON", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "Printf", "REPL", "Random", "RecipesBase", "Reexport", "Requires", "Showoff", "SparseArrays", "Statistics", "StatsBase", "UUIDs"]
git-tree-sha1 = "f226ff9b8e391f6a10891563c370aae8beb5d792"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "0.29.9"

[[PlutoUI]]
deps = ["Base64", "Dates", "InteractiveUtils", "Logging", "Markdown", "Random", "Suppressor"]
git-tree-sha1 = "45ce174d36d3931cd4e37a47f93e07d1455f038d"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.1"

[[PooledArrays]]
deps = ["DataAPI"]
git-tree-sha1 = "b1333d4eced1826e15adbdf01a4ecaccca9d353c"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "0.5.3"

[[PositiveFactorizations]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "17275485f373e6673f7e7f97051f703ed5b15b20"
uuid = "85a6dd25-e78a-55b7-8502-1745935b8125"
version = "0.2.4"

[[Preferences]]
deps = ["TOML"]
git-tree-sha1 = "00cfd92944ca9c760982747e9a1d0d5d86ab1e5a"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.2.2"

[[Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[Profile]]
deps = ["Printf"]
uuid = "9abbd945-dff8-562f-b5e8-e1ebf5ef1b79"

[[PyCall]]
deps = ["Conda", "Dates", "Libdl", "LinearAlgebra", "MacroTools", "Serialization", "VersionParsing"]
git-tree-sha1 = "4ba3651d33ef76e24fef6a598b63ffd1c5e1cd17"
uuid = "438e738f-606a-5dbb-bf0a-cddfbfd45ab0"
version = "1.92.5"

[[QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "78aadffb3efd2155af139781b8a8df1ef279ea39"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.4.2"

[[REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[ROCAnalysis]]
deps = ["DataFrames", "LinearAlgebra", "Random", "Requires", "SpecialFunctions", "Test"]
git-tree-sha1 = "fed3f005bb6d4a39f848b7f713cf83c820d7bea4"
uuid = "f535d66d-59bb-5153-8d2b-ef0a426c6aff"
version = "0.3.0"

[[Random]]
deps = ["Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[Ratios]]
deps = ["Requires"]
git-tree-sha1 = "01d341f502250e81f6fec0afe662aa861392a3aa"
uuid = "c84ed2f1-dad5-54f0-aa8e-dbefe2724439"
version = "0.4.2"

[[RecipesBase]]
git-tree-sha1 = "b4ed4a7f988ea2340017916f7c9e5d7560b52cae"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "0.8.0"

[[Reexport]]
deps = ["Pkg"]
git-tree-sha1 = "7b1d07f411bc8ddb7977ec7f377b97b158514fe0"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "0.2.0"

[[Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "4036a3bd08ac7e968e27c203d45f5fff15020621"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.1.3"

[[Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "bf3188feca147ce108c76ad82c2792c57abe7b1f"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.7.0"

[[Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "68db32dff12bb6127bac73c209881191bf0efbb7"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.3.0+0"

[[SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"

[[ScikitLearn]]
deps = ["Compat", "Conda", "DataFrames", "Distributed", "IterTools", "LinearAlgebra", "MacroTools", "Parameters", "Printf", "PyCall", "Random", "ScikitLearnBase", "SparseArrays", "StatsBase", "VersionParsing"]
git-tree-sha1 = "b2dbb141575879beb3ad771fb0314a22617586d3"
uuid = "3646fa90-6ef7-5e7e-9f22-8aca16db6324"
version = "0.6.2"

[[ScikitLearnBase]]
deps = ["LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "7877e55c1523a4b336b433da39c8e8c08d2f221f"
uuid = "6e75b9c4-186b-50bd-896f-2d2496a4843e"
version = "0.5.0"

[[SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "f45b34656397a1f6e729901dc9ef679610bd12b5"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.3.8"

[[Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[ShiftedArrays]]
git-tree-sha1 = "22395afdcf37d6709a5a0766cc4a5ca52cb85ea0"
uuid = "1277b4bf-5013-50f5-be3d-901d8477a67a"
version = "1.0.0"

[[Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "ee010d8f103468309b8afac4abb9be2e18ff1182"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "0.3.2"

[[SimpleTraits]]
deps = ["InteractiveUtils", "MacroTools"]
git-tree-sha1 = "5d7e3f4e11935503d3ecaf7186eac40602e7d231"
uuid = "699a6c99-e7fa-54fc-8d76-47d257e15c1d"
version = "0.9.4"

[[Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[SortingAlgorithms]]
deps = ["DataStructures", "Random", "Test"]
git-tree-sha1 = "03f5898c9959f8115e30bc7226ada7d0df554ddd"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "0.3.1"

[[SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[SpecialFunctions]]
deps = ["OpenSpecFun_jll"]
git-tree-sha1 = "d8d8b8a9f4119829410ecd706da4cc8594a1e020"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "0.10.3"

[[StatPlots]]
deps = ["Clustering", "DataStructures", "DataValues", "Distributions", "IterableTables", "KernelDensity", "Observables", "Plots", "RecipesBase", "Reexport", "StatsBase", "TableTraits", "TableTraitsUtils", "Test", "Widgets"]
git-tree-sha1 = "245c50f8a6534bb16ada031e064363f8298b61b9"
uuid = "60ddc479-9b66-56df-82fc-76a74619b69c"
version = "0.9.2"

[[StaticArrays]]
deps = ["LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "da4cf579416c81994afd6322365d00916c79b8ae"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "0.12.5"

[[Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[StatsAPI]]
git-tree-sha1 = "0f2aa8e32d511f758a2ce49208181f7733a0936a"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.1.0"

[[StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics"]
git-tree-sha1 = "19bfcb46245f69ff4013b3df3b977a289852c3a1"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.32.2"

[[StatsFuns]]
deps = ["Rmath", "SpecialFunctions"]
git-tree-sha1 = "ced55fd4bae008a8ea12508314e725df61f0ba45"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "0.9.7"

[[StatsModels]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "Printf", "ShiftedArrays", "SparseArrays", "StatsBase", "StatsFuns", "Tables"]
git-tree-sha1 = "3db41a7e4ae7106a6bcff8aa41833a4567c04655"
uuid = "3eaba693-59b7-5ba5-a881-562e759f1c8d"
version = "0.6.21"

[[SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[Suppressor]]
git-tree-sha1 = "a819d77f31f83e5792a76081eee1ea6342ab8787"
uuid = "fd094767-a336-5f1f-9728-57cf17d0bbfb"
version = "0.2.0"

[[TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"

[[TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[TableTraitsUtils]]
deps = ["DataValues", "IteratorInterfaceExtensions", "Missings", "TableTraits"]
git-tree-sha1 = "78fecfe140d7abb480b53a44f3f85b6aa373c293"
uuid = "382cd787-c1b6-5bf2-a167-d5b971a19bda"
version = "1.0.2"

[[Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "TableTraits", "Test"]
git-tree-sha1 = "fed34d0e71b91734bf0a7e10eb1bb05296ddbcd0"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.6.0"

[[Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"

[[Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[Tracker]]
deps = ["Adapt", "DiffRules", "ForwardDiff", "LinearAlgebra", "MacroTools", "NNlib", "NaNMath", "Printf", "Random", "Requires", "SpecialFunctions", "Statistics"]
git-tree-sha1 = "bf4adf36062afc921f251af4db58f06235504eff"
uuid = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"
version = "0.2.16"

[[TranscodingStreams]]
deps = ["Random", "Test"]
git-tree-sha1 = "216b95ea110b5972db65aa90f88d8d89dcb8851c"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.9.6"

[[URIs]]
git-tree-sha1 = "97bbe755a53fe859669cd907f2d96aee8d2c1355"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.3.0"

[[UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[UnPack]]
git-tree-sha1 = "387c1f73762231e86e0c9c5443ce3b4a0a9a0c2b"
uuid = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"
version = "1.0.2"

[[Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[VersionParsing]]
git-tree-sha1 = "e575cf85535c7c3292b4d89d89cc29e8c3098e47"
uuid = "81def892-9a0e-5fdd-b105-ffc91e053289"
version = "1.2.1"

[[Widgets]]
deps = ["Colors", "Dates", "Observables", "OrderedCollections"]
git-tree-sha1 = "fc0feda91b3fef7fe6948ee09bb628f882b49ca4"
uuid = "cc8bc4a8-27d6-5769-a93b-9d913e69aa62"
version = "0.6.2"

[[WoodburyMatrices]]
deps = ["LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "de67fa59e33ad156a590055375a30b23c40299d3"
uuid = "efce3f68-66dc-5838-9240-27a6d6f5f9b6"
version = "0.5.5"

[[XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "1acf5bdf07aa0907e0a37d3718bb88d4b687b74a"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.9.12+0"

[[XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "Pkg", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "91844873c4085240b95e795f692c4cec4d805f8a"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.34+0"

[[Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "5be649d550f3f4b95308bf0183b82e2582876527"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.6.9+4"

[[Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4e490d5c960c314f33885790ed410ff3a94ce67e"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.9+4"

[[Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fe47bd2247248125c428978740e18a681372dd4"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.3+4"

[[Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "b7c0aa8c376b31e4852b360222848637f481f8c3"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.4+4"

[[Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "19560f30fd49f4d4efbe7002a1037f8c43d43b96"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.10+4"

[[Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6783737e45d3c59a4a4c4091f5f88cdcf0908cbb"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.0+3"

[[Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "daf17f441228e7a3833846cd048892861cff16d6"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.13.0+3"

[[Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "79c31e7844f6ecf779705fbc12146eb190b7d845"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.4.0+3"

[[ZipFile]]
deps = ["BinaryProvider", "Libdl", "Printf"]
git-tree-sha1 = "7fbfbc51c186f0ccdbe091f32d3dff8608973f8e"
uuid = "a5390f91-8eb1-5f08-bee0-b1d1ffed6cea"
version = "0.8.4"

[[Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"

[[libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "5982a94fcba20f02f42ace44b9894ee2b140fe47"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.1+0"

[[libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "daacc84a041563f965be61859a36e17c4e4fcd55"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.2+0"

[[libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "94d180a6d2b5e55e447e2d27a29ed04fe79eb30c"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.38+0"

[[libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "c45f4e40e7aafe9d086379e5578947ec8b95a8fb"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+0"

[[nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"

[[p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"

[[x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fea590b89e6ec504593146bf8b988b2c00922b2"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2021.5.5+0"

[[x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ee567a171cce03570d77ad3a43e90218e38937a9"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.5.0+0"
"""

# ╔═╡ Cell order:
# ╟─3240aaf8-3be2-4d08-a25b-594bc8f8ab98
# ╠═f6926a24-41bb-488f-962d-d65dd025f756
# ╟─eefce769-e42a-4a5e-8c19-a14483cd3889
# ╟─87036def-7c5f-4b87-958f-957fc4e43405
# ╟─1469bac3-ce83-4128-8b33-07402c91cee5
# ╟─94a60d30-3d8d-465d-a18b-674a8bb9e83d
# ╟─1230245b-a6b4-4256-9ee0-6256e919d3bf
# ╟─4af40015-6fc9-406d-a922-220ab7348744
# ╟─f90ee966-b34d-41d7-aa4e-bfd228bf2326
# ╟─50260c0f-2547-44d5-b0d6-99ebe629a079
# ╟─3333fe7c-363f-43c4-bddc-d6cbda66bcd6
# ╟─31540eb1-c35c-4b8f-928c-8fc073b4c307
# ╟─931a48af-7a07-4bf0-a782-e1efbfea909d
# ╟─58a4676f-0b74-4ddc-84f6-7485b6759314
# ╟─e2a1a04b-1ac5-4728-9470-6d4525213f0f
# ╟─8f196927-2f16-43ce-b015-aef96240c5a9
# ╟─a3a0cea0-4b04-11ec-2ecc-b993f598f926
# ╟─a81e4ca1-a160-4637-8349-730c70035533
# ╠═63527852-1244-42b5-8dc5-c3194deba174
# ╠═95f2146b-6532-4a16-8bfe-b1b265fe6bd6
# ╟─bb66558c-7587-496b-b322-38d3efd7be4c
# ╠═35245072-0b15-4c8b-bf6a-ec48edab7776
# ╟─b15f8f75-46bc-4730-b8cd-7eeb12d01046
# ╟─c1b676ab-7514-463b-acac-85afb8d4ecc9
# ╠═375a9719-0f4b-4389-ae7f-39b6d82a81bf
# ╟─60d02ba5-9bee-4db7-8e5e-40fc39586272
# ╠═cea7c852-4c86-4bf4-a41a-e82bca37ffae
# ╟─b3be9d3c-9b08-41e6-bae0-d19ce27041da
# ╠═0fc7329d-c245-48bc-b3d5-e366209bfd26
# ╟─59d2f136-b789-4750-9f88-a481637e58ad
# ╠═9f9381ae-5953-4112-8a05-21d76cabfafe
# ╟─20a5bedb-0726-4a9b-b837-be57defb6a3a
# ╠═c9d0da04-df05-4869-ba0e-d34a11a90dfb
# ╟─232d6a0c-fd94-4596-a41a-213c10fd5076
# ╠═35f9a901-c7f4-4561-8e24-f62598b167de
# ╟─b7bf2aa8-4d6b-4f58-bf6b-062e049b8296
# ╠═614aa0dc-66a2-4a4f-9ed2-d902aad255a1
# ╟─0b410342-8190-4dc7-9dac-24ee09d8953f
# ╟─512fce0c-accb-44b9-88e6-975a630065ea
# ╠═3553dff9-2efd-41b1-8306-4498bef05220
# ╟─fbe653ea-5194-465a-aaeb-e77a2ad3022e
# ╟─ef857760-9ab5-42c8-ba73-1453545fa28e
# ╠═f8dbd0ca-7588-4dfa-924f-4551090785e3
# ╠═cb4a85f5-f13e-4016-bb8f-a6e58e5f942f
# ╠═d775da74-1e40-4599-b3ce-1a17ebf2a34a
# ╠═d5ba919c-24cc-41aa-97e9-01a5fbb52f5c
# ╟─08220005-2405-4cc7-a450-7a8e67c78bb9
# ╟─2e823df4-76f6-4621-a56a-a29a4a4ddbff
# ╠═dee47806-c344-47e4-8d83-6249150a36e4
# ╟─6495eb72-d48f-4205-9f80-f8adc51ed1e0
# ╟─6cbb913b-b724-4bf7-abaa-515bf21f12cd
# ╠═100c654a-f24d-4d27-8357-f0df29dcf059
# ╟─8fe39705-4ce6-4c8b-832f-6410c6040c36
# ╠═22dc8567-22c4-40ef-a364-5c61b57a9f38
# ╟─194582b0-3bbd-4977-9e49-d8bb1bf4615c
# ╟─08d201ac-8208-4138-b5bd-f724f43dbc52
# ╠═c3ced2f2-a065-4f46-acb9-ce44f2c03ec1
# ╠═23072f60-5cf6-441d-bc90-973e9d6ec0c4
# ╠═90c45f07-1123-4515-ade8-a20e1c0542f3
# ╟─311b7852-1640-4277-acb0-8caa1e0e0762
# ╠═2ab7d7c5-ca5d-4b8f-9e2c-72a2ee650894
# ╠═a2ca9c5a-cba9-489b-9904-f72738bd1313
# ╠═0e0f83b2-938f-4e84-aa0c-470d16c6b390
# ╟─2eb602c3-8f17-43fc-a300-e75fe4e08452
# ╠═6cb34c59-03fe-4d8c-8adf-e72a980b5779
# ╟─a1b85414-de6e-46fc-bab1-19cacb51e3d4
# ╠═aa503a18-0ee1-4791-8256-1c9b0ce2eada
# ╟─8e3e3199-c8fc-4b76-b110-e103a28802b3
# ╠═3a6f467d-54f9-4e0e-be13-5688bda941eb
# ╟─175e7045-71b9-44f5-86ca-653d76bdf047
# ╟─2665a487-0e19-4f05-84cb-abf0e888a30b
# ╟─57de6a90-eafe-4fee-bab9-afb7c090e27f
# ╟─a661f1e7-a1c1-4deb-9e1b-861873e88d82
# ╟─51c237c4-5ded-44f4-b9d4-f90263bd0253
# ╠═f48573fc-95b7-40a6-9a1e-78315bbe37d3
# ╠═a07482c6-fbf5-4d87-8ada-5b574fb02246
# ╠═a7f520b8-f5bd-4996-a5c2-7b2578f04c43
# ╠═5dca076f-3741-43d6-8515-f41e0248b27b
# ╠═443d7404-ae6b-4e38-9a38-f47a723986f8
# ╠═330fcd4f-82ee-459b-b4e2-12ae8367e3e9
# ╠═8c1185de-4a4c-45db-a5b3-89204796af36
# ╠═5e483882-a375-4a3f-a98f-ea2aad376092
# ╠═d1eab486-39af-4ef1-998a-f9b07d6bf40c
# ╟─a4020f2a-604e-46e3-aa2b-a5a43373dc6c
# ╟─6b715f0b-9258-4c43-a26c-0808ce65154b
# ╠═594147ec-3407-4cc2-ad41-fd872be13229
# ╟─def9e444-131e-444d-9867-ef01396b8c37
# ╟─69fcb1b5-3580-4cec-a30f-b1a5b904a912
# ╟─b5a3600d-e87a-48aa-93a5-cbbc69b687d2
# ╠═09586d02-bc6a-40cf-a6f3-f0ef94f9febd
# ╠═a5641a7c-0555-45f1-9457-7dbe27a5af31
# ╠═a10d4b6e-43a6-4271-a20c-a27ca667011e
# ╠═e4bc3079-697a-4154-a56b-2163d0839852
# ╠═0589811c-c94b-4e7a-9d63-b567dfc25705
# ╠═05aab783-a9ef-49b7-848a-858d6bc9c875
# ╠═3526950c-7f02-47e3-8ebb-251010f605ee
# ╟─662c558e-d93a-4010-827f-8b4c23ea0c52
# ╠═6c91c3b6-53cf-41f8-ad57-ce1ad54eb6b1
# ╟─5aed60ca-6db6-4bc3-91bf-685833409d20
# ╠═e9947565-f679-4bbd-9378-2b4a2b5727b0
# ╠═96d3577a-fb89-44c9-84f1-150d7d73bcae
# ╟─45efafaa-2bad-4394-a383-a6b5a79f3f26
# ╟─c50a699b-790e-43ca-9413-64d2ce7e9295
# ╟─9065469d-db09-47f8-be67-9d9915bca6bf
# ╠═26e63026-8ea9-443a-ab33-34d19889eaf3
# ╟─1afb97a2-8d8e-4c7c-8b9e-7dacc11249c6
# ╠═879c5498-b877-44ee-9352-d0982391832e
# ╟─d4d47777-22af-4c63-80cd-ecabe6279509
# ╟─1f66bc09-e6e0-4bab-9eb9-74f97a3d5a5b
# ╠═cbb45603-f310-4556-aa61-bb56a95c2140
# ╠═d51e8039-defb-4ff1-b0bc-3ca8785c8b76
# ╠═d7b559e7-fe91-4eed-9882-7f9920cd9ee5
# ╠═dd36605d-91e8-4e0e-8b00-b6aabbcc3109
# ╟─8a263263-7ee7-4033-bba7-4c74247d0eb2
# ╟─268c190c-3bca-43ff-93b3-4cd7ddefa731
# ╟─39a8003c-2b53-45b6-a101-d634156304d1
# ╟─be8a5ec7-fcab-4f89-a4ce-a396416c0f67
# ╠═6f128cc5-66df-4f58-803c-0dba77342083
# ╠═b5476740-7825-4dfc-b309-d99a754967da
# ╠═9728951b-61b5-4344-b989-6de2209da8c6
# ╠═4c68c76e-62a7-4ecb-a737-74c814fb041a
# ╠═82b774cd-7254-49b7-9293-f0c3dad7e431
# ╠═77e121f0-a72b-4e59-b2a0-9a3dcfad2901
# ╠═eb312d12-c2ba-416a-9735-ebcf5e190918
# ╟─bc97c003-f8db-4fa9-b9c4-7203b80b3bef
# ╠═044c883d-a9e1-49de-a2c2-b775728632ac
# ╠═579f31dd-a765-498d-b0ae-5093672ba65f
# ╠═87759a8f-0b4b-41ce-88a7-27dacb30d9e5
# ╠═2616a5dc-4a16-4581-a757-50acf50ff3ea
# ╟─ed87de0c-7ac6-4393-a902-cc4af64f2c78
# ╠═98dc6349-0cb6-4bd4-98e1-9f5253c75d65
# ╠═61bfbbf8-dc65-4953-a855-0766b1ee0861
# ╠═60bb22e2-1b8a-4aee-8c46-ca83db675cb5
# ╠═a014cb4c-edeb-4124-8feb-81339742e195
# ╟─b49a0dff-0901-4eaa-bbf4-750ecfcfd096
# ╠═291a07b8-6903-4364-8057-8780a87818d4
# ╠═366ce026-ccd8-4b6e-bcc6-438486bd9e60
# ╠═f4da99ea-bacb-487e-9843-474e88a6305c
# ╠═69d8267e-8f28-4d30-a9e7-89075ed27a6a
# ╟─e5f90438-eb52-455a-a968-8ea76f0589fd
# ╠═3b19829c-434d-4fd7-8195-ec847a055bbd
# ╠═1f375328-b435-4b5b-9fd1-2647bfb09a36
# ╟─93bc4c0a-4921-48ad-8c63-bd899b49a594
# ╟─a0878f0c-b4a6-4c5e-b9ea-4e9b2f070c49
# ╠═5eee4ca3-61f7-46fc-a85e-697045a98a41
# ╠═502050f5-c3a8-498e-89fa-231f0f11d7e1
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002

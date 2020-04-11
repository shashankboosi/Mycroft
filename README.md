
# Mycroft

An AI approach to detect Semantic Data Types using NLP and Deep Learning.

## Implementation

Mycroft is divided into 3 different stages because of the different types of conversions we implement in the datasets by formatting them completely.

### Stage 1 
Converting Web Data Commons 2015 to the format that Mycroft takes in as input

Example structure of **Web Data Commons 2015**:

```json
{
  "relation":[ 
	["#","1","2","3","4","5","6","7","8","9","10"], 
	["Club","Barcelona","Real Madrid","Bayern München","Paris Saint Germain","Atlético Madrid","Juventus","Manchester City","Arsenal","FC Porto","Manchester United"], 
	["Country","ESP","ESP","GER","FRE","ESP","ITA","ENG","ENG","POR","ENG"],
	["Points","2037","2008","1973","1914","1880","1863","1862","1853","1850","1804"]
   ]
  "pageTitle": "FootballDatabase - Club Rankings and Statistics",
  "title": "",
  "url": "http://footballdatabase.com/index.php?page\u003dplayer\u0026Id\u003d660",
  "hasHeader": true,
  "headerPosition": "FIRST_ROW",
  "tableType": "RELATION",
  "tableNum": 0,
  "s3Link": "common-crawl/crawl-data/CC-MAIN-2015-32/segments/1438042981460.12/warc/CC-MAIN-20150728002301-00000-ip-10-236-191-2.ec2.internal.warc.gz",
  "recordEndOffset": 99246001,
  "recordOffset": 99230046,
  "tableOrientation": "HORIZONTAL",
  "TableContextTimeStampBeforeTable": "{10283\u003dOn Wednesday, December 6, 2006 Islanders General Manager Garth Snow attended the Fifth Annual John Theissen Holiday Fundraiser.}",
  "TableContextTimeStampAfterTable": "{37811\u003dIn 2005, Slovakian champion FC Artmedia upset 39-time Scottish league champion Celtic 5-0 in their European Champions League second-round qualifying match.}",  
  "lastModified": "Sat, 19 Jun 2010 15:14:57 GMT",
  "textBeforeTable": "Chelsea Ronnie MacDonald Bayern München Peter P. Juventus Mitsurinho Real Madrid Jan S0L0 Barcelona Globovision Football",
  "textAfterTable": "Full World Ranking Match Centre Argentina Primera 2015 26 July 2015 Vélez Sarsfield 0 - 2 Olimpo Brazil Serie A 2015 26 July 2015 Vasco da Gama 1 - 4 Palmeiras Mexico Liga",
  "hasKeyColumn": true,
  "keyColumnIndex": 1,
  "headerRowIndex": 0  
}
```

In this stage we convert the above structure into Mycroft format shown below

```
["label", "data"]
["Club", ["Barcelona","Real Madrid","Bayern München","Paris Saint Germain","Atlético Madrid","Juventus","Manchester City","Arsenal","FC Porto","Manchester United"]] 
["Country", ["ESP","ESP","GER","FRE","ESP","ITA","ENG","ENG","POR","ENG"]]
["Points", ["2037","2008","1973","1914","1880","1863","1862","1853","1850","1804"]]
```

**Important Considerations while conversion:**

1. Remove the webTable which doesnot contain `hasHeader`
2. Only consider webTables with `tableType` being `RELATION`
3. Remove a few special characters from the label
4. Remove labels which are empty

### Stage 2

**Features to talk about **

1. Use `pageTitle` to increase more relevancy in the dataset
2. Add the feature of Similarity matching whilst Data Collection


### Stage 3 

**Features to talk about **

1. Use TF-IDF
2. Change in the dimensions used in the feature extractors
3. Use of a multi-input NN for better associations between the features.

# Todos

1. Deeply understand Sherlock by going through the paper thoroughly and understand the feature classifications.
2. Try exploiting sherlock and try adding more features to our research.

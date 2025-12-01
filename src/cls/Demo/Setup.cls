Class Demo.Setup Extends %RegisteredObject
{

ClassMethod hf()
{
	#; set config = {"modelName":"sentence-transformers/embeddinggemma-300m-medical",
    #;         "hfCachePath":"/iris.d/hfcache",
    #;         "maxTokens": 256,
    #;         "checkTokenCount": true}
	#; set config = {"modelName":"sentence-transformers/embeddinggemma-300m-medical",
	set config = {"modelName":"sentence-transformers/all-MiniLM-L6-v2",
            "hfCachePath":"/irisdata/hfcache",
			#; "hfToken":(..#TOKEN),
            #; "checkTokenCount": true}.%ToJSON()
			"checkTokenCount": false}.%ToJSON()
	
	&SQL(INSERT INTO %Embedding.Config 
	-- (Name, Configuration, EmbeddingClass, Description)
	SET Name = 'sentence-transformers-config',
	Configuration = :config,
	Description = 'a small SentenceTransformers embedding model',
	EmbeddingClass='%Embedding.SentenceTransformers')
	break
}

ClassMethod Drop()
{
	&SQL(DROP TABLE Embedding.Example)
}

ClassMethod Table()
{
	
	&SQL(CREATE TABLE Embedding.Example (
		Description VARCHAR(200),
		Code VARCHAR(10),
		System VARCHAR(240),
		DescriptionEmbedding EMBEDDING('sentence-transformers-config','Description')))
	b:SQLCODE<0
	&SQL(CREATE INDEX HNSWDescriptionIndex ON TABLE Embedding.Example 
		(DescriptionEmbedding) AS HNSW(Distance='Cosine'))
	&SQL(CREATE INDEX CodeIndex ON Table Embedding.Example
		(Code))
	&SQL(CREATE INDEX DescriptionIndex ON Table Embedding.Example
		(Description))

	break
}

ClassMethod Demo()
{
		set component =  [
		{
			"code": {
				"coding": [
					{
						"system": "http://loinc.org",
						"code": "8480-6",
						"display": "Systolic blood pressure"
					},
					{
						"system": "http://snomed.info/sct",
						"code": "271649006"
					}
				],
				"text": "Systolic blood pressure"
			},
			"valueQuantity": {
				"value": 109,
				"unit": "mmHg",
				"system": "http://unitsofmeasure.org",
				"code": "mm[Hg]"
			}
		},
		{
			"code": {
				"coding": [
					{
						"system": "http://loinc.org",
						"code": "8462-4",
						"display": "Diastolic blood pressure"
					},
					{
						"system": "http://snomed.info/sct",
						"code": "271650006"
					}
				],
				"text": "Diastolic blood pressure"
			},
			"valueQuantity": {
				"value": 44,
				"unit": "mmHg",
				"system": "http://unitsofmeasure.org",
				"code": "mm[Hg]"
			}
		}
	]
	set compIt = component.%GetIterator()
	while compIt.%GetNext(.key,.item) {
		set text=item.code.text
		write !,text
		set coding = item.code.coding
		set codeIt = coding.%GetIterator()
		while codeIt.%GetNext(,.codeItem) {
			set code = codeItem.code
			set system = codeItem.system
			write !,$$$FormatText("Inserting code for %1 : %2, Code system : %3",code,text,system)
			&SQL(INSERT INTO Embedding.Example  (Code,Description,System) VALUES (:code,:text,:system))
			b:SQLCODE<0
		}
	}
}

ClassMethod Test2()
{
#; 	&SQL(SELECT TOP 2 Description FROM Embedding.Example
#; 	  ORDER BY VECTOR_COSINE(Description, 
#; 	                              EMBEDDING('Blood Pressure','sentence-transformers-config')) DESC
#; )
	set query($i(query))="SELECT TOP 2 Description FROM Embedding.Example"
	set query($i(query))="ORDER BY VECTOR_COSINE(DescriptionEmbedding, "
	set query($i(query))="      EMBEDDING('Blood Pressure','sentence-transformers-config')) DESC"
	set result = ##class(%SQL.Statement).%ExecDirect(,.query)
	break
	do result.%Display()
}

}

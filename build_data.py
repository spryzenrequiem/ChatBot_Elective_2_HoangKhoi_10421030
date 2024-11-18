from src import ingest_pipeline, index_builder

if __name__ == "__main__":
   nodes = ingest_pipeline.ingest_documents()
   vector_index = index_builder.build_indexes(nodes)
   # query_engine = vector_index.as_query_engine()
   
   # # Run a sample query
   # query_result = query_engine.query("List common diseases of peanut?")
   # print("Query Result:", query_result)
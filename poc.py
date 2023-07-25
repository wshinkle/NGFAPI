from flask import Flask, request
from haystack.document_stores import InMemoryDocumentStore
from haystack.utils import fetch_archive_from_http
from haystack import Pipeline
from haystack.nodes import TextConverter, PreProcessor, BM25Retriever, FARMReader, PromptNode, PromptTemplate, AnswerParser
import os
from haystack.utils import print_answers


def setup_retriever()-> BM25Retriever:
    document_store = InMemoryDocumentStore(use_bm25=True)

    doc_dir = "data/files"

    print('Fetching data from HTTP...\n')
    fetch_archive_from_http(
        url="https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-qa/datasets/documents/wiki_gameofthrones_txt3.zip", 
        output_dir=doc_dir
    )
    print('Data fetched!\n')

    indexing_pipeline = Pipeline()
    text_converter = TextConverter()
    preprocessor = PreProcessor(
        clean_whitespace=True,
        clean_header_footer=True,
        clean_empty_lines=True,
        split_by="word",
        split_length=200,
        split_overlap=20,
        split_respect_sentence_boundary=True,
    )

    indexing_pipeline.add_node(component=text_converter, name="TextConverter", inputs=["File"])
    indexing_pipeline.add_node(component=preprocessor, name="PreProcessor", inputs=["TextConverter"])
    indexing_pipeline.add_node(component=document_store, name="DocumentStore", inputs=["PreProcessor"])

    files_to_index = [doc_dir + "/" + f for f in os.listdir(doc_dir)]
    indexing_pipeline.run_batch(file_paths=files_to_index)

    retriever = BM25Retriever(document_store=document_store)
    return retriever



def search_setup() -> Pipeline:
    
    retriever=setup_retriever()
    reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=True)

    querying_pipeline = Pipeline()
    querying_pipeline.add_node(component=retriever, name="Retriever", inputs=["Query"])
    querying_pipeline.add_node(component=reader, name="Reader", inputs=["Retriever"])
    return querying_pipeline

def generate_setup() -> Pipeline:
    retriever=setup_retriever()

    lfqa_prompt = PromptTemplate(
    prompt="""Synthesize a comprehensive answer from the following text for the given question.
                             Provide a clear and concise response that summarizes the key points and information presented in the text.
                             Your answer should be in your own words and be no longer than 50 words.
                             \n\n Related text: {join(documents)} \n\n Question: {query} \n\n Answer:""",
    output_parser=AnswerParser(),
)
    
    prompt_node = PromptNode(model_name_or_path="meta-llama/Llama-2-70b-hf", default_prompt_template=lfqa_prompt)


    pipe = Pipeline()
    pipe.add_node(component=retriever, name="retriever", inputs=["Query"])
    pipe.add_node(component=prompt_node, name="prompt_node", inputs=["retriever"])
    return pipe


def create_app():

    app = Flask(__name__)
    search_pipeline=search_setup()
    generate_pipeline=generate_setup()
    @app.route('/search', methods=['POST'])
    def search():
        data = request.get_json()
        print(data)
        question = data["question"]
        prediction = search_pipeline.run(
        query=question,
        params={
            "Retriever": {"top_k": 10},
            "Reader": {"top_k": 5}
        }
        )

        print_answers(
            prediction,
            details="minimum" ## Choose from `minimum`, `medium` and `all`
        )
        return prediction
    
    @app.route('/generate', methods=['POST'])
    def generate():
        data=request.get_json()
        output = generate_pipeline.run(query=data["question"])
        return output
    return app




if __name__ == '__main__':
    app=create_app()
    app.run()
    
    
    
    
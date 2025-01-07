import os
from langchain import hub
from langchain_community.llms import Ollama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Initialize the LLM
llm = Ollama(model="qwen2")

# Load the RAG prompt
rag_prompt = hub.pull("rlm/rag-prompt")

# Define the RAG pipeline
qa_chain = (
    {"context": RunnablePassthrough(), "question": RunnablePassthrough()}  # Direct passthrough
    | rag_prompt
    | llm
    | StrOutputParser()
)

# Validation prompt pipeline
validation_prompt = """
Você deve avaliar se o seguinte texto está no formato solicitado. Responda apenas "Sim" ou "Não". 
O texto deve ter:

1. A seção "Tarefa 1:" seguida por tópicos limitados a 10 itens, começando com "-".
2. A seção "Tarefa 2:" com uma resposta de uma palavra: "positivo" ou "negativo".

Texto para avaliação:
{response}
"""

def validate_response_with_prompt(response):
    """
    Uses an LLM validation prompt to check if the response follows the required format.
    Returns True if valid, otherwise False.
    """
    validation_inputs = {
        "context": response,
        "question": validation_prompt.format(response=response)
    }
    validation_result = qa_chain.invoke(validation_inputs)
    return validation_result.strip().lower() == "sim"

def process_file(file_path):
    with open(file_path, 'r') as file:
        text = file.read()
    return text

def main():
    folder_path = '/home/arthurblb/mestrado/Divided_text/qna/'
    output_path = '/home/arthurblb/mestrado/Divided_text/output/qwen/unsupervised/'  # Folder to save results
    os.makedirs(output_path, exist_ok=True)  # Ensure the output folder exists
    
    files = os.listdir(folder_path)
    cont = 0
    num_files = len(files)
    for filename in files:
        cont += 1
        if filename.endswith('.txt'):
            file_path = os.path.join(folder_path, filename)
            # Process file to extract content
            text = process_file(file_path)
            
            # Define the question for the LLM
            question = """
                Queria pedir para você realizar quatro tarefas sequencialmente:

                Tarefa 1) Apresentar os tópicos mais importantes desse texto. Limite máximo de 10 tópicos. Os tópicos devem ser de no máximo 5 palavras e devem ser assuntos, não o detalhamento do que foi falado. Liste os tópicos de em tópicos com '-'.
                Tarefa 2) Avaliar pelas perguntas do público se o público teve uma percepção positiva do apresentado. A resposta deve ter 1 palavra: positivo ou negativo.

                Para todas as respostas deve-se começar pelo texto: 'Tarefa x:' e usar tópicos usando '-'
                Não deve-se usar *
                """
            inputs = {"context": text, "question": question}
            
            for attempt in range(3):  # Retry up to 3 times if the response is invalid
                result = qa_chain.invoke(inputs)
                if validate_response_with_prompt(result):
                    break  # Stop retrying if the response is valid
                print(f"Invalid response format for {filename}. Retrying... ({attempt + 1}/3)")
            else:
                print(f"Failed to get a valid response for {filename} after 3 attempts.")
                result = "Erro: Formato inválido após 3 tentativas."
            
            # Save the result to a file
            output_file = os.path.join(output_path, f"{filename}_output.txt")
            with open(output_file, 'w') as f:
                f.write(f"{result}")
            
            print(f"Result for {filename} saved to {output_file}")
            print(f"File {cont} of {num_files} processed")

if __name__ == '__main__':
    main()

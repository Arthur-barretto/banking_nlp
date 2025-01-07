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

def process_file(file_path):
    with open(file_path, 'r') as file:
        text = file.read()
    return text

def main():
    folder_path = '/home/arthurblb/mestrado/Divided_text/qna/'
    output_path = '/home/arthurblb/mestrado/Divided_text/output/qwen/'  # Folder to save results
    os.makedirs(output_path, exist_ok=True)  # Ensure the output folder exists
    
    files = os.listdir(folder_path)
    #files = files[:5]
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
                Para cada um dos temas que vou listar abaixo, quero que você indique se o tema foi mencionado no arquivo e se foi positivo ou negativo.

                1) Segmento agro
                2) Segmento imobiliário
                3) Segmento de veículos
                4) Middle Market
                5) Corporate
                6) Gestão de risco
                7) Inadimplência
                8) Despesas administrativas
                9) Custo de captação
                10) Provisões de Risco
                11) Taxa Selic
                12) SME
                13) Equilíbrio fiscal
                14) Medidas do governo
                15) Câmbio
                16) Cartão de crédito

                Gostaria que a resposta viesse da seguinte maneira e sem nenhum texto adicional:

                {

                    'Segmento agro': ['Sim','Sim'],
                    'Segmento imobiliário': ['Sim','Não'],
                    'Segmento de veículos': ['Não','Não'],
                    'Middle Market': ['Sim','Sim'],
                    'Corporate': ['Sim','Não'],
                    'Gestão de risco': ['Não','Não'],
                    'Inadimplência': ['Sim','Não'],
                    'Despesas administrativas': ['Sim','Não'],
                    'Custo de captação': ['Sim','Não'],
                    'Provisões de Risco': ['Sim','Não'],
                    'Taxa Selic': ['Sim','Não'],
                    'SME': ['Sim','Não'],
                    'Equilíbrio fiscal': ['Sim','Não'],
                    'Medidas do governo': ['Sim','Não'],
                    'Câmbio': ['Sim','Não'],
                    'Cartão de crédito': ['Sim','Não']

                }
                """
            #question = f"What insights can you extract from the file ?"
            
            # Use the RAG pipeline with direct text
            inputs = {"context": text, "question": question}
            result = qa_chain.invoke(inputs)
            
            # Save the result to a file
            output_file = os.path.join(output_path, f"{filename}_output.txt")
            with open(output_file, 'w') as f:
                #f.write(f"Question: {question}\n\n")
                #f.write(f"Result:\n{result}\n")
                f.write(f"{result}")
            
            print(f"Result for {filename} saved to {output_file}")
            print(f"File {cont} of {num_files} processed")

if __name__ == '__main__':
    main()

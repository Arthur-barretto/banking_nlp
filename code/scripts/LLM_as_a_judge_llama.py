import os
from langchain_community.llms import Ollama
import json

# Initialize the local Llama model
llm_evaluator = Ollama(model="llama3.1")

# Original task prompt given to the models
original_prompt = """
      Queria pedir para você realizar duas tarefas sequencialmente:

      Tarefa 1) Apresentar os tópicos mais importantes desse texto. Limite máximo de 10 tópicos. Os tópicos devem ser de no máximo 5 palavras e devem ser assuntos, não o detalhamento do que foi falado. Liste os tópicos de em tópicos com '-'.
      Tarefa 2) Avaliar pelas perguntas do público se o público teve uma percepção positiva do apresentado. A resposta deve ter 1 palavra: positivo ou negativo.

      Para todas as respostas deve-se começar pelo texto: 'Tarefa x:' e usar tópicos usando '-'
      Não deve-se usar *
"""

# Define the evaluation prompt with the original prompt included
evaluation_prompt = """
Você deve avaliar a resposta de um modelo para a tarefa 1 demandada e fornecer uma pontuação de 0 a 10, junto com uma explicação para a pontuação. 
Considere:

1. A aderência ao pedido no prompt original.
2. Os temas serem os mais relevantes.
3. A aderência ao formato solicitado. Seja em escrita e quantidade de tópicos.

Tarefa original:
{original_prompt}

Texto original:
{context}

Resposta do Modelo:
{response}

Qual é a pontuação (0 a 10) e a explicação? Forneça no formato:
{{"score": <pontuação>, "explanation": "<explicação>"}}
"""

def read_file(file_path):
    """Read and return the contents of a file."""
    with open(file_path, 'r') as file:
        return file.read()

def evaluate_response(original_prompt, context, response):
    """Evaluate a single response using the local Llama model and ensure the output is correctly formatted."""
    for attempt in range(3):  # Retry up to 3 times
        evaluation_question = evaluation_prompt.format(
            original_prompt=original_prompt,
            context=context,
            response=response
        )
        evaluation_result = llm_evaluator.invoke(evaluation_question).strip()

        try:
            # Parse the result as JSON to validate the format
            result = json.loads(evaluation_result)
            if "score" in result and "explanation" in result:
                return result  # Return the parsed dictionary if valid
        except json.JSONDecodeError:
            print(f"Attempt {attempt + 1}: Invalid format, retrying...")

    raise ValueError("Failed to get a valid response after 3 attempts.")

def main():
    original_folder = '/home/arthurblb/mestrado/Divided_text/qna/'  
    qwen_folder = '/home/arthurblb/mestrado/Divided_text/output/qwen/unsupervised/'
    chatgpt_folder = '/home/arthurblb/mestrado/Divided_text/output/chatgpt/unsupervised/'
    output_folder = '/home/arthurblb/mestrado/Divided_text/output/results/judge_llama/contextualized/'  
    error_file = os.path.join(output_folder, "error.txt")  # File to save errors

    os.makedirs(output_folder, exist_ok=True)  # Ensure the output folder exists

    files = os.listdir(original_folder)

    num_files = len(files)
    cont = 0
    for filename in files:
        if filename.endswith('.txt'):
            name_file = filename.split(".")[0]
            print(f"Processing file {name_file}")
            cont += 1
            print(f"File {cont} of {num_files} processed")

            try:
                # Read the original text and model responses
                original_text = read_file(os.path.join(original_folder, filename))
                qwen_response = read_file(os.path.join(qwen_folder, name_file + ".txt_output.txt"))
                chatgpt_response = read_file(os.path.join(chatgpt_folder, name_file + "txt_output.txt"))

                # Evaluate each response
                qwen_result = evaluate_response(original_prompt, original_text, qwen_response)
                chatgpt_result = evaluate_response(original_prompt, original_text, chatgpt_response)

                # Prepare the result dictionary
                evaluation_data = {
                    "chatgpt": chatgpt_result,
                    "qwen": qwen_result
                }

                # Save the evaluation result as JSON
                output_file = os.path.join(output_folder, f"{name_file}_evaluation.json")
                with open(output_file, 'w') as f:
                    json.dump(evaluation_data, f, indent=4, ensure_ascii=False)

                print(f"Evaluation for {filename} completed. Results saved.")

            except Exception as e:
                # Log the filename to the error file if evaluation fails
                with open(error_file, 'a') as ef:
                    ef.write(f"{filename}\n")
                print(f"Evaluation for {filename} failed. Error logged.")

if __name__ == '__main__':
    main()

import os
from langchain_community.llms import Ollama
import json

# Initialize the local Llama model
llm_evaluator = Ollama(model="llama3.1")

evaluation_prompt = """
Você deve avaliar a resposta de um modelo para uma tarefa específica e fornecer uma pontuação de 0 a 10, junto com uma explicação para a pontuação. 
Considere:

1. A clareza da resposta.
2. A aderência ao formato solicitado.
3. A profundidade da análise.

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

def evaluate_response(context, response):
    """Evaluate a single response using the local Llama model and ensure the output is correctly formatted."""
    for attempt in range(3):  # Retry up to 3 times
        evaluation_question = evaluation_prompt.format(
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
    output_folder = '/home/arthurblb/mestrado/Divided_text/output/results/judge_llama/'
    error_file = os.path.join(output_folder, "error.txt")  # File to save errors

    os.makedirs(output_folder, exist_ok=True)  # Ensure the output folder exists

    files = os.listdir(original_folder)

    for filename in files:
        if filename.endswith('.txt'):
            name_file = filename.split(".")[0]
            print(f"Processing file {name_file}")

            try:
                # Read the original text and model responses
                original_text = read_file(os.path.join(original_folder, filename))
                qwen_response = read_file(os.path.join(qwen_folder, name_file + ".txt_output.txt"))
                chatgpt_response = read_file(os.path.join(chatgpt_folder, name_file + "txt_output.txt"))

                # Evaluate each response
                qwen_result = evaluate_response(original_text, qwen_response)
                chatgpt_result = evaluate_response(original_text, chatgpt_response)

                # Prepare the result dictionary
                evaluation_data = {
                    "chatgpt": chatgpt_result,
                    "qwen": qwen_result
                }

                # Save the evaluation result as JSON
                output_file = os.path.join(output_folder, f"{filename}_evaluation.json")
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

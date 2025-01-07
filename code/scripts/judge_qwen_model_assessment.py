import os
import json
from langchain_community.llms import Ollama

# Initialize the local Llama model
llm_evaluator = Ollama(model="qwen2")

# Prompt for model-level assessment
assessment_prompt = """
Você deve fornecer uma avaliação geral para o modelo '{model_name}' com base nos dados a seguir:

- Pontuação média: {average_score:.2f}
- Explicações agregadas:
{explanations}

Avalie os seguintes aspectos:
1. Os pontos mais fortes do modelo.
2. Os pontos mais fracos do modelo.
3. Recomendações para melhoria.
4. Um resumo geral da performance.

Forneça a resposta no seguinte formato:
{{
    "strengths": ["<forte1>", "<forte2>", ...],
    "weaknesses": ["<fraqueza1>", "<fraqueza2>", ...],
    "recommendations": ["<recomendação1>", "<recomendação2>", ...],
    "summary": "<resumo>"
}}
"""

def aggregate_evaluations(input_folder):
    """Aggregate evaluations from all per-document JSON files."""
    aggregated_scores = {"llama": [], "chatgpt": []}
    aggregated_explanations = {"llama": [], "chatgpt": []}

    # Collect evaluation files
    evaluation_files = [f for f in os.listdir(input_folder) if f.endswith('.json')]

    for eval_file in evaluation_files:
        file_path = os.path.join(input_folder, eval_file)
        with open(file_path, 'r') as f:
            evaluations = json.load(f)
            for model in ["llama", "chatgpt"]:
                if model in evaluations:
                    aggregated_scores[model].append(evaluations[model]["score"])
                    aggregated_explanations[model].append(evaluations[model]["explanation"])

    return aggregated_scores, aggregated_explanations


def generate_model_assessment(model_name, scores, explanations):
    """Generate a general assessment for a model using Llama."""
    average_score = sum(scores) / len(scores)
    explanations_text = "\n\n".join(explanations)

    prompt = assessment_prompt.format(
        model_name=model_name,
        average_score=average_score,
        explanations=explanations_text
    )

    for attempt in range(3):  # Retry up to 3 times if needed
        response = llm_evaluator.invoke(prompt).strip()
        try:
            result = json.loads(response)
            if "strengths" in result and "weaknesses" in result:
                return result
        except json.JSONDecodeError:
            print(f"Attempt {attempt + 1}: Invalid format, retrying...")

    raise ValueError(f"Failed to generate assessment for {model_name} after 3 attempts.")


def main():
    # Directories
    input_folder = '/home/arthurblb/mestrado/Divided_text/output/results/judge_qwen/contextualized/'
    output_folder = '/home/arthurblb/mestrado/Divided_text/output/results/judge_qwen/model_assessments/'

    os.makedirs(output_folder, exist_ok=True)  # Ensure output folder exists

    # Aggregate per-document evaluations
    scores, explanations = aggregate_evaluations(input_folder)

    # Generate assessments for each model
    for model in ["llama", "chatgpt"]:
        if scores[model] and explanations[model]:
            print(f"Generating assessment for model: {model}")
            try:
                assessment = generate_model_assessment(model, scores[model], explanations[model])
                output_file = os.path.join(output_folder, f"{model}_assessment.json")
                with open(output_file, 'w') as f:
                    json.dump(assessment, f, indent=4, ensure_ascii=False)
                print(f"Assessment for {model} saved to {output_file}.")
            except Exception as e:
                print(f"Failed to generate assessment for {model}: {e}")
        else:
            print(f"No data available for model: {model}.")


if __name__ == '__main__':
    main()

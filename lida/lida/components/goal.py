import json
import logging
from typing import Optional

from lida.utils import clean_code_snippet
from llmx import TextGenerator
from lida.datamodel import Goal, TextGenerationConfig, Persona


SYSTEM_INSTRUCTIONS = """
You are a an experienced data analyst who can generate a given number of insightful GOALS about data, when given a summary of the data, and a specified persona. The VISUALIZATIONS YOU RECOMMEND MUST FOLLOW VISUALIZATION BEST PRACTICES (e.g., must use bar charts instead of pie charts for comparing quantities) AND BE MEANINGFUL (e.g., plot longitude and latitude on maps where appropriate). They must also be relevant to the specified persona. Each goal must include a question, a visualization (THE VISUALIZATION MUST REFERENCE THE EXACT COLUMN FIELDS FROM THE SUMMARY), and a rationale (JUSTIFICATION FOR WHICH dataset FIELDS ARE USED and what we will learn from the visualization). Each goal MUST mention the exact fields from the dataset summary above
"""

NPF_SYSTEM_INSTRUCTIONS = """
You are an experienced networking/systems performance analyst working experiment results. You generate insightful
visualization GOALS that follow visualization best practices. Each goal must
include a question, a visualization (THE VISUALIZATION MUST REFERENCE THE EXACT
COLUMN FIELDS FROM THE SUMMARY), and a rationale. Prefer bivariate or small
multivariate charts. Do not recommend pie charts for comparing quantities.
"""

FORMAT_INSTRUCTIONS = """
THE OUTPUT MUST BE A CODE SNIPPET OF A VALID LIST OF JSON OBJECTS. IT MUST USE THE FOLLOWING FORMAT:

```[
    { "index": 0,  "question": "What is the distribution of X", "visualization": "histogram of X", "rationale": "This tells about "} ..
    ]
```
THE OUTPUT SHOULD ONLY USE THE JSON FORMAT ABOVE.
"""

logger = logging.getLogger("lida")


class GoalExplorer():
    """Generate goals given a summary of data"""

    def __init__(self) -> None:
        pass

    def generate(
        self,
        summary: dict,
        textgen_config: TextGenerationConfig,
        text_gen: TextGenerator,
        n=5,
        persona: Persona = None,
        result_type: Optional[str] = None
    ) -> list[Goal]:
        """
        Generate goals given a summary of data.

        When ``result_type`` is set (NPF result metric column name, e.g.
        THROUGHPUT), goals are constrained to focus on that metric versus
        independent experiment variables.
        """

        user_prompt = (
            f"The number of GOALS to generate is {n}. "
            f"The goals should be based on the data summary below,\n\n"
            f"{summary}\n\n"
        )

        if result_type:
            user_prompt += (
                f"IMPORTANT visualization constraints:\n"
                f"- The primary result metric / dependent variable is the column "
                f"named '{result_type}'.\n"
                f"- EVERY goal MUST focus on how '{result_type}' varies with the "
                f"independent variables (other columns in the summary).\n"
                f"- Prefer small multivariate or bivariate charts with "
                f"'{result_type}' as the primary measure.\n"
                f"- Do NOT invent unrelated metrics or goals that ignore "
                f"'{result_type}'.\n"
                f"- Use ONLY exact column field names from the dataset summary.\n"
            )

        if not persona:
            if result_type:
                persona = Persona(
                    persona=(
                        "A networking/systems performance analyst "
                        f"focused on how '{result_type}' "
                        "depends on other variables"
                    ),
                    rationale="",
                )
            else:
                persona = Persona(
                    persona="A highly skilled data analyst who can come up with complex, insightful goals about data",
                    rationale="")

        user_prompt += (
            f"\n The generated goals SHOULD BE FOCUSED ON THE INTERESTS AND "
            f"PERSPECTIVE of a '{persona.persona}' persona, who is interested in "
            f"complex, insightful goals about the data. \n"
        )

        system = NPF_SYSTEM_INSTRUCTIONS if result_type else SYSTEM_INSTRUCTIONS
        messages = [
            {"role": "system", "content": system},
            {"role": "assistant",
             "content":
             f"{user_prompt}\n\n {FORMAT_INSTRUCTIONS} \n\n. The generated {n} goals are: \n "}]

        result: list[Goal] = text_gen.generate(messages=messages, config=textgen_config)

        try:
            json_string = clean_code_snippet(result.text[0]["content"])
            result = json.loads(json_string)
            # cast each item in the list to a Goal object
            if isinstance(result, dict):
                result = [result]
            result = [Goal(**x) for x in result]
        except json.decoder.JSONDecodeError:
            logger.info(f"Error decoding JSON: {result.text[0]['content']}")
            print(f"Error decoding JSON: {result.text[0]['content']}")
            raise ValueError(
                "The model did not return a valid JSON object while attempting generate goals. Consider using a larger model or a model with higher max token length.")
        return result

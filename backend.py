from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
import os
os.environ['GOOGLE_API_KEY']='AIzaSyDDtJhXO1a3ZNosTaCyAyUsoxQIxXYkwZQ'

llm=ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.2,)

class Prompt(TypedDict):
    user_input: str
    generator_output: str
    enhancer_output: str
    critique_output: str
    
def generate_prompt(prompt:Prompt):
    query=prompt['user_input']
    if not query:
        raise ValueError("User input cannot be empty")
    messages = [
    SystemMessage(
        content="""
        You are the Prompt Generator Agent. Your purpose is to take the user’s request or idea and craft an initial, high-quality prompt that is clear, specific, and optimized for producing the best possible output from an AI model.

        Core Directives:

        Understand Intent – Carefully interpret the user’s goal, context, and constraints before generating the prompt.

        Be Clear & Complete – Write prompts that leave no room for ambiguity, providing all necessary details, context, and structure.

        Be Creative & Flexible – Offer original and imaginative prompt wording that fits the user’s needs while encouraging rich, useful responses from the AI.

        Consider Output Format – Specify the desired style, tone, and format (e.g., bullet points, step-by-step, narrative) where relevant.

        Make it Actionable – Frame the prompt so that the AI knows exactly what to do, without overcomplicating instructions.

        Output Requirements:

        Always return one single, fully-formed prompt ready to use.

        Use clear, concise language while keeping enough detail for precision.

        Avoid vague or overly broad phrasing unless explicitly requested.

        If the user’s query is unclear, intelligently infer missing details without asking clarifying questions.

        Tone: Professional, precise, and adaptable to the user’s desired style.

        You are not answering the user’s request directly — you are only creating the perfect prompt that another AI will use to answer the request.
                """
    ),
    HumanMessage(
        content=query
    )
]
    response = llm.invoke(messages)
    return {
        'generator_output': response.content,
    }
    
    

def prompt_enhancer(prompt: Prompt):
    query = prompt['generator_output']
    if not query:
        raise ValueError("Generator output cannot be empty")
    SystemMessage(
        content="""
         You are the Prompt Enhancer Agent. Your task is to take an existing prompt and refine, expand, and optimize it so it becomes more effective, precise, and capable of eliciting the highest-quality responses from an AI model.

        Core Directives:

        Preserve Original Intent – Keep the core meaning and purpose of the original prompt intact.

        Add Depth & Clarity – Fill in missing details, remove ambiguity, and improve structure without overcomplicating.

        Boost Creativity & Engagement – When suitable, enrich the prompt with imaginative framing, examples, or context to inspire richer AI outputs.

        Optimize for Accuracy – Include constraints, role definitions, tone specifications, or formatting requirements when helpful.

        Balance Conciseness & Detail – Ensure the enhanced prompt is both thorough and easy to read.

        Output Requirements:

        Return only the improved version of the prompt — not explanations or comparisons.

        Ensure language is polished, specific, and error-free.

        Maintain a style that suits the intended use case.

        Where beneficial, restructure prompts into step-by-step or role-based instructions for better AI comprehension.

        Tone: Clear, confident, and tailored to the intended output style also take care of tokens as well."""
    )
    messages = [
        HumanMessage(
            content=query
        )
    ]
    response = llm.invoke(messages)
    return {
        'enhancer_output': response.content,
    }
    
    
def critique_prompt(prompt: dict):
    query = prompt['enhancer_output']

    if not query:
        raise ValueError("Enhancer output cannot be empty") 

    critique = llm.invoke(
        [
            SystemMessage(
                content="""
                You are the Prompt Critique Agent. Your task is to:
                1. Review the given enhanced prompt.
                2. Assess its clarity, completeness, creativity, and precision.
                3. Decide if it is 'final' (good to go) or 'regenerate' (send back to the Generator).

                Rules:
                - Respond ONLY with one of these two words:
                  - "end" → The prompt is excellent and ready for use.
                  - "generate_prompt" → The prompt needs major changes, missing details, or rethinking.

                Be strict: only approve if the prompt is crystal clear, specific, and optimized for high-quality AI responses.
                """
            ),
            HumanMessage(content=query),
        ]
    )

    return {
        'critique_output': critique.content.strip()
    }



def routed_prompts(prompt: Prompt) -> str:
    router=prompt['critique_output'].strip().lower() 
    if router == 'end':
        return END
    else:
        return 'generate_prompt'
  
def graph():  
        
    graph = StateGraph(Prompt)
    graph.add_node('generate_prompt',generate_prompt)
    graph.add_node('prompt_enhancer', prompt_enhancer)
    graph.add_node('critique_prompt', critique_prompt)

    graph.add_edge(START, 'generate_prompt')
    graph.add_edge('generate_prompt', 'prompt_enhancer')
    graph.add_edge('prompt_enhancer', 'critique_prompt')
    graph.add_conditional_edges('critique_prompt', routed_prompts,{
        END: END,
        'generate_prompt': 'generate_prompt'
    })
    graph.set_entry_point('generate_prompt')

    prompt=graph.compile()
    return prompt['ehancer_output']
 
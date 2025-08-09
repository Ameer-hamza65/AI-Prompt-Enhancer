from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
import os
os.environ['GOOGLE_API_KEY']=os.getenv('GOOGLE_API_KEY')
llm=ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
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
    ROLE: Critical Aspect Identifier
    TASK: Extract the essential components needed to create a perfect prompt from the user's query
    
    DIRECTIVES:
    1. DECONSTRUCT the query to identify:
       - Core subject matter
       - Required depth and specificity
       - Implicit constraints
       - Domain-specific requirements
       - Unstated audience/context
    2. IDENTIFY knowledge gaps that need addressing
    3. MAP the query to optimal prompt structures:
       - Comparative analysis
       - Technical explanation
       - Creative generation
       - Step-by-step process
    4. OUTPUT only the structured metadata in this format:
       {
         "core_subject": "...",
         "depth_level": "...",  // beginner, intermediate, expert
         "constraints": ["...", "..."],
         "required_components": ["...", "..."],
         "output_structure": "..."
       }
    
    RULES:
    - NEVER generate content or final prompts
    - NEVER add your own interpretations beyond what's implied
    - ALWAYS maintain original intent
    - KEEP output machine-readable
    
    EXAMPLE:
    User Query: "Compare renewable energy sources"
    Output:
    {
      "core_subject": "Renewable energy comparison",
      "depth_level": "intermediate",
      "constraints": ["technical accuracy", "balanced perspective"],
      "required_components": ["efficiency metrics", "cost analysis", "environmental impact", "adoption challenges"],
      "output_structure": "comparative analysis"
    }
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
    
    # Fixed: Include SystemMessage in messages list
    messages = [
        SystemMessage(
            content="""
            ROLE: Elite Prompt Architect
            TASK: Transform extracted aspects into flawless, production-grade prompts
            
            PROCESSING PROTOCOL:
            1. RECEIVE structured metadata from Aspect Extractor
            2. APPLY advanced prompt engineering techniques:
               - Role specialization framework
               - Cognitive chain scaffolding
               - Multi-stage verification systems
               - Anti-hallucination safeguards
            3. IMPLEMENT these enhancement layers:
               a) Precision Engineering: Quantifiable metrics, exact format specs
               b) Knowledge Integration: Required sources, validation methods
               c) Structural Optimization: Step-by-step reasoning flow
               d) Quality Assurance: Self-correction mechanisms
            
            4. OUTPUT REQUIREMENTS:
               - Single executable prompt
               - Also include meaningful emojis to enhance user engagement
               - No explanations or metadata    
               - Fully self-contained
               - Token-optimized (max 300 tokens)
            
            CONSTRUCTION TEMPLATE:
            "Act as [SPECIALIZED_ROLE]. [TASK_DESCRIPTION] using [METHODOLOGY]. 
            Required: [COMPONENTS]. Structure: [OUTPUT_FORMAT]. 
            Validation: [QUALITY_CONTROLS]. Execute via: [PROCESS_FLOW]."
            
            EXAMPLE TRANSFORMATION:
            Input Metadata: {
              "core_subject": "Renewable energy comparison",
              "depth_level": "intermediate",
              "constraints": ["technical accuracy", "balanced perspective"],
              "required_components": ["efficiency metrics", "cost analysis", "environmental impact", "adoption challenges"],
              "output_structure": "comparative analysis"
            }
            
            Output Prompt:
            "Act as Senior Energy Analyst. Compare solar, wind, and hydro energy across efficiency metrics (LCOE calculations), installation costs (2024 USD), environmental impact (carbon footprint/lifecycle), and adoption barriers. Structure: 1) Executive summary 2) Comparative matrix 3) Regional viability analysis 4) Policy recommendations. Validate: Cross-check stats against IEA/NREL databases. Execute: Research → Analysis → Peer-review simulation → Finalize."
            """
        ),
        HumanMessage(
            content=f"Transform these extracted aspects into a production-grade prompt:\n{query}"
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
        
    workflow = StateGraph(Prompt)
    workflow.add_node('generate_prompt',generate_prompt)
    workflow.add_node('prompt_enhancer', prompt_enhancer)
    workflow.add_node('critique_prompt', critique_prompt)

    workflow.add_edge(START, 'generate_prompt')
    workflow.add_edge('generate_prompt', 'prompt_enhancer')
    workflow.add_edge('prompt_enhancer', 'critique_prompt')
    workflow.add_conditional_edges('critique_prompt', routed_prompts,{
        END: END,
        'generate_prompt': 'generate_prompt' 
    })
    workflow.set_entry_point('generate_prompt') 

    return workflow.compile() 
 

app=graph()  
 

def run_optimization(user_query:str)->str:
    initial_state=Prompt(
        user_input=user_query,
        generator_output='',
        enhancer_output='', 
        critique_output=''      
    )
    
    final_state=app.invoke(initial_state)
    return final_state['enhancer_output']



if __name__ == "__main__":
    user_input='write a blog for the reniewable energy sources.'
    enhanced_prompt=run_optimization(user_input)
    print('Optimized Prompt: ',enhanced_prompt)
    



    
           
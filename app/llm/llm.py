from langchain_ollama import OllamaLLM

llm = OllamaLLM(model="llama3.2" , temperature=0)

def generate_answer(prompt):
    return llm.invoke(prompt)
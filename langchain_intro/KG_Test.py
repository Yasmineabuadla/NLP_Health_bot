import pandas as pd
import networkx as nx

# Load the CSV file
csv_file_path = './Data/Disease_symptom_and_patient_profile_dataset 2.csv'
df = pd.read_csv(csv_file_path)

# Create an empty graph
G = nx.Graph()

# Iterate through the DataFrame and add nodes and edges
for index, row in df.iterrows():
    patient_id = f"Patient_{index}"
    disease = row['Disease']
    age = row['Age']
    gender = row['Gender']
    blood_pressure = row['Blood Pressure']
    cholesterol_level = row['Cholesterol Level']
    outcome = row['Outcome Variable']

    # Symptoms
    symptoms = {
        'Fever': row['Fever'],
        'Cough': row['Cough'],
        'Fatigue': row['Fatigue'],
        'Difficulty Breathing': row['Difficulty Breathing']
    }

    # Add patient node
    G.add_node(patient_id, type='Patient', age=age, gender=gender, blood_pressure=blood_pressure, cholesterol_level=cholesterol_level, outcome=outcome)

    # Add disease node and edge
    G.add_node(disease, type='Disease')
    G.add_edge(patient_id, disease, relation='has_disease')

    # Add symptom nodes and edges
    for symptom, present in symptoms.items():
        if present == 'Yes':
            G.add_node(symptom, type='Symptom')
            G.add_edge(patient_id, symptom, relation='has_symptom')

# Print the graph information
num_nodes = G.number_of_nodes()
num_edges = G.number_of_edges()
graph_info = f'Graph with {num_nodes} nodes and {num_edges} edges'
print(graph_info)

def query_knowledge_graph(graph, entity_type, entity_value):
    print(f"Querying graph for {entity_type} = {entity_value}")
    result = []

    if entity_type == 'Patient':
        # Find all symptoms and diseases for a given patient
        neighbors = list(graph.neighbors(entity_value))
        for neighbor in neighbors:
            if graph.nodes[neighbor]['type'] == 'Symptom':
                result.append(f"Symptom: {neighbor}")
            elif graph.nodes[neighbor]['type'] == 'Disease':
                result.append(f"Disease: {neighbor}")

    elif entity_type == 'Disease':
        # Find all patients with a specific disease
        patients = []
        for node in graph.nodes:
            if graph.nodes[node]['type'] == 'Patient':
                neighbors = list(graph.neighbors(node))
                if entity_value in neighbors:
                    patients.append(node)

        # Find all symptoms for these patients
        symptoms = set()
        for patient in patients:
            neighbors = list(graph.neighbors(patient))
            for neighbor in neighbors:
                if graph.nodes[neighbor]['type'] == 'Symptom':
                    symptoms.add(neighbor)
        result.extend([f"Symptom: {symptom}" for symptom in symptoms])

    elif entity_type == 'Symptom':
        # Find all patients with a specific symptom
        for node in graph.nodes:
            if graph.nodes[node]['type'] == 'Patient':
                neighbors = list(graph.neighbors(node))
                if entity_value in neighbors:
                    result.append(f"Patient: {node}")

    print(f"Query result: {result}")
    return result


# Example usage
response = query_knowledge_graph(G, "Patient", "Patient_0")
print("Response:", response)

# Integration with an agent (simplified version)
class Tool:
    def __init__(self, name, func, description):
        self.name = name
        self.func = func
        self.description = description

knowledge_graph_tool = Tool(
    name="KnowledgeGraph",
    func=lambda query: query_knowledge_graph(G, query['entity_type'], query['entity_value']),
    description="Useful for retrieving structured information and relationships between entities. Use for complex queries that require detailed and interconnected information."
)

class Agent:
    def __init__(self, tools):
        self.tools = tools

    def invoke(self, query):
        tool_name = query['tool']
        tool_query = query['input']
        for tool in self.tools:
            if tool.name == tool_name:
                return tool.func(tool_query)

# Initialize agent with the knowledge graph tool
agent = Agent(tools=[knowledge_graph_tool])

# Example usage of the agent
query = {
    'tool': 'KnowledgeGraph',
    'input': {
        'entity_type': 'Patient',
        'entity_value': 'Patient_0'
    }
}

response = agent.invoke(query)
print("Response from agent:", response)
